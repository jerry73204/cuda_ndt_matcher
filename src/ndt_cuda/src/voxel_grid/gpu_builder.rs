//! GPU-accelerated voxel grid construction.
//!
//! This module provides GPU-accelerated voxel grid construction using CubeCL.
//!
//! # Approaches
//!
//! ## 1. Hybrid (current default)
//! - GPU: Compute voxel IDs for all points (parallel per-point)
//! - CPU: Group points by voxel ID and compute statistics
//!
//! ## 2. Segmented Reduction (new, more GPU work)
//! - GPU: Morton codes (CubeCL) + radix sort (CUB via cuda_ffi)
//! - CPU: Segment detection (GPU planned)
//! - GPU: Statistics accumulation (sum, mean, covariance)
//! - CPU: Finalization (eigendecomposition)
//!
//! The segmented approach avoids atomic operations by sorting points so that
//! same-voxel points are contiguous, then using one thread per voxel.
//!
//! See `docs/gpu-voxel-statistics.md` for detailed algorithm analysis.

use std::collections::HashMap;

use anyhow::{Context, Result};
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use super::gpu::morton::{compute_morton_codes_kernel, MortonCodeResult};
use super::gpu::pipeline::GpuPipelineBuffers;
use super::gpu::radix_sort::radix_sort_by_key;
use super::gpu::segments::detect_segments;
use super::gpu::statistics::{
    accumulate_segment_covariances_kernel, accumulate_segment_sums_kernel, compute_means_kernel,
    finalize_voxels_cpu,
};
use super::kernels::compute_voxel_ids_kernel;
use super::types::{Voxel, VoxelCoord, VoxelGridConfig};
use super::VoxelGrid;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// GPU-accelerated voxel grid builder.
///
/// Holds a CubeCL client for GPU operations and provides methods to
/// construct voxel grids from point clouds.
pub struct GpuVoxelGridBuilder {
    client: CudaClient,
}

impl GpuVoxelGridBuilder {
    /// Create a new GPU voxel grid builder.
    ///
    /// # Returns
    /// A new builder, or error if CUDA is not available.
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);
        Ok(Self { client })
    }

    /// Compute Morton codes on GPU.
    ///
    /// # Arguments
    /// * `points_flat` - Flattened point coordinates [x0, y0, z0, x1, y1, z1, ...]
    /// * `resolution` - Voxel resolution
    ///
    /// # Returns
    /// Morton codes and indices, with bounds information.
    fn compute_morton_codes_gpu(&self, points_flat: &[f32], resolution: f32) -> MortonCodeResult {
        let num_points = points_flat.len() / 3;

        if num_points == 0 {
            return MortonCodeResult {
                codes: Vec::new(),
                indices: Vec::new(),
                num_points: 0,
                grid_min: [0.0; 3],
                grid_max: [0.0; 3],
            };
        }

        // Step 1: Compute bounds on CPU (small reduction, not worth GPU overhead)
        let (grid_min, grid_max) = compute_bounds_flat(points_flat);
        let inv_resolution = 1.0 / resolution;

        // Step 2: Upload data to GPU
        let points_gpu = self.client.create(f32::as_bytes(points_flat));
        let min_bound_gpu = self.client.create(f32::as_bytes(&grid_min));

        // Allocate output buffers
        let codes_low_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());
        let codes_high_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());
        let indices_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        // Step 3: Launch kernel
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            compute_morton_codes_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&min_bound_gpu, 3, 1),
                ScalarArg::new(inv_resolution),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<u32>(&codes_low_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&codes_high_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&indices_gpu, num_points, 1),
            );
        }

        // Step 4: Download results
        let codes_low_bytes = self.client.read_one(codes_low_gpu);
        let codes_high_bytes = self.client.read_one(codes_high_gpu);
        let indices_bytes = self.client.read_one(indices_gpu);

        let codes_low = u32::from_bytes(&codes_low_bytes);
        let codes_high = u32::from_bytes(&codes_high_bytes);

        // Combine low/high into u64 Morton codes
        let mut codes = Vec::with_capacity(num_points * 8);
        for i in 0..num_points {
            let code = (codes_high[i] as u64) << 32 | codes_low[i] as u64;
            codes.extend_from_slice(&code.to_le_bytes());
        }

        MortonCodeResult {
            codes,
            indices: indices_bytes.to_vec(),
            num_points: num_points as u32,
            grid_min,
            grid_max,
        }
    }

    /// Build a voxel grid from a point cloud using GPU acceleration.
    ///
    /// The GPU is used to compute voxel IDs in parallel, then CPU handles
    /// the accumulation and statistics computation.
    ///
    /// # Arguments
    /// * `points` - Input point cloud
    /// * `config` - Voxel grid configuration
    ///
    /// # Returns
    /// A VoxelGrid with computed statistics.
    pub fn build(&self, points: &[[f32; 3]], config: &VoxelGridConfig) -> Result<VoxelGrid> {
        if points.is_empty() {
            return Ok(VoxelGrid::new(config.clone()));
        }

        let num_points = points.len();

        // Step 1: Compute bounds to determine grid parameters
        let (min_bound, max_bound) = compute_point_bounds(points);
        let inv_resolution = 1.0 / config.resolution;

        // Compute grid dimensions
        let grid_dim_x = ((max_bound[0] - min_bound[0]) * inv_resolution).ceil() as u32 + 1;
        let grid_dim_y = ((max_bound[1] - min_bound[1]) * inv_resolution).ceil() as u32 + 1;

        // Step 2: Flatten points for GPU upload
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();

        // Step 3: Upload to GPU
        let points_gpu = self.client.create(f32::as_bytes(&points_flat));
        let min_bound_gpu = self.client.create(f32::as_bytes(&min_bound));
        let voxel_ids_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        // Step 4: Launch kernel
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            compute_voxel_ids_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&min_bound_gpu, 3, 1),
                ScalarArg::new(inv_resolution),
                ScalarArg::new(grid_dim_x),
                ScalarArg::new(grid_dim_y),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<u32>(&voxel_ids_gpu, num_points, 1),
            );
        }

        // Step 5: Download voxel IDs
        let voxel_ids_bytes = self.client.read_one(voxel_ids_gpu);
        let voxel_ids = u32::from_bytes(&voxel_ids_bytes);

        // Step 6: Convert voxel IDs back to coordinates and group points
        let min_coord = VoxelCoord::new(
            (min_bound[0] * inv_resolution).floor() as i32,
            (min_bound[1] * inv_resolution).floor() as i32,
            (min_bound[2] * inv_resolution).floor() as i32,
        );

        // Group points by voxel ID
        let mut voxel_points: HashMap<u32, Vec<usize>> = HashMap::new();
        for (point_idx, &voxel_id) in voxel_ids.iter().enumerate() {
            voxel_points.entry(voxel_id).or_default().push(point_idx);
        }

        // Step 7: Compute statistics per voxel (parallel with rayon)
        let voxel_entries: Vec<_> = voxel_points.into_iter().collect();

        let voxels: Vec<_> = voxel_entries
            .into_par_iter()
            .filter_map(|(voxel_id, point_indices)| {
                if point_indices.len() < config.min_points_per_voxel {
                    return None;
                }

                // Convert voxel ID back to coordinate
                let coord = voxel_id_to_coord(voxel_id, min_coord, grid_dim_x, grid_dim_y);

                // Accumulate statistics
                let mut sum = Vector3::zeros();
                let mut sum_sq = Matrix3::zeros();

                for &idx in &point_indices {
                    let p = &points[idx];
                    let v = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                    sum += v;
                    sum_sq += v * v.transpose();
                }

                // Create voxel with statistics
                let voxel = Voxel::from_statistics(&sum, &sum_sq, point_indices.len(), config)?;
                Some((coord, voxel))
            })
            .collect();

        // Step 8: Build VoxelGrid from computed voxels
        let mut result = VoxelGrid::new(config.clone());
        for (coord, voxel) in voxels {
            result.insert(coord, voxel);
        }

        // Build search index
        result.build_search_index();

        Ok(result)
    }

    /// Build a voxel grid using segmented reduction (GPU statistics).
    ///
    /// This approach uses more GPU for statistics computation:
    /// 1. CPU: Morton codes + radix sort (to group points by voxel)
    /// 2. CPU: Segment detection (identify voxel boundaries)
    /// 3. GPU: Position sum accumulation (one thread per voxel)
    /// 4. GPU: Mean computation
    /// 5. GPU: Covariance accumulation (one thread per voxel)
    /// 6. CPU: Finalization (eigendecomposition for regularization)
    ///
    /// This avoids atomic operations by exploiting the sorted data structure.
    ///
    /// # Arguments
    /// * `points` - Input point cloud
    /// * `config` - Voxel grid configuration
    ///
    /// # Returns
    /// A VoxelGrid with computed statistics.
    pub fn build_segmented(
        &self,
        points: &[[f32; 3]],
        config: &VoxelGridConfig,
    ) -> Result<VoxelGrid> {
        if points.is_empty() {
            return Ok(VoxelGrid::new(config.clone()));
        }

        let num_points = points.len();

        // Step 1: Flatten points for processing
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();

        // Step 2: Compute Morton codes (GPU)
        let morton_result = self.compute_morton_codes_gpu(&points_flat, config.resolution);

        // Parse Morton codes from bytes
        let morton_codes: Vec<u64> = morton_result
            .codes
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let original_indices: Vec<u32> = morton_result
            .indices
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Step 3: Radix sort by Morton code (GPU via CUB with CPU fallback)
        let sort_result = radix_sort_by_key(&morton_codes, &original_indices);

        let sorted_codes: Vec<u64> = sort_result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let sorted_indices: Vec<u32> = sort_result
            .values
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Step 4: Detect segments (voxel boundaries) - GPU via CUB with CPU fallback
        let segment_result = detect_segments(&sorted_codes);
        let num_segments = segment_result.num_segments as usize;

        if num_segments == 0 {
            return Ok(VoxelGrid::new(config.clone()));
        }

        // Step 5: Upload data to GPU
        let points_gpu = self.client.create(f32::as_bytes(&points_flat));
        let sorted_indices_gpu = self.client.create(u32::as_bytes(&sorted_indices));
        let segment_starts_gpu = self
            .client
            .create(u32::as_bytes(&segment_result.segment_starts));

        // Allocate output buffers
        let position_sums_gpu = self
            .client
            .empty(num_segments * 3 * std::mem::size_of::<f32>());
        let counts_gpu = self.client.empty(num_segments * std::mem::size_of::<u32>());
        let means_gpu = self
            .client
            .empty(num_segments * 3 * std::mem::size_of::<f32>());
        let cov_sums_gpu = self
            .client
            .empty(num_segments * 9 * std::mem::size_of::<f32>());

        // Step 6: Launch position sums kernel
        let cube_count = num_segments.div_ceil(256) as u32;
        unsafe {
            accumulate_segment_sums_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&sorted_indices_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&segment_starts_gpu, num_segments, 1),
                ScalarArg::new(num_segments as u32),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&position_sums_gpu, num_segments * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&counts_gpu, num_segments, 1),
            );
        }

        // Step 7: Launch means kernel
        unsafe {
            compute_means_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&position_sums_gpu, num_segments * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&counts_gpu, num_segments, 1),
                ScalarArg::new(num_segments as u32),
                ArrayArg::from_raw_parts::<f32>(&means_gpu, num_segments * 3, 1),
            );
        }

        // Step 8: Launch covariance kernel
        unsafe {
            accumulate_segment_covariances_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&sorted_indices_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&segment_starts_gpu, num_segments, 1),
                ArrayArg::from_raw_parts::<f32>(&means_gpu, num_segments * 3, 1),
                ScalarArg::new(num_segments as u32),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&cov_sums_gpu, num_segments * 9, 1),
            );
        }

        // Step 9: Download results from GPU
        let means_bytes = self.client.read_one(means_gpu);
        let means = f32::from_bytes(&means_bytes).to_vec();

        let cov_sums_bytes = self.client.read_one(cov_sums_gpu);
        let cov_sums = f32::from_bytes(&cov_sums_bytes).to_vec();

        let counts_bytes = self.client.read_one(counts_gpu);
        let counts = u32::from_bytes(&counts_bytes).to_vec();

        // Step 10: Finalize voxels on CPU (eigendecomposition)
        let stats =
            finalize_voxels_cpu(means, cov_sums, counts, config.min_points_per_voxel as u32);

        // Step 11: Build VoxelGrid from statistics
        let inv_resolution = 1.0 / config.resolution;
        let mut result = VoxelGrid::new(config.clone());

        for seg_idx in 0..num_segments {
            if !stats.valid[seg_idx] {
                continue;
            }

            // Decode Morton code to get voxel coordinate
            let morton_code = segment_result.segment_codes[seg_idx];
            let (gx, gy, gz) = super::gpu::morton::morton_decode_3d(morton_code);

            // Convert grid coordinates to voxel coordinates
            let vx = morton_result.grid_min[0] * inv_resolution + gx as f32;
            let vy = morton_result.grid_min[1] * inv_resolution + gy as f32;
            let vz = morton_result.grid_min[2] * inv_resolution + gz as f32;

            let coord = VoxelCoord::new(vx.floor() as i32, vy.floor() as i32, vz.floor() as i32);

            // Create voxel with precomputed statistics
            let mean = Vector3::new(
                stats.means[seg_idx * 3],
                stats.means[seg_idx * 3 + 1],
                stats.means[seg_idx * 3 + 2],
            );

            let covariance = Matrix3::new(
                stats.covariances[seg_idx * 9],
                stats.covariances[seg_idx * 9 + 1],
                stats.covariances[seg_idx * 9 + 2],
                stats.covariances[seg_idx * 9 + 3],
                stats.covariances[seg_idx * 9 + 4],
                stats.covariances[seg_idx * 9 + 5],
                stats.covariances[seg_idx * 9 + 6],
                stats.covariances[seg_idx * 9 + 7],
                stats.covariances[seg_idx * 9 + 8],
            );

            let inv_covariance = Matrix3::new(
                stats.inv_covariances[seg_idx * 9],
                stats.inv_covariances[seg_idx * 9 + 1],
                stats.inv_covariances[seg_idx * 9 + 2],
                stats.inv_covariances[seg_idx * 9 + 3],
                stats.inv_covariances[seg_idx * 9 + 4],
                stats.inv_covariances[seg_idx * 9 + 5],
                stats.inv_covariances[seg_idx * 9 + 6],
                stats.inv_covariances[seg_idx * 9 + 7],
                stats.inv_covariances[seg_idx * 9 + 8],
            );

            let principal_axis = Vector3::new(
                stats.principal_axes[seg_idx * 3],
                stats.principal_axes[seg_idx * 3 + 1],
                stats.principal_axes[seg_idx * 3 + 2],
            );

            let voxel = Voxel {
                mean,
                covariance,
                inv_covariance,
                principal_axis,
                point_count: stats.point_counts[seg_idx] as usize,
            };

            result.insert(coord, voxel);
        }

        // Build search index
        result.build_search_index();

        Ok(result)
    }

    /// Build a voxel grid using zero-copy GPU pipeline.
    ///
    /// This is the most efficient approach, minimizing CPU-GPU transfers:
    /// 1. Single CPU→GPU transfer: points
    /// 2. All GPU operations run without intermediate transfers
    /// 3. Single GPU→CPU transfer: final statistics
    /// 4. CPU: Eigendecomposition (small data, ~1000x less work)
    ///
    /// # Arguments
    /// * `points` - Input point cloud
    /// * `config` - Voxel grid configuration
    ///
    /// # Returns
    /// A VoxelGrid with computed statistics.
    pub fn build_zero_copy(
        &self,
        points: &[[f32; 3]],
        config: &VoxelGridConfig,
    ) -> Result<VoxelGrid> {
        if points.is_empty() {
            return Ok(VoxelGrid::new(config.clone()));
        }

        // Estimate max segments as num_points / min_points_per_voxel
        let max_segments = points.len() / config.min_points_per_voxel.max(1);
        let max_segments = max_segments.max(1000); // At least 1000 segments

        // Create pipeline buffers
        let mut pipeline = GpuPipelineBuffers::new(points.len(), max_segments)
            .context("Failed to create pipeline buffers")?;

        // Run the zero-copy pipeline
        let pipeline_result = pipeline
            .build(points, config.resolution, config.min_points_per_voxel)
            .context("Pipeline execution failed")?;

        if pipeline_result.num_segments == 0 {
            return Ok(VoxelGrid::new(config.clone()));
        }

        // Convert pipeline result to VoxelGrid
        let inv_resolution = 1.0 / config.resolution;
        let mut result = VoxelGrid::new(config.clone());

        for seg_idx in 0..pipeline_result.num_segments {
            if !pipeline_result.stats.valid[seg_idx] {
                continue;
            }

            // Decode Morton code to get voxel coordinate
            let morton_code = pipeline_result.segment_codes[seg_idx];
            let (gx, gy, gz) = super::gpu::morton::morton_decode_3d(morton_code);

            // Convert grid coordinates to voxel coordinates
            let vx = pipeline_result.grid_min[0] * inv_resolution + gx as f32;
            let vy = pipeline_result.grid_min[1] * inv_resolution + gy as f32;
            let vz = pipeline_result.grid_min[2] * inv_resolution + gz as f32;

            let coord = VoxelCoord::new(vx.floor() as i32, vy.floor() as i32, vz.floor() as i32);

            // Create voxel with precomputed statistics
            let mean = Vector3::new(
                pipeline_result.stats.means[seg_idx * 3],
                pipeline_result.stats.means[seg_idx * 3 + 1],
                pipeline_result.stats.means[seg_idx * 3 + 2],
            );

            let covariance = Matrix3::new(
                pipeline_result.stats.covariances[seg_idx * 9],
                pipeline_result.stats.covariances[seg_idx * 9 + 1],
                pipeline_result.stats.covariances[seg_idx * 9 + 2],
                pipeline_result.stats.covariances[seg_idx * 9 + 3],
                pipeline_result.stats.covariances[seg_idx * 9 + 4],
                pipeline_result.stats.covariances[seg_idx * 9 + 5],
                pipeline_result.stats.covariances[seg_idx * 9 + 6],
                pipeline_result.stats.covariances[seg_idx * 9 + 7],
                pipeline_result.stats.covariances[seg_idx * 9 + 8],
            );

            let inv_covariance = Matrix3::new(
                pipeline_result.stats.inv_covariances[seg_idx * 9],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 1],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 2],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 3],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 4],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 5],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 6],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 7],
                pipeline_result.stats.inv_covariances[seg_idx * 9 + 8],
            );

            let principal_axis = Vector3::new(
                pipeline_result.stats.principal_axes[seg_idx * 3],
                pipeline_result.stats.principal_axes[seg_idx * 3 + 1],
                pipeline_result.stats.principal_axes[seg_idx * 3 + 2],
            );

            let voxel = Voxel {
                mean,
                covariance,
                inv_covariance,
                principal_axis,
                point_count: pipeline_result.stats.point_counts[seg_idx] as usize,
            };

            result.insert(coord, voxel);
        }

        // Build search index
        result.build_search_index();

        Ok(result)
    }
}

/// Compute min/max bounds of a point cloud.
fn compute_point_bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for p in points {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }

    (min, max)
}

/// Compute min/max bounds of a flattened point cloud.
fn compute_bounds_flat(points: &[f32]) -> ([f32; 3], [f32; 3]) {
    let num_points = points.len() / 3;
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for i in 0..num_points {
        for j in 0..3 {
            let v = points[i * 3 + j];
            min[j] = min[j].min(v);
            max[j] = max[j].max(v);
        }
    }

    (min, max)
}

/// Convert voxel ID back to coordinate.
fn voxel_id_to_coord(
    id: u32,
    min_coord: VoxelCoord,
    grid_dim_x: u32,
    grid_dim_y: u32,
) -> VoxelCoord {
    let dim_xy = grid_dim_x * grid_dim_y;
    let z = id / dim_xy;
    let remainder = id % dim_xy;
    let y = remainder / grid_dim_x;
    let x = remainder % grid_dim_x;

    VoxelCoord::new(
        x as i32 + min_coord.x,
        y as i32 + min_coord.y,
        z as i32 + min_coord.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_default_half_cubic_pcd;

    #[test]
    fn test_gpu_voxel_grid_construction() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let grid = builder
            .build(&points, &config)
            .expect("Failed to build grid");

        // Should produce voxels
        assert!(!grid.is_empty());
        assert!(grid.len() > 100, "Expected >100 voxels for half-cubic PCD");
    }

    #[test]
    fn test_gpu_cpu_consistency() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        // Build with GPU
        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let gpu_grid = builder
            .build(&points, &config)
            .expect("Failed to build GPU grid");

        // Build with CPU
        let cpu_grid = VoxelGrid::from_points_with_config(&points, config.clone())
            .expect("Failed to build CPU grid");

        // Should have similar voxel counts
        let diff = (gpu_grid.len() as i32 - cpu_grid.len() as i32).abs();
        assert!(
            diff <= cpu_grid.len() as i32 / 10,
            "GPU ({}) and CPU ({}) voxel counts should be similar",
            gpu_grid.len(),
            cpu_grid.len()
        );
    }

    #[test]
    fn test_gpu_segmented_voxel_grid_construction() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let grid = builder
            .build_segmented(&points, &config)
            .expect("Failed to build grid with segmented reduction");

        // Should produce voxels
        assert!(!grid.is_empty());
        assert!(
            grid.len() > 100,
            "Expected >100 voxels for half-cubic PCD, got {}",
            grid.len()
        );
    }

    #[test]
    fn test_gpu_segmented_vs_cpu_consistency() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        // Build with GPU segmented reduction
        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let gpu_grid = builder
            .build_segmented(&points, &config)
            .expect("Failed to build GPU segmented grid");

        // Build with CPU
        let cpu_grid = VoxelGrid::from_points_with_config(&points, config.clone())
            .expect("Failed to build CPU grid");

        // Should have similar voxel counts (allow some tolerance due to different grouping)
        let diff = (gpu_grid.len() as i32 - cpu_grid.len() as i32).abs();
        assert!(
            diff <= cpu_grid.len() as i32 / 5, // 20% tolerance
            "GPU segmented ({}) and CPU ({}) voxel counts differ too much",
            gpu_grid.len(),
            cpu_grid.len()
        );

        // Check that at least some voxels have valid statistics
        let valid_count = gpu_grid
            .voxels()
            .iter()
            .filter(|v| v.point_count >= config.min_points_per_voxel)
            .count();
        assert!(
            valid_count > 50,
            "Expected >50 valid voxels, got {}",
            valid_count
        );
    }

    #[test]
    fn test_gpu_segmented_vs_hybrid_consistency() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");

        // Build with hybrid approach
        let hybrid_grid = builder
            .build(&points, &config)
            .expect("Failed to build hybrid grid");

        // Build with segmented reduction
        let segmented_grid = builder
            .build_segmented(&points, &config)
            .expect("Failed to build segmented grid");

        // Should have similar voxel counts
        let diff = (hybrid_grid.len() as i32 - segmented_grid.len() as i32).abs();
        assert!(
            diff <= hybrid_grid.len() as i32 / 5, // 20% tolerance
            "Hybrid ({}) and segmented ({}) voxel counts differ too much",
            hybrid_grid.len(),
            segmented_grid.len()
        );
    }

    #[test]
    fn test_gpu_zero_copy_voxel_grid_construction() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let grid = builder
            .build_zero_copy(&points, &config)
            .expect("Failed to build grid with zero-copy pipeline");

        // Should produce voxels
        assert!(!grid.is_empty());
        assert!(
            grid.len() > 100,
            "Expected >100 voxels for half-cubic PCD, got {}",
            grid.len()
        );
    }

    #[test]
    fn test_gpu_zero_copy_vs_cpu_consistency() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        // Build with GPU zero-copy pipeline
        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let gpu_grid = builder
            .build_zero_copy(&points, &config)
            .expect("Failed to build GPU zero-copy grid");

        // Build with CPU
        let cpu_grid = VoxelGrid::from_points_with_config(&points, config.clone())
            .expect("Failed to build CPU grid");

        // Should have similar voxel counts (allow some tolerance due to different grouping)
        let diff = (gpu_grid.len() as i32 - cpu_grid.len() as i32).abs();
        assert!(
            diff <= cpu_grid.len() as i32 / 5, // 20% tolerance
            "GPU zero-copy ({}) and CPU ({}) voxel counts differ too much",
            gpu_grid.len(),
            cpu_grid.len()
        );

        // Check that at least some voxels have valid statistics
        let valid_count = gpu_grid
            .voxels()
            .iter()
            .filter(|v| v.point_count >= config.min_points_per_voxel)
            .count();
        assert!(
            valid_count > 50,
            "Expected >50 valid voxels, got {}",
            valid_count
        );
    }

    #[test]
    fn test_gpu_zero_copy_vs_segmented_consistency() {
        let points = make_default_half_cubic_pcd();
        let config = VoxelGridConfig::default();

        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");

        // Build with segmented reduction
        let segmented_grid = builder
            .build_segmented(&points, &config)
            .expect("Failed to build segmented grid");

        // Build with zero-copy pipeline
        let zero_copy_grid = builder
            .build_zero_copy(&points, &config)
            .expect("Failed to build zero-copy grid");

        // Should have similar voxel counts
        let diff = (segmented_grid.len() as i32 - zero_copy_grid.len() as i32).abs();
        assert!(
            diff <= segmented_grid.len() as i32 / 5, // 20% tolerance
            "Segmented ({}) and zero-copy ({}) voxel counts differ too much",
            segmented_grid.len(),
            zero_copy_grid.len()
        );
    }

    #[test]
    fn test_gpu_zero_copy_empty_input() {
        let points: Vec<[f32; 3]> = vec![];
        let config = VoxelGridConfig::default();

        let builder = GpuVoxelGridBuilder::new().expect("Failed to create GPU builder");
        let grid = builder
            .build_zero_copy(&points, &config)
            .expect("Failed to build grid with empty input");

        assert!(grid.is_empty());
    }
}
