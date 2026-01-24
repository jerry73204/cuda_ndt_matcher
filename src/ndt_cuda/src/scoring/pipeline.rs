//! GPU Scoring Pipeline for batch NVTL/TP computation.
//!
//! This module provides a zero-copy GPU pipeline for batch scoring,
//! replacing the Rayon-parallel CPU implementation in `evaluate_nvtl_batch()`.
//!
//! # Architecture
//!
//! ```text
//! Once per map (set_target):
//!   Upload: voxel_means [V×3], voxel_inv_covs [V×9]
//!
//! Per batch call (M poses, N points):
//!   Upload: source_points [N×3], transforms [M×16]
//!
//!   GPU Kernel (M×N threads):
//!     1. Transform point by pose
//!     2. Radius search for neighbors (per transformed point)
//!     3. Accumulate sum_score and max_score across neighbors
//!     4. Output: outputs[M×N×4] = [score, max_score, has_neighbor, correspondences]
//!
//!   CUB DeviceSegmentedReduce:
//!     - Reduce scores, max_scores per pose
//!
//!   Download and compute:
//!     transform_probability[m] = total_scores[m] / correspondences[m]
//!     nvtl[m] = nvtl_sums[m] / nvtl_counts[m]
//! ```

use anyhow::{Context, Result};
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::gpu::{compute_scores_batch_kernel, pose_to_transform_matrix_f32};
use crate::derivatives::gpu::GpuVoxelData;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// Result of batch scoring for a single pose.
#[derive(Debug, Clone)]
pub struct BatchScoringResult {
    /// Transform probability: sum(scores) / num_correspondences
    pub transform_probability: f64,
    /// NVTL: sum(max_scores) / num_points_with_neighbors
    pub nvtl: f64,
    /// Number of point-voxel correspondences
    pub num_correspondences: usize,
    /// Number of points with at least one neighbor
    pub num_with_neighbors: usize,
}

/// GPU pipeline for batch scoring (NVTL and transform probability).
///
/// This pipeline processes multiple poses in a single GPU kernel launch,
/// providing significant speedup over the Rayon-parallel CPU implementation.
pub struct GpuScoringPipeline {
    client: CudaClient,

    // Capacity
    max_points: usize,
    max_voxels: usize,
    max_poses: usize,

    // Current sizes
    num_voxels: usize,

    // Persistent voxel data (uploaded once per map via set_target)
    voxel_means: Handle,    // [V × 3]
    voxel_inv_covs: Handle, // [V × 9]
    voxel_valid: Handle,    // [V]

    // Per-batch buffers
    source_points: Handle, // [N × 3]
    transforms: Handle,    // [M × 16]
    gauss_params: Handle,  // [3] = [d1, d2, search_radius_sq]

    // Combined output buffer [M × N × 4]
    // Layout: [score, max_score, has_neighbor, correspondences] per (pose, point)
    outputs: Handle,

    // CUB reduction buffers
    reduce_temp: Handle,
    reduce_temp_bytes: usize,
    reduce_offsets: Handle, // [num_segments + 1]
    reduce_output: Handle,  // [num_segments]

    // Separate arrays for strided reduction
    scores_strided: Handle,     // [M × N]
    max_scores_strided: Handle, // [M × N]
    #[allow(dead_code)]
    has_neighbor_strided: Handle, // [M × N] - reserved for future GPU reduction
    #[allow(dead_code)]
    correspondences_strided: Handle, // [M × N] - reserved for future GPU reduction

    // Parameters
    gauss_d1: f32,
    gauss_d2: f32,
    search_radius_sq: f32,

    // Flag indicating if target has been set
    target_set: bool,
}

impl GpuScoringPipeline {
    /// Create a new GPU scoring pipeline with given capacity.
    ///
    /// # Arguments
    /// * `max_points` - Maximum number of source points
    /// * `max_voxels` - Maximum number of voxels in target grid
    /// * `max_poses` - Maximum number of poses per batch
    pub fn new(max_points: usize, max_voxels: usize, max_poses: usize) -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        // Allocate voxel buffers
        let voxel_means = client.empty(max_voxels * 3 * std::mem::size_of::<f32>());
        let voxel_inv_covs = client.empty(max_voxels * 9 * std::mem::size_of::<f32>());
        let voxel_valid = client.empty(max_voxels * std::mem::size_of::<u32>());

        // Allocate per-batch buffers
        let source_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let transforms = client.empty(max_poses * 16 * std::mem::size_of::<f32>());
        let gauss_params = client.empty(3 * std::mem::size_of::<f32>());

        // Combined output buffer [M × N × 4]
        let max_output = max_poses * max_points;
        let outputs = client.empty(max_output * 4 * std::mem::size_of::<f32>());

        // Separate strided arrays for CUB reduction
        let scores_strided = client.empty(max_output * std::mem::size_of::<f32>());
        let max_scores_strided = client.empty(max_output * std::mem::size_of::<f32>());
        let has_neighbor_strided = client.empty(max_output * std::mem::size_of::<f32>());
        let correspondences_strided = client.empty(max_output * std::mem::size_of::<f32>());

        // CUB reduction buffers
        let num_segments = max_poses;
        let total_items = max_output;
        let reduce_temp_bytes =
            cuda_ffi::segmented_reduce_sum_f32_temp_size(total_items, num_segments)
                .context("Failed to query CUB reduce temp size")?;
        let reduce_temp = client.empty(reduce_temp_bytes.max(1));
        let reduce_offsets = client.empty((num_segments + 1) * std::mem::size_of::<i32>());
        let reduce_output = client.empty(num_segments * std::mem::size_of::<f32>());

        Ok(Self {
            client,
            max_points,
            max_voxels,
            max_poses,
            num_voxels: 0,
            voxel_means,
            voxel_inv_covs,
            voxel_valid,
            source_points,
            transforms,
            gauss_params,
            outputs,
            reduce_temp,
            reduce_temp_bytes,
            reduce_offsets,
            reduce_output,
            scores_strided,
            max_scores_strided,
            has_neighbor_strided,
            correspondences_strided,
            gauss_d1: 1.0,
            gauss_d2: 1.0,
            search_radius_sq: 4.0, // Default 2.0^2
            target_set: false,
        })
    }

    /// Get the maximum points capacity.
    pub fn max_points(&self) -> usize {
        self.max_points
    }

    /// Get the maximum voxels capacity.
    pub fn max_voxels(&self) -> usize {
        self.max_voxels
    }

    /// Get the maximum poses capacity.
    pub fn max_poses(&self) -> usize {
        self.max_poses
    }

    /// Get raw CUDA pointer from a CubeCL handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let binding = handle.clone().binding();
        let resource = self.client.get_resource(binding);
        resource.resource().ptr
    }

    /// Upload target voxel data (call once per map).
    ///
    /// # Arguments
    /// * `voxel_data` - Voxel grid data from map
    /// * `gauss_d1` - Gaussian d1 parameter
    /// * `gauss_d2` - Gaussian d2 parameter
    /// * `search_radius` - Radius for voxel search (typically = voxel resolution)
    pub fn set_target(
        &mut self,
        voxel_data: &GpuVoxelData,
        gauss_d1: f64,
        gauss_d2: f64,
        search_radius: f64,
    ) -> Result<()> {
        let num_voxels = voxel_data.num_voxels;

        if num_voxels > self.max_voxels {
            anyhow::bail!("Too many voxels: {} > max {}", num_voxels, self.max_voxels);
        }

        // Upload voxel data
        self.voxel_means = self.client.create(f32::as_bytes(&voxel_data.means));
        self.voxel_inv_covs = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));

        let valid_bytes: Vec<u8> = voxel_data
            .valid
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        self.voxel_valid = self.client.create(&valid_bytes);

        self.num_voxels = num_voxels;
        self.gauss_d1 = gauss_d1 as f32;
        self.gauss_d2 = gauss_d2 as f32;
        self.search_radius_sq = (search_radius * search_radius) as f32;
        self.target_set = true;

        Ok(())
    }

    /// Compute batch scores for multiple poses.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud
    /// * `poses` - Array of poses [x, y, z, roll, pitch, yaw]
    ///
    /// # Returns
    /// Vector of scoring results, one per pose.
    pub fn compute_scores_batch(
        &mut self,
        source_points: &[[f32; 3]],
        poses: &[[f64; 6]],
    ) -> Result<Vec<BatchScoringResult>> {
        if !self.target_set {
            anyhow::bail!("Target not set. Call set_target() first.");
        }

        let num_points = source_points.len();
        let num_poses = poses.len();

        if num_points == 0 || num_poses == 0 {
            return Ok(vec![]);
        }

        if num_points > self.max_points {
            anyhow::bail!("Too many points: {} > max {}", num_points, self.max_points);
        }
        if num_poses > self.max_poses {
            anyhow::bail!("Too many poses: {} > max {}", num_poses, self.max_poses);
        }

        // Upload source points
        let points_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();
        self.source_points = self.client.create(f32::as_bytes(&points_flat));

        // Convert poses to transform matrices and upload
        let transforms_flat: Vec<f32> = poses
            .iter()
            .flat_map(|pose| pose_to_transform_matrix_f32(pose).into_iter())
            .collect();
        self.transforms = self.client.create(f32::as_bytes(&transforms_flat));

        // Upload Gaussian parameters
        let gauss_params_data = [self.gauss_d1, self.gauss_d2, self.search_radius_sq];
        self.gauss_params = self.client.create(f32::as_bytes(&gauss_params_data));

        // Launch batched kernel
        // Grid: (num_poses, ceil(num_points / 256))
        // Block: (256,)
        let block_size = 256u32;
        let num_point_blocks = (num_points as u32).div_ceil(block_size);

        unsafe {
            compute_scores_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(num_poses as u32, num_point_blocks, 1),
                CubeDim::new(block_size, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transforms, num_poses * 16, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, self.num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, self.num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<u32>(&self.voxel_valid, self.num_voxels, 1),
                ArrayArg::from_raw_parts::<f32>(&self.gauss_params, 3, 1),
                ScalarArg::new(num_poses as u32),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(self.num_voxels as u32),
                ArrayArg::from_raw_parts::<f32>(&self.outputs, num_poses * num_points * 4, 1),
            );
        }

        // Sync before reading results
        cubecl::future::block_on(self.client.sync());

        // Read combined output and parse on CPU
        // Output layout: [M × N × 4] where each entry is [score, max_score, has_neighbor, correspondences]
        let output_bytes = self.client.read_one(self.outputs.clone());
        let output_floats = f32::from_bytes(&output_bytes);

        // Build results
        let mut results = Vec::with_capacity(num_poses);
        for pose_idx in 0..num_poses {
            let mut total_score = 0.0f64;
            let mut total_max_score = 0.0f64;
            let mut total_with_neighbors = 0usize;
            let mut total_correspondences = 0usize;

            for point_idx in 0..num_points {
                let base = (pose_idx * num_points + point_idx) * 4;
                let score = output_floats[base] as f64;
                let max_score = output_floats[base + 1] as f64;
                let has_neighbor = output_floats[base + 2];
                let correspondences = output_floats[base + 3];

                total_score += score;
                total_max_score += max_score;
                if has_neighbor > 0.5 {
                    total_with_neighbors += 1;
                }
                total_correspondences += correspondences as usize;
            }

            let transform_probability = if total_correspondences > 0 {
                total_score / total_correspondences as f64
            } else {
                0.0
            };

            let nvtl = if total_with_neighbors > 0 {
                total_max_score / total_with_neighbors as f64
            } else {
                0.0
            };

            results.push(BatchScoringResult {
                transform_probability,
                nvtl,
                num_correspondences: total_correspondences,
                num_with_neighbors: total_with_neighbors,
            });
        }

        Ok(results)
    }

    /// Compute batch scores using GPU reduction (more efficient for large batches).
    ///
    /// This method uses CUB DeviceSegmentedReduce for GPU-side aggregation,
    /// reducing data transfer. For small batches, `compute_scores_batch` may be faster.
    pub fn compute_scores_batch_gpu_reduce(
        &mut self,
        source_points: &[[f32; 3]],
        poses: &[[f64; 6]],
    ) -> Result<Vec<BatchScoringResult>> {
        if !self.target_set {
            anyhow::bail!("Target not set. Call set_target() first.");
        }

        let num_points = source_points.len();
        let num_poses = poses.len();

        if num_points == 0 || num_poses == 0 {
            return Ok(vec![]);
        }

        if num_points > self.max_points {
            anyhow::bail!("Too many points: {} > max {}", num_points, self.max_points);
        }
        if num_poses > self.max_poses {
            anyhow::bail!("Too many poses: {} > max {}", num_poses, self.max_poses);
        }

        // Upload source points
        let points_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();
        self.source_points = self.client.create(f32::as_bytes(&points_flat));

        // Convert poses to transform matrices and upload
        let transforms_flat: Vec<f32> = poses
            .iter()
            .flat_map(|pose| pose_to_transform_matrix_f32(pose).into_iter())
            .collect();
        self.transforms = self.client.create(f32::as_bytes(&transforms_flat));

        // Upload Gaussian parameters
        let gauss_params_data = [self.gauss_d1, self.gauss_d2, self.search_radius_sq];
        self.gauss_params = self.client.create(f32::as_bytes(&gauss_params_data));

        // Launch batched kernel
        let block_size = 256u32;
        let num_point_blocks = (num_points as u32).div_ceil(block_size);

        unsafe {
            compute_scores_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(num_poses as u32, num_point_blocks, 1),
                CubeDim::new(block_size, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transforms, num_poses * 16, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, self.num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, self.num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<u32>(&self.voxel_valid, self.num_voxels, 1),
                ArrayArg::from_raw_parts::<f32>(&self.gauss_params, 3, 1),
                ScalarArg::new(num_poses as u32),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(self.num_voxels as u32),
                ArrayArg::from_raw_parts::<f32>(&self.outputs, num_poses * num_points * 4, 1),
            );
        }

        // Sync before CUB operations
        cubecl::future::block_on(self.client.sync());

        // Extract strided arrays from combined output for CUB reduction
        // Output layout: [M × N × 4] = [score, max_score, has_neighbor, correspondences]
        // We need to transpose to get contiguous [M × N] arrays for each component

        let output_bytes = self.client.read_one(self.outputs.clone());
        let output_floats = f32::from_bytes(&output_bytes);

        // Extract components into separate arrays
        let total_elements = num_poses * num_points;
        let mut scores_data = Vec::with_capacity(total_elements);
        let mut max_scores_data = Vec::with_capacity(total_elements);
        let mut has_neighbor_data = Vec::with_capacity(total_elements);
        let mut correspondences_data = Vec::with_capacity(total_elements);

        for i in 0..total_elements {
            scores_data.push(output_floats[i * 4]);
            max_scores_data.push(output_floats[i * 4 + 1]);
            has_neighbor_data.push(output_floats[i * 4 + 2]);
            correspondences_data.push(output_floats[i * 4 + 3]);
        }

        // Upload strided arrays
        self.scores_strided = self.client.create(f32::as_bytes(&scores_data));
        self.max_scores_strided = self.client.create(f32::as_bytes(&max_scores_data));
        cubecl::future::block_on(self.client.sync());

        let n = num_points as i32;
        let m = num_poses;

        // Build segment offsets: [0, N, 2N, ..., M*N]
        let offsets: Vec<i32> = (0..=m as i32).map(|i| i * n).collect();
        self.reduce_offsets = self.client.create(i32::as_bytes(&offsets));
        cubecl::future::block_on(self.client.sync());

        // 1. Reduce scores: sum per pose
        unsafe {
            cuda_ffi::segmented_reduce_sum_f32_inplace(
                self.raw_ptr(&self.reduce_temp),
                self.reduce_temp_bytes,
                self.raw_ptr(&self.scores_strided),
                self.raw_ptr(&self.reduce_output),
                m,
                self.raw_ptr(&self.reduce_offsets),
            )
            .context("CUB reduce scores failed")?;
        }
        let score_bytes = self.client.read_one(self.reduce_output.clone());
        let total_scores = f32::from_bytes(&score_bytes);

        // 2. Reduce max_scores: sum per pose (for NVTL numerator)
        unsafe {
            cuda_ffi::segmented_reduce_sum_f32_inplace(
                self.raw_ptr(&self.reduce_temp),
                self.reduce_temp_bytes,
                self.raw_ptr(&self.max_scores_strided),
                self.raw_ptr(&self.reduce_output),
                m,
                self.raw_ptr(&self.reduce_offsets),
            )
            .context("CUB reduce max_scores failed")?;
        }
        let nvtl_sum_bytes = self.client.read_one(self.reduce_output.clone());
        let nvtl_sums = f32::from_bytes(&nvtl_sum_bytes);

        // 3. Sum correspondences and has_neighbor on CPU
        let mut results = Vec::with_capacity(num_poses);
        for pose_idx in 0..num_poses {
            let base = pose_idx * num_points;
            let mut total_correspondences = 0usize;
            let mut total_with_neighbors = 0usize;

            for point_idx in 0..num_points {
                let idx = base + point_idx;
                total_correspondences += correspondences_data[idx] as usize;
                if has_neighbor_data[idx] > 0.5 {
                    total_with_neighbors += 1;
                }
            }

            let transform_probability = if total_correspondences > 0 {
                total_scores[pose_idx] as f64 / total_correspondences as f64
            } else {
                0.0
            };

            let nvtl = if total_with_neighbors > 0 {
                nvtl_sums[pose_idx] as f64 / total_with_neighbors as f64
            } else {
                0.0
            };

            results.push(BatchScoringResult {
                transform_probability,
                nvtl,
                num_correspondences: total_correspondences,
                num_with_neighbors: total_with_neighbors,
            });
        }

        Ok(results)
    }

    /// Compute per-point max scores for visualization (single pose).
    ///
    /// This method returns the max NDT score at each point, suitable for
    /// per-point color visualization (like Autoware's voxel_score_points).
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud
    /// * `pose` - Single pose [x, y, z, roll, pitch, yaw]
    ///
    /// # Returns
    /// Tuple of (transformed_points, max_scores) where:
    /// - transformed_points[i] is source_points[i] transformed by pose (in map frame)
    /// - max_scores[i] is the maximum NDT score for point i across all neighbor voxels
    pub fn compute_per_point_scores(
        &mut self,
        source_points: &[[f32; 3]],
        pose: &[f64; 6],
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>)> {
        if !self.target_set {
            anyhow::bail!("Target not set. Call set_target() first.");
        }

        let num_points = source_points.len();
        if num_points == 0 {
            return Ok((vec![], vec![]));
        }

        if num_points > self.max_points {
            anyhow::bail!("Too many points: {} > max {}", num_points, self.max_points);
        }

        // Upload source points
        let points_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();
        self.source_points = self.client.create(f32::as_bytes(&points_flat));

        // Convert pose to transform matrix and upload
        let transform = pose_to_transform_matrix_f32(pose);
        self.transforms = self.client.create(f32::as_bytes(&transform));

        // Upload Gaussian parameters
        let gauss_params_data = [self.gauss_d1, self.gauss_d2, self.search_radius_sq];
        self.gauss_params = self.client.create(f32::as_bytes(&gauss_params_data));

        // Launch kernel for single pose
        let block_size = 256u32;
        let num_point_blocks = (num_points as u32).div_ceil(block_size);

        unsafe {
            compute_scores_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(1, num_point_blocks, 1),
                CubeDim::new(block_size, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transforms, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, self.num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, self.num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<u32>(&self.voxel_valid, self.num_voxels, 1),
                ArrayArg::from_raw_parts::<f32>(&self.gauss_params, 3, 1),
                ScalarArg::new(1u32),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(self.num_voxels as u32),
                ArrayArg::from_raw_parts::<f32>(&self.outputs, num_points * 4, 1),
            );
        }

        // Sync and read results
        cubecl::future::block_on(self.client.sync());
        let output_bytes = self.client.read_one(self.outputs.clone());
        let output_floats = f32::from_bytes(&output_bytes);

        // Extract max_scores (index 1 in each 4-element group)
        let max_scores: Vec<f32> = (0..num_points).map(|i| output_floats[i * 4 + 1]).collect();

        // Transform source points to map frame
        let transformed_points: Vec<[f32; 3]> = source_points
            .iter()
            .map(|p| {
                // Apply 4x4 transform matrix (row-major)
                let x = p[0] as f64;
                let y = p[1] as f64;
                let z = p[2] as f64;
                let tx = transform[0] as f64 * x
                    + transform[1] as f64 * y
                    + transform[2] as f64 * z
                    + transform[3] as f64;
                let ty = transform[4] as f64 * x
                    + transform[5] as f64 * y
                    + transform[6] as f64 * z
                    + transform[7] as f64;
                let tz = transform[8] as f64 * x
                    + transform[9] as f64 * y
                    + transform[10] as f64 * z
                    + transform[11] as f64;
                [tx as f32, ty as f32, tz as f32]
            })
            .collect();

        Ok((transformed_points, max_scores))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::derivatives::GaussianParams;
    use crate::scoring::{compute_nvtl, compute_transform_probability, NvtlConfig};
    use crate::voxel_grid::VoxelGrid;
    use approx::assert_relative_eq;
    use nalgebra::Isometry3;

    fn make_test_grid() -> (VoxelGrid, GpuVoxelData) {
        // Create a simple voxel grid
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.1, 0.0],
            [0.1, 0.0, 0.1],
            [0.0, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.05, 0.05, 0.05],
            [0.15, 0.05, 0.05],
        ];
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();
        let voxel_data = GpuVoxelData::from_voxel_grid(&grid);
        (grid, voxel_data)
    }
    #[test]
    fn test_gpu_scoring_single_pose() {
        let (grid, voxel_data) = make_test_grid();
        let gauss = GaussianParams::default();

        let mut pipeline = GpuScoringPipeline::new(1000, 1000, 100).unwrap();
        pipeline
            .set_target(&voxel_data, gauss.d1, gauss.d2, grid.resolution() as f64)
            .unwrap();

        // Source points at voxel center
        let source_points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]];

        // Identity pose
        let poses: Vec<[f64; 6]> = vec![[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        let results = pipeline
            .compute_scores_batch(&source_points, &poses)
            .unwrap();

        assert_eq!(results.len(), 1);
        // NDT scores are positive when d1 is negative (default)
        // Score = -d1 * exp(...), with d1 < 0, so -d1 > 0
        assert!(
            results[0].transform_probability > 0.0,
            "TP should be positive (NDT score = -d1*exp(...) where d1 < 0)"
        );
        assert!(
            results[0].nvtl > 0.0,
            "NVTL should be positive (NDT score = -d1*exp(...) where d1 < 0)"
        );
        assert!(results[0].num_with_neighbors > 0, "Should find neighbors");
    }
    #[test]
    fn test_gpu_scoring_multiple_poses() {
        let (grid, voxel_data) = make_test_grid();
        let gauss = GaussianParams::default();

        let mut pipeline = GpuScoringPipeline::new(1000, 1000, 100).unwrap();
        pipeline
            .set_target(&voxel_data, gauss.d1, gauss.d2, grid.resolution() as f64)
            .unwrap();

        let source_points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];

        // Multiple poses: identity and translated
        let poses: Vec<[f64; 6]> = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // At voxel
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Far away
        ];

        let results = pipeline
            .compute_scores_batch(&source_points, &poses)
            .unwrap();

        assert_eq!(results.len(), 2);

        // First pose should have neighbors
        assert!(
            results[0].num_with_neighbors > 0,
            "Pose at origin should find neighbors"
        );

        // Second pose should have no neighbors (far from map)
        assert_eq!(
            results[1].num_with_neighbors, 0,
            "Pose far away should have no neighbors"
        );
    }
    #[test]
    fn test_gpu_vs_cpu_scoring() {
        let (grid, voxel_data) = make_test_grid();
        let gauss = GaussianParams::default();

        let mut pipeline = GpuScoringPipeline::new(1000, 1000, 100).unwrap();
        pipeline
            .set_target(&voxel_data, gauss.d1, gauss.d2, grid.resolution() as f64)
            .unwrap();

        let source_points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ];

        let pose = [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0];
        let poses = vec![pose];

        // GPU result
        let gpu_results = pipeline
            .compute_scores_batch(&source_points, &poses)
            .unwrap();

        // CPU result
        let isometry = Isometry3::identity();
        let cpu_tp = compute_transform_probability(&source_points, &grid, &isometry, &gauss);
        let _cpu_nvtl = compute_nvtl(
            &source_points,
            &grid,
            &isometry,
            &gauss,
            &NvtlConfig::default(),
        );

        // Compare (with some tolerance due to f32 vs f64 differences)
        // Note: The GPU uses brute-force search while CPU uses KD-tree,
        // so results may differ slightly for edge cases
        crate::test_println!(
            "GPU TP: {}, CPU TP: {}",
            gpu_results[0].transform_probability,
            cpu_tp.transform_probability
        );
        crate::test_println!(
            "GPU NVTL: {}, CPU NVTL: {}",
            gpu_results[0].nvtl,
            _cpu_nvtl.nvtl
        );

        // Both should be positive (NDT score = -d1*exp(...) where d1 < 0)
        assert!(gpu_results[0].transform_probability > 0.0);
        assert!(cpu_tp.transform_probability > 0.0);

        // Should be in same ballpark (GPU uses f32, CPU uses f64)
        assert_relative_eq!(
            gpu_results[0].transform_probability,
            cpu_tp.transform_probability,
            epsilon = 0.001
        );
    }
    #[test]
    fn test_gpu_scoring_empty_input() {
        let (grid, voxel_data) = make_test_grid();
        let gauss = GaussianParams::default();

        let mut pipeline = GpuScoringPipeline::new(1000, 1000, 100).unwrap();
        pipeline
            .set_target(&voxel_data, gauss.d1, gauss.d2, grid.resolution() as f64)
            .unwrap();

        // Empty source points
        let source_points: Vec<[f32; 3]> = vec![];
        let poses: Vec<[f64; 6]> = vec![[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        let results = pipeline
            .compute_scores_batch(&source_points, &poses)
            .unwrap();
        assert!(results.is_empty());

        // Empty poses
        let source_points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
        let poses: Vec<[f64; 6]> = vec![];

        let results = pipeline
            .compute_scores_batch(&source_points, &poses)
            .unwrap();
        assert!(results.is_empty());
    }
    #[test]
    fn test_gpu_scoring_no_correspondences() {
        let (grid, voxel_data) = make_test_grid();
        let gauss = GaussianParams::default();

        let mut pipeline = GpuScoringPipeline::new(1000, 1000, 100).unwrap();
        pipeline
            .set_target(&voxel_data, gauss.d1, gauss.d2, grid.resolution() as f64)
            .unwrap();

        // Points far from map
        let source_points: Vec<[f32; 3]> = vec![[1000.0, 1000.0, 1000.0]];
        let poses: Vec<[f64; 6]> = vec![[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        let results = pipeline
            .compute_scores_batch(&source_points, &poses)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_correspondences, 0);
        assert_eq!(results[0].num_with_neighbors, 0);
        assert_eq!(results[0].transform_probability, 0.0);
        assert_eq!(results[0].nvtl, 0.0);
    }
    #[test]
    fn test_gpu_scoring_gpu_reduce() {
        let (grid, voxel_data) = make_test_grid();
        let gauss = GaussianParams::default();

        let mut pipeline = GpuScoringPipeline::new(1000, 1000, 100).unwrap();
        pipeline
            .set_target(&voxel_data, gauss.d1, gauss.d2, grid.resolution() as f64)
            .unwrap();

        let source_points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ];

        let poses: Vec<[f64; 6]> = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        // Compare CPU aggregation vs GPU reduce
        let cpu_agg_results = pipeline
            .compute_scores_batch(&source_points, &poses)
            .unwrap();
        let gpu_reduce_results = pipeline
            .compute_scores_batch_gpu_reduce(&source_points, &poses)
            .unwrap();

        assert_eq!(cpu_agg_results.len(), gpu_reduce_results.len());
        for i in 0..cpu_agg_results.len() {
            assert_relative_eq!(
                cpu_agg_results[i].transform_probability,
                gpu_reduce_results[i].transform_probability,
                epsilon = 1e-5
            );
            assert_relative_eq!(
                cpu_agg_results[i].nvtl,
                gpu_reduce_results[i].nvtl,
                epsilon = 1e-5
            );
        }
    }
}
