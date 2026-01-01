//! GPU runtime management for CubeCL CUDA execution.
//!
//! This module provides the GPU runtime infrastructure for NDT computation:
//! - Device initialization and client management
//! - GPU buffer allocation and data transfer
//! - Kernel launch wrappers
//!
//! # Example
//!
//! ```ignore
//! use ndt_cuda::runtime::GpuRuntime;
//!
//! let runtime = GpuRuntime::new()?;
//! let scores = runtime.compute_scores(&source_points, &voxel_data, &transform)?;
//! ```

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;

use crate::derivatives::gpu::{
    compute_ndt_gradient_kernel, compute_ndt_nvtl_kernel, compute_ndt_score_kernel,
    compute_point_jacobians_cpu, pose_to_transform_matrix, radius_search_kernel, GpuVoxelData,
    MAX_NEIGHBORS,
};
use crate::voxel_grid::kernels::transform_points_kernel;

/// Type alias for CUDA compute client
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// GPU runtime for NDT computation.
///
/// Manages CUDA device initialization and provides high-level APIs
/// for GPU-accelerated NDT operations.
pub struct GpuRuntime {
    /// CUDA device (kept alive for runtime lifetime)
    #[allow(dead_code)]
    device: CudaDevice,
    /// Compute client for kernel execution
    client: CudaClient,
}

impl GpuRuntime {
    /// Create a new GPU runtime with the default CUDA device.
    pub fn new() -> Result<Self> {
        Self::with_device_id(0)
    }

    /// Create a new GPU runtime with a specific CUDA device.
    pub fn with_device_id(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id);
        let client = CudaRuntime::client(&device);

        Ok(Self { device, client })
    }

    /// Get the underlying compute client.
    pub fn client(&self) -> &CudaClient {
        &self.client
    }

    /// Compute NDT scores for source points against voxel grid.
    ///
    /// Returns per-point scores and total correspondence count.
    pub fn compute_scores(
        &self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        transform: &[f32; 16],
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<GpuScoreResult> {
        if source_points.is_empty() {
            return Ok(GpuScoreResult {
                scores: Vec::new(),
                total_score: 0.0,
                correspondences: Vec::new(),
                total_correspondences: 0,
            });
        }

        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload data to GPU
        let source_gpu = self.client.create(f32::as_bytes(&source_flat));
        let transform_gpu = self.client.create(f32::as_bytes(transform));
        let voxel_means_gpu = self.client.create(f32::as_bytes(&voxel_data.means));
        let voxel_inv_covs_gpu = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        let voxel_valid_gpu = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Allocate transformed points buffer
        let transformed_gpu = self
            .client
            .empty(num_points * 3 * std::mem::size_of::<f32>());

        // Step 1: Transform points
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
            );
        }

        // Step 2: Radius search
        let neighbor_indices_gpu = self
            .client
            .empty(num_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        let radius_sq = search_radius * search_radius;
        unsafe {
            radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&voxel_valid_gpu, num_voxels, 1),
                ScalarArg::new(radius_sq),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(num_voxels as u32),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
            );
        }

        // Step 3: Compute scores
        let scores_gpu = self.client.empty(num_points * std::mem::size_of::<f32>());
        let correspondences_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        unsafe {
            compute_ndt_score_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                ScalarArg::new(gauss_d1),
                ScalarArg::new(gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&scores_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&correspondences_gpu, num_points, 1),
            );
        }

        // Read results back
        let scores_bytes = self.client.read_one(scores_gpu);
        let scores = f32::from_bytes(&scores_bytes).to_vec();

        let correspondences_bytes = self.client.read_one(correspondences_gpu);
        let correspondences = u32::from_bytes(&correspondences_bytes).to_vec();

        let total_score: f32 = scores.iter().sum();
        let total_correspondences: u32 = correspondences.iter().sum();

        Ok(GpuScoreResult {
            scores,
            total_score: total_score as f64,
            correspondences,
            total_correspondences: total_correspondences as usize,
        })
    }

    /// Compute NVTL (Nearest Voxel Transformation Likelihood) scores using GPU.
    ///
    /// This matches Autoware's NVTL algorithm:
    /// - For each point, find the **maximum** score across all neighbor voxels
    /// - Final NVTL = average of these max scores
    ///
    /// This differs from `compute_scores()` which sums all voxel contributions
    /// (used for transform probability).
    pub fn compute_nvtl_scores(
        &self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        transform: &[f32; 16],
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<GpuNvtlResult> {
        if source_points.is_empty() {
            return Ok(GpuNvtlResult {
                max_scores: Vec::new(),
                nvtl: 0.0,
                num_with_neighbors: 0,
            });
        }

        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload data to GPU
        let source_gpu = self.client.create(f32::as_bytes(&source_flat));
        let transform_gpu = self.client.create(f32::as_bytes(transform));
        let voxel_means_gpu = self.client.create(f32::as_bytes(&voxel_data.means));
        let voxel_inv_covs_gpu = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        let voxel_valid_gpu = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Allocate transformed points buffer
        let transformed_gpu = self
            .client
            .empty(num_points * 3 * std::mem::size_of::<f32>());

        // Step 1: Transform points
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
            );
        }

        // Step 2: Radius search
        let neighbor_indices_gpu = self
            .client
            .empty(num_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        let radius_sq = search_radius * search_radius;
        unsafe {
            radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&voxel_valid_gpu, num_voxels, 1),
                ScalarArg::new(radius_sq),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(num_voxels as u32),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
            );
        }

        // Step 3: Compute NVTL scores (max per point, not sum)
        let max_scores_gpu = self.client.empty(num_points * std::mem::size_of::<f32>());
        let has_neighbor_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        unsafe {
            compute_ndt_nvtl_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                ScalarArg::new(gauss_d1),
                ScalarArg::new(gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&max_scores_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&has_neighbor_gpu, num_points, 1),
            );
        }

        // Read results back
        let max_scores_bytes = self.client.read_one(max_scores_gpu);
        let max_scores = f32::from_bytes(&max_scores_bytes).to_vec();

        let has_neighbor_bytes = self.client.read_one(has_neighbor_gpu);
        let has_neighbor = u32::from_bytes(&has_neighbor_bytes);

        // Compute NVTL: average of max scores for points with neighbors
        let num_with_neighbors: usize = has_neighbor.iter().map(|&h| h as usize).sum();
        let total_max_score: f64 = max_scores
            .iter()
            .zip(has_neighbor.iter())
            .filter(|(_, &h)| h > 0)
            .map(|(&s, _)| s as f64)
            .sum();

        let nvtl = if num_with_neighbors > 0 {
            total_max_score / num_with_neighbors as f64
        } else {
            0.0
        };

        Ok(GpuNvtlResult {
            max_scores,
            nvtl,
            num_with_neighbors,
        })
    }

    /// Compute NDT derivatives (score, gradient) for optimization.
    ///
    /// This is the main function used during NDT alignment iterations.
    pub fn compute_derivatives(
        &self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        pose: &[f64; 6],
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<GpuDerivativeResult> {
        if source_points.is_empty() {
            return Ok(GpuDerivativeResult {
                score: 0.0,
                gradient: [0.0; 6],
                num_correspondences: 0,
            });
        }

        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        // Convert pose to transform matrix
        let transform = pose_to_transform_matrix(pose);

        // Compute point Jacobians on CPU (small overhead, complex computation)
        let jacobians = compute_point_jacobians_cpu(source_points, pose);

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload data to GPU
        let source_gpu = self.client.create(f32::as_bytes(&source_flat));
        let transform_gpu = self.client.create(f32::as_bytes(&transform));
        let jacobians_gpu = self.client.create(f32::as_bytes(&jacobians));
        let voxel_means_gpu = self.client.create(f32::as_bytes(&voxel_data.means));
        let voxel_inv_covs_gpu = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        let voxel_valid_gpu = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Allocate transformed points buffer
        let transformed_gpu = self
            .client
            .empty(num_points * 3 * std::mem::size_of::<f32>());

        // Step 1: Transform points
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
            );
        }

        // Step 2: Radius search
        let neighbor_indices_gpu = self
            .client
            .empty(num_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        let radius_sq = search_radius * search_radius;
        unsafe {
            radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&transformed_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&voxel_valid_gpu, num_voxels, 1),
                ScalarArg::new(radius_sq),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(num_voxels as u32),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
            );
        }

        // Step 3: Compute scores
        let scores_gpu = self.client.empty(num_points * std::mem::size_of::<f32>());
        let correspondences_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());

        unsafe {
            compute_ndt_score_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                ScalarArg::new(gauss_d1),
                ScalarArg::new(gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&scores_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&correspondences_gpu, num_points, 1),
            );
        }

        // Step 4: Compute gradients
        let gradients_gpu = self
            .client
            .empty(num_points * 6 * std::mem::size_of::<f32>());

        unsafe {
            compute_ndt_gradient_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&jacobians_gpu, num_points * 18, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                ScalarArg::new(gauss_d1),
                ScalarArg::new(gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&gradients_gpu, num_points * 6, 1),
            );
        }

        // Read results back
        let scores_bytes = self.client.read_one(scores_gpu);
        let scores = f32::from_bytes(&scores_bytes);

        let correspondences_bytes = self.client.read_one(correspondences_gpu);
        let correspondences = u32::from_bytes(&correspondences_bytes);

        let gradients_bytes = self.client.read_one(gradients_gpu);
        let gradients = f32::from_bytes(&gradients_bytes);

        // Reduce on CPU (small reduction, not worth GPU overhead)
        let total_score: f64 = scores.iter().map(|&s| s as f64).sum();
        let total_correspondences: usize = correspondences.iter().map(|&c| c as usize).sum();

        let mut total_gradient = [0.0f64; 6];
        for i in 0..num_points {
            for j in 0..6 {
                total_gradient[j] += gradients[i * 6 + j] as f64;
            }
        }

        Ok(GpuDerivativeResult {
            score: total_score,
            gradient: total_gradient,
            num_correspondences: total_correspondences,
        })
    }

    /// Transform points using the GPU.
    pub fn transform_points(
        &self,
        points: &[[f32; 3]],
        transform: &[f32; 16],
    ) -> Result<Vec<[f32; 3]>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let num_points = points.len();
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();

        let points_gpu = self.client.create(f32::as_bytes(&points_flat));
        let transform_gpu = self.client.create(f32::as_bytes(transform));
        let output_gpu = self
            .client
            .empty(num_points * 3 * std::mem::size_of::<f32>());

        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&output_gpu, num_points * 3, 1),
            );
        }

        let output_bytes = self.client.read_one(output_gpu);
        let output_flat = f32::from_bytes(&output_bytes);

        let mut result = Vec::with_capacity(num_points);
        for i in 0..num_points {
            result.push([
                output_flat[i * 3],
                output_flat[i * 3 + 1],
                output_flat[i * 3 + 2],
            ]);
        }

        Ok(result)
    }
}

/// Result of GPU score computation.
#[derive(Debug, Clone)]
pub struct GpuScoreResult {
    /// Per-point scores
    pub scores: Vec<f32>,
    /// Total score (sum of all per-point scores)
    pub total_score: f64,
    /// Per-point correspondence counts
    pub correspondences: Vec<u32>,
    /// Total number of correspondences
    pub total_correspondences: usize,
}

/// Result of GPU derivative computation.
#[derive(Debug, Clone)]
pub struct GpuDerivativeResult {
    /// Total score
    pub score: f64,
    /// Total gradient (6 elements)
    pub gradient: [f64; 6],
    /// Total number of correspondences
    pub num_correspondences: usize,
}

/// Result of GPU NVTL (Nearest Voxel Transformation Likelihood) computation.
///
/// NVTL takes the **maximum** score per point across all neighbor voxels,
/// then computes the average. This matches Autoware's NVTL algorithm.
#[derive(Debug, Clone)]
pub struct GpuNvtlResult {
    /// Per-point max scores (0.0 for points with no neighbors)
    pub max_scores: Vec<f32>,
    /// NVTL = average of max scores for points with neighbors
    pub nvtl: f64,
    /// Number of points that had at least one neighbor voxel
    pub num_with_neighbors: usize,
}

/// Check if CUDA is available on this system.
pub fn is_cuda_available() -> bool {
    // Try to create a device - if it fails, CUDA is not available
    std::panic::catch_unwind(|| {
        let _device = CudaDevice::new(0);
    })
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test just checks if we can query CUDA availability
        let available = is_cuda_available();
        println!("CUDA available: {available}");
    }

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_transform_points_gpu() {
        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");

        let points = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Identity transform
        let transform = [
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ];

        let result = runtime.transform_points(&points, &transform).unwrap();

        assert_eq!(result.len(), 2);
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (result[i][j] - points[i][j]).abs() < 1e-5,
                    "Mismatch at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_transform_points_translation() {
        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");

        let points = vec![[0.0, 0.0, 0.0]];

        // Translation by (1, 2, 3)
        let transform = [
            1.0, 0.0, 0.0, 1.0, // row 0: translate x by 1
            0.0, 1.0, 0.0, 2.0, // row 1: translate y by 2
            0.0, 0.0, 1.0, 3.0, // row 2: translate z by 3
            0.0, 0.0, 0.0, 1.0, // row 3
        ];

        let result = runtime.transform_points(&points, &transform).unwrap();

        assert_eq!(result.len(), 1);
        assert!((result[0][0] - 1.0).abs() < 1e-5);
        assert!((result[0][1] - 2.0).abs() < 1e-5);
        assert!((result[0][2] - 3.0).abs() < 1e-5);
    }
}
