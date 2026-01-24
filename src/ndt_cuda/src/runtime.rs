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
    compute_ndt_gradient_kernel, compute_ndt_gradient_point_to_plane_kernel,
    compute_ndt_hessian_kernel, compute_ndt_nvtl_kernel, compute_ndt_score_kernel,
    compute_ndt_score_point_to_plane_kernel, compute_point_hessians_cpu,
    compute_point_jacobians_cpu, pose_to_transform_matrix, radius_search_kernel, GpuVoxelData,
    MAX_NEIGHBORS,
};
use crate::derivatives::DistanceMetric;
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

    /// Compute NDT derivatives (score, gradient, Hessian) for optimization.
    ///
    /// This is the main function used during NDT alignment iterations.
    #[allow(clippy::too_many_arguments)]
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
                hessian: [[0.0; 6]; 6],
                num_correspondences: 0,
            });
        }

        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        // Convert pose to transform matrix
        let transform = pose_to_transform_matrix(pose);

        // Compute point Jacobians and Hessians on CPU (small overhead, complex computation)
        let jacobians = compute_point_jacobians_cpu(source_points, pose);
        let point_hessians = compute_point_hessians_cpu(source_points, pose);

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload data to GPU
        let source_gpu = self.client.create(f32::as_bytes(&source_flat));
        let transform_gpu = self.client.create(f32::as_bytes(&transform));
        let jacobians_gpu = self.client.create(f32::as_bytes(&jacobians));
        // Note: point_hessians is combined with jacobians later for Hessian kernel
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

        // Step 5: Compute Hessians
        let hessians_gpu = self
            .client
            .empty(num_points * 36 * std::mem::size_of::<f32>());

        // Combine jacobians and point_hessians into single buffer for kernel
        // (CubeCL has parameter count limits)
        let mut jacobians_and_hessians = jacobians.clone();
        jacobians_and_hessians.extend_from_slice(&point_hessians);
        let jacobians_and_hessians_gpu = self.client.create(f32::as_bytes(&jacobians_and_hessians));

        let gauss_params = [gauss_d1, gauss_d2];
        let gauss_params_gpu = self.client.create(f32::as_bytes(&gauss_params));

        unsafe {
            compute_ndt_hessian_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &jacobians_and_hessians_gpu,
                    num_points * 18 + num_points * 144,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &neighbor_indices_gpu,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                ArrayArg::from_raw_parts::<f32>(&gauss_params_gpu, 2, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&hessians_gpu, num_points * 36, 1),
            );
        }

        // Read results back
        let scores_bytes = self.client.read_one(scores_gpu);
        let scores = f32::from_bytes(&scores_bytes);

        let correspondences_bytes = self.client.read_one(correspondences_gpu);
        let correspondences = u32::from_bytes(&correspondences_bytes);

        let gradients_bytes = self.client.read_one(gradients_gpu);
        let gradients = f32::from_bytes(&gradients_bytes);

        let hessians_bytes = self.client.read_one(hessians_gpu);
        let hessians = f32::from_bytes(&hessians_bytes);

        // Reduce on CPU (small reduction, not worth GPU overhead)
        let total_score: f64 = scores.iter().map(|&s| s as f64).sum();
        let total_correspondences: usize = correspondences.iter().map(|&c| c as usize).sum();

        // Reduce on CPU (column-major layout: component * num_points + point_idx)
        let mut total_gradient = [0.0f64; 6];
        for j in 0..6 {
            for i in 0..num_points {
                total_gradient[j] += gradients[j * num_points + i] as f64;
            }
        }

        let mut total_hessian = [[0.0f64; 6]; 6];
        for (row, row_arr) in total_hessian.iter_mut().enumerate() {
            for (col, cell) in row_arr.iter_mut().enumerate() {
                let component = row * 6 + col;
                for i in 0..num_points {
                    *cell += hessians[component * num_points + i] as f64;
                }
            }
        }

        Ok(GpuDerivativeResult {
            score: total_score,
            gradient: total_gradient,
            hessian: total_hessian,
            num_correspondences: total_correspondences,
        })
    }

    /// Compute NDT derivatives with selectable distance metric.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud
    /// * `voxel_data` - Target voxel grid data
    /// * `pose` - Current pose [tx, ty, tz, roll, pitch, yaw]
    /// * `gauss_d1`, `gauss_d2` - Gaussian parameters
    /// * `search_radius` - Radius for voxel search
    /// * `metric` - Distance metric (PointToDistribution or PointToPlane)
    #[allow(clippy::too_many_arguments)]
    pub fn compute_derivatives_with_metric(
        &self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        pose: &[f64; 6],
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
        metric: DistanceMetric,
    ) -> Result<GpuDerivativeResult> {
        if source_points.is_empty() {
            return Ok(GpuDerivativeResult {
                score: 0.0,
                gradient: [0.0; 6],
                hessian: [[0.0; 6]; 6],
                num_correspondences: 0,
            });
        }

        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        // Convert pose to transform matrix
        let transform = pose_to_transform_matrix(pose);

        // Compute point Jacobians and Hessians on CPU (small overhead, complex computation)
        let jacobians = compute_point_jacobians_cpu(source_points, pose);
        let point_hessians = compute_point_hessians_cpu(source_points, pose);

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload data to GPU
        let source_gpu = self.client.create(f32::as_bytes(&source_flat));
        let transform_gpu = self.client.create(f32::as_bytes(&transform));
        let jacobians_gpu = self.client.create(f32::as_bytes(&jacobians));
        // Note: point_hessians is combined with jacobians later for Hessian kernel
        let voxel_means_gpu = self.client.create(f32::as_bytes(&voxel_data.means));
        let voxel_inv_covs_gpu = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        let voxel_principal_axes_gpu = self
            .client
            .create(f32::as_bytes(&voxel_data.principal_axes));
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

        // Step 3: Compute scores and gradients using the selected metric
        let scores_gpu = self.client.empty(num_points * std::mem::size_of::<f32>());
        let correspondences_gpu = self.client.empty(num_points * std::mem::size_of::<u32>());
        let gradients_gpu = self
            .client
            .empty(num_points * 6 * std::mem::size_of::<f32>());

        match metric {
            DistanceMetric::PointToDistribution => {
                // Use full Mahalanobis distance (original kernels)
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
            }
            DistanceMetric::PointToPlane => {
                // Use simplified point-to-plane distance
                unsafe {
                    compute_ndt_score_point_to_plane_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(cube_count, 1, 1),
                        CubeDim::new(256, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                        ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                        ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &voxel_principal_axes_gpu,
                            num_voxels * 3,
                            1,
                        ),
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

                    compute_ndt_gradient_point_to_plane_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(cube_count, 1, 1),
                        CubeDim::new(256, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                        ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                        ArrayArg::from_raw_parts::<f32>(&jacobians_gpu, num_points * 18, 1),
                        ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &voxel_principal_axes_gpu,
                            num_voxels * 3,
                            1,
                        ),
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
            }
        }

        // Step 4: Compute Hessians (only for PointToDistribution metric)
        let hessians_gpu = self
            .client
            .empty(num_points * 36 * std::mem::size_of::<f32>());

        if metric == DistanceMetric::PointToDistribution {
            // Combine jacobians and point_hessians into single buffer for kernel
            // (CubeCL has parameter count limits)
            let mut jacobians_and_hessians = jacobians.clone();
            jacobians_and_hessians.extend_from_slice(&point_hessians);
            let jacobians_and_hessians_gpu =
                self.client.create(f32::as_bytes(&jacobians_and_hessians));

            let gauss_params = [gauss_d1, gauss_d2];
            let gauss_params_gpu = self.client.create(f32::as_bytes(&gauss_params));

            unsafe {
                compute_ndt_hessian_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&source_gpu, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&transform_gpu, 16, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &jacobians_and_hessians_gpu,
                        num_points * 18 + num_points * 144,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<f32>(&voxel_means_gpu, num_voxels * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&voxel_inv_covs_gpu, num_voxels * 9, 1),
                    ArrayArg::from_raw_parts::<i32>(
                        &neighbor_indices_gpu,
                        num_points * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&neighbor_counts_gpu, num_points, 1),
                    ArrayArg::from_raw_parts::<f32>(&gauss_params_gpu, 2, 1),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&hessians_gpu, num_points * 36, 1),
                );
            }
        }

        // Read results back
        let scores_bytes = self.client.read_one(scores_gpu);
        let scores = f32::from_bytes(&scores_bytes);

        let correspondences_bytes = self.client.read_one(correspondences_gpu);
        let correspondences = u32::from_bytes(&correspondences_bytes);

        let gradients_bytes = self.client.read_one(gradients_gpu);
        let gradients = f32::from_bytes(&gradients_bytes);

        let hessians_bytes = self.client.read_one(hessians_gpu);
        let hessians = f32::from_bytes(&hessians_bytes);

        // Reduce on CPU (column-major layout: component * num_points + point_idx)
        let total_score: f64 = scores.iter().map(|&s| s as f64).sum();
        let total_correspondences: usize = correspondences.iter().map(|&c| c as usize).sum();

        let mut total_gradient = [0.0f64; 6];
        for j in 0..6 {
            for i in 0..num_points {
                total_gradient[j] += gradients[j * num_points + i] as f64;
            }
        }

        let mut total_hessian = [[0.0f64; 6]; 6];
        if metric == DistanceMetric::PointToDistribution {
            for (row, row_arr) in total_hessian.iter_mut().enumerate() {
                for (col, cell) in row_arr.iter_mut().enumerate() {
                    let component = row * 6 + col;
                    for i in 0..num_points {
                        *cell += hessians[component * num_points + i] as f64;
                    }
                }
            }
        }

        Ok(GpuDerivativeResult {
            score: total_score,
            gradient: total_gradient,
            hessian: total_hessian,
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
    /// Total Hessian (6x6 matrix, row-major)
    pub hessian: [[f64; 6]; 6],
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

    /// Skip test at runtime if CUDA is not available.
    /// This allows GPU tests to run on machines with CUDA while
    /// gracefully skipping on machines without GPU.
    macro_rules! require_cuda {
        () => {
            if !is_cuda_available() {
                crate::test_println!("Skipping test: CUDA not available");
                return;
            }
        };
    }
    #[test]
    fn test_cuda_availability() {
        let _available = is_cuda_available();
        crate::test_println!("CUDA available: {_available}");
    }
    #[test]
    fn test_transform_points_gpu() {
        require_cuda!();

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
    fn test_transform_points_translation() {
        require_cuda!();

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
    #[test]
    fn test_compute_scores_gpu() {
        require_cuda!();

        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");

        // Create simple voxel data: one voxel at origin
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0], // One voxel at origin
            inv_covariances: vec![
                1.0, 0.0, 0.0, // Identity inverse covariance
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
            ],
            principal_axes: vec![0.0, 0.0, 1.0], // Z-axis as principal axis
            valid: vec![1],                      // One valid voxel
            num_voxels: 1,
        };

        // Source point at origin (should have high score)
        let source_points = vec![[0.0, 0.0, 0.0]];

        // Identity transform
        let transform = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];

        let result = runtime
            .compute_scores(
                &source_points,
                &voxel_data,
                &transform,
                1.0,  // gauss_d1
                1.0,  // gauss_d2
                10.0, // search_radius (large enough to find the voxel)
            )
            .expect("compute_scores failed");

        crate::test_println!("GPU score result: {:?}", result);

        // Should have one correspondence
        assert_eq!(result.total_correspondences, 1);
        // Score should be negative (NDT score formula: -d1 * exp(...))
        assert!(result.total_score < 0.0);
    }
    #[test]
    fn test_compute_nvtl_scores_gpu() {
        require_cuda!();

        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");

        // Create two voxels at different positions
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin
                1.0, 0.0, 0.0, // Voxel 1 at (1, 0, 0)
            ],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 1
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0: Z-axis
                0.0, 0.0, 1.0, // Voxel 1: Z-axis
            ],
            valid: vec![1, 1],
            num_voxels: 2,
        };

        // Source point between the two voxels
        let source_points = vec![[0.5, 0.0, 0.0]];

        // Identity transform
        let transform = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];

        let result = runtime
            .compute_nvtl_scores(
                &source_points,
                &voxel_data,
                &transform,
                1.0,  // gauss_d1
                1.0,  // gauss_d2
                10.0, // search_radius
            )
            .expect("compute_nvtl_scores failed");

        crate::test_println!("GPU NVTL result: {:?}", result);

        // Should have one point with neighbors
        assert_eq!(result.num_with_neighbors, 1);
        // NVTL should be negative (max of negative scores)
        assert!(result.nvtl < 0.0);
        // max_scores should have one entry
        assert_eq!(result.max_scores.len(), 1);
    }
    #[test]
    fn test_cpu_vs_gpu_derivatives() {
        require_cuda!();

        use crate::derivatives::{compute_derivatives_cpu, GpuVoxelData};
        use crate::test_utils::make_half_cubic_pcd;
        use crate::voxel_grid::VoxelGrid;

        // Create a simple point cloud for both target and source
        let target_points = make_half_cubic_pcd(10.0, 0.5); // 10m sides, 0.5m spacing
        let resolution = 2.0;

        // Build CPU voxel grid
        let target_grid =
            VoxelGrid::from_points(&target_points, resolution).expect("Failed to build voxel grid");

        // Convert to GPU format
        let gpu_voxel_data = GpuVoxelData::from_voxel_grid(&target_grid);

        // Create source points (subset of target, slightly offset)
        let source_points: Vec<[f32; 3]> = target_points
            .iter()
            .step_by(10)
            .take(50) // Take 50 points
            .map(|p| [p[0] + 0.1, p[1] + 0.1, p[2]]) // Small offset
            .collect();

        // Test pose: small translation and rotation
        let pose = [0.05, 0.05, 0.0, 0.01, 0.01, 0.02]; // tx, ty, tz, roll, pitch, yaw

        // Gaussian parameters matching Autoware defaults
        // GaussianParams::new(resolution, outlier_ratio)
        let gauss = crate::GaussianParams::new(resolution as f64, 0.55);
        let gauss_d1 = gauss.d1 as f32;
        let gauss_d2 = gauss.d2 as f32;

        // Compute CPU derivatives
        let cpu_result = compute_derivatives_cpu(&source_points, &target_grid, &pose, &gauss, true);

        // Compute GPU derivatives
        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");
        let gpu_result = runtime
            .compute_derivatives(
                &source_points,
                &gpu_voxel_data,
                &pose,
                gauss_d1,
                gauss_d2,
                resolution,
            )
            .expect("GPU derivatives failed");

        crate::test_println!("=== CPU vs GPU Derivative Comparison ===");
        crate::test_println!("Source points: {}", source_points.len());
        crate::test_println!("CPU correspondences: {}", cpu_result.num_correspondences);
        crate::test_println!("GPU correspondences: {}", gpu_result.num_correspondences);

        crate::test_println!("\n--- Score ---");
        crate::test_println!("CPU score: {:.6}", cpu_result.score);
        crate::test_println!("GPU score: {:.6}", gpu_result.score);
        let score_diff = (cpu_result.score - gpu_result.score).abs();
        crate::test_println!("Score diff: {:.6}", score_diff);

        crate::test_println!("\n--- Gradient ---");
        for i in 0..6 {
            let cpu_g = cpu_result.gradient[i];
            let gpu_g = gpu_result.gradient[i];
            let _diff = (cpu_g - gpu_g).abs();
            crate::test_println!(
                "  g[{i}]: CPU={:12.4}, GPU={:12.4}, diff={:12.4}",
                cpu_g,
                gpu_g,
                _diff
            );
        }

        crate::test_println!("\n--- Hessian Diagonal ---");
        for i in 0..6 {
            let cpu_h = cpu_result.hessian[(i, i)];
            let gpu_h = gpu_result.hessian[i][i];
            let _diff = (cpu_h - gpu_h).abs();
            let _sign_match = (cpu_h * gpu_h) > 0.0;
            crate::test_println!(
                "  h[{i},{i}]: CPU={:12.2}, GPU={:12.2}, diff={:12.2}, sign_match={}",
                cpu_h,
                gpu_h,
                _diff,
                _sign_match
            );
        }

        crate::test_println!("\n--- Full Hessian (upper triangle) ---");
        for i in 0..6 {
            for j in i..6 {
                let cpu_h = cpu_result.hessian[(i, j)];
                let gpu_h = gpu_result.hessian[i][j];
                let diff = (cpu_h - gpu_h).abs();
                if diff > 1.0 {
                    crate::test_println!(
                        "  h[{i},{j}]: CPU={:12.2}, GPU={:12.2}, diff={:12.2}",
                        cpu_h,
                        gpu_h,
                        diff
                    );
                }
            }
        }

        // Check that scores match (tolerance for f32/f64 differences)
        let score_rel_diff = if cpu_result.score.abs() > 1e-10 {
            score_diff / cpu_result.score.abs()
        } else {
            score_diff
        };
        assert!(
            score_rel_diff < 0.05,
            "Score relative difference too large: {score_rel_diff}"
        );

        // Check gradient sign consistency
        for i in 0..6 {
            let cpu_g = cpu_result.gradient[i];
            let gpu_g = gpu_result.gradient[i];
            if cpu_g.abs() > 1.0 && gpu_g.abs() > 1.0 {
                assert!(
                    (cpu_g * gpu_g) > 0.0,
                    "Gradient sign mismatch at {i}: CPU={cpu_g}, GPU={gpu_g}"
                );
            }
        }

        // Check Hessian diagonal sign consistency (main bug we're looking for)
        let mut sign_mismatches = Vec::new();
        for i in 0..6 {
            let cpu_h = cpu_result.hessian[(i, i)];
            let gpu_h = gpu_result.hessian[i][i];
            // Only check if values are significant
            if cpu_h.abs() > 10.0 && gpu_h.abs() > 10.0 && (cpu_h * gpu_h) < 0.0 {
                sign_mismatches.push((i, cpu_h, gpu_h));
            }
        }

        if !sign_mismatches.is_empty() {
            crate::test_println!("\n!!! HESSIAN DIAGONAL SIGN MISMATCHES !!!");
            for (_i, _cpu_h, _gpu_h) in &sign_mismatches {
                crate::test_println!(
                    "  h[{_i},{_i}]: CPU={_cpu_h:.2} ({}), GPU={_gpu_h:.2} ({})",
                    if *_cpu_h < 0.0 { "neg" } else { "pos" },
                    if *_gpu_h < 0.0 { "neg" } else { "pos" }
                );
            }
            panic!(
                "Found {} Hessian diagonal sign mismatches - GPU likely has a sign bug",
                sign_mismatches.len()
            );
        }
    }
    #[test]
    fn test_cpu_vs_gpu_single_point_single_voxel() {
        require_cuda!();

        use crate::derivatives::{compute_derivatives_cpu, GpuVoxelData};
        use crate::voxel_grid::VoxelGrid;

        // Create a minimal test case: points clustered at origin
        let target_points: Vec<[f32; 3]> = (0..20)
            .flat_map(|i| (0..20).map(move |j| [0.1 * i as f32, 0.1 * j as f32, 0.0]))
            .collect();
        let resolution = 2.0;

        let target_grid =
            VoxelGrid::from_points(&target_points, resolution).expect("Failed to build voxel grid");
        let gpu_voxel_data = GpuVoxelData::from_voxel_grid(&target_grid);

        crate::test_println!("Target grid has {} voxels", target_grid.len());

        // Single source point near the voxel mean
        let source_points = vec![[0.5, 0.5, 0.0]];

        // Identity pose (no transformation)
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let gauss = crate::GaussianParams::new(resolution as f64, 0.55);
        let gauss_d1 = gauss.d1 as f32;
        let gauss_d2 = gauss.d2 as f32;

        crate::test_println!("Gaussian params: d1={:.4}, d2={:.4}", gauss.d1, gauss.d2);

        // CPU
        let cpu_result = compute_derivatives_cpu(&source_points, &target_grid, &pose, &gauss, true);

        // GPU
        let runtime = GpuRuntime::new().expect("Failed to create GPU runtime");
        let gpu_result = runtime
            .compute_derivatives(
                &source_points,
                &gpu_voxel_data,
                &pose,
                gauss_d1,
                gauss_d2,
                resolution,
            )
            .expect("GPU derivatives failed");

        crate::test_println!("\n=== Single Point Test ===");
        crate::test_println!(
            "CPU score: {:.6}, GPU score: {:.6}",
            cpu_result.score,
            gpu_result.score
        );
        crate::test_println!(
            "CPU correspondences: {}, GPU correspondences: {}",
            cpu_result.num_correspondences,
            gpu_result.num_correspondences
        );

        crate::test_println!("\nGradient comparison:");
        for _i in 0..6 {
            crate::test_println!(
                "  g[{_i}]: CPU={:12.6}, GPU={:12.6}",
                cpu_result.gradient[_i],
                gpu_result.gradient[_i]
            );
        }

        crate::test_println!("\nHessian diagonal:");
        for _i in 0..6 {
            let _cpu_h = cpu_result.hessian[(_i, _i)];
            let _gpu_h = gpu_result.hessian[_i][_i];
            crate::test_println!("  h[{_i},{_i}]: CPU={:12.4}, GPU={:12.4}", _cpu_h, _gpu_h);
        }

        // In this simple case, values should match closely
        for i in 0..6 {
            let cpu_h = cpu_result.hessian[(i, i)];
            let gpu_h = gpu_result.hessian[i][i];
            if cpu_h.abs() > 0.1 && gpu_h.abs() > 0.1 {
                assert!(
                    (cpu_h * gpu_h) > 0.0,
                    "Hessian diagonal sign mismatch at [{i},{i}]: CPU={cpu_h:.4}, GPU={gpu_h:.4}"
                );
            }
        }
    }
}
