//! GPU Zero-Copy Derivative Pipeline.
//!
//! This module provides a zero-copy GPU pipeline that keeps data on the GPU
//! between optimization iterations, minimizing CPU-GPU transfers.
//!
//! # Architecture
//!
//! ```text
//! Once per alignment:
//!   Upload: source_points [N×3], voxel_data [V×12]
//!
//! Per iteration:
//!   Upload: transform [16 floats]
//!   GPU: transform → radius_search → score → gradient → hessian → reduce
//!   Download: score [1] + gradient [6] + hessian [36] = 43 floats
//! ```
//!
//! With CUB GPU reduction, only 43 floats are downloaded per iteration instead
//! of N×43 floats (1000× reduction for N=1000 points).

use anyhow::{Context, Result};
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::gpu::{
    compute_ndt_gradient_kernel, compute_ndt_hessian_kernel, compute_ndt_score_kernel,
    compute_point_hessians_cpu, compute_point_jacobians_cpu, pose_to_transform_matrix,
    radius_search_kernel, GpuDerivativeResult, GpuVoxelData, MAX_NEIGHBORS,
};
use crate::voxel_grid::kernels::transform_points_kernel;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// Pre-allocated GPU buffers for derivative computation pipeline.
///
/// All buffers are owned by CubeCL's memory manager, ensuring proper
/// lifetime management. Data is uploaded once per alignment, and only
/// the pose transform is uploaded per iteration.
pub struct GpuDerivativePipeline {
    client: CudaClient,

    // Capacity tracking
    max_points: usize,
    max_voxels: usize,

    // Current sizes (set during upload_alignment_data)
    num_points: usize,
    num_voxels: usize,

    // Persistent data (uploaded once per alignment)
    source_points: Handle,  // [N × 3]
    voxel_means: Handle,    // [V × 3]
    voxel_inv_covs: Handle, // [V × 9]
    voxel_valid: Handle,    // [V]

    // Per-iteration buffers (reused)
    transform: Handle,          // [16]
    transformed_points: Handle, // [N × 3]
    neighbor_indices: Handle,   // [N × MAX_NEIGHBORS]
    neighbor_counts: Handle,    // [N]
    scores: Handle,             // [N]
    correspondences: Handle,    // [N]
    gradients: Handle,          // [N × 6]
    hessians: Handle,           // [N × 36]

    // Jacobians and point Hessians (uploaded once, depend on source points)
    jacobians: Handle, // [N × 18]
    #[allow(dead_code)]
    point_hessians: Handle, // [N × 144] - allocated but managed via jacobians_combined
    jacobians_combined: Handle, // [N × 18 + N × 144] for Hessian kernel

    // Gaussian parameters
    gauss_params: Handle, // [2] - d1, d2

    // Search radius squared
    search_radius_sq: f32,

    // CUB reduction buffers
    reduce_temp: Handle,      // Temporary storage for CUB
    reduce_temp_bytes: usize, // Size of temp storage
    reduce_offsets: Handle,   // Segment offsets [44] for 43 segments
    reduce_output: Handle,    // Reduction output [43] floats

    // CPU-side cache to avoid GPU downloads per iteration (P2 optimization)
    cached_source_points: Vec<[f32; 3]>, // Cached source points for Jacobian computation
    cached_gauss_d1: f32,                // Cached Gaussian d1 parameter
    cached_gauss_d2: f32,                // Cached Gaussian d2 parameter
}

impl GpuDerivativePipeline {
    /// Create a new derivative pipeline with given capacity.
    ///
    /// # Arguments
    /// * `max_points` - Maximum number of source points
    /// * `max_voxels` - Maximum number of voxels in target grid
    pub fn new(max_points: usize, max_voxels: usize) -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        // Allocate all buffers
        let source_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let voxel_means = client.empty(max_voxels * 3 * std::mem::size_of::<f32>());
        let voxel_inv_covs = client.empty(max_voxels * 9 * std::mem::size_of::<f32>());
        let voxel_valid = client.empty(max_voxels * std::mem::size_of::<u32>());

        let transform = client.empty(16 * std::mem::size_of::<f32>());
        let transformed_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let neighbor_indices =
            client.empty(max_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts = client.empty(max_points * std::mem::size_of::<u32>());
        let scores = client.empty(max_points * std::mem::size_of::<f32>());
        let correspondences = client.empty(max_points * std::mem::size_of::<u32>());
        let gradients = client.empty(max_points * 6 * std::mem::size_of::<f32>());
        let hessians = client.empty(max_points * 36 * std::mem::size_of::<f32>());

        let jacobians = client.empty(max_points * 18 * std::mem::size_of::<f32>());
        let point_hessians = client.empty(max_points * 144 * std::mem::size_of::<f32>());
        let jacobians_combined = client.empty(max_points * (18 + 144) * std::mem::size_of::<f32>());

        let gauss_params = client.empty(2 * std::mem::size_of::<f32>());

        // CUB reduction buffers
        // 43 segments: 1 score + 6 gradient + 36 hessian
        // Total items = num_points * 43 (but we reduce separately for simplicity)
        let num_segments = 43;
        let total_items = max_points * num_segments;
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
            num_points: 0,
            num_voxels: 0,
            source_points,
            voxel_means,
            voxel_inv_covs,
            voxel_valid,
            transform,
            transformed_points,
            neighbor_indices,
            neighbor_counts,
            scores,
            correspondences,
            gradients,
            hessians,
            jacobians,
            point_hessians,
            jacobians_combined,
            gauss_params,
            search_radius_sq: 4.0, // Default 2.0^2
            reduce_temp,
            reduce_temp_bytes,
            reduce_offsets,
            reduce_output,
            cached_source_points: Vec::new(),
            cached_gauss_d1: 0.0,
            cached_gauss_d2: 0.0,
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

    /// Upload alignment data (call once per align()).
    ///
    /// This uploads:
    /// - Source points
    /// - Voxel data (means, inverse covariances, validity flags)
    /// - Gaussian parameters
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud
    /// * `voxel_data` - Target voxel grid data
    /// * `gauss_d1` - Gaussian d1 parameter
    /// * `gauss_d2` - Gaussian d2 parameter
    /// * `search_radius` - Radius for voxel search
    pub fn upload_alignment_data(
        &mut self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<()> {
        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;

        if num_points > self.max_points {
            anyhow::bail!("Too many points: {} > max {}", num_points, self.max_points);
        }
        if num_voxels > self.max_voxels {
            anyhow::bail!("Too many voxels: {} > max {}", num_voxels, self.max_voxels);
        }

        self.num_points = num_points;
        self.num_voxels = num_voxels;
        self.search_radius_sq = search_radius * search_radius;

        // Cache source points on CPU for Jacobian computation (P2 optimization)
        // This avoids downloading source_points every iteration
        self.cached_source_points = source_points.to_vec();
        self.cached_gauss_d1 = gauss_d1;
        self.cached_gauss_d2 = gauss_d2;

        // Flatten source points
        let source_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();

        // Upload source points
        let source_bytes = f32::as_bytes(&source_flat);
        self.source_points = self.client.create(source_bytes);

        // Upload voxel data
        self.voxel_means = self.client.create(f32::as_bytes(&voxel_data.means));
        self.voxel_inv_covs = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        self.voxel_valid = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Upload Gaussian parameters
        let gauss_params = [gauss_d1, gauss_d2];
        self.gauss_params = self.client.create(f32::as_bytes(&gauss_params));

        Ok(())
    }

    /// Compute derivatives for one iteration.
    ///
    /// This method:
    /// 1. Uploads only the pose transform (16 floats)
    /// 2. Runs all GPU kernels
    /// 3. Downloads only the reduced results (43 floats)
    ///
    /// # Arguments
    /// * `pose` - Current pose [tx, ty, tz, roll, pitch, yaw]
    ///
    /// # Returns
    /// Aggregated derivative result (score, gradient, Hessian, correspondences)
    pub fn compute_iteration(&mut self, pose: &[f64; 6]) -> Result<GpuDerivativeResult> {
        if self.num_points == 0 {
            return Ok(GpuDerivativeResult {
                score: 0.0,
                gradient: [0.0; 6],
                hessian: [[0.0; 6]; 6],
                num_correspondences: 0,
            });
        }

        let num_points = self.num_points;
        let num_voxels = self.num_voxels;

        // Convert pose to transform matrix and upload
        let transform = pose_to_transform_matrix(pose);
        self.transform = self.client.create(f32::as_bytes(&transform));

        // Compute Jacobians and point Hessians on CPU using cached source points
        // (P2 optimization: avoids downloading source_points every iteration)
        let jacobians = compute_point_jacobians_cpu(&self.cached_source_points, pose);
        let point_hessians = compute_point_hessians_cpu(&self.cached_source_points, pose);

        // Upload Jacobians
        self.jacobians = self.client.create(f32::as_bytes(&jacobians));

        // Combine jacobians and point_hessians for Hessian kernel
        let mut jacobians_combined = jacobians.clone();
        jacobians_combined.extend_from_slice(&point_hessians);
        self.jacobians_combined = self.client.create(f32::as_bytes(&jacobians_combined));

        // Step 1: Transform points
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.transformed_points, num_points * 3, 1),
            );
        }

        // Step 2: Radius search
        unsafe {
            radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transformed_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&self.voxel_valid, num_voxels, 1),
                ScalarArg::new(self.search_radius_sq),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(num_voxels as u32),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
            );
        }

        // Step 3: Compute scores (using cached Gaussian parameters - P2 optimization)
        unsafe {
            compute_ndt_score_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                ScalarArg::new(self.cached_gauss_d1),
                ScalarArg::new(self.cached_gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.scores, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&self.correspondences, num_points, 1),
            );
        }

        // Step 4: Compute gradients
        unsafe {
            compute_ndt_gradient_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&self.jacobians, num_points * 18, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                ScalarArg::new(self.cached_gauss_d1),
                ScalarArg::new(self.cached_gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.gradients, num_points * 6, 1),
            );
        }

        // Step 5: Compute Hessians
        unsafe {
            compute_ndt_hessian_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.jacobians_combined,
                    num_points * 18 + num_points * 144,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                ArrayArg::from_raw_parts::<f32>(&self.gauss_params, 2, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.hessians, num_points * 36, 1),
            );
        }

        // Step 6: Download and reduce on CPU
        // TODO: Implement GPU reduction kernel to avoid downloading N×43 floats
        // For now, we do CPU reduction which still benefits from persistent GPU buffers
        let scores_bytes = self.client.read_one(self.scores.clone());
        let scores = f32::from_bytes(&scores_bytes);

        let correspondences_bytes = self.client.read_one(self.correspondences.clone());
        let correspondences = u32::from_bytes(&correspondences_bytes);

        let gradients_bytes = self.client.read_one(self.gradients.clone());
        let gradients = f32::from_bytes(&gradients_bytes);

        let hessians_bytes = self.client.read_one(self.hessians.clone());
        let hessians = f32::from_bytes(&hessians_bytes);

        // Reduce on CPU (column-major layout: component * num_points + point_idx)
        let total_score: f64 = scores[..num_points].iter().map(|&s| s as f64).sum();
        let total_correspondences: usize = correspondences[..num_points]
            .iter()
            .map(|&c| c as usize)
            .sum();

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

    /// Get raw CUDA device pointer from CubeCL handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let binding = handle.clone().binding();
        let resource = self.client.get_resource(binding);
        resource.resource().ptr
    }

    /// Compute a single optimization iteration with GPU reduction.
    ///
    /// This variant uses CUB DeviceSegmentedReduce to sum gradients and Hessians
    /// on the GPU, downloading only 43 floats instead of N×43 floats.
    ///
    /// # Arguments
    /// * `pose` - Current pose [x, y, z, roll, pitch, yaw]
    ///
    /// # Returns
    /// `GpuDerivativeResult` with score, gradient, and Hessian.
    pub fn compute_iteration_gpu_reduce(&mut self, pose: &[f64; 6]) -> Result<GpuDerivativeResult> {
        let num_points = self.num_points;
        let num_voxels = self.num_voxels;
        if num_points == 0 {
            return Ok(GpuDerivativeResult {
                score: 0.0,
                gradient: [0.0; 6],
                hessian: [[0.0; 6]; 6],
                num_correspondences: 0,
            });
        }

        // Step 1-5: Same as compute_iteration (kernels already produce column-major output)
        // Run all GPU kernels (transform, radius search, score, gradient, hessian)
        let transform = pose_to_transform_matrix(pose);
        self.transform = self.client.create(f32::as_bytes(&transform));

        // Compute Jacobians and point Hessians on CPU using cached source points
        // (P2 optimization: avoids downloading source_points every iteration)
        let jacobians = compute_point_jacobians_cpu(&self.cached_source_points, pose);
        let point_hessians = compute_point_hessians_cpu(&self.cached_source_points, pose);

        // Upload Jacobians
        self.jacobians = self.client.create(f32::as_bytes(&jacobians));

        // Combine jacobians and point_hessians for Hessian kernel
        let mut jacobians_combined = jacobians.clone();
        jacobians_combined.extend_from_slice(&point_hessians);
        self.jacobians_combined = self.client.create(f32::as_bytes(&jacobians_combined));

        let cube_count = num_points.div_ceil(256) as u32;

        // Transform points
        unsafe {
            transform_points_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.transformed_points, num_points * 3, 1),
            );
        }

        // Radius search
        unsafe {
            radius_search_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transformed_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&self.voxel_valid, num_voxels, 1),
                ScalarArg::new(self.search_radius_sq),
                ScalarArg::new(num_points as u32),
                ScalarArg::new(num_voxels as u32),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
            );
        }

        // Score kernel (using cached Gaussian parameters - P2 optimization)
        unsafe {
            compute_ndt_score_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                ScalarArg::new(self.cached_gauss_d1),
                ScalarArg::new(self.cached_gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.scores, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&self.correspondences, num_points, 1),
            );
        }

        // Gradient kernel
        unsafe {
            compute_ndt_gradient_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ArrayArg::from_raw_parts::<f32>(&self.jacobians, num_points * 18, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                ScalarArg::new(self.cached_gauss_d1),
                ScalarArg::new(self.cached_gauss_d2),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.gradients, num_points * 6, 1),
            );
        }

        // Hessian kernel
        unsafe {
            compute_ndt_hessian_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.jacobians_combined,
                    num_points * (18 + 144),
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                ArrayArg::from_raw_parts::<i32>(
                    &self.neighbor_indices,
                    num_points * MAX_NEIGHBORS as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                ArrayArg::from_raw_parts::<f32>(&self.gauss_params, 2, 1),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.hessians, num_points * 36, 1),
            );
        }

        // Step 6: GPU reduction using CUB DeviceSegmentedReduce
        // We do 3 separate reductions:
        // 1. scores [N] -> [1]
        // 2. gradients [6*N] -> [6]
        // 3. hessians [36*N] -> [36]

        // Sync CubeCL before cuda_ffi
        cubecl::future::block_on(self.client.sync());

        let n = num_points as i32;

        // Reduce scores (1 segment)
        let score_offsets: Vec<i32> = vec![0, n];
        self.reduce_offsets = self.client.create(i32::as_bytes(&score_offsets));
        cubecl::future::block_on(self.client.sync());

        unsafe {
            cuda_ffi::segmented_reduce_sum_f32_inplace(
                self.raw_ptr(&self.reduce_temp),
                self.reduce_temp_bytes,
                self.raw_ptr(&self.scores),
                self.raw_ptr(&self.reduce_output),
                1,
                self.raw_ptr(&self.reduce_offsets),
            )
            .context("CUB reduce scores failed")?;
        }

        let score_bytes = self.client.read_one(self.reduce_output.clone());
        let total_score = f32::from_le_bytes(score_bytes[..4].try_into().unwrap()) as f64;

        // Reduce correspondences on CPU (it's u32, not f32)
        let correspondences_bytes = self.client.read_one(self.correspondences.clone());
        let correspondences = u32::from_bytes(&correspondences_bytes);
        let total_correspondences: usize = correspondences[..num_points]
            .iter()
            .map(|&c| c as usize)
            .sum();

        // Reduce gradients (6 segments)
        let grad_offsets: Vec<i32> = (0..=6).map(|i| i * n).collect();
        self.reduce_offsets = self.client.create(i32::as_bytes(&grad_offsets));
        cubecl::future::block_on(self.client.sync());

        unsafe {
            cuda_ffi::segmented_reduce_sum_f32_inplace(
                self.raw_ptr(&self.reduce_temp),
                self.reduce_temp_bytes,
                self.raw_ptr(&self.gradients),
                self.raw_ptr(&self.reduce_output),
                6,
                self.raw_ptr(&self.reduce_offsets),
            )
            .context("CUB reduce gradients failed")?;
        }

        let grad_bytes = self.client.read_one(self.reduce_output.clone());
        let grad_floats = f32::from_bytes(&grad_bytes);
        let mut total_gradient = [0.0f64; 6];
        for i in 0..6 {
            total_gradient[i] = grad_floats[i] as f64;
        }

        // Reduce hessians (36 segments)
        let hess_offsets: Vec<i32> = (0..=36).map(|i| i * n).collect();
        self.reduce_offsets = self.client.create(i32::as_bytes(&hess_offsets));
        cubecl::future::block_on(self.client.sync());

        unsafe {
            cuda_ffi::segmented_reduce_sum_f32_inplace(
                self.raw_ptr(&self.reduce_temp),
                self.reduce_temp_bytes,
                self.raw_ptr(&self.hessians),
                self.raw_ptr(&self.reduce_output),
                36,
                self.raw_ptr(&self.reduce_offsets),
            )
            .context("CUB reduce hessians failed")?;
        }

        let hess_bytes = self.client.read_one(self.reduce_output.clone());
        let hess_floats = f32::from_bytes(&hess_bytes);
        let mut total_hessian = [[0.0f64; 6]; 6];
        for row in 0..6 {
            for col in 0..6 {
                total_hessian[row][col] = hess_floats[row * 6 + col] as f64;
            }
        }

        Ok(GpuDerivativeResult {
            score: total_score,
            gradient: total_gradient,
            hessian: total_hessian,
            num_correspondences: total_correspondences,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_grid::{VoxelGrid, VoxelGridConfig};

    fn make_test_points() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    }

    fn make_test_voxel_data() -> GpuVoxelData {
        // Create a simple voxel grid
        let config = VoxelGridConfig {
            resolution: 2.0,
            min_points_per_voxel: 1,
            ..Default::default()
        };
        let points = make_test_points();
        let grid = VoxelGrid::from_points_with_config(&points, config).unwrap();
        GpuVoxelData::from_voxel_grid(&grid)
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = GpuDerivativePipeline::new(1000, 500);
        assert!(pipeline.is_ok());
        let p = pipeline.unwrap();
        assert_eq!(p.max_points(), 1000);
        assert_eq!(p.max_voxels(), 500);
    }

    #[test]
    fn test_pipeline_upload_data() {
        let mut pipeline = GpuDerivativePipeline::new(1000, 500).unwrap();
        let points = make_test_points();
        let voxel_data = make_test_voxel_data();

        let result = pipeline.upload_alignment_data(&points, &voxel_data, -0.5, 1.0, 2.0);
        assert!(result.is_ok());
        assert_eq!(pipeline.num_points, 10);
    }

    #[test]
    fn test_pipeline_compute_iteration() {
        let mut pipeline = GpuDerivativePipeline::new(1000, 500).unwrap();
        let points = make_test_points();
        let voxel_data = make_test_voxel_data();

        pipeline
            .upload_alignment_data(&points, &voxel_data, -0.5, 1.0, 2.0)
            .unwrap();

        // Identity pose
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = pipeline.compute_iteration(&pose);
        assert!(result.is_ok());

        let r = result.unwrap();
        // With identity pose and points at origin, we should have correspondences
        assert!(r.num_correspondences > 0);
    }

    #[test]
    fn test_pipeline_multiple_iterations() {
        let mut pipeline = GpuDerivativePipeline::new(1000, 500).unwrap();
        let points = make_test_points();
        let voxel_data = make_test_voxel_data();

        pipeline
            .upload_alignment_data(&points, &voxel_data, -0.5, 1.0, 2.0)
            .unwrap();

        // Run multiple iterations with different poses
        let poses = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        ];

        for pose in &poses {
            let result = pipeline.compute_iteration(pose);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_pipeline_empty_input() {
        let mut pipeline = GpuDerivativePipeline::new(1000, 500).unwrap();
        let points: Vec<[f32; 3]> = vec![];
        let voxel_data = GpuVoxelData {
            means: vec![],
            inv_covariances: vec![],
            principal_axes: vec![],
            valid: vec![],
            num_voxels: 0,
        };

        pipeline
            .upload_alignment_data(&points, &voxel_data, -0.5, 1.0, 2.0)
            .unwrap();

        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = pipeline.compute_iteration(&pose).unwrap();
        assert_eq!(result.num_correspondences, 0);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_pipeline_gpu_reduce_vs_cpu_reduce() {
        let mut pipeline_cpu = GpuDerivativePipeline::new(1000, 500).unwrap();
        let mut pipeline_gpu = GpuDerivativePipeline::new(1000, 500).unwrap();
        let points = make_test_points();
        let voxel_data = make_test_voxel_data();

        pipeline_cpu
            .upload_alignment_data(&points, &voxel_data, -0.5, 1.0, 2.0)
            .unwrap();
        pipeline_gpu
            .upload_alignment_data(&points, &voxel_data, -0.5, 1.0, 2.0)
            .unwrap();

        // Test with multiple poses
        let poses = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.05],
        ];

        for pose in &poses {
            let cpu_result = pipeline_cpu.compute_iteration(pose).unwrap();
            let gpu_result = pipeline_gpu.compute_iteration_gpu_reduce(pose).unwrap();

            // Score should match closely
            let score_diff = (cpu_result.score - gpu_result.score).abs();
            assert!(
                score_diff < 0.001,
                "Score mismatch: CPU={}, GPU={}, diff={}",
                cpu_result.score,
                gpu_result.score,
                score_diff
            );

            // Correspondences should match exactly
            assert_eq!(
                cpu_result.num_correspondences, gpu_result.num_correspondences,
                "Correspondence count mismatch"
            );

            // Gradients should match
            for i in 0..6 {
                let grad_diff = (cpu_result.gradient[i] - gpu_result.gradient[i]).abs();
                let rel_diff = if cpu_result.gradient[i].abs() > 1e-6 {
                    grad_diff / cpu_result.gradient[i].abs()
                } else {
                    grad_diff
                };
                assert!(
                    rel_diff < 0.001,
                    "Gradient[{}] mismatch: CPU={}, GPU={}, rel_diff={}",
                    i,
                    cpu_result.gradient[i],
                    gpu_result.gradient[i],
                    rel_diff
                );
            }

            // Hessian diagonal should match
            for i in 0..6 {
                let hess_diff = (cpu_result.hessian[i][i] - gpu_result.hessian[i][i]).abs();
                let rel_diff = if cpu_result.hessian[i][i].abs() > 1e-6 {
                    hess_diff / cpu_result.hessian[i][i].abs()
                } else {
                    hess_diff
                };
                assert!(
                    rel_diff < 0.001,
                    "Hessian[{},{}] mismatch: CPU={}, GPU={}, rel_diff={}",
                    i,
                    i,
                    cpu_result.hessian[i][i],
                    gpu_result.hessian[i][i],
                    rel_diff
                );
            }
        }
    }

    #[test]
    fn test_pipeline_gpu_reduce_empty_input() {
        let mut pipeline = GpuDerivativePipeline::new(1000, 500).unwrap();
        let points: Vec<[f32; 3]> = vec![];
        let voxel_data = GpuVoxelData {
            means: vec![],
            inv_covariances: vec![],
            principal_axes: vec![],
            valid: vec![],
            num_voxels: 0,
        };

        pipeline
            .upload_alignment_data(&points, &voxel_data, -0.5, 1.0, 2.0)
            .unwrap();

        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = pipeline.compute_iteration_gpu_reduce(&pose).unwrap();
        assert_eq!(result.num_correspondences, 0);
        assert_eq!(result.score, 0.0);
    }
}
