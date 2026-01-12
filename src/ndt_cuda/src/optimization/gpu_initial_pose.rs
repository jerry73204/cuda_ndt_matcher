//! GPU-accelerated initial pose estimation pipeline.
//!
//! This module provides batched GPU processing for the startup phase of
//! initial pose estimation, where N random particles can be evaluated in parallel.
//!
//! # Architecture
//!
//! ```text
//! Startup Phase (K particles, fully parallel):
//! ┌──────────────────────────────────────────────────────────────┐
//! │  CPU: Sample K random particles around initial guess         │
//! │       Upload K poses [K×6 floats = 48 bytes]                 │
//! └──────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌──────────────────────────────────────────────────────────────┐
//! │  GPU Batch Transform: K×N transformed points                 │
//! │       Grid: (ceil(N/256), K)                                 │
//! └──────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌──────────────────────────────────────────────────────────────┐
//! │  GPU Batch Radius Search: K×N×8 neighbor indices             │
//! └──────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌──────────────────────────────────────────────────────────────┐
//! │  GPU Batch NDT: K iterations of Newton optimization          │
//! │       (score, gradient, Hessian, solve) per iteration        │
//! └──────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌──────────────────────────────────────────────────────────────┐
//! │  Download: K final scores [K×1 float = 4K bytes]             │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Expected Speedup
//!
//! With K=8 particles and max_iterations=30:
//! - Current (sequential): K × (16ms NDT align) = 128ms
//! - GPU batch: ~25ms (single GPU pass for all particles)
//! - Speedup: ~5-6x for startup phase

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::derivatives::gpu::{GpuVoxelData, MAX_NEIGHBORS};
use crate::derivatives::gpu_batch::{
    check_convergence_batch_kernel, compute_jacobians_batch_kernel,
    compute_ndt_gradient_batch_kernel, compute_ndt_hessian_batch_kernel,
    compute_ndt_score_batch_kernel, radius_search_batch_kernel, update_poses_batch_kernel,
};
use crate::voxel_grid::kernels::{
    compute_sin_cos_batch_kernel, compute_transforms_batch_kernel, transform_points_batch_kernel,
};
use crate::voxel_grid::VoxelGrid;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// Configuration for GPU initial pose pipeline.
#[derive(Debug, Clone)]
pub struct GpuInitialPoseConfig {
    /// Maximum number of source points (N).
    pub max_points: usize,
    /// Maximum number of voxels (V).
    pub max_voxels: usize,
    /// Maximum batch size (K particles).
    pub max_batch_size: usize,
    /// Maximum Newton iterations per particle.
    pub max_iterations: u32,
    /// Convergence threshold (squared).
    pub epsilon_sq: f64,
    /// Gaussian d1 parameter.
    pub gauss_d1: f64,
    /// Gaussian d2 parameter.
    pub gauss_d2: f64,
    /// Search radius (typically = resolution).
    pub search_radius: f32,
}

impl Default for GpuInitialPoseConfig {
    fn default() -> Self {
        Self {
            max_points: 1000,
            max_voxels: 20000,
            max_batch_size: 8,
            max_iterations: 30,
            epsilon_sq: 0.01 * 0.01, // 1cm threshold
            gauss_d1: 1.0,
            gauss_d2: 1.0,
            search_radius: 2.0,
        }
    }
}

/// Result from a single particle in the batch.
#[derive(Debug, Clone)]
pub struct BatchedNdtResult {
    /// Final optimized pose [tx, ty, tz, roll, pitch, yaw].
    pub pose: [f64; 6],
    /// Final NDT score (more negative = better).
    pub score: f64,
    /// Number of iterations taken.
    pub iterations: u32,
    /// Whether the particle converged.
    pub converged: bool,
}

/// GPU-accelerated batch initial pose pipeline.
///
/// Evaluates K particles simultaneously on GPU for the startup phase
/// of initial pose estimation.
///
/// # Example
///
/// ```ignore
/// let mut pipeline = GpuInitialPosePipeline::new(config)?;
///
/// // Upload source points and voxel data once
/// pipeline.upload_data(&source_points, &voxel_grid)?;
///
/// // Batch evaluate K particles
/// let results = pipeline.evaluate_batch(&particles)?;
///
/// // Find best particle
/// let best = results.iter().min_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
/// ```
pub struct GpuInitialPosePipeline {
    config: GpuInitialPoseConfig,

    // GPU device and client
    client: CudaClient,
    #[allow(dead_code)]
    device: CudaDevice,

    // Current data sizes
    num_points: usize,
    num_voxels: usize,

    // ========================================================================
    // Persistent data (uploaded once per alignment)
    // ========================================================================
    source_points: Handle,  // [N × 3]
    voxel_means: Handle,    // [V × 3]
    voxel_inv_covs: Handle, // [V × 9]
    voxel_valid: Handle,    // [V]

    // ========================================================================
    // Batch state buffers
    // ========================================================================
    poses_gpu: Handle,        // [K × 6] current poses
    sin_cos: Handle,          // [K × 6] sin/cos of angles
    transforms: Handle,       // [K × 16] transform matrices
    jacobians: Handle,        // [K × N × 18]
    transformed: Handle,      // [K × N × 3]
    neighbor_indices: Handle, // [K × N × MAX_NEIGHBORS]
    neighbor_counts: Handle,  // [K × N]

    // ========================================================================
    // Derivative buffers
    // ========================================================================
    scores: Handle,          // [K × N]
    correspondences: Handle, // [K × N]
    gradients: Handle,       // [K × 6 × N] column-major
    hessians: Handle,        // [K × 36 × N] column-major

    // ========================================================================
    // Reduction output buffers
    // ========================================================================
    score_sums: Handle,    // [K]
    gradient_sums: Handle, // [K × 6]
    hessian_sums: Handle,  // [K × 36]

    // ========================================================================
    // Newton solve buffers
    // ========================================================================
    deltas: Handle,          // [K × 6] Newton steps
    alphas: Handle,          // [K] step sizes (1.0 for no line search)
    converged_flags: Handle, // [K] u32

    // ========================================================================
    // Parameter buffers
    // ========================================================================
    params: Handle, // [4] = [gauss_d1, gauss_d2, num_points, batch_size]

    // ========================================================================
    // CUB reduction buffers
    // ========================================================================
    reduce_temp: Handle,
    reduce_temp_bytes: usize,
    score_offsets: Handle, // [K+1]
    grad_offsets: Handle,  // [K*6 + 1]
    hess_offsets: Handle,  // [K*36 + 1]

    // ========================================================================
    // Batched Cholesky solver (planned for future GPU solver integration)
    // ========================================================================
    #[allow(dead_code)]
    batched_solver: cuda_ffi::BatchedCholeskySolver,
}

impl GpuInitialPosePipeline {
    /// Get raw CUDA device pointer from CubeCL handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let binding = handle.clone().binding();
        let resource = self.client.get_resource(binding);
        resource.resource().ptr
    }

    /// Create a new GPU initial pose pipeline.
    pub fn new(config: GpuInitialPoseConfig) -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        let n = config.max_points;
        let v = config.max_voxels;
        let k = config.max_batch_size;

        // Persistent data buffers
        let source_points = client.empty(n * 3 * std::mem::size_of::<f32>());
        let voxel_means = client.empty(v * 3 * std::mem::size_of::<f32>());
        let voxel_inv_covs = client.empty(v * 9 * std::mem::size_of::<f32>());
        let voxel_valid = client.empty(v * std::mem::size_of::<u32>());

        // Batch state buffers
        let poses_gpu = client.empty(k * 6 * std::mem::size_of::<f32>());
        let sin_cos = client.empty(k * 6 * std::mem::size_of::<f32>());
        let transforms = client.empty(k * 16 * std::mem::size_of::<f32>());
        let jacobians = client.empty(k * n * 18 * std::mem::size_of::<f32>());
        let transformed = client.empty(k * n * 3 * std::mem::size_of::<f32>());
        let neighbor_indices =
            client.empty(k * n * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts = client.empty(k * n * std::mem::size_of::<u32>());

        // Derivative buffers
        let scores = client.empty(k * n * std::mem::size_of::<f32>());
        let correspondences = client.empty(k * n * std::mem::size_of::<u32>());
        let gradients = client.empty(k * 6 * n * std::mem::size_of::<f32>());
        let hessians = client.empty(k * 36 * n * std::mem::size_of::<f32>());

        // Reduction output buffers
        let score_sums = client.empty(k * std::mem::size_of::<f32>());
        let gradient_sums = client.empty(k * 6 * std::mem::size_of::<f32>());
        let hessian_sums = client.empty(k * 36 * std::mem::size_of::<f32>());

        // Newton solve buffers
        let deltas = client.empty(k * 6 * std::mem::size_of::<f32>());
        let alphas_data = vec![1.0f32; k];
        let alphas = client.create(f32::as_bytes(&alphas_data));
        let converged_flags = client.empty(k * std::mem::size_of::<u32>());

        // Parameter buffer
        let params = client.empty(4 * std::mem::size_of::<f32>());

        // CUB reduction buffers
        // Calculate temp storage size for the largest reduction (hessians: K*36 segments)
        let max_elements = k * 36 * n;
        let max_segments = k * 36;
        let reduce_temp_bytes =
            cuda_ffi::segmented_reduce_sum_f32_temp_size(max_elements, max_segments)? as usize;
        let reduce_temp = client.empty(reduce_temp_bytes.max(256));

        // Score offsets: [0, N, 2N, ..., K*N]
        let score_offsets_data: Vec<i32> = (0..=k).map(|i| (i * n) as i32).collect();
        let score_offsets = client.create(i32::as_bytes(&score_offsets_data));

        // Gradient offsets: [0, N, 2N, ..., K*6*N]
        let grad_offsets_data: Vec<i32> = (0..=k * 6).map(|i| (i * n) as i32).collect();
        let grad_offsets = client.create(i32::as_bytes(&grad_offsets_data));

        // Hessian offsets: [0, N, 2N, ..., K*36*N]
        let hess_offsets_data: Vec<i32> = (0..=k * 36).map(|i| (i * n) as i32).collect();
        let hess_offsets = client.create(i32::as_bytes(&hess_offsets_data));

        // Batched Cholesky solver
        let batched_solver = cuda_ffi::BatchedCholeskySolver::new()?;

        Ok(Self {
            config,
            client,
            device,
            num_points: 0,
            num_voxels: 0,
            source_points,
            voxel_means,
            voxel_inv_covs,
            voxel_valid,
            poses_gpu,
            sin_cos,
            transforms,
            jacobians,
            transformed,
            neighbor_indices,
            neighbor_counts,
            scores,
            correspondences,
            gradients,
            hessians,
            score_sums,
            gradient_sums,
            hessian_sums,
            deltas,
            alphas,
            converged_flags,
            params,
            reduce_temp,
            reduce_temp_bytes,
            score_offsets,
            grad_offsets,
            hess_offsets,
            batched_solver,
        })
    }

    /// Create pipeline with default configuration.
    pub fn with_defaults(max_points: usize, max_voxels: usize, batch_size: usize) -> Result<Self> {
        let config = GpuInitialPoseConfig {
            max_points,
            max_voxels,
            max_batch_size: batch_size,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Get the configuration.
    pub fn config(&self) -> &GpuInitialPoseConfig {
        &self.config
    }

    /// Upload source points and voxel grid data to GPU.
    ///
    /// This should be called once per scan, before evaluating batches.
    pub fn upload_data(
        &mut self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
    ) -> Result<()> {
        let num_points = source_points.len();
        let num_voxels = voxel_data.num_voxels;
        let k = self.config.max_batch_size;

        if num_points > self.config.max_points {
            anyhow::bail!(
                "Too many source points: {num_points} > {}",
                self.config.max_points
            );
        }
        if num_voxels > self.config.max_voxels {
            anyhow::bail!("Too many voxels: {num_voxels} > {}", self.config.max_voxels);
        }

        self.num_points = num_points;
        self.num_voxels = num_voxels;

        // Upload source points
        let points_flat: Vec<f32> = source_points
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();
        self.source_points = self.client.create(f32::as_bytes(&points_flat));

        // Upload voxel data
        self.voxel_means = self.client.create(f32::as_bytes(&voxel_data.means));
        self.voxel_inv_covs = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));
        self.voxel_valid = self.client.create(u32::as_bytes(&voxel_data.valid));

        // Update parameter buffer
        let params_data = [
            self.config.gauss_d1 as f32,
            self.config.gauss_d2 as f32,
            num_points as f32,
            k as f32,
        ];
        self.params = self.client.create(f32::as_bytes(&params_data));

        // Update CUB offsets for actual num_points
        let n = num_points;
        let score_offsets_data: Vec<i32> = (0..=k).map(|i| (i * n) as i32).collect();
        self.score_offsets = self.client.create(i32::as_bytes(&score_offsets_data));

        let grad_offsets_data: Vec<i32> = (0..=k * 6).map(|i| (i * n) as i32).collect();
        self.grad_offsets = self.client.create(i32::as_bytes(&grad_offsets_data));

        let hess_offsets_data: Vec<i32> = (0..=k * 36).map(|i| (i * n) as i32).collect();
        self.hess_offsets = self.client.create(i32::as_bytes(&hess_offsets_data));

        Ok(())
    }

    /// Upload source points and voxel grid data from VoxelGrid.
    ///
    /// This is a convenience wrapper around `upload_data` that extracts
    /// GpuVoxelData from a VoxelGrid.
    pub fn upload_from_voxel_grid(
        &mut self,
        source_points: &[[f32; 3]],
        voxel_grid: &VoxelGrid,
    ) -> Result<()> {
        let voxel_data = GpuVoxelData::from_voxel_grid(voxel_grid);
        self.upload_data(source_points, &voxel_data)
    }

    /// Evaluate a batch of particles on GPU.
    ///
    /// # Arguments
    /// * `particles` - K initial poses to evaluate [tx, ty, tz, roll, pitch, yaw]
    ///
    /// # Returns
    /// K results, one per particle, with final pose, score, and convergence status.
    pub fn evaluate_batch(&mut self, particles: &[[f64; 6]]) -> Result<Vec<BatchedNdtResult>> {
        let k = particles.len();
        let n = self.num_points;
        let v = self.num_voxels;

        if k == 0 {
            return Ok(vec![]);
        }

        if k > self.config.max_batch_size {
            anyhow::bail!("Batch size {k} exceeds max {}", self.config.max_batch_size);
        }

        if n == 0 {
            // No points - return trivial results
            return Ok(particles
                .iter()
                .map(|pose| BatchedNdtResult {
                    pose: *pose,
                    score: 0.0,
                    iterations: 0,
                    converged: true,
                })
                .collect());
        }

        // Upload initial poses to GPU (convert f64 to f32)
        let poses_flat: Vec<f32> = particles
            .iter()
            .flat_map(|p| p.iter().map(|&x| x as f32))
            .collect();
        self.poses_gpu = self.client.create(f32::as_bytes(&poses_flat));

        // Initialize alphas to 1.0 and convergence flags to 0
        let alphas_data = vec![1.0f32; k];
        self.alphas = self.client.create(f32::as_bytes(&alphas_data));
        let converged_init = vec![0u32; k];
        self.converged_flags = self.client.create(u32::as_bytes(&converged_init));

        // Grid dimensions for batched kernels
        let cube_count_n = n.div_ceil(256) as u32;
        let epsilon_sq = self.config.epsilon_sq as f32;
        let radius_sq = self.config.search_radius * self.config.search_radius;

        // Track iterations per particle
        let mut iterations = vec![0u32; k];

        for iter in 0..self.config.max_iterations {
            // Step 1: Compute transforms for all K poses (computes sin/cos internally)
            unsafe {
                compute_transforms_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(k as u32, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.poses_gpu, k * 6, 1),
                    ScalarArg::new(k as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.transforms, k * 16, 1),
                );
            }

            // Step 2: Compute sin/cos for all K poses (for Jacobians)
            unsafe {
                compute_sin_cos_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(k as u32, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.poses_gpu, k * 6, 1),
                    ScalarArg::new(k as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, k * 6, 1),
                );
            }

            // Step 3: Transform all K×N points
            unsafe {
                transform_points_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count_n, k as u32, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, n * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transforms, k * 16, 1),
                    ScalarArg::new(n as u32),
                    ScalarArg::new(k as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.transformed, k * n * 3, 1),
                );
            }

            // Step 4: Radius search for all K×N points
            unsafe {
                radius_search_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count_n, k as u32, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transformed, k * n * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, v * 3, 1),
                    ArrayArg::from_raw_parts::<u32>(&self.voxel_valid, v, 1),
                    ScalarArg::new(radius_sq),
                    ScalarArg::new(n as u32),
                    ScalarArg::new(v as u32),
                    ScalarArg::new(k as u32),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        k * n * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, k * n, 1),
                );
            }

            // Step 5: Compute Jacobians for all K×N points
            unsafe {
                compute_jacobians_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count_n, k as u32, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, n * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, k * 6, 1),
                    ScalarArg::new(n as u32),
                    ScalarArg::new(k as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.jacobians, k * n * 18, 1),
                );
            }

            // Step 6: Compute scores for all K×N points
            unsafe {
                compute_ndt_score_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count_n, k as u32, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, n * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transforms, k * 16, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, v * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, v * 9, 1),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        k * n * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, k * n, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.params, 4, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.scores, k * n, 1),
                    ArrayArg::from_raw_parts::<u32>(&self.correspondences, k * n, 1),
                );
            }

            // Step 7: Compute gradients for all K×N points
            unsafe {
                compute_ndt_gradient_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count_n, k as u32, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, n * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transforms, k * 16, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.jacobians, k * n * 18, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, v * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, v * 9, 1),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        k * n * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, k * n, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.params, 4, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.gradients, k * 6 * n, 1),
                );
            }

            // Step 8: Compute Hessians for all K×N points
            unsafe {
                compute_ndt_hessian_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count_n, k as u32, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, n * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transforms, k * 16, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.jacobians, k * n * 18, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_means, v * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, v * 9, 1),
                    ArrayArg::from_raw_parts::<i32>(
                        &self.neighbor_indices,
                        k * n * MAX_NEIGHBORS as usize,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, k * n, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.params, 4, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.hessians, k * 36 * n, 1),
                );
            }

            // Step 9: CUB reductions
            cubecl::future::block_on(self.client.sync());

            // Reduce scores: [K×N] → [K]
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.scores),
                    self.raw_ptr(&self.score_sums),
                    k,
                    self.raw_ptr(&self.score_offsets),
                )?;
            }

            // Reduce gradients: [K×6×N] → [K×6]
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.gradients),
                    self.raw_ptr(&self.gradient_sums),
                    k * 6,
                    self.raw_ptr(&self.grad_offsets),
                )?;
            }

            // Reduce hessians: [K×36×N] → [K×36]
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.hessians),
                    self.raw_ptr(&self.hessian_sums),
                    k * 36,
                    self.raw_ptr(&self.hess_offsets),
                )?;
            }

            // Step 10: Newton solve for each particle
            // Download reduced gradients and Hessians
            let grad_bytes = self.client.read_one(self.gradient_sums.clone());
            let grad_vals = f32::from_bytes(&grad_bytes);
            let hess_bytes = self.client.read_one(self.hessian_sums.clone());
            let hess_vals = f32::from_bytes(&hess_bytes);

            // Solve K Newton systems on CPU (cuSOLVER batched would be used for larger K)
            let mut deltas_flat = vec![0.0f32; k * 6];
            for batch_idx in 0..k {
                // Extract gradient and Hessian for this particle
                let grad: [f64; 6] = std::array::from_fn(|i| grad_vals[batch_idx * 6 + i] as f64);
                let hess_flat: [f64; 36] =
                    std::array::from_fn(|i| hess_vals[batch_idx * 36 + i] as f64);

                // Solve H * delta = -g using simple Cholesky/LU
                // For now, use a simple Newton step (could use batched cuSOLVER)
                if let Ok(delta) = solve_newton_step(&hess_flat, &grad) {
                    for i in 0..6 {
                        deltas_flat[batch_idx * 6 + i] = delta[i] as f32;
                    }
                }
            }

            // Upload deltas
            self.deltas = self.client.create(f32::as_bytes(&deltas_flat));

            // Step 11: Update poses
            unsafe {
                update_poses_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(6, k as u32, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.poses_gpu, k * 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.deltas, k * 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.alphas, k, 1),
                    ScalarArg::new(k as u32),
                );
            }

            // Step 12: Check convergence
            unsafe {
                check_convergence_batch_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(k as u32, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.deltas, k * 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.alphas, k, 1),
                    ScalarArg::new(epsilon_sq),
                    ScalarArg::new(k as u32),
                    ArrayArg::from_raw_parts::<u32>(&self.converged_flags, k, 1),
                );
            }

            // Download convergence flags
            let flag_bytes = self.client.read_one(self.converged_flags.clone());
            let flags = u32::from_bytes(&flag_bytes);

            // Update iteration counts for non-converged particles
            let mut num_converged = 0;
            for (idx, &flag) in flags.iter().take(k).enumerate() {
                if flag != 0 {
                    if iterations[idx] == 0 {
                        iterations[idx] = iter + 1;
                    }
                    num_converged += 1;
                }
            }

            // Early exit if all converged
            if num_converged == k {
                break;
            }
        }

        // Download final poses and scores
        let final_poses_bytes = self.client.read_one(self.poses_gpu.clone());
        let final_poses = f32::from_bytes(&final_poses_bytes);
        let final_scores_bytes = self.client.read_one(self.score_sums.clone());
        let final_scores = f32::from_bytes(&final_scores_bytes);
        let final_flags_bytes = self.client.read_one(self.converged_flags.clone());
        let final_flags = u32::from_bytes(&final_flags_bytes);

        // Build results
        let results: Vec<BatchedNdtResult> = (0..k)
            .map(|i| {
                let pose: [f64; 6] = std::array::from_fn(|j| final_poses[i * 6 + j] as f64);
                let score = final_scores[i] as f64;
                let converged = final_flags[i] != 0;
                let iters = if iterations[i] > 0 {
                    iterations[i]
                } else {
                    self.config.max_iterations
                };

                BatchedNdtResult {
                    pose,
                    score,
                    iterations: iters,
                    converged,
                }
            })
            .collect();

        Ok(results)
    }

    /// Get memory requirements for the pipeline.
    pub fn memory_requirements(&self) -> PipelineMemoryRequirements {
        let n = self.config.max_points;
        let v = self.config.max_voxels;
        let k = self.config.max_batch_size;

        PipelineMemoryRequirements {
            source_points_bytes: n * 3 * 4,                            // N×3 f32
            voxel_means_bytes: v * 3 * 4,                              // V×3 f32
            voxel_inv_covs_bytes: v * 9 * 4,                           // V×9 f32
            voxel_valid_bytes: v * 4,                                  // V u32
            batch_poses_bytes: k * 6 * 8,                              // K×6 f64
            batch_transforms_bytes: k * 16 * 4,                        // K×16 f32
            batch_transformed_bytes: k * n * 3 * 4,                    // K×N×3 f32
            batch_neighbors_bytes: k * n * MAX_NEIGHBORS as usize * 4, // K×N×8 i32
            batch_neighbor_counts_bytes: k * n * 4,                    // K×N u32
            batch_scores_bytes: k * n * 4,                             // K×N f32
            batch_gradients_bytes: k * 6 * n * 4,                      // K×6×N f32
            batch_hessians_bytes: k * 36 * n * 4,                      // K×36×N f32
            reduced_gradients_bytes: k * 6 * 8,                        // K×6 f64
            reduced_hessians_bytes: k * 36 * 8,                        // K×36 f64
            solver_workspace_bytes: k * 36 * 8 + k * 6 * 8,            // Hessians + gradients copy
        }
    }
}

/// Memory requirements breakdown for the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineMemoryRequirements {
    pub source_points_bytes: usize,
    pub voxel_means_bytes: usize,
    pub voxel_inv_covs_bytes: usize,
    pub voxel_valid_bytes: usize,
    pub batch_poses_bytes: usize,
    pub batch_transforms_bytes: usize,
    pub batch_transformed_bytes: usize,
    pub batch_neighbors_bytes: usize,
    pub batch_neighbor_counts_bytes: usize,
    pub batch_scores_bytes: usize,
    pub batch_gradients_bytes: usize,
    pub batch_hessians_bytes: usize,
    pub reduced_gradients_bytes: usize,
    pub reduced_hessians_bytes: usize,
    pub solver_workspace_bytes: usize,
}

impl PipelineMemoryRequirements {
    /// Total GPU memory required in bytes.
    pub fn total_bytes(&self) -> usize {
        self.source_points_bytes
            + self.voxel_means_bytes
            + self.voxel_inv_covs_bytes
            + self.voxel_valid_bytes
            + self.batch_poses_bytes
            + self.batch_transforms_bytes
            + self.batch_transformed_bytes
            + self.batch_neighbors_bytes
            + self.batch_neighbor_counts_bytes
            + self.batch_scores_bytes
            + self.batch_gradients_bytes
            + self.batch_hessians_bytes
            + self.reduced_gradients_bytes
            + self.reduced_hessians_bytes
            + self.solver_workspace_bytes
    }

    /// Total GPU memory required in megabytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Solve Newton step: H * delta = -g
///
/// Uses LU decomposition for a 6x6 system.
fn solve_newton_step(hessian_flat: &[f64; 36], gradient: &[f64; 6]) -> Result<[f64; 6]> {
    use nalgebra::{Matrix6, Vector6};

    // Convert flat array to matrix (row-major)
    let mut h = Matrix6::zeros();
    for i in 0..6 {
        for j in 0..6 {
            h[(i, j)] = hessian_flat[i * 6 + j];
        }
    }

    // Make gradient into vector and negate
    let g = Vector6::from_column_slice(gradient);
    let neg_g = -g;

    // Solve using LU decomposition
    let lu = h.lu();
    match lu.solve(&neg_g) {
        Some(delta) => {
            let mut result = [0.0; 6];
            for i in 0..6 {
                result[i] = delta[i];
            }
            Ok(result)
        }
        None => {
            // Fall back to gradient descent if LU fails
            let scale = 0.01;
            let mut result = [0.0; 6];
            for i in 0..6 {
                result[i] = -scale * gradient[i];
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_pipeline_creation() {
        let config = GpuInitialPoseConfig {
            max_points: 1000,
            max_voxels: 10000,
            max_batch_size: 8,
            ..Default::default()
        };

        let pipeline = GpuInitialPosePipeline::new(config).unwrap();
        // Pipeline created successfully
        assert_eq!(pipeline.config().max_points, 1000);
    }

    #[test]
    fn test_memory_requirements_calculation() {
        // Test memory requirement calculations without creating actual GPU pipeline
        let config = GpuInitialPoseConfig {
            max_points: 1000,
            max_voxels: 10000,
            max_batch_size: 8,
            ..Default::default()
        };

        let n = config.max_points;
        let v = config.max_voxels;
        let k = config.max_batch_size;

        let mem = PipelineMemoryRequirements {
            source_points_bytes: n * 3 * 4,
            voxel_means_bytes: v * 3 * 4,
            voxel_inv_covs_bytes: v * 9 * 4,
            voxel_valid_bytes: v * 4,
            batch_poses_bytes: k * 6 * 8,
            batch_transforms_bytes: k * 16 * 4,
            batch_transformed_bytes: k * n * 3 * 4,
            batch_neighbors_bytes: k * n * MAX_NEIGHBORS as usize * 4,
            batch_neighbor_counts_bytes: k * n * 4,
            batch_scores_bytes: k * n * 4,
            batch_gradients_bytes: k * 6 * n * 4,
            batch_hessians_bytes: k * 36 * n * 4,
            reduced_gradients_bytes: k * 6 * 8,
            reduced_hessians_bytes: k * 36 * 8,
            solver_workspace_bytes: k * 36 * 8 + k * 6 * 8,
        };

        // Verify memory calculations are reasonable
        assert!(mem.total_mb() > 0.0);
        assert!(mem.total_mb() < 1000.0); // Less than 1GB

        println!("Total memory: {:.2} MB", mem.total_mb());
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_evaluate_batch_empty() {
        let config = GpuInitialPoseConfig {
            max_points: 100,
            max_voxels: 1000,
            max_batch_size: 4,
            ..Default::default()
        };

        let mut pipeline = GpuInitialPosePipeline::new(config).unwrap();

        // Empty batch should return empty results
        let results = pipeline.evaluate_batch(&[]).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_solve_newton_step() {
        // Test with identity Hessian
        let hessian: [f64; 36] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
        ];
        let gradient = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let delta = solve_newton_step(&hessian, &gradient).unwrap();

        // With identity H, delta = -g
        for i in 0..6 {
            assert!((delta[i] + gradient[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_default_config() {
        let config = GpuInitialPoseConfig::default();
        assert_eq!(config.max_points, 1000);
        assert_eq!(config.max_voxels, 20000);
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.max_iterations, 30);
    }

    #[test]
    fn test_batched_ndt_result() {
        let result = BatchedNdtResult {
            pose: [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            score: -100.0,
            iterations: 5,
            converged: true,
        };

        assert!(result.converged);
        assert_eq!(result.iterations, 5);
        assert_eq!(result.pose[0], 1.0);
    }
}
