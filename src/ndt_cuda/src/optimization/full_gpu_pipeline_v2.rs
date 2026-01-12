//! Full GPU Newton Pipeline V2 with Integrated Line Search.
//!
//! This module implements Phase 15.11: a complete GPU Newton optimization pipeline
//! with minimal CPU-GPU data transfers during iterations.
//!
//! # Architecture
//!
//! ```text
//! Once at start (upload ~500 KB):
//!   Upload: source_points [N×3], voxel_data [V×12], initial_pose [6]
//!
//! GPU Iteration Loop:
//!   PHASE A: Compute Newton direction δ = -H⁻¹g
//!     1. compute_sin_cos_kernel(pose → sin_cos)
//!     2. compute_transform_from_sincos_kernel(sin_cos, pose → transform)
//!     3. compute_jacobians_kernel(sin_cos → jacobians)
//!     4. compute_point_hessians_kernel(sin_cos → point_hessians)
//!     5. transform_points_kernel(transform → transformed)
//!     6. radius_search_kernel(→ neighbors)
//!     7. score/gradient/hessian_v2 kernels (→ scores, grads, hess)
//!     8. CUB segmented reduce (→ score[1], gradient[6], H[36])
//!     9. cuSOLVER solve (H, -g → delta) [requires f64, small download]
//!
//!   PHASE B: Batched line search
//!     10. dot_product_6_kernel(gradient · delta → dphi_0)
//!     11. generate_candidates_kernel(→ candidates[K])
//!     12. batch_transform_kernel(pose, delta, candidates → K transforms)
//!     13. batch_score_gradient_kernel(→ batch_scores[K×N], batch_dir_derivs[K×N])
//!     14. CUB reduce per candidate (→ phi[K], dphi[K])
//!     15. more_thuente_kernel(phi_0, dphi_0, phi, dphi → best_alpha)
//!
//!   PHASE C: Update state
//!     16. update_pose_kernel(pose += best_alpha × delta)
//!     17. check_convergence_kernel(→ converged_flag)
//!     18. Download converged_flag (4 bytes)
//!
//! Once at end (download ~220 bytes):
//!   Download: final_pose [48B], score [8B], H [288B]
//! ```
//!
//! # Transfer Analysis
//!
//! | Phase | Transfer Size | Notes |
//! |-------|---------------|-------|
//! | Phase 14 (current) | ~490 KB/iter | J/PH combine roundtrip |
//! | Phase 15 V2 | ~200 bytes/iter | Newton solve (f32→f64) + convergence |

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::derivatives::gpu::{
    compute_ndt_gradient_kernel, compute_ndt_hessian_kernel_v2, compute_ndt_score_kernel,
    radius_search_kernel, GpuVoxelData, MAX_NEIGHBORS,
};
use crate::derivatives::gpu_jacobian::{
    compute_jacobians_kernel, compute_point_hessians_kernel, compute_sin_cos_kernel,
};
use crate::optimization::gpu_newton::GpuNewtonSolver;
use crate::optimization::gpu_pipeline_kernels::{
    apply_regularization_kernel, batch_score_gradient_kernel, batch_transform_kernel,
    cast_u32_to_f32_kernel, check_convergence_kernel, compute_transform_from_sincos_kernel,
    dot_product_6_kernel, generate_candidates_kernel, more_thuente_kernel, update_pose_kernel,
    DEFAULT_NUM_CANDIDATES,
};
use crate::voxel_grid::kernels::transform_points_kernel;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// Result of V2 full GPU optimization.
#[derive(Debug, Clone)]
pub struct FullGpuOptimizationResultV2 {
    /// Final pose [tx, ty, tz, roll, pitch, yaw]
    pub pose: [f64; 6],
    /// Final NDT score (negative log-likelihood)
    pub score: f64,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: u32,
    /// Final Hessian matrix (6x6, row-major)
    pub hessian: [[f64; 6]; 6],
    /// Number of point-voxel correspondences
    pub num_correspondences: usize,
    /// Whether line search was used
    pub used_line_search: bool,
    /// Average alpha (step size) across iterations
    pub avg_alpha: f64,
}

/// Configuration for the V2 pipeline.
#[derive(Debug, Clone)]
pub struct PipelineV2Config {
    /// Number of line search candidates (K)
    pub num_candidates: u32,
    /// Whether to enable line search
    pub use_line_search: bool,
    /// Initial step size for line search
    pub initial_step: f32,
    /// Minimum step size
    pub step_min: f32,
    /// Maximum step size
    pub step_max: f32,
    /// Sufficient decrease parameter (mu) for Armijo condition
    pub armijo_mu: f32,
    /// Curvature parameter (nu) for Strong Wolfe condition
    pub wolfe_nu: f32,
    /// Whether GNSS regularization is enabled
    pub regularization_enabled: bool,
    /// Scale factor for regularization term
    pub regularization_scale_factor: f32,
}

impl Default for PipelineV2Config {
    fn default() -> Self {
        Self {
            num_candidates: DEFAULT_NUM_CANDIDATES,
            use_line_search: true,
            initial_step: 1.0,
            step_min: 0.001,
            step_max: 10.0,
            armijo_mu: 1e-4,
            wolfe_nu: 0.9,
            regularization_enabled: false,
            regularization_scale_factor: 0.01,
        }
    }
}

/// Full GPU Newton Pipeline V2 with integrated line search.
///
/// This pipeline minimizes CPU-GPU transfers by keeping all intermediate
/// state on GPU. Per-iteration transfer is ~200 bytes (Newton solve requires
/// f64 precision which needs download/upload).
pub struct FullGpuPipelineV2 {
    client: CudaClient,
    #[allow(dead_code)]
    device: CudaDevice,

    // Capacity tracking
    max_points: usize,
    max_voxels: usize,

    // Current sizes
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
    // Iteration state (GPU-resident)
    // ========================================================================
    pose_gpu: Handle,     // [6] - current pose
    delta_gpu: Handle,    // [6] - Newton direction
    sin_cos: Handle,      // [6] - sin/cos of pose angles
    transform: Handle,    // [16] - 4x4 transform matrix
    best_alpha: Handle,   // [1] - step size from line search
    ls_converged: Handle, // [1] - 1.0 if Wolfe satisfied

    // ========================================================================
    // Derivative buffers (reused each iteration)
    // ========================================================================
    jacobians: Handle,          // [N × 18]
    point_hessians: Handle,     // [N × 144]
    transformed_points: Handle, // [N × 3]
    neighbor_indices: Handle,   // [N × MAX_NEIGHBORS]
    neighbor_counts: Handle,    // [N]

    // ========================================================================
    // Reduction buffers
    // ========================================================================
    scores: Handle,              // [N]
    correspondences: Handle,     // [N] u32
    correspondences_f32: Handle, // [N] f32 - for reduction (cast from u32)
    gradients: Handle,           // [N × 6] column-major
    hessians: Handle,            // [N × 36] column-major
    reduce_output: Handle,       // [43] = score + grad[6] + H[36]
    gradient_reduced: Handle,    // [6] - copy of reduced gradient for dot product

    // ========================================================================
    // Line search buffers
    // ========================================================================
    candidates: Handle,        // [K]
    batch_transformed: Handle, // [K × N × 3]
    batch_scores: Handle,      // [K × N]
    batch_dir_derivs: Handle,  // [K × N]
    phi_cache: Handle,         // [K] - reduced scores per candidate
    dphi_cache: Handle,        // [K] - reduced directional derivs per candidate
    phi_0: Handle,             // [1] - score at current pose
    dphi_0: Handle,            // [1] - directional derivative at current pose

    // ========================================================================
    // Convergence flag
    // ========================================================================
    converged_flag: Handle, // [1] u32

    // ========================================================================
    // CUB reduction buffers (pre-allocated)
    // ========================================================================
    reduce_temp: Handle,
    reduce_temp_bytes: usize,
    score_offsets: Handle, // [2] for 1 segment
    grad_offsets: Handle,  // [7] for 6 segments
    hess_offsets: Handle,  // [37] for 36 segments
    phi_offsets: Handle,   // [K+1] for K segments

    // ========================================================================
    // Pre-allocated parameter buffers
    // ========================================================================
    gauss_params: Handle, // [2] - gauss_d1, gauss_d2
    batch_params: Handle, // [4] - gauss_d1, gauss_d2, num_points, num_candidates
    ls_params: Handle,    // [3] - num_candidates, mu, nu

    // ========================================================================
    // Regularization buffers
    // ========================================================================
    reg_params: Handle,         // [4] - ref_x, ref_y, scale_factor, enabled
    correspondence_sum: Handle, // [1] - sum of all correspondences
    corr_offsets: Handle,       // [2] - offsets for correspondence sum reduction

    // ========================================================================
    // Gaussian parameters (cached values)
    // ========================================================================
    gauss_d1: f32,
    gauss_d2: f32,
    search_radius_sq: f32,

    // ========================================================================
    // Newton solver
    // ========================================================================
    newton_solver: GpuNewtonSolver,

    // ========================================================================
    // Configuration
    // ========================================================================
    config: PipelineV2Config,
}

impl FullGpuPipelineV2 {
    /// Get raw CUDA device pointer from CubeCL handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let binding = handle.clone().binding();
        let resource = self.client.get_resource(binding);
        resource.resource().ptr
    }

    /// Create a new V2 pipeline with given capacity.
    pub fn new(max_points: usize, max_voxels: usize) -> Result<Self> {
        Self::with_config(max_points, max_voxels, PipelineV2Config::default())
    }

    /// Create a new V2 pipeline with custom configuration.
    pub fn with_config(
        max_points: usize,
        max_voxels: usize,
        config: PipelineV2Config,
    ) -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        let k = config.num_candidates as usize;

        // Persistent data buffers
        let source_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let voxel_means = client.empty(max_voxels * 3 * std::mem::size_of::<f32>());
        let voxel_inv_covs = client.empty(max_voxels * 9 * std::mem::size_of::<f32>());
        let voxel_valid = client.empty(max_voxels * std::mem::size_of::<u32>());

        // Iteration state buffers
        let pose_gpu = client.empty(6 * std::mem::size_of::<f32>());
        let delta_gpu = client.empty(6 * std::mem::size_of::<f32>());
        let sin_cos = client.empty(6 * std::mem::size_of::<f32>());
        let transform = client.empty(16 * std::mem::size_of::<f32>());
        let best_alpha = client.empty(std::mem::size_of::<f32>());
        let ls_converged = client.empty(std::mem::size_of::<f32>());

        // Derivative buffers
        let jacobians = client.empty(max_points * 18 * std::mem::size_of::<f32>());
        let point_hessians = client.empty(max_points * 144 * std::mem::size_of::<f32>());
        let transformed_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let neighbor_indices =
            client.empty(max_points * MAX_NEIGHBORS as usize * std::mem::size_of::<i32>());
        let neighbor_counts = client.empty(max_points * std::mem::size_of::<u32>());

        // Reduction buffers
        let scores = client.empty(max_points * std::mem::size_of::<f32>());
        let correspondences = client.empty(max_points * std::mem::size_of::<u32>());
        let correspondences_f32 = client.empty(max_points * std::mem::size_of::<f32>());
        let gradients = client.empty(max_points * 6 * std::mem::size_of::<f32>());
        let hessians = client.empty(max_points * 36 * std::mem::size_of::<f32>());
        let reduce_output = client.empty(43 * std::mem::size_of::<f32>());
        let gradient_reduced = client.empty(6 * std::mem::size_of::<f32>());

        // Line search buffers
        let candidates = client.empty(k * std::mem::size_of::<f32>());
        let batch_transformed = client.empty(k * max_points * 3 * std::mem::size_of::<f32>());
        let batch_scores = client.empty(k * max_points * std::mem::size_of::<f32>());
        let batch_dir_derivs = client.empty(k * max_points * std::mem::size_of::<f32>());
        let phi_cache = client.empty(k * std::mem::size_of::<f32>());
        let dphi_cache = client.empty(k * std::mem::size_of::<f32>());
        let phi_0 = client.empty(std::mem::size_of::<f32>());
        let dphi_0 = client.empty(std::mem::size_of::<f32>());

        // Convergence flag
        let converged_flag = client.empty(std::mem::size_of::<u32>());

        // CUB reduction temp storage
        let reduce_temp_bytes =
            cuda_ffi::segmented_reduce_sum_f32_temp_size(max_points * 43, 43)? as usize;
        let reduce_temp = client.empty(reduce_temp_bytes.max(256));

        // Pre-allocate offset buffers (will be updated in upload_alignment_data)
        let score_offsets = client.create(i32::as_bytes(&[0i32, max_points as i32]));
        let grad_offsets: Vec<i32> = (0..=6).map(|i| (i * max_points) as i32).collect();
        let grad_offsets = client.create(i32::as_bytes(&grad_offsets));
        let hess_offsets: Vec<i32> = (0..=36).map(|i| (i * max_points) as i32).collect();
        let hess_offsets = client.create(i32::as_bytes(&hess_offsets));
        let phi_offsets: Vec<i32> = (0..=k).map(|i| (i * max_points) as i32).collect();
        let phi_offsets = client.create(i32::as_bytes(&phi_offsets));

        // Pre-allocate parameter buffers
        let gauss_params = client.empty(2 * std::mem::size_of::<f32>());
        let batch_params = client.empty(4 * std::mem::size_of::<f32>());
        let ls_params_data = [
            config.num_candidates as f32,
            config.armijo_mu,
            config.wolfe_nu,
        ];
        let ls_params = client.create(f32::as_bytes(&ls_params_data));

        // Regularization buffers
        // [ref_x, ref_y, scale_factor, enabled] - enabled as 0.0/1.0
        let reg_params_data = [0.0f32, 0.0, config.regularization_scale_factor, 0.0];
        let reg_params = client.create(f32::as_bytes(&reg_params_data));
        let correspondence_sum = client.empty(std::mem::size_of::<f32>());
        let corr_offsets = client.create(i32::as_bytes(&[0i32, max_points as i32]));

        // Newton solver
        let newton_solver = GpuNewtonSolver::new(0)?;

        Ok(Self {
            client,
            device,
            max_points,
            max_voxels,
            num_points: 0,
            num_voxels: 0,
            source_points,
            voxel_means,
            voxel_inv_covs,
            voxel_valid,
            pose_gpu,
            delta_gpu,
            sin_cos,
            transform,
            best_alpha,
            ls_converged,
            jacobians,
            point_hessians,
            transformed_points,
            neighbor_indices,
            neighbor_counts,
            scores,
            correspondences,
            correspondences_f32,
            gradients,
            hessians,
            reduce_output,
            gradient_reduced,
            candidates,
            batch_transformed,
            batch_scores,
            batch_dir_derivs,
            phi_cache,
            dphi_cache,
            phi_0,
            dphi_0,
            converged_flag,
            reduce_temp,
            reduce_temp_bytes,
            score_offsets,
            grad_offsets,
            hess_offsets,
            phi_offsets,
            gauss_params,
            batch_params,
            ls_params,
            reg_params,
            correspondence_sum,
            corr_offsets,
            gauss_d1: 0.0,
            gauss_d2: 0.0,
            search_radius_sq: 0.0,
            newton_solver,
            config,
        })
    }

    /// Upload alignment data to GPU.
    ///
    /// This is called once per alignment before running iterations.
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
        let k = self.config.num_candidates as usize;

        if num_points > self.max_points {
            anyhow::bail!("Too many source points: {num_points} > {}", self.max_points);
        }
        if num_voxels > self.max_voxels {
            anyhow::bail!("Too many voxels: {num_voxels} > {}", self.max_voxels);
        }

        self.num_points = num_points;
        self.num_voxels = num_voxels;
        self.gauss_d1 = gauss_d1;
        self.gauss_d2 = gauss_d2;
        self.search_radius_sq = search_radius * search_radius;

        // Flatten source points
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

        // Update CUB offsets for actual num_points
        let n = num_points as i32;
        self.score_offsets = self.client.create(i32::as_bytes(&[0i32, n]));

        let grad_offsets: Vec<i32> = (0..=6).map(|i| i * n).collect();
        self.grad_offsets = self.client.create(i32::as_bytes(&grad_offsets));

        let hess_offsets: Vec<i32> = (0..=36).map(|i| i * n).collect();
        self.hess_offsets = self.client.create(i32::as_bytes(&hess_offsets));

        let phi_offsets: Vec<i32> = (0..=k).map(|i| (i * num_points) as i32).collect();
        self.phi_offsets = self.client.create(i32::as_bytes(&phi_offsets));

        // Update correspondence sum offsets
        self.corr_offsets = self.client.create(i32::as_bytes(&[0i32, n]));

        // Update parameter buffers
        self.gauss_params = self.client.create(f32::as_bytes(&[gauss_d1, gauss_d2]));
        self.batch_params = self.client.create(f32::as_bytes(&[
            gauss_d1,
            gauss_d2,
            num_points as f32,
            self.config.num_candidates as f32,
        ]));

        Ok(())
    }

    /// Set the regularization reference pose (from GNSS).
    ///
    /// This enables GNSS regularization if the pipeline is configured for it.
    /// The reference pose provides the x, y coordinates that the optimizer
    /// will be penalized for deviating from in the vehicle's longitudinal direction.
    ///
    /// # Arguments
    /// * `ref_x` - Reference x coordinate (from GNSS)
    /// * `ref_y` - Reference y coordinate (from GNSS)
    pub fn set_regularization_pose(&mut self, ref_x: f64, ref_y: f64) {
        let enabled = if self.config.regularization_enabled {
            1.0f32
        } else {
            0.0f32
        };
        let reg_params_data = [
            ref_x as f32,
            ref_y as f32,
            self.config.regularization_scale_factor,
            enabled,
        ];
        self.reg_params = self.client.create(f32::as_bytes(&reg_params_data));
    }

    /// Clear the regularization reference pose (disables regularization for this alignment).
    pub fn clear_regularization_pose(&mut self) {
        let reg_params_data = [
            0.0f32,
            0.0f32,
            self.config.regularization_scale_factor,
            0.0f32, // disabled
        ];
        self.reg_params = self.client.create(f32::as_bytes(&reg_params_data));
    }

    /// Run full GPU Newton optimization with integrated line search.
    pub fn optimize(
        &mut self,
        initial_pose: &[f64; 6],
        max_iterations: u32,
        transformation_epsilon: f64,
    ) -> Result<FullGpuOptimizationResultV2> {
        let num_points = self.num_points;
        let num_voxels = self.num_voxels;

        if num_points == 0 {
            return Ok(FullGpuOptimizationResultV2 {
                pose: *initial_pose,
                score: 0.0,
                converged: true,
                iterations: 0,
                hessian: [[0.0; 6]; 6],
                num_correspondences: 0,
                used_line_search: false,
                avg_alpha: 1.0,
            });
        }

        // Upload initial pose to GPU
        let pose_f32: [f32; 6] = initial_pose.map(|x| x as f32);
        self.pose_gpu = self.client.create(f32::as_bytes(&pose_f32));

        let epsilon_sq = (transformation_epsilon * transformation_epsilon) as f32;
        let cube_count = num_points.div_ceil(256) as u32;
        let num_candidates = self.config.num_candidates;
        let k = num_candidates as usize;

        let mut iterations = 0u32;
        let mut alpha_sum = 0.0f32;
        let mut alpha_count = 0u32;

        // Cache for final results (updated each iteration)
        let mut last_reduce_output = vec![0.0f32; 43];

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // ==================================================================
            // PHASE A: Compute Newton direction δ = -H⁻¹g
            // ==================================================================

            // Step 1: Compute sin/cos from pose
            unsafe {
                compute_sin_cos_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.pose_gpu, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, 6, 1),
                );
            }

            // Step 2: Compute transform matrix on GPU
            unsafe {
                compute_transform_from_sincos_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.pose_gpu, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                );
            }

            // Step 3: Compute Jacobians
            unsafe {
                compute_jacobians_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, 6, 1),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.jacobians, num_points * 18, 1),
                );
            }

            // Step 4: Compute Point Hessians
            unsafe {
                compute_point_hessians_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.sin_cos, 6, 1),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.point_hessians, num_points * 144, 1),
                );
            }

            // Step 5: Transform points
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

            // Step 6: Radius search
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

            // Step 7a: Score kernel
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
                    ScalarArg::new(self.gauss_d1),
                    ScalarArg::new(self.gauss_d2),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.scores, num_points, 1),
                    ArrayArg::from_raw_parts::<u32>(&self.correspondences, num_points, 1),
                );
            }

            // Step 7b: Gradient kernel
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
                    ScalarArg::new(self.gauss_d1),
                    ScalarArg::new(self.gauss_d2),
                    ScalarArg::new(num_points as u32),
                    ArrayArg::from_raw_parts::<f32>(&self.gradients, num_points * 6, 1),
                );
            }

            // Step 7c: Hessian kernel V2 (separate buffers, no CPU combine)
            unsafe {
                compute_ndt_hessian_kernel_v2::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.transform, 16, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.jacobians, num_points * 18, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.point_hessians, num_points * 144, 1),
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

            // Step 8: CUB reduction (using pre-allocated offsets)
            cubecl::future::block_on(self.client.sync());

            // Reduce scores -> reduce_output[0]
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.scores),
                    self.raw_ptr(&self.reduce_output),
                    1,
                    self.raw_ptr(&self.score_offsets),
                )?;
            }

            // Reduce gradients -> reduce_output[1..7]
            let grad_output_ptr = self.raw_ptr(&self.reduce_output) + 4;
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.gradients),
                    grad_output_ptr,
                    6,
                    self.raw_ptr(&self.grad_offsets),
                )?;
            }

            // Reduce hessians -> reduce_output[7..43]
            let hess_output_ptr = self.raw_ptr(&self.reduce_output) + 28;
            unsafe {
                cuda_ffi::segmented_reduce_sum_f32_inplace(
                    self.raw_ptr(&self.reduce_temp),
                    self.reduce_temp_bytes,
                    self.raw_ptr(&self.hessians),
                    hess_output_ptr,
                    36,
                    self.raw_ptr(&self.hess_offsets),
                )?;
            }

            // Step 8b: Apply regularization (if enabled)
            if self.config.regularization_enabled {
                // Cast correspondences (u32) to f32 for reduction
                unsafe {
                    cast_u32_to_f32_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(cube_count, 1, 1),
                        CubeDim::new(256, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(&self.correspondences, num_points, 1),
                        ScalarArg::new(num_points as u32),
                        ArrayArg::from_raw_parts::<f32>(&self.correspondences_f32, num_points, 1),
                    );
                }

                // Reduce correspondences_f32 -> correspondence_sum
                cubecl::future::block_on(self.client.sync());
                unsafe {
                    cuda_ffi::segmented_reduce_sum_f32_inplace(
                        self.raw_ptr(&self.reduce_temp),
                        self.reduce_temp_bytes,
                        self.raw_ptr(&self.correspondences_f32),
                        self.raw_ptr(&self.correspondence_sum),
                        1,
                        self.raw_ptr(&self.corr_offsets),
                    )?;
                }

                // Apply regularization kernel
                unsafe {
                    apply_regularization_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.pose_gpu, 6, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.reg_params, 4, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.correspondence_sum, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.reduce_output, 43, 1),
                    );
                }
            }

            // Step 9: Newton solve
            // NOTE: Download required because cuSOLVER needs f64 precision
            // This is ~172 bytes download + ~24 bytes upload per iteration
            let reduce_output_bytes = self.client.read_one(self.reduce_output.clone());
            let reduce_output_vals = f32::from_bytes(&reduce_output_bytes);
            last_reduce_output.copy_from_slice(reduce_output_vals);

            let gradient: [f64; 6] = std::array::from_fn(|i| reduce_output_vals[1 + i] as f64);
            let hessian_flat: [f64; 36] = std::array::from_fn(|i| reduce_output_vals[7 + i] as f64);

            let delta = self.newton_solver.solve(&hessian_flat, &gradient)?;
            let delta_f32: [f32; 6] = delta.map(|x| x as f32);
            // Upload delta to pre-allocated GPU buffer (avoids allocation each iteration)
            unsafe {
                cuda_ffi::cuda_memcpy_htod(
                    self.raw_ptr(&self.delta_gpu),
                    delta_f32.as_ptr() as *const u8,
                    6 * std::mem::size_of::<f32>(),
                )?;
            }

            // ==================================================================
            // PHASE B: Line search (if enabled)
            // ==================================================================

            let alpha = if self.config.use_line_search {
                // Copy phi_0 from reduce_output[0] (score at current pose)
                // Use GPU memcpy via raw pointers
                unsafe {
                    cuda_ffi::cuda_memcpy_dtod(
                        self.raw_ptr(&self.phi_0),
                        self.raw_ptr(&self.reduce_output),
                        std::mem::size_of::<f32>(),
                    )?;
                }

                // Copy gradient_reduced from reduce_output[1..7] for dot product
                unsafe {
                    cuda_ffi::cuda_memcpy_dtod(
                        self.raw_ptr(&self.gradient_reduced),
                        self.raw_ptr(&self.reduce_output) + 4,
                        6 * std::mem::size_of::<f32>(),
                    )?;
                }

                // Step 10: Compute directional derivative dphi_0 = gradient · delta
                unsafe {
                    dot_product_6_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.gradient_reduced, 6, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.delta_gpu, 6, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.dphi_0, 1, 1),
                    );
                }

                // Step 11: Generate candidates
                unsafe {
                    generate_candidates_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(num_candidates, 1, 1),
                        ScalarArg::new(self.config.initial_step),
                        ScalarArg::new(self.config.step_min),
                        ScalarArg::new(self.config.step_max),
                        ScalarArg::new(num_candidates),
                        ArrayArg::from_raw_parts::<f32>(&self.candidates, k, 1),
                    );
                }

                // Step 12: Batch transform
                let batch_total = (num_candidates as usize) * num_points;
                let batch_cube_count = batch_total.div_ceil(256) as u32;

                unsafe {
                    batch_transform_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(batch_cube_count, 1, 1),
                        CubeDim::new(256, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.source_points, num_points * 3, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.pose_gpu, 6, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.delta_gpu, 6, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.candidates, k, 1),
                        ScalarArg::new(num_points as u32),
                        ScalarArg::new(num_candidates),
                        ArrayArg::from_raw_parts::<f32>(
                            &self.batch_transformed,
                            k * num_points * 3,
                            1,
                        ),
                    );
                }

                // Step 13: Batch score/gradient
                unsafe {
                    batch_score_gradient_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(batch_cube_count, 1, 1),
                        CubeDim::new(256, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &self.batch_transformed,
                            k * num_points * 3,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(&self.delta_gpu, 6, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.jacobians, num_points * 18, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.voxel_means, num_voxels * 3, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.voxel_inv_covs, num_voxels * 9, 1),
                        ArrayArg::from_raw_parts::<i32>(
                            &self.neighbor_indices,
                            num_points * MAX_NEIGHBORS as usize,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<u32>(&self.neighbor_counts, num_points, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.batch_params, 4, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.batch_scores, k * num_points, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.batch_dir_derivs, k * num_points, 1),
                    );
                }

                // Step 14: Reduce per candidate -> phi_cache[K], dphi_cache[K]
                cubecl::future::block_on(self.client.sync());

                // Reduce batch_scores -> phi_cache
                unsafe {
                    cuda_ffi::segmented_reduce_sum_f32_inplace(
                        self.raw_ptr(&self.reduce_temp),
                        self.reduce_temp_bytes,
                        self.raw_ptr(&self.batch_scores),
                        self.raw_ptr(&self.phi_cache),
                        k,
                        self.raw_ptr(&self.phi_offsets),
                    )?;
                }

                // Reduce batch_dir_derivs -> dphi_cache
                unsafe {
                    cuda_ffi::segmented_reduce_sum_f32_inplace(
                        self.raw_ptr(&self.reduce_temp),
                        self.reduce_temp_bytes,
                        self.raw_ptr(&self.batch_dir_derivs),
                        self.raw_ptr(&self.dphi_cache),
                        k,
                        self.raw_ptr(&self.phi_offsets),
                    )?;
                }

                // Step 15: More-Thuente kernel
                unsafe {
                    more_thuente_kernel::launch_unchecked::<f32, CudaRuntime>(
                        &self.client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.phi_0, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.dphi_0, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.candidates, k, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.phi_cache, k, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.dphi_cache, k, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.ls_params, 3, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.best_alpha, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&self.ls_converged, 1, 1),
                    );
                }

                // Download best_alpha (4 bytes) for statistics
                let alpha_bytes = self.client.read_one(self.best_alpha.clone());
                f32::from_bytes(&alpha_bytes)[0]
            } else {
                // No line search - use full step (write to pre-allocated buffer)
                let alpha_val = [1.0f32];
                unsafe {
                    cuda_ffi::cuda_memcpy_htod(
                        self.raw_ptr(&self.best_alpha),
                        alpha_val.as_ptr() as *const u8,
                        std::mem::size_of::<f32>(),
                    )?;
                }
                1.0f32
            };

            alpha_sum += alpha;
            alpha_count += 1;

            // ==================================================================
            // PHASE C: Update state
            // ==================================================================

            // Step 16: Update pose on GPU
            unsafe {
                update_pose_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(6, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.pose_gpu, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.delta_gpu, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.best_alpha, 1, 1),
                );
            }

            // Step 17: Check convergence on GPU
            unsafe {
                check_convergence_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.delta_gpu, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.best_alpha, 1, 1),
                    ScalarArg::new(epsilon_sq),
                    ArrayArg::from_raw_parts::<u32>(&self.converged_flag, 1, 1),
                );
            }

            // Step 18: Download ONLY convergence flag (4 bytes!)
            let flag_bytes = self.client.read_one(self.converged_flag.clone());
            if u32::from_bytes(&flag_bytes)[0] != 0 {
                break;
            }
        }

        // Download final results
        self.download_final_results(
            iterations,
            if alpha_count > 0 {
                alpha_sum / alpha_count as f32
            } else {
                1.0
            },
            &last_reduce_output,
        )
    }

    /// Download final results from GPU.
    fn download_final_results(
        &self,
        iterations: u32,
        avg_alpha: f32,
        last_reduce_output: &[f32],
    ) -> Result<FullGpuOptimizationResultV2> {
        // Download final pose
        let pose_bytes = self.client.read_one(self.pose_gpu.clone());
        let pose_f32 = f32::from_bytes(&pose_bytes);
        let pose: [f64; 6] = std::array::from_fn(|i| pose_f32[i] as f64);

        // Use cached reduce_output from last iteration
        let score = last_reduce_output[0] as f64;
        let hessian_flat: [f64; 36] = std::array::from_fn(|i| last_reduce_output[7 + i] as f64);

        // Reconstruct hessian matrix
        let mut hessian = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                hessian[i][j] = hessian_flat[i * 6 + j];
            }
        }

        // Download correspondences count (only num_points elements are valid)
        let corr_bytes = self.client.read_one(self.correspondences.clone());
        let correspondences = u32::from_bytes(&corr_bytes);
        // Only sum the first num_points entries; the rest may contain garbage
        let num_correspondences: u32 = correspondences.iter().take(self.num_points).copied().sum();

        // Download convergence flag
        let flag_bytes = self.client.read_one(self.converged_flag.clone());
        let converged = u32::from_bytes(&flag_bytes)[0] != 0;

        Ok(FullGpuOptimizationResultV2 {
            pose,
            score,
            converged,
            iterations,
            hessian,
            num_correspondences: num_correspondences as usize,
            used_line_search: self.config.use_line_search,
            avg_alpha: avg_alpha as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_v2_creation() {
        let pipeline = FullGpuPipelineV2::new(1000, 5000);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_v2_with_config() {
        let config = PipelineV2Config {
            num_candidates: 4,
            use_line_search: false,
            ..Default::default()
        };
        let pipeline = FullGpuPipelineV2::with_config(1000, 5000, config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_v2_empty_input() {
        let mut pipeline = FullGpuPipelineV2::new(1000, 5000).unwrap();

        // Create empty voxel data
        let voxel_data = GpuVoxelData {
            means: vec![],
            inv_covariances: vec![],
            principal_axes: vec![],
            valid: vec![],
            num_voxels: 0,
        };

        // Upload empty data
        pipeline
            .upload_alignment_data(&[], &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        // Run optimization with empty input
        let result = pipeline
            .optimize(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 30, 0.01)
            .unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_pipeline_v2_multiple_points() {
        let mut pipeline = FullGpuPipelineV2::new(1000, 5000).unwrap();

        // Create source points - multiple points needed to constrain all 6 DOF
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ];

        // Create voxel at origin with identity covariance
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0],
            inv_covariances: vec![
                1.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, // row 1
                0.0, 0.0, 1.0, // row 2
            ],
            principal_axes: vec![0.0, 0.0, 1.0],
            valid: vec![1],
            num_voxels: 1,
        };

        // Upload data
        let gauss_d1 = 0.55f32;
        let gauss_d2 = 0.4f32;
        let search_radius = 2.0f32;

        pipeline
            .upload_alignment_data(
                &source_points,
                &voxel_data,
                gauss_d1,
                gauss_d2,
                search_radius,
            )
            .unwrap();

        // Run optimization from a small offset
        let initial_pose = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = pipeline.optimize(&initial_pose, 30, 0.01).unwrap();

        // Should complete some iterations
        assert!(result.iterations > 0, "Should run at least one iteration");
        println!(
            "Multiple points test: {} iterations, converged={}, score={}",
            result.iterations, result.converged, result.score
        );
    }

    #[test]
    fn test_pipeline_v2_no_line_search() {
        let config = PipelineV2Config {
            use_line_search: false,
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        // Create source points
        let source_points = vec![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // Create voxel at origin
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0],
            inv_covariances: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            principal_axes: vec![0.0, 0.0, 1.0],
            valid: vec![1],
            num_voxels: 1,
        };

        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        let result = pipeline
            .optimize(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10, 0.01)
            .unwrap();

        // Should run without line search
        assert!(!result.used_line_search);
        assert!(
            (result.avg_alpha - 1.0).abs() < 1e-6,
            "Without line search, alpha should be 1.0"
        );
        println!(
            "No line search test: {} iterations, converged={}",
            result.iterations, result.converged
        );
    }

    #[test]
    fn test_pipeline_v2_with_line_search() {
        let config = PipelineV2Config {
            use_line_search: true,
            num_candidates: 8,
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        // Create source points arranged around origin
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];

        // Create voxel at origin
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0],
            inv_covariances: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            principal_axes: vec![0.0, 0.0, 1.0],
            valid: vec![1],
            num_voxels: 1,
        };

        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        let result = pipeline
            .optimize(&[0.1, 0.05, 0.0, 0.0, 0.0, 0.0], 30, 0.001)
            .unwrap();

        // Should run with line search
        assert!(result.used_line_search);
        println!(
            "With line search test: {} iterations, converged={}, avg_alpha={}",
            result.iterations, result.converged, result.avg_alpha
        );
    }

    #[test]
    fn test_pipeline_v2_with_regularization() {
        // Use a small scale factor to avoid making the Hessian non-positive-definite
        let config = PipelineV2Config {
            use_line_search: false,
            regularization_enabled: true,
            regularization_scale_factor: 0.001, // Small scale to keep Hessian PD
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        // Create more source points for better conditioning
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ];

        // Create voxel at origin with smaller inv covariance (less constraining)
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0],
            inv_covariances: vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            principal_axes: vec![0.0, 0.0, 1.0],
            valid: vec![1],
            num_voxels: 1,
        };

        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        // Set regularization reference pose at (0.1, 0.0) - small offset
        pipeline.set_regularization_pose(0.1, 0.0);

        // Run from identity pose
        let result = pipeline
            .optimize(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20, 0.001)
            .unwrap();

        println!(
            "Regularization test: {} iterations, converged={}, score={}, pose=({:.4}, {:.4})",
            result.iterations, result.converged, result.score, result.pose[0], result.pose[1]
        );

        // Should complete iterations with regularization active
        assert!(
            result.iterations > 0,
            "Should run at least one iteration with regularization"
        );
    }

    #[test]
    fn test_pipeline_v2_regularization_disabled() {
        // Regularization disabled in config
        let config = PipelineV2Config {
            use_line_search: false,
            regularization_enabled: false,
            regularization_scale_factor: 0.01,
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        let source_points = vec![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0],
            inv_covariances: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            principal_axes: vec![0.0, 0.0, 1.0],
            valid: vec![1],
            num_voxels: 1,
        };

        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        // Set a regularization pose (should be ignored since disabled)
        pipeline.set_regularization_pose(10.0, 10.0);

        let result = pipeline
            .optimize(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10, 0.01)
            .unwrap();

        println!(
            "Regularization disabled test: {} iterations, pose=({:.4}, {:.4})",
            result.iterations, result.pose[0], result.pose[1]
        );

        // Should complete normally (regularization shouldn't affect result)
        assert!(result.iterations > 0 || result.converged);
    }
}
