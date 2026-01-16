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
//!     6. voxel_hash_query(→ neighbors) [O(27) instead of O(V)]
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
//!     17. Download pose for oscillation tracking (24 bytes)
//!     18. check_convergence_kernel(→ converged_flag)
//!     19. Download converged_flag (4 bytes)
//!
//! Once at end:
//!   CPU: Compute oscillation count from pose history
//!   Download: final_pose (already in history), score, H [~300 bytes]
//! ```
//!
//! # Transfer Analysis
//!
//! | Phase | Transfer Size | Notes |
//! |-------|---------------|-------|
//! | Phase 14 (legacy) | ~490 KB/iter | J/PH combine roundtrip |
//! | Phase 15 V2 | ~224 bytes/iter | Newton solve (f32→f64) + pose (oscillation) + convergence |

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::derivatives::gpu::GpuVoxelData;
use crate::optimization::gpu_pipeline_kernels::DEFAULT_NUM_CANDIDATES;

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
    /// Maximum consecutive oscillation count
    pub oscillation_count: usize,
    /// Per-iteration debug data (populated when `enable_debug` is true in config)
    pub iterations_debug: Option<Vec<super::debug::IterationDebug>>,
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
    /// Fixed step size when line search is disabled (matches Autoware's default 0.1)
    pub fixed_step_size: f32,
    /// Whether GNSS regularization is enabled
    pub regularization_enabled: bool,
    /// Scale factor for regularization term
    pub regularization_scale_factor: f32,
    /// Whether to collect per-iteration debug data
    pub enable_debug: bool,
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
            fixed_step_size: 0.1, // Matches Autoware default when line search disabled
            regularization_enabled: false,
            regularization_scale_factor: 0.01,
            enable_debug: false,
        }
    }
}

/// Full GPU Newton Pipeline V2 using persistent kernel.
///
/// This pipeline uses a single cooperative kernel launch for the entire
/// Newton optimization loop, minimizing kernel launch overhead and
/// CPU-GPU synchronization.
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
    // Spatial hash table for O(27) voxel lookup
    // ========================================================================
    hash_table: Handle, // [capacity × 16] bytes (HashEntry)
    hash_capacity: u32, // Hash table capacity (power of 2)
    resolution: f32,    // Voxel grid resolution for grid coordinate conversion

    // ========================================================================
    // Gaussian parameters (cached values)
    // ========================================================================
    gauss_d1: f32,
    gauss_d2: f32,

    // ========================================================================
    // Configuration
    // ========================================================================
    config: PipelineV2Config,

    // ========================================================================
    // Persistent kernel buffers (Phase 17)
    // ========================================================================
    persistent_reduce_buffer: Handle, // [96] for persistent kernel reduction + state
    persistent_initial_pose: Handle,  // [6] input pose for persistent kernel
    persistent_out_pose: Handle,      // [6] output pose from persistent kernel
    persistent_out_iterations: Handle, // [1] i32
    persistent_out_converged: Handle, // [1] u32
    persistent_out_score: Handle,     // [1] f32
    persistent_out_hessian: Handle,   // [36] f32
    persistent_out_correspondences: Handle, // [1] u32 - Phase 18.3
    persistent_out_oscillation: Handle, // [1] u32 - Phase 18.4
    persistent_out_alpha_sum: Handle, // [1] f32 - Phase 19.3

    // Phase 19.4: Debug buffer (only allocated when enable_debug is true)
    persistent_debug_buffer: Option<Handle>, // [max_iterations * 50] f32
    max_iterations_for_debug: u32,           // Cached for buffer sizing

    // Regularization state for persistent kernel - Phase 18.2
    regularization_ref_x: f32,
    regularization_ref_y: f32,
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

        // Synchronize device to ensure any pending operations from previous contexts are complete
        // This prevents race conditions when multiple pipelines are created in rapid succession
        cubecl::future::block_on(client.sync());

        // Persistent data buffers
        let source_points = client.empty(max_points * 3 * std::mem::size_of::<f32>());
        let voxel_means = client.empty(max_voxels * 3 * std::mem::size_of::<f32>());
        let voxel_inv_covs = client.empty(max_voxels * 9 * std::mem::size_of::<f32>());
        let voxel_valid = client.empty(max_voxels * std::mem::size_of::<u32>());

        // Spatial hash table for O(27) voxel lookup
        let hash_capacity = cuda_ffi::hash_table_capacity(max_voxels)?;
        let hash_table_bytes = cuda_ffi::hash_table_size(hash_capacity)?;
        let hash_table = client.empty(hash_table_bytes);

        // Persistent kernel buffers (Phase 17)
        let persistent_reduce_buffer = client.empty(cuda_ffi::persistent_ndt_buffer_size());
        let persistent_initial_pose = client.empty(6 * std::mem::size_of::<f32>());
        let persistent_out_pose = client.empty(6 * std::mem::size_of::<f32>());
        let persistent_out_iterations = client.empty(std::mem::size_of::<i32>());
        let persistent_out_converged = client.empty(std::mem::size_of::<u32>());
        let persistent_out_score = client.empty(std::mem::size_of::<f32>());
        let persistent_out_hessian = client.empty(36 * std::mem::size_of::<f32>());
        let persistent_out_correspondences = client.empty(std::mem::size_of::<u32>());
        let persistent_out_oscillation = client.empty(std::mem::size_of::<u32>());
        let persistent_out_alpha_sum = client.empty(std::mem::size_of::<f32>());

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
            hash_table,
            hash_capacity,
            resolution: 0.0, // Set in upload_alignment_data
            gauss_d1: 0.0,
            gauss_d2: 0.0,
            config,
            persistent_reduce_buffer,
            persistent_initial_pose,
            persistent_out_pose,
            persistent_out_iterations,
            persistent_out_converged,
            persistent_out_score,
            persistent_out_hessian,
            persistent_out_correspondences,
            persistent_out_oscillation,
            persistent_out_alpha_sum,
            persistent_debug_buffer: None, // Allocated on-demand in optimize()
            max_iterations_for_debug: 0,
            regularization_ref_x: 0.0,
            regularization_ref_y: 0.0,
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

        // Build spatial hash table for O(27) voxel lookup
        self.resolution = search_radius; // Resolution = search_radius in NDT

        // Sync CubeCL buffers before using raw pointers with CUDA FFI
        // This ensures all CubeCL writes are visible to CUDA
        cubecl::future::block_on(self.client.sync());

        let voxel_means_ptr = self.raw_ptr(&self.voxel_means);
        let voxel_valid_ptr = self.raw_ptr(&self.voxel_valid);
        let hash_table_ptr = self.raw_ptr(&self.hash_table);

        unsafe {
            cuda_ffi::hash_table_init(hash_table_ptr, self.hash_capacity)?;
            cuda_ffi::hash_table_build(
                voxel_means_ptr,
                voxel_valid_ptr,
                num_voxels,
                self.resolution,
                hash_table_ptr,
                self.hash_capacity,
            )?;
        }

        // Ensure hash table build is complete before any subsequent GPU operations
        // This is critical for persistent kernel which reads the hash table
        cuda_ffi::cuda_device_synchronize()?;

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
        self.regularization_ref_x = ref_x as f32;
        self.regularization_ref_y = ref_y as f32;
    }

    /// Clear the regularization reference pose (disables regularization for this alignment).
    pub fn clear_regularization_pose(&mut self) {
        self.regularization_ref_x = 0.0;
        self.regularization_ref_y = 0.0;
    }

    /// Parse debug buffer into Vec<IterationDebug>.
    ///
    /// # Buffer Layout (per iteration, 50 floats)
    ///
    /// | Offset | Count | Field |
    /// |--------|-------|-------|
    /// | 0 | 1 | iteration |
    /// | 1 | 1 | score |
    /// | 2-7 | 6 | pose_before |
    /// | 8-13 | 6 | gradient |
    /// | 14-34 | 21 | hessian_upper_triangle |
    /// | 35-40 | 6 | delta (Newton step) |
    /// | 41 | 1 | alpha |
    /// | 42 | 1 | correspondences |
    /// | 43 | 1 | direction_reversed |
    /// | 44-49 | 6 | pose_after |
    fn parse_debug_buffer(
        &self,
        num_iterations: usize,
    ) -> Result<Vec<super::debug::IterationDebug>> {
        use super::debug::IterationDebug;

        let buffer = self
            .persistent_debug_buffer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Debug buffer not allocated"))?;
        let bytes = self.client.read_one(buffer.clone());
        let floats = f32::from_bytes(&bytes);

        const FLOATS_PER_ITER: usize =
            cuda_ffi::persistent_ndt::PersistentNdt::DEBUG_FLOATS_PER_ITER;

        let mut result = Vec::with_capacity(num_iterations);
        for iter in 0..num_iterations {
            let base = iter * FLOATS_PER_ITER;
            let mut debug = IterationDebug::new(iter);

            // Parse fields from buffer
            debug.score = floats[base + 1] as f64;
            debug.pose = (0..6).map(|i| floats[base + 2 + i] as f64).collect();
            debug.gradient = (0..6).map(|i| floats[base + 8 + i] as f64).collect();

            // Expand Hessian from upper triangle (21 floats -> 36 floats)
            debug.hessian = Self::expand_upper_triangle(&floats[base + 14..base + 35]);

            debug.newton_step = (0..6).map(|i| floats[base + 35 + i] as f64).collect();
            debug.newton_step_norm = debug.newton_step.iter().map(|x| x * x).sum::<f64>().sqrt();

            debug.step_length = floats[base + 41] as f64;
            debug.num_correspondences = floats[base + 42] as usize;
            debug.direction_reversed = floats[base + 43] > 0.5;
            debug.pose_after = (0..6).map(|i| floats[base + 44 + i] as f64).collect();

            // Set derived fields
            debug.used_line_search = self.config.use_line_search;
            // step_direction is the same as newton_step for our kernel
            debug.step_direction = debug.newton_step.clone();
            // directional_derivative = gradient · step_direction
            debug.directional_derivative = debug
                .gradient
                .iter()
                .zip(debug.step_direction.iter())
                .map(|(g, d)| g * d)
                .sum();

            result.push(debug);
        }

        Ok(result)
    }

    /// Expand upper triangle Hessian (21 floats) to full 6x6 matrix (36 floats).
    ///
    /// Upper triangle layout:
    /// ```text
    /// H[0,0] H[0,1] H[0,2] H[0,3] H[0,4] H[0,5]    →  [0]  [1]  [2]  [3]  [4]  [5]
    ///        H[1,1] H[1,2] H[1,3] H[1,4] H[1,5]    →       [6]  [7]  [8]  [9] [10]
    ///               H[2,2] H[2,3] H[2,4] H[2,5]    →           [11] [12] [13] [14]
    ///                      H[3,3] H[3,4] H[3,5]    →                [15] [16] [17]
    ///                             H[4,4] H[4,5]    →                     [18] [19]
    ///                                    H[5,5]    →                          [20]
    /// ```
    fn expand_upper_triangle(ut: &[f32]) -> Vec<f64> {
        let mut full = vec![0.0f64; 36];

        // Map from upper triangle index to (row, col)
        let mut ut_idx = 0;
        for row in 0..6 {
            for col in row..6 {
                let val = ut[ut_idx] as f64;
                full[row * 6 + col] = val;
                full[col * 6 + row] = val; // Symmetric
                ut_idx += 1;
            }
        }

        full
    }

    /// Run full GPU Newton optimization using the persistent kernel.
    ///
    /// This runs the entire Newton optimization loop in a single kernel launch,
    /// eliminating per-iteration CPU-GPU transfers. Supports all features:
    /// - Line search with Strong Wolfe conditions (if enabled in config)
    /// - GNSS regularization (if enabled)
    /// - Oscillation detection
    /// - Correspondence counting
    /// - Hessian output for covariance estimation
    ///
    /// # Arguments
    ///
    /// * `initial_pose` - Initial pose estimate [tx, ty, tz, roll, pitch, yaw]
    /// * `max_iterations` - Maximum Newton iterations
    /// * `transformation_epsilon` - Convergence threshold for pose delta norm
    ///
    /// # Returns
    ///
    /// Optimization result with final pose, score, convergence status, and diagnostics.
    pub fn optimize(
        &mut self,
        initial_pose: &[f64; 6],
        max_iterations: u32,
        transformation_epsilon: f64,
    ) -> Result<FullGpuOptimizationResultV2> {
        if self.num_points == 0 {
            return Ok(FullGpuOptimizationResultV2 {
                pose: *initial_pose,
                score: 0.0,
                converged: true,
                iterations: 0,
                hessian: [[0.0; 6]; 6],
                num_correspondences: 0,
                used_line_search: false,
                avg_alpha: 1.0,
                oscillation_count: 0,
                iterations_debug: None,
            });
        }

        // Upload initial pose
        let pose_f32: [f32; 6] = initial_pose.map(|x| x as f32);
        self.persistent_initial_pose = self.client.create(f32::as_bytes(&pose_f32));

        // Clear reduce buffer (96 floats for line search support)
        let zeros = [0.0f32; 96];
        self.persistent_reduce_buffer = self.client.create(f32::as_bytes(&zeros));

        // Force CubeCL to sync all pending operations before kernel launch
        // by reading from a buffer (this ensures all previous writes are flushed)
        let _ = self.client.read_one(self.persistent_reduce_buffer.clone());

        // Recreate output buffers to ensure fresh memory (avoids CubeCL caching issues)
        let zeros6 = [0.0f32; 6];
        let zero_i32 = [0i32; 1];
        let zero_u32 = [0u32; 1];
        let zeros36 = [0.0f32; 36];
        self.persistent_out_pose = self.client.create(f32::as_bytes(&zeros6));
        self.persistent_out_iterations = self.client.create(i32::as_bytes(&zero_i32));
        self.persistent_out_converged = self.client.create(u32::as_bytes(&zero_u32));
        self.persistent_out_score = self.client.create(f32::as_bytes(&[0.0f32]));
        self.persistent_out_hessian = self.client.create(f32::as_bytes(&zeros36));
        self.persistent_out_correspondences = self.client.create(u32::as_bytes(&zero_u32));
        self.persistent_out_oscillation = self.client.create(u32::as_bytes(&zero_u32));
        self.persistent_out_alpha_sum = self.client.create(f32::as_bytes(&[0.0f32]));

        // Phase 19.4: Allocate debug buffer if enabled
        let (debug_enabled, debug_ptr) = if self.config.enable_debug {
            let buffer_size = max_iterations as usize
                * cuda_ffi::persistent_ndt::PersistentNdt::DEBUG_FLOATS_PER_ITER
                * std::mem::size_of::<f32>();
            self.persistent_debug_buffer = Some(self.client.empty(buffer_size));
            self.max_iterations_for_debug = max_iterations;
            (
                true,
                self.raw_ptr(self.persistent_debug_buffer.as_ref().unwrap()),
            )
        } else {
            (false, 0)
        };

        // Ensure all CubeCL operations are complete before cooperative kernel launch.
        // This ensures:
        // 1. All CubeCL buffer writes (source points, voxels, hash table) are complete
        // 2. Hash table from upload_alignment_data is fully visible
        // 3. All output buffers are initialized
        // Without this, the cooperative kernel may read uninitialized hash table
        // causing it to loop through all slots (slow) or produce wrong results
        cuda_ffi::cuda_device_synchronize()?;

        // Launch persistent kernel with all features (Phase 17-18)
        unsafe {
            cuda_ffi::persistent_ndt_launch_raw(
                self.raw_ptr(&self.source_points),
                self.raw_ptr(&self.voxel_means),
                self.raw_ptr(&self.voxel_inv_covs),
                self.raw_ptr(&self.hash_table),
                self.gauss_d1,
                self.gauss_d2,
                self.resolution,
                self.num_points,
                self.num_voxels,
                self.hash_capacity,
                max_iterations as i32,
                transformation_epsilon as f32,
                self.regularization_ref_x,
                self.regularization_ref_y,
                self.config.regularization_scale_factor,
                self.config.regularization_enabled,
                self.config.use_line_search,
                8,                           // ls_num_candidates (default)
                1e-4,                        // ls_mu (Armijo constant)
                0.9,                         // ls_nu (curvature constant)
                self.config.fixed_step_size, // Step size when line search disabled
                self.raw_ptr(&self.persistent_initial_pose),
                self.raw_ptr(&self.persistent_reduce_buffer),
                self.raw_ptr(&self.persistent_out_pose),
                self.raw_ptr(&self.persistent_out_iterations),
                self.raw_ptr(&self.persistent_out_converged),
                self.raw_ptr(&self.persistent_out_score),
                self.raw_ptr(&self.persistent_out_hessian),
                self.raw_ptr(&self.persistent_out_correspondences),
                self.raw_ptr(&self.persistent_out_oscillation),
                self.raw_ptr(&self.persistent_out_alpha_sum),
                debug_enabled,
                debug_ptr,
            )?;
        }

        // Download results
        let pose_bytes = self.client.read_one(self.persistent_out_pose.clone());
        let pose_f32 = f32::from_bytes(&pose_bytes);
        let pose: [f64; 6] = std::array::from_fn(|i| pose_f32[i] as f64);

        let iter_bytes = self.client.read_one(self.persistent_out_iterations.clone());
        let iterations = i32::from_bytes(&iter_bytes)[0] as u32;

        let converged_bytes = self.client.read_one(self.persistent_out_converged.clone());
        let converged = u32::from_bytes(&converged_bytes)[0] != 0;

        let score_bytes = self.client.read_one(self.persistent_out_score.clone());
        let score = f32::from_bytes(&score_bytes)[0] as f64;

        let hess_bytes = self.client.read_one(self.persistent_out_hessian.clone());
        let hessian_flat = f32::from_bytes(&hess_bytes);
        let mut hessian = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                hessian[i][j] = hessian_flat[i * 6 + j] as f64;
            }
        }

        // Download correspondence count
        let corr_bytes = self
            .client
            .read_one(self.persistent_out_correspondences.clone());
        let num_correspondences = u32::from_bytes(&corr_bytes)[0] as usize;

        // Download oscillation count
        let osc_bytes = self
            .client
            .read_one(self.persistent_out_oscillation.clone());
        let oscillation_count = u32::from_bytes(&osc_bytes)[0] as usize;

        // Download alpha sum and compute average (Phase 19.3)
        let alpha_sum_bytes = self.client.read_one(self.persistent_out_alpha_sum.clone());
        let alpha_sum = f32::from_bytes(&alpha_sum_bytes)[0] as f64;
        let avg_alpha = if iterations > 0 {
            alpha_sum / (iterations as f64)
        } else {
            1.0
        };

        // Phase 19.4: Parse debug buffer if enabled
        let iterations_debug = if self.config.enable_debug && iterations > 0 {
            Some(self.parse_debug_buffer(iterations as usize)?)
        } else {
            None
        };

        Ok(FullGpuOptimizationResultV2 {
            pose,
            score,
            converged,
            iterations,
            hessian,
            num_correspondences,
            used_line_search: self.config.use_line_search,
            avg_alpha,
            oscillation_count,
            iterations_debug,
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

        // Create multiple voxels at different positions for well-conditioned Hessian
        // Voxels must be in different grid cells (spacing >= resolution)
        // With resolution=2.0: (0,0,0)->cell(0,0,0), (3,0,0)->cell(1,0,0), (0,3,0)->cell(0,1,0)
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin
                3.0, 0.0, 0.0, // Voxel 1 at (3,0,0)
                0.0, 3.0, 0.0, // Voxel 2 at (0,3,0)
            ],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 1
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 2
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0
                0.0, 0.0, 1.0, // Voxel 1
                0.0, 0.0, 1.0, // Voxel 2
            ],
            valid: vec![1, 1, 1],
            num_voxels: 3,
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

        // Create source points at different positions
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
        ];

        // Create multiple voxels at different positions for well-conditioned Hessian
        // Voxels must be in different grid cells (spacing >= resolution)
        // With resolution=2.0: (0,0,0)->cell(0,0,0), (3,0,0)->cell(1,0,0), (0,3,0)->cell(0,1,0)
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin -> grid cell (0,0,0)
                3.0, 0.0, 0.0, // Voxel 1 at (3,0,0) -> grid cell (1,0,0)
                0.0, 3.0, 0.0, // Voxel 2 at (0,3,0) -> grid cell (0,1,0)
            ],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 1
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 2
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0
                0.0, 0.0, 1.0, // Voxel 1
                0.0, 0.0, 1.0, // Voxel 2
            ],
            valid: vec![1, 1, 1],
            num_voxels: 3,
        };

        // Use resolution=2.0 so each voxel is in its own grid cell
        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        let result = pipeline
            .optimize(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10, 0.01)
            .unwrap();

        // Should run without line search, using fixed_step_size (default 0.1)
        assert!(!result.used_line_search);
        assert!(
            (result.avg_alpha - 0.1).abs() < 1e-6,
            "Without line search, alpha should be fixed_step_size (0.1)"
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

        // Create source points at different positions
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
        ];

        // Create multiple voxels at different positions for well-conditioned Hessian
        // Voxels must be in different grid cells (spacing >= resolution)
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin -> grid cell (0,0,0)
                3.0, 0.0, 0.0, // Voxel 1 at (3,0,0) -> grid cell (1,0,0)
                0.0, 3.0, 0.0, // Voxel 2 at (0,3,0) -> grid cell (0,1,0)
            ],
            inv_covariances: vec![
                0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, // Voxel 0
                0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, // Voxel 1
                0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, // Voxel 2
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0
                0.0, 0.0, 1.0, // Voxel 1
                0.0, 0.0, 1.0, // Voxel 2
            ],
            valid: vec![1, 1, 1],
            num_voxels: 3,
        };

        // Use resolution=2.0 so each voxel is in its own grid cell
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

        // Create source points at different positions
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
        ];

        // Create multiple voxels at different positions for well-conditioned Hessian
        // Voxels must be in different grid cells (spacing >= resolution)
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin -> grid cell (0,0,0)
                3.0, 0.0, 0.0, // Voxel 1 at (3,0,0) -> grid cell (1,0,0)
                0.0, 3.0, 0.0, // Voxel 2 at (0,3,0) -> grid cell (0,1,0)
            ],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 1
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 2
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0
                0.0, 0.0, 1.0, // Voxel 1
                0.0, 0.0, 1.0, // Voxel 2
            ],
            valid: vec![1, 1, 1],
            num_voxels: 3,
        };

        // Use resolution=2.0 so each voxel is in its own grid cell
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

    #[test]
    fn test_pipeline_v2_oscillation_tracking() {
        // Test that oscillation count is tracked in GPU path
        let config = PipelineV2Config {
            use_line_search: true,
            num_candidates: 8,
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        // Create source points
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

        // Run optimization from a small offset
        let result = pipeline
            .optimize(&[0.1, 0.05, 0.0, 0.0, 0.0, 0.0], 30, 0.001)
            .unwrap();

        // Verify that oscillation_count field is populated (not necessarily > 0)
        // The optimization may or may not oscillate depending on the problem geometry
        println!(
            "Oscillation tracking test: {} iterations, converged={}, oscillation_count={}",
            result.iterations, result.converged, result.oscillation_count
        );

        // The field should exist and the result should be reasonable (not some garbage value)
        // Note: oscillation_count is usize, so it's always >= 0. Just verify it's reasonable.
        assert!(
            result.oscillation_count <= result.iterations as usize,
            "Oscillation count should not exceed iteration count"
        );

        // With line search on this geometry, we expect some iterations
        assert!(result.iterations > 0, "Should run at least one iteration");
    }

    #[test]
    fn test_pipeline_v2_debug_collection() {
        // Test that debug data is collected when enabled
        let config = PipelineV2Config {
            use_line_search: false, // Simpler path for testing
            enable_debug: true,
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        // Create source points - need enough points to constrain all DOF
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

        // Run optimization from a very small offset (to ensure convergence)
        let result = pipeline
            .optimize(&[0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 10, 0.01)
            .unwrap();

        // Verify debug data is populated
        assert!(
            result.iterations_debug.is_some(),
            "Debug data should be collected when enable_debug is true"
        );

        let debug_vec = result.iterations_debug.as_ref().unwrap();
        assert_eq!(
            debug_vec.len(),
            result.iterations as usize,
            "Should have debug entry for each iteration"
        );

        // Verify first iteration has sensible data
        if !debug_vec.is_empty() {
            let first = &debug_vec[0];
            assert_eq!(first.iteration, 0);
            assert_eq!(first.pose.len(), 6);
            assert_eq!(first.gradient.len(), 6);
            assert_eq!(first.hessian.len(), 36);
            assert_eq!(first.newton_step.len(), 6);

            println!(
                "Debug test: {} iterations, first iter score={:.4}, step_len={:.6}",
                debug_vec.len(),
                first.score,
                first.step_length
            );

            // Print all iterations for verification
            for iter_debug in debug_vec {
                println!(
                    "  iter={} score={:.4} step={:.6} reversed={}",
                    iter_debug.iteration,
                    iter_debug.score,
                    iter_debug.step_length,
                    iter_debug.direction_reversed
                );
            }
        }
    }

    #[test]
    fn test_pipeline_v2_debug_disabled() {
        // Test that debug data is NOT collected when disabled (default)
        let config = PipelineV2Config {
            use_line_search: false,
            enable_debug: false, // Explicitly disabled
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        // Create source points at different positions
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
        ];

        // Create multiple voxels at different positions for well-conditioned Hessian
        // Voxels must be in different grid cells (spacing >= resolution)
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin -> grid cell (0,0,0)
                3.0, 0.0, 0.0, // Voxel 1 at (3,0,0) -> grid cell (1,0,0)
                0.0, 3.0, 0.0, // Voxel 2 at (0,3,0) -> grid cell (0,1,0)
            ],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 1
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 2
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0
                0.0, 0.0, 1.0, // Voxel 1
                0.0, 0.0, 1.0, // Voxel 2
            ],
            valid: vec![1, 1, 1],
            num_voxels: 3,
        };

        // Use resolution=2.0 so each voxel is in its own grid cell
        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        let result = pipeline
            .optimize(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5, 0.01)
            .unwrap();

        // Verify debug data is NOT populated when disabled
        assert!(
            result.iterations_debug.is_none(),
            "Debug data should be None when enable_debug is false"
        );
    }

    #[test]
    fn test_persistent_kernel_optimization() {
        // Test the persistent kernel optimization path
        let config = PipelineV2Config {
            use_line_search: false,
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(1000, 5000, config).unwrap();

        // Points in a sphere around origin
        let source_points: Vec<[f32; 3]> = (0..300)
            .map(|i| {
                let phi = (i as f32) * 0.1;
                let theta = (i as f32) * 0.05;
                let r = 0.5;
                [
                    r * phi.sin() * theta.cos(),
                    r * phi.sin() * theta.sin(),
                    r * phi.cos(),
                ]
            })
            .collect();

        // Create multiple voxels at different positions for well-conditioned Hessian
        let voxel_data = GpuVoxelData {
            means: vec![
                0.0, 0.0, 0.0, // Voxel 0 at origin
                3.0, 0.0, 0.0, // Voxel 1 at (3,0,0)
                0.0, 3.0, 0.0, // Voxel 2 at (0,3,0)
            ],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 1
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Voxel 2
            ],
            principal_axes: vec![
                0.0, 0.0, 1.0, // Voxel 0
                0.0, 0.0, 1.0, // Voxel 1
                0.0, 0.0, 1.0, // Voxel 2
            ],
            valid: vec![1, 1, 1],
            num_voxels: 3,
        };

        // Upload alignment data
        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        // Run optimization
        let initial_pose = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = pipeline.optimize(&initial_pose, 30, 0.01).unwrap();

        println!(
            "Persistent kernel test: {} iterations, converged={}, score={:.4}",
            result.iterations, result.converged, result.score
        );
        println!("  Final pose: {:?}", result.pose);

        // Basic sanity checks
        assert!(result.iterations > 0, "Should run at least one iteration");
        assert!(result.score.is_finite(), "Score should be finite");
        assert!(!result.used_line_search, "Should not use line search");
    }
}
