//! Full GPU Newton Pipeline V2 with Graph-Based Kernels.
//!
//! This module implements a complete GPU Newton optimization pipeline using
//! separated kernels that can work on all CUDA GPUs, including those with
//! limited SM count (e.g., Jetson Orin) where cooperative launch fails.
//!
//! # Architecture (Phase 24)
//!
//! The optimization is split into 5 separate kernels:
//! - K1: Init - Initialize state from initial pose
//! - K2: Compute - Per-point score/gradient/Hessian + block reduction
//! - K3: Solve - Newton solve + regularization
//! - K4: LineSearch - Parallel line search evaluation (optional)
//! - K5: Update - Apply step, check convergence
//!
//! ```text
//! Once at start (upload ~500 KB):
//!   Upload: source_points [N×3], voxel_data [V×12], initial_pose [6]
//!
//! GPU Kernels per iteration:
//!   K2: ndt_graph_compute_kernel - Per-point compute + block reduction + atomic global
//!   K3: ndt_graph_solve_kernel - Newton solve (Cholesky/SVD), regularization
//!   K4: ndt_graph_linesearch_kernel - Evaluate candidates (if enabled)
//!   K5: ndt_graph_update_kernel - Apply step, check convergence, prepare next iter
//!
//! Once at end:
//!   Download: final_pose, score, H, correspondences, oscillation count
//! ```
//!
//! # Benefits over Cooperative Kernel
//!
//! - Works on all CUDA GPUs (no cooperative launch limit)
//! - Same algorithm and numerical results
//! - Slightly higher launch overhead (~10-20μs/iter) but eliminates grid size restrictions

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;
#[cfg(feature = "profiling")]
use tracing::debug;

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
    /// Per-iteration debug data (only available when `debug-iteration` feature is enabled)
    #[cfg(feature = "debug-iteration")]
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
        }
    }
}

/// Full GPU Newton Pipeline V2 using graph-based kernels.
///
/// This pipeline uses separated kernels (K1-K5) that can run on all CUDA GPUs,
/// including those with limited SM count where cooperative launch fails.
/// The kernels are executed sequentially with implicit synchronization.
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
    // Graph-based kernel buffers (Phase 24)
    // ========================================================================
    graph_initial_pose: Handle,  // [6] input pose for graph kernels
    graph_state_buffer: Handle,  // [102] persistent state across iterations
    graph_reduce_buffer: Handle, // [29] per-iteration reduction accumulator
    graph_ls_buffer: Handle,     // [68] line search state
    graph_output_buffer: Handle, // [48] final output

    // Phase 19.4: Debug buffer (only allocated when debug-iteration feature is enabled)
    #[cfg(feature = "debug-iteration")]
    graph_debug_buffer: Option<Handle>, // [max_iterations * 50] f32
    #[cfg(feature = "debug-iteration")]
    max_iterations_for_debug: u32, // Cached for buffer sizing

    // Regularization state - Phase 18.2
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

        // Graph-based kernel buffers (Phase 24)
        let graph_initial_pose = client.empty(6 * std::mem::size_of::<f32>());
        let graph_state_buffer =
            client.empty(cuda_ffi::GRAPH_NDT_STATE_BUFFER_SIZE * std::mem::size_of::<f32>());
        let graph_reduce_buffer =
            client.empty(cuda_ffi::GRAPH_NDT_REDUCE_BUFFER_SIZE * std::mem::size_of::<f32>());
        let graph_ls_buffer =
            client.empty(cuda_ffi::GRAPH_NDT_LS_BUFFER_SIZE * std::mem::size_of::<f32>());
        let graph_output_buffer =
            client.empty(cuda_ffi::GRAPH_NDT_OUTPUT_BUFFER_SIZE * std::mem::size_of::<f32>());

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
            graph_initial_pose,
            graph_state_buffer,
            graph_reduce_buffer,
            graph_ls_buffer,
            graph_output_buffer,
            #[cfg(feature = "debug-iteration")]
            graph_debug_buffer: None, // Allocated on-demand in optimize()
            #[cfg(feature = "debug-iteration")]
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

        // Debug: count hash entries to verify table is populated
        #[cfg(feature = "profiling")]
        {
            let entry_count =
                unsafe { cuda_ffi::hash_table_count_entries(hash_table_ptr, self.hash_capacity)? };
            debug!(
                entry_count,
                num_voxels,
                capacity = self.hash_capacity,
                "Hash table populated"
            );
        }

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
    #[cfg(feature = "debug-iteration")]
    fn parse_debug_buffer(
        &self,
        num_iterations: usize,
    ) -> Result<Vec<super::debug::IterationDebug>> {
        use super::debug::IterationDebug;

        let buffer = self
            .graph_debug_buffer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Debug buffer not allocated"))?;
        let bytes = self.client.read_one(buffer.clone());
        let floats = f32::from_bytes(&bytes);

        const FLOATS_PER_ITER: usize = cuda_ffi::GRAPH_NDT_DEBUG_FLOATS_PER_ITER;

        // Cap iterations to what the buffer can hold to prevent out-of-bounds access
        let max_parseable = floats.len() / FLOATS_PER_ITER;
        let num_iterations = num_iterations.min(max_parseable);

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
    #[cfg(feature = "debug-iteration")]
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

    /// Run full GPU Newton optimization using graph-based kernels.
    ///
    /// This runs the Newton optimization loop using separated kernels (K1-K5)
    /// that work on all CUDA GPUs. Supports all features:
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
                #[cfg(feature = "debug-iteration")]
                iterations_debug: None,
            });
        }

        // Upload initial pose
        let pose_f32: [f32; 6] = initial_pose.map(|x| x as f32);
        self.graph_initial_pose = self.client.create(f32::as_bytes(&pose_f32));

        // Recreate buffers to ensure fresh memory (avoids CubeCL caching issues)
        let state_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_STATE_BUFFER_SIZE];
        let reduce_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_REDUCE_BUFFER_SIZE];
        let ls_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_LS_BUFFER_SIZE];
        let output_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_OUTPUT_BUFFER_SIZE];

        self.graph_state_buffer = self.client.create(f32::as_bytes(&state_zeros));
        self.graph_reduce_buffer = self.client.create(f32::as_bytes(&reduce_zeros));
        self.graph_ls_buffer = self.client.create(f32::as_bytes(&ls_zeros));
        self.graph_output_buffer = self.client.create(f32::as_bytes(&output_zeros));

        // Force CubeCL to sync all pending operations before kernel launch
        let _ = self.client.read_one(self.graph_state_buffer.clone());

        // Phase 19.4: Allocate debug buffer (only when debug-iteration feature is enabled)
        #[cfg(feature = "debug-iteration")]
        let debug_ptr = {
            let buffer_size = max_iterations as usize
                * cuda_ffi::GRAPH_NDT_DEBUG_FLOATS_PER_ITER
                * std::mem::size_of::<f32>();
            self.graph_debug_buffer = Some(self.client.empty(buffer_size));
            self.max_iterations_for_debug = max_iterations;
            self.raw_ptr(self.graph_debug_buffer.as_ref().unwrap())
        };
        #[cfg(not(feature = "debug-iteration"))]
        let debug_ptr = 0u64;

        // Ensure all CubeCL operations are complete before kernel launches
        cuda_ffi::cuda_device_synchronize()?;

        // Build configuration for graph kernels
        let config = cuda_ffi::GraphNdtConfig::new(
            self.gauss_d1,
            self.gauss_d2,
            self.resolution,
            transformation_epsilon as f32,
            self.num_points as u32,
            self.num_voxels as u32,
            self.hash_capacity,
            max_iterations as i32,
        );

        // Apply configuration options
        let config = if self.config.regularization_enabled {
            config.with_regularization(
                self.regularization_ref_x,
                self.regularization_ref_y,
                self.config.regularization_scale_factor,
            )
        } else {
            config
        };

        let config = if self.config.use_line_search {
            config.with_line_search(true, 8, self.config.armijo_mu, self.config.wolfe_nu)
        } else {
            config.with_fixed_step(self.config.fixed_step_size)
        };

        #[cfg(feature = "debug-iteration")]
        let config = config.with_debug(true);
        #[cfg(not(feature = "debug-iteration"))]
        let config = config.with_debug(false);

        // Get raw device pointers
        let d_source_points = self.raw_ptr(&self.source_points);
        let d_voxel_means = self.raw_ptr(&self.voxel_means);
        let d_voxel_inv_covs = self.raw_ptr(&self.voxel_inv_covs);
        let d_hash_table = self.raw_ptr(&self.hash_table);
        let d_initial_pose = self.raw_ptr(&self.graph_initial_pose);
        let d_state_buffer = self.raw_ptr(&self.graph_state_buffer);
        let d_reduce_buffer = self.raw_ptr(&self.graph_reduce_buffer);
        let d_ls_buffer = self.raw_ptr(&self.graph_ls_buffer);
        let d_output_buffer = self.raw_ptr(&self.graph_output_buffer);

        // K1: Initialize state from initial pose
        unsafe {
            cuda_ffi::graph_ndt_launch_init_raw(
                d_initial_pose,
                d_state_buffer,
                d_reduce_buffer,
                d_ls_buffer,
                None, // Use default stream
            )?;
        }
        cuda_ffi::cuda_device_synchronize()?;

        // Iteration loop with K2-K5
        for _ in 0..max_iterations {
            unsafe {
                cuda_ffi::graph_ndt_run_iteration_raw(
                    d_source_points,
                    d_voxel_means,
                    d_voxel_inv_covs,
                    d_hash_table,
                    &config,
                    d_state_buffer,
                    d_reduce_buffer,
                    d_ls_buffer,
                    d_output_buffer,
                    debug_ptr,
                    None, // Use default stream
                )?;
            }
            cuda_ffi::cuda_device_synchronize()?;

            // Check convergence
            let converged = unsafe { cuda_ffi::graph_ndt_check_converged(d_state_buffer)? };
            if converged {
                break;
            }
        }

        // Download results from output buffer
        let output_bytes = self.client.read_one(self.graph_output_buffer.clone());
        let output = f32::from_bytes(&output_bytes);

        // Parse output buffer (see ndt_graph_common.cuh OutputOffset)
        let pose: [f64; 6] = std::array::from_fn(|i| output[i] as f64);
        let iterations = output[6] as u32;
        let converged = output[7] > 0.5;
        let score = output[8] as f64;

        // Extract Hessian (indices 9-44)
        let mut hessian = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                hessian[i][j] = output[9 + i * 6 + j] as f64;
            }
        }

        let num_correspondences = output[45] as usize;
        let oscillation_count = output[46] as usize;
        let avg_alpha = output[47] as f64;

        // Phase 19.4: Parse debug buffer (only when debug-iteration feature is enabled)
        #[cfg(feature = "debug-iteration")]
        let iterations_debug = if iterations > 0 {
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
            #[cfg(feature = "debug-iteration")]
            iterations_debug,
        })
    }

    /// Run optimization with kernel-level profiling.
    ///
    /// Returns both the optimization result and detailed timing breakdown.
    /// This uses CUDA events to measure each kernel's execution time.
    pub fn optimize_profiled(
        &mut self,
        initial_pose: &[f64; 6],
        max_iterations: u32,
        transformation_epsilon: f64,
    ) -> Result<(FullGpuOptimizationResultV2, cuda_ffi::GraphNdtProfile)> {
        use cuda_ffi::async_stream::CudaEvent;
        use std::time::Instant;

        if self.num_points == 0 {
            let mut profile = cuda_ffi::GraphNdtProfile::new();
            profile.iterations = 0;
            return Ok((
                FullGpuOptimizationResultV2 {
                    pose: *initial_pose,
                    score: 0.0,
                    converged: true,
                    iterations: 0,
                    hessian: [[0.0; 6]; 6],
                    num_correspondences: 0,
                    used_line_search: false,
                    avg_alpha: 1.0,
                    oscillation_count: 0,
                    #[cfg(feature = "debug-iteration")]
                    iterations_debug: None,
                },
                profile,
            ));
        }

        let start_time = Instant::now();
        let mut profile = cuda_ffi::GraphNdtProfile::new();

        // Create timing events
        let event_start = CudaEvent::new()?;
        let event_end = CudaEvent::new()?;

        // Upload initial pose
        let pose_f32: [f32; 6] = initial_pose.map(|x| x as f32);
        self.graph_initial_pose = self.client.create(f32::as_bytes(&pose_f32));

        // Recreate buffers to ensure fresh memory
        let state_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_STATE_BUFFER_SIZE];
        let reduce_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_REDUCE_BUFFER_SIZE];
        let ls_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_LS_BUFFER_SIZE];
        let output_zeros = [0.0f32; cuda_ffi::GRAPH_NDT_OUTPUT_BUFFER_SIZE];

        self.graph_state_buffer = self.client.create(f32::as_bytes(&state_zeros));
        self.graph_reduce_buffer = self.client.create(f32::as_bytes(&reduce_zeros));
        self.graph_ls_buffer = self.client.create(f32::as_bytes(&ls_zeros));
        self.graph_output_buffer = self.client.create(f32::as_bytes(&output_zeros));

        // Force CubeCL to sync all pending operations
        let _ = self.client.read_one(self.graph_state_buffer.clone());
        cuda_ffi::cuda_device_synchronize()?;

        // Build configuration for graph kernels
        let config = cuda_ffi::GraphNdtConfig::new(
            self.gauss_d1,
            self.gauss_d2,
            self.resolution,
            transformation_epsilon as f32,
            self.num_points as u32,
            self.num_voxels as u32,
            self.hash_capacity,
            max_iterations as i32,
        );

        let config = if self.config.regularization_enabled {
            config.with_regularization(
                self.regularization_ref_x,
                self.regularization_ref_y,
                self.config.regularization_scale_factor,
            )
        } else {
            config
        };

        let config = if self.config.use_line_search {
            config.with_line_search(true, 8, self.config.armijo_mu, self.config.wolfe_nu)
        } else {
            config.with_fixed_step(self.config.fixed_step_size)
        };

        // Get raw device pointers
        let d_source_points = self.raw_ptr(&self.source_points);
        let d_voxel_means = self.raw_ptr(&self.voxel_means);
        let d_voxel_inv_covs = self.raw_ptr(&self.voxel_inv_covs);
        let d_hash_table = self.raw_ptr(&self.hash_table);
        let d_initial_pose = self.raw_ptr(&self.graph_initial_pose);
        let d_state_buffer = self.raw_ptr(&self.graph_state_buffer);
        let d_reduce_buffer = self.raw_ptr(&self.graph_reduce_buffer);
        let d_ls_buffer = self.raw_ptr(&self.graph_ls_buffer);
        let d_output_buffer = self.raw_ptr(&self.graph_output_buffer);

        // K1: Initialize with timing
        event_start.record_default()?;
        unsafe {
            cuda_ffi::graph_ndt_launch_init_raw(
                d_initial_pose,
                d_state_buffer,
                d_reduce_buffer,
                d_ls_buffer,
                None,
            )?;
        }
        event_end.record_default()?;
        event_end.synchronize()?;
        profile.init.total_ms += event_end.elapsed_time(&event_start)?;
        profile.init.count += 1;

        // Iteration loop with per-kernel timing
        for _ in 0..max_iterations {
            // K2: Compute
            event_start.record_default()?;
            unsafe {
                cuda_ffi::graph_ndt_launch_compute_raw(
                    d_source_points,
                    d_voxel_means,
                    d_voxel_inv_covs,
                    d_hash_table,
                    &config,
                    d_state_buffer,
                    d_reduce_buffer,
                    None,
                )?;
            }
            event_end.record_default()?;
            event_end.synchronize()?;
            profile.compute.total_ms += event_end.elapsed_time(&event_start)?;
            profile.compute.count += 1;

            // K3: Solve
            event_start.record_default()?;
            unsafe {
                cuda_ffi::graph_ndt_launch_solve_raw(
                    &config,
                    d_state_buffer,
                    d_reduce_buffer,
                    d_ls_buffer,
                    d_output_buffer,
                    None,
                )?;
            }
            event_end.record_default()?;
            event_end.synchronize()?;
            profile.solve.total_ms += event_end.elapsed_time(&event_start)?;
            profile.solve.count += 1;

            // K4: Line search (if enabled)
            if self.config.use_line_search {
                event_start.record_default()?;
                unsafe {
                    cuda_ffi::graph_ndt_launch_linesearch_raw(
                        d_source_points,
                        d_voxel_means,
                        d_voxel_inv_covs,
                        d_hash_table,
                        &config,
                        d_state_buffer,
                        d_ls_buffer,
                        None,
                    )?;
                }
                event_end.record_default()?;
                event_end.synchronize()?;
                profile.linesearch.total_ms += event_end.elapsed_time(&event_start)?;
                profile.linesearch.count += 1;
            }

            // K5: Update
            event_start.record_default()?;
            unsafe {
                cuda_ffi::graph_ndt_launch_update_raw(
                    &config,
                    d_state_buffer,
                    d_reduce_buffer,
                    d_ls_buffer,
                    d_output_buffer,
                    0, // No debug buffer for profiling
                    None,
                )?;
            }
            event_end.record_default()?;
            event_end.synchronize()?;
            profile.update.total_ms += event_end.elapsed_time(&event_start)?;
            profile.update.count += 1;

            // Check convergence
            let converged = unsafe { cuda_ffi::graph_ndt_check_converged(d_state_buffer)? };
            if converged {
                break;
            }
        }

        profile.iterations = profile.compute.count;
        profile.total_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Download results from output buffer
        let output_bytes = self.client.read_one(self.graph_output_buffer.clone());
        let output = f32::from_bytes(&output_bytes);

        // Parse output buffer
        let pose: [f64; 6] = std::array::from_fn(|i| output[i] as f64);
        let iterations = output[6] as u32;
        let converged = output[7] > 0.5;
        let score = output[8] as f64;

        // Extract Hessian
        let mut hessian = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                hessian[i][j] = output[9 + i * 6 + j] as f64;
            }
        }

        let num_correspondences = output[45] as usize;
        let oscillation_count = output[46] as usize;
        let avg_alpha = output[47] as f64;

        Ok((
            FullGpuOptimizationResultV2 {
                pose,
                score,
                converged,
                iterations,
                hessian,
                num_correspondences,
                used_line_search: self.config.use_line_search,
                avg_alpha,
                oscillation_count,
                #[cfg(feature = "debug-iteration")]
                iterations_debug: None, // Skip debug parsing in profiling mode
            },
            profile,
        ))
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
        crate::test_println!(
            "Multiple points test: {} iterations, converged={}, score={}",
            result.iterations,
            result.converged,
            result.score
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
        crate::test_println!(
            "No line search test: {} iterations, converged={}",
            result.iterations,
            result.converged
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
        crate::test_println!(
            "With line search test: {} iterations, converged={}, avg_alpha={}",
            result.iterations,
            result.converged,
            result.avg_alpha
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

        crate::test_println!(
            "Regularization test: {} iterations, converged={}, score={}, pose=({:.4}, {:.4})",
            result.iterations,
            result.converged,
            result.score,
            result.pose[0],
            result.pose[1]
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

        crate::test_println!(
            "Regularization disabled test: {} iterations, pose=({:.4}, {:.4})",
            result.iterations,
            result.pose[0],
            result.pose[1]
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
        crate::test_println!(
            "Oscillation tracking test: {} iterations, converged={}, oscillation_count={}",
            result.iterations,
            result.converged,
            result.oscillation_count
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

    #[cfg(feature = "debug-iteration")]
    #[test]
    fn test_pipeline_v2_debug_collection() {
        // Test that debug data is collected when debug-iteration feature is enabled
        let config = PipelineV2Config {
            use_line_search: false, // Simpler path for testing
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
            "Debug data should be collected when debug-iteration feature is enabled"
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

            crate::test_println!(
                "Debug test: {} iterations, first iter score={:.4}, step_len={:.6}",
                debug_vec.len(),
                first.score,
                first.step_length
            );

            // Print all iterations for verification
            for _iter_debug in debug_vec {
                crate::test_println!(
                    "  iter={} score={:.4} step={:.6} reversed={}",
                    _iter_debug.iteration,
                    _iter_debug.score,
                    _iter_debug.step_length,
                    _iter_debug.direction_reversed
                );
            }
        }
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

        crate::test_println!(
            "Persistent kernel test: {} iterations, converged={}, score={:.4}",
            result.iterations,
            result.converged,
            result.score
        );
        crate::test_println!("  Final pose: {:?}", result.pose);

        // Basic sanity checks
        assert!(result.iterations > 0, "Should run at least one iteration");
        assert!(result.score.is_finite(), "Score should be finite");
        assert!(!result.used_line_search, "Should not use line search");
    }

    /// Benchmark test for Phase 24.4 - profiles kernel execution times.
    #[test]
    fn test_pipeline_v2_profiled() {
        let config = PipelineV2Config {
            use_line_search: true,
            ..Default::default()
        };
        let mut pipeline = FullGpuPipelineV2::with_config(10000, 5000, config).unwrap();

        // Create a larger point cloud for more realistic timing
        let source_points: Vec<[f32; 3]> = (0..5000)
            .map(|i| {
                let phi = (i as f32) * 0.02;
                let theta = (i as f32) * 0.01;
                let r = 1.0 + (i % 100) as f32 * 0.01;
                [
                    r * phi.sin() * theta.cos(),
                    r * phi.sin() * theta.sin(),
                    r * phi.cos(),
                ]
            })
            .collect();

        // Create a grid of voxels
        let mut means = Vec::new();
        let mut inv_covs = Vec::new();
        let mut principal_axes = Vec::new();
        let mut valid = Vec::new();

        for x in -5..5 {
            for y in -5..5 {
                for z in -2..2 {
                    means.extend_from_slice(&[x as f32 * 2.0, y as f32 * 2.0, z as f32 * 2.0]);
                    inv_covs.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
                    principal_axes.extend_from_slice(&[0.0, 0.0, 1.0]);
                    valid.push(1);
                }
            }
        }

        let voxel_data = GpuVoxelData {
            means,
            inv_covariances: inv_covs,
            principal_axes,
            valid: valid.clone(),
            num_voxels: valid.len(),
        };

        // Upload alignment data
        pipeline
            .upload_alignment_data(&source_points, &voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        // Run profiled optimization
        let initial_pose = [0.1, 0.05, 0.0, 0.0, 0.0, 0.02];
        let (result, profile) = pipeline.optimize_profiled(&initial_pose, 30, 0.01).unwrap();

        // Print profiling report
        #[cfg(feature = "test-verbose")]
        profile.print_report();

        crate::test_println!("\n=== Phase 24.4 Profiling Results ===");
        crate::test_println!(
            "Total: {:.3} ms, {} iterations, {:.3} ms/iter",
            profile.total_ms,
            result.iterations,
            profile.per_iteration_ms()
        );
        crate::test_println!("Kernel breakdown per iteration (avg):");
        crate::test_println!("  Compute:    {:.3} ms", profile.compute.avg_ms());
        crate::test_println!("  Solve:      {:.3} ms", profile.solve.avg_ms());
        crate::test_println!("  LineSearch: {:.3} ms", profile.linesearch.avg_ms());
        crate::test_println!("  Update:     {:.3} ms", profile.update.avg_ms());
        crate::test_println!(
            "Kernel efficiency: {:.1}% (kernel time / total time)",
            100.0 * profile.kernel_total_ms() / profile.total_ms
        );

        // Basic sanity checks
        assert!(result.iterations > 0, "Should run at least one iteration");
        assert!(
            profile.compute.count > 0,
            "Should have compute kernel calls"
        );
        assert!(profile.solve.count > 0, "Should have solve kernel calls");
        assert!(
            profile.linesearch.count > 0,
            "Should have line search kernel calls"
        );
        assert!(profile.update.count > 0, "Should have update kernel calls");
        assert!(
            profile.kernel_total_ms() > 0.0,
            "Should have non-zero kernel time"
        );
    }
}
