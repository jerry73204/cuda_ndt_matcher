//! CUDA Graph-based NDT kernel FFI bindings.
//!
//! This module provides Rust bindings to the CUDA Graph-based NDT kernels,
//! which replace the cooperative groups persistent kernel on GPUs with
//! limited SM count (e.g., Jetson Orin).
//!
//! # Architecture
//!
//! The optimization is split into 5 separate kernels:
//! - K1: Init - Initialize state from initial pose
//! - K2: Compute - Per-point score/gradient/Hessian + block reduction
//! - K3: Solve - Newton solve + regularization
//! - K4: LineSearch - Parallel line search evaluation (optional)
//! - K5: Update - Apply step, check convergence
//!
//! These can be executed either:
//! 1. Individually via `ndt_graph_launch_*` functions
//! 2. As a captured CUDA Graph for reduced launch overhead
//!
//! # Example
//!
//! ```ignore
//! use cuda_ffi::graph_ndt::{GraphNdtConfig, GraphNdtPipeline};
//!
//! // Create config
//! let config = GraphNdtConfig::new(gauss_d1, gauss_d2, resolution, num_points, ...);
//!
//! // Create pipeline (allocates buffers)
//! let pipeline = GraphNdtPipeline::new(&config)?;
//!
//! // Run optimization
//! let result = pipeline.align(
//!     source_points, voxel_means, voxel_inv_covs, hash_table,
//!     initial_pose, max_iterations
//! )?;
//! ```

use crate::async_stream::CudaStream;
use crate::radix_sort::{check_cuda, CudaError};
use std::ffi::c_int;

// ============================================================================
// Buffer size constants (must match ndt_graph_common.cuh)
// ============================================================================

/// State buffer size in floats
pub const STATE_BUFFER_SIZE: usize = 102;

/// Reduce buffer size in floats
pub const REDUCE_BUFFER_SIZE: usize = 29;

/// Line search buffer size in floats
pub const LS_BUFFER_SIZE: usize = 68;

/// Output buffer size in floats
pub const OUTPUT_BUFFER_SIZE: usize = 48;

/// Debug buffer size per iteration in floats
pub const DEBUG_FLOATS_PER_ITER: usize = 50;

/// Block size for compute/linesearch kernels
pub const BLOCK_SIZE: usize = 256;

// ============================================================================
// Configuration structure (must match ndt_graph_common.cuh)
// ============================================================================

/// Configuration for graph-based NDT optimization.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GraphNdtConfig {
    // NDT parameters
    pub gauss_d1: f32,
    pub gauss_d2: f32,
    pub resolution: f32,
    pub epsilon_sq: f32,

    // Regularization parameters
    pub reg_ref_x: f32,
    pub reg_ref_y: f32,
    pub reg_scale: f32,
    pub reg_enabled: i32,

    // Line search parameters
    pub ls_enabled: i32,
    pub ls_num_candidates: i32,
    pub ls_mu: f32,
    pub ls_nu: f32,
    pub fixed_step_size: f32,

    // Data sizes
    pub num_points: u32,
    pub num_voxels: u32,
    pub hash_capacity: u32,
    pub max_iterations: i32,

    // Debug
    pub debug_enabled: i32,
}

impl GraphNdtConfig {
    /// Create a new configuration with default line search parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
        epsilon: f32,
        num_points: u32,
        num_voxels: u32,
        hash_capacity: u32,
        max_iterations: i32,
    ) -> Self {
        Self {
            gauss_d1,
            gauss_d2,
            resolution,
            epsilon_sq: epsilon * epsilon,
            reg_ref_x: 0.0,
            reg_ref_y: 0.0,
            reg_scale: 0.0,
            reg_enabled: 0,
            ls_enabled: 1,
            ls_num_candidates: 8,
            ls_mu: 1e-4,
            ls_nu: 0.9,
            fixed_step_size: 0.1,
            num_points,
            num_voxels,
            hash_capacity,
            max_iterations,
            debug_enabled: 0,
        }
    }

    /// Enable GNSS regularization.
    pub fn with_regularization(mut self, ref_x: f32, ref_y: f32, scale: f32) -> Self {
        self.reg_ref_x = ref_x;
        self.reg_ref_y = ref_y;
        self.reg_scale = scale;
        self.reg_enabled = 1;
        self
    }

    /// Configure line search.
    pub fn with_line_search(
        mut self,
        enabled: bool,
        num_candidates: i32,
        mu: f32,
        nu: f32,
    ) -> Self {
        self.ls_enabled = if enabled { 1 } else { 0 };
        self.ls_num_candidates = num_candidates;
        self.ls_mu = mu;
        self.ls_nu = nu;
        self
    }

    /// Disable line search and use fixed step size.
    pub fn with_fixed_step(mut self, step_size: f32) -> Self {
        self.ls_enabled = 0;
        self.fixed_step_size = step_size;
        self
    }

    /// Enable debug output.
    pub fn with_debug(mut self, enabled: bool) -> Self {
        self.debug_enabled = if enabled { 1 } else { 0 };
        self
    }
}

// ============================================================================
// Output structure
// ============================================================================

/// Output from graph-based NDT optimization.
#[derive(Debug, Clone)]
pub struct GraphNdtOutput {
    /// Final optimized pose (x, y, z, roll, pitch, yaw)
    pub pose: [f32; 6],
    /// Number of iterations run
    pub iterations: i32,
    /// Whether optimization converged
    pub converged: bool,
    /// Final NDT score
    pub score: f32,
    /// Final 6x6 Hessian matrix (row-major)
    pub hessian: [f32; 36],
    /// Number of point-voxel correspondences
    pub num_correspondences: u32,
    /// Maximum oscillation count observed
    pub max_oscillation_count: u32,
    /// Average step size (alpha)
    pub avg_alpha: f32,
}

// ============================================================================
// FFI declarations
// ============================================================================

extern "C" {
    fn ndt_graph_get_buffer_sizes(
        state_size: *mut u32,
        reduce_size: *mut u32,
        ls_size: *mut u32,
        output_size: *mut u32,
    ) -> c_int;

    fn ndt_graph_compute_shared_mem_size() -> u32;
    fn ndt_graph_linesearch_shared_mem_size() -> u32;

    fn ndt_graph_launch_init(
        initial_pose: *const f32,
        state_buffer: *mut f32,
        reduce_buffer: *mut f32,
        ls_buffer: *mut f32,
        stream: *mut std::ffi::c_void,
    ) -> c_int;

    fn ndt_graph_launch_compute(
        source_points: *const f32,
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        config: *const GraphNdtConfig,
        state_buffer: *const f32,
        reduce_buffer: *mut f32,
        num_points: u32,
        stream: *mut std::ffi::c_void,
    ) -> c_int;

    fn ndt_graph_launch_solve(
        config: *const GraphNdtConfig,
        state_buffer: *mut f32,
        reduce_buffer: *mut f32,
        ls_buffer: *mut f32,
        output_buffer: *mut f32,
        stream: *mut std::ffi::c_void,
    ) -> c_int;

    fn ndt_graph_launch_linesearch(
        source_points: *const f32,
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        config: *const GraphNdtConfig,
        state_buffer: *const f32,
        ls_buffer: *mut f32,
        num_points: u32,
        stream: *mut std::ffi::c_void,
    ) -> c_int;

    fn ndt_graph_launch_update(
        config: *const GraphNdtConfig,
        state_buffer: *mut f32,
        reduce_buffer: *mut f32,
        ls_buffer: *mut f32,
        output_buffer: *mut f32,
        debug_buffer: *mut f32,
        stream: *mut std::ffi::c_void,
    ) -> c_int;

    fn ndt_graph_check_converged(state_buffer: *const f32, converged: *mut bool) -> c_int;

    fn ndt_graph_get_iterations(state_buffer: *const f32, iterations: *mut i32) -> c_int;

    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
}

const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// ============================================================================
// Public API functions
// ============================================================================

/// Get required buffer sizes in bytes.
pub fn get_buffer_sizes() -> Result<(usize, usize, usize, usize), CudaError> {
    let mut state_size: u32 = 0;
    let mut reduce_size: u32 = 0;
    let mut ls_size: u32 = 0;
    let mut output_size: u32 = 0;

    unsafe {
        check_cuda(ndt_graph_get_buffer_sizes(
            &mut state_size,
            &mut reduce_size,
            &mut ls_size,
            &mut output_size,
        ))?;
    }

    Ok((
        state_size as usize,
        reduce_size as usize,
        ls_size as usize,
        output_size as usize,
    ))
}

/// Get shared memory size for compute kernel.
pub fn compute_shared_mem_size() -> usize {
    unsafe { ndt_graph_compute_shared_mem_size() as usize }
}

/// Get shared memory size for line search kernel.
pub fn linesearch_shared_mem_size() -> usize {
    unsafe { ndt_graph_linesearch_shared_mem_size() as usize }
}

/// Calculate number of blocks needed for given point count.
pub fn num_blocks(num_points: usize) -> usize {
    num_points.div_ceil(BLOCK_SIZE)
}

// ============================================================================
// Raw pointer API (for CubeCL interop)
// ============================================================================

/// Launch init kernel using raw device pointers.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_launch_init_raw(
    d_initial_pose: u64,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    d_ls_buffer: u64,
    stream: Option<&CudaStream>,
) -> Result<(), CudaError> {
    let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    check_cuda(ndt_graph_launch_init(
        d_initial_pose as *const f32,
        d_state_buffer as *mut f32,
        d_reduce_buffer as *mut f32,
        d_ls_buffer as *mut f32,
        stream_ptr,
    ))
}

/// Launch compute kernel using raw device pointers.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_launch_compute_raw(
    d_source_points: u64,
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    config: &GraphNdtConfig,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    stream: Option<&CudaStream>,
) -> Result<(), CudaError> {
    let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    check_cuda(ndt_graph_launch_compute(
        d_source_points as *const f32,
        d_voxel_means as *const f32,
        d_voxel_inv_covs as *const f32,
        d_hash_table as *const std::ffi::c_void,
        config,
        d_state_buffer as *const f32,
        d_reduce_buffer as *mut f32,
        config.num_points,
        stream_ptr,
    ))
}

/// Launch solve kernel using raw device pointers.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_launch_solve_raw(
    config: &GraphNdtConfig,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    d_ls_buffer: u64,
    d_output_buffer: u64,
    stream: Option<&CudaStream>,
) -> Result<(), CudaError> {
    let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    check_cuda(ndt_graph_launch_solve(
        config,
        d_state_buffer as *mut f32,
        d_reduce_buffer as *mut f32,
        d_ls_buffer as *mut f32,
        d_output_buffer as *mut f32,
        stream_ptr,
    ))
}

/// Launch line search kernel using raw device pointers.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_launch_linesearch_raw(
    d_source_points: u64,
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    config: &GraphNdtConfig,
    d_state_buffer: u64,
    d_ls_buffer: u64,
    stream: Option<&CudaStream>,
) -> Result<(), CudaError> {
    let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    check_cuda(ndt_graph_launch_linesearch(
        d_source_points as *const f32,
        d_voxel_means as *const f32,
        d_voxel_inv_covs as *const f32,
        d_hash_table as *const std::ffi::c_void,
        config,
        d_state_buffer as *const f32,
        d_ls_buffer as *mut f32,
        config.num_points,
        stream_ptr,
    ))
}

/// Launch update kernel using raw device pointers.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_launch_update_raw(
    config: &GraphNdtConfig,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    d_ls_buffer: u64,
    d_output_buffer: u64,
    d_debug_buffer: u64,
    stream: Option<&CudaStream>,
) -> Result<(), CudaError> {
    let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_raw());
    let debug_ptr = if d_debug_buffer == 0 {
        std::ptr::null_mut()
    } else {
        d_debug_buffer as *mut f32
    };
    check_cuda(ndt_graph_launch_update(
        config,
        d_state_buffer as *mut f32,
        d_reduce_buffer as *mut f32,
        d_ls_buffer as *mut f32,
        d_output_buffer as *mut f32,
        debug_ptr,
        stream_ptr,
    ))
}

/// Check if optimization has converged.
///
/// # Safety
/// `d_state_buffer` must be a valid device pointer.
pub unsafe fn graph_ndt_check_converged(d_state_buffer: u64) -> Result<bool, CudaError> {
    let mut converged = false;
    check_cuda(ndt_graph_check_converged(
        d_state_buffer as *const f32,
        &mut converged,
    ))?;
    Ok(converged)
}

/// Get current iteration count.
///
/// # Safety
/// `d_state_buffer` must be a valid device pointer.
pub unsafe fn graph_ndt_get_iterations(d_state_buffer: u64) -> Result<i32, CudaError> {
    let mut iterations: i32 = 0;
    check_cuda(ndt_graph_get_iterations(
        d_state_buffer as *const f32,
        &mut iterations,
    ))?;
    Ok(iterations)
}

/// Run a single iteration (compute + solve + [linesearch] + update).
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_run_iteration_raw(
    d_source_points: u64,
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    config: &GraphNdtConfig,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    d_ls_buffer: u64,
    d_output_buffer: u64,
    d_debug_buffer: u64,
    stream: Option<&CudaStream>,
) -> Result<(), CudaError> {
    // K2: Compute
    graph_ndt_launch_compute_raw(
        d_source_points,
        d_voxel_means,
        d_voxel_inv_covs,
        d_hash_table,
        config,
        d_state_buffer,
        d_reduce_buffer,
        stream,
    )?;

    // K3: Solve
    graph_ndt_launch_solve_raw(
        config,
        d_state_buffer,
        d_reduce_buffer,
        d_ls_buffer,
        d_output_buffer,
        stream,
    )?;

    // K4: Line search (if enabled)
    if config.ls_enabled != 0 {
        graph_ndt_launch_linesearch_raw(
            d_source_points,
            d_voxel_means,
            d_voxel_inv_covs,
            d_hash_table,
            config,
            d_state_buffer,
            d_ls_buffer,
            stream,
        )?;
    }

    // K5: Update
    graph_ndt_launch_update_raw(
        config,
        d_state_buffer,
        d_reduce_buffer,
        d_ls_buffer,
        d_output_buffer,
        d_debug_buffer,
        stream,
    )?;

    Ok(())
}

// ============================================================================
// Phase 24.4: Iteration Batching & Profiling
// ============================================================================

/// Timing statistics for a single kernel.
#[derive(Debug, Clone, Default)]
pub struct KernelTiming {
    /// Kernel name
    pub name: &'static str,
    /// Total time in milliseconds
    pub total_ms: f32,
    /// Number of invocations
    pub count: u32,
}

impl KernelTiming {
    /// Average time per invocation in milliseconds.
    pub fn avg_ms(&self) -> f32 {
        if self.count > 0 {
            self.total_ms / self.count as f32
        } else {
            0.0
        }
    }
}

/// Profiling statistics for graph-based NDT optimization.
#[derive(Debug, Clone, Default)]
pub struct GraphNdtProfile {
    /// K1: Init kernel timing
    pub init: KernelTiming,
    /// K2: Compute kernel timing
    pub compute: KernelTiming,
    /// K3: Solve kernel timing
    pub solve: KernelTiming,
    /// K4: Line search kernel timing
    pub linesearch: KernelTiming,
    /// K5: Update kernel timing
    pub update: KernelTiming,
    /// Total alignment time including host overhead
    pub total_ms: f32,
    /// Number of iterations executed
    pub iterations: u32,
}

impl GraphNdtProfile {
    /// Create a new empty profile.
    pub fn new() -> Self {
        Self {
            init: KernelTiming {
                name: "init",
                ..Default::default()
            },
            compute: KernelTiming {
                name: "compute",
                ..Default::default()
            },
            solve: KernelTiming {
                name: "solve",
                ..Default::default()
            },
            linesearch: KernelTiming {
                name: "linesearch",
                ..Default::default()
            },
            update: KernelTiming {
                name: "update",
                ..Default::default()
            },
            total_ms: 0.0,
            iterations: 0,
        }
    }

    /// Total kernel time (sum of all kernels) in milliseconds.
    pub fn kernel_total_ms(&self) -> f32 {
        self.init.total_ms
            + self.compute.total_ms
            + self.solve.total_ms
            + self.linesearch.total_ms
            + self.update.total_ms
    }

    /// Per-iteration time in milliseconds.
    pub fn per_iteration_ms(&self) -> f32 {
        if self.iterations > 0 {
            self.kernel_total_ms() / self.iterations as f32
        } else {
            0.0
        }
    }

    /// Print formatted profiling report.
    pub fn print_report(&self) {
        println!("=== Graph NDT Profiling Report ===");
        println!(
            "Total alignment: {:.3} ms ({} iterations)",
            self.total_ms, self.iterations
        );
        println!("Per-iteration avg: {:.3} ms", self.per_iteration_ms());
        println!();
        println!("Kernel breakdown:");
        println!(
            "  Init:       {:.3} ms (×{}, avg {:.3} ms)",
            self.init.total_ms,
            self.init.count,
            self.init.avg_ms()
        );
        println!(
            "  Compute:    {:.3} ms (×{}, avg {:.3} ms)",
            self.compute.total_ms,
            self.compute.count,
            self.compute.avg_ms()
        );
        println!(
            "  Solve:      {:.3} ms (×{}, avg {:.3} ms)",
            self.solve.total_ms,
            self.solve.count,
            self.solve.avg_ms()
        );
        if self.linesearch.count > 0 {
            println!(
                "  LineSearch: {:.3} ms (×{}, avg {:.3} ms)",
                self.linesearch.total_ms,
                self.linesearch.count,
                self.linesearch.avg_ms()
            );
        }
        println!(
            "  Update:     {:.3} ms (×{}, avg {:.3} ms)",
            self.update.total_ms,
            self.update.count,
            self.update.avg_ms()
        );
        println!(
            "Kernel total: {:.3} ms ({:.1}% of total)",
            self.kernel_total_ms(),
            100.0 * self.kernel_total_ms() / self.total_ms
        );
    }
}

/// Run N iterations without convergence checking (for benchmarking).
///
/// This function batches multiple iterations to minimize host-device sync overhead.
/// Useful for benchmarking kernel performance without early exit.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_run_iterations_batched_raw(
    d_source_points: u64,
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    config: &GraphNdtConfig,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    d_ls_buffer: u64,
    d_output_buffer: u64,
    d_debug_buffer: u64,
    num_iterations: u32,
    stream: Option<&CudaStream>,
) -> Result<(), CudaError> {
    for _ in 0..num_iterations {
        graph_ndt_run_iteration_raw(
            d_source_points,
            d_voxel_means,
            d_voxel_inv_covs,
            d_hash_table,
            config,
            d_state_buffer,
            d_reduce_buffer,
            d_ls_buffer,
            d_output_buffer,
            d_debug_buffer,
            stream,
        )?;
    }
    Ok(())
}

/// Run full NDT alignment using graph-based kernels.
///
/// This is the main entry point for graph-based NDT optimization.
/// It runs the init kernel once, then iterates until convergence or max iterations.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_align_raw(
    d_source_points: u64,
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    config: &GraphNdtConfig,
    d_initial_pose: u64,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    d_ls_buffer: u64,
    d_output_buffer: u64,
    d_debug_buffer: u64,
) -> Result<GraphNdtOutput, CudaError> {
    // K1: Initialize
    graph_ndt_launch_init_raw(
        d_initial_pose,
        d_state_buffer,
        d_reduce_buffer,
        d_ls_buffer,
        None,
    )?;
    check_cuda(cudaDeviceSynchronize())?;

    // Iteration loop
    for _ in 0..config.max_iterations {
        graph_ndt_run_iteration_raw(
            d_source_points,
            d_voxel_means,
            d_voxel_inv_covs,
            d_hash_table,
            config,
            d_state_buffer,
            d_reduce_buffer,
            d_ls_buffer,
            d_output_buffer,
            d_debug_buffer,
            None,
        )?;
        check_cuda(cudaDeviceSynchronize())?;

        // Check convergence
        if graph_ndt_check_converged(d_state_buffer)? {
            break;
        }
    }

    // Read output
    let mut output_data = [0.0f32; OUTPUT_BUFFER_SIZE];
    check_cuda(cudaMemcpy(
        output_data.as_mut_ptr() as *mut std::ffi::c_void,
        d_output_buffer as *const std::ffi::c_void,
        OUTPUT_BUFFER_SIZE * std::mem::size_of::<f32>(),
        CUDA_MEMCPY_DEVICE_TO_HOST,
    ))?;

    // Parse output
    let mut pose = [0.0f32; 6];
    pose.copy_from_slice(&output_data[0..6]);

    let mut hessian = [0.0f32; 36];
    hessian.copy_from_slice(&output_data[9..45]);

    Ok(GraphNdtOutput {
        pose,
        iterations: output_data[6] as i32,
        converged: output_data[7] > 0.5,
        score: output_data[8],
        hessian,
        num_correspondences: output_data[45] as u32,
        max_oscillation_count: output_data[46] as u32,
        avg_alpha: output_data[47],
    })
}

/// Run full NDT alignment with profiling.
///
/// Same as `graph_ndt_align_raw` but records timing for each kernel.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn graph_ndt_align_profiled_raw(
    d_source_points: u64,
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    config: &GraphNdtConfig,
    d_initial_pose: u64,
    d_state_buffer: u64,
    d_reduce_buffer: u64,
    d_ls_buffer: u64,
    d_output_buffer: u64,
    d_debug_buffer: u64,
) -> Result<(GraphNdtOutput, GraphNdtProfile), CudaError> {
    use crate::async_stream::CudaEvent;
    use std::time::Instant;

    let mut profile = GraphNdtProfile::new();
    let start_time = Instant::now();

    // Create events for timing
    let event_start = CudaEvent::new()?;
    let event_end = CudaEvent::new()?;

    // K1: Initialize
    event_start.record_default()?;
    graph_ndt_launch_init_raw(
        d_initial_pose,
        d_state_buffer,
        d_reduce_buffer,
        d_ls_buffer,
        None,
    )?;
    event_end.record_default()?;
    event_end.synchronize()?;
    profile.init.total_ms += event_end.elapsed_time(&event_start)?;
    profile.init.count += 1;

    // Iteration loop
    let mut iterations = 0u32;
    for _ in 0..config.max_iterations {
        iterations += 1;

        // K2: Compute
        event_start.record_default()?;
        graph_ndt_launch_compute_raw(
            d_source_points,
            d_voxel_means,
            d_voxel_inv_covs,
            d_hash_table,
            config,
            d_state_buffer,
            d_reduce_buffer,
            None,
        )?;
        event_end.record_default()?;
        event_end.synchronize()?;
        profile.compute.total_ms += event_end.elapsed_time(&event_start)?;
        profile.compute.count += 1;

        // K3: Solve
        event_start.record_default()?;
        graph_ndt_launch_solve_raw(
            config,
            d_state_buffer,
            d_reduce_buffer,
            d_ls_buffer,
            d_output_buffer,
            None,
        )?;
        event_end.record_default()?;
        event_end.synchronize()?;
        profile.solve.total_ms += event_end.elapsed_time(&event_start)?;
        profile.solve.count += 1;

        // K4: Line search (if enabled)
        if config.ls_enabled != 0 {
            event_start.record_default()?;
            graph_ndt_launch_linesearch_raw(
                d_source_points,
                d_voxel_means,
                d_voxel_inv_covs,
                d_hash_table,
                config,
                d_state_buffer,
                d_ls_buffer,
                None,
            )?;
            event_end.record_default()?;
            event_end.synchronize()?;
            profile.linesearch.total_ms += event_end.elapsed_time(&event_start)?;
            profile.linesearch.count += 1;
        }

        // K5: Update
        event_start.record_default()?;
        graph_ndt_launch_update_raw(
            config,
            d_state_buffer,
            d_reduce_buffer,
            d_ls_buffer,
            d_output_buffer,
            d_debug_buffer,
            None,
        )?;
        event_end.record_default()?;
        event_end.synchronize()?;
        profile.update.total_ms += event_end.elapsed_time(&event_start)?;
        profile.update.count += 1;

        // Check convergence
        if graph_ndt_check_converged(d_state_buffer)? {
            break;
        }
    }

    profile.iterations = iterations;
    profile.total_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    // Read output
    let mut output_data = [0.0f32; OUTPUT_BUFFER_SIZE];
    check_cuda(cudaMemcpy(
        output_data.as_mut_ptr() as *mut std::ffi::c_void,
        d_output_buffer as *const std::ffi::c_void,
        OUTPUT_BUFFER_SIZE * std::mem::size_of::<f32>(),
        CUDA_MEMCPY_DEVICE_TO_HOST,
    ))?;

    // Parse output
    let mut pose = [0.0f32; 6];
    pose.copy_from_slice(&output_data[0..6]);

    let mut hessian = [0.0f32; 36];
    hessian.copy_from_slice(&output_data[9..45]);

    let output = GraphNdtOutput {
        pose,
        iterations: output_data[6] as i32,
        converged: output_data[7] > 0.5,
        score: output_data[8],
        hessian,
        num_correspondences: output_data[45] as u32,
        max_oscillation_count: output_data[46] as u32,
        avg_alpha: output_data[47],
    };

    Ok((output, profile))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_sizes() {
        let result = get_buffer_sizes();
        assert!(result.is_ok(), "get_buffer_sizes should succeed");
        let (state, reduce, ls, output) = result.unwrap();

        assert_eq!(state, STATE_BUFFER_SIZE * 4, "State buffer size mismatch");
        assert_eq!(
            reduce,
            REDUCE_BUFFER_SIZE * 4,
            "Reduce buffer size mismatch"
        );
        assert_eq!(ls, LS_BUFFER_SIZE * 4, "Line search buffer size mismatch");
        assert_eq!(
            output,
            OUTPUT_BUFFER_SIZE * 4,
            "Output buffer size mismatch"
        );

        #[cfg(feature = "test-verbose")]
        println!(
            "Buffer sizes: state={}, reduce={}, ls={}, output={}",
            state, reduce, ls, output
        );
    }

    #[test]
    fn test_shared_mem_sizes() {
        let compute_smem = compute_shared_mem_size();
        let ls_smem = linesearch_shared_mem_size();

        assert!(compute_smem > 0, "Compute shared memory should be positive");
        assert!(ls_smem > 0, "Line search shared memory should be positive");

        #[cfg(feature = "test-verbose")]
        println!(
            "Shared memory: compute={} bytes, linesearch={} bytes",
            compute_smem, ls_smem
        );
    }

    #[test]
    fn test_config_builder() {
        let config = GraphNdtConfig::new(
            0.55,  // gauss_d1
            0.5,   // gauss_d2
            2.0,   // resolution
            0.01,  // epsilon
            10000, // num_points
            5000,  // num_voxels
            16384, // hash_capacity
            30,    // max_iterations
        )
        .with_regularization(100.0, 200.0, 0.001)
        .with_line_search(true, 8, 1e-4, 0.9)
        .with_debug(true);

        assert_eq!(config.gauss_d1, 0.55);
        assert_eq!(config.reg_enabled, 1);
        assert_eq!(config.reg_ref_x, 100.0);
        assert_eq!(config.ls_enabled, 1);
        assert_eq!(config.debug_enabled, 1);
    }

    #[test]
    fn test_num_blocks() {
        assert_eq!(num_blocks(256), 1);
        assert_eq!(num_blocks(257), 2);
        assert_eq!(num_blocks(512), 2);
        assert_eq!(num_blocks(100000), 391);
    }
}
