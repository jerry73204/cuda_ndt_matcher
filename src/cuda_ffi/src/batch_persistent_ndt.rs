//! Batch Persistent NDT kernel with atomic barriers.
//!
//! This module provides FFI bindings to the CUDA batch persistent NDT kernel,
//! which processes M alignments in parallel by partitioning blocks into
//! independent slots. Each slot runs a complete Newton optimization using
//! atomic barriers for intra-slot synchronization instead of cooperative
//! grid-wide sync.
//!
//! # Usage
//!
//! ```ignore
//! use cuda_ffi::batch_persistent_ndt::{BatchPersistentNdt, batch_reduce_buffer_size};
//!
//! // Get buffer sizes
//! let reduce_size = batch_reduce_buffer_size();
//! let shared_mem = BatchPersistentNdt::shared_mem_size();
//!
//! // Calculate blocks
//! let blocks_per_slot = BatchPersistentNdt::blocks_per_slot(num_points);
//! let total_blocks = BatchPersistentNdt::total_blocks(num_slots, blocks_per_slot);
//!
//! // Launch batch optimization
//! BatchPersistentNdt::launch(
//!     voxel_means, voxel_inv_covs, hash_table, hash_capacity,
//!     gauss_d1, gauss_d2, resolution,
//!     all_source_points, all_initial_poses, points_per_slot,
//!     all_reduce_buffers, barrier_counters, barrier_senses,
//!     all_out_poses, all_out_iterations, ...
//! )?;
//! ```

use crate::async_stream::RawCudaStream;
use crate::radix_sort::{check_cuda, CudaError};
use std::ffi::c_int;

// ============================================================================
// FFI Declarations
// ============================================================================

extern "C" {
    fn batch_persistent_ndt_blocks_per_slot(num_points: c_int) -> c_int;

    fn batch_persistent_ndt_total_blocks(num_slots: c_int, blocks_per_slot: c_int) -> c_int;

    fn batch_persistent_ndt_shared_mem_size() -> u32;

    fn batch_persistent_ndt_reduce_buffer_size() -> u32;

    fn batch_persistent_ndt_launch(
        // Shared data
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        hash_capacity: u32,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,

        // Per-slot input
        all_source_points: *const f32,
        all_initial_poses: *const f32,
        points_per_slot: *const c_int,

        // Per-slot working memory
        all_reduce_buffers: *mut f32,
        barrier_counters: *mut c_int,
        barrier_senses: *mut c_int,

        // Per-slot outputs
        all_out_poses: *mut f32,
        all_out_iterations: *mut c_int,
        all_out_converged: *mut u32,
        all_out_scores: *mut f32,
        all_out_hessians: *mut f32,
        all_out_correspondences: *mut u32,
        all_out_oscillations: *mut u32,
        all_out_alpha_sums: *mut f32,

        // Control
        num_slots: c_int,
        blocks_per_slot: c_int,
        max_points_per_slot: c_int,
        max_iterations: c_int,
        epsilon: f32,

        // Line search
        ls_enabled: c_int,
        ls_num_candidates: c_int,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32,

        // Regularization
        reg_ref_x: *const f32,
        reg_ref_y: *const f32,
        reg_scale: f32,
        reg_enabled: c_int,
    ) -> c_int;

    fn batch_persistent_ndt_sync() -> c_int;

    fn batch_persistent_ndt_launch_async(
        // Shared data
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        hash_capacity: u32,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,

        // Per-slot input
        all_source_points: *const f32,
        all_initial_poses: *const f32,
        points_per_slot: *const c_int,

        // Per-slot working memory
        all_reduce_buffers: *mut f32,
        barrier_counters: *mut c_int,
        barrier_senses: *mut c_int,

        // Per-slot outputs
        all_out_poses: *mut f32,
        all_out_iterations: *mut c_int,
        all_out_converged: *mut u32,
        all_out_scores: *mut f32,
        all_out_hessians: *mut f32,
        all_out_correspondences: *mut u32,
        all_out_oscillations: *mut u32,
        all_out_alpha_sums: *mut f32,

        // Control
        num_slots: c_int,
        blocks_per_slot: c_int,
        max_points_per_slot: c_int,
        max_iterations: c_int,
        epsilon: f32,

        // Line search
        ls_enabled: c_int,
        ls_num_candidates: c_int,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32,

        // Regularization
        reg_ref_x: *const f32,
        reg_ref_y: *const f32,
        reg_scale: f32,
        reg_enabled: c_int,

        // Stream
        stream: RawCudaStream,
    ) -> c_int;

    fn batch_persistent_ndt_stream_sync(stream: RawCudaStream) -> c_int;

    fn batch_persistent_ndt_warp_shared_mem_size() -> u32;

    fn batch_persistent_ndt_launch_warp_optimized(
        // Shared data
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        hash_capacity: u32,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,

        // Per-slot input
        all_source_points: *const f32,
        all_initial_poses: *const f32,
        points_per_slot: *const c_int,

        // Per-slot working memory
        all_reduce_buffers: *mut f32,
        barrier_counters: *mut c_int,
        barrier_senses: *mut c_int,

        // Per-slot outputs
        all_out_poses: *mut f32,
        all_out_iterations: *mut c_int,
        all_out_converged: *mut u32,
        all_out_scores: *mut f32,
        all_out_hessians: *mut f32,
        all_out_correspondences: *mut u32,
        all_out_oscillations: *mut u32,
        all_out_alpha_sums: *mut f32,

        // Control
        num_slots: c_int,
        blocks_per_slot: c_int,
        max_points_per_slot: c_int,
        max_iterations: c_int,
        epsilon: f32,

        // Line search
        ls_enabled: c_int,
        ls_num_candidates: c_int,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32,

        // Regularization
        reg_ref_x: *const f32,
        reg_ref_y: *const f32,
        reg_scale: f32,
        reg_enabled: c_int,

        // Stream
        stream: RawCudaStream,
    ) -> c_int;

    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: c_int, count: usize) -> c_int;
    fn cudaMemsetAsync(
        devPtr: *mut std::ffi::c_void,
        value: c_int,
        count: usize,
        stream: RawCudaStream,
    ) -> c_int;
}

// ============================================================================
// Public API
// ============================================================================

/// Get required reduce buffer size per slot in bytes.
pub fn batch_reduce_buffer_size() -> usize {
    unsafe { batch_persistent_ndt_reduce_buffer_size() as usize }
}

/// Get shared memory size per block in bytes.
pub fn batch_shared_mem_size() -> usize {
    unsafe { batch_persistent_ndt_shared_mem_size() as usize }
}

/// Batch Persistent NDT kernel interface.
pub struct BatchPersistentNdt;

impl BatchPersistentNdt {
    /// Block size used by the batch kernel (256 threads per block).
    pub const BLOCK_SIZE: usize = 256;

    /// Number of reduce values per thread (score + gradient + hessian + correspondences).
    pub const REDUCE_SIZE: usize = 29;

    /// Total reduce buffer size per slot in floats.
    pub const REDUCE_BUFFER_FLOATS: usize = 160;

    /// Maximum number of line search candidates.
    pub const MAX_LS_CANDIDATES: usize = 8;

    /// Get recommended number of blocks per slot for given point count.
    pub fn blocks_per_slot(num_points: usize) -> usize {
        unsafe { batch_persistent_ndt_blocks_per_slot(num_points as c_int) as usize }
    }

    /// Get total grid size for M slots.
    pub fn total_blocks(num_slots: usize, blocks_per_slot: usize) -> usize {
        unsafe {
            batch_persistent_ndt_total_blocks(num_slots as c_int, blocks_per_slot as c_int) as usize
        }
    }

    /// Get shared memory size per block in bytes.
    pub fn shared_mem_size() -> usize {
        batch_shared_mem_size()
    }

    /// Get reduce buffer size per slot in bytes.
    pub fn reduce_buffer_size() -> usize {
        batch_reduce_buffer_size()
    }

    /// Launch batch persistent NDT optimization kernel.
    ///
    /// # Arguments
    ///
    /// * `voxel_means` - Device pointer to voxel means [V * 3]
    /// * `voxel_inv_covs` - Device pointer to inverse covariances [V * 9]
    /// * `hash_table` - Device pointer to hash table
    /// * `hash_capacity` - Hash table capacity
    /// * `gauss_d1` - NDT Gaussian parameter d1
    /// * `gauss_d2` - NDT Gaussian parameter d2
    /// * `resolution` - Voxel resolution
    /// * `all_source_points` - Device pointer to all source points [num_slots * max_points_per_slot * 3]
    /// * `all_initial_poses` - Device pointer to initial poses [num_slots * 6]
    /// * `points_per_slot` - Device pointer to point counts per slot [num_slots]
    /// * `all_reduce_buffers` - Device pointer to reduce buffers [num_slots * 160]
    /// * `barrier_counters` - Device pointer to barrier counters [num_slots]
    /// * `barrier_senses` - Device pointer to barrier senses [num_slots]
    /// * `all_out_poses` - Device pointer to output poses [num_slots * 6]
    /// * `all_out_iterations` - Device pointer to output iteration counts [num_slots]
    /// * `all_out_converged` - Device pointer to output convergence flags [num_slots]
    /// * `all_out_scores` - Device pointer to output scores [num_slots]
    /// * `all_out_hessians` - Device pointer to output Hessians [num_slots * 36]
    /// * `all_out_correspondences` - Device pointer to output correspondence counts [num_slots]
    /// * `all_out_oscillations` - Device pointer to output oscillation counts [num_slots]
    /// * `all_out_alpha_sums` - Device pointer to output alpha sums [num_slots]
    /// * `num_slots` - Number of parallel alignments
    /// * `blocks_per_slot` - Number of blocks per slot
    /// * `max_points_per_slot` - Maximum points per slot (for buffer indexing)
    /// * `max_iterations` - Maximum Newton iterations
    /// * `epsilon` - Convergence threshold
    /// * `ls_enabled` - Whether line search is enabled
    /// * `ls_num_candidates` - Number of line search candidates
    /// * `ls_mu` - Armijo constant for line search
    /// * `ls_nu` - Curvature constant for line search
    /// * `fixed_step_size` - Step size when line search disabled
    /// * `reg_ref_x` - Device pointer to regularization reference X [num_slots] or null
    /// * `reg_ref_y` - Device pointer to regularization reference Y [num_slots] or null
    /// * `reg_scale` - Regularization scale factor
    /// * `reg_enabled` - Whether regularization is enabled
    ///
    /// # Safety
    ///
    /// All device pointers must be valid with appropriate sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        hash_capacity: u32,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
        all_source_points: *const f32,
        all_initial_poses: *const f32,
        points_per_slot: *const i32,
        all_reduce_buffers: *mut f32,
        barrier_counters: *mut i32,
        barrier_senses: *mut i32,
        all_out_poses: *mut f32,
        all_out_iterations: *mut i32,
        all_out_converged: *mut u32,
        all_out_scores: *mut f32,
        all_out_hessians: *mut f32,
        all_out_correspondences: *mut u32,
        all_out_oscillations: *mut u32,
        all_out_alpha_sums: *mut f32,
        num_slots: usize,
        blocks_per_slot: usize,
        max_points_per_slot: usize,
        max_iterations: i32,
        epsilon: f32,
        ls_enabled: bool,
        ls_num_candidates: i32,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32,
        reg_ref_x: *const f32,
        reg_ref_y: *const f32,
        reg_scale: f32,
        reg_enabled: bool,
    ) -> Result<(), CudaError> {
        let result = batch_persistent_ndt_launch(
            voxel_means,
            voxel_inv_covs,
            hash_table,
            hash_capacity,
            gauss_d1,
            gauss_d2,
            resolution,
            all_source_points,
            all_initial_poses,
            points_per_slot,
            all_reduce_buffers,
            barrier_counters,
            barrier_senses,
            all_out_poses,
            all_out_iterations,
            all_out_converged,
            all_out_scores,
            all_out_hessians,
            all_out_correspondences,
            all_out_oscillations,
            all_out_alpha_sums,
            num_slots as c_int,
            blocks_per_slot as c_int,
            max_points_per_slot as c_int,
            max_iterations,
            epsilon,
            if ls_enabled { 1 } else { 0 },
            ls_num_candidates,
            ls_mu,
            ls_nu,
            fixed_step_size,
            reg_ref_x,
            reg_ref_y,
            reg_scale,
            if reg_enabled { 1 } else { 0 },
        );

        check_cuda(result)
    }

    /// Synchronize device - wait for kernel completion.
    pub fn sync() -> Result<(), CudaError> {
        unsafe { check_cuda(batch_persistent_ndt_sync()) }
    }

    /// Synchronize a specific stream - wait for stream completion.
    ///
    /// # Safety
    ///
    /// The stream handle must be valid.
    pub unsafe fn stream_sync(stream: RawCudaStream) -> Result<(), CudaError> {
        check_cuda(batch_persistent_ndt_stream_sync(stream))
    }

    /// Launch batch persistent NDT optimization kernel asynchronously.
    ///
    /// This version accepts a CUDA stream for async execution and pipelining.
    /// The kernel and memory operations will be enqueued to the specified stream.
    ///
    /// # Arguments
    ///
    /// Same as `launch()`, plus:
    /// * `stream` - CUDA stream handle (use `CudaStream::as_raw()`)
    ///
    /// # Safety
    ///
    /// All device pointers must be valid with appropriate sizes.
    /// The stream must be valid and not destroyed until operations complete.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_async(
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        hash_capacity: u32,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
        all_source_points: *const f32,
        all_initial_poses: *const f32,
        points_per_slot: *const i32,
        all_reduce_buffers: *mut f32,
        barrier_counters: *mut i32,
        barrier_senses: *mut i32,
        all_out_poses: *mut f32,
        all_out_iterations: *mut i32,
        all_out_converged: *mut u32,
        all_out_scores: *mut f32,
        all_out_hessians: *mut f32,
        all_out_correspondences: *mut u32,
        all_out_oscillations: *mut u32,
        all_out_alpha_sums: *mut f32,
        num_slots: usize,
        blocks_per_slot: usize,
        max_points_per_slot: usize,
        max_iterations: i32,
        epsilon: f32,
        ls_enabled: bool,
        ls_num_candidates: i32,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32,
        reg_ref_x: *const f32,
        reg_ref_y: *const f32,
        reg_scale: f32,
        reg_enabled: bool,
        stream: RawCudaStream,
    ) -> Result<(), CudaError> {
        let result = batch_persistent_ndt_launch_async(
            voxel_means,
            voxel_inv_covs,
            hash_table,
            hash_capacity,
            gauss_d1,
            gauss_d2,
            resolution,
            all_source_points,
            all_initial_poses,
            points_per_slot,
            all_reduce_buffers,
            barrier_counters,
            barrier_senses,
            all_out_poses,
            all_out_iterations,
            all_out_converged,
            all_out_scores,
            all_out_hessians,
            all_out_correspondences,
            all_out_oscillations,
            all_out_alpha_sums,
            num_slots as c_int,
            blocks_per_slot as c_int,
            max_points_per_slot as c_int,
            max_iterations,
            epsilon,
            if ls_enabled { 1 } else { 0 },
            ls_num_candidates,
            ls_mu,
            ls_nu,
            fixed_step_size,
            reg_ref_x,
            reg_ref_y,
            reg_scale,
            if reg_enabled { 1 } else { 0 },
            stream,
        );

        check_cuda(result)
    }

    /// Initialize barrier counters and senses to zero asynchronously.
    ///
    /// # Safety
    ///
    /// Pointers must be valid device memory with num_slots elements.
    pub unsafe fn init_barriers_async(
        barrier_counters: *mut i32,
        barrier_senses: *mut i32,
        num_slots: usize,
        stream: RawCudaStream,
    ) -> Result<(), CudaError> {
        check_cuda(cudaMemsetAsync(
            barrier_counters as *mut std::ffi::c_void,
            0,
            num_slots * std::mem::size_of::<i32>(),
            stream,
        ))?;
        check_cuda(cudaMemsetAsync(
            barrier_senses as *mut std::ffi::c_void,
            0,
            num_slots * std::mem::size_of::<i32>(),
            stream,
        ))
    }

    /// Initialize barrier counters and senses to zero.
    ///
    /// Must be called before launching the kernel for a new batch.
    ///
    /// # Safety
    ///
    /// Pointers must be valid device memory with num_slots elements.
    pub unsafe fn init_barriers(
        barrier_counters: *mut i32,
        barrier_senses: *mut i32,
        num_slots: usize,
    ) -> Result<(), CudaError> {
        check_cuda(cudaMemset(
            barrier_counters as *mut std::ffi::c_void,
            0,
            num_slots * std::mem::size_of::<i32>(),
        ))?;
        check_cuda(cudaMemset(
            barrier_senses as *mut std::ffi::c_void,
            0,
            num_slots * std::mem::size_of::<i32>(),
        ))
    }
}

// ============================================================================
// Raw pointer API (for CubeCL handle interop)
// ============================================================================

/// Get blocks per slot for point count.
pub fn batch_ndt_blocks_per_slot(num_points: usize) -> usize {
    BatchPersistentNdt::blocks_per_slot(num_points)
}

/// Get total blocks for batch.
pub fn batch_ndt_total_blocks(num_slots: usize, blocks_per_slot: usize) -> usize {
    BatchPersistentNdt::total_blocks(num_slots, blocks_per_slot)
}

/// Get reduce buffer size per slot in bytes.
pub fn batch_ndt_reduce_buffer_size() -> usize {
    BatchPersistentNdt::reduce_buffer_size()
}

/// Launch batch NDT kernel using raw device pointers (u64 for CubeCL interop).
///
/// # Safety
///
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batch_persistent_ndt_launch_raw(
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    hash_capacity: u32,
    gauss_d1: f32,
    gauss_d2: f32,
    resolution: f32,
    d_all_source_points: u64,
    d_all_initial_poses: u64,
    d_points_per_slot: u64,
    d_all_reduce_buffers: u64,
    d_barrier_counters: u64,
    d_barrier_senses: u64,
    d_all_out_poses: u64,
    d_all_out_iterations: u64,
    d_all_out_converged: u64,
    d_all_out_scores: u64,
    d_all_out_hessians: u64,
    d_all_out_correspondences: u64,
    d_all_out_oscillations: u64,
    d_all_out_alpha_sums: u64,
    num_slots: usize,
    blocks_per_slot: usize,
    max_points_per_slot: usize,
    max_iterations: i32,
    epsilon: f32,
    ls_enabled: bool,
    ls_num_candidates: i32,
    ls_mu: f32,
    ls_nu: f32,
    fixed_step_size: f32,
    d_reg_ref_x: u64,
    d_reg_ref_y: u64,
    reg_scale: f32,
    reg_enabled: bool,
) -> Result<(), CudaError> {
    BatchPersistentNdt::launch(
        d_voxel_means as *const f32,
        d_voxel_inv_covs as *const f32,
        d_hash_table as *const std::ffi::c_void,
        hash_capacity,
        gauss_d1,
        gauss_d2,
        resolution,
        d_all_source_points as *const f32,
        d_all_initial_poses as *const f32,
        d_points_per_slot as *const i32,
        d_all_reduce_buffers as *mut f32,
        d_barrier_counters as *mut i32,
        d_barrier_senses as *mut i32,
        d_all_out_poses as *mut f32,
        d_all_out_iterations as *mut i32,
        d_all_out_converged as *mut u32,
        d_all_out_scores as *mut f32,
        d_all_out_hessians as *mut f32,
        d_all_out_correspondences as *mut u32,
        d_all_out_oscillations as *mut u32,
        d_all_out_alpha_sums as *mut f32,
        num_slots,
        blocks_per_slot,
        max_points_per_slot,
        max_iterations,
        epsilon,
        ls_enabled,
        ls_num_candidates,
        ls_mu,
        ls_nu,
        fixed_step_size,
        if d_reg_ref_x == 0 {
            std::ptr::null()
        } else {
            d_reg_ref_x as *const f32
        },
        if d_reg_ref_y == 0 {
            std::ptr::null()
        } else {
            d_reg_ref_y as *const f32
        },
        reg_scale,
        reg_enabled,
    )
}

/// Initialize barriers using raw device pointers.
///
/// # Safety
///
/// Pointers must be valid device memory with num_slots elements.
pub unsafe fn batch_persistent_ndt_init_barriers_raw(
    d_barrier_counters: u64,
    d_barrier_senses: u64,
    num_slots: usize,
) -> Result<(), CudaError> {
    BatchPersistentNdt::init_barriers(
        d_barrier_counters as *mut i32,
        d_barrier_senses as *mut i32,
        num_slots,
    )
}

/// Synchronize device.
pub fn batch_persistent_ndt_sync_raw() -> Result<(), CudaError> {
    BatchPersistentNdt::sync()
}

/// Synchronize a specific stream.
///
/// # Safety
///
/// The stream handle must be valid.
pub unsafe fn batch_persistent_ndt_stream_sync_raw(stream: RawCudaStream) -> Result<(), CudaError> {
    BatchPersistentNdt::stream_sync(stream)
}

/// Launch batch NDT kernel asynchronously using raw device pointers.
///
/// # Safety
///
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
/// The stream must be valid and not destroyed until operations complete.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batch_persistent_ndt_launch_async_raw(
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    hash_capacity: u32,
    gauss_d1: f32,
    gauss_d2: f32,
    resolution: f32,
    d_all_source_points: u64,
    d_all_initial_poses: u64,
    d_points_per_slot: u64,
    d_all_reduce_buffers: u64,
    d_barrier_counters: u64,
    d_barrier_senses: u64,
    d_all_out_poses: u64,
    d_all_out_iterations: u64,
    d_all_out_converged: u64,
    d_all_out_scores: u64,
    d_all_out_hessians: u64,
    d_all_out_correspondences: u64,
    d_all_out_oscillations: u64,
    d_all_out_alpha_sums: u64,
    num_slots: usize,
    blocks_per_slot: usize,
    max_points_per_slot: usize,
    max_iterations: i32,
    epsilon: f32,
    ls_enabled: bool,
    ls_num_candidates: i32,
    ls_mu: f32,
    ls_nu: f32,
    fixed_step_size: f32,
    d_reg_ref_x: u64,
    d_reg_ref_y: u64,
    reg_scale: f32,
    reg_enabled: bool,
    stream: RawCudaStream,
) -> Result<(), CudaError> {
    BatchPersistentNdt::launch_async(
        d_voxel_means as *const f32,
        d_voxel_inv_covs as *const f32,
        d_hash_table as *const std::ffi::c_void,
        hash_capacity,
        gauss_d1,
        gauss_d2,
        resolution,
        d_all_source_points as *const f32,
        d_all_initial_poses as *const f32,
        d_points_per_slot as *const i32,
        d_all_reduce_buffers as *mut f32,
        d_barrier_counters as *mut i32,
        d_barrier_senses as *mut i32,
        d_all_out_poses as *mut f32,
        d_all_out_iterations as *mut i32,
        d_all_out_converged as *mut u32,
        d_all_out_scores as *mut f32,
        d_all_out_hessians as *mut f32,
        d_all_out_correspondences as *mut u32,
        d_all_out_oscillations as *mut u32,
        d_all_out_alpha_sums as *mut f32,
        num_slots,
        blocks_per_slot,
        max_points_per_slot,
        max_iterations,
        epsilon,
        ls_enabled,
        ls_num_candidates,
        ls_mu,
        ls_nu,
        fixed_step_size,
        if d_reg_ref_x == 0 {
            std::ptr::null()
        } else {
            d_reg_ref_x as *const f32
        },
        if d_reg_ref_y == 0 {
            std::ptr::null()
        } else {
            d_reg_ref_y as *const f32
        },
        reg_scale,
        reg_enabled,
        stream,
    )
}

/// Initialize barriers asynchronously using raw device pointers.
///
/// # Safety
///
/// Pointers must be valid device memory with num_slots elements.
pub unsafe fn batch_persistent_ndt_init_barriers_async_raw(
    d_barrier_counters: u64,
    d_barrier_senses: u64,
    num_slots: usize,
    stream: RawCudaStream,
) -> Result<(), CudaError> {
    BatchPersistentNdt::init_barriers_async(
        d_barrier_counters as *mut i32,
        d_barrier_senses as *mut i32,
        num_slots,
        stream,
    )
}

// ============================================================================
// Warp-Optimized Kernel API
// ============================================================================

/// Get shared memory size for warp-optimized kernel in bytes.
///
/// The warp-optimized kernel uses less shared memory because it uses
/// warp-level reduction instead of full block-level reduction.
pub fn batch_warp_shared_mem_size() -> usize {
    unsafe { batch_persistent_ndt_warp_shared_mem_size() as usize }
}

/// Launch warp-optimized batch NDT kernel using raw device pointers.
///
/// This version uses warp-level reduction and warp-cooperative Newton solve
/// for improved GPU utilization. It uses less shared memory than the original.
///
/// # Safety
///
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batch_persistent_ndt_launch_warp_optimized_raw(
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    hash_capacity: u32,
    gauss_d1: f32,
    gauss_d2: f32,
    resolution: f32,
    d_all_source_points: u64,
    d_all_initial_poses: u64,
    d_points_per_slot: u64,
    d_all_reduce_buffers: u64,
    d_barrier_counters: u64,
    d_barrier_senses: u64,
    d_all_out_poses: u64,
    d_all_out_iterations: u64,
    d_all_out_converged: u64,
    d_all_out_scores: u64,
    d_all_out_hessians: u64,
    d_all_out_correspondences: u64,
    d_all_out_oscillations: u64,
    d_all_out_alpha_sums: u64,
    num_slots: usize,
    blocks_per_slot: usize,
    max_points_per_slot: usize,
    max_iterations: i32,
    epsilon: f32,
    ls_enabled: bool,
    ls_num_candidates: i32,
    ls_mu: f32,
    ls_nu: f32,
    fixed_step_size: f32,
    d_reg_ref_x: u64,
    d_reg_ref_y: u64,
    reg_scale: f32,
    reg_enabled: bool,
    stream: RawCudaStream,
) -> Result<(), CudaError> {
    let result = batch_persistent_ndt_launch_warp_optimized(
        d_voxel_means as *const f32,
        d_voxel_inv_covs as *const f32,
        d_hash_table as *const std::ffi::c_void,
        hash_capacity,
        gauss_d1,
        gauss_d2,
        resolution,
        d_all_source_points as *const f32,
        d_all_initial_poses as *const f32,
        d_points_per_slot as *const c_int,
        d_all_reduce_buffers as *mut f32,
        d_barrier_counters as *mut c_int,
        d_barrier_senses as *mut c_int,
        d_all_out_poses as *mut f32,
        d_all_out_iterations as *mut c_int,
        d_all_out_converged as *mut u32,
        d_all_out_scores as *mut f32,
        d_all_out_hessians as *mut f32,
        d_all_out_correspondences as *mut u32,
        d_all_out_oscillations as *mut u32,
        d_all_out_alpha_sums as *mut f32,
        num_slots as c_int,
        blocks_per_slot as c_int,
        max_points_per_slot as c_int,
        max_iterations,
        epsilon,
        if ls_enabled { 1 } else { 0 },
        ls_num_candidates,
        ls_mu,
        ls_nu,
        fixed_step_size,
        if d_reg_ref_x == 0 {
            std::ptr::null()
        } else {
            d_reg_ref_x as *const f32
        },
        if d_reg_ref_y == 0 {
            std::ptr::null()
        } else {
            d_reg_ref_y as *const f32
        },
        reg_scale,
        if reg_enabled { 1 } else { 0 },
        stream,
    );
    check_cuda(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_blocks_per_slot() {
        // 256 points -> 1 block
        assert_eq!(BatchPersistentNdt::blocks_per_slot(256), 1);
        // 257 points -> 2 blocks
        assert_eq!(BatchPersistentNdt::blocks_per_slot(257), 2);
        // 1000 points -> 4 blocks
        assert_eq!(BatchPersistentNdt::blocks_per_slot(1000), 4);
    }
    #[test]
    fn test_total_blocks() {
        // 4 slots, 4 blocks each -> 16 total
        assert_eq!(BatchPersistentNdt::total_blocks(4, 4), 16);
    }
    #[test]
    fn test_reduce_buffer_size() {
        let size = BatchPersistentNdt::reduce_buffer_size();
        // 160 floats * 4 bytes = 640 bytes
        assert_eq!(
            size,
            160 * 4,
            "Reduce buffer should be 160 floats (640 bytes)"
        );
    }
    #[test]
    fn test_shared_mem_size() {
        let size = BatchPersistentNdt::shared_mem_size();
        // 256 threads * 29 values * 4 bytes = 29696 bytes
        assert_eq!(
            size,
            256 * 29 * 4,
            "Shared memory should be 256 * 29 * 4 bytes"
        );
    }
    #[test]
    fn test_warp_shared_mem_size() {
        let size = batch_warp_shared_mem_size();
        // 8 warps * 29 values * 4 bytes = 928 bytes (much smaller!)
        assert_eq!(
            size,
            8 * 29 * 4,
            "Warp-optimized shared memory should be 8 * 29 * 4 = 928 bytes"
        );
    }
}
