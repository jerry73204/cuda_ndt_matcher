//! FFI bindings for CUDA libraries used by cuda_ndt_matcher.
//!
//! This crate provides Rust bindings for:
//! - CUB DeviceRadixSort (GPU radix sort)
//! - CUB DeviceScan (prefix sum)
//! - CUB DeviceSelect (stream compaction)
//! - GPU segment detection for voxel boundaries
//! - cuSOLVER batched Cholesky solver
//! - Spatial hash table for GPU-accelerated voxel lookup
//! - Persistent NDT kernel with cooperative groups
//!
//! # Example
//!
//! ```ignore
//! use cuda_ffi::{RadixSorter, SegmentDetector};
//!
//! let sorter = RadixSorter::new()?;
//! let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values)?;
//!
//! let detector = SegmentDetector::new()?;
//! let segments = detector.detect_segments(&sorted_keys)?;
//! ```

pub mod async_stream;
pub mod batch_persistent_ndt;
pub mod batched_solve;
pub mod graph_ndt;
pub mod persistent_ndt;
pub mod radix_sort;
pub mod segment_detect;
pub mod segmented_reduce;
pub mod texture;
pub mod voxel_hash;

pub use batched_solve::{BatchedCholeskySolver, CusolverDnHandle, CusolverError};
pub use persistent_ndt::{
    is_supported as persistent_ndt_is_supported, persistent_ndt_buffer_size,
    persistent_ndt_can_launch, persistent_ndt_launch_raw, persistent_ndt_max_blocks,
    persistent_ndt_supported, reduce_buffer_size as persistent_ndt_reduce_buffer_size,
    GridTooLargeError, PersistentNdt,
};
pub use radix_sort::{
    radix_sort_temp_size, sort_pairs_inplace, CudaError, DeviceBuffer, RadixSorter,
};
pub use segment_detect::{
    detect_segments_inplace, segment_detect_temp_sizes, SegmentCounts, SegmentDetector,
    SegmentResult,
};
pub use segmented_reduce::{
    segmented_reduce_sum_f32_inplace, segmented_reduce_sum_f32_temp_size,
    segmented_reduce_sum_f64_inplace, segmented_reduce_sum_f64_temp_size, SegmentedReducer,
};
pub use voxel_hash::{
    hash_table_build, hash_table_capacity, hash_table_count_entries, hash_table_init,
    hash_table_query, hash_table_size, max_neighbors as voxel_hash_max_neighbors, VoxelHash,
};

// Batch persistent NDT kernel (parallel multi-scan alignment)
pub use batch_persistent_ndt::{
    batch_ndt_blocks_per_slot, batch_ndt_reduce_buffer_size, batch_ndt_total_blocks,
    batch_persistent_ndt_init_barriers_async_raw, batch_persistent_ndt_init_barriers_raw,
    batch_persistent_ndt_launch_async_raw, batch_persistent_ndt_launch_raw,
    batch_persistent_ndt_launch_warp_optimized_raw, batch_persistent_ndt_stream_sync_raw,
    batch_persistent_ndt_sync_raw, batch_reduce_buffer_size, batch_shared_mem_size,
    batch_warp_shared_mem_size, BatchPersistentNdt,
};

// Async stream utilities (pinned memory, streams, events)
pub use async_stream::{
    AsyncDeviceBuffer, CudaEvent, CudaStream, PinnedBuffer, RawCudaEvent, RawCudaStream,
};

// Texture memory for voxel data
pub use texture::{
    batch_persistent_ndt_launch_textured_raw, texture_handle_size, CudaTextureObject, TextureError,
    TexturedBatchNdtParams, VoxelInvCovsTexture, VoxelMeansTexture,
};

// Graph-based NDT kernels (Phase 24 - alternative to cooperative kernel)
pub use graph_ndt::{
    compute_shared_mem_size as graph_ndt_compute_shared_mem_size,
    get_buffer_sizes as graph_ndt_get_buffer_sizes, graph_ndt_align_profiled_raw,
    graph_ndt_align_raw, graph_ndt_check_converged, graph_ndt_get_iterations,
    graph_ndt_launch_compute_raw, graph_ndt_launch_init_raw, graph_ndt_launch_linesearch_raw,
    graph_ndt_launch_solve_raw, graph_ndt_launch_update_raw, graph_ndt_run_iteration_raw,
    graph_ndt_run_iterations_batched_raw,
    linesearch_shared_mem_size as graph_ndt_linesearch_shared_mem_size,
    num_blocks as graph_ndt_num_blocks, GraphNdtConfig, GraphNdtOutput, GraphNdtProfile,
    KernelTiming, BLOCK_SIZE as GRAPH_NDT_BLOCK_SIZE,
    DEBUG_FLOATS_PER_ITER as GRAPH_NDT_DEBUG_FLOATS_PER_ITER,
    LS_BUFFER_SIZE as GRAPH_NDT_LS_BUFFER_SIZE, OUTPUT_BUFFER_SIZE as GRAPH_NDT_OUTPUT_BUFFER_SIZE,
    REDUCE_BUFFER_SIZE as GRAPH_NDT_REDUCE_BUFFER_SIZE,
    STATE_BUFFER_SIZE as GRAPH_NDT_STATE_BUFFER_SIZE,
};

/// Device-to-device memory copy using CUDA.
///
/// # Safety
/// Both `dst` and `src` must be valid device pointers with at least `size` bytes.
pub unsafe fn cuda_memcpy_dtod(dst: u64, src: u64, size: usize) -> Result<(), CudaError> {
    use std::ffi::c_int;

    extern "C" {
        fn cudaMemcpy(
            dst: *mut std::ffi::c_void,
            src: *const std::ffi::c_void,
            count: usize,
            kind: c_int,
        ) -> c_int;
    }

    const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

    let result = cudaMemcpy(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_DEVICE_TO_DEVICE,
    );
    if result != 0 {
        Err(CudaError::from(result))
    } else {
        Ok(())
    }
}

/// Synchronize the CUDA device - wait for all pending operations to complete.
///
/// This is a blocking call that ensures all previous CUDA operations
/// (kernel launches, memory copies, etc.) have completed.
pub fn cuda_device_synchronize() -> Result<(), CudaError> {
    use std::ffi::c_int;

    extern "C" {
        fn cudaDeviceSynchronize() -> c_int;
    }

    let result = unsafe { cudaDeviceSynchronize() };
    if result != 0 {
        Err(CudaError::from(result))
    } else {
        Ok(())
    }
}

/// Host-to-device memory copy using CUDA.
///
/// # Safety
/// `dst` must be a valid device pointer with at least `size` bytes.
/// `src` must be a valid host pointer with at least `size` bytes.
pub unsafe fn cuda_memcpy_htod(dst: u64, src: *const u8, size: usize) -> Result<(), CudaError> {
    use std::ffi::c_int;

    extern "C" {
        fn cudaMemcpy(
            dst: *mut std::ffi::c_void,
            src: *const std::ffi::c_void,
            count: usize,
            kind: c_int,
        ) -> c_int;
    }

    const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;

    let result = cudaMemcpy(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_HOST_TO_DEVICE,
    );
    if result != 0 {
        Err(CudaError::from(result))
    } else {
        Ok(())
    }
}
