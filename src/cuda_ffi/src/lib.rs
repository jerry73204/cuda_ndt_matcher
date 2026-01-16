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

pub mod batched_solve;
pub mod persistent_ndt;
pub mod radix_sort;
pub mod segment_detect;
pub mod segmented_reduce;
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
    hash_table_build, hash_table_capacity, hash_table_init, hash_table_query, hash_table_size,
    max_neighbors as voxel_hash_max_neighbors, VoxelHash,
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
