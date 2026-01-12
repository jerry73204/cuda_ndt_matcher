//! FFI bindings for CUDA libraries used by cuda_ndt_matcher.
//!
//! This crate provides Rust bindings for:
//! - CUB DeviceRadixSort (GPU radix sort)
//! - CUB DeviceScan (prefix sum)
//! - CUB DeviceSelect (stream compaction)
//! - GPU segment detection for voxel boundaries
//! - cuSOLVER batched Cholesky solver
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
pub mod radix_sort;
pub mod segment_detect;
pub mod segmented_reduce;

pub use batched_solve::{BatchedCholeskySolver, CusolverDnHandle, CusolverError};
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
