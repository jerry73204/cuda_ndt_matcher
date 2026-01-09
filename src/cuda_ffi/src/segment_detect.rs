//! GPU segment detection for voxel boundary identification.
//!
//! Detects segment boundaries in sorted Morton codes using CUB primitives.
//! A segment is a contiguous run of identical Morton codes (same voxel).

use std::ffi::c_int;
use std::ptr;

use crate::radix_sort::{check_cuda, CudaError, DeviceBuffer};

// ============================================================================
// FFI Declarations
// ============================================================================

type CudaStream = *mut std::ffi::c_void;

extern "C" {
    // Boundary detection
    fn cub_detect_boundaries(
        d_sorted_codes: *const u64,
        d_boundaries: *mut u32,
        num_items: c_int,
        stream: CudaStream,
    ) -> c_int;

    // Inclusive prefix sum
    fn cub_inclusive_sum_u32_temp_size(temp_storage_bytes: *mut usize, num_items: c_int) -> c_int;

    fn cub_inclusive_sum_u32(
        d_temp_storage: *mut std::ffi::c_void,
        temp_storage_bytes: usize,
        d_in: *const u32,
        d_out: *mut u32,
        num_items: c_int,
        stream: CudaStream,
    ) -> c_int;

    // Stream compaction
    fn cub_select_flagged_u32_temp_size(temp_storage_bytes: *mut usize, num_items: c_int) -> c_int;

    fn cub_select_flagged_u32(
        d_temp_storage: *mut std::ffi::c_void,
        temp_storage_bytes: usize,
        d_in: *const u32,
        d_flags: *const u32,
        d_out: *mut u32,
        d_num_selected: *mut c_int,
        num_items: c_int,
        stream: CudaStream,
    ) -> c_int;

    // Index sequence generation
    fn cub_iota_u32(d_output: *mut u32, num_items: c_int, stream: CudaStream) -> c_int;

    // CUDA runtime
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMalloc(dev_ptr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> c_int;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
}

const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// ============================================================================
// High-Level API
// ============================================================================

/// Result of segment detection.
#[derive(Debug, Clone)]
pub struct SegmentResult {
    /// Segment ID for each point (which voxel it belongs to).
    pub segment_ids: Vec<u32>,
    /// Start index of each segment (one per unique voxel).
    pub segment_starts: Vec<u32>,
    /// Morton code for each segment.
    pub segment_codes: Vec<u64>,
    /// Number of points.
    pub num_points: u32,
    /// Number of unique segments (voxels).
    pub num_segments: u32,
}

/// GPU segment detector.
pub struct SegmentDetector;

impl SegmentDetector {
    /// Create a new segment detector.
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self)
    }

    /// Detect segments in sorted Morton codes using GPU.
    ///
    /// # Arguments
    /// * `sorted_codes` - Morton codes sorted in ascending order
    ///
    /// # Returns
    /// Segment information including starts, codes, and per-point segment IDs.
    pub fn detect_segments(&self, sorted_codes: &[u64]) -> Result<SegmentResult, CudaError> {
        let n = sorted_codes.len();
        if n == 0 {
            return Ok(SegmentResult {
                segment_ids: Vec::new(),
                segment_starts: Vec::new(),
                segment_codes: Vec::new(),
                num_points: 0,
                num_segments: 0,
            });
        }

        let num_items = n as c_int;

        // Allocate device memory for input
        let d_codes = DeviceBuffer::new(std::mem::size_of_val(sorted_codes))?;
        unsafe {
            check_cuda(cudaMemcpy(
                d_codes.as_ptr(),
                sorted_codes.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(sorted_codes),
                1, // cudaMemcpyHostToDevice
            ))?;
        }

        // Step 1: Detect boundaries
        let d_boundaries = DeviceBuffer::new(n * std::mem::size_of::<u32>())?;
        unsafe {
            check_cuda(cub_detect_boundaries(
                d_codes.as_ptr() as *const u64,
                d_boundaries.as_ptr() as *mut u32,
                num_items,
                ptr::null_mut(),
            ))?;
        }

        // Step 2: Inclusive prefix sum on boundaries -> segment_ids
        // For codes [1,1,1,5,5,9,9,9,9]:
        //   boundaries = [0, 0, 0, 1, 0, 1, 0, 0, 0]
        //   inclusive_sum = [0, 0, 0, 1, 1, 2, 2, 2, 2] <- segment_ids
        let mut temp_bytes: usize = 0;
        unsafe {
            check_cuda(cub_inclusive_sum_u32_temp_size(&mut temp_bytes, num_items))?;
        }

        let d_temp = DeviceBuffer::new(temp_bytes)?;
        let d_segment_ids = DeviceBuffer::new(n * std::mem::size_of::<u32>())?;
        unsafe {
            check_cuda(cub_inclusive_sum_u32(
                d_temp.as_ptr(),
                temp_bytes,
                d_boundaries.as_ptr() as *const u32,
                d_segment_ids.as_ptr() as *mut u32,
                num_items,
                ptr::null_mut(),
            ))?;
        }

        // Step 3: Stream compaction to get segment starts
        // First, generate index sequence [0, 1, 2, ..., n-1]
        let d_indices = DeviceBuffer::new(n * std::mem::size_of::<u32>())?;
        unsafe {
            check_cuda(cub_iota_u32(
                d_indices.as_ptr() as *mut u32,
                num_items,
                ptr::null_mut(),
            ))?;
        }

        // Query temp storage for select
        let mut select_temp_bytes: usize = 0;
        unsafe {
            check_cuda(cub_select_flagged_u32_temp_size(
                &mut select_temp_bytes,
                num_items,
            ))?;
        }

        let d_select_temp = DeviceBuffer::new(select_temp_bytes)?;
        let d_segment_starts = DeviceBuffer::new(n * std::mem::size_of::<u32>())?; // Max size

        // Allocate device memory for num_selected
        let mut d_num_selected_ptr: *mut std::ffi::c_void = ptr::null_mut();
        unsafe {
            check_cuda(cudaMalloc(
                &mut d_num_selected_ptr,
                std::mem::size_of::<c_int>(),
            ))?;
        }

        // Select indices where boundaries[i] == 1
        unsafe {
            check_cuda(cub_select_flagged_u32(
                d_select_temp.as_ptr(),
                select_temp_bytes,
                d_indices.as_ptr() as *const u32,
                d_boundaries.as_ptr() as *const u32,
                d_segment_starts.as_ptr() as *mut u32,
                d_num_selected_ptr as *mut c_int,
                num_items,
                ptr::null_mut(),
            ))?;
            check_cuda(cudaDeviceSynchronize())?;
        }

        // Copy num_selected (boundary count) back to host
        // num_segments = num_boundaries + 1 (first segment has no boundary marker)
        let mut num_boundaries: c_int = 0;
        unsafe {
            check_cuda(cudaMemcpy(
                &mut num_boundaries as *mut c_int as *mut std::ffi::c_void,
                d_num_selected_ptr,
                std::mem::size_of::<c_int>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ))?;
            let _ = cudaFree(d_num_selected_ptr);
        }
        let num_segments = num_boundaries + 1;

        // Copy intermediate segment starts (without the first 0) to host
        let mut boundary_starts = vec![0u32; num_boundaries as usize];
        if num_boundaries > 0 {
            unsafe {
                check_cuda(cudaMemcpy(
                    boundary_starts.as_mut_ptr() as *mut std::ffi::c_void,
                    d_segment_starts.as_ptr(),
                    num_boundaries as usize * std::mem::size_of::<u32>(),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ))?;
            }
        }

        // Build complete segment_starts by prepending 0
        let mut segment_starts = Vec::with_capacity(num_segments as usize);
        segment_starts.push(0u32); // First segment starts at index 0
        segment_starts.extend(boundary_starts);

        // Gather segment codes at segment_starts (on CPU since we have the data)
        let mut segment_codes = Vec::with_capacity(num_segments as usize);
        for &start in &segment_starts {
            segment_codes.push(sorted_codes[start as usize]);
        }

        // Copy segment_ids back to host
        let mut segment_ids = vec![0u32; n];
        unsafe {
            check_cuda(cudaMemcpy(
                segment_ids.as_mut_ptr() as *mut std::ffi::c_void,
                d_segment_ids.as_ptr(),
                n * std::mem::size_of::<u32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ))?;
        }

        Ok(SegmentResult {
            segment_ids,
            segment_starts,
            segment_codes,
            num_points: n as u32,
            num_segments: num_segments as u32,
        })
    }
}

impl Default for SegmentDetector {
    fn default() -> Self {
        Self::new().expect("Failed to create SegmentDetector")
    }
}

// ============================================================================
// In-Place API (for zero-copy pipeline)
// ============================================================================

/// Query temporary storage sizes for segment detection operations.
///
/// # Arguments
/// * `num_items` - Number of input items (sorted codes)
///
/// # Returns
/// Tuple of (inclusive_sum_temp_bytes, select_temp_bytes)
pub fn segment_detect_temp_sizes(num_items: usize) -> Result<(usize, usize), CudaError> {
    let mut sum_temp: usize = 0;
    let mut select_temp: usize = 0;

    unsafe {
        check_cuda(cub_inclusive_sum_u32_temp_size(
            &mut sum_temp,
            num_items as c_int,
        ))?;
        check_cuda(cub_select_flagged_u32_temp_size(
            &mut select_temp,
            num_items as c_int,
        ))?;
    }

    Ok((sum_temp, select_temp))
}

/// Segment counts returned by in-place detection.
#[derive(Debug, Clone, Copy)]
pub struct SegmentCounts {
    /// Number of unique segments (voxels).
    pub num_segments: u32,
}

/// Detect segment boundaries in-place using pre-allocated GPU buffers.
///
/// This function operates directly on GPU memory without any CPU-GPU transfers.
/// All pointers must be valid CUDA device pointers (CUdeviceptr).
///
/// # Arguments
/// * `d_sorted_codes` - Device pointer to sorted Morton codes (u64, input)
/// * `d_boundaries` - Device pointer for boundary flags (u32, scratch)
/// * `d_segment_ids` - Device pointer for segment IDs (u32, output)
/// * `d_indices` - Device pointer for index sequence (u32, scratch)
/// * `d_segment_starts` - Device pointer for segment starts (u32, output, max num_items size)
/// * `d_num_selected` - Device pointer for output count (single i32)
/// * `d_sum_temp` - Device pointer to inclusive sum temp storage
/// * `sum_temp_bytes` - Size of inclusive sum temp storage
/// * `d_select_temp` - Device pointer to select temp storage
/// * `select_temp_bytes` - Size of select temp storage
/// * `num_items` - Number of input codes
///
/// # Returns
/// SegmentCounts with num_segments. The segment_starts buffer will contain
/// num_segments-1 entries (boundaries without the implicit first start at 0).
///
/// # Safety
/// All device pointers must be valid and have sufficient allocated size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn detect_segments_inplace(
    d_sorted_codes: u64,
    d_boundaries: u64,
    d_segment_ids: u64,
    d_indices: u64,
    d_segment_starts: u64,
    d_num_selected: u64,
    d_sum_temp: u64,
    sum_temp_bytes: usize,
    d_select_temp: u64,
    select_temp_bytes: usize,
    num_items: usize,
) -> Result<SegmentCounts, CudaError> {
    if num_items == 0 {
        return Ok(SegmentCounts { num_segments: 0 });
    }

    let n = num_items as c_int;

    // Step 1: Detect boundaries
    check_cuda(cub_detect_boundaries(
        d_sorted_codes as *const u64,
        d_boundaries as *mut u32,
        n,
        ptr::null_mut(),
    ))?;

    // Step 2: Inclusive prefix sum on boundaries -> segment_ids
    check_cuda(cub_inclusive_sum_u32(
        d_sum_temp as *mut std::ffi::c_void,
        sum_temp_bytes,
        d_boundaries as *const u32,
        d_segment_ids as *mut u32,
        n,
        ptr::null_mut(),
    ))?;

    // Step 3: Generate index sequence
    check_cuda(cub_iota_u32(d_indices as *mut u32, n, ptr::null_mut()))?;

    // Step 4: Stream compaction to get segment starts
    check_cuda(cub_select_flagged_u32(
        d_select_temp as *mut std::ffi::c_void,
        select_temp_bytes,
        d_indices as *const u32,
        d_boundaries as *const u32,
        d_segment_starts as *mut u32,
        d_num_selected as *mut c_int,
        n,
        ptr::null_mut(),
    ))?;

    check_cuda(cudaDeviceSynchronize())?;

    // Read num_selected from device
    let mut num_boundaries: c_int = 0;
    check_cuda(cudaMemcpy(
        &mut num_boundaries as *mut c_int as *mut std::ffi::c_void,
        d_num_selected as *const std::ffi::c_void,
        std::mem::size_of::<c_int>(),
        CUDA_MEMCPY_DEVICE_TO_HOST,
    ))?;

    let num_segments = (num_boundaries + 1) as u32;

    Ok(SegmentCounts { num_segments })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_segments_empty() {
        let detector = SegmentDetector::new().unwrap();
        let result = detector.detect_segments(&[]).unwrap();
        assert_eq!(result.num_segments, 0);
        assert!(result.segment_ids.is_empty());
    }

    #[test]
    fn test_detect_segments_single() {
        let detector = SegmentDetector::new().unwrap();
        let codes = vec![42u64];
        let result = detector.detect_segments(&codes).unwrap();

        assert_eq!(result.num_segments, 1);
        assert_eq!(result.segment_ids, vec![0]);
        assert_eq!(result.segment_starts, vec![0]);
        assert_eq!(result.segment_codes, vec![42]);
    }

    #[test]
    fn test_detect_segments_all_same() {
        let detector = SegmentDetector::new().unwrap();
        let codes = vec![42u64; 100];
        let result = detector.detect_segments(&codes).unwrap();

        assert_eq!(result.num_segments, 1);
        assert_eq!(result.segment_ids, vec![0; 100]);
        assert_eq!(result.segment_starts, vec![0]);
        assert_eq!(result.segment_codes, vec![42]);
    }

    #[test]
    fn test_detect_segments_all_different() {
        let detector = SegmentDetector::new().unwrap();
        let codes: Vec<u64> = (0..50).collect();
        let result = detector.detect_segments(&codes).unwrap();

        assert_eq!(result.num_segments, 50);
        let expected_ids: Vec<u32> = (0..50).collect();
        assert_eq!(result.segment_ids, expected_ids);
    }

    #[test]
    fn test_detect_segments_three_groups() {
        let detector = SegmentDetector::new().unwrap();
        // 3 segments: [1,1,1], [5,5], [9,9,9,9]
        let codes = vec![1u64, 1, 1, 5, 5, 9, 9, 9, 9];
        let result = detector.detect_segments(&codes).unwrap();

        assert_eq!(result.num_segments, 3);
        assert_eq!(result.segment_ids, vec![0, 0, 0, 1, 1, 2, 2, 2, 2]);
        assert_eq!(result.segment_starts, vec![0, 3, 5]);
        assert_eq!(result.segment_codes, vec![1, 5, 9]);
    }

    #[test]
    fn test_detect_segments_gpu_vs_expected() {
        let detector = SegmentDetector::new().unwrap();
        let codes = vec![10u64, 10, 20, 20, 20, 30];
        let result = detector.detect_segments(&codes).unwrap();

        assert_eq!(result.num_segments, 3);
        assert_eq!(result.segment_ids, vec![0, 0, 1, 1, 1, 2]);
        assert_eq!(result.segment_starts, vec![0, 2, 5]);
        assert_eq!(result.segment_codes, vec![10, 20, 30]);
    }
}
