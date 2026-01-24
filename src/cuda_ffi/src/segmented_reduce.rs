//! CUB DeviceSegmentedReduce bindings.
//!
//! Provides GPU-accelerated segmented reduction for summing data in segments,
//! used for reducing per-point gradients and Hessians to totals in NDT.

use std::ffi::c_int;
use std::ptr;

use crate::radix_sort::{check_cuda, CudaError, DeviceBuffer};

// ============================================================================
// FFI Declarations
// ============================================================================

type CudaStream = *mut std::ffi::c_void;

extern "C" {
    fn cub_segmented_reduce_sum_f32_temp_size(
        temp_storage_bytes: *mut usize,
        num_items: c_int,
        num_segments: c_int,
    ) -> c_int;

    fn cub_segmented_reduce_sum_f32(
        d_temp_storage: *mut std::ffi::c_void,
        temp_storage_bytes: usize,
        d_in: *const f32,
        d_out: *mut f32,
        num_segments: c_int,
        d_offsets: *const c_int,
        stream: CudaStream,
    ) -> c_int;

    fn cub_segmented_reduce_sum_f64_temp_size(
        temp_storage_bytes: *mut usize,
        num_items: c_int,
        num_segments: c_int,
    ) -> c_int;

    fn cub_segmented_reduce_sum_f64(
        d_temp_storage: *mut std::ffi::c_void,
        temp_storage_bytes: usize,
        d_in: *const f64,
        d_out: *mut f64,
        num_segments: c_int,
        d_offsets: *const c_int,
        stream: CudaStream,
    ) -> c_int;

    // CUDA runtime functions (from radix_sort.rs, but we need them here too)
    fn cudaDeviceSynchronize() -> c_int;
}

// ============================================================================
// High-Level API
// ============================================================================

/// GPU segmented reducer for summing data in segments.
///
/// Used for reducing per-point scores, gradients, and Hessians to totals.
///
/// # Example
///
/// ```ignore
/// let reducer = SegmentedReducer::new()?;
///
/// // Sum 3 segments: [0..4), [4..7), [7..10)
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let offsets = vec![0, 4, 7, 10];
/// let sums = reducer.sum_f32(&data, &offsets)?;
/// // sums = [10.0, 18.0, 27.0]
/// ```
pub struct SegmentedReducer {
    // No state needed; each call allocates temporary storage
}

impl SegmentedReducer {
    /// Create a new segmented reducer.
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self {})
    }

    /// Sum f32 data in segments.
    ///
    /// # Arguments
    /// * `data` - Input data (all segments concatenated)
    /// * `offsets` - Segment offsets, length = num_segments + 1
    ///
    /// # Returns
    /// Vector of sums, one per segment.
    pub fn sum_f32(&self, data: &[f32], offsets: &[i32]) -> Result<Vec<f32>, CudaError> {
        if offsets.len() < 2 {
            return Ok(Vec::new());
        }

        let num_segments = (offsets.len() - 1) as c_int;
        let num_items = data.len() as c_int;

        // Query temporary storage size
        let mut temp_storage_bytes: usize = 0;
        unsafe {
            check_cuda(cub_segmented_reduce_sum_f32_temp_size(
                &mut temp_storage_bytes,
                num_items,
                num_segments,
            ))?;
        }

        // Allocate device memory
        let d_temp = DeviceBuffer::new(temp_storage_bytes.max(1))?;
        let mut d_in = DeviceBuffer::new(std::mem::size_of_val(data).max(4))?;
        let d_out = DeviceBuffer::new((num_segments as usize) * std::mem::size_of::<f32>())?;
        let mut d_offsets = DeviceBuffer::new(std::mem::size_of_val(offsets))?;

        // Copy input to device
        d_in.copy_from_host(data)?;
        d_offsets.copy_from_host(offsets)?;

        // Execute segmented reduce
        unsafe {
            check_cuda(cub_segmented_reduce_sum_f32(
                d_temp.as_ptr(),
                temp_storage_bytes,
                d_in.as_ptr() as *const f32,
                d_out.as_ptr() as *mut f32,
                num_segments,
                d_offsets.as_ptr() as *const c_int,
                ptr::null_mut(), // default stream
            ))?;

            check_cuda(cudaDeviceSynchronize())?;
        }

        // Copy results back
        let mut sums = vec![0.0f32; num_segments as usize];
        d_out.copy_to_host(&mut sums)?;

        Ok(sums)
    }

    /// Sum f64 data in segments.
    ///
    /// # Arguments
    /// * `data` - Input data (all segments concatenated)
    /// * `offsets` - Segment offsets, length = num_segments + 1
    ///
    /// # Returns
    /// Vector of sums, one per segment.
    pub fn sum_f64(&self, data: &[f64], offsets: &[i32]) -> Result<Vec<f64>, CudaError> {
        if offsets.len() < 2 {
            return Ok(Vec::new());
        }

        let num_segments = (offsets.len() - 1) as c_int;
        let num_items = data.len() as c_int;

        // Query temporary storage size
        let mut temp_storage_bytes: usize = 0;
        unsafe {
            check_cuda(cub_segmented_reduce_sum_f64_temp_size(
                &mut temp_storage_bytes,
                num_items,
                num_segments,
            ))?;
        }

        // Allocate device memory
        let d_temp = DeviceBuffer::new(temp_storage_bytes.max(1))?;
        let mut d_in = DeviceBuffer::new(std::mem::size_of_val(data).max(8))?;
        let d_out = DeviceBuffer::new((num_segments as usize) * std::mem::size_of::<f64>())?;
        let mut d_offsets = DeviceBuffer::new(std::mem::size_of_val(offsets))?;

        // Copy input to device
        d_in.copy_from_host(data)?;
        d_offsets.copy_from_host(offsets)?;

        // Execute segmented reduce
        unsafe {
            check_cuda(cub_segmented_reduce_sum_f64(
                d_temp.as_ptr(),
                temp_storage_bytes,
                d_in.as_ptr() as *const f64,
                d_out.as_ptr() as *mut f64,
                num_segments,
                d_offsets.as_ptr() as *const c_int,
                ptr::null_mut(), // default stream
            ))?;

            check_cuda(cudaDeviceSynchronize())?;
        }

        // Copy results back
        let mut sums = vec![0.0f64; num_segments as usize];
        d_out.copy_to_host(&mut sums)?;

        Ok(sums)
    }
}

impl Default for SegmentedReducer {
    fn default() -> Self {
        Self::new().expect("Failed to create SegmentedReducer")
    }
}

// ============================================================================
// In-Place API (for zero-copy pipeline)
// ============================================================================

/// Query temporary storage size for segmented reduce sum (f32).
///
/// # Arguments
/// * `num_items` - Total number of items across all segments
/// * `num_segments` - Number of segments
///
/// # Returns
/// Required temporary storage size in bytes.
pub fn segmented_reduce_sum_f32_temp_size(
    num_items: usize,
    num_segments: usize,
) -> Result<usize, CudaError> {
    let mut temp_bytes: usize = 0;
    unsafe {
        check_cuda(cub_segmented_reduce_sum_f32_temp_size(
            &mut temp_bytes,
            num_items as c_int,
            num_segments as c_int,
        ))?;
    }
    Ok(temp_bytes)
}

/// Perform segmented sum on f32 data using pre-allocated GPU buffers.
///
/// This function operates directly on GPU memory without any CPU-GPU transfers.
/// All pointers must be valid CUDA device pointers (CUdeviceptr).
///
/// # Arguments
/// * `d_temp` - Device pointer to temporary storage
/// * `temp_bytes` - Size of temporary storage
/// * `d_in` - Device pointer to input data (f32)
/// * `d_out` - Device pointer to output sums (f32), one per segment
/// * `num_segments` - Number of segments
/// * `d_offsets` - Device pointer to segment offsets (i32), length = num_segments + 1
///
/// # Safety
/// All device pointers must be valid and have sufficient allocated size.
pub unsafe fn segmented_reduce_sum_f32_inplace(
    d_temp: u64,
    temp_bytes: usize,
    d_in: u64,
    d_out: u64,
    num_segments: usize,
    d_offsets: u64,
) -> Result<(), CudaError> {
    if num_segments == 0 {
        return Ok(());
    }

    check_cuda(cub_segmented_reduce_sum_f32(
        d_temp as *mut std::ffi::c_void,
        temp_bytes,
        d_in as *const f32,
        d_out as *mut f32,
        num_segments as c_int,
        d_offsets as *const c_int,
        ptr::null_mut(), // default stream
    ))?;

    check_cuda(cudaDeviceSynchronize())
}

/// Query temporary storage size for segmented reduce sum (f64).
pub fn segmented_reduce_sum_f64_temp_size(
    num_items: usize,
    num_segments: usize,
) -> Result<usize, CudaError> {
    let mut temp_bytes: usize = 0;
    unsafe {
        check_cuda(cub_segmented_reduce_sum_f64_temp_size(
            &mut temp_bytes,
            num_items as c_int,
            num_segments as c_int,
        ))?;
    }
    Ok(temp_bytes)
}

/// Perform segmented sum on f64 data using pre-allocated GPU buffers.
///
/// # Safety
/// All device pointers must be valid and have sufficient allocated size.
pub unsafe fn segmented_reduce_sum_f64_inplace(
    d_temp: u64,
    temp_bytes: usize,
    d_in: u64,
    d_out: u64,
    num_segments: usize,
    d_offsets: u64,
) -> Result<(), CudaError> {
    if num_segments == 0 {
        return Ok(());
    }

    check_cuda(cub_segmented_reduce_sum_f64(
        d_temp as *mut std::ffi::c_void,
        temp_bytes,
        d_in as *const f64,
        d_out as *mut f64,
        num_segments as c_int,
        d_offsets as *const c_int,
        ptr::null_mut(), // default stream
    ))?;

    check_cuda(cudaDeviceSynchronize())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_sum_empty() {
        let reducer = SegmentedReducer::new().unwrap();
        let sums = reducer.sum_f32(&[], &[]).unwrap();
        assert!(sums.is_empty());
    }
    #[test]
    fn test_sum_single_segment() {
        let reducer = SegmentedReducer::new().unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let offsets = vec![0, 4];
        let sums = reducer.sum_f32(&data, &offsets).unwrap();
        assert_eq!(sums.len(), 1);
        assert!((sums[0] - 10.0).abs() < 1e-5);
    }
    #[test]
    fn test_sum_multiple_segments() {
        let reducer = SegmentedReducer::new().unwrap();
        // 3 segments: [0..4), [4..7), [7..10)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let offsets = vec![0, 4, 7, 10];
        let sums = reducer.sum_f32(&data, &offsets).unwrap();
        assert_eq!(sums.len(), 3);
        assert!((sums[0] - 10.0).abs() < 1e-5); // 1+2+3+4
        assert!((sums[1] - 18.0).abs() < 1e-5); // 5+6+7
        assert!((sums[2] - 27.0).abs() < 1e-5); // 8+9+10
    }
    #[test]
    fn test_sum_equal_segments() {
        let reducer = SegmentedReducer::new().unwrap();
        // 4 segments of size 3 each
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let offsets = vec![0, 3, 6, 9, 12];
        let sums = reducer.sum_f32(&data, &offsets).unwrap();
        assert_eq!(sums.len(), 4);
        assert!((sums[0] - 6.0).abs() < 1e-5); // 1+2+3
        assert!((sums[1] - 15.0).abs() < 1e-5); // 4+5+6
        assert!((sums[2] - 24.0).abs() < 1e-5); // 7+8+9
        assert!((sums[3] - 33.0).abs() < 1e-5); // 10+11+12
    }
    #[test]
    fn test_sum_f64() {
        let reducer = SegmentedReducer::new().unwrap();
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let offsets = vec![0, 3, 6];
        let sums = reducer.sum_f64(&data, &offsets).unwrap();
        assert_eq!(sums.len(), 2);
        assert!((sums[0] - 6.0).abs() < 1e-10);
        assert!((sums[1] - 15.0).abs() < 1e-10);
    }
    #[test]
    fn test_sum_large() {
        let reducer = SegmentedReducer::new().unwrap();
        let n = 10000;
        let num_segments = 43; // Matching NDT use case
        let segment_size = n / num_segments;

        // Create data where each segment sums to segment_index * segment_size
        let mut data = Vec::with_capacity(n);
        for seg in 0..num_segments {
            for _ in 0..segment_size {
                data.push(seg as f32);
            }
        }
        // Pad to exact size if needed
        while data.len() < n {
            data.push(0.0);
        }

        let offsets: Vec<i32> = (0..=num_segments)
            .map(|i| (i * segment_size) as i32)
            .collect();

        let sums = reducer.sum_f32(&data, &offsets).unwrap();
        assert_eq!(sums.len(), num_segments);

        for (seg, sum) in sums.iter().enumerate() {
            let expected = (seg * segment_size) as f32;
            assert!(
                (sum - expected).abs() < 1e-3,
                "Segment {seg}: expected {expected}, got {sum}"
            );
        }
    }
    #[test]
    fn test_sum_43_segments() {
        // Test the exact NDT use case: 43 segments (1 score + 6 gradient + 36 hessian)
        let reducer = SegmentedReducer::new().unwrap();
        let num_points = 1000;
        let num_segments = 43;

        // Create column-major data: [seg0_pt0, seg0_pt1, ..., seg0_ptN, seg1_pt0, ...]
        let mut data = vec![0.0f32; num_points * num_segments];
        for seg in 0..num_segments {
            for pt in 0..num_points {
                // Each segment sums to (seg + 1) * num_points
                data[seg * num_points + pt] = (seg + 1) as f32;
            }
        }

        let offsets: Vec<i32> = (0..=num_segments)
            .map(|i| (i * num_points) as i32)
            .collect();

        let sums = reducer.sum_f32(&data, &offsets).unwrap();
        assert_eq!(sums.len(), num_segments);

        for (seg, sum) in sums.iter().enumerate() {
            let expected = ((seg + 1) * num_points) as f32;
            assert!(
                (sum - expected).abs() < 1e-2,
                "Segment {seg}: expected {expected}, got {sum}"
            );
        }
    }
}
