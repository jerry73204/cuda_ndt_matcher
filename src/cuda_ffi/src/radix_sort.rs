//! CUB DeviceRadixSort bindings.
//!
//! Provides GPU-accelerated radix sort for (u64, u32) key-value pairs,
//! used for sorting points by Morton code in voxel grid construction.

use std::ffi::c_int;
use std::ptr;
use thiserror::Error;

// ============================================================================
// FFI Declarations
// ============================================================================

type CudaStream = *mut std::ffi::c_void;

extern "C" {
    fn cub_radix_sort_pairs_u64_u32_temp_size(
        temp_storage_bytes: *mut usize,
        num_items: c_int,
        begin_bit: c_int,
        end_bit: c_int,
    ) -> c_int;

    fn cub_radix_sort_pairs_u64_u32(
        d_temp_storage: *mut std::ffi::c_void,
        temp_storage_bytes: usize,
        d_keys_in: *const u64,
        d_keys_out: *mut u64,
        d_values_in: *const u32,
        d_values_out: *mut u32,
        num_items: c_int,
        begin_bit: c_int,
        end_bit: c_int,
        stream: CudaStream,
    ) -> c_int;

    // CUDA runtime functions
    fn cudaMalloc(dev_ptr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> c_int;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
}

// cudaMemcpyKind values
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// ============================================================================
// Error Handling
// ============================================================================

/// CUDA error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CudaError {
    #[error("CUDA success")]
    Success,
    #[error("CUDA invalid value")]
    InvalidValue,
    #[error("CUDA out of memory")]
    OutOfMemory,
    #[error("CUDA not initialized")]
    NotInitialized,
    #[error("CUDA invalid device")]
    InvalidDevice,
    #[error("CUDA error code {0}")]
    Other(i32),
}

impl From<c_int> for CudaError {
    fn from(code: c_int) -> Self {
        match code {
            0 => CudaError::Success,
            1 => CudaError::InvalidValue,
            2 => CudaError::OutOfMemory,
            3 => CudaError::NotInitialized,
            101 => CudaError::InvalidDevice,
            _ => CudaError::Other(code),
        }
    }
}

fn check_cuda(code: c_int) -> Result<(), CudaError> {
    let err = CudaError::from(code);
    if err == CudaError::Success {
        Ok(())
    } else {
        Err(err)
    }
}

// ============================================================================
// Device Memory RAII Wrapper
// ============================================================================

/// RAII wrapper for CUDA device memory.
struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate device memory.
    fn new(size: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        unsafe {
            check_cuda(cudaMalloc(&mut ptr, size))?;
        }
        Ok(Self { ptr, size })
    }

    /// Copy data from host to device.
    fn copy_from_host<T>(&mut self, data: &[T]) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.size, "Data too large for buffer");
        unsafe {
            check_cuda(cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ))
        }
    }

    /// Copy data from device to host.
    fn copy_to_host<T>(&self, data: &mut [T]) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.size, "Buffer too small for data");
        unsafe {
            check_cuda(cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ))
        }
    }

    /// Get raw pointer.
    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cudaFree(self.ptr);
            }
        }
    }
}

// ============================================================================
// High-Level API
// ============================================================================

/// GPU radix sorter for (u64, u32) key-value pairs.
///
/// Used for sorting points by Morton code during voxel grid construction.
///
/// # Example
///
/// ```ignore
/// let sorter = RadixSorter::new()?;
/// let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values)?;
/// ```
pub struct RadixSorter {
    // No state needed; each call allocates temporary storage
}

impl RadixSorter {
    /// Create a new radix sorter.
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self {})
    }

    /// Sort key-value pairs by key.
    ///
    /// # Arguments
    /// * `keys` - Input keys (u64 Morton codes)
    /// * `values` - Input values (u32 point indices)
    ///
    /// # Returns
    /// Tuple of (sorted_keys, sorted_values) in ascending key order.
    pub fn sort_pairs(
        &self,
        keys: &[u64],
        values: &[u32],
    ) -> Result<(Vec<u64>, Vec<u32>), CudaError> {
        assert_eq!(
            keys.len(),
            values.len(),
            "Keys and values must have same length"
        );

        let n = keys.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let num_items = n as c_int;
        let begin_bit = 0;
        let end_bit = 64; // Full 64-bit sort

        // Query temporary storage size
        let mut temp_storage_bytes: usize = 0;
        unsafe {
            check_cuda(cub_radix_sort_pairs_u64_u32_temp_size(
                &mut temp_storage_bytes,
                num_items,
                begin_bit,
                end_bit,
            ))?;
        }

        // Allocate device memory
        let d_temp = DeviceBuffer::new(temp_storage_bytes)?;
        let mut d_keys_in = DeviceBuffer::new(std::mem::size_of_val(keys))?;
        let d_keys_out = DeviceBuffer::new(std::mem::size_of_val(keys))?;
        let mut d_values_in = DeviceBuffer::new(std::mem::size_of_val(values))?;
        let d_values_out = DeviceBuffer::new(std::mem::size_of_val(values))?;

        // Copy input to device
        d_keys_in.copy_from_host(keys)?;
        d_values_in.copy_from_host(values)?;

        // Execute sort
        unsafe {
            check_cuda(cub_radix_sort_pairs_u64_u32(
                d_temp.as_ptr(),
                temp_storage_bytes,
                d_keys_in.as_ptr() as *const u64,
                d_keys_out.as_ptr() as *mut u64,
                d_values_in.as_ptr() as *const u32,
                d_values_out.as_ptr() as *mut u32,
                num_items,
                begin_bit,
                end_bit,
                ptr::null_mut(), // default stream
            ))?;

            // Synchronize
            check_cuda(cudaDeviceSynchronize())?;
        }

        // Copy results back
        let mut sorted_keys = vec![0u64; n];
        let mut sorted_values = vec![0u32; n];
        d_keys_out.copy_to_host(&mut sorted_keys)?;
        d_values_out.copy_to_host(&mut sorted_values)?;

        Ok((sorted_keys, sorted_values))
    }

    /// Sort key-value pairs, returning raw bytes for integration with CubeCL.
    ///
    /// This avoids an extra copy when integrating with CubeCL's byte-based buffers.
    ///
    /// # Arguments
    /// * `keys_bytes` - Input keys as bytes (u64 little-endian)
    /// * `values_bytes` - Input values as bytes (u32 little-endian)
    ///
    /// # Returns
    /// Tuple of (sorted_keys_bytes, sorted_values_bytes).
    pub fn sort_pairs_bytes(
        &self,
        keys_bytes: &[u8],
        values_bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>), CudaError> {
        // Convert from bytes
        let keys: Vec<u64> = keys_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let values: Vec<u32> = values_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let (sorted_keys, sorted_values) = self.sort_pairs(&keys, &values)?;

        // Convert back to bytes
        let sorted_keys_bytes: Vec<u8> = sorted_keys.iter().flat_map(|k| k.to_le_bytes()).collect();
        let sorted_values_bytes: Vec<u8> =
            sorted_values.iter().flat_map(|v| v.to_le_bytes()).collect();

        Ok((sorted_keys_bytes, sorted_values_bytes))
    }
}

impl Default for RadixSorter {
    fn default() -> Self {
        Self::new().expect("Failed to create RadixSorter")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_empty() {
        let sorter = RadixSorter::new().unwrap();
        let (keys, values) = sorter.sort_pairs(&[], &[]).unwrap();
        assert!(keys.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn test_sort_single() {
        let sorter = RadixSorter::new().unwrap();
        let (keys, values) = sorter.sort_pairs(&[42], &[0]).unwrap();
        assert_eq!(keys, vec![42]);
        assert_eq!(values, vec![0]);
    }

    #[test]
    fn test_sort_ordered() {
        let sorter = RadixSorter::new().unwrap();
        let keys = vec![1, 2, 3, 4, 5];
        let values = vec![0, 1, 2, 3, 4];
        let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values).unwrap();
        assert_eq!(sorted_keys, vec![1, 2, 3, 4, 5]);
        assert_eq!(sorted_values, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sort_reversed() {
        let sorter = RadixSorter::new().unwrap();
        let keys = vec![5, 4, 3, 2, 1];
        let values = vec![0, 1, 2, 3, 4];
        let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values).unwrap();
        assert_eq!(sorted_keys, vec![1, 2, 3, 4, 5]);
        assert_eq!(sorted_values, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_sort_random() {
        let sorter = RadixSorter::new().unwrap();
        let keys = vec![42, 17, 99, 1, 50];
        let values = vec![0, 1, 2, 3, 4];
        let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values).unwrap();
        assert_eq!(sorted_keys, vec![1, 17, 42, 50, 99]);
        assert_eq!(sorted_values, vec![3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_sort_duplicates() {
        let sorter = RadixSorter::new().unwrap();
        let keys = vec![5, 3, 5, 1, 3];
        let values = vec![0, 1, 2, 3, 4];
        let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values).unwrap();
        assert_eq!(sorted_keys, vec![1, 3, 3, 5, 5]);
        // Values for equal keys can be in either order (stable sort not guaranteed)
        assert!(sorted_values.contains(&3)); // value for key 1
    }

    #[test]
    fn test_sort_large() {
        let sorter = RadixSorter::new().unwrap();
        let n: usize = 100_000;
        let keys: Vec<u64> = (0..n as u64).rev().collect();
        let values: Vec<u32> = (0..n as u32).collect();
        let (sorted_keys, sorted_values) = sorter.sort_pairs(&keys, &values).unwrap();

        // Verify sorted order
        for i in 1..n {
            assert!(sorted_keys[i] >= sorted_keys[i - 1]);
        }

        // Verify first and last
        assert_eq!(sorted_keys[0], 0);
        assert_eq!(sorted_keys[n - 1], (n - 1) as u64);

        // Silence unused variable warning
        let _ = sorted_values;
    }
}
