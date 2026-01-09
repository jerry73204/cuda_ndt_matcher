//! GPU radix sort for Morton codes.
//!
//! Provides GPU-accelerated radix sort using CUB DeviceRadixSort via cuda_ffi.
//! Falls back to CPU implementation if GPU sort fails.
//!
//! # Algorithm
//!
//! Uses NVIDIA CUB's DeviceRadixSort which implements an efficient
//! parallel radix sort optimized for GPU execution.

use cuda_ffi::RadixSorter;

/// Radix (number of possible values per digit).
/// 4-bit radix = 16 values per digit.
const RADIX_BITS: u32 = 4;
const RADIX: usize = 1 << RADIX_BITS; // 16

/// Number of passes for 64-bit keys.
const NUM_PASSES: u32 = 64 / RADIX_BITS; // 16

/// Result of radix sort operation.
#[derive(Debug)]
pub struct RadixSortResult {
    /// Sorted keys (raw bytes, interpret as u64).
    pub keys: Vec<u8>,
    /// Reordered values (raw bytes, interpret as u32).
    pub values: Vec<u8>,
    /// Number of elements.
    pub num_elements: u32,
}

/// Perform radix sort on Morton codes with associated indices using GPU (CUB).
///
/// Uses NVIDIA CUB's DeviceRadixSort for GPU-accelerated sorting.
///
/// # Arguments
/// * `keys` - Morton codes to sort
/// * `values` - Associated values (original indices)
///
/// # Returns
/// Sorted keys and reordered values, or error if GPU sort fails.
pub fn radix_sort_by_key_gpu(
    keys: &[u64],
    values: &[u32],
) -> Result<RadixSortResult, cuda_ffi::CudaError> {
    let n = keys.len();
    if n == 0 {
        return Ok(RadixSortResult {
            keys: Vec::new(),
            values: Vec::new(),
            num_elements: 0,
        });
    }

    let sorter = RadixSorter::new()?;
    let (sorted_keys, sorted_values) = sorter.sort_pairs(keys, values)?;

    // Convert to bytes
    let key_bytes: Vec<u8> = sorted_keys.iter().flat_map(|k| k.to_le_bytes()).collect();
    let value_bytes: Vec<u8> = sorted_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    Ok(RadixSortResult {
        keys: key_bytes,
        values: value_bytes,
        num_elements: n as u32,
    })
}

/// Perform radix sort on Morton codes, using GPU with CPU fallback.
///
/// Tries GPU sort first, falls back to CPU if GPU fails.
///
/// # Arguments
/// * `keys` - Morton codes to sort
/// * `values` - Associated values (original indices)
///
/// # Returns
/// Sorted keys and reordered values.
pub fn radix_sort_by_key(keys: &[u64], values: &[u32]) -> RadixSortResult {
    match radix_sort_by_key_gpu(keys, values) {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!("GPU radix sort failed ({e}), falling back to CPU");
            radix_sort_by_key_cpu(keys, values)
        }
    }
}

/// Perform radix sort on Morton codes with associated indices (CPU reference).
///
/// # Arguments
/// * `keys` - Morton codes to sort
/// * `values` - Associated values (original indices)
///
/// # Returns
/// Sorted keys and reordered values.
pub fn radix_sort_by_key_cpu(keys: &[u64], values: &[u32]) -> RadixSortResult {
    let n = keys.len();
    if n == 0 {
        return RadixSortResult {
            keys: Vec::new(),
            values: Vec::new(),
            num_elements: 0,
        };
    }

    let mut keys_a = keys.to_vec();
    let mut values_a = values.to_vec();
    let mut keys_b = vec![0u64; n];
    let mut values_b = vec![0u32; n];

    for pass in 0..NUM_PASSES {
        let shift = pass * RADIX_BITS;

        // Count histogram
        let mut hist = [0usize; RADIX];
        for &k in &keys_a {
            let digit = ((k >> shift) & (RADIX as u64 - 1)) as usize;
            hist[digit] += 1;
        }

        // Prefix sum
        let mut sum = 0;
        let mut offsets = [0usize; RADIX];
        for i in 0..RADIX {
            offsets[i] = sum;
            sum += hist[i];
        }

        // Scatter
        let mut counts = [0usize; RADIX];
        for i in 0..n {
            let digit = ((keys_a[i] >> shift) & (RADIX as u64 - 1)) as usize;
            let dest = offsets[digit] + counts[digit];
            counts[digit] += 1;
            keys_b[dest] = keys_a[i];
            values_b[dest] = values_a[i];
        }

        std::mem::swap(&mut keys_a, &mut keys_b);
        std::mem::swap(&mut values_a, &mut values_b);
    }

    // Convert to bytes
    let mut key_bytes = Vec::with_capacity(n * 8);
    let mut value_bytes = Vec::with_capacity(n * 4);
    for i in 0..n {
        key_bytes.extend_from_slice(&keys_a[i].to_le_bytes());
        value_bytes.extend_from_slice(&values_a[i].to_le_bytes());
    }

    RadixSortResult {
        keys: key_bytes,
        values: value_bytes,
        num_elements: n as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort_cpu() {
        let keys = vec![5u64, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let values: Vec<u32> = (0..10).collect();

        let result = radix_sort_by_key_cpu(&keys, &values);

        // Parse result keys
        let sorted_keys: Vec<u64> = result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();

        assert_eq!(sorted_keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_radix_sort_preserves_order() {
        // Test stability: equal keys preserve original order
        let keys = vec![1u64, 1, 1, 1, 1];
        let values: Vec<u32> = vec![0, 1, 2, 3, 4];

        let result = radix_sort_by_key_cpu(&keys, &values);

        let sorted_values: Vec<u32> = result
            .values
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Stable sort should preserve order
        assert_eq!(sorted_values, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_radix_sort_empty() {
        let result = radix_sort_by_key_cpu(&[], &[]);
        assert_eq!(result.num_elements, 0);
    }

    #[test]
    fn test_radix_sort_gpu() {
        let keys = vec![5u64, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let values: Vec<u32> = (0..10).collect();

        let result = radix_sort_by_key_gpu(&keys, &values).expect("GPU sort should succeed");

        let sorted_keys: Vec<u64> = result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();

        assert_eq!(sorted_keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_radix_sort_gpu_vs_cpu() {
        // Test that GPU and CPU produce identical results
        let keys = vec![42u64, 17, 99, 1, 50, 33, 88, 5, 77, 23];
        let values: Vec<u32> = (0..10).collect();

        let cpu_result = radix_sort_by_key_cpu(&keys, &values);
        let gpu_result = radix_sort_by_key_gpu(&keys, &values).expect("GPU sort should succeed");

        assert_eq!(cpu_result.keys, gpu_result.keys);
        assert_eq!(cpu_result.values, gpu_result.values);
    }

    #[test]
    fn test_radix_sort_with_fallback() {
        // Test the unified function that uses GPU with CPU fallback
        let keys = vec![100u64, 50, 75, 25, 0];
        let values: Vec<u32> = (0..5).collect();

        let result = radix_sort_by_key(&keys, &values);

        let sorted_keys: Vec<u64> = result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();

        assert_eq!(sorted_keys, vec![0, 25, 50, 75, 100]);
    }

    #[test]
    fn test_radix_sort_gpu_large() {
        // Test with larger dataset to exercise GPU parallelism
        let n: usize = 100_000;
        let keys: Vec<u64> = (0..n as u64).rev().collect();
        let values: Vec<u32> = (0..n as u32).collect();

        let result = radix_sort_by_key_gpu(&keys, &values).expect("GPU sort should succeed");

        let sorted_keys: Vec<u64> = result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Verify sorted order
        for i in 1..n {
            assert!(sorted_keys[i] >= sorted_keys[i - 1]);
        }

        assert_eq!(sorted_keys[0], 0);
        assert_eq!(sorted_keys[n - 1], (n - 1) as u64);
    }
}
