// CUB DeviceRadixSort wrapper for Rust FFI
//
// CUB is a header-only C++ template library. This file instantiates the
// templates for the specific types we need and provides a C interface.

#include <cub/cub.cuh>
#include <cuda_runtime.h>

extern "C" {

// Error codes matching cudaError_t
typedef int CudaError;

/// Query the temporary storage size needed for radix sort.
///
/// # Arguments
/// * `temp_storage_bytes` - Output: required temporary storage size
/// * `num_items` - Number of items to sort
/// * `begin_bit` - First bit to sort (0 for full sort)
/// * `end_bit` - Last bit to sort (64 for u64 keys)
///
/// # Returns
/// cudaSuccess (0) on success, error code otherwise.
CudaError cub_radix_sort_pairs_u64_u32_temp_size(
    size_t* temp_storage_bytes,
    int num_items,
    int begin_bit,
    int end_bit
) {
    return cub::DeviceRadixSort::SortPairs(
        nullptr,                    // d_temp_storage (nullptr to query size)
        *temp_storage_bytes,        // temp_storage_bytes (output)
        (const uint64_t*)nullptr,   // d_keys_in
        (uint64_t*)nullptr,         // d_keys_out
        (const uint32_t*)nullptr,   // d_values_in
        (uint32_t*)nullptr,         // d_values_out
        num_items,
        begin_bit,
        end_bit,
        0                           // stream (default)
    );
}

/// Sort key-value pairs using radix sort.
///
/// Sorts (u64, u32) pairs by key in ascending order.
///
/// # Arguments
/// * `d_temp_storage` - Device temporary storage
/// * `temp_storage_bytes` - Size of temporary storage
/// * `d_keys_in` - Input keys (device memory)
/// * `d_keys_out` - Output keys (device memory)
/// * `d_values_in` - Input values (device memory)
/// * `d_values_out` - Output values (device memory)
/// * `num_items` - Number of items to sort
/// * `begin_bit` - First bit to sort (0 for full sort)
/// * `end_bit` - Last bit to sort (64 for u64 keys)
/// * `stream` - CUDA stream (0 for default stream)
///
/// # Returns
/// cudaSuccess (0) on success, error code otherwise.
CudaError cub_radix_sort_pairs_u64_u32(
    void* d_temp_storage,
    size_t temp_storage_bytes,
    const uint64_t* d_keys_in,
    uint64_t* d_keys_out,
    const uint32_t* d_values_in,
    uint32_t* d_values_out,
    int num_items,
    int begin_bit,
    int end_bit,
    cudaStream_t stream
) {
    return cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items,
        begin_bit,
        end_bit,
        stream
    );
}

/// Sort key-value pairs in-place using double-buffer technique.
///
/// Uses DoubleBuffer to minimize memory allocation. After sorting,
/// the current buffer selector indicates which buffer contains the result.
///
/// # Arguments
/// * `d_temp_storage` - Device temporary storage
/// * `temp_storage_bytes` - Size of temporary storage
/// * `d_keys` - Keys buffer (device memory, modified in place)
/// * `d_keys_alt` - Alternate keys buffer (device memory)
/// * `d_values` - Values buffer (device memory, modified in place)
/// * `d_values_alt` - Alternate values buffer (device memory)
/// * `num_items` - Number of items to sort
/// * `begin_bit` - First bit to sort (0 for full sort)
/// * `end_bit` - Last bit to sort (64 for u64 keys)
/// * `stream` - CUDA stream (0 for default stream)
/// * `keys_selector` - Output: 0 if result in d_keys, 1 if in d_keys_alt
/// * `values_selector` - Output: 0 if result in d_values, 1 if in d_values_alt
///
/// # Returns
/// cudaSuccess (0) on success, error code otherwise.
CudaError cub_radix_sort_pairs_u64_u32_double_buffer(
    void* d_temp_storage,
    size_t temp_storage_bytes,
    uint64_t* d_keys,
    uint64_t* d_keys_alt,
    uint32_t* d_values,
    uint32_t* d_values_alt,
    int num_items,
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    int* keys_selector,
    int* values_selector
) {
    cub::DoubleBuffer<uint64_t> d_keys_buf(d_keys, d_keys_alt);
    cub::DoubleBuffer<uint32_t> d_values_buf(d_values, d_values_alt);

    cudaError_t err = cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_buf,
        d_values_buf,
        num_items,
        begin_bit,
        end_bit,
        stream
    );

    if (err == cudaSuccess) {
        *keys_selector = d_keys_buf.selector;
        *values_selector = d_values_buf.selector;
    }

    return err;
}

} // extern "C"
