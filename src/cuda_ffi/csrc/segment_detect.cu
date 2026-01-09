// CUB-based segment detection for voxel boundary identification
//
// Detects segment boundaries in sorted Morton codes and computes:
// 1. Boundaries array (1 where code changes, 0 otherwise)
// 2. Segment IDs via exclusive prefix sum
// 3. Segment starts via stream compaction

#include <cub/cub.cuh>
#include <cuda_runtime.h>

extern "C" {

typedef int CudaError;

// =============================================================================
// Boundary Detection Kernel
// =============================================================================

/// Marks boundaries where Morton codes change.
/// boundary[i] = 1 if codes[i] != codes[i-1], else 0
/// boundary[0] = 0 (first element starts segment 0, no increment needed)
///
/// For segment_ids via exclusive prefix sum:
///   codes = [1, 1, 1, 5, 5, 9, 9, 9, 9]
///   boundaries = [0, 0, 0, 1, 0, 1, 0, 0, 0]
///   exclusive_sum = [0, 0, 0, 0, 1, 1, 2, 2, 2]  <- segment_ids
__global__ void detect_boundaries_kernel(
    const uint64_t* __restrict__ sorted_codes,
    uint32_t* __restrict__ boundaries,
    int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    if (idx == 0) {
        boundaries[idx] = 0;  // First element is segment 0
    } else {
        boundaries[idx] = (sorted_codes[idx] != sorted_codes[idx - 1]) ? 1 : 0;
    }
}

/// Launch boundary detection kernel.
CudaError cub_detect_boundaries(
    const uint64_t* d_sorted_codes,
    uint32_t* d_boundaries,
    int num_items,
    cudaStream_t stream
) {
    if (num_items == 0) return cudaSuccess;

    int block_size = 256;
    int num_blocks = (num_items + block_size - 1) / block_size;

    detect_boundaries_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_sorted_codes,
        d_boundaries,
        num_items
    );

    return cudaGetLastError();
}

// =============================================================================
// Inclusive Prefix Sum (for segment IDs)
// =============================================================================

/// Query temporary storage size for inclusive sum.
CudaError cub_inclusive_sum_u32_temp_size(
    size_t* temp_storage_bytes,
    int num_items
) {
    return cub::DeviceScan::InclusiveSum(
        nullptr,
        *temp_storage_bytes,
        (const uint32_t*)nullptr,
        (uint32_t*)nullptr,
        num_items,
        0  // stream
    );
}

/// Perform inclusive prefix sum.
/// output[i] = input[0] + input[1] + ... + input[i]
CudaError cub_inclusive_sum_u32(
    void* d_temp_storage,
    size_t temp_storage_bytes,
    const uint32_t* d_in,
    uint32_t* d_out,
    int num_items,
    cudaStream_t stream
) {
    return cub::DeviceScan::InclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        stream
    );
}

// =============================================================================
// Stream Compaction (for segment starts)
// =============================================================================

/// Query temporary storage size for flagged selection.
CudaError cub_select_flagged_u32_temp_size(
    size_t* temp_storage_bytes,
    int num_items
) {
    int* d_num_selected = nullptr;
    return cub::DeviceSelect::Flagged(
        nullptr,
        *temp_storage_bytes,
        (const uint32_t*)nullptr,  // d_in (indices)
        (const uint32_t*)nullptr,  // d_flags
        (uint32_t*)nullptr,        // d_out
        d_num_selected,
        num_items,
        0  // stream
    );
}

/// Select elements where flag is non-zero.
/// Used to extract indices where boundaries[i] == 1.
CudaError cub_select_flagged_u32(
    void* d_temp_storage,
    size_t temp_storage_bytes,
    const uint32_t* d_in,
    const uint32_t* d_flags,
    uint32_t* d_out,
    int* d_num_selected,  // Device memory for output count
    int num_items,
    cudaStream_t stream
) {
    return cub::DeviceSelect::Flagged(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_flags,
        d_out,
        d_num_selected,
        num_items,
        stream
    );
}

// =============================================================================
// Generate Index Sequence
// =============================================================================

/// Generate sequence 0, 1, 2, ..., n-1
__global__ void iota_kernel(
    uint32_t* __restrict__ output,
    int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_items) {
        output[idx] = idx;
    }
}

/// Launch iota kernel to generate index sequence.
CudaError cub_iota_u32(
    uint32_t* d_output,
    int num_items,
    cudaStream_t stream
) {
    if (num_items == 0) return cudaSuccess;

    int block_size = 256;
    int num_blocks = (num_items + block_size - 1) / block_size;

    iota_kernel<<<num_blocks, block_size, 0, stream>>>(d_output, num_items);

    return cudaGetLastError();
}

// =============================================================================
// Gather (for segment codes)
// =============================================================================

/// Gather codes at specified indices.
/// output[i] = codes[indices[i]]
__global__ void gather_u64_kernel(
    const uint64_t* __restrict__ codes,
    const uint32_t* __restrict__ indices,
    uint64_t* __restrict__ output,
    int num_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        output[idx] = codes[indices[idx]];
    }
}

/// Launch gather kernel for u64.
CudaError cub_gather_u64(
    const uint64_t* d_codes,
    const uint32_t* d_indices,
    uint64_t* d_output,
    int num_indices,
    cudaStream_t stream
) {
    if (num_indices == 0) return cudaSuccess;

    int block_size = 256;
    int num_blocks = (num_indices + block_size - 1) / block_size;

    gather_u64_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_codes, d_indices, d_output, num_indices
    );

    return cudaGetLastError();
}

} // extern "C"
