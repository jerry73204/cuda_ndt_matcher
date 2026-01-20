// Warp-level reduction primitives using shuffle instructions
//
// These functions provide efficient reduction within a warp (32 threads)
// using __shfl_down_sync, avoiding shared memory and __syncthreads().
//
// Benefits over block-level reduction:
// - No shared memory allocation needed
// - No __syncthreads() synchronization
// - 32x fewer atomic operations (one per warp vs one per thread)
// - Better instruction-level parallelism

#pragma once

#include <cuda_runtime.h>

// Full warp mask for shuffle operations
constexpr unsigned FULL_WARP_MASK = 0xffffffff;

// Warp size constant
constexpr int WARP_SIZE = 32;

// ============================================================================
// Single-value reductions
// ============================================================================

/// Reduce a single float value across the warp using sum.
/// Result is valid only in lane 0.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

/// Reduce a single double value across the warp using sum.
/// Result is valid only in lane 0.
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

/// Reduce a single float value across the warp using max.
/// Result is valid only in lane 0.
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

/// Reduce a single float value across the warp using min.
/// Result is valid only in lane 0.
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

// ============================================================================
// Multi-value reductions (for NDT gradient/Hessian)
// ============================================================================

/// Reduce 6 float values (gradient) across the warp.
/// Results are valid only in lane 0.
__device__ __forceinline__ void warp_reduce_sum_6(float* vals) {
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        vals[i] = warp_reduce_sum(vals[i]);
    }
}

/// Reduce 21 float values (upper triangle of 6x6 Hessian) across the warp.
/// Results are valid only in lane 0.
__device__ __forceinline__ void warp_reduce_sum_21(float* vals) {
    #pragma unroll
    for (int i = 0; i < 21; i++) {
        vals[i] = warp_reduce_sum(vals[i]);
    }
}

/// Reduce the full NDT reduction buffer (29 values: 1 score + 6 grad + 21 hess + 1 corr).
/// Results are valid only in lane 0.
__device__ __forceinline__ void warp_reduce_ndt_values(
    float* score,
    float* grad,      // [6]
    float* hess,      // [21]
    float* corr
) {
    *score = warp_reduce_sum(*score);
    warp_reduce_sum_6(grad);
    warp_reduce_sum_21(hess);
    *corr = warp_reduce_sum(*corr);
}

// ============================================================================
// Block-level reduction using warp primitives
// ============================================================================

/// Perform block-wide reduction using warp shuffles + shared memory for cross-warp.
/// This is more efficient than full shared memory tree reduction.
///
/// @param val The thread's value to reduce
/// @param warp_results Shared memory array [num_warps] for cross-warp reduction
/// @return The reduced sum (valid in thread 0 only)
__device__ __forceinline__ float block_reduce_sum_warp(
    float val,
    volatile float* warp_results
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // First: reduce within warp
    val = warp_reduce_sum(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

/// Block-wide reduction for NDT values (29 floats) using warp primitives.
/// More efficient than full shared memory tree reduction.
///
/// @param score Thread's score contribution
/// @param grad Thread's gradient contribution [6]
/// @param hess Thread's Hessian contribution [21]
/// @param corr Thread's correspondence count
/// @param warp_scratch Shared memory [num_warps * 29] for cross-warp reduction
/// @param out_score Output score (valid in thread 0)
/// @param out_grad Output gradient [6] (valid in thread 0)
/// @param out_hess Output Hessian [21] (valid in thread 0)
/// @param out_corr Output correspondence count (valid in thread 0)
__device__ __forceinline__ void block_reduce_ndt_warp(
    float score,
    const float* grad,
    const float* hess,
    float corr,
    volatile float* warp_scratch,
    float* out_score,
    float* out_grad,
    float* out_hess,
    float* out_corr
) {
    constexpr int NDT_REDUCE_SIZE = 29;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Pack values for reduction
    float my_vals[NDT_REDUCE_SIZE];
    my_vals[0] = score;
    #pragma unroll
    for (int i = 0; i < 6; i++) my_vals[1 + i] = grad[i];
    #pragma unroll
    for (int i = 0; i < 21; i++) my_vals[7 + i] = hess[i];
    my_vals[28] = corr;

    // Reduce within warp
    #pragma unroll
    for (int i = 0; i < NDT_REDUCE_SIZE; i++) {
        my_vals[i] = warp_reduce_sum(my_vals[i]);
    }

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < NDT_REDUCE_SIZE; i++) {
            warp_scratch[warp_id * NDT_REDUCE_SIZE + i] = my_vals[i];
        }
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < NDT_REDUCE_SIZE; i++) {
            my_vals[i] = (lane < num_warps) ? warp_scratch[lane * NDT_REDUCE_SIZE + i] : 0.0f;
            my_vals[i] = warp_reduce_sum(my_vals[i]);
        }

        // Thread 0 has the final results
        if (lane == 0) {
            *out_score = my_vals[0];
            #pragma unroll
            for (int i = 0; i < 6; i++) out_grad[i] = my_vals[1 + i];
            #pragma unroll
            for (int i = 0; i < 21; i++) out_hess[i] = my_vals[7 + i];
            *out_corr = my_vals[28];
        }
    }
}

// ============================================================================
// Atomic accumulation helpers
// ============================================================================

/// Atomically add NDT reduction values to global memory.
/// Should be called by lane 0 of each warp after warp reduction.
__device__ __forceinline__ void atomic_add_ndt_values(
    float* global_buffer,  // [29] in global memory
    float score,
    const float* grad,     // [6]
    const float* hess,     // [21]
    float corr
) {
    atomicAdd(&global_buffer[0], score);
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        atomicAdd(&global_buffer[1 + i], grad[i]);
    }
    #pragma unroll
    for (int i = 0; i < 21; i++) {
        atomicAdd(&global_buffer[7 + i], hess[i]);
    }
    atomicAdd(&global_buffer[28], corr);
}
