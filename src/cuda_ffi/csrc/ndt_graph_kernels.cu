// CUDA Graph-based NDT Kernels
//
// This file implements the NDT optimization as separate kernels that can be
// captured into a CUDA Graph, replacing the cooperative groups persistent kernel.
// This approach works on GPUs with limited SM count (e.g., Jetson Orin) where
// cooperative launch fails due to grid size limits.
//
// Kernels:
//   K1: ndt_graph_init_kernel       - Initialize optimization state
//   K2: ndt_graph_compute_kernel    - Per-point score/gradient/Hessian + reduction
//   K3: ndt_graph_solve_kernel      - Newton solve + regularization
//   K4: ndt_graph_linesearch_kernel - Parallel line search evaluation
//   K5: ndt_graph_update_kernel     - Apply step, check convergence
//
// Unlike the cooperative kernel, these use standard launches with no grid sync.
// Synchronization between kernels is implicit in CUDA Graphs.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#include "ndt_graph_common.cuh"
#include "persistent_ndt_device.cuh"
#include "cholesky_6x6.cuh"
#include "jacobi_svd_6x6.cuh"

// 27 neighbor offsets (3x3x3 cube)
__constant__ int8_t GRAPH_NEIGHBOR_OFFSETS[27][3] = {
    {-1, -1, -1}, {-1, -1,  0}, {-1, -1,  1},
    {-1,  0, -1}, {-1,  0,  0}, {-1,  0,  1},
    {-1,  1, -1}, {-1,  1,  0}, {-1,  1,  1},
    { 0, -1, -1}, { 0, -1,  0}, { 0, -1,  1},
    { 0,  0, -1}, { 0,  0,  0}, { 0,  0,  1},
    { 0,  1, -1}, { 0,  1,  0}, { 0,  1,  1},
    { 1, -1, -1}, { 1, -1,  0}, { 1, -1,  1},
    { 1,  0, -1}, { 1,  0,  0}, { 1,  0,  1},
    { 1,  1, -1}, { 1,  1,  0}, { 1,  1,  1}
};

extern "C" {

// ============================================================================
// Hash query (for graph kernels)
// ============================================================================

__device__ __forceinline__ int graph_hash_query_inline(
    float qx, float qy, float qz,
    const GraphHashEntry* hash_table,
    uint32_t capacity,
    float inv_resolution,
    float radius_sq,
    const float* voxel_means,
    int32_t* neighbor_indices  // [MAX_NEIGHBORS]
) {
    int32_t gx = graph_pos_to_grid(qx, inv_resolution);
    int32_t gy = graph_pos_to_grid(qy, inv_resolution);
    int32_t gz = graph_pos_to_grid(qz, inv_resolution);

    int count = 0;

    // Check all 27 neighboring cells
    for (uint32_t n = 0; n < 27 && count < GRAPH_MAX_NEIGHBORS; n++) {
        int32_t nx = gx + GRAPH_NEIGHBOR_OFFSETS[n][0];
        int32_t ny = gy + GRAPH_NEIGHBOR_OFFSETS[n][1];
        int32_t nz = gz + GRAPH_NEIGHBOR_OFFSETS[n][2];

        int64_t key = graph_pack_key(nx, ny, nz);
        uint32_t slot = graph_hash_key(key, capacity);

        // Linear probing search
        for (uint32_t i = 0; i < capacity && count < GRAPH_MAX_NEIGHBORS; i++) {
            uint32_t probe_slot = (slot + i) % capacity;
            int64_t stored_key = hash_table[probe_slot].key;

            if (stored_key == GRAPH_EMPTY_SLOT) break;

            if (stored_key == key) {
                int32_t voxel_idx = hash_table[probe_slot].value;

                float vx = voxel_means[voxel_idx * 3 + 0];
                float vy = voxel_means[voxel_idx * 3 + 1];
                float vz = voxel_means[voxel_idx * 3 + 2];

                float dx = qx - vx;
                float dy = qy - vy;
                float dz = qz - vz;
                float dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq <= radius_sq) {
                    neighbor_indices[count++] = voxel_idx;
                }
                break;
            }
        }
    }

    // Fill remaining with -1
    for (int i = count; i < GRAPH_MAX_NEIGHBORS; i++) {
        neighbor_indices[i] = -1;
    }

    return count;
}

// ============================================================================
// K1: Initialization Kernel
// ============================================================================
//
// Purpose: Initialize persistent state from initial pose
// Grid: 1 block × 1 thread

__global__ void ndt_graph_init_kernel(
    const float* __restrict__ initial_pose,  // [6]
    float* __restrict__ state_buffer,        // [StateOffset::TOTAL_SIZE]
    float* __restrict__ reduce_buffer,       // [ReduceOffset::TOTAL_SIZE]
    float* __restrict__ ls_buffer            // [LineSearchOffset::TOTAL_SIZE]
) {
    // Copy initial pose to current pose
    for (int i = 0; i < 6; i++) {
        state_buffer[StateOffset::POSE + i] = initial_pose[i];
    }

    // Initialize position history (for oscillation detection)
    state_buffer[StateOffset::PREV_POS + 0] = initial_pose[0];
    state_buffer[StateOffset::PREV_POS + 1] = initial_pose[1];
    state_buffer[StateOffset::PREV_POS + 2] = initial_pose[2];
    state_buffer[StateOffset::PREV_PREV_POS + 0] = initial_pose[0];
    state_buffer[StateOffset::PREV_PREV_POS + 1] = initial_pose[1];
    state_buffer[StateOffset::PREV_PREV_POS + 2] = initial_pose[2];

    // Clear counters and flags
    state_buffer[StateOffset::CONVERGED] = 0.0f;
    state_buffer[StateOffset::ITERATIONS] = 0.0f;
    state_buffer[StateOffset::OSC_COUNT] = 0.0f;
    state_buffer[StateOffset::MAX_OSC_COUNT] = 0.0f;
    state_buffer[StateOffset::ALPHA_SUM] = 0.0f;
    state_buffer[StateOffset::ACTUAL_STEP_LEN] = 0.0f;

    // Clear delta
    for (int i = 0; i < 6; i++) {
        state_buffer[StateOffset::DELTA + i] = 0.0f;
    }

    // Clear reduction buffer
    for (int i = 0; i < ReduceOffset::TOTAL_SIZE; i++) {
        reduce_buffer[i] = 0.0f;
    }

    // Clear line search buffer
    for (int i = 0; i < LineSearchOffset::TOTAL_SIZE; i++) {
        ls_buffer[i] = 0.0f;
    }
}

// ============================================================================
// K2: Compute Kernel
// ============================================================================
//
// Purpose: Compute per-point NDT score, gradient, Hessian contributions
// Grid: ceil(num_points / 256) blocks × 256 threads

__global__ void ndt_graph_compute_kernel(
    // Read-only inputs
    const float* __restrict__ source_points,      // [N * 3]
    const float* __restrict__ voxel_means,        // [V * 3]
    const float* __restrict__ voxel_inv_covs,     // [V * 9]
    const GraphHashEntry* __restrict__ hash_table,
    // Configuration
    const GraphNdtConfig* __restrict__ config,
    // State (read)
    const float* __restrict__ state_buffer,       // Read current pose
    // Reduction output (atomic add)
    float* __restrict__ reduce_buffer             // [ReduceOffset::TOTAL_SIZE]
) {
    // Shared memory for block-level reduction
    extern __shared__ float smem[];
    float* partial_sums = smem;  // [REDUCE_SIZE * blockDim.x]

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_points = config->num_points;

    // Pre-computed constants
    float inv_resolution = 1.0f / config->resolution;
    float radius_sq = config->resolution * config->resolution;
    float gauss_d1 = config->gauss_d1;
    float gauss_d2 = config->gauss_d2;

    // Read current pose
    float pose[6];
    for (int i = 0; i < 6; i++) {
        pose[i] = state_buffer[StateOffset::POSE + i];
    }

    // Compute sin/cos from pose
    float sr, cr, sp, cp, sy, cy;
    compute_sincos_inline(pose, &sr, &cr, &sp, &cp, &sy, &cy);

    // Compute transform matrix
    float T[16];
    compute_transform_inline(pose, sr, cr, sp, cp, sy, cy, T);

    // Local accumulators
    float my_score = 0.0f;
    float my_grad[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float my_hess[21] = {0};
    float my_correspondences = 0.0f;

    if (tid < num_points) {
        // Load source point
        float px = source_points[tid * 3 + 0];
        float py = source_points[tid * 3 + 1];
        float pz = source_points[tid * 3 + 2];

        // Transform point
        float tx, ty, tz;
        transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

        // Hash lookup for neighbors
        int32_t neighbor_indices[GRAPH_MAX_NEIGHBORS];
        int num_neighbors = graph_hash_query_inline(
            tx, ty, tz,
            hash_table, config->hash_capacity, inv_resolution, radius_sq,
            voxel_means, neighbor_indices
        );

        // Compute Jacobian
        float J[18];
        compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

        // Compute point Hessians (sparse)
        float pH[15];
        compute_point_hessians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, pH);

        // Accumulate contributions from all neighbors
        for (int n = 0; n < num_neighbors; n++) {
            int32_t vidx = neighbor_indices[n];
            if (vidx < 0) continue;

            float voxel_mean[3] = {
                voxel_means[vidx * 3 + 0],
                voxel_means[vidx * 3 + 1],
                voxel_means[vidx * 3 + 2]
            };
            float voxel_inv_cov[9];
            for (int i = 0; i < 9; i++) {
                voxel_inv_cov[i] = voxel_inv_covs[vidx * 9 + i];
            }

            float score_contrib;
            float grad_contrib[6];
            float hess_contrib[21];

            compute_ndt_contribution(
                tx, ty, tz,
                voxel_mean, voxel_inv_cov,
                J, pH,
                gauss_d1, gauss_d2,
                &score_contrib, grad_contrib, hess_contrib
            );

            my_score += score_contrib;
            for (int i = 0; i < 6; i++) {
                my_grad[i] += grad_contrib[i];
            }
            for (int i = 0; i < 21; i++) {
                my_hess[i] += hess_contrib[i];
            }
            my_correspondences += 1.0f;
        }
    }

    // Block-level reduction
    // Store to shared memory (29 values: score + grad[6] + hess[21] + correspondences)
    partial_sums[threadIdx.x * GRAPH_REDUCE_SIZE + 0] = my_score;
    for (int i = 0; i < 6; i++) {
        partial_sums[threadIdx.x * GRAPH_REDUCE_SIZE + 1 + i] = my_grad[i];
    }
    for (int i = 0; i < 21; i++) {
        partial_sums[threadIdx.x * GRAPH_REDUCE_SIZE + 7 + i] = my_hess[i];
    }
    partial_sums[threadIdx.x * GRAPH_REDUCE_SIZE + 28] = my_correspondences;
    __syncthreads();

    // Tree reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int i = 0; i < GRAPH_REDUCE_SIZE; i++) {
                partial_sums[threadIdx.x * GRAPH_REDUCE_SIZE + i] +=
                    partial_sums[(threadIdx.x + stride) * GRAPH_REDUCE_SIZE + i];
            }
        }
        __syncthreads();
    }

    // Thread 0 of each block atomically adds to global reduce buffer
    if (threadIdx.x == 0) {
        for (int i = 0; i < GRAPH_REDUCE_SIZE; i++) {
            atomicAdd(&reduce_buffer[i], partial_sums[i]);
        }
    }
}

// ============================================================================
// K3: Solve Kernel
// ============================================================================
//
// Purpose: Solve Newton system and compute step direction
// Grid: 1 block × 32 threads (or 1 thread for simplicity)

__global__ void ndt_graph_solve_kernel(
    const GraphNdtConfig* __restrict__ config,
    float* __restrict__ state_buffer,
    float* __restrict__ reduce_buffer,
    float* __restrict__ ls_buffer,
    float* __restrict__ output_buffer    // [OutputOffset::TOTAL_SIZE] for Hessian output
) {
    if (threadIdx.x != 0) return;

    // Load reduced values
    float score = reduce_buffer[ReduceOffset::SCORE];
    float correspondence_count = reduce_buffer[ReduceOffset::CORRESPONDENCES];

    // Apply GNSS regularization if enabled
    if (config->reg_enabled) {
        float pose_x = state_buffer[StateOffset::POSE + 0];
        float pose_y = state_buffer[StateOffset::POSE + 1];
        float yaw = state_buffer[StateOffset::POSE + 5];

        float dx = config->reg_ref_x - pose_x;
        float dy = config->reg_ref_y - pose_y;
        float sin_yaw = sinf(yaw);
        float cos_yaw = cosf(yaw);

        // Longitudinal distance in vehicle frame
        float longitudinal = dy * sin_yaw + dx * cos_yaw;

        // Score adjustment
        float weight = correspondence_count;
        score += -config->reg_scale * weight * longitudinal * longitudinal;

        // Gradient adjustments
        float grad_x_delta = config->reg_scale * weight * 2.0f * cos_yaw * longitudinal;
        float grad_y_delta = config->reg_scale * weight * 2.0f * sin_yaw * longitudinal;
        reduce_buffer[ReduceOffset::GRADIENT + 0] += grad_x_delta;
        reduce_buffer[ReduceOffset::GRADIENT + 1] += grad_y_delta;

        // Hessian adjustments (upper triangle indices)
        float h00_delta = -config->reg_scale * weight * 2.0f * cos_yaw * cos_yaw;
        float h01_delta = -config->reg_scale * weight * 2.0f * cos_yaw * sin_yaw;
        float h11_delta = -config->reg_scale * weight * 2.0f * sin_yaw * sin_yaw;
        reduce_buffer[ReduceOffset::HESSIAN_UT + 0] += h00_delta;  // H[0,0]
        reduce_buffer[ReduceOffset::HESSIAN_UT + 1] += h01_delta;  // H[0,1]
        reduce_buffer[ReduceOffset::HESSIAN_UT + 6] += h11_delta;  // H[1,1]
    }

    // Convert to f64 for solve
    double g[6];
    double H[36];

    for (int i = 0; i < 6; i++) {
        g[i] = (double)reduce_buffer[ReduceOffset::GRADIENT + i];
    }

    // Expand upper triangle to full matrix
    expand_upper_triangle_f32_to_f64(reduce_buffer + ReduceOffset::HESSIAN_UT, H);

    // Save Hessian to output buffer (for covariance estimation)
    expand_upper_triangle_f32(reduce_buffer + ReduceOffset::HESSIAN_UT,
                              output_buffer + OutputOffset::HESSIAN);

    // Jacobi SVD solve: H * delta = -g
    double delta[6];
    bool solve_success;
    jacobi_svd_solve_6x6_f64(H, g, delta, &solve_success);

    if (solve_success) {
        for (int i = 0; i < 6; i++) {
            state_buffer[StateOffset::DELTA + i] = (float)delta[i];
        }
    } else {
        // Fallback: steepest descent with small step
        for (int i = 0; i < 6; i++) {
            state_buffer[StateOffset::DELTA + i] = (float)(-g[i] * 1e-4);
        }
    }

    // Prepare for line search or direct update
    if (config->ls_enabled) {
        // Direction check for maximization
        float dphi_0_val = 0.0f;
        for (int i = 0; i < 6; i++) {
            dphi_0_val += reduce_buffer[ReduceOffset::GRADIENT + i] *
                         state_buffer[StateOffset::DELTA + i];
        }
        if (dphi_0_val <= 0.0f) {
            // Wrong direction - reverse
            for (int i = 0; i < 6; i++) {
                state_buffer[StateOffset::DELTA + i] = -state_buffer[StateOffset::DELTA + i];
            }
            dphi_0_val = -dphi_0_val;
        }

        // Save state for line search
        for (int i = 0; i < 6; i++) {
            state_buffer[StateOffset::ORIGINAL_POSE + i] = state_buffer[StateOffset::POSE + i];
        }
        ls_buffer[LineSearchOffset::PHI_0] = score;
        ls_buffer[LineSearchOffset::DPHI_0] = dphi_0_val;

        // Generate candidates (golden ratio decay from 1.0)
        float step = 1.0f;
        int num_cands = (config->ls_num_candidates < GRAPH_MAX_LS_CANDIDATES)
                        ? config->ls_num_candidates : GRAPH_MAX_LS_CANDIDATES;
        for (int k = 0; k < num_cands; k++) {
            state_buffer[StateOffset::ALPHA_CANDIDATES + k] = step;
            step *= 0.618f;
        }
        ls_buffer[LineSearchOffset::EARLY_TERM] = 0.0f;
        ls_buffer[LineSearchOffset::BEST_ALPHA] = 1.0f;

        // Clear line search reduction slots
        for (int k = 0; k < GRAPH_MAX_LS_CANDIDATES; k++) {
            ls_buffer[LineSearchOffset::CAND_SCORES + k] = 0.0f;
            ls_buffer[LineSearchOffset::CAND_CORR + k] = 0.0f;
            for (int i = 0; i < 6; i++) {
                ls_buffer[LineSearchOffset::CAND_GRADS + k * 6 + i] = 0.0f;
            }
        }
    } else {
        // No line search - apply clamped step directly
        // Direction check
        float dphi_0 = 0.0f;
        for (int i = 0; i < 6; i++) {
            dphi_0 += reduce_buffer[ReduceOffset::GRADIENT + i] *
                     state_buffer[StateOffset::DELTA + i];
        }
        if (dphi_0 <= 0.0f) {
            for (int i = 0; i < 6; i++) {
                state_buffer[StateOffset::DELTA + i] = -state_buffer[StateOffset::DELTA + i];
            }
        }

        // Compute delta norm and clamp
        float delta_norm_sq = 0.0f;
        for (int i = 0; i < 6; i++) {
            float d = state_buffer[StateOffset::DELTA + i];
            delta_norm_sq += d * d;
        }
        float delta_norm = sqrtf(delta_norm_sq);

        float step_length = delta_norm;
        if (step_length > config->fixed_step_size) step_length = config->fixed_step_size;

        // Scale delta
        float scale = (delta_norm > 1e-10f) ? (step_length / delta_norm) : 0.0f;
        for (int i = 0; i < 6; i++) {
            state_buffer[StateOffset::POSE + i] += scale * state_buffer[StateOffset::DELTA + i];
        }

        state_buffer[StateOffset::ACTUAL_STEP_LEN] = step_length;
    }
}

// ============================================================================
// K4: Line Search Kernel
// ============================================================================
//
// Purpose: Evaluate multiple step size candidates in parallel
// Grid: ceil(num_points / 256) blocks × 256 threads
// Only runs if line search is enabled

__global__ void ndt_graph_linesearch_kernel(
    // Read-only inputs
    const float* __restrict__ source_points,
    const float* __restrict__ voxel_means,
    const float* __restrict__ voxel_inv_covs,
    const GraphHashEntry* __restrict__ hash_table,
    // Configuration
    const GraphNdtConfig* __restrict__ config,
    // State (read)
    const float* __restrict__ state_buffer,
    // Line search buffer (read/write)
    float* __restrict__ ls_buffer
) {
    // Shared memory for block-level reduction
    extern __shared__ float smem[];
    float* partial_sums = smem;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_points = config->num_points;

    float inv_resolution = 1.0f / config->resolution;
    float radius_sq = config->resolution * config->resolution;
    float gauss_d1 = config->gauss_d1;
    float gauss_d2 = config->gauss_d2;

    int num_cands = (config->ls_num_candidates < GRAPH_MAX_LS_CANDIDATES)
                    ? config->ls_num_candidates : GRAPH_MAX_LS_CANDIDATES;

    // Per-candidate accumulators
    constexpr int LS_VALUES_PER_CAND = 8;  // score + gradient[6] + corr
    float my_cand_data[GRAPH_MAX_LS_CANDIDATES * LS_VALUES_PER_CAND] = {0};

    if (tid < num_points) {
        float px = source_points[tid * 3 + 0];
        float py = source_points[tid * 3 + 1];
        float pz = source_points[tid * 3 + 2];

        // Evaluate each candidate
        for (int c = 0; c < num_cands; c++) {
            float alpha = state_buffer[StateOffset::ALPHA_CANDIDATES + c];

            // Compute trial pose
            float trial_pose[6];
            for (int i = 0; i < 6; i++) {
                trial_pose[i] = state_buffer[StateOffset::ORIGINAL_POSE + i] +
                               alpha * state_buffer[StateOffset::DELTA + i];
            }

            float sr, cr, sp, cp, sy, cy;
            compute_sincos_inline(trial_pose, &sr, &cr, &sp, &cp, &sy, &cy);

            float T[16];
            compute_transform_inline(trial_pose, sr, cr, sp, cp, sy, cy, T);

            float tx, ty, tz;
            transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

            int32_t neighbor_indices[GRAPH_MAX_NEIGHBORS];
            int num_neighbors = graph_hash_query_inline(
                tx, ty, tz,
                hash_table, config->hash_capacity, inv_resolution, radius_sq,
                voxel_means, neighbor_indices
            );

            float J[18];
            compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

            // Accumulate score and gradient for this candidate
            for (int n = 0; n < num_neighbors; n++) {
                int32_t vidx = neighbor_indices[n];
                if (vidx < 0) continue;

                float voxel_mean[3] = {
                    voxel_means[vidx * 3 + 0],
                    voxel_means[vidx * 3 + 1],
                    voxel_means[vidx * 3 + 2]
                };
                float voxel_inv_cov[9];
                for (int i = 0; i < 9; i++) {
                    voxel_inv_cov[i] = voxel_inv_covs[vidx * 9 + i];
                }

                float dx = tx - voxel_mean[0];
                float dy = ty - voxel_mean[1];
                float dz = tz - voxel_mean[2];

                float Sd_0 = voxel_inv_cov[0] * dx + voxel_inv_cov[1] * dy + voxel_inv_cov[2] * dz;
                float Sd_1 = voxel_inv_cov[3] * dx + voxel_inv_cov[4] * dy + voxel_inv_cov[5] * dz;
                float Sd_2 = voxel_inv_cov[6] * dx + voxel_inv_cov[7] * dy + voxel_inv_cov[8] * dz;

                float dSd = dx * Sd_0 + dy * Sd_1 + dz * Sd_2;
                float exp_val = expf(-gauss_d2 * 0.5f * dSd);

                // Score
                my_cand_data[c * LS_VALUES_PER_CAND + 0] += -gauss_d1 * exp_val;

                // Gradient
                float scale = gauss_d1 * gauss_d2 * exp_val;
                for (int j = 0; j < 6; j++) {
                    float Jt_Sd = J[j * 3 + 0] * Sd_0 + J[j * 3 + 1] * Sd_1 + J[j * 3 + 2] * Sd_2;
                    my_cand_data[c * LS_VALUES_PER_CAND + 1 + j] += scale * Jt_Sd;
                }

                // Correspondence count
                my_cand_data[c * LS_VALUES_PER_CAND + 7] += 1.0f;
            }
        }
    }

    // Block reduction for each candidate
    for (int c = 0; c < num_cands; c++) {
        for (int i = 0; i < LS_VALUES_PER_CAND; i++) {
            partial_sums[threadIdx.x * LS_VALUES_PER_CAND + i] = my_cand_data[c * LS_VALUES_PER_CAND + i];
        }
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                for (int i = 0; i < LS_VALUES_PER_CAND; i++) {
                    partial_sums[threadIdx.x * LS_VALUES_PER_CAND + i] +=
                        partial_sums[(threadIdx.x + stride) * LS_VALUES_PER_CAND + i];
                }
            }
            __syncthreads();
        }

        // Thread 0 atomically adds to global buffers
        if (threadIdx.x == 0) {
            atomicAdd(&ls_buffer[LineSearchOffset::CAND_SCORES + c], partial_sums[0]);
            for (int i = 0; i < 6; i++) {
                atomicAdd(&ls_buffer[LineSearchOffset::CAND_GRADS + c * 6 + i], partial_sums[1 + i]);
            }
            atomicAdd(&ls_buffer[LineSearchOffset::CAND_CORR + c], partial_sums[7]);
        }
        __syncthreads();
    }
}

// ============================================================================
// K5: Update Kernel
// ============================================================================
//
// Purpose: Apply step, check convergence, prepare for next iteration
// Grid: 1 block × 1 thread

__global__ void ndt_graph_update_kernel(
    const GraphNdtConfig* __restrict__ config,
    float* __restrict__ state_buffer,
    float* __restrict__ reduce_buffer,
    float* __restrict__ ls_buffer,
    float* __restrict__ output_buffer,
    float* __restrict__ debug_buffer    // [max_iterations * DEBUG_PER_ITER_SIZE] or nullptr
) {
    if (threadIdx.x != 0) return;

    int iter = (int)state_buffer[StateOffset::ITERATIONS];

    // Process line search results if enabled
    if (config->ls_enabled) {
        int num_cands = (config->ls_num_candidates < GRAPH_MAX_LS_CANDIDATES)
                        ? config->ls_num_candidates : GRAPH_MAX_LS_CANDIDATES;

        float phi_0 = ls_buffer[LineSearchOffset::PHI_0];
        float dphi_0 = ls_buffer[LineSearchOffset::DPHI_0];

        float best_alpha = 0.0f;
        float best_score = phi_0;
        bool found_wolfe = false;

        // Check Wolfe conditions for each candidate
        for (int k = 0; k < num_cands && !found_wolfe; k++) {
            float phi_k = ls_buffer[LineSearchOffset::CAND_SCORES + k];
            float alpha = state_buffer[StateOffset::ALPHA_CANDIDATES + k];
            float corr_k = ls_buffer[LineSearchOffset::CAND_CORR + k];

            // Apply regularization to candidate score if enabled
            if (config->reg_enabled) {
                float trial_pose_x = state_buffer[StateOffset::ORIGINAL_POSE + 0] +
                                    alpha * state_buffer[StateOffset::DELTA + 0];
                float trial_pose_y = state_buffer[StateOffset::ORIGINAL_POSE + 1] +
                                    alpha * state_buffer[StateOffset::DELTA + 1];
                float trial_yaw = state_buffer[StateOffset::ORIGINAL_POSE + 5] +
                                 alpha * state_buffer[StateOffset::DELTA + 5];

                float dx = config->reg_ref_x - trial_pose_x;
                float dy = config->reg_ref_y - trial_pose_y;
                float sin_yaw = sinf(trial_yaw);
                float cos_yaw = cosf(trial_yaw);
                float longitudinal = dy * sin_yaw + dx * cos_yaw;
                phi_k += -config->reg_scale * corr_k * longitudinal * longitudinal;
            }

            // Compute directional derivative
            float dphi_k = 0.0f;
            for (int i = 0; i < 6; i++) {
                dphi_k += ls_buffer[LineSearchOffset::CAND_GRADS + k * 6 + i] *
                         state_buffer[StateOffset::DELTA + i];
            }

            // Strong Wolfe conditions
            bool armijo = phi_k >= phi_0 + config->ls_mu * alpha * dphi_0;
            bool curvature = fabsf(dphi_k) <= config->ls_nu * fabsf(dphi_0);

            if (armijo && curvature) {
                best_alpha = alpha;
                found_wolfe = true;
            } else if (phi_k > best_score) {
                best_score = phi_k;
                best_alpha = alpha;
            }
        }

        // Apply final pose update with best_alpha
        for (int i = 0; i < 6; i++) {
            state_buffer[StateOffset::POSE + i] = state_buffer[StateOffset::ORIGINAL_POSE + i] +
                                                  best_alpha * state_buffer[StateOffset::DELTA + i];
        }

        // Compute actual step length
        float delta_norm_sq = 0.0f;
        for (int i = 0; i < 6; i++) {
            float d = state_buffer[StateOffset::DELTA + i];
            delta_norm_sq += d * d;
        }
        state_buffer[StateOffset::ACTUAL_STEP_LEN] = best_alpha * sqrtf(delta_norm_sq);
        ls_buffer[LineSearchOffset::BEST_ALPHA] = best_alpha;
    }

    // Oscillation detection
    if (iter >= 1) {
        float curr_x = state_buffer[StateOffset::POSE + 0];
        float curr_y = state_buffer[StateOffset::POSE + 1];
        float curr_z = state_buffer[StateOffset::POSE + 2];

        float curr_vec_x = curr_x - state_buffer[StateOffset::PREV_POS + 0];
        float curr_vec_y = curr_y - state_buffer[StateOffset::PREV_POS + 1];
        float curr_vec_z = curr_z - state_buffer[StateOffset::PREV_POS + 2];

        float prev_vec_x = state_buffer[StateOffset::PREV_POS + 0] - state_buffer[StateOffset::PREV_PREV_POS + 0];
        float prev_vec_y = state_buffer[StateOffset::PREV_POS + 1] - state_buffer[StateOffset::PREV_PREV_POS + 1];
        float prev_vec_z = state_buffer[StateOffset::PREV_POS + 2] - state_buffer[StateOffset::PREV_PREV_POS + 2];

        float curr_norm = sqrtf(curr_vec_x * curr_vec_x + curr_vec_y * curr_vec_y + curr_vec_z * curr_vec_z);
        float prev_norm = sqrtf(prev_vec_x * prev_vec_x + prev_vec_y * prev_vec_y + prev_vec_z * prev_vec_z);

        if (curr_norm > 1e-10f && prev_norm > 1e-10f) {
            float inv_curr = 1.0f / curr_norm;
            float inv_prev = 1.0f / prev_norm;
            curr_vec_x *= inv_curr; curr_vec_y *= inv_curr; curr_vec_z *= inv_curr;
            prev_vec_x *= inv_prev; prev_vec_y *= inv_prev; prev_vec_z *= inv_prev;

            float cosine = curr_vec_x * prev_vec_x + curr_vec_y * prev_vec_y + curr_vec_z * prev_vec_z;

            constexpr float INVERSION_THRESHOLD = -0.9f;
            if (cosine < INVERSION_THRESHOLD) {
                state_buffer[StateOffset::OSC_COUNT] += 1.0f;
            } else {
                state_buffer[StateOffset::OSC_COUNT] = 0.0f;
            }
            if (state_buffer[StateOffset::OSC_COUNT] > state_buffer[StateOffset::MAX_OSC_COUNT]) {
                state_buffer[StateOffset::MAX_OSC_COUNT] = state_buffer[StateOffset::OSC_COUNT];
            }
        } else {
            state_buffer[StateOffset::OSC_COUNT] = 0.0f;
        }
    }

    // Update position history
    state_buffer[StateOffset::PREV_PREV_POS + 0] = state_buffer[StateOffset::PREV_POS + 0];
    state_buffer[StateOffset::PREV_PREV_POS + 1] = state_buffer[StateOffset::PREV_POS + 1];
    state_buffer[StateOffset::PREV_PREV_POS + 2] = state_buffer[StateOffset::PREV_POS + 2];
    state_buffer[StateOffset::PREV_POS + 0] = state_buffer[StateOffset::POSE + 0];
    state_buffer[StateOffset::PREV_POS + 1] = state_buffer[StateOffset::POSE + 1];
    state_buffer[StateOffset::PREV_POS + 2] = state_buffer[StateOffset::POSE + 2];

    // Accumulate alpha
    float alpha_this_iter = config->ls_enabled ?
        ls_buffer[LineSearchOffset::BEST_ALPHA] : config->fixed_step_size;
    state_buffer[StateOffset::ALPHA_SUM] += alpha_this_iter;

    // Check convergence
    float step_len = state_buffer[StateOffset::ACTUAL_STEP_LEN];
    bool converged = step_len < sqrtf(config->epsilon_sq);
    state_buffer[StateOffset::CONVERGED] = converged ? 1.0f : 0.0f;

    // Write debug data if enabled
    if (config->debug_enabled && debug_buffer != nullptr) {
        float* iter_debug = &debug_buffer[iter * DebugOffset::PER_ITER_SIZE];
        iter_debug[DebugOffset::ITERATION] = (float)iter;
        iter_debug[DebugOffset::SCORE] = reduce_buffer[ReduceOffset::SCORE];

        // Pose before is in ORIGINAL_POSE if line search, else approximate
        for (int i = 0; i < 6; i++) {
            iter_debug[DebugOffset::POSE_BEFORE + i] =
                config->ls_enabled ? state_buffer[StateOffset::ORIGINAL_POSE + i]
                                  : state_buffer[StateOffset::POSE + i] -
                                    state_buffer[StateOffset::ACTUAL_STEP_LEN] *
                                    state_buffer[StateOffset::DELTA + i] /
                                    (sqrtf(state_buffer[StateOffset::DELTA + 0] *
                                           state_buffer[StateOffset::DELTA + 0] +
                                           state_buffer[StateOffset::DELTA + 1] *
                                           state_buffer[StateOffset::DELTA + 1] +
                                           state_buffer[StateOffset::DELTA + 2] *
                                           state_buffer[StateOffset::DELTA + 2] +
                                           state_buffer[StateOffset::DELTA + 3] *
                                           state_buffer[StateOffset::DELTA + 3] +
                                           state_buffer[StateOffset::DELTA + 4] *
                                           state_buffer[StateOffset::DELTA + 4] +
                                           state_buffer[StateOffset::DELTA + 5] *
                                           state_buffer[StateOffset::DELTA + 5]) + 1e-10f);
        }

        for (int i = 0; i < 6; i++) {
            iter_debug[DebugOffset::GRADIENT + i] = reduce_buffer[ReduceOffset::GRADIENT + i];
        }
        for (int i = 0; i < 21; i++) {
            iter_debug[DebugOffset::HESSIAN_UT + i] = reduce_buffer[ReduceOffset::HESSIAN_UT + i];
        }
        for (int i = 0; i < 6; i++) {
            iter_debug[DebugOffset::DELTA + i] = state_buffer[StateOffset::DELTA + i];
        }
        iter_debug[DebugOffset::ALPHA] = alpha_this_iter;
        iter_debug[DebugOffset::CORRESPONDENCES] = reduce_buffer[ReduceOffset::CORRESPONDENCES];
        iter_debug[DebugOffset::DIR_REVERSED] = 0.0f;  // TODO: track properly
        for (int i = 0; i < 6; i++) {
            iter_debug[DebugOffset::POSE_AFTER + i] = state_buffer[StateOffset::POSE + i];
        }
    }

    // Clear reduce buffer for next iteration
    for (int i = 0; i < ReduceOffset::TOTAL_SIZE; i++) {
        reduce_buffer[i] = 0.0f;
    }

    // Clear line search reduction slots for next iteration
    if (config->ls_enabled) {
        for (int k = 0; k < GRAPH_MAX_LS_CANDIDATES; k++) {
            ls_buffer[LineSearchOffset::CAND_SCORES + k] = 0.0f;
            ls_buffer[LineSearchOffset::CAND_CORR + k] = 0.0f;
            for (int i = 0; i < 6; i++) {
                ls_buffer[LineSearchOffset::CAND_GRADS + k * 6 + i] = 0.0f;
            }
        }
    }

    // Increment iteration counter
    state_buffer[StateOffset::ITERATIONS] = (float)(iter + 1);

    // Write final output if converged or at max iterations
    if (converged || (iter + 1) >= config->max_iterations) {
        for (int i = 0; i < 6; i++) {
            output_buffer[OutputOffset::FINAL_POSE + i] = state_buffer[StateOffset::POSE + i];
        }
        output_buffer[OutputOffset::ITERATIONS] = state_buffer[StateOffset::ITERATIONS];
        output_buffer[OutputOffset::CONVERGED] = state_buffer[StateOffset::CONVERGED];
        output_buffer[OutputOffset::FINAL_SCORE] = reduce_buffer[ReduceOffset::SCORE];
        output_buffer[OutputOffset::NUM_CORRESPONDENCES] = reduce_buffer[ReduceOffset::CORRESPONDENCES];
        output_buffer[OutputOffset::MAX_OSC_COUNT] = state_buffer[StateOffset::MAX_OSC_COUNT];
        float total_iters = state_buffer[StateOffset::ITERATIONS];
        output_buffer[OutputOffset::AVG_ALPHA] = (total_iters > 0.0f) ?
            state_buffer[StateOffset::ALPHA_SUM] / total_iters : 0.0f;
    }
}

// ============================================================================
// Host API: Direct kernel launches (for testing without graphs)
// ============================================================================

typedef int CudaError;

/// Get required buffer sizes
CudaError ndt_graph_get_buffer_sizes(
    uint32_t* state_size,
    uint32_t* reduce_size,
    uint32_t* ls_size,
    uint32_t* output_size
) {
    *state_size = StateOffset::TOTAL_SIZE * sizeof(float);
    *reduce_size = ReduceOffset::TOTAL_SIZE * sizeof(float);
    *ls_size = LineSearchOffset::TOTAL_SIZE * sizeof(float);
    *output_size = OutputOffset::TOTAL_SIZE * sizeof(float);
    return cudaSuccess;
}

/// Get shared memory size for compute kernel
uint32_t ndt_graph_compute_shared_mem_size() {
    return GRAPH_BLOCK_SIZE * GRAPH_REDUCE_SIZE * sizeof(float);
}

/// Get shared memory size for line search kernel
uint32_t ndt_graph_linesearch_shared_mem_size() {
    return GRAPH_BLOCK_SIZE * 8 * sizeof(float);  // 8 values per candidate
}

/// Launch init kernel
CudaError ndt_graph_launch_init(
    const float* initial_pose,
    float* state_buffer,
    float* reduce_buffer,
    float* ls_buffer,
    cudaStream_t stream
) {
    ndt_graph_init_kernel<<<1, 1, 0, stream>>>(
        initial_pose, state_buffer, reduce_buffer, ls_buffer
    );
    return cudaGetLastError();
}

/// Launch compute kernel
CudaError ndt_graph_launch_compute(
    const float* source_points,
    const float* voxel_means,
    const float* voxel_inv_covs,
    const void* hash_table,
    const GraphNdtConfig* config,
    const float* state_buffer,
    float* reduce_buffer,
    uint32_t num_points,
    cudaStream_t stream
) {
    int num_blocks = (num_points + GRAPH_BLOCK_SIZE - 1) / GRAPH_BLOCK_SIZE;
    size_t shared_mem = ndt_graph_compute_shared_mem_size();

    ndt_graph_compute_kernel<<<num_blocks, GRAPH_BLOCK_SIZE, shared_mem, stream>>>(
        source_points, voxel_means, voxel_inv_covs,
        (const GraphHashEntry*)hash_table,
        config, state_buffer, reduce_buffer
    );
    return cudaGetLastError();
}

/// Launch solve kernel
CudaError ndt_graph_launch_solve(
    const GraphNdtConfig* config,
    float* state_buffer,
    float* reduce_buffer,
    float* ls_buffer,
    float* output_buffer,
    cudaStream_t stream
) {
    ndt_graph_solve_kernel<<<1, 1, 0, stream>>>(
        config, state_buffer, reduce_buffer, ls_buffer, output_buffer
    );
    return cudaGetLastError();
}

/// Launch line search kernel
CudaError ndt_graph_launch_linesearch(
    const float* source_points,
    const float* voxel_means,
    const float* voxel_inv_covs,
    const void* hash_table,
    const GraphNdtConfig* config,
    const float* state_buffer,
    float* ls_buffer,
    uint32_t num_points,
    cudaStream_t stream
) {
    int num_blocks = (num_points + GRAPH_BLOCK_SIZE - 1) / GRAPH_BLOCK_SIZE;
    size_t shared_mem = ndt_graph_linesearch_shared_mem_size();

    ndt_graph_linesearch_kernel<<<num_blocks, GRAPH_BLOCK_SIZE, shared_mem, stream>>>(
        source_points, voxel_means, voxel_inv_covs,
        (const GraphHashEntry*)hash_table,
        config, state_buffer, ls_buffer
    );
    return cudaGetLastError();
}

/// Launch update kernel
CudaError ndt_graph_launch_update(
    const GraphNdtConfig* config,
    float* state_buffer,
    float* reduce_buffer,
    float* ls_buffer,
    float* output_buffer,
    float* debug_buffer,
    cudaStream_t stream
) {
    ndt_graph_update_kernel<<<1, 1, 0, stream>>>(
        config, state_buffer, reduce_buffer, ls_buffer, output_buffer, debug_buffer
    );
    return cudaGetLastError();
}

/// Check if converged (reads state_buffer[CONVERGED])
CudaError ndt_graph_check_converged(
    const float* state_buffer,
    bool* converged
) {
    float conv_val;
    cudaError_t err = cudaMemcpy(&conv_val, &state_buffer[StateOffset::CONVERGED],
                                  sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    *converged = conv_val > 0.5f;
    return cudaSuccess;
}

/// Get current iteration count
CudaError ndt_graph_get_iterations(
    const float* state_buffer,
    int32_t* iterations
) {
    float iter_val;
    cudaError_t err = cudaMemcpy(&iter_val, &state_buffer[StateOffset::ITERATIONS],
                                  sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    *iterations = (int32_t)iter_val;
    return cudaSuccess;
}

} // extern "C"
