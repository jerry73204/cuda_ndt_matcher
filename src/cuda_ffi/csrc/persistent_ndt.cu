// Persistent NDT kernel using cooperative groups
//
// This kernel runs the entire Newton optimization loop inside a single kernel
// launch, eliminating CPU-GPU transfers during iteration. Uses cooperative
// groups for grid-wide synchronization between phases.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

#include "persistent_ndt_device.cuh"
#include "cholesky_6x6.cuh"

namespace cg = cooperative_groups;

extern "C" {

// ============================================================================
// Hash table structures (must match voxel_hash.cu)
// ============================================================================

struct HashEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};

constexpr int32_t EMPTY_SLOT = -1;

__device__ __forceinline__ int64_t pack_key(int32_t gx, int32_t gy, int32_t gz) {
    int64_t ux = (int64_t)(gx + (1 << 20));
    int64_t uy = (int64_t)(gy + (1 << 20));
    int64_t uz = (int64_t)(gz + (1 << 20));
    return (ux << 42) | (uy << 21) | uz;
}

__device__ __forceinline__ uint32_t hash_key(int64_t key, uint32_t capacity) {
    uint64_t k = (uint64_t)key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)(k % capacity);
}

__device__ __forceinline__ int32_t pos_to_grid(float pos, float inv_resolution) {
    return (int32_t)floorf(pos * inv_resolution);
}

// 27 neighbor offsets (3x3x3 cube)
__constant__ int8_t NEIGHBOR_OFFSETS[27][3] = {
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

// ============================================================================
// Inline hash query (for persistent kernel)
// ============================================================================

__device__ __forceinline__ int hash_query_inline(
    float qx, float qy, float qz,
    const HashEntry* hash_table,
    uint32_t capacity,
    float inv_resolution,
    float radius_sq,
    const float* voxel_means,
    int32_t* neighbor_indices  // [MAX_NEIGHBORS]
) {
    int32_t gx = pos_to_grid(qx, inv_resolution);
    int32_t gy = pos_to_grid(qy, inv_resolution);
    int32_t gz = pos_to_grid(qz, inv_resolution);

    int count = 0;

    // Check all 27 neighboring cells
    for (uint32_t n = 0; n < 27 && count < MAX_NEIGHBORS; n++) {
        int32_t nx = gx + NEIGHBOR_OFFSETS[n][0];
        int32_t ny = gy + NEIGHBOR_OFFSETS[n][1];
        int32_t nz = gz + NEIGHBOR_OFFSETS[n][2];

        int64_t key = pack_key(nx, ny, nz);
        uint32_t slot = hash_key(key, capacity);

        // Linear probing search
        for (uint32_t i = 0; i < capacity && count < MAX_NEIGHBORS; i++) {
            uint32_t probe_slot = (slot + i) % capacity;
            int64_t stored_key = hash_table[probe_slot].key;

            if (stored_key == EMPTY_SLOT) break;

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
    for (int i = count; i < MAX_NEIGHBORS; i++) {
        neighbor_indices[i] = -1;
    }

    return count;
}

// ============================================================================
// Persistent NDT Kernel
// ============================================================================

__global__ void persistent_ndt_kernel(
    // Read-only inputs
    const float* __restrict__ source_points,      // [N * 3]
    const float* __restrict__ voxel_means,        // [V * 3]
    const float* __restrict__ voxel_inv_covs,     // [V * 9]
    const HashEntry* __restrict__ hash_table,     // [capacity]
    // Configuration
    float gauss_d1,
    float gauss_d2,
    float resolution,
    uint32_t num_points,
    uint32_t num_voxels,
    uint32_t hash_capacity,
    int32_t max_iterations,
    float epsilon_sq,
    // Regularization parameters (Phase 18.2)
    float reg_ref_x,                              // GNSS reference X
    float reg_ref_y,                              // GNSS reference Y
    float reg_scale,                              // Regularization scale factor
    int32_t reg_enabled,                          // 1 if regularization enabled
    // Line search parameters (Phase 18.1)
    int32_t ls_enabled,                           // 1 if line search enabled
    int32_t ls_num_candidates,                    // Number of candidates (default: 8)
    float ls_mu,                                  // Armijo constant (default: 1e-4)
    float ls_nu,                                  // Curvature constant (default: 0.9)
    // Initial pose input
    const float* __restrict__ initial_pose,       // [6]
    // Scratch memory for reduction
    float* __restrict__ reduce_buffer,            // [29] - added 1 for correspondence count
    // Outputs
    float* __restrict__ out_pose,                 // [6]
    int32_t* __restrict__ out_iterations,
    uint32_t* __restrict__ out_converged,
    float* __restrict__ out_final_score,
    float* __restrict__ out_hessian,              // [36] for covariance
    uint32_t* __restrict__ out_num_correspondences, // Phase 18.3
    uint32_t* __restrict__ out_max_oscillation_count, // Phase 18.4
    float* __restrict__ out_alpha_sum, // Phase 19.3: accumulated step sizes
    // Debug output (Phase 19.4)
    int32_t debug_enabled,              // 0 = disabled, 1 = enabled
    float* __restrict__ debug_buffer    // [max_iterations * 50] or nullptr
) {
    // Debug buffer layout per iteration (50 floats):
    // [0]: iteration, [1]: score, [2-7]: pose_before, [8-13]: gradient,
    // [14-34]: hessian_ut, [35-40]: delta, [41]: alpha, [42]: correspondences,
    // [43]: direction_reversed, [44-49]: pose_after
    constexpr int DEBUG_FLOATS_PER_ITER = 50;
    cg::grid_group grid = cg::this_grid();

    // Shared memory for block-level reduction
    // Layout: [29 values per thread] = score(1) + gradient(6) + hessian(21) + correspondences(1)
    extern __shared__ float smem[];
    float* partial_sums = smem;  // [29 * blockDim.x]

    // Use reduce_buffer for cross-block shared state:
    // [0..28]   = reduction values (score[1], gradient[6], hessian[21], correspondences[1])
    // [29]      = converged flag (as float: 0.0 = not converged, 1.0 = converged)
    // [30..35]  = current pose [6]
    // [36..41]  = delta [6]
    // [42]      = final score
    // [43]      = total correspondences (accumulated across iterations for output)
    // [44..46]  = prev_prev_pos [3] (for oscillation detection)
    // [47..49]  = prev_pos [3] (for oscillation detection)
    // [50]      = current oscillation count
    // [51]      = max oscillation count
    // Phase 18.1: Line search state
    // [52..59]  = phi_candidates [8] (scores at each candidate)
    // [60..67]  = dphi_candidates [8] (directional derivatives)
    // [68..75]  = alpha_candidates [8] (step sizes)
    // [76..81]  = original_pose [6] (saved before line search)
    // [82]      = phi_0 (score at current pose)
    // [83]      = dphi_0 (directional derivative at current pose)
    // [84]      = best_alpha (selected step size)
    // [85]      = ls_early_term (early termination flag)
    // [86]      = alpha_sum (accumulated step sizes for avg_alpha)
    // [87..95]  = reserved for alignment (total: 96 floats)
    constexpr int REDUCE_SIZE = 29;  // 1 + 6 + 21 + 1
    constexpr int MAX_LS_CANDIDATES = 8;

    // Pointers into reduce_buffer for state
    float* g_converged = &reduce_buffer[29];
    float* g_pose = &reduce_buffer[30];
    float* g_delta = &reduce_buffer[36];
    float* g_final_score = &reduce_buffer[42];
    float* g_total_corr = &reduce_buffer[43];
    // Phase 18.4: Oscillation detection state
    float* g_prev_prev_pos = &reduce_buffer[44];  // [3]
    float* g_prev_pos = &reduce_buffer[47];       // [3]
    float* g_curr_osc_count = &reduce_buffer[50];
    float* g_max_osc_count = &reduce_buffer[51];
    // Phase 18.1: Line search state
    float* g_phi_candidates = &reduce_buffer[52];     // [8]
    float* g_dphi_candidates = &reduce_buffer[60];    // [8]
    float* g_alpha_candidates = &reduce_buffer[68];   // [8]
    float* g_original_pose = &reduce_buffer[76];      // [6]
    float* g_phi_0 = &reduce_buffer[82];
    float* g_dphi_0 = &reduce_buffer[83];
    float* g_best_alpha = &reduce_buffer[84];
    float* g_ls_early_term = &reduce_buffer[85];
    float* g_alpha_sum = &reduce_buffer[86];

    // Initialize state from input (only thread 0 of block 0)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 6; i++) {
            g_pose[i] = initial_pose[i];
        }
        *g_converged = 0.0f;
        *g_final_score = 0.0f;
        *g_total_corr = 0.0f;

        // Phase 18.4: Initialize oscillation detection state
        // Set prev_prev_pos and prev_pos to initial position
        g_prev_prev_pos[0] = initial_pose[0];
        g_prev_prev_pos[1] = initial_pose[1];
        g_prev_prev_pos[2] = initial_pose[2];
        g_prev_pos[0] = initial_pose[0];
        g_prev_pos[1] = initial_pose[1];
        g_prev_pos[2] = initial_pose[2];
        *g_curr_osc_count = 0.0f;
        *g_max_osc_count = 0.0f;
        *g_alpha_sum = 0.0f;

        // Clear reduction values
        for (int i = 0; i < REDUCE_SIZE; i++) {
            reduce_buffer[i] = 0.0f;
        }
    }
    grid.sync();

    // Pre-computed constants
    float inv_resolution = 1.0f / resolution;
    float radius_sq = resolution * resolution;

    // ========================================================================
    // ITERATION LOOP
    // ========================================================================

    // Phase 19.4: Storage for pose at iteration start (for debug output)
    float pose_before[6];

    int iter;
    for (iter = 0; iter < max_iterations; iter++) {

        // Phase 19.4: Save pose at iteration start (before any updates)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (int i = 0; i < 6; i++) {
                pose_before[i] = g_pose[i];
            }
        }

        // --------------------------------------------------------------------
        // PHASE A: Per-point computation
        // --------------------------------------------------------------------

        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Local accumulators
        float my_score = 0.0f;
        float my_grad[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float my_hess[21] = {0};
        float my_correspondences = 0.0f;  // Phase 18.3: correspondence count

        if (tid < num_points) {
            // Load source point
            float px = source_points[tid * 3 + 0];
            float py = source_points[tid * 3 + 1];
            float pz = source_points[tid * 3 + 2];

            // Compute sin/cos from pose (in global memory)
            float sr, cr, sp, cp, sy, cy;
            compute_sincos_inline(g_pose, &sr, &cr, &sp, &cp, &sy, &cy);

            // Compute transform matrix
            float T[16];
            compute_transform_inline(g_pose, sr, cr, sp, cp, sy, cy, T);

            // Transform point
            float tx, ty, tz;
            transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

            // Hash lookup for neighbors
            int32_t neighbor_indices[MAX_NEIGHBORS];
            int num_neighbors = hash_query_inline(
                tx, ty, tz,
                hash_table, hash_capacity, inv_resolution, radius_sq,
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
                my_correspondences += 1.0f;  // Phase 18.3: count correspondence
            }
        }

        // --------------------------------------------------------------------
        // PHASE B: Block-level reduction
        // --------------------------------------------------------------------

        // Store to shared memory (29 values: score + grad[6] + hess[21] + correspondences)
        partial_sums[threadIdx.x * REDUCE_SIZE + 0] = my_score;
        for (int i = 0; i < 6; i++) {
            partial_sums[threadIdx.x * REDUCE_SIZE + 1 + i] = my_grad[i];
        }
        for (int i = 0; i < 21; i++) {
            partial_sums[threadIdx.x * REDUCE_SIZE + 7 + i] = my_hess[i];
        }
        partial_sums[threadIdx.x * REDUCE_SIZE + 28] = my_correspondences;
        __syncthreads();

        // Tree reduction within block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                for (int i = 0; i < REDUCE_SIZE; i++) {
                    partial_sums[threadIdx.x * REDUCE_SIZE + i] +=
                        partial_sums[(threadIdx.x + stride) * REDUCE_SIZE + i];
                }
            }
            __syncthreads();
        }

        // Thread 0 of each block atomically adds to global reduce buffer
        if (threadIdx.x == 0) {
            for (int i = 0; i < REDUCE_SIZE; i++) {
                atomicAdd(&reduce_buffer[i], partial_sums[i]);
            }
        }
        grid.sync();

        // --------------------------------------------------------------------
        // PHASE C: Newton solve + Regularization (single thread)
        // --------------------------------------------------------------------

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Load reduced values
            float score = reduce_buffer[0];
            float correspondence_count = reduce_buffer[28];

            // Phase 18.2: Apply GNSS regularization to score, gradient, Hessian
            if (reg_enabled) {
                float dx = reg_ref_x - g_pose[0];
                float dy = reg_ref_y - g_pose[1];
                float yaw = g_pose[5];
                float sin_yaw = sinf(yaw);
                float cos_yaw = cosf(yaw);

                // Longitudinal distance in vehicle frame
                float longitudinal = dy * sin_yaw + dx * cos_yaw;

                // Score delta: -scale × weight × distance²
                float weight = correspondence_count;
                score += -reg_scale * weight * longitudinal * longitudinal;

                // Gradient deltas
                float grad_x_delta = reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                float grad_y_delta = reg_scale * weight * 2.0f * sin_yaw * longitudinal;
                reduce_buffer[1] += grad_x_delta;
                reduce_buffer[2] += grad_y_delta;

                // Hessian deltas (upper triangle indices: H[0,0]=0, H[0,1]=1, H[1,1]=6)
                float h00_delta = -reg_scale * weight * 2.0f * cos_yaw * cos_yaw;
                float h01_delta = -reg_scale * weight * 2.0f * cos_yaw * sin_yaw;
                float h11_delta = -reg_scale * weight * 2.0f * sin_yaw * sin_yaw;
                reduce_buffer[7 + 0] += h00_delta;  // H[0,0]
                reduce_buffer[7 + 1] += h01_delta;  // H[0,1]
                reduce_buffer[7 + 6] += h11_delta;  // H[1,1]
            }

            *g_final_score = score;
            *g_total_corr = correspondence_count;  // Save for output

            // Convert to f64 for solve
            double g[6];
            double H[36];

            for (int i = 0; i < 6; i++) {
                g[i] = (double)reduce_buffer[1 + i];
            }

            // Expand upper triangle to full matrix
            expand_upper_triangle_f32_to_f64(reduce_buffer + 7, H);

            // Cholesky solve: H * delta = -g
            bool solve_success;
            cholesky_solve_6x6_f64(H, g, &solve_success);

            if (solve_success) {
                for (int i = 0; i < 6; i++) {
                    g_delta[i] = (float)g[i];
                }
            } else {
                // Fallback: gradient descent with small step
                for (int i = 0; i < 6; i++) {
                    g_delta[i] = -0.001f * reduce_buffer[1 + i];
                }
            }

            // Phase 18.1: Prepare for line search or direct update
            if (ls_enabled) {
                // Save state for line search
                for (int i = 0; i < 6; i++) {
                    g_original_pose[i] = g_pose[i];
                }
                *g_phi_0 = score;
                // Compute dphi_0 = gradient · delta
                float dphi_0_val = 0.0f;
                for (int i = 0; i < 6; i++) {
                    dphi_0_val += reduce_buffer[1 + i] * g_delta[i];
                }
                *g_dphi_0 = dphi_0_val;

                // Generate candidates (golden ratio decay from 1.0)
                float step = 1.0f;
                int num_cands = (ls_num_candidates < MAX_LS_CANDIDATES) ? ls_num_candidates : MAX_LS_CANDIDATES;
                for (int k = 0; k < num_cands; k++) {
                    g_alpha_candidates[k] = step;
                    step *= 0.618f;
                }
                *g_ls_early_term = 0.0f;
                *g_best_alpha = 1.0f;  // Default to full step

                // Clear reduce buffer before line search (it will be reused for each candidate)
                for (int i = 0; i < REDUCE_SIZE; i++) {
                    reduce_buffer[i] = 0.0f;
                }
            } else {
                // No line search - update pose directly with alpha=1.0
                for (int i = 0; i < 6; i++) {
                    g_pose[i] += g_delta[i];
                }
            }
        }
        grid.sync();

        // --------------------------------------------------------------------
        // PHASE C.2: Line search evaluation loop (all threads participate)
        // --------------------------------------------------------------------

        if (ls_enabled && *g_converged < 0.5f) {
            int num_cands = (ls_num_candidates < MAX_LS_CANDIDATES) ? ls_num_candidates : MAX_LS_CANDIDATES;

            for (int k = 0; k < num_cands; k++) {
                // Thread 0 of block 0 sets trial pose
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    float alpha = g_alpha_candidates[k];
                    for (int i = 0; i < 6; i++) {
                        g_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                    }
                }
                grid.sync();

                // All threads compute score and gradient for their points
                // (reuses Phase A code structure, but skips Hessian)
                float my_ls_score = 0.0f;
                float my_ls_grad[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float my_ls_corr = 0.0f;

                if (tid < num_points) {
                    float px = source_points[tid * 3 + 0];
                    float py = source_points[tid * 3 + 1];
                    float pz = source_points[tid * 3 + 2];

                    float sr, cr, sp, cp, sy, cy;
                    compute_sincos_inline(g_pose, &sr, &cr, &sp, &cp, &sy, &cy);

                    float T[16];
                    compute_transform_inline(g_pose, sr, cr, sp, cp, sy, cy, T);

                    float tx, ty, tz;
                    transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

                    int32_t neighbor_indices[MAX_NEIGHBORS];
                    int num_neighbors = hash_query_inline(
                        tx, ty, tz,
                        hash_table, hash_capacity, inv_resolution, radius_sq,
                        voxel_means, neighbor_indices
                    );

                    float J[18];
                    compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

                    // Accumulate score and gradient (skip Hessian)
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

                        // Compute score and gradient contribution only
                        float dx = tx - voxel_mean[0];
                        float dy = ty - voxel_mean[1];
                        float dz = tz - voxel_mean[2];

                        // S⁻¹ * d
                        float Sd_0 = voxel_inv_cov[0] * dx + voxel_inv_cov[1] * dy + voxel_inv_cov[2] * dz;
                        float Sd_1 = voxel_inv_cov[3] * dx + voxel_inv_cov[4] * dy + voxel_inv_cov[5] * dz;
                        float Sd_2 = voxel_inv_cov[6] * dx + voxel_inv_cov[7] * dy + voxel_inv_cov[8] * dz;

                        // d^T * S⁻¹ * d
                        float dSd = dx * Sd_0 + dy * Sd_1 + dz * Sd_2;
                        float exp_val = expf(-gauss_d2 * 0.5f * dSd);
                        my_ls_score += -gauss_d1 * exp_val;

                        // Gradient: d1 * d2 * exp * J^T * S⁻¹ * d
                        float scale = gauss_d1 * gauss_d2 * exp_val;
                        for (int j = 0; j < 6; j++) {
                            float Jt_Sd = J[j * 3 + 0] * Sd_0 + J[j * 3 + 1] * Sd_1 + J[j * 3 + 2] * Sd_2;
                            my_ls_grad[j] += scale * Jt_Sd;
                        }
                        my_ls_corr += 1.0f;
                    }
                }

                // Block reduction for score + gradient (7 values)
                constexpr int LS_REDUCE_SIZE = 8;  // score(1) + gradient(6) + corr(1)
                partial_sums[threadIdx.x * LS_REDUCE_SIZE + 0] = my_ls_score;
                for (int i = 0; i < 6; i++) {
                    partial_sums[threadIdx.x * LS_REDUCE_SIZE + 1 + i] = my_ls_grad[i];
                }
                partial_sums[threadIdx.x * LS_REDUCE_SIZE + 7] = my_ls_corr;
                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (threadIdx.x < stride) {
                        for (int i = 0; i < LS_REDUCE_SIZE; i++) {
                            partial_sums[threadIdx.x * LS_REDUCE_SIZE + i] +=
                                partial_sums[(threadIdx.x + stride) * LS_REDUCE_SIZE + i];
                        }
                    }
                    __syncthreads();
                }

                // Thread 0 of each block atomically adds to global
                if (threadIdx.x == 0) {
                    atomicAdd(&reduce_buffer[0], partial_sums[0]);  // score
                    for (int i = 0; i < 6; i++) {
                        atomicAdd(&reduce_buffer[1 + i], partial_sums[1 + i]);  // gradient
                    }
                    atomicAdd(&reduce_buffer[28], partial_sums[7]);  // correspondences (reuse slot 28)
                }
                grid.sync();

                // Thread 0 of block 0 stores results and checks early termination
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    float phi_k = reduce_buffer[0];

                    // Apply regularization if enabled
                    if (reg_enabled) {
                        float correspondence_count = reduce_buffer[28];
                        float dx = reg_ref_x - g_pose[0];
                        float dy = reg_ref_y - g_pose[1];
                        float yaw = g_pose[5];
                        float sin_yaw = sinf(yaw);
                        float cos_yaw = cosf(yaw);
                        float longitudinal = dy * sin_yaw + dx * cos_yaw;
                        float weight = correspondence_count;
                        phi_k += -reg_scale * weight * longitudinal * longitudinal;

                        // Gradient adjustment for dphi
                        float grad_x_delta = reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                        float grad_y_delta = reg_scale * weight * 2.0f * sin_yaw * longitudinal;
                        reduce_buffer[1] += grad_x_delta;
                        reduce_buffer[2] += grad_y_delta;
                    }

                    g_phi_candidates[k] = phi_k;

                    // Compute dphi = gradient · delta
                    float dphi_k = 0.0f;
                    for (int i = 0; i < 6; i++) {
                        dphi_k += reduce_buffer[1 + i] * g_delta[i];
                    }
                    g_dphi_candidates[k] = dphi_k;

                    // Check strong Wolfe conditions for early termination
                    float phi_0 = *g_phi_0;
                    float dphi_0 = *g_dphi_0;
                    float alpha = g_alpha_candidates[k];

                    // Armijo (sufficient decrease): phi(alpha) >= phi(0) + mu * alpha * dphi(0)
                    // Note: We're maximizing, and dphi_0 should be positive for ascent
                    bool armijo = phi_k >= phi_0 + ls_mu * alpha * dphi_0;

                    // Curvature (strong Wolfe): |dphi(alpha)| <= nu * |dphi(0)|
                    bool curvature = fabsf(dphi_k) <= ls_nu * fabsf(dphi_0);

                    if (armijo && curvature) {
                        *g_best_alpha = alpha;
                        *g_ls_early_term = 1.0f;
                    }

                    // Clear reduce buffer for next candidate
                    for (int i = 0; i < REDUCE_SIZE; i++) {
                        reduce_buffer[i] = 0.0f;
                    }
                }
                grid.sync();

                // Check early termination (all threads read same global value)
                if (*g_ls_early_term > 0.5f) {
                    break;
                }
            }  // end for each candidate

            // More-Thuente selection if no early termination
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                if (*g_ls_early_term < 0.5f) {
                    // Simple selection: find best score among candidates
                    // (Full More-Thuente interpolation is complex; use best score for now)
                    float best_phi = *g_phi_0;
                    float best_alpha_val = 0.0f;  // Stay at current if nothing better
                    int num_cands_mt = (ls_num_candidates < MAX_LS_CANDIDATES) ? ls_num_candidates : MAX_LS_CANDIDATES;
                    for (int k = 0; k < num_cands_mt; k++) {
                        if (g_phi_candidates[k] > best_phi) {
                            best_phi = g_phi_candidates[k];
                            best_alpha_val = g_alpha_candidates[k];
                        }
                    }
                    *g_best_alpha = best_alpha_val;
                }

                // Apply final pose update with best_alpha
                float alpha = *g_best_alpha;
                for (int i = 0; i < 6; i++) {
                    g_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                }
            }
            grid.sync();
        }

        // --------------------------------------------------------------------
        // PHASE C.3: Post-update processing (oscillation, convergence check)
        // --------------------------------------------------------------------

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Phase 18.4: Oscillation detection
            // Compare current movement direction with previous movement direction
            if (iter >= 1) {  // Need at least 2 iterations to compare
                // Current position (after update)
                float curr_x = g_pose[0];
                float curr_y = g_pose[1];
                float curr_z = g_pose[2];

                // Movement vectors
                float curr_vec_x = curr_x - g_prev_pos[0];
                float curr_vec_y = curr_y - g_prev_pos[1];
                float curr_vec_z = curr_z - g_prev_pos[2];

                float prev_vec_x = g_prev_pos[0] - g_prev_prev_pos[0];
                float prev_vec_y = g_prev_pos[1] - g_prev_prev_pos[1];
                float prev_vec_z = g_prev_pos[2] - g_prev_prev_pos[2];

                // Compute norms
                float curr_norm = sqrtf(curr_vec_x * curr_vec_x + curr_vec_y * curr_vec_y + curr_vec_z * curr_vec_z);
                float prev_norm = sqrtf(prev_vec_x * prev_vec_x + prev_vec_y * prev_vec_y + prev_vec_z * prev_vec_z);

                // Check for oscillation if both vectors are non-zero
                if (curr_norm > 1e-10f && prev_norm > 1e-10f) {
                    // Normalize
                    float inv_curr = 1.0f / curr_norm;
                    float inv_prev = 1.0f / prev_norm;
                    curr_vec_x *= inv_curr; curr_vec_y *= inv_curr; curr_vec_z *= inv_curr;
                    prev_vec_x *= inv_prev; prev_vec_y *= inv_prev; prev_vec_z *= inv_prev;

                    // Cosine of angle between vectors
                    float cosine = curr_vec_x * prev_vec_x + curr_vec_y * prev_vec_y + curr_vec_z * prev_vec_z;

                    // Oscillation threshold: -0.9 (about 154 degrees, almost opposite)
                    constexpr float INVERSION_THRESHOLD = -0.9f;
                    if (cosine < INVERSION_THRESHOLD) {
                        *g_curr_osc_count += 1.0f;
                    } else {
                        *g_curr_osc_count = 0.0f;  // Reset on forward motion
                    }
                    if (*g_curr_osc_count > *g_max_osc_count) {
                        *g_max_osc_count = *g_curr_osc_count;
                    }
                } else {
                    *g_curr_osc_count = 0.0f;  // Reset if movement too small
                }
            }

            // Shift position history for next iteration
            g_prev_prev_pos[0] = g_prev_pos[0];
            g_prev_prev_pos[1] = g_prev_pos[1];
            g_prev_prev_pos[2] = g_prev_pos[2];
            g_prev_pos[0] = g_pose[0];
            g_prev_pos[1] = g_pose[1];
            g_prev_pos[2] = g_pose[2];

            // Accumulate alpha for avg_alpha calculation
            // If line search is enabled, use best_alpha; otherwise alpha = 1.0
            float alpha_this_iter = ls_enabled ? *g_best_alpha : 1.0f;
            *g_alpha_sum += alpha_this_iter;

            // Check convergence: ||delta||^2 < epsilon^2
            float delta_norm_sq = 0.0f;
            for (int i = 0; i < 6; i++) {
                delta_norm_sq += g_delta[i] * g_delta[i];
            }
            *g_converged = (delta_norm_sq < epsilon_sq) ? 1.0f : 0.0f;

            // Save Hessian to output BEFORE clearing (for covariance estimation)
            // Expand upper triangle [21] to full symmetric matrix [36]
            expand_upper_triangle_f32(reduce_buffer + 7, out_hessian);

            // Phase 19.4: Write debug data (only if enabled)
            if (debug_enabled && debug_buffer != nullptr) {
                float* iter_debug = &debug_buffer[iter * DEBUG_FLOATS_PER_ITER];

                // [0]: iteration number
                iter_debug[0] = (float)iter;

                // [1]: score
                iter_debug[1] = *g_final_score;

                // [2-7]: pose_before (saved at iteration start)
                for (int i = 0; i < 6; i++) {
                    iter_debug[2 + i] = pose_before[i];
                }

                // [8-13]: gradient (from reduce_buffer, before clearing)
                for (int i = 0; i < 6; i++) {
                    iter_debug[8 + i] = reduce_buffer[1 + i];
                }

                // [14-34]: hessian upper triangle (21 values)
                for (int i = 0; i < 21; i++) {
                    iter_debug[14 + i] = reduce_buffer[7 + i];
                }

                // [35-40]: delta (Newton step)
                for (int i = 0; i < 6; i++) {
                    iter_debug[35 + i] = g_delta[i];
                }

                // [41]: alpha (step size)
                iter_debug[41] = ls_enabled ? *g_best_alpha : 1.0f;

                // [42]: correspondences
                iter_debug[42] = *g_total_corr;

                // [43]: direction_reversed
                // Check if gradient · delta < 0 (not an ascent direction)
                float gd = 0.0f;
                for (int i = 0; i < 6; i++) {
                    gd += reduce_buffer[1 + i] * g_delta[i];
                }
                iter_debug[43] = (gd < 0.0f) ? 1.0f : 0.0f;

                // [44-49]: pose_after (current g_pose after update)
                for (int i = 0; i < 6; i++) {
                    iter_debug[44 + i] = g_pose[i];
                }
            }

            // Clear reduce buffer for next iteration
            for (int i = 0; i < REDUCE_SIZE; i++) {
                reduce_buffer[i] = 0.0f;
            }
        }
        grid.sync();

        // --------------------------------------------------------------------
        // PHASE D: Check convergence (uniform branch - all threads read same global value)
        // --------------------------------------------------------------------

        if (*g_converged > 0.5f) {
            break;
        }
    }

    // ========================================================================
    // Write final outputs
    // ========================================================================

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 6; i++) {
            out_pose[i] = g_pose[i];
        }
        *out_iterations = iter + 1;
        *out_converged = (*g_converged > 0.5f) ? 1 : 0;
        *out_final_score = *g_final_score;
        *out_num_correspondences = (uint32_t)(*g_total_corr);
        *out_max_oscillation_count = (uint32_t)(*g_max_osc_count);  // Phase 18.4
        *out_alpha_sum = *g_alpha_sum;  // Phase 19.3
        // Note: out_hessian is already written in the iteration loop
    }
}

// ============================================================================
// Host API
// ============================================================================

typedef int CudaError;

/// Query maximum grid size for cooperative launch.
CudaError persistent_ndt_get_max_blocks(
    int block_size,
    int shared_mem_bytes,
    int* max_blocks
) {
    int num_blocks_per_sm;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        persistent_ndt_kernel,
        block_size,
        shared_mem_bytes
    );
    if (err != cudaSuccess) return err;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return err;

    *max_blocks = prop.multiProcessorCount * num_blocks_per_sm;
    return cudaSuccess;
}

/// Check if cooperative launch is supported on current device.
CudaError persistent_ndt_is_supported(int* supported) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return err;

    *supported = prop.cooperativeLaunch ? 1 : 0;
    return cudaSuccess;
}

/// Launch persistent NDT optimization kernel.
///
/// @param source_points     Device pointer to source points [N * 3]
/// @param voxel_means       Device pointer to voxel means [V * 3]
/// @param voxel_inv_covs    Device pointer to inverse covariances [V * 9]
/// @param hash_table        Device pointer to hash table
/// @param gauss_d1          NDT Gaussian parameter d1
/// @param gauss_d2          NDT Gaussian parameter d2
/// @param resolution        Voxel resolution
/// @param num_points        Number of source points
/// @param num_voxels        Number of voxels
/// @param hash_capacity     Hash table capacity
/// @param max_iterations    Maximum Newton iterations
/// @param epsilon           Convergence threshold (||delta|| < epsilon)
/// @param reg_ref_x         Regularization reference X (GNSS)
/// @param reg_ref_y         Regularization reference Y (GNSS)
/// @param reg_scale         Regularization scale factor
/// @param reg_enabled       1 if regularization is enabled, 0 otherwise
/// @param ls_enabled        1 if line search is enabled, 0 otherwise
/// @param ls_num_candidates Number of line search candidates (default: 8)
/// @param ls_mu             Armijo constant (default: 1e-4)
/// @param ls_nu             Curvature constant (default: 0.9)
/// @param initial_pose      Device pointer to initial pose [6]
/// @param reduce_buffer     Device pointer to reduction scratch [96]
/// @param out_pose          Device pointer to output pose [6]
/// @param out_iterations    Device pointer to output iteration count
/// @param out_converged     Device pointer to output convergence flag
/// @param out_final_score   Device pointer to output final score
/// @param out_hessian       Device pointer to output Hessian [36]
/// @param out_num_correspondences Device pointer to output correspondence count
/// @param out_max_oscillation_count Device pointer to output max oscillation count (Phase 18.4)
/// @param out_alpha_sum Device pointer to output accumulated step sizes (Phase 19.3)
/// @param debug_enabled 1 to enable debug output, 0 to disable (Phase 19.4)
/// @param debug_buffer Device pointer to debug buffer [max_iterations * 50] or nullptr
CudaError persistent_ndt_launch(
    const float* source_points,
    const float* voxel_means,
    const float* voxel_inv_covs,
    const void* hash_table,
    float gauss_d1,
    float gauss_d2,
    float resolution,
    uint32_t num_points,
    uint32_t num_voxels,
    uint32_t hash_capacity,
    int32_t max_iterations,
    float epsilon,
    float reg_ref_x,
    float reg_ref_y,
    float reg_scale,
    int32_t reg_enabled,
    int32_t ls_enabled,
    int32_t ls_num_candidates,
    float ls_mu,
    float ls_nu,
    const float* initial_pose,
    float* reduce_buffer,
    float* out_pose,
    int32_t* out_iterations,
    uint32_t* out_converged,
    float* out_final_score,
    float* out_hessian,
    uint32_t* out_num_correspondences,
    uint32_t* out_max_oscillation_count,
    float* out_alpha_sum,
    int32_t debug_enabled,
    float* debug_buffer
) {
    if (num_points == 0) {
        // Handle empty input
        cudaMemset(out_pose, 0, 6 * sizeof(float));
        cudaMemcpy(out_pose, initial_pose, 6 * sizeof(float), cudaMemcpyDeviceToDevice);
        int32_t zero_iter = 0;
        uint32_t one = 1;
        uint32_t zero_u32 = 0;
        float zero = 0.0f;
        cudaMemcpy(out_iterations, &zero_iter, sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(out_converged, &one, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(out_final_score, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(out_num_correspondences, &zero_u32, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(out_max_oscillation_count, &zero_u32, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(out_alpha_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);
        return cudaSuccess;
    }

    // Calculate grid dimensions
    const int BLOCK_SIZE = 256;
    const int REDUCE_SIZE = 29;  // score(1) + gradient(6) + hessian(21) + correspondences(1)
    int num_blocks = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Check against max cooperative blocks
    int max_blocks;
    size_t shared_mem_bytes = BLOCK_SIZE * REDUCE_SIZE * sizeof(float);
    CudaError err = persistent_ndt_get_max_blocks(BLOCK_SIZE, shared_mem_bytes, &max_blocks);
    if (err != cudaSuccess) return err;

    if (num_blocks > max_blocks) {
        // Grid too large for cooperative launch
        // Caller should fall back to legacy pipeline
        return cudaErrorCooperativeLaunchTooLarge;
    }

    // Prepare kernel arguments
    float epsilon_sq = epsilon * epsilon;

    void* args[] = {
        (void*)&source_points,
        (void*)&voxel_means,
        (void*)&voxel_inv_covs,
        (void*)&hash_table,
        (void*)&gauss_d1,
        (void*)&gauss_d2,
        (void*)&resolution,
        (void*)&num_points,
        (void*)&num_voxels,
        (void*)&hash_capacity,
        (void*)&max_iterations,
        (void*)&epsilon_sq,
        (void*)&reg_ref_x,
        (void*)&reg_ref_y,
        (void*)&reg_scale,
        (void*)&reg_enabled,
        (void*)&ls_enabled,
        (void*)&ls_num_candidates,
        (void*)&ls_mu,
        (void*)&ls_nu,
        (void*)&initial_pose,
        (void*)&reduce_buffer,
        (void*)&out_pose,
        (void*)&out_iterations,
        (void*)&out_converged,
        (void*)&out_final_score,
        (void*)&out_hessian,
        (void*)&out_num_correspondences,
        (void*)&out_max_oscillation_count,
        (void*)&out_alpha_sum,
        (void*)&debug_enabled,
        (void*)&debug_buffer
    };

    // Launch cooperative kernel
    return cudaLaunchCooperativeKernel(
        (void*)persistent_ndt_kernel,
        dim3(num_blocks),
        dim3(BLOCK_SIZE),
        args,
        shared_mem_bytes
    );
}

/// Get required reduce buffer size in bytes.
/// Layout: [0..28] reduction values (29), [29] converged, [30..35] pose, [36..41] delta,
///         [42] final_score, [43] total_correspondences,
///         [44..46] prev_prev_pos, [47..49] prev_pos, [50] curr_osc, [51] max_osc
///         [52..59] phi_candidates, [60..67] dphi_candidates, [68..75] alpha_candidates,
///         [76..81] original_pose, [82] phi_0, [83] dphi_0, [84] best_alpha, [85] ls_early_term
uint32_t persistent_ndt_reduce_buffer_size() {
    return 96 * sizeof(float);  // 86 needed, 96 for alignment
}

} // extern "C"
