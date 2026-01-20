// Batch Persistent NDT kernel with atomic barriers
//
// This kernel processes M alignments in parallel by partitioning blocks into
// independent slots. Each slot runs a complete Newton optimization using
// atomic barriers for intra-slot synchronization instead of cooperative
// grid-wide sync.
//
// Key differences from persistent_ndt.cu:
// - No cooperative groups (cudaLaunchCooperativeKernel not required)
// - Atomic barriers per slot instead of grid.sync()
// - Multiple alignments processed in single kernel launch
// - Shared voxel data, independent per-slot working memory

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "persistent_ndt_device.cuh"
#include "cholesky_6x6.cuh"
#include "warp_reduce.cuh"
#include "warp_cholesky.cuh"

extern "C" {

// ============================================================================
// Constants
// ============================================================================

constexpr int BATCH_BLOCK_SIZE = 256;
constexpr int BATCH_REDUCE_SIZE = 29;  // 1 + 6 + 21 + 1 (score, grad, hess, corr)
constexpr int BATCH_MAX_LS_CANDIDATES = 8;
constexpr int BATCH_LS_BATCH_SIZE = 4;
constexpr int BATCH_REDUCE_BUFFER_SIZE = 160;

// ============================================================================
// Hash table structures (must match voxel_hash.cu and persistent_ndt.cu)
// ============================================================================

struct BatchHashEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};

constexpr int32_t BATCH_EMPTY_SLOT = -1;

__device__ __forceinline__ int64_t batch_pack_key(int32_t gx, int32_t gy, int32_t gz) {
    int64_t ux = (int64_t)(gx + (1 << 20));
    int64_t uy = (int64_t)(gy + (1 << 20));
    int64_t uz = (int64_t)(gz + (1 << 20));
    return (ux << 42) | (uy << 21) | uz;
}

__device__ __forceinline__ uint32_t batch_hash_key(int64_t key, uint32_t capacity) {
    uint64_t k = (uint64_t)key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)(k % capacity);
}

__device__ __forceinline__ int32_t batch_pos_to_grid(float pos, float inv_resolution) {
    return (int32_t)floorf(pos * inv_resolution);
}

// 27 neighbor offsets (3x3x3 cube)
__constant__ int8_t BATCH_NEIGHBOR_OFFSETS[27][3] = {
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
// Texture-enabled helper functions
// ============================================================================

// Read voxel mean from texture memory
__device__ __forceinline__ void tex_read_voxel_mean(
    cudaTextureObject_t tex_means,
    int32_t vidx,
    float* mean_out
) {
    mean_out[0] = tex1Dfetch<float>(tex_means, vidx * 3 + 0);
    mean_out[1] = tex1Dfetch<float>(tex_means, vidx * 3 + 1);
    mean_out[2] = tex1Dfetch<float>(tex_means, vidx * 3 + 2);
}

// Read voxel inverse covariance from texture memory
__device__ __forceinline__ void tex_read_voxel_inv_cov(
    cudaTextureObject_t tex_inv_covs,
    int32_t vidx,
    float* inv_cov_out
) {
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        inv_cov_out[i] = tex1Dfetch<float>(tex_inv_covs, vidx * 9 + i);
    }
}

// Texture-enabled hash query
__device__ __forceinline__ int batch_hash_query_textured(
    float qx, float qy, float qz,
    const BatchHashEntry* hash_table,
    uint32_t capacity,
    float inv_resolution,
    float radius_sq,
    cudaTextureObject_t tex_voxel_means,
    int32_t* neighbor_indices
) {
    int32_t gx = batch_pos_to_grid(qx, inv_resolution);
    int32_t gy = batch_pos_to_grid(qy, inv_resolution);
    int32_t gz = batch_pos_to_grid(qz, inv_resolution);

    int count = 0;

    for (uint32_t n = 0; n < 27 && count < MAX_NEIGHBORS; n++) {
        int32_t nx = gx + BATCH_NEIGHBOR_OFFSETS[n][0];
        int32_t ny = gy + BATCH_NEIGHBOR_OFFSETS[n][1];
        int32_t nz = gz + BATCH_NEIGHBOR_OFFSETS[n][2];

        int64_t key = batch_pack_key(nx, ny, nz);
        uint32_t slot = batch_hash_key(key, capacity);

        for (uint32_t i = 0; i < capacity && count < MAX_NEIGHBORS; i++) {
            uint32_t probe_slot = (slot + i) % capacity;
            int64_t stored_key = hash_table[probe_slot].key;

            if (stored_key == BATCH_EMPTY_SLOT) break;

            if (stored_key == key) {
                int32_t voxel_idx = hash_table[probe_slot].value;

                // Read mean from texture
                float mx = tex1Dfetch<float>(tex_voxel_means, voxel_idx * 3 + 0);
                float my = tex1Dfetch<float>(tex_voxel_means, voxel_idx * 3 + 1);
                float mz = tex1Dfetch<float>(tex_voxel_means, voxel_idx * 3 + 2);

                float dx = qx - mx;
                float dy = qy - my;
                float dz = qz - mz;
                float dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq <= radius_sq) {
                    neighbor_indices[count++] = voxel_idx;
                }
                break;
            }
        }
    }

    return count;
}

// ============================================================================
// Hash query (inline, same as persistent_ndt.cu)
// ============================================================================

__device__ __forceinline__ int batch_hash_query_inline(
    float qx, float qy, float qz,
    const BatchHashEntry* hash_table,
    uint32_t capacity,
    float inv_resolution,
    float radius_sq,
    const float* voxel_means,
    int32_t* neighbor_indices
) {
    int32_t gx = batch_pos_to_grid(qx, inv_resolution);
    int32_t gy = batch_pos_to_grid(qy, inv_resolution);
    int32_t gz = batch_pos_to_grid(qz, inv_resolution);

    int count = 0;

    for (uint32_t n = 0; n < 27 && count < MAX_NEIGHBORS; n++) {
        int32_t nx = gx + BATCH_NEIGHBOR_OFFSETS[n][0];
        int32_t ny = gy + BATCH_NEIGHBOR_OFFSETS[n][1];
        int32_t nz = gz + BATCH_NEIGHBOR_OFFSETS[n][2];

        int64_t key = batch_pack_key(nx, ny, nz);
        uint32_t slot = batch_hash_key(key, capacity);

        for (uint32_t i = 0; i < capacity && count < MAX_NEIGHBORS; i++) {
            uint32_t probe_slot = (slot + i) % capacity;
            int64_t stored_key = hash_table[probe_slot].key;

            if (stored_key == BATCH_EMPTY_SLOT) break;

            if (stored_key == key) {
                int32_t voxel_idx = hash_table[probe_slot].value;
                float mx = voxel_means[voxel_idx * 3 + 0];
                float my = voxel_means[voxel_idx * 3 + 1];
                float mz = voxel_means[voxel_idx * 3 + 2];

                float dx = qx - mx;
                float dy = qy - my;
                float dz = qz - mz;
                float dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq <= radius_sq) {
                    neighbor_indices[count++] = voxel_idx;
                }
                break;
            }
        }
    }

    return count;
}

// ============================================================================
// Atomic Barrier for Slot Synchronization
// ============================================================================

// Per-slot barrier using atomic operations.
// This replaces cooperative grid.sync() for independent slot synchronization.
__device__ void slot_barrier(
    volatile int* barrier_counter,
    volatile int* barrier_sense,
    int num_blocks_in_slot
) {
    // First ensure all threads in this block are ready
    __syncthreads();

    if (threadIdx.x == 0) {
        // Read current sense value before arriving
        int old_sense = *barrier_sense;

        // Atomically increment arrival counter
        int arrived = atomicAdd((int*)barrier_counter, 1);

        if (arrived == num_blocks_in_slot - 1) {
            // Last block to arrive: reset counter and flip sense
            *barrier_counter = 0;
            __threadfence();  // Ensure counter reset is visible before sense flip
            atomicAdd((int*)barrier_sense, 1);  // Flip sense to release waiters
        } else {
            // Wait for sense to change (spin)
            while (*barrier_sense == old_sense) {
                // Spin-wait (could add backoff for efficiency)
            }
        }
    }

    // Ensure all threads in block see the barrier completion
    __syncthreads();
}

// ============================================================================
// Batch Persistent NDT Kernel
// ============================================================================

__global__ void batch_persistent_ndt_kernel(
    // Shared data (read-only, same for all slots)
    const float* __restrict__ voxel_means,
    const float* __restrict__ voxel_inv_covs,
    const BatchHashEntry* __restrict__ hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input data
    const float* __restrict__ all_source_points,  // [num_slots * max_points_per_slot * 3]
    const float* __restrict__ all_initial_poses,  // [num_slots * 6]
    const int* __restrict__ points_per_slot,      // [num_slots]

    // Per-slot working memory
    float* __restrict__ all_reduce_buffers,       // [num_slots * BATCH_REDUCE_BUFFER_SIZE]
    int* __restrict__ barrier_counters,           // [num_slots]
    int* __restrict__ barrier_senses,             // [num_slots]

    // Per-slot outputs
    float* __restrict__ all_out_poses,            // [num_slots * 6]
    int* __restrict__ all_out_iterations,         // [num_slots]
    uint32_t* __restrict__ all_out_converged,     // [num_slots]
    float* __restrict__ all_out_scores,           // [num_slots]
    float* __restrict__ all_out_hessians,         // [num_slots * 36]
    uint32_t* __restrict__ all_out_correspondences, // [num_slots]
    uint32_t* __restrict__ all_out_oscillations,  // [num_slots]
    float* __restrict__ all_out_alpha_sums,       // [num_slots]

    // Control parameters
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon_sq,

    // Line search parameters
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization parameters
    const float* __restrict__ reg_ref_x,          // [num_slots] or nullptr
    const float* __restrict__ reg_ref_y,          // [num_slots] or nullptr
    float reg_scale,
    int reg_enabled
) {
    // ========================================================================
    // Determine slot assignment
    // ========================================================================

    int slot_id = blockIdx.x / blocks_per_slot;
    int local_block_id = blockIdx.x % blocks_per_slot;

    // Early exit if this block is beyond the last slot
    if (slot_id >= num_slots) return;

    // ========================================================================
    // Get per-slot pointers
    // ========================================================================

    int num_points = points_per_slot[slot_id];
    const float* source_points = all_source_points + slot_id * max_points_per_slot * 3;
    float* reduce_buffer = all_reduce_buffers + slot_id * BATCH_REDUCE_BUFFER_SIZE;
    volatile int* my_barrier_counter = &barrier_counters[slot_id];
    volatile int* my_barrier_sense = &barrier_senses[slot_id];

    // Regularization reference for this slot
    float my_reg_ref_x = reg_enabled && reg_ref_x ? reg_ref_x[slot_id] : 0.0f;
    float my_reg_ref_y = reg_enabled && reg_ref_y ? reg_ref_y[slot_id] : 0.0f;

    // ========================================================================
    // Shared memory for block-level reduction
    // ========================================================================

    extern __shared__ float smem[];
    float* partial_sums = smem;

    // ========================================================================
    // Reduce buffer layout (same as persistent_ndt.cu, 160 floats)
    // ========================================================================

    float* g_converged = &reduce_buffer[29];
    float* g_pose = &reduce_buffer[30];
    float* g_delta = &reduce_buffer[36];
    float* g_final_score = &reduce_buffer[42];
    float* g_total_corr = &reduce_buffer[43];
    float* g_prev_prev_pos = &reduce_buffer[44];
    float* g_prev_pos = &reduce_buffer[47];
    float* g_curr_osc_count = &reduce_buffer[50];
    float* g_max_osc_count = &reduce_buffer[51];
    float* g_phi_candidates = &reduce_buffer[52];
    float* g_dphi_candidates = &reduce_buffer[60];
    float* g_alpha_candidates = &reduce_buffer[68];
    float* g_original_pose = &reduce_buffer[76];
    float* g_phi_0 = &reduce_buffer[82];
    float* g_dphi_0 = &reduce_buffer[83];
    float* g_best_alpha = &reduce_buffer[84];
    float* g_ls_early_term = &reduce_buffer[85];
    float* g_alpha_sum = &reduce_buffer[86];
    float* g_cand_scores = &reduce_buffer[96];
    float* g_cand_corr = &reduce_buffer[104];
    float* g_cand_grads = &reduce_buffer[112];

    // ========================================================================
    // Initialize state (only first block in slot, thread 0)
    // ========================================================================

    if (local_block_id == 0 && threadIdx.x == 0) {
        const float* initial_pose = all_initial_poses + slot_id * 6;
        for (int i = 0; i < 6; i++) {
            g_pose[i] = initial_pose[i];
        }
        *g_converged = 0.0f;
        *g_final_score = 0.0f;
        *g_total_corr = 0.0f;

        // Oscillation detection state
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
        for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
            reduce_buffer[i] = 0.0f;
        }
    }

    // Barrier: wait for initialization
    slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

    // ========================================================================
    // Pre-computed constants
    // ========================================================================

    float inv_resolution = 1.0f / resolution;
    float radius_sq = resolution * resolution;

    // ========================================================================
    // Newton iteration loop
    // ========================================================================

    int iter;
    for (iter = 0; iter < max_iterations; iter++) {

        // --------------------------------------------------------------------
        // PHASE A: Per-point computation
        // --------------------------------------------------------------------

        // Global thread ID within this slot
        uint32_t slot_tid = local_block_id * blockDim.x + threadIdx.x;

        float my_score = 0.0f;
        float my_grad[6] = {0};
        float my_hess[21] = {0};
        float my_correspondences = 0.0f;

        if (slot_tid < (uint32_t)num_points) {
            float px = source_points[slot_tid * 3 + 0];
            float py = source_points[slot_tid * 3 + 1];
            float pz = source_points[slot_tid * 3 + 2];

            float sr, cr, sp, cp, sy, cy;
            compute_sincos_inline(g_pose, &sr, &cr, &sp, &cp, &sy, &cy);

            float T[16];
            compute_transform_inline(g_pose, sr, cr, sp, cp, sy, cy, T);

            float tx, ty, tz;
            transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

            int32_t neighbor_indices[MAX_NEIGHBORS];
            int num_neighbors = batch_hash_query_inline(
                tx, ty, tz,
                hash_table, hash_capacity, inv_resolution, radius_sq,
                voxel_means, neighbor_indices
            );

            float J[18];
            compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

            float pH[15];
            compute_point_hessians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, pH);

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
                for (int i = 0; i < 6; i++) my_grad[i] += grad_contrib[i];
                for (int i = 0; i < 21; i++) my_hess[i] += hess_contrib[i];
                my_correspondences += 1.0f;
            }
        }

        // --------------------------------------------------------------------
        // PHASE B: Block reduction
        // --------------------------------------------------------------------

        // Store to shared memory
        partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 0] = my_score;
        for (int i = 0; i < 6; i++) {
            partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 1 + i] = my_grad[i];
        }
        for (int i = 0; i < 21; i++) {
            partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 7 + i] = my_hess[i];
        }
        partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 28] = my_correspondences;
        __syncthreads();

        // Tree reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                    partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + i] +=
                        partial_sums[(threadIdx.x + stride) * BATCH_REDUCE_SIZE + i];
                }
            }
            __syncthreads();
        }

        // Atomic add to slot's reduce buffer
        if (threadIdx.x == 0) {
            for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                atomicAdd(&reduce_buffer[i], partial_sums[i]);
            }
        }

        // Barrier: wait for all blocks in slot to finish reduction
        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        // --------------------------------------------------------------------
        // PHASE C: Newton solve (single thread per slot)
        // --------------------------------------------------------------------

        if (local_block_id == 0 && threadIdx.x == 0) {
            float score = reduce_buffer[0];
            float correspondence_count = reduce_buffer[28];

            // Store score and correspondence count
            *g_final_score = score;
            *g_total_corr += correspondence_count;

            // Prepare gradient and Hessian
            float grad[6];
            for (int i = 0; i < 6; i++) {
                grad[i] = reduce_buffer[1 + i];
            }

            float H[36];
            // Expand upper triangle to full matrix
            int ut_idx = 0;
            for (int i = 0; i < 6; i++) {
                for (int j = i; j < 6; j++) {
                    float val = reduce_buffer[7 + ut_idx];
                    H[i * 6 + j] = val;
                    H[j * 6 + i] = val;
                    ut_idx++;
                }
            }

            // Store Hessian in output
            float* out_hessian = all_out_hessians + slot_id * 36;
            for (int i = 0; i < 36; i++) {
                out_hessian[i] = H[i];
            }

            // Apply regularization if enabled
            if (reg_enabled && correspondence_count > 0) {
                float dx = my_reg_ref_x - g_pose[0];
                float dy = my_reg_ref_y - g_pose[1];
                float yaw = g_pose[5];
                float sin_yaw = sinf(yaw);
                float cos_yaw = cosf(yaw);
                float longitudinal = dy * sin_yaw + dx * cos_yaw;
                float weight = correspondence_count;
                score += -reg_scale * weight * longitudinal * longitudinal;

                grad[0] += reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                grad[1] += reg_scale * weight * 2.0f * sin_yaw * longitudinal;

                float d2L_dx2 = 2.0f * reg_scale * weight * cos_yaw * cos_yaw;
                float d2L_dy2 = 2.0f * reg_scale * weight * sin_yaw * sin_yaw;
                float d2L_dxdy = 2.0f * reg_scale * weight * cos_yaw * sin_yaw;

                H[0] += d2L_dx2;
                H[7] += d2L_dy2;
                H[1] += d2L_dxdy;
                H[6] += d2L_dxdy;

                *g_final_score = score;
            }

            // Convert to f64 for Cholesky solve
            double g_f64[6];
            double H_f64[36];

            for (int i = 0; i < 6; i++) {
                g_f64[i] = (double)grad[i];
            }

            // Copy H (already full matrix) to f64
            for (int i = 0; i < 36; i++) {
                H_f64[i] = (double)H[i];
            }

            // Cholesky solve: H * delta = -g
            bool solve_success;
            cholesky_solve_6x6_f64(H_f64, g_f64, &solve_success);

            if (!solve_success || correspondence_count < 1.0f) {
                *g_converged = 1.0f;  // Signal to stop
            } else {
                // Convert result back to f32
                for (int i = 0; i < 6; i++) {
                    g_delta[i] = (float)g_f64[i];
                }

                // Check convergence
                float delta_sq = 0.0f;
                for (int i = 0; i < 6; i++) {
                    delta_sq += g_delta[i] * g_delta[i];
                }
                if (delta_sq < epsilon_sq) {
                    *g_converged = 1.0f;
                }
            }

            // Line search setup
            if (ls_enabled && *g_converged < 0.5f) {
                // Store original pose and current score
                for (int i = 0; i < 6; i++) {
                    g_original_pose[i] = g_pose[i];
                }
                *g_phi_0 = score;

                // Directional derivative: gradient · delta
                float dphi_0 = 0.0f;
                for (int i = 0; i < 6; i++) {
                    dphi_0 += grad[i] * g_delta[i];
                }
                *g_dphi_0 = dphi_0;

                // Generate candidates (golden ratio)
                float phi = 1.618033988749895f;
                float alpha = 1.0f;
                for (int k = 0; k < ls_num_candidates && k < BATCH_MAX_LS_CANDIDATES; k++) {
                    g_alpha_candidates[k] = (k == 0) ? 1.0f : alpha;
                    alpha /= phi;
                }

                *g_best_alpha = fixed_step_size;
                *g_ls_early_term = 0.0f;
            } else if (*g_converged < 0.5f) {
                // No line search: use fixed step with clamping
                constexpr float STEP_MIN = 0.005f;
                float delta_norm_sq = 0.0f;
                for (int i = 0; i < 6; i++) {
                    delta_norm_sq += g_delta[i] * g_delta[i];
                }
                float delta_norm = sqrtf(delta_norm_sq);
                float step_length = delta_norm;
                if (step_length > fixed_step_size) step_length = fixed_step_size;
                if (step_length < STEP_MIN) step_length = STEP_MIN;

                float scale = (delta_norm > 1e-10f) ? (step_length / delta_norm) : 0.0f;
                for (int i = 0; i < 6; i++) {
                    g_pose[i] += scale * g_delta[i];
                }
                *g_alpha_sum += step_length;
            }

            // Clear reduce buffer for next iteration
            for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                reduce_buffer[i] = 0.0f;
            }
        }

        // Barrier: wait for Newton solve
        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        // --------------------------------------------------------------------
        // PHASE C.2: Line search (if enabled)
        // --------------------------------------------------------------------

        if (ls_enabled && *g_converged < 0.5f) {
            int num_cands = (ls_num_candidates < BATCH_MAX_LS_CANDIDATES) ?
                            ls_num_candidates : BATCH_MAX_LS_CANDIDATES;
            int num_batches = (num_cands + BATCH_LS_BATCH_SIZE - 1) / BATCH_LS_BATCH_SIZE;

            for (int batch = 0; batch < num_batches; batch++) {
                int batch_start = batch * BATCH_LS_BATCH_SIZE;
                int batch_end = batch_start + BATCH_LS_BATCH_SIZE;
                if (batch_end > num_cands) batch_end = num_cands;
                int batch_count = batch_end - batch_start;

                // Clear per-candidate slots
                if (local_block_id == 0 && threadIdx.x == 0) {
                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        g_cand_scores[cand_idx] = 0.0f;
                        g_cand_corr[cand_idx] = 0.0f;
                        for (int i = 0; i < 6; i++) {
                            g_cand_grads[cand_idx * 6 + i] = 0.0f;
                        }
                    }
                }
                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                // Evaluate candidates
                float my_batch_scores[BATCH_LS_BATCH_SIZE] = {0};
                float my_batch_grads[BATCH_LS_BATCH_SIZE * 6] = {0};
                float my_batch_corr[BATCH_LS_BATCH_SIZE] = {0};

                if (slot_tid < (uint32_t)num_points) {
                    float px = source_points[slot_tid * 3 + 0];
                    float py = source_points[slot_tid * 3 + 1];
                    float pz = source_points[slot_tid * 3 + 2];

                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        float alpha = g_alpha_candidates[cand_idx];

                        float trial_pose[6];
                        for (int i = 0; i < 6; i++) {
                            trial_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                        }

                        float sr, cr, sp, cp, sy, cy;
                        compute_sincos_inline(trial_pose, &sr, &cr, &sp, &cp, &sy, &cy);

                        float T[16];
                        compute_transform_inline(trial_pose, sr, cr, sp, cp, sy, cy, T);

                        float tx, ty, tz;
                        transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

                        int32_t neighbor_indices[MAX_NEIGHBORS];
                        int num_neighbors = batch_hash_query_inline(
                            tx, ty, tz,
                            hash_table, hash_capacity, inv_resolution, radius_sq,
                            voxel_means, neighbor_indices
                        );

                        float J[18];
                        compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

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
                            my_batch_scores[c] += -gauss_d1 * exp_val;

                            float scale = gauss_d1 * gauss_d2 * exp_val;
                            for (int j = 0; j < 6; j++) {
                                float Jt_Sd = J[j * 3 + 0] * Sd_0 + J[j * 3 + 1] * Sd_1 + J[j * 3 + 2] * Sd_2;
                                my_batch_grads[c * 6 + j] += scale * Jt_Sd;
                            }
                            my_batch_corr[c] += 1.0f;
                        }
                    }
                }

                // Block reduction per candidate
                constexpr int LS_VALUES_PER_CAND = 8;
                for (int c = 0; c < batch_count; c++) {
                    partial_sums[threadIdx.x * LS_VALUES_PER_CAND + 0] = my_batch_scores[c];
                    for (int i = 0; i < 6; i++) {
                        partial_sums[threadIdx.x * LS_VALUES_PER_CAND + 1 + i] = my_batch_grads[c * 6 + i];
                    }
                    partial_sums[threadIdx.x * LS_VALUES_PER_CAND + 7] = my_batch_corr[c];
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

                    if (threadIdx.x == 0) {
                        int cand_idx = batch_start + c;
                        atomicAdd(&g_cand_scores[cand_idx], partial_sums[0]);
                        for (int i = 0; i < 6; i++) {
                            atomicAdd(&g_cand_grads[cand_idx * 6 + i], partial_sums[1 + i]);
                        }
                        atomicAdd(&g_cand_corr[cand_idx], partial_sums[7]);
                    }
                    __syncthreads();
                }

                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                // Check Wolfe conditions
                if (local_block_id == 0 && threadIdx.x == 0) {
                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        float phi_k = g_cand_scores[cand_idx];
                        float alpha = g_alpha_candidates[cand_idx];

                        // Compute trial pose for regularization
                        float trial_pose[6];
                        for (int i = 0; i < 6; i++) {
                            trial_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                        }

                        float grad[6];
                        for (int i = 0; i < 6; i++) {
                            grad[i] = g_cand_grads[cand_idx * 6 + i];
                        }

                        if (reg_enabled) {
                            float corr = g_cand_corr[cand_idx];
                            float dx = my_reg_ref_x - trial_pose[0];
                            float dy = my_reg_ref_y - trial_pose[1];
                            float yaw = trial_pose[5];
                            float sin_yaw = sinf(yaw);
                            float cos_yaw = cosf(yaw);
                            float longitudinal = dy * sin_yaw + dx * cos_yaw;
                            float weight = corr;
                            phi_k += -reg_scale * weight * longitudinal * longitudinal;
                            grad[0] += reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                            grad[1] += reg_scale * weight * 2.0f * sin_yaw * longitudinal;
                        }

                        g_phi_candidates[cand_idx] = phi_k;

                        float dphi_k = 0.0f;
                        for (int i = 0; i < 6; i++) {
                            dphi_k += grad[i] * g_delta[i];
                        }
                        g_dphi_candidates[cand_idx] = dphi_k;

                        bool armijo = phi_k >= *g_phi_0 + ls_mu * alpha * (*g_dphi_0);
                        bool curvature = fabsf(dphi_k) <= ls_nu * fabsf(*g_dphi_0);

                        if (armijo && curvature) {
                            *g_best_alpha = alpha;
                            *g_ls_early_term = 1.0f;
                            break;
                        }
                    }
                }

                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                if (*g_ls_early_term > 0.5f) break;
            }

            // More-Thuente selection and pose update
            if (local_block_id == 0 && threadIdx.x == 0) {
                if (*g_ls_early_term < 0.5f) {
                    float best_phi = *g_phi_0;
                    float best_alpha_val = 0.0f;
                    int num_cands_mt = (ls_num_candidates < BATCH_MAX_LS_CANDIDATES) ?
                                       ls_num_candidates : BATCH_MAX_LS_CANDIDATES;
                    for (int k = 0; k < num_cands_mt; k++) {
                        if (g_phi_candidates[k] > best_phi) {
                            best_phi = g_phi_candidates[k];
                            best_alpha_val = g_alpha_candidates[k];
                        }
                    }
                    *g_best_alpha = best_alpha_val;
                }

                float alpha = *g_best_alpha;
                for (int i = 0; i < 6; i++) {
                    g_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                }
                *g_alpha_sum += alpha;
            }

            slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);
        }

        // --------------------------------------------------------------------
        // PHASE C.3: Oscillation detection and convergence
        // --------------------------------------------------------------------

        if (local_block_id == 0 && threadIdx.x == 0) {
            if (iter >= 1) {
                float curr_x = g_pose[0];
                float curr_y = g_pose[1];
                float curr_z = g_pose[2];

                float curr_vec_x = curr_x - g_prev_pos[0];
                float curr_vec_y = curr_y - g_prev_pos[1];
                float curr_vec_z = curr_z - g_prev_pos[2];

                float prev_vec_x = g_prev_pos[0] - g_prev_prev_pos[0];
                float prev_vec_y = g_prev_pos[1] - g_prev_prev_pos[1];
                float prev_vec_z = g_prev_pos[2] - g_prev_prev_pos[2];

                float dot_product = curr_vec_x * prev_vec_x + curr_vec_y * prev_vec_y + curr_vec_z * prev_vec_z;

                if (dot_product < 0.0f) {
                    *g_curr_osc_count += 1.0f;
                    if (*g_curr_osc_count > *g_max_osc_count) {
                        *g_max_osc_count = *g_curr_osc_count;
                    }
                } else {
                    *g_curr_osc_count = 0.0f;
                }
            }

            g_prev_prev_pos[0] = g_prev_pos[0];
            g_prev_prev_pos[1] = g_prev_pos[1];
            g_prev_prev_pos[2] = g_prev_pos[2];
            g_prev_pos[0] = g_pose[0];
            g_prev_pos[1] = g_pose[1];
            g_prev_pos[2] = g_pose[2];

            for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                reduce_buffer[i] = 0.0f;
            }
        }

        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        if (*g_converged > 0.5f) break;
    }

    // ========================================================================
    // Write final outputs
    // ========================================================================

    if (local_block_id == 0 && threadIdx.x == 0) {
        float* out_pose = all_out_poses + slot_id * 6;
        for (int i = 0; i < 6; i++) {
            out_pose[i] = g_pose[i];
        }
        all_out_iterations[slot_id] = iter + 1;
        all_out_converged[slot_id] = (*g_converged > 0.5f) ? 1 : 0;
        all_out_scores[slot_id] = *g_final_score;
        all_out_correspondences[slot_id] = (uint32_t)(*g_total_corr);
        all_out_oscillations[slot_id] = (uint32_t)(*g_max_osc_count);
        all_out_alpha_sums[slot_id] = *g_alpha_sum;
    }
}

// ============================================================================
// Texture-enabled Batch Persistent NDT Kernel
// ============================================================================

__global__ void batch_persistent_ndt_kernel_textured(
    // Texture objects for voxel data (read via texture cache)
    cudaTextureObject_t tex_voxel_means,
    cudaTextureObject_t tex_voxel_inv_covs,

    // Hash table (still raw pointer - not worth texturing)
    const BatchHashEntry* __restrict__ hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input data
    const float* __restrict__ all_source_points,
    const float* __restrict__ all_initial_poses,
    const int* __restrict__ points_per_slot,

    // Per-slot working memory
    float* __restrict__ all_reduce_buffers,
    int* __restrict__ barrier_counters,
    int* __restrict__ barrier_senses,

    // Per-slot outputs
    float* __restrict__ all_out_poses,
    int* __restrict__ all_out_iterations,
    uint32_t* __restrict__ all_out_converged,
    float* __restrict__ all_out_scores,
    float* __restrict__ all_out_hessians,
    uint32_t* __restrict__ all_out_correspondences,
    uint32_t* __restrict__ all_out_oscillations,
    float* __restrict__ all_out_alpha_sums,

    // Control parameters
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon_sq,

    // Line search parameters
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization parameters
    const float* __restrict__ reg_ref_x,
    const float* __restrict__ reg_ref_y,
    float reg_scale,
    int reg_enabled
) {
    // ========================================================================
    // Determine slot assignment
    // ========================================================================

    int slot_id = blockIdx.x / blocks_per_slot;
    int local_block_id = blockIdx.x % blocks_per_slot;

    if (slot_id >= num_slots) return;

    // ========================================================================
    // Get per-slot pointers
    // ========================================================================

    int num_points = points_per_slot[slot_id];
    const float* source_points = all_source_points + slot_id * max_points_per_slot * 3;
    float* reduce_buffer = all_reduce_buffers + slot_id * BATCH_REDUCE_BUFFER_SIZE;
    volatile int* my_barrier_counter = &barrier_counters[slot_id];
    volatile int* my_barrier_sense = &barrier_senses[slot_id];

    float my_reg_ref_x = reg_enabled && reg_ref_x ? reg_ref_x[slot_id] : 0.0f;
    float my_reg_ref_y = reg_enabled && reg_ref_y ? reg_ref_y[slot_id] : 0.0f;

    // ========================================================================
    // Shared memory for block-level reduction
    // ========================================================================

    extern __shared__ float smem[];
    float* partial_sums = smem;

    // ========================================================================
    // Reduce buffer layout (same as non-textured version)
    // ========================================================================

    float* g_converged = &reduce_buffer[29];
    float* g_pose = &reduce_buffer[30];
    float* g_delta = &reduce_buffer[36];
    float* g_final_score = &reduce_buffer[42];
    float* g_total_corr = &reduce_buffer[43];
    float* g_prev_prev_pos = &reduce_buffer[44];
    float* g_prev_pos = &reduce_buffer[47];
    float* g_curr_osc_count = &reduce_buffer[50];
    float* g_max_osc_count = &reduce_buffer[51];
    float* g_phi_candidates = &reduce_buffer[52];
    float* g_dphi_candidates = &reduce_buffer[60];
    float* g_alpha_candidates = &reduce_buffer[68];
    float* g_original_pose = &reduce_buffer[76];
    float* g_phi_0 = &reduce_buffer[82];
    float* g_dphi_0 = &reduce_buffer[83];
    float* g_best_alpha = &reduce_buffer[84];
    float* g_ls_early_term = &reduce_buffer[85];
    float* g_alpha_sum = &reduce_buffer[86];
    float* g_cand_scores = &reduce_buffer[96];
    float* g_cand_corr = &reduce_buffer[104];
    float* g_cand_grads = &reduce_buffer[112];

    // ========================================================================
    // Initialize state (only first block in slot, thread 0)
    // ========================================================================

    if (local_block_id == 0 && threadIdx.x == 0) {
        const float* initial_pose = all_initial_poses + slot_id * 6;
        for (int i = 0; i < 6; i++) {
            g_pose[i] = initial_pose[i];
        }
        *g_converged = 0.0f;
        *g_final_score = 0.0f;
        *g_total_corr = 0.0f;

        g_prev_prev_pos[0] = initial_pose[0];
        g_prev_prev_pos[1] = initial_pose[1];
        g_prev_prev_pos[2] = initial_pose[2];
        g_prev_pos[0] = initial_pose[0];
        g_prev_pos[1] = initial_pose[1];
        g_prev_pos[2] = initial_pose[2];
        *g_curr_osc_count = 0.0f;
        *g_max_osc_count = 0.0f;
        *g_alpha_sum = 0.0f;

        for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
            reduce_buffer[i] = 0.0f;
        }
    }

    slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

    // ========================================================================
    // Pre-computed constants
    // ========================================================================

    float inv_resolution = 1.0f / resolution;
    float radius_sq = resolution * resolution;

    // ========================================================================
    // Newton iteration loop
    // ========================================================================

    int iter;
    for (iter = 0; iter < max_iterations; iter++) {

        // --------------------------------------------------------------------
        // PHASE A: Per-point computation (texture reads)
        // --------------------------------------------------------------------

        uint32_t slot_tid = local_block_id * blockDim.x + threadIdx.x;

        float my_score = 0.0f;
        float my_grad[6] = {0};
        float my_hess[21] = {0};
        float my_correspondences = 0.0f;

        if (slot_tid < (uint32_t)num_points) {
            float px = source_points[slot_tid * 3 + 0];
            float py = source_points[slot_tid * 3 + 1];
            float pz = source_points[slot_tid * 3 + 2];

            float sr, cr, sp, cp, sy, cy;
            compute_sincos_inline(g_pose, &sr, &cr, &sp, &cp, &sy, &cy);

            float T[16];
            compute_transform_inline(g_pose, sr, cr, sp, cp, sy, cy, T);

            float tx, ty, tz;
            transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

            int32_t neighbor_indices[MAX_NEIGHBORS];
            int num_neighbors = batch_hash_query_textured(
                tx, ty, tz,
                hash_table, hash_capacity, inv_resolution, radius_sq,
                tex_voxel_means, neighbor_indices
            );

            float J[18];
            compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

            float pH[15];
            compute_point_hessians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, pH);

            for (int n = 0; n < num_neighbors; n++) {
                int32_t vidx = neighbor_indices[n];
                if (vidx < 0) continue;

                // Read voxel data from texture memory
                float voxel_mean[3];
                tex_read_voxel_mean(tex_voxel_means, vidx, voxel_mean);

                float voxel_inv_cov[9];
                tex_read_voxel_inv_cov(tex_voxel_inv_covs, vidx, voxel_inv_cov);

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
                for (int i = 0; i < 6; i++) my_grad[i] += grad_contrib[i];
                for (int i = 0; i < 21; i++) my_hess[i] += hess_contrib[i];
                my_correspondences += 1.0f;
            }
        }

        // --------------------------------------------------------------------
        // PHASE B: Block reduction (same as non-textured)
        // --------------------------------------------------------------------

        partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 0] = my_score;
        for (int i = 0; i < 6; i++) {
            partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 1 + i] = my_grad[i];
        }
        for (int i = 0; i < 21; i++) {
            partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 7 + i] = my_hess[i];
        }
        partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + 28] = my_correspondences;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                    partial_sums[threadIdx.x * BATCH_REDUCE_SIZE + i] +=
                        partial_sums[(threadIdx.x + stride) * BATCH_REDUCE_SIZE + i];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                atomicAdd(&reduce_buffer[i], partial_sums[i]);
            }
        }

        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        // --------------------------------------------------------------------
        // PHASE C: Newton solve (single thread per slot) - same as non-textured
        // --------------------------------------------------------------------

        if (local_block_id == 0 && threadIdx.x == 0) {
            float score = reduce_buffer[0];
            float correspondence_count = reduce_buffer[28];

            *g_final_score = score;
            *g_total_corr += correspondence_count;

            float grad[6];
            for (int i = 0; i < 6; i++) {
                grad[i] = reduce_buffer[1 + i];
            }

            float H[36];
            int ut_idx = 0;
            for (int i = 0; i < 6; i++) {
                for (int j = i; j < 6; j++) {
                    float val = reduce_buffer[7 + ut_idx];
                    H[i * 6 + j] = val;
                    H[j * 6 + i] = val;
                    ut_idx++;
                }
            }

            float* out_hessian = all_out_hessians + slot_id * 36;
            for (int i = 0; i < 36; i++) {
                out_hessian[i] = H[i];
            }

            if (reg_enabled && correspondence_count > 0) {
                float dx = my_reg_ref_x - g_pose[0];
                float dy = my_reg_ref_y - g_pose[1];
                float yaw = g_pose[5];
                float sin_yaw = sinf(yaw);
                float cos_yaw = cosf(yaw);
                float longitudinal = dy * sin_yaw + dx * cos_yaw;
                float weight = correspondence_count;
                score += -reg_scale * weight * longitudinal * longitudinal;

                grad[0] += reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                grad[1] += reg_scale * weight * 2.0f * sin_yaw * longitudinal;

                float d2L_dx2 = 2.0f * reg_scale * weight * cos_yaw * cos_yaw;
                float d2L_dy2 = 2.0f * reg_scale * weight * sin_yaw * sin_yaw;
                float d2L_dxdy = 2.0f * reg_scale * weight * cos_yaw * sin_yaw;

                H[0] += d2L_dx2;
                H[7] += d2L_dy2;
                H[1] += d2L_dxdy;
                H[6] += d2L_dxdy;

                *g_final_score = score;
            }

            double g_f64[6];
            double H_f64[36];

            for (int i = 0; i < 6; i++) {
                g_f64[i] = (double)grad[i];
            }

            for (int i = 0; i < 36; i++) {
                H_f64[i] = (double)H[i];
            }

            bool solve_success;
            cholesky_solve_6x6_f64(H_f64, g_f64, &solve_success);

            if (!solve_success || correspondence_count < 1.0f) {
                *g_converged = 1.0f;
            } else {
                for (int i = 0; i < 6; i++) {
                    g_delta[i] = (float)g_f64[i];
                }

                float delta_sq = 0.0f;
                for (int i = 0; i < 6; i++) {
                    delta_sq += g_delta[i] * g_delta[i];
                }
                if (delta_sq < epsilon_sq) {
                    *g_converged = 1.0f;
                }
            }

            if (ls_enabled && *g_converged < 0.5f) {
                for (int i = 0; i < 6; i++) {
                    g_original_pose[i] = g_pose[i];
                }
                *g_phi_0 = score;

                float dphi_0 = 0.0f;
                for (int i = 0; i < 6; i++) {
                    dphi_0 += grad[i] * g_delta[i];
                }
                *g_dphi_0 = dphi_0;

                float phi = 1.618033988749895f;
                float alpha = 1.0f;
                for (int k = 0; k < ls_num_candidates && k < BATCH_MAX_LS_CANDIDATES; k++) {
                    g_alpha_candidates[k] = (k == 0) ? 1.0f : alpha;
                    alpha /= phi;
                }

                *g_best_alpha = fixed_step_size;
                *g_ls_early_term = 0.0f;
            } else if (*g_converged < 0.5f) {
                constexpr float STEP_MIN = 0.005f;
                float delta_norm_sq = 0.0f;
                for (int i = 0; i < 6; i++) {
                    delta_norm_sq += g_delta[i] * g_delta[i];
                }
                float delta_norm = sqrtf(delta_norm_sq);
                float step_length = delta_norm;
                if (step_length > fixed_step_size) step_length = fixed_step_size;
                if (step_length < STEP_MIN) step_length = STEP_MIN;

                float scale = (delta_norm > 1e-10f) ? (step_length / delta_norm) : 0.0f;
                for (int i = 0; i < 6; i++) {
                    g_pose[i] += scale * g_delta[i];
                }
                *g_alpha_sum += step_length;
            }

            for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                reduce_buffer[i] = 0.0f;
            }
        }

        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        // --------------------------------------------------------------------
        // PHASE C.2: Line search with texture reads
        // --------------------------------------------------------------------

        if (ls_enabled && *g_converged < 0.5f) {
            int num_cands = (ls_num_candidates < BATCH_MAX_LS_CANDIDATES) ?
                            ls_num_candidates : BATCH_MAX_LS_CANDIDATES;
            int num_batches = (num_cands + BATCH_LS_BATCH_SIZE - 1) / BATCH_LS_BATCH_SIZE;

            for (int batch = 0; batch < num_batches; batch++) {
                int batch_start = batch * BATCH_LS_BATCH_SIZE;
                int batch_end = batch_start + BATCH_LS_BATCH_SIZE;
                if (batch_end > num_cands) batch_end = num_cands;
                int batch_count = batch_end - batch_start;

                if (local_block_id == 0 && threadIdx.x == 0) {
                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        g_cand_scores[cand_idx] = 0.0f;
                        g_cand_corr[cand_idx] = 0.0f;
                        for (int i = 0; i < 6; i++) {
                            g_cand_grads[cand_idx * 6 + i] = 0.0f;
                        }
                    }
                }
                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                float my_batch_scores[BATCH_LS_BATCH_SIZE] = {0};
                float my_batch_grads[BATCH_LS_BATCH_SIZE * 6] = {0};
                float my_batch_corr[BATCH_LS_BATCH_SIZE] = {0};

                if (slot_tid < (uint32_t)num_points) {
                    float px = source_points[slot_tid * 3 + 0];
                    float py = source_points[slot_tid * 3 + 1];
                    float pz = source_points[slot_tid * 3 + 2];

                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        float alpha = g_alpha_candidates[cand_idx];

                        float trial_pose[6];
                        for (int i = 0; i < 6; i++) {
                            trial_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                        }

                        float sr, cr, sp, cp, sy, cy;
                        compute_sincos_inline(trial_pose, &sr, &cr, &sp, &cp, &sy, &cy);

                        float T[16];
                        compute_transform_inline(trial_pose, sr, cr, sp, cp, sy, cy, T);

                        float tx, ty, tz;
                        transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

                        int32_t neighbor_indices[MAX_NEIGHBORS];
                        int num_neighbors = batch_hash_query_textured(
                            tx, ty, tz,
                            hash_table, hash_capacity, inv_resolution, radius_sq,
                            tex_voxel_means, neighbor_indices
                        );

                        float J[18];
                        compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

                        for (int n = 0; n < num_neighbors; n++) {
                            int32_t vidx = neighbor_indices[n];
                            if (vidx < 0) continue;

                            // Read voxel data from texture memory
                            float voxel_mean[3];
                            tex_read_voxel_mean(tex_voxel_means, vidx, voxel_mean);

                            float voxel_inv_cov[9];
                            tex_read_voxel_inv_cov(tex_voxel_inv_covs, vidx, voxel_inv_cov);

                            float dx = tx - voxel_mean[0];
                            float dy = ty - voxel_mean[1];
                            float dz = tz - voxel_mean[2];

                            float Sd_0 = voxel_inv_cov[0] * dx + voxel_inv_cov[1] * dy + voxel_inv_cov[2] * dz;
                            float Sd_1 = voxel_inv_cov[3] * dx + voxel_inv_cov[4] * dy + voxel_inv_cov[5] * dz;
                            float Sd_2 = voxel_inv_cov[6] * dx + voxel_inv_cov[7] * dy + voxel_inv_cov[8] * dz;

                            float dSd = dx * Sd_0 + dy * Sd_1 + dz * Sd_2;
                            float exp_val = expf(-gauss_d2 * 0.5f * dSd);
                            my_batch_scores[c] += -gauss_d1 * exp_val;

                            float scale = gauss_d1 * gauss_d2 * exp_val;
                            for (int j = 0; j < 6; j++) {
                                float Jt_Sd = J[j * 3 + 0] * Sd_0 + J[j * 3 + 1] * Sd_1 + J[j * 3 + 2] * Sd_2;
                                my_batch_grads[c * 6 + j] += scale * Jt_Sd;
                            }
                            my_batch_corr[c] += 1.0f;
                        }
                    }
                }

                constexpr int LS_VALUES_PER_CAND = 8;
                for (int c = 0; c < batch_count; c++) {
                    partial_sums[threadIdx.x * LS_VALUES_PER_CAND + 0] = my_batch_scores[c];
                    for (int i = 0; i < 6; i++) {
                        partial_sums[threadIdx.x * LS_VALUES_PER_CAND + 1 + i] = my_batch_grads[c * 6 + i];
                    }
                    partial_sums[threadIdx.x * LS_VALUES_PER_CAND + 7] = my_batch_corr[c];
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

                    if (threadIdx.x == 0) {
                        int cand_idx = batch_start + c;
                        atomicAdd(&g_cand_scores[cand_idx], partial_sums[0]);
                        for (int i = 0; i < 6; i++) {
                            atomicAdd(&g_cand_grads[cand_idx * 6 + i], partial_sums[1 + i]);
                        }
                        atomicAdd(&g_cand_corr[cand_idx], partial_sums[7]);
                    }
                    __syncthreads();
                }

                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                if (local_block_id == 0 && threadIdx.x == 0) {
                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        float phi_k = g_cand_scores[cand_idx];
                        float alpha = g_alpha_candidates[cand_idx];

                        float trial_pose[6];
                        for (int i = 0; i < 6; i++) {
                            trial_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                        }

                        float grad[6];
                        for (int i = 0; i < 6; i++) {
                            grad[i] = g_cand_grads[cand_idx * 6 + i];
                        }

                        if (reg_enabled) {
                            float corr = g_cand_corr[cand_idx];
                            float dx = my_reg_ref_x - trial_pose[0];
                            float dy = my_reg_ref_y - trial_pose[1];
                            float yaw = trial_pose[5];
                            float sin_yaw = sinf(yaw);
                            float cos_yaw = cosf(yaw);
                            float longitudinal = dy * sin_yaw + dx * cos_yaw;
                            float weight = corr;
                            phi_k += -reg_scale * weight * longitudinal * longitudinal;
                            grad[0] += reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                            grad[1] += reg_scale * weight * 2.0f * sin_yaw * longitudinal;
                        }

                        g_phi_candidates[cand_idx] = phi_k;

                        float dphi_k = 0.0f;
                        for (int i = 0; i < 6; i++) {
                            dphi_k += grad[i] * g_delta[i];
                        }
                        g_dphi_candidates[cand_idx] = dphi_k;

                        bool armijo = phi_k >= *g_phi_0 + ls_mu * alpha * (*g_dphi_0);
                        bool curvature = fabsf(dphi_k) <= ls_nu * fabsf(*g_dphi_0);

                        if (armijo && curvature) {
                            *g_best_alpha = alpha;
                            *g_ls_early_term = 1.0f;
                            break;
                        }
                    }
                }

                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                if (*g_ls_early_term > 0.5f) break;
            }

            if (local_block_id == 0 && threadIdx.x == 0) {
                if (*g_ls_early_term < 0.5f) {
                    float best_phi = *g_phi_0;
                    float best_alpha_val = 0.0f;
                    int num_cands_mt = (ls_num_candidates < BATCH_MAX_LS_CANDIDATES) ?
                                       ls_num_candidates : BATCH_MAX_LS_CANDIDATES;
                    for (int k = 0; k < num_cands_mt; k++) {
                        if (g_phi_candidates[k] > best_phi) {
                            best_phi = g_phi_candidates[k];
                            best_alpha_val = g_alpha_candidates[k];
                        }
                    }
                    *g_best_alpha = best_alpha_val;
                }

                float alpha = *g_best_alpha;
                for (int i = 0; i < 6; i++) {
                    g_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                }
                *g_alpha_sum += alpha;
            }

            slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);
        }

        // --------------------------------------------------------------------
        // PHASE C.3: Oscillation detection and convergence
        // --------------------------------------------------------------------

        if (local_block_id == 0 && threadIdx.x == 0) {
            if (iter >= 1) {
                float curr_x = g_pose[0];
                float curr_y = g_pose[1];
                float curr_z = g_pose[2];

                float curr_vec_x = curr_x - g_prev_pos[0];
                float curr_vec_y = curr_y - g_prev_pos[1];
                float curr_vec_z = curr_z - g_prev_pos[2];

                float prev_vec_x = g_prev_pos[0] - g_prev_prev_pos[0];
                float prev_vec_y = g_prev_pos[1] - g_prev_prev_pos[1];
                float prev_vec_z = g_prev_pos[2] - g_prev_prev_pos[2];

                float dot_product = curr_vec_x * prev_vec_x + curr_vec_y * prev_vec_y + curr_vec_z * prev_vec_z;

                if (dot_product < 0.0f) {
                    *g_curr_osc_count += 1.0f;
                    if (*g_curr_osc_count > *g_max_osc_count) {
                        *g_max_osc_count = *g_curr_osc_count;
                    }
                } else {
                    *g_curr_osc_count = 0.0f;
                }
            }

            g_prev_prev_pos[0] = g_prev_pos[0];
            g_prev_prev_pos[1] = g_prev_pos[1];
            g_prev_prev_pos[2] = g_prev_pos[2];
            g_prev_pos[0] = g_pose[0];
            g_prev_pos[1] = g_pose[1];
            g_prev_pos[2] = g_pose[2];

            for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                reduce_buffer[i] = 0.0f;
            }
        }

        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        if (*g_converged > 0.5f) break;
    }

    // ========================================================================
    // Write final outputs
    // ========================================================================

    if (local_block_id == 0 && threadIdx.x == 0) {
        float* out_pose = all_out_poses + slot_id * 6;
        for (int i = 0; i < 6; i++) {
            out_pose[i] = g_pose[i];
        }
        all_out_iterations[slot_id] = iter + 1;
        all_out_converged[slot_id] = (*g_converged > 0.5f) ? 1 : 0;
        all_out_scores[slot_id] = *g_final_score;
        all_out_correspondences[slot_id] = (uint32_t)(*g_total_corr);
        all_out_oscillations[slot_id] = (uint32_t)(*g_max_osc_count);
        all_out_alpha_sums[slot_id] = *g_alpha_sum;
    }
}

// ============================================================================
// Warp-Optimized Batch Persistent NDT Kernel
// ============================================================================
//
// This kernel uses:
// - Warp-level reduction (Phase B) instead of full shared memory tree reduction
// - Warp-cooperative Cholesky solve (Phase C) instead of single-thread solve
//
// Benefits:
// - Reduced shared memory usage (8 warps * 29 values vs 256 threads * 29 values)
// - No __syncthreads() in reduction (uses warp shuffles)
// - Parallelized Newton solve across first warp

// Shared memory size for warp-optimized kernel
// Only need space for cross-warp reduction + line search
constexpr int WARP_OPT_NUM_WARPS = BATCH_BLOCK_SIZE / WARP_SIZE;  // 256/32 = 8
constexpr int WARP_OPT_SMEM_NDT = WARP_OPT_NUM_WARPS * BATCH_REDUCE_SIZE;  // 8 * 29 = 232
constexpr int WARP_OPT_SMEM_LS = BATCH_BLOCK_SIZE * 8;  // For line search (256 * 8)

__global__ void batch_persistent_ndt_kernel_warp_optimized(
    // Shared data (read-only, same for all slots)
    const float* __restrict__ voxel_means,
    const float* __restrict__ voxel_inv_covs,
    const BatchHashEntry* __restrict__ hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input data
    const float* __restrict__ all_source_points,
    const float* __restrict__ all_initial_poses,
    const int* __restrict__ points_per_slot,

    // Per-slot working memory
    float* __restrict__ all_reduce_buffers,
    int* __restrict__ barrier_counters,
    int* __restrict__ barrier_senses,

    // Per-slot outputs
    float* __restrict__ all_out_poses,
    int* __restrict__ all_out_iterations,
    uint32_t* __restrict__ all_out_converged,
    float* __restrict__ all_out_scores,
    float* __restrict__ all_out_hessians,
    uint32_t* __restrict__ all_out_correspondences,
    uint32_t* __restrict__ all_out_oscillations,
    float* __restrict__ all_out_alpha_sums,

    // Control parameters
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon_sq,

    // Line search parameters
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization parameters
    const float* __restrict__ reg_ref_x,
    const float* __restrict__ reg_ref_y,
    float reg_scale,
    int reg_enabled
) {
    // Slot and block identification
    int slot_id = blockIdx.x / blocks_per_slot;
    int local_block_id = blockIdx.x % blocks_per_slot;

    if (slot_id >= num_slots) return;

    // Thread identification
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Per-slot pointers
    int num_points = points_per_slot[slot_id];
    const float* source_points = all_source_points + slot_id * max_points_per_slot * 3;
    float* reduce_buffer = all_reduce_buffers + slot_id * BATCH_REDUCE_BUFFER_SIZE;
    volatile int* my_barrier_counter = &barrier_counters[slot_id];
    volatile int* my_barrier_sense = &barrier_senses[slot_id];

    float my_reg_ref_x = reg_enabled && reg_ref_x ? reg_ref_x[slot_id] : 0.0f;
    float my_reg_ref_y = reg_enabled && reg_ref_y ? reg_ref_y[slot_id] : 0.0f;

    // Shared memory: smaller than original due to warp reduction
    extern __shared__ float smem[];
    float* warp_scratch = smem;  // [num_warps * BATCH_REDUCE_SIZE] for cross-warp reduction
    float* ls_scratch = smem;    // Reused for line search

    // Reduce buffer layout (same as original)
    float* g_converged = &reduce_buffer[29];
    float* g_pose = &reduce_buffer[30];
    float* g_delta = &reduce_buffer[36];
    float* g_final_score = &reduce_buffer[42];
    float* g_total_corr = &reduce_buffer[43];
    float* g_prev_prev_pos = &reduce_buffer[44];
    float* g_prev_pos = &reduce_buffer[47];
    float* g_curr_osc_count = &reduce_buffer[50];
    float* g_max_osc_count = &reduce_buffer[51];
    float* g_phi_candidates = &reduce_buffer[52];
    float* g_dphi_candidates = &reduce_buffer[60];
    float* g_alpha_candidates = &reduce_buffer[68];
    float* g_original_pose = &reduce_buffer[76];
    float* g_phi_0 = &reduce_buffer[82];
    float* g_dphi_0 = &reduce_buffer[83];
    float* g_best_alpha = &reduce_buffer[84];
    float* g_ls_early_term = &reduce_buffer[85];
    float* g_alpha_sum = &reduce_buffer[86];
    float* g_cand_scores = &reduce_buffer[96];
    float* g_cand_corr = &reduce_buffer[104];
    float* g_cand_grads = &reduce_buffer[112];

    // Initialize state
    if (local_block_id == 0 && threadIdx.x == 0) {
        const float* initial_pose = all_initial_poses + slot_id * 6;
        for (int i = 0; i < 6; i++) {
            g_pose[i] = initial_pose[i];
        }
        *g_converged = 0.0f;
        *g_final_score = 0.0f;
        *g_total_corr = 0.0f;

        g_prev_prev_pos[0] = initial_pose[0];
        g_prev_prev_pos[1] = initial_pose[1];
        g_prev_prev_pos[2] = initial_pose[2];
        g_prev_pos[0] = initial_pose[0];
        g_prev_pos[1] = initial_pose[1];
        g_prev_pos[2] = initial_pose[2];
        *g_curr_osc_count = 0.0f;
        *g_max_osc_count = 0.0f;
        *g_alpha_sum = 0.0f;

        for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
            reduce_buffer[i] = 0.0f;
        }
    }

    slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

    float inv_resolution = 1.0f / resolution;
    float radius_sq = resolution * resolution;

    // Newton iteration loop
    int iter;
    for (iter = 0; iter < max_iterations; iter++) {

        // ====================================================================
        // PHASE A: Per-point computation (same as original)
        // ====================================================================

        uint32_t slot_tid = local_block_id * blockDim.x + threadIdx.x;

        float my_score = 0.0f;
        float my_grad[6] = {0};
        float my_hess[21] = {0};
        float my_correspondences = 0.0f;

        if (slot_tid < (uint32_t)num_points) {
            float px = source_points[slot_tid * 3 + 0];
            float py = source_points[slot_tid * 3 + 1];
            float pz = source_points[slot_tid * 3 + 2];

            float sr, cr, sp, cp, sy, cy;
            compute_sincos_inline(g_pose, &sr, &cr, &sp, &cp, &sy, &cy);

            float T[16];
            compute_transform_inline(g_pose, sr, cr, sp, cp, sy, cy, T);

            float tx, ty, tz;
            transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

            int32_t neighbor_indices[MAX_NEIGHBORS];
            int num_neighbors = batch_hash_query_inline(
                tx, ty, tz,
                hash_table, hash_capacity, inv_resolution, radius_sq,
                voxel_means, neighbor_indices
            );

            float J[18];
            compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

            float pH[15];
            compute_point_hessians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, pH);

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
                for (int i = 0; i < 6; i++) my_grad[i] += grad_contrib[i];
                for (int i = 0; i < 21; i++) my_hess[i] += hess_contrib[i];
                my_correspondences += 1.0f;
            }
        }

        // ====================================================================
        // PHASE B: Warp-level reduction (optimized)
        // ====================================================================

        // Step 1: Reduce within each warp using shuffles
        my_score = warp_reduce_sum(my_score);
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            my_grad[i] = warp_reduce_sum(my_grad[i]);
        }
        #pragma unroll
        for (int i = 0; i < 21; i++) {
            my_hess[i] = warp_reduce_sum(my_hess[i]);
        }
        my_correspondences = warp_reduce_sum(my_correspondences);

        // Step 2: Lane 0 of each warp writes to shared memory
        if (lane == 0) {
            warp_scratch[warp_id * BATCH_REDUCE_SIZE + 0] = my_score;
            #pragma unroll
            for (int i = 0; i < 6; i++) {
                warp_scratch[warp_id * BATCH_REDUCE_SIZE + 1 + i] = my_grad[i];
            }
            #pragma unroll
            for (int i = 0; i < 21; i++) {
                warp_scratch[warp_id * BATCH_REDUCE_SIZE + 7 + i] = my_hess[i];
            }
            warp_scratch[warp_id * BATCH_REDUCE_SIZE + 28] = my_correspondences;
        }
        __syncthreads();

        // Step 3: First warp reduces across warps and atomically adds to global
        if (warp_id == 0) {
            // Each lane reads one warp's contribution (if it exists)
            float block_score = (lane < num_warps) ? warp_scratch[lane * BATCH_REDUCE_SIZE + 0] : 0.0f;
            float block_grad[6], block_hess[21], block_corr;

            #pragma unroll
            for (int i = 0; i < 6; i++) {
                block_grad[i] = (lane < num_warps) ? warp_scratch[lane * BATCH_REDUCE_SIZE + 1 + i] : 0.0f;
            }
            #pragma unroll
            for (int i = 0; i < 21; i++) {
                block_hess[i] = (lane < num_warps) ? warp_scratch[lane * BATCH_REDUCE_SIZE + 7 + i] : 0.0f;
            }
            block_corr = (lane < num_warps) ? warp_scratch[lane * BATCH_REDUCE_SIZE + 28] : 0.0f;

            // Reduce across lanes
            block_score = warp_reduce_sum(block_score);
            #pragma unroll
            for (int i = 0; i < 6; i++) {
                block_grad[i] = warp_reduce_sum(block_grad[i]);
            }
            #pragma unroll
            for (int i = 0; i < 21; i++) {
                block_hess[i] = warp_reduce_sum(block_hess[i]);
            }
            block_corr = warp_reduce_sum(block_corr);

            // Lane 0 atomically adds to global reduce buffer
            if (lane == 0) {
                atomicAdd(&reduce_buffer[0], block_score);
                #pragma unroll
                for (int i = 0; i < 6; i++) {
                    atomicAdd(&reduce_buffer[1 + i], block_grad[i]);
                }
                #pragma unroll
                for (int i = 0; i < 21; i++) {
                    atomicAdd(&reduce_buffer[7 + i], block_hess[i]);
                }
                atomicAdd(&reduce_buffer[28], block_corr);
            }
        }

        // Barrier: wait for all blocks in slot
        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        // ====================================================================
        // PHASE C: Warp-cooperative Newton solve
        // ====================================================================

        if (local_block_id == 0) {
            // Only first block does Newton solve
            // First warp (lanes 0-31) collaborates on the solve

            float score = reduce_buffer[0];
            float correspondence_count = reduce_buffer[28];

            if (threadIdx.x == 0) {
                *g_final_score = score;
                *g_total_corr += correspondence_count;
            }

            // Prepare gradient and Hessian (all threads read for consistency)
            float grad[6];
            float H[36];

            #pragma unroll
            for (int i = 0; i < 6; i++) {
                grad[i] = reduce_buffer[1 + i];
            }

            // Expand upper triangle to full matrix
            int ut_idx = 0;
            #pragma unroll
            for (int i = 0; i < 6; i++) {
                #pragma unroll
                for (int j = i; j < 6; j++) {
                    float val = reduce_buffer[7 + ut_idx];
                    H[i * 6 + j] = val;
                    H[j * 6 + i] = val;
                    ut_idx++;
                }
            }

            // Store Hessian in output
            if (threadIdx.x == 0) {
                float* out_hessian = all_out_hessians + slot_id * 36;
                #pragma unroll
                for (int i = 0; i < 36; i++) {
                    out_hessian[i] = H[i];
                }
            }

            // Apply regularization if enabled
            if (reg_enabled && correspondence_count > 0) {
                if (threadIdx.x == 0) {
                    float dx = my_reg_ref_x - g_pose[0];
                    float dy = my_reg_ref_y - g_pose[1];
                    float yaw = g_pose[5];
                    float sin_yaw = sinf(yaw);
                    float cos_yaw = cosf(yaw);
                    float longitudinal = dy * sin_yaw + dx * cos_yaw;
                    float weight = correspondence_count;
                    score += -reg_scale * weight * longitudinal * longitudinal;

                    grad[0] += reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                    grad[1] += reg_scale * weight * 2.0f * sin_yaw * longitudinal;

                    float d2L_dx2 = 2.0f * reg_scale * weight * cos_yaw * cos_yaw;
                    float d2L_dy2 = 2.0f * reg_scale * weight * sin_yaw * sin_yaw;
                    float d2L_dxdy = 2.0f * reg_scale * weight * cos_yaw * sin_yaw;

                    H[0] += d2L_dx2;
                    H[7] += d2L_dy2;
                    H[1] += d2L_dxdy;
                    H[6] += d2L_dxdy;

                    *g_final_score = score;
                }
                __syncthreads();

                // Broadcast updated values from thread 0
                #pragma unroll
                for (int i = 0; i < 6; i++) {
                    grad[i] = __shfl_sync(FULL_WARP_MASK, grad[i], 0);
                }
                #pragma unroll
                for (int i = 0; i < 36; i++) {
                    H[i] = __shfl_sync(FULL_WARP_MASK, H[i], 0);
                }
            }

            // Warp-cooperative Cholesky solve (first warp only)
            if (warp_id == 0) {
                // Convert to f64 for Cholesky
                double g_f64[6];
                double H_f64[36];
                double delta_f64[6];

                #pragma unroll
                for (int i = 0; i < 6; i++) {
                    g_f64[i] = (double)grad[i];
                }
                #pragma unroll
                for (int i = 0; i < 36; i++) {
                    H_f64[i] = (double)H[i];
                }

                bool solve_success;
                warp_solve_6x6_with_fallback(H_f64, g_f64, delta_f64, lane, &solve_success);

                // Convert back to f32 and store
                if (lane == 0) {
                    if (!solve_success || correspondence_count < 1.0f) {
                        *g_converged = 1.0f;
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            g_delta[i] = (float)delta_f64[i];
                        }

                        // Check convergence
                        float delta_sq = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            delta_sq += g_delta[i] * g_delta[i];
                        }
                        if (delta_sq < epsilon_sq) {
                            *g_converged = 1.0f;
                        }
                    }

                    // Line search setup or direct step
                    if (ls_enabled && *g_converged < 0.5f) {
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            g_original_pose[i] = g_pose[i];
                        }
                        *g_phi_0 = score;

                        float dphi_0 = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            dphi_0 += grad[i] * g_delta[i];
                        }
                        *g_dphi_0 = dphi_0;

                        float phi = 1.618033988749895f;
                        float alpha = 1.0f;
                        for (int k = 0; k < ls_num_candidates && k < BATCH_MAX_LS_CANDIDATES; k++) {
                            g_alpha_candidates[k] = (k == 0) ? 1.0f : alpha;
                            alpha /= phi;
                        }

                        *g_best_alpha = fixed_step_size;
                        *g_ls_early_term = 0.0f;
                    } else if (*g_converged < 0.5f) {
                        // No line search: use fixed step with clamping
                        constexpr float STEP_MIN = 0.005f;
                        float delta_norm_sq = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            delta_norm_sq += g_delta[i] * g_delta[i];
                        }
                        float delta_norm = sqrtf(delta_norm_sq);
                        float step_length = delta_norm;
                        if (step_length > fixed_step_size) step_length = fixed_step_size;
                        if (step_length < STEP_MIN) step_length = STEP_MIN;

                        float scale = (delta_norm > 1e-10f) ? (step_length / delta_norm) : 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            g_pose[i] += scale * g_delta[i];
                        }
                        *g_alpha_sum += step_length;
                    }

                    // Clear reduce buffer
                    #pragma unroll
                    for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                        reduce_buffer[i] = 0.0f;
                    }
                }
            }
        }

        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        // ====================================================================
        // PHASE C.2: Line search (uses original shared memory approach)
        // ====================================================================

        if (ls_enabled && *g_converged < 0.5f) {
            int num_cands = (ls_num_candidates < BATCH_MAX_LS_CANDIDATES) ?
                            ls_num_candidates : BATCH_MAX_LS_CANDIDATES;
            int num_batches = (num_cands + BATCH_LS_BATCH_SIZE - 1) / BATCH_LS_BATCH_SIZE;

            for (int batch = 0; batch < num_batches; batch++) {
                int batch_start = batch * BATCH_LS_BATCH_SIZE;
                int batch_end = batch_start + BATCH_LS_BATCH_SIZE;
                if (batch_end > num_cands) batch_end = num_cands;
                int batch_count = batch_end - batch_start;

                if (local_block_id == 0 && threadIdx.x == 0) {
                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        g_cand_scores[cand_idx] = 0.0f;
                        g_cand_corr[cand_idx] = 0.0f;
                        for (int i = 0; i < 6; i++) {
                            g_cand_grads[cand_idx * 6 + i] = 0.0f;
                        }
                    }
                }
                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                float my_batch_scores[BATCH_LS_BATCH_SIZE] = {0};
                float my_batch_grads[BATCH_LS_BATCH_SIZE * 6] = {0};
                float my_batch_corr[BATCH_LS_BATCH_SIZE] = {0};

                if (slot_tid < (uint32_t)num_points) {
                    float px = source_points[slot_tid * 3 + 0];
                    float py = source_points[slot_tid * 3 + 1];
                    float pz = source_points[slot_tid * 3 + 2];

                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        float alpha = g_alpha_candidates[cand_idx];

                        float trial_pose[6];
                        for (int i = 0; i < 6; i++) {
                            trial_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                        }

                        float sr, cr, sp, cp, sy, cy;
                        compute_sincos_inline(trial_pose, &sr, &cr, &sp, &cp, &sy, &cy);

                        float T[16];
                        compute_transform_inline(trial_pose, sr, cr, sp, cp, sy, cy, T);

                        float tx, ty, tz;
                        transform_point_inline(px, py, pz, T, &tx, &ty, &tz);

                        int32_t neighbor_indices[MAX_NEIGHBORS];
                        int num_neighbors = batch_hash_query_inline(
                            tx, ty, tz,
                            hash_table, hash_capacity, inv_resolution, radius_sq,
                            voxel_means, neighbor_indices
                        );

                        float J[18];
                        compute_jacobians_inline(px, py, pz, sr, cr, sp, cp, sy, cy, J);

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
                            my_batch_scores[c] += -gauss_d1 * exp_val;

                            float scale = gauss_d1 * gauss_d2 * exp_val;
                            for (int j = 0; j < 6; j++) {
                                float Jt_Sd = J[j * 3 + 0] * Sd_0 + J[j * 3 + 1] * Sd_1 + J[j * 3 + 2] * Sd_2;
                                my_batch_grads[c * 6 + j] += scale * Jt_Sd;
                            }
                            my_batch_corr[c] += 1.0f;
                        }
                    }
                }

                // Warp-level reduction for line search candidates
                constexpr int LS_VALUES_PER_CAND = 8;
                for (int c = 0; c < batch_count; c++) {
                    // Warp reduce
                    float cand_score = warp_reduce_sum(my_batch_scores[c]);
                    float cand_grads[6];
                    #pragma unroll
                    for (int i = 0; i < 6; i++) {
                        cand_grads[i] = warp_reduce_sum(my_batch_grads[c * 6 + i]);
                    }
                    float cand_corr = warp_reduce_sum(my_batch_corr[c]);

                    // Lane 0 of each warp writes to shared memory
                    if (lane == 0) {
                        ls_scratch[warp_id * LS_VALUES_PER_CAND + 0] = cand_score;
                        for (int i = 0; i < 6; i++) {
                            ls_scratch[warp_id * LS_VALUES_PER_CAND + 1 + i] = cand_grads[i];
                        }
                        ls_scratch[warp_id * LS_VALUES_PER_CAND + 7] = cand_corr;
                    }
                    __syncthreads();

                    // First warp reduces across warps
                    if (warp_id == 0) {
                        float final_score = (lane < num_warps) ? ls_scratch[lane * LS_VALUES_PER_CAND + 0] : 0.0f;
                        float final_grads[6];
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            final_grads[i] = (lane < num_warps) ? ls_scratch[lane * LS_VALUES_PER_CAND + 1 + i] : 0.0f;
                        }
                        float final_corr = (lane < num_warps) ? ls_scratch[lane * LS_VALUES_PER_CAND + 7] : 0.0f;

                        final_score = warp_reduce_sum(final_score);
                        #pragma unroll
                        for (int i = 0; i < 6; i++) {
                            final_grads[i] = warp_reduce_sum(final_grads[i]);
                        }
                        final_corr = warp_reduce_sum(final_corr);

                        if (lane == 0) {
                            int cand_idx = batch_start + c;
                            atomicAdd(&g_cand_scores[cand_idx], final_score);
                            for (int i = 0; i < 6; i++) {
                                atomicAdd(&g_cand_grads[cand_idx * 6 + i], final_grads[i]);
                            }
                            atomicAdd(&g_cand_corr[cand_idx], final_corr);
                        }
                    }
                    __syncthreads();
                }

                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                // Wolfe condition check (single thread)
                if (local_block_id == 0 && threadIdx.x == 0) {
                    for (int c = 0; c < batch_count; c++) {
                        int cand_idx = batch_start + c;
                        float phi_k = g_cand_scores[cand_idx];
                        float alpha = g_alpha_candidates[cand_idx];

                        float trial_pose[6];
                        for (int i = 0; i < 6; i++) {
                            trial_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                        }

                        float grad[6];
                        for (int i = 0; i < 6; i++) {
                            grad[i] = g_cand_grads[cand_idx * 6 + i];
                        }

                        if (reg_enabled) {
                            float corr = g_cand_corr[cand_idx];
                            float dx = my_reg_ref_x - trial_pose[0];
                            float dy = my_reg_ref_y - trial_pose[1];
                            float yaw = trial_pose[5];
                            float sin_yaw = sinf(yaw);
                            float cos_yaw = cosf(yaw);
                            float longitudinal = dy * sin_yaw + dx * cos_yaw;
                            float weight = corr;
                            phi_k += -reg_scale * weight * longitudinal * longitudinal;
                            grad[0] += reg_scale * weight * 2.0f * cos_yaw * longitudinal;
                            grad[1] += reg_scale * weight * 2.0f * sin_yaw * longitudinal;
                        }

                        g_phi_candidates[cand_idx] = phi_k;

                        float dphi_k = 0.0f;
                        for (int i = 0; i < 6; i++) {
                            dphi_k += grad[i] * g_delta[i];
                        }
                        g_dphi_candidates[cand_idx] = dphi_k;

                        bool armijo = phi_k >= *g_phi_0 + ls_mu * alpha * (*g_dphi_0);
                        bool curvature = fabsf(dphi_k) <= ls_nu * fabsf(*g_dphi_0);

                        if (armijo && curvature) {
                            *g_best_alpha = alpha;
                            *g_ls_early_term = 1.0f;
                            break;
                        }
                    }
                }

                slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

                if (*g_ls_early_term > 0.5f) break;
            }

            // More-Thuente selection and pose update
            if (local_block_id == 0 && threadIdx.x == 0) {
                if (*g_ls_early_term < 0.5f) {
                    float best_phi = *g_phi_0;
                    float best_alpha_val = 0.0f;
                    int num_cands_mt = (ls_num_candidates < BATCH_MAX_LS_CANDIDATES) ?
                                       ls_num_candidates : BATCH_MAX_LS_CANDIDATES;
                    for (int k = 0; k < num_cands_mt; k++) {
                        if (g_phi_candidates[k] > best_phi) {
                            best_phi = g_phi_candidates[k];
                            best_alpha_val = g_alpha_candidates[k];
                        }
                    }
                    *g_best_alpha = best_alpha_val;
                }

                float alpha = *g_best_alpha;
                for (int i = 0; i < 6; i++) {
                    g_pose[i] = g_original_pose[i] + alpha * g_delta[i];
                }
                *g_alpha_sum += alpha;
            }

            slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);
        }

        // ====================================================================
        // PHASE C.3: Oscillation detection and convergence
        // ====================================================================

        if (local_block_id == 0 && threadIdx.x == 0) {
            if (iter >= 1) {
                float curr_x = g_pose[0];
                float curr_y = g_pose[1];
                float curr_z = g_pose[2];

                float curr_vec_x = curr_x - g_prev_pos[0];
                float curr_vec_y = curr_y - g_prev_pos[1];
                float curr_vec_z = curr_z - g_prev_pos[2];

                float prev_vec_x = g_prev_pos[0] - g_prev_prev_pos[0];
                float prev_vec_y = g_prev_pos[1] - g_prev_prev_pos[1];
                float prev_vec_z = g_prev_pos[2] - g_prev_prev_pos[2];

                float dot_product = curr_vec_x * prev_vec_x + curr_vec_y * prev_vec_y + curr_vec_z * prev_vec_z;

                if (dot_product < 0.0f) {
                    *g_curr_osc_count += 1.0f;
                    if (*g_curr_osc_count > *g_max_osc_count) {
                        *g_max_osc_count = *g_curr_osc_count;
                    }
                } else {
                    *g_curr_osc_count = 0.0f;
                }
            }

            g_prev_prev_pos[0] = g_prev_pos[0];
            g_prev_prev_pos[1] = g_prev_pos[1];
            g_prev_prev_pos[2] = g_prev_pos[2];
            g_prev_pos[0] = g_pose[0];
            g_prev_pos[1] = g_pose[1];
            g_prev_pos[2] = g_pose[2];

            for (int i = 0; i < BATCH_REDUCE_SIZE; i++) {
                reduce_buffer[i] = 0.0f;
            }
        }

        slot_barrier(my_barrier_counter, my_barrier_sense, blocks_per_slot);

        if (*g_converged > 0.5f) break;
    }

    // Write final outputs
    if (local_block_id == 0 && threadIdx.x == 0) {
        float* out_pose = all_out_poses + slot_id * 6;
        for (int i = 0; i < 6; i++) {
            out_pose[i] = g_pose[i];
        }
        all_out_iterations[slot_id] = iter + 1;
        all_out_converged[slot_id] = (*g_converged > 0.5f) ? 1 : 0;
        all_out_scores[slot_id] = *g_final_score;
        all_out_correspondences[slot_id] = (uint32_t)(*g_total_corr);
        all_out_oscillations[slot_id] = (uint32_t)(*g_max_osc_count);
        all_out_alpha_sums[slot_id] = *g_alpha_sum;
    }
}

// ============================================================================
// Host API
// ============================================================================

typedef int CudaError;

/// Get shared memory size for warp-optimized kernel
uint32_t batch_persistent_ndt_warp_shared_mem_size() {
    // Max of: cross-warp NDT reduction OR line search scratch
    int ndt_smem = WARP_OPT_NUM_WARPS * BATCH_REDUCE_SIZE * sizeof(float);  // 8 * 29 * 4 = 928 bytes
    int ls_smem = WARP_OPT_NUM_WARPS * 8 * sizeof(float);  // 8 * 8 * 4 = 256 bytes
    return (ndt_smem > ls_smem) ? ndt_smem : ls_smem;
}

/// Get recommended blocks per slot for given point count
int batch_persistent_ndt_blocks_per_slot(int num_points) {
    // One block per 256 points, minimum 1
    int blocks = (num_points + BATCH_BLOCK_SIZE - 1) / BATCH_BLOCK_SIZE;
    return blocks > 0 ? blocks : 1;
}

/// Get total grid size for M slots
int batch_persistent_ndt_total_blocks(int num_slots, int blocks_per_slot) {
    return num_slots * blocks_per_slot;
}

/// Get shared memory size per block
uint32_t batch_persistent_ndt_shared_mem_size() {
    return BATCH_BLOCK_SIZE * BATCH_REDUCE_SIZE * sizeof(float);
}

/// Get reduce buffer size per slot (bytes)
uint32_t batch_persistent_ndt_reduce_buffer_size() {
    return BATCH_REDUCE_BUFFER_SIZE * sizeof(float);
}

/// Launch batch persistent NDT kernel
CudaError batch_persistent_ndt_launch(
    // Shared data
    const float* voxel_means,
    const float* voxel_inv_covs,
    const void* hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input
    const float* all_source_points,
    const float* all_initial_poses,
    const int* points_per_slot,

    // Per-slot working memory
    float* all_reduce_buffers,
    int* barrier_counters,
    int* barrier_senses,

    // Per-slot outputs
    float* all_out_poses,
    int* all_out_iterations,
    uint32_t* all_out_converged,
    float* all_out_scores,
    float* all_out_hessians,
    uint32_t* all_out_correspondences,
    uint32_t* all_out_oscillations,
    float* all_out_alpha_sums,

    // Control
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon,

    // Line search
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization
    const float* reg_ref_x,
    const float* reg_ref_y,
    float reg_scale,
    int reg_enabled
) {
    int total_blocks = num_slots * blocks_per_slot;
    size_t shared_mem = BATCH_BLOCK_SIZE * BATCH_REDUCE_SIZE * sizeof(float);
    float epsilon_sq = epsilon * epsilon;

    batch_persistent_ndt_kernel<<<total_blocks, BATCH_BLOCK_SIZE, shared_mem>>>(
        voxel_means, voxel_inv_covs,
        (const BatchHashEntry*)hash_table, hash_capacity,
        gauss_d1, gauss_d2, resolution,
        all_source_points, all_initial_poses, points_per_slot,
        all_reduce_buffers, barrier_counters, barrier_senses,
        all_out_poses, all_out_iterations, all_out_converged,
        all_out_scores, all_out_hessians,
        all_out_correspondences, all_out_oscillations, all_out_alpha_sums,
        num_slots, blocks_per_slot, max_points_per_slot,
        max_iterations, epsilon_sq,
        ls_enabled, ls_num_candidates, ls_mu, ls_nu, fixed_step_size,
        reg_ref_x, reg_ref_y, reg_scale, reg_enabled
    );

    return cudaGetLastError();
}

/// Synchronize device (wait for kernel completion)
CudaError batch_persistent_ndt_sync() {
    return cudaDeviceSynchronize();
}

/// Launch batch persistent NDT kernel with stream support (async)
///
/// This version accepts a CUDA stream for async execution and pipelining.
/// Pass nullptr for the default stream.
CudaError batch_persistent_ndt_launch_async(
    // Shared data
    const float* voxel_means,
    const float* voxel_inv_covs,
    const void* hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input
    const float* all_source_points,
    const float* all_initial_poses,
    const int* points_per_slot,

    // Per-slot working memory
    float* all_reduce_buffers,
    int* barrier_counters,
    int* barrier_senses,

    // Per-slot outputs
    float* all_out_poses,
    int* all_out_iterations,
    uint32_t* all_out_converged,
    float* all_out_scores,
    float* all_out_hessians,
    uint32_t* all_out_correspondences,
    uint32_t* all_out_oscillations,
    float* all_out_alpha_sums,

    // Control
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon,

    // Line search
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization
    const float* reg_ref_x,
    const float* reg_ref_y,
    float reg_scale,
    int reg_enabled,

    // Stream (nullptr for default stream)
    cudaStream_t stream
) {
    int total_blocks = num_slots * blocks_per_slot;
    size_t shared_mem = BATCH_BLOCK_SIZE * BATCH_REDUCE_SIZE * sizeof(float);
    float epsilon_sq = epsilon * epsilon;

    batch_persistent_ndt_kernel<<<total_blocks, BATCH_BLOCK_SIZE, shared_mem, stream>>>(
        voxel_means, voxel_inv_covs,
        (const BatchHashEntry*)hash_table, hash_capacity,
        gauss_d1, gauss_d2, resolution,
        all_source_points, all_initial_poses, points_per_slot,
        all_reduce_buffers, barrier_counters, barrier_senses,
        all_out_poses, all_out_iterations, all_out_converged,
        all_out_scores, all_out_hessians,
        all_out_correspondences, all_out_oscillations, all_out_alpha_sums,
        num_slots, blocks_per_slot, max_points_per_slot,
        max_iterations, epsilon_sq,
        ls_enabled, ls_num_candidates, ls_mu, ls_nu, fixed_step_size,
        reg_ref_x, reg_ref_y, reg_scale, reg_enabled
    );

    return cudaGetLastError();
}

/// Synchronize a specific stream (wait for stream completion)
CudaError batch_persistent_ndt_stream_sync(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

/// Launch batch persistent NDT kernel with texture memory for voxel data
///
/// This version uses texture objects for voxel_means and voxel_inv_covs,
/// which may provide better cache performance for scattered reads.
int batch_persistent_ndt_launch_with_textures(
    // Texture objects (instead of raw pointers)
    cudaTextureObject_t tex_voxel_means,
    cudaTextureObject_t tex_voxel_inv_covs,

    // Hash table (still raw pointer - not worth texturing)
    const void* hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input (same as before)
    const float* all_source_points,
    const float* all_initial_poses,
    const int* points_per_slot,

    // Per-slot working memory
    float* all_reduce_buffers,
    int* barrier_counters,
    int* barrier_senses,

    // Per-slot outputs
    float* all_out_poses,
    int* all_out_iterations,
    uint32_t* all_out_converged,
    float* all_out_scores,
    float* all_out_hessians,
    uint32_t* all_out_correspondences,
    uint32_t* all_out_oscillations,
    float* all_out_alpha_sums,

    // Control
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon,

    // Line search
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization
    const float* reg_ref_x,
    const float* reg_ref_y,
    float reg_scale,
    int reg_enabled,

    // Stream
    cudaStream_t stream
) {
    int total_blocks = num_slots * blocks_per_slot;
    size_t shared_mem = BATCH_BLOCK_SIZE * BATCH_REDUCE_SIZE * sizeof(float);
    float epsilon_sq = epsilon * epsilon;

    batch_persistent_ndt_kernel_textured<<<total_blocks, BATCH_BLOCK_SIZE, shared_mem, stream>>>(
        tex_voxel_means, tex_voxel_inv_covs,
        (const BatchHashEntry*)hash_table, hash_capacity,
        gauss_d1, gauss_d2, resolution,
        all_source_points, all_initial_poses, points_per_slot,
        all_reduce_buffers, barrier_counters, barrier_senses,
        all_out_poses, all_out_iterations, all_out_converged,
        all_out_scores, all_out_hessians,
        all_out_correspondences, all_out_oscillations, all_out_alpha_sums,
        num_slots, blocks_per_slot, max_points_per_slot,
        max_iterations, epsilon_sq,
        ls_enabled, ls_num_candidates, ls_mu, ls_nu, fixed_step_size,
        reg_ref_x, reg_ref_y, reg_scale, reg_enabled
    );

    return cudaGetLastError();
}

/// Launch warp-optimized batch persistent NDT kernel with stream support
///
/// This version uses warp-level reduction and warp-cooperative Newton solve
/// for improved GPU utilization. Uses less shared memory than the original.
CudaError batch_persistent_ndt_launch_warp_optimized(
    // Shared data
    const float* voxel_means,
    const float* voxel_inv_covs,
    const void* hash_table,
    uint32_t hash_capacity,
    float gauss_d1,
    float gauss_d2,
    float resolution,

    // Per-slot input
    const float* all_source_points,
    const float* all_initial_poses,
    const int* points_per_slot,

    // Per-slot working memory
    float* all_reduce_buffers,
    int* barrier_counters,
    int* barrier_senses,

    // Per-slot outputs
    float* all_out_poses,
    int* all_out_iterations,
    uint32_t* all_out_converged,
    float* all_out_scores,
    float* all_out_hessians,
    uint32_t* all_out_correspondences,
    uint32_t* all_out_oscillations,
    float* all_out_alpha_sums,

    // Control
    int num_slots,
    int blocks_per_slot,
    int max_points_per_slot,
    int max_iterations,
    float epsilon,

    // Line search
    int ls_enabled,
    int ls_num_candidates,
    float ls_mu,
    float ls_nu,
    float fixed_step_size,

    // Regularization
    const float* reg_ref_x,
    const float* reg_ref_y,
    float reg_scale,
    int reg_enabled,

    // Stream
    cudaStream_t stream
) {
    int total_blocks = num_slots * blocks_per_slot;
    size_t shared_mem = batch_persistent_ndt_warp_shared_mem_size();
    float epsilon_sq = epsilon * epsilon;

    batch_persistent_ndt_kernel_warp_optimized<<<total_blocks, BATCH_BLOCK_SIZE, shared_mem, stream>>>(
        voxel_means, voxel_inv_covs,
        (const BatchHashEntry*)hash_table, hash_capacity,
        gauss_d1, gauss_d2, resolution,
        all_source_points, all_initial_poses, points_per_slot,
        all_reduce_buffers, barrier_counters, barrier_senses,
        all_out_poses, all_out_iterations, all_out_converged,
        all_out_scores, all_out_hessians,
        all_out_correspondences, all_out_oscillations, all_out_alpha_sums,
        num_slots, blocks_per_slot, max_points_per_slot,
        max_iterations, epsilon_sq,
        ls_enabled, ls_num_candidates, ls_mu, ls_nu, fixed_step_size,
        reg_ref_x, reg_ref_y, reg_scale, reg_enabled
    );

    return cudaGetLastError();
}

} // extern "C"
