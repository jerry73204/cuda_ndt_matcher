// Common definitions for CUDA Graph-based NDT kernels
//
// This header defines buffer layouts and constants shared between the
// graph-based NDT kernels and the cooperative kernel implementation.

#ifndef NDT_GRAPH_COMMON_CUH
#define NDT_GRAPH_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Configuration constants
// ============================================================================

constexpr int GRAPH_BLOCK_SIZE = 256;
constexpr int GRAPH_REDUCE_SIZE = 29;  // score(1) + gradient(6) + hessian(21) + corr(1)
constexpr int GRAPH_MAX_LS_CANDIDATES = 8;
constexpr int GRAPH_LS_BATCH_SIZE = 4;
constexpr int GRAPH_MAX_NEIGHBORS = 8;

// ============================================================================
// State Buffer Layout (102 floats = 408 bytes)
// Persistent across iterations - holds optimization state
// ============================================================================

namespace StateOffset {
    constexpr int POSE = 0;                // [0-5]   Current pose (x, y, z, roll, pitch, yaw)
    constexpr int DELTA = 6;               // [6-11]  Newton step direction
    constexpr int PREV_POS = 12;           // [12-14] Previous position (for oscillation)
    constexpr int PREV_PREV_POS = 15;      // [15-17] Position before previous
    constexpr int CONVERGED = 18;          // [18]    Convergence flag (0.0 or 1.0)
    constexpr int ITERATIONS = 19;         // [19]    Iteration count (as float)
    constexpr int OSC_COUNT = 20;          // [20]    Current oscillation streak
    constexpr int MAX_OSC_COUNT = 21;      // [21]    Maximum observed oscillation count
    constexpr int ALPHA_SUM = 22;          // [22]    Accumulated step sizes
    constexpr int ACTUAL_STEP_LEN = 23;    // [23]    Step length for convergence check
    constexpr int ORIGINAL_POSE = 24;      // [24-29] Saved pose for line search
    constexpr int ALPHA_CANDIDATES = 30;   // [30-37] Line search step sizes [8]
    constexpr int CANDIDATE_SCORES = 38;   // [38-45] Scores at each candidate [8]
    constexpr int CANDIDATE_GRADS = 46;    // [46-93] Gradients at each candidate [8×6=48]
    constexpr int CANDIDATE_CORR = 94;     // [94-101] Correspondences at each candidate [8]

    constexpr int TOTAL_SIZE = 102;
}

// ============================================================================
// Reduce Buffer Layout (29 floats = 116 bytes)
// Cleared each iteration - holds accumulation results
// ============================================================================

namespace ReduceOffset {
    constexpr int SCORE = 0;           // [0]     Accumulated NDT score
    constexpr int GRADIENT = 1;        // [1-6]   Accumulated gradient
    constexpr int HESSIAN_UT = 7;      // [7-27]  Accumulated Hessian upper triangle (21)
    constexpr int CORRESPONDENCES = 28; // [28]   Point-voxel match count

    constexpr int TOTAL_SIZE = 29;
}

// ============================================================================
// Line Search Buffer Layout (64 floats = 256 bytes)
// Per-candidate reduction slots for parallel line search
// ============================================================================

namespace LineSearchOffset {
    constexpr int CAND_SCORES = 0;      // [0-7]   Per-candidate scores
    constexpr int CAND_CORR = 8;        // [8-15]  Per-candidate correspondences
    constexpr int CAND_GRADS = 16;      // [16-63] Per-candidate gradients [8×6=48]
    constexpr int PHI_0 = 64;           // [64]    Score at current pose
    constexpr int DPHI_0 = 65;          // [65]    Directional derivative at current pose
    constexpr int BEST_ALPHA = 66;      // [66]    Selected step size
    constexpr int EARLY_TERM = 67;      // [67]    Early termination flag

    constexpr int TOTAL_SIZE = 68;
}

// ============================================================================
// Output Buffer Layout (48 floats = 192 bytes)
// Final results after optimization
// ============================================================================

namespace OutputOffset {
    constexpr int FINAL_POSE = 0;        // [0-5]   Final optimized pose
    constexpr int ITERATIONS = 6;        // [6]     Number of iterations run
    constexpr int CONVERGED = 7;         // [7]     Convergence flag (as float)
    constexpr int FINAL_SCORE = 8;       // [8]     Final NDT score
    constexpr int HESSIAN = 9;           // [9-44]  Full 6×6 Hessian (for covariance)
    constexpr int NUM_CORRESPONDENCES = 45; // [45] Total correspondences
    constexpr int MAX_OSC_COUNT = 46;    // [46]    Maximum oscillation count
    constexpr int AVG_ALPHA = 47;        // [47]    Average step size

    constexpr int TOTAL_SIZE = 48;
}

// ============================================================================
// Debug Buffer Layout (50 floats per iteration)
// Optional debug output for each iteration
// ============================================================================

namespace DebugOffset {
    constexpr int ITERATION = 0;         // [0]     Iteration number
    constexpr int SCORE = 1;             // [1]     Score at iteration
    constexpr int POSE_BEFORE = 2;       // [2-7]   Pose before update
    constexpr int GRADIENT = 8;          // [8-13]  Gradient
    constexpr int HESSIAN_UT = 14;       // [14-34] Hessian upper triangle (21)
    constexpr int DELTA = 35;            // [35-40] Newton step
    constexpr int ALPHA = 41;            // [41]    Step size used
    constexpr int CORRESPONDENCES = 42;  // [42]    Correspondence count
    constexpr int DIR_REVERSED = 43;     // [43]    Direction reversal flag
    constexpr int POSE_AFTER = 44;       // [44-49] Pose after update

    constexpr int PER_ITER_SIZE = 50;
}

// ============================================================================
// Kernel configuration structure
// ============================================================================

struct GraphNdtConfig {
    // NDT parameters
    float gauss_d1;
    float gauss_d2;
    float resolution;
    float epsilon_sq;

    // Regularization parameters
    float reg_ref_x;
    float reg_ref_y;
    float reg_scale;
    int32_t reg_enabled;

    // Line search parameters
    int32_t ls_enabled;
    int32_t ls_num_candidates;
    float ls_mu;
    float ls_nu;
    float fixed_step_size;

    // Data sizes
    uint32_t num_points;
    uint32_t num_voxels;
    uint32_t hash_capacity;
    int32_t max_iterations;

    // Debug
    int32_t debug_enabled;
};

// ============================================================================
// Hash table structures (must match voxel_hash.cu and persistent_ndt.cu)
// ============================================================================

struct GraphHashEntry {
    int64_t key;
    int32_t value;
    int32_t padding;
};

constexpr int32_t GRAPH_EMPTY_SLOT = -1;

// ============================================================================
// Inline helper functions
// ============================================================================

__device__ __forceinline__ int64_t graph_pack_key(int32_t gx, int32_t gy, int32_t gz) {
    int64_t ux = (int64_t)(gx + (1 << 20));
    int64_t uy = (int64_t)(gy + (1 << 20));
    int64_t uz = (int64_t)(gz + (1 << 20));
    return (ux << 42) | (uy << 21) | uz;
}

__device__ __forceinline__ uint32_t graph_hash_key(int64_t key, uint32_t capacity) {
    uint64_t k = (uint64_t)key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)(k % capacity);
}

__device__ __forceinline__ int32_t graph_pos_to_grid(float pos, float inv_resolution) {
    return (int32_t)floorf(pos * inv_resolution);
}

#endif // NDT_GRAPH_COMMON_CUH
