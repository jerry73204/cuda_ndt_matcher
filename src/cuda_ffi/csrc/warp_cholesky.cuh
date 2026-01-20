// Warp-cooperative Cholesky decomposition and solve for 6x6 systems
//
// This implements a warp-parallel solver for the Newton step:
//   H * delta = -g
// where H is the 6x6 Hessian and g is the 6-element gradient.
//
// Benefits:
// - Parallelizes the serial Newton solve across 6 lanes
// - Uses warp shuffles for efficient data exchange
// - No shared memory needed for the solve
// - Reduces idle time (previously only thread 0 worked)
//
// Algorithm:
// 1. Cholesky decomposition: H = L * L^T (lower triangular)
// 2. Forward substitution: L * y = -g
// 3. Backward substitution: L^T * delta = y

#pragma once

#include <cuda_runtime.h>

// Full warp mask
constexpr unsigned CHOL_WARP_MASK = 0xffffffff;

// ============================================================================
// Warp-parallel Cholesky decomposition for 6x6 symmetric positive definite
// ============================================================================

/// Warp-parallel Cholesky decomposition of a 6x6 symmetric matrix.
///
/// Each of lanes 0-5 owns one column of the output L matrix.
/// Lanes 6-31 participate in shuffles but their results are unused.
///
/// @param H Input 6x6 symmetric matrix in row-major [36]
/// @param L Output 6x6 lower triangular matrix in row-major [36]
/// @param lane Thread's lane ID (threadIdx.x % 32)
/// @return true if decomposition succeeded, false if matrix is not positive definite
__device__ __forceinline__ bool warp_cholesky_6x6(
    const double* H,
    double* L,
    int lane
) {
    // Each lane 0-5 will compute column `lane` of L
    // L is lower triangular, so L[i][j] = 0 for j > i

    // Initialize L to zero
    if (lane < 6) {
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            L[i * 6 + lane] = 0.0;
        }
    }
    __syncwarp();

    // Cholesky algorithm: for each column j
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        // Compute L[j][j] = sqrt(H[j][j] - sum_{k<j} L[j][k]^2)
        double diag_sum = 0.0;
        if (lane < j) {
            double Ljk = L[j * 6 + lane];
            diag_sum = Ljk * Ljk;
        }

        // Reduce sum across lanes 0 to j-1
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            diag_sum += __shfl_down_sync(CHOL_WARP_MASK, diag_sum, offset);
        }

        // Lane 0 has the sum, broadcast to all
        diag_sum = __shfl_sync(CHOL_WARP_MASK, diag_sum, 0);

        double Ljj_sq = H[j * 6 + j] - diag_sum;
        if (Ljj_sq <= 0.0) {
            return false;  // Not positive definite
        }
        double Ljj = sqrt(Ljj_sq);

        // Store diagonal element
        if (lane == j) {
            L[j * 6 + j] = Ljj;
        }
        __syncwarp();

        // Compute L[i][j] for i > j (elements below diagonal in column j)
        // L[i][j] = (H[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j]
        #pragma unroll
        for (int i = j + 1; i < 6; i++) {
            double off_sum = 0.0;
            if (lane < j) {
                off_sum = L[i * 6 + lane] * L[j * 6 + lane];
            }

            // Reduce
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                off_sum += __shfl_down_sync(CHOL_WARP_MASK, off_sum, offset);
            }
            off_sum = __shfl_sync(CHOL_WARP_MASK, off_sum, 0);

            double Lij = (H[i * 6 + j] - off_sum) / Ljj;

            if (lane == j) {
                L[i * 6 + j] = Lij;
            }
            __syncwarp();
        }
    }

    return true;
}

/// Warp-parallel forward substitution: L * y = b
///
/// @param L 6x6 lower triangular matrix [36]
/// @param b Right-hand side [6]
/// @param y Output solution [6]
/// @param lane Thread's lane ID
__device__ __forceinline__ void warp_forward_substitution_6(
    const double* L,
    const double* b,
    double* y,
    int lane
) {
    // y[0] = b[0] / L[0][0]
    // y[i] = (b[i] - sum_{j<i} L[i][j]*y[j]) / L[i][i]

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        double sum = 0.0;
        if (lane < i) {
            sum = L[i * 6 + lane] * y[lane];
        }

        // Reduce sum across lanes 0 to i-1
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(CHOL_WARP_MASK, sum, offset);
        }
        sum = __shfl_sync(CHOL_WARP_MASK, sum, 0);

        double yi = (b[i] - sum) / L[i * 6 + i];

        if (lane == i) {
            y[i] = yi;
        }
        // Broadcast y[i] to all lanes for next iteration
        y[i] = __shfl_sync(CHOL_WARP_MASK, yi, i);
    }
}

/// Warp-parallel backward substitution: L^T * x = y
///
/// @param L 6x6 lower triangular matrix [36] (we use its transpose)
/// @param y Right-hand side [6]
/// @param x Output solution [6]
/// @param lane Thread's lane ID
__device__ __forceinline__ void warp_backward_substitution_6(
    const double* L,
    const double* y,
    double* x,
    int lane
) {
    // x[5] = y[5] / L[5][5]
    // x[i] = (y[i] - sum_{j>i} L[j][i]*x[j]) / L[i][i]
    // Note: L^T[i][j] = L[j][i]

    #pragma unroll
    for (int i = 5; i >= 0; i--) {
        double sum = 0.0;
        if (lane > i && lane < 6) {
            // L^T[i][lane] = L[lane][i]
            sum = L[lane * 6 + i] * x[lane];
        }

        // Reduce sum across lanes i+1 to 5
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(CHOL_WARP_MASK, sum, offset);
        }
        sum = __shfl_sync(CHOL_WARP_MASK, sum, 0);

        double xi = (y[i] - sum) / L[i * 6 + i];

        if (lane == i) {
            x[i] = xi;
        }
        // Broadcast x[i] to all lanes for next iteration
        x[i] = __shfl_sync(CHOL_WARP_MASK, xi, i);
    }
}

/// Complete warp-parallel solve: H * delta = -g using Cholesky decomposition.
///
/// This solves the Newton system where H is symmetric positive definite.
/// Falls back to returning false if H is indefinite.
///
/// @param H 6x6 symmetric Hessian matrix [36] (double precision)
/// @param g 6-element gradient [6] (double precision)
/// @param delta Output: Newton step delta = -H^{-1} * g [6] (double precision)
/// @param lane Thread's lane ID (threadIdx.x % 32)
/// @return true if solve succeeded, false if Hessian is indefinite
__device__ __forceinline__ bool warp_cholesky_solve_6x6(
    const double* H,
    const double* g,
    double* delta,
    int lane
) {
    // Temporary storage in registers (each lane has its own copy)
    double L[36];
    double neg_g[6];
    double y[6];

    // Negate g for the solve: H * delta = -g
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        neg_g[i] = -g[i];
    }

    // Step 1: Cholesky decomposition H = L * L^T
    if (!warp_cholesky_6x6(H, L, lane)) {
        return false;
    }

    // Step 2: Forward substitution L * y = -g
    warp_forward_substitution_6(L, neg_g, y, lane);

    // Step 3: Backward substitution L^T * delta = y
    warp_backward_substitution_6(L, y, delta, lane);

    return true;
}

// ============================================================================
// Single-thread fallback (for comparison/debugging)
// ============================================================================

/// Single-thread Cholesky solve (fallback for indefinite matrices).
/// Uses regularization to make the matrix positive definite.
__device__ __forceinline__ void single_thread_regularized_solve_6x6(
    const double* H,
    const double* g,
    double* delta
) {
    // Add small regularization to diagonal
    double H_reg[36];
    #pragma unroll
    for (int i = 0; i < 36; i++) {
        H_reg[i] = H[i];
    }
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        H_reg[i * 6 + i] += 1e-6;  // Small regularization
    }

    // Simple Gauss-Jordan elimination (not optimal but works for fallback)
    double A[6][7];  // Augmented matrix [H | -g]
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        #pragma unroll
        for (int j = 0; j < 6; j++) {
            A[i][j] = H_reg[i * 6 + j];
        }
        A[i][6] = -g[i];
    }

    // Forward elimination with partial pivoting
    #pragma unroll
    for (int k = 0; k < 6; k++) {
        // Find pivot
        int max_row = k;
        double max_val = fabs(A[k][k]);
        for (int i = k + 1; i < 6; i++) {
            if (fabs(A[i][k]) > max_val) {
                max_val = fabs(A[i][k]);
                max_row = i;
            }
        }

        // Swap rows
        if (max_row != k) {
            for (int j = k; j < 7; j++) {
                double tmp = A[k][j];
                A[k][j] = A[max_row][j];
                A[max_row][j] = tmp;
            }
        }

        // Eliminate
        double pivot = A[k][k];
        if (fabs(pivot) < 1e-12) {
            // Singular, use steepest descent
            double g_norm_sq = 0.0;
            for (int i = 0; i < 6; i++) g_norm_sq += g[i] * g[i];
            double scale = (g_norm_sq > 1e-12) ? 0.01 / sqrt(g_norm_sq) : 0.0;
            for (int i = 0; i < 6; i++) delta[i] = -scale * g[i];
            return;
        }

        for (int j = k; j < 7; j++) {
            A[k][j] /= pivot;
        }

        for (int i = k + 1; i < 6; i++) {
            double factor = A[i][k];
            for (int j = k; j < 7; j++) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }

    // Back substitution
    for (int i = 5; i >= 0; i--) {
        delta[i] = A[i][6];
        for (int j = i + 1; j < 6; j++) {
            delta[i] -= A[i][j] * delta[j];
        }
    }
}

/// Warp-cooperative solve with fallback for indefinite Hessians.
///
/// First tries Cholesky decomposition. If that fails (indefinite matrix),
/// thread 0 performs a regularized single-thread solve and broadcasts results.
///
/// @param H 6x6 symmetric Hessian [36]
/// @param g 6-element gradient [6]
/// @param delta Output Newton step [6]
/// @param lane Thread's lane ID
/// @param solve_success Output: true if Cholesky succeeded
__device__ __forceinline__ void warp_solve_6x6_with_fallback(
    const double* H,
    const double* g,
    double* delta,
    int lane,
    bool* solve_success
) {
    // Try warp-parallel Cholesky first
    bool success = warp_cholesky_solve_6x6(H, g, delta, lane);

    // Broadcast success flag from lane 0
    int success_int = success ? 1 : 0;
    success_int = __shfl_sync(CHOL_WARP_MASK, success_int, 0);
    success = (success_int != 0);

    if (!success) {
        // Cholesky failed - use regularized single-thread solve on lane 0
        if (lane == 0) {
            single_thread_regularized_solve_6x6(H, g, delta);
        }
        __syncwarp();

        // Broadcast delta from lane 0 to all lanes
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            delta[i] = __shfl_sync(CHOL_WARP_MASK, delta[i], 0);
        }
    }

    *solve_success = success;
}
