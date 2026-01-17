// 6x6 Jacobi SVD solver in registers for GPU
//
// This implements Jacobi eigendecomposition for 6x6 symmetric matrices
// entirely in registers. Unlike Cholesky, this handles indefinite matrices
// natively, which is critical for NDT Hessians that may have mixed eigenvalue
// signs far from the optimum.
//
// Algorithm: Classical Jacobi eigendecomposition
// For symmetric H, compute H = V * D * V^T where D is diagonal (eigenvalues)
// and V is orthogonal (eigenvectors). Then solve H*x = -g using pseudo-inverse.
//
// Reference: https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm

#ifndef JACOBI_SVD_6X6_CUH
#define JACOBI_SVD_6X6_CUH

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// Constants
// ============================================================================

// Convergence threshold for off-diagonal elements
#define JACOBI_CONV_TOL 1e-14

// Minimum eigenvalue threshold for pseudo-inverse (relative to max eigenvalue)
#define JACOBI_PINV_TOL 1e-10

// Maximum number of Jacobi sweeps (typically converges in 5-10)
#define JACOBI_MAX_SWEEPS 30

// Threshold for skipping tiny off-diagonal elements
#define JACOBI_ZERO_TOL 1e-15

// ============================================================================
// Helper: Apply Jacobi rotation to symmetric matrix A
// ============================================================================

/// Apply Jacobi rotation J(p,q,theta) to symmetric matrix A.
/// This zeros out A[p,q] and A[q,p].
///
/// The rotation is: A' = J^T * A * J
/// For symmetric A, we only update the affected elements.
///
/// @param A   6x6 symmetric matrix, row-major [36 doubles]. Modified in place.
/// @param p   First index (p < q)
/// @param q   Second index (q > p)
/// @param c   cos(theta)
/// @param s   sin(theta)
__device__ __forceinline__ void jacobi_rotate_A(
    double* A,
    int p, int q,
    double c, double s
) {
    // Compute rotated diagonal elements
    double App = A[p * 6 + p];
    double Aqq = A[q * 6 + q];
    double Apq = A[p * 6 + q];

    // New diagonal elements (Apq' becomes 0 by construction)
    double c2 = c * c;
    double s2 = s * s;
    double cs = c * s;

    A[p * 6 + p] = c2 * App + s2 * Aqq - 2.0 * cs * Apq;
    A[q * 6 + q] = s2 * App + c2 * Aqq + 2.0 * cs * Apq;
    A[p * 6 + q] = 0.0;  // Zeroed by design
    A[q * 6 + p] = 0.0;

    // Update off-diagonal elements in rows/columns p and q
    // For k != p, q:
    //   A'[p,k] = c*A[p,k] - s*A[q,k]
    //   A'[q,k] = s*A[p,k] + c*A[q,k]
    for (int k = 0; k < 6; k++) {
        if (k == p || k == q) continue;

        double Apk = A[p * 6 + k];
        double Aqk = A[q * 6 + k];

        double Apk_new = c * Apk - s * Aqk;
        double Aqk_new = s * Apk + c * Aqk;

        // Update both (row, col) since A is symmetric
        A[p * 6 + k] = Apk_new;
        A[k * 6 + p] = Apk_new;
        A[q * 6 + k] = Aqk_new;
        A[k * 6 + q] = Aqk_new;
    }
}

// ============================================================================
// Helper: Accumulate Jacobi rotation in eigenvector matrix V
// ============================================================================

/// Accumulate Jacobi rotation J(p,q,theta) in eigenvector matrix V.
/// V' = V * J (rotation applied from the right)
///
/// @param V   6x6 eigenvector matrix, row-major [36 doubles]. Modified in place.
/// @param p   First index
/// @param q   Second index
/// @param c   cos(theta)
/// @param s   sin(theta)
__device__ __forceinline__ void jacobi_rotate_V(
    double* V,
    int p, int q,
    double c, double s
) {
    // V' = V * J means:
    // V'[i,p] = c*V[i,p] - s*V[i,q]
    // V'[i,q] = s*V[i,p] + c*V[i,q]
    for (int i = 0; i < 6; i++) {
        double Vip = V[i * 6 + p];
        double Viq = V[i * 6 + q];

        V[i * 6 + p] = c * Vip - s * Viq;
        V[i * 6 + q] = s * Vip + c * Viq;
    }
}

// ============================================================================
// Main solver: Jacobi SVD for 6x6 symmetric matrices
// ============================================================================

/// Solve H * x = -g using Jacobi eigendecomposition.
///
/// Unlike Cholesky, this handles indefinite and near-singular matrices.
/// The pseudo-inverse is computed as: x = V * D^+ * V^T * (-g)
/// where D^+[i] = 1/D[i] if |D[i]| > threshold, else 0.
///
/// @param H_orig   6x6 Hessian matrix, row-major [36 doubles]. NOT modified.
/// @param g_orig   6-element gradient vector [6 doubles]. NOT modified.
/// @param x_out    6-element solution output.
/// @param success  Output: true unless all eigenvalues are below threshold.
__device__ __forceinline__ void jacobi_svd_solve_6x6_f64(
    const double* H_orig,
    const double* g_orig,
    double* x_out,
    bool* success
) {
    // Working matrices in registers
    double A[36];  // Working copy of H, becomes diagonal
    double V[36];  // Eigenvector accumulator, starts as identity

    // Initialize A = H_orig
    for (int i = 0; i < 36; i++) {
        A[i] = H_orig[i];
    }

    // Initialize V = I (identity)
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            V[i * 6 + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // ========================================================================
    // Jacobi sweeps: rotate pairs (p,q) to zero off-diagonal elements
    // ========================================================================

    for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
        // Check convergence: ||off-diagonal||^2 < tol^2 * ||diagonal||^2
        double off_diag_sq = 0.0;
        double diag_sq = 0.0;

        for (int i = 0; i < 6; i++) {
            diag_sq += A[i * 6 + i] * A[i * 6 + i];
            for (int j = i + 1; j < 6; j++) {
                off_diag_sq += A[i * 6 + j] * A[i * 6 + j];
            }
        }

        // Note: off_diag_sq counts upper triangle only, multiply by 2 for full
        off_diag_sq *= 2.0;

        if (off_diag_sq < JACOBI_CONV_TOL * JACOBI_CONV_TOL * diag_sq) {
            break;  // Converged
        }

        // Sweep through all 15 pairs (p,q) where p < q
        for (int p = 0; p < 5; p++) {
            for (int q = p + 1; q < 6; q++) {
                double Apq = A[p * 6 + q];

                // Skip if already essentially zero
                if (fabs(Apq) < JACOBI_ZERO_TOL) continue;

                // Compute Jacobi rotation angle
                // tau = (A[q,q] - A[p,p]) / (2 * A[p,q])
                double App = A[p * 6 + p];
                double Aqq = A[q * 6 + q];
                double tau = (Aqq - App) / (2.0 * Apq);

                // Numerically stable formula for t = tan(theta)
                // t = sign(tau) / (|tau| + sqrt(tau^2 + 1))
                double t;
                if (tau >= 0.0) {
                    t = 1.0 / (tau + sqrt(tau * tau + 1.0));
                } else {
                    t = -1.0 / (-tau + sqrt(tau * tau + 1.0));
                }

                // c = cos(theta), s = sin(theta)
                double c = 1.0 / sqrt(t * t + 1.0);
                double s = t * c;

                // Apply rotation to A and V
                jacobi_rotate_A(A, p, q, c, s);
                jacobi_rotate_V(V, p, q, c, s);
            }
        }
    }

    // ========================================================================
    // Solve using pseudo-inverse: x = V * D^+ * V^T * (-g)
    // ========================================================================

    // Eigenvalues are now on diagonal of A
    double eigenvalues[6];
    double max_abs_eig = 0.0;

    for (int i = 0; i < 6; i++) {
        eigenvalues[i] = A[i * 6 + i];
        double abs_eig = fabs(eigenvalues[i]);
        if (abs_eig > max_abs_eig) max_abs_eig = abs_eig;
    }

    // Threshold for pseudo-inverse
    double threshold = JACOBI_PINV_TOL * max_abs_eig;
    if (threshold < 1e-15) threshold = 1e-15;  // Minimum threshold

    // Step 1: y = V^T * (-g)
    double y[6];
    for (int i = 0; i < 6; i++) {
        double sum = 0.0;
        for (int j = 0; j < 6; j++) {
            sum += V[j * 6 + i] * (-g_orig[j]);  // V^T[i,j] = V[j,i]
        }
        y[i] = sum;
    }

    // Step 2: z[i] = y[i] / |eigenvalue[i]| (pseudo-inverse like SVD)
    // NOTE: We use ABSOLUTE VALUE of eigenvalues to match Autoware's JacobiSVD.
    // For symmetric H with eigendecomposition H = V*D*V^T, SVD gives H = V*|D|*V^T.
    //
    // For NDT score MAXIMIZATION, this may give the wrong direction when H is
    // negative definite (normal at a maximum). The caller must check direction
    // and reverse if necessary (matching Autoware's behavior).
    double z[6];
    int num_nonzero = 0;
    for (int i = 0; i < 6; i++) {
        double abs_eig = fabs(eigenvalues[i]);
        if (abs_eig > threshold) {
            z[i] = y[i] / abs_eig;  // Use |eigenvalue| for pseudo-inverse (like SVD)
            num_nonzero++;
        } else {
            z[i] = 0.0;  // Zero out directions with tiny eigenvalues
        }
    }

    // Step 3: x = V * z
    for (int i = 0; i < 6; i++) {
        double sum = 0.0;
        for (int j = 0; j < 6; j++) {
            sum += V[i * 6 + j] * z[j];
        }
        x_out[i] = sum;
    }

    // Success if at least one eigenvalue was usable
    *success = (num_nonzero > 0);
}

#endif // JACOBI_SVD_6X6_CUH
