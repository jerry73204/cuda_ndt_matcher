// cuSOLVER batched Cholesky solver wrapper for Rust FFI
//
// Provides batched 6×6 Cholesky factorization and solve for Newton optimization.
// Used for Phase 16: GPU Initial Pose Pipeline.

#include <cuda_runtime.h>
#include <cusolverDn.h>

extern "C" {

// Error codes
typedef int CudaError;
typedef int CusolverError;

/// Create a cuSOLVER dense handle.
///
/// # Arguments
/// * `handle` - Output: cuSOLVER handle
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_create_handle(cusolverDnHandle_t* handle) {
    return cusolverDnCreate(handle);
}

/// Destroy a cuSOLVER dense handle.
///
/// # Arguments
/// * `handle` - cuSOLVER handle to destroy
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_destroy_handle(cusolverDnHandle_t handle) {
    return cusolverDnDestroy(handle);
}

/// Batched Cholesky factorization for K 6×6 positive-definite matrices.
///
/// Computes A = L * L^T for K matrices simultaneously.
/// The matrices are modified in-place with L stored in the lower triangle.
///
/// # Arguments
/// * `handle` - cuSOLVER handle
/// * `batch_size` - Number of matrices (K)
/// * `d_A_array` - Device array of K pointers to 6×6 matrices (column-major, in-place)
/// * `d_info` - Device array of K status codes (0 = success, >0 = leading minor not PD)
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_batched_potrf_f64(
    cusolverDnHandle_t handle,
    int batch_size,
    double** d_A_array,
    int* d_info
) {
    const int n = 6;  // 6×6 matrices for NDT pose optimization
    const int lda = 6;

    return cusolverDnDpotrfBatched(
        handle,
        CUBLAS_FILL_MODE_LOWER,  // Use lower triangle
        n,
        d_A_array,
        lda,
        d_info,
        batch_size
    );
}

/// Batched triangular solve after Cholesky factorization.
///
/// Solves L * L^T * x = b for K systems using pre-factored matrices.
/// The right-hand side b is modified in-place with the solution x.
///
/// # Arguments
/// * `handle` - cuSOLVER handle
/// * `batch_size` - Number of systems (K)
/// * `d_A_array` - Device array of K pointers to factored 6×6 matrices (from potrf)
/// * `d_B_array` - Device array of K pointers to 6-element RHS vectors (in-place solution)
/// * `d_info` - Device array of K status codes (0 = success)
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_batched_potrs_f64(
    cusolverDnHandle_t handle,
    int batch_size,
    double** d_A_array,
    double** d_B_array,
    int* d_info
) {
    const int n = 6;
    const int nrhs = 1;  // One right-hand side per system
    const int lda = 6;
    const int ldb = 6;

    return cusolverDnDpotrsBatched(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,
        d_A_array,
        lda,
        d_B_array,
        ldb,
        d_info,
        batch_size
    );
}

/// Combined batched Cholesky factorization and solve.
///
/// Convenience function that performs:
/// 1. A = L * L^T (factorization)
/// 2. L * L^T * x = b (solve)
///
/// Both A and b are modified in-place.
///
/// # Arguments
/// * `handle` - cuSOLVER handle
/// * `batch_size` - Number of systems (K)
/// * `d_A_array` - Device array of K pointers to 6×6 Hessian matrices (in-place factored)
/// * `d_B_array` - Device array of K pointers to 6-element gradient vectors (in-place solved)
/// * `d_info` - Device array of K status codes
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_batched_cholesky_solve_f64(
    cusolverDnHandle_t handle,
    int batch_size,
    double** d_A_array,
    double** d_B_array,
    int* d_info
) {
    // Step 1: Factorization
    CusolverError status = cusolver_batched_potrf_f64(handle, batch_size, d_A_array, d_info);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }

    // Synchronize to ensure factorization is complete before solve
    cudaDeviceSynchronize();

    // Step 2: Solve
    return cusolver_batched_potrs_f64(handle, batch_size, d_A_array, d_B_array, d_info);
}

/// Single matrix Cholesky factorization (6×6 f64).
///
/// For non-batched cases or when batch_size == 1.
///
/// # Arguments
/// * `handle` - cuSOLVER handle
/// * `d_A` - Device pointer to 6×6 matrix (column-major, in-place)
/// * `d_workspace` - Device workspace (size from cusolver_potrf_workspace_size)
/// * `workspace_size` - Size of workspace in bytes
/// * `d_info` - Device pointer to status code
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_potrf_f64(
    cusolverDnHandle_t handle,
    double* d_A,
    double* d_workspace,
    int workspace_size,
    int* d_info
) {
    const int n = 6;
    const int lda = 6;

    return cusolverDnDpotrf(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,
        lda,
        d_workspace,
        workspace_size,
        d_info
    );
}

/// Query workspace size for single matrix Cholesky.
///
/// # Arguments
/// * `handle` - cuSOLVER handle
/// * `workspace_size` - Output: required workspace size in elements (not bytes)
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_potrf_workspace_size_f64(
    cusolverDnHandle_t handle,
    int* workspace_size
) {
    const int n = 6;
    const int lda = 6;

    return cusolverDnDpotrf_bufferSize(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        (double*)nullptr,
        lda,
        workspace_size
    );
}

/// Single matrix triangular solve (6×6 f64).
///
/// # Arguments
/// * `handle` - cuSOLVER handle
/// * `d_A` - Device pointer to factored 6×6 matrix
/// * `d_B` - Device pointer to 6-element RHS (in-place solution)
/// * `d_info` - Device pointer to status code
///
/// # Returns
/// CUSOLVER_STATUS_SUCCESS (0) on success.
CusolverError cusolver_potrs_f64(
    cusolverDnHandle_t handle,
    double* d_A,
    double* d_B,
    int* d_info
) {
    const int n = 6;
    const int nrhs = 1;
    const int lda = 6;
    const int ldb = 6;

    return cusolverDnDpotrs(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,
        d_A,
        lda,
        d_B,
        ldb,
        d_info
    );
}

} // extern "C"
