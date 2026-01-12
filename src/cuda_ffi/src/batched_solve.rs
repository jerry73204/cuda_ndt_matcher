//! Rust bindings for cuSOLVER batched Cholesky solver.
//!
//! Provides GPU-accelerated batched linear system solving for Newton optimization.
//! Used in Phase 16 for solving K 6×6 systems simultaneously.

#![allow(non_snake_case)] // Follow CUDA naming conventions

use std::ffi::c_int;
use std::ptr;

/// cuSOLVER dense handle (opaque pointer).
pub type CusolverDnHandle = *mut std::ffi::c_void;

extern "C" {
    fn cusolver_create_handle(handle: *mut CusolverDnHandle) -> c_int;
    fn cusolver_destroy_handle(handle: CusolverDnHandle) -> c_int;

    fn cusolver_batched_potrf_f64(
        handle: CusolverDnHandle,
        batch_size: c_int,
        d_A_array: *mut *mut f64,
        d_info: *mut c_int,
    ) -> c_int;

    fn cusolver_batched_potrs_f64(
        handle: CusolverDnHandle,
        batch_size: c_int,
        d_A_array: *mut *mut f64,
        d_B_array: *mut *mut f64,
        d_info: *mut c_int,
    ) -> c_int;

    fn cusolver_batched_cholesky_solve_f64(
        handle: CusolverDnHandle,
        batch_size: c_int,
        d_A_array: *mut *mut f64,
        d_B_array: *mut *mut f64,
        d_info: *mut c_int,
    ) -> c_int;

    fn cusolver_potrf_f64(
        handle: CusolverDnHandle,
        d_A: *mut f64,
        d_workspace: *mut f64,
        workspace_size: c_int,
        d_info: *mut c_int,
    ) -> c_int;

    fn cusolver_potrf_workspace_size_f64(
        handle: CusolverDnHandle,
        workspace_size: *mut c_int,
    ) -> c_int;

    fn cusolver_potrs_f64(
        handle: CusolverDnHandle,
        d_A: *mut f64,
        d_B: *mut f64,
        d_info: *mut c_int,
    ) -> c_int;
}

/// Error type for cuSOLVER operations.
#[derive(Debug, Clone, Copy)]
pub struct CusolverError(c_int);

impl CusolverError {
    /// Check if this is a success status.
    pub fn is_success(&self) -> bool {
        self.0 == 0
    }

    /// Get the underlying error code.
    pub fn code(&self) -> c_int {
        self.0
    }
}

impl std::fmt::Display for CusolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuSOLVER error: {}", self.0)
    }
}

impl std::error::Error for CusolverError {}

/// Batched Cholesky solver for 6×6 positive-definite systems.
///
/// Solves K systems of the form H × δ = -g simultaneously on GPU.
///
/// # Example
///
/// ```ignore
/// let solver = BatchedCholeskySolver::new()?;
///
/// // Solve batch of K systems
/// solver.solve_batch_inplace(
///     d_hessians_ptr_array,  // K pointers to 6×6 Hessians
///     d_gradients_ptr_array, // K pointers to 6-element gradients
///     batch_size,
///     d_info,
/// )?;
/// ```
pub struct BatchedCholeskySolver {
    handle: CusolverDnHandle,
}

impl BatchedCholeskySolver {
    /// Create a new batched Cholesky solver.
    pub fn new() -> Result<Self, CusolverError> {
        let mut handle: CusolverDnHandle = ptr::null_mut();
        let status = unsafe { cusolver_create_handle(&mut handle) };
        if status != 0 {
            return Err(CusolverError(status));
        }
        Ok(Self { handle })
    }

    /// Get the raw cuSOLVER handle for interop.
    pub fn handle(&self) -> CusolverDnHandle {
        self.handle
    }

    /// Perform batched Cholesky factorization.
    ///
    /// Computes A = L × L^T for K matrices.
    /// The matrices are modified in-place.
    ///
    /// # Safety
    ///
    /// - `d_A_array` must be a valid device pointer to K device pointers
    /// - Each pointer in d_A_array must point to 36 f64 values (6×6 column-major)
    /// - `d_info` must be a valid device pointer to K c_int values
    pub unsafe fn factorize_batch(
        &self,
        d_A_array: u64,
        batch_size: usize,
        d_info: u64,
    ) -> Result<(), CusolverError> {
        let status = cusolver_batched_potrf_f64(
            self.handle,
            batch_size as c_int,
            d_A_array as *mut *mut f64,
            d_info as *mut c_int,
        );
        if status != 0 {
            return Err(CusolverError(status));
        }
        Ok(())
    }

    /// Perform batched triangular solve after factorization.
    ///
    /// Solves L × L^T × x = b for K systems.
    /// The right-hand sides are modified in-place with solutions.
    ///
    /// # Safety
    ///
    /// - `d_A_array` must contain factored matrices from `factorize_batch`
    /// - `d_B_array` must be a valid device pointer to K device pointers
    /// - Each pointer in d_B_array must point to 6 f64 values
    /// - `d_info` must be a valid device pointer to K c_int values
    pub unsafe fn solve_batch(
        &self,
        d_A_array: u64,
        d_B_array: u64,
        batch_size: usize,
        d_info: u64,
    ) -> Result<(), CusolverError> {
        let status = cusolver_batched_potrs_f64(
            self.handle,
            batch_size as c_int,
            d_A_array as *mut *mut f64,
            d_B_array as *mut *mut f64,
            d_info as *mut c_int,
        );
        if status != 0 {
            return Err(CusolverError(status));
        }
        Ok(())
    }

    /// Perform combined batched Cholesky factorization and solve.
    ///
    /// Solves H × δ = b for K systems:
    /// 1. Factorize: H = L × L^T
    /// 2. Solve: L × L^T × δ = b
    ///
    /// Both H and b are modified in-place.
    ///
    /// # Safety
    ///
    /// - `d_A_array` must be a valid device pointer to K device pointers
    /// - Each pointer in d_A_array must point to 36 f64 values (6×6 column-major)
    /// - `d_B_array` must be a valid device pointer to K device pointers
    /// - Each pointer in d_B_array must point to 6 f64 values
    /// - `d_info` must be a valid device pointer to K c_int values
    pub unsafe fn solve_batch_inplace(
        &self,
        d_A_array: u64,
        d_B_array: u64,
        batch_size: usize,
        d_info: u64,
    ) -> Result<(), CusolverError> {
        let status = cusolver_batched_cholesky_solve_f64(
            self.handle,
            batch_size as c_int,
            d_A_array as *mut *mut f64,
            d_B_array as *mut *mut f64,
            d_info as *mut c_int,
        );
        if status != 0 {
            return Err(CusolverError(status));
        }
        Ok(())
    }

    /// Single matrix Cholesky factorization.
    ///
    /// For non-batched cases or batch_size == 1.
    ///
    /// # Safety
    ///
    /// - `d_A` must be a valid device pointer to 36 f64 values
    /// - `d_workspace` must be a valid device pointer with sufficient space
    /// - `d_info` must be a valid device pointer to 1 c_int value
    pub unsafe fn factorize_single(
        &self,
        d_A: u64,
        d_workspace: u64,
        workspace_size: usize,
        d_info: u64,
    ) -> Result<(), CusolverError> {
        let status = cusolver_potrf_f64(
            self.handle,
            d_A as *mut f64,
            d_workspace as *mut f64,
            workspace_size as c_int,
            d_info as *mut c_int,
        );
        if status != 0 {
            return Err(CusolverError(status));
        }
        Ok(())
    }

    /// Query workspace size for single matrix factorization.
    pub fn factorize_workspace_size(&self) -> Result<usize, CusolverError> {
        let mut size: c_int = 0;
        let status = unsafe { cusolver_potrf_workspace_size_f64(self.handle, &mut size) };
        if status != 0 {
            return Err(CusolverError(status));
        }
        Ok(size as usize)
    }

    /// Single matrix triangular solve.
    ///
    /// # Safety
    ///
    /// - `d_A` must contain a factored matrix from `factorize_single`
    /// - `d_B` must be a valid device pointer to 6 f64 values
    /// - `d_info` must be a valid device pointer to 1 c_int value
    pub unsafe fn solve_single(
        &self,
        d_A: u64,
        d_B: u64,
        d_info: u64,
    ) -> Result<(), CusolverError> {
        let status = cusolver_potrs_f64(
            self.handle,
            d_A as *mut f64,
            d_B as *mut f64,
            d_info as *mut c_int,
        );
        if status != 0 {
            return Err(CusolverError(status));
        }
        Ok(())
    }
}

impl Drop for BatchedCholeskySolver {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                cusolver_destroy_handle(self.handle);
            }
        }
    }
}

// Safety: cuSOLVER handle is thread-safe when not being actively used
unsafe impl Send for BatchedCholeskySolver {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = BatchedCholeskySolver::new();
        assert!(solver.is_ok(), "Failed to create batched solver");
    }

    #[test]
    fn test_workspace_size_query() {
        let solver = BatchedCholeskySolver::new().expect("Failed to create solver");
        let size = solver.factorize_workspace_size();
        assert!(size.is_ok(), "Failed to query workspace size");
        let size = size.unwrap();
        assert!(size > 0, "Workspace size should be positive");
    }
}
