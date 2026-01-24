//! GPU-accelerated Newton solver using cuSOLVER.
//!
//! This module provides a 6×6 linear system solver for the Newton step:
//!   H·δ = -g
//! where H is the Hessian (6×6) and g is the gradient (6×1).
//!
//! Uses Cholesky factorization (cusolverDnDpotrf/Dpotrs) for positive definite
//! Hessian matrices, with LU factorization (cusolverDnDgetrf/Dgetrs) as fallback.

use std::sync::Arc;

use cudarc::cusolver::safe::DnHandle;
use cudarc::cusolver::sys as cusolver_sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use thiserror::Error;

/// Errors from GPU Newton solver operations.
#[derive(Error, Debug)]
pub enum GpuNewtonError {
    #[error("cuSOLVER error: {0:?}")]
    CusolverError(cusolver_sys::cusolverStatus_t),

    #[error("CUDA driver error: {0}")]
    CudaError(#[from] cudarc::driver::DriverError),

    #[error("Matrix factorization failed (info={0})")]
    FactorizationFailed(i32),

    #[error("Matrix is singular or not positive definite")]
    SingularMatrix,
}

impl From<cudarc::cusolver::result::CusolverError> for GpuNewtonError {
    fn from(e: cudarc::cusolver::result::CusolverError) -> Self {
        GpuNewtonError::CusolverError(e.0)
    }
}

/// GPU Newton solver for 6×6 linear systems.
///
/// Pre-allocates all GPU buffers and cuSOLVER workspace to avoid
/// per-iteration allocations.
pub struct GpuNewtonSolver {
    /// cuSOLVER dense handle
    handle: DnHandle,

    /// CUDA context for memory operations
    ctx: Arc<CudaContext>,

    /// CUDA stream
    stream: Arc<CudaStream>,

    /// Pre-allocated Hessian matrix buffer [36] (6×6, column-major)
    d_hessian: CudaSlice<f64>,

    /// Pre-allocated gradient/solution buffer [6]
    d_rhs: CudaSlice<f64>,

    /// Pivot indices for LU factorization [6]
    d_ipiv: CudaSlice<i32>,

    /// cuSOLVER info output [1]
    d_info: CudaSlice<i32>,

    /// cuSOLVER workspace
    d_workspace: CudaSlice<f64>,

    /// Workspace size (in elements)
    workspace_size: usize,
}

impl GpuNewtonSolver {
    /// Create a new GPU Newton solver.
    ///
    /// Initializes cuSOLVER handle and pre-allocates all GPU buffers.
    pub fn new(device_id: usize) -> Result<Self, GpuNewtonError> {
        // Create CUDA context and stream
        let ctx = CudaContext::new(device_id)?;
        let stream = ctx.default_stream();

        // Create cuSOLVER handle
        let handle = DnHandle::new(stream.clone())?;

        // Query workspace size for 6×6 Cholesky (larger of potrf and getrf)
        let potrf_workspace = Self::query_potrf_workspace_size(&handle)?;
        let getrf_workspace = Self::query_getrf_workspace_size(&handle)?;
        let workspace_size = potrf_workspace.max(getrf_workspace);

        // Allocate device buffers
        let d_hessian = stream.alloc_zeros::<f64>(36)?; // 6×6
        let d_rhs = stream.alloc_zeros::<f64>(6)?;
        let d_ipiv = stream.alloc_zeros::<i32>(6)?;
        let d_info = stream.alloc_zeros::<i32>(1)?;
        let d_workspace = stream.alloc_zeros::<f64>(workspace_size.max(1))?;

        Ok(Self {
            handle,
            ctx,
            stream,
            d_hessian,
            d_rhs,
            d_ipiv,
            d_info,
            d_workspace,
            workspace_size,
        })
    }

    /// Query workspace size for Cholesky factorization.
    fn query_potrf_workspace_size(handle: &DnHandle) -> Result<usize, GpuNewtonError> {
        let mut lwork: i32 = 0;
        let status = unsafe {
            cusolver_sys::cusolverDnDpotrf_bufferSize(
                handle.cu(),
                cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                6,                    // n
                std::ptr::null_mut(), // A (not used for size query)
                6,                    // lda
                &mut lwork,
            )
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }
        Ok(lwork as usize)
    }

    /// Query workspace size for LU factorization.
    fn query_getrf_workspace_size(handle: &DnHandle) -> Result<usize, GpuNewtonError> {
        let mut lwork: i32 = 0;
        let status = unsafe {
            cusolver_sys::cusolverDnDgetrf_bufferSize(
                handle.cu(),
                6,                    // m
                6,                    // n
                std::ptr::null_mut(), // A (not used for size query)
                6,                    // lda
                &mut lwork,
            )
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }
        Ok(lwork as usize)
    }

    /// Solve H·δ = -g on GPU using Cholesky factorization.
    ///
    /// # Arguments
    /// * `hessian` - 6×6 Hessian matrix (row-major on host, will be converted to column-major)
    /// * `gradient` - 6-element gradient vector
    ///
    /// # Returns
    /// Solution vector δ = -H⁻¹g, or error if matrix is singular.
    pub fn solve(
        &mut self,
        hessian: &[f64; 36],
        gradient: &[f64; 6],
    ) -> Result<[f64; 6], GpuNewtonError> {
        // Convert row-major hessian to column-major for cuSOLVER
        let hessian_col_major = Self::row_to_col_major_6x6(hessian);

        // Compute -gradient (RHS = -g)
        let neg_gradient: [f64; 6] = [
            -gradient[0],
            -gradient[1],
            -gradient[2],
            -gradient[3],
            -gradient[4],
            -gradient[5],
        ];

        // Upload to GPU
        self.stream
            .memcpy_htod(&hessian_col_major, &mut self.d_hessian)?;
        self.stream.memcpy_htod(&neg_gradient, &mut self.d_rhs)?;

        // Try Cholesky first (for positive definite matrices)
        let cholesky_result = self.solve_cholesky();

        if cholesky_result.is_ok() {
            return cholesky_result;
        }

        // Fallback to LU factorization
        // Re-upload Hessian (Cholesky modified it)
        self.stream
            .memcpy_htod(&hessian_col_major, &mut self.d_hessian)?;
        self.stream.memcpy_htod(&neg_gradient, &mut self.d_rhs)?;

        self.solve_lu()
    }

    /// Solve using Cholesky factorization (for positive definite Hessian).
    fn solve_cholesky(&mut self) -> Result<[f64; 6], GpuNewtonError> {
        // Get device pointers with proper synchronization
        let (hessian_ptr, _hessian_guard) = self.d_hessian.device_ptr_mut(&self.stream);
        let (workspace_ptr, _workspace_guard) = self.d_workspace.device_ptr_mut(&self.stream);
        let (info_ptr, _info_guard) = self.d_info.device_ptr_mut(&self.stream);

        // Cholesky factorization: A = U^T * U
        let status = unsafe {
            cusolver_sys::cusolverDnDpotrf(
                self.handle.cu(),
                cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                6, // n
                hessian_ptr as *mut f64,
                6, // lda
                workspace_ptr as *mut f64,
                self.workspace_size as i32,
                info_ptr as *mut i32,
            )
        };

        // Drop guards to ensure operations are recorded
        drop(_hessian_guard);
        drop(_workspace_guard);
        drop(_info_guard);

        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }

        // Check factorization result
        let mut info = [0i32];
        self.stream.memcpy_dtoh(&self.d_info, &mut info)?;
        self.stream.synchronize()?;

        if info[0] != 0 {
            // Matrix is not positive definite
            return Err(GpuNewtonError::FactorizationFailed(info[0]));
        }

        // Get pointers for solve step
        let (hessian_ptr, _hessian_guard) = self.d_hessian.device_ptr(&self.stream);
        let (rhs_ptr, _rhs_guard) = self.d_rhs.device_ptr_mut(&self.stream);
        let (info_ptr, _info_guard) = self.d_info.device_ptr_mut(&self.stream);

        // Solve using Cholesky factors
        let status = unsafe {
            cusolver_sys::cusolverDnDpotrs(
                self.handle.cu(),
                cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                6, // n
                1, // nrhs
                hessian_ptr as *const f64,
                6, // lda
                rhs_ptr as *mut f64,
                6, // ldb
                info_ptr as *mut i32,
            )
        };

        drop(_hessian_guard);
        drop(_rhs_guard);
        drop(_info_guard);

        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }

        // Download solution
        let mut solution = [0.0f64; 6];
        self.stream.memcpy_dtoh(&self.d_rhs, &mut solution)?;
        self.stream.synchronize()?;

        Ok(solution)
    }

    /// Solve using LU factorization (general fallback).
    fn solve_lu(&mut self) -> Result<[f64; 6], GpuNewtonError> {
        // Get device pointers with proper synchronization
        let (hessian_ptr, _hessian_guard) = self.d_hessian.device_ptr_mut(&self.stream);
        let (workspace_ptr, _workspace_guard) = self.d_workspace.device_ptr_mut(&self.stream);
        let (ipiv_ptr, _ipiv_guard) = self.d_ipiv.device_ptr_mut(&self.stream);
        let (info_ptr, _info_guard) = self.d_info.device_ptr_mut(&self.stream);

        // LU factorization: A = P * L * U
        let status = unsafe {
            cusolver_sys::cusolverDnDgetrf(
                self.handle.cu(),
                6, // m
                6, // n
                hessian_ptr as *mut f64,
                6, // lda
                workspace_ptr as *mut f64,
                ipiv_ptr as *mut i32,
                info_ptr as *mut i32,
            )
        };

        drop(_hessian_guard);
        drop(_workspace_guard);
        drop(_ipiv_guard);
        drop(_info_guard);

        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }

        // Check factorization result
        let mut info = [0i32];
        self.stream.memcpy_dtoh(&self.d_info, &mut info)?;
        self.stream.synchronize()?;

        if info[0] != 0 {
            return Err(GpuNewtonError::FactorizationFailed(info[0]));
        }

        // Get pointers for solve step
        let (hessian_ptr, _hessian_guard) = self.d_hessian.device_ptr(&self.stream);
        let (ipiv_ptr, _ipiv_guard) = self.d_ipiv.device_ptr(&self.stream);
        let (rhs_ptr, _rhs_guard) = self.d_rhs.device_ptr_mut(&self.stream);
        let (info_ptr, _info_guard) = self.d_info.device_ptr_mut(&self.stream);

        // Solve using LU factors
        let status = unsafe {
            cusolver_sys::cusolverDnDgetrs(
                self.handle.cu(),
                cusolver_sys::cublasOperation_t::CUBLAS_OP_N,
                6, // n
                1, // nrhs
                hessian_ptr as *const f64,
                6, // lda
                ipiv_ptr as *const i32,
                rhs_ptr as *mut f64,
                6, // ldb
                info_ptr as *mut i32,
            )
        };

        drop(_hessian_guard);
        drop(_ipiv_guard);
        drop(_rhs_guard);
        drop(_info_guard);

        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }

        // Download solution
        let mut solution = [0.0f64; 6];
        self.stream.memcpy_dtoh(&self.d_rhs, &mut solution)?;
        self.stream.synchronize()?;

        Ok(solution)
    }

    /// Solve H·δ = -g entirely on GPU with result staying on GPU.
    ///
    /// This is the Phase 15.3 implementation for zero-transfer Newton iterations.
    /// Unlike `solve()`, this method:
    /// - Takes GPU device slices directly (no host->device copy)
    /// - Leaves the result on GPU (no device->host copy)
    ///
    /// # Arguments
    /// * `hessian` - 6×6 Hessian matrix (column-major, already on GPU)
    /// * `gradient` - 6-element gradient vector (on GPU)
    /// * `delta` - 6-element output buffer (on GPU, will contain -H⁻¹g)
    ///
    /// # Note
    /// The hessian buffer is modified during factorization.
    /// The gradient buffer is NOT modified; its negation is copied to delta first.
    pub fn solve_inplace(
        &mut self,
        hessian: &mut CudaSlice<f64>,
        gradient: &CudaSlice<f64>,
        delta: &mut CudaSlice<f64>,
    ) -> Result<(), GpuNewtonError> {
        // Copy -gradient to delta (b = -g)
        self.negate_and_copy(gradient, delta)?;

        // Try Cholesky first (for positive definite matrices)
        let cholesky_result = self.solve_cholesky_inplace(hessian, delta);

        if cholesky_result.is_ok() {
            return Ok(());
        }

        // Cholesky failed - Hessian was modified, so caller must ensure we can
        // still use LU. For now, we'll return an error since we can't restore
        // the original Hessian without a copy.
        // In practice, NDT Hessians should be positive definite near convergence.
        Err(GpuNewtonError::SingularMatrix)
    }

    /// Copy -src to dst on GPU.
    fn negate_and_copy(
        &self,
        src: &CudaSlice<f64>,
        dst: &mut CudaSlice<f64>,
    ) -> Result<(), GpuNewtonError> {
        // Download, negate, upload (simple but adds latency)
        // TODO: Replace with a simple CUDA kernel for better performance
        let mut host = [0.0f64; 6];
        self.stream.memcpy_dtoh(src, &mut host)?;
        for v in &mut host {
            *v = -*v;
        }
        self.stream.memcpy_htod(&host, dst)?;
        Ok(())
    }

    /// Solve using Cholesky factorization with GPU-resident buffers.
    fn solve_cholesky_inplace(
        &mut self,
        hessian: &mut CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
    ) -> Result<(), GpuNewtonError> {
        // Get device pointers with proper synchronization
        let (hessian_ptr, _hessian_guard) = hessian.device_ptr_mut(&self.stream);
        let (workspace_ptr, _workspace_guard) = self.d_workspace.device_ptr_mut(&self.stream);
        let (info_ptr, _info_guard) = self.d_info.device_ptr_mut(&self.stream);

        // Cholesky factorization: A = U^T * U
        let status = unsafe {
            cusolver_sys::cusolverDnDpotrf(
                self.handle.cu(),
                cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                6, // n
                hessian_ptr as *mut f64,
                6, // lda
                workspace_ptr as *mut f64,
                self.workspace_size as i32,
                info_ptr as *mut i32,
            )
        };

        // Drop guards to ensure operations are recorded
        drop(_hessian_guard);
        drop(_workspace_guard);
        drop(_info_guard);

        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }

        // Check factorization result
        let mut info = [0i32];
        self.stream.memcpy_dtoh(&self.d_info, &mut info)?;
        self.stream.synchronize()?;

        if info[0] != 0 {
            // Matrix is not positive definite
            return Err(GpuNewtonError::FactorizationFailed(info[0]));
        }

        // Get pointers for solve step
        let (hessian_ptr, _hessian_guard) = hessian.device_ptr(&self.stream);
        let (rhs_ptr, _rhs_guard) = rhs.device_ptr_mut(&self.stream);
        let (info_ptr, _info_guard) = self.d_info.device_ptr_mut(&self.stream);

        // Solve using Cholesky factors
        let status = unsafe {
            cusolver_sys::cusolverDnDpotrs(
                self.handle.cu(),
                cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                6, // n
                1, // nrhs
                hessian_ptr as *const f64,
                6, // lda
                rhs_ptr as *mut f64,
                6, // ldb
                info_ptr as *mut i32,
            )
        };

        drop(_hessian_guard);
        drop(_rhs_guard);
        drop(_info_guard);

        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuNewtonError::CusolverError(status));
        }

        // Synchronize to ensure solve is complete
        self.stream.synchronize()?;

        Ok(())
    }

    /// Get access to the internal Hessian buffer for direct GPU operations.
    pub fn hessian_buffer(&self) -> &CudaSlice<f64> {
        &self.d_hessian
    }

    /// Get mutable access to the internal Hessian buffer.
    pub fn hessian_buffer_mut(&mut self) -> &mut CudaSlice<f64> {
        &mut self.d_hessian
    }

    /// Get access to the internal RHS/solution buffer.
    pub fn rhs_buffer(&self) -> &CudaSlice<f64> {
        &self.d_rhs
    }

    /// Get mutable access to the internal RHS/solution buffer.
    pub fn rhs_buffer_mut(&mut self) -> &mut CudaSlice<f64> {
        &mut self.d_rhs
    }

    /// Convert 6×6 matrix from row-major to column-major order.
    fn row_to_col_major_6x6(row_major: &[f64; 36]) -> [f64; 36] {
        let mut col_major = [0.0f64; 36];
        for i in 0..6 {
            for j in 0..6 {
                // row_major[i, j] = row_major[i * 6 + j]
                // col_major[i, j] = col_major[j * 6 + i]
                col_major[j * 6 + i] = row_major[i * 6 + j];
            }
        }
        col_major
    }

    /// Get the CUDA stream for this solver.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get the CUDA context for this solver.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_solve_identity() {
        let mut solver = GpuNewtonSolver::new(0).unwrap();

        // H = I, g = [1, 2, 3, 4, 5, 6]
        // δ = -H⁻¹g = -g = [-1, -2, -3, -4, -5, -6]
        let hessian = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
        ];
        let gradient = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let solution = solver.solve(&hessian, &gradient).unwrap();

        for (i, &val) in solution.iter().enumerate() {
            assert_relative_eq!(val, -(i as f64 + 1.0), epsilon = 1e-10);
        }
    }
    #[test]
    fn test_solve_scaled_identity() {
        let mut solver = GpuNewtonSolver::new(0).unwrap();

        // H = 2I, g = [2, 4, 6, 8, 10, 12]
        // δ = -H⁻¹g = -g/2 = [-1, -2, -3, -4, -5, -6]
        let hessian = [
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0,
        ];
        let gradient = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let solution = solver.solve(&hessian, &gradient).unwrap();

        for (i, &val) in solution.iter().enumerate() {
            assert_relative_eq!(val, -(i as f64 + 1.0), epsilon = 1e-10);
        }
    }
    #[test]
    fn test_solve_positive_definite() {
        let mut solver = GpuNewtonSolver::new(0).unwrap();

        // Create a positive definite matrix: H = A^T * A + I
        // A simple symmetric positive definite matrix
        let hessian = [
            4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 0.2, 0.0,
            0.0, 0.0, 0.0, 0.2, 2.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 1.5, 0.3, 0.0, 0.0, 0.0, 0.0,
            0.3, 1.0,
        ];
        let gradient = [1.0, 2.0, 3.0, 0.5, 0.2, 0.1];

        let solution = solver.solve(&hessian, &gradient).unwrap();

        // Verify: H * solution ≈ -gradient
        let mut result = [0.0f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                result[i] += hessian[i * 6 + j] * solution[j];
            }
        }

        for i in 0..6 {
            assert_relative_eq!(result[i], -gradient[i], epsilon = 1e-8);
        }
    }
    #[test]
    fn test_solve_non_positive_definite() {
        let mut solver = GpuNewtonSolver::new(0).unwrap();

        // Matrix with negative eigenvalue (not positive definite)
        // Should fall back to LU factorization
        let hessian = [
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0,
        ];
        let gradient = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let solution = solver.solve(&hessian, &gradient).unwrap();

        // Verify: H * solution ≈ -gradient
        let mut result = [0.0f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                result[i] += hessian[i * 6 + j] * solution[j];
            }
        }

        for i in 0..6 {
            assert_relative_eq!(result[i], -gradient[i], epsilon = 1e-8);
        }
    }
    #[test]
    fn test_row_to_col_major() {
        let row_major = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        ];

        let col_major = GpuNewtonSolver::row_to_col_major_6x6(&row_major);

        // Verify: col_major[j * 6 + i] == row_major[i * 6 + j]
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(col_major[j * 6 + i], row_major[i * 6 + j]);
            }
        }
    }
    #[test]
    fn test_solve_inplace() {
        let mut solver = GpuNewtonSolver::new(0).unwrap();

        // H = 2I, g = [2, 4, 6, 8, 10, 12]
        // δ = -H⁻¹g = -g/2 = [-1, -2, -3, -4, -5, -6]
        let hessian_row_major = [
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0,
        ];
        let hessian_col_major = GpuNewtonSolver::row_to_col_major_6x6(&hessian_row_major);
        let gradient = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        // Upload to GPU
        let stream = solver.stream().clone();
        let mut d_hessian = stream.alloc_zeros::<f64>(36).unwrap();
        let mut d_gradient = stream.alloc_zeros::<f64>(6).unwrap();
        let mut d_delta = stream.alloc_zeros::<f64>(6).unwrap();

        stream
            .memcpy_htod(&hessian_col_major, &mut d_hessian)
            .unwrap();
        stream.memcpy_htod(&gradient, &mut d_gradient).unwrap();
        stream.synchronize().unwrap();

        // Solve inplace
        solver
            .solve_inplace(&mut d_hessian, &d_gradient, &mut d_delta)
            .unwrap();

        // Download result
        let mut delta = [0.0f64; 6];
        stream.memcpy_dtoh(&d_delta, &mut delta).unwrap();
        stream.synchronize().unwrap();

        for (i, &val) in delta.iter().enumerate() {
            assert_relative_eq!(val, -(i as f64 + 1.0), epsilon = 1e-10);
        }
    }
}
