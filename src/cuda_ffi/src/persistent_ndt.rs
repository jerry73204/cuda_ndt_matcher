//! Persistent NDT kernel using cooperative groups.
//!
//! This module provides FFI bindings to the CUDA persistent NDT kernel,
//! which runs the entire Newton optimization loop in a single kernel launch
//! using cooperative groups for grid synchronization.
//!
//! # Usage
//!
//! ```ignore
//! use cuda_ffi::persistent_ndt::{PersistentNdt, is_supported};
//!
//! // Check if device supports cooperative launch
//! if !is_supported()? {
//!     return Err("Cooperative launch not supported");
//! }
//!
//! // Check if point count fits in max grid
//! let max_blocks = PersistentNdt::get_max_blocks()?;
//! let num_blocks = (num_points + 255) / 256;
//! if num_blocks > max_blocks {
//!     return use_legacy_pipeline();
//! }
//!
//! // Launch optimization
//! PersistentNdt::launch(
//!     source_points, voxel_means, voxel_inv_covs, hash_table,
//!     gauss_d1, gauss_d2, resolution,
//!     num_points, num_voxels, hash_capacity,
//!     max_iterations, epsilon,
//!     initial_pose, reduce_buffer,
//!     out_pose, out_iterations, out_converged, out_score, out_hessian
//! )?;
//! ```

use crate::radix_sort::{check_cuda, CudaError};
use std::ffi::c_int;

// ============================================================================
// FFI Declarations
// ============================================================================

// Custom error code for grid too large
const CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: c_int = 720;

extern "C" {
    fn persistent_ndt_get_max_blocks(
        block_size: c_int,
        shared_mem_bytes: c_int,
        max_blocks: *mut c_int,
    ) -> c_int;

    fn persistent_ndt_is_supported(supported: *mut c_int) -> c_int;

    fn persistent_ndt_launch(
        source_points: *const f32,
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
        num_points: u32,
        num_voxels: u32,
        hash_capacity: u32,
        max_iterations: i32,
        epsilon: f32,
        // Phase 18.2: Regularization parameters
        reg_ref_x: f32,
        reg_ref_y: f32,
        reg_scale: f32,
        reg_enabled: i32,
        // Phase 18.1: Line search parameters
        ls_enabled: i32,
        ls_num_candidates: i32,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32, // Step size when line search disabled (default: 0.1)
        // Buffers
        initial_pose: *const f32,
        reduce_buffer: *mut f32,
        out_pose: *mut f32,
        out_iterations: *mut i32,
        out_converged: *mut u32,
        out_final_score: *mut f32,
        out_hessian: *mut f32,
        out_num_correspondences: *mut u32,   // Phase 18.3
        out_max_oscillation_count: *mut u32, // Phase 18.4
        out_alpha_sum: *mut f32,             // Phase 19.3
        // Phase 19.4: Debug output
        debug_enabled: i32,
        debug_buffer: *mut f32, // [max_iterations * 50] or nullptr
    ) -> c_int;

    fn persistent_ndt_reduce_buffer_size() -> u32;

    fn cudaDeviceSynchronize() -> c_int;
}

// ============================================================================
// Public API
// ============================================================================

/// Error returned when grid is too large for cooperative launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridTooLargeError {
    pub requested_blocks: usize,
    pub max_blocks: usize,
}

impl std::fmt::Display for GridTooLargeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Grid too large for cooperative launch: {} blocks requested, {} max",
            self.requested_blocks, self.max_blocks
        )
    }
}

impl std::error::Error for GridTooLargeError {}

/// Check if current device supports cooperative kernel launch.
pub fn is_supported() -> Result<bool, CudaError> {
    let mut supported: c_int = 0;
    unsafe {
        check_cuda(persistent_ndt_is_supported(&mut supported))?;
    }
    Ok(supported != 0)
}

/// Get required reduce buffer size in bytes.
pub fn reduce_buffer_size() -> usize {
    unsafe { persistent_ndt_reduce_buffer_size() as usize }
}

/// Persistent NDT kernel interface.
pub struct PersistentNdt;

impl PersistentNdt {
    /// Block size used by the persistent kernel.
    pub const BLOCK_SIZE: usize = 256;

    /// Number of reduce values per thread (score + gradient + hessian + correspondences).
    pub const REDUCE_SIZE: usize = 29;

    /// Number of floats per iteration in debug buffer (Phase 19.4).
    pub const DEBUG_FLOATS_PER_ITER: usize = 50;

    /// Get maximum number of blocks for cooperative launch.
    pub fn get_max_blocks() -> Result<usize, CudaError> {
        let shared_mem_bytes =
            (Self::BLOCK_SIZE * Self::REDUCE_SIZE * std::mem::size_of::<f32>()) as c_int;
        let mut max_blocks: c_int = 0;
        unsafe {
            check_cuda(persistent_ndt_get_max_blocks(
                Self::BLOCK_SIZE as c_int,
                shared_mem_bytes,
                &mut max_blocks,
            ))?;
        }
        Ok(max_blocks as usize)
    }

    /// Check if the given number of points can be processed with cooperative launch.
    pub fn can_launch(num_points: usize) -> Result<bool, CudaError> {
        let num_blocks = num_points.div_ceil(Self::BLOCK_SIZE);
        let max_blocks = Self::get_max_blocks()?;
        Ok(num_blocks <= max_blocks)
    }

    /// Launch persistent NDT optimization kernel.
    ///
    /// # Arguments
    ///
    /// * `source_points` - Device pointer to source points [N * 3]
    /// * `voxel_means` - Device pointer to voxel means [V * 3]
    /// * `voxel_inv_covs` - Device pointer to inverse covariances [V * 9]
    /// * `hash_table` - Device pointer to hash table
    /// * `gauss_d1` - NDT Gaussian parameter d1
    /// * `gauss_d2` - NDT Gaussian parameter d2
    /// * `resolution` - Voxel resolution
    /// * `num_points` - Number of source points
    /// * `num_voxels` - Number of voxels
    /// * `hash_capacity` - Hash table capacity
    /// * `max_iterations` - Maximum Newton iterations
    /// * `epsilon` - Convergence threshold
    /// * `reg_ref_x` - Regularization reference X (GNSS)
    /// * `reg_ref_y` - Regularization reference Y (GNSS)
    /// * `reg_scale` - Regularization scale factor
    /// * `reg_enabled` - Whether regularization is enabled
    /// * `ls_enabled` - Whether line search is enabled
    /// * `ls_num_candidates` - Number of line search candidates (default: 8)
    /// * `ls_mu` - Armijo constant for line search (default: 1e-4)
    /// * `ls_nu` - Curvature constant for line search (default: 0.9)
    /// * `fixed_step_size` - Step size when line search disabled (default: 0.1)
    /// * `initial_pose` - Device pointer to initial pose [6]
    /// * `reduce_buffer` - Device pointer to reduction scratch and state [96 floats]
    /// * `out_pose` - Device pointer to output pose [6]
    /// * `out_iterations` - Device pointer to output iteration count
    /// * `out_converged` - Device pointer to output convergence flag
    /// * `out_final_score` - Device pointer to output final score
    /// * `out_hessian` - Device pointer to output Hessian [36]
    /// * `out_num_correspondences` - Device pointer to output correspondence count
    /// * `out_max_oscillation_count` - Device pointer to output max oscillation count
    /// * `out_alpha_sum` - Device pointer to output accumulated step sizes
    /// * `debug_enabled` - Whether to enable debug output
    /// * `debug_buffer` - Device pointer to debug buffer [max_iterations * 50] or null
    ///
    /// # Errors
    ///
    /// Returns `GridTooLargeError` if the grid size exceeds the cooperative launch limit.
    /// Returns `CudaError` for other CUDA errors.
    ///
    /// # Safety
    ///
    /// All device pointers must be valid with appropriate sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        source_points: *const f32,
        voxel_means: *const f32,
        voxel_inv_covs: *const f32,
        hash_table: *const std::ffi::c_void,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
        num_points: u32,
        num_voxels: u32,
        hash_capacity: u32,
        max_iterations: i32,
        epsilon: f32,
        reg_ref_x: f32,
        reg_ref_y: f32,
        reg_scale: f32,
        reg_enabled: bool,
        ls_enabled: bool,
        ls_num_candidates: i32,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32,
        initial_pose: *const f32,
        reduce_buffer: *mut f32,
        out_pose: *mut f32,
        out_iterations: *mut i32,
        out_converged: *mut u32,
        out_final_score: *mut f32,
        out_hessian: *mut f32,
        out_num_correspondences: *mut u32,
        out_max_oscillation_count: *mut u32,
        out_alpha_sum: *mut f32,
        debug_enabled: bool,
        debug_buffer: *mut f32,
    ) -> Result<(), CudaError> {
        let result = persistent_ndt_launch(
            source_points,
            voxel_means,
            voxel_inv_covs,
            hash_table,
            gauss_d1,
            gauss_d2,
            resolution,
            num_points,
            num_voxels,
            hash_capacity,
            max_iterations,
            epsilon,
            reg_ref_x,
            reg_ref_y,
            reg_scale,
            if reg_enabled { 1 } else { 0 },
            if ls_enabled { 1 } else { 0 },
            ls_num_candidates,
            ls_mu,
            ls_nu,
            fixed_step_size,
            initial_pose,
            reduce_buffer,
            out_pose,
            out_iterations,
            out_converged,
            out_final_score,
            out_hessian,
            out_num_correspondences,
            out_max_oscillation_count,
            out_alpha_sum,
            if debug_enabled { 1 } else { 0 },
            debug_buffer,
        );

        if result == CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE {
            // This shouldn't happen if caller checked can_launch(), but handle it
            return Err(CudaError::Other(result));
        }

        check_cuda(result)?;
        check_cuda(cudaDeviceSynchronize())
    }
}

// ============================================================================
// Raw pointer API (for CubeCL handle interop)
// ============================================================================

/// Check if cooperative launch is supported using raw device pointers.
pub fn persistent_ndt_supported() -> Result<bool, CudaError> {
    is_supported()
}

/// Get maximum blocks for cooperative launch.
pub fn persistent_ndt_max_blocks() -> Result<usize, CudaError> {
    PersistentNdt::get_max_blocks()
}

/// Check if point count can be processed with cooperative launch.
pub fn persistent_ndt_can_launch(num_points: usize) -> Result<bool, CudaError> {
    PersistentNdt::can_launch(num_points)
}

/// Get reduce buffer size in bytes.
pub fn persistent_ndt_buffer_size() -> usize {
    reduce_buffer_size()
}

/// Launch persistent NDT kernel using raw device pointers (u64 for CubeCL interop).
///
/// # Safety
///
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn persistent_ndt_launch_raw(
    d_source_points: u64,
    d_voxel_means: u64,
    d_voxel_inv_covs: u64,
    d_hash_table: u64,
    gauss_d1: f32,
    gauss_d2: f32,
    resolution: f32,
    num_points: usize,
    num_voxels: usize,
    hash_capacity: u32,
    max_iterations: i32,
    epsilon: f32,
    reg_ref_x: f32,
    reg_ref_y: f32,
    reg_scale: f32,
    reg_enabled: bool,
    ls_enabled: bool,
    ls_num_candidates: i32,
    ls_mu: f32,
    ls_nu: f32,
    fixed_step_size: f32,
    d_initial_pose: u64,
    d_reduce_buffer: u64,
    d_out_pose: u64,
    d_out_iterations: u64,
    d_out_converged: u64,
    d_out_final_score: u64,
    d_out_hessian: u64,
    d_out_num_correspondences: u64,
    d_out_max_oscillation_count: u64,
    d_out_alpha_sum: u64,
    debug_enabled: bool,
    d_debug_buffer: u64,
) -> Result<(), CudaError> {
    PersistentNdt::launch(
        d_source_points as *const f32,
        d_voxel_means as *const f32,
        d_voxel_inv_covs as *const f32,
        d_hash_table as *const std::ffi::c_void,
        gauss_d1,
        gauss_d2,
        resolution,
        num_points as u32,
        num_voxels as u32,
        hash_capacity,
        max_iterations,
        epsilon,
        reg_ref_x,
        reg_ref_y,
        reg_scale,
        reg_enabled,
        ls_enabled,
        ls_num_candidates,
        ls_mu,
        ls_nu,
        fixed_step_size,
        d_initial_pose as *const f32,
        d_reduce_buffer as *mut f32,
        d_out_pose as *mut f32,
        d_out_iterations as *mut i32,
        d_out_converged as *mut u32,
        d_out_final_score as *mut f32,
        d_out_hessian as *mut f32,
        d_out_num_correspondences as *mut u32,
        d_out_max_oscillation_count as *mut u32,
        d_out_alpha_sum as *mut f32,
        debug_enabled,
        d_debug_buffer as *mut f32,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_is_supported() {
        let result = is_supported();
        assert!(result.is_ok(), "is_supported should not fail");
        #[cfg(feature = "test-verbose")]
        println!("Cooperative launch supported: {}", result.unwrap());
    }
    #[test]
    fn test_get_max_blocks() {
        let result = PersistentNdt::get_max_blocks();
        assert!(result.is_ok(), "get_max_blocks should not fail");
        let max_blocks = result.unwrap();
        #[cfg(feature = "test-verbose")]
        println!("Max cooperative blocks: {max_blocks}");
        assert!(max_blocks > 0, "Max blocks should be positive");
    }
    #[test]
    fn test_reduce_buffer_size() {
        let size = reduce_buffer_size();
        assert_eq!(
            size,
            160 * 4,
            "Reduce buffer should be 160 floats (640 bytes) for parallel line search"
        );
    }
    #[test]
    fn test_can_launch() {
        // Small point count should always work
        let result = PersistentNdt::can_launch(1000);
        assert!(result.is_ok());
        assert!(
            result.unwrap(),
            "1000 points should fit in cooperative launch"
        );

        // Very large point count might not work
        let result = PersistentNdt::can_launch(10_000_000);
        assert!(result.is_ok());
        // Don't assert on the result - depends on GPU
        #[cfg(feature = "test-verbose")]
        println!("10M points can_launch: {}", result.unwrap());
    }
}
