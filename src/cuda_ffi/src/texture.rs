//! Texture memory support for voxel data.
//!
//! This module provides RAII wrappers for CUDA texture objects
//! that can be used to access voxel means and inverse covariances
//! through the texture cache, potentially improving memory bandwidth
//! for scattered read patterns.
//!
//! # Example
//!
//! ```ignore
//! use cuda_ffi::texture::{VoxelMeansTexture, VoxelInvCovsTexture};
//!
//! // Create texture objects from device buffers
//! let means_tex = VoxelMeansTexture::new(d_means_ptr, num_voxels)?;
//! let inv_covs_tex = VoxelInvCovsTexture::new(d_inv_covs_ptr, num_voxels)?;
//!
//! // Use in texture-enabled kernel launch
//! batch_persistent_ndt_launch_textured(
//!     means_tex.handle(),
//!     inv_covs_tex.handle(),
//!     // ... other parameters
//! );
//! ```

use std::ffi::c_int;

/// Opaque handle for CUDA texture objects.
///
/// cudaTextureObject_t is defined as `unsigned long long` in CUDA.
pub type CudaTextureObject = u64;

/// Raw stream handle (alias for compatibility with async_stream module)
pub type RawCudaStream = *mut std::ffi::c_void;

// FFI declarations for texture operations
extern "C" {
    fn create_voxel_means_texture(
        tex_out: *mut CudaTextureObject,
        d_means: *const f32,
        num_voxels: usize,
    ) -> c_int;

    fn create_voxel_inv_covs_texture(
        tex_out: *mut CudaTextureObject,
        d_inv_covs: *const f32,
        num_voxels: usize,
    ) -> c_int;

    fn destroy_texture_object(tex: CudaTextureObject) -> c_int;

    fn texture_object_size() -> usize;

    fn batch_persistent_ndt_launch_textured(
        tex_voxel_means: CudaTextureObject,
        tex_voxel_inv_covs: CudaTextureObject,
        hash_table: *const std::ffi::c_void,
        hash_capacity: u32,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
        all_source_points: *const f32,
        all_initial_poses: *const f32,
        points_per_slot: *const c_int,
        all_reduce_buffers: *mut f32,
        barrier_counters: *mut c_int,
        barrier_senses: *mut c_int,
        all_out_poses: *mut f32,
        all_out_iterations: *mut c_int,
        all_out_converged: *mut u32,
        all_out_scores: *mut f32,
        all_out_hessians: *mut f32,
        all_out_correspondences: *mut u32,
        all_out_oscillations: *mut u32,
        all_out_alpha_sums: *mut f32,
        num_slots: c_int,
        blocks_per_slot: c_int,
        max_points_per_slot: c_int,
        max_iterations: c_int,
        epsilon: f32,
        ls_enabled: c_int,
        ls_num_candidates: c_int,
        ls_mu: f32,
        ls_nu: f32,
        fixed_step_size: f32,
        reg_ref_x: *const f32,
        reg_ref_y: *const f32,
        reg_scale: f32,
        reg_enabled: c_int,
        stream: RawCudaStream,
    ) -> c_int;
}

/// Error type for texture operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureError(pub i32);

impl std::fmt::Display for TextureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA texture error: {}", self.0)
    }
}

impl std::error::Error for TextureError {}

/// Get the size of a CUDA texture object handle.
pub fn texture_handle_size() -> usize {
    unsafe { texture_object_size() }
}

/// RAII wrapper for a CUDA texture object bound to voxel means data.
///
/// The texture provides read-only access to voxel means through the
/// texture cache, which can improve performance for scattered reads.
pub struct VoxelMeansTexture {
    handle: CudaTextureObject,
}

impl VoxelMeansTexture {
    /// Create a texture object for voxel means.
    ///
    /// # Arguments
    /// * `d_means` - Device pointer to voxel means array [num_voxels * 3] floats
    /// * `num_voxels` - Number of voxels
    ///
    /// # Safety
    /// The device pointer must remain valid for the lifetime of this texture.
    pub fn new(d_means: u64, num_voxels: usize) -> Result<Self, TextureError> {
        let mut handle: CudaTextureObject = 0;
        let result =
            unsafe { create_voxel_means_texture(&mut handle, d_means as *const f32, num_voxels) };
        if result != 0 {
            return Err(TextureError(result));
        }
        Ok(Self { handle })
    }

    /// Get the raw texture object handle for use in kernel launches.
    pub fn handle(&self) -> CudaTextureObject {
        self.handle
    }
}

impl Drop for VoxelMeansTexture {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe {
                destroy_texture_object(self.handle);
            }
        }
    }
}

/// RAII wrapper for a CUDA texture object bound to voxel inverse covariances.
///
/// The texture provides read-only access to inverse covariances through the
/// texture cache, which can improve performance for scattered reads.
pub struct VoxelInvCovsTexture {
    handle: CudaTextureObject,
}

impl VoxelInvCovsTexture {
    /// Create a texture object for voxel inverse covariances.
    ///
    /// # Arguments
    /// * `d_inv_covs` - Device pointer to inverse covariances [num_voxels * 9] floats
    /// * `num_voxels` - Number of voxels
    ///
    /// # Safety
    /// The device pointer must remain valid for the lifetime of this texture.
    pub fn new(d_inv_covs: u64, num_voxels: usize) -> Result<Self, TextureError> {
        let mut handle: CudaTextureObject = 0;
        let result = unsafe {
            create_voxel_inv_covs_texture(&mut handle, d_inv_covs as *const f32, num_voxels)
        };
        if result != 0 {
            return Err(TextureError(result));
        }
        Ok(Self { handle })
    }

    /// Get the raw texture object handle for use in kernel launches.
    pub fn handle(&self) -> CudaTextureObject {
        self.handle
    }
}

impl Drop for VoxelInvCovsTexture {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe {
                destroy_texture_object(self.handle);
            }
        }
    }
}

/// Parameters for texture-enabled batch NDT kernel launch.
#[derive(Debug, Clone)]
pub struct TexturedBatchNdtParams {
    /// Hash table device pointer
    pub hash_table: u64,
    /// Hash table capacity
    pub hash_capacity: u32,
    /// Gaussian d1 parameter
    pub gauss_d1: f32,
    /// Gaussian d2 parameter
    pub gauss_d2: f32,
    /// Voxel resolution
    pub resolution: f32,
    /// Number of slots in the batch
    pub num_slots: i32,
    /// Blocks per slot
    pub blocks_per_slot: i32,
    /// Maximum points per slot
    pub max_points_per_slot: i32,
    /// Maximum Newton iterations
    pub max_iterations: i32,
    /// Convergence epsilon
    pub epsilon: f32,
    /// Enable line search
    pub ls_enabled: bool,
    /// Number of line search candidates
    pub ls_num_candidates: i32,
    /// Line search mu parameter
    pub ls_mu: f32,
    /// Line search nu parameter
    pub ls_nu: f32,
    /// Fixed step size (when line search disabled)
    pub fixed_step_size: f32,
    /// Regularization scale
    pub reg_scale: f32,
    /// Enable regularization
    pub reg_enabled: bool,
}

/// Launch the texture-enabled batch persistent NDT kernel.
///
/// This version uses texture memory for voxel_means and voxel_inv_covs,
/// potentially providing better cache performance for scattered reads.
///
/// # Safety
/// All device pointers must be valid and properly aligned.
/// The stream must be a valid CUDA stream or null for the default stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn batch_persistent_ndt_launch_textured_raw(
    tex_voxel_means: CudaTextureObject,
    tex_voxel_inv_covs: CudaTextureObject,
    params: &TexturedBatchNdtParams,
    all_source_points: u64,
    all_initial_poses: u64,
    points_per_slot: u64,
    all_reduce_buffers: u64,
    barrier_counters: u64,
    barrier_senses: u64,
    all_out_poses: u64,
    all_out_iterations: u64,
    all_out_converged: u64,
    all_out_scores: u64,
    all_out_hessians: u64,
    all_out_correspondences: u64,
    all_out_oscillations: u64,
    all_out_alpha_sums: u64,
    reg_ref_x: u64,
    reg_ref_y: u64,
    stream: RawCudaStream,
) -> Result<(), TextureError> {
    let result = batch_persistent_ndt_launch_textured(
        tex_voxel_means,
        tex_voxel_inv_covs,
        params.hash_table as *const std::ffi::c_void,
        params.hash_capacity,
        params.gauss_d1,
        params.gauss_d2,
        params.resolution,
        all_source_points as *const f32,
        all_initial_poses as *const f32,
        points_per_slot as *const c_int,
        all_reduce_buffers as *mut f32,
        barrier_counters as *mut c_int,
        barrier_senses as *mut c_int,
        all_out_poses as *mut f32,
        all_out_iterations as *mut c_int,
        all_out_converged as *mut u32,
        all_out_scores as *mut f32,
        all_out_hessians as *mut f32,
        all_out_correspondences as *mut u32,
        all_out_oscillations as *mut u32,
        all_out_alpha_sums as *mut f32,
        params.num_slots,
        params.blocks_per_slot,
        params.max_points_per_slot,
        params.max_iterations,
        params.epsilon,
        if params.ls_enabled { 1 } else { 0 },
        params.ls_num_candidates,
        params.ls_mu,
        params.ls_nu,
        params.fixed_step_size,
        if reg_ref_x == 0 {
            std::ptr::null()
        } else {
            reg_ref_x as *const f32
        },
        if reg_ref_y == 0 {
            std::ptr::null()
        } else {
            reg_ref_y as *const f32
        },
        params.reg_scale,
        if params.reg_enabled { 1 } else { 0 },
        stream,
    );

    if result != 0 {
        return Err(TextureError(result));
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_texture_handle_size() {
        let size = texture_handle_size();
        // cudaTextureObject_t is unsigned long long = 8 bytes
        assert_eq!(size, 8);
    }
    #[test]
    fn test_texture_error_display() {
        let err = TextureError(1);
        assert_eq!(format!("{err}"), "CUDA texture error: 1");
    }
}
