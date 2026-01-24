//! Spatial hash table for GPU-accelerated voxel lookup.
//!
//! This module provides FFI bindings to the CUDA spatial hash table implementation.
//! The hash table maps 3D grid coordinates to voxel indices, enabling O(27) neighbor
//! lookups instead of O(V) brute-force search.
//!
//! # Usage
//!
//! ```ignore
//! // During map loading (once):
//! let capacity = VoxelHash::get_capacity(num_voxels);
//! let table_size = VoxelHash::get_table_size(capacity);
//! let hash_table = DeviceBuffer::new(table_size)?;
//! VoxelHash::init(&hash_table, capacity)?;
//! VoxelHash::build(&voxel_means, &voxel_valid, num_voxels, resolution, &hash_table, capacity)?;
//!
//! // Per iteration:
//! VoxelHash::query(&query_points, &voxel_means, num_queries, resolution, search_radius,
//!                  &hash_table, capacity, &neighbor_indices, &neighbor_counts)?;
//! ```

#[cfg(test)]
use crate::radix_sort::DeviceBuffer;
use crate::radix_sort::{check_cuda, CudaError};
use std::ffi::c_int;

// ============================================================================
// FFI Declarations
// ============================================================================

type CudaStream = *mut std::ffi::c_void;

extern "C" {
    fn voxel_hash_get_capacity(num_voxels: u32, capacity: *mut u32) -> c_int;

    fn voxel_hash_get_table_size(capacity: u32, bytes: *mut usize) -> c_int;

    fn voxel_hash_init(
        d_hash_table: *mut std::ffi::c_void,
        capacity: u32,
        stream: CudaStream,
    ) -> c_int;

    fn voxel_hash_build(
        d_voxel_means: *const f32,
        d_voxel_valid: *const u32,
        num_voxels: u32,
        resolution: f32,
        d_hash_table: *mut std::ffi::c_void,
        capacity: u32,
        stream: CudaStream,
    ) -> c_int;

    fn voxel_hash_query(
        d_query_points: *const f32,
        d_voxel_means: *const f32,
        num_queries: u32,
        resolution: f32,
        search_radius: f32,
        d_hash_table: *const std::ffi::c_void,
        capacity: u32,
        d_neighbor_indices: *mut i32,
        d_neighbor_counts: *mut u32,
        stream: CudaStream,
    ) -> c_int;

    fn voxel_hash_max_neighbors() -> u32;

    fn voxel_hash_count_entries(
        d_hash_table: *const std::ffi::c_void,
        capacity: u32,
        h_count: *mut u32,
        stream: CudaStream,
    ) -> c_int;

    // CUDA runtime
    fn cudaDeviceSynchronize() -> c_int;
}

// ============================================================================
// Public API
// ============================================================================

/// Maximum number of neighbors returned per query point.
pub fn max_neighbors() -> u32 {
    unsafe { voxel_hash_max_neighbors() }
}

/// Spatial hash table for voxel neighbor lookup.
///
/// This struct provides a high-level interface to the CUDA hash table.
/// Call methods in order: `get_capacity` → `get_table_size` → `init` → `build` → `query`.
pub struct VoxelHash;

impl VoxelHash {
    /// Get recommended hash table capacity for the given number of voxels.
    ///
    /// Returns a capacity with ~50% load factor for good performance.
    pub fn get_capacity(num_voxels: u32) -> Result<u32, CudaError> {
        let mut capacity: u32 = 0;
        unsafe {
            check_cuda(voxel_hash_get_capacity(num_voxels, &mut capacity))?;
        }
        Ok(capacity)
    }

    /// Get required memory size in bytes for hash table with given capacity.
    pub fn get_table_size(capacity: u32) -> Result<usize, CudaError> {
        let mut bytes: usize = 0;
        unsafe {
            check_cuda(voxel_hash_get_table_size(capacity, &mut bytes))?;
        }
        Ok(bytes)
    }

    /// Initialize hash table (set all entries to empty).
    ///
    /// # Safety
    /// `d_hash_table` must be a valid device pointer with at least `get_table_size(capacity)` bytes.
    pub unsafe fn init(
        d_hash_table: *mut std::ffi::c_void,
        capacity: u32,
    ) -> Result<(), CudaError> {
        check_cuda(voxel_hash_init(
            d_hash_table,
            capacity,
            std::ptr::null_mut(),
        ))?;
        check_cuda(cudaDeviceSynchronize())
    }

    /// Build hash table from voxel means.
    ///
    /// # Arguments
    /// * `d_voxel_means` - Device pointer to voxel means [V * 3]
    /// * `d_voxel_valid` - Device pointer to voxel validity flags [V]
    /// * `num_voxels` - Number of voxels
    /// * `resolution` - Voxel grid resolution (e.g., 2.0)
    /// * `d_hash_table` - Device pointer to initialized hash table
    /// * `capacity` - Hash table capacity
    ///
    /// # Safety
    /// All device pointers must be valid with appropriate sizes.
    pub unsafe fn build(
        d_voxel_means: *const f32,
        d_voxel_valid: *const u32,
        num_voxels: u32,
        resolution: f32,
        d_hash_table: *mut std::ffi::c_void,
        capacity: u32,
    ) -> Result<(), CudaError> {
        check_cuda(voxel_hash_build(
            d_voxel_means,
            d_voxel_valid,
            num_voxels,
            resolution,
            d_hash_table,
            capacity,
            std::ptr::null_mut(),
        ))?;
        check_cuda(cudaDeviceSynchronize())
    }

    /// Query neighbors for multiple points using hash table.
    ///
    /// # Arguments
    /// * `d_query_points` - Device pointer to query points [N * 3]
    /// * `d_voxel_means` - Device pointer to voxel means [V * 3]
    /// * `num_queries` - Number of query points
    /// * `resolution` - Voxel grid resolution
    /// * `search_radius` - Search radius (typically = resolution)
    /// * `d_hash_table` - Device pointer to built hash table
    /// * `capacity` - Hash table capacity
    /// * `d_neighbor_indices` - Device pointer to output indices [N * MAX_NEIGHBORS]
    /// * `d_neighbor_counts` - Device pointer to output counts [N]
    ///
    /// # Safety
    /// All device pointers must be valid with appropriate sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn query(
        d_query_points: *const f32,
        d_voxel_means: *const f32,
        num_queries: u32,
        resolution: f32,
        search_radius: f32,
        d_hash_table: *const std::ffi::c_void,
        capacity: u32,
        d_neighbor_indices: *mut i32,
        d_neighbor_counts: *mut u32,
    ) -> Result<(), CudaError> {
        check_cuda(voxel_hash_query(
            d_query_points,
            d_voxel_means,
            num_queries,
            resolution,
            search_radius,
            d_hash_table,
            capacity,
            d_neighbor_indices,
            d_neighbor_counts,
            std::ptr::null_mut(),
        ))?;
        check_cuda(cudaDeviceSynchronize())
    }

    /// Count non-empty entries in hash table (for debugging).
    ///
    /// # Safety
    /// - `d_hash_table` must be a valid device pointer to a hash table created by `build`
    /// - `capacity` must match the capacity used when building the hash table
    pub unsafe fn count_entries(
        d_hash_table: *const std::ffi::c_void,
        capacity: u32,
    ) -> Result<u32, CudaError> {
        let mut count: u32 = 0;
        check_cuda(voxel_hash_count_entries(
            d_hash_table,
            capacity,
            &mut count,
            std::ptr::null_mut(),
        ))?;
        check_cuda(cudaDeviceSynchronize())?;
        Ok(count)
    }
}

// ============================================================================
// In-Place API (for zero-copy pipeline with CubeCL handles)
// ============================================================================

/// Query required hash table capacity for given number of voxels.
pub fn hash_table_capacity(num_voxels: usize) -> Result<u32, CudaError> {
    VoxelHash::get_capacity(num_voxels as u32)
}

/// Query required memory size for hash table.
pub fn hash_table_size(capacity: u32) -> Result<usize, CudaError> {
    VoxelHash::get_table_size(capacity)
}

/// Initialize hash table using raw device pointer (CUdeviceptr as u64).
///
/// # Safety
/// `d_hash_table` must be a valid CUDA device pointer with at least `hash_table_size(capacity)` bytes.
pub unsafe fn hash_table_init(d_hash_table: u64, capacity: u32) -> Result<(), CudaError> {
    VoxelHash::init(d_hash_table as *mut std::ffi::c_void, capacity)
}

/// Build hash table using raw device pointers.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
pub unsafe fn hash_table_build(
    d_voxel_means: u64,
    d_voxel_valid: u64,
    num_voxels: usize,
    resolution: f32,
    d_hash_table: u64,
    capacity: u32,
) -> Result<(), CudaError> {
    VoxelHash::build(
        d_voxel_means as *const f32,
        d_voxel_valid as *const u32,
        num_voxels as u32,
        resolution,
        d_hash_table as *mut std::ffi::c_void,
        capacity,
    )
}

/// Query neighbors using raw device pointers.
///
/// # Safety
/// All device pointers must be valid CUDA device pointers with appropriate sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hash_table_query(
    d_query_points: u64,
    d_voxel_means: u64,
    num_queries: usize,
    resolution: f32,
    search_radius: f32,
    d_hash_table: u64,
    capacity: u32,
    d_neighbor_indices: u64,
    d_neighbor_counts: u64,
) -> Result<(), CudaError> {
    VoxelHash::query(
        d_query_points as *const f32,
        d_voxel_means as *const f32,
        num_queries as u32,
        resolution,
        search_radius,
        d_hash_table as *const std::ffi::c_void,
        capacity,
        d_neighbor_indices as *mut i32,
        d_neighbor_counts as *mut u32,
    )
}

/// Count non-empty entries in hash table (for debugging).
///
/// # Safety
/// d_hash_table must be a valid CUDA device pointer.
pub unsafe fn hash_table_count_entries(d_hash_table: u64, capacity: u32) -> Result<u32, CudaError> {
    VoxelHash::count_entries(d_hash_table as *const std::ffi::c_void, capacity)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_max_neighbors() {
        let max = max_neighbors();
        assert_eq!(max, 8, "MAX_NEIGHBORS should be 8");
    }
    #[test]
    fn test_get_capacity() {
        let cap = VoxelHash::get_capacity(1000).unwrap();
        assert!(cap >= 2000, "Capacity should be at least 2x voxels");
        assert!(cap.is_power_of_two(), "Capacity should be power of 2");
    }
    #[test]
    fn test_get_table_size() {
        let cap = VoxelHash::get_capacity(1000).unwrap();
        let size = VoxelHash::get_table_size(cap).unwrap();
        // Each entry is 16 bytes (int64 key + int32 value + int32 padding)
        assert_eq!(size, cap as usize * 16);
    }
    #[test]
    fn test_build_and_query() {
        // Create test data: 4 voxels at corners of a 4x4x4 grid
        let voxel_means: Vec<f32> = vec![
            0.5, 0.5, 0.5, // Voxel 0 at (0,0,0) cell
            2.5, 0.5, 0.5, // Voxel 1 at (1,0,0) cell
            0.5, 2.5, 0.5, // Voxel 2 at (0,1,0) cell
            0.5, 0.5, 2.5, // Voxel 3 at (0,0,1) cell
        ];
        let voxel_valid: Vec<u32> = vec![1, 1, 1, 1];
        let num_voxels = 4u32;
        let resolution = 2.0f32;

        // Query points
        let query_points: Vec<f32> = vec![
            0.5, 0.5, 0.5, // Should find voxel 0
            1.5, 0.5, 0.5, // Should find voxels 0 and 1 (at boundary)
        ];
        let num_queries = 2u32;
        let search_radius = 2.0f32;

        // Get capacity and allocate
        let capacity = VoxelHash::get_capacity(num_voxels).unwrap();
        let table_size = VoxelHash::get_table_size(capacity).unwrap();

        // Allocate device memory
        let mut d_voxel_means = DeviceBuffer::new(voxel_means.len() * 4).unwrap();
        let mut d_voxel_valid = DeviceBuffer::new(voxel_valid.len() * 4).unwrap();
        let d_hash_table = DeviceBuffer::new(table_size).unwrap();
        let mut d_query_points = DeviceBuffer::new(query_points.len() * 4).unwrap();
        let d_neighbor_indices =
            DeviceBuffer::new(num_queries as usize * max_neighbors() as usize * 4).unwrap();
        let d_neighbor_counts = DeviceBuffer::new(num_queries as usize * 4).unwrap();

        // Copy input data
        d_voxel_means.copy_from_host(&voxel_means).unwrap();
        d_voxel_valid.copy_from_host(&voxel_valid).unwrap();
        d_query_points.copy_from_host(&query_points).unwrap();

        // Build hash table
        unsafe {
            VoxelHash::init(d_hash_table.as_ptr(), capacity).unwrap();
            VoxelHash::build(
                d_voxel_means.as_ptr() as *const f32,
                d_voxel_valid.as_ptr() as *const u32,
                num_voxels,
                resolution,
                d_hash_table.as_ptr(),
                capacity,
            )
            .unwrap();

            // Query
            VoxelHash::query(
                d_query_points.as_ptr() as *const f32,
                d_voxel_means.as_ptr() as *const f32,
                num_queries,
                resolution,
                search_radius,
                d_hash_table.as_ptr(),
                capacity,
                d_neighbor_indices.as_ptr() as *mut i32,
                d_neighbor_counts.as_ptr() as *mut u32,
            )
            .unwrap();
        }

        // Copy results back
        let mut neighbor_indices = vec![-1i32; num_queries as usize * max_neighbors() as usize];
        let mut neighbor_counts = vec![0u32; num_queries as usize];
        d_neighbor_indices
            .copy_to_host(&mut neighbor_indices)
            .unwrap();
        d_neighbor_counts
            .copy_to_host(&mut neighbor_counts)
            .unwrap();

        // Verify results
        assert!(
            neighbor_counts[0] >= 1,
            "Query 0 should find at least 1 neighbor"
        );
        assert!(
            neighbor_indices[0] == 0,
            "Query 0 should find voxel 0 first"
        );

        #[cfg(feature = "test-verbose")]
        {
            println!(
                "Query 0: count={}, indices={:?}",
                neighbor_counts[0], neighbor_indices
            );
            println!("Query 1: count={}", neighbor_counts[1]);
        }
    }
}
