//! CubeCL GPU kernels for voxel grid operations.
//!
//! This module provides GPU-accelerated kernels for:
//! - Voxel ID computation
//! - Point transformation
//! - Future: covariance accumulation

use cubecl::prelude::*;

/// Compute voxel IDs for a batch of points.
///
/// Each point is assigned to a voxel based on its position and the grid resolution.
/// The voxel ID is computed as: id = ix + iy * dim_x + iz * dim_x * dim_y
///
/// # Inputs
/// - `points`: [N * 3] point coordinates (x, y, z) flattened
/// - `min_bound`: [3] minimum bounds of the grid
/// - `inv_resolution`: 1.0 / resolution (precomputed)
/// - `grid_dim_x`: grid X dimension
/// - `grid_dim_y`: grid Y dimension
///
/// # Outputs
/// - `voxel_ids`: [N] voxel ID for each point
#[cube(launch_unchecked)]
pub fn compute_voxel_ids_kernel<F: Float>(
    points: &Array<F>,
    min_bound: &Array<F>,
    inv_resolution: F,
    grid_dim_x: u32,
    grid_dim_y: u32,
    num_points: u32,
    voxel_ids: &mut Array<u32>,
) {
    let idx = ABSOLUTE_POS;

    if idx >= num_points {
        terminate!();
    }

    // Load point coordinates
    let base = idx * 3;
    let x = points[base];
    let y = points[base + 1];
    let z = points[base + 2];

    // Load min bounds
    let min_x = min_bound[0];
    let min_y = min_bound[1];
    let min_z = min_bound[2];

    // Compute local coordinates
    let local_x = (x - min_x) * inv_resolution;
    let local_y = (y - min_y) * inv_resolution;
    let local_z = (z - min_z) * inv_resolution;

    // Floor to get grid indices
    let ix = u32::cast_from(F::floor(local_x));
    let iy = u32::cast_from(F::floor(local_y));
    let iz = u32::cast_from(F::floor(local_z));

    // Compute linear voxel ID
    let voxel_id = ix + iy * grid_dim_x + iz * grid_dim_x * grid_dim_y;
    voxel_ids[idx] = voxel_id;
}

/// Transform points by a 4x4 transformation matrix.
///
/// # Inputs
/// - `points`: [N * 3] input point coordinates (flattened)
/// - `transform`: [16] 4x4 transformation matrix (row-major)
/// - `num_points`: number of points
///
/// # Outputs
/// - `output`: [N * 3] transformed point coordinates
#[cube(launch_unchecked)]
pub fn transform_points_kernel<F: Float>(
    points: &Array<F>,
    transform: &Array<F>,
    num_points: u32,
    output: &mut Array<F>,
) {
    let idx = ABSOLUTE_POS;

    if idx >= num_points {
        terminate!();
    }

    // Load point (homogeneous coordinate w=1 is implicit)
    let base = idx * 3;
    let x = points[base];
    let y = points[base + 1];
    let z = points[base + 2];

    // Load transformation matrix (row-major 4x4)
    // Row 0: transform[0..4]
    // Row 1: transform[4..8]
    // Row 2: transform[8..12]

    // Apply transformation: T * [x, y, z, 1]^T
    let out_x = transform[0] * x + transform[1] * y + transform[2] * z + transform[3];
    let out_y = transform[4] * x + transform[5] * y + transform[6] * z + transform[7];
    let out_z = transform[8] * x + transform[9] * y + transform[10] * z + transform[11];

    // Store result
    output[base] = out_x;
    output[base + 1] = out_y;
    output[base + 2] = out_z;
}

#[cfg(test)]
mod tests {
    // GPU tests require CUDA runtime, so we test the kernel structure
    // by ensuring they compile. Actual execution tests are in integration tests.

    #[test]
    fn test_kernels_compile() {
        // This test just verifies that the kernel macros expand correctly.
        // Actual GPU execution is tested elsewhere.
    }
}
