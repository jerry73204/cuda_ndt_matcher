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

/// Transform points by K different 4x4 transformation matrices (batched).
///
/// This kernel transforms N source points by K different poses simultaneously,
/// producing K×N transformed points. Used for batched initial pose estimation.
///
/// # Grid Layout
/// - Block: (256, 1, 1)
/// - Grid: (ceil(N/256), K, 1) where K is batch_size
///
/// # Inputs
/// - `points`: [N * 3] source point coordinates (shared across batches)
/// - `transforms`: [K * 16] K transformation matrices (row-major 4x4)
/// - `num_points`: N
/// - `batch_size`: K
///
/// # Outputs
/// - `output`: [K * N * 3] transformed points, batch-major order
#[cube(launch_unchecked)]
pub fn transform_points_batch_kernel<F: Float>(
    points: &Array<F>,
    transforms: &Array<F>,
    num_points: u32,
    batch_size: u32,
    output: &mut Array<F>,
) {
    // Grid: (ceil(N/256), K, 1), Block: (256, 1, 1)
    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let batch_idx = CUBE_POS_Y;

    if point_idx >= num_points || batch_idx >= batch_size {
        terminate!();
    }

    // Load point (shared across all batches)
    let pt_base = point_idx * 3;
    let x = points[pt_base];
    let y = points[pt_base + 1];
    let z = points[pt_base + 2];

    // Load transform for this batch (row-major 4x4)
    let t_offset = batch_idx * 16;
    let t00 = transforms[t_offset];
    let t01 = transforms[t_offset + 1];
    let t02 = transforms[t_offset + 2];
    let t03 = transforms[t_offset + 3];
    let t10 = transforms[t_offset + 4];
    let t11 = transforms[t_offset + 5];
    let t12 = transforms[t_offset + 6];
    let t13 = transforms[t_offset + 7];
    let t20 = transforms[t_offset + 8];
    let t21 = transforms[t_offset + 9];
    let t22 = transforms[t_offset + 10];
    let t23 = transforms[t_offset + 11];

    // Apply transform: T * [x, y, z, 1]^T
    let out_x = t00 * x + t01 * y + t02 * z + t03;
    let out_y = t10 * x + t11 * y + t12 * z + t13;
    let out_z = t20 * x + t21 * y + t22 * z + t23;

    // Write to batch-specific output location (batch-major: [K][N][3])
    let out_offset = (batch_idx * num_points + point_idx) * 3;
    output[out_offset] = out_x;
    output[out_offset + 1] = out_y;
    output[out_offset + 2] = out_z;
}

/// Compute K 4x4 transform matrices from K poses.
///
/// Each pose is [tx, ty, tz, roll, pitch, yaw] (Euler angles in radians).
///
/// # Grid Layout
/// - Block: (K, 1, 1) - one thread per pose
/// - Grid: (1, 1, 1)
///
/// # Inputs
/// - `poses`: [K * 6] K poses
/// - `batch_size`: K
///
/// # Outputs
/// - `transforms`: [K * 16] K transform matrices (row-major)
#[cube(launch_unchecked)]
pub fn compute_transforms_batch_kernel<F: Float>(
    poses: &Array<F>,
    batch_size: u32,
    transforms: &mut Array<F>,
) {
    let batch_idx = UNIT_POS_X;

    if batch_idx >= batch_size {
        terminate!();
    }

    // Load pose for this batch
    let pose_offset = batch_idx * 6;
    let tx = poses[pose_offset];
    let ty = poses[pose_offset + 1];
    let tz = poses[pose_offset + 2];
    let roll = poses[pose_offset + 3];
    let pitch = poses[pose_offset + 4];
    let yaw = poses[pose_offset + 5];

    // Compute sin/cos
    let sr = F::sin(roll);
    let cr = F::cos(roll);
    let sp = F::sin(pitch);
    let cp = F::cos(pitch);
    let sy = F::sin(yaw);
    let cy = F::cos(yaw);

    // Build rotation matrix (ZYX Euler angles)
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    let r00 = cy * cp;
    let r01 = cy * sp * sr - sy * cr;
    let r02 = cy * sp * cr + sy * sr;
    let r10 = sy * cp;
    let r11 = sy * sp * sr + cy * cr;
    let r12 = sy * sp * cr - cy * sr;
    let r20 = F::new(0.0) - sp;
    let r21 = cp * sr;
    let r22 = cp * cr;

    // Write transform matrix (row-major 4x4)
    let t_offset = batch_idx * 16;

    // Row 0
    transforms[t_offset] = r00;
    transforms[t_offset + 1] = r01;
    transforms[t_offset + 2] = r02;
    transforms[t_offset + 3] = tx;

    // Row 1
    transforms[t_offset + 4] = r10;
    transforms[t_offset + 5] = r11;
    transforms[t_offset + 6] = r12;
    transforms[t_offset + 7] = ty;

    // Row 2
    transforms[t_offset + 8] = r20;
    transforms[t_offset + 9] = r21;
    transforms[t_offset + 10] = r22;
    transforms[t_offset + 11] = tz;

    // Row 3 (homogeneous)
    transforms[t_offset + 12] = F::new(0.0);
    transforms[t_offset + 13] = F::new(0.0);
    transforms[t_offset + 14] = F::new(0.0);
    transforms[t_offset + 15] = F::new(1.0);
}

/// Compute sin/cos values for K poses.
///
/// # Grid Layout
/// - Block: (K, 1, 1) - one thread per pose
/// - Grid: (1, 1, 1)
///
/// # Inputs
/// - `poses`: [K * 6] K poses [tx, ty, tz, roll, pitch, yaw]
/// - `batch_size`: K
///
/// # Outputs
/// - `sin_cos`: [K * 6] [sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw]
#[cube(launch_unchecked)]
pub fn compute_sin_cos_batch_kernel<F: Float>(
    poses: &Array<F>,
    batch_size: u32,
    sin_cos: &mut Array<F>,
) {
    let batch_idx = UNIT_POS_X;

    if batch_idx >= batch_size {
        terminate!();
    }

    // Load angles for this batch
    let pose_offset = batch_idx * 6;
    let roll = poses[pose_offset + 3];
    let pitch = poses[pose_offset + 4];
    let yaw = poses[pose_offset + 5];

    // Compute sin/cos
    let sc_offset = batch_idx * 6;
    sin_cos[sc_offset] = F::sin(roll);
    sin_cos[sc_offset + 1] = F::cos(roll);
    sin_cos[sc_offset + 2] = F::sin(pitch);
    sin_cos[sc_offset + 3] = F::cos(pitch);
    sin_cos[sc_offset + 4] = F::sin(yaw);
    sin_cos[sc_offset + 5] = F::cos(yaw);
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
