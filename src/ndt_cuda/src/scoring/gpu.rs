//! GPU-accelerated batch scoring kernels using CubeCL.
//!
//! This module provides CUDA-accelerated batch scoring for multiple poses.
//! The key algorithm processes M poses × N points in a single kernel launch.
//!
//! # Architecture
//!
//! ```text
//! Per batch call (M poses, N points):
//!   GPU Kernel (M×N threads):
//!     1. Transform point by pose
//!     2. Brute-force radius search for neighbors
//!     3. Accumulate sum_score and track max_score across neighbors
//!     4. Output: scores[M×N], max_scores[M×N], has_neighbor[M×N]
//!
//!   CUB DeviceSegmentedReduce:
//!     - total_scores[M], nvtl_sums[M], nvtl_counts[M]
//! ```

use cubecl::prelude::*;

/// Transform a point by a 4x4 transformation matrix.
#[cube]
fn transform_point_batch<F: Float>(px: F, py: F, pz: F, transform: &Array<F>) -> (F, F, F) {
    let tx = transform[0] * px + transform[1] * py + transform[2] * pz + transform[3];
    let ty = transform[4] * px + transform[5] * py + transform[6] * pz + transform[7];
    let tz = transform[8] * px + transform[9] * py + transform[10] * pz + transform[11];
    (tx, ty, tz)
}

/// Batched scoring kernel for M poses × N points.
///
/// Each thread processes one (pose, point) pair. For each pair:
/// 1. Transform the source point using the pose's transform matrix
/// 2. Find all voxels within search radius (brute-force)
/// 3. Compute NDT score for each neighbor voxel
/// 4. Output sum_score (for transform probability) and max_score (for NVTL)
///
/// # Memory Layout (Column-Major for CUB Reduction)
///
/// Output is in a single array `outputs[M × N × 4]`:
/// - `outputs[(m * N + n) * 4 + 0]` - score for pose m, point n
/// - `outputs[(m * N + n) * 4 + 1]` - max_score for pose m, point n
/// - `outputs[(m * N + n) * 4 + 2]` - has_neighbor (0 or 1)
/// - `outputs[(m * N + n) * 4 + 3]` - num_correspondences
///
/// Note: Parameters are split to keep under CubeCL's 12-parameter limit.
/// gauss_params[0] = d1, gauss_params[1] = d2, gauss_params[2] = search_radius_sq
#[cube(launch_unchecked)]
pub fn compute_scores_batch_kernel<F: Float>(
    // Source points [N × 3]
    source_points: &Array<F>,
    // Transform matrices [M × 16] - one 4x4 matrix per pose
    transforms: &Array<F>,
    // Voxel means [V × 3]
    voxel_means: &Array<F>,
    // Voxel inverse covariances [V × 9]
    voxel_inv_covs: &Array<F>,
    // Voxel validity flags [V]
    voxel_valid: &Array<u32>,
    // Gaussian params: [d1, d2, search_radius_sq]
    gauss_params: &Array<F>,
    // Number of poses
    num_poses: u32,
    // Number of points
    num_points: u32,
    // Number of voxels
    num_voxels: u32,
    // Output: [M × N × 4] = [score, max_score, has_neighbor, correspondences]
    outputs: &mut Array<F>,
) {
    // 2D grid: (pose_idx, point_idx)
    // pose_idx = CUBE_POS_X
    // point_idx = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X
    let pose_idx = CUBE_POS_X;
    let point_idx = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;

    if pose_idx >= num_poses || point_idx >= num_points {
        terminate!();
    }

    // Load Gaussian parameters
    let gauss_d1 = gauss_params[0];
    let gauss_d2 = gauss_params[1];
    let search_radius_sq = gauss_params[2];

    // Load source point
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    // Load transform for this pose (16 elements starting at pose_idx * 16)
    let tbase = pose_idx * 16;

    // Manual transform since we need to index into array
    let tx = transforms[tbase] * sx
        + transforms[tbase + 1] * sy
        + transforms[tbase + 2] * sz
        + transforms[tbase + 3];
    let ty = transforms[tbase + 4] * sx
        + transforms[tbase + 5] * sy
        + transforms[tbase + 6] * sz
        + transforms[tbase + 7];
    let tz = transforms[tbase + 8] * sx
        + transforms[tbase + 9] * sy
        + transforms[tbase + 10] * sz
        + transforms[tbase + 11];

    // Accumulate scores across all neighbors
    let mut total_score = F::new(0.0);
    let mut max_score = F::new(0.0);
    let mut found_neighbor = F::new(0.0);
    let mut num_correspondences = F::new(0.0);

    // Max neighbors limit (same as MAX_NEIGHBORS = 8)
    let max_neighbors_limit = F::new(8.0);

    // Brute-force neighbor search: check all voxels within radius
    // Note: We avoid using `break` due to CubeCL optimizer bug
    for v in 0..num_voxels {
        // Only process if we haven't exceeded a reasonable limit
        let should_process = num_correspondences < max_neighbors_limit;

        if should_process {
            let is_valid = voxel_valid[v];
            if is_valid != 0u32 {
                let vbase = v * 3;
                let vx = voxel_means[vbase];
                let vy = voxel_means[vbase + 1];
                let vz = voxel_means[vbase + 2];

                // Distance check
                let dx = tx - vx;
                let dy = ty - vy;
                let dz = tz - vz;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq <= search_radius_sq {
                    // Compute x_trans = transformed - mean
                    let x0 = tx - vx;
                    let x1 = ty - vy;
                    let x2 = tz - vz;

                    // Load inverse covariance (row-major 3x3)
                    let cbase = v * 9;
                    let c00 = voxel_inv_covs[cbase];
                    let c01 = voxel_inv_covs[cbase + 1];
                    let c02 = voxel_inv_covs[cbase + 2];
                    let c10 = voxel_inv_covs[cbase + 3];
                    let c11 = voxel_inv_covs[cbase + 4];
                    let c12 = voxel_inv_covs[cbase + 5];
                    let c20 = voxel_inv_covs[cbase + 6];
                    let c21 = voxel_inv_covs[cbase + 7];
                    let c22 = voxel_inv_covs[cbase + 8];

                    // Compute x' * Σ⁻¹ * x
                    let cx0 = c00 * x0 + c01 * x1 + c02 * x2;
                    let cx1 = c10 * x0 + c11 * x1 + c12 * x2;
                    let cx2 = c20 * x0 + c21 * x1 + c22 * x2;
                    let x_c_x = x0 * cx0 + x1 * cx1 + x2 * cx2;

                    // Score: -d1 * exp(-d2/2 * x'Σ⁻¹x)
                    let exponent = gauss_d2 * x_c_x * F::new(-0.5);
                    let score = gauss_d1 * F::new(-1.0) * F::exp(exponent);

                    // Accumulate for transform probability
                    total_score += score;

                    // Track maximum for NVTL
                    if found_neighbor == F::new(0.0) || score > max_score {
                        max_score = score;
                    }

                    found_neighbor = F::new(1.0);
                    num_correspondences += F::new(1.0);
                }
            }
        }
    }

    // Column-major output with 4 values per (pose, point)
    let out_base = (pose_idx * num_points + point_idx) * 4;
    outputs[out_base] = total_score;
    outputs[out_base + 1] = max_score;
    outputs[out_base + 2] = found_neighbor;
    outputs[out_base + 3] = num_correspondences;
}

/// Convert a pose [x, y, z, roll, pitch, yaw] to a 4x4 transform matrix.
///
/// This matches the CPU implementation in `derivatives/gpu.rs::pose_to_transform_matrix`.
pub fn pose_to_transform_matrix_f32(pose: &[f64; 6]) -> [f32; 16] {
    let (x, y, z, roll, pitch, yaw) = (pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]);

    let cr = roll.cos();
    let sr = roll.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let cy = yaw.cos();
    let sy = yaw.sin();

    // Rotation matrix R = Rx(roll) * Ry(pitch) * Rz(yaw)
    // This matches Autoware's convention used in:
    // - eulerAngles(0, 1, 2) extraction
    // - AngleAxis composition: Translation * Rx * Ry * Rz
    // - Jacobian/Hessian angular derivatives (j_ang, h_ang)
    let r00 = cp * cy;
    let r01 = -cp * sy;
    let r02 = sp;
    let r10 = sr * sp * cy + cr * sy;
    let r11 = cr * cy - sr * sp * sy;
    let r12 = -sr * cp;
    let r20 = sr * sy - cr * sp * cy;
    let r21 = cr * sp * sy + sr * cy;
    let r22 = cr * cp;

    // Row-major 4x4 matrix
    [
        r00 as f32, r01 as f32, r02 as f32, x as f32, r10 as f32, r11 as f32, r12 as f32, y as f32,
        r20 as f32, r21 as f32, r22 as f32, z as f32, 0.0, 0.0, 0.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose_to_transform_identity() {
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let matrix = pose_to_transform_matrix_f32(&pose);

        // Should be identity matrix
        assert!((matrix[0] - 1.0).abs() < 1e-6);
        assert!((matrix[5] - 1.0).abs() < 1e-6);
        assert!((matrix[10] - 1.0).abs() < 1e-6);
        assert!((matrix[15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pose_to_transform_translation() {
        let pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let matrix = pose_to_transform_matrix_f32(&pose);

        assert!((matrix[3] - 1.0).abs() < 1e-6);
        assert!((matrix[7] - 2.0).abs() < 1e-6);
        assert!((matrix[11] - 3.0).abs() < 1e-6);
    }
}
