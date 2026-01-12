//! GPU kernels for Phase 15: True Full GPU Newton Pipeline.
//!
//! This module contains kernels that enable zero-transfer Newton iterations:
//! - Transform matrix computation on GPU (15.1)
//! - Hessian kernel with separate buffers (15.2)
//! - Directional derivative (dot product) kernel (15.4)
//! - Candidate generation kernel (15.5)
//! - Batch transform kernel for line search (15.6)
//! - Batch score/gradient kernel for line search (15.7)
//! - More-Thuente logic kernel (15.8)

use cubecl::prelude::*;

/// Maximum number of neighbors per point (used in batch score/gradient kernel).
pub const MAX_NEIGHBORS: u32 = 8;

/// Default number of line search candidates.
pub const DEFAULT_NUM_CANDIDATES: u32 = 8;

// ============================================================================
// Utility: U32 to F32 Conversion Kernel
// ============================================================================

/// Convert u32 values to f32 for reduction.
///
/// This is needed because CUB segmented reduce only supports f32/f64.
///
/// # Arguments
/// * `input` - [N] u32 values
/// * `num_elements` - Number of elements
/// * `output` - [N] f32 values
#[cube(launch_unchecked)]
pub fn cast_u32_to_f32_kernel<F: Float>(
    input: &Array<u32>,
    num_elements: u32,
    output: &mut Array<F>,
) {
    let i = ABSOLUTE_POS;

    if i >= num_elements {
        terminate!();
    }

    output[i] = F::cast_from(input[i]);
}

// ============================================================================
// Phase 15.1: GPU Transform Matrix Kernel
// ============================================================================

/// Compute 4x4 transform matrix from sin/cos values on GPU.
///
/// This eliminates the need to compute the transform matrix on CPU and upload it.
///
/// # Arguments
/// * `sin_cos` - [6]: sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw)
/// * `pose` - [6]: tx, ty, tz, roll, pitch, yaw
/// * `transform` - [16] output: 4x4 row-major transform matrix
///
/// # Rotation Convention
/// Uses ZYX Euler angles: Rz(yaw) * Ry(pitch) * Rx(roll)
#[cube(launch_unchecked)]
pub fn compute_transform_from_sincos_kernel<F: Float>(
    sin_cos: &Array<F>,
    pose: &Array<F>,
    transform: &mut Array<F>,
) {
    // Single thread kernel
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    let sr = sin_cos[0]; // sin(roll)
    let cr = sin_cos[1]; // cos(roll)
    let sp = sin_cos[2]; // sin(pitch)
    let cp = sin_cos[3]; // cos(pitch)
    let sy = sin_cos[4]; // sin(yaw)
    let cy = sin_cos[5]; // cos(yaw)

    // Rotation matrix: Rz(yaw) * Ry(pitch) * Rx(roll)
    // Row 0
    transform[0] = cy * cp;
    transform[1] = cy * sp * sr - sy * cr;
    transform[2] = cy * sp * cr + sy * sr;
    transform[3] = pose[0]; // tx

    // Row 1
    transform[4] = sy * cp;
    transform[5] = sy * sp * sr + cy * cr;
    transform[6] = sy * sp * cr - cy * sr;
    transform[7] = pose[1]; // ty

    // Row 2
    transform[8] = F::new(0.0) - sp;
    transform[9] = cp * sr;
    transform[10] = cp * cr;
    transform[11] = pose[2]; // tz

    // Row 3 (homogeneous)
    transform[12] = F::new(0.0);
    transform[13] = F::new(0.0);
    transform[14] = F::new(0.0);
    transform[15] = F::new(1.0);
}

// ============================================================================
// Phase 15.4: Directional Derivative Kernel (Dot Product)
// ============================================================================

/// Compute dot product of two 6-element vectors.
///
/// Used for computing directional derivative: φ'(0) = g · δ
///
/// # Arguments
/// * `a` - [6] first vector (e.g., gradient)
/// * `b` - [6] second vector (e.g., delta/direction)
/// * `result` - [1] output: a · b
#[cube(launch_unchecked)]
pub fn dot_product_6_kernel<F: Float>(a: &Array<F>, b: &Array<F>, result: &mut Array<F>) {
    // Single thread kernel
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    result[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5];
}

// ============================================================================
// Phase 15.5: Candidate Generation Kernel
// ============================================================================

/// Generate K candidate step sizes for batched line search evaluation.
///
/// Uses a geometric progression with golden ratio points for good coverage:
/// - Ratios: [1.0, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625]
///
/// # Arguments
/// * `initial_step` - Base step size (typically 1.0 for Newton direction)
/// * `step_min` - Minimum allowed step size
/// * `step_max` - Maximum allowed step size
/// * `num_candidates` - Number of candidates to generate (K)
/// * `candidates` - [K] output: candidate step sizes
#[cube(launch_unchecked)]
pub fn generate_candidates_kernel<F: Float>(
    initial_step: F,
    step_min: F,
    step_max: F,
    num_candidates: u32,
    candidates: &mut Array<F>,
) {
    let k = ABSOLUTE_POS;

    if k >= num_candidates {
        terminate!();
    }

    // Pre-defined ratios for good coverage of the step space
    // Includes geometric series + golden ratio points (0.618, 0.382)
    let ratio = select_ratio::<F>(k);

    let alpha = initial_step * ratio;

    // Clamp to valid range
    let clamped = if alpha < step_min {
        step_min
    } else if alpha > step_max {
        step_max
    } else {
        alpha
    };

    candidates[k] = clamped;
}

/// Select ratio based on index (unrolled for CubeCL compatibility).
#[cube]
fn select_ratio<F: Float>(k: u32) -> F {
    // CubeCL doesn't support match on runtime values in all cases,
    // so we use if-else chain
    if k == 0u32 {
        F::new(1.0)
    } else if k == 1u32 {
        F::new(0.75)
    } else if k == 2u32 {
        F::new(0.5)
    } else if k == 3u32 {
        F::new(0.25)
    } else if k == 4u32 {
        F::new(0.125)
    } else if k == 5u32 {
        F::new(0.618) // Golden ratio
    } else if k == 6u32 {
        F::new(0.382) // 1 - golden ratio
    } else {
        F::new(0.0625)
    }
}

// ============================================================================
// Phase 15.9: Pose Update Kernel
// ============================================================================

/// Update pose in-place on GPU: pose += alpha * delta
///
/// # Arguments
/// * `pose` - [6] pose to update (modified in place)
/// * `delta` - [6] Newton direction
/// * `alpha` - [1] step size
#[cube(launch_unchecked)]
pub fn update_pose_kernel<F: Float>(pose: &mut Array<F>, delta: &Array<F>, alpha: &Array<F>) {
    let i = ABSOLUTE_POS;

    if i >= 6 {
        terminate!();
    }

    pose[i] = pose[i] + alpha[0] * delta[i];
}

// ============================================================================
// Phase 15.10: Convergence Check Kernel
// ============================================================================

/// Check convergence on GPU: ||alpha * delta|| < epsilon
///
/// # Arguments
/// * `delta` - [6] Newton direction
/// * `alpha` - [1] step size
/// * `epsilon_sq` - Convergence threshold squared
/// * `converged` - [1] output: 1 if converged, 0 otherwise
#[cube(launch_unchecked)]
pub fn check_convergence_kernel<F: Float>(
    delta: &Array<F>,
    alpha: &Array<F>,
    epsilon_sq: F,
    converged: &mut Array<u32>,
) {
    // Single thread kernel
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    let a = alpha[0];

    // Compute ||alpha * delta||²
    let step_sq = (a * delta[0]) * (a * delta[0])
        + (a * delta[1]) * (a * delta[1])
        + (a * delta[2]) * (a * delta[2])
        + (a * delta[3]) * (a * delta[3])
        + (a * delta[4]) * (a * delta[4])
        + (a * delta[5]) * (a * delta[5]);

    if step_sq < epsilon_sq {
        converged[0] = 1u32;
    } else {
        converged[0] = 0u32;
    }
}

// ============================================================================
// GNSS Regularization Kernel
// ============================================================================

/// Apply GNSS regularization penalty to score, gradient, and Hessian.
///
/// This kernel adds a quadratic penalty in the vehicle's longitudinal direction,
/// penalizing deviation from a reference GNSS pose.
///
/// The regularization term is:
/// - Score: -scale × weight × longitudinal_distance²
/// - Gradient[0]: scale × weight × 2 × cos(yaw) × longitudinal_distance
/// - Gradient[1]: scale × weight × 2 × sin(yaw) × longitudinal_distance
/// - Hessian[0,0]: -scale × weight × 2 × cos²(yaw)
/// - Hessian[0,1] = Hessian[1,0]: -scale × weight × 2 × cos(yaw) × sin(yaw)
/// - Hessian[1,1]: -scale × weight × 2 × sin²(yaw)
///
/// Where:
/// - longitudinal_distance = (ref_y - y) × sin(yaw) + (ref_x - x) × cos(yaw)
/// - weight = correspondence_sum (number of point-voxel correspondences)
///
/// # Arguments
/// * `pose` - [6]: tx, ty, tz, roll, pitch, yaw
/// * `reg_params` - [4]: ref_x, ref_y, scale_factor, enabled (0.0 or 1.0)
/// * `correspondence_sum` - [1]: sum of correspondences (weight)
/// * `reduce_output` - [43] in/out: score[1] + gradient[6] + hessian[36]
#[cube(launch_unchecked)]
pub fn apply_regularization_kernel<F: Float>(
    pose: &Array<F>,
    reg_params: &Array<F>,
    correspondence_sum: &Array<F>,
    reduce_output: &mut Array<F>,
) {
    // Single thread kernel
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    let enabled = reg_params[3];

    // Early exit if not enabled (enabled == 0.0)
    if enabled < F::new(0.5) {
        terminate!();
    }

    let ref_x = reg_params[0];
    let ref_y = reg_params[1];
    let scale = reg_params[2];
    let weight = correspondence_sum[0];

    // Current pose
    let x = pose[0];
    let y = pose[1];
    let yaw = pose[5];

    // Compute sin/cos of yaw
    let sin_yaw = F::sin(yaw);
    let cos_yaw = F::cos(yaw);

    // Difference from reference
    let dx = ref_x - x;
    let dy = ref_y - y;

    // Longitudinal distance in vehicle frame
    let longitudinal = dy * sin_yaw + dx * cos_yaw;

    // Score delta: -scale × weight × distance²
    let score_delta = F::new(0.0) - scale * weight * longitudinal * longitudinal;

    // Gradient deltas
    let two = F::new(2.0);
    let grad_x_delta = scale * weight * two * cos_yaw * longitudinal;
    let grad_y_delta = scale * weight * two * sin_yaw * longitudinal;

    // Hessian deltas (note: these are negative because the score is negative)
    let h00_delta = F::new(0.0) - scale * weight * two * cos_yaw * cos_yaw;
    let h01_delta = F::new(0.0) - scale * weight * two * cos_yaw * sin_yaw;
    let h11_delta = F::new(0.0) - scale * weight * two * sin_yaw * sin_yaw;

    // Add to reduce_output
    // reduce_output layout: [score, grad[6], hess[36]] (row-major Hessian)
    reduce_output[0] = reduce_output[0] + score_delta; // score
    reduce_output[1] = reduce_output[1] + grad_x_delta; // gradient[0]
    reduce_output[2] = reduce_output[2] + grad_y_delta; // gradient[1]

    // Hessian is row-major 6×6, so:
    // H[0,0] is at index 7+0 = 7
    // H[0,1] is at index 7+1 = 8
    // H[1,0] is at index 7+6 = 13
    // H[1,1] is at index 7+7 = 14
    reduce_output[7] = reduce_output[7] + h00_delta; // H[0,0]
    reduce_output[8] = reduce_output[8] + h01_delta; // H[0,1]
    reduce_output[13] = reduce_output[13] + h01_delta; // H[1,0]
    reduce_output[14] = reduce_output[14] + h11_delta; // H[1,1]
}

// ============================================================================
// Phase 15.6: Batch Transform Kernel
// ============================================================================

/// Transform all source points for K candidate step sizes in parallel.
///
/// Each thread handles one (candidate, point) pair, computing:
/// `transformed[k][i] = R(pose + candidates[k] * delta) * source[i] + t(pose + candidates[k] * delta)`
///
/// Parallelism: K × N threads.
///
/// # Arguments
/// * `source_points` - [N × 3] source point cloud
/// * `pose` - [6] current pose (tx, ty, tz, roll, pitch, yaw)
/// * `delta` - [6] Newton direction
/// * `candidates` - [K] candidate step sizes
/// * `num_points` - N, number of source points
/// * `num_candidates` - K, number of candidates
/// * `batch_transformed` - [K × N × 3] output: transformed points for each candidate
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn batch_transform_kernel<F: Float>(
    source_points: &Array<F>,
    pose: &Array<F>,
    delta: &Array<F>,
    candidates: &Array<F>,
    num_points: u32,
    num_candidates: u32,
    batch_transformed: &mut Array<F>,
) {
    let idx = ABSOLUTE_POS;
    let k = idx / num_points;
    let i = idx % num_points;

    if k >= num_candidates {
        terminate!();
    }

    let alpha = candidates[k];

    // Compute trial pose: pose + alpha * delta
    let tx = pose[0] + alpha * delta[0];
    let ty = pose[1] + alpha * delta[1];
    let tz = pose[2] + alpha * delta[2];
    let roll = pose[3] + alpha * delta[3];
    let pitch = pose[4] + alpha * delta[4];
    let yaw = pose[5] + alpha * delta[5];

    // Compute sin/cos for trial pose
    let sr = F::sin(roll);
    let cr = F::cos(roll);
    let sp = F::sin(pitch);
    let cp = F::cos(pitch);
    let sy = F::sin(yaw);
    let cy = F::cos(yaw);

    // Build rotation matrix: Rz(yaw) * Ry(pitch) * Rx(roll)
    let r00 = cy * cp;
    let r01 = cy * sp * sr - sy * cr;
    let r02 = cy * sp * cr + sy * sr;
    let r10 = sy * cp;
    let r11 = sy * sp * sr + cy * cr;
    let r12 = sy * sp * cr - cy * sr;
    let r20 = F::new(0.0) - sp;
    let r21 = cp * sr;
    let r22 = cp * cr;

    // Read source point
    let px = source_points[i * 3];
    let py = source_points[i * 3 + 1];
    let pz = source_points[i * 3 + 2];

    // Transform point: R * p + t
    let out_idx = (k * num_points + i) * 3;
    batch_transformed[out_idx] = r00 * px + r01 * py + r02 * pz + tx;
    batch_transformed[out_idx + 1] = r10 * px + r11 * py + r12 * pz + ty;
    batch_transformed[out_idx + 2] = r20 * px + r21 * py + r22 * pz + tz;
}

// ============================================================================
// Phase 15.7: Batch Score/Gradient Kernel
// ============================================================================

/// Compute NDT score and directional derivative for K candidate step sizes.
///
/// For line search, we need φ(α) = score and φ'(α) = gradient · delta.
/// This kernel computes per-point contributions that will be reduced.
///
/// Parallelism: K × N threads.
///
/// # Arguments
/// * `batch_transformed` - [K × N × 3] transformed points for each candidate
/// * `delta` - [6] Newton direction (for directional derivative)
/// * `jacobians` - [N × 18] pose-to-point Jacobians (shared across candidates)
/// * `voxel_means` - [V × 3] voxel centroids
/// * `voxel_inv_covs` - [V × 9] inverse covariance matrices (row-major)
/// * `neighbor_indices` - [N × MAX_NEIGHBORS] neighbor voxel indices per point
/// * `neighbor_counts` - [N] number of neighbors per point
/// * `params` - [4]: gauss_d1, gauss_d2, num_points (as f32), num_candidates (as f32)
/// * `batch_scores` - [K × N] output: per-point score for each candidate
/// * `batch_dir_derivs` - [K × N] output: per-point directional derivative
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn batch_score_gradient_kernel<F: Float>(
    batch_transformed: &Array<F>,
    delta: &Array<F>,
    jacobians: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    neighbor_indices: &Array<i32>,
    neighbor_counts: &Array<u32>,
    params: &Array<F>,
    batch_scores: &mut Array<F>,
    batch_dir_derivs: &mut Array<F>,
) {
    // Extract params: [gauss_d1, gauss_d2, num_points, num_candidates]
    let gauss_d1 = params[0];
    let gauss_d2 = params[1];
    let num_points = u32::cast_from(params[2]);
    let num_candidates = u32::cast_from(params[3]);

    let idx = ABSOLUTE_POS;
    let k = idx / num_points;
    let i = idx % num_points;

    if k >= num_candidates {
        terminate!();
    }

    // Get transformed point for this candidate
    let tp_idx = (k * num_points + i) * 3;
    let tx = batch_transformed[tp_idx];
    let ty = batch_transformed[tp_idx + 1];
    let tz = batch_transformed[tp_idx + 2];

    // Read Jacobians for this point (shared across candidates)
    // Jacobian layout: [j_tx, j_ty, j_tz] for each of 6 pose params
    // j_tx[0..6], j_ty[0..6], j_tz[0..6] = 18 values
    let j_base = i * 18;
    let j_tx_0 = jacobians[j_base];
    let j_tx_1 = jacobians[j_base + 1];
    let j_tx_2 = jacobians[j_base + 2];
    let j_tx_3 = jacobians[j_base + 3];
    let j_tx_4 = jacobians[j_base + 4];
    let j_tx_5 = jacobians[j_base + 5];
    let j_ty_0 = jacobians[j_base + 6];
    let j_ty_1 = jacobians[j_base + 7];
    let j_ty_2 = jacobians[j_base + 8];
    let j_ty_3 = jacobians[j_base + 9];
    let j_ty_4 = jacobians[j_base + 10];
    let j_ty_5 = jacobians[j_base + 11];
    let j_tz_0 = jacobians[j_base + 12];
    let j_tz_1 = jacobians[j_base + 13];
    let j_tz_2 = jacobians[j_base + 14];
    let j_tz_3 = jacobians[j_base + 15];
    let j_tz_4 = jacobians[j_base + 16];
    let j_tz_5 = jacobians[j_base + 17];

    // Accumulate score and gradient across neighbors
    let mut score_sum = F::new(0.0);
    let mut g0 = F::new(0.0);
    let mut g1 = F::new(0.0);
    let mut g2 = F::new(0.0);
    let mut g3 = F::new(0.0);
    let mut g4 = F::new(0.0);
    let mut g5 = F::new(0.0);

    let num_neighbors = neighbor_counts[i];
    let neighbor_base = i * MAX_NEIGHBORS;

    // Unrolled neighbor loop (up to MAX_NEIGHBORS = 8)
    // Neighbor 0
    if 0u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Neighbor 1
    if 1u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base + 1];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Neighbor 2
    if 2u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base + 2];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Neighbor 3
    if 3u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base + 3];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Neighbor 4
    if 4u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base + 4];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Neighbor 5
    if 5u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base + 5];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Neighbor 6
    if 6u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base + 6];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Neighbor 7
    if 7u32 < num_neighbors {
        let v_idx = neighbor_indices[neighbor_base + 7];
        if v_idx >= 0 {
            let (s, gr0, gr1, gr2, gr3, gr4, gr5) = compute_score_gradient_contribution::<F>(
                tx,
                ty,
                tz,
                v_idx as u32,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                j_tx_0,
                j_tx_1,
                j_tx_2,
                j_tx_3,
                j_tx_4,
                j_tx_5,
                j_ty_0,
                j_ty_1,
                j_ty_2,
                j_ty_3,
                j_ty_4,
                j_ty_5,
                j_tz_0,
                j_tz_1,
                j_tz_2,
                j_tz_3,
                j_tz_4,
                j_tz_5,
            );
            score_sum += s;
            g0 += gr0;
            g1 += gr1;
            g2 += gr2;
            g3 += gr3;
            g4 += gr4;
            g5 += gr5;
        }
    }

    // Write score
    let out_idx = k * num_points + i;
    batch_scores[out_idx] = score_sum;

    // Compute directional derivative: grad · delta
    let dir_deriv = g0 * delta[0]
        + g1 * delta[1]
        + g2 * delta[2]
        + g3 * delta[3]
        + g4 * delta[4]
        + g5 * delta[5];
    batch_dir_derivs[out_idx] = dir_deriv;
}

/// Compute score and gradient contribution from a single voxel.
#[cube]
#[allow(clippy::too_many_arguments)]
fn compute_score_gradient_contribution<F: Float>(
    tx: F,
    ty: F,
    tz: F,
    v_idx: u32,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    gauss_d1: F,
    gauss_d2: F,
    j_tx_0: F,
    j_tx_1: F,
    j_tx_2: F,
    j_tx_3: F,
    j_tx_4: F,
    j_tx_5: F,
    j_ty_0: F,
    j_ty_1: F,
    j_ty_2: F,
    j_ty_3: F,
    j_ty_4: F,
    j_ty_5: F,
    j_tz_0: F,
    j_tz_1: F,
    j_tz_2: F,
    j_tz_3: F,
    j_tz_4: F,
    j_tz_5: F,
) -> (F, F, F, F, F, F, F) {
    // Read voxel mean
    let mean_base = v_idx * 3;
    let mx = voxel_means[mean_base];
    let my = voxel_means[mean_base + 1];
    let mz = voxel_means[mean_base + 2];

    // Compute difference: x - μ
    let dx = tx - mx;
    let dy = ty - my;
    let dz = tz - mz;

    // Read inverse covariance (row-major 3x3)
    let cov_base = v_idx * 9;
    let s00 = voxel_inv_covs[cov_base];
    let s01 = voxel_inv_covs[cov_base + 1];
    let s02 = voxel_inv_covs[cov_base + 2];
    let s10 = voxel_inv_covs[cov_base + 3];
    let s11 = voxel_inv_covs[cov_base + 4];
    let s12 = voxel_inv_covs[cov_base + 5];
    let s20 = voxel_inv_covs[cov_base + 6];
    let s21 = voxel_inv_covs[cov_base + 7];
    let s22 = voxel_inv_covs[cov_base + 8];

    // Compute Σ⁻¹(x - μ)
    let q0 = s00 * dx + s01 * dy + s02 * dz;
    let q1 = s10 * dx + s11 * dy + s12 * dz;
    let q2 = s20 * dx + s21 * dy + s22 * dz;

    // Compute Mahalanobis distance: (x-μ)ᵀ Σ⁻¹ (x-μ)
    let mahal = dx * q0 + dy * q1 + dz * q2;

    // Score: -d1 * exp(-d2/2 * mahal)
    let half = F::new(0.5);
    let exp_term = F::exp(F::new(0.0) - half * gauss_d2 * mahal);
    let score = F::new(0.0) - gauss_d1 * exp_term;

    // Gradient: d1 * d2 * exp(...) * Σ⁻¹(x-μ) · J
    // For each pose parameter p_i: ∂score/∂p_i = d1*d2*exp(...) * q · ∂T/∂p_i
    let coeff = gauss_d1 * gauss_d2 * exp_term;

    // Gradient components: coeff * (q · J_col[i])
    let g0 = coeff * (q0 * j_tx_0 + q1 * j_ty_0 + q2 * j_tz_0);
    let g1 = coeff * (q0 * j_tx_1 + q1 * j_ty_1 + q2 * j_tz_1);
    let g2 = coeff * (q0 * j_tx_2 + q1 * j_ty_2 + q2 * j_tz_2);
    let g3 = coeff * (q0 * j_tx_3 + q1 * j_ty_3 + q2 * j_tz_3);
    let g4 = coeff * (q0 * j_tx_4 + q1 * j_ty_4 + q2 * j_tz_4);
    let g5 = coeff * (q0 * j_tx_5 + q1 * j_ty_5 + q2 * j_tz_5);

    (score, g0, g1, g2, g3, g4, g5)
}

// ============================================================================
// Phase 15.8: More-Thuente Logic Kernel
// ============================================================================

/// More-Thuente line search using pre-computed candidate evaluations.
///
/// This is a single-thread kernel that runs the sequential More-Thuente algorithm
/// using cached function evaluations from the batched kernels.
///
/// # Arguments
/// * `phi_0` - [1] Score at current pose (φ(0))
/// * `dphi_0` - [1] Directional derivative at current pose (φ'(0))
/// * `candidates` - [K] candidate step sizes
/// * `cached_phi` - [K] reduced scores for each candidate
/// * `cached_dphi` - [K] reduced directional derivatives for each candidate
/// * `ls_params` - [3]: num_candidates (as f32), mu, nu
/// * `best_alpha` - [1] output: best step size
/// * `ls_converged` - [1] output: 1.0 if Wolfe conditions satisfied, 0.0 otherwise
#[cube(launch_unchecked)]
pub fn more_thuente_kernel<F: Float>(
    phi_0: &Array<F>,
    dphi_0: &Array<F>,
    candidates: &Array<F>,
    cached_phi: &Array<F>,
    cached_dphi: &Array<F>,
    ls_params: &Array<F>,
    best_alpha: &mut Array<F>,
    ls_converged: &mut Array<F>,
) {
    // Single thread kernel
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    // Extract params: [num_candidates, mu, nu]
    let num_candidates = u32::cast_from(ls_params[0]);
    let mu = ls_params[1];
    let nu = ls_params[2];

    let f0 = phi_0[0];
    let g0 = dphi_0[0];

    // First, check if any candidate satisfies Strong Wolfe conditions
    // Look through candidates in order of step size (largest first is at index 0)
    let mut best_k: u32 = 0u32;
    let mut found_wolfe: u32 = 0u32; // Use u32 instead of bool for CubeCL compatibility

    // Check each candidate for Wolfe conditions
    // Unrolled loop for CubeCL compatibility
    if found_wolfe == 0u32 && 0u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            0u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 0u32;
            found_wolfe = 1u32;
        }
    }
    if found_wolfe == 0u32 && 1u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            1u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 1u32;
            found_wolfe = 1u32;
        }
    }
    if found_wolfe == 0u32 && 2u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            2u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 2u32;
            found_wolfe = 1u32;
        }
    }
    if found_wolfe == 0u32 && 3u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            3u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 3u32;
            found_wolfe = 1u32;
        }
    }
    if found_wolfe == 0u32 && 4u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            4u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 4u32;
            found_wolfe = 1u32;
        }
    }
    if found_wolfe == 0u32 && 5u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            5u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 5u32;
            found_wolfe = 1u32;
        }
    }
    if found_wolfe == 0u32 && 6u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            6u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 6u32;
            found_wolfe = 1u32;
        }
    }
    if found_wolfe == 0u32 && 7u32 < num_candidates {
        let ok = check_wolfe_at_candidate::<F>(
            7u32,
            candidates,
            cached_phi,
            cached_dphi,
            f0,
            g0,
            mu,
            nu,
        );
        if ok == 1u32 {
            best_k = 7u32;
            found_wolfe = 1u32;
        }
    }

    if found_wolfe == 1u32 {
        best_alpha[0] = candidates[best_k];
        ls_converged[0] = F::new(1.0);
        terminate!();
    }

    // No candidate satisfies Wolfe conditions exactly
    // Fall back to finding the candidate with lowest score that satisfies Armijo
    let mut best_score: F = F::new(1e10);
    let mut armijo_found: u32 = 0u32;

    // Unrolled search for best Armijo-satisfying candidate
    if 0u32 < num_candidates {
        let alpha_k = candidates[0u32];
        let phi_k = cached_phi[0u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 0u32;
            armijo_found = 1u32;
        }
    }
    if 1u32 < num_candidates {
        let alpha_k = candidates[1u32];
        let phi_k = cached_phi[1u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 1u32;
            armijo_found = 1u32;
        }
    }
    if 2u32 < num_candidates {
        let alpha_k = candidates[2u32];
        let phi_k = cached_phi[2u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 2u32;
            armijo_found = 1u32;
        }
    }
    if 3u32 < num_candidates {
        let alpha_k = candidates[3u32];
        let phi_k = cached_phi[3u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 3u32;
            armijo_found = 1u32;
        }
    }
    if 4u32 < num_candidates {
        let alpha_k = candidates[4u32];
        let phi_k = cached_phi[4u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 4u32;
            armijo_found = 1u32;
        }
    }
    if 5u32 < num_candidates {
        let alpha_k = candidates[5u32];
        let phi_k = cached_phi[5u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 5u32;
            armijo_found = 1u32;
        }
    }
    if 6u32 < num_candidates {
        let alpha_k = candidates[6u32];
        let phi_k = cached_phi[6u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 6u32;
            armijo_found = 1u32;
        }
    }
    if 7u32 < num_candidates {
        let alpha_k = candidates[7u32];
        let phi_k = cached_phi[7u32];
        let armijo_bound = f0 + mu * alpha_k * g0;
        if phi_k <= armijo_bound && phi_k < best_score {
            best_score = phi_k;
            best_k = 7u32;
            armijo_found = 1u32;
        }
    }

    if armijo_found == 1u32 {
        best_alpha[0] = candidates[best_k];
        ls_converged[0] = F::new(0.0); // Armijo only, not full Wolfe
    } else {
        // No candidate satisfies even Armijo - use smallest step
        // Find smallest candidate (should be at index with smallest ratio)
        best_alpha[0] = candidates[7u32]; // 0.0625 ratio is smallest
        ls_converged[0] = F::new(0.0);
    }

    // Suppress unused variable warning
    let _ = best_score;
}

/// Check Strong Wolfe conditions at a candidate.
/// Returns 1u32 if both Armijo and curvature conditions are satisfied, 0u32 otherwise.
#[cube]
#[allow(clippy::too_many_arguments)]
fn check_wolfe_at_candidate<F: Float>(
    k: u32,
    candidates: &Array<F>,
    cached_phi: &Array<F>,
    cached_dphi: &Array<F>,
    f0: F,
    g0: F,
    mu: F,
    nu: F,
) -> u32 {
    let alpha_k = candidates[k];
    let phi_k = cached_phi[k];
    let dphi_k = cached_dphi[k];

    // Armijo (sufficient decrease): φ(α) ≤ φ(0) + μ·α·φ'(0)
    let armijo_bound = f0 + mu * alpha_k * g0;
    let armijo_ok = phi_k <= armijo_bound;

    // Strong curvature: |φ'(α)| ≤ ν·|φ'(0)|
    let abs_dphi_k = if dphi_k < F::new(0.0) {
        F::new(0.0) - dphi_k
    } else {
        dphi_k
    };
    let abs_g0 = if g0 < F::new(0.0) {
        F::new(0.0) - g0
    } else {
        g0
    };
    let curvature_bound = nu * abs_g0;
    let curvature_ok = abs_dphi_k <= curvature_bound;

    let mut result: u32 = 0u32;
    if armijo_ok && curvature_ok {
        result = 1u32;
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    fn get_test_client() -> (
        CudaDevice,
        cubecl::client::ComputeClient<<CudaRuntime as cubecl::Runtime>::Server>,
    ) {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);
        (device, client)
    }

    /// CPU reference for transform matrix computation.
    fn compute_transform_cpu(pose: &[f64; 6]) -> [f32; 16] {
        let (sr, cr) = (pose[3] as f32).sin_cos();
        let (sp, cp) = (pose[4] as f32).sin_cos();
        let (sy, cy) = (pose[5] as f32).sin_cos();

        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            pose[0] as f32,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            pose[1] as f32,
            -sp,
            cp * sr,
            cp * cr,
            pose[2] as f32,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }

    #[test]
    fn test_transform_kernel_matches_cpu() {
        let (_device, client) = get_test_client();

        // Test poses
        let test_poses: Vec<[f64; 6]> = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      // Identity
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],      // Translation only
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],      // Rotation only
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],      // Full transform
            [-1.0, 0.5, -0.3, -0.2, 0.15, -0.1], // Negative values
            [
                10.0,
                20.0,
                30.0,
                std::f64::consts::FRAC_PI_2,
                0.0,
                std::f64::consts::PI,
            ], // Large rotation
        ];

        for pose in &test_poses {
            // Compute CPU reference
            let cpu_transform = compute_transform_cpu(pose);

            // Prepare GPU inputs
            let pose_f32: [f32; 6] = pose.map(|x| x as f32);
            let (sr, cr) = pose_f32[3].sin_cos();
            let (sp, cp) = pose_f32[4].sin_cos();
            let (sy, cy) = pose_f32[5].sin_cos();
            let sin_cos: [f32; 6] = [sr, cr, sp, cp, sy, cy];

            let d_sin_cos = client.create(f32::as_bytes(&sin_cos));
            let d_pose = client.create(f32::as_bytes(&pose_f32));
            let d_transform = client.empty(16 * std::mem::size_of::<f32>());

            // Launch kernel
            unsafe {
                compute_transform_from_sincos_kernel::launch_unchecked::<f32, CudaRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&d_sin_cos, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&d_pose, 6, 1),
                    ArrayArg::from_raw_parts::<f32>(&d_transform, 16, 1),
                );
            }

            // Download and compare
            let gpu_transform_bytes = client.read_one(d_transform);
            let gpu_transform = f32::from_bytes(&gpu_transform_bytes);

            for (i, (gpu, cpu)) in gpu_transform.iter().zip(cpu_transform.iter()).enumerate() {
                let diff = (gpu - cpu).abs();
                assert!(
                    diff < 1e-5,
                    "Transform mismatch at index {i} for pose {pose:?}: GPU={gpu}, CPU={cpu}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_dot_product_kernel() {
        let (_device, client) = get_test_client();

        // Test vectors
        let a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: [f32; 6] = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        // Expected: 1*6 + 2*5 + 3*4 + 4*3 + 5*2 + 6*1 = 6 + 10 + 12 + 12 + 10 + 6 = 56
        let expected = 56.0f32;

        let d_a = client.create(f32::as_bytes(&a));
        let d_b = client.create(f32::as_bytes(&b));
        let d_result = client.empty(std::mem::size_of::<f32>());

        unsafe {
            dot_product_6_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_a, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_b, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_result, 1, 1),
            );
        }

        let result_bytes = client.read_one(d_result);
        let result = f32::from_bytes(&result_bytes)[0];

        assert!(
            (result - expected).abs() < 1e-5,
            "Dot product mismatch: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_product_kernel_orthogonal() {
        let (_device, client) = get_test_client();

        // Orthogonal vectors should have dot product 0
        let a: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b: [f32; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let d_a = client.create(f32::as_bytes(&a));
        let d_b = client.create(f32::as_bytes(&b));
        let d_result = client.empty(std::mem::size_of::<f32>());

        unsafe {
            dot_product_6_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_a, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_b, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_result, 1, 1),
            );
        }

        let result_bytes = client.read_one(d_result);
        let result = f32::from_bytes(&result_bytes)[0];

        assert!(
            result.abs() < 1e-6,
            "Orthogonal vectors should have dot product 0, got {result}"
        );
    }

    #[test]
    fn test_generate_candidates_kernel() {
        let (_device, client) = get_test_client();

        let initial_step: f32 = 1.0;
        let step_min: f32 = 0.001;
        let step_max: f32 = 10.0;
        let num_candidates: u32 = 8;

        let d_candidates = client.empty(num_candidates as usize * std::mem::size_of::<f32>());

        unsafe {
            generate_candidates_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(num_candidates, 1, 1),
                ScalarArg::new(initial_step),
                ScalarArg::new(step_min),
                ScalarArg::new(step_max),
                ScalarArg::new(num_candidates),
                ArrayArg::from_raw_parts::<f32>(&d_candidates, num_candidates as usize, 1),
            );
        }

        let candidates_bytes = client.read_one(d_candidates);
        let candidates = f32::from_bytes(&candidates_bytes);

        // Expected ratios: [1.0, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625]
        let expected_ratios = [1.0f32, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625];

        assert_eq!(candidates.len(), 8);
        for (i, (got, expected)) in candidates.iter().zip(expected_ratios.iter()).enumerate() {
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-5,
                "Candidate {i} mismatch: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_generate_candidates_with_clamping() {
        let (_device, client) = get_test_client();

        // Use a small initial step so some candidates get clamped to step_min
        let initial_step: f32 = 0.01;
        let step_min: f32 = 0.001;
        let step_max: f32 = 0.005; // Small max so some get clamped
        let num_candidates: u32 = 8;

        let d_candidates = client.empty(num_candidates as usize * std::mem::size_of::<f32>());

        unsafe {
            generate_candidates_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(num_candidates, 1, 1),
                ScalarArg::new(initial_step),
                ScalarArg::new(step_min),
                ScalarArg::new(step_max),
                ScalarArg::new(num_candidates),
                ArrayArg::from_raw_parts::<f32>(&d_candidates, num_candidates as usize, 1),
            );
        }

        let candidates_bytes = client.read_one(d_candidates);
        let candidates = f32::from_bytes(&candidates_bytes);

        // All candidates should be within [step_min, step_max]
        for (i, &c) in candidates.iter().enumerate() {
            assert!(
                c >= step_min && c <= step_max,
                "Candidate {i} = {c} out of range [{step_min}, {step_max}]"
            );
        }
    }

    #[test]
    fn test_update_pose_kernel() {
        let (_device, client) = get_test_client();

        let pose: [f32; 6] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let delta: [f32; 6] = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03];
        let alpha: [f32; 1] = [0.5];

        // Expected: pose + alpha * delta
        let expected: [f32; 6] = [
            1.0 + 0.5 * 0.1,
            2.0 + 0.5 * 0.2,
            3.0 + 0.5 * 0.3,
            0.1 + 0.5 * 0.01,
            0.2 + 0.5 * 0.02,
            0.3 + 0.5 * 0.03,
        ];

        let d_pose = client.create(f32::as_bytes(&pose));
        let d_delta = client.create(f32::as_bytes(&delta));
        let d_alpha = client.create(f32::as_bytes(&alpha));

        unsafe {
            update_pose_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(6, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_pose, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_delta, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_alpha, 1, 1),
            );
        }

        let pose_bytes = client.read_one(d_pose);
        let updated_pose = f32::from_bytes(&pose_bytes);

        for (i, (got, exp)) in updated_pose.iter().zip(expected.iter()).enumerate() {
            let diff = (got - exp).abs();
            assert!(diff < 1e-6, "Pose[{i}] mismatch: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_convergence_kernel_converged() {
        let (_device, client) = get_test_client();

        // Small delta * alpha should converge
        let delta: [f32; 6] = [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001];
        let alpha: [f32; 1] = [0.5];
        let epsilon_sq: f32 = 0.01 * 0.01; // epsilon = 0.01

        let d_delta = client.create(f32::as_bytes(&delta));
        let d_alpha = client.create(f32::as_bytes(&alpha));
        let d_converged = client.empty(std::mem::size_of::<u32>());

        unsafe {
            check_convergence_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_delta, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_alpha, 1, 1),
                ScalarArg::new(epsilon_sq),
                ArrayArg::from_raw_parts::<u32>(&d_converged, 1, 1),
            );
        }

        let converged_bytes = client.read_one(d_converged);
        let converged = u32::from_bytes(&converged_bytes)[0];

        assert_eq!(converged, 1, "Should be converged with small step");
    }

    #[test]
    fn test_convergence_kernel_not_converged() {
        let (_device, client) = get_test_client();

        // Large delta * alpha should not converge
        let delta: [f32; 6] = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1];
        let alpha: [f32; 1] = [1.0];
        let epsilon_sq: f32 = 0.01 * 0.01; // epsilon = 0.01

        let d_delta = client.create(f32::as_bytes(&delta));
        let d_alpha = client.create(f32::as_bytes(&alpha));
        let d_converged = client.empty(std::mem::size_of::<u32>());

        unsafe {
            check_convergence_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_delta, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_alpha, 1, 1),
                ScalarArg::new(epsilon_sq),
                ArrayArg::from_raw_parts::<u32>(&d_converged, 1, 1),
            );
        }

        let converged_bytes = client.read_one(d_converged);
        let converged = u32::from_bytes(&converged_bytes)[0];

        assert_eq!(converged, 0, "Should not be converged with large step");
    }

    // ========================================================================
    // Phase 15.6: Batch Transform Kernel Tests
    // ========================================================================

    /// CPU reference for transforming a single point with a given pose.
    fn transform_point_cpu(point: &[f32; 3], pose: &[f32; 6]) -> [f32; 3] {
        let (sr, cr) = pose[3].sin_cos();
        let (sp, cp) = pose[4].sin_cos();
        let (sy, cy) = pose[5].sin_cos();

        // Rotation matrix: Rz(yaw) * Ry(pitch) * Rx(roll)
        let r00 = cy * cp;
        let r01 = cy * sp * sr - sy * cr;
        let r02 = cy * sp * cr + sy * sr;
        let r10 = sy * cp;
        let r11 = sy * sp * sr + cy * cr;
        let r12 = sy * sp * cr - cy * sr;
        let r20 = -sp;
        let r21 = cp * sr;
        let r22 = cp * cr;

        [
            r00 * point[0] + r01 * point[1] + r02 * point[2] + pose[0],
            r10 * point[0] + r11 * point[1] + r12 * point[2] + pose[1],
            r20 * point[0] + r21 * point[1] + r22 * point[2] + pose[2],
        ]
    }

    #[test]
    fn test_batch_transform_kernel() {
        let (_device, client) = get_test_client();

        // Test data
        let num_points: u32 = 4;
        let num_candidates: u32 = 3;

        let source_points: [f32; 12] = [
            1.0, 0.0, 0.0, // Point 0
            0.0, 1.0, 0.0, // Point 1
            0.0, 0.0, 1.0, // Point 2
            1.0, 1.0, 1.0, // Point 3
        ];

        let pose: [f32; 6] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let delta: [f32; 6] = [0.1, 0.1, 0.1, 0.01, 0.02, 0.03];
        let candidates: [f32; 3] = [1.0, 0.5, 0.25];

        // Compute CPU reference
        let mut expected: Vec<[f32; 3]> = Vec::new();
        for &alpha in &candidates {
            let trial_pose: [f32; 6] = [
                pose[0] + alpha * delta[0],
                pose[1] + alpha * delta[1],
                pose[2] + alpha * delta[2],
                pose[3] + alpha * delta[3],
                pose[4] + alpha * delta[4],
                pose[5] + alpha * delta[5],
            ];
            for i in 0..num_points as usize {
                let point = [
                    source_points[i * 3],
                    source_points[i * 3 + 1],
                    source_points[i * 3 + 2],
                ];
                expected.push(transform_point_cpu(&point, &trial_pose));
            }
        }

        // GPU computation
        let d_source = client.create(f32::as_bytes(&source_points));
        let d_pose = client.create(f32::as_bytes(&pose));
        let d_delta = client.create(f32::as_bytes(&delta));
        let d_candidates = client.create(f32::as_bytes(&candidates));
        let batch_size = (num_candidates * num_points * 3) as usize;
        let d_batch_transformed = client.empty(batch_size * std::mem::size_of::<f32>());

        let total_threads = num_candidates * num_points;
        let block_size = 256u32;
        let grid_size = total_threads.div_ceil(block_size);

        unsafe {
            batch_transform_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(grid_size, 1, 1),
                CubeDim::new(block_size, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_source, (num_points * 3) as usize, 1),
                ArrayArg::from_raw_parts::<f32>(&d_pose, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_delta, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_candidates, num_candidates as usize, 1),
                ScalarArg::new(num_points),
                ScalarArg::new(num_candidates),
                ArrayArg::from_raw_parts::<f32>(&d_batch_transformed, batch_size, 1),
            );
        }

        let result_bytes = client.read_one(d_batch_transformed);
        let result = f32::from_bytes(&result_bytes);

        // Compare
        for k in 0..num_candidates as usize {
            for i in 0..num_points as usize {
                let idx = k * num_points as usize + i;
                let gpu_point = [result[idx * 3], result[idx * 3 + 1], result[idx * 3 + 2]];
                let cpu_point = expected[idx];

                for d in 0..3 {
                    let diff = (gpu_point[d] - cpu_point[d]).abs();
                    assert!(
                        diff < 1e-5,
                        "Batch transform mismatch at candidate={k}, point={i}, dim={d}: \
                         GPU={}, CPU={}, diff={diff}",
                        gpu_point[d],
                        cpu_point[d]
                    );
                }
            }
        }
    }

    #[test]
    fn test_batch_transform_kernel_identity() {
        let (_device, client) = get_test_client();

        // Test with identity pose and zero delta - should return original points
        let num_points: u32 = 2;
        let num_candidates: u32 = 2;

        let source_points: [f32; 6] = [
            1.0, 2.0, 3.0, // Point 0
            4.0, 5.0, 6.0, // Point 1
        ];

        let pose: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delta: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let candidates: [f32; 2] = [1.0, 0.5];

        let d_source = client.create(f32::as_bytes(&source_points));
        let d_pose = client.create(f32::as_bytes(&pose));
        let d_delta = client.create(f32::as_bytes(&delta));
        let d_candidates = client.create(f32::as_bytes(&candidates));
        let batch_size = (num_candidates * num_points * 3) as usize;
        let d_batch_transformed = client.empty(batch_size * std::mem::size_of::<f32>());

        let total_threads = num_candidates * num_points;

        unsafe {
            batch_transform_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(total_threads, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_source, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_pose, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_delta, 6, 1),
                ArrayArg::from_raw_parts::<f32>(&d_candidates, 2, 1),
                ScalarArg::new(num_points),
                ScalarArg::new(num_candidates),
                ArrayArg::from_raw_parts::<f32>(&d_batch_transformed, batch_size, 1),
            );
        }

        let result_bytes = client.read_one(d_batch_transformed);
        let result = f32::from_bytes(&result_bytes);

        // All candidates should return original points with identity transform
        for k in 0..num_candidates as usize {
            for i in 0..num_points as usize {
                let idx = k * num_points as usize + i;
                for d in 0..3 {
                    let diff = (result[idx * 3 + d] - source_points[i * 3 + d]).abs();
                    assert!(
                        diff < 1e-6,
                        "Identity transform should preserve points: k={k}, i={i}, d={d}"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Phase 15.8: More-Thuente Kernel Tests
    // ========================================================================

    #[test]
    fn test_more_thuente_kernel_wolfe_satisfied() {
        let (_device, client) = get_test_client();

        // Setup: candidate at index 2 (alpha=0.5) satisfies Wolfe conditions
        let phi_0: [f32; 1] = [-100.0]; // Starting score
        let dphi_0: [f32; 1] = [-10.0]; // Negative gradient (descent direction)
        let num_candidates: u32 = 8;

        // Candidate step sizes
        let candidates: [f32; 8] = [1.0, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625];

        // Scores and derivatives at each candidate
        // Design so alpha=0.5 (index 2) satisfies Strong Wolfe:
        // Armijo: phi(0.5) <= phi(0) + mu * 0.5 * phi'(0) = -100 + 1e-4 * 0.5 * (-10) = -100.0005
        // Curvature: |phi'(0.5)| <= nu * |phi'(0)| = 0.9 * 10 = 9
        let cached_phi: [f32; 8] = [
            -95.0,  // alpha=1.0: too high (doesn't satisfy Armijo)
            -98.0,  // alpha=0.75: borderline
            -101.0, // alpha=0.5: satisfies Armijo (< -100.0005)
            -100.5, // alpha=0.25
            -100.1, // alpha=0.125
            -99.0,  // alpha=0.618
            -100.2, // alpha=0.382
            -100.0, // alpha=0.0625
        ];
        let cached_dphi: [f32; 8] = [
            -5.0, // alpha=1.0: |dphi| = 5 < 9, ok for curvature
            -6.0, // alpha=0.75
            -2.0, // alpha=0.5: |dphi| = 2 < 9, satisfies curvature
            -8.0, // alpha=0.25
            -9.5, // alpha=0.125: |dphi| = 9.5 > 9, fails curvature
            -4.0, // alpha=0.618
            -7.0, // alpha=0.382
            -9.8, // alpha=0.0625
        ];

        // ls_params: [num_candidates as f32, mu, nu]
        let ls_params: [f32; 3] = [num_candidates as f32, 1e-4, 0.9];

        let d_phi_0 = client.create(f32::as_bytes(&phi_0));
        let d_dphi_0 = client.create(f32::as_bytes(&dphi_0));
        let d_candidates = client.create(f32::as_bytes(&candidates));
        let d_cached_phi = client.create(f32::as_bytes(&cached_phi));
        let d_cached_dphi = client.create(f32::as_bytes(&cached_dphi));
        let d_ls_params = client.create(f32::as_bytes(&ls_params));
        let d_best_alpha = client.empty(std::mem::size_of::<f32>());
        let d_ls_converged = client.empty(std::mem::size_of::<f32>());

        unsafe {
            more_thuente_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_phi_0, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_dphi_0, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_candidates, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_cached_phi, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_cached_dphi, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_ls_params, 3, 1),
                ArrayArg::from_raw_parts::<f32>(&d_best_alpha, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_ls_converged, 1, 1),
            );
        }

        let alpha_bytes = client.read_one(d_best_alpha);
        let best_alpha = f32::from_bytes(&alpha_bytes)[0];

        let converged_bytes = client.read_one(d_ls_converged);
        let ls_converged = f32::from_bytes(&converged_bytes)[0];

        // Should find alpha=0.5 (index 2) which satisfies both conditions
        assert!(
            (best_alpha - 0.5).abs() < 1e-5,
            "Expected best_alpha=0.5, got {best_alpha}"
        );
        assert!(
            ls_converged > 0.5,
            "Expected ls_converged=1.0, got {ls_converged}"
        );
    }

    #[test]
    fn test_more_thuente_kernel_armijo_only() {
        let (_device, client) = get_test_client();

        // Setup: no candidate satisfies Strong Wolfe, but some satisfy Armijo
        let phi_0: [f32; 1] = [-100.0];
        let dphi_0: [f32; 1] = [-10.0];
        let num_candidates: u32 = 8;

        let candidates: [f32; 8] = [1.0, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625];

        // All have |dphi| > nu * |dphi_0| = 9, so none satisfy curvature
        // But some satisfy Armijo
        let cached_phi: [f32; 8] = [
            -95.0,  // alpha=1.0: fails Armijo
            -98.0,  // alpha=0.75: fails Armijo
            -101.0, // alpha=0.5: satisfies Armijo, lowest score
            -100.5, // alpha=0.25: satisfies Armijo
            -100.1, // alpha=0.125: satisfies Armijo
            -99.0,  // alpha=0.618: fails Armijo
            -100.2, // alpha=0.382: satisfies Armijo
            -100.0, // alpha=0.0625: borderline
        ];
        // All fail curvature (|dphi| > 9)
        let cached_dphi: [f32; 8] = [-15.0, -14.0, -12.0, -11.0, -10.5, -13.0, -10.0, -9.5];

        // ls_params: [num_candidates as f32, mu, nu]
        let ls_params: [f32; 3] = [num_candidates as f32, 1e-4, 0.9];

        let d_phi_0 = client.create(f32::as_bytes(&phi_0));
        let d_dphi_0 = client.create(f32::as_bytes(&dphi_0));
        let d_candidates = client.create(f32::as_bytes(&candidates));
        let d_cached_phi = client.create(f32::as_bytes(&cached_phi));
        let d_cached_dphi = client.create(f32::as_bytes(&cached_dphi));
        let d_ls_params = client.create(f32::as_bytes(&ls_params));
        let d_best_alpha = client.empty(std::mem::size_of::<f32>());
        let d_ls_converged = client.empty(std::mem::size_of::<f32>());

        unsafe {
            more_thuente_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_phi_0, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_dphi_0, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_candidates, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_cached_phi, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_cached_dphi, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_ls_params, 3, 1),
                ArrayArg::from_raw_parts::<f32>(&d_best_alpha, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_ls_converged, 1, 1),
            );
        }

        let alpha_bytes = client.read_one(d_best_alpha);
        let best_alpha = f32::from_bytes(&alpha_bytes)[0];

        let converged_bytes = client.read_one(d_ls_converged);
        let ls_converged = f32::from_bytes(&converged_bytes)[0];

        // Should find alpha=0.5 which has lowest score among Armijo-satisfying candidates
        assert!(
            (best_alpha - 0.5).abs() < 1e-5,
            "Expected best_alpha=0.5 (lowest Armijo score), got {best_alpha}"
        );
        // Should NOT be fully converged (Wolfe not satisfied)
        assert!(
            ls_converged < 0.5,
            "Expected ls_converged=0.0, got {ls_converged}"
        );
    }

    #[test]
    fn test_more_thuente_kernel_no_armijo() {
        let (_device, client) = get_test_client();

        // Setup: no candidate satisfies even Armijo
        let phi_0: [f32; 1] = [-100.0];
        let dphi_0: [f32; 1] = [-10.0];
        let num_candidates: u32 = 8;

        let candidates: [f32; 8] = [1.0, 0.75, 0.5, 0.25, 0.125, 0.618, 0.382, 0.0625];

        // All scores are higher than starting score (bad line search direction)
        let cached_phi: [f32; 8] = [-50.0, -60.0, -70.0, -80.0, -85.0, -65.0, -75.0, -90.0];
        let cached_dphi: [f32; 8] = [5.0, 4.0, 3.0, 2.0, 1.0, 3.5, 2.5, 0.5];

        // ls_params: [num_candidates as f32, mu, nu]
        let ls_params: [f32; 3] = [num_candidates as f32, 1e-4, 0.9];

        let d_phi_0 = client.create(f32::as_bytes(&phi_0));
        let d_dphi_0 = client.create(f32::as_bytes(&dphi_0));
        let d_candidates = client.create(f32::as_bytes(&candidates));
        let d_cached_phi = client.create(f32::as_bytes(&cached_phi));
        let d_cached_dphi = client.create(f32::as_bytes(&cached_dphi));
        let d_ls_params = client.create(f32::as_bytes(&ls_params));
        let d_best_alpha = client.empty(std::mem::size_of::<f32>());
        let d_ls_converged = client.empty(std::mem::size_of::<f32>());

        unsafe {
            more_thuente_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_phi_0, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_dphi_0, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_candidates, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_cached_phi, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_cached_dphi, 8, 1),
                ArrayArg::from_raw_parts::<f32>(&d_ls_params, 3, 1),
                ArrayArg::from_raw_parts::<f32>(&d_best_alpha, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_ls_converged, 1, 1),
            );
        }

        let alpha_bytes = client.read_one(d_best_alpha);
        let best_alpha = f32::from_bytes(&alpha_bytes)[0];

        let converged_bytes = client.read_one(d_ls_converged);
        let ls_converged = f32::from_bytes(&converged_bytes)[0];

        // Should fall back to smallest step (0.0625)
        assert!(
            (best_alpha - 0.0625).abs() < 1e-5,
            "Expected best_alpha=0.0625 (smallest step), got {best_alpha}"
        );
        assert!(
            ls_converged < 0.5,
            "Expected ls_converged=0.0, got {ls_converged}"
        );
    }
}
