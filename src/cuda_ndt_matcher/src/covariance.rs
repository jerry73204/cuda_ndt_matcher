//! Covariance estimation for NDT scan matching.
//!
//! This module implements covariance estimation modes:
//! - FIXED: Static covariance matrix from configuration
//! - LAPLACE: Inverse of Hessian matrix (2x2 XY block)
//! - MULTI_NDT: Run alignments from offset poses, compute sample covariance
//! - MULTI_NDT_SCORE: Compute NVTL at offset poses, use softmax-weighted covariance

// Allow dead_code: Functions are called from main.rs via dynamic covariance estimation
// mode selection. Rust's analysis doesn't track usage through config-driven dispatch.
#![allow(dead_code)]

use crate::ndt_manager::NdtManager;
use crate::params::{CovarianceEstimationParams, CovarianceEstimationType, CovarianceParams};
use geometry_msgs::msg::Pose;
use nalgebra::{Matrix2, UnitQuaternion, Vector2};

/// Result of covariance estimation
#[derive(Debug, Clone)]
pub struct CovarianceEstimationResult {
    /// 6x6 covariance matrix (row-major order)
    pub covariance: [f64; 36],
    /// Estimated 2x2 XY covariance (if dynamic estimation was used)
    pub xy_covariance: Option<[[f64; 2]; 2]>,
}

/// Estimate covariance using ndt_cuda result
///
/// Supports FIXED and LAPLACE modes. MULTI_NDT modes fall back to LAPLACE.
pub fn estimate_covariance(
    params: &CovarianceParams,
    hessian: &[[f64; 6]; 6],
    result_pose: &Pose,
) -> CovarianceEstimationResult {
    // Start with static covariance rotated to match the result pose
    let mut covariance = rotate_covariance(&params.output_pose_covariance, result_pose);

    let xy_covariance = match params.covariance_estimation_type {
        CovarianceEstimationType::Fixed => None,
        CovarianceEstimationType::LaplaceApproximation
        | CovarianceEstimationType::MultiNdt
        | CovarianceEstimationType::MultiNdtScore => {
            // For MULTI_NDT modes, fall back to Laplace approximation
            // (MULTI_NDT requires expensive multiple alignments)
            let estimated_2d = estimate_xy_covariance_by_laplace(hessian);

            // Apply scale factor
            let scaled = scale_covariance_2d(&estimated_2d, params.estimation.scale_factor);

            // Ensure minimum covariance values from the static matrix
            let adjusted = adjust_diagonal_covariance(
                &scaled,
                params.output_pose_covariance[0], // default cov_xx
                params.output_pose_covariance[7], // default cov_yy
            );

            // Update the XY block of the 6x6 covariance matrix
            covariance[0] = adjusted[0][0]; // [0][0]
            covariance[1] = adjusted[0][1]; // [0][1]
            covariance[6] = adjusted[1][0]; // [1][0]
            covariance[7] = adjusted[1][1]; // [1][1]

            Some(adjusted)
        }
    };

    CovarianceEstimationResult {
        covariance,
        xy_covariance,
    }
}

/// Estimate 2D covariance using Laplace approximation (inverse of Hessian XY block)
fn estimate_xy_covariance_by_laplace(hessian: &[[f64; 6]; 6]) -> [[f64; 2]; 2] {
    // Extract 2x2 XY block from the 6x6 Hessian
    let h = Matrix2::new(hessian[0][0], hessian[0][1], hessian[1][0], hessian[1][1]);

    // Covariance = -H^(-1)
    // The negative is because the Hessian of the NDT score function is negative definite
    match h.try_inverse() {
        Some(inv) => {
            let cov = -inv;
            [[cov[(0, 0)], cov[(0, 1)]], [cov[(1, 0)], cov[(1, 1)]]]
        }
        None => {
            // If Hessian is singular, return large uncertainty
            [[1.0, 0.0], [0.0, 1.0]]
        }
    }
}

/// Rotate covariance matrix based on result pose orientation
fn rotate_covariance(covariance: &[f64; 36], pose: &Pose) -> [f64; 36] {
    let q = &pose.orientation;
    let rotation = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z));
    let rotation_matrix = rotation.to_rotation_matrix();

    // Create 6x6 rotation block matrix (rotation for position and rotation for orientation)
    let r = rotation_matrix.matrix();
    let mut rot_6x6 = nalgebra::Matrix6::<f64>::zeros();

    // Top-left 3x3 block for position
    for i in 0..3 {
        for j in 0..3 {
            rot_6x6[(i, j)] = r[(i, j)];
        }
    }

    // Bottom-right 3x3 block for orientation
    for i in 0..3 {
        for j in 0..3 {
            rot_6x6[(i + 3, j + 3)] = r[(i, j)];
        }
    }

    // Build covariance matrix
    let mut cov = nalgebra::Matrix6::<f64>::zeros();
    for i in 0..6 {
        for j in 0..6 {
            cov[(i, j)] = covariance[i * 6 + j];
        }
    }

    // Rotate covariance: R * C * R^T
    let rotated = rot_6x6 * cov * rot_6x6.transpose();

    let mut result = [0.0; 36];
    for i in 0..6 {
        for j in 0..6 {
            result[i * 6 + j] = rotated[(i, j)];
        }
    }
    result
}

/// Scale 2D covariance by a factor
fn scale_covariance_2d(cov: &[[f64; 2]; 2], scale: f64) -> [[f64; 2]; 2] {
    [
        [cov[0][0] * scale, cov[0][1] * scale],
        [cov[1][0] * scale, cov[1][1] * scale],
    ]
}

/// Adjust diagonal covariance to ensure minimum values
pub fn adjust_diagonal_covariance(cov: &[[f64; 2]; 2], min_xx: f64, min_yy: f64) -> [[f64; 2]; 2] {
    [
        [cov[0][0].max(min_xx), cov[0][1]],
        [cov[1][0], cov[1][1].max(min_yy)],
    ]
}

/// Calculate sample covariance from 2D points (equal weights)
pub fn calculate_sample_covariance(points: &[Vector2<f64>]) -> [[f64; 2]; 2] {
    let n = points.len();
    if n < 2 {
        return [[1.0, 0.0], [0.0, 1.0]];
    }

    // Calculate mean
    let mut mean = Vector2::zeros();
    for p in points {
        mean += p;
    }
    mean /= n as f64;

    // Calculate covariance
    let mut cov = Matrix2::zeros();
    for p in points {
        let diff = p - mean;
        cov += diff * diff.transpose();
    }
    cov /= (n - 1) as f64; // Unbiased estimator

    [[cov[(0, 0)], cov[(0, 1)]], [cov[(1, 0)], cov[(1, 1)]]]
}

/// Calculate softmax weights from scores
pub fn calculate_softmax_weights(scores: &[f64], temperature: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Calculate exponentials
    let exp_scores: Vec<f64> = scores
        .iter()
        .map(|&s| ((s - max_score) / temperature).exp())
        .collect();

    // Normalize
    let sum: f64 = exp_scores.iter().sum();
    if sum > 0.0 {
        exp_scores.iter().map(|&e| e / sum).collect()
    } else {
        // Equal weights if all scores are very negative
        vec![1.0 / scores.len() as f64; scores.len()]
    }
}

// ============================================================================
// Multi-NDT Covariance Estimation
// ============================================================================

/// Result of multi-NDT covariance estimation
#[derive(Debug, Clone)]
pub struct MultiNdtResult {
    /// Estimated 2D covariance matrix
    pub covariance: [[f64; 2]; 2],
    /// Poses used for estimation (offset poses)
    pub poses_searched: Vec<Pose>,
    /// NVTL scores at each pose (for MULTI_NDT_SCORE mode)
    pub nvtl_scores: Vec<f64>,
}

/// Create offset poses from a result pose for multi-NDT estimation.
///
/// The offsets are rotated by the result pose orientation so they are
/// in the vehicle's local coordinate frame.
pub fn propose_offset_poses(result_pose: &Pose, offset_x: &[f64], offset_y: &[f64]) -> Vec<Pose> {
    assert_eq!(
        offset_x.len(),
        offset_y.len(),
        "Offset arrays must have same length"
    );

    // Extract 2D rotation from result pose quaternion
    let q = &result_pose.orientation;
    let rotation = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z));
    let rot_matrix = rotation.to_rotation_matrix();
    let rot_2d = Matrix2::new(
        rot_matrix[(0, 0)],
        rot_matrix[(0, 1)],
        rot_matrix[(1, 0)],
        rot_matrix[(1, 1)],
    );

    let mut poses = Vec::with_capacity(offset_x.len());
    for i in 0..offset_x.len() {
        let offset = Vector2::new(offset_x[i], offset_y[i]);
        let rotated_offset = rot_2d * offset;

        let mut pose = result_pose.clone();
        pose.position.x += rotated_offset.x;
        pose.position.y += rotated_offset.y;
        poses.push(pose);
    }

    poses
}

/// Estimate covariance using MULTI_NDT mode (parallel via Rayon).
///
/// This runs NDT alignment from each offset pose in parallel and computes the sample
/// covariance from all resulting aligned poses.
pub fn estimate_xy_covariance_by_multi_ndt(
    ndt_manager: &NdtManager,
    sensor_points: &[[f32; 3]],
    _map_points: &[[f32; 3]], // Not needed, target already set
    result_pose: &Pose,
    estimation_params: &CovarianceEstimationParams,
) -> MultiNdtResult {
    // Create offset poses
    let offset_poses = propose_offset_poses(
        result_pose,
        &estimation_params.initial_pose_offset_model_x,
        &estimation_params.initial_pose_offset_model_y,
    );

    // Run parallel batch alignment for all offset poses
    let batch_results = ndt_manager.align_batch(sensor_points, &offset_poses);

    // Collect aligned poses (including the original result)
    let mut pose_2d_vec: Vec<Vector2<f64>> =
        vec![Vector2::new(result_pose.position.x, result_pose.position.y)];
    let mut poses_searched = Vec::new();

    match batch_results {
        Ok(results) => {
            for align_result in results {
                pose_2d_vec.push(Vector2::new(
                    align_result.pose.position.x,
                    align_result.pose.position.y,
                ));
                poses_searched.push(align_result.pose);
            }
        }
        Err(_) => {
            // Fallback: use offset poses directly if batch alignment fails
            for offset_pose in &offset_poses {
                pose_2d_vec.push(Vector2::new(offset_pose.position.x, offset_pose.position.y));
                poses_searched.push(offset_pose.clone());
            }
        }
    }

    // Calculate sample covariance with equal weights
    let n = pose_2d_vec.len();
    let weights = vec![1.0 / n as f64; n];
    let covariance = calculate_weighted_covariance(&pose_2d_vec, &weights);

    // Apply unbiased correction: multiply by (n-1)/n
    let correction = (n - 1) as f64 / n as f64;
    let corrected_cov = [
        [covariance[0][0] * correction, covariance[0][1] * correction],
        [covariance[1][0] * correction, covariance[1][1] * correction],
    ];

    MultiNdtResult {
        covariance: corrected_cov,
        poses_searched,
        nvtl_scores: Vec::new(),
    }
}

/// Estimate covariance using MULTI_NDT_SCORE mode (parallel via Rayon).
///
/// This does NOT run alignments for offset poses. Instead, it:
/// 1. Computes NVTL at all poses in parallel (fast, no iteration)
/// 2. Uses softmax weights based on NVTL scores
/// 3. Computes weighted covariance from offset poses
pub fn estimate_xy_covariance_by_multi_ndt_score(
    ndt_manager: &mut NdtManager,
    sensor_points: &[[f32; 3]],
    _map_points: &[[f32; 3]], // Not needed, target already set
    result_pose: &Pose,
    estimation_params: &CovarianceEstimationParams,
) -> MultiNdtResult {
    // Create offset poses
    let offset_poses = propose_offset_poses(
        result_pose,
        &estimation_params.initial_pose_offset_model_x,
        &estimation_params.initial_pose_offset_model_y,
    );

    // Collect all poses for batch evaluation (result pose + offset poses)
    let mut all_poses = vec![result_pose.clone()];
    all_poses.extend(offset_poses.iter().cloned());

    // Run parallel batch NVTL evaluation for all poses at once
    let nvtl_scores = ndt_manager
        .evaluate_nvtl_batch(sensor_points, &all_poses)
        .unwrap_or_else(|_| vec![0.0; all_poses.len()]);

    // Build 2D pose vectors for covariance calculation
    let pose_2d_vec: Vec<Vector2<f64>> = all_poses
        .iter()
        .map(|p| Vector2::new(p.position.x, p.position.y))
        .collect();

    // Calculate softmax weights from NVTL scores
    let weights = calculate_softmax_weights(&nvtl_scores, estimation_params.temperature);

    // Calculate weighted covariance
    let covariance = calculate_weighted_covariance(&pose_2d_vec, &weights);

    MultiNdtResult {
        covariance,
        poses_searched: offset_poses,
        nvtl_scores,
    }
}

/// Calculate weighted mean and covariance from 2D points.
fn calculate_weighted_covariance(points: &[Vector2<f64>], weights: &[f64]) -> [[f64; 2]; 2] {
    assert_eq!(points.len(), weights.len());

    if points.is_empty() {
        return [[1.0, 0.0], [0.0, 1.0]];
    }

    // Calculate weighted mean
    let mut mean = Vector2::zeros();
    for (p, &w) in points.iter().zip(weights.iter()) {
        mean += w * p;
    }

    // Calculate weighted covariance
    let mut cov = Matrix2::zeros();
    for (p, &w) in points.iter().zip(weights.iter()) {
        let diff = p - mean;
        cov += w * diff * diff.transpose();
    }

    [[cov[(0, 0)], cov[(0, 1)]], [cov[(1, 0)], cov[(1, 1)]]]
}

/// Full covariance estimation including multi-NDT modes.
///
/// This is the main entry point when multi-NDT estimation is needed.
/// For modes that don't require multi-NDT, use `estimate_covariance` instead.
///
/// Note: Multi-NDT modes use parallel batch evaluation (via Rayon) for
/// efficient covariance estimation from multiple poses.
pub fn estimate_covariance_full(
    params: &CovarianceParams,
    hessian: &[[f64; 6]; 6],
    result_pose: &Pose,
    ndt_manager: Option<&mut NdtManager>,
    sensor_points: Option<&[[f32; 3]]>,
    map_points: Option<&[[f32; 3]]>,
) -> CovarianceEstimationResult {
    // Start with static covariance rotated to match the result pose
    let mut covariance = rotate_covariance(&params.output_pose_covariance, result_pose);

    let xy_covariance = match params.covariance_estimation_type {
        CovarianceEstimationType::Fixed => None,

        CovarianceEstimationType::LaplaceApproximation => {
            let estimated_2d = estimate_xy_covariance_by_laplace(hessian);
            let scaled = scale_covariance_2d(&estimated_2d, params.estimation.scale_factor);
            let adjusted = adjust_diagonal_covariance(
                &scaled,
                params.output_pose_covariance[0],
                params.output_pose_covariance[7],
            );

            covariance[0] = adjusted[0][0];
            covariance[1] = adjusted[0][1];
            covariance[6] = adjusted[1][0];
            covariance[7] = adjusted[1][1];

            Some(adjusted)
        }

        CovarianceEstimationType::MultiNdt => {
            // Use multi-NDT if we have the required data, otherwise fall back to Laplace
            match (ndt_manager, sensor_points, map_points) {
                (Some(manager), Some(sensor), Some(map)) => {
                    let multi_result = estimate_xy_covariance_by_multi_ndt(
                        manager,
                        sensor,
                        map,
                        result_pose,
                        &params.estimation,
                    );
                    let scaled = scale_covariance_2d(
                        &multi_result.covariance,
                        params.estimation.scale_factor,
                    );
                    let adjusted = adjust_diagonal_covariance(
                        &scaled,
                        params.output_pose_covariance[0],
                        params.output_pose_covariance[7],
                    );

                    covariance[0] = adjusted[0][0];
                    covariance[1] = adjusted[0][1];
                    covariance[6] = adjusted[1][0];
                    covariance[7] = adjusted[1][1];

                    Some(adjusted)
                }
                _ => {
                    // Fallback to Laplace
                    let estimated_2d = estimate_xy_covariance_by_laplace(hessian);
                    let scaled = scale_covariance_2d(&estimated_2d, params.estimation.scale_factor);
                    let adjusted = adjust_diagonal_covariance(
                        &scaled,
                        params.output_pose_covariance[0],
                        params.output_pose_covariance[7],
                    );

                    covariance[0] = adjusted[0][0];
                    covariance[1] = adjusted[0][1];
                    covariance[6] = adjusted[1][0];
                    covariance[7] = adjusted[1][1];

                    Some(adjusted)
                }
            }
        }

        CovarianceEstimationType::MultiNdtScore => {
            // Use multi-NDT score if we have the required data, otherwise fall back to Laplace
            match (ndt_manager, sensor_points, map_points) {
                (Some(manager), Some(sensor), Some(map)) => {
                    let multi_result = estimate_xy_covariance_by_multi_ndt_score(
                        manager,
                        sensor,
                        map,
                        result_pose,
                        &params.estimation,
                    );
                    let scaled = scale_covariance_2d(
                        &multi_result.covariance,
                        params.estimation.scale_factor,
                    );
                    let adjusted = adjust_diagonal_covariance(
                        &scaled,
                        params.output_pose_covariance[0],
                        params.output_pose_covariance[7],
                    );

                    covariance[0] = adjusted[0][0];
                    covariance[1] = adjusted[0][1];
                    covariance[6] = adjusted[1][0];
                    covariance[7] = adjusted[1][1];

                    Some(adjusted)
                }
                _ => {
                    // Fallback to Laplace
                    let estimated_2d = estimate_xy_covariance_by_laplace(hessian);
                    let scaled = scale_covariance_2d(&estimated_2d, params.estimation.scale_factor);
                    let adjusted = adjust_diagonal_covariance(
                        &scaled,
                        params.output_pose_covariance[0],
                        params.output_pose_covariance[7],
                    );

                    covariance[0] = adjusted[0][0];
                    covariance[1] = adjusted[0][1];
                    covariance[6] = adjusted[1][0];
                    covariance[7] = adjusted[1][1];

                    Some(adjusted)
                }
            }
        }
    };

    CovarianceEstimationResult {
        covariance,
        xy_covariance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_weights() {
        let scores = vec![1.0, 2.0, 3.0];
        let weights = calculate_softmax_weights(&scores, 1.0);
        assert_eq!(weights.len(), 3);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Higher score should have higher weight
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_sample_covariance() {
        let points = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];
        let cov = calculate_sample_covariance(&points);
        // Should have positive diagonal
        assert!(cov[0][0] > 0.0);
        assert!(cov[1][1] > 0.0);
    }

    #[test]
    fn test_adjust_diagonal() {
        let cov = [[0.001, 0.0], [0.0, 0.002]];
        let adjusted = adjust_diagonal_covariance(&cov, 0.01, 0.01);
        assert_eq!(adjusted[0][0], 0.01); // Should be adjusted to minimum
        assert_eq!(adjusted[1][1], 0.01); // Should be adjusted to minimum
    }

    #[test]
    fn test_propose_offset_poses_identity() {
        // Test with identity orientation (no rotation)
        let pose = Pose {
            position: geometry_msgs::msg::Point {
                x: 10.0,
                y: 20.0,
                z: 0.0,
            },
            orientation: geometry_msgs::msg::Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };

        let offset_x = vec![1.0, -1.0, 0.0];
        let offset_y = vec![0.0, 0.0, 0.5];

        let poses = propose_offset_poses(&pose, &offset_x, &offset_y);

        assert_eq!(poses.len(), 3);
        // With identity rotation, offsets should be applied directly
        assert!((poses[0].position.x - 11.0).abs() < 1e-10);
        assert!((poses[0].position.y - 20.0).abs() < 1e-10);
        assert!((poses[1].position.x - 9.0).abs() < 1e-10);
        assert!((poses[1].position.y - 20.0).abs() < 1e-10);
        assert!((poses[2].position.x - 10.0).abs() < 1e-10);
        assert!((poses[2].position.y - 20.5).abs() < 1e-10);
    }

    #[test]
    fn test_propose_offset_poses_rotated() {
        // Test with 90-degree rotation (yaw = pi/2)
        // quaternion for 90-degree rotation around Z: (0, 0, sin(pi/4), cos(pi/4))
        let half_angle = std::f64::consts::FRAC_PI_4;
        let pose = Pose {
            position: geometry_msgs::msg::Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            orientation: geometry_msgs::msg::Quaternion {
                x: 0.0,
                y: 0.0,
                z: half_angle.sin(),
                w: half_angle.cos(),
            },
        };

        let offset_x = vec![1.0]; // 1m forward in vehicle frame
        let offset_y = vec![0.0];

        let poses = propose_offset_poses(&pose, &offset_x, &offset_y);

        assert_eq!(poses.len(), 1);
        // With 90-degree rotation, x offset becomes y offset in world frame
        assert!((poses[0].position.x).abs() < 1e-10);
        assert!((poses[0].position.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_covariance_equal_weights() {
        let points = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(2.0, 0.0),
            Vector2::new(0.0, 2.0),
            Vector2::new(2.0, 2.0),
        ];
        let weights = vec![0.25, 0.25, 0.25, 0.25];

        let cov = calculate_weighted_covariance(&points, &weights);

        // Mean is (1, 1), variance should be 1 in both directions
        assert!(cov[0][0] > 0.0);
        assert!(cov[1][1] > 0.0);
        // Should be symmetric
        assert!((cov[0][1] - cov[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_covariance_unequal_weights() {
        let points = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(10.0, 0.0), // Far outlier
        ];
        // High weight on first point, low on second
        let weights = vec![0.99, 0.01];

        let cov = calculate_weighted_covariance(&points, &weights);

        // Covariance should be small since we're mostly weighting the origin
        // The weighted mean is close to (0,0), so variance is small
        assert!(cov[0][0] < 1.0);
    }
}
