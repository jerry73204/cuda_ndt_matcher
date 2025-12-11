//! Covariance estimation for NDT scan matching.
//!
//! This module implements four covariance estimation modes:
//! - FIXED: Static covariance matrix from configuration
//! - LAPLACE: Inverse of Hessian matrix (2x2 XY block)
//! - MULTI_NDT: Sample covariance from multiple NDT alignments
//! - MULTI_NDT_SCORE: Score-weighted covariance from offset poses

use crate::params::{CovarianceEstimationParams, CovarianceEstimationType, CovarianceParams};
use fast_gicp::{Hessian6x6, NDTCuda, PointCloudXYZ, Transform3f};
use geometry_msgs::msg::Pose;
use nalgebra::{Matrix2, Matrix4, UnitQuaternion, Vector2};

/// Result of covariance estimation
#[derive(Debug, Clone)]
pub struct CovarianceEstimationResult {
    /// 6x6 covariance matrix (row-major order)
    pub covariance: [f64; 36],
    /// Estimated 2x2 XY covariance (if dynamic estimation was used)
    pub xy_covariance: Option<[[f64; 2]; 2]>,
}

/// Estimate covariance based on the configured estimation type
pub fn estimate_covariance(
    params: &CovarianceParams,
    hessian: &Hessian6x6,
    result_pose: &Pose,
    ndt: &NDTCuda,
    source: &PointCloudXYZ,
    target: &PointCloudXYZ,
    initial_transform: &Transform3f,
) -> CovarianceEstimationResult {
    // Start with static covariance rotated to match the result pose
    let mut covariance = rotate_covariance(&params.output_pose_covariance, result_pose);

    let xy_covariance = if params.covariance_estimation_type == CovarianceEstimationType::Fixed {
        // FIXED mode: just use the static covariance
        None
    } else {
        // Estimate 2D covariance based on mode
        let estimated_2d = match params.covariance_estimation_type {
            CovarianceEstimationType::Fixed => unreachable!(),
            CovarianceEstimationType::LaplaceApproximation => {
                estimate_xy_covariance_by_laplace(hessian)
            }
            CovarianceEstimationType::MultiNdt => estimate_xy_covariance_by_multi_ndt(
                ndt,
                source,
                target,
                initial_transform,
                &params.estimation,
            ),
            CovarianceEstimationType::MultiNdtScore => estimate_xy_covariance_by_multi_ndt_score(
                ndt,
                source,
                target,
                initial_transform,
                &params.estimation,
            ),
        };

        // Apply scale factor
        let scaled = scale_covariance_2d(&estimated_2d, params.estimation.scale_factor);

        // Ensure minimum covariance values from the static matrix
        let adjusted = adjust_diagonal_covariance(
            &scaled,
            params.output_pose_covariance[0],  // default cov_xx
            params.output_pose_covariance[7],  // default cov_yy
        );

        // Update the XY block of the 6x6 covariance matrix
        covariance[0] = adjusted[0][0];     // [0][0]
        covariance[1] = adjusted[0][1];     // [0][1]
        covariance[6] = adjusted[1][0];     // [1][0]
        covariance[7] = adjusted[1][1];     // [1][1]

        Some(adjusted)
    };

    CovarianceEstimationResult {
        covariance,
        xy_covariance,
    }
}

/// Estimate 2D covariance using Laplace approximation (inverse of Hessian XY block)
fn estimate_xy_covariance_by_laplace(hessian: &Hessian6x6) -> [[f64; 2]; 2] {
    let hessian_xy = hessian.xy_block();

    // Create nalgebra matrix for inversion
    let h = Matrix2::new(
        hessian_xy[0][0],
        hessian_xy[0][1],
        hessian_xy[1][0],
        hessian_xy[1][1],
    );

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

/// Estimate 2D covariance using multiple NDT alignments from offset poses
fn estimate_xy_covariance_by_multi_ndt(
    ndt: &NDTCuda,
    source: &PointCloudXYZ,
    target: &PointCloudXYZ,
    center_transform: &Transform3f,
    params: &CovarianceEstimationParams,
) -> [[f64; 2]; 2] {
    // Generate offset poses
    let offset_poses = propose_poses_to_search(center_transform, params);

    // Collect convergence points
    let mut results_2d: Vec<Vector2<f64>> = Vec::with_capacity(offset_poses.len() + 1);

    // Add main result
    let center_xy = extract_xy_from_transform(center_transform);
    results_2d.push(center_xy);

    // Run NDT from each offset pose
    for offset_pose in &offset_poses {
        match ndt.align_with_guess(source, target, Some(offset_pose)) {
            Ok(result) => {
                let xy = extract_xy_from_transform(&result.final_transformation);
                results_2d.push(xy);
            }
            Err(_) => {
                // Skip failed alignments
            }
        }
    }

    // Calculate equal-weighted mean and covariance
    calculate_sample_covariance(&results_2d)
}

/// Estimate 2D covariance using score-weighted offset poses (faster than full alignment)
fn estimate_xy_covariance_by_multi_ndt_score(
    ndt: &NDTCuda,
    source: &PointCloudXYZ,
    target: &PointCloudXYZ,
    center_transform: &Transform3f,
    params: &CovarianceEstimationParams,
) -> [[f64; 2]; 2] {
    // Generate offset poses
    let offset_poses = propose_poses_to_search(center_transform, params);

    // Collect poses and scores
    let mut poses_2d: Vec<Vector2<f64>> = Vec::with_capacity(offset_poses.len() + 1);
    let mut scores: Vec<f64> = Vec::with_capacity(offset_poses.len() + 1);

    // Add main result
    let center_xy = extract_xy_from_transform(center_transform);
    poses_2d.push(center_xy);

    // Evaluate score at center pose
    match ndt.evaluate_cost_at_pose(source, target, center_transform) {
        Ok(cost) => scores.push(-cost), // Negate because lower cost = better match
        Err(_) => scores.push(0.0),
    }

    // Evaluate scores at offset poses
    for offset_pose in &offset_poses {
        let xy = extract_xy_from_transform(offset_pose);
        poses_2d.push(xy);

        match ndt.evaluate_cost_at_pose(source, target, offset_pose) {
            Ok(cost) => scores.push(-cost),
            Err(_) => scores.push(0.0),
        }
    }

    // Calculate score-weighted covariance
    let weights = calculate_softmax_weights(&scores, params.temperature);
    calculate_weighted_covariance(&poses_2d, &weights)
}

/// Generate offset poses for multi-NDT search
fn propose_poses_to_search(
    center_transform: &Transform3f,
    params: &CovarianceEstimationParams,
) -> Vec<Transform3f> {
    let n = params
        .initial_pose_offset_model_x
        .len()
        .min(params.initial_pose_offset_model_y.len());

    if n == 0 {
        return Vec::new();
    }

    // Extract rotation from center pose to rotate offsets
    let center_matrix = transform_to_matrix4(center_transform);
    let rotation_2d = Matrix2::new(
        center_matrix[(0, 0)],
        center_matrix[(0, 1)],
        center_matrix[(1, 0)],
        center_matrix[(1, 1)],
    );

    let mut poses = Vec::with_capacity(n);

    for i in 0..n {
        let offset = Vector2::new(
            params.initial_pose_offset_model_x[i],
            params.initial_pose_offset_model_y[i],
        );

        // Rotate offset to align with vehicle heading
        let rotated_offset = rotation_2d * offset;

        // Create new transform with offset applied
        let mut new_matrix = center_matrix;
        new_matrix[(0, 3)] += rotated_offset[0];
        new_matrix[(1, 3)] += rotated_offset[1];

        poses.push(matrix4_to_transform(&new_matrix));
    }

    poses
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
    let rotated = &rot_6x6 * &cov * rot_6x6.transpose();

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
fn adjust_diagonal_covariance(
    cov: &[[f64; 2]; 2],
    min_xx: f64,
    min_yy: f64,
) -> [[f64; 2]; 2] {
    [
        [cov[0][0].max(min_xx), cov[0][1]],
        [cov[1][0], cov[1][1].max(min_yy)],
    ]
}

/// Calculate sample covariance from 2D points (equal weights)
fn calculate_sample_covariance(points: &[Vector2<f64>]) -> [[f64; 2]; 2] {
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

/// Calculate weighted covariance from 2D points
fn calculate_weighted_covariance(
    points: &[Vector2<f64>],
    weights: &[f64],
) -> [[f64; 2]; 2] {
    let n = points.len();
    if n < 2 || weights.len() != n {
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

/// Calculate softmax weights from scores
fn calculate_softmax_weights(scores: &[f64], temperature: f64) -> Vec<f64> {
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

/// Extract XY coordinates from transform
fn extract_xy_from_transform(transform: &Transform3f) -> Vector2<f64> {
    let flat = transform.to_flat();
    Vector2::new(flat[3] as f64, flat[7] as f64)
}

/// Convert Transform3f to nalgebra Matrix4
fn transform_to_matrix4(transform: &Transform3f) -> Matrix4<f64> {
    let flat = transform.to_flat();
    Matrix4::new(
        flat[0] as f64, flat[1] as f64, flat[2] as f64, flat[3] as f64,
        flat[4] as f64, flat[5] as f64, flat[6] as f64, flat[7] as f64,
        flat[8] as f64, flat[9] as f64, flat[10] as f64, flat[11] as f64,
        flat[12] as f64, flat[13] as f64, flat[14] as f64, flat[15] as f64,
    )
}

/// Convert nalgebra Matrix4 to Transform3f
fn matrix4_to_transform(matrix: &Matrix4<f64>) -> Transform3f {
    Transform3f::from_flat(&[
        matrix[(0, 0)] as f32, matrix[(0, 1)] as f32, matrix[(0, 2)] as f32, matrix[(0, 3)] as f32,
        matrix[(1, 0)] as f32, matrix[(1, 1)] as f32, matrix[(1, 2)] as f32, matrix[(1, 3)] as f32,
        matrix[(2, 0)] as f32, matrix[(2, 1)] as f32, matrix[(2, 2)] as f32, matrix[(2, 3)] as f32,
        matrix[(3, 0)] as f32, matrix[(3, 1)] as f32, matrix[(3, 2)] as f32, matrix[(3, 3)] as f32,
    ])
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
}
