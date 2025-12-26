//! Covariance estimation for NDT scan matching.
//!
//! This module implements covariance estimation modes:
//! - FIXED: Static covariance matrix from configuration
//! - LAPLACE: Inverse of Hessian matrix (2x2 XY block)

use crate::params::{CovarianceEstimationType, CovarianceParams};
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
