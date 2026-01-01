//! Oscillation detection for NDT optimization.
//!
//! Detects when the optimizer is oscillating between poses, which can indicate
//! that it's stuck in a local minimum. This is used to determine convergence
//! status even when max iterations is reached.
//!
//! Based on Autoware's ndt_scan_matcher implementation.

use nalgebra::{Isometry3, Vector3};

/// Threshold for detecting direction reversal.
/// When the cosine of the angle between consecutive movement vectors is below
/// this threshold, it indicates the optimizer is moving in the opposite direction.
/// -0.9 corresponds to an angle of about 154 degrees (almost opposite).
const INVERSION_VECTOR_THRESHOLD: f64 = -0.9;

/// Default threshold for oscillation count that triggers a warning.
pub const DEFAULT_OSCILLATION_THRESHOLD: usize = 10;

/// Result of oscillation analysis.
#[derive(Debug, Clone)]
pub struct OscillationResult {
    /// Maximum consecutive oscillation count detected.
    pub max_oscillation_count: usize,
    /// Whether oscillation was detected (count exceeds threshold).
    pub is_oscillating: bool,
    /// Indices where oscillation was detected.
    pub oscillation_indices: Vec<usize>,
}

impl Default for OscillationResult {
    fn default() -> Self {
        Self {
            max_oscillation_count: 0,
            is_oscillating: false,
            oscillation_indices: Vec::new(),
        }
    }
}

/// Count oscillations in a sequence of poses.
///
/// An oscillation is detected when consecutive movement vectors point in
/// approximately opposite directions (cosine < -0.9).
///
/// # Arguments
/// * `poses` - Sequence of poses from each optimization iteration
///
/// # Returns
/// The maximum number of consecutive oscillations detected.
///
/// # Example
/// ```ignore
/// let poses = vec![pose1, pose2, pose3, pose4];
/// let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
/// if result.is_oscillating {
///     println!("Warning: Optimizer oscillating with {} reversals", result.max_oscillation_count);
/// }
/// ```
pub fn count_oscillation(poses: &[Isometry3<f64>], threshold: usize) -> OscillationResult {
    if poses.len() < 3 {
        return OscillationResult::default();
    }

    let mut oscillation_cnt = 0;
    let mut max_oscillation_cnt = 0;
    let mut oscillation_indices = Vec::new();

    for i in 2..poses.len() {
        let current_pos = poses[i].translation.vector;
        let prev_pos = poses[i - 1].translation.vector;
        let prev_prev_pos = poses[i - 2].translation.vector;

        // Compute movement vectors
        let current_vec = current_pos - prev_pos;
        let prev_vec = prev_pos - prev_prev_pos;

        // Skip if either vector is too small (avoid division by zero)
        let current_norm = current_vec.norm();
        let prev_norm = prev_vec.norm();
        if current_norm < 1e-10 || prev_norm < 1e-10 {
            oscillation_cnt = 0;
            continue;
        }

        // Normalize and compute cosine
        let current_normalized = current_vec / current_norm;
        let prev_normalized = prev_vec / prev_norm;
        let cosine_value = current_normalized.dot(&prev_normalized);

        // Check for oscillation (direction reversal)
        let is_oscillation = cosine_value < INVERSION_VECTOR_THRESHOLD;

        if is_oscillation {
            oscillation_cnt += 1;
            oscillation_indices.push(i);
        } else {
            oscillation_cnt = 0;
        }

        max_oscillation_cnt = max_oscillation_cnt.max(oscillation_cnt);
    }

    OscillationResult {
        max_oscillation_count: max_oscillation_cnt,
        is_oscillating: max_oscillation_cnt > threshold,
        oscillation_indices,
    }
}

/// Count oscillations from a sequence of [x, y, z, roll, pitch, yaw] poses.
///
/// This is a convenience function for when poses are stored as arrays.
pub fn count_oscillation_from_arrays(poses: &[[f64; 6]], threshold: usize) -> OscillationResult {
    if poses.len() < 3 {
        return OscillationResult::default();
    }

    let mut oscillation_cnt = 0;
    let mut max_oscillation_cnt = 0;
    let mut oscillation_indices = Vec::new();

    for i in 2..poses.len() {
        let current_pos = Vector3::new(poses[i][0], poses[i][1], poses[i][2]);
        let prev_pos = Vector3::new(poses[i - 1][0], poses[i - 1][1], poses[i - 1][2]);
        let prev_prev_pos = Vector3::new(poses[i - 2][0], poses[i - 2][1], poses[i - 2][2]);

        // Compute movement vectors
        let current_vec = current_pos - prev_pos;
        let prev_vec = prev_pos - prev_prev_pos;

        // Skip if either vector is too small
        let current_norm = current_vec.norm();
        let prev_norm = prev_vec.norm();
        if current_norm < 1e-10 || prev_norm < 1e-10 {
            oscillation_cnt = 0;
            continue;
        }

        // Normalize and compute cosine
        let current_normalized = current_vec / current_norm;
        let prev_normalized = prev_vec / prev_norm;
        let cosine_value = current_normalized.dot(&prev_normalized);

        // Check for oscillation
        let is_oscillation = cosine_value < INVERSION_VECTOR_THRESHOLD;

        if is_oscillation {
            oscillation_cnt += 1;
            oscillation_indices.push(i);
        } else {
            oscillation_cnt = 0;
        }

        max_oscillation_cnt = max_oscillation_cnt.max(oscillation_cnt);
    }

    OscillationResult {
        max_oscillation_count: max_oscillation_cnt,
        is_oscillating: max_oscillation_cnt > threshold,
        oscillation_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    fn make_pose(x: f64, y: f64, z: f64) -> Isometry3<f64> {
        Isometry3::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity())
    }

    #[test]
    fn test_no_oscillation_linear_motion() {
        // Linear motion in +X direction - no oscillation
        let poses = vec![
            make_pose(0.0, 0.0, 0.0),
            make_pose(1.0, 0.0, 0.0),
            make_pose(2.0, 0.0, 0.0),
            make_pose(3.0, 0.0, 0.0),
            make_pose(4.0, 0.0, 0.0),
        ];

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert_eq!(result.max_oscillation_count, 0);
        assert!(!result.is_oscillating);
    }

    #[test]
    fn test_no_oscillation_curved_motion() {
        // Curved motion (not reversing direction) - no oscillation
        let poses = vec![
            make_pose(0.0, 0.0, 0.0),
            make_pose(1.0, 0.0, 0.0),
            make_pose(1.5, 0.5, 0.0),
            make_pose(1.8, 1.0, 0.0),
            make_pose(2.0, 1.5, 0.0),
        ];

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert_eq!(result.max_oscillation_count, 0);
        assert!(!result.is_oscillating);
    }

    #[test]
    fn test_oscillation_back_and_forth() {
        // Oscillating back and forth in X direction
        let poses = vec![
            make_pose(0.0, 0.0, 0.0),
            make_pose(1.0, 0.0, 0.0),
            make_pose(0.0, 0.0, 0.0), // reversal
            make_pose(1.0, 0.0, 0.0), // reversal
            make_pose(0.0, 0.0, 0.0), // reversal
        ];

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        // Oscillations at indices 2, 3, 4 (consecutive)
        assert!(result.max_oscillation_count >= 2);
        assert!(!result.is_oscillating); // Still below threshold of 10
    }

    #[test]
    fn test_oscillation_threshold_exceeded() {
        // Many oscillations that exceed the threshold
        let mut poses = vec![make_pose(0.0, 0.0, 0.0)];
        for i in 1..=25 {
            let x = if i % 2 == 0 { 0.0 } else { 1.0 };
            poses.push(make_pose(x, 0.0, 0.0));
        }

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert!(result.max_oscillation_count > 10);
        assert!(result.is_oscillating);
    }

    #[test]
    fn test_oscillation_resets_on_forward_motion() {
        // Some oscillation followed by forward motion should reset counter
        let poses = vec![
            make_pose(0.0, 0.0, 0.0),
            make_pose(1.0, 0.0, 0.0),
            make_pose(0.0, 0.0, 0.0), // reversal 1
            make_pose(1.0, 0.0, 0.0), // reversal 2
            // Now forward motion
            make_pose(2.0, 0.0, 0.0),
            make_pose(3.0, 0.0, 0.0),
            make_pose(4.0, 0.0, 0.0),
        ];

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        // Max consecutive oscillations is 2 (at indices 2 and 3)
        assert_eq!(result.max_oscillation_count, 2);
        assert!(!result.is_oscillating);
    }

    #[test]
    fn test_too_few_poses() {
        // Less than 3 poses - can't detect oscillation
        let poses = vec![make_pose(0.0, 0.0, 0.0), make_pose(1.0, 0.0, 0.0)];

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert_eq!(result.max_oscillation_count, 0);
        assert!(!result.is_oscillating);
    }

    #[test]
    fn test_empty_poses() {
        let poses: Vec<Isometry3<f64>> = vec![];
        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert_eq!(result.max_oscillation_count, 0);
        assert!(!result.is_oscillating);
    }

    #[test]
    fn test_stationary_poses() {
        // All poses at the same location - no movement, no oscillation
        let poses = vec![
            make_pose(1.0, 2.0, 3.0),
            make_pose(1.0, 2.0, 3.0),
            make_pose(1.0, 2.0, 3.0),
            make_pose(1.0, 2.0, 3.0),
        ];

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert_eq!(result.max_oscillation_count, 0);
        assert!(!result.is_oscillating);
    }

    #[test]
    fn test_oscillation_from_arrays() {
        // Test the array-based interface
        let poses = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // reversal
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // reversal
        ];

        let result = count_oscillation_from_arrays(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert!(result.max_oscillation_count >= 1);
    }

    #[test]
    fn test_3d_oscillation() {
        // Oscillation in 3D space
        let poses = vec![
            make_pose(0.0, 0.0, 0.0),
            make_pose(1.0, 1.0, 1.0),
            make_pose(0.0, 0.0, 0.0), // reversal in 3D
            make_pose(1.0, 1.0, 1.0), // reversal in 3D
        ];

        let result = count_oscillation(&poses, DEFAULT_OSCILLATION_THRESHOLD);
        assert!(result.max_oscillation_count >= 1);
    }
}
