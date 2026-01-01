//! Type definitions for NDT optimization.

use nalgebra::{Isometry3, Matrix6, UnitQuaternion, Vector3, Vector6};

/// Configuration for NDT scan matching.
#[derive(Debug, Clone)]
pub struct NdtConfig {
    /// Voxel resolution in meters (typically 1.0 - 4.0m).
    pub resolution: f64,

    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Convergence threshold for pose delta norm.
    /// Iteration stops when ||Δp|| < trans_epsilon.
    pub trans_epsilon: f64,

    /// Maximum step length for Newton update (Autoware default: 0.1).
    ///
    /// The Newton step direction is normalized, then scaled by
    /// `min(newton_step_norm, step_size)`. This prevents large steps
    /// when far from the optimum while allowing full steps when close.
    ///
    /// NOTE: This is NOT a damping factor - it's the maximum allowed step length.
    pub step_size: f64,

    /// Probability that a point is an outlier (typically 0.55).
    pub outlier_ratio: f64,

    /// Whether to use line search for step size.
    pub use_line_search: bool,

    /// Regularization factor for Hessian when near-singular.
    pub regularization: f64,
}

impl Default for NdtConfig {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            max_iterations: 30,
            trans_epsilon: 0.01,
            step_size: 0.1, // Autoware default: max step length (NOT a damping factor)
            outlier_ratio: 0.55,
            use_line_search: true, // Enable More-Thuente line search (matches Autoware)
            regularization: 1e-6,
        }
    }
}

impl NdtConfig {
    /// Create a new configuration with custom resolution.
    pub fn with_resolution(resolution: f64) -> Self {
        Self {
            resolution,
            ..Default::default()
        }
    }
}

/// Status of NDT optimization convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Converged: delta norm below threshold.
    Converged,

    /// Reached maximum iterations without convergence.
    MaxIterations,

    /// No valid correspondences found.
    NoCorrespondences,

    /// Hessian is singular (cannot compute Newton step).
    SingularHessian,

    /// Score increased (diverging).
    Diverged,
}

impl ConvergenceStatus {
    /// Check if the optimization converged successfully.
    pub fn is_converged(&self) -> bool {
        matches!(self, ConvergenceStatus::Converged)
    }

    /// Check if the result is usable (converged or max iterations).
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            ConvergenceStatus::Converged | ConvergenceStatus::MaxIterations
        )
    }
}

/// Result of NDT scan matching.
#[derive(Debug, Clone)]
pub struct NdtResult {
    /// Final pose estimate (transformation from source to target frame).
    pub pose: Isometry3<f64>,

    /// Convergence status.
    pub status: ConvergenceStatus,

    /// Final NDT score (higher is better, maximum at perfect alignment).
    pub score: f64,

    /// Transform probability (normalized score).
    pub transform_probability: f64,

    /// Nearest voxel transformation likelihood (NVTL).
    pub nvtl: f64,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Final Hessian matrix (useful for covariance estimation).
    pub hessian: Matrix6<f64>,

    /// Number of valid correspondences in final iteration.
    pub num_correspondences: usize,

    /// Maximum consecutive oscillation count detected during optimization.
    /// Oscillation indicates the optimizer is bouncing between poses,
    /// potentially stuck in a local minimum.
    pub oscillation_count: usize,
}

impl NdtResult {
    /// Create a result indicating no correspondences were found.
    pub fn no_correspondences(initial_pose: Isometry3<f64>) -> Self {
        Self {
            pose: initial_pose,
            status: ConvergenceStatus::NoCorrespondences,
            score: 0.0,
            transform_probability: 0.0,
            nvtl: 0.0,
            iterations: 0,
            hessian: Matrix6::zeros(),
            num_correspondences: 0,
            oscillation_count: 0,
        }
    }

    /// Check if oscillation was detected (count exceeds threshold).
    pub fn is_oscillating(&self) -> bool {
        self.oscillation_count > super::oscillation::DEFAULT_OSCILLATION_THRESHOLD
    }
}

/// Convert a 6-DOF pose vector [tx, ty, tz, roll, pitch, yaw] to an Isometry3.
pub fn pose_vector_to_isometry(pose: &[f64; 6]) -> Isometry3<f64> {
    let translation = Vector3::new(pose[0], pose[1], pose[2]);
    let rotation = UnitQuaternion::from_euler_angles(pose[3], pose[4], pose[5]);
    Isometry3::from_parts(translation.into(), rotation)
}

/// Convert an Isometry3 to a 6-DOF pose vector [tx, ty, tz, roll, pitch, yaw].
pub fn isometry_to_pose_vector(isometry: &Isometry3<f64>) -> [f64; 6] {
    let translation = isometry.translation.vector;
    let (roll, pitch, yaw) = isometry.rotation.euler_angles();
    [
        translation.x,
        translation.y,
        translation.z,
        roll,
        pitch,
        yaw,
    ]
}

/// Apply a delta vector to a pose, returning the updated pose.
///
/// The delta is in the local frame and is applied as:
/// new_pose = old_pose * exp(delta)
///
/// For small deltas, this is approximately:
/// new_pose ≈ old_pose + delta
pub fn apply_pose_delta(pose: &[f64; 6], delta: &Vector6<f64>, step_size: f64) -> [f64; 6] {
    let scaled_delta = delta * step_size;

    // Apply translation delta
    let new_tx = pose[0] + scaled_delta[0];
    let new_ty = pose[1] + scaled_delta[1];
    let new_tz = pose[2] + scaled_delta[2];

    // Apply rotation delta (Euler angle increment)
    let new_roll = pose[3] + scaled_delta[3];
    let new_pitch = pose[4] + scaled_delta[4];
    let new_yaw = pose[5] + scaled_delta[5];

    [new_tx, new_ty, new_tz, new_roll, new_pitch, new_yaw]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn test_config_default() {
        let config = NdtConfig::default();
        assert_eq!(config.resolution, 2.0);
        assert_eq!(config.max_iterations, 30);
        assert!(config.use_line_search); // More-Thuente is now default
    }

    #[test]
    fn test_config_with_resolution() {
        let config = NdtConfig::with_resolution(1.0);
        assert_eq!(config.resolution, 1.0);
        assert_eq!(config.max_iterations, 30); // Other defaults preserved
    }

    #[test]
    fn test_convergence_status() {
        assert!(ConvergenceStatus::Converged.is_converged());
        assert!(!ConvergenceStatus::MaxIterations.is_converged());

        assert!(ConvergenceStatus::Converged.is_usable());
        assert!(ConvergenceStatus::MaxIterations.is_usable());
        assert!(!ConvergenceStatus::NoCorrespondences.is_usable());
    }

    #[test]
    fn test_pose_vector_roundtrip() {
        let pose = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let isometry = pose_vector_to_isometry(&pose);
        let recovered = isometry_to_pose_vector(&isometry);

        for i in 0..6 {
            assert_relative_eq!(pose[i], recovered[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pose_vector_identity() {
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let isometry = pose_vector_to_isometry(&pose);

        assert!(isometry.rotation.angle() < 1e-10);
        assert!(isometry.translation.vector.norm() < 1e-10);
    }

    #[test]
    fn test_apply_pose_delta_translation() {
        let pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let delta = Vector6::new(0.1, 0.2, 0.3, 0.0, 0.0, 0.0);

        let new_pose = apply_pose_delta(&pose, &delta, 1.0);

        assert_relative_eq!(new_pose[0], 1.1, epsilon = 1e-10);
        assert_relative_eq!(new_pose[1], 2.2, epsilon = 1e-10);
        assert_relative_eq!(new_pose[2], 3.3, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_pose_delta_with_step_size() {
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delta = Vector6::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

        let new_pose = apply_pose_delta(&pose, &delta, 0.5);

        assert_relative_eq!(new_pose[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(new_pose[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(new_pose[2], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_pose_delta_rotation() {
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let delta = Vector6::new(0.0, 0.0, 0.0, FRAC_PI_4, 0.0, 0.0);

        let new_pose = apply_pose_delta(&pose, &delta, 1.0);

        assert_relative_eq!(new_pose[3], FRAC_PI_4, epsilon = 1e-10);
    }
}
