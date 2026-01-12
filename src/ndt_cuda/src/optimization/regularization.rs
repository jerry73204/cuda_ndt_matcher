//! GNSS regularization for NDT optimization.
//!
//! This module implements the regularization term that penalizes deviation from
//! a GNSS pose, helping to prevent drift in open areas where scan matching may be
//! unreliable.
//!
//! The regularization term adds a quadratic penalty in the vehicle's longitudinal
//! direction (forward/backward), weighted by the number of voxel correspondences.
//!
//! Based on Autoware's ndt_omp implementation.

use nalgebra::{Isometry3, Matrix6, Vector6};

/// Configuration for GNSS regularization.
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// Whether regularization is enabled.
    pub enabled: bool,

    /// Scale factor for regularization term (default: 0.01).
    /// Higher values give more weight to GNSS poses.
    pub scale_factor: f64,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            scale_factor: 0.01,
        }
    }
}

/// GNSS regularization term for NDT optimization.
///
/// This penalizes deviation from a reference GNSS pose in the vehicle's
/// longitudinal direction (forward/backward), which helps prevent drift
/// in open areas.
#[derive(Debug, Clone)]
pub struct RegularizationTerm {
    /// Configuration
    config: RegularizationConfig,

    /// Reference pose (from GNSS)
    reference_pose: Option<Isometry3<f64>>,

    /// Cached reference translation
    reference_translation: Option<[f64; 3]>,
}

impl RegularizationTerm {
    /// Create a new regularization term with the given configuration.
    pub fn new(config: RegularizationConfig) -> Self {
        Self {
            config,
            reference_pose: None,
            reference_translation: None,
        }
    }

    /// Create a new regularization term with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RegularizationConfig::default())
    }

    /// Check if regularization is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if a reference pose is set.
    pub fn has_reference_pose(&self) -> bool {
        self.reference_pose.is_some()
    }

    /// Get the reference translation [x, y, z] if set.
    pub fn reference_translation(&self) -> Option<[f64; 3]> {
        self.reference_translation
    }

    /// Set the reference pose (from GNSS).
    pub fn set_reference_pose(&mut self, pose: Isometry3<f64>) {
        let translation = pose.translation;
        self.reference_translation = Some([translation.x, translation.y, translation.z]);
        self.reference_pose = Some(pose);
    }

    /// Clear the reference pose.
    pub fn clear_reference_pose(&mut self) {
        self.reference_pose = None;
        self.reference_translation = None;
    }

    /// Compute the regularization contribution to score, gradient, and Hessian.
    ///
    /// # Arguments
    /// * `current_pose` - Current pose as [x, y, z, roll, pitch, yaw]
    /// * `neighborhood_count` - Number of voxel correspondences (used as weight)
    ///
    /// # Returns
    /// Tuple of (score_delta, gradient_delta, hessian_delta)
    pub fn compute_derivatives(
        &self,
        current_pose: &[f64; 6],
        neighborhood_count: usize,
    ) -> (f64, Vector6<f64>, Matrix6<f64>) {
        // If not enabled or no reference pose, return zeros
        if !self.config.enabled {
            return (0.0, Vector6::zeros(), Matrix6::zeros());
        }

        let reference = match &self.reference_translation {
            Some(t) => t,
            None => return (0.0, Vector6::zeros(), Matrix6::zeros()),
        };

        let scale = self.config.scale_factor;
        let weight = neighborhood_count as f64;

        // Current pose components
        let x = current_pose[0];
        let y = current_pose[1];
        let yaw = current_pose[5];

        // Difference from reference
        let dx = reference[0] - x;
        let dy = reference[1] - y;

        // Longitudinal distance (in vehicle frame)
        let sin_yaw = yaw.sin();
        let cos_yaw = yaw.cos();
        let longitudinal_distance = dy * sin_yaw + dx * cos_yaw;

        // Score: -scale * weight * longitudinal_distance^2
        // This is negative because we're maximizing the score (NDT score is positive)
        let score = -scale * weight * longitudinal_distance * longitudinal_distance;

        // Gradient: derivative of score with respect to pose parameters
        // grad_x = d/dx(-scale * weight * (dy*sin + dx*cos)^2)
        //        = -scale * weight * 2 * (dy*sin + dx*cos) * d/dx(dy*sin + dx*cos)
        //        = -scale * weight * 2 * longitudinal * (-cos)  // dx decreases when x increases
        //        = scale * weight * 2 * cos * longitudinal
        let mut gradient = Vector6::zeros();
        gradient[0] = scale * weight * 2.0 * cos_yaw * longitudinal_distance;
        gradient[1] = scale * weight * 2.0 * sin_yaw * longitudinal_distance;
        // Note: Autoware does not compute yaw gradient for regularization

        // Hessian: second derivatives
        // H[0,0] = d/dx(grad_x) = scale * weight * 2 * cos * d/dx(longitudinal)
        //        = scale * weight * 2 * cos * (-cos)
        //        = -scale * weight * 2 * cos^2
        let mut hessian = Matrix6::zeros();
        hessian[(0, 0)] = -scale * weight * 2.0 * cos_yaw * cos_yaw;
        hessian[(0, 1)] = -scale * weight * 2.0 * cos_yaw * sin_yaw;
        hessian[(1, 0)] = hessian[(0, 1)];
        hessian[(1, 1)] = -scale * weight * 2.0 * sin_yaw * sin_yaw;

        (score, gradient, hessian)
    }

    /// Get the scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.config.scale_factor
    }

    /// Set the scale factor.
    pub fn set_scale_factor(&mut self, scale: f64) {
        self.config.scale_factor = scale;
    }

    /// Enable or disable regularization.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_regularization_disabled() {
        let term = RegularizationTerm::with_defaults();
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (score, gradient, hessian) = term.compute_derivatives(&pose, 100);

        assert_eq!(score, 0.0);
        assert_eq!(gradient, Vector6::zeros());
        assert_eq!(hessian, Matrix6::zeros());
    }

    #[test]
    fn test_regularization_no_reference() {
        let config = RegularizationConfig {
            enabled: true,
            scale_factor: 0.01,
        };
        let term = RegularizationTerm::new(config);
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (score, gradient, hessian) = term.compute_derivatives(&pose, 100);

        assert_eq!(score, 0.0);
        assert_eq!(gradient, Vector6::zeros());
        assert_eq!(hessian, Matrix6::zeros());
    }

    #[test]
    fn test_regularization_at_reference() {
        let config = RegularizationConfig {
            enabled: true,
            scale_factor: 0.01,
        };
        let mut term = RegularizationTerm::new(config);

        // Set reference at origin
        term.set_reference_pose(Isometry3::identity());

        // Current pose at origin - no deviation
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (score, gradient, _hessian) = term.compute_derivatives(&pose, 100);

        // Score should be zero (no deviation)
        assert_relative_eq!(score, 0.0, epsilon = 1e-10);

        // Gradient should be zero at minimum
        assert_relative_eq!(gradient[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regularization_with_offset() {
        let config = RegularizationConfig {
            enabled: true,
            scale_factor: 0.01,
        };
        let mut term = RegularizationTerm::new(config);

        // Set reference at (1, 0, 0)
        term.set_reference_pose(Isometry3::translation(1.0, 0.0, 0.0));

        // Current pose at origin, yaw = 0
        // dx = 1, dy = 0, longitudinal_distance = dx * cos(0) + dy * sin(0) = 1
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let weight = 100.0;
        let scale = 0.01;
        let (score, gradient, hessian) = term.compute_derivatives(&pose, 100);

        // Score = -scale * weight * 1^2 = -0.01 * 100 * 1 = -1.0
        assert_relative_eq!(score, -scale * weight * 1.0, epsilon = 1e-10);

        // Gradient in x: scale * weight * 2 * cos(0) * 1 = 0.01 * 100 * 2 * 1 * 1 = 2.0
        assert_relative_eq!(gradient[0], scale * weight * 2.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[1], 0.0, epsilon = 1e-10);

        // Hessian[0,0] = -scale * weight * 2 * cos^2(0) = -2.0
        assert_relative_eq!(hessian[(0, 0)], -scale * weight * 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regularization_with_yaw() {
        let config = RegularizationConfig {
            enabled: true,
            scale_factor: 0.01,
        };
        let mut term = RegularizationTerm::new(config);

        // Set reference at (0, 1, 0)
        term.set_reference_pose(Isometry3::translation(0.0, 1.0, 0.0));

        // Current pose at origin, yaw = 90 degrees (pi/2)
        // dx = 0, dy = 1, longitudinal_distance = dy * sin(pi/2) + dx * cos(pi/2) = 1 * 1 + 0 * 0 = 1
        let yaw = std::f64::consts::FRAC_PI_2;
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, yaw];
        let weight = 100.0;
        let scale = 0.01;
        let (score, gradient, _hessian) = term.compute_derivatives(&pose, 100);

        // Score = -scale * weight * 1^2 = -1.0
        assert_relative_eq!(score, -scale * weight * 1.0, epsilon = 1e-10);

        // Gradient in x: scale * weight * 2 * cos(pi/2) * 1 ≈ 0
        assert_relative_eq!(gradient[0], 0.0, epsilon = 1e-10);

        // Gradient in y: scale * weight * 2 * sin(pi/2) * 1 = 2.0
        assert_relative_eq!(gradient[1], scale * weight * 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_set_and_clear_reference() {
        let config = RegularizationConfig {
            enabled: true,
            scale_factor: 0.01,
        };
        let mut term = RegularizationTerm::new(config);

        assert!(!term.has_reference_pose());

        term.set_reference_pose(Isometry3::translation(1.0, 2.0, 3.0));
        assert!(term.has_reference_pose());

        term.clear_reference_pose();
        assert!(!term.has_reference_pose());
    }
}
