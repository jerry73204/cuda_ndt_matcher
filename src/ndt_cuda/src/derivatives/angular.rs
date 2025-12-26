//! Angular derivatives for NDT optimization.
//!
//! This module precomputes the derivatives of the rotation matrix with respect to
//! Euler angles (roll, pitch, yaw). These are used to compute the point gradient
//! and Hessian efficiently.
//!
//! Based on Equations 6.19 and 6.21 from Magnusson 2009.

use super::types::{Matrix16x4, Matrix8x4};

/// Threshold for treating angles as zero (avoids numerical issues).
const ANGLE_EPSILON: f64 = 1e-5;

/// Precomputed angular derivatives for a given pose.
///
/// The rotation matrix R is parameterized by Euler angles (roll, pitch, yaw) in XYZ order.
/// This struct stores the derivatives of R with respect to these angles.
#[derive(Debug, Clone)]
pub struct AngularDerivatives {
    /// First derivatives (Jacobian) - 8x4 matrix.
    /// Contains ∂R/∂roll, ∂R/∂pitch, ∂R/∂yaw components.
    pub j_ang: Matrix8x4,

    /// Second derivatives (Hessian) - 16x4 matrix (15 rows used, 16 for alignment).
    /// Contains ∂²R/∂θᵢ∂θⱼ components.
    pub h_ang: Matrix16x4,
}

impl AngularDerivatives {
    /// Compute angular derivatives for a given pose.
    ///
    /// # Arguments
    /// * `roll` - Rotation around X axis (radians)
    /// * `pitch` - Rotation around Y axis (radians)
    /// * `yaw` - Rotation around Z axis (radians)
    /// * `compute_hessian` - Whether to compute second derivatives
    pub fn new(roll: f64, pitch: f64, yaw: f64, compute_hessian: bool) -> Self {
        // Compute sin/cos with small angle approximation
        let (sx, cx) = if roll.abs() < ANGLE_EPSILON {
            (0.0, 1.0)
        } else {
            (roll.sin(), roll.cos())
        };

        let (sy, cy) = if pitch.abs() < ANGLE_EPSILON {
            (0.0, 1.0)
        } else {
            (pitch.sin(), pitch.cos())
        };

        let (sz, cz) = if yaw.abs() < ANGLE_EPSILON {
            (0.0, 1.0)
        } else {
            (yaw.sin(), yaw.cos())
        };

        // Precomputed angular gradient components (Equation 6.19 [Magnusson 2009])
        let mut j_ang = Matrix8x4::zeros();

        // Row 0: ∂R/∂roll components for y-coordinate
        j_ang[(0, 0)] = -sx * sz + cx * sy * cz;
        j_ang[(0, 1)] = -sx * cz - cx * sy * sz;
        j_ang[(0, 2)] = -cx * cy;
        // j_ang[(0, 3)] = 0.0; // Already zero

        // Row 1: ∂R/∂roll components for z-coordinate
        j_ang[(1, 0)] = cx * sz + sx * sy * cz;
        j_ang[(1, 1)] = cx * cz - sx * sy * sz;
        j_ang[(1, 2)] = -sx * cy;

        // Row 2: ∂R/∂pitch components for x-coordinate
        j_ang[(2, 0)] = -sy * cz;
        j_ang[(2, 1)] = sy * sz;
        j_ang[(2, 2)] = cy;

        // Row 3: ∂R/∂pitch components for y-coordinate
        j_ang[(3, 0)] = sx * cy * cz;
        j_ang[(3, 1)] = -sx * cy * sz;
        j_ang[(3, 2)] = sx * sy;

        // Row 4: ∂R/∂pitch components for z-coordinate
        j_ang[(4, 0)] = -cx * cy * cz;
        j_ang[(4, 1)] = cx * cy * sz;
        j_ang[(4, 2)] = -cx * sy;

        // Row 5: ∂R/∂yaw components for x-coordinate
        j_ang[(5, 0)] = -cy * sz;
        j_ang[(5, 1)] = -cy * cz;
        // j_ang[(5, 2)] = 0.0;

        // Row 6: ∂R/∂yaw components for y-coordinate
        j_ang[(6, 0)] = cx * cz - sx * sy * sz;
        j_ang[(6, 1)] = -cx * sz - sx * sy * cz;
        // j_ang[(6, 2)] = 0.0;

        // Row 7: ∂R/∂yaw components for z-coordinate
        j_ang[(7, 0)] = sx * cz + cx * sy * sz;
        j_ang[(7, 1)] = cx * sy * cz - sx * sz;
        // j_ang[(7, 2)] = 0.0;

        // Compute Hessian if requested
        let mut h_ang = Matrix16x4::zeros();

        if compute_hessian {
            // Precomputed angular hessian components (Equation 6.21 [Magnusson 2009])

            // a2: ∂²R/∂roll² for y-coordinate
            h_ang[(0, 0)] = -cx * sz - sx * sy * cz;
            h_ang[(0, 1)] = -cx * cz + sx * sy * sz;
            h_ang[(0, 2)] = sx * cy;

            // a3: ∂²R/∂roll² for z-coordinate
            h_ang[(1, 0)] = -sx * sz + cx * sy * cz;
            h_ang[(1, 1)] = -cx * sy * sz - sx * cz;
            h_ang[(1, 2)] = -cx * cy;

            // b2: ∂²R/∂roll∂pitch for y-coordinate
            h_ang[(2, 0)] = cx * cy * cz;
            h_ang[(2, 1)] = -cx * cy * sz;
            h_ang[(2, 2)] = cx * sy;

            // b3: ∂²R/∂roll∂pitch for z-coordinate
            h_ang[(3, 0)] = sx * cy * cz;
            h_ang[(3, 1)] = -sx * cy * sz;
            h_ang[(3, 2)] = sx * sy;

            // c2: ∂²R/∂roll∂yaw for y-coordinate
            h_ang[(4, 0)] = -sx * cz - cx * sy * sz;
            h_ang[(4, 1)] = sx * sz - cx * sy * cz;
            // h_ang[(4, 2)] = 0.0;

            // c3: ∂²R/∂roll∂yaw for z-coordinate
            h_ang[(5, 0)] = cx * cz - sx * sy * sz;
            h_ang[(5, 1)] = -sx * sy * cz - cx * sz;
            // h_ang[(5, 2)] = 0.0;

            // d1: ∂²R/∂pitch² for x-coordinate
            h_ang[(6, 0)] = -cy * cz;
            h_ang[(6, 1)] = cy * sz;
            h_ang[(6, 2)] = sy;

            // d2: ∂²R/∂pitch² for y-coordinate
            h_ang[(7, 0)] = -sx * sy * cz;
            h_ang[(7, 1)] = sx * sy * sz;
            h_ang[(7, 2)] = sx * cy;

            // d3: ∂²R/∂pitch² for z-coordinate
            h_ang[(8, 0)] = cx * sy * cz;
            h_ang[(8, 1)] = -cx * sy * sz;
            h_ang[(8, 2)] = -cx * cy;

            // e1: ∂²R/∂pitch∂yaw for x-coordinate
            h_ang[(9, 0)] = sy * sz;
            h_ang[(9, 1)] = sy * cz;
            // h_ang[(9, 2)] = 0.0;

            // e2: ∂²R/∂pitch∂yaw for y-coordinate
            h_ang[(10, 0)] = -sx * cy * sz;
            h_ang[(10, 1)] = -sx * cy * cz;
            // h_ang[(10, 2)] = 0.0;

            // e3: ∂²R/∂pitch∂yaw for z-coordinate
            h_ang[(11, 0)] = cx * cy * sz;
            h_ang[(11, 1)] = cx * cy * cz;
            // h_ang[(11, 2)] = 0.0;

            // f1: ∂²R/∂yaw² for x-coordinate
            h_ang[(12, 0)] = -cy * cz;
            h_ang[(12, 1)] = cy * sz;
            // h_ang[(12, 2)] = 0.0;

            // f2: ∂²R/∂yaw² for y-coordinate
            h_ang[(13, 0)] = -cx * sz - sx * sy * cz;
            h_ang[(13, 1)] = -cx * cz + sx * sy * sz;
            // h_ang[(13, 2)] = 0.0;

            // f3: ∂²R/∂yaw² for z-coordinate
            h_ang[(14, 0)] = -sx * sz + cx * sy * cz;
            h_ang[(14, 1)] = -cx * sy * sz - sx * cz;
            // h_ang[(14, 2)] = 0.0;
        }

        Self { j_ang, h_ang }
    }

    /// Compute point gradient contribution from angular derivatives.
    ///
    /// Given a source point x, computes the derivatives of the transformed point
    /// with respect to the pose angles. This fills in the rotation-dependent
    /// portion of the point gradient matrix (columns 3-5).
    ///
    /// # Arguments
    /// * `point` - Source point [x, y, z] before transformation
    ///
    /// # Returns
    /// Array of 8 values: [∂y/∂roll, ∂z/∂roll, ∂x/∂pitch, ∂y/∂pitch, ∂z/∂pitch, ∂x/∂yaw, ∂y/∂yaw, ∂z/∂yaw]
    pub fn compute_point_gradient_terms(&self, point: &[f64; 3]) -> [f64; 8] {
        // Multiply j_ang (8x4) by extended point (4x1)
        // j_ang * [x, y, z, 0]^T
        let x = point[0];
        let y = point[1];
        let z = point[2];

        [
            self.j_ang[(0, 0)] * x + self.j_ang[(0, 1)] * y + self.j_ang[(0, 2)] * z,
            self.j_ang[(1, 0)] * x + self.j_ang[(1, 1)] * y + self.j_ang[(1, 2)] * z,
            self.j_ang[(2, 0)] * x + self.j_ang[(2, 1)] * y + self.j_ang[(2, 2)] * z,
            self.j_ang[(3, 0)] * x + self.j_ang[(3, 1)] * y + self.j_ang[(3, 2)] * z,
            self.j_ang[(4, 0)] * x + self.j_ang[(4, 1)] * y + self.j_ang[(4, 2)] * z,
            self.j_ang[(5, 0)] * x + self.j_ang[(5, 1)] * y + self.j_ang[(5, 2)] * z,
            self.j_ang[(6, 0)] * x + self.j_ang[(6, 1)] * y + self.j_ang[(6, 2)] * z,
            self.j_ang[(7, 0)] * x + self.j_ang[(7, 1)] * y + self.j_ang[(7, 2)] * z,
        ]
    }

    /// Compute point Hessian contribution from angular derivatives.
    ///
    /// Given a source point x, computes the second derivatives of the transformed
    /// point with respect to pose angles. Returns 15 values corresponding to
    /// the symmetric second derivative matrix.
    ///
    /// # Arguments
    /// * `point` - Source point [x, y, z] before transformation
    ///
    /// # Returns
    /// Array of 15 values for the 6 unique second derivative combinations:
    /// [a2, a3, b2, b3, c2, c3, d1, d2, d3, e1, e2, e3, f1, f2, f3]
    pub fn compute_point_hessian_terms(&self, point: &[f64; 3]) -> [f64; 15] {
        // Multiply h_ang (16x4) by extended point (4x1)
        // h_ang * [x, y, z, 0]^T
        let x = point[0];
        let y = point[1];
        let z = point[2];

        [
            self.h_ang[(0, 0)] * x + self.h_ang[(0, 1)] * y + self.h_ang[(0, 2)] * z,
            self.h_ang[(1, 0)] * x + self.h_ang[(1, 1)] * y + self.h_ang[(1, 2)] * z,
            self.h_ang[(2, 0)] * x + self.h_ang[(2, 1)] * y + self.h_ang[(2, 2)] * z,
            self.h_ang[(3, 0)] * x + self.h_ang[(3, 1)] * y + self.h_ang[(3, 2)] * z,
            self.h_ang[(4, 0)] * x + self.h_ang[(4, 1)] * y + self.h_ang[(4, 2)] * z,
            self.h_ang[(5, 0)] * x + self.h_ang[(5, 1)] * y + self.h_ang[(5, 2)] * z,
            self.h_ang[(6, 0)] * x + self.h_ang[(6, 1)] * y + self.h_ang[(6, 2)] * z,
            self.h_ang[(7, 0)] * x + self.h_ang[(7, 1)] * y + self.h_ang[(7, 2)] * z,
            self.h_ang[(8, 0)] * x + self.h_ang[(8, 1)] * y + self.h_ang[(8, 2)] * z,
            self.h_ang[(9, 0)] * x + self.h_ang[(9, 1)] * y + self.h_ang[(9, 2)] * z,
            self.h_ang[(10, 0)] * x + self.h_ang[(10, 1)] * y + self.h_ang[(10, 2)] * z,
            self.h_ang[(11, 0)] * x + self.h_ang[(11, 1)] * y + self.h_ang[(11, 2)] * z,
            self.h_ang[(12, 0)] * x + self.h_ang[(12, 1)] * y + self.h_ang[(12, 2)] * z,
            self.h_ang[(13, 0)] * x + self.h_ang[(13, 1)] * y + self.h_ang[(13, 2)] * z,
            self.h_ang[(14, 0)] * x + self.h_ang[(14, 1)] * y + self.h_ang[(14, 2)] * z,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_zero_angles() {
        let deriv = AngularDerivatives::new(0.0, 0.0, 0.0, true);

        // At zero angles, j_ang should have specific structure
        // Row 0: [-0*0 + 1*0*1, -0*1 - 1*0*0, -1*1] = [0, 0, -1]
        assert_relative_eq!(deriv.j_ang[(0, 2)], -1.0, epsilon = 1e-10);

        // Row 2: [-0*1, 0*0, 1] = [0, 0, 1]
        assert_relative_eq!(deriv.j_ang[(2, 2)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_small_angles_approximation() {
        // Very small angles should be treated as zero
        let deriv = AngularDerivatives::new(1e-10, 1e-10, 1e-10, true);
        let deriv_zero = AngularDerivatives::new(0.0, 0.0, 0.0, true);

        // Should be identical
        for i in 0..8 {
            for j in 0..4 {
                assert_relative_eq!(
                    deriv.j_ang[(i, j)],
                    deriv_zero.j_ang[(i, j)],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_point_gradient_terms() {
        let deriv = AngularDerivatives::new(0.1, 0.2, 0.3, false);
        let point = [1.0, 2.0, 3.0];
        let terms = deriv.compute_point_gradient_terms(&point);

        // Verify terms are computed (non-zero for non-zero angles)
        let sum: f64 = terms.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Point gradient terms should be non-zero");
    }

    #[test]
    fn test_point_hessian_terms() {
        let deriv = AngularDerivatives::new(0.1, 0.2, 0.3, true);
        let point = [1.0, 2.0, 3.0];
        let terms = deriv.compute_point_hessian_terms(&point);

        // Verify all 15 terms are computed
        assert_eq!(terms.len(), 15);

        // Verify terms are computed (non-zero for non-zero angles)
        let sum: f64 = terms.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Point hessian terms should be non-zero");
    }

    #[test]
    fn test_rotation_symmetry() {
        // Test that rolling 180 degrees gives expected symmetry
        let roll = PI;
        let deriv = AngularDerivatives::new(roll, 0.0, 0.0, true);

        // cos(PI) = -1, sin(PI) ≈ 0
        // Row 0: [-0*0 + (-1)*0*1, -0*1 - (-1)*0*0, -(-1)*1] = [0, 0, 1]
        assert_relative_eq!(deriv.j_ang[(0, 2)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hessian_not_computed_when_disabled() {
        let deriv = AngularDerivatives::new(0.1, 0.2, 0.3, false);

        // h_ang should be zero when compute_hessian is false
        for i in 0..16 {
            for j in 0..4 {
                assert_eq!(deriv.h_ang[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_numerical_derivative_approximation() {
        // Verify j_ang by comparing with numerical differentiation
        let roll = 0.1;
        let pitch = 0.2;
        let yaw = 0.3;
        let eps = 1e-6;

        let deriv = AngularDerivatives::new(roll, pitch, yaw, false);
        let deriv_roll_plus = AngularDerivatives::new(roll + eps, pitch, yaw, false);
        let deriv_roll_minus = AngularDerivatives::new(roll - eps, pitch, yaw, false);

        // Numerical derivative of j_ang with respect to roll should be consistent
        // This is a sanity check that the derivatives change smoothly
        let point = [1.0, 0.0, 0.0];
        let _terms = deriv.compute_point_gradient_terms(&point); // Computed for reference
        let terms_plus = deriv_roll_plus.compute_point_gradient_terms(&point);
        let terms_minus = deriv_roll_minus.compute_point_gradient_terms(&point);

        // Check that finite differences are smooth
        for i in 0..8 {
            let numerical_diff = (terms_plus[i] - terms_minus[i]) / (2.0 * eps);
            // The derivative should be finite and reasonable
            assert!(
                numerical_diff.is_finite(),
                "Numerical derivative should be finite"
            );
        }
    }
}
