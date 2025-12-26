//! Type definitions for NDT derivative computation.
//!
//! Based on Magnusson 2009, Chapter 6.

use nalgebra::{Matrix6, Vector6};

/// Gaussian fitting parameters for NDT score function.
///
/// The NDT score function (Eq. 6.9) is:
/// `p(x) = -d1 * exp(-d2/2 * (x-μ)ᵀΣ⁻¹(x-μ))`
///
/// These parameters control the shape of the probability distribution.
#[derive(Debug, Clone, Copy)]
pub struct GaussianParams {
    /// Amplitude of the Gaussian (-d1 in the score function).
    /// Computed as: -gauss_d1 = -(1 - outlier_ratio) / (gauss_c1 + gauss_c2)
    pub d1: f64,

    /// Exponent coefficient (-d2/2 in the score function).
    /// Computed as: -gauss_d2 / 2
    pub d2: f64,

    /// Outlier ratio (probability that a point is an outlier).
    /// Typically 0.55 for NDT.
    pub outlier_ratio: f64,
}

impl GaussianParams {
    /// Create Gaussian parameters from resolution and outlier ratio.
    ///
    /// This follows the Autoware/pclomp computation:
    /// - gauss_c1 = 10 * (1 - outlier_ratio)
    /// - gauss_c2 = outlier_ratio / resolution^3
    /// - gauss_d3 = -log(gauss_c2)
    /// - gauss_d1 = -log(gauss_c1 + gauss_c2) - gauss_d3
    /// - gauss_d2 = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1)
    ///
    /// # Arguments
    /// * `resolution` - Voxel resolution in meters
    /// * `outlier_ratio` - Probability that a point is an outlier (typically 0.55)
    pub fn new(resolution: f64, outlier_ratio: f64) -> Self {
        let gauss_c1 = 10.0 * (1.0 - outlier_ratio);
        let gauss_c2 = outlier_ratio / (resolution * resolution * resolution);
        let gauss_d3 = -gauss_c2.ln();
        let gauss_d1 = -(gauss_c1 + gauss_c2).ln() - gauss_d3;
        let gauss_d2_nom = -(gauss_c1 * (-0.5_f64).exp() + gauss_c2).ln() - gauss_d3;
        let gauss_d2 = -2.0 * (gauss_d2_nom / gauss_d1).ln();

        Self {
            d1: gauss_d1,
            d2: gauss_d2,
            outlier_ratio,
        }
    }
}

impl Default for GaussianParams {
    fn default() -> Self {
        // Default values matching Autoware's NDT
        Self::new(2.0, 0.55)
    }
}

/// Point derivatives with respect to pose parameters.
///
/// The pose is parameterized as [tx, ty, tz, roll, pitch, yaw] (6 DOF).
/// For a point x transformed by pose p, we compute:
/// - point_gradient: ∂T(x)/∂p (4x6 matrix, but stored as 3x6 since w=1)
/// - point_hessian: ∂²T(x)/∂p² (stored compactly)
#[derive(Debug, Clone)]
pub struct PointDerivatives {
    /// Gradient of transformed point w.r.t. pose [3x6].
    /// Row i is ∂(Tx)_i/∂p where (Tx)_i is the i-th component of transformed point.
    /// Column j is the derivative w.r.t. pose parameter j.
    pub point_gradient: Matrix4x6,

    /// Hessian of transformed point w.r.t. pose [24x6].
    /// This stores the second derivatives ∂²(Tx)/∂p_i∂p_j.
    /// Organized as 4 blocks of 6x6 (one per output coordinate x,y,z,w).
    pub point_hessian: Matrix24x6,
}

/// 4x6 matrix type for point gradient.
pub type Matrix4x6 =
    nalgebra::Matrix<f64, nalgebra::U4, nalgebra::U6, nalgebra::ArrayStorage<f64, 4, 6>>;

/// 24x6 matrix type for point hessian (4 coordinates × 6 pose params × 6 pose params).
pub type Matrix24x6 =
    nalgebra::Matrix<f64, nalgebra::U24, nalgebra::U6, nalgebra::ArrayStorage<f64, 24, 6>>;

/// 8x4 matrix type for angular Jacobian.
pub type Matrix8x4 =
    nalgebra::Matrix<f64, nalgebra::U8, nalgebra::U4, nalgebra::ArrayStorage<f64, 8, 4>>;

/// 16x4 matrix type for angular Hessian (was 15x4 but we use 16 for alignment).
pub type Matrix16x4 =
    nalgebra::Matrix<f64, nalgebra::U16, nalgebra::U4, nalgebra::ArrayStorage<f64, 16, 4>>;

impl PointDerivatives {
    /// Create zero-initialized point derivatives.
    pub fn zeros() -> Self {
        Self {
            point_gradient: Matrix4x6::zeros(),
            point_hessian: Matrix24x6::zeros(),
        }
    }
}

/// Result of derivative computation for a single point-voxel pair.
#[derive(Debug, Clone)]
pub struct DerivativeResult {
    /// NDT score contribution from this point-voxel pair.
    pub score: f64,

    /// Gradient of score w.r.t. pose [6x1].
    pub gradient: Vector6<f64>,

    /// Hessian of score w.r.t. pose [6x6].
    pub hessian: Matrix6<f64>,
}

impl DerivativeResult {
    /// Create zero-initialized derivative result.
    pub fn zeros() -> Self {
        Self {
            score: 0.0,
            gradient: Vector6::zeros(),
            hessian: Matrix6::zeros(),
        }
    }

    /// Add another derivative result to this one.
    pub fn accumulate(&mut self, other: &DerivativeResult) {
        self.score += other.score;
        self.gradient += other.gradient;
        self.hessian += other.hessian;
    }
}

/// Aggregated derivatives for the entire point cloud.
#[derive(Debug, Clone)]
pub struct AggregatedDerivatives {
    /// Total NDT score (sum over all point-voxel pairs).
    pub score: f64,

    /// Total gradient [6x1].
    pub gradient: Vector6<f64>,

    /// Total Hessian [6x6].
    pub hessian: Matrix6<f64>,

    /// Number of valid correspondences (points that matched voxels).
    pub num_correspondences: usize,
}

impl AggregatedDerivatives {
    /// Create zero-initialized aggregated derivatives.
    pub fn zeros() -> Self {
        Self {
            score: 0.0,
            gradient: Vector6::zeros(),
            hessian: Matrix6::zeros(),
            num_correspondences: 0,
        }
    }

    /// Add a single point-voxel derivative result.
    pub fn add(&mut self, result: &DerivativeResult) {
        self.score += result.score;
        self.gradient += result.gradient;
        self.hessian += result.hessian;
        self.num_correspondences += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_params_default() {
        let params = GaussianParams::default();

        // d1 is negative per the Autoware implementation
        // (gauss_d1 = -log(gauss_c1 + gauss_c2) - gauss_d3)
        // d2 should be positive
        assert!(params.d1 < 0.0, "d1 should be negative: {}", params.d1);
        assert!(params.d2 > 0.0, "d2 should be positive: {}", params.d2);
        assert_eq!(params.outlier_ratio, 0.55);
    }

    #[test]
    fn test_gaussian_params_custom() {
        let params = GaussianParams::new(1.0, 0.3);

        // Different resolution/outlier_ratio should give different values
        let default = GaussianParams::default();
        assert!((params.d1 - default.d1).abs() > 0.01);
    }

    #[test]
    fn test_derivative_result_accumulate() {
        let mut result1 = DerivativeResult {
            score: 1.0,
            gradient: Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            hessian: Matrix6::identity(),
        };

        let result2 = DerivativeResult {
            score: 2.0,
            gradient: Vector6::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            hessian: Matrix6::identity() * 2.0,
        };

        result1.accumulate(&result2);

        assert_eq!(result1.score, 3.0);
        assert_eq!(result1.gradient[0], 1.5);
        assert_eq!(result1.hessian[(0, 0)], 3.0);
    }

    #[test]
    fn test_aggregated_derivatives_add() {
        let mut agg = AggregatedDerivatives::zeros();

        let result = DerivativeResult {
            score: 1.5,
            gradient: Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            hessian: Matrix6::identity(),
        };

        agg.add(&result);
        assert_eq!(agg.score, 1.5);
        assert_eq!(agg.num_correspondences, 1);

        agg.add(&result);
        assert_eq!(agg.score, 3.0);
        assert_eq!(agg.num_correspondences, 2);
    }
}
