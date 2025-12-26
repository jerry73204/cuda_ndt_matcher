//! Type definitions for voxel grid structures.

use nalgebra::{Matrix3, Vector3};

/// Configuration for voxel grid construction.
#[derive(Debug, Clone)]
pub struct VoxelGridConfig {
    /// Voxel resolution (side length in meters).
    pub resolution: f32,
    /// Minimum number of points per voxel for a valid Gaussian.
    /// Voxels with fewer points are discarded.
    pub min_points_per_voxel: usize,
    /// Regularization factor for covariance eigenvalues.
    /// Small eigenvalues are clamped to this fraction of the largest.
    pub eigenvalue_ratio_threshold: f32,
}

impl Default for VoxelGridConfig {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            min_points_per_voxel: 6,
            eigenvalue_ratio_threshold: 0.01,
        }
    }
}

/// A single voxel containing Gaussian distribution parameters.
#[derive(Debug, Clone)]
pub struct Voxel {
    /// Voxel center (mean of contained points).
    pub mean: Vector3<f32>,
    /// 3x3 covariance matrix.
    pub covariance: Matrix3<f32>,
    /// Inverse of the covariance matrix.
    pub inv_covariance: Matrix3<f32>,
    /// Number of points used to compute this voxel's statistics.
    pub point_count: usize,
}

impl Voxel {
    /// Create a new voxel from accumulated point statistics.
    ///
    /// # Arguments
    /// * `sum` - Sum of all points in this voxel
    /// * `sum_sq` - Sum of outer products (x * x^T) for all points
    /// * `count` - Number of points
    /// * `config` - Configuration for regularization
    ///
    /// Returns `None` if the voxel has too few points or the covariance is degenerate.
    pub fn from_statistics(
        sum: &Vector3<f64>,
        sum_sq: &Matrix3<f64>,
        count: usize,
        config: &VoxelGridConfig,
    ) -> Option<Self> {
        if count < config.min_points_per_voxel {
            return None;
        }

        let n = count as f64;

        // Compute mean
        let mean = sum / n;

        // Compute covariance using the formula:
        // Cov = (sum_sq - n * mean * mean^T) / (n - 1)
        let mean_outer = mean * mean.transpose();
        let covariance = (sum_sq - mean_outer * n) / (n - 1.0);

        // Regularize covariance to avoid singularity
        let (regularized, inv_cov) =
            regularize_covariance(&covariance, config.eigenvalue_ratio_threshold)?;

        Some(Self {
            mean: Vector3::new(mean.x as f32, mean.y as f32, mean.z as f32),
            covariance: regularized.cast::<f32>(),
            inv_covariance: inv_cov.cast::<f32>(),
            point_count: count,
        })
    }
}

/// Regularize a covariance matrix by inflating small eigenvalues.
///
/// This prevents numerical issues when inverting near-singular matrices.
/// Small eigenvalues are clamped to `ratio_threshold * max_eigenvalue`.
///
/// Returns `(regularized_covariance, inverse_covariance)` or `None` if all
/// eigenvalues are zero.
fn regularize_covariance(
    cov: &Matrix3<f64>,
    ratio_threshold: f32,
) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
    // Symmetric eigenvalue decomposition
    let eigen = cov.symmetric_eigen();
    let mut eigenvalues = eigen.eigenvalues;

    // Find max eigenvalue
    let max_eigenvalue = eigenvalues.iter().copied().fold(0.0_f64, f64::max);

    if max_eigenvalue <= 0.0 {
        return None;
    }

    let min_eigenvalue = max_eigenvalue * ratio_threshold as f64;

    // Clamp small eigenvalues
    for ev in eigenvalues.iter_mut() {
        if *ev < min_eigenvalue {
            *ev = min_eigenvalue;
        }
    }

    // Reconstruct covariance: V * D * V^T
    let eigenvectors = &eigen.eigenvectors;
    let diag = Matrix3::from_diagonal(&eigenvalues);
    let regularized = eigenvectors * diag * eigenvectors.transpose();

    // Compute inverse: V * D^{-1} * V^T
    let inv_eigenvalues = Vector3::new(
        1.0 / eigenvalues[0],
        1.0 / eigenvalues[1],
        1.0 / eigenvalues[2],
    );
    let inv_diag = Matrix3::from_diagonal(&inv_eigenvalues);
    let inverse = eigenvectors * inv_diag * eigenvectors.transpose();

    Some((regularized, inverse))
}

/// 3D voxel coordinates (integer grid indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Compute voxel coordinates from a 3D point.
    pub fn from_point(point: &[f32; 3], resolution: f32) -> Self {
        Self {
            x: (point[0] / resolution).floor() as i32,
            y: (point[1] / resolution).floor() as i32,
            z: (point[2] / resolution).floor() as i32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_voxel_coord_from_point() {
        let resolution = 2.0;

        // Point at origin
        let coord = VoxelCoord::from_point(&[0.0, 0.0, 0.0], resolution);
        assert_eq!(coord, VoxelCoord::new(0, 0, 0));

        // Point in positive quadrant
        let coord = VoxelCoord::from_point(&[3.5, 5.1, 1.9], resolution);
        assert_eq!(coord, VoxelCoord::new(1, 2, 0));

        // Point in negative quadrant
        let coord = VoxelCoord::from_point(&[-3.5, -1.1, -0.1], resolution);
        assert_eq!(coord, VoxelCoord::new(-2, -1, -1));
    }

    #[test]
    fn test_voxel_from_statistics() {
        let config = VoxelGridConfig::default();

        // Create 10 points clustered around (1, 2, 3)
        let points = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9],
            [1.05, 2.05, 3.05],
            [0.95, 1.95, 2.95],
            [1.02, 2.02, 3.02],
            [0.98, 1.98, 2.98],
            [1.03, 2.03, 3.03],
            [0.97, 1.97, 2.97],
            [1.0, 2.0, 3.0],
        ];

        let mut sum = Vector3::zeros();
        let mut sum_sq = Matrix3::zeros();

        for p in &points {
            let v = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            sum += v;
            sum_sq += v * v.transpose();
        }

        let voxel = Voxel::from_statistics(&sum, &sum_sq, points.len(), &config);
        assert!(voxel.is_some());

        let voxel = voxel.unwrap();

        // Mean should be close to (1, 2, 3)
        assert_relative_eq!(voxel.mean.x, 1.0, epsilon = 0.1);
        assert_relative_eq!(voxel.mean.y, 2.0, epsilon = 0.1);
        assert_relative_eq!(voxel.mean.z, 3.0, epsilon = 0.1);

        // Covariance should be small (tight cluster)
        assert!(voxel.covariance.norm() < 0.1);

        // Inverse covariance should exist and be symmetric
        let identity = voxel.covariance * voxel.inv_covariance;
        assert_relative_eq!(identity[(0, 0)], 1.0, epsilon = 0.01);
        assert_relative_eq!(identity[(1, 1)], 1.0, epsilon = 0.01);
        assert_relative_eq!(identity[(2, 2)], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_voxel_too_few_points() {
        let config = VoxelGridConfig {
            min_points_per_voxel: 6,
            ..Default::default()
        };

        let sum = Vector3::new(1.0, 2.0, 3.0);
        let sum_sq = Matrix3::identity();

        // Only 3 points - should fail
        let voxel = Voxel::from_statistics(&sum, &sum_sq, 3, &config);
        assert!(voxel.is_none());
    }

    #[test]
    fn test_regularize_covariance() {
        // Create a nearly-singular covariance (planar points)
        let cov = Matrix3::new(
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0001, // Very small z variance
        );

        let result = regularize_covariance(&cov, 0.01);
        assert!(result.is_some());

        let (reg_cov, inv_cov) = result.unwrap();

        // Small eigenvalue should be inflated to at least ratio_threshold * max_eigenvalue
        let eigen = reg_cov.symmetric_eigen();
        let min_ev = eigen.eigenvalues.iter().copied().fold(f64::MAX, f64::min);
        let max_ev = eigen.eigenvalues.iter().copied().fold(0.0, f64::max);

        // With ratio 0.01, min should be at least 0.01 * max
        assert!(
            min_ev >= max_ev * 0.009, // Allow small tolerance
            "min_ev={min_ev}, max_ev={max_ev}, expected min >= {}",
            max_ev * 0.01
        );

        // Inverse should work
        let identity = reg_cov * inv_cov;
        assert_relative_eq!(identity, Matrix3::identity(), epsilon = 1e-8);
    }
}
