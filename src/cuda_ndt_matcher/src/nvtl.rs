//! NVTL (Nearest Voxel Transformation Likelihood) score computation.
//!
//! This module implements Autoware's NVTL metric for evaluating how well
//! a transformed point cloud matches a target map.
//!
//! The NVTL score is computed as:
//! 1. Build voxel grid from target points with per-voxel mean and covariance
//! 2. Transform source points using the candidate pose
//! 3. For each transformed point, find neighbor voxels
//! 4. Compute Gaussian score using Mahalanobis distance to each voxel
//! 5. Take max score per point, average across all points

use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;

/// Configuration for NVTL computation
pub struct NvtlConfig {
    /// Voxel resolution (same as NDT resolution)
    pub resolution: f64,
    /// Outlier ratio for Gaussian parameters (Autoware default: 0.55)
    pub outlier_ratio: f64,
}

impl Default for NvtlConfig {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            outlier_ratio: 0.55,
        }
    }
}

/// Gaussian fitting parameters from Autoware's NDT implementation
/// Based on [Magnusson 2009] equations
struct GaussParams {
    d1: f64,
    d2: f64,
}

impl GaussParams {
    fn new(resolution: f64, outlier_ratio: f64) -> Self {
        // Autoware's Gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
        let gauss_c1 = 10.0 * (1.0 - outlier_ratio);
        let gauss_c2 = outlier_ratio / resolution.powi(3);
        let gauss_d3 = -gauss_c2.ln();
        let d1 = -(gauss_c1 + gauss_c2).ln() - gauss_d3;
        let d2 = -2.0 * ((-(gauss_c1 * (-0.5_f64).exp() + gauss_c2).ln() - gauss_d3) / d1).ln();

        Self { d1, d2 }
    }

    /// Compute score using Mahalanobis distance
    /// This is Autoware's score_inc = -gauss_d1_ * exp(-gauss_d2_ * mahal_dist_sq / 2)
    fn score(&self, mahal_dist_sq: f64) -> f64 {
        -self.d1 * (-self.d2 * mahal_dist_sq / 2.0).exp()
    }
}

/// Voxel with Gaussian distribution (mean and covariance)
struct GaussianVoxel {
    /// Mean of points in this voxel
    mean: Vector3<f64>,
    /// Inverse covariance matrix (for Mahalanobis distance)
    cov_inv: Matrix3<f64>,
}

impl GaussianVoxel {
    /// Create a voxel from a set of points
    fn from_points(points: &[[f32; 3]]) -> Option<Self> {
        let n = points.len();
        if n < 6 {
            // Autoware requires at least 6 points for stable covariance
            return None;
        }

        // Compute mean
        let mut mean = Vector3::zeros();
        for p in points {
            mean += Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        }
        mean /= n as f64;

        // Compute covariance
        let mut cov = Matrix3::zeros();
        for p in points {
            let d = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64) - mean;
            cov += d * d.transpose();
        }
        cov /= n as f64;

        // Regularize covariance to ensure positive definiteness
        // Add small value to diagonal (similar to Autoware's MIN_EIG regularization)
        let min_eig = 0.01;
        cov[(0, 0)] += min_eig;
        cov[(1, 1)] += min_eig;
        cov[(2, 2)] += min_eig;

        // Compute inverse
        let cov_inv = cov.try_inverse()?;

        Some(Self { mean, cov_inv })
    }

    /// Compute Mahalanobis distance squared from a point to this voxel
    fn mahalanobis_sq(&self, point: &Vector3<f64>) -> f64 {
        let d = point - self.mean;
        d.dot(&(self.cov_inv * d))
    }
}

/// NDT voxel grid with Gaussian distributions
struct NdtVoxelGrid {
    /// Map from voxel key to Gaussian voxel
    voxels: HashMap<(i64, i64, i64), GaussianVoxel>,
    /// Inverse resolution for faster computation
    inv_resolution: f64,
}

impl NdtVoxelGrid {
    fn new(points: &[[f32; 3]], resolution: f64) -> Self {
        let inv_resolution = 1.0 / resolution;

        // First, group points by voxel
        let mut point_groups: HashMap<(i64, i64, i64), Vec<[f32; 3]>> = HashMap::new();
        for point in points {
            let key = Self::point_to_key(point, inv_resolution);
            point_groups.entry(key).or_default().push(*point);
        }

        // Then, compute Gaussian for each voxel
        let mut voxels = HashMap::new();
        for (key, voxel_points) in point_groups {
            if let Some(gaussian) = GaussianVoxel::from_points(&voxel_points) {
                voxels.insert(key, gaussian);
            }
        }

        Self {
            voxels,
            inv_resolution,
        }
    }

    fn point_to_key(point: &[f32; 3], inv_resolution: f64) -> (i64, i64, i64) {
        (
            (point[0] as f64 * inv_resolution).floor() as i64,
            (point[1] as f64 * inv_resolution).floor() as i64,
            (point[2] as f64 * inv_resolution).floor() as i64,
        )
    }

    /// Find neighbor voxels and compute max score for a query point
    fn compute_max_score(&self, point: &[f32; 3], gauss: &GaussParams) -> Option<f64> {
        let key = Self::point_to_key(point, self.inv_resolution);
        let query_pt = Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);

        let mut max_score = f64::MIN;
        let mut found = false;

        // Search in 3x3x3 neighborhood (like Autoware's radiusSearch with resolution)
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    let neighbor_key = (key.0 + di, key.1 + dj, key.2 + dk);
                    if let Some(voxel) = self.voxels.get(&neighbor_key) {
                        let mahal_sq = voxel.mahalanobis_sq(&query_pt);
                        let score = gauss.score(mahal_sq);
                        if score > max_score {
                            max_score = score;
                            found = true;
                        }
                    }
                }
            }
        }

        if found {
            Some(max_score)
        } else {
            None
        }
    }
}

/// Pre-computed NDT voxel grid for efficient NVTL computation
///
/// Build this once for the target (map) point cloud, then reuse for
/// scoring multiple candidate poses.
pub struct NvtlVoxelGrid {
    grid: NdtVoxelGrid,
    gauss: GaussParams,
}

impl NvtlVoxelGrid {
    /// Create a new NVTL voxel grid from target points
    pub fn new(target_points: &[[f32; 3]], config: &NvtlConfig) -> Self {
        Self {
            grid: NdtVoxelGrid::new(target_points, config.resolution),
            gauss: GaussParams::new(config.resolution, config.outlier_ratio),
        }
    }

    /// Compute NVTL score for transformed source points
    pub fn compute_score(&self, transformed_source: &[[f32; 3]]) -> f64 {
        if transformed_source.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut count = 0;

        for source_pt in transformed_source {
            if let Some(score) = self.grid.compute_max_score(source_pt, &self.gauss) {
                total_score += score;
                count += 1;
            }
        }

        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }

    /// Compute NVTL score for source points transformed by a pose
    pub fn compute_score_with_pose(
        &self,
        source_points: &[[f32; 3]],
        pose: &geometry_msgs::msg::Pose,
    ) -> f64 {
        let transformed = transform_points(source_points, pose);
        self.compute_score(&transformed)
    }
}

/// Compute NVTL score for a transformed point cloud
///
/// # Arguments
/// * `transformed_source` - Source points already transformed by the candidate pose
/// * `target_points` - Target (map) points
/// * `config` - NVTL configuration
///
/// # Returns
/// * NVTL score (higher = better alignment), typically in range [0, ~5]
#[allow(dead_code)]
pub fn compute_nvtl(
    transformed_source: &[[f32; 3]],
    target_points: &[[f32; 3]],
    config: &NvtlConfig,
) -> f64 {
    if transformed_source.is_empty() || target_points.is_empty() {
        return 0.0;
    }

    let gauss = GaussParams::new(config.resolution, config.outlier_ratio);
    let voxel_grid = NdtVoxelGrid::new(target_points, config.resolution);

    let mut total_score = 0.0;
    let mut count = 0;

    for source_pt in transformed_source {
        if let Some(score) = voxel_grid.compute_max_score(source_pt, &gauss) {
            total_score += score;
            count += 1;
        }
    }

    if count > 0 {
        total_score / count as f64
    } else {
        0.0
    }
}

/// Transform source points using a pose and compute NVTL
///
/// # Arguments
/// * `source_points` - Original source points
/// * `pose` - Transformation pose
/// * `target_points` - Target (map) points
/// * `config` - NVTL configuration
#[allow(dead_code)]
pub fn compute_nvtl_with_pose(
    source_points: &[[f32; 3]],
    pose: &geometry_msgs::msg::Pose,
    target_points: &[[f32; 3]],
    config: &NvtlConfig,
) -> f64 {
    // Transform source points
    let transformed = transform_points(source_points, pose);
    compute_nvtl(&transformed, target_points, config)
}

/// Transform points using a pose
fn transform_points(points: &[[f32; 3]], pose: &geometry_msgs::msg::Pose) -> Vec<[f32; 3]> {
    use nalgebra::{Isometry3, Quaternion as NaQuaternion, Translation3, UnitQuaternion};

    let p = &pose.position;
    let q = &pose.orientation;

    let translation = Translation3::new(p.x, p.y, p.z);
    let quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));
    let isometry = Isometry3::from_parts(translation, quaternion);

    points
        .iter()
        .map(|pt| {
            let transformed = isometry * nalgebra::Point3::new(pt[0] as f64, pt[1] as f64, pt[2] as f64);
            [transformed.x as f32, transformed.y as f32, transformed.z as f32]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry_msgs::msg::{Point, Quaternion};

    fn identity_pose() -> geometry_msgs::msg::Pose {
        geometry_msgs::msg::Pose {
            position: Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        }
    }

    #[test]
    fn test_nvtl_perfect_match() {
        let config = NvtlConfig::default();
        let points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        // Same points should give high score
        let score = compute_nvtl(&points, &points, &config);
        assert!(score > 0.0, "Perfect match should have positive score");
    }

    #[test]
    fn test_nvtl_with_offset() {
        let config = NvtlConfig::default();
        let source: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let target: Vec<[f32; 3]> = vec![[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]];

        // Small offset should still give reasonable score
        let score = compute_nvtl(&source, &target, &config);
        assert!(score > 0.0, "Small offset should have positive score");
    }

    #[test]
    fn test_nvtl_no_overlap() {
        let config = NvtlConfig::default();
        let source: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
        let target: Vec<[f32; 3]> = vec![[100.0, 100.0, 100.0]];

        // No overlap should give 0 score (no neighbors found)
        let score = compute_nvtl(&source, &target, &config);
        assert_eq!(score, 0.0, "No overlap should give zero score");
    }

    #[test]
    fn test_gauss_params() {
        // Test with Autoware defaults
        let gauss = GaussParams::new(2.0, 0.55);

        // Score at distance 0 should be high
        let score_0 = gauss.score(0.0);
        assert!(score_0 > 0.0, "Score at distance 0 should be positive");

        // Score should decrease with distance
        let score_1 = gauss.score(1.0);
        assert!(score_1 < score_0, "Score should decrease with distance");
    }
}
