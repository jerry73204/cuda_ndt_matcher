//! Transform probability and per-point scoring metrics.
//!
//! Transform probability measures how well the transformed source points
//! match the target voxel grid. It's computed as the sum of NDT scores
//! normalized by the number of correspondences.

use nalgebra::{Isometry3, Matrix3, Vector3};

use crate::derivatives::GaussianParams;
use crate::voxel_grid::VoxelGrid;

/// Result of scoring computation.
#[derive(Debug, Clone)]
pub struct ScoringResult {
    /// Total NDT score (sum over all point-voxel pairs).
    pub total_score: f64,

    /// Transform probability (total_score / num_correspondences).
    /// This is the average score per correspondence.
    pub transform_probability: f64,

    /// Number of source points that found at least one voxel correspondence.
    pub num_correspondences: usize,

    /// Number of source points with no voxel correspondence.
    pub num_no_correspondence: usize,

    /// Per-point scores (if requested).
    pub per_point_scores: Option<Vec<f64>>,
}

impl ScoringResult {
    /// Create a result with no correspondences.
    pub fn no_correspondences(num_points: usize) -> Self {
        Self {
            total_score: 0.0,
            transform_probability: 0.0,
            num_correspondences: 0,
            num_no_correspondence: num_points,
            per_point_scores: None,
        }
    }
}

/// Compute transform probability for a point cloud against a voxel grid.
///
/// This is the standard NDT score normalized by the number of correspondences.
/// Higher values indicate better alignment.
///
/// # Arguments
/// * `source_points` - Source point cloud (will be transformed)
/// * `target_grid` - Target voxel grid (map)
/// * `pose` - Transform to apply to source points
/// * `gauss` - Gaussian parameters for NDT score function
///
/// # Returns
/// Scoring result with transform probability and correspondence counts.
pub fn compute_transform_probability(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &Isometry3<f64>,
    gauss: &GaussianParams,
) -> ScoringResult {
    if source_points.is_empty() {
        return ScoringResult::no_correspondences(0);
    }

    let mut total_score = 0.0;
    let mut num_correspondences = 0;
    let mut num_no_correspondence = 0;

    // Use voxel resolution as search radius (matches Autoware's radiusSearch behavior)
    let search_radius = target_grid.resolution();

    for source_point in source_points {
        // Transform point
        let pt = nalgebra::Point3::new(
            source_point[0] as f64,
            source_point[1] as f64,
            source_point[2] as f64,
        );
        let transformed = pose * pt;
        let transformed_f32 = [
            transformed.x as f32,
            transformed.y as f32,
            transformed.z as f32,
        ];

        // Find corresponding voxels using radius search (like Autoware)
        let nearby_voxels = target_grid.radius_search(&transformed_f32, search_radius);

        if nearby_voxels.is_empty() {
            num_no_correspondence += 1;
        } else {
            // Accumulate scores from ALL nearby voxels (key difference from single-voxel lookup)
            for voxel in nearby_voxels {
                let score = compute_point_score(
                    &transformed,
                    &voxel.mean.cast::<f64>(),
                    &voxel.inv_covariance.cast::<f64>(),
                    gauss,
                );
                total_score += score;
                num_correspondences += 1;
            }
        }
    }

    let transform_probability = if num_correspondences > 0 {
        total_score / num_correspondences as f64
    } else {
        0.0
    };

    ScoringResult {
        total_score,
        transform_probability,
        num_correspondences,
        num_no_correspondence,
        per_point_scores: None,
    }
}

/// Compute per-point scores for visualization and debugging.
///
/// Each point's score is the sum of NDT scores from all nearby voxels
/// (using radius search like Autoware's radiusSearch).
///
/// # Arguments
/// * `source_points` - Source point cloud (will be transformed)
/// * `target_grid` - Target voxel grid (map)
/// * `pose` - Transform to apply to source points
/// * `gauss` - Gaussian parameters for NDT score function
///
/// # Returns
/// Vector of scores for each source point (0.0 if no correspondence).
pub fn compute_per_point_scores(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &Isometry3<f64>,
    gauss: &GaussianParams,
) -> Vec<f64> {
    // Use voxel resolution as search radius (matches Autoware's radiusSearch behavior)
    let search_radius = target_grid.resolution();

    source_points
        .iter()
        .map(|source_point| {
            // Transform point
            let pt = nalgebra::Point3::new(
                source_point[0] as f64,
                source_point[1] as f64,
                source_point[2] as f64,
            );
            let transformed = pose * pt;
            let transformed_f32 = [
                transformed.x as f32,
                transformed.y as f32,
                transformed.z as f32,
            ];

            // Find corresponding voxels and sum their scores
            let nearby_voxels = target_grid.radius_search(&transformed_f32, search_radius);

            nearby_voxels
                .iter()
                .map(|voxel| {
                    compute_point_score(
                        &transformed,
                        &voxel.mean.cast::<f64>(),
                        &voxel.inv_covariance.cast::<f64>(),
                        gauss,
                    )
                })
                .sum()
        })
        .collect()
}

/// Compute the NDT score for a single point-voxel pair.
///
/// Score = -d1 * exp(-d2/2 * (x-μ)ᵀΣ⁻¹(x-μ))
///
/// # Arguments
/// * `point` - Transformed source point
/// * `mean` - Voxel mean
/// * `inv_covariance` - Voxel inverse covariance matrix
/// * `gauss` - Gaussian parameters
fn compute_point_score(
    point: &nalgebra::Point3<f64>,
    mean: &Vector3<f64>,
    inv_covariance: &Matrix3<f64>,
    gauss: &GaussianParams,
) -> f64 {
    // x - μ
    let diff = point.coords - mean;

    // (x-μ)ᵀ Σ⁻¹ (x-μ) = Mahalanobis distance squared
    let mahal_sq = diff.dot(&(inv_covariance * diff));

    // Score = -d1 * exp(-d2/2 * mahal_sq)
    // Note: d1 is negative, so -d1 is positive
    -gauss.d1 * (-gauss.d2 * mahal_sq / 2.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        make_default_half_cubic_pcd, make_half_cubic_pcd_offset, voxelize_pcd,
    };
    use approx::assert_relative_eq;

    fn create_test_grid() -> VoxelGrid {
        // Create a grid with points clustered around [1, 1, 1]
        // Using center at 1.0 ensures all points stay in voxel [0,0,0] with resolution 2.0
        // (voxel [0,0,0] covers [0, 2) in each dimension)
        use rand::prelude::*;
        use rand::SeedableRng;
        use rand_distr::Normal;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.1).unwrap();
        let center = 1.0f32; // Center at 1.0 to stay in voxel [0,0,0]

        let mut points = Vec::new();
        for _ in 0..50 {
            points.push([
                center + dist.sample(&mut rng) as f32,
                center + dist.sample(&mut rng) as f32,
                center + dist.sample(&mut rng) as f32,
            ]);
        }
        VoxelGrid::from_points(&points, 2.0).unwrap()
    }

    #[test]
    fn test_transform_probability_basic() {
        let grid = create_test_grid();
        // Query at [1, 1, 1] where the grid is centered
        let source_points: Vec<[f32; 3]> = vec![[1.0, 1.0, 1.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();

        let result = compute_transform_probability(&source_points, &grid, &pose, &gauss);

        // Should find correspondence at grid center
        assert_eq!(result.num_correspondences, 1);
        assert!(result.total_score > 0.0);
        assert!(result.transform_probability > 0.0);
    }

    #[test]
    fn test_transform_probability_no_correspondence() {
        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![[1000.0, 1000.0, 1000.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();

        let result = compute_transform_probability(&source_points, &grid, &pose, &gauss);

        assert_eq!(result.num_correspondences, 0);
        assert_eq!(result.num_no_correspondence, 1);
        assert_eq!(result.transform_probability, 0.0);
    }

    #[test]
    fn test_transform_probability_empty() {
        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();

        let result = compute_transform_probability(&source_points, &grid, &pose, &gauss);

        assert_eq!(result.num_correspondences, 0);
    }

    #[test]
    fn test_per_point_scores() {
        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![
            [1.0, 1.0, 1.0],    // Should have score (at grid center)
            [1000.0, 0.0, 0.0], // No correspondence
        ];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();

        let scores = compute_per_point_scores(&source_points, &grid, &pose, &gauss);

        assert_eq!(scores.len(), 2);
        assert!(scores[0] > 0.0, "First point should have positive score");
        assert_eq!(scores[1], 0.0, "Second point should have zero score");
    }

    #[test]
    fn test_point_score_at_mean() {
        // Score at voxel mean should be maximum
        let point = nalgebra::Point3::new(1.0, 2.0, 3.0);
        let mean = Vector3::new(1.0, 2.0, 3.0);
        let inv_cov = Matrix3::identity();
        let gauss = GaussianParams::default();

        let score = compute_point_score(&point, &mean, &inv_cov, &gauss);

        // At mean, mahal_sq = 0, so score = -d1 * exp(0) = -d1
        // d1 is negative, so -d1 is positive
        assert_relative_eq!(score, -gauss.d1, epsilon = 1e-10);
        assert!(score > 0.0);
    }

    #[test]
    fn test_point_score_decreases_with_distance() {
        let mean = Vector3::new(0.0, 0.0, 0.0);
        let inv_cov = Matrix3::identity();
        let gauss = GaussianParams::default();

        let score_0 = compute_point_score(
            &nalgebra::Point3::new(0.0, 0.0, 0.0),
            &mean,
            &inv_cov,
            &gauss,
        );
        let score_1 = compute_point_score(
            &nalgebra::Point3::new(1.0, 0.0, 0.0),
            &mean,
            &inv_cov,
            &gauss,
        );
        let score_2 = compute_point_score(
            &nalgebra::Point3::new(2.0, 0.0, 0.0),
            &mean,
            &inv_cov,
            &gauss,
        );

        assert!(score_0 > score_1, "Score should decrease with distance");
        assert!(score_1 > score_2, "Score should decrease with distance");
    }

    #[test]
    fn test_transform_probability_with_pose() {
        let grid = create_test_grid();
        // Source point at [2, 1, 1], translated by [-1, 0, 0] should land at [1, 1, 1] (grid center)
        let source_points: Vec<[f32; 3]> = vec![[2.0, 1.0, 1.0]];
        let pose = Isometry3::translation(-1.0, 0.0, 0.0);
        let gauss = GaussianParams::default();

        let result = compute_transform_probability(&source_points, &grid, &pose, &gauss);

        // After translation, point should be at [1, 1, 1] where grid is
        assert_eq!(result.num_correspondences, 1);
        assert!(result.transform_probability > 0.0);
    }

    // ========================================================================
    // Phase 4: Transform probability tests using Autoware-style test data
    // ========================================================================

    /// Test transform probability on half-cubic point cloud.
    #[test]
    fn test_transform_probability_half_cubic() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();

        let result =
            compute_transform_probability(&sensor_scan, &grid, &Isometry3::identity(), &gauss);

        // Should have many correspondences
        assert!(
            result.num_correspondences > sensor_scan.len() / 2,
            "Should have many correspondences: {} out of {}",
            result.num_correspondences,
            sensor_scan.len()
        );

        // Transform probability should be positive
        assert!(
            result.transform_probability > 0.0,
            "Transform probability should be positive, got {}",
            result.transform_probability
        );
    }

    /// Test transform probability is higher at origin than far outside map.
    ///
    /// Note: With multi-voxel radius search, score ordering can be non-monotonic
    /// for small offsets, but large offsets (outside map) should score lower.
    #[test]
    fn test_transform_probability_decreases_with_offset() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();

        let tp_at_0 =
            compute_transform_probability(&sensor_scan, &grid, &Isometry3::identity(), &gauss)
                .transform_probability;

        let tp_at_far = compute_transform_probability(
            &sensor_scan,
            &grid,
            &Isometry3::translation(50.0, 50.0, 0.0),
            &gauss,
        )
        .transform_probability;

        // TP at origin should be positive
        assert!(
            tp_at_0 > 0.0,
            "TP at origin should be positive, got {}",
            tp_at_0
        );

        // TP far outside map should be lower
        assert!(
            tp_at_far < tp_at_0,
            "TP far from map ({}) should be < TP at origin ({})",
            tp_at_far,
            tp_at_0
        );
    }

    /// Test transform probability is non-negative and finite.
    ///
    /// Note: With multi-voxel radius search, TP can exceed 1.0 because
    /// each point can contribute to multiple voxels' scores.
    #[test]
    fn test_transform_probability_valid_range() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();

        for offset in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let pose = Isometry3::translation(offset, 0.0, 0.0);
            let result = compute_transform_probability(&sensor_scan, &grid, &pose, &gauss);

            assert!(
                result.transform_probability >= 0.0 && result.transform_probability.is_finite(),
                "TP should be >= 0 and finite, got {} at offset {}",
                result.transform_probability,
                offset
            );
        }
    }

    /// Test transform probability with offset map.
    #[test]
    fn test_transform_probability_with_offset_map() {
        // Map at (100, 100)
        let map_points = make_half_cubic_pcd_offset(100.0, 100.0, 0.0);
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        // Sensor at origin
        let sensor_scan = voxelize_pcd(&make_default_half_cubic_pcd(), 1.0);
        let gauss = GaussianParams::default();

        // Without offset: no overlap
        let result_no_offset =
            compute_transform_probability(&sensor_scan, &grid, &Isometry3::identity(), &gauss);

        // With correct offset: should overlap
        let result_with_offset = compute_transform_probability(
            &sensor_scan,
            &grid,
            &Isometry3::translation(100.0, 100.0, 0.0),
            &gauss,
        );

        assert!(
            result_with_offset.transform_probability > result_no_offset.transform_probability,
            "TP with offset ({}) should be > without ({})",
            result_with_offset.transform_probability,
            result_no_offset.transform_probability
        );
        assert!(
            result_with_offset.num_correspondences > result_no_offset.num_correspondences,
            "Correspondences with offset ({}) should be > without ({})",
            result_with_offset.num_correspondences,
            result_no_offset.num_correspondences
        );
    }

    /// Test per-point scores on half-cubic.
    #[test]
    fn test_per_point_scores_half_cubic() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();

        let scores = compute_per_point_scores(&sensor_scan, &grid, &Isometry3::identity(), &gauss);

        assert_eq!(scores.len(), sensor_scan.len());

        // Most points should have positive scores
        let positive_count = scores.iter().filter(|&&s| s > 0.0).count();
        assert!(
            positive_count > sensor_scan.len() / 2,
            "Most points should have positive scores: {} out of {}",
            positive_count,
            sensor_scan.len()
        );
    }

    /// Test total score consistency.
    #[test]
    fn test_total_score_consistency() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();

        let result =
            compute_transform_probability(&sensor_scan, &grid, &Isometry3::identity(), &gauss);
        let per_point =
            compute_per_point_scores(&sensor_scan, &grid, &Isometry3::identity(), &gauss);

        // Total score from per-point should match
        let per_point_total: f64 = per_point.iter().sum();
        assert_relative_eq!(result.total_score, per_point_total, epsilon = 1e-6);
    }
}
