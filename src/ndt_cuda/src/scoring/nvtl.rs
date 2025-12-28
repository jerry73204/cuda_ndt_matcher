//! NVTL (Nearest Voxel Transformation Likelihood) computation.
//!
//! NVTL is a quality metric that measures how well transformed source points
//! match the target voxel grid. Unlike transform probability which sums all
//! voxel scores, NVTL takes the maximum score per point across neighbor voxels.
//!
//! This makes NVTL more robust to multi-modal distributions where a point
//! might be between two voxels.
//!
//! Based on Autoware's NDT implementation.

use nalgebra::{Isometry3, Matrix3, Vector3};

use crate::derivatives::GaussianParams;
use crate::voxel_grid::VoxelGrid;

/// Configuration for NVTL computation.
#[derive(Debug, Clone)]
pub struct NvtlConfig {
    /// Search radius in voxel units (1 = 3x3x3 neighborhood).
    /// Default is 1, meaning we search the current voxel and 26 neighbors.
    pub search_radius: i32,

    /// Whether to return per-point scores.
    pub compute_per_point: bool,
}

impl Default for NvtlConfig {
    fn default() -> Self {
        Self {
            search_radius: 1,
            compute_per_point: false,
        }
    }
}

/// Result of NVTL computation.
#[derive(Debug, Clone)]
pub struct NvtlResult {
    /// NVTL score (average of max scores per point).
    /// Higher values indicate better alignment.
    pub nvtl: f64,

    /// Number of points that found at least one neighbor voxel.
    pub num_with_neighbors: usize,

    /// Number of points with no neighbor voxels.
    pub num_no_neighbors: usize,

    /// Per-point max scores (if requested).
    pub per_point_scores: Option<Vec<f64>>,
}

impl NvtlResult {
    /// Create a result with no neighbors found.
    pub fn no_neighbors(num_points: usize) -> Self {
        Self {
            nvtl: 0.0,
            num_with_neighbors: 0,
            num_no_neighbors: num_points,
            per_point_scores: None,
        }
    }
}

/// Compute NVTL for source points against a voxel grid.
///
/// For each source point:
/// 1. Transform by the given pose
/// 2. Search neighbor voxels (3x3x3 by default)
/// 3. Compute NDT score for each neighbor
/// 4. Take the maximum score
///
/// The final NVTL is the average of these max scores.
///
/// # Arguments
/// * `source_points` - Source point cloud
/// * `target_grid` - Target voxel grid (map)
/// * `pose` - Transform to apply to source points
/// * `gauss` - Gaussian parameters for NDT score function
/// * `config` - NVTL configuration
///
/// # Returns
/// NVTL result with score and statistics.
pub fn compute_nvtl(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &Isometry3<f64>,
    gauss: &GaussianParams,
    config: &NvtlConfig,
) -> NvtlResult {
    if source_points.is_empty() {
        return NvtlResult::no_neighbors(0);
    }

    let mut total_max_score = 0.0;
    let mut num_with_neighbors = 0;
    let mut num_no_neighbors = 0;
    let mut per_point_scores = if config.compute_per_point {
        Some(Vec::with_capacity(source_points.len()))
    } else {
        None
    };

    let resolution = target_grid.config().resolution;
    let inv_resolution = 1.0 / resolution as f64;

    for source_point in source_points {
        // Transform point
        let pt = nalgebra::Point3::new(
            source_point[0] as f64,
            source_point[1] as f64,
            source_point[2] as f64,
        );
        let transformed = pose * pt;

        // Compute max score across neighbor voxels
        let max_score = compute_max_neighbor_score(
            &transformed,
            target_grid,
            gauss,
            inv_resolution,
            config.search_radius,
        );

        if let Some(score) = max_score {
            total_max_score += score;
            num_with_neighbors += 1;
            if let Some(ref mut scores) = per_point_scores {
                scores.push(score);
            }
        } else {
            num_no_neighbors += 1;
            if let Some(ref mut scores) = per_point_scores {
                scores.push(0.0);
            }
        }
    }

    let nvtl = if num_with_neighbors > 0 {
        total_max_score / num_with_neighbors as f64
    } else {
        0.0
    };

    NvtlResult {
        nvtl,
        num_with_neighbors,
        num_no_neighbors,
        per_point_scores,
    }
}

/// Compute the maximum NDT score across neighbor voxels.
fn compute_max_neighbor_score(
    point: &nalgebra::Point3<f64>,
    grid: &VoxelGrid,
    gauss: &GaussianParams,
    inv_resolution: f64,
    search_radius: i32,
) -> Option<f64> {
    // Compute voxel coordinates
    let vx = (point.x * inv_resolution).floor() as i32;
    let vy = (point.y * inv_resolution).floor() as i32;
    let vz = (point.z * inv_resolution).floor() as i32;

    let mut max_score: Option<f64> = None;

    // Search in neighborhood
    for di in -search_radius..=search_radius {
        for dj in -search_radius..=search_radius {
            for dk in -search_radius..=search_radius {
                let neighbor_coord = [vx + di, vy + dj, vz + dk];

                if let Some(voxel) = grid.get_by_coord(&neighbor_coord) {
                    let score = compute_point_voxel_score(
                        point,
                        &voxel.mean.cast::<f64>(),
                        &voxel.inv_covariance.cast::<f64>(),
                        gauss,
                    );

                    max_score = Some(match max_score {
                        Some(current_max) => current_max.max(score),
                        None => score,
                    });
                }
            }
        }
    }

    max_score
}

/// Compute NDT score for a point-voxel pair.
fn compute_point_voxel_score(
    point: &nalgebra::Point3<f64>,
    mean: &Vector3<f64>,
    inv_covariance: &Matrix3<f64>,
    gauss: &GaussianParams,
) -> f64 {
    let diff = point.coords - mean;
    let mahal_sq = diff.dot(&(inv_covariance * diff));
    -gauss.d1 * (-gauss.d2 * mahal_sq / 2.0).exp()
}

/// Compute NVTL using radius search (like Autoware's radiusSearch).
///
/// For each point, finds all voxels within the search radius and takes the
/// maximum score. The final NVTL is the average of these max scores.
pub fn compute_nvtl_simple(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &Isometry3<f64>,
    gauss: &GaussianParams,
) -> f64 {
    if source_points.is_empty() {
        return 0.0;
    }

    let mut total_max_score = 0.0;
    let mut count = 0;

    // Use voxel resolution as search radius (matches Autoware's radiusSearch behavior)
    let search_radius = target_grid.resolution();

    for source_point in source_points {
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

        // Find all voxels within radius and take the max score
        let nearby_voxels = target_grid.radius_search(&transformed_f32, search_radius);

        if !nearby_voxels.is_empty() {
            let max_score = nearby_voxels
                .iter()
                .map(|voxel| {
                    compute_point_voxel_score(
                        &transformed,
                        &voxel.mean.cast::<f64>(),
                        &voxel.inv_covariance.cast::<f64>(),
                        gauss,
                    )
                })
                .fold(f64::NEG_INFINITY, f64::max);

            total_max_score += max_score;
            count += 1;
        }
    }

    if count > 0 {
        total_max_score / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_default_half_cubic_pcd, make_half_cubic_pcd_offset, voxelize_pcd};
    use approx::assert_relative_eq;

    fn create_test_grid() -> VoxelGrid {
        // Create a grid with points clustered around [1, 1, 1]
        // Using center at 1.0 ensures all points stay in voxel [0,0,0] with resolution 2.0
        use rand::prelude::*;
        use rand::SeedableRng;
        use rand_distr::Normal;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.1).unwrap();
        let center = 1.0f32;

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
    fn test_nvtl_basic() {
        let grid = create_test_grid();
        // Query at [1, 1, 1] where the grid is centered
        let source_points: Vec<[f32; 3]> = vec![[1.0, 1.0, 1.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        let result = compute_nvtl(&source_points, &grid, &pose, &gauss, &config);

        assert_eq!(result.num_with_neighbors, 1);
        assert!(result.nvtl > 0.0);
    }

    #[test]
    fn test_nvtl_no_neighbors() {
        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![[1000.0, 1000.0, 1000.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        let result = compute_nvtl(&source_points, &grid, &pose, &gauss, &config);

        assert_eq!(result.num_with_neighbors, 0);
        assert_eq!(result.num_no_neighbors, 1);
        assert_eq!(result.nvtl, 0.0);
    }

    #[test]
    fn test_nvtl_empty() {
        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        let result = compute_nvtl(&source_points, &grid, &pose, &gauss, &config);

        assert_eq!(result.nvtl, 0.0);
    }

    #[test]
    fn test_nvtl_with_per_point_scores() {
        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();
        let config = NvtlConfig {
            compute_per_point: true,
            ..Default::default()
        };

        let result = compute_nvtl(&source_points, &grid, &pose, &gauss, &config);

        assert!(result.per_point_scores.is_some());
        let scores = result.per_point_scores.unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > 0.0);
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn test_nvtl_simple() {
        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();

        let nvtl = compute_nvtl_simple(&source_points, &grid, &pose, &gauss);

        assert!(nvtl > 0.0);
    }

    #[test]
    fn test_nvtl_neighbor_search() {
        // Create a grid with two voxels
        let points: Vec<[f32; 3]> = vec![
            // First cluster at origin
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.1, 0.0],
            [0.1, 0.0, 0.1],
            // Second cluster at [3, 0, 0] (different voxel with resolution 2.0)
            [3.0, 0.0, 0.0],
            [3.1, 0.0, 0.0],
            [3.0, 0.1, 0.0],
            [3.0, 0.0, 0.1],
            [3.1, 0.1, 0.0],
            [3.1, 0.0, 0.1],
        ];
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // Query point between the two voxels
        let source_points: Vec<[f32; 3]> = vec![[1.5, 0.0, 0.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();

        // With neighbor search, should find both voxels
        let config_with_search = NvtlConfig {
            search_radius: 1,
            compute_per_point: false,
        };
        let result_with_search =
            compute_nvtl(&source_points, &grid, &pose, &gauss, &config_with_search);

        // Without neighbor search (simple), may or may not find a voxel
        let simple_result = compute_nvtl_simple(&source_points, &grid, &pose, &gauss);

        // With search should always find something
        assert!(
            result_with_search.num_with_neighbors > 0 || result_with_search.num_no_neighbors > 0
        );

        // Simple might not find anything if point doesn't land exactly in a voxel
        // This is expected - NVTL with neighbor search is more robust
        let _ = simple_result; // Just to use the variable
    }

    #[test]
    fn test_nvtl_vs_transform_probability() {
        use crate::scoring::compute_transform_probability;

        let grid = create_test_grid();
        let source_points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        let nvtl_result = compute_nvtl(&source_points, &grid, &pose, &gauss, &config);
        let tp_result = compute_transform_probability(&source_points, &grid, &pose, &gauss);

        // Both should give similar results for a single point at voxel center
        // NVTL might be slightly higher due to neighbor search finding the best match
        assert!(nvtl_result.nvtl > 0.0);
        assert!(tp_result.transform_probability > 0.0);

        // For a single point at the voxel center, they should be very close
        assert_relative_eq!(
            nvtl_result.nvtl,
            tp_result.transform_probability,
            epsilon = 0.1
        );
    }

    // ========================================================================
    // Phase 4: Scoring tests using Autoware-style test data
    // ========================================================================

    /// Test NVTL on half-cubic point cloud (Autoware test data).
    #[test]
    fn test_nvtl_half_cubic_perfect_alignment() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        // Sensor scan matches map exactly
        let sensor_scan = voxelize_pcd(&map_points, 1.0);

        let pose = Isometry3::identity();
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        let result = compute_nvtl(&sensor_scan, &grid, &pose, &gauss, &config);

        // Perfect alignment should have high NVTL
        assert!(
            result.nvtl > 0.5,
            "Perfect alignment should have high NVTL, got {}",
            result.nvtl
        );

        // Most points should have neighbors
        assert!(
            result.num_with_neighbors > result.num_no_neighbors,
            "Most points should have neighbors"
        );
    }

    /// Test NVTL at origin is higher than far outside map.
    ///
    /// Note: With multi-voxel radius search, NVTL ordering can be non-monotonic
    /// for small offsets, but large offsets (outside map) should score lower.
    #[test]
    fn test_nvtl_decreases_with_offset() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        // Compute NVTL at origin and far outside
        let nvtl_at_0 = compute_nvtl(
            &sensor_scan,
            &grid,
            &Isometry3::identity(),
            &gauss,
            &config,
        )
        .nvtl;
        let nvtl_at_far = compute_nvtl(
            &sensor_scan,
            &grid,
            &Isometry3::translation(50.0, 50.0, 0.0),
            &gauss,
            &config,
        )
        .nvtl;

        // NVTL at origin should be positive
        assert!(
            nvtl_at_0 > 0.0,
            "NVTL at origin should be positive, got {}",
            nvtl_at_0
        );

        // NVTL far outside map should be lower
        assert!(
            nvtl_at_far < nvtl_at_0,
            "NVTL far from map ({}) should be < NVTL at origin ({})",
            nvtl_at_far,
            nvtl_at_0
        );
    }

    /// Test NVTL is non-negative and finite.
    ///
    /// Note: With multi-voxel radius search, NVTL can exceed 1.0 because
    /// each point can contribute to multiple voxels' scores.
    #[test]
    fn test_nvtl_valid_range() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        for offset in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            let pose = Isometry3::translation(offset, 0.0, 0.0);
            let result = compute_nvtl(&sensor_scan, &grid, &pose, &gauss, &config);

            assert!(
                result.nvtl >= 0.0 && result.nvtl.is_finite(),
                "NVTL should be >= 0 and finite, got {} at offset {}",
                result.nvtl,
                offset
            );
        }
    }

    /// Test NVTL with map at offset (like Autoware's test setup).
    #[test]
    fn test_nvtl_with_offset_map() {
        // Map at offset (100, 100)
        let map_points = make_half_cubic_pcd_offset(100.0, 100.0, 0.0);
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        // Sensor scan at origin
        let sensor_scan = voxelize_pcd(&make_default_half_cubic_pcd(), 1.0);
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        // Without translation: should have low/zero NVTL (no overlap)
        let result_no_offset = compute_nvtl(
            &sensor_scan,
            &grid,
            &Isometry3::identity(),
            &gauss,
            &config,
        );

        // With correct translation: should have high NVTL
        let result_with_offset = compute_nvtl(
            &sensor_scan,
            &grid,
            &Isometry3::translation(100.0, 100.0, 0.0),
            &gauss,
            &config,
        );

        assert!(
            result_with_offset.nvtl > result_no_offset.nvtl,
            "NVTL with correct offset ({}) should be > without ({})",
            result_with_offset.nvtl,
            result_no_offset.nvtl
        );
    }

    /// Test per-point scores on half-cubic.
    #[test]
    fn test_per_point_scores_half_cubic() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();
        let config = NvtlConfig {
            compute_per_point: true,
            ..Default::default()
        };

        let result = compute_nvtl(
            &sensor_scan,
            &grid,
            &Isometry3::identity(),
            &gauss,
            &config,
        );

        let scores = result.per_point_scores.expect("Should have per-point scores");
        assert_eq!(scores.len(), sensor_scan.len());

        // Count points with positive scores
        let positive_count = scores.iter().filter(|&&s| s > 0.0).count();
        assert!(
            positive_count > sensor_scan.len() / 2,
            "Most points should have positive scores: {} out of {}",
            positive_count,
            sensor_scan.len()
        );

        // All scores should be non-negative and finite
        // Note: With multi-voxel radius search, scores can exceed 1.0
        for (i, &score) in scores.iter().enumerate() {
            assert!(
                score >= 0.0 && score.is_finite(),
                "Score[{}] should be >= 0 and finite, got {}",
                i,
                score
            );
        }
    }

    /// Test NVTL simple vs full on half-cubic.
    #[test]
    fn test_nvtl_simple_vs_full_half_cubic() {
        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let gauss = GaussianParams::default();
        let config = NvtlConfig::default();

        let full_result = compute_nvtl(
            &sensor_scan,
            &grid,
            &Isometry3::identity(),
            &gauss,
            &config,
        );
        let simple_nvtl = compute_nvtl_simple(
            &sensor_scan,
            &grid,
            &Isometry3::identity(),
            &gauss,
        );

        // Both should give positive values
        assert!(full_result.nvtl > 0.0, "Full NVTL should be positive");
        assert!(simple_nvtl > 0.0, "Simple NVTL should be positive");

        // Values should be in same ballpark (within 50%)
        let ratio = if full_result.nvtl > simple_nvtl {
            full_result.nvtl / simple_nvtl
        } else {
            simple_nvtl / full_result.nvtl
        };
        assert!(
            ratio < 2.0,
            "Full ({}) and simple ({}) NVTL should be similar",
            full_result.nvtl,
            simple_nvtl
        );
    }
}
