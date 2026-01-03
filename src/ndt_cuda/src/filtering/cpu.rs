//! CPU fallback implementations for point cloud filtering.

use super::{FilterParams, FilterResult};
use std::collections::HashMap;

/// Filter points using CPU (fallback when CUDA not available or for small point clouds).
pub fn filter_points_cpu(points: &[[f32; 3]], params: &FilterParams) -> FilterResult {
    let original_count = points.len();

    let min_dist_sq = params.min_distance * params.min_distance;
    let max_dist_sq = params.max_distance * params.max_distance;

    // Apply distance and z filters
    let mut filtered: Vec<[f32; 3]> = Vec::with_capacity(points.len());
    let mut removed_by_distance = 0usize;
    let mut removed_by_z = 0usize;

    for p in points {
        let dist_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];

        // Distance filter
        if dist_sq < min_dist_sq || dist_sq > max_dist_sq {
            removed_by_distance += 1;
            continue;
        }

        // Z filter
        if p[2] < params.min_z || p[2] > params.max_z {
            removed_by_z += 1;
            continue;
        }

        filtered.push(*p);
    }

    // Apply voxel downsampling if enabled
    let removed_by_downsampling = if let Some(resolution) = params.downsample_resolution {
        let before = filtered.len();
        filtered = voxel_downsample_cpu(&filtered, resolution);
        before - filtered.len()
    } else {
        0
    };

    // Sanity check
    debug_assert_eq!(
        original_count,
        filtered.len() + removed_by_distance + removed_by_z + removed_by_downsampling
    );

    FilterResult {
        points: filtered,
        removed_by_distance,
        removed_by_z,
        removed_by_downsampling,
        used_gpu: false,
    }
}

/// Voxel accumulator: (sum_x, sum_y, sum_z, count)
type VoxelAccum = (f64, f64, f64, usize);

/// Downsample points using a voxel grid filter.
///
/// For each voxel, keeps the centroid of all points within that voxel.
/// Uses f64 for accumulation to avoid precision loss with large sums.
pub fn voxel_downsample_cpu(points: &[[f32; 3]], resolution: f32) -> Vec<[f32; 3]> {
    if points.is_empty() || resolution <= 0.0 {
        return points.to_vec();
    }

    let inv_resolution = 1.0 / resolution;

    // Accumulate points per voxel
    // Key: (voxel_x, voxel_y, voxel_z) as integers
    let mut voxel_accum: HashMap<(i32, i32, i32), VoxelAccum> = HashMap::new();

    for p in points {
        let vx = (p[0] * inv_resolution).floor() as i32;
        let vy = (p[1] * inv_resolution).floor() as i32;
        let vz = (p[2] * inv_resolution).floor() as i32;

        let entry = voxel_accum
            .entry((vx, vy, vz))
            .or_insert((0.0, 0.0, 0.0, 0));
        entry.0 += p[0] as f64;
        entry.1 += p[1] as f64;
        entry.2 += p[2] as f64;
        entry.3 += 1;
    }

    // Compute centroids
    voxel_accum
        .values()
        .map(|(sx, sy, sz, count)| {
            let n = *count as f64;
            [(sx / n) as f32, (sy / n) as f32, (sz / n) as f32]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_empty() {
        let params = FilterParams::default();
        let result = filter_points_cpu(&[], &params);
        assert!(result.points.is_empty());
        assert_eq!(result.removed_by_distance, 0);
        assert_eq!(result.removed_by_z, 0);
    }

    #[test]
    fn test_filter_distance_min() {
        let points = vec![
            [0.5, 0.0, 0.0], // distance = 0.5 (too close)
            [2.0, 0.0, 0.0], // distance = 2.0 (pass)
            [5.0, 0.0, 0.0], // distance = 5.0 (pass)
        ];

        let params = FilterParams {
            min_distance: 1.0,
            ..Default::default()
        };

        let result = filter_points_cpu(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_distance, 1);
    }

    #[test]
    fn test_filter_distance_max() {
        let points = vec![
            [5.0, 0.0, 0.0],   // distance = 5.0 (pass)
            [50.0, 0.0, 0.0],  // distance = 50.0 (pass)
            [150.0, 0.0, 0.0], // distance = 150.0 (too far)
        ];

        let params = FilterParams {
            max_distance: 100.0,
            ..Default::default()
        };

        let result = filter_points_cpu(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_distance, 1);
    }

    #[test]
    fn test_filter_z_height() {
        let points = vec![
            [1.0, 0.0, -5.0], // below min_z
            [1.0, 0.0, 0.0],  // within range
            [1.0, 0.0, 5.0],  // within range
            [1.0, 0.0, 15.0], // above max_z
        ];

        let params = FilterParams {
            min_z: -2.0,
            max_z: 10.0,
            ..Default::default()
        };

        let result = filter_points_cpu(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_z, 2);
    }

    #[test]
    fn test_filter_combined() {
        let points = vec![
            [0.5, 0.0, 0.0],   // too close
            [5.0, 0.0, -5.0],  // z too low
            [5.0, 0.0, 0.0],   // pass
            [5.0, 0.0, 15.0],  // z too high
            [150.0, 0.0, 0.0], // too far
        ];

        let params = FilterParams {
            min_distance: 1.0,
            max_distance: 100.0,
            min_z: -2.0,
            max_z: 10.0,
            downsample_resolution: None,
        };

        let result = filter_points_cpu(&points, &params);
        assert_eq!(result.points.len(), 1);
        assert_eq!(result.points[0], [5.0, 0.0, 0.0]);
    }

    #[test]
    fn test_voxel_downsample_empty() {
        let result = voxel_downsample_cpu(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_voxel_downsample_invalid_resolution() {
        let points = vec![[1.0, 2.0, 3.0]];
        let result = voxel_downsample_cpu(&points, 0.0);
        assert_eq!(result.len(), 1);

        let result = voxel_downsample_cpu(&points, -1.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_voxel_downsample_single_voxel() {
        // Points all within same voxel (resolution 1.0)
        let points = vec![
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
        ];

        let result = voxel_downsample_cpu(&points, 1.0);
        assert_eq!(result.len(), 1);

        // Centroid should be (0.25, 0.25, 0.25)
        let c = result[0];
        assert!((c[0] - 0.25).abs() < 1e-5);
        assert!((c[1] - 0.25).abs() < 1e-5);
        assert!((c[2] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_voxel_downsample_multiple_voxels() {
        // Points in different voxels
        let points = vec![
            [0.1, 0.1, 0.1], // voxel (0, 0, 0)
            [1.1, 0.1, 0.1], // voxel (1, 0, 0)
            [0.1, 1.1, 0.1], // voxel (0, 1, 0)
            [0.1, 0.1, 1.1], // voxel (0, 0, 1)
        ];

        let result = voxel_downsample_cpu(&points, 1.0);
        assert_eq!(result.len(), 4); // Each point in its own voxel
    }

    #[test]
    fn test_voxel_downsample_negative_coords() {
        // Points with negative coordinates
        let points = vec![[-0.1, -0.1, -0.1], [-0.2, -0.2, -0.2]];

        let result = voxel_downsample_cpu(&points, 1.0);
        assert_eq!(result.len(), 1);

        // Centroid should be (-0.15, -0.15, -0.15)
        let c = result[0];
        assert!((c[0] - (-0.15)).abs() < 1e-5);
        assert!((c[1] - (-0.15)).abs() < 1e-5);
        assert!((c[2] - (-0.15)).abs() < 1e-5);
    }

    #[test]
    fn test_filter_with_downsampling() {
        let points = vec![
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [5.0, 0.0, 0.0], // different voxel
        ];

        let params = FilterParams {
            downsample_resolution: Some(1.0),
            ..Default::default()
        };

        let result = filter_points_cpu(&points, &params);
        assert_eq!(result.points.len(), 2); // Two voxels
        assert_eq!(result.removed_by_downsampling, 2); // 3 points merged to 1
    }
}
