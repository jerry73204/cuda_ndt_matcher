//! CPU reference implementation for voxel grid construction.
//!
//! This provides a baseline for correctness testing of GPU kernels.

use std::collections::HashMap;

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use super::types::{Voxel, VoxelCoord, VoxelGridConfig};

/// Accumulated statistics for a single voxel during construction.
#[derive(Debug, Clone, Default)]
struct VoxelAccumulator {
    /// Sum of all points (for mean computation).
    sum: Vector3<f64>,
    /// Sum of outer products (x * x^T) for covariance computation.
    sum_sq: Matrix3<f64>,
    /// Number of points accumulated.
    count: usize,
}

impl VoxelAccumulator {
    /// Add a point to this accumulator.
    fn add_point(&mut self, point: &[f32; 3]) {
        let v = Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);
        self.sum += v;
        self.sum_sq += v * v.transpose();
        self.count += 1;
    }
}

/// Build a voxel grid from a point cloud using CPU.
///
/// This is the reference implementation for correctness testing.
///
/// # Arguments
/// * `points` - Input point cloud (array of [x, y, z] coordinates)
/// * `config` - Voxel grid configuration
///
/// # Returns
/// A HashMap mapping voxel coordinates to Voxel structures.
pub fn build_voxel_grid_cpu(
    points: &[[f32; 3]],
    config: &VoxelGridConfig,
) -> HashMap<VoxelCoord, Voxel> {
    // Phase 1: Accumulate points into voxels
    let mut accumulators: HashMap<VoxelCoord, VoxelAccumulator> = HashMap::new();

    for point in points {
        let coord = VoxelCoord::from_point(point, config.resolution);
        accumulators.entry(coord).or_default().add_point(point);
    }

    // Phase 2: Convert accumulators to voxels (parallel)
    let entries: Vec<_> = accumulators.into_iter().collect();

    let voxels: Vec<_> = entries
        .into_par_iter()
        .filter_map(|(coord, acc)| {
            let voxel = Voxel::from_statistics(&acc.sum, &acc.sum_sq, acc.count, config)?;
            Some((coord, voxel))
        })
        .collect();

    voxels.into_iter().collect()
}

/// Compute voxel ID from coordinates (for GPU compatibility testing).
///
/// Uses a simple linear index: id = x + y * dim_x + z * dim_x * dim_y
///
/// # Arguments
/// * `coord` - Voxel coordinate
/// * `min_bound` - Minimum bounds (offset for non-negative indices)
/// * `grid_dims` - Grid dimensions [dim_x, dim_y, dim_z]
pub fn voxel_coord_to_id(coord: VoxelCoord, min_bound: VoxelCoord, grid_dims: [u32; 3]) -> u32 {
    let local_x = (coord.x - min_bound.x) as u32;
    let local_y = (coord.y - min_bound.y) as u32;
    let local_z = (coord.z - min_bound.z) as u32;

    local_x + local_y * grid_dims[0] + local_z * grid_dims[0] * grid_dims[1]
}

/// Compute voxel coordinate from ID (inverse of voxel_coord_to_id).
pub fn voxel_id_to_coord(id: u32, min_bound: VoxelCoord, grid_dims: [u32; 3]) -> VoxelCoord {
    let dim_xy = grid_dims[0] * grid_dims[1];
    let z = id / dim_xy;
    let remainder = id % dim_xy;
    let y = remainder / grid_dims[0];
    let x = remainder % grid_dims[0];

    VoxelCoord::new(
        x as i32 + min_bound.x,
        y as i32 + min_bound.y,
        z as i32 + min_bound.z,
    )
}

/// Compute bounding box of voxel coordinates.
///
/// Returns (min_bound, max_bound, grid_dims).
pub fn compute_voxel_bounds(coords: &[VoxelCoord]) -> (VoxelCoord, VoxelCoord, [u32; 3]) {
    if coords.is_empty() {
        return (
            VoxelCoord::new(0, 0, 0),
            VoxelCoord::new(0, 0, 0),
            [1, 1, 1],
        );
    }

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut max_z = i32::MIN;

    for coord in coords {
        min_x = min_x.min(coord.x);
        min_y = min_y.min(coord.y);
        min_z = min_z.min(coord.z);
        max_x = max_x.max(coord.x);
        max_y = max_y.max(coord.y);
        max_z = max_z.max(coord.z);
    }

    let dims = [
        (max_x - min_x + 1) as u32,
        (max_y - min_y + 1) as u32,
        (max_z - min_z + 1) as u32,
    ];

    (
        VoxelCoord::new(min_x, min_y, min_z),
        VoxelCoord::new(max_x, max_y, max_z),
        dims,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_clustered_points(center: [f32; 3], spread: f32, count: usize) -> Vec<[f32; 3]> {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, spread as f64).unwrap();

        (0..count)
            .map(|_| {
                [
                    center[0] + dist.sample(&mut rng) as f32,
                    center[1] + dist.sample(&mut rng) as f32,
                    center[2] + dist.sample(&mut rng) as f32,
                ]
            })
            .collect()
    }

    #[test]
    fn test_build_voxel_grid_single_cluster() {
        let config = VoxelGridConfig {
            resolution: 2.0,
            min_points_per_voxel: 6,
            ..Default::default()
        };

        // Create 20 points clustered around (5, 5, 5) with small spread
        // Center at 5.0 is well inside voxel [4, 6) with resolution 2.0
        let points = generate_clustered_points([5.0, 5.0, 5.0], 0.1, 20);

        let grid = build_voxel_grid_cpu(&points, &config);

        // Should have at least one voxel
        assert!(!grid.is_empty());

        // Check that we have a voxel near (5, 5, 5)
        let expected_coord = VoxelCoord::from_point(&[5.0, 5.0, 5.0], config.resolution);
        assert!(grid.contains_key(&expected_coord));

        let voxel = &grid[&expected_coord];
        assert!(voxel.point_count >= 6);
    }

    #[test]
    fn test_build_voxel_grid_multiple_clusters() {
        let config = VoxelGridConfig {
            resolution: 2.0,
            min_points_per_voxel: 6,
            ..Default::default()
        };

        // Create clusters at different locations with tight spread to stay within voxels
        // Centers chosen to be well inside voxel boundaries
        // [1,1,1] -> voxel [0,0,0], covers [0,2)
        // [11,1,1] -> voxel [5,0,0], covers [10,12) in x
        // [1,11,1] -> voxel [0,5,0], covers [10,12) in y
        let mut points = Vec::new();
        points.extend(generate_clustered_points([1.0, 1.0, 1.0], 0.1, 10));
        points.extend(generate_clustered_points([11.0, 1.0, 1.0], 0.1, 10));
        points.extend(generate_clustered_points([1.0, 11.0, 1.0], 0.1, 10));

        let grid = build_voxel_grid_cpu(&points, &config);

        // Should have 3 voxels
        assert_eq!(grid.len(), 3, "Expected 3 voxels but got {}", grid.len());
    }

    #[test]
    fn test_voxel_coord_id_roundtrip() {
        let min_bound = VoxelCoord::new(-5, -3, -2);
        let grid_dims = [10, 8, 5];

        for x in -5..5 {
            for y in -3..5 {
                for z in -2..3 {
                    let coord = VoxelCoord::new(x, y, z);
                    let id = voxel_coord_to_id(coord, min_bound, grid_dims);
                    let recovered = voxel_id_to_coord(id, min_bound, grid_dims);
                    assert_eq!(coord, recovered);
                }
            }
        }
    }

    #[test]
    fn test_compute_voxel_bounds() {
        let coords = vec![
            VoxelCoord::new(-2, 1, 0),
            VoxelCoord::new(3, -1, 2),
            VoxelCoord::new(0, 0, 1),
        ];

        let (min_bound, max_bound, dims) = compute_voxel_bounds(&coords);

        assert_eq!(min_bound, VoxelCoord::new(-2, -1, 0));
        assert_eq!(max_bound, VoxelCoord::new(3, 1, 2));
        assert_eq!(dims, [6, 3, 3]);
    }

    #[test]
    fn test_empty_point_cloud() {
        let config = VoxelGridConfig::default();
        let points: Vec<[f32; 3]> = Vec::new();

        let grid = build_voxel_grid_cpu(&points, &config);
        assert!(grid.is_empty());
    }

    #[test]
    fn test_sparse_points_filtered() {
        let config = VoxelGridConfig {
            resolution: 10.0, // Large voxels
            min_points_per_voxel: 6,
            ..Default::default()
        };

        // Create 5 points - should be filtered out
        let points: Vec<[f32; 3]> = (0..5).map(|i| [i as f32, 0.0, 0.0]).collect();

        let grid = build_voxel_grid_cpu(&points, &config);
        assert!(grid.is_empty());
    }
}
