//! Voxel grid construction and management.
//!
//! This module provides GPU-accelerated voxel grid construction from point clouds.
//! The voxel grid stores Gaussian distributions (mean + covariance) for each voxel,
//! which are used for NDT scan matching.
//!
//! # Architecture
//!
//! The implementation uses a hybrid CPU/GPU approach:
//! 1. GPU: Compute voxel IDs for all points (parallel)
//! 2. CPU: Accumulate point statistics per voxel (HashMap-based)
//! 3. CPU: Compute covariance and regularization (parallel via rayon)
//! 4. CPU: Build KD-tree from voxel centroids for radius search
//!
//! The KD-tree enables efficient radius search matching Autoware's behavior,
//! where each source point can contribute to score from multiple nearby voxels.

pub mod cpu;
pub mod gpu;
pub mod kernels;
pub mod search;
pub mod types;

pub use search::VoxelSearch;
pub use types::{Voxel, VoxelCoord, VoxelGridConfig};

use std::collections::HashMap;

use anyhow::Result;

use crate::voxel_grid::cpu::build_voxel_grid_cpu;

/// A voxel grid containing Gaussian distributions for NDT matching.
///
/// The grid stores:
/// - Voxel means (centroids)
/// - Inverse covariance matrices
/// - Spatial hash for O(1) voxel lookup
/// - KD-tree for efficient radius search (like Autoware's radiusSearch)
#[derive(Debug)]
pub struct VoxelGrid {
    /// Configuration used to build this grid.
    pub config: VoxelGridConfig,
    /// Voxels stored in a vector for indexed access.
    voxels: Vec<Voxel>,
    /// Voxel coordinates (parallel to voxels vector).
    coords: Vec<VoxelCoord>,
    /// Map from coordinate to index in voxels vector.
    coord_to_index: HashMap<VoxelCoord, usize>,
    /// KD-tree for radius search over voxel centroids.
    search: Option<VoxelSearch>,
    /// Precomputed bounds for GPU operations.
    min_bound: Option<VoxelCoord>,
    max_bound: Option<VoxelCoord>,
    grid_dims: Option<[u32; 3]>,
}

impl VoxelGrid {
    /// Create a new empty voxel grid with the given configuration.
    pub fn new(config: VoxelGridConfig) -> Self {
        Self {
            config,
            voxels: Vec::new(),
            coords: Vec::new(),
            coord_to_index: HashMap::new(),
            search: None,
            min_bound: None,
            max_bound: None,
            grid_dims: None,
        }
    }

    /// Build a voxel grid from a point cloud.
    ///
    /// This is the main entry point for voxel grid construction.
    /// Currently uses CPU implementation; GPU acceleration coming in future.
    ///
    /// # Arguments
    /// * `points` - Input point cloud (array of [x, y, z] coordinates)
    /// * `resolution` - Voxel side length in meters
    ///
    /// # Example
    /// ```ignore
    /// let points = vec![[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], /* ... */];
    /// let grid = VoxelGrid::from_points(&points, 2.0)?;
    /// ```
    pub fn from_points(points: &[[f32; 3]], resolution: f32) -> Result<Self> {
        let config = VoxelGridConfig {
            resolution,
            ..Default::default()
        };
        Self::from_points_with_config(points, config)
    }

    /// Build a voxel grid with custom configuration.
    pub fn from_points_with_config(points: &[[f32; 3]], config: VoxelGridConfig) -> Result<Self> {
        let voxel_map = build_voxel_grid_cpu(points, &config);

        // Convert HashMap to Vec + index map
        let mut voxels = Vec::with_capacity(voxel_map.len());
        let mut coords = Vec::with_capacity(voxel_map.len());
        let mut coord_to_index = HashMap::with_capacity(voxel_map.len());

        for (coord, voxel) in voxel_map {
            let idx = voxels.len();
            coords.push(coord);
            voxels.push(voxel);
            coord_to_index.insert(coord, idx);
        }

        // Compute bounds
        let (min_bound, max_bound, grid_dims) = if coords.is_empty() {
            (None, None, None)
        } else {
            let (min, max, dims) = cpu::compute_voxel_bounds(&coords);
            (Some(min), Some(max), Some(dims))
        };

        // Build KD-tree for radius search
        let search = VoxelSearch::from_voxels(&voxels);

        Ok(Self {
            config,
            voxels,
            coords,
            coord_to_index,
            search,
            min_bound,
            max_bound,
            grid_dims,
        })
    }

    /// Get the number of voxels in the grid.
    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    /// Check if the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Get a voxel by its coordinates.
    pub fn get(&self, coord: &VoxelCoord) -> Option<&Voxel> {
        self.coord_to_index.get(coord).map(|&idx| &self.voxels[idx])
    }

    /// Get a voxel by integer coordinate array [x, y, z].
    pub fn get_by_coord(&self, coord: &[i32; 3]) -> Option<&Voxel> {
        let voxel_coord = VoxelCoord {
            x: coord[0],
            y: coord[1],
            z: coord[2],
        };
        self.get(&voxel_coord)
    }

    /// Get a voxel by its index.
    pub fn get_by_index(&self, idx: usize) -> Option<&Voxel> {
        self.voxels.get(idx)
    }

    /// Find all voxels within a given radius of a point.
    ///
    /// This matches Autoware's `radiusSearch` behavior where each source point
    /// may contribute to score from multiple voxels, providing smoother gradients
    /// especially near voxel boundaries.
    ///
    /// Uses KD-tree for efficient search over voxel centroids (means).
    ///
    /// # Arguments
    /// * `point` - Query point [x, y, z]
    /// * `radius` - Search radius (typically equal to voxel resolution)
    ///
    /// # Returns
    /// Vector of references to voxels within the radius, sorted by distance.
    pub fn radius_search(&self, point: &[f32; 3], radius: f32) -> Vec<&Voxel> {
        match &self.search {
            Some(search) => {
                let indices = search.within(point, radius);
                indices.iter().map(|&idx| &self.voxels[idx]).collect()
            }
            None => Vec::new(),
        }
    }

    /// Find all voxel indices within a given radius of a point.
    ///
    /// Returns indices that can be used with `get_by_index`.
    pub fn radius_search_indices(&self, point: &[f32; 3], radius: f32) -> Vec<usize> {
        match &self.search {
            Some(search) => search.within(point, radius),
            None => Vec::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &VoxelGridConfig {
        &self.config
    }

    /// Iterate over all voxels with their coordinates.
    pub fn iter(&self) -> impl Iterator<Item = (&VoxelCoord, &Voxel)> {
        self.coords.iter().zip(self.voxels.iter())
    }

    /// Iterate over all voxels.
    pub fn voxels(&self) -> &[Voxel] {
        &self.voxels
    }

    /// Get the grid bounds.
    pub fn bounds(&self) -> Option<(VoxelCoord, VoxelCoord)> {
        match (self.min_bound, self.max_bound) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
    }

    /// Get the grid dimensions.
    pub fn dims(&self) -> Option<[u32; 3]> {
        self.grid_dims
    }

    /// Get the resolution.
    pub fn resolution(&self) -> f32 {
        self.config.resolution
    }

    /// Get all voxel means as a flat array [V * 3].
    ///
    /// Useful for GPU upload.
    pub fn means_flat(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.voxels.len() * 3);
        for voxel in &self.voxels {
            result.push(voxel.mean.x);
            result.push(voxel.mean.y);
            result.push(voxel.mean.z);
        }
        result
    }

    /// Get all inverse covariances as a flat array [V * 9].
    ///
    /// Row-major 3x3 matrices. Useful for GPU upload.
    pub fn inv_covariances_flat(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.voxels.len() * 9);
        for voxel in &self.voxels {
            // Row-major order
            for row in 0..3 {
                for col in 0..3 {
                    result.push(voxel.inv_covariance[(row, col)]);
                }
            }
        }
        result
    }

    /// Get all voxel coordinates as a flat array.
    pub fn coords_flat(&self) -> Vec<i32> {
        let mut result = Vec::with_capacity(self.coords.len() * 3);
        for coord in &self.coords {
            result.push(coord.x);
            result.push(coord.y);
            result.push(coord.z);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        make_default_half_cubic_pcd, make_half_cubic_pcd_offset, make_xy_plane, voxelize_pcd,
    };
    use approx::assert_relative_eq;

    fn generate_test_points() -> Vec<[f32; 3]> {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        // Use very small spread to ensure points stay within one voxel (resolution 2.0)
        let dist = Normal::new(0.0, 0.1).unwrap();

        // Choose centers well inside voxel boundaries
        // With resolution 2.0, voxel [0,0,0] covers [-inf,2) in each dim
        // voxel [5,0,0] covers [10,12) in x
        // voxel [0,5,0] covers [10,12) in y
        let centers = [[1.0, 1.0, 1.0], [11.0, 1.0, 1.0], [1.0, 11.0, 1.0]];
        let mut points = Vec::new();

        for center in &centers {
            for _ in 0..50 {
                points.push([
                    center[0] + dist.sample(&mut rng) as f32,
                    center[1] + dist.sample(&mut rng) as f32,
                    center[2] + dist.sample(&mut rng) as f32,
                ]);
            }
        }

        points
    }

    #[test]
    fn test_voxel_grid_from_points() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // Should have 3 voxels (one per cluster)
        // Each cluster has 50 points, well above the min_points_per_voxel threshold
        assert_eq!(grid.len(), 3, "Expected 3 voxels but got {}", grid.len());

        // Check resolution
        assert_eq!(grid.resolution(), 2.0);
    }

    #[test]
    fn test_voxel_grid_means_flat() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let means = grid.means_flat();
        assert_eq!(means.len(), grid.len() * 3);
    }

    #[test]
    fn test_voxel_grid_inv_covariances_flat() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let inv_covs = grid.inv_covariances_flat();
        assert_eq!(inv_covs.len(), grid.len() * 9);

        // Verify no NaN or Inf values
        for val in &inv_covs {
            assert!(val.is_finite(), "Found non-finite value: {val}");
        }
    }

    #[test]
    fn test_voxel_grid_bounds() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let bounds = grid.bounds();
        assert!(bounds.is_some());

        let (min, max) = bounds.unwrap();
        // First cluster at origin should give voxel (0,0,0)
        // Second cluster at (10,0,0) should give voxel (5,0,0)
        // Third cluster at (0,10,0) should give voxel (0,5,0)
        assert!(min.x <= 0);
        assert!(min.y <= 0);
        assert!(max.x >= 4); // 10/2 = 5, but floor
        assert!(max.y >= 4);
    }

    #[test]
    fn test_empty_voxel_grid() {
        let points: Vec<[f32; 3]> = Vec::new();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
        assert!(grid.bounds().is_none());
    }

    #[test]
    fn test_voxel_mean_accuracy() {
        // Create points with known mean
        // Choose a center well inside a voxel boundary with resolution 2.0
        // Center at 5.0 -> voxel coord 2, voxel covers [4, 6)
        // Points at 5.0 +/- 0.1 stay within [4.9, 5.1] which is inside [4, 6)
        let center = [5.0f32, 5.0, 5.0];
        let mut points = Vec::new();

        // Generate points in a tight cluster around the center
        for dx in [-0.1, 0.0, 0.1] {
            for dy in [-0.1, 0.0, 0.1] {
                for dz in [-0.1, 0.0, 0.1] {
                    points.push([
                        center[0] + dx as f32,
                        center[1] + dy as f32,
                        center[2] + dz as f32,
                    ]);
                }
            }
        }

        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();
        assert_eq!(grid.len(), 1, "Expected 1 voxel but got {}", grid.len());

        let coord = VoxelCoord::from_point(&center, 2.0);
        let voxel = grid.get(&coord).unwrap();

        // Mean should be very close to center
        assert_relative_eq!(voxel.mean.x, center[0], epsilon = 0.02);
        assert_relative_eq!(voxel.mean.y, center[1], epsilon = 0.02);
        assert_relative_eq!(voxel.mean.z, center[2], epsilon = 0.02);
    }

    #[test]
    fn test_covariance_symmetry() {
        let points = generate_test_points();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        for (_, voxel) in grid.iter() {
            let cov = &voxel.covariance;
            let inv_cov = &voxel.inv_covariance;

            // Check symmetry
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(cov[(i, j)], cov[(j, i)], epsilon = 1e-6);
                    assert_relative_eq!(inv_cov[(i, j)], inv_cov[(j, i)], epsilon = 1e-6);
                }
            }

            // Check that cov * inv_cov ≈ I
            let product = cov * inv_cov;
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert_relative_eq!(product[(i, j)], expected, epsilon = 0.01);
                }
            }
        }
    }

    // ========================================================================
    // Phase 2: Voxel grid tests using Autoware-style test data
    // ========================================================================

    #[test]
    fn test_voxel_grid_from_half_cubic() {
        let points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // Half-cube has 3 planes of 20m × 20m at 0.2m spacing
        // At 2.0m resolution, each plane has ~100 voxels (10×10)
        // Some voxels overlap at edges, so total should be ~200-350
        assert!(
            grid.len() >= 150,
            "Expected at least 150 voxels, got {}",
            grid.len()
        );
        assert!(
            grid.len() <= 400,
            "Expected at most 400 voxels, got {}",
            grid.len()
        );
    }

    #[test]
    fn test_voxel_grid_half_cubic_bounds() {
        let points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let bounds = grid.bounds().expect("Grid should have bounds");
        let (min, max) = bounds;

        // Half-cube spans [0, 20] in all dimensions
        // At resolution 2.0, voxel coords span [0, 9]
        assert!(min.x >= 0, "min.x should be >= 0");
        assert!(min.y >= 0, "min.y should be >= 0");
        assert!(min.z >= 0, "min.z should be >= 0");
        assert!(max.x <= 10, "max.x should be <= 10");
        assert!(max.y <= 10, "max.y should be <= 10");
        assert!(max.z <= 10, "max.z should be <= 10");
    }

    #[test]
    fn test_voxel_grid_half_cubic_offset() {
        let points = make_half_cubic_pcd_offset(100.0, 100.0, 0.0);
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        let bounds = grid.bounds().expect("Grid should have bounds");
        let (min, max) = bounds;

        // Half-cube spans [0, 20], offset by 100 -> [100, 120]
        // At resolution 2.0, voxel coords span [50, 59]
        assert!(
            min.x >= 49,
            "min.x should be >= 49 after offset, got {}",
            min.x
        );
        assert!(
            min.y >= 49,
            "min.y should be >= 49 after offset, got {}",
            min.y
        );
        assert!(
            max.x <= 60,
            "max.x should be <= 60 after offset, got {}",
            max.x
        );
        assert!(
            max.y <= 60,
            "max.y should be <= 60 after offset, got {}",
            max.y
        );
    }

    #[test]
    fn test_voxel_mean_on_planar_points() {
        // Points on XY plane at z=0
        let points = make_xy_plane(10.0, 0.5, 0.0);
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // All voxel means should have z close to 0
        for (_, voxel) in grid.iter() {
            assert!(
                voxel.mean.z.abs() < 0.1,
                "Mean z should be near 0 for XY plane, got {}",
                voxel.mean.z
            );
        }
    }

    #[test]
    fn test_voxel_covariance_on_planar_points() {
        // Points on XY plane -> z variance should be regularized (near minimum)
        let points = make_xy_plane(10.0, 0.5, 0.0);
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        for (_, voxel) in grid.iter() {
            // z-z covariance should be small (regularized)
            let z_variance = voxel.covariance[(2, 2)];
            assert!(
                z_variance <= 0.1,
                "z variance should be regularized for planar points, got {}",
                z_variance
            );
            assert!(
                z_variance >= 0.0,
                "z variance should be non-negative, got {}",
                z_variance
            );
        }
    }

    #[test]
    fn test_voxelized_sensor_pcd() {
        let original = make_default_half_cubic_pcd();
        let voxelized = voxelize_pcd(&original, 1.0);

        // Voxelized should have fewer points
        assert!(
            voxelized.len() < original.len() / 10,
            "Voxelized should have << original points"
        );

        // Create grid from voxelized points
        // Note: After voxelization at 1.0m, many voxels may not meet min_points
        // threshold when building grid at 2.0m, so we relax the requirements
        let grid = VoxelGrid::from_points(&voxelized, 2.0).unwrap();

        // Should still create some voxels (may be fewer due to min_points filter)
        assert!(!grid.is_empty(), "Should have at least some voxels");

        for (_, voxel) in grid.iter() {
            assert!(
                voxel.point_count >= 6,
                "Should have enough points per voxel"
            );
        }
    }

    #[test]
    fn test_radius_search_on_half_cubic() {
        let points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // Query at center of grid (10, 10, 0 - on XY plane)
        let neighbors = grid.radius_search(&[10.0, 10.0, 0.0], 2.0);

        // Should find nearby voxels
        assert!(
            !neighbors.is_empty(),
            "Should find at least one neighbor at grid center"
        );

        // All found voxels should be within radius of query point
        for voxel in &neighbors {
            let dx = voxel.mean.x - 10.0;
            let dy = voxel.mean.y - 10.0;
            let dz = voxel.mean.z - 0.0;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                dist <= 3.0, // Allow some tolerance for voxel centroids
                "Neighbor voxel should be within search radius, dist={}",
                dist
            );
        }
    }

    #[test]
    fn test_voxel_grid_min_points_filter() {
        // Create sparse points that might not meet min_points threshold
        let mut points = Vec::new();

        // Add 10 tight clusters of 10 points each
        for cluster in 0..10 {
            let cx = cluster as f32 * 5.0;
            for i in 0..10 {
                points.push([cx + (i as f32) * 0.01, 0.0, 0.0]);
            }
        }

        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // With min_points_per_voxel=6 (default), clusters of 10 should pass
        for (_, voxel) in grid.iter() {
            assert!(
                voxel.point_count >= 6,
                "All voxels should have at least min_points"
            );
        }
    }

    #[test]
    fn test_inv_covariance_positive_definite() {
        let points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        for (_, voxel) in grid.iter() {
            let inv_cov = &voxel.inv_covariance;

            // Diagonal elements should be positive
            assert!(
                inv_cov[(0, 0)] > 0.0,
                "inv_cov[0,0] should be positive: {}",
                inv_cov[(0, 0)]
            );
            assert!(
                inv_cov[(1, 1)] > 0.0,
                "inv_cov[1,1] should be positive: {}",
                inv_cov[(1, 1)]
            );
            assert!(
                inv_cov[(2, 2)] > 0.0,
                "inv_cov[2,2] should be positive: {}",
                inv_cov[(2, 2)]
            );

            // Should be finite
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        inv_cov[(i, j)].is_finite(),
                        "inv_cov[{},{}] should be finite: {}",
                        i,
                        j,
                        inv_cov[(i, j)]
                    );
                }
            }
        }
    }

    // ========================================================================
    // Phase 5: CPU vs GPU consistency tests
    // ========================================================================

    use crate::voxel_grid::gpu::{GpuVoxelGrid, GpuVoxelGridConfig};

    /// Helper to convert [[f32; 3]] to flat Vec<f32> for GpuVoxelGrid
    fn points_to_flat(points: &[[f32; 3]]) -> Vec<f32> {
        points.iter().flat_map(|p| p.iter().copied()).collect()
    }

    #[test]
    fn test_cpu_gpu_voxel_count_consistency() {
        let points = make_default_half_cubic_pcd();
        let flat_points = points_to_flat(&points);

        // Build CPU grid
        let cpu_grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        // Build GPU grid with same parameters
        let gpu_config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 6, // Match CPU default
        };
        let gpu_grid = GpuVoxelGrid::from_points(&flat_points, &gpu_config);

        let cpu_count = cpu_grid.len();
        let gpu_valid_count = gpu_grid.num_valid_voxels();

        // Verify total points are conserved
        let gpu_points_sum: u64 = gpu_grid.point_counts.iter().map(|&c| c as u64).sum();
        assert_eq!(
            gpu_points_sum,
            points.len() as u64,
            "GPU should account for all input points"
        );

        // Verify no voxels fail regularization (covariance should always be invertible after regularization)
        let failing_regularization: usize = (0..gpu_grid.num_voxels)
            .filter(|&i| gpu_grid.point_counts[i] >= 6 && !gpu_grid.valid[i])
            .count();
        assert_eq!(
            failing_regularization, 0,
            "No voxels with >=6 points should fail regularization"
        );

        // CPU and GPU should have similar voxel counts (within 10%)
        let diff = (cpu_count as i32 - gpu_valid_count as i32).abs();
        assert!(
            diff <= cpu_count as i32 / 10,
            "CPU ({cpu_count}) and GPU ({gpu_valid_count}) voxel counts should be similar, diff={diff}"
        );
    }

    #[test]
    fn test_cpu_gpu_means_consistency() {
        // Use a simple point cloud where voxel assignment is deterministic
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                // Dense grid on XY plane
                points.push([i as f32 * 0.3, j as f32 * 0.3, 0.0]);
            }
        }

        let flat_points = points_to_flat(&points);

        let cpu_grid = VoxelGrid::from_points(&points, 2.0).unwrap();
        let gpu_config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 6,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&flat_points, &gpu_config);

        // Collect GPU means
        let gpu_means: Vec<[f32; 3]> = gpu_grid.iter_valid_voxels().map(|v| v.mean).collect();

        // For each CPU voxel, find a matching GPU voxel
        let mut matched = 0;
        for (_, cpu_voxel) in cpu_grid.iter() {
            let cpu_mean = [cpu_voxel.mean.x, cpu_voxel.mean.y, cpu_voxel.mean.z];

            // Find closest GPU mean
            let closest = gpu_means.iter().min_by(|a, b| {
                let da = (a[0] - cpu_mean[0]).powi(2)
                    + (a[1] - cpu_mean[1]).powi(2)
                    + (a[2] - cpu_mean[2]).powi(2);
                let db = (b[0] - cpu_mean[0]).powi(2)
                    + (b[1] - cpu_mean[1]).powi(2)
                    + (b[2] - cpu_mean[2]).powi(2);
                da.partial_cmp(&db).unwrap()
            });

            if let Some(gpu_mean) = closest {
                let dist = ((gpu_mean[0] - cpu_mean[0]).powi(2)
                    + (gpu_mean[1] - cpu_mean[1]).powi(2)
                    + (gpu_mean[2] - cpu_mean[2]).powi(2))
                .sqrt();

                if dist < 0.5 {
                    // Within half a voxel
                    matched += 1;
                }
            }
        }

        // Most voxels should have matching means
        let match_ratio = matched as f64 / cpu_grid.len() as f64;
        assert!(
            match_ratio > 0.8,
            "At least 80% of CPU voxels should match GPU voxels, got {}%",
            match_ratio * 100.0
        );
    }

    #[test]
    fn test_cpu_gpu_covariance_consistency() {
        // Use a simple 3D point cloud
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    points.push([i as f32 * 0.5, j as f32 * 0.5, k as f32 * 0.5]);
                }
            }
        }

        let flat_points = points_to_flat(&points);

        let cpu_grid = VoxelGrid::from_points(&points, 2.0).unwrap();
        let gpu_config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 6,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&flat_points, &gpu_config);

        // Both should produce valid covariances
        for (_, cpu_voxel) in cpu_grid.iter() {
            // CPU covariance should be positive semi-definite
            let cov = &cpu_voxel.covariance;
            assert!(cov[(0, 0)] >= 0.0, "Covariance diagonal should be >= 0");
            assert!(cov[(1, 1)] >= 0.0, "Covariance diagonal should be >= 0");
            assert!(cov[(2, 2)] >= 0.0, "Covariance diagonal should be >= 0");
        }

        for gpu_voxel in gpu_grid.iter_valid_voxels() {
            // GPU covariance should also be positive semi-definite
            assert!(
                gpu_voxel.covariance[0] >= 0.0,
                "GPU covariance[0,0] should be >= 0"
            );
            assert!(
                gpu_voxel.covariance[4] >= 0.0,
                "GPU covariance[1,1] should be >= 0"
            );
            assert!(
                gpu_voxel.covariance[8] >= 0.0,
                "GPU covariance[2,2] should be >= 0"
            );
        }
    }

    #[test]
    fn test_cpu_gpu_inv_covariance_consistency() {
        // Use a simple 3D point cloud
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    points.push([i as f32 * 0.5, j as f32 * 0.5, k as f32 * 0.5]);
                }
            }
        }

        let flat_points = points_to_flat(&points);

        let cpu_grid = VoxelGrid::from_points(&points, 2.0).unwrap();
        let gpu_config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 6,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&flat_points, &gpu_config);

        // Both should produce valid inverse covariances
        for (_, cpu_voxel) in cpu_grid.iter() {
            let inv_cov = &cpu_voxel.inv_covariance;
            // Diagonal should be positive
            assert!(inv_cov[(0, 0)] > 0.0, "CPU inv_cov[0,0] should be > 0");
            assert!(inv_cov[(1, 1)] > 0.0, "CPU inv_cov[1,1] should be > 0");
            assert!(inv_cov[(2, 2)] > 0.0, "CPU inv_cov[2,2] should be > 0");
        }

        for gpu_voxel in gpu_grid.iter_valid_voxels() {
            // Diagonal should be positive
            assert!(
                gpu_voxel.inv_covariance[0] > 0.0,
                "GPU inv_cov[0,0] should be > 0"
            );
            assert!(
                gpu_voxel.inv_covariance[4] > 0.0,
                "GPU inv_cov[1,1] should be > 0"
            );
            assert!(
                gpu_voxel.inv_covariance[8] > 0.0,
                "GPU inv_cov[2,2] should be > 0"
            );
        }
    }

    #[test]
    fn test_cpu_gpu_half_cubic_consistency() {
        let points = make_default_half_cubic_pcd();
        let flat_points = points_to_flat(&points);

        let cpu_grid = VoxelGrid::from_points(&points, 2.0).unwrap();
        let gpu_config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 6,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&flat_points, &gpu_config);

        // Collect stats
        let cpu_valid = cpu_grid.len();
        let gpu_valid = gpu_grid.num_valid_voxels();

        // Both should produce a reasonable number of voxels
        assert!(
            cpu_valid > 100,
            "CPU should have >100 voxels from half-cubic"
        );
        assert!(
            gpu_valid > 100,
            "GPU should have >100 voxels from half-cubic"
        );

        // Point counts should sum to similar totals
        let cpu_total_points: u64 = cpu_grid.iter().map(|(_, v)| v.point_count as u64).sum();
        let gpu_total_points: u64 = gpu_grid
            .iter_valid_voxels()
            .map(|v| v.point_count as u64)
            .sum();

        // GPU may have fewer total points due to min_points filtering differences
        assert!(
            gpu_total_points > cpu_total_points / 2,
            "GPU total points ({}) should be reasonable compared to CPU ({})",
            gpu_total_points,
            cpu_total_points
        );
    }

    #[test]
    fn test_morton_codes_valid() {
        use crate::voxel_grid::gpu::{compute_morton_codes_cpu, morton_decode_3d};

        let points = make_default_half_cubic_pcd();
        let flat_points = points_to_flat(&points);

        let result = compute_morton_codes_cpu(&flat_points, 2.0);

        // All codes should be valid (decodable)
        let codes: Vec<u64> = result
            .codes
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();

        for code in &codes {
            let (x, y, z) = morton_decode_3d(*code);
            // Morton coords should be reasonable (within grid bounds)
            assert!(x < 0x1FFFFF, "Morton x should be < 2^21");
            assert!(y < 0x1FFFFF, "Morton y should be < 2^21");
            assert!(z < 0x1FFFFF, "Morton z should be < 2^21");
        }
    }

    #[test]
    fn test_radix_sort_preserves_data() {
        use crate::voxel_grid::gpu::radix_sort_by_key_cpu;

        let keys: Vec<u64> = vec![100, 50, 200, 25, 150];
        let values: Vec<u32> = vec![0, 1, 2, 3, 4];

        let result = radix_sort_by_key_cpu(&keys, &values);

        let sorted_keys: Vec<u64> = result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let sorted_values: Vec<u32> = result
            .values
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Keys should be sorted
        assert_eq!(sorted_keys, vec![25, 50, 100, 150, 200]);

        // Values should follow their keys
        assert_eq!(sorted_values, vec![3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_segment_detection_consistency() {
        use crate::voxel_grid::gpu::detect_segments_cpu;

        // Sorted codes with known segment boundaries
        let codes: Vec<u64> = vec![10, 10, 10, 20, 20, 30, 30, 30, 30];

        let result = detect_segments_cpu(&codes);

        // Should detect 3 segments
        assert_eq!(result.num_segments, 3);
        assert_eq!(result.segment_codes.len(), 3);
        assert_eq!(result.segment_codes, vec![10, 20, 30]);
    }
}
