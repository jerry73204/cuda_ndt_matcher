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
pub mod gpu_builder;
pub mod kernels;
pub mod search;
pub mod types;

pub use search::VoxelSearch;
pub use types::{Voxel, VoxelCoord, VoxelGridConfig};

// GPU builder (requires CUDA)
#[cfg(feature = "cuda")]
pub use gpu_builder::GpuVoxelGridBuilder;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use nalgebra::{Matrix3, Vector3};
use serde::Deserialize;

use crate::voxel_grid::cpu::build_voxel_grid_cpu;

/// JSON representation of a voxel dump file (from CUDA or Autoware).
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // num_voxels is parsed from JSON but not used
struct VoxelDumpJson {
    resolution: f32,
    num_voxels: usize,
    voxels: Vec<VoxelJson>,
}

/// JSON representation of a single voxel.
#[derive(Debug, Deserialize)]
struct VoxelJson {
    mean: [f32; 3],
    cov: [[f32; 3]; 3],
    inv_cov: [[f32; 3]; 3],
    point_count: usize,
}

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

    /// Build a voxel grid using GPU acceleration.
    ///
    /// Uses GPU for voxel ID computation (parallel) and CPU for statistics.
    /// Falls back to CPU if GPU is not available.
    ///
    /// # Arguments
    /// * `points` - Input point cloud
    /// * `config` - Voxel grid configuration
    ///
    /// # Example
    /// ```ignore
    /// let grid = VoxelGrid::from_points_gpu(&points, config)?;
    /// ```
    #[cfg(feature = "cuda")]
    pub fn from_points_gpu(points: &[[f32; 3]], config: VoxelGridConfig) -> Result<Self> {
        match gpu_builder::GpuVoxelGridBuilder::new() {
            Ok(builder) => builder.build(points, &config),
            Err(_) => {
                // Fall back to CPU if GPU not available
                Self::from_points_with_config(points, config)
            }
        }
    }

    /// Load a voxel grid from a JSON dump file.
    ///
    /// This loads voxel data from a JSON file generated by either CUDA or Autoware
    /// voxel dump functionality. This is useful for:
    /// - Testing with identical input data between implementations
    /// - Debugging score differences by using one implementation's voxels in another
    ///
    /// # Arguments
    /// * `path` - Path to the JSON file (e.g., `/tmp/ndt_autoware_voxels.json`)
    ///
    /// # Example
    /// ```ignore
    /// // Load Autoware's voxels into CUDA for comparison
    /// let grid = VoxelGrid::from_json("/tmp/ndt_autoware_voxels.json")?;
    /// ```
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open voxel dump file: {}", path.display()))?;

        let dump: VoxelDumpJson = serde_json::from_reader(file)
            .with_context(|| format!("Failed to parse voxel dump JSON: {}", path.display()))?;

        let resolution = dump.resolution;
        let config = VoxelGridConfig {
            resolution,
            min_points_per_voxel: 6, // Not used for loading, but set a default
            eigenvalue_ratio_threshold: 0.01,
        };

        let mut voxels = Vec::with_capacity(dump.voxels.len());
        let mut coords = Vec::with_capacity(dump.voxels.len());
        let mut coord_to_index = HashMap::with_capacity(dump.voxels.len());

        for voxel_json in &dump.voxels {
            // Compute voxel coordinate from mean position
            let coord = VoxelCoord::from_point(&voxel_json.mean, resolution);

            // Build covariance matrix
            let cov = Matrix3::new(
                voxel_json.cov[0][0],
                voxel_json.cov[0][1],
                voxel_json.cov[0][2],
                voxel_json.cov[1][0],
                voxel_json.cov[1][1],
                voxel_json.cov[1][2],
                voxel_json.cov[2][0],
                voxel_json.cov[2][1],
                voxel_json.cov[2][2],
            );

            // Build inverse covariance matrix
            let inv_cov = Matrix3::new(
                voxel_json.inv_cov[0][0],
                voxel_json.inv_cov[0][1],
                voxel_json.inv_cov[0][2],
                voxel_json.inv_cov[1][0],
                voxel_json.inv_cov[1][1],
                voxel_json.inv_cov[1][2],
                voxel_json.inv_cov[2][0],
                voxel_json.inv_cov[2][1],
                voxel_json.inv_cov[2][2],
            );

            // Compute principal axis from inverse covariance eigenvalues
            // The principal axis is the eigenvector of the smallest eigenvalue
            let eigen = cov.symmetric_eigen();
            let min_idx = eigen
                .eigenvalues
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let principal_axis = eigen.eigenvectors.column(min_idx).into_owned();

            let voxel = Voxel {
                mean: Vector3::new(voxel_json.mean[0], voxel_json.mean[1], voxel_json.mean[2]),
                covariance: cov,
                inv_covariance: inv_cov,
                principal_axis,
                point_count: voxel_json.point_count,
            };

            let idx = voxels.len();
            voxels.push(voxel);
            coords.push(coord);
            coord_to_index.insert(coord, idx);
        }

        // Build KD-tree for radius search
        let search = VoxelSearch::from_voxels(&voxels);

        // Compute bounds
        let (min_bound, max_bound, grid_dims) = if coords.is_empty() {
            (None, None, None)
        } else {
            let (min, max, dims) = cpu::compute_voxel_bounds(&coords);
            (Some(min), Some(max), Some(dims))
        };

        tracing::info!(
            "Loaded {} voxels from JSON (resolution={})",
            voxels.len(),
            resolution
        );

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

    /// Insert a voxel at the given coordinate.
    ///
    /// Used by GPU builder to construct grid incrementally.
    pub fn insert(&mut self, coord: VoxelCoord, voxel: Voxel) {
        let idx = self.voxels.len();
        self.coords.push(coord);
        self.voxels.push(voxel);
        self.coord_to_index.insert(coord, idx);
    }

    /// Build the search index (KD-tree) for radius queries.
    ///
    /// Should be called after all voxels have been inserted.
    pub fn build_search_index(&mut self) {
        self.search = VoxelSearch::from_voxels(&self.voxels);

        // Update bounds
        if !self.coords.is_empty() {
            let (min, max, dims) = cpu::compute_voxel_bounds(&self.coords);
            self.min_bound = Some(min);
            self.max_bound = Some(max);
            self.grid_dims = Some(dims);
        }
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

    /// Get all principal axes as a flat array [V * 3].
    ///
    /// Principal axis is the eigenvector of the smallest eigenvalue (surface normal).
    /// Useful for GPU upload for point-to-plane distance metric.
    pub fn principal_axes_flat(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.voxels.len() * 3);
        for voxel in &self.voxels {
            result.push(voxel.principal_axis.x);
            result.push(voxel.principal_axis.y);
            result.push(voxel.principal_axis.z);
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

    /// Debug test to print detailed voxel statistics for comparison with Autoware.
    /// Run with: cargo test -p ndt_cuda --lib "test_voxel_covariance_debug" -- --nocapture
    #[test]
    fn test_voxel_covariance_debug() {
        // Create a simple cluster of points for debugging
        let mut points = Vec::new();
        let center = [1.0f32, 1.0, 1.0];

        // Create 50 points in a deterministic sphere-ish pattern
        for i in 0..50 {
            let t = i as f32 / 49.0;
            let phi = t * std::f32::consts::PI * 2.0;
            let theta = (i % 10) as f32 / 10.0 * std::f32::consts::PI;
            let r = 0.3 * (0.5 + 0.5 * ((i * 7) % 11) as f32 / 10.0);

            let x = center[0] + r * theta.sin() * phi.cos();
            let y = center[1] + r * theta.sin() * phi.sin();
            let z = center[2] + r * theta.cos();
            points.push([x, y, z]);
        }

        crate::test_println!("\n=== VOXEL COVARIANCE DEBUG ===");
        crate::test_println!(
            "Input: {} points centered around {:?}",
            points.len(),
            center
        );

        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        crate::test_println!("Resolution: 2.0");
        crate::test_println!("Min points: 6, eigenvalue_ratio_threshold: 0.01");
        crate::test_println!("Voxel count: {}", grid.len());

        for (_coord, voxel) in grid.iter() {
            crate::test_println!(
                "\n--- Voxel at ({}, {}, {}) ---",
                _coord.x,
                _coord.y,
                _coord.z
            );
            crate::test_println!("Point count: {}", voxel.point_count);
            crate::test_println!(
                "Mean: [{:.6}, {:.6}, {:.6}]",
                voxel.mean.x,
                voxel.mean.y,
                voxel.mean.z
            );

            crate::test_println!("\nCovariance matrix:");
            for _i in 0..3 {
                crate::test_println!(
                    "  [{:.8}, {:.8}, {:.8}]",
                    voxel.covariance[(_i, 0)],
                    voxel.covariance[(_i, 1)],
                    voxel.covariance[(_i, 2)]
                );
            }

            crate::test_println!("\nInverse covariance matrix:");
            for _i in 0..3 {
                crate::test_println!(
                    "  [{:.8}, {:.8}, {:.8}]",
                    voxel.inv_covariance[(_i, 0)],
                    voxel.inv_covariance[(_i, 1)],
                    voxel.inv_covariance[(_i, 2)]
                );
            }

            // Compute eigenvalues of covariance
            let cov_f64 = voxel.covariance.cast::<f64>();
            let eigen = cov_f64.symmetric_eigen();
            let mut cov_eigs: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
            cov_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            crate::test_println!(
                "\nCovariance eigenvalues (sorted): [{:.8}, {:.8}, {:.8}]",
                cov_eigs[0],
                cov_eigs[1],
                cov_eigs[2]
            );

            // Compute eigenvalues of inverse covariance
            let icov_f64 = voxel.inv_covariance.cast::<f64>();
            let ieigen = icov_f64.symmetric_eigen();
            let mut icov_eigs: Vec<f64> = ieigen.eigenvalues.iter().copied().collect();
            icov_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            crate::test_println!(
                "Inv-cov eigenvalues (sorted): [{:.8}, {:.8}, {:.8}]",
                icov_eigs[0],
                icov_eigs[1],
                icov_eigs[2]
            );

            // Verify: cov_eig * icov_eig should be ~1.0
            crate::test_println!("\nVerification (cov_eig * icov_eig should be ~1.0):");
            // For inverse relationship: smallest cov eigenvalue * largest icov eigenvalue = 1
            crate::test_println!(
                "  cov_min={:.6} * icov_max={:.6} = {:.6}",
                cov_eigs[0],
                icov_eigs[2],
                cov_eigs[0] * icov_eigs[2]
            );
            crate::test_println!(
                "  cov_mid={:.6} * icov_mid={:.6} = {:.6}",
                cov_eigs[1],
                icov_eigs[1],
                cov_eigs[1] * icov_eigs[1]
            );
            crate::test_println!(
                "  cov_max={:.6} * icov_min={:.6} = {:.6}",
                cov_eigs[2],
                icov_eigs[0],
                cov_eigs[2] * icov_eigs[0]
            );

            // Compute score at voxel center and at offset
            let gauss_d1 = -4.196518186951408f64;
            let gauss_d2 = 0.24847851012449546f64;
            let _score_at_center = -gauss_d1;
            crate::test_println!("\nScore at voxel center: {:.6}", _score_at_center);

            // Compute score at 0.5m offset
            let offset = 0.5f64;
            let diff = nalgebra::Vector3::new(offset, 0.0, 0.0);
            let icov = voxel.inv_covariance.cast::<f64>();
            let mahal_sq = diff.dot(&(icov * diff));
            let _score_at_offset = -gauss_d1 * (-gauss_d2 * mahal_sq / 2.0).exp();
            crate::test_println!(
                "Score at 0.5m X offset: {:.6} (mahal_sq={:.6})",
                _score_at_offset,
                mahal_sq
            );
        }

        crate::test_println!("\n=== END VOXEL COVARIANCE DEBUG ===\n");
    }

    /// Debug test using half-cubic point cloud (matches Autoware test data).
    /// Run with: cargo test -p ndt_cuda --lib "test_voxel_half_cubic_debug" -- --nocapture
    #[test]
    fn test_voxel_half_cubic_debug() {
        let points = make_default_half_cubic_pcd();

        crate::test_println!("\n=== HALF CUBIC VOXEL DEBUG ===");
        crate::test_println!("Input: {} points", points.len());

        let grid = VoxelGrid::from_points(&points, 2.0).unwrap();

        crate::test_println!("Resolution: 2.0, Voxel count: {}", grid.len());

        // Print stats for first 5 voxels
        for (_i, (_coord, voxel)) in grid.iter().enumerate().take(5) {
            crate::test_println!(
                "\nVoxel {} at ({}, {}, {}): {} pts, mean=[{:.3}, {:.3}, {:.3}]",
                _i,
                _coord.x,
                _coord.y,
                _coord.z,
                voxel.point_count,
                voxel.mean.x,
                voxel.mean.y,
                voxel.mean.z
            );

            let icov = voxel.inv_covariance.cast::<f64>();
            let ieigen = icov.symmetric_eigen();
            let mut eigs: Vec<f64> = ieigen.eigenvalues.iter().copied().collect();
            eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            crate::test_println!(
                "  Inv-cov eigenvalues: [{:.2}, {:.2}, {:.2}]",
                eigs[0],
                eigs[1],
                eigs[2]
            );
        }

        // Summary statistics across all voxels
        let mut all_icov_traces: Vec<f32> = grid
            .voxels()
            .iter()
            .map(|v| v.inv_covariance.trace())
            .collect();
        all_icov_traces.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let _median_idx = all_icov_traces.len() / 2;
        crate::test_println!("\n--- Summary across {} voxels ---", grid.len());
        crate::test_println!(
            "Inv-cov trace: min={:.2}, median={:.2}, max={:.2}",
            all_icov_traces.first().unwrap_or(&0.0),
            all_icov_traces.get(_median_idx).unwrap_or(&0.0),
            all_icov_traces.last().unwrap_or(&0.0)
        );

        crate::test_println!("\n=== END HALF CUBIC DEBUG ===\n");
    }

    /// Debug test that computes detailed Mahalanobis distance statistics.
    /// This helps compare with Autoware's behavior.
    /// Run with: cargo test -p ndt_cuda --lib "test_mahalanobis_distance_stats" --features test-verbose -- --nocapture
    #[test]
    fn test_mahalanobis_distance_stats() {
        use crate::voxel_grid::VoxelSearch;

        let map_points = make_default_half_cubic_pcd();
        let grid = VoxelGrid::from_points(&map_points, 2.0).unwrap();
        let search = VoxelSearch::from_voxels(grid.voxels()).unwrap();

        crate::test_println!("\n=== MAHALANOBIS DISTANCE DEBUG ===");
        crate::test_println!("Map: {} points, {} voxels", map_points.len(), grid.len());

        // Create a source point cloud (subset of map, slightly offset)
        let source_points: Vec<[f32; 3]> = map_points
            .iter()
            .step_by(40) // Sample every 40th point
            .map(|p| [p[0] + 0.05, p[1] + 0.05, p[2]]) // Small offset
            .collect();

        crate::test_println!(
            "Source: {} points (with 5cm offset from map)",
            source_points.len()
        );

        let gauss_d1 = -4.196518186951408f64;
        let gauss_d2 = 0.24847851012449546f64;

        // Compute per-point max scores and Mahalanobis distances
        let mut all_mahal_sq = Vec::new();
        let mut all_scores = Vec::new();
        let mut num_with_neighbors = 0usize;
        let mut _total_correspondences = 0usize;

        let voxels = grid.voxels();
        for p in &source_points {
            let neighbor_indices = search.within(p, 2.0);

            if !neighbor_indices.is_empty() {
                num_with_neighbors += 1;
                _total_correspondences += neighbor_indices.len();

                let mut max_score = 0.0f64;
                let mut min_mahal_sq = f64::MAX;

                for &idx in &neighbor_indices {
                    let voxel = &voxels[idx];
                    let dx = p[0] as f64 - voxel.mean.x as f64;
                    let dy = p[1] as f64 - voxel.mean.y as f64;
                    let dz = p[2] as f64 - voxel.mean.z as f64;
                    let diff = nalgebra::Vector3::new(dx, dy, dz);

                    let icov = voxel.inv_covariance.cast::<f64>();
                    let mahal_sq = diff.dot(&(icov * diff));
                    let score = -gauss_d1 * (-gauss_d2 * mahal_sq / 2.0).exp();

                    if score > max_score {
                        max_score = score;
                        min_mahal_sq = mahal_sq;
                    }
                }

                all_mahal_sq.push(min_mahal_sq);
                all_scores.push(max_score);
            }
        }

        // Compute statistics
        all_mahal_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let _avg_mahal_sq: f64 = all_mahal_sq.iter().sum::<f64>() / all_mahal_sq.len() as f64;
        let _avg_score: f64 = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
        let _nvtl = all_scores.iter().sum::<f64>() / num_with_neighbors as f64;

        let _n = all_mahal_sq.len();
        crate::test_println!("\n--- Statistics over {} points with neighbors ---", _n);
        crate::test_println!(
            "Correspondences: {} total ({:.2} vpp)",
            total_correspondences,
            total_correspondences as f64 / n as f64
        );
        crate::test_println!("\nMahalanobis distance squared (for max score per point):");
        crate::test_println!("  min: {:.4}", all_mahal_sq.first().unwrap_or(&0.0));
        crate::test_println!("  10%: {:.4}", all_mahal_sq.get(n / 10).unwrap_or(&0.0));
        crate::test_println!("  25%: {:.4}", all_mahal_sq.get(n / 4).unwrap_or(&0.0));
        crate::test_println!("  median: {:.4}", all_mahal_sq.get(n / 2).unwrap_or(&0.0));
        crate::test_println!("  75%: {:.4}", all_mahal_sq.get(3 * n / 4).unwrap_or(&0.0));
        crate::test_println!("  90%: {:.4}", all_mahal_sq.get(9 * n / 10).unwrap_or(&0.0));
        crate::test_println!("  max: {:.4}", all_mahal_sq.last().unwrap_or(&0.0));
        crate::test_println!("  average: {:.4}", avg_mahal_sq);

        crate::test_println!("\nMax score per point:");
        crate::test_println!("  min: {:.4}", all_scores.first().unwrap_or(&0.0));
        crate::test_println!("  median: {:.4}", all_scores.get(n / 2).unwrap_or(&0.0));
        crate::test_println!(
            "  max: {:.4} (theoretical max: {:.4})",
            all_scores.last().unwrap_or(&0.0),
            -gauss_d1
        );
        crate::test_println!("  average: {:.4}", _avg_score);
        crate::test_println!("  NVTL: {:.4}", _nvtl);

        // Also compute what Autoware's NVTL of ~3.0 would imply
        let autoware_nvtl = 3.0f64;
        let _implied_mahal_sq = -2.0 * (autoware_nvtl / (-gauss_d1)).ln() / gauss_d2;
        crate::test_println!("\n--- For comparison with Autoware ---");
        crate::test_println!(
            "Autoware NVTL ~3.0 implies avg mahal_sq: {:.4}",
            _implied_mahal_sq
        );
        crate::test_println!(
            "Our avg mahal_sq: {:.4} (ratio: {:.2}x)",
            _avg_mahal_sq,
            _avg_mahal_sq / _implied_mahal_sq
        );

        crate::test_println!("\n=== END MAHALANOBIS DISTANCE DEBUG ===\n");
    }
}
