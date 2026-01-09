//! GPU-accelerated voxel grid implementation.
//!
//! This module integrates all GPU voxel grid components into a single
//! `GpuVoxelGrid` struct that provides the same interface as the CPU
//! `VoxelGrid` but uses GPU-accelerated algorithms.
//!
//! # Pipeline
//!
//! ```text
//! Points → Morton Codes → Radix Sort → Segments → Statistics → VoxelGrid
//!                                                      ↓
//!                                              Radius Search Index
//! ```
//!
//! # Current Status
//!
//! CPU reference implementation provided. GPU execution requires CubeCL
//! kernel fixes and will be added in a future update.

use super::{
    compute_morton_codes_cpu, compute_voxel_statistics_cpu, detect_segments_cpu, radius_search_cpu,
    radix_sort_by_key, RadiusSearchConfig,
};

/// GPU-accelerated voxel grid for NDT scan matching.
///
/// Contains voxel statistics (mean, covariance) and provides
/// efficient radius search for derivative computation.
#[derive(Debug, Clone)]
pub struct GpuVoxelGrid {
    /// Mean position for each voxel, [V, 3] flattened.
    pub means: Vec<f32>,
    /// Inverse covariance matrix for each voxel, [V, 9] flattened.
    pub inv_covariances: Vec<f32>,
    /// Covariance matrix for each voxel, [V, 9] flattened.
    pub covariances: Vec<f32>,
    /// Morton codes for each voxel (sorted).
    pub morton_codes: Vec<u64>,
    /// Number of points in each voxel.
    pub point_counts: Vec<u32>,
    /// Whether each voxel is valid (has enough points).
    pub valid: Vec<bool>,
    /// Grid resolution (voxel size).
    pub resolution: f32,
    /// Grid minimum bounds (for Morton code computation).
    pub grid_min: [f32; 3],
    /// Grid maximum bounds.
    pub grid_max: [f32; 3],
    /// Number of voxels.
    pub num_voxels: usize,
    /// Minimum points required for a valid voxel.
    pub min_points: u32,
}

/// Configuration for GPU voxel grid construction.
#[derive(Debug, Clone)]
pub struct GpuVoxelGridConfig {
    /// Voxel resolution (size of each voxel).
    pub resolution: f32,
    /// Minimum number of points required per voxel.
    pub min_points: u32,
}

impl Default for GpuVoxelGridConfig {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            min_points: 3,
        }
    }
}

impl GpuVoxelGrid {
    /// Construct a GPU voxel grid from a point cloud.
    ///
    /// # Arguments
    /// * `points` - Point cloud as flat array [x0, y0, z0, x1, y1, z1, ...]
    /// * `config` - Grid configuration
    ///
    /// # Returns
    /// A new GpuVoxelGrid with computed voxel statistics.
    pub fn from_points(points: &[f32], config: &GpuVoxelGridConfig) -> Self {
        let num_points = points.len() / 3;

        if num_points == 0 {
            return Self {
                means: Vec::new(),
                inv_covariances: Vec::new(),
                covariances: Vec::new(),
                morton_codes: Vec::new(),
                point_counts: Vec::new(),
                valid: Vec::new(),
                resolution: config.resolution,
                grid_min: [0.0; 3],
                grid_max: [0.0; 3],
                num_voxels: 0,
                min_points: config.min_points,
            };
        }

        // Step 1: Compute Morton codes
        let morton_result = compute_morton_codes_cpu(points, config.resolution);

        // Step 2: Sort by Morton code
        let codes: Vec<u64> = morton_result
            .codes
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let indices: Vec<u32> = morton_result
            .indices
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let sort_result = radix_sort_by_key(&codes, &indices);

        let sorted_codes: Vec<u64> = sort_result
            .keys
            .chunks(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let sorted_indices: Vec<u32> = sort_result
            .values
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Step 3: Detect segments (voxel boundaries)
        let segments = detect_segments_cpu(&sorted_codes);

        // Step 4: Compute voxel statistics
        let stats = compute_voxel_statistics_cpu(
            points,
            &sorted_indices,
            &segments.segment_ids,
            segments.num_segments,
            config.min_points,
        );

        // Extract unique Morton codes for each voxel
        let voxel_morton_codes = segments.segment_codes;

        Self {
            means: stats.means,
            inv_covariances: stats.inv_covariances,
            covariances: stats.covariances,
            morton_codes: voxel_morton_codes,
            point_counts: stats.point_counts,
            valid: stats.valid,
            resolution: config.resolution,
            grid_min: morton_result.grid_min,
            grid_max: morton_result.grid_max,
            num_voxels: segments.num_segments as usize,
            min_points: config.min_points,
        }
    }

    /// Get the number of valid voxels.
    pub fn num_valid_voxels(&self) -> usize {
        self.valid.iter().filter(|&&v| v).count()
    }

    /// Perform radius search to find voxels near a query point.
    ///
    /// # Arguments
    /// * `query_point` - Query point [x, y, z]
    /// * `radius` - Search radius
    /// * `max_neighbors` - Maximum number of neighbors to return
    ///
    /// # Returns
    /// Indices of voxels within the search radius.
    pub fn radius_search(
        &self,
        query_point: &[f32; 3],
        radius: f32,
        max_neighbors: usize,
    ) -> Vec<u32> {
        if self.num_voxels == 0 {
            return Vec::new();
        }

        let config = RadiusSearchConfig {
            radius,
            max_neighbors,
            resolution: self.resolution,
            grid_min: self.grid_min,
        };

        let query_flat = [query_point[0], query_point[1], query_point[2]];
        let results = radius_search_cpu(&query_flat, &self.means, &self.morton_codes, &config);

        if results.is_empty() {
            return Vec::new();
        }

        // Filter to only valid voxels
        results[0]
            .voxel_indices
            .iter()
            .filter(|&&idx| self.valid.get(idx as usize).copied().unwrap_or(false))
            .copied()
            .collect()
    }

    /// Get voxel mean by index.
    pub fn get_mean(&self, voxel_idx: usize) -> Option<[f32; 3]> {
        if voxel_idx >= self.num_voxels || !self.valid[voxel_idx] {
            return None;
        }

        Some([
            self.means[voxel_idx * 3],
            self.means[voxel_idx * 3 + 1],
            self.means[voxel_idx * 3 + 2],
        ])
    }

    /// Get voxel inverse covariance by index.
    pub fn get_inv_covariance(&self, voxel_idx: usize) -> Option<[[f32; 3]; 3]> {
        if voxel_idx >= self.num_voxels || !self.valid[voxel_idx] {
            return None;
        }

        let base = voxel_idx * 9;
        Some([
            [
                self.inv_covariances[base],
                self.inv_covariances[base + 1],
                self.inv_covariances[base + 2],
            ],
            [
                self.inv_covariances[base + 3],
                self.inv_covariances[base + 4],
                self.inv_covariances[base + 5],
            ],
            [
                self.inv_covariances[base + 6],
                self.inv_covariances[base + 7],
                self.inv_covariances[base + 8],
            ],
        ])
    }

    /// Get voxel covariance by index.
    pub fn get_covariance(&self, voxel_idx: usize) -> Option<[[f32; 3]; 3]> {
        if voxel_idx >= self.num_voxels || !self.valid[voxel_idx] {
            return None;
        }

        let base = voxel_idx * 9;
        Some([
            [
                self.covariances[base],
                self.covariances[base + 1],
                self.covariances[base + 2],
            ],
            [
                self.covariances[base + 3],
                self.covariances[base + 4],
                self.covariances[base + 5],
            ],
            [
                self.covariances[base + 6],
                self.covariances[base + 7],
                self.covariances[base + 8],
            ],
        ])
    }

    /// Iterate over all valid voxels.
    pub fn iter_valid_voxels(&self) -> impl Iterator<Item = ValidVoxel<'_>> {
        (0..self.num_voxels)
            .filter(|&idx| self.valid[idx])
            .map(move |idx| ValidVoxel {
                index: idx,
                mean: [
                    self.means[idx * 3],
                    self.means[idx * 3 + 1],
                    self.means[idx * 3 + 2],
                ],
                inv_covariance: &self.inv_covariances[idx * 9..idx * 9 + 9],
                covariance: &self.covariances[idx * 9..idx * 9 + 9],
                point_count: self.point_counts[idx],
            })
    }
}

/// A valid voxel from the grid.
#[derive(Debug)]
pub struct ValidVoxel<'a> {
    /// Voxel index.
    pub index: usize,
    /// Mean position.
    pub mean: [f32; 3],
    /// Inverse covariance (9 elements, row-major 3x3).
    pub inv_covariance: &'a [f32],
    /// Covariance (9 elements, row-major 3x3).
    pub covariance: &'a [f32],
    /// Number of points in this voxel.
    pub point_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_point_cloud() -> Vec<f32> {
        let mut points = Vec::new();

        // Cluster 1: around (0, 0, 0)
        for i in 0..10 {
            let offset = i as f32 * 0.1 - 0.45;
            points.push(offset);
            points.push(offset);
            points.push(offset);
        }

        // Cluster 2: around (5, 5, 5)
        for i in 0..10 {
            let offset = 5.0 + i as f32 * 0.1 - 0.45;
            points.push(offset);
            points.push(offset);
            points.push(offset);
        }

        // Cluster 3: around (10, 0, 0)
        for i in 0..10 {
            let offset = i as f32 * 0.1 - 0.45;
            points.push(10.0 + offset);
            points.push(offset);
            points.push(offset);
        }

        points
    }

    #[test]
    fn test_gpu_voxel_grid_construction() {
        let points = create_test_point_cloud();
        let config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 3,
        };

        let grid = GpuVoxelGrid::from_points(&points, &config);

        assert!(grid.num_voxels > 0);
        assert!(grid.num_valid_voxels() > 0);
    }

    #[test]
    fn test_gpu_voxel_grid_empty() {
        let config = GpuVoxelGridConfig::default();
        let grid = GpuVoxelGrid::from_points(&[], &config);

        assert_eq!(grid.num_voxels, 0);
        assert_eq!(grid.num_valid_voxels(), 0);
    }

    #[test]
    fn test_radius_search() {
        let points = create_test_point_cloud();
        let config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 3,
        };

        let grid = GpuVoxelGrid::from_points(&points, &config);

        // Search near cluster 1
        let results = grid.radius_search(&[0.0, 0.0, 0.0], 3.0, 10);
        assert!(!results.is_empty(), "Should find voxels near origin");

        // Search far from any cluster
        let results = grid.radius_search(&[100.0, 100.0, 100.0], 1.0, 10);
        assert!(results.is_empty(), "Should find no voxels far away");
    }

    #[test]
    fn test_get_mean_and_covariance() {
        let points = create_test_point_cloud();
        let config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 3,
        };

        let grid = GpuVoxelGrid::from_points(&points, &config);

        // Get first valid voxel
        let valid_voxels: Vec<_> = grid.iter_valid_voxels().collect();
        assert!(!valid_voxels.is_empty());

        let first = &valid_voxels[0];
        let mean = grid.get_mean(first.index).unwrap();
        let inv_cov = grid.get_inv_covariance(first.index).unwrap();

        // Mean should be finite
        assert!(mean.iter().all(|&v| v.is_finite()));

        // Inverse covariance should be finite
        assert!(inv_cov
            .iter()
            .flat_map(|row| row.iter())
            .all(|&v| v.is_finite()));
    }

    #[test]
    fn test_iter_valid_voxels() {
        let points = create_test_point_cloud();
        let config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 3,
        };

        let grid = GpuVoxelGrid::from_points(&points, &config);

        let valid_count = grid.iter_valid_voxels().count();
        assert_eq!(valid_count, grid.num_valid_voxels());

        for voxel in grid.iter_valid_voxels() {
            assert!(voxel.point_count >= config.min_points);
            assert_eq!(voxel.inv_covariance.len(), 9);
            assert_eq!(voxel.covariance.len(), 9);
        }
    }

    #[test]
    fn test_voxel_means_in_cluster() {
        // Create a simple cluster
        let points = vec![
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4,
        ];
        let config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 3,
        };

        let grid = GpuVoxelGrid::from_points(&points, &config);

        assert_eq!(grid.num_valid_voxels(), 1);

        let voxel = grid.iter_valid_voxels().next().unwrap();

        // Mean should be approximately (0.2, 0.2, 0.2)
        assert!((voxel.mean[0] - 0.2).abs() < 0.01);
        assert!((voxel.mean[1] - 0.2).abs() < 0.01);
        assert!((voxel.mean[2] - 0.2).abs() < 0.01);
    }
}
