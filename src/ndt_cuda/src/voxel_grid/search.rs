//! KD-tree based voxel search for radius queries.
//!
//! This module provides efficient radius search over voxel centroids,
//! matching Autoware's `MultiVoxelGridCovariance::radiusSearch` behavior.
//!
//! # Algorithm
//!
//! Autoware builds a KD-tree from voxel centroids (mean positions), then
//! uses radius search to find all voxels whose centroids are within a given
//! distance of a query point. This provides smoother gradients near voxel
//! boundaries compared to simple geometric containment lookup.
//!
//! # Usage
//!
//! ```ignore
//! let search = VoxelSearch::from_voxels(&voxels);
//! let nearby = search.within(&query_point, radius);
//! for idx in nearby {
//!     let voxel = &voxels[idx];
//!     // Process voxel...
//! }
//! ```

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;

use super::Voxel;

/// Bucket size for the KD-tree.
///
/// Increased from default 32 to handle planar point clouds where many
/// voxel centroids may have similar coordinates on one axis.
const BUCKET_SIZE: usize = 256;

/// KD-tree based search structure for voxel centroids.
///
/// This enables efficient radius search over voxel mean positions,
/// matching Autoware's behavior where each source point can contribute
/// to score from multiple nearby voxels.
#[derive(Debug)]
pub struct VoxelSearch {
    /// KD-tree built from voxel centroids.
    /// Generic args: A=f32 (coordinate type), T=u64 (item/index type), K=3 (dimensions), B=bucket size
    kdtree: ImmutableKdTree<f32, u64, 3, BUCKET_SIZE>,
}

impl VoxelSearch {
    /// Build a search index from a slice of voxels.
    ///
    /// The KD-tree is built from voxel mean positions (centroids).
    /// The index of each entry in the KD-tree corresponds to the index
    /// in the original voxels slice.
    ///
    /// # Arguments
    /// * `voxels` - Slice of voxels to index
    ///
    /// # Returns
    /// A new VoxelSearch instance, or None if voxels is empty.
    pub fn from_voxels(voxels: &[Voxel]) -> Option<Self> {
        if voxels.is_empty() {
            return None;
        }

        // Extract centroids from voxels
        let centroids: Vec<[f32; 3]> = voxels
            .iter()
            .map(|v| [v.mean.x, v.mean.y, v.mean.z])
            .collect();

        // Build immutable KD-tree (more efficient for static data)
        // The item value (u64) is automatically set to the array index
        // Use (&*centroids) to convert Vec to slice reference as required by From trait
        let kdtree: ImmutableKdTree<f32, u64, 3, BUCKET_SIZE> = (&*centroids).into();

        Some(Self { kdtree })
    }

    /// Find all voxel indices within a given radius of a query point.
    ///
    /// This matches Autoware's `radiusSearch(point, radius, neighborhood)` behavior.
    /// Returns indices into the original voxels slice that was used to build
    /// this search index.
    ///
    /// # Arguments
    /// * `point` - Query point [x, y, z]
    /// * `radius` - Search radius in meters
    ///
    /// # Returns
    /// Vector of voxel indices within the radius, sorted by distance (nearest first).
    pub fn within(&self, point: &[f32; 3], radius: f32) -> Vec<usize> {
        // Kiddo uses squared distance
        let radius_sq = radius * radius;

        // Query the KD-tree
        let results = self.kdtree.within::<SquaredEuclidean>(point, radius_sq);

        // Extract indices (item field contains the index)
        results.iter().map(|nn| nn.item as usize).collect()
    }

    /// Find all voxel indices within a given radius, also returning distances.
    ///
    /// # Arguments
    /// * `point` - Query point [x, y, z]
    /// * `radius` - Search radius in meters
    ///
    /// # Returns
    /// Vector of (index, squared_distance) pairs, sorted by distance (nearest first).
    pub fn within_with_distances(&self, point: &[f32; 3], radius: f32) -> Vec<(usize, f32)> {
        let radius_sq = radius * radius;
        let results = self.kdtree.within::<SquaredEuclidean>(point, radius_sq);

        results
            .iter()
            .map(|nn| (nn.item as usize, nn.distance))
            .collect()
    }

    /// Get the number of indexed voxels.
    pub fn len(&self) -> usize {
        self.kdtree.size()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.kdtree.size() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use nalgebra::{Matrix3, Vector3};

    fn make_test_voxel(mean: [f32; 3]) -> Voxel {
        Voxel {
            mean: Vector3::new(mean[0], mean[1], mean[2]),
            covariance: Matrix3::identity(),
            inv_covariance: Matrix3::identity(),
            point_count: 10,
        }
    }

    #[test]
    fn test_empty_voxels() {
        let voxels: Vec<Voxel> = vec![];
        let search = VoxelSearch::from_voxels(&voxels);
        assert!(search.is_none());
    }

    #[test]
    fn test_single_voxel() {
        let voxels = vec![make_test_voxel([5.0, 5.0, 5.0])];
        let search = VoxelSearch::from_voxels(&voxels).unwrap();

        assert_eq!(search.len(), 1);

        // Query at the same point
        let results = search.within(&[5.0, 5.0, 5.0], 1.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0);

        // Query too far away
        let results = search.within(&[100.0, 100.0, 100.0], 1.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_multiple_voxels_radius_search() {
        // Create voxels at different positions
        let voxels = vec![
            make_test_voxel([0.0, 0.0, 0.0]),
            make_test_voxel([1.0, 0.0, 0.0]),
            make_test_voxel([2.0, 0.0, 0.0]),
            make_test_voxel([10.0, 0.0, 0.0]), // Far away
        ];
        let search = VoxelSearch::from_voxels(&voxels).unwrap();

        // Query at origin with radius 1.5 should find voxels 0 and 1
        let results = search.within(&[0.0, 0.0, 0.0], 1.5);
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));

        // Query with larger radius should find voxels 0, 1, 2
        let results = search.within(&[0.0, 0.0, 0.0], 2.5);
        assert_eq!(results.len(), 3);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(results.contains(&2));

        // Query at center of all close voxels
        let results = search.within(&[1.0, 0.0, 0.0], 1.5);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_within_with_distances() {
        let voxels = vec![
            make_test_voxel([0.0, 0.0, 0.0]),
            make_test_voxel([1.0, 0.0, 0.0]),
            make_test_voxel([2.0, 0.0, 0.0]),
        ];
        let search = VoxelSearch::from_voxels(&voxels).unwrap();

        let results = search.within_with_distances(&[0.0, 0.0, 0.0], 2.5);

        // Check that distances are correct (squared)
        for (idx, dist_sq) in &results {
            let voxel = &voxels[*idx];
            let dx = voxel.mean.x;
            let expected_dist_sq = dx * dx;
            assert!((dist_sq - expected_dist_sq).abs() < 1e-6);
        }

        // Check sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_3d_radius_search() {
        // Create a 3D grid of voxels
        let mut voxels = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    voxels.push(make_test_voxel([x as f32, y as f32, z as f32]));
                }
            }
        }
        let search = VoxelSearch::from_voxels(&voxels).unwrap();

        // Query at center (1,1,1) with radius sqrt(3) ~ 1.73 should find all adjacent voxels
        let results = search.within(&[1.0, 1.0, 1.0], 1.8);

        // Should find center + 6 face-adjacent + some edge-adjacent
        // Face adjacent: distance 1.0
        // Edge adjacent: distance sqrt(2) ~ 1.41
        // Corner adjacent: distance sqrt(3) ~ 1.73
        assert!(results.len() >= 7); // At least center + 6 face-adjacent
    }
}
