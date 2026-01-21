//! Visualization utilities for NDT debugging.
//!
//! This module provides functions for visualizing NDT data, including:
//! - Per-point score visualization (colored point clouds)
//! - Pose history markers
//! - Particle filter visualization

use ndt_cuda::scoring::colors::{
    color_to_rgb_packed, ndt_score_to_color, DEFAULT_SCORE_LOWER, DEFAULT_SCORE_UPPER,
};
use ndt_cuda::scoring::nvtl::{compute_nvtl, NvtlConfig};
use ndt_cuda::GaussianParams;
use ndt_cuda::VoxelGrid;
use sensor_msgs::msg::PointCloud2;
use sensor_msgs::msg::PointField;
use std_msgs::msg::Header;

/// Configuration for point score visualization.
#[derive(Debug, Clone)]
pub struct PointScoreConfig {
    /// Lower bound for score color mapping.
    pub score_lower: f32,
    /// Upper bound for score color mapping.
    pub score_upper: f32,
    /// Search radius for NVTL computation (in voxel units).
    pub search_radius: i32,
}

impl Default for PointScoreConfig {
    fn default() -> Self {
        Self {
            score_lower: DEFAULT_SCORE_LOWER,
            score_upper: DEFAULT_SCORE_UPPER,
            search_radius: 1,
        }
    }
}

/// Generate a colored point cloud showing per-point NDT scores.
///
/// For each source point:
/// 1. Transform by the given pose
/// 2. Compute nearest voxel score (max score across neighbors)
/// 3. Map score to RGB color
/// 4. Output as PointXYZRGB
///
/// This matches Autoware's `visualize_point_score` function.
///
/// # Arguments
/// * `source_points` - Source point cloud
/// * `target_grid` - Target voxel grid (map)
/// * `pose` - Transform to apply to source points
/// * `gauss` - Gaussian parameters for NDT score function
/// * `header` - ROS message header
/// * `config` - Visualization configuration
///
/// # Returns
/// PointCloud2 message with RGB-colored points
pub fn visualize_point_scores(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &nalgebra::Isometry3<f64>,
    gauss: &GaussianParams,
    header: &Header,
    config: &PointScoreConfig,
) -> PointCloud2 {
    // Compute per-point NVTL scores
    let nvtl_config = NvtlConfig {
        search_radius: config.search_radius,
        compute_per_point: true,
    };
    let nvtl_result = compute_nvtl(source_points, target_grid, pose, gauss, &nvtl_config);

    // Get per-point scores (or use empty vec if NVTL failed)
    let scores = nvtl_result.per_point_scores.unwrap_or_default();

    // Transform source points to world frame
    let transformed_points: Vec<[f32; 3]> = source_points
        .iter()
        .map(|p| {
            let pt = nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            let transformed = pose * pt;
            [
                transformed.x as f32,
                transformed.y as f32,
                transformed.z as f32,
            ]
        })
        .collect();

    // Create PointCloud2 with XYZRGB fields
    let mut msg = PointCloud2::default();
    msg.header = header.clone();
    msg.height = 1;
    msg.width = source_points.len() as u32;
    msg.is_dense = true;
    msg.is_bigendian = false;

    // Define fields: x, y, z (float32), rgb (uint32)
    msg.fields = vec![
        PointField {
            name: "x".to_string(),
            offset: 0,
            datatype: PointField::FLOAT32,
            count: 1,
        },
        PointField {
            name: "y".to_string(),
            offset: 4,
            datatype: PointField::FLOAT32,
            count: 1,
        },
        PointField {
            name: "z".to_string(),
            offset: 8,
            datatype: PointField::FLOAT32,
            count: 1,
        },
        PointField {
            name: "rgb".to_string(),
            offset: 12,
            datatype: PointField::UINT32,
            count: 1,
        },
    ];

    msg.point_step = 16; // 3 floats (12 bytes) + 1 uint32 (4 bytes)
    msg.row_step = msg.point_step * msg.width;

    // Allocate data buffer
    let data_size = (msg.point_step * msg.width) as usize;
    msg.data = vec![0u8; data_size];

    // Fill point data
    for (i, (point, score)) in transformed_points.iter().zip(scores.iter()).enumerate() {
        let offset = (i * msg.point_step as usize) as usize;

        // Write XYZ
        msg.data[offset..offset + 4].copy_from_slice(&point[0].to_le_bytes());
        msg.data[offset + 4..offset + 8].copy_from_slice(&point[1].to_le_bytes());
        msg.data[offset + 8..offset + 12].copy_from_slice(&point[2].to_le_bytes());

        // Convert score to color
        let color = ndt_score_to_color(*score as f32, config.score_lower, config.score_upper);
        let rgb_packed = color_to_rgb_packed(&color);

        // Write RGB as uint32
        msg.data[offset + 12..offset + 16].copy_from_slice(&rgb_packed.to_le_bytes());
    }

    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualize_point_scores_empty() {
        let points: Vec<[f32; 3]> = vec![];
        let grid_config = ndt_cuda::VoxelGridConfig {
            resolution: 2.0,
            ..Default::default()
        };
        let grid = VoxelGrid::new(grid_config);
        let pose = nalgebra::Isometry3::identity();
        let gauss = GaussianParams::new(2.0, 0.55);
        let header = Header::default();
        let config = PointScoreConfig::default();

        let cloud = visualize_point_scores(&points, &grid, &pose, &gauss, &header, &config);
        assert_eq!(cloud.width, 0);
        assert!(cloud.data.is_empty());
    }

    #[test]
    fn test_visualize_point_scores_structure() {
        let points = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let grid_config = ndt_cuda::VoxelGridConfig {
            resolution: 2.0,
            ..Default::default()
        };
        let grid = VoxelGrid::new(grid_config);
        let pose = nalgebra::Isometry3::identity();
        let gauss = GaussianParams::new(2.0, 0.55);
        let header = Header::default();
        let config = PointScoreConfig::default();

        let cloud = visualize_point_scores(&points, &grid, &pose, &gauss, &header, &config);
        assert_eq!(cloud.width, 2);
        assert_eq!(cloud.height, 1);
        assert_eq!(cloud.point_step, 16);
        assert_eq!(cloud.fields.len(), 4);
        assert_eq!(cloud.fields[0].name, "x");
        assert_eq!(cloud.fields[3].name, "rgb");
    }
}
