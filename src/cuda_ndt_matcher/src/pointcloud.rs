//! Point cloud conversion utilities

use anyhow::{bail, Result};
use ndt_cuda::filtering::{CpuPointFilter, FilterParams as GpuFilterParams};
use sensor_msgs::msg::{PointCloud2, PointField};
use std_msgs::msg::Header;

/// Parameters for filtering sensor points
#[derive(Clone, Debug)]
pub struct PointFilterParams {
    /// Minimum distance from sensor origin (default: 0.0)
    pub min_distance: f32,
    /// Maximum distance from sensor origin (default: f32::MAX)
    pub max_distance: f32,
    /// Minimum z value (ground filtering, default: -f32::MAX)
    pub min_z: f32,
    /// Maximum z value (ceiling filtering, default: f32::MAX)
    pub max_z: f32,
    /// Voxel grid downsampling resolution (None = no downsampling)
    pub downsample_resolution: Option<f32>,
}

impl Default for PointFilterParams {
    fn default() -> Self {
        Self {
            min_distance: 0.0,
            max_distance: f32::MAX,
            min_z: f32::MIN,
            max_z: f32::MAX,
            downsample_resolution: None,
        }
    }
}

/// Result of point filtering operation
#[derive(Debug)]
pub struct FilterResult {
    /// Filtered points
    pub points: Vec<[f32; 3]>,
    /// Number of points removed by distance filter
    pub removed_by_distance: usize,
    /// Number of points removed by z filter
    pub removed_by_z: usize,
    /// Number of points removed by downsampling
    pub removed_by_downsampling: usize,
    /// Whether GPU acceleration was used
    pub used_gpu: bool,
}

/// Filter sensor points based on distance and z-height constraints
///
/// This implements Autoware's sensor point preprocessing:
/// - Distance-based filtering (min/max distance from sensor origin)
/// - Z-height filtering (ground/ceiling removal)
/// - Optional voxel grid downsampling
///
/// Uses GPU acceleration for large point clouds (>10k points) when available,
/// falling back to CPU for small clouds or when GPU is unavailable.
pub fn filter_sensor_points(points: &[[f32; 3]], params: &PointFilterParams) -> FilterResult {
    // Convert to ndt_cuda filter params
    let gpu_params = GpuFilterParams {
        min_distance: params.min_distance,
        max_distance: params.max_distance,
        min_z: params.min_z,
        max_z: params.max_z,
        downsample_resolution: params.downsample_resolution,
    };

    // Try GPU filter first if point cloud is large enough
    if points.len() >= 10000 {
        if let Ok(gpu_filter) = ndt_cuda::GpuPointFilter::new() {
            if let Ok(result) = gpu_filter.filter(points, &gpu_params) {
                return FilterResult {
                    points: result.points,
                    removed_by_distance: result.removed_by_distance,
                    removed_by_z: result.removed_by_z,
                    removed_by_downsampling: result.removed_by_downsampling,
                    used_gpu: result.used_gpu,
                };
            }
        }
    }

    // Fall back to CPU filter
    let cpu_filter = CpuPointFilter::new();
    let result = cpu_filter.filter(points, &gpu_params);

    FilterResult {
        points: result.points,
        removed_by_distance: result.removed_by_distance,
        removed_by_z: result.removed_by_z,
        removed_by_downsampling: result.removed_by_downsampling,
        used_gpu: false,
    }
}

/// Field offsets for XYZ point cloud
struct XyzOffsets {
    x: usize,
    y: usize,
    z: usize,
    point_step: usize,
}

impl XyzOffsets {
    fn from_pointcloud2(msg: &PointCloud2) -> Result<Self> {
        let mut x_offset = None;
        let mut y_offset = None;
        let mut z_offset = None;

        for field in &msg.fields {
            match field.name.as_str() {
                "x" => x_offset = Some(field.offset as usize),
                "y" => y_offset = Some(field.offset as usize),
                "z" => z_offset = Some(field.offset as usize),
                _ => {}
            }
        }

        let x = x_offset.ok_or_else(|| anyhow::anyhow!("Missing 'x' field"))?;
        let y = y_offset.ok_or_else(|| anyhow::anyhow!("Missing 'y' field"))?;
        let z = z_offset.ok_or_else(|| anyhow::anyhow!("Missing 'z' field"))?;

        Ok(Self {
            x,
            y,
            z,
            point_step: msg.point_step as usize,
        })
    }
}

/// Convert PointCloud2 message to Vec of [x, y, z] points
pub fn from_pointcloud2(msg: &PointCloud2) -> Result<Vec<[f32; 3]>> {
    if msg.data.is_empty() {
        return Ok(Vec::new());
    }

    let offsets = XyzOffsets::from_pointcloud2(msg)?;
    let num_points = (msg.width * msg.height) as usize;

    if msg.data.len() < num_points * offsets.point_step {
        bail!(
            "PointCloud2 data too short: {} < {}",
            msg.data.len(),
            num_points * offsets.point_step
        );
    }

    let mut points = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let base = i * offsets.point_step;

        let x = read_f32(&msg.data, base + offsets.x);
        let y = read_f32(&msg.data, base + offsets.y);
        let z = read_f32(&msg.data, base + offsets.z);

        // Skip NaN points
        if x.is_finite() && y.is_finite() && z.is_finite() {
            points.push([x, y, z]);
        }
    }

    Ok(points)
}

/// Read f32 from byte slice (little endian)
fn read_f32(data: &[u8], offset: usize) -> f32 {
    let bytes = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    f32::from_le_bytes(bytes)
}

/// Convert Vec of [x, y, z] points with RGB colors to PointCloud2 message.
///
/// Each point has xyz coordinates and a packed RGB value (0x00RRGGBB format).
/// This is used for per-point score visualization where colors indicate quality.
pub fn to_pointcloud2_with_rgb(
    points: &[[f32; 3]],
    rgb_values: &[u32],
    header: &Header,
) -> PointCloud2 {
    // Point format: x, y, z (float32), rgb (packed as float32 by reinterpreting bits)
    // Total: 16 bytes per point (same as Autoware's XYZI format)
    let point_step = 16u32;
    let mut data = Vec::with_capacity(points.len() * point_step as usize);

    for (i, p) in points.iter().enumerate() {
        data.extend_from_slice(&p[0].to_le_bytes()); // x
        data.extend_from_slice(&p[1].to_le_bytes()); // y
        data.extend_from_slice(&p[2].to_le_bytes()); // z
                                                     // RGB is packed as a float32 by reinterpreting the bits (ROS convention)
        let rgb = rgb_values.get(i).copied().unwrap_or(0);
        data.extend_from_slice(&f32::from_bits(rgb).to_le_bytes());
    }

    PointCloud2 {
        header: header.clone(),
        height: 1,
        width: points.len() as u32,
        fields: vec![
            PointField {
                name: "x".into(),
                offset: 0,
                datatype: 7, // FLOAT32
                count: 1,
            },
            PointField {
                name: "y".into(),
                offset: 4,
                datatype: 7,
                count: 1,
            },
            PointField {
                name: "z".into(),
                offset: 8,
                datatype: 7,
                count: 1,
            },
            PointField {
                name: "rgb".into(),
                offset: 12,
                datatype: 7, // FLOAT32 (bits reinterpreted as RGB)
                count: 1,
            },
        ],
        is_bigendian: false,
        point_step,
        row_step: point_step * points.len() as u32,
        data,
        is_dense: true,
    }
}

/// Convert Vec of [x, y, z] points to PointCloud2 message
pub fn to_pointcloud2(points: &[[f32; 3]], header: &Header) -> PointCloud2 {
    let point_step = 12u32; // 3 * sizeof(f32)
    let mut data = Vec::with_capacity(points.len() * point_step as usize);

    for p in points {
        data.extend_from_slice(&p[0].to_le_bytes());
        data.extend_from_slice(&p[1].to_le_bytes());
        data.extend_from_slice(&p[2].to_le_bytes());
    }

    PointCloud2 {
        header: header.clone(),
        height: 1,
        width: points.len() as u32,
        fields: vec![
            PointField {
                name: "x".into(),
                offset: 0,
                datatype: 7, // FLOAT32
                count: 1,
            },
            PointField {
                name: "y".into(),
                offset: 4,
                datatype: 7,
                count: 1,
            },
            PointField {
                name: "z".into(),
                offset: 8,
                datatype: 7,
                count: 1,
            },
        ],
        is_bigendian: false,
        point_step,
        row_step: point_step * points.len() as u32,
        data,
        is_dense: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_pointcloud(points: &[[f32; 3]]) -> PointCloud2 {
        let point_step = 12u32; // 3 * sizeof(f32)
        let mut data = Vec::with_capacity(points.len() * point_step as usize);

        for p in points {
            data.extend_from_slice(&p[0].to_le_bytes());
            data.extend_from_slice(&p[1].to_le_bytes());
            data.extend_from_slice(&p[2].to_le_bytes());
        }

        PointCloud2 {
            header: Default::default(),
            height: 1,
            width: points.len() as u32,
            fields: vec![
                PointField {
                    name: "x".into(),
                    offset: 0,
                    datatype: 7, // FLOAT32
                    count: 1,
                },
                PointField {
                    name: "y".into(),
                    offset: 4,
                    datatype: 7,
                    count: 1,
                },
                PointField {
                    name: "z".into(),
                    offset: 8,
                    datatype: 7,
                    count: 1,
                },
            ],
            is_bigendian: false,
            point_step,
            row_step: point_step * points.len() as u32,
            data,
            is_dense: true,
        }
    }

    #[test]
    fn test_from_pointcloud2() {
        let input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let msg = make_test_pointcloud(&input);

        let result = from_pointcloud2(&msg).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], [1.0, 2.0, 3.0]);
        assert_eq!(result[1], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_empty_pointcloud() {
        let msg = make_test_pointcloud(&[]);
        let result = from_pointcloud2(&msg).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_distance() {
        let points = vec![
            [1.0, 0.0, 0.0],  // distance = 1.0
            [5.0, 0.0, 0.0],  // distance = 5.0
            [10.0, 0.0, 0.0], // distance = 10.0
            [15.0, 0.0, 0.0], // distance = 15.0
        ];

        let params = PointFilterParams {
            min_distance: 3.0,
            max_distance: 12.0,
            ..Default::default()
        };

        let result = filter_sensor_points(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.points[0], [5.0, 0.0, 0.0]);
        assert_eq!(result.points[1], [10.0, 0.0, 0.0]);
    }

    #[test]
    fn test_filter_z_height() {
        let points = vec![
            [1.0, 0.0, -2.0], // below min_z
            [1.0, 0.0, 0.5],  // within range
            [1.0, 0.0, 1.5],  // within range
            [1.0, 0.0, 5.0],  // above max_z
        ];

        let params = PointFilterParams {
            min_z: -1.0,
            max_z: 3.0,
            ..Default::default()
        };

        let result = filter_sensor_points(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_z, 2);
    }

    #[test]
    fn test_filter_with_downsampling() {
        let points = vec![
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.0, 0.0, 0.0], // different voxel
        ];

        let params = PointFilterParams {
            downsample_resolution: Some(1.0),
            ..Default::default()
        };

        let result = filter_sensor_points(&points, &params);
        // First two points merge into one voxel, third stays separate
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_downsampling, 1);
    }

    #[test]
    fn test_filter_default_params() {
        let points = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let params = PointFilterParams::default();

        let result = filter_sensor_points(&points, &params);
        assert_eq!(result.points.len(), 2);
        assert_eq!(result.removed_by_distance, 0);
        assert_eq!(result.removed_by_z, 0);
        assert_eq!(result.removed_by_downsampling, 0);
    }
}
