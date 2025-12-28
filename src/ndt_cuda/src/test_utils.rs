//! Test utilities matching Autoware's test_util.hpp.
//!
//! Provides synthetic point cloud generators and pose utilities for testing
//! NDT algorithms with known ground truth.

/// Generate a half-cubic point cloud matching Autoware's `make_sample_half_cubic_pcd()`.
///
/// Creates 3 orthogonal planes (XY, YZ, ZX) forming a half-cube:
/// - XY plane: z=0, x∈[0,20], y∈[0,20]
/// - YZ plane: x=0, y∈[0,20], z∈[0,20]
/// - ZX plane: y=0, x∈[0,20], z∈[0,20]
///
/// # Arguments
/// * `length` - Side length of each plane (default 20.0m)
/// * `interval` - Grid spacing between points (default 0.2m)
///
/// # Returns
/// Vector of 3D points, approximately `3 * (length/interval + 1)²` points.
pub fn make_half_cubic_pcd(length: f32, interval: f32) -> Vec<[f32; 3]> {
    let num_points_per_line = ((length / interval) as usize) + 1;
    let num_points_per_plane = num_points_per_line * num_points_per_line;
    let total_points = 3 * num_points_per_plane;

    let mut points = Vec::with_capacity(total_points);

    for i in 0..num_points_per_line {
        for j in 0..num_points_per_line {
            let u = interval * (j as f32);
            let v = interval * (i as f32);

            // XY plane (z=0)
            points.push([u, v, 0.0]);

            // YZ plane (x=0)
            points.push([0.0, u, v]);

            // ZX plane (y=0)
            points.push([u, 0.0, v]);
        }
    }

    points
}

/// Generate half-cubic point cloud with default Autoware parameters.
///
/// - Length: 20m
/// - Interval: 0.2m
/// - Total: ~30,603 points (3 × 101²)
pub fn make_default_half_cubic_pcd() -> Vec<[f32; 3]> {
    make_half_cubic_pcd(20.0, 0.2)
}

/// Generate half-cubic point cloud with offset.
///
/// Useful for simulating a map at a different location.
pub fn make_half_cubic_pcd_offset(offset_x: f32, offset_y: f32, offset_z: f32) -> Vec<[f32; 3]> {
    make_default_half_cubic_pcd()
        .into_iter()
        .map(|p| [p[0] + offset_x, p[1] + offset_y, p[2] + offset_z])
        .collect()
}

/// Voxelize a point cloud using simple grid-based downsampling.
///
/// This is a simplified version of PCL's VoxelGrid filter.
/// For each voxel, keeps the centroid of all points within it.
///
/// # Arguments
/// * `points` - Input point cloud
/// * `leaf_size` - Voxel size in all dimensions
///
/// # Returns
/// Downsampled point cloud with one point per occupied voxel.
pub fn voxelize_pcd(points: &[[f32; 3]], leaf_size: f32) -> Vec<[f32; 3]> {
    use std::collections::HashMap;

    let inv_leaf = 1.0 / leaf_size;

    // Group points by voxel coordinate
    let mut voxel_map: HashMap<(i32, i32, i32), (f64, f64, f64, usize)> = HashMap::new();

    for p in points {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }

        let vx = (p[0] * inv_leaf).floor() as i32;
        let vy = (p[1] * inv_leaf).floor() as i32;
        let vz = (p[2] * inv_leaf).floor() as i32;

        let entry = voxel_map.entry((vx, vy, vz)).or_insert((0.0, 0.0, 0.0, 0));
        entry.0 += p[0] as f64;
        entry.1 += p[1] as f64;
        entry.2 += p[2] as f64;
        entry.3 += 1;
    }

    // Compute centroids
    voxel_map
        .values()
        .map(|(sum_x, sum_y, sum_z, count)| {
            let n = *count as f64;
            [(sum_x / n) as f32, (sum_y / n) as f32, (sum_z / n) as f32]
        })
        .collect()
}

/// Generate a sensor point cloud matching Autoware's `make_default_sensor_pcd()`.
///
/// Creates a voxelized half-cube with 1.0m leaf size.
pub fn make_default_sensor_pcd() -> Vec<[f32; 3]> {
    let cloud = make_default_half_cubic_pcd();
    voxelize_pcd(&cloud, 1.0)
}

/// Generate a simple XY plane for testing planar point distributions.
///
/// # Arguments
/// * `size` - Side length of the square plane
/// * `interval` - Grid spacing between points
/// * `z` - Z coordinate of the plane
pub fn make_xy_plane(size: f32, interval: f32, z: f32) -> Vec<[f32; 3]> {
    let num_points = ((size / interval) as usize) + 1;
    let mut points = Vec::with_capacity(num_points * num_points);

    for i in 0..num_points {
        for j in 0..num_points {
            let x = interval * (j as f32);
            let y = interval * (i as f32);
            points.push([x, y, z]);
        }
    }

    points
}

/// Generate a 3D grid of points (cube).
///
/// # Arguments
/// * `size` - Side length of the cube
/// * `interval` - Grid spacing between points
/// * `offset` - Offset to apply to all points
pub fn make_cube_grid(size: f32, interval: f32, offset: [f32; 3]) -> Vec<[f32; 3]> {
    let num_points = ((size / interval) as usize) + 1;
    let mut points = Vec::with_capacity(num_points * num_points * num_points);

    for i in 0..num_points {
        for j in 0..num_points {
            for k in 0..num_points {
                let x = offset[0] + interval * (k as f32);
                let y = offset[1] + interval * (j as f32);
                let z = offset[2] + interval * (i as f32);
                points.push([x, y, z]);
            }
        }
    }

    points
}

/// Generate random points within a sphere.
///
/// # Arguments
/// * `center` - Center of the sphere
/// * `radius` - Radius of the sphere
/// * `num_points` - Number of points to generate
/// * `seed` - Random seed for reproducibility
pub fn make_random_sphere(
    center: [f32; 3],
    radius: f32,
    num_points: usize,
    seed: u64,
) -> Vec<[f32; 3]> {
    // Simple LCG for reproducibility
    let mut rng_state = seed;
    let mut next_random = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    let mut points = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        // Uniform distribution in sphere using rejection sampling
        loop {
            let u = next_random() * 2.0 - 1.0;
            let v = next_random() * 2.0 - 1.0;
            let w = next_random() * 2.0 - 1.0;

            let r2 = u * u + v * v + w * w;
            if r2 <= 1.0 && r2 > 0.0 {
                let r = next_random().powf(1.0 / 3.0) * radius;
                let scale = r / r2.sqrt();
                points.push([
                    center[0] + u * scale,
                    center[1] + v * scale,
                    center[2] + w * scale,
                ]);
                break;
            }
        }
    }

    points
}

/// Pose representation for testing (position + orientation).
#[derive(Debug, Clone)]
pub struct TestPose {
    /// Translation (x, y, z)
    pub translation: [f64; 3],
    /// Quaternion (w, x, y, z)
    pub quaternion: [f64; 4],
    /// 6x6 covariance matrix (row-major)
    pub covariance: [f64; 36],
}

impl TestPose {
    /// Create a pose at (x, y, 0) with identity orientation.
    ///
    /// Uses Autoware's default covariance values from test_util.hpp.
    pub fn from_xy(x: f64, y: f64) -> Self {
        let mut covariance = [0.0; 36];
        // Diagonal elements matching Autoware's make_pose()
        covariance[0] = 0.25; // x variance
        covariance[7] = 0.25; // y variance
        covariance[14] = 0.0025; // z variance
        covariance[21] = 0.0006853891909122467; // roll variance
        covariance[28] = 0.0006853891909122467; // pitch variance
        covariance[35] = 0.06853891909122467; // yaw variance

        Self {
            translation: [x, y, 0.0],
            quaternion: [1.0, 0.0, 0.0, 0.0], // Identity quaternion (w, x, y, z)
            covariance,
        }
    }

    /// Create a pose with full position.
    pub fn from_xyz(x: f64, y: f64, z: f64) -> Self {
        let mut pose = Self::from_xy(x, y);
        pose.translation[2] = z;
        pose
    }

    /// Convert to 4x4 transformation matrix (row-major).
    pub fn to_matrix(&self) -> [[f64; 4]; 4] {
        let [w, x, y, z] = self.quaternion;
        let [tx, ty, tz] = self.translation;

        // Rotation matrix from quaternion
        let r00 = 1.0 - 2.0 * (y * y + z * z);
        let r01 = 2.0 * (x * y - z * w);
        let r02 = 2.0 * (x * z + y * w);

        let r10 = 2.0 * (x * y + z * w);
        let r11 = 1.0 - 2.0 * (x * x + z * z);
        let r12 = 2.0 * (y * z - x * w);

        let r20 = 2.0 * (x * z - y * w);
        let r21 = 2.0 * (y * z + x * w);
        let r22 = 1.0 - 2.0 * (x * x + y * y);

        [
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

/// Transform a point cloud by a pose.
pub fn transform_points(points: &[[f32; 3]], pose: &TestPose) -> Vec<[f32; 3]> {
    let mat = pose.to_matrix();

    points
        .iter()
        .map(|p| {
            let x = p[0] as f64;
            let y = p[1] as f64;
            let z = p[2] as f64;

            let nx = mat[0][0] * x + mat[0][1] * y + mat[0][2] * z + mat[0][3];
            let ny = mat[1][0] * x + mat[1][1] * y + mat[1][2] * z + mat[1][3];
            let nz = mat[2][0] * x + mat[2][1] * y + mat[2][2] * z + mat[2][3];

            [nx as f32, ny as f32, nz as f32]
        })
        .collect()
}

/// Compute bounds of a point cloud.
pub fn compute_bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for p in points {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_cubic_pcd_point_count() {
        let pcd = make_default_half_cubic_pcd();
        // 3 planes × 101² points each = 30,603
        assert_eq!(pcd.len(), 3 * 101 * 101);
    }

    #[test]
    fn test_half_cubic_pcd_bounds() {
        let pcd = make_default_half_cubic_pcd();
        let (min, max) = compute_bounds(&pcd);

        // All points should be within [0, 20] × [0, 20] × [0, 20]
        for i in 0..3 {
            assert!(min[i] >= 0.0, "min[{i}] = {} should be >= 0", min[i]);
            assert!(max[i] <= 20.0, "max[{i}] = {} should be <= 20", max[i]);
        }
    }

    #[test]
    fn test_half_cubic_pcd_planes() {
        let pcd = make_default_half_cubic_pcd();
        let n = 101; // points per line

        // Count points strictly on each plane (excluding edges shared with other planes)
        // XY plane: z=0, but exclude x=0 (shared with YZ) and y=0 (shared with ZX)
        let xy_strict = pcd
            .iter()
            .filter(|p| p[2].abs() < 0.01 && p[0].abs() > 0.01 && p[1].abs() > 0.01)
            .count();
        // YZ plane: x=0, but exclude y=0 (shared with ZX) and z=0 (shared with XY)
        let yz_strict = pcd
            .iter()
            .filter(|p| p[0].abs() < 0.01 && p[1].abs() > 0.01 && p[2].abs() > 0.01)
            .count();
        // ZX plane: y=0, but exclude x=0 (shared with YZ) and z=0 (shared with XY)
        let zx_strict = pcd
            .iter()
            .filter(|p| p[1].abs() < 0.01 && p[0].abs() > 0.01 && p[2].abs() > 0.01)
            .count();

        // Each plane's interior (excluding 2 edges) = (n-1) × (n-1) = 100 × 100 = 10,000
        let expected_interior = (n - 1) * (n - 1);
        assert_eq!(
            xy_strict, expected_interior,
            "XY plane interior point count"
        );
        assert_eq!(
            yz_strict, expected_interior,
            "YZ plane interior point count"
        );
        assert_eq!(
            zx_strict, expected_interior,
            "ZX plane interior point count"
        );

        // Total should be 3 × n² (each plane generates n² points)
        assert_eq!(pcd.len(), 3 * n * n);
    }

    #[test]
    fn test_half_cubic_pcd_offset() {
        let pcd = make_half_cubic_pcd_offset(100.0, 100.0, 0.0);
        let (min, max) = compute_bounds(&pcd);

        assert!(min[0] >= 100.0);
        assert!(min[1] >= 100.0);
        assert!(max[0] <= 120.0);
        assert!(max[1] <= 120.0);
    }

    #[test]
    fn test_voxelize_reduces_points() {
        let pcd = make_default_half_cubic_pcd();
        let voxelized = voxelize_pcd(&pcd, 1.0);

        // Original: ~30,000 points at 0.2m spacing
        // Voxelized at 1.0m: should be ~3,000 points (5x reduction per dimension)
        assert!(voxelized.len() < pcd.len() / 10);
        assert!(voxelized.len() > 100); // Should still have many points
    }

    #[test]
    fn test_voxelize_preserves_bounds() {
        let pcd = make_default_half_cubic_pcd();
        let voxelized = voxelize_pcd(&pcd, 1.0);

        let (orig_min, orig_max) = compute_bounds(&pcd);
        let (vox_min, vox_max) = compute_bounds(&voxelized);

        // Bounds should be similar (within half a voxel)
        for i in 0..3 {
            assert!((orig_min[i] - vox_min[i]).abs() < 0.6);
            assert!((orig_max[i] - vox_max[i]).abs() < 0.6);
        }
    }

    #[test]
    fn test_default_sensor_pcd() {
        let pcd = make_default_sensor_pcd();

        // Should be voxelized version of half-cube
        assert!(pcd.len() > 100);
        assert!(pcd.len() < 5000);
    }

    #[test]
    fn test_xy_plane() {
        let plane = make_xy_plane(10.0, 0.5, 5.0);

        // 21 × 21 = 441 points
        assert_eq!(plane.len(), 21 * 21);

        // All points should be at z=5
        for p in &plane {
            assert!((p[2] - 5.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_cube_grid() {
        let cube = make_cube_grid(2.0, 1.0, [0.0, 0.0, 0.0]);

        // 3³ = 27 points
        assert_eq!(cube.len(), 27);
    }

    #[test]
    fn test_random_sphere() {
        let sphere = make_random_sphere([0.0, 0.0, 0.0], 1.0, 1000, 42);

        assert_eq!(sphere.len(), 1000);

        // All points should be within radius
        for p in &sphere {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(r <= 1.0 + 0.001, "Point at radius {} exceeds 1.0", r);
        }
    }

    #[test]
    fn test_random_sphere_reproducible() {
        let sphere1 = make_random_sphere([0.0, 0.0, 0.0], 1.0, 100, 42);
        let sphere2 = make_random_sphere([0.0, 0.0, 0.0], 1.0, 100, 42);

        for (p1, p2) in sphere1.iter().zip(sphere2.iter()) {
            assert_eq!(p1, p2);
        }
    }

    #[test]
    fn test_pose_from_xy() {
        let pose = TestPose::from_xy(100.0, 200.0);

        assert_eq!(pose.translation, [100.0, 200.0, 0.0]);
        assert_eq!(pose.quaternion, [1.0, 0.0, 0.0, 0.0]); // Identity
        assert!(pose.covariance[0] > 0.0); // Has covariance
    }

    #[test]
    fn test_pose_to_matrix_identity() {
        let pose = TestPose::from_xy(0.0, 0.0);
        let mat = pose.to_matrix();

        // Should be identity matrix
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (mat[i][j] - expected).abs() < 1e-10,
                    "mat[{i}][{j}] = {}, expected {}",
                    mat[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_pose_to_matrix_translation() {
        let pose = TestPose::from_xyz(1.0, 2.0, 3.0);
        let mat = pose.to_matrix();

        assert!((mat[0][3] - 1.0).abs() < 1e-10);
        assert!((mat[1][3] - 2.0).abs() < 1e-10);
        assert!((mat[2][3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_points() {
        let points = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pose = TestPose::from_xyz(10.0, 20.0, 30.0);

        let transformed = transform_points(&points, &pose);

        assert!((transformed[0][0] - 11.0).abs() < 0.001);
        assert!((transformed[0][1] - 20.0).abs() < 0.001);
        assert!((transformed[0][2] - 30.0).abs() < 0.001);
    }
}
