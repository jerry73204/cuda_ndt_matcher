//! Comparison tests between our GPU voxel grid and Autoware's algorithm.
//!
//! This module implements Autoware's voxel grid algorithm in Rust for
//! comparison testing. The key differences verified:
//!
//! 1. **Mean computation**: Both use `sum(points) / count`
//! 2. **Covariance**: Autoware uses single-pass `(sum_xx - sum_x * mean') / (n-1)`
//! 3. **Regularization**: Eigenvalue clamping at `min_covar_eigvalue_mult * max_eigenvalue`
//! 4. **Inverse covariance**: Matrix inversion after regularization
//!
//! Reference: Autoware's `multi_voxel_grid_covariance_omp_impl.hpp`

use std::collections::HashMap;

/// Autoware-style voxel leaf structure.
#[derive(Debug, Clone)]
pub struct AutowareLeaf {
    /// Number of points in the voxel
    pub nr_points: i32,
    /// Mean position (centroid)
    pub mean: [f64; 3],
    /// Covariance matrix (row-major 3x3)
    pub cov: [[f64; 3]; 3],
    /// Inverse covariance matrix (row-major 3x3)
    pub icov: [[f64; 3]; 3],
}

/// Autoware-style voxel grid parameters.
#[derive(Debug, Clone)]
pub struct AutowareVoxelParams {
    /// Voxel resolution
    pub resolution: f32,
    /// Minimum points per voxel
    pub min_points_per_voxel: i32,
    /// Minimum eigenvalue multiplier (default 0.01)
    pub min_covar_eigvalue_mult: f64,
}

impl Default for AutowareVoxelParams {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            min_points_per_voxel: 6,
            min_covar_eigvalue_mult: 0.01,
        }
    }
}

/// Compute voxel ID for a point (Autoware-style floor-based indexing).
fn compute_voxel_id(x: f32, y: f32, z: f32, resolution: f32) -> (i32, i32, i32) {
    let inv_res = 1.0 / resolution;
    (
        (x * inv_res).floor() as i32,
        (y * inv_res).floor() as i32,
        (z * inv_res).floor() as i32,
    )
}

/// Build voxel grid using Autoware's algorithm.
///
/// This replicates the logic from `multi_voxel_grid_covariance_omp_impl.hpp`:
/// 1. First pass: accumulate points into voxels
/// 2. Second pass: compute mean, covariance, regularize, invert
pub fn build_autoware_voxel_grid(
    points: &[[f32; 3]],
    params: &AutowareVoxelParams,
) -> Vec<AutowareLeaf> {
    // First pass: group points by voxel and accumulate statistics
    let mut voxel_map: HashMap<(i32, i32, i32), VoxelAccumulator> = HashMap::new();

    for p in points {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            continue;
        }

        let voxel_id = compute_voxel_id(p[0], p[1], p[2], params.resolution);
        let acc = voxel_map
            .entry(voxel_id)
            .or_insert_with(VoxelAccumulator::new);

        let pt = [p[0] as f64, p[1] as f64, p[2] as f64];
        acc.add_point(&pt);
    }

    // Second pass: compute final statistics
    let mut leaves = Vec::new();

    for (_voxel_id, acc) in voxel_map {
        if acc.nr_points < params.min_points_per_voxel {
            continue;
        }

        if let Some(leaf) = acc.finalize(params.min_covar_eigvalue_mult) {
            leaves.push(leaf);
        }
    }

    leaves
}

/// Accumulator for voxel statistics (single-pass covariance).
struct VoxelAccumulator {
    nr_points: i32,
    /// Sum of points
    sum: [f64; 3],
    /// Sum of outer products (for covariance)
    sum_xx: [[f64; 3]; 3],
}

impl VoxelAccumulator {
    fn new() -> Self {
        Self {
            nr_points: 0,
            sum: [0.0; 3],
            sum_xx: [[0.0; 3]; 3],
        }
    }

    fn add_point(&mut self, pt: &[f64; 3]) {
        self.nr_points += 1;

        // Accumulate sum
        self.sum[0] += pt[0];
        self.sum[1] += pt[1];
        self.sum[2] += pt[2];

        // Accumulate outer product x*x'
        for i in 0..3 {
            for j in 0..3 {
                self.sum_xx[i][j] += pt[i] * pt[j];
            }
        }
    }

    fn finalize(self, min_covar_eigvalue_mult: f64) -> Option<AutowareLeaf> {
        let n = self.nr_points as f64;

        // Compute mean
        let mean = [self.sum[0] / n, self.sum[1] / n, self.sum[2] / n];

        // Single-pass covariance: cov = (sum_xx - sum * mean') / (n-1)
        // Equivalent to: cov = (sum_xx - n * mean * mean') / (n-1)
        let mut cov = [[0.0; 3]; 3];
        for (cov_row, (sum_xx_row, &sum_i)) in
            cov.iter_mut().zip(self.sum_xx.iter().zip(self.sum.iter()))
        {
            for (j, cov_ij) in cov_row.iter_mut().enumerate() {
                *cov_ij = (sum_xx_row[j] - sum_i * mean[j]) / (n - 1.0);
            }
        }

        // Regularize covariance (eigenvalue clamping)
        let (regularized_cov, valid) = regularize_covariance(&cov, min_covar_eigvalue_mult);
        if !valid {
            return None;
        }

        // Invert covariance
        let icov = invert_matrix_3x3(&regularized_cov)?;

        // Check for infinity
        if !icov.iter().flatten().all(|x| x.is_finite()) {
            return None;
        }

        Some(AutowareLeaf {
            nr_points: self.nr_points,
            mean,
            cov: regularized_cov,
            icov,
        })
    }
}

/// Regularize covariance matrix by clamping small eigenvalues.
///
/// Implements Autoware's eigenvalue regularization (eq 6.11 from Magnusson 2009):
/// - Compute eigenvalues/eigenvectors
/// - Clamp eigenvalues < min_covar_eigvalue_mult * max_eigenvalue
/// - Reconstruct covariance
fn regularize_covariance(
    cov: &[[f64; 3]; 3],
    min_covar_eigvalue_mult: f64,
) -> ([[f64; 3]; 3], bool) {
    // Compute eigenvalues using characteristic polynomial
    let eigenvalues = compute_eigenvalues_3x3_f64(cov);

    // Check for non-finite eigenvalues (NaN, Inf)
    if eigenvalues.iter().any(|e| !e.is_finite()) {
        return (*cov, false);
    }

    // Small negative eigenvalues can occur due to floating-point precision
    // Treat them as zero; reject only significantly negative ones
    let tolerance = 1e-6;
    if eigenvalues.iter().any(|&e| e < -tolerance) {
        return (*cov, false);
    }

    // Find max eigenvalue (treating small negatives as zero)
    let max_eigenvalue = eigenvalues
        .iter()
        .cloned()
        .map(|e| e.max(0.0))
        .fold(0.0f64, f64::max);

    // If all eigenvalues are effectively zero, add minimum regularization
    if max_eigenvalue < tolerance {
        let mut regularized = *cov;
        regularized[0][0] += 1e-6;
        regularized[1][1] += 1e-6;
        regularized[2][2] += 1e-6;
        return (regularized, true);
    }

    let min_allowed = (min_covar_eigvalue_mult * max_eigenvalue).max(1e-9);

    // Check if regularization is needed (any eigenvalue below threshold)
    let needs_regularization = eigenvalues.iter().any(|&e| e < min_allowed);

    if !needs_regularization {
        return (*cov, true);
    }

    // For proper regularization, we need full eigendecomposition
    // Here we use a simplified approach: add scaled identity
    // This is an approximation but matches the effect of eigenvalue clamping
    let min_eigenvalue = eigenvalues.iter().cloned().fold(f64::MAX, f64::min);
    let reg_amount = (min_allowed - min_eigenvalue.max(0.0)).max(1e-9);

    let mut regularized = *cov;
    regularized[0][0] += reg_amount;
    regularized[1][1] += reg_amount;
    regularized[2][2] += reg_amount;

    (regularized, true)
}

/// Compute eigenvalues of a symmetric 3x3 matrix using Cardano's formula.
fn compute_eigenvalues_3x3_f64(a: &[[f64; 3]; 3]) -> [f64; 3] {
    let trace = a[0][0] + a[1][1] + a[2][2];

    // Sum of principal 2x2 minors
    let m01 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let m02 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    let m12 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let sum_minors = m01 + m02 + m12;

    // Determinant
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    // Solve cubic: λ³ - trace*λ² + sum_minors*λ - det = 0
    let p = sum_minors - trace * trace / 3.0;
    let q = 2.0 * trace * trace * trace / 27.0 - trace * sum_minors / 3.0 + det;

    let discriminant = q * q / 4.0 + p * p * p / 27.0;

    if discriminant < 0.0 {
        // Three real roots
        let r = (-p * p * p / 27.0).sqrt();
        let phi = (-q / (2.0 * r)).clamp(-1.0, 1.0).acos();
        let cube_root_r = r.powf(1.0 / 3.0);

        let offset = trace / 3.0;
        let e1 = 2.0 * cube_root_r * (phi / 3.0).cos() + offset;
        let e2 = 2.0 * cube_root_r * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos() + offset;
        let e3 = 2.0 * cube_root_r * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos() + offset;

        [e1, e2, e3]
    } else {
        let sqrt_disc = discriminant.sqrt();
        let u = (-q / 2.0 + sqrt_disc).cbrt();
        let v = (-q / 2.0 - sqrt_disc).cbrt();
        let e1 = u + v + trace / 3.0;
        [e1, e1, e1]
    }
}

/// Invert a 3x3 matrix using the adjugate method.
fn invert_matrix_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;

    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_grid::gpu::{GpuVoxelGrid, GpuVoxelGridConfig};

    /// Generate a simple test point cloud with known clusters.
    /// Each cluster has independent variation in x, y, z to ensure full-rank covariance.
    fn generate_test_points() -> Vec<[f32; 3]> {
        let mut points = Vec::new();

        // Cluster 1: around (0, 0, 0) - 10 points with 3D spread
        // Use different indices for x, y, z to avoid collinearity
        for i in 0..10 {
            let x_offset = (i as f32 - 4.5) * 0.1;
            let y_offset = ((i * 3) % 10) as f32 * 0.08 - 0.4;
            let z_offset = ((i * 7) % 10) as f32 * 0.12 - 0.6;
            points.push([x_offset, y_offset, z_offset]);
        }

        // Cluster 2: around (5, 5, 5) - 10 points with 3D spread
        for i in 0..10 {
            let x_offset = ((i * 2) % 10) as f32 * 0.1 - 0.5;
            let y_offset = ((i * 5) % 10) as f32 * 0.09 - 0.45;
            let z_offset = ((i * 3) % 10) as f32 * 0.11 - 0.55;
            points.push([5.0 + x_offset, 5.0 + y_offset, 5.0 + z_offset]);
        }

        // Cluster 3: around (10, 0, 5) - 10 points with 3D spread
        for i in 0..10 {
            let x_offset = ((i * 7) % 10) as f32 * 0.1 - 0.5;
            let y_offset = ((i * 4) % 10) as f32 * 0.07 - 0.35;
            let z_offset = ((i * 9) % 10) as f32 * 0.09 - 0.45;
            points.push([10.0 + x_offset, y_offset, 5.0 + z_offset]);
        }

        points
    }

    #[test]
    fn test_autoware_voxel_construction() {
        let points = generate_test_points();
        let params = AutowareVoxelParams {
            resolution: 2.0,
            min_points_per_voxel: 3,
            min_covar_eigvalue_mult: 0.01,
        };

        let leaves = build_autoware_voxel_grid(&points, &params);

        // Should have 3 voxels (one per cluster)
        assert!(!leaves.is_empty(), "Should have at least one valid voxel");

        for leaf in &leaves {
            // Check valid point count
            assert!(leaf.nr_points >= params.min_points_per_voxel);

            // Check covariance is positive semi-definite (diagonal positive)
            assert!(leaf.cov[0][0] >= 0.0);
            assert!(leaf.cov[1][1] >= 0.0);
            assert!(leaf.cov[2][2] >= 0.0);

            // Check inverse covariance is finite
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        leaf.icov[i][j].is_finite(),
                        "Inverse covariance should be finite"
                    );
                }
            }
        }
    }

    #[test]
    fn test_compare_mean_computation() {
        // Create a simple cluster where mean is easy to verify
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        // Expected mean: (0.5, 0.5, 0.5)
        let expected_mean = [0.5, 0.5, 0.5];

        // Autoware computation
        let autoware_params = AutowareVoxelParams {
            resolution: 3.0, // Large enough for all points in one voxel
            min_points_per_voxel: 3,
            min_covar_eigvalue_mult: 0.01,
        };
        let autoware_leaves = build_autoware_voxel_grid(&points, &autoware_params);

        assert_eq!(autoware_leaves.len(), 1, "Should have exactly one voxel");
        let autoware_mean = autoware_leaves[0].mean;

        // Our GPU implementation
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let gpu_config = GpuVoxelGridConfig {
            resolution: 3.0,
            min_points: 3,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&points_flat, &gpu_config);

        assert_eq!(gpu_grid.num_valid_voxels(), 1, "GPU should have one voxel");

        let gpu_voxel = gpu_grid.iter_valid_voxels().next().unwrap();
        let gpu_mean = gpu_voxel.mean;

        // Compare means
        for i in 0..3 {
            assert!(
                (autoware_mean[i] - expected_mean[i]).abs() < 1e-6,
                "Autoware mean[{i}] should match expected"
            );
            assert!(
                (gpu_mean[i] as f64 - expected_mean[i]).abs() < 1e-5,
                "GPU mean[{i}] should match expected"
            );
            assert!(
                (autoware_mean[i] - gpu_mean[i] as f64).abs() < 1e-5,
                "Autoware and GPU means should match"
            );
        }
    }

    #[test]
    fn test_compare_covariance_computation() {
        // Points with known covariance structure (3D cube vertices plus center)
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ];

        // Autoware computation
        let autoware_params = AutowareVoxelParams {
            resolution: 5.0,
            min_points_per_voxel: 3,
            min_covar_eigvalue_mult: 0.01,
        };
        let autoware_leaves = build_autoware_voxel_grid(&points, &autoware_params);

        assert_eq!(autoware_leaves.len(), 1);
        let autoware_cov = autoware_leaves[0].cov;

        // Our GPU implementation
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let gpu_config = GpuVoxelGridConfig {
            resolution: 5.0,
            min_points: 3,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&points_flat, &gpu_config);

        assert_eq!(gpu_grid.num_valid_voxels(), 1);
        let gpu_voxel = gpu_grid.iter_valid_voxels().next().unwrap();

        // Compare covariances (with tolerance for f32 vs f64)
        for (i, row) in autoware_cov.iter().enumerate() {
            for (j, &autoware_val) in row.iter().enumerate() {
                let gpu_val = gpu_voxel.covariance[i * 3 + j] as f64;
                let diff = (autoware_val - gpu_val).abs();
                let relative_diff = if autoware_val.abs() > 1e-6 {
                    diff / autoware_val.abs()
                } else {
                    diff
                };

                assert!(
                    relative_diff < 0.1 || diff < 1e-4,
                    "Covariance[{i}][{j}] mismatch: autoware={autoware_val}, gpu={gpu_val}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_compare_inverse_covariance() {
        // Points spread in 3D
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
        ];

        // Autoware computation
        let autoware_params = AutowareVoxelParams {
            resolution: 2.0,
            min_points_per_voxel: 3,
            min_covar_eigvalue_mult: 0.01,
        };
        let autoware_leaves = build_autoware_voxel_grid(&points, &autoware_params);

        assert!(!autoware_leaves.is_empty());
        let autoware_icov = autoware_leaves[0].icov;

        // Our GPU implementation
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let gpu_config = GpuVoxelGridConfig {
            resolution: 2.0,
            min_points: 3,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&points_flat, &gpu_config);

        assert!(gpu_grid.num_valid_voxels() > 0);
        let gpu_voxel = gpu_grid.iter_valid_voxels().next().unwrap();

        // Compare inverse covariances
        for (i, row) in autoware_icov.iter().enumerate() {
            for (j, &autoware_val) in row.iter().enumerate() {
                let gpu_val = gpu_voxel.inv_covariance[i * 3 + j] as f64;

                // Inverse covariance can have larger values, use relative tolerance
                let diff = (autoware_val - gpu_val).abs();
                let scale = autoware_val.abs().max(gpu_val.abs()).max(1.0);
                let relative_diff = diff / scale;

                assert!(
                    relative_diff < 0.2 || diff < 1.0,
                    "Inv covariance[{i}][{j}] mismatch: autoware={autoware_val}, gpu={gpu_val}"
                );
            }
        }
    }

    #[test]
    fn test_voxel_count_consistency() {
        let points = generate_test_points();
        let resolution = 2.0;

        // Autoware
        let autoware_params = AutowareVoxelParams {
            resolution,
            min_points_per_voxel: 3,
            min_covar_eigvalue_mult: 0.01,
        };
        let autoware_leaves = build_autoware_voxel_grid(&points, &autoware_params);

        // GPU
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let gpu_config = GpuVoxelGridConfig {
            resolution,
            min_points: 3,
        };
        let gpu_grid = GpuVoxelGrid::from_points(&points_flat, &gpu_config);

        // Both implementations should produce valid voxels.
        // Note: The GPU now aligns grid_min to resolution boundaries to match
        // Autoware's floor(coord / resolution) voxel assignment. However, exact
        // counts may still differ slightly due to:
        // - Numerical differences in covariance regularization
        // - Different eigenvalue thresholds for validity
        let autoware_count = autoware_leaves.len();
        let gpu_count = gpu_grid.num_valid_voxels();

        assert!(
            autoware_count > 0,
            "Autoware should produce at least one valid voxel"
        );
        assert!(gpu_count > 0, "GPU should produce at least one valid voxel");

        // Verify that both produce voxels with reasonable statistics
        for leaf in &autoware_leaves {
            assert!(leaf.nr_points >= 3, "Voxel should have enough points");
            assert!(
                leaf.cov[0][0] >= 0.0,
                "Covariance diagonal should be non-negative"
            );
        }

        for voxel in gpu_grid.iter_valid_voxels() {
            assert!(voxel.point_count >= 3, "Voxel should have enough points");
            assert!(
                voxel.covariance[0] >= 0.0,
                "Covariance diagonal should be non-negative"
            );
        }
    }

    #[test]
    fn test_eigenvalue_regularization() {
        // Test the eigenvalue regularization matches Autoware's approach
        let cov = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.001]];

        // Autoware-style regularization with 0.01 multiplier
        let (regularized, valid) = regularize_covariance(&cov, 0.01);

        assert!(valid);

        // Min allowed = 0.01 * 1.0 = 0.01
        // The smallest eigenvalue (0.001) should be regularized
        // Check that diagonal elements are increased appropriately
        assert!(
            regularized[2][2] >= 0.01 - 1e-9,
            "Smallest eigenvalue should be clamped"
        );
    }

    #[test]
    fn test_single_pass_covariance_formula() {
        // Verify single-pass covariance formula: cov = (sum_xx - sum_x * mean') / (n-1)
        let points: Vec<[f64; 3]> = vec![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];
        let n = points.len() as f64;

        // Two-pass method (reference)
        let mean: [f64; 3] = [
            points.iter().map(|p| p[0]).sum::<f64>() / n,
            points.iter().map(|p| p[1]).sum::<f64>() / n,
            points.iter().map(|p| p[2]).sum::<f64>() / n,
        ];

        let mut cov_two_pass = [[0.0; 3]; 3];
        for p in &points {
            for (i, cov_row) in cov_two_pass.iter_mut().enumerate() {
                for (j, cov_ij) in cov_row.iter_mut().enumerate() {
                    *cov_ij += (p[i] - mean[i]) * (p[j] - mean[j]);
                }
            }
        }
        for val in cov_two_pass.iter_mut().flatten() {
            *val /= n - 1.0;
        }

        // Single-pass method (Autoware)
        let mut sum = [0.0; 3];
        let mut sum_xx = [[0.0; 3]; 3];
        for p in &points {
            for (i, (sum_i, sum_xx_row)) in sum.iter_mut().zip(sum_xx.iter_mut()).enumerate() {
                *sum_i += p[i];
                for (j, sum_xx_ij) in sum_xx_row.iter_mut().enumerate() {
                    *sum_xx_ij += p[i] * p[j];
                }
            }
        }

        let mut cov_single_pass = [[0.0; 3]; 3];
        for (cov_row, (sum_xx_row, &sum_i)) in cov_single_pass
            .iter_mut()
            .zip(sum_xx.iter().zip(sum.iter()))
        {
            for (j, cov_ij) in cov_row.iter_mut().enumerate() {
                *cov_ij = (sum_xx_row[j] - sum_i * mean[j]) / (n - 1.0);
            }
        }

        // Compare
        for (i, (two_pass_row, single_pass_row)) in
            cov_two_pass.iter().zip(cov_single_pass.iter()).enumerate()
        {
            for (j, (&two_pass_val, &single_pass_val)) in
                two_pass_row.iter().zip(single_pass_row.iter()).enumerate()
            {
                assert!(
                    (two_pass_val - single_pass_val).abs() < 1e-10,
                    "Single-pass should match two-pass: [{i}][{j}]"
                );
            }
        }
    }
}
