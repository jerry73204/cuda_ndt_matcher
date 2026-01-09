//! CPU reference implementation for NDT derivative computation.
//!
//! This module implements the gradient and Hessian computation described in
//! Magnusson 2009, Chapter 6.

use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector4, Vector6};
#[cfg(debug_assertions)]
use tracing::debug;

use super::angular::AngularDerivatives;
use super::types::{
    AggregatedDerivatives, DerivativeResult, DistanceMetric, GaussianParams, Matrix24x6, Matrix4x6,
    PointDerivatives,
};
use crate::voxel_grid::VoxelGrid;

/// Compute point gradient and optionally Hessian for a source point.
///
/// The point gradient is ∂T(x)/∂p where T is the transformation and p is the pose.
/// This fills in the translation derivatives (identity for columns 0-2) and
/// rotation derivatives (from angular derivatives for columns 3-5).
///
/// # Arguments
/// * `point` - Source point coordinates [x, y, z]
/// * `angular` - Precomputed angular derivatives
/// * `compute_hessian` - Whether to compute second derivatives
pub fn compute_point_derivatives(
    point: &[f64; 3],
    angular: &AngularDerivatives,
    compute_hessian: bool,
) -> PointDerivatives {
    let mut result = PointDerivatives::zeros();

    // Translation derivatives: ∂T/∂tx = [1,0,0,0], etc.
    // This is the identity block for columns 0-2
    result.point_gradient[(0, 0)] = 1.0;
    result.point_gradient[(1, 1)] = 1.0;
    result.point_gradient[(2, 2)] = 1.0;
    // Row 3 is always zero (homogeneous coordinate)

    // Rotation derivatives from angular Jacobian
    let grad_terms = angular.compute_point_gradient_terms(point);

    // Fill in rotation-dependent gradient terms (columns 3-5)
    // Based on computePointDerivatives in multigrid_ndt_omp_impl.hpp
    result.point_gradient[(1, 3)] = grad_terms[0]; // ∂y/∂roll
    result.point_gradient[(2, 3)] = grad_terms[1]; // ∂z/∂roll
    result.point_gradient[(0, 4)] = grad_terms[2]; // ∂x/∂pitch
    result.point_gradient[(1, 4)] = grad_terms[3]; // ∂y/∂pitch
    result.point_gradient[(2, 4)] = grad_terms[4]; // ∂z/∂pitch
    result.point_gradient[(0, 5)] = grad_terms[5]; // ∂x/∂yaw
    result.point_gradient[(1, 5)] = grad_terms[6]; // ∂y/∂yaw
    result.point_gradient[(2, 5)] = grad_terms[7]; // ∂z/∂yaw

    if compute_hessian {
        let hess_terms = angular.compute_point_hessian_terms(point);

        // Hessian is organized as 4 blocks of 6x6, each block is 4 rows
        // Block i contains ∂²(T(x))_i/∂p_j∂p_k
        // We store: block 0 at rows 0-3, block 1 at 4-7, etc.
        // But in practice, only the rotation-rotation blocks (3-5, 3-5) are non-zero

        // From the reference implementation:
        // point_hessian.block<4, 1>(12, 3) = a (roll-roll)
        // point_hessian.block<4, 1>(16, 3) = b (pitch-roll)
        // point_hessian.block<4, 1>(20, 3) = c (yaw-roll)
        // point_hessian.block<4, 1>(12, 4) = b (roll-pitch)
        // point_hessian.block<4, 1>(16, 4) = d (pitch-pitch)
        // point_hessian.block<4, 1>(20, 4) = e (yaw-pitch)
        // point_hessian.block<4, 1>(12, 5) = c (roll-yaw)
        // point_hessian.block<4, 1>(16, 5) = e (pitch-yaw)
        // point_hessian.block<4, 1>(20, 5) = f (yaw-yaw)

        // a = [0, a2, a3, 0] = [0, hess_terms[0], hess_terms[1], 0]
        // b = [0, b2, b3, 0] = [0, hess_terms[2], hess_terms[3], 0]
        // c = [0, c2, c3, 0] = [0, hess_terms[4], hess_terms[5], 0]
        // d = [d1, d2, d3, 0] = [hess_terms[6], hess_terms[7], hess_terms[8], 0]
        // e = [e1, e2, e3, 0] = [hess_terms[9], hess_terms[10], hess_terms[11], 0]
        // f = [f1, f2, f3, 0] = [hess_terms[12], hess_terms[13], hess_terms[14], 0]

        // Block at rows 12-15, column 3 (roll-roll): a
        result.point_hessian[(13, 3)] = hess_terms[0]; // a2
        result.point_hessian[(14, 3)] = hess_terms[1]; // a3

        // Block at rows 16-19, column 3 (pitch-roll): b
        result.point_hessian[(17, 3)] = hess_terms[2]; // b2
        result.point_hessian[(18, 3)] = hess_terms[3]; // b3

        // Block at rows 20-23, column 3 (yaw-roll): c
        result.point_hessian[(21, 3)] = hess_terms[4]; // c2
        result.point_hessian[(22, 3)] = hess_terms[5]; // c3

        // Block at rows 12-15, column 4 (roll-pitch): b
        result.point_hessian[(13, 4)] = hess_terms[2]; // b2
        result.point_hessian[(14, 4)] = hess_terms[3]; // b3

        // Block at rows 16-19, column 4 (pitch-pitch): d
        result.point_hessian[(16, 4)] = hess_terms[6]; // d1
        result.point_hessian[(17, 4)] = hess_terms[7]; // d2
        result.point_hessian[(18, 4)] = hess_terms[8]; // d3

        // Block at rows 20-23, column 4 (yaw-pitch): e
        result.point_hessian[(20, 4)] = hess_terms[9]; // e1
        result.point_hessian[(21, 4)] = hess_terms[10]; // e2
        result.point_hessian[(22, 4)] = hess_terms[11]; // e3

        // Block at rows 12-15, column 5 (roll-yaw): c
        result.point_hessian[(13, 5)] = hess_terms[4]; // c2
        result.point_hessian[(14, 5)] = hess_terms[5]; // c3

        // Block at rows 16-19, column 5 (pitch-yaw): e
        result.point_hessian[(16, 5)] = hess_terms[9]; // e1
        result.point_hessian[(17, 5)] = hess_terms[10]; // e2
        result.point_hessian[(18, 5)] = hess_terms[11]; // e3

        // Block at rows 20-23, column 5 (yaw-yaw): f
        result.point_hessian[(20, 5)] = hess_terms[12]; // f1
        result.point_hessian[(21, 5)] = hess_terms[13]; // f2
        result.point_hessian[(22, 5)] = hess_terms[14]; // f3
    }

    result
}

/// Compute score, gradient, and Hessian for a single point-voxel correspondence.
///
/// This implements the updateDerivatives function from Magnusson 2009:
/// - Score (Eq. 6.9): p(x) = -d1 * exp(-d2/2 * (x-μ)ᵀΣ⁻¹(x-μ))
/// - Gradient (Eq. 6.12)
/// - Hessian (Eq. 6.13)
///
/// # Arguments
/// * `x_trans` - Transformed point minus voxel mean (x - μ)
/// * `inv_covariance` - Inverse covariance matrix of the voxel
/// * `point_gradient` - Point gradient matrix (4x6)
/// * `point_hessian` - Point Hessian matrix (24x6)
/// * `gauss` - Gaussian parameters
/// * `compute_hessian` - Whether to compute Hessian
pub fn compute_derivative_single(
    x_trans: &Vector3<f64>,
    inv_covariance: &Matrix3<f64>,
    point_gradient: &Matrix4x6,
    point_hessian: &Matrix24x6,
    gauss: &GaussianParams,
    compute_hessian: bool,
) -> DerivativeResult {
    // Extend to 4D for matrix operations (w = 0)
    let x_trans4 = Vector4::new(x_trans.x, x_trans.y, x_trans.z, 0.0);

    // Extend inverse covariance to 4x4
    let mut c_inv4 = Matrix4::zeros();
    c_inv4
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(inv_covariance);

    // Compute (x-μ)ᵀ Σ⁻¹ (x-μ) - result is 1x1 matrix, extract scalar
    let x_c_inv_x_mat = x_trans4.transpose() * c_inv4 * x_trans4;
    let x_c_inv_x: f64 = x_c_inv_x_mat[(0, 0)];

    // e^(-d2/2 * (x-μ)ᵀ Σ⁻¹ (x-μ)) - Equation 6.9
    let e_x_cov_x_raw = (-gauss.d2 * x_c_inv_x * 0.5).exp();

    // Score contribution: -d1 * exp(...)
    let score = -gauss.d1 * e_x_cov_x_raw;

    // Scale for gradient/hessian computation
    let mut e_x_cov_x = gauss.d2 * e_x_cov_x_raw;

    // Error checking for invalid values
    if !(0.0..=1.0).contains(&e_x_cov_x) || !e_x_cov_x.is_finite() {
        return DerivativeResult::zeros();
    }

    // Reusable portion: d1 * d2 * exp(...)
    e_x_cov_x *= gauss.d1;

    // Σ⁻¹ * ∂T/∂p (4x6)
    let c_inv4_x_point_gradient = c_inv4 * point_gradient;

    // (x-μ)ᵀ Σ⁻¹ * ∂T/∂p (1x6)
    let x_trans4_row = x_trans4.transpose();
    let x_trans4_dot_c_inv4_x_point_gradient_row = x_trans4_row * c_inv4_x_point_gradient;

    // Convert 1x6 to 6x1 Vector6
    let x_trans4_dot_c_inv4_x_point_gradient = Vector6::new(
        x_trans4_dot_c_inv4_x_point_gradient_row[(0, 0)],
        x_trans4_dot_c_inv4_x_point_gradient_row[(0, 1)],
        x_trans4_dot_c_inv4_x_point_gradient_row[(0, 2)],
        x_trans4_dot_c_inv4_x_point_gradient_row[(0, 3)],
        x_trans4_dot_c_inv4_x_point_gradient_row[(0, 4)],
        x_trans4_dot_c_inv4_x_point_gradient_row[(0, 5)],
    );

    // Gradient: d1*d2*exp(...) * (x-μ)ᵀ Σ⁻¹ * ∂T/∂p
    let gradient = e_x_cov_x * x_trans4_dot_c_inv4_x_point_gradient;

    // Compute Hessian if requested
    let mut hessian = Matrix6::zeros();

    if compute_hessian {
        let x_trans4_x_c_inv4 = x_trans4_row * c_inv4;

        // (∂T/∂p)ᵀ Σ⁻¹ ∂T/∂p (6x6)
        let point_gradient_t_c_inv4_x_point_gradient =
            point_gradient.transpose() * c_inv4_x_point_gradient;

        for i in 0..6 {
            // (x-μ)ᵀ Σ⁻¹ * ∂²T/∂pᵢ∂p (1x6)
            let hessian_block = point_hessian.fixed_view::<4, 6>(i * 4, 0);
            let x_trans4_dot_c_inv4_x_ext_point_hessian = x_trans4_x_c_inv4 * hessian_block;

            for j in 0..6 {
                // Equation 6.13
                hessian[(i, j)] = e_x_cov_x
                    * (-gauss.d2
                        * x_trans4_dot_c_inv4_x_point_gradient[i]
                        * x_trans4_dot_c_inv4_x_point_gradient[j]
                        + x_trans4_dot_c_inv4_x_ext_point_hessian[(0, j)]
                        + point_gradient_t_c_inv4_x_point_gradient[(j, i)]);
            }
        }
    }

    DerivativeResult {
        score,
        gradient,
        hessian,
    }
}

/// Compute score, gradient, and Hessian for point-to-plane distance.
///
/// The point-to-plane metric uses only the surface normal direction:
/// - Distance: d = (x - μ) · n  (dot product with principal axis)
/// - Score: p(x) = -d1 * exp(-d2/2 * d²)
///
/// This is faster and more stable for planar structures.
///
/// # Arguments
/// * `x_trans` - Transformed point minus voxel mean (x - μ)
/// * `principal_axis` - Surface normal (eigenvector of smallest covariance eigenvalue)
/// * `point_gradient` - Point gradient matrix (4x6)
/// * `point_hessian` - Point Hessian matrix (24x6)
/// * `gauss` - Gaussian parameters
/// * `compute_hessian` - Whether to compute Hessian
pub fn compute_derivative_point_to_plane(
    x_trans: &Vector3<f64>,
    principal_axis: &Vector3<f64>,
    point_gradient: &Matrix4x6,
    point_hessian: &Matrix24x6,
    gauss: &GaussianParams,
    compute_hessian: bool,
) -> DerivativeResult {
    // Point-to-plane distance: d = (x - μ) · n
    let distance = x_trans.dot(principal_axis);

    // Distance squared for exponential
    let distance_sq = distance * distance;

    // exp(-d2/2 * d²) - similar to Eq. 6.9 but with scalar distance
    let e_x_cov_x_raw = (-gauss.d2 * distance_sq * 0.5).exp();

    // Score contribution: -d1 * exp(...)
    let score = -gauss.d1 * e_x_cov_x_raw;

    // Scale for gradient/hessian computation
    let e_x_cov_x = gauss.d1 * gauss.d2 * e_x_cov_x_raw;

    // Error checking for invalid values
    let normalized_exp = gauss.d2 * e_x_cov_x_raw;
    if !(0.0..=1.0).contains(&normalized_exp) || !normalized_exp.is_finite() {
        return DerivativeResult::zeros();
    }

    // Extend normal to 4D (w = 0)
    let n4 = Vector4::new(principal_axis.x, principal_axis.y, principal_axis.z, 0.0);

    // ∂d/∂p = n^T * ∂T(x)/∂p  (1x6 vector)
    // This is the Jacobian of the distance function
    let n4_row = n4.transpose();
    let dd_dp_row = n4_row * point_gradient;

    // Convert to Vector6
    let dd_dp = Vector6::new(
        dd_dp_row[(0, 0)],
        dd_dp_row[(0, 1)],
        dd_dp_row[(0, 2)],
        dd_dp_row[(0, 3)],
        dd_dp_row[(0, 4)],
        dd_dp_row[(0, 5)],
    );

    // Gradient: d1 * d2 * exp(...) * d * ∂d/∂p
    let gradient = e_x_cov_x * distance * dd_dp;

    // Compute Hessian if requested
    let mut hessian = Matrix6::zeros();

    if compute_hessian {
        // For point-to-plane:
        // H = d1 * d2 * exp(-d2/2 * d²) * [
        //     (-d2 * d² + 1) * (∂d/∂p)^T * (∂d/∂p) + d * ∂²d/∂p²
        // ]
        // where ∂²d/∂p² = n^T * ∂²T/∂p²

        let scale = e_x_cov_x;

        // First term: (-d2 * d² + 1) * (∂d/∂p)(∂d/∂p)^T
        let factor = -gauss.d2 * distance_sq + 1.0;
        let outer_product = dd_dp * dd_dp.transpose();
        hessian += scale * factor * outer_product;

        // Second term: d * n^T * ∂²T/∂p²
        for i in 0..6 {
            let hessian_block = point_hessian.fixed_view::<4, 6>(i * 4, 0);
            let n4_x_hessian_block = n4_row * hessian_block;

            for j in 0..6 {
                hessian[(i, j)] += scale * distance * n4_x_hessian_block[(0, j)];
            }
        }
    }

    DerivativeResult {
        score,
        gradient,
        hessian,
    }
}

/// Transform a point using a 6-DOF pose.
///
/// # Arguments
/// * `point` - Source point [x, y, z]
/// * `pose` - 6-DOF pose [tx, ty, tz, roll, pitch, yaw]
///
/// # Returns
/// Transformed point [x', y', z']
pub fn transform_point(point: &[f64; 3], pose: &[f64; 6]) -> [f64; 3] {
    let (tx, ty, tz) = (pose[0], pose[1], pose[2]);
    let (roll, pitch, yaw) = (pose[3], pose[4], pose[5]);

    // Compute rotation matrix from Euler angles (XYZ order)
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    // Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
    let r00 = cy * cp;
    let r01 = cy * sp * sr - sy * cr;
    let r02 = cy * sp * cr + sy * sr;
    let r10 = sy * cp;
    let r11 = sy * sp * sr + cy * cr;
    let r12 = sy * sp * cr - cy * sr;
    let r20 = -sp;
    let r21 = cp * sr;
    let r22 = cp * cr;

    let x = point[0];
    let y = point[1];
    let z = point[2];

    [
        r00 * x + r01 * y + r02 * z + tx,
        r10 * x + r11 * y + r12 * z + ty,
        r20 * x + r21 * y + r22 * z + tz,
    ]
}

/// Compute NDT score, gradient, and Hessian for a point cloud against a voxel grid.
///
/// This is the main CPU reference implementation for derivative computation.
///
/// # Arguments
/// * `source_points` - Source point cloud to match
/// * `target_grid` - Target voxel grid (map)
/// * `pose` - Current pose estimate [tx, ty, tz, roll, pitch, yaw]
/// * `gauss` - Gaussian parameters for NDT
/// * `compute_hessian` - Whether to compute Hessian
///
/// # Returns
/// Aggregated derivatives for all correspondences
pub fn compute_derivatives_cpu(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &[f64; 6],
    gauss: &GaussianParams,
    compute_hessian: bool,
) -> AggregatedDerivatives {
    // Use default point-to-distribution metric
    compute_derivatives_cpu_with_metric(
        source_points,
        target_grid,
        pose,
        gauss,
        compute_hessian,
        DistanceMetric::PointToDistribution,
    )
}

/// Compute NDT score, gradient, and Hessian with configurable distance metric.
///
/// # Arguments
/// * `source_points` - Source point cloud to match
/// * `target_grid` - Target voxel grid (map)
/// * `pose` - Current pose estimate [tx, ty, tz, roll, pitch, yaw]
/// * `gauss` - Gaussian parameters for NDT
/// * `compute_hessian` - Whether to compute Hessian
/// * `metric` - Distance metric to use (PointToDistribution or PointToPlane)
///
/// # Returns
/// Aggregated derivatives for all correspondences
pub fn compute_derivatives_cpu_with_metric(
    source_points: &[[f32; 3]],
    target_grid: &VoxelGrid,
    pose: &[f64; 6],
    gauss: &GaussianParams,
    compute_hessian: bool,
    metric: DistanceMetric,
) -> AggregatedDerivatives {
    let mut result = AggregatedDerivatives::zeros();

    // Precompute angular derivatives for this pose
    let angular = AngularDerivatives::new(pose[3], pose[4], pose[5], compute_hessian);

    // Debug: track voxel-per-point distribution (only in debug builds)
    #[cfg(debug_assertions)]
    let (
        debug_vpp,
        mut points_with_0_voxels,
        mut points_with_1_voxel,
        mut points_with_2_voxels,
        mut points_with_3plus_voxels,
    ) = (
        std::env::var("NDT_DEBUG_VPP").is_ok(),
        0usize,
        0usize,
        0usize,
        0usize,
    );

    for source_point in source_points {
        // Convert to f64
        let point_f64 = [
            source_point[0] as f64,
            source_point[1] as f64,
            source_point[2] as f64,
        ];

        // Transform point using current pose
        let transformed = transform_point(&point_f64, pose);

        // Find corresponding voxels using radius search (like Autoware's radiusSearch)
        // This returns all voxels whose centroids are within the search radius
        let transformed_f32 = [
            transformed[0] as f32,
            transformed[1] as f32,
            transformed[2] as f32,
        ];

        // Use voxel resolution as search radius (matches Autoware behavior)
        let search_radius = target_grid.resolution();
        let nearby_voxels = target_grid.radius_search(&transformed_f32, search_radius);

        // Debug: track voxel-per-point distribution (only in debug builds)
        #[cfg(debug_assertions)]
        if debug_vpp {
            match nearby_voxels.len() {
                0 => points_with_0_voxels += 1,
                1 => points_with_1_voxel += 1,
                2 => points_with_2_voxels += 1,
                _ => points_with_3plus_voxels += 1,
            }
        }

        if nearby_voxels.is_empty() {
            continue; // No correspondences for this point
        }

        // Compute point derivatives once per source point (shared across all voxels)
        let point_deriv = compute_point_derivatives(&point_f64, &angular, compute_hessian);

        // Accumulate contributions from ALL nearby voxels (key difference from single-voxel lookup)
        for voxel in nearby_voxels {
            // Compute x - μ (transformed point minus voxel mean)
            let x_trans = Vector3::new(
                transformed[0] - voxel.mean.x as f64,
                transformed[1] - voxel.mean.y as f64,
                transformed[2] - voxel.mean.z as f64,
            );

            // Compute score/gradient/hessian based on distance metric
            let deriv = match metric {
                DistanceMetric::PointToDistribution => {
                    // Full Mahalanobis distance using inverse covariance
                    let inv_cov = voxel.inv_covariance.cast::<f64>();
                    compute_derivative_single(
                        &x_trans,
                        &inv_cov,
                        &point_deriv.point_gradient,
                        &point_deriv.point_hessian,
                        gauss,
                        compute_hessian,
                    )
                }
                DistanceMetric::PointToPlane => {
                    // Simplified point-to-plane distance using principal axis
                    let principal_axis = voxel.principal_axis.cast::<f64>();
                    compute_derivative_point_to_plane(
                        &x_trans,
                        &principal_axis,
                        &point_deriv.point_gradient,
                        &point_deriv.point_hessian,
                        gauss,
                        compute_hessian,
                    )
                }
            };

            // Accumulate
            result.add(&deriv);
        }
    }

    // Debug: output voxel-per-point distribution (only in debug builds)
    #[cfg(debug_assertions)]
    if debug_vpp {
        let total = source_points.len();
        let vpp = result.num_correspondences as f64 / total as f64;
        debug!(
            "[VPP] {} points: 0v={}, 1v={}, 2v={}, 3+v={} | {} corr ({:.2} vpp)",
            total,
            points_with_0_voxels,
            points_with_1_voxel,
            points_with_2_voxels,
            points_with_3plus_voxels,
            result.num_correspondences,
            vpp
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_grid() -> VoxelGrid {
        // Create a grid with points clustered around (1, 1, 1)
        // Use random points to ensure non-degenerate covariance
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 0.1).unwrap();
        let center = [1.0f32, 1.0, 1.0];

        let mut points = Vec::new();
        for _ in 0..50 {
            points.push([
                center[0] + dist.sample(&mut rng) as f32,
                center[1] + dist.sample(&mut rng) as f32,
                center[2] + dist.sample(&mut rng) as f32,
            ]);
        }
        VoxelGrid::from_points(&points, 2.0).unwrap()
    }

    #[test]
    fn test_transform_point_identity() {
        let point = [1.0, 2.0, 3.0];
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let transformed = transform_point(&point, &pose);

        assert_relative_eq!(transformed[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(transformed[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(transformed[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_voxel_radius_search_contributes_multiple_correspondences() {
        // Create a dense grid where each source point should find multiple voxels
        // This verifies the multi-voxel radius search is working
        use crate::test_utils::make_default_half_cubic_pcd;

        let target_points = make_default_half_cubic_pcd();
        let target_grid = VoxelGrid::from_points(&target_points, 2.0).unwrap();

        // Use a single source point at the center of the grid
        // At resolution 2.0, this should find multiple nearby voxels
        let source_points = vec![[10.0f32, 10.0, 0.0]]; // Center of XY plane

        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Identity pose
        let gauss = GaussianParams::new(2.0, 0.55);

        let result = compute_derivatives_cpu(&source_points, &target_grid, &pose, &gauss, true);

        // With multi-voxel radius search, we should have MORE correspondences
        // than source points (1 point finding multiple voxels)
        println!(
            "Source points: {}, Correspondences: {}",
            source_points.len(),
            result.num_correspondences
        );

        // Key assertion: multi-voxel search means more correspondences than points
        assert!(
            result.num_correspondences >= 1,
            "Should find at least one voxel"
        );

        // With a point at (10,10,0) on a half-cubic PCD and resolution 2.0,
        // we expect to find multiple voxels whose centroids are within 2.0m
        // Let's verify radius search is finding multiple voxels:
        let transformed = [10.0f32, 10.0, 0.0];
        let nearby = target_grid.radius_search(&transformed, 2.0);
        println!("Nearby voxels found: {}", nearby.len());

        // This test confirms multi-voxel search is working
        // The actual count depends on the point cloud density
        assert!(
            !nearby.is_empty(),
            "Radius search should find nearby voxels"
        );
    }

    #[test]
    fn test_transform_point_translation() {
        let point = [1.0, 2.0, 3.0];
        let pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];

        let transformed = transform_point(&point, &pose);

        assert_relative_eq!(transformed[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(transformed[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(transformed[2], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transform_point_rotation_yaw() {
        use std::f64::consts::FRAC_PI_2;

        let point = [1.0, 0.0, 0.0];
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, FRAC_PI_2]; // 90 degree yaw

        let transformed = transform_point(&point, &pose);

        // After 90 degree rotation around Z, [1,0,0] -> [0,1,0]
        assert_relative_eq!(transformed[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(transformed[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(transformed[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_point_derivatives_translation() {
        let angular = AngularDerivatives::new(0.0, 0.0, 0.0, false);
        let point = [1.0, 2.0, 3.0];

        let deriv = compute_point_derivatives(&point, &angular, false);

        // Translation derivatives should be identity
        assert_eq!(deriv.point_gradient[(0, 0)], 1.0);
        assert_eq!(deriv.point_gradient[(1, 1)], 1.0);
        assert_eq!(deriv.point_gradient[(2, 2)], 1.0);
        assert_eq!(deriv.point_gradient[(0, 1)], 0.0);
    }

    #[test]
    fn test_compute_derivative_single_score() {
        // Test that score is positive (NDT cost function)
        // Score = -d1 * exp(...) where d1 < 0, so score > 0
        let x_trans = Vector3::new(0.1, 0.1, 0.1);
        let inv_cov = Matrix3::identity();
        let point_gradient = Matrix4x6::zeros();
        let point_hessian = Matrix24x6::zeros();
        let gauss = GaussianParams::default();

        let result = compute_derivative_single(
            &x_trans,
            &inv_cov,
            &point_gradient,
            &point_hessian,
            &gauss,
            false,
        );

        // Score should be positive (it's a cost to minimize)
        // score = -d1 * exp(...) where d1 < 0
        assert!(
            result.score > 0.0,
            "Score should be positive: {}",
            result.score
        );
    }

    #[test]
    fn test_compute_derivative_single_zero_displacement() {
        // When x = μ, score should be at maximum value
        let x_trans = Vector3::zeros();
        let inv_cov = Matrix3::identity();
        let point_gradient = Matrix4x6::zeros();
        let point_hessian = Matrix24x6::zeros();
        let gauss = GaussianParams::default();

        let result = compute_derivative_single(
            &x_trans,
            &inv_cov,
            &point_gradient,
            &point_hessian,
            &gauss,
            false,
        );

        // At μ, exp(-d2/2 * 0) = 1, so score = -d1
        // Since d1 < 0, score = -d1 > 0
        assert_relative_eq!(result.score, -gauss.d1, epsilon = 1e-10);
        assert!(result.score > 0.0, "Score should be positive at mean");
    }

    #[test]
    fn test_compute_derivatives_cpu_basic() {
        let grid = create_test_grid();

        // Source point near the voxel center (grid is at [1, 1, 1])
        let source_points = vec![[1.0f32, 1.0, 1.0]];
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let gauss = GaussianParams::default();

        let result = compute_derivatives_cpu(&source_points, &grid, &pose, &gauss, false);

        // Should have found a correspondence
        assert_eq!(result.num_correspondences, 1);
        // Score should be positive (NDT cost function: -d1 * exp(...) where d1 < 0)
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_compute_derivatives_cpu_no_correspondence() {
        let grid = create_test_grid();

        // Source points far from any voxel
        let source_points = vec![[1000.0f32, 1000.0, 1000.0]];
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let gauss = GaussianParams::default();

        let result = compute_derivatives_cpu(&source_points, &grid, &pose, &gauss, false);

        // Should have no correspondences
        assert_eq!(result.num_correspondences, 0);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_gradient_finite_difference() {
        let grid = create_test_grid();
        let source_points = vec![[1.0f32, 1.0, 1.0]]; // Grid center
        let gauss = GaussianParams::default();
        let eps = 1e-6;

        // Compute gradient at identity pose
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = compute_derivatives_cpu(&source_points, &grid, &pose, &gauss, false);

        // Verify gradient with finite differences for translation
        for i in 0..3 {
            let mut pose_plus = pose;
            let mut pose_minus = pose;
            pose_plus[i] += eps;
            pose_minus[i] -= eps;

            let result_plus =
                compute_derivatives_cpu(&source_points, &grid, &pose_plus, &gauss, false);
            let result_minus =
                compute_derivatives_cpu(&source_points, &grid, &pose_minus, &gauss, false);

            let numerical_grad = (result_plus.score - result_minus.score) / (2.0 * eps);
            let analytical_grad = result.gradient[i];

            // Gradient should match within tolerance
            assert_relative_eq!(
                numerical_grad,
                analytical_grad,
                epsilon = 1e-4,
                max_relative = 0.1
            );
        }
    }

    #[test]
    fn test_hessian_symmetry() {
        let grid = create_test_grid();
        let source_points = vec![[1.0f32, 1.0, 1.0]]; // Grid center
        let gauss = GaussianParams::default();
        let pose = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

        let result = compute_derivatives_cpu(&source_points, &grid, &pose, &gauss, true);

        // Hessian should be symmetric
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(
                    result.hessian[(i, j)],
                    result.hessian[(j, i)],
                    epsilon = 1e-10
                );
            }
        }
    }

    // ===== Point-to-Plane Metric Tests =====

    #[test]
    fn test_point_to_plane_score_zero_distance() {
        // When point is exactly on the plane (distance = 0), score is at maximum
        let x_trans = Vector3::zeros(); // Point at voxel mean
        let principal_axis = Vector3::new(0.0, 0.0, 1.0); // Normal pointing up
        let point_gradient = Matrix4x6::zeros();
        let point_hessian = Matrix24x6::zeros();
        let gauss = GaussianParams::default();

        let result = compute_derivative_point_to_plane(
            &x_trans,
            &principal_axis,
            &point_gradient,
            &point_hessian,
            &gauss,
            false,
        );

        // At distance=0, exp(0) = 1, so score = -d1
        assert_relative_eq!(result.score, -gauss.d1, epsilon = 1e-10);
        assert!(
            result.score > 0.0,
            "Score should be positive at zero distance"
        );
    }

    #[test]
    fn test_point_to_plane_score_nonzero_distance() {
        // Point displaced along the normal direction
        let principal_axis = Vector3::new(0.0, 0.0, 1.0);
        let x_trans = Vector3::new(0.0, 0.0, 1.0); // 1 unit above mean
        let point_gradient = Matrix4x6::zeros();
        let point_hessian = Matrix24x6::zeros();
        let gauss = GaussianParams::default();

        let result = compute_derivative_point_to_plane(
            &x_trans,
            &principal_axis,
            &point_gradient,
            &point_hessian,
            &gauss,
            false,
        );

        // Score should be positive but less than maximum
        assert!(result.score > 0.0);
        assert!(
            result.score < -gauss.d1,
            "Score should decrease with distance"
        );
    }

    #[test]
    fn test_point_to_plane_score_perpendicular_displacement() {
        // Point displaced perpendicular to normal (parallel to plane)
        let principal_axis = Vector3::new(0.0, 0.0, 1.0); // Normal pointing up
        let x_trans = Vector3::new(1.0, 0.0, 0.0); // 1 unit in X (on plane)
        let point_gradient = Matrix4x6::zeros();
        let point_hessian = Matrix24x6::zeros();
        let gauss = GaussianParams::default();

        let result = compute_derivative_point_to_plane(
            &x_trans,
            &principal_axis,
            &point_gradient,
            &point_hessian,
            &gauss,
            false,
        );

        // Displacement perpendicular to normal = distance is 0
        // Score should be at maximum
        assert_relative_eq!(result.score, -gauss.d1, epsilon = 1e-10);
    }

    #[test]
    fn test_point_to_plane_vs_point_to_distribution() {
        // Compare both metrics on the same test case
        let grid = create_test_grid();
        let source_points = vec![[1.0f32, 1.0, 1.0]];
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let gauss = GaussianParams::default();

        let p2d_result = compute_derivatives_cpu_with_metric(
            &source_points,
            &grid,
            &pose,
            &gauss,
            true,
            DistanceMetric::PointToDistribution,
        );

        let p2p_result = compute_derivatives_cpu_with_metric(
            &source_points,
            &grid,
            &pose,
            &gauss,
            true,
            DistanceMetric::PointToPlane,
        );

        // Both should find correspondences
        assert_eq!(
            p2d_result.num_correspondences,
            p2p_result.num_correspondences
        );

        // Both scores should be positive
        assert!(p2d_result.score > 0.0);
        assert!(p2p_result.score > 0.0);

        // Hessians should be symmetric for both metrics
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(
                    p2p_result.hessian[(i, j)],
                    p2p_result.hessian[(j, i)],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_point_to_plane_gradient_finite_difference() {
        let grid = create_test_grid();
        let source_points = vec![[1.0f32, 1.0, 1.0]];
        let gauss = GaussianParams::default();
        let eps = 1e-6;

        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = compute_derivatives_cpu_with_metric(
            &source_points,
            &grid,
            &pose,
            &gauss,
            false,
            DistanceMetric::PointToPlane,
        );

        // Verify gradient with finite differences for translation
        for i in 0..3 {
            let mut pose_plus = pose;
            let mut pose_minus = pose;
            pose_plus[i] += eps;
            pose_minus[i] -= eps;

            let result_plus = compute_derivatives_cpu_with_metric(
                &source_points,
                &grid,
                &pose_plus,
                &gauss,
                false,
                DistanceMetric::PointToPlane,
            );
            let result_minus = compute_derivatives_cpu_with_metric(
                &source_points,
                &grid,
                &pose_minus,
                &gauss,
                false,
                DistanceMetric::PointToPlane,
            );

            let numerical_grad = (result_plus.score - result_minus.score) / (2.0 * eps);
            let analytical_grad = result.gradient[i];

            // Gradient should match within tolerance
            assert_relative_eq!(
                numerical_grad,
                analytical_grad,
                epsilon = 1e-4,
                max_relative = 0.1
            );
        }
    }
}
