//! GPU-accelerated NDT derivative computation using CubeCL.
//!
//! This module provides CUDA-accelerated computation of NDT score, gradient, and Hessian.
//! The key algorithm:
//! 1. Transform source points using current pose
//! 2. For each transformed point, find nearby voxels (radius search)
//! 3. Compute per-point-voxel derivatives
//! 4. Reduce across all contributions
//!
//! # GPU Memory Layout
//!
//! - Source points: [N, 3] flattened to [N * 3]
//! - Voxel means: [V, 3] flattened to [V * 3]
//! - Voxel inv_covariances: [V, 9] flattened to [V * 9] (row-major 3x3)
//! - Morton codes: [V] for radius search
//!
//! # Multi-Voxel Radius Search
//!
//! Unlike single-voxel lookup, we search for ALL voxels within a radius of
//! each transformed point. This matches Autoware's `radiusSearch` behavior
//! and provides smoother gradients near voxel boundaries.

use cubecl::prelude::*;

/// Maximum number of neighboring voxels per query point.
/// Autoware typically finds 1-7 neighbors with radius = resolution.
pub const MAX_NEIGHBORS: u32 = 8;

/// Size of per-point derivative output.
/// score (1) + gradient (6) + hessian_upper (21) = 28 elements
/// We store upper triangle of symmetric 6x6 Hessian.
pub const DERIVATIVE_OUTPUT_SIZE: u32 = 28;

// Note: Helper functions like distance_squared are inlined directly
// in kernels due to CubeCL type inference limitations with generics.

/// Transform a point by a 4x4 transformation matrix.
///
/// # Arguments
/// * `px`, `py`, `pz` - Input point
/// * `transform` - 4x4 transformation matrix (row-major)
///
/// # Returns
/// Transformed point (tx, ty, tz)
#[cube]
fn transform_point_inline<F: Float>(px: F, py: F, pz: F, transform: &Array<F>) -> (F, F, F) {
    let tx = transform[0] * px + transform[1] * py + transform[2] * pz + transform[3];
    let ty = transform[4] * px + transform[5] * py + transform[6] * pz + transform[7];
    let tz = transform[8] * px + transform[9] * py + transform[10] * pz + transform[11];
    (tx, ty, tz)
}

/// Brute-force radius search on GPU.
///
/// For each query point, finds up to MAX_NEIGHBORS voxels within the radius.
/// Outputs neighbor indices (-1 for no neighbor).
///
/// This is O(N*V) but with good GPU parallelism for moderate V.
/// For large V, Morton-based search would be more efficient.
#[cube(launch_unchecked)]
pub fn radius_search_kernel<F: Float>(
    // Query points (transformed source points) [N * 3]
    query_points: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel validity flags [V]
    voxel_valid: &Array<u32>,
    // Search radius squared
    radius_sq: F,
    // Number of query points
    num_queries: u32,
    // Number of voxels
    num_voxels: u32,
    // Output: neighbor indices [N * MAX_NEIGHBORS], -1 for no neighbor
    neighbor_indices: &mut Array<i32>,
    // Output: neighbor count per query [N]
    neighbor_counts: &mut Array<u32>,
) {
    let query_idx = ABSOLUTE_POS;

    if query_idx >= num_queries {
        terminate!();
    }

    // Load query point
    let base = query_idx * 3;
    let qx = query_points[base];
    let qy = query_points[base + 1];
    let qz = query_points[base + 2];

    // Initialize neighbor output
    let out_base = query_idx * MAX_NEIGHBORS;
    for i in 0..MAX_NEIGHBORS {
        neighbor_indices[out_base + i] = -1_i32;
    }

    let mut count = 0u32;

    // Search all voxels
    // NOTE: We avoid using `break` here because it triggers a CubeCL optimizer bug
    // in uniformity analysis ("no entry found for key"). Instead, we use a
    // conditional flag to skip processing once we've found enough neighbors.
    for v in 0..num_voxels {
        // Only process if we haven't reached MAX_NEIGHBORS yet
        let should_process = count < MAX_NEIGHBORS;

        if should_process {
            // Skip invalid voxels (use conditional instead of continue)
            let is_valid = voxel_valid[v];
            if is_valid != 0u32 {
                let vbase = v * 3;
                let vx = voxel_means[vbase];
                let vy = voxel_means[vbase + 1];
                let vz = voxel_means[vbase + 2];

                // Inline distance calculation (CubeCL type inference issue with helpers)
                let dx = qx - vx;
                let dy = qy - vy;
                let dz = qz - vz;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq <= radius_sq {
                    neighbor_indices[out_base + count] = v as i32;
                    count += 1u32;
                }
            }
        }
    }

    neighbor_counts[query_idx] = count;
}

/// Compute NDT score for all point-voxel pairs.
///
/// For each source point, computes score contributions from all neighboring voxels.
///
/// # Algorithm (per point)
/// 1. Transform point using current pose
/// 2. Look up neighbor voxels from pre-computed indices
/// 3. For each neighbor voxel:
///    a. Compute x_trans = transformed_point - voxel_mean
///    b. Compute score contribution: -d1 * exp(-d2/2 * x'Σ⁻¹x)
/// 4. Accumulate to per-point output
#[cube(launch_unchecked)]
pub fn compute_ndt_score_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel inverse covariances [V * 9] (row-major 3x3)
    voxel_inv_covs: &Array<F>,
    // Neighbor indices from radius search [N * MAX_NEIGHBORS]
    neighbor_indices: &Array<i32>,
    // Neighbor counts [N]
    neighbor_counts: &Array<u32>,
    // Gaussian d1 parameter
    gauss_d1: F,
    // Gaussian d2 parameter
    gauss_d2: F,
    // Number of source points
    num_points: u32,
    // Output: per-point scores [N]
    scores: &mut Array<F>,
    // Output: per-point correspondence counts [N]
    correspondences: &mut Array<u32>,
) {
    let point_idx = ABSOLUTE_POS;

    if point_idx >= num_points {
        terminate!();
    }

    // Load and transform source point
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    let (tx, ty, tz) = transform_point_inline(sx, sy, sz, transform);

    // Accumulate score across all neighbors
    let mut total_score = F::new(0.0);
    let mut total_correspondences = 0u32;

    let num_neighbors = neighbor_counts[point_idx];
    let neighbor_base = point_idx * MAX_NEIGHBORS;

    for i in 0..MAX_NEIGHBORS {
        // Only process if within neighbor count
        if i < num_neighbors {
            let voxel_idx = neighbor_indices[neighbor_base + i];
            if voxel_idx >= 0 {
                let v = voxel_idx as u32;

                // Load voxel mean
                let vbase = v * 3;
                let mx = voxel_means[vbase];
                let my = voxel_means[vbase + 1];
                let mz = voxel_means[vbase + 2];

                // Compute x_trans = transformed - mean
                let x0 = tx - mx;
                let x1 = ty - my;
                let x2 = tz - mz;

                // Load inverse covariance (row-major 3x3)
                let cbase = v * 9;
                let c00 = voxel_inv_covs[cbase];
                let c01 = voxel_inv_covs[cbase + 1];
                let c02 = voxel_inv_covs[cbase + 2];
                let c10 = voxel_inv_covs[cbase + 3];
                let c11 = voxel_inv_covs[cbase + 4];
                let c12 = voxel_inv_covs[cbase + 5];
                let c20 = voxel_inv_covs[cbase + 6];
                let c21 = voxel_inv_covs[cbase + 7];
                let c22 = voxel_inv_covs[cbase + 8];

                // Compute x' * Σ⁻¹ * x
                // First: Σ⁻¹ * x
                let cx0 = c00 * x0 + c01 * x1 + c02 * x2;
                let cx1 = c10 * x0 + c11 * x1 + c12 * x2;
                let cx2 = c20 * x0 + c21 * x1 + c22 * x2;

                // Then: x' * (Σ⁻¹ * x)
                let x_c_x = x0 * cx0 + x1 * cx1 + x2 * cx2;

                // Score: -d1 * exp(-d2/2 * x'Σ⁻¹x)
                let exponent = gauss_d2 * x_c_x * F::new(-0.5);
                let score = gauss_d1 * F::new(-1.0) * F::exp(exponent);

                total_score += score;
                total_correspondences += 1u32;
            }
        }
    }

    scores[point_idx] = total_score;
    correspondences[point_idx] = total_correspondences;
}

/// Compute NVTL (Nearest Voxel Transformation Likelihood) scores.
///
/// For each source point, computes the **maximum** score across all neighboring voxels.
/// This matches Autoware's NVTL algorithm where NVTL = average of max scores.
///
/// # Algorithm (per point)
/// 1. Transform point using current pose
/// 2. Look up neighbor voxels from pre-computed indices
/// 3. For each neighbor voxel:
///    a. Compute x_trans = transformed_point - voxel_mean
///    b. Compute score: -d1 * exp(-d2/2 * x'Σ⁻¹x)
/// 4. Track the **maximum** score (not sum)
///
/// # Difference from compute_ndt_score_kernel
/// - `compute_ndt_score_kernel`: Sums scores → for transform probability
/// - `compute_ndt_nvtl_kernel`: Takes max score → for NVTL (Autoware-compatible)
#[cube(launch_unchecked)]
pub fn compute_ndt_nvtl_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel inverse covariances [V * 9] (row-major 3x3)
    voxel_inv_covs: &Array<F>,
    // Neighbor indices from radius search [N * MAX_NEIGHBORS]
    neighbor_indices: &Array<i32>,
    // Neighbor counts [N]
    neighbor_counts: &Array<u32>,
    // Gaussian d1 parameter
    gauss_d1: F,
    // Gaussian d2 parameter
    gauss_d2: F,
    // Number of source points
    num_points: u32,
    // Output: max score per point [N]
    max_scores: &mut Array<F>,
    // Output: 1 if point has at least one neighbor, 0 otherwise [N]
    has_neighbor: &mut Array<u32>,
) {
    let point_idx = ABSOLUTE_POS;

    if point_idx >= num_points {
        terminate!();
    }

    // Load and transform source point
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    let (tx, ty, tz) = transform_point_inline(sx, sy, sz, transform);

    // Track maximum score across all neighbors (NVTL algorithm)
    let mut max_score = F::new(0.0);
    let mut found_neighbor = 0u32;

    let num_neighbors = neighbor_counts[point_idx];
    let neighbor_base = point_idx * MAX_NEIGHBORS;

    for i in 0..MAX_NEIGHBORS {
        // Only process if within neighbor count
        if i < num_neighbors {
            let voxel_idx = neighbor_indices[neighbor_base + i];
            if voxel_idx >= 0 {
                let v = voxel_idx as u32;

                // Load voxel mean
                let vbase = v * 3;
                let mx = voxel_means[vbase];
                let my = voxel_means[vbase + 1];
                let mz = voxel_means[vbase + 2];

                // Compute x_trans = transformed - mean
                let x0 = tx - mx;
                let x1 = ty - my;
                let x2 = tz - mz;

                // Load inverse covariance (row-major 3x3)
                let cbase = v * 9;
                let c00 = voxel_inv_covs[cbase];
                let c01 = voxel_inv_covs[cbase + 1];
                let c02 = voxel_inv_covs[cbase + 2];
                let c10 = voxel_inv_covs[cbase + 3];
                let c11 = voxel_inv_covs[cbase + 4];
                let c12 = voxel_inv_covs[cbase + 5];
                let c20 = voxel_inv_covs[cbase + 6];
                let c21 = voxel_inv_covs[cbase + 7];
                let c22 = voxel_inv_covs[cbase + 8];

                // Compute x' * Σ⁻¹ * x
                // First: Σ⁻¹ * x
                let cx0 = c00 * x0 + c01 * x1 + c02 * x2;
                let cx1 = c10 * x0 + c11 * x1 + c12 * x2;
                let cx2 = c20 * x0 + c21 * x1 + c22 * x2;

                // Then: x' * (Σ⁻¹ * x)
                let x_c_x = x0 * cx0 + x1 * cx1 + x2 * cx2;

                // Score: -d1 * exp(-d2/2 * x'Σ⁻¹x)
                let exponent = gauss_d2 * x_c_x * F::new(-0.5);
                let score = gauss_d1 * F::new(-1.0) * F::exp(exponent);

                // Track maximum score (key difference from sum-based kernel)
                if found_neighbor == 0u32 || score > max_score {
                    max_score = score;
                }
                found_neighbor = 1u32;
            }
        }
    }

    max_scores[point_idx] = max_score;
    has_neighbor[point_idx] = found_neighbor;
}

/// Compute gradient contributions per point.
///
/// This kernel computes the gradient vector (6 elements) for each point
/// summed across all its neighboring voxels.
///
/// The gradient is: d1*d2*exp(...) * x' * Σ⁻¹ * ∂T/∂p
///
/// Where ∂T/∂p is the 3x6 Jacobian of the transformation w.r.t. pose.
#[cube(launch_unchecked)]
pub fn compute_ndt_gradient_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Point Jacobians [N * 18] - ∂T(p)/∂pose for each point
    // 3x6 matrix flattened row-major
    point_jacobians: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel inverse covariances [V * 9]
    voxel_inv_covs: &Array<F>,
    // Neighbor indices [N * MAX_NEIGHBORS]
    neighbor_indices: &Array<i32>,
    // Neighbor counts [N]
    neighbor_counts: &Array<u32>,
    // Gaussian d1 parameter
    gauss_d1: F,
    // Gaussian d2 parameter
    gauss_d2: F,
    // Number of points
    num_points: u32,
    // Output: per-point gradients [N * 6]
    gradients: &mut Array<F>,
) {
    let point_idx = ABSOLUTE_POS;

    if point_idx >= num_points {
        terminate!();
    }

    // Load and transform source point
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    let (tx, ty, tz) = transform_point_inline(sx, sy, sz, transform);

    // Load point Jacobian (3x6 row-major)
    let jbase = point_idx * 18;

    // Initialize gradient accumulators (unrolled for CubeCL)
    let mut grad0 = F::new(0.0);
    let mut grad1 = F::new(0.0);
    let mut grad2 = F::new(0.0);
    let mut grad3 = F::new(0.0);
    let mut grad4 = F::new(0.0);
    let mut grad5 = F::new(0.0);

    let num_neighbors = neighbor_counts[point_idx];
    let neighbor_base = point_idx * MAX_NEIGHBORS;

    for i in 0..MAX_NEIGHBORS {
        if i < num_neighbors {
            let voxel_idx = neighbor_indices[neighbor_base + i];
            if voxel_idx >= 0 {
                let v = voxel_idx as u32;

                // Load voxel mean and compute x_trans
                let vbase = v * 3;
                let x0 = tx - voxel_means[vbase];
                let x1 = ty - voxel_means[vbase + 1];
                let x2 = tz - voxel_means[vbase + 2];

                // Load inverse covariance
                let cbase = v * 9;
                let c00 = voxel_inv_covs[cbase];
                let c01 = voxel_inv_covs[cbase + 1];
                let c02 = voxel_inv_covs[cbase + 2];
                let c10 = voxel_inv_covs[cbase + 3];
                let c11 = voxel_inv_covs[cbase + 4];
                let c12 = voxel_inv_covs[cbase + 5];
                let c20 = voxel_inv_covs[cbase + 6];
                let c21 = voxel_inv_covs[cbase + 7];
                let c22 = voxel_inv_covs[cbase + 8];

                // Compute Σ⁻¹ * x
                let cx0 = c00 * x0 + c01 * x1 + c02 * x2;
                let cx1 = c10 * x0 + c11 * x1 + c12 * x2;
                let cx2 = c20 * x0 + c21 * x1 + c22 * x2;

                // x' * Σ⁻¹ * x
                let x_c_x = x0 * cx0 + x1 * cx1 + x2 * cx2;

                // e_x_cov_x = d1 * d2 * exp(-d2/2 * x'Σ⁻¹x)
                let exponent = gauss_d2 * x_c_x * F::new(-0.5);
                let e_x_cov_x = gauss_d1 * gauss_d2 * F::exp(exponent);

                // Check for valid value (use conditional, not continue)
                let is_valid = e_x_cov_x <= F::new(1.0) && e_x_cov_x >= F::new(0.0);
                if is_valid {
                    // x' * Σ⁻¹ (row vector) - for symmetric Σ⁻¹: (Σ⁻¹ * x)' = (cx0, cx1, cx2)

                    // Compute gradient: e_x_cov_x * x'Σ⁻¹ * J
                    // Where J is the 3x6 point Jacobian (row-major)
                    // J[row, col] = point_jacobians[jbase + row*6 + col]

                    // Column 0: (x'Σ⁻¹) * J[:,0]
                    let j0_0 = point_jacobians[jbase]; // J[0,0]
                    let j1_0 = point_jacobians[jbase + 6]; // J[1,0]
                    let j2_0 = point_jacobians[jbase + 12]; // J[2,0]
                    grad0 += e_x_cov_x * (cx0 * j0_0 + cx1 * j1_0 + cx2 * j2_0);

                    // Column 1
                    let j0_1 = point_jacobians[jbase + 1];
                    let j1_1 = point_jacobians[jbase + 7];
                    let j2_1 = point_jacobians[jbase + 13];
                    grad1 += e_x_cov_x * (cx0 * j0_1 + cx1 * j1_1 + cx2 * j2_1);

                    // Column 2
                    let j0_2 = point_jacobians[jbase + 2];
                    let j1_2 = point_jacobians[jbase + 8];
                    let j2_2 = point_jacobians[jbase + 14];
                    grad2 += e_x_cov_x * (cx0 * j0_2 + cx1 * j1_2 + cx2 * j2_2);

                    // Column 3
                    let j0_3 = point_jacobians[jbase + 3];
                    let j1_3 = point_jacobians[jbase + 9];
                    let j2_3 = point_jacobians[jbase + 15];
                    grad3 += e_x_cov_x * (cx0 * j0_3 + cx1 * j1_3 + cx2 * j2_3);

                    // Column 4
                    let j0_4 = point_jacobians[jbase + 4];
                    let j1_4 = point_jacobians[jbase + 10];
                    let j2_4 = point_jacobians[jbase + 16];
                    grad4 += e_x_cov_x * (cx0 * j0_4 + cx1 * j1_4 + cx2 * j2_4);

                    // Column 5
                    let j0_5 = point_jacobians[jbase + 5];
                    let j1_5 = point_jacobians[jbase + 11];
                    let j2_5 = point_jacobians[jbase + 17];
                    grad5 += e_x_cov_x * (cx0 * j0_5 + cx1 * j1_5 + cx2 * j2_5);
                }
            }
        }
    }

    // Write output
    let gbase = point_idx * 6;
    gradients[gbase] = grad0;
    gradients[gbase + 1] = grad1;
    gradients[gbase + 2] = grad2;
    gradients[gbase + 3] = grad3;
    gradients[gbase + 4] = grad4;
    gradients[gbase + 5] = grad5;
}

/// CPU fallback for point Jacobian computation.
///
/// Computes ∂T(p)/∂pose for each source point, where T is the transformation
/// and pose is [tx, ty, tz, roll, pitch, yaw].
///
/// The Jacobian is a 3x6 matrix stored row-major (18 elements per point).
pub fn compute_point_jacobians_cpu(source_points: &[[f32; 3]], pose: &[f64; 6]) -> Vec<f32> {
    let (roll, pitch, yaw) = (pose[3], pose[4], pose[5]);

    // Precompute trig values
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    let mut jacobians = Vec::with_capacity(source_points.len() * 18);

    for point in source_points {
        let (x, y, z) = (point[0] as f64, point[1] as f64, point[2] as f64);

        // Row-major 3x6 Jacobian
        // Columns 0-2: translation derivatives (identity block)
        // Columns 3-5: rotation derivatives

        // Row 0: ∂x'/∂(tx, ty, tz, r, p, y)
        jacobians.push(1.0_f32); // ∂x'/∂tx
        jacobians.push(0.0_f32); // ∂x'/∂ty
        jacobians.push(0.0_f32); // ∂x'/∂tz
                                 // ∂x'/∂roll = 0
        jacobians.push(0.0_f32);
        // ∂x'/∂pitch = -cy*sp*cr*z - cy*sp*sr*y - cy*cp*x + ... simplified
        jacobians.push((cy * cp * sr * y - cy * cp * cr * z - cy * sp * x) as f32);
        // ∂x'/∂yaw = -sy*cp*x - sy*sp*sr*y - sy*sp*cr*z - cy*cr*y + cy*sr*z
        jacobians.push(
            (-sy * cp * x + (-sy * sp * sr - cy * cr) * y + (-sy * sp * cr + cy * sr) * z) as f32,
        );

        // Row 1: ∂y'/∂(tx, ty, tz, r, p, y)
        jacobians.push(0.0_f32); // ∂y'/∂tx
        jacobians.push(1.0_f32); // ∂y'/∂ty
        jacobians.push(0.0_f32); // ∂y'/∂tz
                                 // ∂y'/∂roll
        jacobians.push(((sy * sp * cr + cy * sr) * y + (-sy * sp * sr + cy * cr) * z) as f32);
        // ∂y'/∂pitch
        jacobians.push((sy * cp * sr * y - sy * cp * cr * z - sy * sp * x) as f32);
        // ∂y'/∂yaw
        jacobians.push(
            (cy * cp * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z) as f32,
        );

        // Row 2: ∂z'/∂(tx, ty, tz, r, p, y)
        jacobians.push(0.0_f32); // ∂z'/∂tx
        jacobians.push(0.0_f32); // ∂z'/∂ty
        jacobians.push(1.0_f32); // ∂z'/∂tz
                                 // ∂z'/∂roll
        jacobians.push((cp * cr * y - cp * sr * z) as f32);
        // ∂z'/∂pitch
        jacobians.push((-sp * sr * y - sp * cr * z - cp * x) as f32);
        // ∂z'/∂yaw = 0
        jacobians.push(0.0_f32);
    }

    jacobians
}

/// GPU derivative computation context.
///
/// Manages GPU buffers and provides the main API for computing
/// NDT derivatives on GPU.
pub struct GpuDerivatives {
    /// Search radius for voxel lookup.
    pub search_radius: f32,
    /// Gaussian d1 parameter.
    pub gauss_d1: f32,
    /// Gaussian d2 parameter.
    pub gauss_d2: f32,
}

impl GpuDerivatives {
    /// Create a new GPU derivatives context.
    pub fn new(resolution: f64, outlier_ratio: f64) -> Self {
        // Compute Gaussian parameters (same as CPU GaussianParams)
        let gauss_c1 = 10.0 * (1.0 - outlier_ratio);
        let gauss_c2 = outlier_ratio / (resolution * resolution * resolution);
        let gauss_d3 = -gauss_c2.ln();
        let gauss_d1 = -(gauss_c1 + gauss_c2).ln() - gauss_d3;
        let gauss_d2_nom = -(gauss_c1 * (-0.5_f64).exp() + gauss_c2).ln() - gauss_d3;
        let gauss_d2 = -2.0 * (gauss_d2_nom / gauss_d1).ln();

        Self {
            search_radius: resolution as f32,
            gauss_d1: gauss_d1 as f32,
            gauss_d2: gauss_d2 as f32,
        }
    }
}

// ============================================================================
// GPU Runtime Integration
// ============================================================================
//
// The GPU integration requires properly initializing a CUDA runtime and
// managing GPU memory. This is done lazily when `compute_derivatives_gpu`
// is called.
//
// For now, we provide the kernel definitions above. The actual runtime
// integration will be added when we have a working CUDA environment to test.

/// GPU voxel data prepared for derivative computation.
#[derive(Debug, Clone)]
pub struct GpuVoxelData {
    /// Flattened means [V * 3]
    pub means: Vec<f32>,
    /// Flattened inverse covariances [V * 9]
    pub inv_covariances: Vec<f32>,
    /// Validity flags [V]
    pub valid: Vec<u32>,
    /// Number of voxels
    pub num_voxels: usize,
}

impl GpuVoxelData {
    /// Create GPU voxel data from a VoxelGrid.
    pub fn from_voxel_grid(grid: &crate::voxel_grid::VoxelGrid) -> Self {
        let means = grid.means_flat();
        let inv_covariances = grid.inv_covariances_flat();
        let valid: Vec<u32> = grid.voxels().iter().map(|_| 1u32).collect();
        let num_voxels = grid.len();

        Self {
            means,
            inv_covariances,
            valid,
            num_voxels,
        }
    }
}

/// Aggregated GPU derivative result.
#[derive(Debug, Clone)]
pub struct GpuDerivativeResult {
    /// Total score (sum across all points and voxels)
    pub score: f64,
    /// Gradient (6 elements) - sum across all points
    pub gradient: [f64; 6],
    /// Number of point-voxel correspondences
    pub num_correspondences: usize,
}

/// Convert pose [tx, ty, tz, roll, pitch, yaw] to 4x4 transformation matrix.
pub fn pose_to_transform_matrix(pose: &[f64; 6]) -> [f32; 16] {
    let (tx, ty, tz) = (pose[0], pose[1], pose[2]);
    let (roll, pitch, yaw) = (pose[3], pose[4], pose[5]);

    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    // ZYX Euler angle rotation matrix (yaw * pitch * roll)
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    let r00 = cy * cp;
    let r01 = cy * sp * sr - sy * cr;
    let r02 = cy * sp * cr + sy * sr;
    let r10 = sy * cp;
    let r11 = sy * sp * sr + cy * cr;
    let r12 = sy * sp * cr - cy * sr;
    let r20 = -sp;
    let r21 = cp * sr;
    let r22 = cp * cr;

    // Row-major 4x4 matrix
    [
        r00 as f32, r01 as f32, r02 as f32, tx as f32, r10 as f32, r11 as f32, r12 as f32,
        ty as f32, r20 as f32, r21 as f32, r22 as f32, tz as f32, 0.0, 0.0, 0.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_jacobians_identity() {
        // At identity pose, translation derivatives are identity,
        // rotation derivatives depend on point coordinates
        let points = vec![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let jacobians = compute_point_jacobians_cpu(&points, &pose);

        assert_eq!(jacobians.len(), 3 * 18);

        // First point [1, 0, 0]:
        // Translation block should be identity
        assert_eq!(jacobians[0], 1.0); // ∂x'/∂tx
        assert_eq!(jacobians[1], 0.0); // ∂x'/∂ty
        assert_eq!(jacobians[2], 0.0); // ∂x'/∂tz
        assert_eq!(jacobians[6], 0.0); // ∂y'/∂tx
        assert_eq!(jacobians[7], 1.0); // ∂y'/∂ty
    }

    #[test]
    fn test_gpu_derivatives_params() {
        let ctx = GpuDerivatives::new(2.0, 0.55);

        // d1 should be negative, d2 positive
        assert!(ctx.gauss_d1 < 0.0);
        assert!(ctx.gauss_d2 > 0.0);
        assert_eq!(ctx.search_radius, 2.0);
    }
}
