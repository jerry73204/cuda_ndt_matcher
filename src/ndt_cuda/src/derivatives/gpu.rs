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

/// Compute NDT score using point-to-plane distance metric.
///
/// For each source point, computes score contributions from all neighboring voxels
/// using the simplified point-to-plane distance instead of full Mahalanobis distance.
///
/// # Algorithm (per point)
/// 1. Transform point using current pose
/// 2. Look up neighbor voxels from pre-computed indices
/// 3. For each neighbor voxel:
///    a. Compute x_trans = transformed_point - voxel_mean
///    b. Compute point-to-plane distance: d = dot(x_trans, principal_axis)
///    c. Compute score contribution: -d1 * exp(-d2/2 * d²)
/// 4. Accumulate to per-point output
///
/// This is a simplified but faster alternative to full Mahalanobis distance,
/// especially effective for planar surfaces.
#[cube(launch_unchecked)]
pub fn compute_ndt_score_point_to_plane_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel principal axes [V * 3] (surface normals)
    voxel_principal_axes: &Array<F>,
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

                // Load principal axis (surface normal)
                let nx = voxel_principal_axes[vbase];
                let ny = voxel_principal_axes[vbase + 1];
                let nz = voxel_principal_axes[vbase + 2];

                // Point-to-plane distance: d = dot(x_trans, n)
                let d = x0 * nx + x1 * ny + x2 * nz;

                // Score: -d1 * exp(-d2/2 * d²)
                let d_sq = d * d;
                let exponent = gauss_d2 * d_sq * F::new(-0.5);
                let score = gauss_d1 * F::new(-1.0) * F::exp(exponent);

                total_score += score;
                total_correspondences += 1u32;
            }
        }
    }

    scores[point_idx] = total_score;
    correspondences[point_idx] = total_correspondences;
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
                // Autoware checks validity BEFORE multiplying by d1 (which is negative)
                let exponent = gauss_d2 * x_c_x * F::new(-0.5);
                let e_x_cov_x_check = gauss_d2 * F::exp(exponent);

                // Check for valid value before applying d1
                // e_x_cov_x_check = d2 * exp(-d2/2 * x'Σ⁻¹x) should be in [0, 1]
                let is_valid = e_x_cov_x_check <= F::new(1.0) && e_x_cov_x_check >= F::new(0.0);
                // Now apply d1 for the actual computation (d1 < 0, so result is negative)
                let e_x_cov_x = gauss_d1 * e_x_cov_x_check;
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

    // Write output (column-major: component * num_points + point_idx)
    // This layout enables efficient CUB segmented reduction
    let np = num_points;
    gradients[point_idx] = grad0;
    gradients[np + point_idx] = grad1;
    gradients[np * 2 + point_idx] = grad2;
    gradients[np * 3 + point_idx] = grad3;
    gradients[np * 4 + point_idx] = grad4;
    gradients[np * 5 + point_idx] = grad5;
}

/// Compute gradient contributions using point-to-plane distance metric.
///
/// This kernel computes the gradient vector (6 elements) for each point
/// using the simplified point-to-plane distance instead of full Mahalanobis.
///
/// The gradient is: d1*d2*exp(-d2/2 * d²) * d * n^T * ∂T/∂p
///
/// Where:
/// - d = dot(x_trans, n) is the point-to-plane distance
/// - n is the principal axis (surface normal)
/// - ∂T/∂p is the 3x6 Jacobian of the transformation w.r.t. pose
#[cube(launch_unchecked)]
pub fn compute_ndt_gradient_point_to_plane_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Point Jacobians [N * 18] - ∂T(p)/∂pose for each point
    // 3x6 matrix flattened row-major
    point_jacobians: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel principal axes [V * 3] (surface normals)
    voxel_principal_axes: &Array<F>,
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

                // Load principal axis
                let nx = voxel_principal_axes[vbase];
                let ny = voxel_principal_axes[vbase + 1];
                let nz = voxel_principal_axes[vbase + 2];

                // Point-to-plane distance: d = dot(x_trans, n)
                let d = x0 * nx + x1 * ny + x2 * nz;
                let d_sq = d * d;

                // e_d = d1 * d2 * exp(-d2/2 * d²) - coefficient for gradient
                // Autoware checks validity BEFORE multiplying by d1 (which is negative)
                let exponent = gauss_d2 * d_sq * F::new(-0.5);
                let e_d_check = gauss_d2 * F::exp(exponent);

                // Check for valid value before applying d1
                // e_d_check = d2 * exp(-d2/2 * d²) should be in [0, 1]
                let is_valid = e_d_check <= F::new(1.0) && e_d_check >= F::new(0.0);
                // Now apply d1 for the actual computation (d1 < 0, so result is negative)
                let e_d = gauss_d1 * e_d_check;
                if is_valid {
                    // Gradient: e_d * d * n^T * J
                    // Where n^T * J is dot product of normal with each column of Jacobian
                    let coeff = e_d * d;

                    // Column 0: n^T * J[:,0]
                    let j0_0 = point_jacobians[jbase]; // J[0,0]
                    let j1_0 = point_jacobians[jbase + 6]; // J[1,0]
                    let j2_0 = point_jacobians[jbase + 12]; // J[2,0]
                    grad0 += coeff * (nx * j0_0 + ny * j1_0 + nz * j2_0);

                    // Column 1
                    let j0_1 = point_jacobians[jbase + 1];
                    let j1_1 = point_jacobians[jbase + 7];
                    let j2_1 = point_jacobians[jbase + 13];
                    grad1 += coeff * (nx * j0_1 + ny * j1_1 + nz * j2_1);

                    // Column 2
                    let j0_2 = point_jacobians[jbase + 2];
                    let j1_2 = point_jacobians[jbase + 8];
                    let j2_2 = point_jacobians[jbase + 14];
                    grad2 += coeff * (nx * j0_2 + ny * j1_2 + nz * j2_2);

                    // Column 3
                    let j0_3 = point_jacobians[jbase + 3];
                    let j1_3 = point_jacobians[jbase + 9];
                    let j2_3 = point_jacobians[jbase + 15];
                    grad3 += coeff * (nx * j0_3 + ny * j1_3 + nz * j2_3);

                    // Column 4
                    let j0_4 = point_jacobians[jbase + 4];
                    let j1_4 = point_jacobians[jbase + 10];
                    let j2_4 = point_jacobians[jbase + 16];
                    grad4 += coeff * (nx * j0_4 + ny * j1_4 + nz * j2_4);

                    // Column 5
                    let j0_5 = point_jacobians[jbase + 5];
                    let j1_5 = point_jacobians[jbase + 11];
                    let j2_5 = point_jacobians[jbase + 17];
                    grad5 += coeff * (nx * j0_5 + ny * j1_5 + nz * j2_5);
                }
            }
        }
    }

    // Write output (column-major: component * num_points + point_idx)
    // This layout enables efficient CUB segmented reduction
    let np = num_points;
    gradients[point_idx] = grad0;
    gradients[np + point_idx] = grad1;
    gradients[np * 2 + point_idx] = grad2;
    gradients[np * 3 + point_idx] = grad3;
    gradients[np * 4 + point_idx] = grad4;
    gradients[np * 5 + point_idx] = grad5;
}

/// Compute Hessian matrix (6x6) for NDT optimization.
///
/// This kernel computes the second derivatives of the NDT score function
/// with respect to the pose parameters. The Hessian is used in Newton's method.
///
/// Equation 6.13 from Magnusson 2009:
/// H[i,j] = e_x_cov_x * (
///     -d2 * (x'Σ⁻¹J[:,i]) * (x'Σ⁻¹J[:,j]) +  // outer product term
///     x'Σ⁻¹ * H_block[:,j] +                  // point Hessian term
///     J[:,i]'Σ⁻¹J[:,j]                        // gradient squared term
/// )
///
/// This version uses fully unrolled loops to avoid dynamic array indexing,
/// which is not supported by CubeCL.
///
/// Note: Parameters are combined to stay within CubeCL's tuple size limits:
/// - `jacobians_and_hessians`: first N*18 floats are jacobians, next N*144 are point hessians
/// - `gauss_params`: [d1, d2]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn compute_ndt_hessian_kernel<F: Float>(
    // Source points [N * 3]
    source_points: &Array<F>,
    // Transformation matrix [16]
    transform: &Array<F>,
    // Combined: Point Jacobians [N * 18] + Point Hessians [N * 144]
    jacobians_and_hessians: &Array<F>,
    // Voxel means [V * 3]
    voxel_means: &Array<F>,
    // Voxel inverse covariances [V * 9]
    voxel_inv_covs: &Array<F>,
    // Neighbor indices [N * MAX_NEIGHBORS]
    neighbor_indices: &Array<i32>,
    // Neighbor counts [N]
    neighbor_counts: &Array<u32>,
    // Gaussian parameters [d1, d2]
    gauss_params: &Array<F>,
    // Number of points
    num_points: u32,
    // Output: per-point Hessians [N * 36] (6x6 symmetric, stored full)
    hessians: &mut Array<F>,
) {
    let gauss_d1 = gauss_params[0];
    let gauss_d2 = gauss_params[1];

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

    // Load point Jacobian (3x6 row-major) - 18 values total
    // j_rc = Jacobian[row r][col c]
    // Jacobians are at the start of the combined buffer
    let jbase = point_idx * 18;
    let j_00 = jacobians_and_hessians[jbase];
    let j_01 = jacobians_and_hessians[jbase + 1];
    let j_02 = jacobians_and_hessians[jbase + 2];
    let j_03 = jacobians_and_hessians[jbase + 3];
    let j_04 = jacobians_and_hessians[jbase + 4];
    let j_05 = jacobians_and_hessians[jbase + 5];
    let j_10 = jacobians_and_hessians[jbase + 6];
    let j_11 = jacobians_and_hessians[jbase + 7];
    let j_12 = jacobians_and_hessians[jbase + 8];
    let j_13 = jacobians_and_hessians[jbase + 9];
    let j_14 = jacobians_and_hessians[jbase + 10];
    let j_15 = jacobians_and_hessians[jbase + 11];
    let j_20 = jacobians_and_hessians[jbase + 12];
    let j_21 = jacobians_and_hessians[jbase + 13];
    let j_22 = jacobians_and_hessians[jbase + 14];
    let j_23 = jacobians_and_hessians[jbase + 15];
    let j_24 = jacobians_and_hessians[jbase + 16];
    let j_25 = jacobians_and_hessians[jbase + 17];

    // Load point Hessian non-zero elements
    // Structure: 24x6 matrix, but only rows 13,14,17,18,21,22 have non-zero values
    // and only in columns 3,4,5
    // Point Hessians start after all jacobians: offset = num_points * 18
    let hbase = num_points * 18 + point_idx * 144;

    // Block 3 (roll): rows 12-15, non-zero in rows 13,14 cols 3-5
    let ph_13_3 = jacobians_and_hessians[hbase + 13 * 6 + 3];
    let ph_13_4 = jacobians_and_hessians[hbase + 13 * 6 + 4];
    let ph_13_5 = jacobians_and_hessians[hbase + 13 * 6 + 5];
    let ph_14_3 = jacobians_and_hessians[hbase + 14 * 6 + 3];
    let ph_14_4 = jacobians_and_hessians[hbase + 14 * 6 + 4];
    let ph_14_5 = jacobians_and_hessians[hbase + 14 * 6 + 5];

    // Block 4 (pitch): rows 16-19, non-zero in rows 16 (d1,e1), 17 (b2,d2,e2), 18 (b3,d3,e3)
    // Row 16 (x-component): cols 4,5 contain d1, e1
    let ph_16_4 = jacobians_and_hessians[hbase + 16 * 6 + 4];
    let ph_16_5 = jacobians_and_hessians[hbase + 16 * 6 + 5];
    let ph_17_3 = jacobians_and_hessians[hbase + 17 * 6 + 3];
    let ph_17_4 = jacobians_and_hessians[hbase + 17 * 6 + 4];
    let ph_17_5 = jacobians_and_hessians[hbase + 17 * 6 + 5];
    let ph_18_3 = jacobians_and_hessians[hbase + 18 * 6 + 3];
    let ph_18_4 = jacobians_and_hessians[hbase + 18 * 6 + 4];
    let ph_18_5 = jacobians_and_hessians[hbase + 18 * 6 + 5];

    // Block 5 (yaw): rows 20-23, non-zero in rows 20 (e1,f1), 21 (c2,e2,f2), 22 (c3,e3,f3)
    // Row 20 (x-component): cols 4,5 contain e1, f1
    let ph_20_4 = jacobians_and_hessians[hbase + 20 * 6 + 4];
    let ph_20_5 = jacobians_and_hessians[hbase + 20 * 6 + 5];
    let ph_21_3 = jacobians_and_hessians[hbase + 21 * 6 + 3];
    let ph_21_4 = jacobians_and_hessians[hbase + 21 * 6 + 4];
    let ph_21_5 = jacobians_and_hessians[hbase + 21 * 6 + 5];
    let ph_22_3 = jacobians_and_hessians[hbase + 22 * 6 + 3];
    let ph_22_4 = jacobians_and_hessians[hbase + 22 * 6 + 4];
    let ph_22_5 = jacobians_and_hessians[hbase + 22 * 6 + 5];

    // Initialize Hessian accumulators (21 unique values for symmetric 6x6)
    let mut h00 = F::new(0.0);
    let mut h01 = F::new(0.0);
    let mut h02 = F::new(0.0);
    let mut h03 = F::new(0.0);
    let mut h04 = F::new(0.0);
    let mut h05 = F::new(0.0);
    let mut h11 = F::new(0.0);
    let mut h12 = F::new(0.0);
    let mut h13 = F::new(0.0);
    let mut h14 = F::new(0.0);
    let mut h15 = F::new(0.0);
    let mut h22 = F::new(0.0);
    let mut h23 = F::new(0.0);
    let mut h24 = F::new(0.0);
    let mut h25 = F::new(0.0);
    let mut h33 = F::new(0.0);
    let mut h34 = F::new(0.0);
    let mut h35 = F::new(0.0);
    let mut h44 = F::new(0.0);
    let mut h45 = F::new(0.0);
    let mut h55 = F::new(0.0);

    let num_neighbors = neighbor_counts[point_idx];
    let neighbor_base = point_idx * MAX_NEIGHBORS;

    // Unrolled neighbor loop (up to MAX_NEIGHBORS = 8)
    // Neighbor 0
    if num_neighbors > 0 {
        let voxel_idx = neighbor_indices[neighbor_base];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 1
    if num_neighbors > 1 {
        let voxel_idx = neighbor_indices[neighbor_base + 1];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 2
    if num_neighbors > 2 {
        let voxel_idx = neighbor_indices[neighbor_base + 2];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 3
    if num_neighbors > 3 {
        let voxel_idx = neighbor_indices[neighbor_base + 3];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 4
    if num_neighbors > 4 {
        let voxel_idx = neighbor_indices[neighbor_base + 4];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 5
    if num_neighbors > 5 {
        let voxel_idx = neighbor_indices[neighbor_base + 5];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 6
    if num_neighbors > 6 {
        let voxel_idx = neighbor_indices[neighbor_base + 6];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 7
    if num_neighbors > 7 {
        let voxel_idx = neighbor_indices[neighbor_base + 7];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Write output (full 6x6, column-major: component * num_points + point_idx)
    // This layout enables efficient CUB segmented reduction
    let np = num_points;
    // Row 0
    hessians[point_idx] = h00;
    hessians[np + point_idx] = h01;
    hessians[np * 2 + point_idx] = h02;
    hessians[np * 3 + point_idx] = h03;
    hessians[np * 4 + point_idx] = h04;
    hessians[np * 5 + point_idx] = h05;

    // Row 1 (symmetric)
    hessians[np * 6 + point_idx] = h01;
    hessians[np * 7 + point_idx] = h11;
    hessians[np * 8 + point_idx] = h12;
    hessians[np * 9 + point_idx] = h13;
    hessians[np * 10 + point_idx] = h14;
    hessians[np * 11 + point_idx] = h15;

    // Row 2
    hessians[np * 12 + point_idx] = h02;
    hessians[np * 13 + point_idx] = h12;
    hessians[np * 14 + point_idx] = h22;
    hessians[np * 15 + point_idx] = h23;
    hessians[np * 16 + point_idx] = h24;
    hessians[np * 17 + point_idx] = h25;

    // Row 3
    hessians[np * 18 + point_idx] = h03;
    hessians[np * 19 + point_idx] = h13;
    hessians[np * 20 + point_idx] = h23;
    hessians[np * 21 + point_idx] = h33;
    hessians[np * 22 + point_idx] = h34;
    hessians[np * 23 + point_idx] = h35;

    // Row 4
    hessians[np * 24 + point_idx] = h04;
    hessians[np * 25 + point_idx] = h14;
    hessians[np * 26 + point_idx] = h24;
    hessians[np * 27 + point_idx] = h34;
    hessians[np * 28 + point_idx] = h44;
    hessians[np * 29 + point_idx] = h45;

    // Row 5
    hessians[np * 30 + point_idx] = h05;
    hessians[np * 31 + point_idx] = h15;
    hessians[np * 32 + point_idx] = h25;
    hessians[np * 33 + point_idx] = h35;
    hessians[np * 34 + point_idx] = h45;
    hessians[np * 35 + point_idx] = h55;
}

// ============================================================================
// Phase 15.2: Refactored Hessian Kernel with Separate Buffers
// ============================================================================

/// Compute NDT Hessian with SEPARATE jacobian and point_hessian inputs.
///
/// This version eliminates the CPU roundtrip for combining J and PH buffers
/// used in Phase 14. Both buffers can now be computed and stored separately
/// on GPU, enabling true zero-transfer Newton iterations.
///
/// # Differences from `compute_ndt_hessian_kernel`:
/// - Takes `jacobians` [N × 18] and `point_hessians` [N × 144] as separate inputs
/// - Otherwise identical math
///
/// # Arguments
/// * `source_points` - [N × 3] source point cloud
/// * `transform` - [16] 4×4 transform matrix (row-major)
/// * `jacobians` - [N × 18] point Jacobians (3×6 per point, row-major)
/// * `point_hessians` - [N × 144] point Hessians (24×6 per point, row-major)
/// * `voxel_means` - [V × 3] voxel centroids
/// * `voxel_inv_covs` - [V × 9] inverse covariances (3×3 row-major)
/// * `neighbor_indices` - [N × MAX_NEIGHBORS] neighbor voxel indices
/// * `neighbor_counts` - [N] number of neighbors per point
/// * `gauss_params` - [2] Gaussian parameters [d1, d2]
/// * `num_points` - Number of source points N
/// * `hessians` - [N × 36] output (6×6 per point, column-major for reduction)
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn compute_ndt_hessian_kernel_v2<F: Float>(
    source_points: &Array<F>,
    transform: &Array<F>,
    jacobians: &Array<F>,
    point_hessians: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    neighbor_indices: &Array<i32>,
    neighbor_counts: &Array<u32>,
    gauss_params: &Array<F>,
    num_points: u32,
    hessians: &mut Array<F>,
) {
    let gauss_d1 = gauss_params[0];
    let gauss_d2 = gauss_params[1];

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

    // Load point Jacobian (3x6 row-major) - 18 values total from SEPARATE buffer
    let jbase = point_idx * 18;
    let j_00 = jacobians[jbase];
    let j_01 = jacobians[jbase + 1];
    let j_02 = jacobians[jbase + 2];
    let j_03 = jacobians[jbase + 3];
    let j_04 = jacobians[jbase + 4];
    let j_05 = jacobians[jbase + 5];
    let j_10 = jacobians[jbase + 6];
    let j_11 = jacobians[jbase + 7];
    let j_12 = jacobians[jbase + 8];
    let j_13 = jacobians[jbase + 9];
    let j_14 = jacobians[jbase + 10];
    let j_15 = jacobians[jbase + 11];
    let j_20 = jacobians[jbase + 12];
    let j_21 = jacobians[jbase + 13];
    let j_22 = jacobians[jbase + 14];
    let j_23 = jacobians[jbase + 15];
    let j_24 = jacobians[jbase + 16];
    let j_25 = jacobians[jbase + 17];

    // Load point Hessian non-zero elements from SEPARATE buffer
    // Structure: 24x6 matrix, but only rows 13,14,17,18,21,22 have non-zero values
    // and only in columns 3,4,5
    let hbase = point_idx * 144;

    // Block 3 (roll): rows 12-15, non-zero in rows 13,14 cols 3-5
    let ph_13_3 = point_hessians[hbase + 13 * 6 + 3];
    let ph_13_4 = point_hessians[hbase + 13 * 6 + 4];
    let ph_13_5 = point_hessians[hbase + 13 * 6 + 5];
    let ph_14_3 = point_hessians[hbase + 14 * 6 + 3];
    let ph_14_4 = point_hessians[hbase + 14 * 6 + 4];
    let ph_14_5 = point_hessians[hbase + 14 * 6 + 5];

    // Block 4 (pitch): rows 16-19, non-zero in rows 16 (d1,e1), 17 (b2,d2,e2), 18 (b3,d3,e3)
    // Row 16 (x-component): cols 4,5 contain d1, e1
    let ph_16_4 = point_hessians[hbase + 16 * 6 + 4];
    let ph_16_5 = point_hessians[hbase + 16 * 6 + 5];
    let ph_17_3 = point_hessians[hbase + 17 * 6 + 3];
    let ph_17_4 = point_hessians[hbase + 17 * 6 + 4];
    let ph_17_5 = point_hessians[hbase + 17 * 6 + 5];
    let ph_18_3 = point_hessians[hbase + 18 * 6 + 3];
    let ph_18_4 = point_hessians[hbase + 18 * 6 + 4];
    let ph_18_5 = point_hessians[hbase + 18 * 6 + 5];

    // Block 5 (yaw): rows 20-23, non-zero in rows 20 (e1,f1), 21 (c2,e2,f2), 22 (c3,e3,f3)
    // Row 20 (x-component): cols 4,5 contain e1, f1
    let ph_20_4 = point_hessians[hbase + 20 * 6 + 4];
    let ph_20_5 = point_hessians[hbase + 20 * 6 + 5];
    let ph_21_3 = point_hessians[hbase + 21 * 6 + 3];
    let ph_21_4 = point_hessians[hbase + 21 * 6 + 4];
    let ph_21_5 = point_hessians[hbase + 21 * 6 + 5];
    let ph_22_3 = point_hessians[hbase + 22 * 6 + 3];
    let ph_22_4 = point_hessians[hbase + 22 * 6 + 4];
    let ph_22_5 = point_hessians[hbase + 22 * 6 + 5];

    // Initialize Hessian accumulators (21 unique values for symmetric 6x6)
    let mut h00 = F::new(0.0);
    let mut h01 = F::new(0.0);
    let mut h02 = F::new(0.0);
    let mut h03 = F::new(0.0);
    let mut h04 = F::new(0.0);
    let mut h05 = F::new(0.0);
    let mut h11 = F::new(0.0);
    let mut h12 = F::new(0.0);
    let mut h13 = F::new(0.0);
    let mut h14 = F::new(0.0);
    let mut h15 = F::new(0.0);
    let mut h22 = F::new(0.0);
    let mut h23 = F::new(0.0);
    let mut h24 = F::new(0.0);
    let mut h25 = F::new(0.0);
    let mut h33 = F::new(0.0);
    let mut h34 = F::new(0.0);
    let mut h35 = F::new(0.0);
    let mut h44 = F::new(0.0);
    let mut h45 = F::new(0.0);
    let mut h55 = F::new(0.0);

    let num_neighbors = neighbor_counts[point_idx];
    let neighbor_base = point_idx * MAX_NEIGHBORS;

    // Unrolled neighbor loop (up to MAX_NEIGHBORS = 8)
    // Neighbor 0
    if num_neighbors > 0 {
        let voxel_idx = neighbor_indices[neighbor_base];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 1
    if num_neighbors > 1 {
        let voxel_idx = neighbor_indices[neighbor_base + 1];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 2
    if num_neighbors > 2 {
        let voxel_idx = neighbor_indices[neighbor_base + 2];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 3
    if num_neighbors > 3 {
        let voxel_idx = neighbor_indices[neighbor_base + 3];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 4
    if num_neighbors > 4 {
        let voxel_idx = neighbor_indices[neighbor_base + 4];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 5
    if num_neighbors > 5 {
        let voxel_idx = neighbor_indices[neighbor_base + 5];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 6
    if num_neighbors > 6 {
        let voxel_idx = neighbor_indices[neighbor_base + 6];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Neighbor 7
    if num_neighbors > 7 {
        let voxel_idx = neighbor_indices[neighbor_base + 7];
        if voxel_idx >= 0 {
            accumulate_hessian_contribution(
                voxel_idx as u32,
                tx,
                ty,
                tz,
                j_00,
                j_01,
                j_02,
                j_03,
                j_04,
                j_05,
                j_10,
                j_11,
                j_12,
                j_13,
                j_14,
                j_15,
                j_20,
                j_21,
                j_22,
                j_23,
                j_24,
                j_25,
                ph_13_3,
                ph_13_4,
                ph_13_5,
                ph_14_3,
                ph_14_4,
                ph_14_5,
                ph_16_4,
                ph_16_5,
                ph_17_3,
                ph_17_4,
                ph_17_5,
                ph_18_3,
                ph_18_4,
                ph_18_5,
                ph_20_4,
                ph_20_5,
                ph_21_3,
                ph_21_4,
                ph_21_5,
                ph_22_3,
                ph_22_4,
                ph_22_5,
                voxel_means,
                voxel_inv_covs,
                gauss_d1,
                gauss_d2,
                &mut h00,
                &mut h01,
                &mut h02,
                &mut h03,
                &mut h04,
                &mut h05,
                &mut h11,
                &mut h12,
                &mut h13,
                &mut h14,
                &mut h15,
                &mut h22,
                &mut h23,
                &mut h24,
                &mut h25,
                &mut h33,
                &mut h34,
                &mut h35,
                &mut h44,
                &mut h45,
                &mut h55,
            );
        }
    }

    // Write output (full 6x6, column-major: component * num_points + point_idx)
    let np = num_points;
    // Row 0
    hessians[point_idx] = h00;
    hessians[np + point_idx] = h01;
    hessians[np * 2 + point_idx] = h02;
    hessians[np * 3 + point_idx] = h03;
    hessians[np * 4 + point_idx] = h04;
    hessians[np * 5 + point_idx] = h05;

    // Row 1 (symmetric)
    hessians[np * 6 + point_idx] = h01;
    hessians[np * 7 + point_idx] = h11;
    hessians[np * 8 + point_idx] = h12;
    hessians[np * 9 + point_idx] = h13;
    hessians[np * 10 + point_idx] = h14;
    hessians[np * 11 + point_idx] = h15;

    // Row 2
    hessians[np * 12 + point_idx] = h02;
    hessians[np * 13 + point_idx] = h12;
    hessians[np * 14 + point_idx] = h22;
    hessians[np * 15 + point_idx] = h23;
    hessians[np * 16 + point_idx] = h24;
    hessians[np * 17 + point_idx] = h25;

    // Row 3
    hessians[np * 18 + point_idx] = h03;
    hessians[np * 19 + point_idx] = h13;
    hessians[np * 20 + point_idx] = h23;
    hessians[np * 21 + point_idx] = h33;
    hessians[np * 22 + point_idx] = h34;
    hessians[np * 23 + point_idx] = h35;

    // Row 4
    hessians[np * 24 + point_idx] = h04;
    hessians[np * 25 + point_idx] = h14;
    hessians[np * 26 + point_idx] = h24;
    hessians[np * 27 + point_idx] = h34;
    hessians[np * 28 + point_idx] = h44;
    hessians[np * 29 + point_idx] = h45;

    // Row 5
    hessians[np * 30 + point_idx] = h05;
    hessians[np * 31 + point_idx] = h15;
    hessians[np * 32 + point_idx] = h25;
    hessians[np * 33 + point_idx] = h35;
    hessians[np * 34 + point_idx] = h45;
    hessians[np * 35 + point_idx] = h55;
}

/// Helper function to accumulate Hessian contribution from a single voxel.
///
/// This is a #[cube] function with fully unrolled computations to avoid
/// dynamic array indexing.
#[cube]
#[allow(clippy::too_many_arguments)]
fn accumulate_hessian_contribution<F: Float>(
    voxel_idx: u32,
    tx: F,
    ty: F,
    tz: F,
    // Jacobian elements (3x6 = 18 values)
    j_00: F,
    j_01: F,
    j_02: F,
    j_03: F,
    j_04: F,
    j_05: F,
    j_10: F,
    j_11: F,
    j_12: F,
    j_13: F,
    j_14: F,
    j_15: F,
    j_20: F,
    j_21: F,
    j_22: F,
    j_23: F,
    j_24: F,
    j_25: F,
    // Point Hessian non-zero elements (22 values)
    // Roll block: rows 13,14 (x-row 12 is all zeros for roll)
    ph_13_3: F,
    ph_13_4: F,
    ph_13_5: F,
    ph_14_3: F,
    ph_14_4: F,
    ph_14_5: F,
    // Pitch block: rows 16 (x-row has d1,e1), 17, 18
    ph_16_4: F,
    ph_16_5: F,
    ph_17_3: F,
    ph_17_4: F,
    ph_17_5: F,
    ph_18_3: F,
    ph_18_4: F,
    ph_18_5: F,
    // Yaw block: rows 20 (x-row has e1,f1), 21, 22
    ph_20_4: F,
    ph_20_5: F,
    ph_21_3: F,
    ph_21_4: F,
    ph_21_5: F,
    ph_22_3: F,
    ph_22_4: F,
    ph_22_5: F,
    // Voxel data
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    gauss_d1: F,
    gauss_d2: F,
    // Hessian accumulators (21 unique values)
    h00: &mut F,
    h01: &mut F,
    h02: &mut F,
    h03: &mut F,
    h04: &mut F,
    h05: &mut F,
    h11: &mut F,
    h12: &mut F,
    h13: &mut F,
    h14: &mut F,
    h15: &mut F,
    h22: &mut F,
    h23: &mut F,
    h24: &mut F,
    h25: &mut F,
    h33: &mut F,
    h34: &mut F,
    h35: &mut F,
    h44: &mut F,
    h45: &mut F,
    h55: &mut F,
) {
    // Load voxel mean
    let vbase = voxel_idx * 3;
    let x0 = tx - voxel_means[vbase];
    let x1 = ty - voxel_means[vbase + 1];
    let x2 = tz - voxel_means[vbase + 2];

    // Load inverse covariance (3x3 row-major)
    let cbase = voxel_idx * 9;
    let c00 = voxel_inv_covs[cbase];
    let c01 = voxel_inv_covs[cbase + 1];
    let c02 = voxel_inv_covs[cbase + 2];
    let c10 = voxel_inv_covs[cbase + 3];
    let c11 = voxel_inv_covs[cbase + 4];
    let c12 = voxel_inv_covs[cbase + 5];
    let c20 = voxel_inv_covs[cbase + 6];
    let c21 = voxel_inv_covs[cbase + 7];
    let c22 = voxel_inv_covs[cbase + 8];

    // Compute Σ⁻¹ * x = (cx0, cx1, cx2)
    let cx0 = c00 * x0 + c01 * x1 + c02 * x2;
    let cx1 = c10 * x0 + c11 * x1 + c12 * x2;
    let cx2 = c20 * x0 + c21 * x1 + c22 * x2;

    // x' * Σ⁻¹ * x
    let x_c_x = x0 * cx0 + x1 * cx1 + x2 * cx2;

    // e_x_cov_x = d1 * d2 * exp(-d2/2 * x'Σ⁻¹x)
    // Autoware checks validity BEFORE multiplying by d1 (which is negative)
    let exponent = gauss_d2 * x_c_x * F::new(-0.5);
    let e_x_cov_x_check = gauss_d2 * F::exp(exponent);

    // Check for valid value before applying d1
    // e_x_cov_x_check = d2 * exp(-d2/2 * x'Σ⁻¹x) should be in [0, 1]
    let is_valid = e_x_cov_x_check <= F::new(1.0) && e_x_cov_x_check >= F::new(0.0);
    // Now apply d1 for the actual computation (d1 < 0, so result is negative)
    let e_x_cov_x = gauss_d1 * e_x_cov_x_check;
    if is_valid {
        // Compute cov_dxd_i = (Σ⁻¹x)' * J[:,i] = cx' * J[:,i]
        // This is the gradient coefficient for pose parameter i
        let cov_dxd_0 = cx0 * j_00 + cx1 * j_10 + cx2 * j_20;
        let cov_dxd_1 = cx0 * j_01 + cx1 * j_11 + cx2 * j_21;
        let cov_dxd_2 = cx0 * j_02 + cx1 * j_12 + cx2 * j_22;
        let cov_dxd_3 = cx0 * j_03 + cx1 * j_13 + cx2 * j_23;
        let cov_dxd_4 = cx0 * j_04 + cx1 * j_14 + cx2 * j_24;
        let cov_dxd_5 = cx0 * j_05 + cx1 * j_15 + cx2 * j_25;

        // Compute Σ⁻¹ * J (3x6 matrix, call it cj)
        // cj[row][col] = sum_k C[row][k] * J[k][col]
        let cj_00 = c00 * j_00 + c01 * j_10 + c02 * j_20;
        let cj_01 = c00 * j_01 + c01 * j_11 + c02 * j_21;
        let cj_02 = c00 * j_02 + c01 * j_12 + c02 * j_22;
        let cj_03 = c00 * j_03 + c01 * j_13 + c02 * j_23;
        let cj_04 = c00 * j_04 + c01 * j_14 + c02 * j_24;
        let cj_05 = c00 * j_05 + c01 * j_15 + c02 * j_25;

        let cj_10 = c10 * j_00 + c11 * j_10 + c12 * j_20;
        let cj_11 = c10 * j_01 + c11 * j_11 + c12 * j_21;
        let cj_12 = c10 * j_02 + c11 * j_12 + c12 * j_22;
        let cj_13 = c10 * j_03 + c11 * j_13 + c12 * j_23;
        let cj_14 = c10 * j_04 + c11 * j_14 + c12 * j_24;
        let cj_15 = c10 * j_05 + c11 * j_15 + c12 * j_25;

        let cj_20 = c20 * j_00 + c21 * j_10 + c22 * j_20;
        let cj_21 = c20 * j_01 + c21 * j_11 + c22 * j_21;
        let cj_22 = c20 * j_02 + c21 * j_12 + c22 * j_22;
        let cj_23 = c20 * j_03 + c21 * j_13 + c22 * j_23;
        let cj_24 = c20 * j_04 + c21 * j_14 + c22 * j_24;
        let cj_25 = c20 * j_05 + c21 * j_15 + c22 * j_25;

        // Compute J' * Σ⁻¹ * J (6x6 symmetric matrix)
        // jtcj[i][j] = J[:,i]' * cj[:,j]
        let jtcj_00 = j_00 * cj_00 + j_10 * cj_10 + j_20 * cj_20;
        let jtcj_01 = j_00 * cj_01 + j_10 * cj_11 + j_20 * cj_21;
        let jtcj_02 = j_00 * cj_02 + j_10 * cj_12 + j_20 * cj_22;
        let jtcj_03 = j_00 * cj_03 + j_10 * cj_13 + j_20 * cj_23;
        let jtcj_04 = j_00 * cj_04 + j_10 * cj_14 + j_20 * cj_24;
        let jtcj_05 = j_00 * cj_05 + j_10 * cj_15 + j_20 * cj_25;

        let jtcj_11 = j_01 * cj_01 + j_11 * cj_11 + j_21 * cj_21;
        let jtcj_12 = j_01 * cj_02 + j_11 * cj_12 + j_21 * cj_22;
        let jtcj_13 = j_01 * cj_03 + j_11 * cj_13 + j_21 * cj_23;
        let jtcj_14 = j_01 * cj_04 + j_11 * cj_14 + j_21 * cj_24;
        let jtcj_15 = j_01 * cj_05 + j_11 * cj_15 + j_21 * cj_25;

        let jtcj_22 = j_02 * cj_02 + j_12 * cj_12 + j_22 * cj_22;
        let jtcj_23 = j_02 * cj_03 + j_12 * cj_13 + j_22 * cj_23;
        let jtcj_24 = j_02 * cj_04 + j_12 * cj_14 + j_22 * cj_24;
        let jtcj_25 = j_02 * cj_05 + j_12 * cj_15 + j_22 * cj_25;

        let jtcj_33 = j_03 * cj_03 + j_13 * cj_13 + j_23 * cj_23;
        let jtcj_34 = j_03 * cj_04 + j_13 * cj_14 + j_23 * cj_24;
        let jtcj_35 = j_03 * cj_05 + j_13 * cj_15 + j_23 * cj_25;

        let jtcj_44 = j_04 * cj_04 + j_14 * cj_14 + j_24 * cj_24;
        let jtcj_45 = j_04 * cj_05 + j_14 * cj_15 + j_24 * cj_25;

        let jtcj_55 = j_05 * cj_05 + j_15 * cj_15 + j_25 * cj_25;

        // Compute x'Σ⁻¹ * H_block terms (xch)
        // xch[i][j] = cx' * H_block[i][:,j] = cx0 * ph[i*4, j] + cx1 * ph[i*4+1, j] + cx2 * ph[i*4+2, j]
        // Only non-zero for i >= 3, j >= 3 due to point Hessian structure
        //
        // For block i (pose parameter i), the point Hessian has structure:
        // - Blocks 0,1,2 (translation): all zeros
        // - Block 3 (roll): rows 12-15, x-row is zero, y,z rows non-zero in cols 3-5
        // - Block 4 (pitch): rows 16-19, x-row non-zero in cols 4,5 (d1,e1), y,z rows non-zero in cols 3-5
        // - Block 5 (yaw): rows 20-23, x-row non-zero in cols 4,5 (e1,f1), y,z rows non-zero in cols 3-5

        // Block 3 (roll): row 12 is all zeros, so only need y,z components
        let xch_33 = cx1 * ph_13_3 + cx2 * ph_14_3;
        let xch_34 = cx1 * ph_13_4 + cx2 * ph_14_4;
        let xch_35 = cx1 * ph_13_5 + cx2 * ph_14_5;

        // Block 4 (pitch): row 16 has non-zero entries in cols 4,5 (d1, e1)
        let xch_43 = cx1 * ph_17_3 + cx2 * ph_18_3; // col 3: row 16 is zero
        let xch_44 = cx0 * ph_16_4 + cx1 * ph_17_4 + cx2 * ph_18_4; // col 4: includes d1
        let xch_45 = cx0 * ph_16_5 + cx1 * ph_17_5 + cx2 * ph_18_5; // col 5: includes e1

        // Block 5 (yaw): row 20 has non-zero entries in cols 4,5 (e1, f1)
        let xch_53 = cx1 * ph_21_3 + cx2 * ph_22_3; // col 3: row 20 is zero
        let xch_54 = cx0 * ph_20_4 + cx1 * ph_21_4 + cx2 * ph_22_4; // col 4: includes e1
        let xch_55 = cx0 * ph_20_5 + cx1 * ph_21_5 + cx2 * ph_22_5; // col 5: includes f1

        // Accumulate Hessian: H[i,j] += e_x_cov_x * (-d2 * cov_dxd[i] * cov_dxd[j] + xch[i][j] + jtcj[i][j])
        let neg_d2 = F::new(0.0) - gauss_d2;

        // Row 0: H[0,0..5]
        *h00 += e_x_cov_x * (neg_d2 * cov_dxd_0 * cov_dxd_0 + jtcj_00);
        *h01 += e_x_cov_x * (neg_d2 * cov_dxd_0 * cov_dxd_1 + jtcj_01);
        *h02 += e_x_cov_x * (neg_d2 * cov_dxd_0 * cov_dxd_2 + jtcj_02);
        *h03 += e_x_cov_x * (neg_d2 * cov_dxd_0 * cov_dxd_3 + jtcj_03);
        *h04 += e_x_cov_x * (neg_d2 * cov_dxd_0 * cov_dxd_4 + jtcj_04);
        *h05 += e_x_cov_x * (neg_d2 * cov_dxd_0 * cov_dxd_5 + jtcj_05);

        // Row 1: H[1,1..5]
        *h11 += e_x_cov_x * (neg_d2 * cov_dxd_1 * cov_dxd_1 + jtcj_11);
        *h12 += e_x_cov_x * (neg_d2 * cov_dxd_1 * cov_dxd_2 + jtcj_12);
        *h13 += e_x_cov_x * (neg_d2 * cov_dxd_1 * cov_dxd_3 + jtcj_13);
        *h14 += e_x_cov_x * (neg_d2 * cov_dxd_1 * cov_dxd_4 + jtcj_14);
        *h15 += e_x_cov_x * (neg_d2 * cov_dxd_1 * cov_dxd_5 + jtcj_15);

        // Row 2: H[2,2..5]
        *h22 += e_x_cov_x * (neg_d2 * cov_dxd_2 * cov_dxd_2 + jtcj_22);
        *h23 += e_x_cov_x * (neg_d2 * cov_dxd_2 * cov_dxd_3 + jtcj_23);
        *h24 += e_x_cov_x * (neg_d2 * cov_dxd_2 * cov_dxd_4 + jtcj_24);
        *h25 += e_x_cov_x * (neg_d2 * cov_dxd_2 * cov_dxd_5 + jtcj_25);

        // Row 3: H[3,3..5] - includes xch terms
        // For off-diagonal terms H[i,j], both xch[i,j] and xch[j,i] contribute
        // (from x'Σ⁻¹H_i[:,j] and x'Σ⁻¹H_j[:,i] in Magnusson 2009 Eq. 6.13)
        *h33 += e_x_cov_x * (neg_d2 * cov_dxd_3 * cov_dxd_3 + xch_33 + jtcj_33);
        *h34 += e_x_cov_x * (neg_d2 * cov_dxd_3 * cov_dxd_4 + xch_34 + xch_43 + jtcj_34);
        *h35 += e_x_cov_x * (neg_d2 * cov_dxd_3 * cov_dxd_5 + xch_35 + xch_53 + jtcj_35);

        // Row 4: H[4,4..5] - includes xch terms
        *h44 += e_x_cov_x * (neg_d2 * cov_dxd_4 * cov_dxd_4 + xch_44 + jtcj_44);
        *h45 += e_x_cov_x * (neg_d2 * cov_dxd_4 * cov_dxd_5 + xch_45 + xch_54 + jtcj_45);

        // Row 5: H[5,5] - includes xch term
        *h55 += e_x_cov_x * (neg_d2 * cov_dxd_5 * cov_dxd_5 + xch_55 + jtcj_55);
    }
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
        //
        // Formulas from angular.rs (Magnusson 2009 Equation 6.19)
        // angular.rs uses: sx=sin(roll), cx=cos(roll), sy=sin(pitch), cy=cos(pitch), sz=sin(yaw), cz=cos(yaw)
        // Here we use:     sr=sin(roll), cr=cos(roll), sp=sin(pitch), cp=cos(pitch), sy=sin(yaw), cy=cos(yaw)
        // Mapping: angular sx->sr, cx->cr, sy->sp, cy->cp, sz->sy, cz->cy

        // Row 0: ∂x'/∂(tx, ty, tz, roll, pitch, yaw)
        jacobians.push(1.0_f32); // ∂x'/∂tx
        jacobians.push(0.0_f32); // ∂x'/∂ty
        jacobians.push(0.0_f32); // ∂x'/∂tz
        jacobians.push(0.0_f32); // ∂x'/∂roll = 0
                                 // ∂x'/∂pitch: j_ang[(2,:)] = [-sp*cy, sp*sy, cp]
        jacobians.push(((-sp * cy) * x + (sp * sy) * y + cp * z) as f32);
        // ∂x'/∂yaw: j_ang[(5,:)] = [-cp*sy, -cp*cy, 0]
        jacobians.push(((-cp * sy) * x + (-cp * cy) * y) as f32);

        // Row 1: ∂y'/∂(tx, ty, tz, roll, pitch, yaw)
        jacobians.push(0.0_f32); // ∂y'/∂tx
        jacobians.push(1.0_f32); // ∂y'/∂ty
        jacobians.push(0.0_f32); // ∂y'/∂tz
                                 // ∂y'/∂roll: j_ang[(0,:)] = [-sr*sy + cr*sp*cy, -sr*cy - cr*sp*sy, -cr*cp]
        jacobians.push(
            ((-sr * sy + cr * sp * cy) * x + (-sr * cy - cr * sp * sy) * y + (-cr * cp) * z) as f32,
        );
        // ∂y'/∂pitch: j_ang[(3,:)] = [sr*cp*cy, -sr*cp*sy, sr*sp]
        jacobians.push(((sr * cp * cy) * x + (-sr * cp * sy) * y + (sr * sp) * z) as f32);
        // ∂y'/∂yaw: j_ang[(6,:)] = [cr*cy - sr*sp*sy, -cr*sy - sr*sp*cy, 0]
        jacobians.push(((cr * cy - sr * sp * sy) * x + (-cr * sy - sr * sp * cy) * y) as f32);

        // Row 2: ∂z'/∂(tx, ty, tz, roll, pitch, yaw)
        jacobians.push(0.0_f32); // ∂z'/∂tx
        jacobians.push(0.0_f32); // ∂z'/∂ty
        jacobians.push(1.0_f32); // ∂z'/∂tz
                                 // ∂z'/∂roll: j_ang[(1,:)] = [cr*sy + sr*sp*cy, cr*cy - sr*sp*sy, -sr*cp]
        jacobians.push(
            ((cr * sy + sr * sp * cy) * x + (cr * cy - sr * sp * sy) * y + (-sr * cp) * z) as f32,
        );
        // ∂z'/∂pitch: j_ang[(4,:)] = [-cr*cp*cy, cr*cp*sy, -cr*sp]
        jacobians.push(((-cr * cp * cy) * x + (cr * cp * sy) * y + (-cr * sp) * z) as f32);
        // ∂z'/∂yaw: j_ang[(7,:)] = [sr*cy + cr*sp*sy, cr*sp*cy - sr*sy, 0]
        jacobians.push(((sr * cy + cr * sp * sy) * x + (cr * sp * cy - sr * sy) * y) as f32);
    }

    jacobians
}

/// Compute point Hessians on CPU for GPU kernel.
///
/// Returns a flattened vector of 24x6 Hessian matrices for each point.
/// The structure matches the CPU point_hessian from angular derivatives.
///
/// # Layout
/// Each point has 144 floats (24 rows × 6 columns):
/// - Rows 0-11: translation Hessians (all zeros, second derivatives of translation are 0)
/// - Rows 12-23: angular Hessians (computed from h_ang × point)
///
/// The angular Hessian terms are arranged as:
/// - Block 3 (rows 12-15): ∂²T/∂roll² and ∂²T/∂roll∂pitch
/// - Block 4 (rows 16-19): ∂²T/∂pitch² and ∂²T/∂pitch∂yaw
/// - Block 5 (rows 20-23): ∂²T/∂yaw² and remaining cross terms
pub fn compute_point_hessians_cpu(source_points: &[[f32; 3]], pose: &[f64; 6]) -> Vec<f32> {
    let (roll, pitch, yaw) = (pose[3], pose[4], pose[5]);

    // Precompute trig values
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    // Precompute angular Hessian (h_ang) - 15 terms × 3 components
    // These are the second derivatives of the rotation matrix w.r.t. angles
    // a2, a3: ∂²R/∂roll² (y, z components)
    let a2 = [-cr * sy - sr * sp * cy, -cr * cy + sr * sp * sy, sr * cp];
    let a3 = [-sr * sy + cr * sp * cy, -cr * sp * sy - sr * cy, -cr * cp];

    // b2, b3: ∂²R/∂roll∂pitch
    let b2 = [cr * cp * cy, -cr * cp * sy, cr * sp];
    let b3 = [sr * cp * cy, -sr * cp * sy, sr * sp];

    // c2, c3: ∂²R/∂roll∂yaw
    let c2 = [-sr * cy - cr * sp * sy, sr * sy - cr * sp * cy, 0.0];
    let c3 = [cr * cy - sr * sp * sy, -sr * sp * cy - cr * sy, 0.0];

    // d1, d2, d3: ∂²R/∂pitch²
    let d1 = [-cp * cy, cp * sy, sp];
    let d2 = [-sr * sp * cy, sr * sp * sy, sr * cp];
    let d3 = [cr * sp * cy, -cr * sp * sy, -cr * cp];

    // e1, e2, e3: ∂²R/∂pitch∂yaw
    let e1 = [sp * sy, sp * cy, 0.0];
    let e2 = [-sr * cp * sy, -sr * cp * cy, 0.0];
    let e3 = [cr * cp * sy, cr * cp * cy, 0.0];

    // f1, f2, f3: ∂²R/∂yaw²
    let f1 = [-cp * cy, cp * sy, 0.0];
    let f2 = [-cr * sy - sr * sp * cy, -cr * cy + sr * sp * sy, 0.0];
    let f3 = [-sr * sy + cr * sp * cy, -cr * sp * sy - sr * cy, 0.0];

    let mut hessians = Vec::with_capacity(source_points.len() * 144);

    for point in source_points {
        let (x, y, z) = (point[0] as f64, point[1] as f64, point[2] as f64);

        // The 24x6 point Hessian matrix
        // Rows 0-11: zeros (translation second derivatives)
        // Rows 12-23: angular second derivatives

        // We output row-major: 24 rows × 6 columns = 144 values
        // For GPU efficiency, we only store the non-zero portions

        // Rows 0-11: all zeros (translation has no second derivatives)
        hessians.extend(std::iter::repeat_n(0.0_f32, 72));

        // Rows 12-15: Block for roll-roll, roll-pitch, roll-yaw in column 3,4,5
        // Row 12: zeros (x component of roll-* terms)
        hessians.extend(std::iter::repeat_n(0.0_f32, 6));

        // Row 13: a2·point for col 3, b2·point for col 4, c2·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 3));
        hessians.push((a2[0] * x + a2[1] * y + a2[2] * z) as f32); // col 3
        hessians.push((b2[0] * x + b2[1] * y + b2[2] * z) as f32); // col 4
        hessians.push((c2[0] * x + c2[1] * y + c2[2] * z) as f32); // col 5

        // Row 14: a3·point for col 3, b3·point for col 4, c3·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 3));
        hessians.push((a3[0] * x + a3[1] * y + a3[2] * z) as f32); // col 3
        hessians.push((b3[0] * x + b3[1] * y + b3[2] * z) as f32); // col 4
        hessians.push((c3[0] * x + c3[1] * y + c3[2] * z) as f32); // col 5

        // Row 15: zeros (w component)
        hessians.extend(std::iter::repeat_n(0.0_f32, 6));

        // Rows 16-19: Block for pitch-roll, pitch-pitch, pitch-yaw
        // Row 16: d1·point for col 4, e1·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 4));
        hessians.push((d1[0] * x + d1[1] * y + d1[2] * z) as f32); // col 4
        hessians.push((e1[0] * x + e1[1] * y + e1[2] * z) as f32); // col 5

        // Row 17: b2·point for col 3, d2·point for col 4, e2·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 3));
        hessians.push((b2[0] * x + b2[1] * y + b2[2] * z) as f32); // col 3
        hessians.push((d2[0] * x + d2[1] * y + d2[2] * z) as f32); // col 4
        hessians.push((e2[0] * x + e2[1] * y + e2[2] * z) as f32); // col 5

        // Row 18: b3·point for col 3, d3·point for col 4, e3·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 3));
        hessians.push((b3[0] * x + b3[1] * y + b3[2] * z) as f32); // col 3
        hessians.push((d3[0] * x + d3[1] * y + d3[2] * z) as f32); // col 4
        hessians.push((e3[0] * x + e3[1] * y + e3[2] * z) as f32); // col 5

        // Row 19: zeros (w component)
        hessians.extend(std::iter::repeat_n(0.0_f32, 6));

        // Rows 20-23: Block for yaw-roll, yaw-pitch, yaw-yaw
        // Row 20: e1·point for col 4, f1·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 4));
        hessians.push((e1[0] * x + e1[1] * y + e1[2] * z) as f32); // col 4
        hessians.push((f1[0] * x + f1[1] * y + f1[2] * z) as f32); // col 5

        // Row 21: c2·point for col 3, e2·point for col 4, f2·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 3));
        hessians.push((c2[0] * x + c2[1] * y + c2[2] * z) as f32); // col 3
        hessians.push((e2[0] * x + e2[1] * y + e2[2] * z) as f32); // col 4
        hessians.push((f2[0] * x + f2[1] * y + f2[2] * z) as f32); // col 5

        // Row 22: c3·point for col 3, e3·point for col 4, f3·point for col 5
        hessians.extend(std::iter::repeat_n(0.0_f32, 3));
        hessians.push((c3[0] * x + c3[1] * y + c3[2] * z) as f32); // col 3
        hessians.push((e3[0] * x + e3[1] * y + e3[2] * z) as f32); // col 4
        hessians.push((f3[0] * x + f3[1] * y + f3[2] * z) as f32); // col 5

        // Row 23: zeros (w component)
        hessians.extend(std::iter::repeat_n(0.0_f32, 6));
    }

    hessians
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
    /// Flattened principal axes [V * 3] (surface normals for point-to-plane)
    pub principal_axes: Vec<f32>,
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
        let principal_axes = grid.principal_axes_flat();
        let valid: Vec<u32> = grid.voxels().iter().map(|_| 1u32).collect();
        let num_voxels = grid.len();

        Self {
            means,
            inv_covariances,
            principal_axes,
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
    /// Hessian (6x6 matrix) - sum across all points, row-major
    pub hessian: [[f64; 6]; 6],
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
    use crate::derivatives::AngularDerivatives;

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

    #[test]
    fn test_jacobians_match_angular_derivatives() {
        use crate::derivatives::AngularDerivatives;

        // Test with a non-trivial pose
        let pose = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]; // tx, ty, tz, roll, pitch, yaw

        // Test points
        let points = vec![
            [1.0f32, 2.0, 3.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ];

        // Compute Jacobians using GPU function
        let gpu_jacobians = compute_point_jacobians_cpu(&points, &pose);

        // Compute Jacobians using Angular derivatives (CPU reference)
        let angular = AngularDerivatives::new(pose[3], pose[4], pose[5], false);

        println!("=== Jacobian Comparison ===");
        println!(
            "Pose: roll={:.3}, pitch={:.3}, yaw={:.3}",
            pose[3], pose[4], pose[5]
        );

        for (i, point) in points.iter().enumerate() {
            println!(
                "\nPoint {i}: [{:.1}, {:.1}, {:.1}]",
                point[0], point[1], point[2]
            );

            let point_f64 = [point[0] as f64, point[1] as f64, point[2] as f64];
            let angular_terms = angular.compute_point_gradient_terms(&point_f64);

            // GPU Jacobian for this point (3x6 row-major, starting at i*18)
            let jbase = i * 18;

            // Compare rotation columns (3, 4, 5)
            // Column 3 (roll): GPU [jbase+3, jbase+9, jbase+15] vs Angular [-, 0, 1]
            // Angular gives [∂y'/∂roll, ∂z'/∂roll] (∂x'/∂roll = 0)
            let gpu_dy_droll = gpu_jacobians[jbase + 9]; // J[1,3]
            let gpu_dz_droll = gpu_jacobians[jbase + 15]; // J[2,3]
            let cpu_dy_droll = angular_terms[0] as f32;
            let cpu_dz_droll = angular_terms[1] as f32;

            println!(
                "  Roll:  GPU=[{:.4}, {:.4}], CPU=[{:.4}, {:.4}], diff=[{:.4}, {:.4}]",
                gpu_dy_droll,
                gpu_dz_droll,
                cpu_dy_droll,
                cpu_dz_droll,
                (gpu_dy_droll - cpu_dy_droll).abs(),
                (gpu_dz_droll - cpu_dz_droll).abs()
            );

            // Column 4 (pitch): GPU [jbase+4, jbase+10, jbase+16] vs Angular [2, 3, 4]
            let gpu_dx_dpitch = gpu_jacobians[jbase + 4]; // J[0,4]
            let gpu_dy_dpitch = gpu_jacobians[jbase + 10]; // J[1,4]
            let gpu_dz_dpitch = gpu_jacobians[jbase + 16]; // J[2,4]
            let cpu_dx_dpitch = angular_terms[2] as f32;
            let cpu_dy_dpitch = angular_terms[3] as f32;
            let cpu_dz_dpitch = angular_terms[4] as f32;

            println!(
                "  Pitch: GPU=[{:.4}, {:.4}, {:.4}], CPU=[{:.4}, {:.4}, {:.4}]",
                gpu_dx_dpitch,
                gpu_dy_dpitch,
                gpu_dz_dpitch,
                cpu_dx_dpitch,
                cpu_dy_dpitch,
                cpu_dz_dpitch
            );
            println!(
                "         diff=[{:.4}, {:.4}, {:.4}]",
                (gpu_dx_dpitch - cpu_dx_dpitch).abs(),
                (gpu_dy_dpitch - cpu_dy_dpitch).abs(),
                (gpu_dz_dpitch - cpu_dz_dpitch).abs()
            );

            // Column 5 (yaw): GPU [jbase+5, jbase+11, jbase+17] vs Angular [5, 6, 7]
            let gpu_dx_dyaw = gpu_jacobians[jbase + 5]; // J[0,5]
            let gpu_dy_dyaw = gpu_jacobians[jbase + 11]; // J[1,5]
            let gpu_dz_dyaw = gpu_jacobians[jbase + 17]; // J[2,5]
            let cpu_dx_dyaw = angular_terms[5] as f32;
            let cpu_dy_dyaw = angular_terms[6] as f32;
            let cpu_dz_dyaw = angular_terms[7] as f32;

            println!(
                "  Yaw:   GPU=[{:.4}, {:.4}, {:.4}], CPU=[{:.4}, {:.4}, {:.4}]",
                gpu_dx_dyaw, gpu_dy_dyaw, gpu_dz_dyaw, cpu_dx_dyaw, cpu_dy_dyaw, cpu_dz_dyaw
            );
            println!(
                "         diff=[{:.4}, {:.4}, {:.4}]",
                (gpu_dx_dyaw - cpu_dx_dyaw).abs(),
                (gpu_dy_dyaw - cpu_dy_dyaw).abs(),
                (gpu_dz_dyaw - cpu_dz_dyaw).abs()
            );

            // Assert rotation gradients match
            let eps = 1e-3;
            assert!(
                (gpu_dy_droll - cpu_dy_droll).abs() < eps,
                "∂y'/∂roll mismatch: GPU={gpu_dy_droll}, CPU={cpu_dy_droll}"
            );
            assert!(
                (gpu_dz_droll - cpu_dz_droll).abs() < eps,
                "∂z'/∂roll mismatch: GPU={gpu_dz_droll}, CPU={cpu_dz_droll}"
            );
            assert!(
                (gpu_dx_dpitch - cpu_dx_dpitch).abs() < eps,
                "∂x'/∂pitch mismatch: GPU={gpu_dx_dpitch}, CPU={cpu_dx_dpitch}"
            );
            assert!(
                (gpu_dy_dpitch - cpu_dy_dpitch).abs() < eps,
                "∂y'/∂pitch mismatch: GPU={gpu_dy_dpitch}, CPU={cpu_dy_dpitch}"
            );
            assert!(
                (gpu_dz_dpitch - cpu_dz_dpitch).abs() < eps,
                "∂z'/∂pitch mismatch: GPU={gpu_dz_dpitch}, CPU={cpu_dz_dpitch}"
            );
            assert!(
                (gpu_dx_dyaw - cpu_dx_dyaw).abs() < eps,
                "∂x'/∂yaw mismatch: GPU={gpu_dx_dyaw}, CPU={cpu_dx_dyaw}"
            );
            assert!(
                (gpu_dy_dyaw - cpu_dy_dyaw).abs() < eps,
                "∂y'/∂yaw mismatch: GPU={gpu_dy_dyaw}, CPU={cpu_dy_dyaw}"
            );
            assert!(
                (gpu_dz_dyaw - cpu_dz_dyaw).abs() < eps,
                "∂z'/∂yaw mismatch: GPU={gpu_dz_dyaw}, CPU={cpu_dz_dyaw}"
            );
        }
    }

    #[test]
    fn test_point_hessians_match_angular_derivatives() {
        // Verify that compute_point_hessians_cpu matches AngularDerivatives.compute_point_hessian_terms
        let pose = [0.0, 0.0, 0.0, 0.3, 0.2, 0.1]; // tx, ty, tz, roll, pitch, yaw

        // Test multiple points
        let points: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [5.0, 0.0, 0.0],
        ];

        // Compute Hessians using GPU function
        let gpu_hessians = compute_point_hessians_cpu(&points, &pose);

        // Compute Hessians using Angular derivatives (CPU reference)
        let angular = AngularDerivatives::new(pose[3], pose[4], pose[5], true);

        println!("=== Point Hessian Comparison ===");
        println!(
            "Pose: roll={:.3}, pitch={:.3}, yaw={:.3}",
            pose[3], pose[4], pose[5]
        );

        for (i, point) in points.iter().enumerate() {
            println!(
                "\nPoint {i}: [{:.1}, {:.1}, {:.1}]",
                point[0], point[1], point[2]
            );

            let point_f64 = [point[0] as f64, point[1] as f64, point[2] as f64];
            let angular_terms = angular.compute_point_hessian_terms(&point_f64);

            // GPU Hessian for this point (24x6 row-major, starting at i*144)
            let hbase = i * 144;

            // Extract the 15 angular Hessian terms from GPU output
            // The GPU stores them in a 24x6 matrix with specific layout:
            // Row 13: [0,0,0, a2·p, b2·p, c2·p]
            // Row 14: [0,0,0, a3·p, b3·p, c3·p]
            // Row 16: [0,0,0,0, d1·p, e1·p]
            // Row 17: [0,0,0, b2·p, d2·p, e2·p]
            // Row 18: [0,0,0, b3·p, d3·p, e3·p]
            // Row 20: [0,0,0,0, e1·p, f1·p]  - note: e1 reused here
            // Row 21: [0,0,0, c2·p, e2·p, f2·p]
            // Row 22: [0,0,0, c3·p, e3·p, f3·p]

            // Extract GPU values (row-major indexing: row*6 + col)
            let gpu_a2 = gpu_hessians[hbase + 13 * 6 + 3];
            let gpu_a3 = gpu_hessians[hbase + 14 * 6 + 3];
            let gpu_b2 = gpu_hessians[hbase + 13 * 6 + 4];
            let gpu_b3 = gpu_hessians[hbase + 14 * 6 + 4];
            let gpu_c2 = gpu_hessians[hbase + 13 * 6 + 5];
            let gpu_c3 = gpu_hessians[hbase + 14 * 6 + 5];
            let gpu_d1 = gpu_hessians[hbase + 16 * 6 + 4];
            let gpu_d2 = gpu_hessians[hbase + 17 * 6 + 4];
            let gpu_d3 = gpu_hessians[hbase + 18 * 6 + 4];
            let gpu_e1 = gpu_hessians[hbase + 16 * 6 + 5];
            let gpu_e2 = gpu_hessians[hbase + 17 * 6 + 5];
            let gpu_e3 = gpu_hessians[hbase + 18 * 6 + 5];
            let gpu_f1 = gpu_hessians[hbase + 20 * 6 + 5];
            let gpu_f2 = gpu_hessians[hbase + 21 * 6 + 5];
            let gpu_f3 = gpu_hessians[hbase + 22 * 6 + 5];

            // Angular terms order: [a2, a3, b2, b3, c2, c3, d1, d2, d3, e1, e2, e3, f1, f2, f3]
            let cpu_a2 = angular_terms[0] as f32;
            let cpu_a3 = angular_terms[1] as f32;
            let cpu_b2 = angular_terms[2] as f32;
            let cpu_b3 = angular_terms[3] as f32;
            let cpu_c2 = angular_terms[4] as f32;
            let cpu_c3 = angular_terms[5] as f32;
            let cpu_d1 = angular_terms[6] as f32;
            let cpu_d2 = angular_terms[7] as f32;
            let cpu_d3 = angular_terms[8] as f32;
            let cpu_e1 = angular_terms[9] as f32;
            let cpu_e2 = angular_terms[10] as f32;
            let cpu_e3 = angular_terms[11] as f32;
            let cpu_f1 = angular_terms[12] as f32;
            let cpu_f2 = angular_terms[13] as f32;
            let cpu_f3 = angular_terms[14] as f32;

            // Print comparison
            println!(
                "  a2: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_a2,
                cpu_a2,
                (gpu_a2 - cpu_a2).abs()
            );
            println!(
                "  a3: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_a3,
                cpu_a3,
                (gpu_a3 - cpu_a3).abs()
            );
            println!(
                "  b2: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_b2,
                cpu_b2,
                (gpu_b2 - cpu_b2).abs()
            );
            println!(
                "  b3: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_b3,
                cpu_b3,
                (gpu_b3 - cpu_b3).abs()
            );
            println!(
                "  c2: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_c2,
                cpu_c2,
                (gpu_c2 - cpu_c2).abs()
            );
            println!(
                "  c3: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_c3,
                cpu_c3,
                (gpu_c3 - cpu_c3).abs()
            );
            println!(
                "  d1: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_d1,
                cpu_d1,
                (gpu_d1 - cpu_d1).abs()
            );
            println!(
                "  d2: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_d2,
                cpu_d2,
                (gpu_d2 - cpu_d2).abs()
            );
            println!(
                "  d3: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_d3,
                cpu_d3,
                (gpu_d3 - cpu_d3).abs()
            );
            println!(
                "  e1: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_e1,
                cpu_e1,
                (gpu_e1 - cpu_e1).abs()
            );
            println!(
                "  e2: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_e2,
                cpu_e2,
                (gpu_e2 - cpu_e2).abs()
            );
            println!(
                "  e3: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_e3,
                cpu_e3,
                (gpu_e3 - cpu_e3).abs()
            );
            println!(
                "  f1: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_f1,
                cpu_f1,
                (gpu_f1 - cpu_f1).abs()
            );
            println!(
                "  f2: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_f2,
                cpu_f2,
                (gpu_f2 - cpu_f2).abs()
            );
            println!(
                "  f3: GPU={:.4}, CPU={:.4}, diff={:.4}",
                gpu_f3,
                cpu_f3,
                (gpu_f3 - cpu_f3).abs()
            );

            // Assert all terms match
            let eps = 1e-3;
            assert!(
                (gpu_a2 - cpu_a2).abs() < eps,
                "a2 mismatch: GPU={gpu_a2}, CPU={cpu_a2}"
            );
            assert!(
                (gpu_a3 - cpu_a3).abs() < eps,
                "a3 mismatch: GPU={gpu_a3}, CPU={cpu_a3}"
            );
            assert!(
                (gpu_b2 - cpu_b2).abs() < eps,
                "b2 mismatch: GPU={gpu_b2}, CPU={cpu_b2}"
            );
            assert!(
                (gpu_b3 - cpu_b3).abs() < eps,
                "b3 mismatch: GPU={gpu_b3}, CPU={cpu_b3}"
            );
            assert!(
                (gpu_c2 - cpu_c2).abs() < eps,
                "c2 mismatch: GPU={gpu_c2}, CPU={cpu_c2}"
            );
            assert!(
                (gpu_c3 - cpu_c3).abs() < eps,
                "c3 mismatch: GPU={gpu_c3}, CPU={cpu_c3}"
            );
            assert!(
                (gpu_d1 - cpu_d1).abs() < eps,
                "d1 mismatch: GPU={gpu_d1}, CPU={cpu_d1}"
            );
            assert!(
                (gpu_d2 - cpu_d2).abs() < eps,
                "d2 mismatch: GPU={gpu_d2}, CPU={cpu_d2}"
            );
            assert!(
                (gpu_d3 - cpu_d3).abs() < eps,
                "d3 mismatch: GPU={gpu_d3}, CPU={cpu_d3}"
            );
            assert!(
                (gpu_e1 - cpu_e1).abs() < eps,
                "e1 mismatch: GPU={gpu_e1}, CPU={cpu_e1}"
            );
            assert!(
                (gpu_e2 - cpu_e2).abs() < eps,
                "e2 mismatch: GPU={gpu_e2}, CPU={cpu_e2}"
            );
            assert!(
                (gpu_e3 - cpu_e3).abs() < eps,
                "e3 mismatch: GPU={gpu_e3}, CPU={cpu_e3}"
            );
            assert!(
                (gpu_f1 - cpu_f1).abs() < eps,
                "f1 mismatch: GPU={gpu_f1}, CPU={cpu_f1}"
            );
            assert!(
                (gpu_f2 - cpu_f2).abs() < eps,
                "f2 mismatch: GPU={gpu_f2}, CPU={cpu_f2}"
            );
            assert!(
                (gpu_f3 - cpu_f3).abs() < eps,
                "f3 mismatch: GPU={gpu_f3}, CPU={cpu_f3}"
            );
        }
    }
}
