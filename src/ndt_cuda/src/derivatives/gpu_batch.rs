//! Batched GPU kernels for initial pose estimation.
//!
//! This module provides GPU kernels optimized for batch processing K initial poses
//! simultaneously. Used for Phase 16: GPU Initial Pose Pipeline.
//!
//! # Memory Layout (batch-major ordering)
//!
//! - Transformed points: [K][N][3] = K×N×3 floats
//! - Neighbor indices: [K][N][MAX_NEIGHBORS] = K×N×8 i32s
//! - Scores: [K][N] = K×N floats
//! - Gradients: [K][N][6] = K×N×6 floats (column-major for reduction)
//! - Hessians: [K][N][36] = K×N×36 floats (column-major for reduction)

use cubecl::prelude::*;

use super::gpu::MAX_NEIGHBORS;

/// Batched brute-force radius search on GPU.
///
/// For K batches of N query points each, finds up to MAX_NEIGHBORS voxels within radius.
/// All K batches share the same voxel data but have different transformed points.
///
/// # Grid Layout
/// - Block: (256, 1, 1)
/// - Grid: (ceil(N/256), K, 1)
///
/// # Inputs
/// - `query_points`: [K×N×3] transformed points (batch-major)
/// - `voxel_means`: [V×3] voxel centroids (shared across batches)
/// - `voxel_valid`: [V] validity flags
/// - `radius_sq`: squared search radius
/// - `num_points`: N
/// - `num_voxels`: V
/// - `batch_size`: K
///
/// # Outputs
/// - `neighbor_indices`: [K×N×MAX_NEIGHBORS] neighbor voxel indices (-1 = none)
/// - `neighbor_counts`: [K×N] number of neighbors per point
#[cube(launch_unchecked)]
pub fn radius_search_batch_kernel<F: Float>(
    query_points: &Array<F>,
    voxel_means: &Array<F>,
    voxel_valid: &Array<u32>,
    radius_sq: F,
    num_points: u32,
    num_voxels: u32,
    batch_size: u32,
    neighbor_indices: &mut Array<i32>,
    neighbor_counts: &mut Array<u32>,
) {
    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let batch_idx = CUBE_POS_Y;

    if point_idx >= num_points || batch_idx >= batch_size {
        terminate!();
    }

    // Load query point from batch-specific location
    let pt_base = (batch_idx * num_points + point_idx) * 3;
    let qx = query_points[pt_base];
    let qy = query_points[pt_base + 1];
    let qz = query_points[pt_base + 2];

    // Initialize neighbor output
    let out_base = (batch_idx * num_points + point_idx) * MAX_NEIGHBORS;
    for i in 0..MAX_NEIGHBORS {
        neighbor_indices[out_base + i] = -1_i32;
    }

    let mut count = 0u32;

    // Search all voxels (brute-force)
    for v in 0..num_voxels {
        let should_process = count < MAX_NEIGHBORS;

        if should_process {
            let is_valid = voxel_valid[v];
            if is_valid != 0u32 {
                let vbase = v * 3;
                let vx = voxel_means[vbase];
                let vy = voxel_means[vbase + 1];
                let vz = voxel_means[vbase + 2];

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

    neighbor_counts[batch_idx * num_points + point_idx] = count;
}

/// Batched NDT score computation kernel.
///
/// Computes NDT scores for K batches of N points each, sharing the same voxel data
/// but using different transforms.
///
/// # Grid Layout
/// - Block: (256, 1, 1)
/// - Grid: (ceil(N/256), K, 1)
///
/// # Inputs
/// - `source_points`: [N×3] source points (shared across batches)
/// - `transforms`: [K×16] transformation matrices (one per batch)
/// - `voxel_means`: [V×3] voxel centroids
/// - `voxel_inv_covs`: [V×9] inverse covariances (row-major 3×3)
/// - `neighbor_indices`: [K×N×MAX_NEIGHBORS] from radius search
/// - `neighbor_counts`: [K×N] from radius search
/// - `params`: [4] = [gauss_d1, gauss_d2, num_points, batch_size]
///
/// # Outputs
/// - `scores`: [K×N] per-point scores
/// - `correspondences`: [K×N] correspondence counts (0 or 1+ voxels matched)
#[cube(launch_unchecked)]
pub fn compute_ndt_score_batch_kernel<F: Float>(
    source_points: &Array<F>,
    transforms: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    neighbor_indices: &Array<i32>,
    neighbor_counts: &Array<u32>,
    params: &Array<F>,
    scores: &mut Array<F>,
    correspondences: &mut Array<u32>,
) {
    // Unpack params: [gauss_d1, gauss_d2, num_points_f, batch_size_f]
    let gauss_d1 = params[0];
    let gauss_d2 = params[1];
    let num_points = u32::cast_from(params[2]);
    let batch_size = u32::cast_from(params[3]);

    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let batch_idx = CUBE_POS_Y;

    if point_idx >= num_points || batch_idx >= batch_size {
        terminate!();
    }

    // Load source point (shared across batches)
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    // Load transform for this batch
    let t_offset = batch_idx * 16;
    let t00 = transforms[t_offset];
    let t01 = transforms[t_offset + 1];
    let t02 = transforms[t_offset + 2];
    let t03 = transforms[t_offset + 3];
    let t10 = transforms[t_offset + 4];
    let t11 = transforms[t_offset + 5];
    let t12 = transforms[t_offset + 6];
    let t13 = transforms[t_offset + 7];
    let t20 = transforms[t_offset + 8];
    let t21 = transforms[t_offset + 9];
    let t22 = transforms[t_offset + 10];
    let t23 = transforms[t_offset + 11];

    // Transform point
    let tx = t00 * sx + t01 * sy + t02 * sz + t03;
    let ty = t10 * sx + t11 * sy + t12 * sz + t13;
    let tz = t20 * sx + t21 * sy + t22 * sz + t23;

    // Accumulate score across neighbors
    let mut total_score = F::new(0.0);
    let batch_point_idx = batch_idx * num_points + point_idx;
    let num_neighbors = neighbor_counts[batch_point_idx];
    let neighbor_base = batch_point_idx * MAX_NEIGHBORS;

    for i in 0..MAX_NEIGHBORS {
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
                let x = tx - mx;
                let y = ty - my;
                let z = tz - mz;

                // Load inverse covariance (row-major 3×3)
                let icbase = v * 9;
                let ic00 = voxel_inv_covs[icbase];
                let ic01 = voxel_inv_covs[icbase + 1];
                let ic02 = voxel_inv_covs[icbase + 2];
                let ic10 = voxel_inv_covs[icbase + 3];
                let ic11 = voxel_inv_covs[icbase + 4];
                let ic12 = voxel_inv_covs[icbase + 5];
                let ic20 = voxel_inv_covs[icbase + 6];
                let ic21 = voxel_inv_covs[icbase + 7];
                let ic22 = voxel_inv_covs[icbase + 8];

                // Compute x'Σ⁻¹x (Mahalanobis distance squared)
                let tmp0 = ic00 * x + ic01 * y + ic02 * z;
                let tmp1 = ic10 * x + ic11 * y + ic12 * z;
                let tmp2 = ic20 * x + ic21 * y + ic22 * z;
                let mahal_sq = x * tmp0 + y * tmp1 + z * tmp2;

                // Score: -d1 * exp(-d2/2 * mahal_sq)
                let half = F::new(0.5);
                let exponent = F::new(0.0) - half * gauss_d2 * mahal_sq;
                let score = F::new(0.0) - gauss_d1 * F::exp(exponent);

                total_score += score;
            }
        }
    }

    let out_idx = batch_idx * num_points + point_idx;
    scores[out_idx] = total_score;

    // Set correspondence flag (1 if has neighbors, 0 otherwise)
    if num_neighbors > 0u32 {
        correspondences[out_idx] = 1u32;
    } else {
        correspondences[out_idx] = 0u32;
    }
}

/// Batched NDT gradient computation kernel.
///
/// Computes per-point gradients for K batches, output in column-major order
/// for efficient CUB segmented reduction.
///
/// # Grid Layout
/// - Block: (256, 1, 1)
/// - Grid: (ceil(N/256), K, 1)
///
/// # Inputs
/// - `source_points`: [N×3] source points (shared)
/// - `transforms`: [K×16] transformation matrices
/// - `jacobians`: [K×N×18] pre-computed Jacobians (from sin_cos)
/// - `voxel_means`: [V×3]
/// - `voxel_inv_covs`: [V×9]
/// - `neighbor_indices`: [K×N×MAX_NEIGHBORS]
/// - `neighbor_counts`: [K×N]
/// - `params`: [4] = [gauss_d1, gauss_d2, num_points, batch_size]
///
/// # Outputs
/// - `gradients`: [K×6×N] column-major (6 gradients, each N values per batch)
#[cube(launch_unchecked)]
pub fn compute_ndt_gradient_batch_kernel<F: Float>(
    source_points: &Array<F>,
    transforms: &Array<F>,
    jacobians: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    neighbor_indices: &Array<i32>,
    neighbor_counts: &Array<u32>,
    params: &Array<F>,
    gradients: &mut Array<F>,
) {
    // Unpack params: [gauss_d1, gauss_d2, num_points_f, batch_size_f]
    let gauss_d1 = params[0];
    let gauss_d2 = params[1];
    let num_points = u32::cast_from(params[2]);
    let batch_size = u32::cast_from(params[3]);

    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let batch_idx = CUBE_POS_Y;

    if point_idx >= num_points || batch_idx >= batch_size {
        terminate!();
    }

    // Load source point (shared)
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    // Load transform for this batch
    let t_offset = batch_idx * 16;
    let t00 = transforms[t_offset];
    let t01 = transforms[t_offset + 1];
    let t02 = transforms[t_offset + 2];
    let t03 = transforms[t_offset + 3];
    let t10 = transforms[t_offset + 4];
    let t11 = transforms[t_offset + 5];
    let t12 = transforms[t_offset + 6];
    let t13 = transforms[t_offset + 7];
    let t20 = transforms[t_offset + 8];
    let t21 = transforms[t_offset + 9];
    let t22 = transforms[t_offset + 10];
    let t23 = transforms[t_offset + 11];

    // Transform point
    let tx = t00 * sx + t01 * sy + t02 * sz + t03;
    let ty = t10 * sx + t11 * sy + t12 * sz + t13;
    let tz = t20 * sx + t21 * sy + t22 * sz + t23;

    // Load Jacobian for this batch and point
    // Jacobian layout: [batch][point][18] = [J_tx, J_ty, J_tz, J_rx, J_ry, J_rz] × 3 coords
    let jac_offset = (batch_idx * num_points + point_idx) * 18;

    // J[i][j] = ∂(transformed coord j) / ∂(pose param i)
    // j=0,1,2 → x,y,z; i=0..5 → tx,ty,tz,rx,ry,rz
    // Stored as [J[0][0..3], J[1][0..3], ..., J[5][0..3]]
    let j00 = jacobians[jac_offset]; // ∂x/∂tx
    let j01 = jacobians[jac_offset + 1]; // ∂y/∂tx
    let j02 = jacobians[jac_offset + 2]; // ∂z/∂tx
    let j10 = jacobians[jac_offset + 3]; // ∂x/∂ty
    let j11 = jacobians[jac_offset + 4]; // ∂y/∂ty
    let j12 = jacobians[jac_offset + 5]; // ∂z/∂ty
    let j20 = jacobians[jac_offset + 6]; // ∂x/∂tz
    let j21 = jacobians[jac_offset + 7]; // ∂y/∂tz
    let j22 = jacobians[jac_offset + 8]; // ∂z/∂tz
    let j30 = jacobians[jac_offset + 9]; // ∂x/∂rx
    let j31 = jacobians[jac_offset + 10]; // ∂y/∂rx
    let j32 = jacobians[jac_offset + 11]; // ∂z/∂rx
    let j40 = jacobians[jac_offset + 12]; // ∂x/∂ry
    let j41 = jacobians[jac_offset + 13]; // ∂y/∂ry
    let j42 = jacobians[jac_offset + 14]; // ∂z/∂ry
    let j50 = jacobians[jac_offset + 15]; // ∂x/∂rz
    let j51 = jacobians[jac_offset + 16]; // ∂y/∂rz
    let j52 = jacobians[jac_offset + 17]; // ∂z/∂rz

    // Initialize gradient accumulators
    let mut g0 = F::new(0.0);
    let mut g1 = F::new(0.0);
    let mut g2 = F::new(0.0);
    let mut g3 = F::new(0.0);
    let mut g4 = F::new(0.0);
    let mut g5 = F::new(0.0);

    let batch_point_idx = batch_idx * num_points + point_idx;
    let num_neighbors = neighbor_counts[batch_point_idx];
    let neighbor_base = batch_point_idx * MAX_NEIGHBORS;

    for i in 0..MAX_NEIGHBORS {
        if i < num_neighbors {
            let voxel_idx = neighbor_indices[neighbor_base + i];
            if voxel_idx >= 0 {
                let v = voxel_idx as u32;

                // Load voxel mean
                let vbase = v * 3;
                let mx = voxel_means[vbase];
                let my = voxel_means[vbase + 1];
                let mz = voxel_means[vbase + 2];

                // x_trans = transformed - mean
                let x = tx - mx;
                let y = ty - my;
                let z = tz - mz;

                // Load inverse covariance
                let icbase = v * 9;
                let ic00 = voxel_inv_covs[icbase];
                let ic01 = voxel_inv_covs[icbase + 1];
                let ic02 = voxel_inv_covs[icbase + 2];
                let ic10 = voxel_inv_covs[icbase + 3];
                let ic11 = voxel_inv_covs[icbase + 4];
                let ic12 = voxel_inv_covs[icbase + 5];
                let ic20 = voxel_inv_covs[icbase + 6];
                let ic21 = voxel_inv_covs[icbase + 7];
                let ic22 = voxel_inv_covs[icbase + 8];

                // q = Σ⁻¹ × x_trans
                let q0 = ic00 * x + ic01 * y + ic02 * z;
                let q1 = ic10 * x + ic11 * y + ic12 * z;
                let q2 = ic20 * x + ic21 * y + ic22 * z;

                // Mahalanobis distance squared
                let mahal_sq = x * q0 + y * q1 + z * q2;

                // Score derivative factor: d1 * d2 * exp(-d2/2 * mahal_sq)
                let half = F::new(0.5);
                let exponent = F::new(0.0) - half * gauss_d2 * mahal_sq;
                let score_factor = gauss_d1 * gauss_d2 * F::exp(exponent);

                // Gradient: factor × J^T × q
                g0 += score_factor * (j00 * q0 + j01 * q1 + j02 * q2);
                g1 += score_factor * (j10 * q0 + j11 * q1 + j12 * q2);
                g2 += score_factor * (j20 * q0 + j21 * q1 + j22 * q2);
                g3 += score_factor * (j30 * q0 + j31 * q1 + j32 * q2);
                g4 += score_factor * (j40 * q0 + j41 * q1 + j42 * q2);
                g5 += score_factor * (j50 * q0 + j51 * q1 + j52 * q2);
            }
        }
    }

    // Write gradients in column-major order for efficient reduction
    // Layout: [batch][grad_idx][point] where batch is outer, grad_idx is middle
    // This allows reducing all points for each gradient component in one segment
    let batch_grad_base = batch_idx * 6 * num_points;
    gradients[batch_grad_base + point_idx] = g0;
    gradients[batch_grad_base + num_points + point_idx] = g1;
    gradients[batch_grad_base + 2 * num_points + point_idx] = g2;
    gradients[batch_grad_base + 3 * num_points + point_idx] = g3;
    gradients[batch_grad_base + 4 * num_points + point_idx] = g4;
    gradients[batch_grad_base + 5 * num_points + point_idx] = g5;
}

/// Batched NDT Hessian computation kernel.
///
/// Computes per-point Hessians (6×6 symmetric) for K batches.
/// Output in column-major order for efficient CUB segmented reduction.
///
/// # Grid Layout
/// - Block: (256, 1, 1)
/// - Grid: (ceil(N/256), K, 1)
///
/// # Inputs
/// - `source_points`: [N×3] source points (shared)
/// - `transforms`: [K×16] transformation matrices
/// - `jacobians`: [K×N×18] pre-computed Jacobians
/// - `voxel_means`: [V×3]
/// - `voxel_inv_covs`: [V×9]
/// - `neighbor_indices`: [K×N×MAX_NEIGHBORS]
/// - `neighbor_counts`: [K×N]
/// - `params`: [4] = [gauss_d1, gauss_d2, num_points, batch_size]
///
/// # Outputs
/// - `hessians`: [K×36×N] column-major (36 Hessian elements, each N values per batch)
#[cube(launch_unchecked)]
pub fn compute_ndt_hessian_batch_kernel<F: Float>(
    source_points: &Array<F>,
    transforms: &Array<F>,
    jacobians: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    neighbor_indices: &Array<i32>,
    neighbor_counts: &Array<u32>,
    params: &Array<F>,
    hessians: &mut Array<F>,
) {
    // Unpack params: [gauss_d1, gauss_d2, num_points_f, batch_size_f]
    let gauss_d1 = params[0];
    let gauss_d2 = params[1];
    let num_points = u32::cast_from(params[2]);
    let batch_size = u32::cast_from(params[3]);

    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let batch_idx = CUBE_POS_Y;

    if point_idx >= num_points || batch_idx >= batch_size {
        terminate!();
    }

    // Load source point (shared)
    let sbase = point_idx * 3;
    let sx = source_points[sbase];
    let sy = source_points[sbase + 1];
    let sz = source_points[sbase + 2];

    // Load transform for this batch
    let t_offset = batch_idx * 16;
    let t00 = transforms[t_offset];
    let t01 = transforms[t_offset + 1];
    let t02 = transforms[t_offset + 2];
    let t03 = transforms[t_offset + 3];
    let t10 = transforms[t_offset + 4];
    let t11 = transforms[t_offset + 5];
    let t12 = transforms[t_offset + 6];
    let t13 = transforms[t_offset + 7];
    let t20 = transforms[t_offset + 8];
    let t21 = transforms[t_offset + 9];
    let t22 = transforms[t_offset + 10];
    let t23 = transforms[t_offset + 11];

    // Transform point
    let tx = t00 * sx + t01 * sy + t02 * sz + t03;
    let ty = t10 * sx + t11 * sy + t12 * sz + t13;
    let tz = t20 * sx + t21 * sy + t22 * sz + t23;

    // Initialize Hessian accumulators (6×6 = 36 elements, row-major)
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

    let batch_point_idx = batch_idx * num_points + point_idx;
    let num_neighbors = neighbor_counts[batch_point_idx];
    let neighbor_base = batch_point_idx * MAX_NEIGHBORS;

    // Load Jacobian for this batch and point
    let jac_offset = (batch_idx * num_points + point_idx) * 18;
    let j00 = jacobians[jac_offset];
    let j01 = jacobians[jac_offset + 1];
    let j02 = jacobians[jac_offset + 2];
    let j10 = jacobians[jac_offset + 3];
    let j11 = jacobians[jac_offset + 4];
    let j12 = jacobians[jac_offset + 5];
    let j20 = jacobians[jac_offset + 6];
    let j21 = jacobians[jac_offset + 7];
    let j22 = jacobians[jac_offset + 8];
    let j30 = jacobians[jac_offset + 9];
    let j31 = jacobians[jac_offset + 10];
    let j32 = jacobians[jac_offset + 11];
    let j40 = jacobians[jac_offset + 12];
    let j41 = jacobians[jac_offset + 13];
    let j42 = jacobians[jac_offset + 14];
    let j50 = jacobians[jac_offset + 15];
    let j51 = jacobians[jac_offset + 16];
    let j52 = jacobians[jac_offset + 17];

    // Note: Second-order point Hessian terms (∂²T/∂pose²) are omitted for simplicity.
    // This matches the simplified Hessian approximation used in the main pipeline.

    for i in 0..MAX_NEIGHBORS {
        if i < num_neighbors {
            let voxel_idx = neighbor_indices[neighbor_base + i];
            if voxel_idx >= 0 {
                let v = voxel_idx as u32;

                // Load voxel mean
                let vbase = v * 3;
                let mx = voxel_means[vbase];
                let my = voxel_means[vbase + 1];
                let mz = voxel_means[vbase + 2];

                // x_trans = transformed - mean
                let x = tx - mx;
                let y = ty - my;
                let z = tz - mz;

                // Load inverse covariance
                let icbase = v * 9;
                let ic00 = voxel_inv_covs[icbase];
                let ic01 = voxel_inv_covs[icbase + 1];
                let ic02 = voxel_inv_covs[icbase + 2];
                let ic10 = voxel_inv_covs[icbase + 3];
                let ic11 = voxel_inv_covs[icbase + 4];
                let ic12 = voxel_inv_covs[icbase + 5];
                let ic20 = voxel_inv_covs[icbase + 6];
                let ic21 = voxel_inv_covs[icbase + 7];
                let ic22 = voxel_inv_covs[icbase + 8];

                // q = Σ⁻¹ × x_trans
                let q0 = ic00 * x + ic01 * y + ic02 * z;
                let q1 = ic10 * x + ic11 * y + ic12 * z;
                let q2 = ic20 * x + ic21 * y + ic22 * z;

                // Mahalanobis distance squared
                let mahal_sq = x * q0 + y * q1 + z * q2;

                // Common factors
                let half = F::new(0.5);
                let exponent = F::new(0.0) - half * gauss_d2 * mahal_sq;
                let exp_val = F::exp(exponent);
                let d1_d2 = gauss_d1 * gauss_d2;
                let d1_d2_sq = d1_d2 * gauss_d2;

                // Gradient terms: factor × J^T × q
                let gj0 = j00 * q0 + j01 * q1 + j02 * q2;
                let gj1 = j10 * q0 + j11 * q1 + j12 * q2;
                let gj2 = j20 * q0 + j21 * q1 + j22 * q2;
                let gj3 = j30 * q0 + j31 * q1 + j32 * q2;
                let gj4 = j40 * q0 + j41 * q1 + j42 * q2;
                let gj5 = j50 * q0 + j51 * q1 + j52 * q2;

                // Hessian term 1: -d1*d2^2 * exp * (J^T q)(J^T q)^T
                let factor1 = F::new(0.0) - d1_d2_sq * exp_val;
                h00 += factor1 * gj0 * gj0;
                h01 += factor1 * gj0 * gj1;
                h02 += factor1 * gj0 * gj2;
                h03 += factor1 * gj0 * gj3;
                h04 += factor1 * gj0 * gj4;
                h05 += factor1 * gj0 * gj5;
                h11 += factor1 * gj1 * gj1;
                h12 += factor1 * gj1 * gj2;
                h13 += factor1 * gj1 * gj3;
                h14 += factor1 * gj1 * gj4;
                h15 += factor1 * gj1 * gj5;
                h22 += factor1 * gj2 * gj2;
                h23 += factor1 * gj2 * gj3;
                h24 += factor1 * gj2 * gj4;
                h25 += factor1 * gj2 * gj5;
                h33 += factor1 * gj3 * gj3;
                h34 += factor1 * gj3 * gj4;
                h35 += factor1 * gj3 * gj5;
                h44 += factor1 * gj4 * gj4;
                h45 += factor1 * gj4 * gj5;
                h55 += factor1 * gj5 * gj5;

                // Hessian term 2: d1*d2 * exp * (J^T Σ⁻¹ J)
                let factor2 = d1_d2 * exp_val;

                // Compute J^T Σ⁻¹ J terms
                // This is (∂T/∂p_i)^T Σ⁻¹ (∂T/∂p_j) for each i,j pair
                // For translation parameters (0,1,2), J columns are unit vectors
                // For rotation parameters (3,4,5), J columns depend on sin_cos

                // J^T Σ⁻¹ J[i,j] = sum over k,l of J[k,i] * Σ⁻¹[k,l] * J[l,j]
                // Using Σ⁻¹ J[j] first, then J[i]^T × that

                // Compute Σ⁻¹ J for each pose parameter
                let sj00 = ic00 * j00 + ic01 * j01 + ic02 * j02;
                let sj01 = ic10 * j00 + ic11 * j01 + ic12 * j02;
                let sj02 = ic20 * j00 + ic21 * j01 + ic22 * j02;

                let sj10 = ic00 * j10 + ic01 * j11 + ic02 * j12;
                let sj11 = ic10 * j10 + ic11 * j11 + ic12 * j12;
                let sj12 = ic20 * j10 + ic21 * j11 + ic22 * j12;

                let sj20 = ic00 * j20 + ic01 * j21 + ic02 * j22;
                let sj21 = ic10 * j20 + ic11 * j21 + ic12 * j22;
                let sj22 = ic20 * j20 + ic21 * j21 + ic22 * j22;

                let sj30 = ic00 * j30 + ic01 * j31 + ic02 * j32;
                let sj31 = ic10 * j30 + ic11 * j31 + ic12 * j32;
                let sj32 = ic20 * j30 + ic21 * j31 + ic22 * j32;

                let sj40 = ic00 * j40 + ic01 * j41 + ic02 * j42;
                let sj41 = ic10 * j40 + ic11 * j41 + ic12 * j42;
                let sj42 = ic20 * j40 + ic21 * j41 + ic22 * j42;

                let sj50 = ic00 * j50 + ic01 * j51 + ic02 * j52;
                let sj51 = ic10 * j50 + ic11 * j51 + ic12 * j52;
                let sj52 = ic20 * j50 + ic21 * j51 + ic22 * j52;

                // J^T Σ⁻¹ J terms
                h00 += factor2 * (j00 * sj00 + j01 * sj01 + j02 * sj02);
                h01 += factor2 * (j00 * sj10 + j01 * sj11 + j02 * sj12);
                h02 += factor2 * (j00 * sj20 + j01 * sj21 + j02 * sj22);
                h03 += factor2 * (j00 * sj30 + j01 * sj31 + j02 * sj32);
                h04 += factor2 * (j00 * sj40 + j01 * sj41 + j02 * sj42);
                h05 += factor2 * (j00 * sj50 + j01 * sj51 + j02 * sj52);
                h11 += factor2 * (j10 * sj10 + j11 * sj11 + j12 * sj12);
                h12 += factor2 * (j10 * sj20 + j11 * sj21 + j12 * sj22);
                h13 += factor2 * (j10 * sj30 + j11 * sj31 + j12 * sj32);
                h14 += factor2 * (j10 * sj40 + j11 * sj41 + j12 * sj42);
                h15 += factor2 * (j10 * sj50 + j11 * sj51 + j12 * sj52);
                h22 += factor2 * (j20 * sj20 + j21 * sj21 + j22 * sj22);
                h23 += factor2 * (j20 * sj30 + j21 * sj31 + j22 * sj32);
                h24 += factor2 * (j20 * sj40 + j21 * sj41 + j22 * sj42);
                h25 += factor2 * (j20 * sj50 + j21 * sj51 + j22 * sj52);
                h33 += factor2 * (j30 * sj30 + j31 * sj31 + j32 * sj32);
                h34 += factor2 * (j30 * sj40 + j31 * sj41 + j32 * sj42);
                h35 += factor2 * (j30 * sj50 + j31 * sj51 + j32 * sj52);
                h44 += factor2 * (j40 * sj40 + j41 * sj41 + j42 * sj42);
                h45 += factor2 * (j40 * sj50 + j41 * sj51 + j42 * sj52);
                h55 += factor2 * (j50 * sj50 + j51 * sj51 + j52 * sj52);

                // Hessian term 3: d1*d2 * exp * (H^T q) for rotation-rotation entries
                // This uses the point Hessians (second derivatives of transform)
                // Note: H[i,j,k] = ∂²T_k / ∂p_i ∂p_j (second-order point Hessian)
                // is omitted for simplicity, matching the simplified Hessian used
                // in the main pipeline. Only rotation-rotation terms (i,j >= 3) are non-zero.
            }
        }
    }

    // Write Hessians in column-major order for efficient reduction
    // Layout: [batch][hess_idx][point] where hess_idx is 0..36 (row-major 6×6)
    let batch_hess_base = batch_idx * 36 * num_points;

    // Row 0
    hessians[batch_hess_base + point_idx] = h00;
    hessians[batch_hess_base + num_points + point_idx] = h01;
    hessians[batch_hess_base + 2 * num_points + point_idx] = h02;
    hessians[batch_hess_base + 3 * num_points + point_idx] = h03;
    hessians[batch_hess_base + 4 * num_points + point_idx] = h04;
    hessians[batch_hess_base + 5 * num_points + point_idx] = h05;

    // Row 1 (symmetric)
    hessians[batch_hess_base + 6 * num_points + point_idx] = h01;
    hessians[batch_hess_base + 7 * num_points + point_idx] = h11;
    hessians[batch_hess_base + 8 * num_points + point_idx] = h12;
    hessians[batch_hess_base + 9 * num_points + point_idx] = h13;
    hessians[batch_hess_base + 10 * num_points + point_idx] = h14;
    hessians[batch_hess_base + 11 * num_points + point_idx] = h15;

    // Row 2 (symmetric)
    hessians[batch_hess_base + 12 * num_points + point_idx] = h02;
    hessians[batch_hess_base + 13 * num_points + point_idx] = h12;
    hessians[batch_hess_base + 14 * num_points + point_idx] = h22;
    hessians[batch_hess_base + 15 * num_points + point_idx] = h23;
    hessians[batch_hess_base + 16 * num_points + point_idx] = h24;
    hessians[batch_hess_base + 17 * num_points + point_idx] = h25;

    // Row 3 (symmetric)
    hessians[batch_hess_base + 18 * num_points + point_idx] = h03;
    hessians[batch_hess_base + 19 * num_points + point_idx] = h13;
    hessians[batch_hess_base + 20 * num_points + point_idx] = h23;
    hessians[batch_hess_base + 21 * num_points + point_idx] = h33;
    hessians[batch_hess_base + 22 * num_points + point_idx] = h34;
    hessians[batch_hess_base + 23 * num_points + point_idx] = h35;

    // Row 4 (symmetric)
    hessians[batch_hess_base + 24 * num_points + point_idx] = h04;
    hessians[batch_hess_base + 25 * num_points + point_idx] = h14;
    hessians[batch_hess_base + 26 * num_points + point_idx] = h24;
    hessians[batch_hess_base + 27 * num_points + point_idx] = h34;
    hessians[batch_hess_base + 28 * num_points + point_idx] = h44;
    hessians[batch_hess_base + 29 * num_points + point_idx] = h45;

    // Row 5 (symmetric)
    hessians[batch_hess_base + 30 * num_points + point_idx] = h05;
    hessians[batch_hess_base + 31 * num_points + point_idx] = h15;
    hessians[batch_hess_base + 32 * num_points + point_idx] = h25;
    hessians[batch_hess_base + 33 * num_points + point_idx] = h35;
    hessians[batch_hess_base + 34 * num_points + point_idx] = h45;
    hessians[batch_hess_base + 35 * num_points + point_idx] = h55;
}

/// Batched pose update kernel.
///
/// Updates K poses by their respective Newton steps scaled by line search alphas.
///
/// # Grid Layout
/// - Block: (6, K, 1)
/// - Grid: (1, 1, 1)
///
/// # Inputs
/// - `poses`: [K×6] current poses
/// - `deltas`: [K×6] Newton steps
/// - `alphas`: [K] step sizes
/// - `batch_size`: K
///
/// # Outputs
/// - `poses`: [K×6] updated in-place
#[cube(launch_unchecked)]
pub fn update_poses_batch_kernel<F: Float>(
    poses: &mut Array<F>,
    deltas: &Array<F>,
    alphas: &Array<F>,
    batch_size: u32,
) {
    let param_idx = UNIT_POS_X;
    let batch_idx = UNIT_POS_Y;

    if param_idx >= 6u32 || batch_idx >= batch_size {
        terminate!();
    }

    let idx = batch_idx * 6 + param_idx;
    let alpha = alphas[batch_idx];
    poses[idx] = poses[idx] + alpha * deltas[idx];
}

/// Batched convergence check kernel.
///
/// Checks if each of K poses has converged based on step size.
///
/// # Grid Layout
/// - Block: (K, 1, 1)
/// - Grid: (1, 1, 1)
///
/// # Inputs
/// - `deltas`: [K×6] Newton steps
/// - `alphas`: [K] step sizes
/// - `epsilon_sq`: squared convergence threshold
/// - `batch_size`: K
///
/// # Outputs
/// - `converged`: [K] 1 if converged, 0 otherwise
#[cube(launch_unchecked)]
pub fn check_convergence_batch_kernel<F: Float>(
    deltas: &Array<F>,
    alphas: &Array<F>,
    epsilon_sq: F,
    batch_size: u32,
    converged: &mut Array<u32>,
) {
    let batch_idx = UNIT_POS_X;

    if batch_idx >= batch_size {
        terminate!();
    }

    let delta_offset = batch_idx * 6;
    let alpha = alphas[batch_idx];

    // Compute squared norm of scaled delta
    let mut norm_sq = F::new(0.0);
    for i in 0..6u32 {
        let scaled = alpha * deltas[delta_offset + i];
        norm_sq += scaled * scaled;
    }

    // Set convergence flag (1 if converged, 0 otherwise)
    if norm_sq < epsilon_sq {
        converged[batch_idx] = 1u32;
    } else {
        converged[batch_idx] = 0u32;
    }
}

/// Batched Jacobian computation kernel.
///
/// Computes Jacobians (∂T/∂pose) for K batches of N points.
/// Each Jacobian is 6×3 stored as 18 elements.
///
/// # Grid Layout
/// - Block: (256, 1, 1)
/// - Grid: (ceil(N/256), K, 1)
///
/// # Inputs
/// - `source_points`: [N×3] source points (shared)
/// - `sin_cos`: [K×6] [sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw] per batch
/// - `num_points`: N
/// - `batch_size`: K
///
/// # Outputs
/// - `jacobians`: [K×N×18] Jacobians
#[cube(launch_unchecked)]
pub fn compute_jacobians_batch_kernel<F: Float>(
    source_points: &Array<F>,
    sin_cos: &Array<F>,
    num_points: u32,
    batch_size: u32,
    jacobians: &mut Array<F>,
) {
    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let batch_idx = CUBE_POS_Y;

    if point_idx >= num_points || batch_idx >= batch_size {
        terminate!();
    }

    // Load source point (shared)
    let sbase = point_idx * 3;
    let x = source_points[sbase];
    let y = source_points[sbase + 1];
    let z = source_points[sbase + 2];

    // Load sin/cos for this batch
    let sc_offset = batch_idx * 6;
    let sr = sin_cos[sc_offset];
    let cr = sin_cos[sc_offset + 1];
    let sp = sin_cos[sc_offset + 2];
    let cp = sin_cos[sc_offset + 3];
    let sy = sin_cos[sc_offset + 4];
    let cy = sin_cos[sc_offset + 5];

    // Compute Jacobian terms
    // Translation Jacobians are identity (∂T/∂tx = [1,0,0], etc.)
    // Rotation Jacobians depend on point and angles

    // ∂T/∂roll
    let jr0 = (cy * sp * cr + sy * sr) * y + (F::new(0.0) - cy * sp * sr + sy * cr) * z;
    let jr1 = (sy * sp * cr - cy * sr) * y + (F::new(0.0) - sy * sp * sr - cy * cr) * z;
    let jr2 = cp * cr * y - cp * sr * z;

    // ∂T/∂pitch
    let jp0 = (F::new(0.0) - cy * sp) * x + cy * cp * sr * y + cy * cp * cr * z;
    let jp1 = (F::new(0.0) - sy * sp) * x + sy * cp * sr * y + sy * cp * cr * z;
    let jp2 = (F::new(0.0) - cp) * x - sp * sr * y - sp * cr * z;

    // ∂T/∂yaw
    let jy0 = (F::new(0.0) - sy * cp) * x
        + (F::new(0.0) - sy * sp * sr - cy * cr) * y
        + (F::new(0.0) - sy * sp * cr + cy * sr) * z;
    let jy1 = cy * cp * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z;
    let jy2 = F::new(0.0);

    // Write Jacobian
    let jac_offset = (batch_idx * num_points + point_idx) * 18;

    // Translation derivatives (identity)
    jacobians[jac_offset] = F::new(1.0); // ∂x/∂tx
    jacobians[jac_offset + 1] = F::new(0.0); // ∂y/∂tx
    jacobians[jac_offset + 2] = F::new(0.0); // ∂z/∂tx
    jacobians[jac_offset + 3] = F::new(0.0); // ∂x/∂ty
    jacobians[jac_offset + 4] = F::new(1.0); // ∂y/∂ty
    jacobians[jac_offset + 5] = F::new(0.0); // ∂z/∂ty
    jacobians[jac_offset + 6] = F::new(0.0); // ∂x/∂tz
    jacobians[jac_offset + 7] = F::new(0.0); // ∂y/∂tz
    jacobians[jac_offset + 8] = F::new(1.0); // ∂z/∂tz

    // Rotation derivatives
    jacobians[jac_offset + 9] = jr0; // ∂x/∂roll
    jacobians[jac_offset + 10] = jr1; // ∂y/∂roll
    jacobians[jac_offset + 11] = jr2; // ∂z/∂roll
    jacobians[jac_offset + 12] = jp0; // ∂x/∂pitch
    jacobians[jac_offset + 13] = jp1; // ∂y/∂pitch
    jacobians[jac_offset + 14] = jp2; // ∂z/∂pitch
    jacobians[jac_offset + 15] = jy0; // ∂x/∂yaw
    jacobians[jac_offset + 16] = jy1; // ∂y/∂yaw
    jacobians[jac_offset + 17] = jy2; // ∂z/∂yaw
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_kernels_compile() {
        // This test just verifies that the kernel macros expand correctly.
        // Actual GPU execution is tested elsewhere.
    }
}
