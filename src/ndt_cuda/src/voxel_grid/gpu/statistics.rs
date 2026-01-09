//! Voxel statistics computation (mean, covariance, regularization).
//!
//! After segment detection, each segment represents a voxel. This module
//! computes per-voxel statistics needed for NDT matching:
//! - Mean position (centroid)
//! - Covariance matrix
//! - Inverse covariance matrix (after regularization)
//!
//! # Algorithm
//!
//! 1. **Pass 1**: Accumulate position sums per voxel
//! 2. **Pass 2**: Compute means, then accumulate covariance sums
//! 3. **Pass 3**: Finalize covariance (divide by N-1), regularize, invert
//!
//! # GPU Implementation
//!
//! Uses **segmented reduction** to avoid atomic operations:
//! - Points are sorted by Morton code so same-voxel points are contiguous
//! - Each segment (voxel) is processed by one GPU thread
//! - No atomics needed since each thread owns its segment
//!
//! See `docs/gpu-voxel-statistics.md` for detailed algorithm analysis.

use cubecl::prelude::*;

/// Result of voxel statistics computation.
#[derive(Debug, Clone)]
pub struct VoxelStatistics {
    /// Mean position (centroid) for each voxel, [V, 3] flattened.
    pub means: Vec<f32>,
    /// Covariance matrix for each voxel, [V, 9] flattened (row-major 3x3).
    pub covariances: Vec<f32>,
    /// Inverse covariance matrix for each voxel, [V, 9] flattened.
    pub inv_covariances: Vec<f32>,
    /// Principal axis (eigenvector of smallest eigenvalue) for each voxel, [V, 3] flattened.
    pub principal_axes: Vec<f32>,
    /// Number of points in each voxel.
    pub point_counts: Vec<u32>,
    /// Whether each voxel is valid (has enough points).
    pub valid: Vec<bool>,
    /// Number of voxels.
    pub num_voxels: u32,
    /// Minimum points required for a valid voxel.
    pub min_points: u32,
}

/// Intermediate sums for voxel computation.
#[derive(Debug)]
pub struct VoxelSums {
    /// Sum of positions for each voxel, [V, 3] flattened.
    pub position_sums: Vec<f32>,
    /// Point count for each voxel.
    pub counts: Vec<u32>,
    /// Number of voxels.
    pub num_voxels: u32,
}

// ============================================================================
// GPU Kernels for Segmented Reduction
// ============================================================================
//
// These kernels process sorted points where each segment (contiguous run of
// points with the same Morton code) represents a voxel. One thread per segment
// means no atomic operations are needed.

/// GPU kernel: Accumulate position sums for each segment (voxel).
///
/// Each thread processes one segment, accumulating all points in that segment.
/// Since points are sorted by Morton code, segments are contiguous.
///
/// # Inputs
/// - `points`: [N * 3] sorted point coordinates (by Morton code)
/// - `sorted_indices`: [N] original point indices (to look up coordinates)
/// - `segment_starts`: [S] start index of each segment
/// - `num_segments`: number of segments (voxels)
/// - `num_points`: total number of points
///
/// # Outputs
/// - `position_sums`: [S * 3] accumulated position sums per segment
/// - `counts`: [S] point count per segment
#[cube(launch_unchecked)]
pub fn accumulate_segment_sums_kernel<F: Float>(
    points: &Array<F>,
    sorted_indices: &Array<u32>,
    segment_starts: &Array<u32>,
    num_segments: u32,
    num_points: u32,
    position_sums: &mut Array<F>,
    counts: &mut Array<u32>,
) {
    let segment_idx = ABSOLUTE_POS;

    // Early exit for out-of-bounds threads
    if segment_idx >= num_segments {
        terminate!();
    }

    // Determine segment bounds
    let start = segment_starts[segment_idx];
    let end = if segment_idx + 1 < num_segments {
        segment_starts[segment_idx + 1]
    } else {
        num_points
    };

    // Accumulate position sums for this segment
    let mut sum_x = F::new(0.0);
    let mut sum_y = F::new(0.0);
    let mut sum_z = F::new(0.0);
    let mut count = 0u32;

    // Process all points in this segment
    // Using a bounded loop to avoid CubeCL optimizer issues with break statements
    let max_points_per_segment = end - start;
    for offset in 0..max_points_per_segment {
        let i = start + offset;
        // Get original point index
        let orig_idx = sorted_indices[i];
        let base = orig_idx * 3;

        // Accumulate
        sum_x += points[base];
        sum_y += points[base + 1];
        sum_z += points[base + 2];
        count += 1;
    }

    // Write results (no contention since each thread owns one segment)
    let out_base = segment_idx * 3;
    position_sums[out_base] = sum_x;
    position_sums[out_base + 1] = sum_y;
    position_sums[out_base + 2] = sum_z;
    counts[segment_idx] = count;
}

/// GPU kernel: Accumulate covariance sums for each segment (voxel).
///
/// Second pass after means are computed. Each thread processes one segment,
/// computing (p - mean)(p - mean)^T for all points in that segment.
///
/// # Inputs
/// - `points`: [N * 3] sorted point coordinates
/// - `sorted_indices`: [N] original point indices
/// - `segment_starts`: [S] start index of each segment
/// - `means`: [S * 3] mean position for each segment (from pass 1)
/// - `num_segments`: number of segments
/// - `num_points`: total number of points
///
/// # Outputs
/// - `cov_sums`: [S * 9] accumulated covariance sums (row-major 3x3)
#[cube(launch_unchecked)]
pub fn accumulate_segment_covariances_kernel<F: Float>(
    points: &Array<F>,
    sorted_indices: &Array<u32>,
    segment_starts: &Array<u32>,
    means: &Array<F>,
    num_segments: u32,
    num_points: u32,
    cov_sums: &mut Array<F>,
) {
    let segment_idx = ABSOLUTE_POS;

    if segment_idx >= num_segments {
        terminate!();
    }

    // Determine segment bounds
    let start = segment_starts[segment_idx];
    let end = if segment_idx + 1 < num_segments {
        segment_starts[segment_idx + 1]
    } else {
        num_points
    };

    // Load mean for this segment
    let mean_base = segment_idx * 3;
    let mean_x = means[mean_base];
    let mean_y = means[mean_base + 1];
    let mean_z = means[mean_base + 2];

    // Accumulate covariance components (symmetric 3x3 matrix)
    let mut cov00 = F::new(0.0); // dx*dx
    let mut cov01 = F::new(0.0); // dx*dy
    let mut cov02 = F::new(0.0); // dx*dz
    let mut cov11 = F::new(0.0); // dy*dy
    let mut cov12 = F::new(0.0); // dy*dz
    let mut cov22 = F::new(0.0); // dz*dz

    // Process all points in this segment
    let max_points_per_segment = end - start;
    for offset in 0..max_points_per_segment {
        let i = start + offset;
        let orig_idx = sorted_indices[i];
        let base = orig_idx * 3;

        // Deviation from mean
        let dx = points[base] - mean_x;
        let dy = points[base + 1] - mean_y;
        let dz = points[base + 2] - mean_z;

        // Outer product (only upper triangle since symmetric)
        cov00 += dx * dx;
        cov01 += dx * dy;
        cov02 += dx * dz;
        cov11 += dy * dy;
        cov12 += dy * dz;
        cov22 += dz * dz;
    }

    // Write full 3x3 matrix (row-major)
    let out_base = segment_idx * 9;
    cov_sums[out_base] = cov00; // [0,0]
    cov_sums[out_base + 1] = cov01; // [0,1]
    cov_sums[out_base + 2] = cov02; // [0,2]
    cov_sums[out_base + 3] = cov01; // [1,0] = [0,1] (symmetric)
    cov_sums[out_base + 4] = cov11; // [1,1]
    cov_sums[out_base + 5] = cov12; // [1,2]
    cov_sums[out_base + 6] = cov02; // [2,0] = [0,2] (symmetric)
    cov_sums[out_base + 7] = cov12; // [2,1] = [1,2] (symmetric)
    cov_sums[out_base + 8] = cov22; // [2,2]
}

/// GPU kernel: Compute means from position sums and counts.
///
/// Simple element-wise division: mean = sum / count
///
/// # Inputs
/// - `position_sums`: [S * 3] accumulated position sums
/// - `counts`: [S] point counts per segment
/// - `num_segments`: number of segments
///
/// # Outputs
/// - `means`: [S * 3] mean positions
#[cube(launch_unchecked)]
pub fn compute_means_kernel<F: Float>(
    position_sums: &Array<F>,
    counts: &Array<u32>,
    num_segments: u32,
    means: &mut Array<F>,
) {
    let segment_idx = ABSOLUTE_POS;

    if segment_idx >= num_segments {
        terminate!();
    }

    let count = counts[segment_idx];
    let base = segment_idx * 3;

    if count > 0 {
        let inv_count = F::new(1.0) / F::cast_from(count);
        means[base] = position_sums[base] * inv_count;
        means[base + 1] = position_sums[base + 1] * inv_count;
        means[base + 2] = position_sums[base + 2] * inv_count;
    } else {
        means[base] = F::new(0.0);
        means[base + 1] = F::new(0.0);
        means[base + 2] = F::new(0.0);
    }
}

// ============================================================================
// CPU Reference Implementations
// ============================================================================

/// Compute position sums and counts per voxel (CPU reference implementation).
///
/// # Arguments
/// * `points` - Flat array of point coordinates [x0, y0, z0, x1, y1, z1, ...]
/// * `sorted_indices` - Original point indices after sorting by Morton code
/// * `segment_ids` - Voxel ID for each point (from segment detection)
/// * `num_voxels` - Total number of voxels
///
/// # Returns
/// Position sums and counts per voxel.
pub fn compute_voxel_sums_cpu(
    points: &[f32],
    sorted_indices: &[u32],
    segment_ids: &[u32],
    num_voxels: u32,
) -> VoxelSums {
    let num_points = sorted_indices.len();
    let num_voxels_usize = num_voxels as usize;

    let mut position_sums = vec![0.0f32; num_voxels_usize * 3];
    let mut counts = vec![0u32; num_voxels_usize];

    for i in 0..num_points {
        let orig_idx = sorted_indices[i] as usize;
        let voxel_id = segment_ids[i] as usize;

        if voxel_id >= num_voxels_usize {
            continue;
        }

        let px = points[orig_idx * 3];
        let py = points[orig_idx * 3 + 1];
        let pz = points[orig_idx * 3 + 2];

        position_sums[voxel_id * 3] += px;
        position_sums[voxel_id * 3 + 1] += py;
        position_sums[voxel_id * 3 + 2] += pz;
        counts[voxel_id] += 1;
    }

    VoxelSums {
        position_sums,
        counts,
        num_voxels,
    }
}

/// Compute means from sums (CPU reference implementation).
pub fn compute_means_from_sums(sums: &VoxelSums) -> Vec<f32> {
    let num_voxels = sums.num_voxels as usize;
    let mut means = vec![0.0f32; num_voxels * 3];

    for v in 0..num_voxels {
        let count = sums.counts[v];
        if count > 0 {
            let inv_count = 1.0 / count as f32;
            means[v * 3] = sums.position_sums[v * 3] * inv_count;
            means[v * 3 + 1] = sums.position_sums[v * 3 + 1] * inv_count;
            means[v * 3 + 2] = sums.position_sums[v * 3 + 2] * inv_count;
        }
    }

    means
}

/// Compute covariance sums per voxel (CPU reference implementation).
///
/// Accumulates (p - mean)(p - mean)^T for each point.
///
/// # Arguments
/// * `points` - Flat array of point coordinates
/// * `sorted_indices` - Original point indices after sorting
/// * `segment_ids` - Voxel ID for each point
/// * `means` - Mean position for each voxel [V, 3]
/// * `num_voxels` - Total number of voxels
///
/// # Returns
/// Covariance sums [V, 9] (row-major 3x3 matrices).
pub fn compute_covariance_sums_cpu(
    points: &[f32],
    sorted_indices: &[u32],
    segment_ids: &[u32],
    means: &[f32],
    num_voxels: u32,
) -> Vec<f32> {
    let num_points = sorted_indices.len();
    let num_voxels_usize = num_voxels as usize;

    let mut cov_sums = vec![0.0f32; num_voxels_usize * 9];

    for i in 0..num_points {
        let orig_idx = sorted_indices[i] as usize;
        let voxel_id = segment_ids[i] as usize;

        if voxel_id >= num_voxels_usize {
            continue;
        }

        // Deviation from mean
        let dx = points[orig_idx * 3] - means[voxel_id * 3];
        let dy = points[orig_idx * 3 + 1] - means[voxel_id * 3 + 1];
        let dz = points[orig_idx * 3 + 2] - means[voxel_id * 3 + 2];

        // Outer product (symmetric 3x3 matrix)
        let base = voxel_id * 9;
        cov_sums[base] += dx * dx; // [0,0]
        cov_sums[base + 1] += dx * dy; // [0,1]
        cov_sums[base + 2] += dx * dz; // [0,2]
        cov_sums[base + 3] += dy * dx; // [1,0]
        cov_sums[base + 4] += dy * dy; // [1,1]
        cov_sums[base + 5] += dy * dz; // [1,2]
        cov_sums[base + 6] += dz * dx; // [2,0]
        cov_sums[base + 7] += dz * dy; // [2,1]
        cov_sums[base + 8] += dz * dz; // [2,2]
    }

    cov_sums
}

/// Finalize covariance matrices: divide by (N-1), regularize, and invert.
///
/// # Arguments
/// * `cov_sums` - Covariance sums [V, 9]
/// * `counts` - Point counts per voxel
/// * `min_points` - Minimum points required for valid voxel
///
/// # Returns
/// Complete voxel statistics with covariances, inverse covariances, and principal axes.
pub fn finalize_voxels_cpu(
    means: Vec<f32>,
    cov_sums: Vec<f32>,
    counts: Vec<u32>,
    min_points: u32,
) -> VoxelStatistics {
    let num_voxels = counts.len();
    let mut covariances = vec![0.0f32; num_voxels * 9];
    let mut inv_covariances = vec![0.0f32; num_voxels * 9];
    let mut principal_axes = vec![0.0f32; num_voxels * 3];
    let mut valid = vec![false; num_voxels];

    for v in 0..num_voxels {
        let count = counts[v];

        if count < min_points {
            // Not enough points - mark as invalid
            valid[v] = false;
            // Set identity matrix for safety
            inv_covariances[v * 9] = 1.0;
            inv_covariances[v * 9 + 4] = 1.0;
            inv_covariances[v * 9 + 8] = 1.0;
            // Default principal axis to Z
            principal_axes[v * 3 + 2] = 1.0;
            continue;
        }

        // Finalize covariance: cov = sum / (count - 1)
        let denom = if count > 1 {
            1.0 / (count - 1) as f32
        } else {
            1.0
        };

        for i in 0..9 {
            covariances[v * 9 + i] = cov_sums[v * 9 + i] * denom;
        }

        // Regularize and invert covariance matrix
        let result = regularize_and_invert(&covariances[v * 9..v * 9 + 9]);

        for i in 0..9 {
            inv_covariances[v * 9 + i] = result.inv_covariance[i];
        }
        for i in 0..3 {
            principal_axes[v * 3 + i] = result.principal_axis[i];
        }
        valid[v] = result.is_valid;
    }

    VoxelStatistics {
        means,
        covariances,
        inv_covariances,
        principal_axes,
        point_counts: counts,
        valid,
        num_voxels: num_voxels as u32,
        min_points,
    }
}

/// Result of covariance regularization.
struct RegularizationResult {
    /// Inverse covariance matrix (row-major 3x3).
    inv_covariance: [f32; 9],
    /// Principal axis (eigenvector of smallest eigenvalue).
    principal_axis: [f32; 3],
    /// Whether the regularization succeeded.
    is_valid: bool,
}

/// Regularize and invert a 3x3 covariance matrix.
///
/// Uses eigenvalue decomposition to:
/// 1. Clamp small eigenvalues to min_eigenvalue_ratio * max_eigenvalue
/// 2. Reconstruct the regularized covariance
/// 3. Invert the matrix
/// 4. Extract principal axis (eigenvector of smallest eigenvalue)
///
/// Uses f64 internally for numerical stability with planar point clouds
/// that have near-zero eigenvalues.
fn regularize_and_invert(cov: &[f32]) -> RegularizationResult {
    // Default result for invalid cases
    let invalid_result = RegularizationResult {
        inv_covariance: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        principal_axis: [0.0, 0.0, 1.0], // Default to Z-axis
        is_valid: false,
    };

    // Use f64 for numerical stability (matching CPU implementation)
    let cov64: [f64; 9] = [
        cov[0] as f64,
        cov[1] as f64,
        cov[2] as f64,
        cov[3] as f64,
        cov[4] as f64,
        cov[5] as f64,
        cov[6] as f64,
        cov[7] as f64,
        cov[8] as f64,
    ];

    // Check for any NaN/Inf in input covariance
    if cov64.iter().any(|e| !e.is_finite()) {
        return invalid_result;
    }

    // Use nalgebra for robust eigenvalue decomposition (matching CPU types.rs)
    let cov_matrix = nalgebra::Matrix3::new(
        cov64[0], cov64[1], cov64[2], cov64[3], cov64[4], cov64[5], cov64[6], cov64[7], cov64[8],
    );

    // Symmetric eigenvalue decomposition
    let eigen = cov_matrix.symmetric_eigen();
    let mut eigenvalues = eigen.eigenvalues;

    // Check for non-finite eigenvalues
    if eigenvalues.iter().any(|e| !e.is_finite()) {
        return invalid_result;
    }

    // Find max eigenvalue and min eigenvalue index
    let max_eigenvalue = eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let min_eigenvalue_idx = eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(2);

    // Handle degenerate case where all eigenvalues are near zero
    if max_eigenvalue <= 0.0 {
        return RegularizationResult {
            inv_covariance: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            principal_axis: [0.0, 0.0, 1.0],
            is_valid: true,
        };
    }

    // Minimum allowed eigenvalue (Autoware uses 0.01 ratio)
    let min_eigenvalue = 0.01 * max_eigenvalue;

    // Clamp small eigenvalues
    for ev in eigenvalues.iter_mut() {
        if *ev < min_eigenvalue {
            *ev = min_eigenvalue;
        }
    }

    // Compute inverse: V * D^{-1} * V^T
    let eigenvectors = &eigen.eigenvectors;
    let inv_eigenvalues = nalgebra::Vector3::new(
        1.0 / eigenvalues[0],
        1.0 / eigenvalues[1],
        1.0 / eigenvalues[2],
    );
    let inv_diag = nalgebra::Matrix3::from_diagonal(&inv_eigenvalues);
    let inverse = eigenvectors * inv_diag * eigenvectors.transpose();

    // Check for non-finite values in result
    if inverse.iter().any(|e| !e.is_finite()) {
        return invalid_result;
    }

    // Extract principal axis (eigenvector of smallest eigenvalue)
    let principal_col = eigenvectors.column(min_eigenvalue_idx);

    RegularizationResult {
        inv_covariance: [
            inverse[(0, 0)] as f32,
            inverse[(0, 1)] as f32,
            inverse[(0, 2)] as f32,
            inverse[(1, 0)] as f32,
            inverse[(1, 1)] as f32,
            inverse[(1, 2)] as f32,
            inverse[(2, 0)] as f32,
            inverse[(2, 1)] as f32,
            inverse[(2, 2)] as f32,
        ],
        principal_axis: [
            principal_col[0] as f32,
            principal_col[1] as f32,
            principal_col[2] as f32,
        ],
        is_valid: true,
    }
}

/// Compute complete voxel statistics from sorted points and segments.
///
/// This is the main entry point that chains all three passes.
///
/// # Arguments
/// * `points` - Flat array of point coordinates
/// * `sorted_indices` - Original point indices after Morton sort
/// * `segment_ids` - Voxel ID for each point
/// * `num_voxels` - Total number of voxels
/// * `min_points` - Minimum points required for valid voxel
pub fn compute_voxel_statistics_cpu(
    points: &[f32],
    sorted_indices: &[u32],
    segment_ids: &[u32],
    num_voxels: u32,
    min_points: u32,
) -> VoxelStatistics {
    // Pass 1: Compute sums and counts
    let sums = compute_voxel_sums_cpu(points, sorted_indices, segment_ids, num_voxels);

    // Compute means from sums
    let means = compute_means_from_sums(&sums);

    // Pass 2: Compute covariance sums
    let cov_sums =
        compute_covariance_sums_cpu(points, sorted_indices, segment_ids, &means, num_voxels);

    // Pass 3: Finalize covariances
    finalize_voxels_cpu(means, cov_sums, sums.counts, min_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_sums_single_voxel() {
        // Three points in a single voxel
        let points = vec![
            1.0, 2.0, 3.0, // Point 0
            1.5, 2.5, 3.5, // Point 1
            1.2, 2.2, 3.2, // Point 2
        ];
        let sorted_indices = vec![0, 1, 2];
        let segment_ids = vec![0, 0, 0]; // All same voxel

        let sums = compute_voxel_sums_cpu(&points, &sorted_indices, &segment_ids, 1);

        assert_eq!(sums.counts[0], 3);
        assert!((sums.position_sums[0] - 3.7).abs() < 1e-5); // 1.0 + 1.5 + 1.2
        assert!((sums.position_sums[1] - 6.7).abs() < 1e-5); // 2.0 + 2.5 + 2.2
        assert!((sums.position_sums[2] - 9.7).abs() < 1e-5); // 3.0 + 3.5 + 3.2
    }

    #[test]
    fn test_voxel_sums_multiple_voxels() {
        // Points in two voxels
        let points = vec![
            0.0, 0.0, 0.0, // Point 0 -> voxel 0
            1.0, 1.0, 1.0, // Point 1 -> voxel 0
            10.0, 10.0, 10.0, // Point 2 -> voxel 1
        ];
        let sorted_indices = vec![0, 1, 2];
        let segment_ids = vec![0, 0, 1];

        let sums = compute_voxel_sums_cpu(&points, &sorted_indices, &segment_ids, 2);

        assert_eq!(sums.counts[0], 2);
        assert_eq!(sums.counts[1], 1);
        assert!((sums.position_sums[0] - 1.0).abs() < 1e-5); // voxel 0: 0+1
        assert!((sums.position_sums[3] - 10.0).abs() < 1e-5); // voxel 1: 10
    }

    #[test]
    fn test_means_from_sums() {
        let sums = VoxelSums {
            position_sums: vec![3.0, 6.0, 9.0, 10.0, 20.0, 30.0],
            counts: vec![3, 2],
            num_voxels: 2,
        };

        let means = compute_means_from_sums(&sums);

        assert!((means[0] - 1.0).abs() < 1e-5); // 3/3
        assert!((means[1] - 2.0).abs() < 1e-5); // 6/3
        assert!((means[2] - 3.0).abs() < 1e-5); // 9/3
        assert!((means[3] - 5.0).abs() < 1e-5); // 10/2
        assert!((means[4] - 10.0).abs() < 1e-5); // 20/2
        assert!((means[5] - 15.0).abs() < 1e-5); // 30/2
    }

    #[test]
    fn test_covariance_sums() {
        // Points exactly at mean - should produce zero covariance
        let points = vec![1.0, 2.0, 3.0];
        let sorted_indices = vec![0];
        let segment_ids = vec![0];
        let means = vec![1.0, 2.0, 3.0];

        let cov_sums =
            compute_covariance_sums_cpu(&points, &sorted_indices, &segment_ids, &means, 1);

        // All covariance sums should be zero (point at mean)
        for &val in cov_sums.iter().take(9) {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_full_statistics_pipeline() {
        // Create a simple point cloud with two voxels
        let points = vec![
            // Voxel 0: cluster around (0, 0, 0)
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.05, -0.05, 0.05,
            // Voxel 1: cluster around (10, 10, 10)
            10.0, 10.0, 10.0, 10.1, 10.1, 10.1, 9.9, 9.9, 9.9,
        ];
        let sorted_indices: Vec<u32> = (0..7).collect();
        let segment_ids = vec![0, 0, 0, 0, 1, 1, 1];

        let stats = compute_voxel_statistics_cpu(&points, &sorted_indices, &segment_ids, 2, 3);

        assert_eq!(stats.num_voxels, 2);
        assert_eq!(stats.point_counts[0], 4);
        assert_eq!(stats.point_counts[1], 3);
        assert!(stats.valid[0]);
        assert!(stats.valid[1]);

        // Check means approximately
        assert!((stats.means[0] - 0.0125).abs() < 0.01); // mean x of voxel 0
        assert!((stats.means[3] - 10.0).abs() < 0.1); // mean x of voxel 1
    }

    #[test]
    fn test_min_points_filter() {
        let points = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // Only 2 points
        ];
        let sorted_indices: Vec<u32> = (0..2).collect();
        let segment_ids = vec![0, 0];

        let stats = compute_voxel_statistics_cpu(&points, &sorted_indices, &segment_ids, 1, 3);

        // With min_points=3, this voxel should be invalid
        assert!(!stats.valid[0]);
    }
}
