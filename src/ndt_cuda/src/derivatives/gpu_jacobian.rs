//! GPU kernels for computing Jacobians and Point Hessians.
//!
//! These kernels compute the transformation derivatives directly on GPU,
//! eliminating the need to upload pre-computed Jacobians each iteration.
//!
//! The Jacobian is the 3x6 matrix ∂T(p)/∂pose where T transforms a point
//! by the current pose (translation + rotation).
//!
//! The Point Hessian is the 24x6 matrix of second derivatives.
//!
//! Based on angular.rs formulas (Magnusson 2009, Equations 6.19 and 6.21).

use cubecl::prelude::*;

/// Compute sin/cos values from pose angles.
///
/// # Arguments
/// * `pose` - [6]: tx, ty, tz, roll, pitch, yaw
/// * `sin_cos` - [6] output: sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw)
#[cube(launch_unchecked)]
pub fn compute_sin_cos_kernel<F: Float>(pose: &Array<F>, sin_cos: &mut Array<F>) {
    // Single thread computes 6 trig values
    let roll = pose[3];
    let pitch = pose[4];
    let yaw = pose[5];

    sin_cos[0] = F::sin(roll);
    sin_cos[1] = F::cos(roll);
    sin_cos[2] = F::sin(pitch);
    sin_cos[3] = F::cos(pitch);
    sin_cos[4] = F::sin(yaw);
    sin_cos[5] = F::cos(yaw);
}

/// Compute per-point Jacobians from sin/cos and point coordinates.
///
/// The Jacobian is a 3x6 matrix stored row-major (18 elements per point):
/// - Row 0: ∂x'/∂(tx, ty, tz, roll, pitch, yaw)
/// - Row 1: ∂y'/∂(tx, ty, tz, roll, pitch, yaw)
/// - Row 2: ∂z'/∂(tx, ty, tz, roll, pitch, yaw)
///
/// Columns 0-2 are identity (translation), columns 3-5 are rotation derivatives.
///
/// # Arguments
/// * `source_points` - [N * 3]: source point coordinates
/// * `sin_cos` - [6]: sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw)
/// * `num_points` - number of points
/// * `jacobians` - [N * 18] output: 3x6 Jacobian per point, row-major
#[cube(launch_unchecked)]
pub fn compute_jacobians_kernel<F: Float>(
    source_points: &Array<F>,
    sin_cos: &Array<F>,
    num_points: u32,
    jacobians: &mut Array<F>,
) {
    let idx = ABSOLUTE_POS;

    if idx >= num_points {
        terminate!();
    }

    // Load point
    let base = idx * 3;
    let x = source_points[base];
    let y = source_points[base + 1];
    let z = source_points[base + 2];

    // Load sin/cos values
    let sr = sin_cos[0]; // sin(roll)
    let cr = sin_cos[1]; // cos(roll)
    let sp = sin_cos[2]; // sin(pitch)
    let cp = sin_cos[3]; // cos(pitch)
    let sy = sin_cos[4]; // sin(yaw)
    let cy = sin_cos[5]; // cos(yaw)

    // Output base index
    let jbase = idx * 18;

    // Row 0: ∂x'/∂(tx, ty, tz, roll, pitch, yaw)
    jacobians[jbase] = F::new(1.0); // ∂x'/∂tx
    jacobians[jbase + 1] = F::new(0.0); // ∂x'/∂ty
    jacobians[jbase + 2] = F::new(0.0); // ∂x'/∂tz
    jacobians[jbase + 3] = F::new(0.0); // ∂x'/∂roll = 0
                                        // ∂x'/∂pitch: j_ang[(2,:)] · [x, y, z] = [-sp*cy, sp*sy, cp] · [x, y, z]
    jacobians[jbase + 4] = (F::new(0.0) - sp * cy) * x + (sp * sy) * y + cp * z;
    // ∂x'/∂yaw: j_ang[(5,:)] · [x, y, z] = [-cp*sy, -cp*cy, 0] · [x, y, z]
    jacobians[jbase + 5] = (F::new(0.0) - cp * sy) * x + (F::new(0.0) - cp * cy) * y;

    // Row 1: ∂y'/∂(tx, ty, tz, roll, pitch, yaw)
    jacobians[jbase + 6] = F::new(0.0); // ∂y'/∂tx
    jacobians[jbase + 7] = F::new(1.0); // ∂y'/∂ty
    jacobians[jbase + 8] = F::new(0.0); // ∂y'/∂tz
                                        // ∂y'/∂roll: j_ang[(0,:)] · [x, y, z] = [-sr*sy + cr*sp*cy, -sr*cy - cr*sp*sy, -cr*cp] · [x, y, z]
    jacobians[jbase + 9] = (F::new(0.0) - sr * sy + cr * sp * cy) * x
        + (F::new(0.0) - sr * cy - cr * sp * sy) * y
        + (F::new(0.0) - cr * cp) * z;
    // ∂y'/∂pitch: j_ang[(3,:)] · [x, y, z] = [sr*cp*cy, -sr*cp*sy, sr*sp] · [x, y, z]
    jacobians[jbase + 10] = (sr * cp * cy) * x + (F::new(0.0) - sr * cp * sy) * y + (sr * sp) * z;
    // ∂y'/∂yaw: j_ang[(6,:)] · [x, y, z] = [cr*cy - sr*sp*sy, -cr*sy - sr*sp*cy, 0] · [x, y, z]
    jacobians[jbase + 11] =
        (cr * cy - sr * sp * sy) * x + (F::new(0.0) - cr * sy - sr * sp * cy) * y;

    // Row 2: ∂z'/∂(tx, ty, tz, roll, pitch, yaw)
    jacobians[jbase + 12] = F::new(0.0); // ∂z'/∂tx
    jacobians[jbase + 13] = F::new(0.0); // ∂z'/∂ty
    jacobians[jbase + 14] = F::new(1.0); // ∂z'/∂tz
                                         // ∂z'/∂roll: j_ang[(1,:)] · [x, y, z] = [cr*sy + sr*sp*cy, cr*cy - sr*sp*sy, -sr*cp] · [x, y, z]
    jacobians[jbase + 15] =
        (cr * sy + sr * sp * cy) * x + (cr * cy - sr * sp * sy) * y + (F::new(0.0) - sr * cp) * z;
    // ∂z'/∂pitch: j_ang[(4,:)] · [x, y, z] = [-cr*cp*cy, cr*cp*sy, -cr*sp] · [x, y, z]
    jacobians[jbase + 16] =
        (F::new(0.0) - cr * cp * cy) * x + (cr * cp * sy) * y + (F::new(0.0) - cr * sp) * z;
    // ∂z'/∂yaw: j_ang[(7,:)] · [x, y, z] = [sr*cy + cr*sp*sy, cr*sp*cy - sr*sy, 0] · [x, y, z]
    jacobians[jbase + 17] = (sr * cy + cr * sp * sy) * x + (cr * sp * cy - sr * sy) * y;
}

/// Compute per-point Hessians from sin/cos and point coordinates.
///
/// The Point Hessian is a 24x6 matrix stored row-major (144 elements per point).
/// Only rows 12-23 and columns 3-5 have non-zero values (angular second derivatives).
///
/// # Arguments
/// * `source_points` - [N * 3]: source point coordinates
/// * `sin_cos` - [6]: sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw)
/// * `num_points` - number of points
/// * `point_hessians` - [N * 144] output: 24x6 Hessian per point, row-major
#[cube(launch_unchecked)]
pub fn compute_point_hessians_kernel<F: Float>(
    source_points: &Array<F>,
    sin_cos: &Array<F>,
    num_points: u32,
    point_hessians: &mut Array<F>,
) {
    let idx = ABSOLUTE_POS;

    if idx >= num_points {
        terminate!();
    }

    // Load point
    let pbase = idx * 3;
    let x = source_points[pbase];
    let y = source_points[pbase + 1];
    let z = source_points[pbase + 2];

    // Load sin/cos values
    let sr = sin_cos[0]; // sin(roll)
    let cr = sin_cos[1]; // cos(roll)
    let sp = sin_cos[2]; // sin(pitch)
    let cp = sin_cos[3]; // cos(pitch)
    let sy = sin_cos[4]; // sin(yaw)
    let cy = sin_cos[5]; // cos(yaw)

    // Precompute angular Hessian terms (h_ang from angular.rs)
    // Each term is a 3-element vector that gets dotted with [x, y, z]

    // a2: ∂²R/∂roll² for y-coordinate
    let a2_dot = (F::new(0.0) - cr * sy - sr * sp * cy) * x
        + (F::new(0.0) - cr * cy + sr * sp * sy) * y
        + (sr * cp) * z;

    // a3: ∂²R/∂roll² for z-coordinate
    let a3_dot = (F::new(0.0) - sr * sy + cr * sp * cy) * x
        + (F::new(0.0) - cr * sp * sy - sr * cy) * y
        + (F::new(0.0) - cr * cp) * z;

    // b2: ∂²R/∂roll∂pitch for y-coordinate
    let b2_dot = (cr * cp * cy) * x + (F::new(0.0) - cr * cp * sy) * y + (cr * sp) * z;

    // b3: ∂²R/∂roll∂pitch for z-coordinate
    let b3_dot = (sr * cp * cy) * x + (F::new(0.0) - sr * cp * sy) * y + (sr * sp) * z;

    // c2: ∂²R/∂roll∂yaw for y-coordinate
    let c2_dot = (F::new(0.0) - sr * cy - cr * sp * sy) * x + (sr * sy - cr * sp * cy) * y;

    // c3: ∂²R/∂roll∂yaw for z-coordinate
    let c3_dot = (cr * cy - sr * sp * sy) * x + (F::new(0.0) - sr * sp * cy - cr * sy) * y;

    // d1: ∂²R/∂pitch² for x-coordinate
    let d1_dot = (F::new(0.0) - cp * cy) * x + (cp * sy) * y + (sp) * z;

    // d2: ∂²R/∂pitch² for y-coordinate
    let d2_dot = (F::new(0.0) - sr * sp * cy) * x + (sr * sp * sy) * y + (sr * cp) * z;

    // d3: ∂²R/∂pitch² for z-coordinate
    let d3_dot =
        (cr * sp * cy) * x + (F::new(0.0) - cr * sp * sy) * y + (F::new(0.0) - cr * cp) * z;

    // e1: ∂²R/∂pitch∂yaw for x-coordinate
    let e1_dot = (sp * sy) * x + (sp * cy) * y;

    // e2: ∂²R/∂pitch∂yaw for y-coordinate
    let e2_dot = (F::new(0.0) - sr * cp * sy) * x + (F::new(0.0) - sr * cp * cy) * y;

    // e3: ∂²R/∂pitch∂yaw for z-coordinate
    let e3_dot = (cr * cp * sy) * x + (cr * cp * cy) * y;

    // f1: ∂²R/∂yaw² for x-coordinate
    let f1_dot = (F::new(0.0) - cp * cy) * x + (cp * sy) * y;

    // f2: ∂²R/∂yaw² for y-coordinate
    let f2_dot =
        (F::new(0.0) - cr * sy - sr * sp * cy) * x + (F::new(0.0) - cr * cy + sr * sp * sy) * y;

    // f3: ∂²R/∂yaw² for z-coordinate
    let f3_dot =
        (F::new(0.0) - sr * sy + cr * sp * cy) * x + (F::new(0.0) - cr * sp * sy - sr * cy) * y;

    // Output base index
    let hbase = idx * 144;

    // Initialize all to zero (rows 0-11 are all zeros)
    // CubeCL doesn't support dynamic loops well, so we unroll
    // Rows 0-11: 72 zeros
    let zero = F::new(0.0);

    // Row 0-11: all zeros (72 elements)
    // Unroll manually for first 72 elements
    jacobians_set_zeros_72(point_hessians, hbase);

    // Row 12: all zeros (6 elements) - x component of roll-* terms
    let row12_base = hbase + 72;
    point_hessians[row12_base] = zero;
    point_hessians[row12_base + 1] = zero;
    point_hessians[row12_base + 2] = zero;
    point_hessians[row12_base + 3] = zero;
    point_hessians[row12_base + 4] = zero;
    point_hessians[row12_base + 5] = zero;

    // Row 13: [0, 0, 0, a2, b2, c2] - y component
    let row13_base = hbase + 78;
    point_hessians[row13_base] = zero;
    point_hessians[row13_base + 1] = zero;
    point_hessians[row13_base + 2] = zero;
    point_hessians[row13_base + 3] = a2_dot;
    point_hessians[row13_base + 4] = b2_dot;
    point_hessians[row13_base + 5] = c2_dot;

    // Row 14: [0, 0, 0, a3, b3, c3] - z component
    let row14_base = hbase + 84;
    point_hessians[row14_base] = zero;
    point_hessians[row14_base + 1] = zero;
    point_hessians[row14_base + 2] = zero;
    point_hessians[row14_base + 3] = a3_dot;
    point_hessians[row14_base + 4] = b3_dot;
    point_hessians[row14_base + 5] = c3_dot;

    // Row 15: all zeros (6 elements) - w component
    let row15_base = hbase + 90;
    point_hessians[row15_base] = zero;
    point_hessians[row15_base + 1] = zero;
    point_hessians[row15_base + 2] = zero;
    point_hessians[row15_base + 3] = zero;
    point_hessians[row15_base + 4] = zero;
    point_hessians[row15_base + 5] = zero;

    // Row 16: [0, 0, 0, 0, d1, e1] - x component of pitch-* terms
    let row16_base = hbase + 96;
    point_hessians[row16_base] = zero;
    point_hessians[row16_base + 1] = zero;
    point_hessians[row16_base + 2] = zero;
    point_hessians[row16_base + 3] = zero;
    point_hessians[row16_base + 4] = d1_dot;
    point_hessians[row16_base + 5] = e1_dot;

    // Row 17: [0, 0, 0, b2, d2, e2] - y component
    let row17_base = hbase + 102;
    point_hessians[row17_base] = zero;
    point_hessians[row17_base + 1] = zero;
    point_hessians[row17_base + 2] = zero;
    point_hessians[row17_base + 3] = b2_dot;
    point_hessians[row17_base + 4] = d2_dot;
    point_hessians[row17_base + 5] = e2_dot;

    // Row 18: [0, 0, 0, b3, d3, e3] - z component
    let row18_base = hbase + 108;
    point_hessians[row18_base] = zero;
    point_hessians[row18_base + 1] = zero;
    point_hessians[row18_base + 2] = zero;
    point_hessians[row18_base + 3] = b3_dot;
    point_hessians[row18_base + 4] = d3_dot;
    point_hessians[row18_base + 5] = e3_dot;

    // Row 19: all zeros (6 elements) - w component
    let row19_base = hbase + 114;
    point_hessians[row19_base] = zero;
    point_hessians[row19_base + 1] = zero;
    point_hessians[row19_base + 2] = zero;
    point_hessians[row19_base + 3] = zero;
    point_hessians[row19_base + 4] = zero;
    point_hessians[row19_base + 5] = zero;

    // Row 20: [0, 0, 0, 0, e1, f1] - x component of yaw-* terms
    let row20_base = hbase + 120;
    point_hessians[row20_base] = zero;
    point_hessians[row20_base + 1] = zero;
    point_hessians[row20_base + 2] = zero;
    point_hessians[row20_base + 3] = zero;
    point_hessians[row20_base + 4] = e1_dot;
    point_hessians[row20_base + 5] = f1_dot;

    // Row 21: [0, 0, 0, c2, e2, f2] - y component
    let row21_base = hbase + 126;
    point_hessians[row21_base] = zero;
    point_hessians[row21_base + 1] = zero;
    point_hessians[row21_base + 2] = zero;
    point_hessians[row21_base + 3] = c2_dot;
    point_hessians[row21_base + 4] = e2_dot;
    point_hessians[row21_base + 5] = f2_dot;

    // Row 22: [0, 0, 0, c3, e3, f3] - z component
    let row22_base = hbase + 132;
    point_hessians[row22_base] = zero;
    point_hessians[row22_base + 1] = zero;
    point_hessians[row22_base + 2] = zero;
    point_hessians[row22_base + 3] = c3_dot;
    point_hessians[row22_base + 4] = e3_dot;
    point_hessians[row22_base + 5] = f3_dot;

    // Row 23: all zeros (6 elements) - w component
    let row23_base = hbase + 138;
    point_hessians[row23_base] = zero;
    point_hessians[row23_base + 1] = zero;
    point_hessians[row23_base + 2] = zero;
    point_hessians[row23_base + 3] = zero;
    point_hessians[row23_base + 4] = zero;
    point_hessians[row23_base + 5] = zero;
}

/// Helper to set 72 zeros (rows 0-11 of point Hessian).
/// Fully unrolled to avoid CubeCL dynamic indexing issues.
#[cube]
fn jacobians_set_zeros_72<F: Float>(arr: &mut Array<F>, base: u32) {
    let zero = F::new(0.0);
    // Row 0 (6 elements)
    arr[base] = zero;
    arr[base + 1] = zero;
    arr[base + 2] = zero;
    arr[base + 3] = zero;
    arr[base + 4] = zero;
    arr[base + 5] = zero;
    // Row 1
    arr[base + 6] = zero;
    arr[base + 7] = zero;
    arr[base + 8] = zero;
    arr[base + 9] = zero;
    arr[base + 10] = zero;
    arr[base + 11] = zero;
    // Row 2
    arr[base + 12] = zero;
    arr[base + 13] = zero;
    arr[base + 14] = zero;
    arr[base + 15] = zero;
    arr[base + 16] = zero;
    arr[base + 17] = zero;
    // Row 3
    arr[base + 18] = zero;
    arr[base + 19] = zero;
    arr[base + 20] = zero;
    arr[base + 21] = zero;
    arr[base + 22] = zero;
    arr[base + 23] = zero;
    // Row 4
    arr[base + 24] = zero;
    arr[base + 25] = zero;
    arr[base + 26] = zero;
    arr[base + 27] = zero;
    arr[base + 28] = zero;
    arr[base + 29] = zero;
    // Row 5
    arr[base + 30] = zero;
    arr[base + 31] = zero;
    arr[base + 32] = zero;
    arr[base + 33] = zero;
    arr[base + 34] = zero;
    arr[base + 35] = zero;
    // Row 6
    arr[base + 36] = zero;
    arr[base + 37] = zero;
    arr[base + 38] = zero;
    arr[base + 39] = zero;
    arr[base + 40] = zero;
    arr[base + 41] = zero;
    // Row 7
    arr[base + 42] = zero;
    arr[base + 43] = zero;
    arr[base + 44] = zero;
    arr[base + 45] = zero;
    arr[base + 46] = zero;
    arr[base + 47] = zero;
    // Row 8
    arr[base + 48] = zero;
    arr[base + 49] = zero;
    arr[base + 50] = zero;
    arr[base + 51] = zero;
    arr[base + 52] = zero;
    arr[base + 53] = zero;
    // Row 9
    arr[base + 54] = zero;
    arr[base + 55] = zero;
    arr[base + 56] = zero;
    arr[base + 57] = zero;
    arr[base + 58] = zero;
    arr[base + 59] = zero;
    // Row 10
    arr[base + 60] = zero;
    arr[base + 61] = zero;
    arr[base + 62] = zero;
    arr[base + 63] = zero;
    arr[base + 64] = zero;
    arr[base + 65] = zero;
    // Row 11
    arr[base + 66] = zero;
    arr[base + 67] = zero;
    arr[base + 68] = zero;
    arr[base + 69] = zero;
    arr[base + 70] = zero;
    arr[base + 71] = zero;
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::derivatives::gpu::{compute_point_hessians_cpu, compute_point_jacobians_cpu};
    use cubecl::cuda::{CudaDevice, CudaRuntime};

    fn get_test_client() -> (
        CudaDevice,
        cubecl::client::ComputeClient<<CudaRuntime as cubecl::Runtime>::Server>,
    ) {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);
        (device, client)
    }
    #[test]
    fn test_gpu_jacobians_match_cpu() {
        let (_device, client) = get_test_client();

        // Test pose
        let pose: [f64; 6] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];

        // Test points
        let points: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, -0.3],
        ];
        let num_points = points.len() as u32;

        // Compute CPU reference
        let cpu_jacobians = compute_point_jacobians_cpu(&points, &pose);

        // Prepare GPU inputs
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let (sr, cr) = (pose[3] as f32).sin_cos();
        let (sp, cp) = (pose[4] as f32).sin_cos();
        let (sy, cy) = (pose[5] as f32).sin_cos();
        let sin_cos: Vec<f32> = vec![sr, cr, sp, cp, sy, cy];

        // Upload to GPU
        let d_points = client.create(f32::as_bytes(&points_flat));
        let d_sin_cos = client.create(f32::as_bytes(&sin_cos));
        let d_jacobians = client.empty(num_points as usize * 18 * std::mem::size_of::<f32>());

        // Launch kernel
        unsafe {
            compute_jacobians_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(num_points, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_points, points_flat.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&d_sin_cos, 6, 1),
                ScalarArg::new(num_points),
                ArrayArg::from_raw_parts::<f32>(&d_jacobians, num_points as usize * 18, 1),
            );
        }

        // Download results
        let gpu_jacobians_bytes = client.read_one(d_jacobians);
        let gpu_jacobians = f32::from_bytes(&gpu_jacobians_bytes);

        // Compare
        assert_eq!(gpu_jacobians.len(), cpu_jacobians.len());
        for (i, (gpu, cpu)) in gpu_jacobians.iter().zip(cpu_jacobians.iter()).enumerate() {
            let diff = (gpu - cpu).abs();
            assert!(
                diff < 1e-5,
                "Jacobian mismatch at index {i}: GPU={gpu}, CPU={cpu}, diff={diff}"
            );
        }
    }
    #[test]
    fn test_gpu_point_hessians_match_cpu() {
        let (_device, client) = get_test_client();

        // Test pose
        let pose: [f64; 6] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];

        // Test points
        let points: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        ];
        let num_points = points.len() as u32;

        // Compute CPU reference
        let cpu_hessians = compute_point_hessians_cpu(&points, &pose);

        // Prepare GPU inputs
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();
        let (sr, cr) = (pose[3] as f32).sin_cos();
        let (sp, cp) = (pose[4] as f32).sin_cos();
        let (sy, cy) = (pose[5] as f32).sin_cos();
        let sin_cos: Vec<f32> = vec![sr, cr, sp, cp, sy, cy];

        // Upload to GPU
        let d_points = client.create(f32::as_bytes(&points_flat));
        let d_sin_cos = client.create(f32::as_bytes(&sin_cos));
        let d_hessians = client.empty(num_points as usize * 144 * std::mem::size_of::<f32>());

        // Launch kernel
        unsafe {
            compute_point_hessians_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(num_points, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&d_points, points_flat.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&d_sin_cos, 6, 1),
                ScalarArg::new(num_points),
                ArrayArg::from_raw_parts::<f32>(&d_hessians, num_points as usize * 144, 1),
            );
        }

        // Download results
        let gpu_hessians_bytes = client.read_one(d_hessians);
        let gpu_hessians = f32::from_bytes(&gpu_hessians_bytes);

        // Compare
        assert_eq!(gpu_hessians.len(), cpu_hessians.len());
        for (i, (gpu, cpu)) in gpu_hessians.iter().zip(cpu_hessians.iter()).enumerate() {
            let diff = (gpu - cpu).abs();
            assert!(
                diff < 1e-5,
                "Point Hessian mismatch at index {i}: GPU={gpu}, CPU={cpu}, diff={diff}"
            );
        }
    }
}
