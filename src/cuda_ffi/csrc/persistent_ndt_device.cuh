// Device functions for persistent NDT kernel
//
// These inline functions compute transforms, Jacobians, and Hessians
// for the NDT optimization. Designed to be called from the persistent kernel.

#ifndef PERSISTENT_NDT_DEVICE_CUH
#define PERSISTENT_NDT_DEVICE_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Maximum neighbors per query point (must match voxel_hash.cu)
constexpr uint32_t MAX_NEIGHBORS = 8;

// ============================================================================
// Trigonometric functions
// ============================================================================

__device__ __forceinline__ void compute_sincos_inline(
    const float* pose,  // [6]: tx, ty, tz, roll, pitch, yaw
    float* sr, float* cr,
    float* sp, float* cp,
    float* sy, float* cy
) {
    *sr = sinf(pose[3]);  // roll
    *cr = cosf(pose[3]);
    *sp = sinf(pose[4]);  // pitch
    *cp = cosf(pose[4]);
    *sy = sinf(pose[5]);  // yaw
    *cy = cosf(pose[5]);
}

// ============================================================================
// Transform matrix (4x4 homogeneous)
// ============================================================================

__device__ __forceinline__ void compute_transform_inline(
    const float* pose,
    float sr, float cr, float sp, float cp, float sy, float cy,
    float* T  // [16] row-major
) {
    // Row 0
    T[0]  = cy * cp;
    T[1]  = cy * sp * sr - sy * cr;
    T[2]  = cy * sp * cr + sy * sr;
    T[3]  = pose[0];
    // Row 1
    T[4]  = sy * cp;
    T[5]  = sy * sp * sr + cy * cr;
    T[6]  = sy * sp * cr - cy * sr;
    T[7]  = pose[1];
    // Row 2
    T[8]  = -sp;
    T[9]  = cp * sr;
    T[10] = cp * cr;
    T[11] = pose[2];
    // Row 3
    T[12] = 0.0f;
    T[13] = 0.0f;
    T[14] = 0.0f;
    T[15] = 1.0f;
}

// ============================================================================
// Point transform
// ============================================================================

__device__ __forceinline__ void transform_point_inline(
    float px, float py, float pz,
    const float* T,
    float* tx, float* ty, float* tz
) {
    *tx = T[0] * px + T[1] * py + T[2]  * pz + T[3];
    *ty = T[4] * px + T[5] * py + T[6]  * pz + T[7];
    *tz = T[8] * px + T[9] * py + T[10] * pz + T[11];
}

// ============================================================================
// Jacobian computation (3x6 = 18 values)
// ============================================================================

__device__ __forceinline__ void compute_jacobians_inline(
    float x, float y, float z,  // source point
    float sr, float cr, float sp, float cp, float sy, float cy,
    float* J  // [18] row-major: 3 rows (xyz) x 6 cols (tx,ty,tz,roll,pitch,yaw)
) {
    // dT/d(tx) - column 0
    J[0]  = 1.0f;
    J[6]  = 0.0f;
    J[12] = 0.0f;

    // dT/d(ty) - column 1
    J[1]  = 0.0f;
    J[7]  = 1.0f;
    J[13] = 0.0f;

    // dT/d(tz) - column 2
    J[2]  = 0.0f;
    J[8]  = 0.0f;
    J[14] = 1.0f;

    // dT/d(roll) - column 3 (j_ang_a)
    J[3]  = (cy * sp * cr + sy * sr) * y + (-cy * sp * sr + sy * cr) * z;
    J[9]  = (sy * sp * cr - cy * sr) * y + (-sy * sp * sr - cy * cr) * z;
    J[15] = cp * cr * y - cp * sr * z;

    // dT/d(pitch) - column 4 (j_ang_b)
    J[4]  = -cy * sp * x + cy * cp * sr * y + cy * cp * cr * z;
    J[10] = -sy * sp * x + sy * cp * sr * y + sy * cp * cr * z;
    J[16] = -cp * x - sp * sr * y - sp * cr * z;

    // dT/d(yaw) - column 5 (j_ang_c)
    J[5]  = -sy * cp * x + (-sy * sp * sr - cy * cr) * y + (-sy * sp * cr + cy * sr) * z;
    J[11] = cy * cp * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z;
    J[17] = 0.0f;
}

// ============================================================================
// Point Hessians (sparse: 15 non-zero values)
// ============================================================================

__device__ __forceinline__ void compute_point_hessians_inline(
    float x, float y, float z,
    float sr, float cr, float sp, float cp, float sy, float cy,
    float* pH  // [15]: a2,a3,b2,b3,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3
) {
    // d^2T/d(roll)^2 terms
    pH[0]  = (-cy * sp * sr + sy * cr) * y + (-cy * sp * cr - sy * sr) * z;  // a2
    pH[1]  = (-sy * sp * sr - cy * cr) * y + (-sy * sp * cr + cy * sr) * z;  // a3
    pH[2]  = -cp * sr * y - cp * cr * z;                                      // b2

    // d^2T/d(pitch)^2 terms
    pH[3]  = -cy * cp * x - cy * sp * sr * y - cy * sp * cr * z;             // b3
    pH[4]  = -sy * cp * x - sy * sp * sr * y - sy * sp * cr * z;             // c2
    pH[5]  = sp * x - cp * sr * y - cp * cr * z;                              // c3

    // d^2T/d(yaw)^2 terms
    pH[6]  = -cy * cp * x + (-cy * sp * sr + sy * cr) * y + (-cy * sp * cr - sy * sr) * z;  // d1
    pH[7]  = -sy * cp * x + (-sy * sp * sr - cy * cr) * y + (-sy * sp * cr + cy * sr) * z;  // d2
    pH[8]  = 0.0f;                                                            // d3

    // d^2T/d(roll)d(pitch) terms
    pH[9]  = cy * cp * cr * y - cy * cp * sr * z;                             // e1
    pH[10] = sy * cp * cr * y - sy * cp * sr * z;                             // e2
    pH[11] = -sp * cr * y + sp * sr * z;                                      // e3

    // d^2T/d(roll)d(yaw) terms
    pH[12] = (-sy * sp * cr - cy * sr) * y + (sy * sp * sr - cy * cr) * z;    // f1
    pH[13] = (cy * sp * cr - sy * sr) * y + (-cy * sp * sr - sy * cr) * z;    // f2
    pH[14] = 0.0f;                                                            // f3
}

// ============================================================================
// NDT score, gradient, and Hessian contribution from one voxel
// ============================================================================

__device__ __forceinline__ void compute_ndt_contribution(
    float tx, float ty, float tz,                      // transformed point
    const float* voxel_mean,                           // [3]
    const float* voxel_inv_cov,                        // [9] row-major
    const float* J,                                    // [18] Jacobian
    const float* pH,                                   // [15] sparse point Hessian
    float gauss_d1, float gauss_d2,
    float* score_out,                                  // scalar output
    float* grad_out,                                   // [6] output
    float* hess_out                                    // [21] upper triangle output
) {
    // x_trans = transformed - mean
    float xt = tx - voxel_mean[0];
    float yt = ty - voxel_mean[1];
    float zt = tz - voxel_mean[2];

    // cx = inv_cov * x_trans
    float cx = voxel_inv_cov[0] * xt + voxel_inv_cov[1] * yt + voxel_inv_cov[2] * zt;
    float cy = voxel_inv_cov[3] * xt + voxel_inv_cov[4] * yt + voxel_inv_cov[5] * zt;
    float cz = voxel_inv_cov[6] * xt + voxel_inv_cov[7] * yt + voxel_inv_cov[8] * zt;

    // x_c_x = x_trans' * inv_cov * x_trans
    float x_c_x = xt * cx + yt * cy + zt * cz;

    // NDT score contribution: -d1 * exp(-d2/2 * x_c_x)
    float e_x_cov_x = gauss_d1 * expf(-gauss_d2 * 0.5f * x_c_x);
    *score_out = -e_x_cov_x;

    // Gradient: e_x_cov_x * d2 * J' * cx
    // J is 3x6, cx is 3x1, result is 6x1
    float factor = e_x_cov_x * gauss_d2;
    for (int i = 0; i < 6; i++) {
        // J column i: J[i], J[6+i], J[12+i]
        grad_out[i] = factor * (J[i] * cx + J[6 + i] * cy + J[12 + i] * cz);
    }

    // Hessian computation (upper triangle, 21 values)
    // H_ij = e_x_cov_x * d2 * (J_i' * inv_cov * J_j - d2 * (J_i' * cx) * (J_j' * cx))
    //      + e_x_cov_x * (x_trans' * inv_cov * H_ij) for angular terms

    // Pre-compute J' * inv_cov for each column
    float Jc[6][3];  // Jc[col] = inv_cov * J_col
    for (int col = 0; col < 6; col++) {
        float j0 = J[col];
        float j1 = J[6 + col];
        float j2 = J[12 + col];
        Jc[col][0] = voxel_inv_cov[0] * j0 + voxel_inv_cov[1] * j1 + voxel_inv_cov[2] * j2;
        Jc[col][1] = voxel_inv_cov[3] * j0 + voxel_inv_cov[4] * j1 + voxel_inv_cov[5] * j2;
        Jc[col][2] = voxel_inv_cov[6] * j0 + voxel_inv_cov[7] * j1 + voxel_inv_cov[8] * j2;
    }

    // Pre-compute J' * cx for each column
    float Jcx[6];
    for (int col = 0; col < 6; col++) {
        Jcx[col] = J[col] * cx + J[6 + col] * cy + J[12 + col] * cz;
    }

    // Sparse point Hessian contribution mapping:
    // pH[0..2]   = a2, a3, b2  -> d^2/d(roll)^2, rows 0-2
    // pH[3..5]   = b3, c2, c3  -> d^2/d(pitch)^2, rows 0-2
    // pH[6..8]   = d1, d2, d3  -> d^2/d(yaw)^2, rows 0-2
    // pH[9..11]  = e1, e2, e3  -> d^2/d(roll)d(pitch), rows 0-2
    // pH[12..14] = f1, f2, f3  -> d^2/d(roll)d(yaw), rows 0-2

    // Compute x_trans' * inv_cov = [cx, cy, cz] (already computed as cx, cy, cz)
    // Point Hessian contribution: x_trans' * inv_cov * pH_col

    int idx = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = i; j < 6; j++) {
            // J_i' * inv_cov * J_j
            float JiCJj = J[i] * Jc[j][0] + J[6 + i] * Jc[j][1] + J[12 + i] * Jc[j][2];

            // Main Hessian term
            float h = factor * (JiCJj - gauss_d2 * Jcx[i] * Jcx[j]);

            // Point Hessian contribution (only for angular terms i,j >= 3)
            if (i >= 3 && j >= 3) {
                // Map (i,j) to point Hessian index
                // (3,3)->roll,roll    (3,4)->roll,pitch   (3,5)->roll,yaw
                // (4,4)->pitch,pitch  (4,5)->pitch,yaw
                // (5,5)->yaw,yaw
                int pi = i - 3;  // 0=roll, 1=pitch, 2=yaw
                int pj = j - 3;

                float pH_contrib = 0.0f;
                if (pi == 0 && pj == 0) {
                    // d^2/d(roll)^2: a2, a3, b2
                    pH_contrib = cx * pH[0] + cy * pH[1] + cz * pH[2];
                } else if (pi == 1 && pj == 1) {
                    // d^2/d(pitch)^2: b3, c2, c3
                    pH_contrib = cx * pH[3] + cy * pH[4] + cz * pH[5];
                } else if (pi == 2 && pj == 2) {
                    // d^2/d(yaw)^2: d1, d2, d3
                    pH_contrib = cx * pH[6] + cy * pH[7] + cz * pH[8];
                } else if ((pi == 0 && pj == 1) || (pi == 1 && pj == 0)) {
                    // d^2/d(roll)d(pitch): e1, e2, e3
                    pH_contrib = cx * pH[9] + cy * pH[10] + cz * pH[11];
                } else if ((pi == 0 && pj == 2) || (pi == 2 && pj == 0)) {
                    // d^2/d(roll)d(yaw): f1, f2, f3
                    pH_contrib = cx * pH[12] + cy * pH[13] + cz * pH[14];
                }
                // Note: d^2/d(pitch)d(yaw) has no point Hessian contribution (all zeros)

                h += e_x_cov_x * gauss_d2 * pH_contrib;
            }

            hess_out[idx++] = h;
        }
    }
}

#endif // PERSISTENT_NDT_DEVICE_CUH
