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
//
// Uses rotation order R = Rx(roll) * Ry(pitch) * Rz(yaw) to match Autoware.
// This is "intrinsic" rotation order (rotate about fixed world axes).
//
// R = Rx * Ry * Rz =
// [cp*cy,              -cp*sy,              sp    ]
// [sr*sp*cy + cr*sy,   -sr*sp*sy + cr*cy,  -sr*cp]
// [sr*sy - cr*sp*cy,    cr*sp*sy + sr*cy,   cr*cp]

__device__ __forceinline__ void compute_transform_inline(
    const float* pose,
    float sr, float cr, float sp, float cp, float sy, float cy,
    float* T  // [16] row-major
) {
    // Row 0
    T[0]  = cp * cy;
    T[1]  = -cp * sy;
    T[2]  = sp;
    T[3]  = pose[0];
    // Row 1
    T[4]  = sr * sp * cy + cr * sy;
    T[5]  = -sr * sp * sy + cr * cy;
    T[6]  = -sr * cp;
    T[7]  = pose[1];
    // Row 2
    T[8]  = sr * sy - cr * sp * cy;
    T[9]  = cr * sp * sy + sr * cy;
    T[10] = cr * cp;
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
//
// Jacobians for R = Rx(roll) * Ry(pitch) * Rz(yaw), matching Autoware's j_ang_.
//
// The Jacobian J is 3x6 where J[row][col] = dT_row/d(param_col).
// Storage: J[row*6 + col] in row-major.
//
// Translation Jacobians (columns 0-2): Identity matrix
// Rotation Jacobians (columns 3-5): Derived from R = Rx * Ry * Rz

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

    // dT/d(roll) - column 3
    // j_ang_.row(0): dT_y/d(roll) = (-sr*sy + cr*sp*cy)*x + (-sr*cy - cr*sp*sy)*y + (-cr*cp)*z
    // j_ang_.row(1): dT_z/d(roll) = (cr*sy + sr*sp*cy)*x + (cr*cy - sr*sp*sy)*y + (-sr*cp)*z
    J[3]  = 0.0f;  // dT_x/d(roll) = 0 (R[0,:] doesn't depend on roll)
    J[9]  = (-sr * sy + cr * sp * cy) * x + (-sr * cy - cr * sp * sy) * y + (-cr * cp) * z;
    J[15] = (cr * sy + sr * sp * cy) * x + (cr * cy - sr * sp * sy) * y + (-sr * cp) * z;

    // dT/d(pitch) - column 4
    // j_ang_.row(2): dT_x/d(pitch) = (-sp*cy)*x + (sp*sy)*y + (cp)*z
    // j_ang_.row(3): dT_y/d(pitch) = (sr*cp*cy)*x + (-sr*cp*sy)*y + (sr*sp)*z
    // j_ang_.row(4): dT_z/d(pitch) = (-cr*cp*cy)*x + (cr*cp*sy)*y + (-cr*sp)*z
    J[4]  = (-sp * cy) * x + (sp * sy) * y + cp * z;
    J[10] = (sr * cp * cy) * x + (-sr * cp * sy) * y + (sr * sp) * z;
    J[16] = (-cr * cp * cy) * x + (cr * cp * sy) * y + (-cr * sp) * z;

    // dT/d(yaw) - column 5
    // j_ang_.row(5): dT_x/d(yaw) = (-cp*sy)*x + (-cp*cy)*y + 0
    // j_ang_.row(6): dT_y/d(yaw) = (cr*cy - sr*sp*sy)*x + (-cr*sy - sr*sp*cy)*y + 0
    // j_ang_.row(7): dT_z/d(yaw) = (sr*cy + cr*sp*sy)*x + (cr*sp*cy - sr*sy)*y + 0
    J[5]  = (-cp * sy) * x + (-cp * cy) * y;
    J[11] = (cr * cy - sr * sp * sy) * x + (-cr * sy - sr * sp * cy) * y;
    J[17] = (sr * cy + cr * sp * sy) * x + (cr * sp * cy - sr * sy) * y;
}

// ============================================================================
// Point Hessians (sparse: 15 non-zero values)
// ============================================================================
//
// These formulas match Autoware's h_ang_ matrix in multigrid_ndt_omp_impl.hpp.
// The h_ang_ rows are dot-producted with [x, y, z] to get scalar values.
//
// Variable mapping (Autoware → CUDA):
//   cx → cr (cos roll), sx → sr (sin roll)
//   cy → cp (cos pitch), sy → sp (sin pitch)
//   cz → cy (cos yaw), sz → sy (sin yaw)
//
// Structure:
//   pH[0,1] = a2, a3 (roll-roll, vector [0, a2, a3])
//   pH[2,3] = b2, b3 (roll-pitch, vector [0, b2, b3])
//   pH[4,5] = c2, c3 (roll-yaw, vector [0, c2, c3])
//   pH[6,7,8] = d1, d2, d3 (pitch-pitch, vector [d1, d2, d3])
//   pH[9,10,11] = e1, e2, e3 (pitch-yaw, vector [e1, e2, e3])
//   pH[12,13,14] = f1, f2, f3 (yaw-yaw, vector [f1, f2, f3])

__device__ __forceinline__ void compute_point_hessians_inline(
    float x, float y, float z,
    float sr, float cr, float sp, float cp, float sy, float cy,
    float* pH  // [15]: a2,a3,b2,b3,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3
) {
    // a2 = h_ang_.row(0) · [x,y,z]
    // Autoware: (-cx*sz - sx*sy*cz), (-cx*cz + sx*sy*sz), sx*cy
    // CUDA:     (-cr*sy - sr*sp*cy), (-cr*cy + sr*sp*sy), sr*cp
    pH[0] = (-cr * sy - sr * sp * cy) * x + (-cr * cy + sr * sp * sy) * y + sr * cp * z;

    // a3 = h_ang_.row(1) · [x,y,z]
    // Autoware: (-sx*sz + cx*sy*cz), (-cx*sy*sz - sx*cz), (-cx*cy)
    // CUDA:     (-sr*sy + cr*sp*cy), (-cr*sp*sy - sr*cy), (-cr*cp)
    pH[1] = (-sr * sy + cr * sp * cy) * x + (-cr * sp * sy - sr * cy) * y + (-cr * cp) * z;

    // b2 = h_ang_.row(2) · [x,y,z]
    // Autoware: (cx*cy*cz), (-cx*cy*sz), (cx*sy)
    // CUDA:     (cr*cp*cy), (-cr*cp*sy), (cr*sp)
    pH[2] = (cr * cp * cy) * x + (-cr * cp * sy) * y + (cr * sp) * z;

    // b3 = h_ang_.row(3) · [x,y,z]
    // Autoware: (sx*cy*cz), (-sx*cy*sz), (sx*sy)
    // CUDA:     (sr*cp*cy), (-sr*cp*sy), (sr*sp)
    pH[3] = (sr * cp * cy) * x + (-sr * cp * sy) * y + (sr * sp) * z;

    // c2 = h_ang_.row(4) · [x,y,z]
    // Autoware: (-sx*cz - cx*sy*sz), (sx*sz - cx*sy*cz), 0
    // CUDA:     (-sr*cy - cr*sp*sy), (sr*sy - cr*sp*cy), 0
    pH[4] = (-sr * cy - cr * sp * sy) * x + (sr * sy - cr * sp * cy) * y;

    // c3 = h_ang_.row(5) · [x,y,z]
    // Autoware: (cx*cz - sx*sy*sz), (-sx*sy*cz - cx*sz), 0
    // CUDA:     (cr*cy - sr*sp*sy), (-sr*sp*cy - cr*sy), 0
    pH[5] = (cr * cy - sr * sp * sy) * x + (-sr * sp * cy - cr * sy) * y;

    // d1 = h_ang_.row(6) · [x,y,z]
    // Autoware: (-cy*cz), (cy*sz), (-sy)  [Note: d²R[0]/dp² third element is -sp, not +sp]
    // CUDA:     (-cp*cy), (cp*sy), (-sp)
    pH[6] = (-cp * cy) * x + (cp * sy) * y + (-sp) * z;

    // d2 = h_ang_.row(7) · [x,y,z]
    // Autoware: (-sx*sy*cz), (sx*sy*sz), (sx*cy)
    // CUDA:     (-sr*sp*cy), (sr*sp*sy), (sr*cp)
    pH[7] = (-sr * sp * cy) * x + (sr * sp * sy) * y + (sr * cp) * z;

    // d3 = h_ang_.row(8) · [x,y,z]
    // Autoware: (cx*sy*cz), (-cx*sy*sz), (-cx*cy)
    // CUDA:     (cr*sp*cy), (-cr*sp*sy), (-cr*cp)
    pH[8] = (cr * sp * cy) * x + (-cr * sp * sy) * y + (-cr * cp) * z;

    // e1 = h_ang_.row(9) · [x,y,z]
    // Autoware: (sy*sz), (sy*cz), 0
    // CUDA:     (sp*sy), (sp*cy), 0
    pH[9] = (sp * sy) * x + (sp * cy) * y;

    // e2 = h_ang_.row(10) · [x,y,z]
    // Autoware: (-sx*cy*sz), (-sx*cy*cz), 0
    // CUDA:     (-sr*cp*sy), (-sr*cp*cy), 0
    pH[10] = (-sr * cp * sy) * x + (-sr * cp * cy) * y;

    // e3 = h_ang_.row(11) · [x,y,z]
    // Autoware: (cx*cy*sz), (cx*cy*cz), 0
    // CUDA:     (cr*cp*sy), (cr*cp*cy), 0
    pH[11] = (cr * cp * sy) * x + (cr * cp * cy) * y;

    // f1 = h_ang_.row(12) · [x,y,z]
    // Autoware: (-cy*cz), (cy*sz), 0
    // CUDA:     (-cp*cy), (cp*sy), 0
    pH[12] = (-cp * cy) * x + (cp * sy) * y;

    // f2 = h_ang_.row(13) · [x,y,z]
    // Autoware: (-cx*sz - sx*sy*cz), (-cx*cz + sx*sy*sz), 0
    // CUDA:     (-cr*sy - sr*sp*cy), (-cr*cy + sr*sp*sy), 0
    pH[13] = (-cr * sy - sr * sp * cy) * x + (-cr * cy + sr * sp * sy) * y;

    // f3 = h_ang_.row(14) · [x,y,z]
    // Autoware: (-sx*sz + cx*sy*cz), (-cx*sy*sz - sx*cz), 0
    // CUDA:     (-sr*sy + cr*sp*cy), (-cr*sp*sy - sr*cy), 0
    pH[14] = (-sr * sy + cr * sp * cy) * x + (-cr * sp * sy - sr * cy) * y;
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

    // Sparse point Hessian contribution mapping (matching Autoware's h_ang_ structure):
    //
    // The pH values are scalars from h_ang_ · [x,y,z]. They get assembled into
    // 4D vectors representing the contribution to [x, y, z, 0] components.
    //
    // For roll-related terms (a, b, c): vector is [0, v1, v2, 0]
    //   → Contribution = 0*cx + v1*cy + v2*cz = cy*v1 + cz*v2
    //
    // For pitch/yaw-related terms (d, e, f): vector is [v1, v2, v3, 0]
    //   → Contribution = cx*v1 + cy*v2 + cz*v3
    //
    // pH[0,1] = a2, a3 → roll-roll: [0, a2, a3] → cy*pH[0] + cz*pH[1]
    // pH[2,3] = b2, b3 → roll-pitch: [0, b2, b3] → cy*pH[2] + cz*pH[3]
    // pH[4,5] = c2, c3 → roll-yaw: [0, c2, c3] → cy*pH[4] + cz*pH[5]
    // pH[6,7,8] = d1, d2, d3 → pitch-pitch: [d1, d2, d3] → cx*pH[6] + cy*pH[7] + cz*pH[8]
    // pH[9,10,11] = e1, e2, e3 → pitch-yaw: [e1, e2, e3] → cx*pH[9] + cy*pH[10] + cz*pH[11]
    // pH[12,13,14] = f1, f2, f3 → yaw-yaw: [f1, f2, f3] → cx*pH[12] + cy*pH[13] + cz*pH[14]

    int idx = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = i; j < 6; j++) {
            // J_i' * inv_cov * J_j
            float JiCJj = J[i] * Jc[j][0] + J[6 + i] * Jc[j][1] + J[12 + i] * Jc[j][2];

            // Main Hessian term
            float h = factor * (JiCJj - gauss_d2 * Jcx[i] * Jcx[j]);

            // Point Hessian contribution (only for angular terms i,j >= 3)
            if (i >= 3 && j >= 3) {
                int pi = i - 3;  // 0=roll, 1=pitch, 2=yaw
                int pj = j - 3;

                float pH_contrib = 0.0f;
                if (pi == 0 && pj == 0) {
                    // roll-roll: a = [0, a2, a3] → cy*a2 + cz*a3
                    pH_contrib = cy * pH[0] + cz * pH[1];
                } else if ((pi == 0 && pj == 1) || (pi == 1 && pj == 0)) {
                    // roll-pitch: b = [0, b2, b3] → cy*b2 + cz*b3
                    pH_contrib = cy * pH[2] + cz * pH[3];
                } else if ((pi == 0 && pj == 2) || (pi == 2 && pj == 0)) {
                    // roll-yaw: c = [0, c2, c3] → cy*c2 + cz*c3
                    pH_contrib = cy * pH[4] + cz * pH[5];
                } else if (pi == 1 && pj == 1) {
                    // pitch-pitch: d = [d1, d2, d3] → cx*d1 + cy*d2 + cz*d3
                    pH_contrib = cx * pH[6] + cy * pH[7] + cz * pH[8];
                } else if ((pi == 1 && pj == 2) || (pi == 2 && pj == 1)) {
                    // pitch-yaw: e = [e1, e2, e3] → cx*e1 + cy*e2 + cz*e3
                    pH_contrib = cx * pH[9] + cy * pH[10] + cz * pH[11];
                } else if (pi == 2 && pj == 2) {
                    // yaw-yaw: f = [f1, f2, f3] → cx*f1 + cy*f2 + cz*f3
                    pH_contrib = cx * pH[12] + cy * pH[13] + cz * pH[14];
                }

                h += e_x_cov_x * gauss_d2 * pH_contrib;
            }

            hess_out[idx++] = h;
        }
    }
}

#endif // PERSISTENT_NDT_DEVICE_CUH
