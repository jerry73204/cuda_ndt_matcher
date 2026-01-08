# Phase 2: Derivative Computation

**Status**: ✅ Complete

## Goal

Compute gradient (6x1) and Hessian (6x6) for Newton optimization.

This is the **most compute-intensive** part, parallelized over input points.

## Components

### 2.1 Angular Derivatives (Precompute)

Compute j_ang matrices from current rotation estimate (Euler angles):

```rust
/// Precomputed angular derivatives for current pose
pub struct AngularDerivatives {
    /// 8x4 Jacobian components (Eq. 6.19)
    pub j_ang: [[f32; 4]; 8],
    /// 16x4 Hessian components (Eq. 6.21)
    pub h_ang: [[f32; 4]; 16],
}

fn compute_angular_derivatives(roll: f32, pitch: f32, yaw: f32) -> AngularDerivatives {
    let (sx, cx) = (roll.sin(), roll.cos());
    let (sy, cy) = (pitch.sin(), pitch.cos());
    let (sz, cz) = (yaw.sin(), yaw.cos());

    // Magnusson 2009, Eq. 6.19
    let j_ang = [
        [-sx*sz + cx*sy*cz, -sx*cz - cx*sy*sz, -cx*cy, 0.0],
        [cx*sz + sx*sy*cz, cx*cz - sx*sy*sz, -sx*cy, 0.0],
        // ... remaining 6 rows
    ];
    // ...
}
```

### 2.2 Point Jacobian Kernel

For each input point, compute 4x6 Jacobian:

```rust
#[cube(launch_unchecked)]
fn compute_point_jacobians<F: Float>(
    points: &Array<F>,            // [N, 3] source points
    j_ang: &Array<F>,             // [8, 4] precomputed angular derivatives
    jacobians: &mut Array<F>,     // [N, 4, 6] output Jacobians
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = points[idx * 3 + 0];
    let y = points[idx * 3 + 1];
    let z = points[idx * 3 + 2];

    // Eq. 6.18: point_gradient = [I_3x3 | d(Rx)/d(r,p,y)]
    // First 3 columns: identity for translation
    jacobians[idx * 24 + 0] = F::new(1.0);  // dx/dtx
    jacobians[idx * 24 + 7] = F::new(1.0);  // dy/dty
    jacobians[idx * 24 + 14] = F::new(1.0); // dz/dtz

    // Last 3 columns: rotation derivatives
    // j_ang[0] * x + j_ang[1] * y + j_ang[2] * z
    // ...
}
```

### 2.3 Voxel Correspondence & Score Accumulation

The critical kernel: find neighboring voxels and accumulate gradient/Hessian.

```rust
#[cube(launch_unchecked)]
fn compute_derivatives<F: Float>(
    // Input
    transformed_points: &Array<F>,  // [N, 3] T(p) * source_points
    original_points: &Array<F>,     // [N, 3] source points (for Jacobian)
    jacobians: &Array<F>,           // [N, 4, 6] point Jacobians

    // Voxel grid (target)
    voxel_means: &Array<F>,         // [V, 3]
    voxel_inv_covs: &Array<F>,      // [V, 9]
    voxel_hash: &Array<u32>,        // Hash table for lookup

    // Gaussian parameters
    gauss_d1: F,
    gauss_d2: F,
    resolution: F,

    // Output (per-thread accumulation)
    scores: &mut Array<F>,          // [num_blocks]
    gradients: &mut Array<F>,       // [num_blocks, 6]
    hessians: &mut Array<F>,        // [num_blocks, 36]
) {
    let idx = ABSOLUTE_POS;

    // 1. Get transformed point
    let x_trans = [
        transformed_points[idx * 3 + 0],
        transformed_points[idx * 3 + 1],
        transformed_points[idx * 3 + 2],
    ];

    // 2. Find neighboring voxels (DIRECT7: center + 6 neighbors)
    let voxel_coord = compute_voxel_coord(x_trans, resolution);

    for offset in NEIGHBOR_OFFSETS {
        let neighbor = voxel_coord + offset;
        let voxel_idx = hash_lookup(voxel_hash, neighbor);

        if voxel_idx < 0 {
            continue;
        }

        // 3. Compute residual
        let mean = load_vec3(voxel_means, voxel_idx);
        let x_diff = x_trans - mean;

        // 4. Compute Mahalanobis distance
        let inv_cov = load_mat3x3(voxel_inv_covs, voxel_idx);
        let mahal = dot(x_diff, mat_vec_mul(inv_cov, x_diff));

        // 5. Compute score (Eq. 6.9)
        let e_x_cov_x = F::exp(-gauss_d2 * mahal / F::new(2.0));
        let score_inc = -gauss_d1 * e_x_cov_x;

        // 6. Accumulate gradient (Eq. 6.12)
        // gradient += gauss_d1 * gauss_d2 * e_x_cov_x * J^T * inv_cov * x_diff

        // 7. Accumulate Hessian (Eq. 6.13)
        // Complex: involves point_hessian and second derivatives

        // Use atomic add for thread-safe accumulation
        atomic_add(&scores[CUBE_POS], score_inc);
        // ...
    }
}
```

### 2.4 Block-Level Reduction

Reduce per-block accumulators to final gradient/Hessian:

```rust
#[cube(launch_unchecked)]
fn reduce_derivatives<F: Float>(
    block_scores: &Array<F>,      // [num_blocks]
    block_gradients: &Array<F>,   // [num_blocks, 6]
    block_hessians: &Array<F>,    // [num_blocks, 36]

    total_score: &mut Array<F>,   // [1]
    total_gradient: &mut Array<F>, // [6]
    total_hessian: &mut Array<F>,  // [36]
) {
    // Parallel reduction
    // ...
}
```

## Tests

- [x] Gradient matches CPU within 1e-5 (verified with finite difference test)
- [x] Hessian matches CPU within 1e-4 (verified symmetry test)
- [x] Score matches CPU within 1e-6
- [x] Gaussian parameters (d1, d2, d3) match Autoware exactly
- [x] Multi-voxel radius search accumulates contributions correctly
- [ ] GPU kernel performance: < 5ms for 50K source points
