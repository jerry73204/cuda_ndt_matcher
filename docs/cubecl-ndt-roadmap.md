# CubeCL NDT Implementation Roadmap

This document outlines the plan to implement custom CUDA kernels for NDT scan matching using [CubeCL](https://github.com/tracel-ai/cubecl), phasing out the fast-gicp dependency.

## Background

### Why Replace fast-gicp?

The fast-gicp NDTCuda implementation has fundamental issues:

| Issue | Impact |
|-------|--------|
| Uses Levenberg-Marquardt optimizer | Never converges properly (hits 30 iterations every time) |
| Different cost function (Mahalanobis) | Different optimization landscape vs pclomp |
| SO(3) exponential map parameterization | Different Jacobian structure than Euler angles |
| No exposed iteration count | Cannot diagnose convergence |

### Why CubeCL?

- **Pure Rust**: No C++/CUDA FFI complexity
- **Multi-platform**: CUDA, ROCm, WebGPU from same codebase
- **Type-safe**: Rust's type system for GPU code
- **Automatic vectorization**: `Line<T>` handles SIMD
- **Autotuning**: Runtime optimization of kernel parameters

## Architecture Overview

```
cuda_ndt_matcher/
├── src/
│   ├── cuda_ndt_matcher/           # ROS node (existing)
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── ndt_manager.rs      # Will use new NDT engine
│   │   │   └── ...
│   │   └── Cargo.toml
│   │
│   └── ndt_cuda/                   # NEW: CubeCL NDT library
│       ├── src/
│       │   ├── lib.rs
│       │   ├── voxel_grid/         # Phase 1: Voxelization
│       │   │   ├── mod.rs
│       │   │   ├── kernels.rs      # CubeCL kernels
│       │   │   └── types.rs
│       │   ├── derivatives/        # Phase 2: Derivative computation
│       │   │   ├── mod.rs
│       │   │   ├── jacobian.rs
│       │   │   ├── hessian.rs
│       │   │   └── kernels.rs
│       │   ├── optimization/       # Phase 3: Newton solver
│       │   │   ├── mod.rs
│       │   │   ├── newton.rs
│       │   │   └── line_search.rs
│       │   ├── scoring/            # Phase 4: Probability scoring
│       │   │   ├── mod.rs
│       │   │   └── kernels.rs
│       │   └── ndt.rs              # High-level API
│       ├── benches/
│       └── Cargo.toml
```

## Phase 1: Voxel Grid Construction (2-3 weeks)

### Goal
Build GPU-accelerated voxel grid from point cloud map with covariance computation.

### Components

#### 1.1 Voxel ID Computation
```rust
#[cube(launch_unchecked)]
fn compute_voxel_ids<F: Float>(
    points: &Array<Line<F>>,      // [N, 3] point cloud
    voxel_ids: &mut Array<u32>,   // [N] output voxel IDs
    min_bound: &Array<F>,         // [3] bounding box min
    resolution: F,
    grid_dims: &Array<u32>,       // [3] grid dimensions
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = (points[idx * 3 + 0] - min_bound[0]) / resolution;
    let y = (points[idx * 3 + 1] - min_bound[1]) / resolution;
    let z = (points[idx * 3 + 2] - min_bound[2]) / resolution;

    let ix = Line::cast_from(x);
    let iy = Line::cast_from(y);
    let iz = Line::cast_from(z);

    voxel_ids[idx] = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
}
```

#### 1.2 Point Accumulation (Parallel Histogram)
- Sort points by voxel ID (use `cub::DeviceRadixSort` via cuBLAS or implement in CubeCL)
- Segment reduce to compute per-voxel statistics

#### 1.3 Covariance Computation
```rust
#[cube(launch_unchecked)]
fn compute_covariance<F: Float>(
    // Per-voxel accumulated statistics
    point_sums: &Array<F>,        // [V, 3] sum of points
    point_sq_sums: &Array<F>,     // [V, 6] sum of x*x^T (upper triangle)
    point_counts: &Array<u32>,    // [V] count per voxel

    // Output
    means: &mut Array<F>,         // [V, 3] voxel centroids
    covariances: &mut Array<F>,   // [V, 9] 3x3 covariance matrices
    inv_covariances: &mut Array<F>, // [V, 9] inverse covariances
) {
    let voxel_idx = ABSOLUTE_POS;
    let count = point_counts[voxel_idx];

    if count < 6 {  // Minimum points for valid covariance
        return;
    }

    // Compute mean
    let mean_x = point_sums[voxel_idx * 3 + 0] / F::cast_from(count);
    // ... mean_y, mean_z

    // Compute covariance using single-pass formula:
    // cov = (sum_sq - sum * mean^T) / (n - 1)
    // ...

    // Regularize via eigendecomposition (or simplified approach)
    // ...

    // Compute inverse
    // ...
}
```

#### 1.4 Eigenvalue Regularization
For singular covariances, inflate small eigenvalues:
- Option A: Use cuSOLVER for batched eigendecomposition
- Option B: Implement power iteration in CubeCL (simpler, may be sufficient)
- Option C: Use Jacobi eigenvalue algorithm (pure CubeCL)

### Data Structures
```rust
/// GPU voxel grid
pub struct GpuVoxelGrid {
    /// Voxel centroids [V, 3]
    pub means: CubeBuffer<f32>,
    /// Inverse covariances [V, 9] (row-major 3x3)
    pub inv_covariances: CubeBuffer<f32>,
    /// Point counts per voxel
    pub counts: CubeBuffer<u32>,
    /// Voxel coordinates for spatial lookup
    pub coords: CubeBuffer<i32>,  // [V, 3]
    /// Hash table for O(1) voxel lookup
    pub hash_table: GpuHashTable,
    /// Grid parameters
    pub resolution: f32,
    pub min_bound: [f32; 3],
}
```

### Tests
- [x] Voxel ID matches CPU implementation
- [x] Mean/covariance matches within floating-point tolerance
- [x] Inverse covariance is valid (no NaN/Inf)
- [ ] Performance: < 10ms for 100K point cloud (GPU kernel pending)

---

## Phase 2: Derivative Computation (3-4 weeks)

### Goal
Compute gradient (6x1) and Hessian (6x6) for Newton optimization.

This is the **most compute-intensive** part, parallelized over input points.

### Components

#### 2.1 Angular Derivatives (Precompute)
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

#### 2.2 Point Jacobian Kernel
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

#### 2.3 Voxel Correspondence & Score Accumulation
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

#### 2.4 Block-Level Reduction
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

### Tests
- [x] Gradient matches CPU within 1e-5 (verified with finite difference test)
- [x] Hessian matches CPU within 1e-4 (verified symmetry test)
- [x] Score matches CPU within 1e-6
- [ ] Performance: < 5ms for 50K source points (GPU kernel pending)

---

## Phase 3: Newton Optimization (2 weeks)

### Goal
Implement Newton's method with optional More-Thuente line search.

### Components

#### 3.1 Newton Step (CPU)
The 6x6 linear solve is too small for GPU benefit:
```rust
pub fn newton_step(
    gradient: &Vector6<f64>,
    hessian: &Matrix6<f64>,
) -> Vector6<f64> {
    // SVD solve: delta = -H^{-1} * g
    let svd = hessian.svd(true, true);
    svd.solve(gradient, 1e-10).unwrap() * -1.0
}
```

#### 3.2 Transformation Update (GPU)
Apply delta to transformation and transform all points:
```rust
#[cube(launch_unchecked)]
fn transform_points<F: Float>(
    points: &Array<F>,           // [N, 3] source points
    transform: &Array<F>,        // [16] 4x4 transformation matrix
    output: &mut Array<F>,       // [N, 3] transformed points
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = points[idx * 3 + 0];
    let y = points[idx * 3 + 1];
    let z = points[idx * 3 + 2];

    // T * [x, y, z, 1]^T
    output[idx * 3 + 0] = transform[0]*x + transform[1]*y + transform[2]*z + transform[3];
    output[idx * 3 + 1] = transform[4]*x + transform[5]*y + transform[6]*z + transform[7];
    output[idx * 3 + 2] = transform[8]*x + transform[9]*y + transform[10]*z + transform[11];
}
```

#### 3.3 Convergence Check
```rust
pub fn check_convergence(
    delta: &Vector6<f64>,
    trans_epsilon: f64,
    iteration: usize,
    max_iterations: usize,
) -> bool {
    iteration >= max_iterations || delta.norm() < trans_epsilon
}
```

#### 3.4 Line Search (Optional)
More-Thuente line search - currently disabled in Autoware due to local minima issues.
Implement as optional feature for experimentation.

### Main Loop
```rust
pub fn align(
    &mut self,
    source: &GpuPointCloud,
    initial_guess: Isometry3<f64>,
) -> NdtResult {
    let mut transform = initial_guess;

    for iteration in 0..self.max_iterations {
        // 1. Transform source points (GPU)
        self.transform_points(source, &transform);

        // 2. Compute angular derivatives (CPU, tiny)
        let ang_deriv = compute_angular_derivatives(&transform);

        // 3. Compute point Jacobians (GPU)
        self.compute_point_jacobians(source, &ang_deriv);

        // 4. Compute gradient & Hessian (GPU)
        let (score, gradient, hessian) = self.compute_derivatives();

        // 5. Newton step (CPU, 6x6 solve)
        let delta = newton_step(&gradient, &hessian);

        // 6. Update transform
        transform = apply_delta(transform, delta);

        // 7. Check convergence
        if check_convergence(&delta, self.trans_epsilon, iteration, self.max_iterations) {
            break;
        }
    }

    NdtResult { transform, score, iterations, hessian }
}
```

### Tests
- [x] Convergence within 10 iterations for good initial guess
- [ ] Final pose matches pclomp within 1cm / 0.1 degree (validation pending)
- [x] Handles edge cases (no correspondences, singular Hessian)

---

## Phase 4: Scoring & NVTL (1 week)

### Goal
Compute transform probability and NVTL for quality assessment.

### Components

#### 4.1 Transform Probability
```rust
#[cube(launch_unchecked)]
fn compute_transform_probability<F: Float>(
    transformed_points: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    voxel_hash: &Array<u32>,
    gauss_d1: F,
    gauss_d2: F,
    scores: &mut Array<F>,  // Per-point scores
) {
    // Similar to derivative computation but only score
    // ...
}
```

#### 4.2 NVTL (Nearest Voxel Transformation Likelihood)
```rust
#[cube(launch_unchecked)]
fn compute_nvtl<F: Float>(
    transformed_points: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    voxel_hash: &Array<u32>,
    gauss_d1: F,
    gauss_d2: F,
    nearest_scores: &mut Array<F>,  // Max score per point
) {
    let idx = ABSOLUTE_POS;

    let x_trans = load_vec3(transformed_points, idx);
    let mut max_score = F::new(0.0);

    for offset in NEIGHBOR_OFFSETS {
        let voxel_idx = find_voxel(x_trans, offset, voxel_hash);
        if voxel_idx >= 0 {
            let score = compute_point_score(x_trans, voxel_idx, ...);
            if score > max_score {
                max_score = score;
            }
        }
    }

    nearest_scores[idx] = max_score;
}
```

### Tests
- [x] Transform probability matches CPU implementation
- [x] Per-point scores computed correctly
- [x] NVTL neighbor search finds all relevant voxels
- [x] NVTL vs transform probability comparison
- [ ] Performance: < 2ms for scoring (GPU kernel pending)

---

## Phase 5: Integration & Optimization (2 weeks)

### Goal
Integrate with cuda_ndt_matcher and optimize performance.

### Components

#### 5.1 API Design
```rust
// src/ndt_cuda/src/lib.rs

pub struct NdtCuda {
    client: CubeClient,
    config: NdtConfig,
    voxel_grid: Option<GpuVoxelGrid>,
}

impl NdtCuda {
    pub fn new(config: NdtConfig) -> Result<Self>;

    /// Set target (map) point cloud - builds voxel grid
    pub fn set_target(&mut self, points: &[[f32; 3]]) -> Result<()>;

    /// Align source to target with initial guess
    pub fn align(
        &self,
        source: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
    ) -> Result<NdtResult>;

    /// Compute scores without optimization
    pub fn compute_score(
        &self,
        source: &[[f32; 3]],
        pose: Isometry3<f64>,
    ) -> Result<ScoreResult>;
}

pub struct NdtConfig {
    pub resolution: f32,
    pub max_iterations: u32,
    pub trans_epsilon: f64,
    pub step_size: f64,
    pub outlier_ratio: f64,
}

pub struct NdtResult {
    pub pose: Isometry3<f64>,
    pub converged: bool,
    pub score: f64,
    pub nvtl: f64,
    pub iterations: u32,
    pub hessian: Matrix6<f64>,
}
```

#### 5.2 Memory Management
- Reuse GPU buffers across calls
- Lazy voxel grid rebuild only when target changes
- Stream-ordered memory allocation

#### 5.3 Performance Optimization
- Tune CubeCL launch parameters
- Profile with Nsight Systems
- Optimize memory access patterns (coalescing)
- Consider persistent kernel approach for small workloads

### Tests
- [x] High-level NdtScanMatcher API with builder pattern
- [x] Feature flags for ndt_cuda vs fast-gicp backends
- [x] Unit tests for API (9 new tests)
- [x] Covariance estimation with Laplace approximation
- [ ] Integration test with sample rosbag (Phase 6)
- [ ] A/B comparison with pclomp (Phase 6)
- [ ] Latency < 20ms for typical workload (GPU kernels pending)
- [ ] Memory usage < 500MB

---

## Phase 6: Validation & Production (2 weeks)

### Goal
Validate against pclomp and prepare for production.

### Components

#### 6.1 Numerical Validation
- Compare every intermediate value with pclomp
- Log divergence points
- Create regression test suite

#### 6.2 Edge Cases
- Empty point clouds
- Single-voxel scenes
- Degenerate covariances
- Large initial pose errors

#### 6.3 Documentation
- API documentation
- Performance tuning guide
- Troubleshooting guide

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Voxel Grid | 2-3 weeks | CubeCL setup |
| Phase 2: Derivatives | 3-4 weeks | Phase 1 |
| Phase 3: Newton Solver | 2 weeks | Phase 2 |
| Phase 4: Scoring | 1 week | Phase 2 |
| Phase 5: Integration | 2 weeks | Phases 3, 4 |
| Phase 6: Validation | 2 weeks | Phase 5 |
| **Total** | **12-14 weeks** | |

---

## Dependencies

### Rust Crates
```toml
[dependencies]
cubecl = { version = "0.4", features = ["cuda"] }
cubecl-cuda = "0.4"
nalgebra = "0.33"
```

### Build Requirements
- CUDA Toolkit 12.x
- Rust nightly (for CubeCL proc macros)
- NVIDIA GPU (compute capability 7.0+)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CubeCL alpha stability | Pin version, contribute fixes upstream |
| Eigendecomposition complexity | Start with simplified regularization |
| Hash table performance | Profile early, consider alternatives |
| Numerical precision | Use f64 for accumulation, f32 for storage |

---

## Success Criteria

1. **Correctness**: Final pose within 1cm / 0.1° of pclomp
2. **Convergence**: < 10 iterations for typical scenarios
3. **Performance**: Total latency < 20ms (vs ~50ms CPU)
4. **Reliability**: No crashes or numerical instabilities

---

## References

1. Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform. PhD Thesis.
2. [CubeCL Documentation](https://github.com/tracel-ai/cubecl)
3. [Autoware NDT Implementation](https://github.com/autowarefoundation/autoware.universe)
4. [More-Thuente Line Search](https://www.ii.uib.no/~lennMDL/talks/MT-paper.pdf)

---

## Sources

- [CubeCL GitHub](https://github.com/tracel-ai/cubecl)
- [Rust-CUDA Project](https://github.com/Rust-GPU/Rust-CUDA)
- [Burn Deep Learning Framework](https://burn.dev/blog/going-big-and-small-for-2025/)
- [CubeCL Architecture Overview](https://gist.github.com/nihalpasham/570d4fe01b403985e1eaf620b6613774)
