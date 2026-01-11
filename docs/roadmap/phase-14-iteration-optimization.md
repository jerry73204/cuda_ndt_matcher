# Phase 14: Full GPU Newton Iteration

**Status**: ✅ Complete (Core infrastructure)
**Priority**: High
**Target**: Run entire Newton optimization on GPU, only download final pose

## Overview

This phase implements a full GPU Newton iteration loop that eliminates per-iteration CPU↔GPU synchronization. All derivative computation, Jacobian/Hessian construction, matrix solve, and convergence checking run on GPU.

## Implementation Status (2026-01-10)

### Completed Components

| Component | Location | Status |
|-----------|----------|--------|
| cuSOLVER integration | `optimization/gpu_newton.rs` | ✅ Complete |
| GPU Jacobian kernel | `derivatives/gpu_jacobian.rs` | ✅ Complete |
| GPU Point Hessian kernel | `derivatives/gpu_jacobian.rs` | ✅ Complete |
| GPU sin/cos kernel | `derivatives/gpu_jacobian.rs` | ✅ Complete |
| Full GPU pipeline | `optimization/full_gpu_pipeline.rs` | ✅ Complete |

### New Files Created

- `src/ndt_cuda/src/optimization/gpu_newton.rs` - cuSOLVER wrapper for 6×6 Newton solve (Cholesky + LU fallback)
- `src/ndt_cuda/src/derivatives/gpu_jacobian.rs` - GPU kernels for computing Jacobians and Point Hessians
- `src/ndt_cuda/src/optimization/full_gpu_pipeline.rs` - Full GPU Newton iteration pipeline

### Dependencies Added

- `cudarc` with `cusolver` feature for GPU matrix solve

## Current State

### Per-Iteration Transfer Analysis

Current implementation (`derivatives/pipeline.rs:461-599`) per iteration:

| Direction | Data | Size (N=756 points) | Notes |
|-----------|------|---------------------|-------|
| CPU → GPU | Transform matrix | 64 bytes | Required |
| CPU → GPU | Jacobians | N × 18 × 4 = 54 KB | **Move to GPU** |
| CPU → GPU | Point Hessians | N × 144 × 4 = 435 KB | **Move to GPU** |
| GPU → CPU | Reduced results | 43 × 4 = 172 bytes | **Eliminate** |
| CPU | Newton solve (6×6 SVD) | N/A | **Move to GPU** |

**Total transfer per iteration**: ~490 KB
**Target**: 0 bytes per iteration (only download final pose after all iterations)

### Existing Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| GPU transform kernel | `derivatives/gpu.rs` | ✅ Ready |
| GPU radius search | `derivatives/gpu.rs` | ✅ Ready |
| GPU score kernel | `derivatives/gpu.rs` | ✅ Ready |
| GPU gradient kernel | `derivatives/gpu.rs` | ✅ Ready (needs Jacobians) |
| GPU Hessian kernel | `derivatives/gpu.rs` | ✅ Ready (needs Jacobians+PtHess) |
| CUB segmented reduce | `cuda_ffi` | ✅ Ready |
| cuSOLVER bindings | `cudarc/cusolver` | ✅ Available |

### CUDA Library Availability (cudarc)

cudarc provides bindings for cuSOLVER functions in `external/cudarc/src/cusolver/sys/mod.rs`:

**LU Factorization & Solve** (general matrices):
```c
cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, &Lwork)
cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
```

**Cholesky Factorization & Solve** (positive definite matrices - preferred for Hessian):
```c
cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, &Lwork)
cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
```

---

## Implementation Roadmap

### Step 1: Add FFI Bindings to cuda_ffi

**Goal**: Wrap cuSOLVER functions for 6×6 matrix solve.

**Option A: Use cudarc directly** (Recommended)
- cudarc already has complete bindings in `cusolver/sys/mod.rs`
- Need to add `cudarc` as dependency to `ndt_cuda`
- Use `DnHandle` from `cusolver/safe.rs` for handle management

**Option B: Custom CUDA kernel** (Alternative)
- Implement 6×6 Cholesky in CUDA (simple for fixed size)
- Add to `cuda_ffi/csrc/cholesky_6x6.cu`
- Avoid cuSOLVER handle overhead

**Recommendation**: Start with cudarc/cuSOLVER, switch to custom kernel if overhead is significant.

**Files to modify**:
- `src/ndt_cuda/Cargo.toml` - Add cudarc dependency with cusolver feature
- `src/ndt_cuda/src/lib.rs` - Re-export cuSOLVER types if needed

### Step 2: Implement GPU Jacobian/Point Hessian Kernels

**Goal**: Compute Jacobians and Point Hessians directly on GPU.

**Kernel Design**:
```rust
#[cube(launch_unchecked)]
fn compute_jacobians_kernel<F: Float>(
    source_points: &Array<F>,     // [N×3] - already on GPU
    sin_cos: &Array<F>,           // [6]: sin(r), cos(r), sin(p), cos(p), sin(y), cos(y)
    num_points: u32,
    jacobians: &mut Array<F>,     // [N×18] output
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_points { terminate!(); }

    // Load point
    let x = source_points[idx * 3];
    let y = source_points[idx * 3 + 1];
    let z = source_points[idx * 3 + 2];

    // Load sin/cos values
    let sr = sin_cos[0];  // sin(roll)
    let cr = sin_cos[1];  // cos(roll)
    let sp = sin_cos[2];  // sin(pitch)
    let cp = sin_cos[3];  // cos(pitch)
    let sy = sin_cos[4];  // sin(yaw)
    let cy = sin_cos[5];  // cos(yaw)

    // Compute j_ang terms (from angular.rs)
    // Row 0: ∂y'/∂roll
    let j0_0 = -sr*sy + cr*sp*cy;
    let j0_1 = -sr*cy - cr*sp*sy;
    let j0_2 = -cr*cp;
    // ... (8 rows total)

    // Build 3×6 Jacobian matrix (row-major)
    let base = idx * 18;
    // Row 0: [1, 0, 0, 0, ∂x'/∂pitch, ∂x'/∂yaw]
    jacobians[base + 0] = F::new(1.0);
    jacobians[base + 1] = F::new(0.0);
    jacobians[base + 2] = F::new(0.0);
    jacobians[base + 3] = F::new(0.0);
    jacobians[base + 4] = /* ∂x'/∂pitch formula */;
    jacobians[base + 5] = /* ∂x'/∂yaw formula */;
    // ... rows 1-2
}

#[cube(launch_unchecked)]
fn compute_point_hessians_kernel<F: Float>(
    source_points: &Array<F>,     // [N×3]
    sin_cos: &Array<F>,           // [6]
    num_points: u32,
    point_hessians: &mut Array<F>,// [N×144] output (24×6 per point)
) {
    // Similar structure, compute h_ang terms
    // Output 24×6 matrix per point (only rows 13,14,17,18,21,22 have non-zero values)
}
```

**Files to create/modify**:
- `src/ndt_cuda/src/derivatives/gpu_jacobian.rs` - New kernels
- `src/ndt_cuda/src/derivatives/mod.rs` - Export new module

### Step 3: Implement Full GPU Newton Iteration Loop

**Goal**: Run all 30 iterations on GPU without CPU sync.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU Memory (Persistent)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Source Points [N×3]     Voxel Data [V×12]     ← Upload ONCE       │
│                                                                     │
│  Jacobians [N×18]        Point Hessians [N×144] ← Computed on GPU  │
│                                                                     │
│  Per-point outputs [N×43]: scores, gradients, Hessians             │
│                                                                     │
│  Reduced results [43]: total score, gradient[6], Hessian[36]       │
│                                                                     │
│  Pose [6]               Convergence flags      ← Updated on GPU    │
│                                                                     │
│  Newton workspace: Hessian copy [36], pivot [6], RHS [6]           │
└─────────────────────────────────────────────────────────────────────┘

GPU Iteration Loop (single kernel or fused launch):
  for iter in 0..max_iterations:
    1. Extract sin/cos from pose                    [tiny kernel]
    2. Compute Jacobians from sin/cos + points      [N threads]
    3. Compute Point Hessians from sin/cos + points [N threads]
    4. Transform points by current pose             [N threads]
    5. Radius search for neighbors                  [N threads]
    6. Compute per-point score/gradient/Hessian     [N threads]
    7. CUB segmented reduce → 43 totals             [CUB]
    8. Build 6×6 Hessian matrix from upper triangle [1 thread]
    9. Solve H·δ = -g using Cholesky/LU            [cuSOLVER or custom]
    10. Update pose: pose += step_size * δ          [1 thread]
    11. Check convergence: |δ| < ε                  [1 thread]

  Download final pose [6 floats = 24 bytes]
```

**Implementation Options**:

**Option A: Separate Kernels with GPU State**
- Keep current kernel structure
- Add GPU buffers for Jacobians, Hessians, pose, convergence
- Loop on CPU but only sync for convergence check every N iterations

**Option B: Single Fused Iteration Kernel** (More complex)
- One kernel does steps 1-11
- Uses cooperative groups for synchronization
- Most efficient but harder to debug

**Option C: Hybrid** (Recommended)
- Steps 1-7: Existing kernels (add Jacobian kernels)
- Steps 8-11: New small kernels for Newton solve + update
- CPU loop checks convergence every 5 iterations (not every iteration)

**Files to create/modify**:
- `src/ndt_cuda/src/optimization/gpu_newton.rs` - GPU Newton solver
- `src/ndt_cuda/src/derivatives/pipeline.rs` - Integrate full GPU loop
- `src/ndt_cuda/src/optimization/mod.rs` - Export new module

---

## Detailed Implementation Plan

### Phase 14.1: cuSOLVER Integration

```rust
// In ndt_cuda/src/solver/cusolver.rs

use cudarc::cusolver::{DnHandle, sys};
use cudarc::driver::CudaStream;

pub struct GpuNewtonSolver {
    handle: DnHandle,
    // Pre-allocated workspace
    d_hessian: DeviceBuffer<f64>,    // [36] - 6×6 matrix
    d_gradient: DeviceBuffer<f64>,   // [6]
    d_workspace: DeviceBuffer<f64>,  // cuSOLVER workspace
    d_ipiv: DeviceBuffer<i32>,       // [6] pivot indices
    d_info: DeviceBuffer<i32>,       // [1] status
    workspace_size: usize,
}

impl GpuNewtonSolver {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, Error> {
        let handle = DnHandle::new(stream.clone())?;
        // Query workspace size for 6×6 Cholesky
        let workspace_size = unsafe {
            let mut lwork = 0i32;
            sys::cusolverDnDpotrf_bufferSize(
                handle.cu(),
                sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                6, ptr::null_mut(), 6, &mut lwork
            );
            lwork as usize
        };
        // Allocate buffers...
    }

    /// Solve H·δ = -g on GPU
    /// Inputs: d_hessian [36], d_gradient [6] (GPU pointers)
    /// Output: d_gradient overwritten with solution δ
    pub fn solve(&self, d_hessian: u64, d_gradient: u64) -> Result<(), Error> {
        unsafe {
            // Copy Hessian to workspace (potrf modifies in place)
            // cusolverDnDpotrf - Cholesky factorization
            // cusolverDnDpotrs - Solve using Cholesky factors
        }
    }
}
```

### Phase 14.2: GPU Jacobian Kernels

```rust
// In ndt_cuda/src/derivatives/gpu_jacobian.rs

/// Compute sin/cos values from pose angles
#[cube(launch_unchecked)]
pub fn compute_sin_cos_kernel<F: Float>(
    pose: &Array<F>,       // [6]: tx, ty, tz, roll, pitch, yaw
    sin_cos: &mut Array<F>,// [6]: sr, cr, sp, cp, sy, cy
) {
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

/// Compute per-point Jacobians from sin/cos and point coordinates
#[cube(launch_unchecked)]
pub fn compute_jacobians_kernel<F: Float>(
    source_points: &Array<F>,  // [N×3]
    sin_cos: &Array<F>,        // [6]
    num_points: u32,
    jacobians: &mut Array<F>,  // [N×18]
) {
    // ... implementation from angular.rs formulas
}

/// Compute per-point Hessians from sin/cos and point coordinates
#[cube(launch_unchecked)]
pub fn compute_point_hessians_kernel<F: Float>(
    source_points: &Array<F>,    // [N×3]
    sin_cos: &Array<F>,          // [6]
    num_points: u32,
    point_hessians: &mut Array<F>,// [N×144]
) {
    // ... implementation from angular.rs h_ang formulas
}
```

### Phase 14.3: Full GPU Pipeline

```rust
// In ndt_cuda/src/derivatives/gpu_pipeline.rs

pub struct FullGpuPipeline {
    // Existing buffers
    source_points: Handle,
    voxel_data: Handle,

    // New: Pre-allocated iteration buffers (never reallocated)
    jacobians: Handle,           // [max_points × 18]
    point_hessians: Handle,      // [max_points × 144]
    sin_cos: Handle,             // [6]
    pose: Handle,                // [6]

    // Intermediate results
    per_point_output: Handle,    // [max_points × 43]
    reduced_output: Handle,      // [43]

    // Newton solver
    newton_solver: GpuNewtonSolver,
}

impl FullGpuPipeline {
    /// Run full Newton optimization on GPU
    /// Returns: (final_pose, converged, iterations, final_score)
    pub fn optimize(
        &mut self,
        initial_pose: &[f64; 6],
        max_iterations: u32,
        convergence_threshold: f64,
    ) -> Result<OptimizationResult, Error> {
        // Upload initial pose
        self.pose.copy_from(initial_pose);

        for iter in 0..max_iterations {
            // Step 1: Compute sin/cos on GPU
            launch!(compute_sin_cos_kernel, 1, self.pose, self.sin_cos);

            // Step 2-3: Compute Jacobians and Point Hessians
            launch!(compute_jacobians_kernel, self.num_points,
                    self.source_points, self.sin_cos, self.jacobians);
            launch!(compute_point_hessians_kernel, self.num_points,
                    self.source_points, self.sin_cos, self.point_hessians);

            // Step 4-6: Existing derivative pipeline
            launch!(transform_points_kernel, ...);
            launch!(radius_search_kernel, ...);
            launch!(compute_score_gradient_hessian_kernels, ...);

            // Step 7: CUB reduce
            cub_segmented_reduce(...);

            // Step 8-10: Newton solve and pose update
            self.newton_solver.solve(self.reduced_output, self.pose);

            // Step 11: Check convergence (every 5 iterations to reduce sync)
            if iter % 5 == 4 {
                let delta_norm = self.download_delta_norm();
                if delta_norm < convergence_threshold {
                    return Ok(/* converged */);
                }
            }
        }

        // Download final pose
        let final_pose = self.download_pose();
        Ok(OptimizationResult { pose: final_pose, ... })
    }
}
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Per-iteration upload | ~490 KB | 0 bytes |
| Per-iteration download | ~172 bytes | 0 bytes |
| CPU↔GPU syncs per alignment | 30 | 1-6 (every 5 iters + final) |
| Mean alignment time | 13.07 ms | <3 ms |
| GPU memory allocation per iteration | 2-3 | 0 |

## Files Summary

| File | Changes |
|------|---------|
| `ndt_cuda/Cargo.toml` | Add cudarc/cusolver dependency |
| `derivatives/gpu_jacobian.rs` | **New**: GPU Jacobian/Hessian kernels |
| `optimization/gpu_newton.rs` | **New**: GPU Newton solver (cuSOLVER wrapper) |
| `derivatives/pipeline.rs` | Integrate full GPU loop |
| `derivatives/mod.rs` | Export new modules |
| `optimization/mod.rs` | Export gpu_newton |

## Testing

1. **Unit tests**: Compare GPU vs CPU Jacobian/Hessian computation
2. **Integration test**: Verify `optimize()` produces same results as current CPU loop
3. **Performance test**: Measure per-alignment timing
4. **Stress test**: Run on large point clouds, verify numerical stability

## References

- `derivatives/angular.rs` - j_ang and h_ang formulas
- `derivatives/gpu.rs` - Existing GPU kernels
- `optimization/newton.rs` - CPU Newton solver (reference)
- `external/cudarc/src/cusolver/` - cuSOLVER bindings
- [cuSOLVER docs](https://docs.nvidia.com/cuda/cusolver/)
