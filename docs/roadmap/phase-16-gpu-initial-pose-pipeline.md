# Phase 16: GPU Initial Pose Pipeline

**Status**: ✅ Complete
**Priority**: Medium
**Estimated Speedup**: 6x for startup phase, 1.4-4.6x overall

## Implementation Progress

| Component | Status | Notes |
|-----------|--------|-------|
| Batched transform kernel | ✅ Complete | `voxel_grid/kernels.rs` |
| Batched sin/cos kernel | ✅ Complete | `voxel_grid/kernels.rs` |
| Batched radius search kernel | ✅ Complete | `derivatives/gpu_batch.rs` |
| Batched score kernel | ✅ Complete | `derivatives/gpu_batch.rs` |
| Batched gradient kernel | ✅ Complete | `derivatives/gpu_batch.rs` |
| Batched Hessian kernel | ✅ Complete | `derivatives/gpu_batch.rs` |
| Batched convergence check | ✅ Complete | `derivatives/gpu_batch.rs` |
| Batched pose update | ✅ Complete | `derivatives/gpu_batch.rs` |
| Batched Jacobian kernel | ✅ Complete | `derivatives/gpu_batch.rs` |
| cuSOLVER batched Cholesky FFI | ✅ Complete | `cuda_ffi/csrc/batched_solve.cu` |
| GpuInitialPosePipeline struct | ✅ Complete | `optimization/gpu_initial_pose.rs` |
| Pipeline implementation | ✅ Complete | Full GPU Newton loop with batched kernels |
| Integration with initial_pose.rs | ✅ Complete | Uses `align_batch` for startup phase |

## Integration Notes

The integration with `initial_pose.rs` uses the existing `NdtManager.align_batch()` method for the startup phase. This method leverages `FullGpuPipelineV2` with shared voxel data across alignments. The `GpuInitialPosePipeline` provides an additional option for true K-way parallel batch processing if needed in the future.

## Overview

Batch multiple initial pose particles through a single GPU pass during the startup phase of TPE-based initial pose estimation. This eliminates the sequential N × 16ms latency for startup particles.

## Background

### Current Algorithm

Initial pose estimation uses Tree-Structured Parzen Estimator (TPE) with two phases:

1. **Startup Phase** (first `n_startup_trials`, typically 10):
   - Generate random poses from Gaussian/Uniform distributions
   - For each pose: run NDT alignment (~13ms) + NVTL evaluation (~3ms)
   - **No dependency between particles** - fully parallelizable

2. **TPE-Guided Phase** (remaining particles):
   - Sort trials by score, split into "good" and "bad" sets
   - Generate 100 random candidates
   - Evaluate each candidate using KDE (log-likelihood ratio)
   - Select best candidate, run NDT alignment + NVTL
   - Add result to trials → **sequential dependency**

### Performance Analysis

| Phase | Particles | Current Time | Notes |
|-------|-----------|--------------|-------|
| Startup | 10 | 10 × 16ms = 160ms | Independent, parallelizable |
| TPE-guided | 20 | 20 × 16ms = 320ms | Sequential dependency |
| **Total** | 30 | **480ms** | |

### Code Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| Main estimation loop | `cuda_ndt_matcher/src/initial_pose.rs` | 116-191 |
| TPE implementation | `cuda_ndt_matcher/src/tpe.rs` | 104-155 |
| Particle struct | `cuda_ndt_matcher/src/particle.rs` | 10-42 |
| NDT alignment | `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` | - |
| NVTL scoring | `ndt_cuda/src/scoring/pipeline.rs` | - |

---

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GpuInitialPosePipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│ SETUP (once per map):                                               │
│   Upload: voxel_means [V×3], voxel_inv_covs [V×9], gauss params     │
│                                                                     │
│ BATCH ALIGNMENT (K particles):                                      │
│   1. Upload: source_points [N×3], K initial_poses [K×6]             │
│                                                                     │
│   2. Per iteration (all K poses in parallel):                       │
│      ┌──────────────────────────────────────────────────────────┐   │
│      │ compute_sin_cos_batch_kernel [K×6]                       │   │
│      │ compute_jacobians_batch_kernel [K×N×18]                  │   │
│      │ compute_point_hessians_batch_kernel [K×N×144]            │   │
│      │ transform_points_batch_kernel [K×N×3]                    │   │
│      │ radius_search_batch_kernel [K×N×MAX_NEIGHBORS]           │   │
│      │ score_gradient_hessian_batch_kernel [K×N]                │   │
│      │ reduce_batch_kernel [K×43]                               │   │
│      │ batched_newton_solve (cuSOLVER) [K×6]                    │   │
│      │ update_poses_batch_kernel [K×6]                          │   │
│      │ check_convergence_batch_kernel [K]                       │   │
│      └──────────────────────────────────────────────────────────┘   │
│      Download: K converged flags [K bytes]                          │
│      Continue until all K converged or max_iterations               │
│                                                                     │
│   3. Download: K final poses [K×6], K scores, K iterations          │
│                                                                     │
│ BATCH NVTL (K poses):                                               │
│   4. Use existing GpuScoringPipeline for K poses                    │
│   5. Download: K NVTL scores                                        │
│                                                                     │
│ Total transfers: 3 uploads + 3 downloads (vs K×6 transfers)         │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Batched Kernels

New kernels with batch dimension for K simultaneous poses:

| Kernel | Input | Output | Notes |
|--------|-------|--------|-------|
| `transform_points_batch_kernel` | K×N points, K poses | K×N×3 transformed | Grid: (K, ceil(N/256)) |
| `radius_search_batch_kernel` | K×N×3 points, V voxels | K×N×MAX_NEIGHBORS | Brute-force O(K×N×V) |
| `score_gradient_hessian_batch_kernel` | K×N points, K neighbors | K×N×43 | Per-point derivatives |
| `reduce_batch_kernel` | K×N×43 | K×43 | Segmented reduce |
| `update_poses_batch_kernel` | K poses, K deltas, K alphas | K poses | Apply Newton step |

#### 2. Batched cuSOLVER Integration

Use `cusolverDnDpotrfBatched` and `cusolverDnDpotrsBatched` for K simultaneous 6×6 Cholesky solves:

```c
// In cuda_ffi/csrc/batched_solve.cu
extern "C" int batched_cholesky_solve_f64(
    cusolverDnHandle_t handle,
    int batch_size,      // K
    double* A_array[],   // K pointers to 6×6 Hessians
    double* B_array[],   // K pointers to 6-element gradients
    int* info            // K status codes
);
```

#### 3. Memory Layout

```
GPU Memory Layout (K=10 particles, N=756 points, V=12000 voxels):

Static (per map):
  voxel_means:     V × 3 × 4 bytes = 144 KB
  voxel_inv_covs:  V × 9 × 4 bytes = 432 KB
  voxel_valid:     V × 4 bytes     = 48 KB

Per-alignment:
  source_points:   N × 3 × 4 bytes = 9 KB
  poses:           K × 6 × 4 bytes = 240 bytes

Per-iteration (working memory):
  transformed:     K × N × 3 × 4 bytes = 90 KB
  neighbors:       K × N × 8 × 4 bytes = 240 KB
  jacobians:       K × N × 18 × 4 bytes = 540 KB
  point_hessians:  K × N × 144 × 4 bytes = 4.3 MB
  scores:          K × N × 4 bytes = 30 KB
  gradients:       K × N × 6 × 4 bytes = 180 KB
  hessians:        K × N × 36 × 4 bytes = 1.1 MB

Total GPU memory: ~7 MB (fits easily)
```

---

## Implementation Plan

### Step 1: Batched Transform Kernel

**File**: `src/ndt_cuda/src/voxel_grid/kernels.rs`

```rust
#[cube(launch_unchecked)]
pub fn transform_points_batch_kernel<F: Float>(
    source_points: &Array<F>,      // [N×3] shared across batches
    transforms: &Array<F>,         // [K×16] K transform matrices
    num_points: u32,
    batch_size: u32,
    output: &mut Array<F>,         // [K×N×3]
) {
    let batch_idx = CUBE_POS_Y;
    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;

    if batch_idx < batch_size && point_idx < num_points {
        // Load transform for this batch
        let t_offset = batch_idx * 16;
        // ... apply transform ...

        // Write to batch-specific output location
        let out_offset = (batch_idx * num_points + point_idx) * 3;
        output[out_offset + 0] = transformed_x;
        output[out_offset + 1] = transformed_y;
        output[out_offset + 2] = transformed_z;
    }
}
```

### Step 2: Batched Radius Search Kernel

**File**: `src/ndt_cuda/src/derivatives/gpu.rs`

```rust
#[cube(launch_unchecked)]
pub fn radius_search_batch_kernel<F: Float>(
    transformed_points: &Array<F>,  // [K×N×3]
    voxel_means: &Array<F>,         // [V×3]
    voxel_valid: &Array<u32>,       // [V]
    radius_sq: F,
    num_points: u32,
    num_voxels: u32,
    batch_size: u32,
    neighbor_indices: &mut Array<i32>,  // [K×N×MAX_NEIGHBORS]
    neighbor_counts: &mut Array<u32>,   // [K×N]
) {
    let batch_idx = CUBE_POS_Y;
    let point_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;

    if batch_idx < batch_size && point_idx < num_points {
        let pt_offset = (batch_idx * num_points + point_idx) * 3;
        let px = transformed_points[pt_offset + 0];
        let py = transformed_points[pt_offset + 1];
        let pz = transformed_points[pt_offset + 2];

        // Search voxels (same as single-pose version)
        // Write to batch-specific output location
        let out_base = (batch_idx * num_points + point_idx) * MAX_NEIGHBORS;
        // ...
    }
}
```

### Step 3: Batched Derivative Kernels

**File**: `src/ndt_cuda/src/derivatives/gpu.rs`

Similar pattern - add `batch_idx` dimension to existing kernels.

### Step 4: Batched cuSOLVER FFI

**File**: `src/cuda_ffi/csrc/batched_solve.cu`

```c
#include <cusolverDn.h>

extern "C" {

// Batched Cholesky factorization + solve for K 6×6 systems
int batched_cholesky_solve_f64(
    cusolverDnHandle_t handle,
    int batch_size,
    double** A_ptrs,      // Array of K pointers to 6×6 matrices (in-place factorization)
    double** B_ptrs,      // Array of K pointers to 6-element RHS (in-place solution)
    int* info_array       // Output: K status codes
) {
    const int n = 6;
    const int nrhs = 1;

    // Factorization: A = L * L^T
    cusolverStatus_t status = cusolverDnDpotrfBatched(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        A_ptrs,
        n,      // lda
        info_array,
        batch_size
    );
    if (status != CUSOLVER_STATUS_SUCCESS) return -1;

    // Solve: L * L^T * x = b
    status = cusolverDnDpotrsBatched(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,
        A_ptrs,
        n,      // lda
        B_ptrs,
        n,      // ldb
        info_array,
        batch_size
    );
    if (status != CUSOLVER_STATUS_SUCCESS) return -2;

    return 0;
}

}
```

**File**: `src/cuda_ffi/src/batched_solve.rs`

```rust
extern "C" {
    pub fn batched_cholesky_solve_f64(
        handle: *mut c_void,
        batch_size: c_int,
        a_ptrs: *mut *mut f64,
        b_ptrs: *mut *mut f64,
        info_array: *mut c_int,
    ) -> c_int;
}

pub struct BatchedCholeskySolver {
    handle: cusolverDnHandle_t,
}

impl BatchedCholeskySolver {
    pub fn solve_batch(
        &self,
        hessians: &mut [[f64; 36]],  // K Hessians (modified in place)
        gradients: &mut [[f64; 6]],  // K gradients (becomes solutions)
    ) -> Result<Vec<i32>> {
        // ... setup pointers and call FFI ...
    }
}
```

### Step 5: GpuInitialPosePipeline

**File**: `src/ndt_cuda/src/initial_pose/pipeline.rs`

```rust
pub struct GpuInitialPosePipeline {
    client: CudaClient,

    // Static data (uploaded once per map)
    voxel_means: Handle,
    voxel_inv_covs: Handle,
    voxel_valid: Handle,
    num_voxels: usize,

    // Per-alignment data
    source_points: Handle,
    num_points: usize,

    // Working buffers (sized for max batch)
    poses: Handle,           // [K×6]
    transformed: Handle,     // [K×N×3]
    neighbors: Handle,       // [K×N×MAX_NEIGHBORS]
    // ... other buffers ...

    // Batched solver
    batched_solver: BatchedCholeskySolver,

    // Configuration
    max_batch_size: usize,
    max_iterations: u32,
}

impl GpuInitialPosePipeline {
    /// Align K poses in a single GPU pass
    pub fn align_batch(
        &mut self,
        initial_poses: &[[f64; 6]],
    ) -> Result<BatchAlignmentResult> {
        let k = initial_poses.len();
        assert!(k <= self.max_batch_size);

        // Upload K poses
        self.upload_poses(initial_poses)?;

        // Track convergence per pose
        let mut converged = vec![false; k];
        let mut iterations = vec![0u32; k];

        for iter in 0..self.max_iterations {
            // Run batched kernels for all K poses
            self.run_batch_iteration(k)?;

            // Download convergence flags
            let flags = self.download_convergence_flags(k)?;

            // Update tracking
            for i in 0..k {
                if flags[i] && !converged[i] {
                    converged[i] = true;
                    iterations[i] = iter + 1;
                }
            }

            // Early exit if all converged
            if converged.iter().all(|&c| c) {
                break;
            }
        }

        // Download final results
        self.download_results(k)
    }
}
```

### Step 6: Integration

**File**: `src/cuda_ndt_matcher/src/initial_pose.rs`

```rust
pub fn estimate_initial_pose_gpu(
    initial_pose_with_cov: &PoseWithCovarianceStamped,
    pipeline: &mut GpuInitialPosePipeline,
    source_points: &[[f32; 3]],
    params: &InitialPoseParams,
) -> Result<InitialPoseResult, String> {
    // Setup TPE
    let mut tpe = TreeStructuredParzenEstimator::new(...);
    let mut particles = Vec::new();

    // ===== PHASE 1: Batched Startup =====
    let n_startup = params.n_startup_trials as usize;

    // Generate N random startup poses on CPU (fast)
    let startup_poses: Vec<[f64; 6]> = (0..n_startup)
        .map(|_| tpe.generate_random_input())
        .collect();

    // Batch NDT alignment on GPU (single pass!)
    let align_results = pipeline.align_batch(&startup_poses)?;

    // Batch NVTL scoring on GPU
    let nvtl_scores = pipeline.score_batch(&align_results.poses)?;

    // Add all results to TPE and particles
    for i in 0..n_startup {
        let transform_prob = (-align_results.scores[i] / 10.0).exp();
        tpe.add_trial(Trial {
            input: align_results.poses[i],
            score: transform_prob,
        });
        particles.push(Particle::new(
            startup_poses[i],
            align_results.poses[i],
            nvtl_scores[i].max(transform_prob * 5.0),
            align_results.iterations[i],
        ));
    }

    // ===== PHASE 2: TPE-Guided (Sequential) =====
    for _ in n_startup..params.particles_num as usize {
        // Get next candidate from TPE
        let candidate = tpe.get_next_input();

        // Single NDT alignment (reuses pipeline)
        let result = pipeline.align_single(&candidate)?;

        // Single NVTL scoring
        let nvtl = pipeline.score_single(&result.pose)?;

        // Update TPE and particles
        let transform_prob = (-result.score / 10.0).exp();
        tpe.add_trial(Trial { input: result.pose, score: transform_prob });
        particles.push(Particle::new(...));
    }

    // Select best particle
    select_best_particle(&particles)
}
```

---

## Testing

### Unit Tests

```rust
#[test]
fn test_batched_transform_kernel() {
    // Verify K transforms produce same results as K individual transforms
}

#[test]
fn test_batched_newton_solve() {
    // Verify batched cuSOLVER matches K individual solves
}

#[test]
fn test_batch_alignment_matches_sequential() {
    // Key test: batch of K alignments should match K sequential alignments
    let sequential_results: Vec<_> = poses.iter()
        .map(|p| pipeline.align_single(p))
        .collect();

    let batch_results = pipeline.align_batch(&poses);

    for (seq, batch) in sequential_results.iter().zip(batch_results.iter()) {
        assert_pose_close(seq.pose, batch.pose, 1e-6);
        assert_relative_eq!(seq.score, batch.score, epsilon = 1e-4);
    }
}

#[test]
fn test_gpu_initial_pose_matches_cpu() {
    // Full integration test: GPU pipeline should find similar best particle as CPU
}
```

### Performance Benchmarks

```rust
#[test]
#[ignore]
fn bench_initial_pose_startup() {
    // Compare: K sequential align_single() vs 1 align_batch(K)
    // Expected: 6x speedup for K=10
}
```

---

## Expected Results

| Configuration | Current | GPU Batch | Speedup |
|---------------|---------|-----------|---------|
| 10 startup + 20 guided | 480ms | 25ms + 320ms = 345ms | 1.4x |
| 20 startup + 10 guided | 480ms | 25ms + 160ms = 185ms | 2.6x |
| 25 startup + 5 guided | 480ms | 25ms + 80ms = 105ms | 4.6x |

**Recommendation**: Consider increasing `n_startup_trials` from 10 to 20-25 to maximize GPU utilization. More random exploration in parallel, less sequential exploitation.

---

## Dependencies

- CubeCL for batched kernels
- cuSOLVER for batched Cholesky solve (`cusolverDnDpotrfBatched`)
- Existing `GpuScoringPipeline` for batch NVTL

## Risks

1. **Memory pressure**: K×N×144 point Hessians may be large for K>20
   - Mitigation: Limit max_batch_size based on available GPU memory

2. **Divergent convergence**: Different poses converge at different iterations
   - Mitigation: Track per-pose convergence, allow early exit per pose

3. **cuSOLVER batched API complexity**: Need pointer arrays for batched calls
   - Mitigation: Careful FFI wrapper with proper memory management

---

## Future Extensions

### Phase 2: GPU KDE Evaluation (Low Priority)

Parallelize the 100-candidate KDE evaluation in TPE-guided phase:

```rust
#[cube(launch_unchecked)]
pub fn compute_kde_scores_kernel<F: Float>(
    candidates: &Array<F>,    // [100×6]
    trials: &Array<F>,        // [M×7] (input + score)
    above_num: u32,
    scores: &mut Array<F>,    // [100]
) {
    let candidate_idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if candidate_idx < 100 {
        // Compute log-likelihood ratio for this candidate
        // Sum over all trials (reduction)
    }
}
```

Estimated impact: Marginal (~0.1ms → ~0.05ms per particle, total ~1ms saved)
