# Phase 12: GPU Zero-Copy Derivative Pipeline

**Status**: Planned
**Priority**: High
**Estimated Speedup**: 2-3x per alignment

## Overview

Integrate existing GPU derivative kernels into the optimization loop with a zero-copy pipeline that minimizes CPU-GPU transfers.

## Current State

### What Exists

1. **GPU Kernels** (all working in `derivatives/gpu.rs`):
   - `radius_search_kernel` (line 61) - Brute-force O(N×V) with bounded loop
   - `compute_ndt_score_kernel` (line 145) - Per-point score accumulation
   - `compute_ndt_gradient_kernel` (line 473) - 6-element gradient per point
   - `compute_ndt_hessian_kernel` (line 796) - 36-element Hessian per point

2. **GPU Runtime** (`runtime.rs:345`):
   - `NdtCudaRuntime::compute_derivatives()` chains all kernels
   - Works but has excessive transfers (uploads/downloads per call)

3. **CPU Path** (`solver.rs:156`):
   - `NdtOptimizer` calls `compute_derivatives_cpu_with_metric()`
   - Used because GPU path has too many transfers per iteration

### Transfer Analysis

**Current GPU runtime (if used)**:
```
Per iteration:
  Upload: source_points, transform, jacobians, voxel_data (4 transfers)
  Download: scores, correspondences, gradients, hessians (4 transfers)

30 iterations × 8 transfers = 240 transfers per alignment
```

**Target zero-copy pipeline**:
```
Once per alignment:
  Upload: source_points, voxel_data, jacobians_template (3 transfers)

Per iteration:
  Upload: transform [16 floats] (1 small transfer)
  Download: score + gradient + hessian [1 + 6 + 36 = 43 floats] (1 small transfer)

Total: 3 + (30 × 2) = 63 transfers (74% reduction)
Data volume: ~99% reduction (43 floats vs N×43 floats per iteration)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU Memory (Persistent)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                        │
│  │ Source Points   │    │   Voxel Data    │  ← Upload ONCE         │
│  │ [N × 3]         │    │ means [V × 3]   │    at align() start    │
│  └────────┬────────┘    │ inv_cov [V × 9] │                        │
│           │             │ valid [V]       │                        │
│           │             └────────┬────────┘                        │
│           │                      │                                  │
│           ▼                      ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              PER-ITERATION GPU PIPELINE                     │    │
│  │                                                             │    │
│  │  [transform: 16 floats] ← Upload per iteration              │    │
│  │           │                                                 │    │
│  │           ▼                                                 │    │
│  │  ┌──────────────┐                                          │    │
│  │  │  Transform   │  Point transformation                     │    │
│  │  │   Points     │  (existing kernel)                        │    │
│  │  └──────┬───────┘                                          │    │
│  │         │                                                   │    │
│  │         ▼                                                   │    │
│  │  ┌──────────────┐                                          │    │
│  │  │   Radius     │  Find neighboring voxels                  │    │
│  │  │   Search     │  (existing kernel)                        │    │
│  │  └──────┬───────┘                                          │    │
│  │         │                                                   │    │
│  │         ▼                                                   │    │
│  │  ┌──────────────┐                                          │    │
│  │  │    Score     │  Per-point scores                         │    │
│  │  │   Kernel     │  (existing kernel)                        │    │
│  │  └──────┬───────┘                                          │    │
│  │         │                                                   │    │
│  │         ▼                                                   │    │
│  │  ┌──────────────┐                                          │    │
│  │  │  Gradient    │  Per-point gradients [N × 6]              │    │
│  │  │   Kernel     │  (existing kernel)                        │    │
│  │  └──────┬───────┘                                          │    │
│  │         │                                                   │    │
│  │         ▼                                                   │    │
│  │  ┌──────────────┐                                          │    │
│  │  │   Hessian    │  Per-point Hessians [N × 36]              │    │
│  │  │   Kernel     │  (existing kernel)                        │    │
│  │  └──────┬───────┘                                          │    │
│  │         │                                                   │    │
│  │         ▼                                                   │    │
│  │  ┌──────────────┐                                          │    │
│  │  │ GPU Reduce   │  Sum to single result  ← NEW KERNEL       │    │
│  │  │  (atomic)    │  [1 + 6 + 36 floats]                      │    │
│  │  └──────┬───────┘                                          │    │
│  │         │                                                   │    │
│  └─────────┼───────────────────────────────────────────────────┘    │
│            │                                                        │
└────────────┼────────────────────────────────────────────────────────┘
             │
             ▼ Download: 43 floats
    ┌─────────────────┐
    │  CPU Newton     │  6×6 solve (too small for GPU)
    │     Solve       │
    └─────────────────┘
```

## Implementation Plan

### Phase 12.1: GPU Reduction Kernel

**Goal**: Sum per-point results on GPU instead of downloading N×43 floats.

**File**: `src/ndt_cuda/src/derivatives/gpu.rs`

```rust
/// Reduce per-point derivatives to totals using atomic operations.
#[cube(launch_unchecked)]
pub fn reduce_derivatives_kernel<F: Float>(
    scores: &Array<F>,           // [N]
    correspondences: &Array<u32>, // [N]
    gradients: &Array<F>,        // [N × 6]
    hessians: &Array<F>,         // [N × 36]
    num_points: u32,
    // Output: single aggregated result
    total_score: &mut Array<F>,        // [1]
    total_correspondences: &mut Array<u32>, // [1]
    total_gradient: &mut Array<F>,     // [6]
    total_hessian: &mut Array<F>,      // [36]
);
```

**Approach**: Two-phase reduction
1. Block-level reduction using shared memory
2. Final reduction using atomics

**Tests**:
- `test_reduce_small` - 100 points
- `test_reduce_large` - 10,000 points
- `test_reduce_matches_cpu` - Compare with CPU sum

### Phase 12.2: Derivative Pipeline Buffers

**Goal**: Pre-allocate persistent GPU buffers for the optimization loop.

**File**: `src/ndt_cuda/src/derivatives/pipeline.rs` (new)

```rust
/// Pre-allocated GPU buffers for derivative computation pipeline.
pub struct GpuDerivativePipeline {
    client: CudaClient,

    // Capacity
    max_points: usize,
    max_voxels: usize,

    // Persistent data (uploaded once per alignment)
    source_points: Handle,      // [N × 3]
    voxel_means: Handle,        // [V × 3]
    voxel_inv_covs: Handle,     // [V × 9]
    voxel_valid: Handle,        // [V]

    // Per-iteration buffers (reused)
    transform: Handle,          // [16]
    transformed_points: Handle, // [N × 3]
    neighbor_indices: Handle,   // [N × MAX_NEIGHBORS]
    neighbor_counts: Handle,    // [N]
    scores: Handle,             // [N]
    correspondences: Handle,    // [N]
    gradients: Handle,          // [N × 6]
    hessians: Handle,           // [N × 36]

    // Reduction output
    total_score: Handle,        // [1]
    total_correspondences: Handle, // [1]
    total_gradient: Handle,     // [6]
    total_hessian: Handle,      // [36]
}

impl GpuDerivativePipeline {
    /// Create pipeline with given capacity.
    pub fn new(max_points: usize, max_voxels: usize) -> Result<Self>;

    /// Upload alignment data (call once per align()).
    pub fn upload_alignment_data(
        &mut self,
        source_points: &[[f32; 3]],
        voxel_data: &GpuVoxelData,
    ) -> Result<()>;

    /// Compute derivatives for one iteration (call per iteration).
    pub fn compute_iteration(
        &mut self,
        pose: &[f64; 6],
        gauss_d1: f32,
        gauss_d2: f32,
        search_radius: f32,
    ) -> Result<GpuDerivativeResult>;
}
```

**Tests**:
- `test_pipeline_creation`
- `test_pipeline_single_iteration`
- `test_pipeline_multi_iteration`
- `test_pipeline_matches_cpu`

### Phase 12.3: Jacobian/Hessian Handling

**Goal**: Decide how to handle point Jacobians and point Hessians.

**Options**:

| Option | Jacobians | Point Hessians | Tradeoff |
|--------|-----------|----------------|----------|
| A | CPU, upload once | CPU, upload once | Simple, ~1ms overhead |
| B | GPU kernel | GPU kernel | Complex, saves ~1ms |
| C | CPU, upload per iter | CPU, upload per iter | Current approach |

**Recommendation**: Option A
- Jacobians depend only on source points and pose angles (not position)
- Point Hessians are 144 floats per point but only depend on angles
- Upload once, recompute on pose angle change (rare in practice)

### Phase 12.4: Solver Integration

**Goal**: Replace CPU path in `NdtOptimizer` with GPU pipeline.

**File**: `src/ndt_cuda/src/optimization/solver.rs`

```rust
impl NdtOptimizer {
    /// Align using GPU derivative pipeline.
    pub fn align_gpu(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        initial_guess: Isometry3<f64>,
    ) -> NdtResult {
        // Create pipeline
        let mut pipeline = GpuDerivativePipeline::new(
            source_points.len(),
            target_grid.len(),
        )?;

        // Upload once
        let voxel_data = GpuVoxelData::from_voxel_grid(target_grid);
        pipeline.upload_alignment_data(source_points, &voxel_data)?;

        // Optimization loop
        for iteration in 0..self.config.ndt.max_iterations {
            // GPU derivatives (only uploads pose, downloads 43 floats)
            let derivatives = pipeline.compute_iteration(
                &pose,
                self.gauss.d1 as f32,
                self.gauss.d2 as f32,
                self.config.ndt.resolution as f32,
            )?;

            // CPU Newton solve (unchanged)
            let delta = newton_step_regularized(...)?;
            pose = apply_delta(pose, delta);
        }
    }
}
```

**Tests**:
- `test_align_gpu_matches_cpu` - Results within tolerance
- `test_align_gpu_convergence` - Converges on test data
- `test_align_gpu_performance` - Benchmark vs CPU

### Phase 12.5: Performance Validation

**Goal**: Measure and validate speedup.

**Benchmarks**:
```rust
#[bench]
fn bench_derivatives_cpu(b: &mut Bencher) { ... }

#[bench]
fn bench_derivatives_gpu_current(b: &mut Bencher) { ... }

#[bench]
fn bench_derivatives_gpu_zero_copy(b: &mut Bencher) { ... }

#[bench]
fn bench_full_alignment_cpu(b: &mut Bencher) { ... }

#[bench]
fn bench_full_alignment_gpu(b: &mut Bencher) { ... }
```

**Expected Results**:

| Metric | CPU | GPU (current) | GPU (zero-copy) |
|--------|-----|---------------|-----------------|
| Per-iteration | ~1.5ms | ~2ms (transfer overhead) | ~0.5ms |
| Full alignment (30 iter) | ~45ms | ~60ms | ~20ms |
| Speedup | 1x | 0.75x | 2.25x |

## Dependencies

- Phase 11 (GPU Zero-Copy Voxel Pipeline) - Completed ✅
- CubeCL atomic operations support
- cuda_ffi for potential CUB reduction (fallback)

## Risks

1. **Atomic contention**: Many threads writing to same 43 locations
   - Mitigation: Two-phase reduction (block-level first)

2. **Jacobian recomputation**: If pose angles change significantly
   - Mitigation: Track angle delta, recompute when > threshold

3. **Memory pressure**: Large point clouds need significant GPU memory
   - Mitigation: Streaming for very large clouds (future work)

## Success Criteria

- [ ] GPU derivative results match CPU within 1e-5 tolerance
- [ ] Full alignment speedup ≥ 2x vs CPU
- [ ] No regression in convergence rate
- [ ] Memory usage < 2x current GPU path
