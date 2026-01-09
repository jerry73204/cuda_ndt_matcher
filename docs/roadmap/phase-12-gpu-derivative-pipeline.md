# Phase 12: GPU Zero-Copy Derivative Pipeline

**Status**: ✅ Complete
**Priority**: High
**Measured Speedup**: 1.6x per alignment (500 points, 57 voxels); GPU reduction (12.6) eliminates N×43 download overhead

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

### Phase 12.1: Reduction Strategy

**Status**: ⚠️ CPU reduction implemented, GPU reduction planned in Phase 12.6

**Goal**: Sum per-point results on GPU instead of downloading N×43 floats.

**Current Implementation**: CPU reduction downloads N×43 floats and sums on CPU.
The pipeline still benefits from persistent GPU buffers (data stays on GPU between
kernel launches). See **Phase 12.6** for GPU reduction using CUB.

**Current data flow**:
```
GPU: scores[N], correspondences[N], gradients[N×6], hessians[N×36]
     ↓ Download N×43 floats (bottleneck)
CPU: Sum to 43 totals
```

**Target data flow** (Phase 12.6):
```
GPU: scores[N], correspondences[N], gradients[6×N], hessians[36×N]  ← column-major
     ↓ CUB DeviceSegmentedReduce (single kernel)
GPU: totals[43]
     ↓ Download 43 floats only
CPU: Use totals
```

### Phase 12.2: Derivative Pipeline Buffers

**Status**: ✅ Complete

**Goal**: Pre-allocate persistent GPU buffers for the optimization loop.

**File**: `src/ndt_cuda/src/derivatives/pipeline.rs`

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

**Status**: ✅ Complete (Option A implemented)

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

**Status**: ✅ Complete

**Goal**: Replace CPU path in `NdtOptimizer` with GPU pipeline.

**File**: `src/ndt_cuda/src/optimization/solver.rs`

**Implementation**: Added `align_gpu()` method to `NdtOptimizer` that:
1. Creates `GpuDerivativePipeline` at the start of alignment
2. Uploads alignment data once (source points, voxel data, Gaussian params)
3. Uses `pipeline.compute_iteration()` in the optimization loop
4. Handles regularization, convergence checking, and oscillation detection

**Tests**:
- `test_align_gpu_identity` - Basic alignment with GPU path ✅
- `test_align_gpu_no_correspondences` - Handles no correspondences case ✅
- `test_align_gpu_vs_cpu` - Results match CPU within tolerance ✅

### Phase 12.5: Performance Validation

**Status**: ✅ Complete

**Goal**: Measure and validate speedup.

**Benchmark Test**: `test_align_performance` (run with `--ignored` flag)

**Measured Results** (500 points, 57 voxels):

| Metric | CPU | GPU (zero-copy) | Speedup |
|--------|-----|-----------------|---------|
| Per alignment | 4.26ms | 2.69ms | 1.58x |

Notes:
- GPU path includes CPU reduction (downloads N×43 floats, sums on CPU)
- Larger point clouds (typical real-world: 1000+ points) will show better speedups
- GPU reduction (Phase 12.6) will eliminate the N×43 download bottleneck

### Phase 12.6: CUB GPU Reduction

**Status**: ✅ Complete

**Goal**: Replace CPU reduction with CUB DeviceSegmentedReduce to download only 43 floats.

**Approach**: Option D2 - Column-major kernel output + CUB segmented reduce

**Implementation Summary**:
1. Added `cuda_ffi/csrc/segmented_reduce.cu` - CUB DeviceSegmentedReduce wrapper (f32/f64)
2. Added `cuda_ffi/src/segmented_reduce.rs` - Rust bindings with `SegmentedReducer` API
3. Modified `derivatives/gpu.rs` gradient/Hessian kernels for column-major output
4. Added `compute_iteration_gpu_reduce()` to `GpuDerivativePipeline`
5. Updated `runtime.rs` CPU reduction to match column-major layout

**Tests**:
- 7 segmented reduce tests in `cuda_ffi` (all passing)
- `test_pipeline_gpu_reduce_vs_cpu_reduce` - Verifies GPU/CPU reduction produce same results
- `test_pipeline_gpu_reduce_empty_input` - Edge case handling
- 299 total tests passing in `ndt_cuda`

#### The Layout Problem

Current gradient/Hessian kernels output **row-major** (per-point contiguous):
```
gradients[N×6] = [g0₀,g0₁,g0₂,g0₃,g0₄,g0₅, g1₀,g1₁,..., gN₀,gN₁,...]
                  ├────── point 0 ──────┤  ├─ point 1 ─┤
```

We need to sum **columns** (each gradient component across all points):
```
total_gradient[0] = g0₀ + g1₀ + g2₀ + ... + gN₀  (component 0)
total_gradient[1] = g0₁ + g1₁ + g2₁ + ... + gN₁  (component 1)
...
```

CUB DeviceSegmentedReduce sums **contiguous segments**, not strided data.

#### Solution: Column-Major Output

Modify kernels to output **column-major** (per-component contiguous):
```
gradients[6×N] = [g0₀,g1₀,g2₀,...,gN₀, g0₁,g1₁,...,gN₁, ..., g0₅,g1₅,...,gN₅]
                  ├─── component 0 ───┤ ├── component 1 ──┤    ├── component 5 ──┤
```

Then CUB can sum each segment (component) in a single kernel launch.

#### Implementation Steps

**Step 1: Add CUB DeviceSegmentedReduce to cuda_ffi**

```
src/cuda_ffi/
├── cuda/
│   └── segmented_reduce.cu    (NEW - CUB wrapper)
├── src/
│   ├── lib.rs                 (add pub mod segmented_reduce)
│   └── segmented_reduce.rs    (NEW - Rust bindings)
└── build.rs                   (add compilation)
```

C++ interface:
```cpp
extern "C" void segmented_reduce_sum_f32(
    const float* d_input,      // [total_elements]
    float* d_output,           // [num_segments]
    const int* d_offsets,      // [num_segments + 1]
    int num_segments,
    void* d_temp,
    size_t temp_bytes
);

extern "C" size_t segmented_reduce_temp_size(
    int num_items,
    int num_segments
);
```

Rust interface:
```rust
pub struct SegmentedReducer {
    temp_buffer: *mut c_void,
    temp_size: usize,
}

impl SegmentedReducer {
    pub fn new(max_items: usize, max_segments: usize) -> Result<Self>;

    pub fn sum_f32(
        &self,
        input: *mut f32,         // device pointer
        output: *mut f32,        // device pointer
        offsets: *const i32,     // device pointer [num_segments + 1]
        num_segments: i32,
    ) -> Result<()>;
}
```

**Step 2: Modify gradient/Hessian kernels for column-major output**

File: `src/ndt_cuda/src/derivatives/gpu.rs`

```rust
// Before (row-major):
gradients[point_idx * 6 + component] = value;

// After (column-major):
gradients[component * num_points + point_idx] = value;
```

Same change for `compute_ndt_hessian_kernel`.

**Step 3: Update pipeline to use CUB reduction**

File: `src/ndt_cuda/src/derivatives/pipeline.rs`

```rust
impl GpuDerivativePipeline {
    fn reduce_on_gpu(&self) -> Result<GpuDerivativeResult> {
        // Segment offsets: [0, N, 2N, 3N, ..., 43N]
        // For: scores[N], corr[N], grad[6×N], hess[36×N]
        let offsets = [0, N, 2*N, 3*N, 4*N, ..., 43*N];

        // Concatenate buffers (or use separate calls)
        // Call CUB segmented reduce
        self.reducer.sum_f32(combined, totals, offsets, 43)?;

        // Download only 43 floats
        let totals_bytes = self.client.read_one(self.totals.clone());
        // Parse into GpuDerivativeResult
    }
}
```

**Step 4: Add tests**

- `test_segmented_reduce_simple` - Basic CUB reduce test
- `test_gradient_kernel_column_major` - Verify new layout
- `test_hessian_kernel_column_major` - Verify new layout
- `test_gpu_reduction_matches_cpu` - End-to-end validation

#### Data Transfer Comparison

| Metric | Current (CPU reduce) | With CUB (Phase 12.6) |
|--------|---------------------|----------------------|
| Download per iteration | N × 43 × 4 bytes | 43 × 4 = 172 bytes |
| For N=1000, 30 iters | 5.2 MB | 5.2 KB |
| Reduction | 1000× more data | 1000× less |

#### Files to Modify

| File | Changes |
|------|---------|
| `src/cuda_ffi/cuda/segmented_reduce.cu` | NEW - CUB wrapper |
| `src/cuda_ffi/src/segmented_reduce.rs` | NEW - Rust bindings |
| `src/cuda_ffi/src/lib.rs` | Add module export |
| `src/cuda_ffi/build.rs` | Add CUDA compilation |
| `src/ndt_cuda/src/derivatives/gpu.rs` | Column-major output |
| `src/ndt_cuda/src/derivatives/pipeline.rs` | Use CUB reduction |

#### Estimated Effort

- cuda_ffi CUB bindings: ~150 lines C++ + ~100 lines Rust
- Kernel modifications: ~20 lines
- Pipeline integration: ~50 lines
- Tests: ~100 lines
- **Total**: ~420 lines

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

- [x] GPU derivative results match CPU within 1e-5 tolerance
- [x] Full alignment speedup ≥ 1.5x vs CPU (1.58x measured with small test case)
- [x] No regression in convergence rate
- [x] Memory usage < 2x current GPU path
