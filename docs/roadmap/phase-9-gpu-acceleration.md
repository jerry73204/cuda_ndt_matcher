# Phase 9: Full GPU Acceleration

**Status**: ⚠️ Partial - 9.1 workaround applied, 9.2 complete

## Goal

Enable GPU acceleration for all compute-intensive operations, not just scoring.

## Current GPU Status

| Component               | GPU Kernels Exist | GPU Active            | Reason Disabled         |
|-------------------------|-------------------|-----------------------|-------------------------|
| Voxel Grid Construction | ✅ Yes            | ❌ No                 | CubeCL optimizer bugs   |
| Radius Search           | ✅ Yes            | ✅ Yes (scoring only) | Works in scoring path   |
| Derivative Computation  | ✅ Yes            | ❌ No                 | Only used for scoring   |
| Newton Solve            | ❌ No             | ❌ No                 | Too small for GPU (6x6) |
| Transform Probability   | ✅ Yes            | ✅ Yes                | Working                 |
| NVTL Evaluation         | ✅ Yes            | ✅ Yes                | Working                 |

## 9.1 Fix CubeCL Optimizer Issues ⚠️ WORKAROUND APPLIED

**Problem**: CubeCL uniformity analysis panics on complex control flow.

**Trigger Pattern**:
```rust
// This causes "no entry found for key" panic:
for v in 0..dynamic_bound {
    if condition { break; }
}
```

**Status (2026-01-03)**:

1. **Upgrade CubeCL** - ❌ BLOCKED
   - CubeCL 0.9.0-pre.6 available but has major API breaking changes
   - Migration requires significant refactoring (type system changes, Runtime trait changes)
   - Waiting for 0.9.0 stable release with migration guide
   - Current version: 0.8.1

2. **Simplify Kernels** - ✅ APPLIED
   - All GPU kernels now use conditional flags instead of `break`/`continue`
   - Pattern: `for i in 0..MAX { if i < count { ... } }`
   - Applied in `radius_search_kernel`, score kernels, gradient kernels
   - All 253 unit tests pass

**Workaround Pattern** (in place):
```rust
// Instead of:
for v in 0..num_voxels {
    if count >= MAX_NEIGHBORS { break; }
}

// Use:
for i in 0..MAX_NEIGHBORS {  // Static bound
    if i < num_neighbors {   // Conditional instead of break
        // Process...
    }
}
```

**Files with Workaround Applied**:
- `src/ndt_cuda/src/derivatives/gpu.rs` - All score/gradient kernels
- `src/ndt_cuda/src/voxel_grid/kernels.rs` - Voxel ID and transform kernels
- `src/ndt_cuda/src/filtering/kernels.rs` - Filter mask and compact kernels

**Next Steps**:
- Monitor CubeCL 0.9.0 stable release for easier migration path
- GPU derivative computation for optimization loop (Phase 9.3)

## 9.2 Enable GPU Voxel Grid Construction ✅ COMPLETE

**Status (2026-01-03)**: Hybrid GPU/CPU approach implemented and working.

**Implementation**:
- `GpuVoxelGridBuilder` struct in `src/ndt_cuda/src/voxel_grid/gpu_builder.rs`
- GPU computes voxel IDs in parallel using `compute_voxel_ids_kernel`
- CPU handles statistics accumulation (mean, covariance) with rayon parallelism
- Automatic fallback to pure CPU if CUDA unavailable

**Key Components**:
- `GpuVoxelGridBuilder::new()` - Creates CUDA client
- `GpuVoxelGridBuilder::build()` - Builds voxel grid with GPU acceleration
- `VoxelGrid::from_points_gpu()` - Convenience method with automatic fallback
- `VoxelGrid::insert()` - Direct voxel insertion for builder use
- `VoxelGrid::build_search_index()` - Builds KD-tree after construction

**Why Hybrid Approach**:
- Voxel ID computation parallelizes well on GPU (3 divisions per point)
- Statistics accumulation requires atomic operations not well-supported in CubeCL 0.8
- CPU with rayon provides efficient parallel statistics computation
- Overall speedup for large point clouds (>100K points)

**Files Created/Modified**:
- `src/ndt_cuda/src/voxel_grid/gpu_builder.rs` - NEW: GpuVoxelGridBuilder
- `src/ndt_cuda/src/voxel_grid/mod.rs` - Added insert(), build_search_index(), from_points_gpu()
- `src/ndt_cuda/src/lib.rs` - Exported GpuVoxelGridBuilder

**Tests**: 255 unit tests pass including GPU tests (CUDA hardware available and verified)

## 9.3 Enable GPU Derivatives for Optimization 🔲

**Current State**: GPU kernels compute gradients but only for scoring, not optimization loop.

**Problem**: Optimization loop needs Hessian on CPU for Newton solve.

**Options**:

**Option A: GPU Gradient, CPU Hessian**
```rust
// Each iteration:
1. GPU: Transform points
2. GPU: Compute gradient (6x1)
3. CPU: Compute Hessian (6x6) - too small for GPU
4. CPU: Newton solve
```

**Option B: Batched GPU Hessian**
```rust
// Batch multiple alignment attempts:
1. GPU: Transform N point clouds
2. GPU: Compute N gradients + N Hessians
3. CPU: N Newton solves (can parallelize)
```

**Implementation**:
```rust
// Add to GpuRuntime:
pub fn compute_derivatives_batch(
    &self,
    source_points: &CubeBuffer<f32>,
    poses: &[Isometry3<f64>],
) -> Vec<DerivativeResult> {
    // Single kernel launch for all poses
    // Returns gradient + Hessian for each
}
```

**Files to Modify**:
- `src/ndt_cuda/src/runtime.rs` - Add batch derivative computation
- `src/ndt_cuda/src/optimization/solver.rs` - Use GPU derivatives

## 9.4 GPU Memory Pooling 🔲

**Goal**: Reduce GPU memory allocation overhead during iteration.

**Implementation**:
```rust
pub struct GpuBufferPool {
    /// Pre-allocated buffers for common sizes
    point_buffers: Vec<CubeBuffer<f32>>,
    /// Score accumulation buffers
    score_buffers: Vec<CubeBuffer<f32>>,
    /// Gradient buffers
    gradient_buffers: Vec<CubeBuffer<f32>>,
}

impl GpuBufferPool {
    pub fn acquire_point_buffer(&mut self, size: usize) -> &mut CubeBuffer<f32>;
    pub fn release_point_buffer(&mut self, buffer: CubeBuffer<f32>);
}
```

## 9.5 Async GPU Execution 🔲

**Goal**: Overlap CPU work with GPU execution using CUDA streams.

**Implementation**:
```rust
// Pipeline: while GPU processes iteration N, CPU processes iteration N-1
pub struct AsyncPipeline {
    stream_compute: CudaStream,
    stream_transfer: CudaStream,
}

impl AsyncPipeline {
    pub fn submit_derivatives(&self, ...);
    pub fn get_previous_result(&self) -> Option<DerivativeResult>;
}
```

## Performance Targets

| Metric | Current (CPU) | Target (GPU) |
|--------|---------------|--------------|
| Alignment latency | ~50ms | <20ms |
| Voxel grid build | ~200ms | <50ms |
| Scoring (NVTL) | ~5ms | <2ms |
| Memory usage | ~100MB | <500MB GPU |

## Tests

- [x] GPU voxel grid matches CPU within tolerance (test_gpu_cpu_consistency)
- [ ] GPU derivatives match CPU within tolerance
- [ ] No memory leaks during continuous operation
- [ ] Performance improvement measurable
- [x] Graceful fallback when GPU unavailable (from_points_gpu() fallback)
