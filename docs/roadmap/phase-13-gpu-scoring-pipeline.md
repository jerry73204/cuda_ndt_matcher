# Phase 13: GPU Scoring Pipeline

**Status**: ✅ Complete
**Priority**: Medium
**Depends on**: Phase 12 (CUB reduction infrastructure)

## Overview

Create a zero-copy GPU pipeline for batch scoring (transform probability and NVTL). This replaces the current Rayon-parallel CPU implementation used by `evaluate_nvtl_batch()` in MULTI_NDT_SCORE covariance estimation.

## Current State

**CPU Implementation** (`scoring/metrics.rs`, `scoring/nvtl.rs`):
- `compute_transform_probability()` - Sum of NDT scores / num_correspondences
- `compute_nvtl()` - Average of max score per point
- `evaluate_nvtl_batch()` in `ndt.rs` - Rayon parallel, ~15ms for 25 poses

**Existing GPU Infrastructure**:
- `compute_ndt_score_kernel` in `derivatives/gpu.rs` - Per-point scoring
- `radius_search_kernel` in `derivatives/gpu.rs` - Neighbor finding
- CUB DeviceSegmentedReduce via `cuda_ffi` - GPU reduction

## Design

### Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GpuScoringPipeline                              │
├────────────────────────────────────────────────────────────────────────┤
│ Once per map:                                                          │
│   Upload: voxel_means [V×3], voxel_inv_covs [V×9]                     │
│                                                                        │
│ Per batch call (M poses, N points):                                    │
│   Upload: source_points [N×3], transforms [M×16]                       │
│                                                                        │
│   GPU Kernel (M×N threads):                                            │
│     1. Transform point by pose                                         │
│     2. Radius search for neighbors (per transformed point)             │
│     3. Accumulate sum_score and max_score across neighbors             │
│     4. Output: scores[M×N], max_scores[M×N], has_neighbor[M×N]        │
│                                                                        │
│   CUB DeviceSegmentedReduce (3M segments):                             │
│     - Segments 0..M: SUM(scores) → total_scores[M]                    │
│     - Segments M..2M: SUM(max_scores) → nvtl_sums[M]                  │
│     - Segments 2M..3M: SUM(has_neighbor) → nvtl_counts[M]             │
│                                                                        │
│   Download: [3M floats]                                                │
│     transform_probability[m] = total_scores[m] / correspondences[m]   │
│     nvtl[m] = nvtl_sums[m] / nvtl_counts[m]                           │
└────────────────────────────────────────────────────────────────────────┘
```

### Neighbor Search (Matching Autoware)

Per-pose neighbor search to match Autoware's behavior:
1. Transform source point by pose → lands in map frame
2. Find neighbor voxels around the **transformed** point
3. Compute scores against those neighbors

This requires `neighbor_indices[M × N × MAX_NEIGHBORS]` for full batch.

### Memory Layout

Column-major for efficient CUB segmented reduce:
- `scores[m * N + n]` - all scores for pose 0, then pose 1, etc.
- Enables contiguous segments for reduction

## Work Items

### Phase 13.1: Pipeline Infrastructure
**Status**: ✅ Complete

Create `GpuScoringPipeline` struct with buffer allocation.

**Files**:
- `src/ndt_cuda/src/scoring/pipeline.rs` (NEW)
- `src/ndt_cuda/src/scoring/mod.rs` (modify)

**Tasks**:
- [ ] Define `GpuScoringPipeline` struct with all buffer handles
- [ ] Implement `new(max_points, max_voxels, max_poses)` constructor
- [ ] Allocate CUB reduction buffers (reuse pattern from Phase 12.6)
- [ ] Add `raw_ptr()` helper for CubeCL ↔ cuda_ffi interop

**Struct outline**:
```rust
pub struct GpuScoringPipeline {
    client: CudaClient,
    max_points: usize,
    max_voxels: usize,
    max_poses: usize,

    // Persistent voxel data
    voxel_means: Handle,      // [V × 3]
    voxel_inv_covs: Handle,   // [V × 9]
    num_voxels: usize,

    // Per-batch buffers
    source_points: Handle,      // [N × 3]
    transforms: Handle,         // [M × 16]

    // Intermediate (M × N)
    neighbor_indices: Handle,   // [M × N × MAX_NEIGHBORS]
    neighbor_counts: Handle,    // [M × N]

    // Output (column-major)
    scores: Handle,             // [M × N]
    max_scores: Handle,         // [M × N]
    has_neighbor: Handle,       // [M × N]

    // CUB reduction
    reduce_temp: Handle,
    reduce_offsets: Handle,     // [3M + 1]
    reduce_output: Handle,      // [3M]

    // Parameters
    gauss_d1: f32,
    gauss_d2: f32,
    search_radius_sq: f32,
}
```

---

### Phase 13.2: Batched Score Kernel
**Status**: ✅ Complete

Implement GPU kernel that processes M poses × N points.

**Files**:
- `src/ndt_cuda/src/scoring/gpu.rs` (NEW)

**Tasks**:
- [ ] Create `compute_scores_batch_kernel` with (pose_idx, point_idx) thread mapping
- [ ] Transform point by `transforms[pose_idx]`
- [ ] Inline neighbor search (find voxels near transformed point)
- [ ] Accumulate `sum_score` and track `max_score` per point
- [ ] Output to column-major arrays for CUB reduction

**Kernel signature**:
```rust
#[cube(launch_unchecked)]
pub fn compute_scores_batch_kernel<F: Float>(
    source_points: &Array<F>,      // [N × 3]
    transforms: &Array<F>,         // [M × 16]
    voxel_means: &Array<F>,        // [V × 3]
    voxel_inv_covs: &Array<F>,     // [V × 9]
    gauss_d1: F,
    gauss_d2: F,
    search_radius_sq: F,
    num_poses: u32,
    num_points: u32,
    num_voxels: u32,
    // Outputs (column-major: [M × N])
    scores: &mut Array<F>,
    max_scores: &mut Array<F>,
    has_neighbor: &mut Array<u32>,
    correspondences: &mut Array<u32>,
)
```

**Thread layout**:
- Grid: `(M, ceil(N / 256))`
- Block: `(256,)`
- `pose_idx = blockIdx.x`, `point_idx = blockIdx.y * 256 + threadIdx.x`

---

### Phase 13.3: Brute-Force Neighbor Search
**Status**: ✅ Complete

Implement inline brute-force neighbor search in the batched kernel.

**Approach**: For each transformed point, check all voxels within search radius.

**Tasks**:
- [ ] Compute transformed point position
- [ ] Loop over all voxels (bounded by `num_voxels`)
- [ ] Check distance² < search_radius_sq
- [ ] Accumulate score if within radius
- [ ] Track max score and correspondence count

**Note**: Brute-force O(N×V) is acceptable for scoring (not in optimization loop). Can optimize later with spatial hashing if needed.

---

### Phase 13.4: set_target() Method
**Status**: ✅ Complete

Upload voxel grid data to GPU (once per map).

**Files**:
- `src/ndt_cuda/src/scoring/pipeline.rs`

**Tasks**:
- [ ] Implement `set_target(&mut self, voxel_data: &GpuVoxelData, gauss: &GaussianParams)`
- [ ] Upload voxel_means, voxel_inv_covs
- [ ] Store gauss_d1, gauss_d2, search_radius_sq

---

### Phase 13.5: CUB Reduction Integration
**Status**: ✅ Complete

Wire up CUB DeviceSegmentedReduce for batch results.

**Tasks**:
- [ ] Build segment offsets for 3M segments: `[0, N, 2N, ..., 3M×N]`
- [ ] Concatenate outputs: `[scores | max_scores | has_neighbor]` (or use 3 separate reduce calls)
- [ ] Call `segmented_reduce_sum_f32_inplace` via cuda_ffi
- [ ] Parse results into `(total_score, nvtl_sum, nvtl_count)` per pose

**Reduction layout**:
```
Segments 0..M-1:    scores[m×N..(m+1)×N] → total_scores[m]
Segments M..2M-1:   max_scores[m×N..(m+1)×N] → nvtl_sums[m]
Segments 2M..3M-1:  has_neighbor[m×N..(m+1)×N] → nvtl_counts[m]
```

---

### Phase 13.6: compute_scores_batch() Method
**Status**: ✅ Complete

Orchestrate the full pipeline.

**Files**:
- `src/ndt_cuda/src/scoring/pipeline.rs`

**Tasks**:
- [ ] Implement `compute_scores_batch(&mut self, source_points, poses) -> Result<Vec<ScoringResult>>`
- [ ] Upload source_points and transforms
- [ ] Launch batched kernel
- [ ] Run CUB reduction
- [ ] Download 3M floats
- [ ] Compute `transform_probability = total / correspondences`, `nvtl = sum / count`
- [ ] Return results

**API**:
```rust
pub struct BatchScoringResult {
    pub transform_probability: f64,
    pub nvtl: f64,
    pub num_correspondences: usize,
}

impl GpuScoringPipeline {
    pub fn compute_scores_batch(
        &mut self,
        source_points: &[[f32; 3]],
        poses: &[[f64; 6]],
    ) -> Result<Vec<BatchScoringResult>>;
}
```

---

### Phase 13.7: Tests
**Status**: ✅ Complete

Verify GPU results match CPU implementation.

**Files**:
- `src/ndt_cuda/src/scoring/pipeline.rs` (tests module)

**Tasks**:
- [ ] `test_gpu_scoring_single_pose` - Compare GPU vs CPU for single pose
- [ ] `test_gpu_scoring_batch` - Compare GPU vs CPU for multiple poses
- [ ] `test_gpu_scoring_empty_input` - Edge case handling
- [ ] `test_gpu_scoring_no_correspondences` - Points outside map

---

### Phase 13.8: Integration
**Status**: ✅ Complete

Replace Rayon CPU path with GPU pipeline.

**Files**:
- `src/ndt_cuda/src/ndt.rs`

**Tasks**:
- [ ] Add `gpu_scoring_pipeline: Option<GpuScoringPipeline>` to `NdtScanMatcher`
- [ ] Initialize in `new()` if GPU enabled
- [ ] Update `set_target()` to also call `gpu_scoring_pipeline.set_target()`
- [ ] Replace `evaluate_nvtl_batch()` implementation to use GPU when available
- [ ] Benchmark: compare Rayon vs GPU for 25 poses × 1000 points

---

### Phase 13.9: Ground Filtering (Optional)
**Status**: 🔲 Not Started

Add no-ground scoring support.

**Tasks**:
- [ ] Add `is_ground` output array to kernel
- [ ] Filter by `transformed_z < min_z + margin`
- [ ] Separate reduction for ground vs no-ground points
- [ ] Add `compute_scores_batch_with_ground_filter()` method

---

## Expected Performance

| Scenario | Current (Rayon) | GPU Batched | Speedup |
|----------|-----------------|-------------|---------|
| 1 pose, 1000 pts | ~2ms | ~0.5ms | 4× |
| 25 poses, 1000 pts | ~15ms | ~1ms | 15× |
| 25 poses, 5000 pts | ~75ms | ~3ms | 25× |

## Files Summary

**New files**:
- `src/ndt_cuda/src/scoring/gpu.rs` - Batched GPU kernel
- `src/ndt_cuda/src/scoring/pipeline.rs` - `GpuScoringPipeline`

**Modified files**:
- `src/ndt_cuda/src/scoring/mod.rs` - Export new modules
- `src/ndt_cuda/src/ndt.rs` - Integration with `NdtScanMatcher`
- `docs/autoware-comparison.md` - Update GPU status
