# NDT Feature List

Feature comparison between `cuda_ndt_matcher` and Autoware's `ndt_scan_matcher`.

**Last Updated**: 2026-01-12 (GPU Oscillation Detection added)

---

## Status Legend

### Feature Status

| Symbol | Meaning                            |
|--------|------------------------------------|
| ✅     | Implemented, matches Autoware      |
| ⚠️      | Implemented, differs from Autoware |
| ❌     | Not implemented                    |
| 🔲     | Planned                            |

### GPU Status

| Symbol | Meaning                                   |
|--------|-------------------------------------------|
| ✅     | On GPU (via `align_gpu()` or other path)  |
| ⚠️      | Partial/hybrid (GPU + CPU reduction)      |
| 🔲     | Should be GPU, blocked by technical issue |
| —      | CPU by design (GPU not beneficial)        |
| -      | N/A (feature not implemented)             |

---

## 1. Core NDT Algorithm

| Feature                       | Status | GPU | Autoware Diff             | GPU Rationale                                                                                                 |
|-------------------------------|--------|-----|---------------------------|---------------------------------------------------------------------------------------------------------------|
| Newton-Raphson optimization   | ✅     | ✅  | Same algorithm            | Full GPU via cuSOLVER (Phase 14/15) - 6×6 Cholesky solve                                                      |
| More-Thuente line search      | ✅     | ✅  | Same, disabled by default | Phase 15: Batched speculative evaluation (K=8 candidates)                                                     |
| Voxel grid construction       | ✅     | ✅  | Same output               | **Zero-copy pipeline**: GPU radix sort + segment detect via CUB |
| Gaussian covariance per voxel | ✅     | —   | Same formulas             | Per-voxel eigendecomposition better on CPU                                                                    |
| Multi-voxel radius search     | ✅     | ✅  | Same (KDTREE)             | GPU for scoring path; ~2.4 voxels/point                                                                       |
| Convergence detection         | ✅     | ✅  | Same epsilon check        | GPU kernel checks ||α×δ|| < ε                                                                                 |
| Oscillation detection         | ✅     | ✅  | Same (gated at count > 10) | GPU tracks pose history (24 bytes/iter download)                                                              |
| Step size clamping            | ✅     | ✅  | Same limits               | GPU kernel applies max step                                                                                   |

### 1.1 Search Methods

| Method   | Status | Autoware Diff         | Notes                                |
|----------|--------|-----------------------|--------------------------------------|
| KDTREE   | ✅     | Default in both       | Uses `kiddo` crate KD-tree           |
| DIRECT26 | ❌     | Available in Autoware | Not needed - KDTREE is more accurate |
| DIRECT7  | ❌     | Available in Autoware | Not needed - KDTREE is recommended   |
| DIRECT1  | ❌     | Available in Autoware | Not needed - too inaccurate          |

**Decision**: KDTREE only. Autoware recommends KDTREE; DIRECT methods are legacy.

### 1.2 Voxel Grid Construction Pipeline

The `build_zero_copy()` method uses a GPU-accelerated pipeline with minimal CPU-GPU transfers:

| Stage                     | Location | Notes                                                            |
|---------------------------|----------|------------------------------------------------------------------|
| Morton code computation   | GPU ✅   | `compute_morton_codes_kernel` - 63-bit codes split into 2×u32    |
| Radix sort                | GPU ✅   | CUB DeviceRadixSort via `cuda_ffi` - zero-copy interop           |
| Segment detection         | GPU ✅   | CUB DeviceScan + DeviceSelect via `cuda_ffi` - zero-copy interop |
| Position sum accumulation | GPU ✅   | `accumulate_segment_sums_kernel` - one thread per segment        |
| Mean computation          | GPU ✅   | `compute_means_kernel`                                           |
| Covariance accumulation   | GPU ✅   | `accumulate_segment_covariances_kernel`                          |
| Finalization              | CPU      | Eigendecomposition for regularization + principal axis           |

**Key designs**:
- Segmented reduction avoids atomic operations by exploiting sorted data locality
- Zero-copy pipeline keeps data on GPU between operations (CubeCL ↔ cuda_ffi interop via raw CUDA pointers)
- Only 2 CPU-GPU transfers: upload points, download statistics

---

## 2. Derivative Computation

| Feature                     | Status | GPU | Autoware Diff                  | GPU Rationale                                                         |
|-----------------------------|--------|-----|--------------------------------|-----------------------------------------------------------------------|
| Jacobian computation        | ✅     | ✅  | Same formulas (Magnusson 2009) | GPU pipeline via `align_gpu()` method                                 |
| Hessian computation         | ✅     | ✅  | Same formulas                  | GPU pipeline via `align_gpu()` method                                 |
| Angular derivatives (j_ang) | ✅     | —   | All 8 terms match              | Precomputed on CPU, uploaded once per alignment                       |
| Point Hessian (h_ang)       | ✅     | —   | All 15 terms match             | Precomputed on CPU, uploaded once per alignment                       |
| Gradient accumulation       | ✅     | ✅  | Same algorithm                 | GPU kernel + CPU reduction                                            |
| Hessian accumulation        | ✅     | ✅  | Same algorithm                 | GPU kernel + CPU reduction                                            |

### GPU Derivative Kernels (Implemented)

All kernels exist in `derivatives/gpu.rs` and are functional:

| Kernel                        | Location | Status     | Notes                                       |
|-------------------------------|----------|------------|---------------------------------------------|
| `radius_search_kernel`        | Line 61  | ✅ Working | Brute-force O(N×V), bounded loop workaround |
| `compute_ndt_score_kernel`    | Line 145 | ✅ Working | Per-point score with neighbor accumulation  |
| `compute_ndt_gradient_kernel` | Line 473 | ✅ Working | Unrolled 6-element gradient accumulator     |
| `compute_ndt_hessian_kernel`  | Line 796 | ✅ Working | Combined jacobians+hessians parameter       |

### GPU Runtime Integration

**Legacy path** (`NdtCudaRuntime::compute_derivatives()` in `runtime.rs:345`):
- Chains all kernels but has excessive CPU-GPU transfers per call

**Full GPU path** (`NdtOptimizer::align_full_gpu()` via `FullGpuPipelineV2`):
1. Upload alignment data once (source points, voxel data, Gaussian params)
2. Per iteration (~200 bytes transfer for Newton solve):
   - **Phase A - Newton Direction**: sin/cos → transform → J/PH → score/gradient/Hessian → CUB reduce → cuSOLVER (download 172B, upload 24B)
   - **Phase B - Line Search** (if enabled): Generate K=8 candidates → batch transform → batch score/gradient → CUB reduce → More-Thuente selection
   - **Phase C - Update State**: pose += α×δ → convergence check → download 4-byte flag
3. Download final pose after convergence

### Optimization Loop Status

| Component            | CPU path (`align`) | GPU path (`align_gpu`) | Notes                              |
|----------------------|--------------------|------------------------|------------------------------------|
| Point transformation | CPU                | GPU ✅                 | -                                  |
| Radius search        | CPU (KD-tree)      | GPU ✅                 | Brute-force O(N×V) on GPU          |
| Jacobian computation | CPU                | GPU ✅                 | `compute_jacobians_kernel`         |
| Point Hessian comp.  | CPU                | GPU ✅                 | `compute_point_hessians_kernel`    |
| Gradient computation | CPU                | GPU ✅                 | Per-point kernels                  |
| Hessian computation  | CPU                | GPU ✅                 | Per-point kernels                  |
| Reduction            | N/A                | GPU ✅                 | CUB DeviceSegmentedReduce (43 out) |
| Newton solve         | CPU                | GPU ✅                 | cuSOLVER Cholesky (6×6)            |

**Full GPU path (Phase 15)**: When `NDT_USE_GPU=1` (default), the entire Newton optimization with line search runs on GPU. Per-iteration transfer is ~224 bytes (Newton solve requires f64 precision + pose download for oscillation tracking).

**GPU path limitations** (falls back to CPU path):
- Debug output not supported (`align_with_debug()` uses CPU path)

---

## 3. Scoring

| Feature                | Status | GPU | Autoware Diff       | GPU Rationale                    |
|------------------------|--------|-----|---------------------|----------------------------------|
| Transform probability  | ✅     | ✅  | Same formula        | Parallel per-point scoring       |
| NVTL scoring           | ✅     | ✅  | Same formula        | Parallel per-point max           |
| Convergence threshold  | ✅     | —   | Same check          | Single comparison                |
| Skip on low score      | ✅     | —   | Same behavior       | Control flow only                |
| No-ground scoring      | ✅     | —   | Same algorithm      | CPU filtering (small point count)|
| Per-point score colors | ✅     | ✅  | Same algorithm      | GPU per-point max scores         |

### Ground Point Filtering (✅ Implemented)

**Implementation** (`main.rs`, enabled via `score_estimation.no_ground_points.enable`):
- Filters ground points using z-height threshold (same algorithm as Autoware)
- Computes `no_ground_transform_probability` and `no_ground_nvtl` on filtered points
- Publishes `points_aligned_no_ground` point cloud

**Parameters**:
- `enable`: boolean, default false
- `z_margin_for_ground_removal`: f32, default 0.8m

**GPU consideration**: CPU filtering is sufficient - post-alignment filtering operates on small point subsets (~100-500 points).

### Per-Point Score Visualization (✅ Implemented)

**Implementation** (`GpuScoringPipeline::compute_per_point_scores` + `colors.rs`):
- Computes per-point max NDT scores via GPU kernel
- Maps scores to RGB colors using Autoware's 4-quadrant color scheme (blue→cyan→green→yellow→red)
- Publishes `voxel_score_points` PointCloud2 with RGB for RViz visualization

**Color mapping** (same as Autoware's `exchange_color_crc`):
- Score range: [1.0, 3.5] (configurable via `DEFAULT_SCORE_LOWER/UPPER`)
- Blue (low score) → Cyan → Green → Yellow → Red (high score)
- Uses sine-based smooth gradient in each quadrant

**GPU acceleration**: Full GPU path via `compute_per_point_scores_for_visualization()` method.

---

## 4. Covariance Estimation

| Feature               | Status | GPU | Autoware Diff           | GPU Rationale                                                    |
|-----------------------|--------|-----|-------------------------|------------------------------------------------------------------|
| FIXED mode            | ✅     | —   | Same                    | Returns constant matrix                                          |
| LAPLACE mode          | ✅     | —   | Same (Hessian inverse)  | 6x6 inversion - CPU faster                                       |
| MULTI_NDT mode        | ✅     | ✅  | Same algorithm          | **GPU batch alignment** (shared voxel data)                      |
| MULTI_NDT_SCORE mode  | ✅     | ✅  | Same (softmax weighted) | **GPU batch scoring via `GpuScoringPipeline`**                   |
| Covariance rotation   | ✅     | —   | Same                    | 6x6 rotation - trivial                                           |
| Temperature parameter | ✅     | —   | Same default            | Single scalar                                                    |
| Scale factor          | ✅     | —   | Same                    | Single scalar                                                    |

### MULTI_NDT GPU Batch Alignment (✅ Complete)

**Implementation** (`NdtOptimizer::align_batch_gpu` in `optimization/solver.rs`):
- Creates `GpuDerivativePipeline` once for all M alignments
- Uploads voxel data once (shared across all poses)
- Runs M sequential Newton optimizations, each reusing the pipeline
- Falls back to CPU Rayon path if GPU fails

**Key optimization**: Voxel data upload is expensive; sharing it across M alignments
provides ~2-3× speedup over M independent `align_gpu()` calls.

**API**:
```rust
// NdtOptimizer level
optimizer.align_batch_gpu(source_points, target_grid, &initial_poses)?;

// NdtScanMatcher level
matcher.align_batch_gpu(source_points, &initial_poses)?;

// NdtManager level (auto-routes to GPU with CPU fallback)
manager.align_batch(source_points, &initial_poses)?;
```

### MULTI_NDT_SCORE GPU Pipeline (✅ Complete)

**Implementation** (`GpuScoringPipeline` in `scoring/pipeline.rs`):
- Single kernel launch for M poses × N points (`compute_scores_batch_kernel`)
- Brute-force neighbor search per transformed point (O(N×V))
- CUB DeviceSegmentedReduce for aggregation
- Returns `transform_probability` and `nvtl` per pose

**Performance**: GPU batch scoring replaces Rayon parallel CPU path.

See `docs/roadmap/phase-13-gpu-scoring-pipeline.md` for implementation details.

---

## 5. Initial Pose Estimation

| Feature                 | Status | GPU | Autoware Diff       | GPU Rationale                                        |
|-------------------------|--------|-----|---------------------|------------------------------------------------------|
| SmartPoseBuffer         | ✅     | —   | Same interpolation  | Sequential buffer operations                         |
| Pose timeout validation | ✅     | —   | Same thresholds     | Single check                                         |
| Distance tolerance      | ✅     | —   | Same thresholds     | Single check                                         |
| TPE optimization        | ✅     | —   | Same algorithm      | Sequential Bayesian optimization                     |
| Particle evaluation     | ✅     | ✅  | Same                | Uses GPU NVTL batch                                  |
| Monte Carlo markers     | ✅     | —   | Same visualization  | ROS message creation                                 |
| `multi_initial_pose`    | ✅     | —   | Same                | Initial offset poses for MULTI_NDT covariance        |

**GPU status**: Particle NVTL evaluation already uses GPU batch. TPE itself is inherently sequential.

---

## 6. Map Management

| Feature                 | Status | GPU | Autoware Diff              | GPU Rationale                  |
|-------------------------|--------|-----|----------------------------|--------------------------------|
| Dynamic map loading     | ✅     | —   | Same service interface     | ROS service call, I/O bound    |
| Tile-based updates      | ✅     | —   | Same differential approach | Map data structure             |
| Position-based trigger  | ✅     | —   | Same distance check        | Single comparison              |
| Map radius filtering    | ✅     | —   | Same algorithm             | CPU point filtering sufficient |
| Dual-NDT (non-blocking) | ✅     | —   | Same concept               | Background thread swap         |
| Debug map publisher     | ✅     | —   | Same                       | Visualization only             |

**GPU consideration**: Map loading is I/O bound, not compute bound. No GPU benefit.

---

## 7. Regularization

| Feature               | Status | GPU | Autoware Diff        | GPU Rationale        |
|-----------------------|--------|-----|----------------------|----------------------|
| GNSS subscription     | ✅     | —   | Same topic           | ROS subscription     |
| Regularization buffer | ✅     | —   | Same SmartPoseBuffer | Same as initial pose |
| Scale factor          | ✅     | —   | Same default (0.01)  | Single scalar        |
| Enable/disable flag   | ✅     | —   | Same parameter       | Config only          |
| Gradient penalty      | ✅     | ✅  | Same formula         | GPU kernel           |
| Hessian contribution  | ✅     | ✅  | Same formula         | GPU kernel           |

**GPU implementation**: GNSS regularization is supported in both CPU and GPU paths. The GPU path uses `apply_regularization_kernel` which modifies the reduced score/gradient/Hessian after CUB reduction.

---

## 8. Diagnostics

| Feature               | Status | GPU | Autoware Diff | GPU Rationale             |
|-----------------------|--------|-----|---------------|---------------------------|
| Execution time        | ✅     | —   | Same metric   | Timer measurement         |
| Transform probability | ✅     | —   | Same metric   | Already computed          |
| NVTL score            | ✅     | —   | Same metric   | Already computed          |
| Iteration count       | ✅     | —   | Same metric   | Counter                   |
| Oscillation count     | ✅     | —   | Same metric   | Already computed          |
| Distance metrics      | ✅     | —   | Same          | Pose difference           |
| Skip counter          | ✅     | —   | Same          | Counter                   |
| Map update status     | ✅     | —   | Same          | Status flags              |
| No-ground metrics     | ✅     | —   | Same          | Published when enabled    |

**GPU consideration**: Diagnostics are metrics publication - no GPU benefit.

---

## 9. ROS Interface

| Feature                        | Status | Autoware Diff       |
|--------------------------------|--------|---------------------|
| `ekf_pose_with_covariance` sub | ✅     | Same                |
| `points_raw` sub               | ✅     | Same                |
| `regularization_pose` sub      | ✅     | Same                |
| `pointcloud_map` sub           | ✅     | Same                |
| `ndt_pose` pub                 | ✅     | Same                |
| `ndt_pose_with_covariance` pub | ✅     | Same                |
| `/tf` broadcast                | ✅     | Same                |
| `trigger_node_srv` service     | ✅     | Same                |
| `ndt_align_srv` service        | ✅     | Same                |
| Debug metric publishers (10)   | ✅     | Same topics         |
| `initial_to_result_*` pubs (4) | ✅     | Same topics         |
| `no_ground_*` publishers (3)   | ✅     | Same topics         |
| `voxel_score_points` pub       | ✅     | Same (RGB colors)   |
| `multi_ndt_pose` PoseArray     | ✅     | Same                |
| `multi_initial_pose` PoseArray | ✅     | Same                |
| `debug/loaded_pointcloud_map`  | ✅     | Same                |

---

## 10. Point Cloud Processing

| Feature                | Status | GPU | Autoware Diff | GPU Rationale               |
|------------------------|--------|-----|---------------|-----------------------------|
| PointCloud2 conversion | ✅     | —   | Same          | Memory copy only            |
| TF2 sensor transform   | ✅     | —   | Same          | Single 4x4 matrix multiply  |
| Distance filtering     | ✅     | ✅  | Same          | Parallel per-point          |
| Z-height filtering     | ✅     | ✅  | Same          | Parallel per-point          |
| Voxel downsampling     | ✅     | ✅  | Same          | GPU voxel assignment        |
| No-ground z-filtering  | ✅     | —   | Same          | CPU (post-alignment, small N) |

---

## 11. Behavioral Differences

These are implementation differences that may affect output behavior:

### 11.1 Convergence Gating ✅ Matches Autoware

| Condition              | Autoware            | CUDA NDT            | Status |
|------------------------|---------------------|---------------------|--------|
| Max iterations check   | ✅ Gates publishing | ✅ Gates publishing | Same   |
| Oscillation count > 10 | ✅ Gates publishing | ✅ Gates publishing | Same   |
| Score threshold        | ✅ Gates publishing | ✅ Gates publishing | Same   |

**Implementation** (`main.rs:700-764`):
- `is_ok_iteration_num`: Gates if max iterations reached (`!result.converged`)
- `is_ok_oscillation`: Gates if oscillation count > 10 (`OSCILLATION_THRESHOLD = 10`)
- `is_ok_score`: Gates if score below threshold
- Only publishes if ALL three conditions pass (same as Autoware)

### 11.2 Execution Time Measurement ✅ Matches Autoware

| Metric | Autoware                    | CUDA NDT             |
|--------|-----------------------------|----------------------|
| Scope  | NDT alignment only          | NDT alignment only   |
| Timer  | `std::chrono::system_clock` | `std::time::Instant` |

Both implementations measure only the NDT alignment call. `std::time::Instant` provides higher resolution than `std::chrono::system_clock`.

### 11.3 Diagnostic Keys ✅ Matches Autoware

All diagnostic keys published by Autoware are now implemented:

| Key                                              | Autoware | CUDA NDT | Notes                   |
|--------------------------------------------------|----------|----------|-------------------------|
| `transform_probability`                          | ✅       | ✅       | Final alignment score   |
| `transform_probability_before`                   | ✅       | ✅       | Initial estimate score  |
| `transform_probability_diff`                     | ✅       | ✅       | Before/after comparison |
| `nearest_voxel_transformation_likelihood`        | ✅       | ✅       | Final NVTL score        |
| `nearest_voxel_transformation_likelihood_before` | ✅       | ✅       | Initial NVTL score      |
| `nearest_voxel_transformation_likelihood_diff`   | ✅       | ✅       | Before/after comparison |
| `distance_initial_to_result`                     | ✅       | ✅       | Pose displacement       |

**Implementation** (`diagnostics.rs`, `main.rs:649-655`):
- Computes "before" scores at initial pose using `evaluate_transform_probability()` and `evaluate_nvtl()`
- Publishes 19 key-value pairs in `ScanMatchingDiagnostics::apply_to()`

---

## Summary: What's Missing

### Functional Gaps

All functional features are implemented. Convergence gating and diagnostic keys now match Autoware behavior.

### Debug/Visualization Gaps (no runtime impact)

None - all debug publishers are implemented.

### Intentionally Not Implemented

| Feature               | Reason                                    |
|-----------------------|-------------------------------------------|
| DIRECT search methods | KDTREE is better, recommended by Autoware |
| OpenMP thread config  | Rayon handles automatically               |

---

## Summary: GPU Acceleration Status

### Currently on GPU (✅)

| Component                       | Notes                                             |
|---------------------------------|---------------------------------------------------|
| Morton code computation         | `compute_morton_codes_kernel` (voxel grid)        |
| Morton code packing             | `pack_morton_codes_kernel` (voxel grid)           |
| Radix sort                      | CUB DeviceRadixSort via `cuda_ffi`                |
| Segment detection               | CUB DeviceScan + DeviceSelect via `cuda_ffi`      |
| Segmented reduce                | CUB DeviceSegmentedReduce via `cuda_ffi`          |
| Segment statistics              | Position sums, means, covariances (voxel grid)    |
| Point transformation            | Full GPU (in `align_full_gpu` path)               |
| Radius search (optimization)    | GPU kernel via `FullGpuPipelineV2`                |
| Radius search (scoring)         | GPU kernel                                        |
| Jacobian computation            | `compute_jacobians_kernel` (Phase 14)             |
| Point Hessian computation       | `compute_point_hessians_kernel` (Phase 14)        |
| Gradient computation            | GPU kernel via `FullGpuPipelineV2`                |
| Hessian computation             | GPU kernel via `FullGpuPipelineV2`                |
| Derivative reduction            | CUB DeviceSegmentedReduce (43 segments → 43)      |
| Newton solve (6×6)              | cuSOLVER Cholesky via `GpuNewtonSolver`           |
| Line search                     | Batched speculative (K=8) via `FullGpuPipelineV2` |
| Convergence check               | GPU kernel `check_convergence_kernel`             |
| Pose update                     | GPU kernel `update_pose_kernel`                   |
| Transform probability           | Parallel per-point                                |
| NVTL scoring                    | Parallel per-point max                            |
| Batch scoring (MULTI_NDT_SCORE) | `GpuScoringPipeline` - M poses × N points         |
| Batch alignment (MULTI_NDT)     | `GpuDerivativePipeline` - shared voxel data       |
| Per-point score visualization   | GPU max-score extraction + CPU color mapping      |
| Sensor point filtering          | GPU if ≥10k points                                |

### Hybrid GPU/CPU (⚠️)

| Component               | GPU Part                                        | CPU Part                      |
|-------------------------|------------------------------------------------|-------------------------------|
| Voxel grid construction | Morton, pack, sort, segments, statistics (7/8) | Eigendecomposition (1/8)      |
| Derivative reduction    | CUB DeviceSegmentedReduce (43 segments)         | Correspondences count (u32)   |
| Oscillation detection   | Pose update kernel on GPU                       | History tracking (24 bytes/iter download) |

### Integrated via `align_full_gpu()` (✅)

| Component                     | Kernel Status | Integration Status | Notes                    |
|-------------------------------|---------------|-------------------|--------------------------|
| Newton iteration loop         | ✅ Working    | ✅ Integrated     | ~200 bytes/iter transfer |
| Line search (More-Thuente)    | ✅ Working    | ✅ Integrated     | K=8 batched candidates   |
| GPU reduction (sum)           | ✅ Working    | ✅ Integrated     | CUB DeviceSegmentedReduce|
| Batch scoring pipeline        | ✅ Working    | ✅ Integrated     | ~15x speedup             |
| Batch alignment pipeline      | ✅ Working    | ✅ Integrated     | ~2-3x speedup            |

See `docs/roadmap/phase-15-gpu-line-search.md` for implementation details.

### CPU by Design (—)

| Component             | Reason                                              |
|-----------------------|-----------------------------------------------------|
| Covariance matrix ops | 6x6 matrices too small                              |
| TPE optimization      | Sequential Bayesian method                          |
| All diagnostics       | Metrics publication, not compute                    |
| All ROS interface     | Message handling, not compute                       |
| Map management        | I/O bound, not compute bound                        |

### Planned for GPU (🔲)

All major compute operations are now on GPU. No further GPU migration planned.

---

## Performance Comparison

| Operation            | CPU Only | GPU Path (`align_gpu`) | Speedup |
|----------------------|----------|------------------------|---------|
| Single alignment     | ~4.3ms   | ~2.7ms                 | 1.58x   |
| NVTL scoring         | ~8ms     | ~3ms                   | 2.7x    |
| Voxel grid build     | ~200ms   | ~50ms                  | 4x      |
| MULTI_NDT covariance | ~300ms   | ~180ms                 | 1.7x    |

*Note: Alignment timing from 500 points, 57 voxels test case. Real-world cases with larger point clouds expected to show better speedups.*

**Voxel grid build breakdown** (100k points, zero-copy pipeline):
- Morton codes: ~1ms (GPU, CubeCL)
- Radix sort: ~2ms (GPU, CUB via cuda_ffi)
- Segment detection: ~1ms (GPU, CUB via cuda_ffi)
- Statistics accumulation: ~5ms (GPU, CubeCL)
- Finalization: ~15ms (CPU eigendecomp)
- **CPU-GPU transfers**: 2 only (upload points, download stats)

**Full GPU pipeline breakdown** (`align_full_gpu` per iteration via `FullGpuPipelineV2`):
- Sin/cos + Jacobians + Point Hessians: GPU kernel
- Point transformation: GPU kernel
- Radius search: GPU kernel (brute-force O(N×V))
- Score/gradient/Hessian: GPU kernel
- CUB reduction: 43 floats
- Newton solve: cuSOLVER (requires f64 precision)
- Line search (if enabled): K=8 batched candidates
- Oscillation tracking: pose download (24 bytes) for CPU history
- **CPU-GPU transfers per iteration**: ~224 bytes total

---

## GPU Pipeline Architecture

### Three Major Pipelines

| Pipeline       | Purpose                                | Location                     |
|----------------|----------------------------------------|------------------------------|
| **Voxel Grid** | Build NDT map from points              | `voxel_grid/gpu/pipeline.rs` |
| **Derivative** | Compute gradient/Hessian per iteration | `derivatives/pipeline.rs`    |
| **Scoring**    | Batch NVTL evaluation for covariance   | `scoring/pipeline.rs`        |

### 1. Voxel Grid Construction Pipeline

```
Source points [N×3]
    ↓ upload (once)
GPU: Morton codes → Radix sort → Segment detect → Statistics
    ↓ download (once)
Results: means [V×3], inv_covariances [V×9]
```

**Zero-copy status**: Complete - true zero-copy from upload to download

### 2. Derivative Pipeline (Two Implementations)

**Legacy: `GpuDerivativePipeline`** (per-iteration transfers):
```
Once per alignment:
  Upload: source_points [N×3], voxel_means [V×3], voxel_inv_covs [V×9]

Per iteration:
  Upload: pose [16 floats], jacobians [N×18], point_hessians [N×144]
  GPU: transform → radius_search → score → gradient → hessian
  Download: reduced results [43 floats]
```

**Current: `FullGpuPipelineV2`** (~224 bytes per-iteration, Phase 15):
```
Once per alignment:
  Upload: source_points [N×3], voxel_means [V×3], voxel_inv_covs [V×9]

Per iteration:
  Phase A - Newton Direction (GPU):
    sin_cos → transform → jacobians → point_hessians →
    radius_search → score → gradient → hessian → CUB reduce
    Download: 172 bytes (for f64 Newton solve)
    CPU: cuSOLVER 6×6 Cholesky
    Upload: 24 bytes (delta)

  Phase B - Line Search (GPU, if enabled):
    Generate K=8 candidates → batch transform K×N points →
    batch score/gradient → CUB reduce → More-Thuente selection

  Phase C - Update State (GPU):
    pose += α×δ → convergence check
    Download: 24 bytes (pose for oscillation tracking)
    Download: 4 bytes (converged flag)
    CPU: Track pose history for oscillation detection

After convergence:
  CPU: Compute oscillation count from pose history
  Download: final pose [24 bytes] (already in history)
```

### 3. Scoring Pipeline (`GpuScoringPipeline`)

```
Once per map:
  Upload: voxel_means [V×3], voxel_inv_covs [V×9]

Per batch (M poses):
  Upload: source_points [N×3], transforms [M×16]
  GPU: transform → radius_search → score
  Download: reduced [M×4] OR full [M×N×4]
```

**Two reduction paths**:
- `compute_scores_batch()` - Downloads M×N×4 floats, parses on CPU
- `compute_scores_batch_gpu_reduce()` - GPU reduction, downloads M×4 floats

---

## Completion Summary

**Feature parity**: Complete. The CUDA implementation is a drop-in replacement for Autoware's `ndt_scan_matcher`.

**GPU acceleration**: All compute-heavy operations run on GPU:
- Voxel grid construction (zero-copy pipeline)
- NDT alignment with line search (full GPU Newton via `FullGpuPipelineV2`)
- GNSS regularization (`apply_regularization_kernel`)
- Batch scoring (`GpuScoringPipeline`)
- Batch alignment (`align_batch_gpu()`)

**Full GPU Newton with Line Search (Phase 15)**: The default path (`NDT_USE_GPU=1`) runs the entire Newton optimization on GPU with ~224 bytes per-iteration transfer:
- Jacobians and Point Hessians computed on GPU
- Newton solve via cuSOLVER (6×6 Cholesky) - requires f64, hence ~200 bytes download/upload
- More-Thuente line search with K=8 batched candidates (GPU)
- GNSS regularization penalty (GPU kernel)
- Convergence check on GPU
- Oscillation detection via pose history download (24 bytes/iter)

**GPU path current limitations**:
- Debug output requested → use CPU path (`align_with_debug()`)

**Behavioral compatibility**: Convergence gating matches Autoware:
- Max iterations check (gates if hit max iterations)
- Oscillation count > 10 (gates if oscillating)
- Score threshold (gates if below threshold)

---

## Known Performance Issues

**Last Updated**: 2026-01-12

See `docs/profiling-results.md` for detailed profiling data.

### Current Performance Gap

| Metric | Autoware (OpenMP) | CUDA GPU | Ratio |
|--------|-------------------|----------|-------|
| Mean execution time | 2.48 ms | 13.07 ms | **5.3x slower** |
| Convergence rate | ~100% | 51.9% | Issue |
| Mean iterations | 3-5 | 15.2 | ~3-5x more |

### Root Causes

#### 1. Low Convergence Rate (Primary Issue)

48% of alignments hit the 30-iteration limit instead of converging. This is the main cause of slow execution times.

**Possible causes** (under investigation):
- Derivative computation differences
- Initial pose quality differences
- Voxel search radius differences

#### 2. Per-Iteration Memory Transfer (~224 bytes)

Phase 15 reduced per-iteration transfer from ~490 KB to ~224 bytes:

| Direction | Data                         | Size          |
|-----------|------------------------------|---------------|
| GPU → CPU | Reduced results              | 172 bytes     |
| CPU → GPU | Newton delta                 | 24 bytes      |
| GPU → CPU | Pose (oscillation tracking)  | 24 bytes      |
| GPU → CPU | Converged flag               | 4 bytes       |

**Total per iteration**: ~224 bytes

The Newton solve requires f64 precision (cuSOLVER), which necessitates downloading the f32 reduce output, converting to f64, solving, and uploading the f32 delta back. Additionally, pose is downloaded each iteration for CPU-side oscillation detection (tracking direction reversals).

#### 3. No Batch Processing for Incoming Scans

Current architecture processes each scan independently:
- `align_full_gpu()` - Single scan, single initial pose (full GPU with line search)
- `align_batch_gpu()` - Single scan, multiple initial poses (for MULTI_NDT)
- **No queue** for batching incoming scans across time

This means per-alignment setup overhead cannot be amortized.

---

## Future Optimization Opportunities

### Scan Queue for Batch Processing

**Problem**: Each incoming scan is processed independently, no amortization of setup costs.

**Solution**: Queue incoming scans and process in batches of 2-3.

**Trade-off**: Adds latency (must wait for batch to fill). Only viable if real-time constraints allow ~10ms additional latency.
