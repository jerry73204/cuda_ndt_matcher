# NDT Feature List

Feature comparison between `cuda_ndt_matcher` and Autoware's `ndt_scan_matcher`.

**Last Updated**: 2026-01-09 (Per-point Score Visualization complete)

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
| Newton-Raphson optimization   | ✅     | —   | Same algorithm            | Newton solve is 6x6 matrix - too small for GPU benefit                                                        |
| More-Thuente line search      | ✅     | —   | Same, disabled by default | Sequential algorithm, not parallelizable                                                                      |
| Voxel grid construction       | ✅     | ✅  | Same output               | **Zero-copy pipeline**: GPU radix sort + segment detect via CUB |
| Gaussian covariance per voxel | ✅     | —   | Same formulas             | Per-voxel eigendecomposition better on CPU                                                                    |
| Multi-voxel radius search     | ✅     | ✅  | Same (KDTREE)             | GPU for scoring path; ~2.4 voxels/point                                                                       |
| Convergence detection         | ✅     | —   | Same epsilon check        | Single comparison, no parallelism                                                                             |
| Oscillation detection         | ✅     | —   | Same algorithm            | Sequential history tracking                                                                                   |
| Step size clamping            | ✅     | —   | Same limits               | Single operation                                                                                              |

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

**New GPU path** (`NdtOptimizer::align_gpu()` via `GpuDerivativePipeline`):
1. Upload alignment data once (source points, voxel data, Gaussian params)
2. Per iteration: Upload only pose transform (16 floats)
3. Run full GPU kernel chain: transform → radius_search → score → gradient → hessian
4. GPU reduction via CUB DeviceSegmentedReduce (downloads only 43 floats)
5. Newton solve on CPU (6×6 system)

### Optimization Loop Status

| Component            | CPU path (`align`) | GPU path (`align_gpu`) | Notes                              |
|----------------------|--------------------|------------------------|------------------------------------|
| Point transformation | CPU                | GPU ✅                 | -                                  |
| Radius search        | CPU (KD-tree)      | GPU ✅                 | Brute-force O(N×V) on GPU          |
| Gradient computation | CPU                | GPU ✅                 | Per-point kernels                  |
| Hessian computation  | CPU                | GPU ✅                 | Per-point kernels                  |
| Reduction            | N/A                | GPU ✅                 | CUB DeviceSegmentedReduce (43 out) |
| Newton solve         | CPU                | CPU                    | 6×6 too small for GPU              |

**Measured speedup**: 1.58x with GPU path (500 points, 57 voxels test case).
Larger point clouds (typical 1000+ points) expected to show better speedups.

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
| MULTI_NDT mode        | ✅     | 🔲  | Same algorithm          | Runs batch alignments via Rayon                                  |
| MULTI_NDT_SCORE mode  | ✅     | ✅  | Same (softmax weighted) | **GPU batch scoring via `GpuScoringPipeline`**                   |
| Covariance rotation   | ✅     | —   | Same                    | 6x6 rotation - trivial                                           |
| Temperature parameter | ✅     | —   | Same default            | Single scalar                                                    |
| Scale factor          | ✅     | —   | Same                    | Single scalar                                                    |

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

| Feature                 | Status | GPU | Autoware Diff       | GPU Rationale                    |
|-------------------------|--------|-----|---------------------|----------------------------------|
| SmartPoseBuffer         | ✅     | —   | Same interpolation  | Sequential buffer operations     |
| Pose timeout validation | ✅     | —   | Same thresholds     | Single check                     |
| Distance tolerance      | ✅     | —   | Same thresholds     | Single check                     |
| TPE optimization        | ✅     | —   | Same algorithm      | Sequential Bayesian optimization |
| Particle evaluation     | ✅     | ✅  | Same                | Uses GPU NVTL batch              |
| Monte Carlo markers     | ✅     | —   | Same visualization  | ROS message creation             |
| PoseArray publication   | ❌     | -   | **Missing** (debug) | Not needed for function          |

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
| Debug map publisher     | ❌     | -   | **Missing** (debug)        | Visualization only             |

**GPU consideration**: Map loading is I/O bound, not compute bound. No GPU benefit.

---

## 7. Regularization

| Feature               | Status | GPU | Autoware Diff        | GPU Rationale               |
|-----------------------|--------|-----|----------------------|-----------------------------|
| GNSS subscription     | ✅     | —   | Same topic           | ROS subscription            |
| Regularization buffer | ✅     | —   | Same SmartPoseBuffer | Same as initial pose        |
| Scale factor          | ✅     | —   | Same default (0.01)  | Single scalar               |
| Enable/disable flag   | ✅     | —   | Same parameter       | Config only                 |
| Gradient penalty      | ✅     | —   | Same formula         | Added to 6-element gradient |
| Hessian contribution  | ✅     | —   | Same formula         | Added to 6x6 Hessian        |

**GPU consideration**: Regularization adds to gradient/Hessian - would be GPU if main derivatives were GPU.

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
| 12 debug publishers            | ✅     | Same topics         |
| `no_ground_*` publishers (3)   | ✅     | Same topics         |
| `voxel_score_points` pub       | ✅     | Same (RGB colors)   |
| `multi_ndt_pose` PoseArray     | ❌     | **Missing** (debug) |
| `multi_initial_pose` PoseArray | ❌     | **Missing** (debug) |
| `debug/loaded_pointcloud_map`  | ❌     | **Missing** (debug) |

---

## 10. Point Cloud Processing

| Feature                | Status | GPU | Autoware Diff | GPU Rationale               |
|------------------------|--------|-----|---------------|-----------------------------|
| PointCloud2 conversion | ✅     | —   | Same          | Memory copy only            |
| TF2 sensor transform   | ✅     | —   | Same          | Single 4x4 matrix multiply  |
| Distance filtering     | ✅     | ✅  | Same          | Parallel per-point          |
| Z-height filtering     | ✅     | ✅  | Same          | Parallel per-point          |
| Voxel downsampling     | ✅     | ✅  | Same          | GPU voxel assignment        |
| Ground segmentation    | ❌     | -   | **Missing**   | Would be parallel per-point |

---

## Summary: What's Missing

### Functional Gaps (affects behavior)

All functional features are implemented. Ground point filtering was added in Phase 13.8.

### Debug/Visualization Gaps (no runtime impact)

| Feature                     | Priority | Effort | GPU |
|-----------------------------|----------|--------|-----|
| Multi-NDT PoseArray         | Low      | Low    | No  |
| Debug map publisher         | Low      | Low    | No  |
| Distance old/new publishers | Low      | Low    | No  |

### Intentionally Not Implemented

| Feature               | Reason                                    |
|-----------------------|-------------------------------------------|
| DIRECT search methods | KDTREE is better, recommended by Autoware |
| OpenMP thread config  | Rayon handles automatically               |

---

## Summary: GPU Acceleration Status

### Currently on GPU (✅)

| Component                    | Notes                                           |
|------------------------------|-------------------------------------------------|
| Morton code computation      | `compute_morton_codes_kernel` (voxel grid)      |
| Radix sort                   | CUB DeviceRadixSort via `cuda_ffi`              |
| Segment detection            | CUB DeviceScan + DeviceSelect via `cuda_ffi`    |
| Segmented reduce             | CUB DeviceSegmentedReduce via `cuda_ffi`        |
| Segment statistics           | Position sums, means, covariances (voxel grid)  |
| Point transformation         | Full GPU (in `align_gpu` path)                  |
| Radius search (optimization) | GPU kernel via `GpuDerivativePipeline`          |
| Radius search (scoring)      | GPU kernel                                      |
| Gradient computation         | GPU kernel via `GpuDerivativePipeline`          |
| Hessian computation          | GPU kernel via `GpuDerivativePipeline`          |
| Derivative reduction         | CUB DeviceSegmentedReduce (43 segments → 43)    |
| Transform probability        | Parallel per-point                              |
| NVTL scoring                 | Parallel per-point max                          |
| Batch scoring (MULTI_NDT)    | `GpuScoringPipeline` - M poses × N points       |
| Per-point score visualization| GPU max-score extraction + CPU color mapping    |
| Sensor point filtering       | GPU if ≥10k points                              |

### Hybrid GPU/CPU (⚠️)

| Component               | GPU Part                                   | CPU Part                     |
|-------------------------|--------------------------------------------|------------------------------|
| Voxel grid construction | Morton, sort, segments, statistics (6/7)   | Eigendecomposition (1/7)     |
| Derivative reduction    | CUB DeviceSegmentedReduce (43 segments)    | Correspondences count (u32)  |

### Integrated via `align_gpu()` (✅)

| Component                     | Kernel Status | Integration Status | Speedup   |
|-------------------------------|---------------|-------------------|-----------|
| Gradient in optimization loop | ✅ Working    | ✅ Integrated     | 1.58x     |
| Hessian in optimization loop  | ✅ Working    | ✅ Integrated     | (combined)|
| GPU reduction (sum)           | ✅ Working    | ✅ Integrated     | Minor     |
| Batch scoring pipeline        | ✅ Working    | ✅ Integrated     | ~15x      |
| Batch alignment pipeline      | ❌ Missing    | Not started       | 3-5x      |

See `docs/roadmap/phase-12-gpu-derivative-pipeline.md` for implementation details.

### CPU by Design (—)

| Component             | Reason                           |
|-----------------------|----------------------------------|
| Newton solve          | 6x6 matrix too small for GPU     |
| Line search           | Sequential algorithm             |
| Covariance matrix ops | 6x6 matrices too small           |
| TPE optimization      | Sequential Bayesian method       |
| Oscillation detection | Sequential history               |
| Angular derivatives   | Tiny computation, once per iter  |
| All diagnostics       | Metrics publication, not compute |
| All ROS interface     | Message handling, not compute    |
| Map management        | I/O bound, not compute bound     |

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

**Derivative pipeline breakdown** (`align_gpu` per iteration):
- Jacobian/point Hessian: ~0.5ms (CPU, uploaded once)
- Point transformation: <0.1ms (GPU)
- Radius search: ~0.3ms (GPU, brute-force O(N×V))
- Score/gradient/Hessian: ~0.5ms (GPU)
- Reduction: ~0.1ms (GPU via CUB DeviceSegmentedReduce)
- **CPU-GPU transfers per iteration**: Upload pose (16 floats), download 43 floats only

---

## Recommendations

### High Priority

1. ~~**GPU optimization loop derivatives**~~ ✅ Complete - 1.58x speedup via `align_gpu()`
2. ~~**GPU reduction kernel**~~ ✅ Complete - CUB DeviceSegmentedReduce, downloads only 43 floats
3. ~~**GPU batch scoring (Phase 13)**~~ ✅ Complete - GPU batch scoring for MULTI_NDT_SCORE
4. ~~**Ground point filtering**~~ ✅ Complete - No-ground scoring for flat road robustness

### Medium Priority

5. **GPU batch alignment** - Benefits MULTI_NDT mode (3-5x)

### Low Priority

6. ~~Debug visualization features~~ Partially complete - Per-point score visualization done; remaining debug publishers low impact
