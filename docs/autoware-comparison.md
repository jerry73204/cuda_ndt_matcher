# NDT Feature List

Feature comparison between `cuda_ndt_matcher` and Autoware's `ndt_scan_matcher`.

**Last Updated**: 2026-01-09

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
| ✅     | On GPU                                    |
| ⚠️      | Partial/hybrid (GPU + CPU)                |
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

| Feature                     | Status | GPU | Autoware Diff                  | GPU Rationale                                                             |
|-----------------------------|--------|-----|--------------------------------|---------------------------------------------------------------------------|
| Jacobian computation        | ✅     | 🔲  | Same formulas (Magnusson 2009) | GPU kernels exist but **only used for scoring**, not optimization loop    |
| Hessian computation         | ✅     | 🔲  | Same formulas                  | Same as above                                                             |
| Angular derivatives (j_ang) | ✅     | —   | All 8 terms match              | Precomputed once per iteration, tiny                                      |
| Point Hessian (h_ang)       | ✅     | —   | All 15 terms match             | Precomputed once per iteration                                            |
| Gradient accumulation       | ✅     | 🔲  | Same algorithm                 | **Should be GPU** - main bottleneck. Blocked by CubeCL control flow bugs  |
| Hessian accumulation        | ✅     | 🔲  | Same algorithm                 | **Should be GPU** - paired with gradient                                  |

### GPU Optimization Loop Status

| Component            | Current | Target | Blocker                                  |
|----------------------|---------|--------|------------------------------------------|
| Point transformation | GPU ✅  | GPU    | -                                        |
| Radius search        | GPU ✅  | GPU    | -                                        |
| Gradient computation | CPU     | GPU    | CubeCL optimizer panics on complex loops |
| Hessian computation  | CPU     | GPU    | Same as gradient                         |
| Newton solve         | CPU     | CPU    | 6x6 too small for GPU                    |

**Potential speedup**: 2-3x per alignment if GPU derivatives enabled in optimization loop.

---

## 3. Scoring

| Feature                | Status | GPU | Autoware Diff       | GPU Rationale                    |
|------------------------|--------|-----|---------------------|----------------------------------|
| Transform probability  | ✅     | ✅  | Same formula        | Parallel per-point scoring       |
| NVTL scoring           | ✅     | ✅  | Same formula        | Parallel per-point max           |
| Convergence threshold  | ✅     | —   | Same check          | Single comparison                |
| Skip on low score      | ✅     | —   | Same behavior       | Control flow only                |
| No-ground scoring      | ❌     | -   | **Missing**         | Would use same GPU path          |
| Per-point score colors | ❌     | -   | **Missing** (debug) | CPU sufficient for visualization |

### Missing: Ground Point Filtering

**Autoware behavior**:
- Filters ground points (z < min_z + margin)
- Computes separate `no_ground_transform_probability` and `no_ground_nvtl`
- Publishes `points_aligned_no_ground`

**Why implement**: Improves robustness on flat roads where ground dominates score.

**GPU consideration**: Yes - same parallel filtering as existing z-height filter.

---

## 4. Covariance Estimation

| Feature               | Status | GPU | Autoware Diff           | GPU Rationale                                                    |
|-----------------------|--------|-----|-------------------------|------------------------------------------------------------------|
| FIXED mode            | ✅     | —   | Same                    | Returns constant matrix                                          |
| LAPLACE mode          | ✅     | —   | Same (Hessian inverse)  | 6x6 inversion - CPU faster                                       |
| MULTI_NDT mode        | ✅     | 🔲  | Same algorithm          | Runs batch alignments. **Could benefit from GPU batch pipeline** |
| MULTI_NDT_SCORE mode  | ✅     | ✅  | Same (softmax weighted) | Uses GPU NVTL batch                                              |
| Covariance rotation   | ✅     | —   | Same                    | 6x6 rotation - trivial                                           |
| Temperature parameter | ✅     | —   | Same default            | Single scalar                                                    |
| Scale factor          | ✅     | —   | Same                    | Single scalar                                                    |

### MULTI_NDT GPU Opportunity

**Current**: 6+ alignments run via Rayon parallel CPU.

**Opportunity**: Single GPU kernel with shared voxel grid, different poses.

**Potential speedup**: 3-5x for MULTI_NDT covariance mode.

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
| No-ground metrics     | ❌     | -   | **Missing**   | Requires ground filtering |

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
| `no_ground_*` publishers (2)   | ❌     | **Missing**         |
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

| Feature                   | Priority | Effort | GPU |
|---------------------------|----------|--------|-----|
| Ground point filtering    | Medium   | Low    | Yes |
| No-ground scoring metrics | Medium   | Low    | Yes |

### Debug/Visualization Gaps (no runtime impact)

| Feature                     | Priority | Effort | GPU |
|-----------------------------|----------|--------|-----|
| Per-point score colors      | Low      | Low    | No  |
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

| Component               | Notes                                           |
|-------------------------|-------------------------------------------------|
| Morton code computation | `compute_morton_codes_kernel` (voxel grid)      |
| Radix sort              | CUB DeviceRadixSort via `cuda_ffi`              |
| Segment detection       | CUB DeviceScan + DeviceSelect via `cuda_ffi`    |
| Segment statistics      | Position sums, means, covariances (voxel grid)  |
| Point transformation    | Full GPU                                        |
| Radius search (scoring) | GPU kernel                                      |
| Transform probability   | Parallel per-point                              |
| NVTL scoring            | Parallel per-point max                          |
| Sensor point filtering  | GPU if ≥10k points                              |

### Hybrid GPU/CPU (⚠️)

| Component               | GPU Part                                   | CPU Part                 |
|-------------------------|--------------------------------------------|--------------------------|
| Voxel grid construction | Morton, sort, segments, statistics (6/7)   | Eigendecomposition (1/7) |

### Should Be on GPU, Blocked (🔲)

| Component                     | Blocker                    | Speedup |
|-------------------------------|----------------------------|---------|
| Gradient in optimization loop | CubeCL optimizer bugs      | 2-3x    |
| Hessian in optimization loop  | Same                       | 2-3x    |
| Batch alignment pipeline      | Architecture change needed | 3-5x    |

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

| Operation            | CPU Only | Current (Zero-Copy) | Target (Full GPU) |
|----------------------|----------|---------------------|-------------------|
| Single alignment     | ~50ms    | ~30ms               | <20ms             |
| NVTL scoring         | ~8ms     | ~3ms                | <2ms              |
| Voxel grid build     | ~200ms   | ~50ms               | <40ms             |
| MULTI_NDT covariance | ~300ms   | ~180ms              | <100ms            |

**Voxel grid build breakdown** (100k points, zero-copy pipeline):
- Morton codes: ~1ms (GPU, CubeCL)
- Radix sort: ~2ms (GPU, CUB via cuda_ffi)
- Segment detection: ~1ms (GPU, CUB via cuda_ffi)
- Statistics accumulation: ~5ms (GPU, CubeCL)
- Finalization: ~15ms (CPU eigendecomp)
- **CPU-GPU transfers**: 2 only (upload points, download stats)

---

## Recommendations

### High Priority

1. **GPU optimization loop derivatives** - Largest performance gain (2-3x)
2. **Ground point filtering** - Functional improvement for flat roads

### Medium Priority

3. **No-ground scoring** - Pairs with ground filtering
4. **GPU batch alignment** - Benefits MULTI_NDT mode (3-5x)

### Low Priority

5. Debug visualization features - No runtime benefit
