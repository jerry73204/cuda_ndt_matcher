# NDT Feature List

Feature comparison between `cuda_ndt_matcher` and Autoware's `ndt_scan_matcher`.

**Target Version**: Autoware 1.5.0 (`autoware_core` repository)

**Last Updated**: 2026-01-25 (Updated to CUDA Graph Kernels - Phase 24)

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

| Symbol | Meaning                                           |
|--------|---------------------------------------------------|
| ✅     | On GPU (via `align_full_gpu()` or scoring path)   |
| ⚠️      | Partial/hybrid (GPU + CPU reduction)              |
| 🔲     | Should be GPU, blocked by technical issue         |
| —      | CPU by design (GPU not beneficial)                |
| -      | N/A (feature not implemented)                     |

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
| Jacobian computation        | ✅     | ✅  | Same formulas (Magnusson 2009) | Full GPU via `FullGpuPipelineV2`                                      |
| Hessian computation         | ✅     | ✅  | Same formulas                  | Full GPU via `FullGpuPipelineV2`                                      |
| Angular derivatives (j_ang) | ✅     | ✅  | All 8 terms match              | GPU kernel `compute_jacobians_kernel`                                 |
| Point Hessian (h_ang)       | ✅     | ✅  | All 15 terms match             | GPU kernel `compute_point_hessians_kernel`                            |
| Gradient accumulation       | ✅     | ✅  | Same algorithm                 | GPU kernel + CUB segmented reduce                                     |
| Hessian accumulation        | ✅     | ✅  | Same algorithm                 | GPU kernel + CUB segmented reduce                                     |

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

| Component            | CPU path (`align`) | GPU path (`align_full_gpu`) | Notes                              |
|----------------------|--------------------|-----------------------------|------------------------------------|
| Point transformation | CPU                | GPU ✅                      | -                                  |
| Radius search        | CPU (KD-tree)      | GPU ✅                      | Brute-force O(N×V) on GPU          |
| Jacobian computation | CPU                | GPU ✅                      | `compute_jacobians_kernel`         |
| Point Hessian comp.  | CPU                | GPU ✅                      | `compute_point_hessians_kernel`    |
| Gradient computation | CPU                | GPU ✅                      | Per-point kernels                  |
| Hessian computation  | CPU                | GPU ✅                 | Per-point kernels                  |
| Reduction            | N/A                | GPU ✅                 | CUB DeviceSegmentedReduce (43 out) |
| Newton solve         | CPU                | GPU ✅                 | cuSOLVER Cholesky (6×6)            |

**Full GPU path (Phase 15)**: When `NDT_USE_GPU=1` (default), the entire Newton optimization with line search runs on GPU. Per-iteration transfer is ~224 bytes (Newton solve requires f64 precision + pose download for oscillation tracking).

**GPU path debug support**: Set `enable_debug: true` in `PipelineV2Config` to collect per-iteration debug data with zero additional GPU transfer overhead (uses data already downloaded for Newton solve and oscillation tracking).

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
- Creates `FullGpuPipelineV2` once for all M alignments
- Uploads source points and voxel data once (shared across all poses)
- Runs M sequential Newton optimizations via `pipeline.optimize()`, reusing the pipeline
- Each alignment has full GPU Newton iteration with optional line search
- Falls back to CPU Rayon path if GPU fails

**Key optimization**: Voxel data upload is expensive; sharing it across M alignments
provides ~2-3× speedup over M independent `align_full_gpu()` calls.

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
| TPE optimization        | ✅     | —   | Same algorithm      | Sequential Bayesian optimization (data dependency)   |
| Particle evaluation     | ✅     | ✅  | Same                | Uses GPU NVTL batch                                  |
| Batch startup alignment | ✅     | ✅  | N/A (our extension) | **Phase 16**: Batch N startup particles via `align_batch()` |
| Monte Carlo markers     | ✅     | —   | Same visualization  | ROS message creation                                 |
| `multi_initial_pose`    | ✅     | —   | Same                | Initial offset poses for MULTI_NDT covariance        |

### GPU Parallelization Analysis

The initial pose estimation has two phases with different parallelization potential:

| Phase | Particles | Implementation | GPU Acceleration | Speedup |
|-------|-----------|----------------|------------------|---------|
| **Startup** (random sampling) | First N | Batch via `align_batch()` | ✅ Full GPU | ~6x |
| **TPE-guided** (Bayesian opt) | Remaining | Sequential | — (data dependency) | 1x |

**Why startup is parallelizable**: No dependencies between particles - all use random poses.

**Why TPE-guided is sequential**: Each particle depends on previous results to guide the search via kernel density estimation.

### GPU Batch Startup Pipeline (✅ Complete)

See `docs/roadmap/phase-16-gpu-initial-pose-pipeline.md` for implementation details.

```
Previous flow (sequential):
  FOR i = 1 to N:
    pose[i] = random()           ─┐
    result[i] = NDT_align(pose)   │ 16ms per particle
    nvtl[i] = evaluate_NVTL()    ─┘
  Total: N × 16ms = 160ms (N=10)

Current flow (batch parallel):
  poses[1..N] = random()         ─┐
  results = align_batch(poses)    │ ~25ms total (GPU batch)
  nvtls = batch_NVTL()           ─┘
  Total: ~25ms (6x speedup)
```

**Implementation** (`initial_pose.rs`):
- Startup phase samples all N poses from TPE at once
- Uses `ndt_manager.align_batch()` for GPU batch alignment
- Falls back to sequential CPU alignment on error
- Guided phase remains sequential (TPE data dependency)

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

### 8.1 Per-Iteration Debug Data

When `enable_debug: true` is set in `PipelineV2Config`, the GPU path collects detailed per-iteration debug data with **zero additional GPU transfer overhead** (reuses data already downloaded for Newton solve and oscillation tracking).

| Field                    | Description                                    | Source                          |
|--------------------------|------------------------------------------------|---------------------------------|
| `iteration`              | Iteration number (0-indexed)                   | Loop counter                    |
| `pose`                   | Pose at start of iteration [x,y,z,r,p,y]       | pose_history                    |
| `score`                  | NDT score at current pose                      | reduce_output[0]                |
| `gradient`               | Gradient vector (6 elements)                   | reduce_output[1..7]             |
| `hessian`                | Hessian matrix (6×6, row-major)                | reduce_output[7..43]            |
| `newton_step`            | Newton step before line search                 | cuSOLVER output                 |
| `newton_step_norm`       | Norm of Newton step                            | Computed from newton_step       |
| `step_length`            | Step size from line search (or 1.0)            | Line search result              |
| `direction_reversed`     | Whether Newton step was reversed (not ascent)  | gradient · delta ≤ 0            |
| `directional_derivative` | Gradient · step_direction                      | Computed                        |
| `pose_after`             | Pose after applying step                       | pose_history[i+1]               |
| `used_line_search`       | Whether line search was used                   | Config flag                     |

**Usage** (in `cuda_ndt_matcher` or tests):
```rust
let config = PipelineV2Config {
    enable_debug: true,
    ..Default::default()
};
let mut pipeline = FullGpuPipelineV2::with_config(max_points, max_voxels, config)?;

let result = pipeline.optimize(&initial_pose, max_iterations, epsilon)?;

if let Some(debug_vec) = &result.iterations_debug {
    for iter in debug_vec {
        println!("iter={} score={:.4} step={:.6}", iter.iteration, iter.score, iter.step_length);
    }
}
```

**Output format**: The debug data uses the same `IterationDebug` struct as the CPU path (`optimization/debug.rs`), enabling direct comparison with Autoware's debug output.

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

## 12. Test Coverage

**Last Updated**: 2026-01-13 | **Total**: 422 tests (419 pass, 3 ignored)

### 1. Core NDT Algorithm

| Feature                       | Tests | Test Names                                                                                                                          |
|-------------------------------|-------|-------------------------------------------------------------------------------------------------------------------------------------|
| Newton-Raphson optimization   | 10    | `test_newton_step_*`, `test_align_identity`, `test_align_with_translation`                                                          |
| More-Thuente line search      | 6     | `test_simple_line_search`, `test_quadratic_minimization`, `test_auxiliary_psi`, `test_armijo_condition`, `test_curvature_condition` |
| Voxel grid construction       | 15    | `test_build_voxel_grid_*`, `test_voxel_grid_from_*`, `test_gpu_voxel_grid_construction`                                             |
| Gaussian covariance per voxel | 8     | `test_covariance_symmetry`, `test_regularize_covariance`, `test_autoware_voxel_construction`, `test_compare_covariance_computation` |
| Multi-voxel radius search     | 5     | `test_radius_search_*`, `test_3d_radius_search`, `test_within_with_distances`                                                       |
| Convergence detection         | 3     | `test_check_convergence`, `test_convergence_status`, `test_convergence_kernel_*`                                                    |
| Oscillation detection         | 10    | `test_oscillation_*`, `test_no_oscillation_*`, `test_pipeline_v2_oscillation_tracking`                                              |
| Step size clamping            | 2     | `test_apply_pose_delta_*`, `test_generate_candidates_with_clamping`                                                                 |
| KDTREE search                 | 5     | `test_single_voxel`, `test_multiple_voxels_radius_search`, `test_3d_radius_search`                                                  |

### 1.2 Voxel Grid Pipeline

| Stage                   | Tests | Test Names                                                                                        |
|-------------------------|-------|---------------------------------------------------------------------------------------------------|
| Morton code computation | 3     | `test_morton_codes_valid`, `test_compute_morton_codes_cpu`, `test_morton_encode_decode_roundtrip` |
| Radix sort              | 8     | `test_sort_*`, `test_radix_sort_gpu`, `test_radix_sort_preserves_data`                            |
| Segment detection       | 10    | `test_detect_segments_*`, `test_segment_detection_*`, `test_gpu_segment_detection`                |
| Statistics accumulation | 6     | `test_voxel_sums_*`, `test_means_from_sums`, `test_full_statistics_pipeline`                      |
| GPU vs CPU consistency  | 6     | `test_cpu_gpu_*_consistency`, `test_gpu_zero_copy_vs_*`                                           |

### 2. Derivative Computation

| Feature                     | Tests | Test Names                                                                                                    |
|-----------------------------|-------|---------------------------------------------------------------------------------------------------------------|
| Jacobian computation        | 6     | `test_point_jacobians_identity`, `test_gpu_jacobians_match_cpu`, `test_jacobians_match_angular_derivatives`   |
| Hessian computation         | 4     | `test_gpu_point_hessians_match_cpu`, `test_point_hessians_match_angular_derivatives`, `test_hessian_symmetry` |
| Angular derivatives (j_ang) | 4     | `test_zero_angles`, `test_point_gradient_terms`, `test_small_angles_approximation`                            |
| Point Hessian (h_ang)       | 3     | `test_point_hessian_terms`, `test_hessian_not_computed_when_disabled`                                         |
| Gradient accumulation       | 3     | `test_gradient_finite_difference`, `test_derivative_result_accumulate`                                        |
| CPU vs GPU derivatives      | 4     | `test_cpu_vs_gpu_derivatives`, `test_cpu_vs_gpu_single_point_single_voxel`                                    |

### 3. Scoring

| Feature                | Tests | Test Names                                                                       |
|------------------------|-------|----------------------------------------------------------------------------------|
| Transform probability  | 10    | `test_transform_probability_*`, `test_gpu_cpu_transform_probability_consistency` |
| NVTL scoring           | 13    | `test_nvtl_*`, `test_gpu_cpu_nvtl_consistency`                                   |
| Per-point score colors | 3     | `test_score_to_color_range`, `test_ndt_score_to_color`, `test_color_packing`     |
| GPU scoring pipeline   | 6     | `test_gpu_scoring_*`, `test_gpu_vs_cpu_scoring`                                  |

### 4. Covariance Estimation

| Feature              | Tests | Test Names                                                |
|----------------------|-------|-----------------------------------------------------------|
| FIXED mode           | 1     | `test_adjust_diagonal`                                    |
| LAPLACE mode         | 1     | `test_hessian_returned`                                   |
| MULTI_NDT mode       | 3     | `test_sample_covariance`, `test_weighted_covariance_*`    |
| MULTI_NDT_SCORE mode | 2     | `test_softmax_weights`, `test_gpu_scoring_multiple_poses` |
| Offset pose proposal | 2     | `test_propose_offset_poses_*`                             |

### 5. Initial Pose Estimation

| Feature                    | Tests | Test Names                                                                            |
|----------------------------|-------|---------------------------------------------------------------------------------------|
| SmartPoseBuffer            | 10    | `test_push_back_*`, `test_interpolate_*`, `test_pop_old`, `test_clear`                |
| Pose timeout validation    | 1     | `test_time_validation_rejects_stale_poses`                                            |
| Distance tolerance         | 1     | `test_distance_validation_rejects_position_jumps`                                     |
| TPE optimization           | 4     | `test_tpe_startup_phase`, `test_tpe_guided_phase`, `test_log_sum_exp*`                |
| Particle evaluation        | 2     | `test_select_best_particle`, `test_select_best_empty`                                 |
| Batch startup alignment    | 3     | `test_batch_kernels_compile`, `test_evaluate_batch_empty`*, `test_pipeline_creation`* |
| Quaternion/pose conversion | 2     | `test_quaternion_rpy_roundtrip`, `test_input_pose_roundtrip`                          |

*\* = requires CUDA GPU (ignored on CPU-only)*

### 6. Map Management

| Feature                 | Tests | Test Names                                                                             |
|-------------------------|-------|----------------------------------------------------------------------------------------|
| Dynamic map loading     | 2     | `test_load_full_map`, `test_check_and_update`                                          |
| Tile-based updates      | 2     | `test_add_remove_tiles`, `test_get_stats`                                              |
| Position-based trigger  | 4     | `test_should_update_*`                                                                 |
| Map radius filtering    | 1     | `test_map_radius_filtering`                                                            |
| Dual-NDT (non-blocking) | 3     | `test_dual_ndt_manager_creation`, `test_blocking_set_target`, `test_background_update` |

### 7. Regularization

| Feature               | Tests | Test Names                                                                         |
|-----------------------|-------|------------------------------------------------------------------------------------|
| Regularization buffer | 1     | (uses SmartPoseBuffer tests)                                                       |
| Scale factor          | 1     | `test_regularization_with_offset`                                                  |
| Enable/disable flag   | 2     | `test_regularization_disabled`, `test_pipeline_v2_regularization_disabled`         |
| Gradient penalty      | 2     | `test_regularization_at_reference`, `test_regularization_with_offset`              |
| Hessian contribution  | 1     | `test_regularization_with_yaw`                                                     |
| GPU regularization    | 2     | `test_pipeline_v2_with_regularization`, `test_pipeline_v2_regularization_disabled` |

### 8. Diagnostics

| Feature                   | Tests | Test Names                                              |
|---------------------------|-------|---------------------------------------------------------|
| Execution time            | 1     | `test_execution_timer`                                  |
| Diagnostic levels         | 2     | `test_diagnostic_category`, `test_level_only_increases` |
| Scan matching diagnostics | 1     | `test_scan_matching_diagnostics`                        |

### 9. Point Cloud Processing

| Feature                | Tests | Test Names                                              |
|------------------------|-------|---------------------------------------------------------|
| PointCloud2 conversion | 2     | `test_from_pointcloud2`, `test_empty_pointcloud`        |
| Distance filtering     | 4     | `test_filter_distance*`                                 |
| Z-height filtering     | 2     | `test_filter_z_height`, `test_filter_z`                 |
| Voxel downsampling     | 6     | `test_voxel_downsample_*`                               |
| Combined filtering     | 2     | `test_filter_combined`, `test_filter_with_downsampling` |

### 10. GPU Pipeline Integration

| Feature              | Tests | Test Names                                                                                                       |
|----------------------|-------|------------------------------------------------------------------------------------------------------------------|
| Full GPU Newton      | 3     | `test_align_full_gpu_*`                                                                                          |
| GPU Newton solver    | 6     | `test_solve_*` (gpu_newton.rs)                                                                                   |
| Pipeline creation    | 2     | `test_pipeline_v2_creation`, `test_pipeline_v2_with_config`                                                      |
| Line search pipeline | 2     | `test_pipeline_v2_with_line_search`, `test_pipeline_v2_no_line_search`                                           |
| Debug collection     | 2     | `test_pipeline_v2_debug_collection`, `test_pipeline_v2_debug_disabled`                                           |
| GPU kernels          | 8     | `test_transform_kernel_*`, `test_dot_product_kernel*`, `test_convergence_kernel_*`, `test_more_thuente_kernel_*` |

### 11. CUB FFI Bindings

| Feature           | Tests | Test Names                                          |
|-------------------|-------|-----------------------------------------------------|
| Radix sort        | 8     | `test_sort_*` (radix_sort.rs)                       |
| Segment detection | 6     | `test_detect_segments_*` (segment_detect.rs)        |
| Segmented reduce  | 7     | `test_sum_*` (segmented_reduce.rs)                  |
| Batched Cholesky  | 2     | `test_solver_creation`, `test_workspace_size_query` |

### Autoware Reference Verification

Tests that verify correctness against Autoware's pclomp implementation:

| Test                                            | Verification                             |
|-------------------------------------------------|------------------------------------------|
| `test_autoware_voxel_construction`              | Voxel grid output matches pclomp         |
| `test_compare_covariance_computation`           | Covariance formula matches pclomp        |
| `test_compare_inverse_covariance`               | Inverse covariance matches pclomp        |
| `test_compare_mean_computation`                 | Voxel means match pclomp                 |
| `test_eigenvalue_regularization`                | Eigenvalue regularization matches pclomp |
| `test_jacobians_match_angular_derivatives`      | All 8 Jacobian terms match reference     |
| `test_point_hessians_match_angular_derivatives` | All 15 Hessian terms match reference     |

### Running Tests

```bash
just test                    # Run all tests
just test -- test_name       # Run specific test
just test -- --nocapture     # Run with output
```

---

## Summary: What's Missing

### Target Version Note

This comparison targets **Autoware 1.5.0**, where `ndt_scan_matcher` was moved from `autoware_universe` to `autoware_core`. The reference implementation is at:
- `tests/comparison/autoware_core/localization/autoware_ndt_scan_matcher/`

### Functional Gaps

All functional features are implemented. Convergence gating and diagnostic keys match Autoware 1.5.0 behavior.

### Debug/Visualization Gaps (no runtime impact)

None - all debug publishers are implemented.

### Intentionally Not Implemented

| Feature               | Reason                                    |
|-----------------------|-------------------------------------------|
| DIRECT search methods | KDTREE is better, recommended by Autoware |
| OpenMP thread config  | Rayon handles automatically               |

### Autoware 1.5.0 Specific Notes

| Feature                 | Autoware 1.5.0               | CUDA NDT                   | Notes                                      |
|-------------------------|------------------------------|----------------------------|--------------------------------------------|
| Package location        | `autoware_core`              | N/A                        | Moved from `autoware_universe` in 0.48.0   |
| `search_method` param   | KDTREE (default)             | KDTREE only                | DIRECT methods removed from default config |
| `use_line_search` param | Available (default: false)   | Available (default: false) | Same behavior                              |
| Diagnostic interface    | `autoware_utils_diagnostics` | Custom impl                | Same output format                         |

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
| Batch alignment (MULTI_NDT)     | `FullGpuPipelineV2` - shared voxel data           |
| Batch startup (Initial Pose)    | `align_batch()` - Phase 16, ~6x speedup           |
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
| Batch startup pipeline        | ✅ Working    | ✅ Integrated     | Phase 16, ~6x speedup    |

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

| Operation            | CPU Only | GPU Path (`align_full_gpu`) | Speedup |
|----------------------|----------|-----------------------------|---------|
| Single alignment     | ~4.3ms   | ~2.7ms                      | 1.58x   |
| NVTL scoring         | ~8ms     | ~3ms                        | 2.7x    |
| Voxel grid build     | ~200ms   | ~50ms                       | 4x      |
| MULTI_NDT covariance | ~300ms   | ~180ms                      | 1.7x    |

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

| Pipeline       | Purpose                                | Location                                    |
|----------------|----------------------------------------|---------------------------------------------|
| **Voxel Grid** | Build NDT map from points              | `voxel_grid/gpu/pipeline.rs`                |
| **Derivative** | Full GPU Newton optimization           | `optimization/full_gpu_pipeline_v2.rs`      |
| **Scoring**    | Batch NVTL evaluation for covariance   | `scoring/pipeline.rs`                       |

### 1. Voxel Grid Construction Pipeline

```
Source points [N×3]
    ↓ upload (once)
GPU: Morton codes → Radix sort → Segment detect → Statistics
    ↓ download (once)
Results: means [V×3], inv_covariances [V×9]
```

**Zero-copy status**: Complete - true zero-copy from upload to download

### 2. Derivative Pipeline (`FullGpuPipelineV2`)

**Architecture**: CUDA Graph Kernels (Phase 24)

The optimization pipeline uses 5 separate CUDA kernels instead of a single cooperative kernel.
This approach works on all GPUs, including those with limited SM count (e.g., Jetson Orin),
and eliminates the cooperative groups dependency.

**Kernel architecture** (`ndt_graph_kernels.cu`):

| Kernel | Purpose | Grid/Block |
|--------|---------|------------|
| K1: Init | Initialize state from initial pose | 1×32 threads |
| K2: Compute | Per-point score/gradient/Hessian + block reduction | ceil(N/256)×256 threads |
| K3: Solve | Newton solve (Jacobi SVD), regularization, direction check | 1×32 threads |
| K4: LineSearch | Parallel evaluation of K step size candidates | ceil(N/256)×256 threads |
| K5: Update | Apply step, oscillation detection, convergence check | 1×32 threads |

**Buffer layout** (persistent across iterations):

| Buffer | Size (floats) | Purpose |
|--------|---------------|---------|
| state_buffer | 102 | Pose, delta, oscillation tracking, alpha candidates |
| reduce_buffer | 29 | Score, gradient, Hessian upper triangle, correspondences |
| ls_buffer | 68 | Line search candidates, phi_0/dphi_0, per-candidate scores/grads |
| output_buffer | 48 | Final pose, iterations, convergence, Hessian, statistics |
| debug_buffer | max_iter×50 | Optional per-iteration debug data |

**Iteration flow** (~224 bytes per-iteration transfer):
```
Once per alignment:
  Upload: source_points [N×3], voxel_means [V×3], voxel_inv_covs [V×9]
  K1: Init state from initial_pose

Per iteration:
  K2: Compute score/gradient/Hessian (parallel per-point, atomic reduction)
  K3: Solve Newton system (Jacobi SVD), apply regularization, setup line search
  K4: Line search evaluation (if enabled, parallel per-point for K=8 candidates)
  K5: Apply best step, detect oscillation, check convergence
  Host: cudaDeviceSynchronize + check convergence flag

After convergence:
  Download: output_buffer (192 bytes)
```

**Advantages over cooperative kernel**:
- Works on Jetson Orin (limited SM count prevents cooperative launch)
- No `cudaLaunchCooperativeKernel` requirement
- Explicit synchronization between kernels for easier debugging
- Same algorithmic behavior as the original persistent kernel

**Note**: The pipeline still uses CubeCL for some helper operations (point transformation,
voxel grid construction). These require compatible nvrtc at runtime.

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

**Target**: Autoware 1.5.0 (`autoware_core/localization/autoware_ndt_scan_matcher`)

**Feature parity**: Complete. The CUDA implementation is a drop-in replacement for Autoware 1.5.0's `ndt_scan_matcher`.

**GPU acceleration**: All compute-heavy operations run on GPU:
- Voxel grid construction (zero-copy pipeline)
- NDT alignment with line search (full GPU Newton via `FullGpuPipelineV2`)
- GNSS regularization (`apply_regularization_kernel`)
- Batch scoring (`GpuScoringPipeline`)
- Batch alignment (`align_batch_gpu()`)

**Full GPU Newton with Line Search (Phase 24)**: The default path (`NDT_USE_GPU=1`) runs the entire Newton optimization on GPU using CUDA graph kernels:
- 5 separate kernels (K1-K5) replace the cooperative groups persistent kernel
- Jacobi SVD solver for Newton direction (handles indefinite Hessians)
- More-Thuente line search with K=8 batched candidates (parallel evaluation)
- GNSS regularization in K3 (solve) and K5 (line search evaluation)
- Oscillation detection via pose history in state_buffer
- Per-iteration debug data collection (when `enable_debug: true`)
- Works on Jetson Orin (no cooperative launch requirement)

**Behavioral compatibility**: Convergence gating matches Autoware:
- Max iterations check (gates if hit max iterations)
- Oscillation count > 10 (gates if oscillating)
- Score threshold (gates if below threshold)

---

## Phase 24 Migration Notes

**Date**: 2026-01-25

### Bugs Found and Fixed

During the migration from cooperative groups persistent kernel to CUDA graph kernels,
two bugs were discovered and fixed:

#### 1. Output Buffer Ordering Bug (Fixed)

**Location**: `ndt_graph_kernels.cu`, K5 update kernel

**Bug**: The reduce buffer was cleared BEFORE writing final score and correspondences to
the output buffer, resulting in zero values.

**Impact**: Final score and correspondence count were always 0 in the output, affecting
diagnostics and covariance estimation quality metrics.

**Fix**: Moved output buffer write BEFORE reduce buffer clear:
```cpp
// BEFORE (wrong):
for (int i = 0; i < ReduceOffset::TOTAL_SIZE; i++)
    reduce_buffer[i] = 0.0f;  // Clears score!
// ...later...
output_buffer[OutputOffset::FINAL_SCORE] = reduce_buffer[ReduceOffset::SCORE];  // Always 0!

// AFTER (correct):
if (converged || at_max_iterations) {
    output_buffer[OutputOffset::FINAL_SCORE] = reduce_buffer[ReduceOffset::SCORE];  // Correct value
    // ...
}
for (int i = 0; i < ReduceOffset::TOTAL_SIZE; i++)
    reduce_buffer[i] = 0.0f;  // Now safe to clear
```

#### 2. Regularization Gradient in Line Search (Fixed)

**Location**: `ndt_graph_kernels.cu`, K5 update kernel, Wolfe condition check

**Bug**: When computing the directional derivative (dphi_k) for the curvature condition,
the regularization gradient adjustment was not applied. The old persistent kernel
applied regularization to the gradient before computing dphi_k.

**Impact**: Curvature condition in Strong Wolfe line search was computed incorrectly
when GNSS regularization was enabled, potentially selecting suboptimal step sizes.

**Fix**: Apply regularization gradient adjustment before computing dphi_k:
```cpp
// Copy gradient and apply regularization adjustment
float grad[6];
for (int i = 0; i < 6; i++)
    grad[i] = ls_buffer[LineSearchOffset::CAND_GRADS + k * 6 + i];

if (config.reg_enabled) {
    // Apply gradient adjustment (same as in K3 solve)
    grad[0] += config.reg_scale * corr_k * 2.0f * cos_yaw * longitudinal;
    grad[1] += config.reg_scale * corr_k * 2.0f * sin_yaw * longitudinal;
}

// Now compute dphi_k with correct gradient
float dphi_k = 0.0f;
for (int i = 0; i < 6; i++)
    dphi_k += grad[i] * state_buffer[StateOffset::DELTA + i];
```

### Environment Requirements

The CUDA graph kernels are compiled with nvcc at build time and work correctly.
However, the codebase also uses CubeCL for some helper operations (point transformation,
voxel grid construction), which require nvrtc at runtime.

**Required**: CUDA 12.4 for both build and runtime. Ensure:
- `CUDA_PATH=/usr/local/cuda-12.4`
- `PATH` includes `/usr/local/cuda-12.4/bin`
- `LD_LIBRARY_PATH` includes `/usr/local/cuda-12.4/lib64`

CUDA 13+ has incompatible nvrtc that causes CubeCL JIT compilation to fail with
"invalid value for --gpu-architecture (-arch)" errors.

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
