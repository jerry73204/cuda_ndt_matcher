# Debug Feature Comparison: CUDA NDT vs Autoware NDT

This document compares debug and monitoring features between our CUDA NDT implementation and Autoware's `autoware_ndt_scan_matcher`.

**Last Updated**: 2026-01-21
**Status**: Phase 24 (Debug Features) Complete

---

## Summary

| Category                     | CUDA NDT        | Autoware NDT    | Status                |
|------------------------------|-----------------|-----------------|-----------------------|
| **Per-Iteration Debug Data** | ✅ Full         | ✅ Full         | 🟢 **Parity**         |
| **Debug Topics (ROS)**       | ⚠️ Partial       | ✅ Full         | 🟡 **Gaps Exist**     |
| **Visualization Markers**    | ✅ Full         | ✅ Full         | 🟢 **Parity**         |
| **Diagnostics**              | ⚠️ Partial       | ✅ Full         | 🟡 **Gaps Exist**     |
| **Debug Output Files**       | ✅ Full         | ❌ None         | 🟢 **CUDA Advantage** |
| **Feature Gating**           | ✅ Compile-time | ❌ Runtime only | 🟢 **CUDA Advantage** |

---

## 1. Per-Iteration Debug Data

### 1.1 Core Optimization Metrics

| Feature                                            | CUDA NDT | Autoware NDT | Notes                                                                           |
|----------------------------------------------------|----------|--------------|---------------------------------------------------------------------------------|
| **Pose at each iteration**                         | ✅       | ✅           | CUDA: 6-DOF array + 4x4 matrix<br>Autoware: transformation_array (4x4 matrices) |
| **Score (Transform Probability)**                  | ✅       | ✅           | Both track per-iteration TP                                                     |
| **NVTL (Nearest Voxel Transformation Likelihood)** | ✅       | ✅           | Both compute per-iteration NVTL                                                 |
| **Gradient (6-DOF)**                               | ✅       | ❌           | CUDA stores full gradient vector                                                |
| **Hessian (6x6)**                                  | ✅       | ❌           | CUDA stores full Hessian matrix                                                 |
| **Newton step (6-DOF)**                            | ✅       | ❌           | CUDA tracks step direction                                                      |
| **Step length (alpha)**                            | ✅       | ❌           | CUDA tracks line search result                                                  |
| **Direction reversal flag**                        | ✅       | ❌           | CUDA detects oscillation per iteration                                          |
| **Number of correspondences**                      | ✅       | ❌           | CUDA tracks valid point matches                                                 |

**Implementation:**
- **CUDA NDT**: `src/ndt_cuda/src/optimization/debug.rs` - `IterationDebug` struct
- **Autoware NDT**: `ndt_result.transform_probability_array`, `ndt_result.nearest_voxel_transformation_likelihood_array`

**Status**: 🟢 **CUDA has more detailed per-iteration data**

---

### 1.2 Debug Arrays

| Feature | CUDA NDT | Autoware NDT |
|---------|----------|--------------|
| **Transform probability array** | ✅ | ✅ |
| **NVTL array** | ✅ | ✅ |
| **Transformation array (poses)** | ✅ | ✅ |
| **Array size validation** | ✅ | ✅ |

**Status**: 🟢 **Parity Achieved**

---

## 2. ROS Debug Topics

### 2.1 Performance Metrics

| Topic                                                      | CUDA NDT | Autoware NDT | Notes                              |
|------------------------------------------------------------|----------|--------------|------------------------------------|
| `exe_time_ms` (Float32Stamped)                             | ❌       | ✅           | Autoware publishes execution time  |
| `transform_probability` (Float32Stamped)                   | ❌       | ✅           | Autoware publishes TP score        |
| `nearest_voxel_transformation_likelihood` (Float32Stamped) | ❌       | ✅           | Autoware publishes NVTL score      |
| `iteration_num` (Int32Stamped)                             | ❌       | ✅           | Autoware publishes iteration count |

**Status**: 🔴 **Missing in CUDA NDT**

**Impact**: Medium - useful for real-time monitoring but covered by diagnostics

---

### 2.2 No-Ground Variants

| Topic                                               | CUDA NDT | Autoware NDT | Notes                         |
|-----------------------------------------------------|----------|--------------|-------------------------------|
| `no_ground_transform_probability`                   | ❌       | ✅           | Score for ground-removed scan |
| `no_ground_nearest_voxel_transformation_likelihood` | ❌       | ✅           | NVTL for ground-removed scan  |

**Status**: 🔴 **Missing in CUDA NDT**

**Impact**: Low - ground removal feature not yet implemented in CUDA NDT

**Related**: Autoware param `score_estimation.no_ground_points.enable` (default: false)

---

### 2.3 Pose Relationships

| Topic                             | CUDA NDT | Autoware NDT | Notes                                    |
|-----------------------------------|----------|--------------|------------------------------------------|
| `initial_to_result_relative_pose` | ❌       | ✅           | Relative pose between initial and result |
| `initial_to_result_distance`      | ❌       | ✅           | Distance metric [m]                      |
| `initial_to_result_distance_old`  | ❌       | ✅           | Distance to older interpolated pose      |
| `initial_to_result_distance_new`  | ❌       | ✅           | Distance to newer interpolated pose      |

**Status**: 🔴 **Missing in CUDA NDT**

**Impact**: Low - useful for diagnostics validation but not critical

---

### 2.4 Point Cloud Data

| Topic                          | CUDA NDT | Autoware NDT | Notes                                         |
|--------------------------------|----------|--------------|-----------------------------------------------|
| `points_aligned`               | ❌       | ✅           | Aligned sensor points in map frame            |
| `points_aligned_no_ground`     | ❌       | ✅           | Ground-removed aligned points                 |
| `voxel_score_points`           | ✅       | ✅           | RGB-colored point cloud with per-point scores |
| `initial_pose_with_covariance` | ❌       | ✅           | Initial pose used for alignment               |

**Status**: 🟡 **Partial** - CUDA has voxel score visualization, missing aligned point clouds

**Implementation**:
- **CUDA NDT**: `src/cuda_ndt_matcher/src/main.rs:1282-1303` - Feature-gated with `debug-markers`
- **Autoware NDT**: `src/ndt_scan_matcher_core.cpp` - `visualize_point_score()` function

---

### 2.5 Multi-NDT Pose Arrays

| Topic                | CUDA NDT | Autoware NDT | Notes                                               |
|----------------------|----------|--------------|-----------------------------------------------------|
| `multi_ndt_pose`     | ✅       | ✅           | Multiple estimated poses from covariance estimation |
| `multi_initial_pose` | ✅       | ✅           | Initial poses for real-time covariance estimation   |

**Status**: 🟢 **Parity Achieved**

**Implementation**:
- **CUDA NDT**: `src/cuda_ndt_matcher/src/dual_ndt_manager.rs` - Monte Carlo initial pose estimation
- **Autoware NDT**: `src/particle.cpp` - Particle-based pose estimation

---

### 2.6 Dynamic Map Loading

| Topic                         | CUDA NDT | Autoware NDT | Notes                                            |
|-------------------------------|----------|--------------|--------------------------------------------------|
| `debug/loaded_pointcloud_map` | ❌       | ✅           | Currently loaded voxel map (transient local QoS) |

**Status**: 🔴 **Missing in CUDA NDT**

**Impact**: Low - CUDA NDT uses static map loading

**Note**: Autoware publishes map updates when dynamic map loading is enabled

---

## 3. Visualization Markers

### 3.1 NDT Markers

| Feature                         | CUDA NDT | Autoware NDT | Notes                                                              |
|---------------------------------|----------|--------------|--------------------------------------------------------------------|
| **Pose history markers**        | ✅       | ✅           | Iteration-by-iteration pose trajectory                             |
| **Color gradient by iteration** | ✅       | ✅           | CUDA: blue→cyan→green<br>Autoware: `exchange_color_crc()` encoding |
| **Arrow markers**               | ✅       | ✅           | Both use arrow markers for pose                                    |
| **MarkerArray topic**           | ✅       | ✅           | Both publish to `ndt_marker`                                       |

**Status**: 🟢 **Parity Achieved**

**Implementation**:
- **CUDA NDT**: `src/cuda_ndt_matcher/src/main.rs:1243-1261` - Feature-gated with `debug-markers`
- **Autoware NDT**: `src/ndt_scan_matcher_core.cpp` - Iteration markers with color encoding

---

### 3.2 Monte Carlo / Particle Markers

| Feature                   | CUDA NDT | Autoware NDT | Notes                                                                                          |
|---------------------------|----------|--------------|------------------------------------------------------------------------------------------------|
| **Particle pose markers** | ✅       | ✅           | Both visualize initial pose candidates                                                         |
| **Color by score**        | ✅       | ✅           | CUDA: green→yellow→red gradient<br>Autoware: `initial_pose_transform_probability_color_marker` |
| **Color by iteration**    | ❌       | ✅           | Autoware has `initial_pose_iteration_color_marker` namespace                                   |
| **Color by index**        | ❌       | ✅           | Autoware has `initial_pose_index_color_marker` namespace                                       |
| **Result pose markers**   | ❌       | ✅           | Autoware publishes result pose with multiple color schemes                                     |
| **Marker lifetime**       | ✅       | ✅           | CUDA: 1.0s, Autoware: 10.0s                                                                    |

**Status**: 🟡 **Partial** - CUDA has basic particle markers, missing multiple color schemes

**Implementation**:
- **CUDA NDT**: `src/cuda_ndt_matcher/src/initial_pose.rs:293-364`
- **Autoware NDT**: `src/particle.cpp` - Comprehensive particle marker visualization with multiple namespaces

---

## 4. Diagnostics

### 4.1 Scan Matching Diagnostics

| Diagnostic Key                  | CUDA NDT   | Autoware NDT | Notes                                                             |
|---------------------------------|------------|--------------|-------------------------------------------------------------------|
| **Input validation**            | ✅ Partial | ✅ Full      | CUDA checks sensor points size<br>Autoware: size + timeout checks |
| **Transform success status**    | ✅         | ✅           | Both track alignment success                                      |
| **Iteration count**             | ✅         | ✅           | Both report iteration_num                                         |
| **Transform probability**       | ✅         | ✅           | Both report TP score                                              |
| **NVTL**                        | ✅         | ✅           | Both report NVTL score                                            |
| **Transform probability array** | ❌         | ✅           | Autoware includes full array in diagnostics                       |
| **NVTL array**                  | ❌         | ✅           | Autoware includes full array in diagnostics                       |
| **Transform probability diff**  | ❌         | ✅           | Autoware tracks score improvement                                 |
| **NVTL diff**                   | ❌         | ✅           | Autoware tracks NVTL improvement                                  |
| **Convergence validation**      | ✅         | ✅           | Both validate against thresholds                                  |
| **Initial to result distance**  | ❌         | ✅           | Autoware tracks pose displacement                                 |
| **Execution time monitoring**   | ✅         | ✅           | Both track alignment time                                         |
| **Oscillation detection**       | ✅         | ✅           | Both detect pose trajectory inversions                            |

**Status**: 🟡 **Partial** - CUDA has core diagnostics, missing detailed arrays and metrics

**Implementation**:
- **CUDA NDT**: `src/cuda_ndt_matcher/src/main.rs` - `update_diagnostics()` function
- **Autoware NDT**: `src/diagnostics.cpp` - `DiagnosticsInterface::update_scan_matching_status()`

---

### 4.2 Subscriber Diagnostics

| Diagnostic                                | CUDA NDT | Autoware NDT | Notes                                                   |
|-------------------------------------------|----------|--------------|---------------------------------------------------------|
| **Initial pose subscriber status**        | ❌       | ✅           | Autoware tracks subscription timestamps, frame ID       |
| **Regularization pose subscriber status** | ❌       | ✅           | Autoware tracks regularization subscription             |
| **Map update status**                     | ❌       | ✅           | Autoware tracks map loading state, distance, maps count |
| **NDT align service status**              | ❌       | ✅           | Autoware tracks service call timestamps                 |
| **Trigger node service status**           | ✅       | ✅           | Both track activation service                           |

**Status**: 🔴 **Missing Most Subscriber Diagnostics**

**Impact**: Low - primarily for Autoware's dynamic features not yet in CUDA NDT

---

## 5. Debug Output Files

### 5.1 JSONL Debug Output

| Feature                        | CUDA NDT | Autoware NDT | Notes                                      |
|--------------------------------|----------|--------------|--------------------------------------------|
| **Per-alignment JSONL output** | ✅       | ❌           | CUDA writes structured debug records       |
| **Per-iteration data**         | ✅       | ❌           | CUDA includes full optimization trajectory |
| **Initial pose records**       | ✅       | ❌           | CUDA logs initial pose requests            |
| **Run start markers**          | ✅       | ❌           | CUDA logs session boundaries               |
| **Configurable output path**   | ✅       | ❌           | `NDT_DEBUG_FILE` env var                   |

**Status**: 🟢 **CUDA Advantage** - Autoware has no file-based debug output

**Implementation**: `src/cuda_ndt_matcher/src/main.rs:564-582, 927-937` - Feature-gated with `debug-output`

---

### 5.2 Voxel Grid Dumps

| Feature                      | CUDA NDT | Autoware NDT | Notes                            |
|------------------------------|----------|--------------|----------------------------------|
| **Voxel grid JSON dump**     | ✅       | ❌           | CUDA dumps full voxel statistics |
| **Per-voxel covariance**     | ✅       | ❌           | Mean, cov, inv_cov, point count  |
| **Configurable output path** | ✅       | ❌           | `NDT_DUMP_VOXELS_FILE` env var   |

**Status**: 🟢 **CUDA Advantage** - Used for validation against Autoware

**Implementation**: `src/cuda_ndt_matcher/src/ndt_manager.rs:24-94` - Feature-gated with `debug-voxels`

---

## 6. Feature Gating & Overhead

### 6.1 Debug Feature Control

| Feature                         | CUDA NDT     | Autoware NDT | Notes                                                               |
|---------------------------------|--------------|--------------|---------------------------------------------------------------------|
| **Compile-time feature flags**  | ✅           | ❌           | CUDA uses `#[cfg(feature = "...")]`                                 |
| **Zero-overhead when disabled** | ✅           | ⚠️            | CUDA: code not compiled<br>Autoware: runtime checks only            |
| **Selective debug features**    | ✅           | ⚠️            | CUDA: 7 independent features<br>Autoware: topic subscription checks |
| **Production binary size**      | ✅ Optimized | ⚠️ Larger     | CUDA eliminates debug code completely                               |

**CUDA NDT Features**:
```toml
# ndt_cuda/Cargo.toml
profiling              # Timing instrumentation
debug-iteration        # Per-iteration data (pose, score, gradient, Hessian)
debug-cov              # GPU vs CPU covariance comparison
debug-vpp              # Voxel-per-point distribution tracking
debug                  # Meta-feature: all ndt_cuda debug

# cuda_ndt_matcher/Cargo.toml
debug-output           # JSONL file output
debug-voxels           # Voxel grid JSON dump
debug-markers          # Pose markers + voxel score visualization
debug                  # Meta-feature: all debug features
```

**Status**: 🟢 **CUDA Advantage** - Compile-time guarantees with zero overhead

**Documentation**: `docs/debug-infrastructure.md`

---

## 7. Detailed Feature Breakdown by Category

### Category 1: Per-Iteration Optimization Data

| Data Field           | CUDA NDT | Autoware NDT |
|----------------------|----------|--------------|
| Pose (6-DOF)         | ✅       | ✅           |
| Pose (4x4 matrix)    | ✅       | ✅           |
| Score (TP)           | ✅       | ✅           |
| Score (NVTL)         | ✅       | ✅           |
| Gradient (6-DOF)     | ✅       | ❌           |
| Hessian (6x6)        | ✅       | ❌           |
| Newton step          | ✅       | ❌           |
| Step length (alpha)  | ✅       | ❌           |
| Direction reversal   | ✅       | ❌           |
| Correspondence count | ✅       | ❌           |

**CUDA Files**: `ndt_cuda/src/optimization/debug.rs`, `ndt_cuda/src/optimization/solver.rs`

**Autoware Files**: `ndt_scan_matcher_core.cpp` (ndt_result arrays)

---

### Category 2: JSONL File Output

| Feature              | CUDA NDT | Autoware NDT |
|----------------------|----------|--------------|
| Alignment records    | ✅       | ❌           |
| Initial pose records | ✅       | ❌           |
| Run start markers    | ✅       | ❌           |
| Configurable path    | ✅       | ❌           |

**CUDA Files**: `cuda_ndt_matcher/src/main.rs:564-582, 927-937`

**Feature Gate**: `debug-output`

---

### Category 3: Voxel Grid Dumps

| Feature              | CUDA NDT | Autoware NDT |
|----------------------|----------|--------------|
| Full voxel data      | ✅       | ❌           |
| Per-voxel statistics | ✅       | ❌           |
| Configurable path    | ✅       | ❌           |

**CUDA Files**: `cuda_ndt_matcher/src/ndt_manager.rs:24-94`

**Feature Gate**: `debug-voxels`

---

### Category 4: GPU vs CPU Validation

| Feature                 | CUDA NDT | Autoware NDT |
|-------------------------|----------|--------------|
| Covariance comparison   | ✅       | ❌           |
| Trace ratio validation  | ✅       | ❌           |
| Max difference tracking | ✅       | ❌           |

**CUDA Files**: `ndt_cuda/src/voxel_grid/gpu/pipeline.rs:465-536`

**Feature Gate**: `debug-cov`

---

### Category 5: Voxel-Per-Point Distribution

| Feature               | CUDA NDT | Autoware NDT |
|-----------------------|----------|--------------|
| Distribution tracking | ✅       | ❌           |
| Count per category    | ✅       | ❌           |

**CUDA Files**: `ndt_cuda/src/derivatives/cpu.rs:432-444`

**Feature Gate**: `debug-vpp`

---

### Category 6: Visualization Features

| Feature                                   | CUDA NDT | Autoware NDT |
|-------------------------------------------|----------|--------------|
| Pose history markers                      | ✅       | ✅           |
| Voxel score point cloud                   | ✅       | ✅           |
| Particle markers (basic)                  | ✅       | ✅           |
| Particle markers (multiple color schemes) | ❌       | ✅           |
| Result pose markers                       | ❌       | ✅           |
| Aligned point clouds                      | ❌       | ✅           |

**CUDA Files**: `cuda_ndt_matcher/src/main.rs:1243-1261, 1282-1303`

**Autoware Files**: `ndt_scan_matcher_core.cpp`, `particle.cpp`

**Feature Gate**: `debug-markers` (CUDA)

---

### Category 7: Profiling & Timing

| Feature              | CUDA NDT | Autoware NDT |
|----------------------|----------|--------------|
| Per-iteration timing | ✅       | ❌           |
| Phase breakdown      | ✅       | ❌           |
| Total alignment time | ✅       | ✅           |
| Execution time topic | ❌       | ✅           |

**CUDA Files**: `ndt_cuda/src/timing.rs`, `ndt_cuda/src/optimization/debug.rs`

**Autoware Files**: `ndt_scan_matcher_core.cpp` (exe_time_ms topic)

**Feature Gate**: `profiling` (CUDA)

---

## 8. Missing Features in CUDA NDT

### 8.1 High Priority

None - core debug functionality is complete.

---

### 8.2 Medium Priority

#### ROS Performance Metric Topics

**Missing Topics**:
- `exe_time_ms` (Float32Stamped)
- `transform_probability` (Float32Stamped)
- `nearest_voxel_transformation_likelihood` (Float32Stamped)
- `iteration_num` (Int32Stamped)

**Impact**: Medium - useful for real-time monitoring, already covered in diagnostics

**Effort**: Low (1-2 hours) - simple topic publishers

**Implementation Plan**:
```rust
// Add to src/cuda_ndt_matcher/src/main.rs
struct DebugMetricPublishers {
    exe_time_pub: Publisher<Float32Stamped>,
    transform_probability_pub: Publisher<Float32Stamped>,
    nvtl_pub: Publisher<Float32Stamped>,
    iteration_num_pub: Publisher<Int32Stamped>,
}

#[cfg(feature = "debug-output")]
fn publish_debug_metrics(&self, result: &AlignmentResult, debug: &AlignmentDebug) {
    // Publish execution time, scores, iteration count
}
```

---

#### Enhanced Diagnostics Arrays

**Missing Diagnostics Fields**:
- `transform_probability_array` in diagnostics
- `nearest_voxel_transformation_likelihood_array` in diagnostics
- `transform_probability_diff`
- `nearest_voxel_transformation_likelihood_diff`
- `initial_to_result_distance`

**Impact**: Medium - useful for validation and monitoring

**Effort**: Low (2-3 hours) - extend existing diagnostics

**Implementation Plan**:
```rust
// Add to diagnostics update in src/cuda_ndt_matcher/src/main.rs
#[cfg(feature = "debug-output")]
fn update_diagnostics_with_arrays(&mut self, debug: &AlignmentDebug) {
    // Add per-iteration arrays to key-value pairs
    // Compute score diffs
    // Compute initial-to-result distance
}
```

---

#### Enhanced Particle Markers

**Missing Features**:
- Multiple color schemes (by score, iteration, index)
- Result pose markers with color variants
- Longer marker lifetime (10s vs 1s)

**Impact**: Low - existing particle markers are functional

**Effort**: Low (2-3 hours) - extend existing marker code

**Implementation Plan**:
```rust
// Add to src/cuda_ndt_matcher/src/initial_pose.rs
fn create_particle_markers_multi_scheme(
    particles: &[ParticleResult],
    color_scheme: ColorScheme, // Score, Iteration, Index
) -> MarkerArray {
    // Generate markers with different color mappings
}
```

---

### 8.3 Low Priority

#### Aligned Point Cloud Topics

**Missing Topics**:
- `points_aligned` (PointCloud2)
- `points_aligned_no_ground` (PointCloud2)
- `initial_pose_with_covariance` (PoseWithCovarianceStamped)

**Impact**: Low - mostly for visualization, not critical for functionality

**Effort**: Low (1-2 hours) - transform and publish point clouds

---

#### Pose Relationship Topics

**Missing Topics**:
- `initial_to_result_relative_pose` (PoseStamped)
- `initial_to_result_distance` (Float32Stamped)
- `initial_to_result_distance_old` (Float32Stamped)
- `initial_to_result_distance_new` (Float32Stamped)

**Impact**: Low - useful for debugging initial pose quality

**Effort**: Low (1-2 hours) - compute and publish relative poses

---

#### Subscriber Status Diagnostics

**Missing Diagnostics**:
- Initial pose subscriber status
- Regularization pose subscriber status
- Map update status
- NDT align service status

**Impact**: Very Low - primarily for Autoware's dynamic features

**Effort**: Medium (4-6 hours) - requires tracking subscription state

**Note**: CUDA NDT uses static map loading, so map update diagnostics are less relevant

---

#### No-Ground Features

**Missing Topics**:
- `no_ground_transform_probability` (Float32Stamped)
- `no_ground_nearest_voxel_transformation_likelihood` (Float32Stamped)
- `points_aligned_no_ground` (PointCloud2)

**Impact**: Very Low - ground removal not yet implemented in CUDA NDT

**Effort**: High (full ground removal feature required first)

**Note**: Autoware param `score_estimation.no_ground_points.enable` defaults to false

---

#### Dynamic Map Loading Debug

**Missing Topic**:
- `debug/loaded_pointcloud_map` (PointCloud2)

**Impact**: Very Low - CUDA NDT uses static map loading

**Effort**: N/A - not applicable to current architecture

---

## 9. CUDA NDT Advantages

### 9.1 Compile-Time Feature Gating

**Advantage**: Zero-overhead production builds
- Debug code completely eliminated when features disabled
- Smaller binary size
- Clearer compile-time guarantees
- Better compiler optimization

**Autoware Limitation**: Runtime checks only, debug infrastructure always compiled

---

### 9.2 JSONL Debug Output

**Advantage**: Persistent debug data for post-hoc analysis
- Full optimization trajectory
- Initial pose history
- Session boundaries
- Machine-readable format

**Autoware Limitation**: No file-based debug output (ROS topics only)

---

### 9.3 Voxel Grid Validation

**Advantage**: Ground truth comparison capability
- Full voxel statistics dump
- Per-voxel covariance validation
- Used for algorithm verification

**Autoware Limitation**: No voxel dump capability

---

### 9.4 GPU vs CPU Validation

**Advantage**: Runtime correctness validation
- Covariance comparison
- Trace ratio checks
- Max difference tracking

**Autoware Limitation**: No GPU/CPU comparison (CPU-only implementation)

---

### 9.5 Detailed Per-Iteration Data

**Advantage**: Deeper optimization insights
- Full gradient and Hessian
- Newton step tracking
- Line search results
- Correspondence counts

**Autoware Limitation**: Only stores scores and poses per iteration

---

## 10. Recommendations

### 10.1 Priority Implementation Order

1. **ROS Performance Metric Topics** (Medium Priority, Low Effort)
   - Adds real-time monitoring capability
   - Simple to implement
   - Useful for live debugging

2. **Enhanced Diagnostics Arrays** (Medium Priority, Low Effort)
   - Improves diagnostics detail
   - Helps with convergence analysis
   - Extends existing infrastructure

3. **Enhanced Particle Markers** (Low Priority, Low Effort)
   - Improves visualization quality
   - Quick win for completeness
   - Low risk

4. **Aligned Point Cloud Topics** (Low Priority, Low Effort)
   - Useful for visualization
   - Simple transform + publish
   - Nice-to-have feature

5. **Everything Else** (Low Priority)
   - Implement as needed
   - Not critical for core functionality
   - Some features not applicable to CUDA NDT architecture

---

### 10.2 Keep CUDA Advantages

Do NOT remove or compromise:
- ✅ Compile-time feature gating
- ✅ JSONL debug output
- ✅ Voxel grid dumps
- ✅ GPU vs CPU validation
- ✅ Detailed per-iteration data

These are differentiators that make CUDA NDT superior for debugging and validation.

---

## 11. File Reference Quick Guide

### CUDA NDT Debug Files

| File | Purpose | Feature Gate |
|------|---------|--------------|
| `ndt_cuda/src/optimization/debug.rs` | Per-iteration data structures | `debug-iteration` |
| `ndt_cuda/src/optimization/solver.rs` | Debug data population | `debug-iteration` |
| `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` | GPU debug buffers | `debug-iteration` |
| `ndt_cuda/src/timing.rs` | Timing instrumentation | `profiling` |
| `ndt_cuda/src/voxel_grid/gpu/pipeline.rs` | GPU/CPU cov comparison | `debug-cov` |
| `ndt_cuda/src/derivatives/cpu.rs` | Voxel-per-point tracking | `debug-vpp` |
| `cuda_ndt_matcher/src/main.rs` | ROS topics, markers, JSONL output | `debug-output`, `debug-markers` |
| `cuda_ndt_matcher/src/initial_pose.rs` | Particle markers, JSONL output | `debug-output` |
| `cuda_ndt_matcher/src/ndt_manager.rs` | Voxel dump | `debug-voxels` |
| `cuda_ndt_matcher/src/visualization.rs` | Visualization utilities | `debug-markers` |

### Autoware NDT Debug Files

| File | Purpose |
|------|---------|
| `ndt_scan_matcher_core.hpp` | Core debug topics declaration |
| `ndt_scan_matcher_core.cpp` | Debug topics, markers, voxel score viz |
| `particle.cpp` | Monte Carlo particle markers |
| `diagnostics.cpp` | Comprehensive diagnostics |
| `map_update_module.cpp` | Dynamic map loading debug |
| `ndt_scan_matcher.param.yaml` | Debug-related parameters |

---

## 12. Conclusion

**Overall Status**: 🟢 **CUDA NDT has strong debug parity with unique advantages**

**Core Functionality**: ✅ Complete
- Per-iteration debug data
- Visualization markers
- Diagnostics
- JSONL output
- Voxel dumps

**Gaps**: 🟡 Minor
- Some ROS debug topics missing (easy to add)
- Some enhanced diagnostics missing (non-critical)
- Some visualization variants missing (nice-to-have)

**Advantages**: 🟢 Significant
- Compile-time feature gating (zero overhead)
- JSONL debug output (persistent analysis)
- Voxel grid validation
- GPU/CPU correctness checks
- More detailed per-iteration data

**Recommendation**: Current debug infrastructure is **production-ready**. Missing features are low priority and can be added incrementally as needed.
