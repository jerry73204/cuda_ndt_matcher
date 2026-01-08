# Phase 7: ROS Integration & Production Features

**Status**: ✅ Complete

## Goal

Complete ROS integration features for full Autoware compatibility.

## 7.1 TF Broadcasting ✅ COMPLETE

**Implemented:** `map → ndt_base_link` transform broadcast matching Autoware.

**Implementation Details:**
- Added `tf_pub: Publisher<TFMessage>` to `DebugPublishers` struct
- Publisher publishes to `/tf` topic (absolute topic name)
- `publish_tf()` function converts `Pose` to `TransformStamped`
- Frame IDs are configurable via `frame.map_frame` and `frame.ndt_base_frame` parameters
- Default: `map` → `ndt_base_link` (matching Autoware defaults)

**Code Location:** `src/cuda_ndt_matcher/src/main.rs`

```rust
fn publish_tf(
    tf_pub: &Publisher<TFMessage>,
    stamp: &builtin_interfaces::msg::Time,
    pose: &Pose,
    map_frame: &str,
    ndt_base_frame: &str,
) {
    let transform = geometry_msgs::msg::Transform {
        translation: geometry_msgs::msg::Vector3 {
            x: pose.position.x,
            y: pose.position.y,
            z: pose.position.z,
        },
        rotation: pose.orientation.clone(),
    };

    let transform_stamped = geometry_msgs::msg::TransformStamped {
        header: Header {
            stamp: stamp.clone(),
            frame_id: map_frame.to_string(),
        },
        child_frame_id: ndt_base_frame.to_string(),
        transform,
    };

    let tf_msg = TFMessage {
        transforms: vec![transform_stamped],
    };

    tf_pub.publish(&tf_msg);
}
```

## 7.2 Dynamic Map Loading ✅ COMPLETE

**Implemented:** Full differential map loading via `GetDifferentialPointCloudMap` service.

**What's Working:**
- `MapUpdateModule` with tile management and position-based filtering
- `DynamicMapLoader` service client for `GetDifferentialPointCloudMap`
- Automatic service request during each NDT alignment (`on_points`)
- `should_update()` - checks if position moved beyond `update_distance`
- `out_of_map_range()` - detects when approaching edge of loaded map
- `check_and_update()` - combined check+update with NDT target refresh
- `get_stats()` - map statistics for monitoring
- Points filtered within `map_radius` of current position
- Async service callback handling (node spins for response processing)

**Code Locations:**
- `src/cuda_ndt_matcher/src/map_module.rs` - `MapUpdateModule` and `DynamicMapLoader`
- `src/cuda_ndt_matcher/src/main.rs` - Integration in `on_points()`

**Service Client Implementation:**
```rust
// DynamicMapLoader handles the GetDifferentialPointCloudMap service
pub struct DynamicMapLoader {
    client: Client<GetDifferentialPointCloudMap>,
    map_module: Arc<MapUpdateModule>,
    request_pending: Arc<AtomicBool>,
}

impl DynamicMapLoader {
    // Create service client for pcd_loader_service
    pub fn new(node: &Node, service_name: &str, map_module: Arc<MapUpdateModule>) -> Result<Self>;

    // Request map tiles around position (async, callback-based)
    pub fn request_map_update(&self, position: &Point, map_radius: f32) -> Result<bool>;
}
```

**Behavior:**
1. Service client connects to `pcd_loader_service`
2. On position change beyond `update_distance`, request is sent
3. Request includes current position, radius, and cached tile IDs
4. Response provides new tiles to add and old tile IDs to remove
5. Callback updates `MapUpdateModule` with differential changes
6. Node spinning ensures callback execution

**NOT Implemented:**
- Secondary NDT for non-blocking updates (Autoware feature for smoother transitions)

## 7.3 GNSS Regularization ✅ COMPLETE

Autoware uses GNSS poses to regularize NDT in open areas where scan matching may drift.

```rust
// Add regularization term to NDT cost function
pub struct RegularizationTerm {
    gnss_pose: Option<Isometry3<f64>>,
    scale_factor: f64,  // Default: 0.01
}

impl RegularizationTerm {
    pub fn add_to_derivatives(
        &self,
        current_pose: &Isometry3<f64>,
        gradient: &mut Vector6<f64>,
        hessian: &mut Matrix6<f64>,
    ) {
        if let Some(gnss) = &self.gnss_pose {
            // Add quadratic penalty: scale * ||current - gnss||^2
            let diff = pose_difference(current_pose, gnss);
            *gradient += self.scale_factor * diff;
            // Hessian contribution: scale * I
        }
    }
}
```

## 7.4 Multi-NDT Covariance Estimation ✅ COMPLETE

**Implemented:** Full multi-NDT covariance estimation matching Autoware's algorithm.

**Modes Supported:**
- `MULTI_NDT`: Run NDT alignment from offset poses, compute sample covariance
- `MULTI_NDT_SCORE`: Compute NVTL at offset poses (no alignment), use softmax-weighted covariance

**Code Location:** `src/cuda_ndt_matcher/src/covariance.rs`

**Key Functions:**
```rust
/// Create offset poses rotated by result pose orientation
pub fn propose_offset_poses(
    result_pose: &Pose,
    offset_x: &[f64],
    offset_y: &[f64],
) -> Vec<Pose>;

/// MULTI_NDT: Run alignment from each offset, compute sample covariance
pub fn estimate_xy_covariance_by_multi_ndt(
    ndt_manager: &mut NdtManager,
    sensor_points: &[[f32; 3]],
    map_points: &[[f32; 3]],
    result_pose: &Pose,
    estimation_params: &CovarianceEstimationParams,
) -> MultiNdtResult;

/// MULTI_NDT_SCORE: Compute NVTL at offsets, use softmax weights
pub fn estimate_xy_covariance_by_multi_ndt_score(
    ndt_manager: &mut NdtManager,
    sensor_points: &[[f32; 3]],
    map_points: &[[f32; 3]],
    result_pose: &Pose,
    estimation_params: &CovarianceEstimationParams,
) -> MultiNdtResult;
```

**Default Offset Model (matching Autoware):**
- X offsets: [0.0, 0.0, 0.5, -0.5, 1.0, -1.0]
- Y offsets: [0.5, -0.5, 0.0, 0.0, 0.0, 0.0]

**Integration:**
- `estimate_covariance_full()` handles all modes including multi-NDT
- Falls back to Laplace approximation if required data is not available

## 7.5 Diagnostics Interface ✅ Complete

Add ROS diagnostics for system health monitoring.

**Implementation:**
- `DiagnosticsInterface` publishes to `/diagnostics` topic
- `DiagnosticCategory` for each diagnostic type with key-value pairs and severity levels
- `ScanMatchingDiagnostics` collects scan matching metrics:
  - `sensor_points_size`, `sensor_points_delay_time_sec`, `sensor_points_max_distance`
  - `is_activated`, `is_set_map_points`, `is_succeed_interpolate_initial_pose`
  - `iteration_num`, `oscillation_count`, `transform_probability`, `nvtl`
  - `distance_initial_to_result`, `execution_time_ms`, `skipping_publish_num`
- Diagnostic levels: OK, WARN (oscillation, no map, etc.), ERROR (transform failed)

## 7.6 Oscillation Detection ✅ Complete

Detect when optimization oscillates between poses.

```rust
pub fn count_oscillation(pose_history: &[Pose]) -> usize {
    // Count direction reversals in pose sequence
    let mut reversals = 0;
    for window in pose_history.windows(3) {
        let d1 = pose_difference(&window[0], &window[1]);
        let d2 = pose_difference(&window[1], &window[2]);
        if d1.dot(&d2) < 0.0 {
            reversals += 1;
        }
    }
    reversals
}
```

## 7.7 Monte Carlo Visualization ✅ COMPLETE

Publish visualization markers for Monte Carlo initial pose estimation particles.

**Implementation:**
- `create_monte_carlo_markers()` converts particles to `MarkerArray`
- Initial poses shown as small blue spheres
- Result poses shown as spheres colored by score (red=low, green=high)
- Best particle highlighted with larger size
- Published to `monte_carlo_initial_pose_marker` topic after NDT align service call
- Markers have 10-second lifetime for debugging visibility

## 7.8 GPU Scoring Integration ✅ COMPLETE

Use GPU for transform probability and NVTL evaluation.

**Status:** Both transform probability and NVTL use GPU when available, with CPU fallback.

**Implementation:**

1. **Transform Probability (sum-based aggregation)**
   - `compute_ndt_score_kernel`: Sums scores across all neighbor voxels per point
   - `evaluate_transform_probability()` uses GPU via `evaluate_scores_gpu()`
   - Result: `total_score / num_correspondences`

2. **NVTL (max-based aggregation, Autoware-compatible)**
   - `compute_ndt_nvtl_kernel`: Takes **max** score per point across voxels
   - `evaluate_nvtl()` uses GPU via `evaluate_nvtl_gpu()`
   - Result: Average of max scores across points with neighbors

**Key Files:**
- `src/ndt_cuda/src/derivatives/gpu.rs`: `compute_ndt_nvtl_kernel`
- `src/ndt_cuda/src/runtime.rs`: `compute_nvtl_scores()`, `GpuNvtlResult`
- `src/ndt_cuda/src/ndt.rs`: `evaluate_nvtl_gpu()`, updated `evaluate_nvtl()`

## 7.9 Score Threshold Filtering ✅ COMPLETE

Skip pose publishing when alignment quality is below threshold, matching Autoware's `is_converged` check.

**Implementation** (2026-01-05):

1. **Score computation before publishing**
   - NVTL and transform_probability computed immediately after alignment
   - Scores available before publish decision

2. **Threshold check based on converged_param_type**
   - `converged_param_type = 0`: Use transform_probability (threshold: 3.0)
   - `converged_param_type = 1`: Use NVTL (threshold: 2.3) - **default**

3. **Skip counter tracking**
   - `skip_counter: Arc<AtomicI32>` tracks consecutive skips
   - Incremented when score below threshold
   - Reset to 0 when score above threshold
   - Reported in diagnostics as `skipping_publish_num`

4. **Conditional publishing**
   - Pose, pose_with_covariance, and TF only published when score ≥ threshold
   - Debug metrics (NVTL, iteration_num, exe_time, etc.) always published for monitoring

**Code Location:** `src/cuda_ndt_matcher/src/main.rs:614-707`

```rust
// Score threshold check (like Autoware's is_converged check)
let (score_for_check, threshold, score_name) = if params.score.converged_param_type == 0 {
    (transform_prob, params.score.converged_param_transform_probability, "transform_probability")
} else {
    (nvtl_score, params.score.converged_param_nearest_voxel_transformation_likelihood, "NVTL")
};

let skipping_publish_num = if score_for_check < threshold {
    let skips = skip_counter.fetch_add(1, Ordering::SeqCst) + 1;
    log_warn!(NODE_NAME, "Score below threshold: {score_name}={score_for_check:.3} < {threshold:.3}");
    skips
} else {
    skip_counter.store(0, Ordering::SeqCst);
    0
};

// Only publish pose if score is above threshold
if score_for_check >= threshold {
    // Publish pose, pose_with_covariance, TF
}
```

**Configuration** (from `ndt_scan_matcher.param.yaml`):
- `converged_param_type: 1` (use NVTL)
- `converged_param_nearest_voxel_transformation_likelihood: 2.3`
- `converged_param_transform_probability: 3.0`

## Tests

- [x] TF broadcast implemented (`map` → `ndt_base_link`)
- [ ] TF broadcast verified with `tf2_echo` (runtime test)
- [x] Position-based map update logic implemented
- [x] Map radius filtering works correctly
- [x] `check_and_update()` convenience method
- [x] `get_stats()` provides map statistics
- [x] Dynamic map loading with pcd_loader service
- [x] GNSS regularization implemented (penalizes deviation from GNSS pose)
- [ ] Multi-NDT covariance matches Autoware output
- [x] Diagnostics published to `/diagnostics` (scan_matching + map_update)
- [x] Oscillation detection implemented (publishes to `local_optimal_solution_oscillation_num`)
- [x] Monte Carlo particle visualization (markers to `monte_carlo_initial_pose_marker`)
- [x] GPU scoring for evaluate_transform_probability (sum-based)
- [x] GPU scoring for evaluate_nvtl (max-per-point, Autoware-compatible)
- [x] Score threshold filtering (skip publish when NVTL < 2.3)
- [x] Skip counter tracking in diagnostics (`skipping_publish_num`)
