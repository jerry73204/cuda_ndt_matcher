# Roadmap

## Phase 1: Core Integration

Build minimal working node using fast_gicp_rust.

### Work Items

- [x] Add fast-gicp dependency to workspace
- [x] Implement point cloud conversion (PointCloud2 <-> fast_gicp)
- [x] Implement basic NDT alignment using NDTCuda
- [x] Implement pose output (ndt_pose topic)
- [x] Add basic parameters (resolution, max_iterations)

### Passing Criteria

- Node subscribes to `points_raw` and `ekf_pose_with_covariance`
- Node publishes `ndt_pose` with valid transform
- Alignment runs on GPU (nvidia-smi shows usage)

### Tests

```bash
# Unit: point cloud conversion
cargo test point_cloud_conversion

# Integration: node starts and publishes
ros2 run cuda_ndt_matcher cuda_ndt_matcher &
ros2 topic hz /ndt_pose  # Should show messages

# Smoke: alignment with sample data
ros2 bag play sample.bag
# Verify ndt_pose output is reasonable
```

---

## Phase 2: Full ROS Interface

Match Autoware ndt_scan_matcher interface.

### Work Items

- [x] Add all subscriptions (regularization_pose)
- [x] Add ndt_pose_with_covariance publisher
- [x] Implement trigger_node_srv service
- [ ] Add pcd_loader_service client (deferred to Phase 5)
- [x] Implement all parameters from config
- [x] Add launch file with remapping

### Passing Criteria

- Launch file is drop-in replacement for Autoware
- All topics/services match original names
- Parameters loaded from YAML

### Tests

```bash
# Integration: launch file works
ros2 launch cuda_ndt_matcher_launch ndt_scan_matcher.launch.xml

# Interface: topics exist
ros2 topic list | grep -E "ndt_pose|points_raw"
ros2 service list | grep trigger_node

# Config: parameters loaded
ros2 param list /ndt_scan_matcher
```

---

## Phase 3: Covariance Estimation

Implement covariance estimation modes.

### Work Items

- [x] Implement FIXED covariance mode
- [x] Implement LAPLACE approximation (Hessian inverse)
- [x] Implement MULTI_NDT (multiple alignments)
- [x] Implement MULTI_NDT_SCORE (score-weighted)
- [x] Add covariance parameters
- [x] Add Hessian/cost FFI bindings to fast_gicp_rust

### Passing Criteria

- All 4 covariance modes work
- Output covariance varies with alignment quality
- GPU kernel execution confirmed

### Tests

```bash
# Unit: covariance estimation
cargo test covariance

# Integration: covariance output (change mode via config)
# In config: covariance.covariance_estimation.covariance_estimation_type: 1
ros2 topic echo /ndt_pose_with_covariance --field pose.covariance

# Verify different modes:
# 0 = FIXED: static covariance from config
# 1 = LAPLACE: varies with alignment quality
# 2 = MULTI_NDT: sample covariance from multiple alignments
# 3 = MULTI_NDT_SCORE: score-weighted covariance (faster)
```

---

## Phase 4: Initial Pose Estimation

Implement Monte Carlo initial pose with TPE-guided search.

### Work Items

- [x] Implement particle generation (from prior distributions)
- [x] Implement score evaluation (via NDTCuda alignment)
- [x] Implement best particle selection
- [x] Add TPE (Tree-Structured Parzen Estimator) search
- [x] Add initial_pose_estimation parameters
- [x] Add ndt_align_srv service (with correct message type)
- [x] Fix executor spin pattern for service responses

### Passing Criteria

- Node recovers from poor initial guess
- particles_num affects search coverage
- TPE guides search toward better regions

### Tests

```bash
# Unit: TPE and particle tests pass
cargo test tpe
cargo test initial_pose
cargo test particle

# Integration: trigger initial pose estimation
ros2 service call /ndt_scan_matcher/ndt_align_srv tier4_localization_msgs/srv/PoseWithCovarianceStamped "..."

# Verify response includes score and reliability info
```

---

## Phase 5: Dynamic Map Loading

Implement map management and caching.

### Work Items

- [x] Implement map cache structure (MapUpdateModule with tile storage)
- [x] Implement distance-based update trigger (should_update, out_of_map_range)
- [x] Add map subscription (pointcloud_map topic)
- [x] Add map_update_srv service (trigger map update)
- [x] Implement map radius filtering (filters points within map_radius)
- [x] Add dynamic_map_loading parameters (already in config)

### Passing Criteria

- Map updates when vehicle moves update_distance
- Only loads map within map_radius
- Old map sections unloaded

### Tests

```bash
# Unit: map module tests pass
cargo test map_module

# Integration: receive map via subscription
ros2 topic pub /pointcloud_map sensor_msgs/msg/PointCloud2 ...

# Integration: trigger map update
ros2 service call /ndt_scan_matcher/map_update_srv std_srvs/srv/Trigger
```

---

## Phase 6: Visualization & Debug Topics

Add missing publishers for RViz visualization and debugging.

### Work Items

**Visualization (RViz essential):**
- [x] Add `ndt_marker` publisher (MarkerArray) - NDT result pose as arrow/axis
  - Note: Publishes only final pose. Builtin publishes transformation_array (all iteration poses) but fast-gicp doesn't expose iteration history.
- [x] Add `points_aligned` publisher (PointCloud2) - Source points transformed by result
- [ ] Add `monte_carlo_initial_pose_marker` publisher (MarkerArray) - Initial pose particles

**Debug Metrics:**
- [x] Add `transform_probability` publisher (Float32Stamped) - NDT fitness score
- [x] Add `nearest_voxel_transformation_likelihood` publisher (Float32Stamped) - NVTL score
- [x] Add `iteration_num` publisher (Int32Stamped) - Convergence iterations
- [x] Add `exe_time_ms` publisher (Float32Stamped) - Execution time

**Pose Tracking:**
- [x] Add `initial_pose_with_covariance` publisher (PoseWithCovarianceStamped)
- [x] Add `initial_to_result_distance` publisher (Float32Stamped)
- [x] Add `initial_to_result_relative_pose` publisher (PoseStamped)

**Optional:**
- [ ] Add `multi_ndt_pose` publisher (PoseArray) - Multi-NDT covariance poses
- [ ] Add `multi_initial_pose` publisher (PoseArray) - Multi-initial poses
- [ ] Add `/tf` broadcast (base_link -> ndt_base_link)
- [ ] Add `voxel_score_points` publisher (PointCloud2) - Points colored by score

### Passing Criteria

- NDT result visible in RViz as marker
- Aligned point cloud visible in RViz
- Initial pose particles visible during estimation
- Debug topics show valid values

### Tests

```bash
# Integration: verify all topics exist
ros2 topic list | grep -E "ndt_marker|points_aligned|monte_carlo"

# Visualization: check markers in RViz
rviz2 -d config/ndt_debug.rviz

# Debug: verify metrics
ros2 topic echo /localization/pose_estimator/transform_probability
ros2 topic echo /localization/pose_estimator/nearest_voxel_transformation_likelihood
ros2 topic echo /localization/pose_estimator/iteration_num
```

---

## Phase 7: Validation & Diagnostics

Implement all validation and diagnostic features.

### Work Items

- [ ] Implement timestamp validation
- [ ] Implement distance validation
- [ ] Implement execution time monitoring
- [ ] Implement skip counting
- [ ] Add diagnostics publisher
- [ ] Add no_ground_points score option

### Passing Criteria

- Invalid inputs rejected with diagnostics
- Execution time warnings published
- No ground score computed when enabled

### Tests

```bash
# Unit: validation logic
cargo test validation

# Integration: diagnostics output
ros2 topic echo /diagnostics

# Edge cases: invalid inputs handled
ros2 topic pub /points_raw ...  # Empty cloud
ros2 topic pub /ekf_pose_with_covariance ...  # Old timestamp
```

---

## Phase 8: Performance Optimization

Optimize for real-time performance.

### Work Items

- [ ] Profile GPU kernel execution
- [ ] Optimize memory transfers
- [ ] Implement async processing
- [ ] Add performance metrics
- [ ] Upstream improvements to fast_gicp_rust

### Passing Criteria

- < 50ms total latency for 10K points
- < 100ms for 50K points
- Stable frame rate without drops

### Tests

```bash
# Benchmark: end-to-end latency
cargo bench ndt_pipeline

# Stress: high frequency input
ros2 bag play --rate 2.0 sample.bag
# Verify no message drops

# Profile: GPU utilization
nsys profile ros2 run cuda_ndt_matcher cuda_ndt_matcher
```
