# Phase 10: SmartPoseBuffer (Initial Pose Interpolation)

**Status**: ✅ Complete

## Goal

Implement Autoware's `SmartPoseBuffer` for timestamp-based initial pose interpolation, improving NDT alignment accuracy by providing better initial guesses that match sensor timestamps.

## Background

**Problem**: The EKF pose and sensor data timestamps don't always align. Using the latest EKF pose directly as the initial guess introduces temporal offset error.

**Autoware's Solution**: `SmartPoseBuffer` stores recent poses and interpolates to match the exact sensor timestamp.

**Reference Implementation**:
- Header: `external/autoware_core/localization/autoware_localization_util/include/autoware/localization_util/smart_pose_buffer.hpp`
- Source: `external/autoware_core/localization/autoware_localization_util/src/smart_pose_buffer.cpp`
- Interpolation: `external/autoware_core/localization/autoware_localization_util/src/util_func.cpp`

## Components

### 10.1 PoseBuffer Data Structure ✅ COMPLETE

**File**: `src/cuda_ndt_matcher/src/pose_buffer.rs`

```rust
use geometry_msgs::msg::PoseWithCovarianceStamped;
use parking_lot::Mutex;
use std::collections::VecDeque;

/// Result of pose interpolation
pub struct InterpolateResult {
    /// Pose before target time
    pub old_pose: PoseWithCovarianceStamped,
    /// Pose after target time
    pub new_pose: PoseWithCovarianceStamped,
    /// Interpolated pose at target time
    pub interpolated_pose: PoseWithCovarianceStamped,
}

/// Thread-safe buffer for pose interpolation
pub struct SmartPoseBuffer {
    buffer: Mutex<VecDeque<PoseWithCovarianceStamped>>,
    /// Maximum age for poses (validation)
    pose_timeout_sec: f64,
    /// Maximum position jump between poses (validation)
    pose_distance_tolerance_m: f64,
}

impl SmartPoseBuffer {
    pub fn new(pose_timeout_sec: f64, pose_distance_tolerance_m: f64) -> Self;

    /// Add new pose to buffer
    pub fn push_back(&self, pose: PoseWithCovarianceStamped);

    /// Interpolate pose at target timestamp
    pub fn interpolate(&self, target_time_ns: i64) -> Option<InterpolateResult>;

    /// Remove poses older than target time
    pub fn pop_old(&self, target_time_ns: i64);

    /// Clear all poses
    pub fn clear(&self);
}
```

**Autoware Behavior to Match**:
- Buffer uses `std::deque` - matches Rust's `VecDeque`
- Clears buffer if new pose timestamp < latest (handles rosbag replay)
- Requires at least 2 poses for interpolation
- Returns `None` if target time is before first pose

### 10.2 Pose Interpolation Algorithm ✅ COMPLETE

**Interpolation Logic** (from `util_func.cpp:interpolate_pose`):

```rust
/// Interpolate between two poses at a target timestamp
pub fn interpolate_pose(
    pose_a: &PoseWithCovarianceStamped,  // Old pose
    pose_b: &PoseWithCovarianceStamped,  // New pose
    target_time_ns: i64,
) -> PoseWithCovarianceStamped {
    // 1. Compute twist (velocity) from pose_a to pose_b
    let dt_ab = timestamp_diff_sec(pose_b, pose_a);
    let twist = compute_twist(pose_a, pose_b, dt_ab);

    // 2. Compute time offset from pose_a to target
    let dt = timestamp_diff_sec_from_ns(target_time_ns, pose_a);

    // 3. Linear interpolation for position
    let x = pose_a.pose.pose.position.x + twist.linear.x * dt;
    let y = pose_a.pose.pose.position.y + twist.linear.y * dt;
    let z = pose_a.pose.pose.position.z + twist.linear.z * dt;

    // 4. Angular interpolation via euler angles
    let (roll_a, pitch_a, yaw_a) = quaternion_to_rpy(&pose_a.pose.pose.orientation);
    let roll = roll_a + twist.angular.x * dt;
    let pitch = pitch_a + twist.angular.y * dt;
    let yaw = yaw_a + twist.angular.z * dt;
    let orientation = rpy_to_quaternion(roll, pitch, yaw);

    // 5. Use old_pose covariance (Autoware does not interpolate covariance)
    PoseWithCovarianceStamped {
        header: Header { stamp: ns_to_time(target_time_ns), frame_id: pose_a.header.frame_id.clone() },
        pose: PoseWithCovariance {
            pose: Pose { position: Point { x, y, z }, orientation },
            covariance: pose_a.pose.covariance,
        },
    }
}
```

**Key Detail**: Autoware normalizes angular differences to [-π, π] using `calc_diff_for_radian()`.

### 10.3 Validation Functions ✅ COMPLETE

```rust
impl SmartPoseBuffer {
    /// Check if timestamp difference is within tolerance
    fn validate_time_stamp_difference(
        &self,
        target_time_ns: i64,
        reference_time_ns: i64,
    ) -> bool {
        let dt = (target_time_ns - reference_time_ns).abs() as f64 / 1e9;
        dt < self.pose_timeout_sec
    }

    /// Check if position jump is within tolerance
    fn validate_position_difference(
        &self,
        target: &Point,
        reference: &Point,
    ) -> bool {
        let dx = target.x - reference.x;
        let dy = target.y - reference.y;
        let dz = target.z - reference.z;
        let distance = (dx*dx + dy*dy + dz*dz).sqrt();
        distance < self.pose_distance_tolerance_m
    }
}
```

**Autoware Validations**:
- `is_old_pose_valid`: old_pose timestamp within timeout of target
- `is_new_pose_valid`: new_pose timestamp within timeout of target
- `is_pose_diff_valid`: position difference between old and new within tolerance
- All three must pass for interpolation to succeed

### 10.4 Integration with NDT Node ✅ COMPLETE

**File**: `src/cuda_ndt_matcher/src/main.rs`

**Changes**:

1. Add `pose_buffer` state:
```rust
let pose_buffer = Arc::new(SmartPoseBuffer::new(
    params.validation.initial_pose_timeout_sec,
    params.validation.initial_pose_distance_tolerance_m,
));
```

2. Update `initial_pose_sub` to push to buffer:
```rust
node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
    pose_buffer.push_back(msg);
})?
```

3. Update `on_points` to use interpolation:
```rust
// Instead of: let initial_pose = latest_pose.load();
let sensor_time_ns = msg.header.stamp.sec as i64 * 1_000_000_000
                   + msg.header.stamp.nanosec as i64;

let interpolate_result = match pose_buffer.interpolate(sensor_time_ns) {
    Some(result) => result,
    None => {
        log_warn!(NODE_NAME, "Failed to interpolate initial pose");
        return;  // Skip this scan (matches Autoware behavior)
    }
};

let initial_pose = &interpolate_result.interpolated_pose;

// Pop old poses to prevent unbounded growth
pose_buffer.pop_old(sensor_time_ns);
```

4. Update diagnostics:
```rust
is_succeed_interpolate_initial_pose: interpolate_result.is_some(),
```

### 10.5 Configuration Parameters

**File**: `src/cuda_ndt_matcher/src/params.rs`

Already exists in `ValidationParams`:
```rust
pub struct ValidationParams {
    pub initial_pose_timeout_sec: f64,           // pose_timeout_sec
    pub initial_pose_distance_tolerance_m: f64,  // pose_distance_tolerance_meters
    // ...
}
```

**Default Values** (from Autoware):
- `initial_pose_timeout_sec`: 1.0
- `initial_pose_distance_tolerance_m`: 10.0

## Tests

- [x] Unit test: `push_back` clears buffer on timestamp reversal
- [x] Unit test: `interpolate` requires minimum 2 poses
- [x] Unit test: `interpolate` returns None when target < first pose
- [x] Unit test: Linear position interpolation is correct
- [x] Unit test: Angular interpolation (uses SLERP for smooth quaternion interpolation)
- [x] Unit test: Time validation rejects stale poses
- [x] Unit test: Distance validation rejects position jumps
- [x] Unit test: Covariance comes from old_pose (not interpolated)
- [ ] Integration test: Interpolated pose improves alignment accuracy

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/cuda_ndt_matcher/src/pose_buffer.rs` | NEW | SmartPoseBuffer implementation |
| `src/cuda_ndt_matcher/src/main.rs` | MODIFY | Integrate pose buffer |
| `src/cuda_ndt_matcher/src/lib.rs` or `mod.rs` | MODIFY | Add module declaration |

## Dependencies

No new dependencies required. Uses:
- `nalgebra` for rotation conversions (already in use)
- `parking_lot::Mutex` for thread safety (already in use)
- `std::collections::VecDeque` (stdlib)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quaternion to euler conversion edge cases | Medium | Use nalgebra's robust conversion |
| Buffer growth without pop_old | Low | Call pop_old after each interpolation |
| Time synchronization issues | Medium | Log warnings when validation fails |
