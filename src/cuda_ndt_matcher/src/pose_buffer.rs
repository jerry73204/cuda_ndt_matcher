//! SmartPoseBuffer - Timestamp-based pose interpolation for initial pose estimation
//!
//! This module implements Autoware's SmartPoseBuffer behavior for interpolating
//! EKF poses to match sensor timestamps, improving NDT alignment accuracy.
//!
//! Reference: external/autoware_core/localization/autoware_localization_util/src/smart_pose_buffer.cpp

use geometry_msgs::msg::{Point, Pose, PoseWithCovariance, PoseWithCovarianceStamped, Quaternion};
use nalgebra::{Quaternion as NaQuaternion, UnitQuaternion};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std_msgs::msg::Header;

/// Result of pose interpolation
#[derive(Clone)]
pub struct InterpolateResult {
    /// Pose before target time (for debugging/analysis)
    #[allow(dead_code)]
    pub old_pose: PoseWithCovarianceStamped,
    /// Pose after target time (for debugging/analysis)
    #[allow(dead_code)]
    pub new_pose: PoseWithCovarianceStamped,
    /// Interpolated pose at target time
    pub interpolated_pose: PoseWithCovarianceStamped,
}

/// Thread-safe buffer for pose interpolation
///
/// Stores recent poses with timestamps and provides interpolation
/// to match sensor data timestamps exactly.
pub struct SmartPoseBuffer {
    buffer: Mutex<VecDeque<PoseWithCovarianceStamped>>,
    /// Maximum age for poses in seconds (validation)
    pose_timeout_sec: f64,
    /// Maximum position jump between poses in meters (validation)
    pose_distance_tolerance_m: f64,
}

impl SmartPoseBuffer {
    /// Create a new SmartPoseBuffer
    ///
    /// # Arguments
    /// * `pose_timeout_sec` - Maximum allowed time difference between pose and target (seconds)
    /// * `pose_distance_tolerance_m` - Maximum allowed position jump between poses (meters)
    pub fn new(pose_timeout_sec: f64, pose_distance_tolerance_m: f64) -> Self {
        Self {
            buffer: Mutex::new(VecDeque::with_capacity(100)),
            pose_timeout_sec,
            pose_distance_tolerance_m,
        }
    }

    /// Add a new pose to the buffer
    ///
    /// If the new pose has a timestamp earlier than the latest pose in the buffer,
    /// the buffer is cleared (handles rosbag replay scenarios).
    pub fn push_back(&self, pose: PoseWithCovarianceStamped) {
        let mut buffer = self.buffer.lock();

        if !buffer.is_empty() {
            // Check for non-chronological timestamp order
            let last_time_ns = Self::stamp_to_ns(&buffer.back().unwrap().header.stamp);
            let msg_time_ns = Self::stamp_to_ns(&pose.header.stamp);

            if msg_time_ns < last_time_ns {
                // Clear buffer if timestamps are reversed (rosbag replay)
                buffer.clear();
            }
        }

        buffer.push_back(pose);
    }

    /// Interpolate pose at the target timestamp
    ///
    /// Finds the two poses bracketing the target time and performs linear
    /// interpolation for position and angular interpolation for orientation.
    ///
    /// # Arguments
    /// * `target_time_ns` - Target timestamp in nanoseconds
    ///
    /// # Returns
    /// * `Some(InterpolateResult)` - If interpolation succeeds
    /// * `None` - If buffer has < 2 poses, target is before first pose,
    ///   or validation fails (timeout, position jump)
    pub fn interpolate(&self, target_time_ns: i64) -> Option<InterpolateResult> {
        let buffer = self.buffer.lock();

        // Need at least 2 poses for interpolation
        if buffer.len() < 2 {
            return None;
        }

        let first_time_ns = Self::stamp_to_ns(&buffer.front().unwrap().header.stamp);

        // Target must be after first pose
        if target_time_ns < first_time_ns {
            return None;
        }

        // Find bracketing poses (old_pose <= target_time <= new_pose)
        let mut old_pose = buffer.front().unwrap().clone();
        let mut new_pose = old_pose.clone();

        for pose in buffer.iter() {
            new_pose = pose.clone();
            let pose_time_ns = Self::stamp_to_ns(&pose.header.stamp);
            if pose_time_ns > target_time_ns {
                break;
            }
            old_pose = pose.clone();
        }

        // Release lock before validation (validation doesn't need buffer)
        drop(buffer);

        // Validate time stamps
        let old_time_ns = Self::stamp_to_ns(&old_pose.header.stamp);
        let new_time_ns = Self::stamp_to_ns(&new_pose.header.stamp);

        if !self.validate_time_stamp_difference(target_time_ns, old_time_ns) {
            return None;
        }
        if !self.validate_time_stamp_difference(target_time_ns, new_time_ns) {
            return None;
        }

        // Validate position difference (detect jumps after initial pose reset)
        if !self.validate_position_difference(
            &old_pose.pose.pose.position,
            &new_pose.pose.pose.position,
        ) {
            return None;
        }

        // Perform interpolation
        let interpolated_pose = Self::interpolate_pose(&old_pose, &new_pose, target_time_ns);

        Some(InterpolateResult {
            old_pose,
            new_pose,
            interpolated_pose,
        })
    }

    /// Remove poses older than the target time
    ///
    /// Keeps at least one pose before the target time for future interpolation.
    pub fn pop_old(&self, target_time_ns: i64) {
        let mut buffer = self.buffer.lock();

        while buffer.len() > 1 {
            let front_time_ns = Self::stamp_to_ns(&buffer.front().unwrap().header.stamp);
            if front_time_ns >= target_time_ns {
                break;
            }
            buffer.pop_front();
        }
    }

    /// Clear all poses from the buffer
    #[allow(dead_code)]
    pub fn clear(&self) {
        self.buffer.lock().clear();
    }

    /// Get the number of poses in the buffer
    pub fn len(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Check if the buffer is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.buffer.lock().is_empty()
    }

    /// Get the latest (most recent) pose in the buffer
    ///
    /// Returns `None` if the buffer is empty.
    pub fn latest(&self) -> Option<PoseWithCovarianceStamped> {
        self.buffer.lock().back().cloned()
    }

    // ---- Private helper methods ----

    /// Convert ROS timestamp to nanoseconds
    fn stamp_to_ns(stamp: &builtin_interfaces::msg::Time) -> i64 {
        stamp.sec as i64 * 1_000_000_000 + stamp.nanosec as i64
    }

    /// Convert nanoseconds to ROS timestamp
    fn ns_to_stamp(ns: i64) -> builtin_interfaces::msg::Time {
        builtin_interfaces::msg::Time {
            sec: (ns / 1_000_000_000) as i32,
            nanosec: (ns % 1_000_000_000) as u32,
        }
    }

    /// Check if timestamp difference is within tolerance
    fn validate_time_stamp_difference(&self, target_time_ns: i64, reference_time_ns: i64) -> bool {
        let dt_sec = (target_time_ns - reference_time_ns).abs() as f64 / 1e9;
        dt_sec < self.pose_timeout_sec
    }

    /// Check if position jump is within tolerance
    fn validate_position_difference(&self, p1: &Point, p2: &Point) -> bool {
        let dx = p1.x - p2.x;
        let dy = p1.y - p2.y;
        let dz = p1.z - p2.z;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        distance < self.pose_distance_tolerance_m
    }

    /// Interpolate between two poses at a target timestamp
    ///
    /// Uses linear interpolation for position and SLERP for orientation.
    fn interpolate_pose(
        pose_a: &PoseWithCovarianceStamped,
        pose_b: &PoseWithCovarianceStamped,
        target_time_ns: i64,
    ) -> PoseWithCovarianceStamped {
        let time_a = Self::stamp_to_ns(&pose_a.header.stamp);
        let time_b = Self::stamp_to_ns(&pose_b.header.stamp);

        // Handle edge case: same timestamp
        if time_a == time_b {
            return pose_a.clone();
        }

        // Compute interpolation factor t in [0, 1]
        let t = (target_time_ns - time_a) as f64 / (time_b - time_a) as f64;
        let t = t.clamp(0.0, 1.0);

        // Linear interpolation for position
        let x = pose_a.pose.pose.position.x
            + t * (pose_b.pose.pose.position.x - pose_a.pose.pose.position.x);
        let y = pose_a.pose.pose.position.y
            + t * (pose_b.pose.pose.position.y - pose_a.pose.pose.position.y);
        let z = pose_a.pose.pose.position.z
            + t * (pose_b.pose.pose.position.z - pose_a.pose.pose.position.z);

        // SLERP for orientation
        let q_a = Self::ros_quat_to_unit_quat(&pose_a.pose.pose.orientation);
        let q_b = Self::ros_quat_to_unit_quat(&pose_b.pose.pose.orientation);
        let q_interp = q_a.slerp(&q_b, t);
        let orientation = Self::unit_quat_to_ros_quat(&q_interp);

        // Use old_pose covariance (Autoware does not interpolate covariance)
        PoseWithCovarianceStamped {
            header: Header {
                stamp: Self::ns_to_stamp(target_time_ns),
                frame_id: pose_a.header.frame_id.clone(),
            },
            pose: PoseWithCovariance {
                pose: Pose {
                    position: Point { x, y, z },
                    orientation,
                },
                covariance: pose_a.pose.covariance,
            },
        }
    }

    /// Convert ROS quaternion to nalgebra UnitQuaternion
    fn ros_quat_to_unit_quat(q: &Quaternion) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z))
    }

    /// Convert nalgebra UnitQuaternion to ROS quaternion
    fn unit_quat_to_ros_quat(q: &UnitQuaternion<f64>) -> Quaternion {
        Quaternion {
            x: q.i,
            y: q.j,
            z: q.k,
            w: q.w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pose(sec: i32, nanosec: u32, x: f64, y: f64, z: f64) -> PoseWithCovarianceStamped {
        PoseWithCovarianceStamped {
            header: Header {
                stamp: builtin_interfaces::msg::Time { sec, nanosec },
                frame_id: "map".to_string(),
            },
            pose: PoseWithCovariance {
                pose: Pose {
                    position: Point { x, y, z },
                    orientation: Quaternion {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        w: 1.0,
                    },
                },
                covariance: [0.0; 36],
            },
        }
    }

    #[test]
    fn test_push_back_and_len() {
        let buffer = SmartPoseBuffer::new(1.0, 10.0);
        assert!(buffer.is_empty());

        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        assert_eq!(buffer.len(), 1);

        buffer.push_back(make_pose(2, 0, 1.0, 0.0, 0.0));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_push_back_clears_on_timestamp_reversal() {
        let buffer = SmartPoseBuffer::new(1.0, 10.0);

        buffer.push_back(make_pose(10, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(11, 0, 1.0, 0.0, 0.0));
        assert_eq!(buffer.len(), 2);

        // Push pose with earlier timestamp (simulates rosbag replay restart)
        buffer.push_back(make_pose(5, 0, 2.0, 0.0, 0.0));
        assert_eq!(buffer.len(), 1); // Buffer was cleared, only new pose remains
    }

    #[test]
    fn test_interpolate_requires_two_poses() {
        let buffer = SmartPoseBuffer::new(1.0, 10.0);

        // Empty buffer
        assert!(buffer.interpolate(1_000_000_000).is_none());

        // Single pose
        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        assert!(buffer.interpolate(1_000_000_000).is_none());

        // Two poses - should work
        buffer.push_back(make_pose(2, 0, 1.0, 0.0, 0.0));
        assert!(buffer.interpolate(1_500_000_000).is_some());
    }

    #[test]
    fn test_interpolate_returns_none_before_first_pose() {
        let buffer = SmartPoseBuffer::new(1.0, 10.0);

        buffer.push_back(make_pose(10, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(11, 0, 1.0, 0.0, 0.0));

        // Target before first pose
        let target_ns = 9_000_000_000; // 9 seconds
        assert!(buffer.interpolate(target_ns).is_none());
    }

    #[test]
    fn test_interpolate_position_linear() {
        let buffer = SmartPoseBuffer::new(1.0, 20.0); // 20m tolerance

        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(2, 0, 10.0, 0.0, 0.0)); // 10m < 20m tolerance

        // Interpolate at t=1.5s (midpoint)
        let target_ns = 1_500_000_000;
        let result = buffer.interpolate(target_ns).unwrap();

        let pos = &result.interpolated_pose.pose.pose.position;
        assert!((pos.x - 5.0).abs() < 1e-9);
        assert!((pos.y - 0.0).abs() < 1e-9);
        assert!((pos.z - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_at_exact_pose_time() {
        let buffer = SmartPoseBuffer::new(2.0, 20.0); // 2s timeout, 20m tolerance

        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(2, 0, 10.0, 0.0, 0.0)); // 10m < 20m tolerance

        // Interpolate at exactly t=1s (old_pose time)
        // Need 2s timeout because new_pose is 1s away from target
        let target_ns = 1_000_000_000;
        let result = buffer.interpolate(target_ns).unwrap();

        let pos = &result.interpolated_pose.pose.pose.position;
        assert!((pos.x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_validation_rejects_stale_poses() {
        let buffer = SmartPoseBuffer::new(0.5, 10.0); // 0.5 second timeout

        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(2, 0, 1.0, 0.0, 0.0));

        // Target at t=3s is 1s after last pose, exceeds 0.5s timeout
        let target_ns = 3_000_000_000;
        assert!(buffer.interpolate(target_ns).is_none());
    }

    #[test]
    fn test_distance_validation_rejects_position_jumps() {
        let buffer = SmartPoseBuffer::new(1.0, 5.0); // 5 meter tolerance

        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(2, 0, 10.0, 0.0, 0.0)); // 10m jump > 5m tolerance

        let target_ns = 1_500_000_000;
        assert!(buffer.interpolate(target_ns).is_none());
    }

    #[test]
    fn test_pop_old() {
        let buffer = SmartPoseBuffer::new(1.0, 10.0);

        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(2, 0, 1.0, 0.0, 0.0));
        buffer.push_back(make_pose(3, 0, 2.0, 0.0, 0.0));
        buffer.push_back(make_pose(4, 0, 3.0, 0.0, 0.0));
        assert_eq!(buffer.len(), 4);

        // Pop poses older than t=2.5s
        buffer.pop_old(2_500_000_000);

        // Should keep poses at t=2s, t=3s, t=4s (at least one before target)
        assert!(buffer.len() >= 2);
        assert!(buffer.len() <= 3);
    }

    #[test]
    fn test_clear() {
        let buffer = SmartPoseBuffer::new(1.0, 10.0);

        buffer.push_back(make_pose(1, 0, 0.0, 0.0, 0.0));
        buffer.push_back(make_pose(2, 0, 1.0, 0.0, 0.0));
        assert_eq!(buffer.len(), 2);

        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_covariance_from_old_pose() {
        let buffer = SmartPoseBuffer::new(1.0, 10.0);

        let mut pose1 = make_pose(1, 0, 0.0, 0.0, 0.0);
        pose1.pose.covariance[0] = 1.0; // Mark old pose covariance

        let mut pose2 = make_pose(2, 0, 1.0, 0.0, 0.0);
        pose2.pose.covariance[0] = 2.0; // Different covariance

        buffer.push_back(pose1);
        buffer.push_back(pose2);

        let result = buffer.interpolate(1_500_000_000).unwrap();

        // Should use old_pose covariance
        assert!((result.interpolated_pose.pose.covariance[0] - 1.0).abs() < 1e-9);
    }
}
