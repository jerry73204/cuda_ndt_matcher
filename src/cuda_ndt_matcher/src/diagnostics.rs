//! Diagnostics interface for NDT scan matcher.
//!
//! Mirrors Autoware's DiagnosticsInterface pattern for publishing diagnostic
//! status to `/diagnostics` topic.
//!
//! Note: Some methods/fields are public API for future use (e.g., additional
//! diagnostic categories beyond scan_matching_status).

// Allow dead_code: DiagnosticsInterface methods are called from main.rs.
// Some diagnostic categories are defined for future expansion.
#![allow(dead_code)]

use diagnostic_msgs::msg::{DiagnosticArray, DiagnosticStatus, KeyValue};
use rclrs::{Node, Publisher};
use std::time::Instant;

/// Diagnostic severity levels matching ROS diagnostic_msgs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DiagnosticLevel {
    Ok = 0,
    Warn = 1,
    Error = 2,
    Stale = 3,
}

impl From<DiagnosticLevel> for u8 {
    fn from(level: DiagnosticLevel) -> u8 {
        level as u8
    }
}

/// A single diagnostic category (e.g., "scan_matching_status").
pub struct DiagnosticCategory {
    name: String,
    hardware_id: String,
    level: DiagnosticLevel,
    message: String,
    key_values: Vec<(String, String)>,
}

impl DiagnosticCategory {
    /// Create a new diagnostic category.
    pub fn new(name: &str, hardware_id: &str) -> Self {
        Self {
            name: name.to_string(),
            hardware_id: hardware_id.to_string(),
            level: DiagnosticLevel::Ok,
            message: String::new(),
            key_values: Vec::new(),
        }
    }

    /// Clear all key-value pairs and reset level.
    pub fn clear(&mut self) {
        self.key_values.clear();
        self.level = DiagnosticLevel::Ok;
        self.message.clear();
    }

    /// Add a key-value pair.
    pub fn add_key_value(&mut self, key: &str, value: impl ToString) {
        self.key_values.push((key.to_string(), value.to_string()));
    }

    /// Update the diagnostic level and message.
    /// Only updates if the new level is more severe than current.
    pub fn update_level_and_message(&mut self, level: DiagnosticLevel, message: &str) {
        if level as u8 > self.level as u8 {
            self.level = level;
            self.message = message.to_string();
        }
    }

    /// Set the diagnostic level and message unconditionally.
    pub fn set_level_and_message(&mut self, level: DiagnosticLevel, message: &str) {
        self.level = level;
        self.message = message.to_string();
    }

    /// Build the DiagnosticStatus message.
    fn to_status(&self) -> DiagnosticStatus {
        DiagnosticStatus {
            level: self.level as u8,
            name: self.name.clone(),
            message: self.message.clone(),
            hardware_id: self.hardware_id.clone(),
            values: self
                .key_values
                .iter()
                .map(|(k, v)| KeyValue {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
        }
    }
}

/// Diagnostics interface for NDT scan matcher.
///
/// Provides methods to publish diagnostic status matching Autoware's format.
pub struct DiagnosticsInterface {
    publisher: Publisher<DiagnosticArray>,
    scan_matching: DiagnosticCategory,
    initial_pose: DiagnosticCategory,
    regularization_pose: DiagnosticCategory,
    map_update: DiagnosticCategory,
    trigger_node: DiagnosticCategory,
}

impl DiagnosticsInterface {
    /// Create diagnostics interface with publisher.
    pub fn new(node: &Node) -> Result<Self, rclrs::RclrsError> {
        let publisher = node.create_publisher("/diagnostics")?;
        let hardware_id = "ndt_scan_matcher";

        Ok(Self {
            publisher,
            scan_matching: DiagnosticCategory::new("scan_matching_status", hardware_id),
            initial_pose: DiagnosticCategory::new("initial_pose_subscriber_status", hardware_id),
            regularization_pose: DiagnosticCategory::new(
                "regularization_pose_subscriber_status",
                hardware_id,
            ),
            map_update: DiagnosticCategory::new("map_update_status", hardware_id),
            trigger_node: DiagnosticCategory::new("trigger_node_service_status", hardware_id),
        })
    }

    /// Get mutable reference to scan matching diagnostics.
    pub fn scan_matching_mut(&mut self) -> &mut DiagnosticCategory {
        &mut self.scan_matching
    }

    /// Get mutable reference to initial pose diagnostics.
    pub fn initial_pose_mut(&mut self) -> &mut DiagnosticCategory {
        &mut self.initial_pose
    }

    /// Get mutable reference to regularization pose diagnostics.
    pub fn regularization_pose_mut(&mut self) -> &mut DiagnosticCategory {
        &mut self.regularization_pose
    }

    /// Get mutable reference to map update diagnostics.
    pub fn map_update_mut(&mut self) -> &mut DiagnosticCategory {
        &mut self.map_update
    }

    /// Get mutable reference to trigger node diagnostics.
    pub fn trigger_node_mut(&mut self) -> &mut DiagnosticCategory {
        &mut self.trigger_node
    }

    /// Publish all diagnostic categories.
    pub fn publish(&self, stamp: builtin_interfaces::msg::Time) {
        let msg = DiagnosticArray {
            header: std_msgs::msg::Header {
                stamp,
                frame_id: String::new(),
            },
            status: vec![
                self.scan_matching.to_status(),
                self.initial_pose.to_status(),
                self.regularization_pose.to_status(),
                self.map_update.to_status(),
                self.trigger_node.to_status(),
            ],
        };
        let _ = self.publisher.publish(msg);
    }

    /// Publish only scan matching diagnostics (most common case).
    pub fn publish_scan_matching(&self, stamp: builtin_interfaces::msg::Time) {
        let msg = DiagnosticArray {
            header: std_msgs::msg::Header {
                stamp,
                frame_id: String::new(),
            },
            status: vec![self.scan_matching.to_status()],
        };
        let _ = self.publisher.publish(msg);
    }
}

/// Helper to track execution time for diagnostics.
pub struct ExecutionTimer {
    start: Instant,
}

impl ExecutionTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

impl Default for ExecutionTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of scan matching diagnostics collection.
#[derive(Debug, Clone)]
pub struct ScanMatchingDiagnostics {
    pub topic_time_stamp: f64,
    pub sensor_points_size: usize,
    pub sensor_points_delay_time_sec: f64,
    pub is_succeed_transform_sensor_points: bool,
    pub sensor_points_max_distance: f64,
    pub is_activated: bool,
    pub is_succeed_interpolate_initial_pose: bool,
    pub is_set_map_points: bool,
    pub iteration_num: i32,
    pub oscillation_count: usize,
    pub transform_probability: f64,
    pub nearest_voxel_transformation_likelihood: f64,
    /// Scores at initial pose (before alignment) for comparison
    pub transform_probability_before: f64,
    pub nearest_voxel_transformation_likelihood_before: f64,
    pub distance_initial_to_result: f64,
    pub execution_time_ms: f64,
    pub skipping_publish_num: i32,
    /// Per-iteration transform probability scores (from AlignmentDebug)
    pub transform_probability_array: Option<Vec<f64>>,
    /// Per-iteration NVTL scores (from AlignmentDebug)
    pub nearest_voxel_transformation_likelihood_array: Option<Vec<f64>>,
}

impl ScanMatchingDiagnostics {
    /// Apply diagnostics to a category.
    pub fn apply_to(&self, diag: &mut DiagnosticCategory) {
        diag.clear();

        // Add all key-value pairs
        diag.add_key_value("topic_time_stamp", format!("{:.6}", self.topic_time_stamp));
        diag.add_key_value("sensor_points_size", self.sensor_points_size);
        diag.add_key_value(
            "sensor_points_delay_time_sec",
            format!("{:.6}", self.sensor_points_delay_time_sec),
        );
        diag.add_key_value(
            "is_succeed_transform_sensor_points",
            self.is_succeed_transform_sensor_points,
        );
        diag.add_key_value(
            "sensor_points_max_distance",
            format!("{:.3}", self.sensor_points_max_distance),
        );
        diag.add_key_value("is_activated", self.is_activated);
        diag.add_key_value(
            "is_succeed_interpolate_initial_pose",
            self.is_succeed_interpolate_initial_pose,
        );
        diag.add_key_value("is_set_map_points", self.is_set_map_points);
        diag.add_key_value("iteration_num", self.iteration_num);
        diag.add_key_value(
            "local_optimal_solution_oscillation_num",
            self.oscillation_count,
        );
        diag.add_key_value(
            "transform_probability",
            format!("{:.6}", self.transform_probability),
        );
        diag.add_key_value(
            "transform_probability_before",
            format!("{:.6}", self.transform_probability_before),
        );
        diag.add_key_value(
            "transform_probability_diff",
            format!(
                "{:.6}",
                self.transform_probability - self.transform_probability_before
            ),
        );
        diag.add_key_value(
            "nearest_voxel_transformation_likelihood",
            format!("{:.6}", self.nearest_voxel_transformation_likelihood),
        );
        diag.add_key_value(
            "nearest_voxel_transformation_likelihood_before",
            format!("{:.6}", self.nearest_voxel_transformation_likelihood_before),
        );
        diag.add_key_value(
            "nearest_voxel_transformation_likelihood_diff",
            format!(
                "{:.6}",
                self.nearest_voxel_transformation_likelihood
                    - self.nearest_voxel_transformation_likelihood_before
            ),
        );
        diag.add_key_value(
            "distance_initial_to_result",
            format!("{:.6}", self.distance_initial_to_result),
        );
        diag.add_key_value("execution_time", format!("{:.3}", self.execution_time_ms));
        diag.add_key_value("skipping_publish_num", self.skipping_publish_num);

        // Add per-iteration arrays if available (from AlignmentDebug with debug-iteration feature)
        if let Some(tp_array) = &self.transform_probability_array {
            diag.add_key_value("transform_probability_array", format!("{:?}", tp_array));
        }
        if let Some(nvtl_array) = &self.nearest_voxel_transformation_likelihood_array {
            diag.add_key_value(
                "nearest_voxel_transformation_likelihood_array",
                format!("{:?}", nvtl_array),
            );
        }

        // Set level based on status
        if !self.is_activated {
            diag.set_level_and_message(DiagnosticLevel::Warn, "NDT is not activated");
        } else if !self.is_set_map_points {
            diag.set_level_and_message(DiagnosticLevel::Warn, "Map points not set");
        } else if !self.is_succeed_interpolate_initial_pose {
            diag.set_level_and_message(DiagnosticLevel::Warn, "Failed to interpolate initial pose");
        } else if !self.is_succeed_transform_sensor_points {
            diag.set_level_and_message(DiagnosticLevel::Error, "Failed to transform sensor points");
        } else if self.oscillation_count > 10 {
            diag.set_level_and_message(
                DiagnosticLevel::Warn,
                &format!("Oscillation detected: {} reversals", self.oscillation_count),
            );
        } else {
            diag.set_level_and_message(DiagnosticLevel::Ok, "OK");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_category() {
        let mut cat = DiagnosticCategory::new("test_status", "test_hw");
        cat.add_key_value("key1", "value1");
        cat.add_key_value("key2", 42);
        cat.update_level_and_message(DiagnosticLevel::Warn, "warning message");

        let status = cat.to_status();
        assert_eq!(status.name, "test_status");
        assert_eq!(status.hardware_id, "test_hw");
        assert_eq!(status.level, DiagnosticLevel::Warn as u8);
        assert_eq!(status.message, "warning message");
        assert_eq!(status.values.len(), 2);
    }

    #[test]
    fn test_level_only_increases() {
        let mut cat = DiagnosticCategory::new("test", "hw");
        cat.update_level_and_message(DiagnosticLevel::Warn, "warn");
        cat.update_level_and_message(DiagnosticLevel::Ok, "ok"); // Should not change

        assert_eq!(cat.level, DiagnosticLevel::Warn);
        assert_eq!(cat.message, "warn");

        cat.update_level_and_message(DiagnosticLevel::Error, "error"); // Should change
        assert_eq!(cat.level, DiagnosticLevel::Error);
        assert_eq!(cat.message, "error");
    }

    #[test]
    fn test_scan_matching_diagnostics() {
        let diag = ScanMatchingDiagnostics {
            topic_time_stamp: 1234567890.123,
            sensor_points_size: 10000,
            sensor_points_delay_time_sec: 0.05,
            is_succeed_transform_sensor_points: true,
            sensor_points_max_distance: 120.0,
            is_activated: true,
            is_succeed_interpolate_initial_pose: true,
            is_set_map_points: true,
            iteration_num: 5,
            oscillation_count: 0,
            transform_probability: 3.5,
            nearest_voxel_transformation_likelihood: 2.8,
            transform_probability_before: 2.0,
            nearest_voxel_transformation_likelihood_before: 1.5,
            distance_initial_to_result: 0.15,
            execution_time_ms: 45.2,
            skipping_publish_num: 0,
            transform_probability_array: None,
            nearest_voxel_transformation_likelihood_array: None,
        };

        let mut cat = DiagnosticCategory::new("scan_matching_status", "ndt");
        diag.apply_to(&mut cat);

        assert_eq!(cat.level, DiagnosticLevel::Ok);
        // 15 original + 4 new (before/diff for transform_prob and nvtl) = 19
        assert_eq!(cat.key_values.len(), 19);
    }

    #[test]
    fn test_scan_matching_diagnostics_with_arrays() {
        let diag = ScanMatchingDiagnostics {
            topic_time_stamp: 1234567890.123,
            sensor_points_size: 10000,
            sensor_points_delay_time_sec: 0.05,
            is_succeed_transform_sensor_points: true,
            sensor_points_max_distance: 120.0,
            is_activated: true,
            is_succeed_interpolate_initial_pose: true,
            is_set_map_points: true,
            iteration_num: 5,
            oscillation_count: 0,
            transform_probability: 3.5,
            nearest_voxel_transformation_likelihood: 2.8,
            transform_probability_before: 2.0,
            nearest_voxel_transformation_likelihood_before: 1.5,
            distance_initial_to_result: 0.15,
            execution_time_ms: 45.2,
            skipping_publish_num: 0,
            transform_probability_array: Some(vec![2.0, 2.5, 3.0, 3.5]),
            nearest_voxel_transformation_likelihood_array: Some(vec![1.5, 2.0, 2.5, 2.8]),
        };

        let mut cat = DiagnosticCategory::new("scan_matching_status", "ndt");
        diag.apply_to(&mut cat);

        assert_eq!(cat.level, DiagnosticLevel::Ok);
        // 19 base + 2 arrays = 21
        assert_eq!(cat.key_values.len(), 21);

        // Verify array keys are present
        let keys: Vec<_> = cat.key_values.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"transform_probability_array"));
        assert!(keys.contains(&"nearest_voxel_transformation_likelihood_array"));
    }

    #[test]
    fn test_execution_timer() {
        let timer = ExecutionTimer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }
}
