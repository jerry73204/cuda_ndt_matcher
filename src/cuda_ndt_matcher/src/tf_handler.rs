//! TF2 Transform Handler for ROS2 Rust.
//!
//! This module provides a simple TF2 buffer that subscribes to `/tf` and `/tf_static`
//! topics and allows transform lookups between frames.
//!
//! # Example
//!
//! ```ignore
//! use cuda_ndt_matcher::tf_handler::TfHandler;
//!
//! let tf_handler = TfHandler::new(&node)?;
//!
//! // Look up transform from sensor to base_link
//! if let Some(transform) = tf_handler.lookup_transform("velodyne", "base_link", None) {
//!     // Use transform to transform points
//! }
//! ```

use geometry_msgs::msg::Transform;
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use parking_lot::RwLock;
use rclrs::{log_debug, log_warn, Node, Subscription};
use std::collections::HashMap;
use std::sync::Arc;
use tf2_msgs::msg::TFMessage;

const LOGGER_NAME: &str = "ndt_scan_matcher.tf_handler";

/// Type alias for the transform buffer: (parent_frame, child_frame) -> list of transforms
type TransformBuffer = HashMap<(String, String), Vec<TimestampedTransform>>;

/// Maximum age of a transform before it's considered stale (seconds).
const MAX_TRANSFORM_AGE_SECS: f64 = 10.0;

/// A single transform with timestamp.
#[derive(Clone, Debug)]
struct TimestampedTransform {
    /// The transform data.
    transform: Transform,
    /// Timestamp in nanoseconds since epoch.
    stamp_ns: i64,
    /// Whether this is a static transform.
    is_static: bool,
}

/// TF2 transform buffer and handler.
///
/// Subscribes to `/tf` and `/tf_static` and maintains a buffer of transforms
/// for lookup operations.
#[allow(dead_code)]
pub struct TfHandler {
    /// Transform buffer: (parent_frame, child_frame) -> transform
    /// Multiple transforms can exist for the same frame pair (at different times)
    buffer: Arc<RwLock<TransformBuffer>>,

    /// Subscription to /tf (dynamic transforms)
    tf_sub: Subscription<TFMessage>,

    /// Subscription to /tf_static (static transforms)
    tf_static_sub: Subscription<TFMessage>,
}

impl TfHandler {
    /// Create a new TF handler with subscriptions to /tf and /tf_static.
    pub fn new(node: &Node) -> anyhow::Result<Arc<Self>> {
        let buffer = Arc::new(RwLock::new(HashMap::new()));

        // Create subscriptions
        let buffer_clone = Arc::clone(&buffer);
        let tf_sub = node.create_subscription::<TFMessage, _>("/tf", move |msg: TFMessage| {
            Self::on_tf_message(&buffer_clone, msg, false);
        })?;

        let buffer_clone = Arc::clone(&buffer);
        let tf_static_sub =
            node.create_subscription::<TFMessage, _>("/tf_static", move |msg: TFMessage| {
                Self::on_tf_message(&buffer_clone, msg, true);
            })?;

        log_debug!(
            LOGGER_NAME,
            "TF handler initialized, listening to /tf and /tf_static"
        );

        Ok(Arc::new(Self {
            buffer,
            tf_sub,
            tf_static_sub,
        }))
    }

    /// Handle incoming TF messages.
    fn on_tf_message(buffer: &Arc<RwLock<TransformBuffer>>, msg: TFMessage, is_static: bool) {
        let mut buf = buffer.write();

        for ts in msg.transforms {
            let key = (ts.header.frame_id.clone(), ts.child_frame_id.clone());
            let stamp_ns =
                ts.header.stamp.sec as i64 * 1_000_000_000 + ts.header.stamp.nanosec as i64;

            let timestamped = TimestampedTransform {
                transform: ts.transform,
                stamp_ns,
                is_static,
            };

            let transforms = buf.entry(key).or_default();

            if is_static {
                // Static transforms replace any existing
                transforms.clear();
                transforms.push(timestamped);
            } else {
                // Keep a limited history of dynamic transforms
                transforms.push(timestamped);

                // Limit buffer size to avoid memory growth
                if transforms.len() > 100 {
                    transforms.remove(0);
                }
            }
        }
    }

    /// Look up a transform from source_frame to target_frame.
    ///
    /// # Arguments
    /// * `source_frame` - The frame to transform from (e.g., "velodyne")
    /// * `target_frame` - The frame to transform to (e.g., "base_link")
    /// * `time_ns` - Optional timestamp in nanoseconds. If None, uses latest transform.
    ///
    /// # Returns
    /// The transform if found, None otherwise.
    pub fn lookup_transform(
        &self,
        source_frame: &str,
        target_frame: &str,
        time_ns: Option<i64>,
    ) -> Option<Isometry3<f64>> {
        // Normalize frame names (remove leading slashes)
        let source = source_frame.trim_start_matches('/');
        let target = target_frame.trim_start_matches('/');

        // Same frame - return identity
        if source == target {
            return Some(Isometry3::identity());
        }

        let buf = self.buffer.read();

        // Try direct transform: source -> target
        if let Some(transform) = self.find_transform(&buf, source, target, time_ns) {
            return Some(transform);
        }

        // Try inverse transform: target -> source (then invert)
        if let Some(transform) = self.find_transform(&buf, target, source, time_ns) {
            return Some(transform.inverse());
        }

        // Try common parent frames (base_link, map, etc.)
        for common_parent in ["base_link", "base_footprint", "sensor_kit_base_link"] {
            // source -> common_parent -> target
            if let (Some(source_to_parent), Some(target_to_parent)) = (
                self.find_transform(&buf, source, common_parent, time_ns)
                    .or_else(|| {
                        self.find_transform(&buf, common_parent, source, time_ns)
                            .map(|t| t.inverse())
                    }),
                self.find_transform(&buf, target, common_parent, time_ns)
                    .or_else(|| {
                        self.find_transform(&buf, common_parent, target, time_ns)
                            .map(|t| t.inverse())
                    }),
            ) {
                // source -> common_parent, then common_parent -> target
                // (target -> common_parent)^-1 = common_parent -> target
                return Some(target_to_parent.inverse() * source_to_parent);
            }
        }

        log_debug!(
            LOGGER_NAME,
            "Transform not found: {} -> {}",
            source_frame,
            target_frame
        );
        None
    }

    /// Find a direct transform in the buffer.
    fn find_transform(
        &self,
        buf: &HashMap<(String, String), Vec<TimestampedTransform>>,
        parent: &str,
        child: &str,
        time_ns: Option<i64>,
    ) -> Option<Isometry3<f64>> {
        let key = (parent.to_string(), child.to_string());
        let transforms = buf.get(&key)?;

        if transforms.is_empty() {
            return None;
        }

        // Find the best matching transform
        let transform = if let Some(target_time) = time_ns {
            // Find closest transform to requested time
            transforms
                .iter()
                .min_by_key(|t| (t.stamp_ns - target_time).abs())?
        } else {
            // Use latest transform
            transforms.iter().max_by_key(|t| t.stamp_ns)?
        };

        // Check if transform is too old (only for dynamic transforms)
        if let Some(query_time) = time_ns {
            if !transform.is_static {
                let age_secs = (query_time - transform.stamp_ns).abs() as f64 / 1e9;
                if age_secs > MAX_TRANSFORM_AGE_SECS {
                    log_warn!(
                        LOGGER_NAME,
                        "Transform {} -> {} is {:.1}s old (max: {:.1}s)",
                        parent,
                        child,
                        age_secs,
                        MAX_TRANSFORM_AGE_SECS
                    );
                }
            }
        }

        Some(Self::transform_to_isometry(&transform.transform))
    }

    /// Convert ROS Transform to nalgebra Isometry3.
    fn transform_to_isometry(transform: &Transform) -> Isometry3<f64> {
        let translation = Translation3::new(
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        );

        let quaternion = UnitQuaternion::from_quaternion(Quaternion::new(
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
        ));

        Isometry3::from_parts(translation, quaternion)
    }

    /// Transform a point cloud from source_frame to target_frame.
    ///
    /// # Arguments
    /// * `points` - Points to transform
    /// * `source_frame` - Frame the points are in
    /// * `target_frame` - Frame to transform to
    /// * `time_ns` - Optional timestamp
    ///
    /// # Returns
    /// Transformed points if transform is available, None otherwise.
    pub fn transform_points(
        &self,
        points: &[[f32; 3]],
        source_frame: &str,
        target_frame: &str,
        time_ns: Option<i64>,
    ) -> Option<Vec<[f32; 3]>> {
        let transform = self.lookup_transform(source_frame, target_frame, time_ns)?;

        let transformed: Vec<[f32; 3]> = points
            .iter()
            .map(|p| {
                let pt = nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                let transformed = transform * pt;
                [
                    transformed.x as f32,
                    transformed.y as f32,
                    transformed.z as f32,
                ]
            })
            .collect();

        Some(transformed)
    }

    /// Check if a transform is available between two frames.
    #[allow(dead_code)]
    pub fn can_transform(&self, source_frame: &str, target_frame: &str) -> bool {
        self.lookup_transform(source_frame, target_frame, None)
            .is_some()
    }

    /// Get the number of frame pairs in the buffer.
    #[allow(dead_code)]
    pub fn buffer_size(&self) -> usize {
        self.buffer.read().len()
    }

    /// Get all available frames.
    #[allow(dead_code)]
    pub fn get_frames(&self) -> Vec<String> {
        let buf = self.buffer.read();
        let mut frames: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (parent, child) in buf.keys() {
            frames.insert(parent.clone());
            frames.insert(child.clone());
        }

        frames.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_to_isometry_identity() {
        let transform = Transform {
            translation: geometry_msgs::msg::Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: geometry_msgs::msg::Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };

        let isometry = TfHandler::transform_to_isometry(&transform);

        assert!((isometry.translation.x).abs() < 1e-10);
        assert!((isometry.translation.y).abs() < 1e-10);
        assert!((isometry.translation.z).abs() < 1e-10);
    }

    #[test]
    fn test_transform_to_isometry_translation() {
        let transform = Transform {
            translation: geometry_msgs::msg::Vector3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            rotation: geometry_msgs::msg::Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };

        let isometry = TfHandler::transform_to_isometry(&transform);

        assert!((isometry.translation.x - 1.0).abs() < 1e-10);
        assert!((isometry.translation.y - 2.0).abs() < 1e-10);
        assert!((isometry.translation.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_to_isometry_rotation_90_z() {
        // 90 degree rotation around Z axis
        let angle = std::f64::consts::FRAC_PI_2;
        let transform = Transform {
            translation: geometry_msgs::msg::Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: geometry_msgs::msg::Quaternion {
                x: 0.0,
                y: 0.0,
                z: (angle / 2.0).sin(),
                w: (angle / 2.0).cos(),
            },
        };

        let isometry = TfHandler::transform_to_isometry(&transform);

        // Transform point (1, 0, 0) should become (0, 1, 0)
        let pt = nalgebra::Point3::new(1.0, 0.0, 0.0);
        let transformed = isometry * pt;

        assert!((transformed.x - 0.0).abs() < 1e-10);
        assert!((transformed.y - 1.0).abs() < 1e-10);
        assert!((transformed.z - 0.0).abs() < 1e-10);
    }
}
