//! NDT alignment manager using our CubeCL implementation.

// Allow dead_code: NdtManager methods are called from main.rs callbacks.
// Rust doesn't track usage through Arc<Mutex<T>> and closure captures.
#![allow(dead_code)]

use crate::params::NdtParams;
use anyhow::{bail, Result};
use geometry_msgs::msg::{Point, Pose, Quaternion};
use nalgebra::{Isometry3, Matrix6, Quaternion as NaQuaternion, Translation3, UnitQuaternion};
pub use ndt_cuda::AlignmentDebug;
use ndt_cuda::NdtScanMatcher;
use rclrs::log_debug;

const LOGGER_NAME: &str = "ndt_scan_matcher.ndt_manager";

/// Extract yaw from quaternion for debug logging
fn quaternion_to_yaw(q: &Quaternion) -> f64 {
    let unit_q = UnitQuaternion::new_normalize(NaQuaternion::new(q.w, q.x, q.y, q.z));
    unit_q.euler_angles().2
}

/// Result of NDT alignment with Hessian for covariance estimation
pub struct AlignResult {
    pub pose: Pose,
    pub converged: bool,
    pub score: f64,
    pub iterations: i32,
    /// Hessian matrix from NDT optimization (for Laplace covariance estimation)
    pub hessian: [[f64; 6]; 6],
    /// NVTL score
    pub nvtl: f64,
    /// Transform probability
    pub transform_probability: f64,
    /// Number of correspondences
    pub num_correspondences: usize,
    /// Oscillation count (direction reversals during optimization)
    pub oscillation_count: usize,
}

/// NDT alignment manager using our CubeCL implementation
pub struct NdtManager {
    matcher: NdtScanMatcher,
}

impl NdtManager {
    /// Create new NDT manager with parameters
    pub fn new(params: &NdtParams) -> Result<Self> {
        let matcher = NdtScanMatcher::builder()
            .resolution(params.ndt.resolution as f32)
            .max_iterations(params.ndt.max_iterations as usize)
            .transformation_epsilon(params.ndt.trans_epsilon)
            .step_size(params.ndt.step_size) // From config (default 0.1)
            .outlier_ratio(0.55) // Autoware default
            .regularization_enabled(params.regularization.enabled)
            .regularization_scale_factor(params.regularization.scale_factor)
            .build()?;

        if params.regularization.enabled {
            log_debug!(
                LOGGER_NAME,
                "GNSS regularization enabled with scale factor {}",
                params.regularization.scale_factor
            );
        }

        Ok(Self { matcher })
    }

    /// Set target (map) point cloud
    pub fn set_target(&mut self, points: &[[f32; 3]]) -> Result<()> {
        self.matcher.set_target(points)
    }

    /// Check if a target has been set
    pub fn has_target(&self) -> bool {
        self.matcher.has_target()
    }

    /// Align source points to target with initial guess
    pub fn align(
        &mut self,
        source_points: &[[f32; 3]],
        _target_points: &[[f32; 3]], // Not needed, we use stored target
        initial_pose: &Pose,
    ) -> Result<AlignResult> {
        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Debug: log initial pose
        let initial_yaw = quaternion_to_yaw(&initial_pose.orientation);
        log_debug!(
            LOGGER_NAME,
            "Initial: pos=({:.2}, {:.2}, {:.2}), yaw={:.1}°",
            initial_pose.position.x,
            initial_pose.position.y,
            initial_pose.position.z,
            initial_yaw.to_degrees()
        );

        // Convert initial pose to isometry
        let initial_guess = pose_to_isometry(initial_pose);

        // Run alignment
        let result = self.matcher.align(source_points, initial_guess)?;

        // Convert result to ROS pose
        let pose = isometry_to_pose(&result.pose);

        // Debug: log result pose
        let result_yaw = quaternion_to_yaw(&pose.orientation);
        log_debug!(
            LOGGER_NAME,
            "Result:  pos=({:.2}, {:.2}, {:.2}), yaw={:.1}°, iter={}, score={:.3}, converged={}",
            pose.position.x,
            pose.position.y,
            pose.position.z,
            result_yaw.to_degrees(),
            result.iterations,
            result.score,
            result.converged
        );

        // Convert Hessian from nalgebra to array
        let hessian = matrix6_to_array(&result.hessian);

        Ok(AlignResult {
            pose,
            converged: result.converged,
            score: result.score,
            iterations: result.iterations as i32,
            hessian,
            nvtl: result.nvtl,
            transform_probability: result.transform_probability,
            num_correspondences: result.num_correspondences,
            oscillation_count: result.oscillation_count,
        })
    }

    /// Align source points to target with initial guess and debug output.
    ///
    /// This is the same as `align()` but also returns detailed iteration debug info.
    pub fn align_with_debug(
        &mut self,
        source_points: &[[f32; 3]],
        _target_points: &[[f32; 3]], // Not needed, we use stored target
        initial_pose: &Pose,
        timestamp_ns: u64,
    ) -> Result<(AlignResult, AlignmentDebug)> {
        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Convert initial pose to isometry
        let initial_guess = pose_to_isometry(initial_pose);

        // Run alignment with debug
        let (result, debug) =
            self.matcher
                .align_with_debug(source_points, initial_guess, timestamp_ns)?;

        // Convert result to ROS pose
        let pose = isometry_to_pose(&result.pose);

        // Convert Hessian from nalgebra to array
        let hessian = matrix6_to_array(&result.hessian);

        Ok((
            AlignResult {
                pose,
                converged: result.converged,
                score: result.score,
                iterations: result.iterations as i32,
                hessian,
                nvtl: result.nvtl,
                transform_probability: result.transform_probability,
                num_correspondences: result.num_correspondences,
                oscillation_count: result.oscillation_count,
            },
            debug,
        ))
    }

    /// Evaluate NVTL score at a given pose
    pub fn evaluate_nvtl(
        &self,
        source_points: &[[f32; 3]],
        _target_points: &[[f32; 3]], // Not needed, we use stored target
        pose: &Pose,
        _outlier_ratio: f64, // Already configured
    ) -> Result<f64> {
        let isometry = pose_to_isometry(pose);
        self.matcher.evaluate_nvtl(source_points, &isometry)
    }

    /// Evaluate NVTL at multiple poses in parallel (GPU-accelerated via Rayon).
    ///
    /// This is optimized for multi-NDT covariance estimation where we need
    /// to evaluate NVTL at many offset poses quickly.
    pub fn evaluate_nvtl_batch(
        &self,
        source_points: &[[f32; 3]],
        poses: &[Pose],
    ) -> Result<Vec<f64>> {
        let isometries: Vec<Isometry3<f64>> = poses.iter().map(pose_to_isometry).collect();
        self.matcher.evaluate_nvtl_batch(source_points, &isometries)
    }

    /// Align from multiple initial poses in parallel and return all results.
    ///
    /// This is useful for multi-NDT covariance estimation where we need
    /// to run alignment from multiple offset poses.
    pub fn align_batch(
        &self,
        source_points: &[[f32; 3]],
        initial_poses: &[Pose],
    ) -> Result<Vec<AlignResult>> {
        let isometries: Vec<Isometry3<f64>> = initial_poses.iter().map(pose_to_isometry).collect();
        let results = self.matcher.align_batch(source_points, &isometries)?;

        Ok(results
            .into_iter()
            .map(|r| {
                let pose = isometry_to_pose(&r.pose);
                let hessian = matrix6_to_array(&r.hessian);
                AlignResult {
                    pose,
                    converged: r.converged,
                    score: r.score,
                    iterations: r.iterations as i32,
                    hessian,
                    nvtl: r.nvtl,
                    transform_probability: r.transform_probability,
                    num_correspondences: r.num_correspondences,
                    oscillation_count: r.oscillation_count,
                }
            })
            .collect())
    }

    /// Set the regularization reference pose (from GNSS).
    ///
    /// When regularization is enabled, this pose is used to penalize deviation
    /// in the vehicle's longitudinal direction, helping prevent drift in open areas.
    pub fn set_regularization_pose(&mut self, pose: &Pose) {
        let isometry = pose_to_isometry(pose);
        self.matcher.set_regularization_pose(isometry);
        log_debug!(
            LOGGER_NAME,
            "Regularization pose set: ({:.2}, {:.2}, {:.2})",
            pose.position.x,
            pose.position.y,
            pose.position.z
        );
    }

    /// Clear the regularization reference pose.
    pub fn clear_regularization_pose(&mut self) {
        self.matcher.clear_regularization_pose();
        log_debug!(LOGGER_NAME, "Regularization pose cleared");
    }

    /// Check if regularization is enabled.
    pub fn is_regularization_enabled(&self) -> bool {
        self.matcher.is_regularization_enabled()
    }
}

fn matrix6_to_array(m: &Matrix6<f64>) -> [[f64; 6]; 6] {
    let mut result = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            result[i][j] = m[(i, j)];
        }
    }
    result
}

/// Convert ROS Pose to nalgebra Isometry3
fn pose_to_isometry(pose: &Pose) -> Isometry3<f64> {
    let p = &pose.position;
    let q = &pose.orientation;

    let translation = Translation3::new(p.x, p.y, p.z);
    let quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));

    Isometry3::from_parts(translation, quaternion)
}

/// Convert nalgebra Isometry3 to ROS Pose
fn isometry_to_pose(isometry: &Isometry3<f64>) -> Pose {
    let translation = isometry.translation;
    let quaternion = isometry.rotation;

    Pose {
        position: Point {
            x: translation.x,
            y: translation.y,
            z: translation.z,
        },
        orientation: Quaternion {
            x: quaternion.i,
            y: quaternion.j,
            z: quaternion.k,
            w: quaternion.w,
        },
    }
}
