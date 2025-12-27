//! NDT alignment manager using our CubeCL implementation.

use crate::params::NdtParams;
use anyhow::{bail, Result};
use geometry_msgs::msg::{Point, Pose, Quaternion};
use nalgebra::{Isometry3, Matrix6, Quaternion as NaQuaternion, Translation3, UnitQuaternion};
use ndt_cuda::NdtScanMatcher;

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
            .build()?;

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

        // Convert initial pose to isometry
        let initial_guess = pose_to_isometry(initial_pose);

        // Run alignment
        let result = self.matcher.align(source_points, initial_guess)?;

        // Convert result to ROS pose
        let pose = isometry_to_pose(&result.pose);

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
        })
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
