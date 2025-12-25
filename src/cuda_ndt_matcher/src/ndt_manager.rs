//! NDT alignment manager using fast-gicp

use crate::params::NdtParams;
use anyhow::{bail, Result};
use fast_gicp::{
    Hessian6x6, NDTCuda, NdtDistanceMode, NeighborSearchMethod, PointCloudXYZ, Transform3f,
};
use geometry_msgs::msg::{Point, Pose, Quaternion};
use nalgebra::{Isometry3, Quaternion as NaQuaternion, Translation3, UnitQuaternion};

/// Result of NDT alignment with Hessian for covariance estimation
pub struct AlignResult {
    pub pose: Pose,
    pub converged: bool,
    pub score: f64,
    pub iterations: i32,
    /// Hessian matrix from NDT optimization (for Laplace covariance estimation)
    pub hessian: Hessian6x6,
    /// The final transformation as Transform3f (for multi-NDT covariance estimation)
    pub final_transform: Transform3f,
}

/// NDT alignment manager
pub struct NdtManager {
    ndt: NDTCuda,
    has_target: bool,
}

impl NdtManager {
    /// Create new NDT manager with parameters
    pub fn new(params: &NdtParams) -> Result<Self> {
        let ndt = NDTCuda::builder()
            .resolution(params.ndt.resolution)
            .max_iterations(params.ndt.max_iterations as u32)
            .transformation_epsilon(params.ndt.trans_epsilon)
            .distance_mode(NdtDistanceMode::P2D)
            .neighbor_search_method(NeighborSearchMethod::Direct7)
            .build()?;

        Ok(Self {
            ndt,
            has_target: false,
        })
    }

    /// Set target (map) point cloud
    pub fn set_target(&mut self, points: &[[f32; 3]]) -> Result<()> {
        if points.is_empty() {
            bail!("Target point cloud is empty");
        }
        self.has_target = true;
        Ok(())
    }

    /// Align source points to target with initial guess
    pub fn align(
        &mut self,
        source_points: &[[f32; 3]],
        target_points: &[[f32; 3]],
        initial_pose: &Pose,
    ) -> Result<AlignResult> {
        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        if target_points.is_empty() {
            bail!("Target point cloud is empty");
        }

        let source = PointCloudXYZ::from_points(source_points);
        let target = PointCloudXYZ::from_points(target_points);

        // Convert initial pose to transform
        let initial_transform = pose_to_transform(initial_pose);

        // Run alignment with full result (includes Hessian)
        let ndt_result =
            self.ndt
                .align_with_guess_full(&source, &target, Some(&initial_transform))?;

        // Convert result transform to pose
        let pose = transform_to_pose(&ndt_result.result.final_transformation);

        Ok(AlignResult {
            pose,
            converged: ndt_result.result.has_converged,
            score: ndt_result.result.fitness_score,
            iterations: ndt_result.result.num_iterations,
            hessian: ndt_result.hessian,
            final_transform: ndt_result.result.final_transformation,
        })
    }

    /// Get the NDT instance for covariance estimation
    pub fn ndt(&self) -> &NDTCuda {
        &self.ndt
    }

    /// Evaluate NVTL (Nearest Voxel Transformation Likelihood) score at a given pose.
    ///
    /// NVTL is Autoware's metric for evaluating alignment quality.
    /// Higher scores indicate better alignment (typically in range [0, ~5]).
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor data)
    /// * `target_points` - Target point cloud (map data)
    /// * `pose` - The pose to evaluate
    /// * `outlier_ratio` - Outlier ratio for Gaussian parameters (Autoware default: 0.55)
    ///
    /// # Returns
    /// NVTL score (higher = better alignment)
    pub fn evaluate_nvtl(
        &self,
        source_points: &[[f32; 3]],
        target_points: &[[f32; 3]],
        pose: &Pose,
        outlier_ratio: f64,
    ) -> Result<f64> {
        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }
        if target_points.is_empty() {
            bail!("Target point cloud is empty");
        }

        let source = PointCloudXYZ::from_points(source_points);
        let target = PointCloudXYZ::from_points(target_points);
        let transform = pose_to_transform(pose);

        let nvtl = self
            .ndt
            .evaluate_nvtl(&source, &target, &transform, outlier_ratio)?;
        Ok(nvtl)
    }
}

/// Convert ROS Pose to fast-gicp Transform3f using nalgebra
fn pose_to_transform(pose: &Pose) -> Transform3f {
    let p = &pose.position;
    let q = &pose.orientation;

    // Create nalgebra types
    let translation = Translation3::new(p.x, p.y, p.z);
    let quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));

    // Create isometry (rotation + translation)
    let isometry = Isometry3::from_parts(translation, quaternion);

    // Convert to 4x4 matrix (row-major for fast-gicp)
    let matrix = isometry.to_homogeneous();

    Transform3f::from_flat(&[
        matrix[(0, 0)] as f32,
        matrix[(0, 1)] as f32,
        matrix[(0, 2)] as f32,
        matrix[(0, 3)] as f32,
        matrix[(1, 0)] as f32,
        matrix[(1, 1)] as f32,
        matrix[(1, 2)] as f32,
        matrix[(1, 3)] as f32,
        matrix[(2, 0)] as f32,
        matrix[(2, 1)] as f32,
        matrix[(2, 2)] as f32,
        matrix[(2, 3)] as f32,
        matrix[(3, 0)] as f32,
        matrix[(3, 1)] as f32,
        matrix[(3, 2)] as f32,
        matrix[(3, 3)] as f32,
    ])
}

/// Convert fast-gicp Transform3f to ROS Pose using nalgebra
fn transform_to_pose(transform: &Transform3f) -> Pose {
    let m = transform.to_flat();

    // Reconstruct 4x4 matrix from flat array (row-major)
    let matrix = nalgebra::Matrix4::new(
        m[0] as f64,
        m[1] as f64,
        m[2] as f64,
        m[3] as f64,
        m[4] as f64,
        m[5] as f64,
        m[6] as f64,
        m[7] as f64,
        m[8] as f64,
        m[9] as f64,
        m[10] as f64,
        m[11] as f64,
        m[12] as f64,
        m[13] as f64,
        m[14] as f64,
        m[15] as f64,
    );

    // Extract rotation matrix (top-left 3x3)
    let rotation_matrix = matrix.fixed_view::<3, 3>(0, 0).into_owned();

    // Convert rotation matrix to quaternion using nalgebra
    let rotation = nalgebra::Rotation3::from_matrix_unchecked(rotation_matrix);
    let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);

    // Extract translation
    let position = Point {
        x: matrix[(0, 3)],
        y: matrix[(1, 3)],
        z: matrix[(2, 3)],
    };

    // Convert nalgebra quaternion to ROS quaternion
    let orientation = Quaternion {
        x: quaternion.i,
        y: quaternion.j,
        z: quaternion.k,
        w: quaternion.w,
    };

    Pose {
        position,
        orientation,
    }
}
