//! NDT scan matcher parameters

use anyhow::Result;
use rclrs::Node;
use std::sync::Arc;

/// Frame configuration
pub struct FrameParams {
    pub base_frame: String,
    pub ndt_base_frame: String,
    pub map_frame: String,
}

/// Sensor points configuration
pub struct SensorPointsParams {
    pub timeout_sec: f64,
    pub required_distance: f32,
}

/// NDT algorithm configuration
pub struct NdtAlgorithmParams {
    pub trans_epsilon: f64,
    pub step_size: f64,
    pub resolution: f64,
    pub max_iterations: i32,
    pub num_threads: i32,
}

/// Initial pose estimation configuration
pub struct InitialPoseParams {
    pub particles_num: i32,
    pub n_startup_trials: i32,
}

/// Validation configuration
pub struct ValidationParams {
    pub initial_pose_timeout_sec: f64,
    pub initial_pose_distance_tolerance_m: f64,
    pub initial_to_result_distance_tolerance_m: f64,
    pub critical_upper_bound_exe_time_ms: f64,
    pub skipping_publish_num: i32,
}

/// Score estimation configuration
pub struct ScoreParams {
    pub converged_param_type: i32,
    pub converged_param_transform_probability: f64,
    pub converged_param_nearest_voxel_transformation_likelihood: f64,
}

/// Covariance estimation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CovarianceEstimationType {
    /// Use fixed covariance matrix
    Fixed = 0,
    /// Use Laplace approximation (inverse of Hessian)
    LaplaceApproximation = 1,
    /// Use multi-NDT (multiple alignments from offset poses)
    MultiNdt = 2,
    /// Use multi-NDT with score-based weighting (faster than MultiNdt)
    MultiNdtScore = 3,
}

impl CovarianceEstimationType {
    pub fn from_i32(value: i32) -> Self {
        match value {
            0 => Self::Fixed,
            1 => Self::LaplaceApproximation,
            2 => Self::MultiNdt,
            3 => Self::MultiNdtScore,
            _ => Self::Fixed, // Default to fixed for unknown values
        }
    }
}

/// Covariance estimation sub-parameters
pub struct CovarianceEstimationParams {
    /// Initial pose offset model X coordinates (meters)
    pub initial_pose_offset_model_x: Vec<f64>,
    /// Initial pose offset model Y coordinates (meters)
    pub initial_pose_offset_model_y: Vec<f64>,
    /// Softmax temperature for MULTI_NDT_SCORE (lower = sharper weights)
    pub temperature: f64,
    /// Scale factor for estimated covariance
    pub scale_factor: f64,
}

impl Default for CovarianceEstimationParams {
    fn default() -> Self {
        Self {
            // Default offsets matching Autoware reference
            initial_pose_offset_model_x: vec![0.0, 0.0, 0.5, -0.5, 1.0, -1.0],
            initial_pose_offset_model_y: vec![0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
            temperature: 0.05,
            scale_factor: 1.0,
        }
    }
}

/// Covariance configuration
pub struct CovarianceParams {
    /// Static 6x6 covariance matrix (used for FIXED mode and as fallback)
    pub output_pose_covariance: [f64; 36],
    /// Covariance estimation type
    pub covariance_estimation_type: CovarianceEstimationType,
    /// Estimation parameters for dynamic modes
    pub estimation: CovarianceEstimationParams,
}

/// Dynamic map loading configuration
pub struct DynamicMapParams {
    pub update_distance: f64,
    pub map_radius: f64,
    pub lidar_radius: f64,
}

/// All NDT parameters
pub struct NdtParams {
    pub frame: FrameParams,
    pub sensor_points: SensorPointsParams,
    pub ndt: NdtAlgorithmParams,
    pub initial_pose: InitialPoseParams,
    pub validation: ValidationParams,
    pub score: ScoreParams,
    pub covariance: CovarianceParams,
    pub dynamic_map: DynamicMapParams,
}

impl NdtParams {
    /// Load parameters from ROS node
    pub fn from_node(node: &Node) -> Result<Self> {
        Ok(Self {
            frame: FrameParams {
                base_frame: node
                    .declare_parameter::<Arc<str>>("frame.base_frame")
                    .default("base_link".into())
                    .mandatory()?
                    .get()
                    .to_string(),
                ndt_base_frame: node
                    .declare_parameter::<Arc<str>>("frame.ndt_base_frame")
                    .default("ndt_base_link".into())
                    .mandatory()?
                    .get()
                    .to_string(),
                map_frame: node
                    .declare_parameter::<Arc<str>>("frame.map_frame")
                    .default("map".into())
                    .mandatory()?
                    .get()
                    .to_string(),
            },
            sensor_points: SensorPointsParams {
                timeout_sec: node
                    .declare_parameter("sensor_points.timeout_sec")
                    .default(1.0)
                    .mandatory()?
                    .get(),
                required_distance: node
                    .declare_parameter("sensor_points.required_distance")
                    .default(10.0)
                    .mandatory()?
                    .get() as f32,
            },
            ndt: NdtAlgorithmParams {
                trans_epsilon: node
                    .declare_parameter("ndt.trans_epsilon")
                    .default(0.01)
                    .mandatory()?
                    .get(),
                step_size: node
                    .declare_parameter("ndt.step_size")
                    .default(0.1)
                    .mandatory()?
                    .get(),
                resolution: node
                    .declare_parameter("ndt.resolution")
                    .default(2.0)
                    .mandatory()?
                    .get(),
                max_iterations: node
                    .declare_parameter("ndt.max_iterations")
                    .default(30)
                    .mandatory()?
                    .get() as i32,
                num_threads: node
                    .declare_parameter("ndt.num_threads")
                    .default(4)
                    .mandatory()?
                    .get() as i32,
            },
            initial_pose: InitialPoseParams {
                particles_num: node
                    .declare_parameter("initial_pose_estimation.particles_num")
                    .default(200)
                    .mandatory()?
                    .get() as i32,
                n_startup_trials: node
                    .declare_parameter("initial_pose_estimation.n_startup_trials")
                    .default(100)
                    .mandatory()?
                    .get() as i32,
            },
            validation: ValidationParams {
                initial_pose_timeout_sec: node
                    .declare_parameter("validation.initial_pose_timeout_sec")
                    .default(1.0)
                    .mandatory()?
                    .get(),
                initial_pose_distance_tolerance_m: node
                    .declare_parameter("validation.initial_pose_distance_tolerance_m")
                    .default(10.0)
                    .mandatory()?
                    .get(),
                initial_to_result_distance_tolerance_m: node
                    .declare_parameter("validation.initial_to_result_distance_tolerance_m")
                    .default(3.0)
                    .mandatory()?
                    .get(),
                critical_upper_bound_exe_time_ms: node
                    .declare_parameter("validation.critical_upper_bound_exe_time_ms")
                    .default(100.0)
                    .mandatory()?
                    .get(),
                skipping_publish_num: node
                    .declare_parameter("validation.skipping_publish_num")
                    .default(5)
                    .mandatory()?
                    .get() as i32,
            },
            score: ScoreParams {
                converged_param_type: node
                    .declare_parameter("score_estimation.converged_param_type")
                    .default(1)
                    .mandatory()?
                    .get() as i32,
                converged_param_transform_probability: node
                    .declare_parameter("score_estimation.converged_param_transform_probability")
                    .default(3.0)
                    .mandatory()?
                    .get(),
                converged_param_nearest_voxel_transformation_likelihood: node
                    .declare_parameter(
                        "score_estimation.converged_param_nearest_voxel_transformation_likelihood",
                    )
                    .default(2.3)
                    .mandatory()?
                    .get(),
            },
            covariance: CovarianceParams {
                output_pose_covariance: default_covariance(),
                covariance_estimation_type: CovarianceEstimationType::from_i32(
                    node.declare_parameter(
                        "covariance.covariance_estimation.covariance_estimation_type",
                    )
                    .default(0)
                    .mandatory()?
                    .get() as i32,
                ),
                estimation: CovarianceEstimationParams {
                    initial_pose_offset_model_x: node
                        .declare_parameter::<Arc<[f64]>>(
                            "covariance.covariance_estimation.initial_pose_offset_model_x",
                        )
                        .default(Arc::from([0.0, 0.0, 0.5, -0.5, 1.0, -1.0]))
                        .mandatory()?
                        .get()
                        .to_vec(),
                    initial_pose_offset_model_y: node
                        .declare_parameter::<Arc<[f64]>>(
                            "covariance.covariance_estimation.initial_pose_offset_model_y",
                        )
                        .default(Arc::from([0.5, -0.5, 0.0, 0.0, 0.0, 0.0]))
                        .mandatory()?
                        .get()
                        .to_vec(),
                    temperature: node
                        .declare_parameter("covariance.covariance_estimation.temperature")
                        .default(0.05)
                        .mandatory()?
                        .get(),
                    scale_factor: node
                        .declare_parameter("covariance.covariance_estimation.scale_factor")
                        .default(1.0)
                        .mandatory()?
                        .get(),
                },
            },
            dynamic_map: DynamicMapParams {
                update_distance: node
                    .declare_parameter("dynamic_map_loading.update_distance")
                    .default(20.0)
                    .mandatory()?
                    .get(),
                map_radius: node
                    .declare_parameter("dynamic_map_loading.map_radius")
                    .default(150.0)
                    .mandatory()?
                    .get(),
                lidar_radius: node
                    .declare_parameter("dynamic_map_loading.lidar_radius")
                    .default(100.0)
                    .mandatory()?
                    .get(),
            },
        })
    }
}

/// Default 6x6 covariance matrix (diagonal)
fn default_covariance() -> [f64; 36] {
    [
        0.0225, 0.0, 0.0, 0.0, 0.0, 0.0, // row 0
        0.0, 0.0225, 0.0, 0.0, 0.0, 0.0, // row 1
        0.0, 0.0, 0.0225, 0.0, 0.0, 0.0, // row 2
        0.0, 0.0, 0.0, 0.000625, 0.0, 0.0, // row 3
        0.0, 0.0, 0.0, 0.0, 0.000625, 0.0, // row 4
        0.0, 0.0, 0.0, 0.0, 0.0, 0.000625, // row 5
    ]
}
