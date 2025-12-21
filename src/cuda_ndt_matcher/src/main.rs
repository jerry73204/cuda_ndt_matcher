mod covariance;
mod initial_pose;
mod map_module;
mod ndt_manager;
mod params;
mod particle;
mod pointcloud;
mod tpe;

use anyhow::Result;
use arc_swap::ArcSwap;
use geometry_msgs::msg::{Point, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped};
use map_module::MapUpdateModule;
use ndt_manager::NdtManager;
use params::NdtParams;
use parking_lot::Mutex;
use rclrs::{
    log_error, log_info, log_warn, Context, CreateBasicExecutor, Node, Publisher, QoSHistoryPolicy,
    QoSProfile, RclrsErrorFilter, Service, SpinOptions, Subscription, SubscriptionOptions,
};
use sensor_msgs::msg::PointCloud2;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std_msgs::msg::Header;
use std_srvs::srv::{SetBool, Trigger};
use tier4_localization_msgs::srv::PoseWithCovarianceStamped as PoseWithCovSrv;

// Type aliases
type SetBoolRequest = std_srvs::srv::SetBool_Request;
type SetBoolResponse = std_srvs::srv::SetBool_Response;
type TriggerRequest = std_srvs::srv::Trigger_Request;
type TriggerResponse = std_srvs::srv::Trigger_Response;
type PoseWithCovSrvRequest = tier4_localization_msgs::srv::PoseWithCovarianceStamped_Request;
type PoseWithCovSrvResponse = tier4_localization_msgs::srv::PoseWithCovarianceStamped_Response;

const NODE_NAME: &str = "ndt_scan_matcher";

struct NdtScanMatcherNode {
    // Subscriptions (stored to keep alive)
    _points_sub: Subscription<PointCloud2>,
    _initial_pose_sub: Subscription<PoseWithCovarianceStamped>,
    _regularization_pose_sub: Subscription<PoseWithCovarianceStamped>,
    _map_sub: Subscription<PointCloud2>,

    // Publishers
    pose_pub: Publisher<PoseStamped>,
    pose_cov_pub: Publisher<PoseWithCovarianceStamped>,

    // Services
    _trigger_srv: Service<SetBool>,
    _ndt_align_srv: Service<PoseWithCovSrv>,
    _map_update_srv: Service<Trigger>,

    // State
    ndt_manager: Arc<Mutex<NdtManager>>,
    map_module: Arc<MapUpdateModule>,
    map_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    latest_pose: Arc<ArcSwap<Option<PoseWithCovarianceStamped>>>,
    latest_sensor_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    enabled: Arc<AtomicBool>,
    params: Arc<NdtParams>,
}

impl NdtScanMatcherNode {
    fn new(node: &Node) -> Result<Self> {
        // Load parameters
        let params = Arc::new(NdtParams::from_node(node)?);
        log_info!(
            NODE_NAME,
            "NDT params: resolution={}, max_iter={}, epsilon={}",
            params.ndt.resolution,
            params.ndt.max_iterations,
            params.ndt.trans_epsilon
        );

        // Initialize NDT manager
        let ndt_manager = Arc::new(Mutex::new(NdtManager::new(&params)?));

        // Shared state
        let map_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>> = Arc::new(ArcSwap::from_pointee(None));
        let latest_pose: Arc<ArcSwap<Option<PoseWithCovarianceStamped>>> =
            Arc::new(ArcSwap::from_pointee(None));
        let latest_sensor_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>> =
            Arc::new(ArcSwap::from_pointee(None));
        let enabled = Arc::new(AtomicBool::new(true));

        // Initialize map update module
        let map_module = Arc::new(MapUpdateModule::new(params.dynamic_map.clone()));
        log_info!(
            NODE_NAME,
            "Map module: update_distance={}, map_radius={}, lidar_radius={}",
            params.dynamic_map.update_distance,
            params.dynamic_map.map_radius,
            params.dynamic_map.lidar_radius
        );

        // QoS for sensor data
        let sensor_qos = QoSProfile {
            history: QoSHistoryPolicy::KeepLast { depth: 1 },
            ..QoSProfile::sensor_data_default()
        };

        // Publishers
        let pose_pub = node.create_publisher("ndt_pose")?;
        let pose_cov_pub = node.create_publisher("ndt_pose_with_covariance")?;

        // Points subscription
        let points_sub = {
            let ndt_manager = Arc::clone(&ndt_manager);
            let map_points = Arc::clone(&map_points);
            let latest_pose = Arc::clone(&latest_pose);
            let latest_sensor_points = Arc::clone(&latest_sensor_points);
            let enabled = Arc::clone(&enabled);
            let pose_pub = pose_pub.clone();
            let pose_cov_pub = pose_cov_pub.clone();
            let params = Arc::clone(&params);

            let mut opts = SubscriptionOptions::new("points_raw");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |msg: PointCloud2| {
                Self::on_points(
                    msg,
                    &ndt_manager,
                    &map_points,
                    &latest_pose,
                    &latest_sensor_points,
                    &enabled,
                    &pose_pub,
                    &pose_cov_pub,
                    &params,
                );
            })?
        };

        // Initial pose subscription
        let initial_pose_sub = {
            let latest_pose = Arc::clone(&latest_pose);

            let mut opts = SubscriptionOptions::new("ekf_pose_with_covariance");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
                latest_pose.store(Arc::new(Some(msg)));
            })?
        };

        // Regularization pose subscription
        let regularization_pose_sub = {
            let mut opts = SubscriptionOptions::new("regularization_pose_with_covariance");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |_msg: PoseWithCovarianceStamped| {
                // TODO: Implement regularization in later phase
            })?
        };

        // Map subscription (for receiving point cloud map data)
        let map_sub = {
            let map_module = Arc::clone(&map_module);
            let map_points = Arc::clone(&map_points);
            let ndt_manager = Arc::clone(&ndt_manager);

            let mut opts = SubscriptionOptions::new("pointcloud_map");
            opts.qos = QoSProfile::default(); // Reliable for map data

            node.create_subscription(opts, move |msg: PointCloud2| {
                Self::on_map_received(msg, &map_module, &map_points, &ndt_manager);
            })?
        };

        // Trigger service
        let trigger_srv = {
            let enabled = Arc::clone(&enabled);

            node.create_service::<SetBool, _>(
                "trigger_node_srv",
                move |req: SetBoolRequest, _info: rclrs::ServiceInfo| {
                    enabled.store(req.data, Ordering::SeqCst);
                    log_info!(NODE_NAME, "Node enabled: {}", req.data);
                    SetBoolResponse {
                        success: true,
                        message: format!(
                            "NDT scan matcher {}",
                            if req.data { "enabled" } else { "disabled" }
                        ),
                    }
                },
            )?
        };

        // NDT align service (initial pose estimation)
        // This service is called by pose_initializer with an initial pose guess
        let ndt_align_srv = {
            let ndt_manager = Arc::clone(&ndt_manager);
            let map_points = Arc::clone(&map_points);
            let latest_sensor_points = Arc::clone(&latest_sensor_points);
            let params = Arc::clone(&params);

            node.create_service::<PoseWithCovSrv, _>(
                "ndt_align_srv",
                move |req: PoseWithCovSrvRequest, _info: rclrs::ServiceInfo| {
                    Self::on_ndt_align(
                        req,
                        &ndt_manager,
                        &map_points,
                        &latest_sensor_points,
                        &params,
                    )
                },
            )?
        };

        // Map update service (triggers map update based on current position)
        let map_update_srv = {
            let map_module = Arc::clone(&map_module);
            let map_points = Arc::clone(&map_points);
            let ndt_manager = Arc::clone(&ndt_manager);
            let latest_pose = Arc::clone(&latest_pose);

            node.create_service::<Trigger, _>(
                "map_update_srv",
                move |_req: TriggerRequest, _info: rclrs::ServiceInfo| {
                    Self::on_map_update(&map_module, &map_points, &ndt_manager, &latest_pose)
                },
            )?
        };

        log_info!(NODE_NAME, "NDT scan matcher node initialized");

        Ok(Self {
            _points_sub: points_sub,
            _initial_pose_sub: initial_pose_sub,
            _regularization_pose_sub: regularization_pose_sub,
            _map_sub: map_sub,
            pose_pub,
            pose_cov_pub,
            _trigger_srv: trigger_srv,
            _ndt_align_srv: ndt_align_srv,
            _map_update_srv: map_update_srv,
            ndt_manager,
            map_module,
            map_points,
            latest_pose,
            latest_sensor_points,
            enabled,
            params,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn on_points(
        msg: PointCloud2,
        ndt_manager: &Arc<Mutex<NdtManager>>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        latest_pose: &Arc<ArcSwap<Option<PoseWithCovarianceStamped>>>,
        latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        enabled: &Arc<AtomicBool>,
        pose_pub: &Publisher<PoseStamped>,
        pose_cov_pub: &Publisher<PoseWithCovarianceStamped>,
        params: &NdtParams,
    ) {
        // Check if enabled
        if !enabled.load(Ordering::SeqCst) {
            return;
        }

        // Get map points
        let map = map_points.load();
        let map = match map.as_ref() {
            Some(m) => m,
            None => {
                log_warn!(NODE_NAME, "No map loaded, skipping alignment");
                return;
            }
        };

        // Get initial pose
        let initial_pose = latest_pose.load();
        let initial_pose = match initial_pose.as_ref() {
            Some(p) => p,
            None => {
                log_warn!(NODE_NAME, "No initial pose, skipping alignment");
                return;
            }
        };

        // Convert sensor points
        let sensor_points = match pointcloud::from_pointcloud2(&msg) {
            Ok(pts) => pts,
            Err(e) => {
                log_error!(NODE_NAME, "Failed to convert point cloud: {e}");
                return;
            }
        };

        // Store sensor points for initial pose estimation service
        latest_sensor_points.store(Arc::new(Some(sensor_points.clone())));

        // Check minimum distance
        let max_dist = sensor_points
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
            .fold(0.0f32, f32::max);

        if max_dist < params.sensor_points.required_distance {
            log_warn!(
                NODE_NAME,
                "Sensor points max distance {max_dist:.1}m < required {:.1}m",
                params.sensor_points.required_distance
            );
            return;
        }

        // Run NDT alignment
        let mut manager = ndt_manager.lock();
        let result = match manager.align(&sensor_points, map, &initial_pose.pose.pose) {
            Ok(r) => r,
            Err(e) => {
                log_error!(NODE_NAME, "NDT alignment failed: {e}");
                return;
            }
        };

        if !result.converged {
            log_warn!(NODE_NAME, "NDT did not converge, score={:.3}", result.score);
            return;
        }

        // Estimate covariance based on configured mode
        let source_cloud = fast_gicp::PointCloudXYZ::from_points(&sensor_points);
        let target_cloud = fast_gicp::PointCloudXYZ::from_points(map);

        let covariance_result = covariance::estimate_covariance(
            &params.covariance,
            &result.hessian,
            &result.pose,
            manager.ndt(),
            &source_cloud,
            &target_cloud,
            &result.final_transform,
        );

        // Create output header
        let header = Header {
            stamp: msg.header.stamp.clone(),
            frame_id: params.frame.map_frame.clone(),
        };

        // Publish PoseStamped
        let pose_msg = PoseStamped {
            header: header.clone(),
            pose: result.pose.clone(),
        };
        if let Err(e) = pose_pub.publish(&pose_msg) {
            log_error!(NODE_NAME, "Failed to publish pose: {e}");
        }

        // Publish PoseWithCovarianceStamped with estimated covariance
        let pose_cov_msg = PoseWithCovarianceStamped {
            header,
            pose: PoseWithCovariance {
                pose: result.pose,
                covariance: covariance_result.covariance,
            },
        };
        if let Err(e) = pose_cov_pub.publish(&pose_cov_msg) {
            log_error!(NODE_NAME, "Failed to publish pose with covariance: {e}");
        }
    }

    /// Handle NDT align service request
    /// This service is called by pose_initializer with an initial pose guess.
    /// It performs a single NDT alignment and returns the aligned pose.
    fn on_ndt_align(
        req: PoseWithCovSrvRequest,
        ndt_manager: &Arc<Mutex<NdtManager>>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        params: &NdtParams,
    ) -> PoseWithCovSrvResponse {
        log_info!(NODE_NAME, "NDT align service called");

        // Get initial pose from request
        let initial_pose = req.pose_with_covariance;

        // Get map points
        let map = map_points.load();
        let map = match map.as_ref() {
            Some(m) => m,
            None => {
                log_error!(NODE_NAME, "NDT align failed: No map loaded");
                return PoseWithCovSrvResponse {
                    success: false,
                    reliable: false,
                    pose_with_covariance: initial_pose,
                };
            }
        };

        // Get sensor points
        let sensor_points = latest_sensor_points.load();
        let sensor_points = match sensor_points.as_ref() {
            Some(p) => p,
            None => {
                log_error!(NODE_NAME, "NDT align failed: No sensor points available");
                return PoseWithCovSrvResponse {
                    success: false,
                    reliable: false,
                    pose_with_covariance: initial_pose,
                };
            }
        };

        // Run single NDT alignment from the provided initial pose
        let mut manager = ndt_manager.lock();
        let result = match manager.align(sensor_points, map, &initial_pose.pose.pose) {
            Ok(r) => r,
            Err(e) => {
                log_error!(NODE_NAME, "NDT alignment failed: {e}");
                return PoseWithCovSrvResponse {
                    success: false,
                    reliable: false,
                    pose_with_covariance: initial_pose,
                };
            }
        };

        // Check convergence
        let reliable = result.converged
            && result.score >= params.score.converged_param_nearest_voxel_transformation_likelihood;

        log_info!(
            NODE_NAME,
            "NDT align complete: converged={}, score={:.3}, reliable={}",
            result.converged,
            result.score,
            reliable
        );

        // Build result with aligned pose
        let result_pose = PoseWithCovarianceStamped {
            header: initial_pose.header,
            pose: PoseWithCovariance {
                pose: result.pose,
                covariance: params.covariance.output_pose_covariance,
            },
        };

        PoseWithCovSrvResponse {
            success: result.converged,
            reliable,
            pose_with_covariance: result_pose,
        }
    }

    /// Handle map point cloud received
    fn on_map_received(
        msg: PointCloud2,
        map_module: &Arc<MapUpdateModule>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        ndt_manager: &Arc<Mutex<NdtManager>>,
    ) {
        // Convert point cloud
        let points = match pointcloud::from_pointcloud2(&msg) {
            Ok(pts) => pts,
            Err(e) => {
                log_error!(NODE_NAME, "Failed to convert map point cloud: {e}");
                return;
            }
        };

        log_info!(NODE_NAME, "Received map with {} points", points.len());

        // Load into map module
        map_module.load_full_map(points.clone());

        // Update shared map points
        map_points.store(Arc::new(Some(points.clone())));

        // Update NDT target
        let mut manager = ndt_manager.lock();
        if let Err(e) = manager.set_target(&points) {
            log_error!(NODE_NAME, "Failed to set NDT target: {e}");
        } else {
            log_info!(NODE_NAME, "NDT target updated with map");
        }
    }

    /// Handle map update service request
    fn on_map_update(
        map_module: &Arc<MapUpdateModule>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        ndt_manager: &Arc<Mutex<NdtManager>>,
        latest_pose: &Arc<ArcSwap<Option<PoseWithCovarianceStamped>>>,
    ) -> TriggerResponse {
        // Get current position
        let pose = latest_pose.load();
        let position = match pose.as_ref() {
            Some(p) => Point {
                x: p.pose.pose.position.x,
                y: p.pose.pose.position.y,
                z: p.pose.pose.position.z,
            },
            None => {
                return TriggerResponse {
                    success: false,
                    message: "No position available for map update".to_string(),
                };
            }
        };

        // Check if update is needed
        let should_update = map_module.should_update(&position);
        let out_of_range = map_module.out_of_map_range(&position);

        if out_of_range {
            log_warn!(
                NODE_NAME,
                "Position is out of map range - may need new map data"
            );
        }

        // Perform map update
        let result = map_module.update_map(&position);

        if result.updated {
            log_info!(
                NODE_NAME,
                "Map updated: {} tiles, {} points, distance={:.1}m",
                result.tiles_loaded,
                result.total_points,
                result.distance_from_last_update
            );

            // Update shared map points with filtered map
            if let Some(filtered_points) = map_module.get_map_points() {
                map_points.store(Arc::new(Some(filtered_points.clone())));

                // Update NDT target
                let mut manager = ndt_manager.lock();
                if let Err(e) = manager.set_target(&filtered_points) {
                    log_error!(NODE_NAME, "Failed to update NDT target: {e}");
                    return TriggerResponse {
                        success: false,
                        message: format!("Failed to update NDT target: {e}"),
                    };
                }
            }
        }

        TriggerResponse {
            success: true,
            message: format!(
                "updated={}, tiles={}, points={}, distance={:.1}m, should_update={}, out_of_range={}",
                result.updated,
                result.tiles_loaded,
                result.total_points,
                result.distance_from_last_update,
                should_update,
                out_of_range
            ),
        }
    }

    /// Load map from points (called externally or via service)
    #[allow(dead_code)]
    pub fn set_map(&self, points: Vec<[f32; 3]>) {
        log_info!(NODE_NAME, "Loading map with {} points", points.len());
        self.map_points.store(Arc::new(Some(points.clone())));

        let mut manager = self.ndt_manager.lock();
        if let Err(e) = manager.set_target(&points) {
            log_error!(NODE_NAME, "Failed to set NDT target: {e}");
        }
    }
}

fn main() -> Result<()> {
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let node = executor.create_node(NODE_NAME)?;

    let _ndt_node = NdtScanMatcherNode::new(&node)?;

    // TODO: For testing, load a dummy map
    // In production, this would come from pcd_loader_service
    log_info!(NODE_NAME, "Waiting for map and initial pose...");

    // Spin until shutdown (Ctrl-C or ROS shutdown)
    // Using default spin options which properly handles service responses
    executor.spin(SpinOptions::default()).first_error()?;

    log_info!(NODE_NAME, "Shutdown complete");
    Ok(())
}
