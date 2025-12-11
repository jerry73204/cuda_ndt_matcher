mod covariance;
mod ndt_manager;
mod params;
mod pointcloud;

use anyhow::Result;
use arc_swap::ArcSwap;
use geometry_msgs::msg::{PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped};
use ndt_manager::NdtManager;
use params::NdtParams;
use parking_lot::Mutex;
use rclrs::{
    log_error, log_info, log_warn, Context, CreateBasicExecutor, Node, Publisher, QoSHistoryPolicy,
    QoSProfile, Service, SpinOptions, Subscription, SubscriptionOptions,
};
use sensor_msgs::msg::PointCloud2;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std_msgs::msg::Header;
use std_srvs::srv::SetBool;

// Type aliases
type SetBoolRequest = std_srvs::srv::SetBool_Request;
type SetBoolResponse = std_srvs::srv::SetBool_Response;

const NODE_NAME: &str = "ndt_scan_matcher";

struct NdtScanMatcherNode {
    // Subscriptions (stored to keep alive)
    _points_sub: Subscription<PointCloud2>,
    _initial_pose_sub: Subscription<PoseWithCovarianceStamped>,
    _regularization_pose_sub: Subscription<PoseWithCovarianceStamped>,

    // Publishers
    pose_pub: Publisher<PoseStamped>,
    pose_cov_pub: Publisher<PoseWithCovarianceStamped>,

    // Service
    _trigger_srv: Service<SetBool>,

    // State
    ndt_manager: Arc<Mutex<NdtManager>>,
    map_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    latest_pose: Arc<ArcSwap<Option<PoseWithCovarianceStamped>>>,
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
        let enabled = Arc::new(AtomicBool::new(true));

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
            let enabled = Arc::clone(&enabled);
            let pose_pub = pose_pub.clone();
            let pose_cov_pub = pose_cov_pub.clone();
            let params = Arc::clone(&params);

            let mut opts = SubscriptionOptions::new("points_raw");
            opts.qos = sensor_qos.clone();

            node.create_subscription(opts, move |msg: PointCloud2| {
                Self::on_points(
                    msg,
                    &ndt_manager,
                    &map_points,
                    &latest_pose,
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
            opts.qos = sensor_qos.clone();

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

        log_info!(NODE_NAME, "NDT scan matcher node initialized");

        Ok(Self {
            _points_sub: points_sub,
            _initial_pose_sub: initial_pose_sub,
            _regularization_pose_sub: regularization_pose_sub,
            pose_pub,
            pose_cov_pub,
            _trigger_srv: trigger_srv,
            ndt_manager,
            map_points,
            latest_pose,
            enabled,
            params,
        })
    }

    fn on_points(
        msg: PointCloud2,
        ndt_manager: &Arc<Mutex<NdtManager>>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        latest_pose: &Arc<ArcSwap<Option<PoseWithCovarianceStamped>>>,
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
            log_warn!(
                NODE_NAME,
                "NDT did not converge, score={:.3}",
                result.score
            );
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

    // Signal handling
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        log_info!(NODE_NAME, "Shutting down...");
        r.store(false, Ordering::SeqCst);
    })?;

    // Spin
    while running.load(Ordering::SeqCst) {
        let opts = SpinOptions::spin_once().timeout(Duration::from_millis(100));
        let _ = executor.spin(opts);
    }

    log_info!(NODE_NAME, "Shutdown complete");
    Ok(())
}
