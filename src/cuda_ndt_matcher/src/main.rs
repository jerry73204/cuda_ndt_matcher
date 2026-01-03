mod covariance;
mod diagnostics;
mod dual_ndt_manager;
mod initial_pose;
mod map_module;
mod ndt_manager;
mod nvtl;
mod params;
mod particle;
mod pointcloud;
mod tpe;

use anyhow::Result;
use arc_swap::ArcSwap;
use diagnostics::{DiagnosticLevel, DiagnosticsInterface, ScanMatchingDiagnostics};
use dual_ndt_manager::DualNdtManager;
use geometry_msgs::msg::{Point, Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped};
use map_module::{DynamicMapLoader, MapUpdateModule};
use params::NdtParams;
use parking_lot::Mutex;
use rclrs::{
    log_debug, log_error, log_info, log_warn, Context, CreateBasicExecutor, Node, Publisher,
    QoSHistoryPolicy, QoSProfile, RclrsErrorFilter, Service, SpinOptions, Subscription,
    SubscriptionOptions,
};
use sensor_msgs::msg::PointCloud2;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std_msgs::msg::Header;
use std_srvs::srv::{SetBool, Trigger};
use tf2_msgs::msg::TFMessage;
use tier4_debug_msgs::msg::{Float32Stamped, Int32Stamped};
use tier4_localization_msgs::srv::PoseWithCovarianceStamped as PoseWithCovSrv;
use visualization_msgs::msg::{Marker, MarkerArray};

// Type aliases
type SetBoolRequest = std_srvs::srv::SetBool_Request;
type SetBoolResponse = std_srvs::srv::SetBool_Response;
type TriggerRequest = std_srvs::srv::Trigger_Request;
type TriggerResponse = std_srvs::srv::Trigger_Response;
type PoseWithCovSrvRequest = tier4_localization_msgs::srv::PoseWithCovarianceStamped_Request;
type PoseWithCovSrvResponse = tier4_localization_msgs::srv::PoseWithCovarianceStamped_Response;

const NODE_NAME: &str = "ndt_scan_matcher";

/// Holds debug and visualization publishers
#[derive(Clone)]
struct DebugPublishers {
    // TF broadcaster (publishes to /tf)
    tf_pub: Publisher<TFMessage>,

    // Visualization
    ndt_marker_pub: Publisher<MarkerArray>,
    points_aligned_pub: Publisher<PointCloud2>,
    monte_carlo_marker_pub: Publisher<MarkerArray>,

    // Debug metrics
    transform_probability_pub: Publisher<Float32Stamped>,
    nvtl_pub: Publisher<Float32Stamped>,
    iteration_num_pub: Publisher<Int32Stamped>,
    exe_time_pub: Publisher<Float32Stamped>,
    oscillation_count_pub: Publisher<Int32Stamped>,

    // Pose tracking
    initial_pose_cov_pub: Publisher<PoseWithCovarianceStamped>,
    initial_to_result_distance_pub: Publisher<Float32Stamped>,
    initial_to_result_relative_pose_pub: Publisher<PoseStamped>,
}

// Note: Many fields appear "unused" but are actually used via cloned references
// passed to subscription/service callbacks. Rust's dead code analysis doesn't
// track usage through closures.
#[allow(dead_code)]
struct NdtScanMatcherNode {
    // Subscriptions (stored to keep alive)
    _points_sub: Subscription<PointCloud2>,
    _initial_pose_sub: Subscription<PoseWithCovarianceStamped>,
    _regularization_pose_sub: Subscription<PoseWithCovarianceStamped>,
    _map_sub: Subscription<PointCloud2>,

    // Publishers - Core pose output
    pose_pub: Publisher<PoseStamped>,
    pose_cov_pub: Publisher<PoseWithCovarianceStamped>,

    // Publishers - Debug and visualization
    debug_pubs: DebugPublishers,

    // Diagnostics
    diagnostics: Arc<Mutex<DiagnosticsInterface>>,

    // Services
    _trigger_srv: Service<SetBool>,
    _ndt_align_srv: Service<PoseWithCovSrv>,
    _map_update_srv: Service<Trigger>,

    // State
    ndt_manager: Arc<DualNdtManager>,
    map_module: Arc<MapUpdateModule>,
    map_loader: Arc<DynamicMapLoader>,
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

        // Initialize NDT manager (dual for non-blocking updates)
        let ndt_manager = Arc::new(DualNdtManager::new((*params).clone())?);

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

        // Initialize dynamic map loader (service client for pcd_loader_service)
        let map_loader = Arc::new(DynamicMapLoader::new(
            node,
            "pcd_loader_service",
            Arc::clone(&map_module),
        )?);

        // QoS for sensor data
        let sensor_qos = QoSProfile {
            history: QoSHistoryPolicy::KeepLast { depth: 1 },
            ..QoSProfile::sensor_data_default()
        };

        // Publishers - Core pose output
        let pose_pub = node.create_publisher("pose")?;
        let pose_cov_pub = node.create_publisher("pose_with_covariance")?;

        // Publishers - Debug and visualization
        let debug_pubs = DebugPublishers {
            // TF broadcaster - publishes to /tf (absolute topic name)
            tf_pub: node.create_publisher("/tf")?,
            ndt_marker_pub: node.create_publisher("ndt_marker")?,
            points_aligned_pub: node.create_publisher("points_aligned")?,
            monte_carlo_marker_pub: node.create_publisher("monte_carlo_initial_pose_marker")?,
            transform_probability_pub: node.create_publisher("transform_probability")?,
            nvtl_pub: node.create_publisher("nearest_voxel_transformation_likelihood")?,
            iteration_num_pub: node.create_publisher("iteration_num")?,
            exe_time_pub: node.create_publisher("exe_time_ms")?,
            oscillation_count_pub: node
                .create_publisher("local_optimal_solution_oscillation_num")?,
            initial_pose_cov_pub: node.create_publisher("initial_pose_with_covariance")?,
            initial_to_result_distance_pub: node.create_publisher("initial_to_result_distance")?,
            initial_to_result_relative_pose_pub: node
                .create_publisher("initial_to_result_relative_pose")?,
        };

        // Create diagnostics interface
        let diagnostics = Arc::new(Mutex::new(DiagnosticsInterface::new(node)?));

        // Points subscription
        let points_sub = {
            let ndt_manager = Arc::clone(&ndt_manager);
            let map_module = Arc::clone(&map_module);
            let map_loader = Arc::clone(&map_loader);
            let map_points = Arc::clone(&map_points);
            let latest_pose = Arc::clone(&latest_pose);
            let latest_sensor_points = Arc::clone(&latest_sensor_points);
            let enabled = Arc::clone(&enabled);
            let pose_pub = pose_pub.clone();
            let pose_cov_pub = pose_cov_pub.clone();
            let debug_pubs = debug_pubs.clone();
            let diagnostics = Arc::clone(&diagnostics);
            let params = Arc::clone(&params);

            let mut opts = SubscriptionOptions::new("points_raw");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |msg: PointCloud2| {
                Self::on_points(
                    msg,
                    &ndt_manager,
                    &map_module,
                    &map_loader,
                    &map_points,
                    &latest_pose,
                    &latest_sensor_points,
                    &enabled,
                    &pose_pub,
                    &pose_cov_pub,
                    &debug_pubs,
                    &diagnostics,
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

        // Regularization pose subscription (GNSS pose for regularization)
        let regularization_pose_sub = {
            let ndt_manager = Arc::clone(&ndt_manager);
            let params = Arc::clone(&params);

            let mut opts = SubscriptionOptions::new("regularization_pose_with_covariance");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
                // Only process if regularization is enabled
                if !params.regularization.enabled {
                    return;
                }

                // Set the regularization reference pose in the NDT matcher
                ndt_manager.set_regularization_pose(&msg.pose.pose);
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
            let monte_carlo_pub = debug_pubs.monte_carlo_marker_pub.clone();

            node.create_service::<PoseWithCovSrv, _>(
                "ndt_align_srv",
                move |req: PoseWithCovSrvRequest, _info: rclrs::ServiceInfo| {
                    Self::on_ndt_align(
                        req,
                        &ndt_manager,
                        &map_points,
                        &latest_sensor_points,
                        &params,
                        &monte_carlo_pub,
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
            debug_pubs,
            diagnostics,
            _trigger_srv: trigger_srv,
            _ndt_align_srv: ndt_align_srv,
            _map_update_srv: map_update_srv,
            ndt_manager,
            map_module,
            map_loader,
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
        ndt_manager: &Arc<DualNdtManager>,
        map_module: &Arc<MapUpdateModule>,
        map_loader: &Arc<DynamicMapLoader>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        latest_pose: &Arc<ArcSwap<Option<PoseWithCovarianceStamped>>>,
        latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        enabled: &Arc<AtomicBool>,
        pose_pub: &Publisher<PoseStamped>,
        pose_cov_pub: &Publisher<PoseWithCovarianceStamped>,
        debug_pubs: &DebugPublishers,
        diagnostics: &Arc<Mutex<DiagnosticsInterface>>,
        params: &NdtParams,
    ) {
        let start_time = Instant::now();

        // Convert sensor points first - needed for align service even before we have initial pose
        let sensor_points = match pointcloud::from_pointcloud2(&msg) {
            Ok(pts) => pts,
            Err(e) => {
                log_error!(NODE_NAME, "Failed to convert point cloud: {e}");
                return;
            }
        };

        // Always store sensor points for initial pose estimation service (ndt_align_srv)
        // This must happen before any early returns so the align service can work
        latest_sensor_points.store(Arc::new(Some(sensor_points.clone())));

        // Check if enabled for regular NDT alignment
        if !enabled.load(Ordering::SeqCst) {
            return;
        }

        // Get initial pose (needed for map update check and NDT alignment)
        let initial_pose = latest_pose.load();
        let initial_pose = match initial_pose.as_ref() {
            Some(p) => p,
            None => {
                log_warn!(NODE_NAME, "No initial pose, skipping alignment");
                return;
            }
        };

        // Check if map needs updating based on current position
        // This implements Autoware's dynamic map loading behavior
        let current_position = Point {
            x: initial_pose.pose.pose.position.x,
            y: initial_pose.pose.pose.position.y,
            z: initial_pose.pose.pose.position.z,
        };

        // Check if we should request new map tiles via service
        // This is non-blocking - the callback will update map_module when response arrives
        if map_module.should_update(&current_position) {
            if let Err(e) = map_loader
                .request_map_update(&current_position, params.dynamic_map.map_radius as f32)
            {
                log_error!(NODE_NAME, "Failed to request map update: {e}");
            }
        }

        // Check and apply any pending updates from the map module (local filtering)
        if let Some(filtered_map) = map_module.check_and_update(&current_position) {
            // Map was updated - refresh the shared map points
            map_points.store(Arc::new(Some(filtered_map.clone())));

            // Start non-blocking NDT target update in background thread
            let started = ndt_manager.start_background_update(filtered_map.clone());
            log_debug!(
                NODE_NAME,
                "Background NDT update started={started} with {} points",
                filtered_map.len()
            );
        }

        // Get map points (may have been updated above)
        let map = map_points.load();
        let map = match map.as_ref() {
            Some(m) => m,
            None => {
                log_warn!(NODE_NAME, "No map loaded, skipping alignment");
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

        // Run NDT alignment (with optional debug output)
        let debug_enabled = std::env::var("NDT_DEBUG").is_ok();
        let timestamp_ns =
            msg.header.stamp.sec as u64 * 1_000_000_000 + msg.header.stamp.nanosec as u64;

        // Get lock on active NDT manager (also checks for pending swap from background update)
        let mut manager = ndt_manager.lock();
        let result = if debug_enabled {
            // Use debug variant and write to file
            match manager.align_with_debug(
                &sensor_points,
                map,
                &initial_pose.pose.pose,
                timestamp_ns,
            ) {
                Ok((r, debug)) => {
                    // Write debug JSON to file
                    if let Ok(json) = debug.to_json() {
                        let debug_file = std::env::var("NDT_DEBUG_FILE")
                            .unwrap_or_else(|_| "/tmp/ndt_cuda_debug.jsonl".to_string());
                        if let Ok(mut file) = OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&debug_file)
                        {
                            let _ = writeln!(file, "{json}");
                        }
                    }
                    r
                }
                Err(e) => {
                    log_error!(NODE_NAME, "NDT alignment failed: {e}");
                    return;
                }
            }
        } else {
            match manager.align(&sensor_points, map, &initial_pose.pose.pose) {
                Ok(r) => r,
                Err(e) => {
                    log_error!(NODE_NAME, "NDT alignment failed: {e}");
                    return;
                }
            }
        };

        // Log warning if not converged, but still use the result
        // (Autoware publishes even when max iterations reached)
        if !result.converged {
            log_warn!(NODE_NAME, "NDT did not converge, score={:.3}", result.score);
        }

        // Estimate covariance based on configured mode
        // For MULTI_NDT modes, we use parallel batch evaluation (Rayon)
        // NOTE: We reuse the manager lock from the alignment - don't try to lock again!
        let covariance_result = covariance::estimate_covariance_full(
            &params.covariance,
            &result.hessian,
            &result.pose,
            Some(&*manager), // Reuse existing lock to avoid deadlock
            Some(&sensor_points),
            Some(map),
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
            header: header.clone(),
            pose: PoseWithCovariance {
                pose: result.pose.clone(),
                covariance: covariance_result.covariance,
            },
        };
        if let Err(e) = pose_cov_pub.publish(&pose_cov_msg) {
            log_error!(NODE_NAME, "Failed to publish pose with covariance: {e}");
        }

        // Publish TF transform (map -> ndt_base_link)
        // This matches Autoware's publish_tf() behavior
        Self::publish_tf(
            &debug_pubs.tf_pub,
            &msg.header.stamp,
            &result.pose,
            &params.frame.map_frame,
            &params.frame.ndt_base_frame,
        );

        // ---- Debug Publishers ----

        // Calculate execution time
        let exe_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Publish execution time
        let exe_time_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: exe_time_ms,
        };
        let _ = debug_pubs.exe_time_pub.publish(&exe_time_msg);

        // Publish iteration count
        let iteration_msg = Int32Stamped {
            stamp: msg.header.stamp.clone(),
            data: result.iterations,
        };
        let _ = debug_pubs.iteration_num_pub.publish(&iteration_msg);

        // Publish oscillation count (detects if optimizer is bouncing between poses)
        let oscillation_msg = Int32Stamped {
            stamp: msg.header.stamp.clone(),
            data: result.oscillation_count as i32,
        };
        let _ = debug_pubs.oscillation_count_pub.publish(&oscillation_msg);

        // Publish transform probability (fitness score converted to probability)
        let transform_prob = (-result.score / 10.0).exp();
        let transform_prob_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: transform_prob as f32,
        };
        let _ = debug_pubs
            .transform_probability_pub
            .publish(&transform_prob_msg);

        // Compute and publish NVTL score
        let nvtl_score = manager
            .evaluate_nvtl(&sensor_points, map, &result.pose, 0.55)
            .unwrap_or(0.0);
        let nvtl_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: nvtl_score as f32,
        };
        let _ = debug_pubs.nvtl_pub.publish(&nvtl_msg);

        // Publish initial pose with covariance
        let _ = debug_pubs.initial_pose_cov_pub.publish(initial_pose);

        // Calculate initial to result distance
        let dx = result.pose.position.x - initial_pose.pose.pose.position.x;
        let dy = result.pose.position.y - initial_pose.pose.pose.position.y;
        let dz = result.pose.position.z - initial_pose.pose.pose.position.z;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let distance_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: distance as f32,
        };
        let _ = debug_pubs
            .initial_to_result_distance_pub
            .publish(&distance_msg);

        // Publish relative pose (result relative to initial)
        let relative_pose_msg = PoseStamped {
            header: header.clone(),
            pose: Pose {
                position: Point {
                    x: dx,
                    y: dy,
                    z: dz,
                },
                orientation: result.pose.orientation.clone(),
            },
        };
        let _ = debug_pubs
            .initial_to_result_relative_pose_pub
            .publish(&relative_pose_msg);

        // Publish NDT marker (arrow showing result pose)
        // Note: Autoware's builtin publishes transformation_array (all iteration poses as markers).
        // fast-gicp doesn't expose iteration history, so we only publish the final pose.
        let ndt_marker = Self::create_pose_marker(&header, &result.pose, 0);
        let marker_array = MarkerArray {
            markers: vec![ndt_marker],
        };
        let _ = debug_pubs.ndt_marker_pub.publish(&marker_array);

        // Publish aligned points (transformed sensor points)
        let aligned_points: Vec<[f32; 3]> = sensor_points
            .iter()
            .map(|p| {
                // Transform point by result pose
                let px = p[0] as f64;
                let py = p[1] as f64;
                let pz = p[2] as f64;
                // Simple translation (full transform would need quaternion rotation)
                [
                    (px + result.pose.position.x) as f32,
                    (py + result.pose.position.y) as f32,
                    (pz + result.pose.position.z) as f32,
                ]
            })
            .collect();
        let aligned_msg = pointcloud::to_pointcloud2(&aligned_points, &header);
        let _ = debug_pubs.points_aligned_pub.publish(&aligned_msg);

        // ---- Diagnostics ----
        // Collect and publish scan matching diagnostics
        let topic_time_stamp = msg.header.stamp.sec as f64 + msg.header.stamp.nanosec as f64 * 1e-9;
        let scan_diag = ScanMatchingDiagnostics {
            topic_time_stamp,
            sensor_points_size: sensor_points.len(),
            sensor_points_delay_time_sec: 0.0, // Would need current time to compute
            is_succeed_transform_sensor_points: true,
            sensor_points_max_distance: max_dist as f64,
            is_activated: true, // We're here, so we're activated
            is_succeed_interpolate_initial_pose: true,
            is_set_map_points: true,
            iteration_num: result.iterations,
            oscillation_count: result.oscillation_count,
            transform_probability: transform_prob,
            nearest_voxel_transformation_likelihood: nvtl_score,
            distance_initial_to_result: distance,
            execution_time_ms: exe_time_ms as f64,
            skipping_publish_num: 0,
        };

        {
            let mut diag = diagnostics.lock();
            scan_diag.apply_to(diag.scan_matching_mut());

            // Add map update diagnostics
            let map_status = map_loader.get_status();
            let map_diag = diag.map_update_mut();
            map_diag.clear();
            map_diag.add_key_value(
                "is_succeed_call_pcd_loader",
                map_status.last_request_success,
            );
            map_diag.add_key_value("pcd_loader_service_available", map_status.service_available);
            map_diag.add_key_value("tiles_loaded", map_module.tile_count());
            map_diag.add_key_value("tiles_added", map_status.tiles_added);
            map_diag.add_key_value("tiles_removed", map_status.tiles_removed);
            map_diag.add_key_value("points_added", map_status.points_added);
            if let Some(err) = &map_status.error_message {
                map_diag.add_key_value("error_message", err);
                map_diag.set_level_and_message(DiagnosticLevel::Warn, err);
            } else if !map_status.service_available && map_module.tile_count() == 0 {
                map_diag.set_level_and_message(
                    DiagnosticLevel::Warn,
                    "pcd_loader_service not available, no map loaded",
                );
            } else {
                map_diag.set_level_and_message(DiagnosticLevel::Ok, "OK");
            }

            diag.publish(msg.header.stamp);
        }
    }

    /// Create an arrow marker representing a pose
    fn create_pose_marker(header: &Header, pose: &Pose, id: i32) -> Marker {
        Marker {
            header: header.clone(),
            ns: "result_pose_matrix_array".to_string(),
            id,
            type_: 0,  // ARROW
            action: 0, // ADD
            pose: pose.clone(),
            scale: geometry_msgs::msg::Vector3 {
                x: 0.3,
                y: 0.1,
                z: 0.1,
            },
            color: std_msgs::msg::ColorRGBA {
                r: 0.0,
                g: 0.7,
                b: 1.0,
                a: 0.999,
            },
            lifetime: builtin_interfaces::msg::Duration { sec: 0, nanosec: 0 },
            frame_locked: false,
            points: vec![],
            colors: vec![],
            texture_resource: String::new(),
            texture: sensor_msgs::msg::CompressedImage::default(),
            uv_coordinates: vec![],
            text: String::new(),
            mesh_resource: String::new(),
            mesh_file: visualization_msgs::msg::MeshFile::default(),
            mesh_use_embedded_materials: false,
        }
    }

    /// Create visualization markers for Monte Carlo particles.
    ///
    /// Visualizes the initial pose estimation particles:
    /// - Initial poses: small blue spheres
    /// - Result poses: spheres colored by score (red=low, green=high)
    /// - Best particle: larger green sphere
    fn create_monte_carlo_markers(
        header: &Header,
        particles: &[crate::particle::Particle],
        best_score: f64,
    ) -> MarkerArray {
        let mut markers = Vec::new();
        let mut id = 0;

        // Find score range for color normalization
        let min_score = particles
            .iter()
            .map(|p| p.score)
            .fold(f64::INFINITY, f64::min);
        let max_score = particles
            .iter()
            .map(|p| p.score)
            .fold(f64::NEG_INFINITY, f64::max);
        let score_range = (max_score - min_score).max(0.001); // Avoid division by zero

        for particle in particles {
            // Normalize score to 0-1 range for color
            let normalized_score = (particle.score - min_score) / score_range;
            let is_best = (particle.score - best_score).abs() < 1e-10;

            // Initial pose marker (small blue sphere)
            markers.push(Marker {
                header: header.clone(),
                ns: "monte_carlo_initial".to_string(),
                id,
                type_: 2,  // SPHERE
                action: 0, // ADD
                pose: particle.initial_pose.clone(),
                scale: geometry_msgs::msg::Vector3 {
                    x: 0.15,
                    y: 0.15,
                    z: 0.15,
                },
                color: std_msgs::msg::ColorRGBA {
                    r: 0.3,
                    g: 0.5,
                    b: 1.0,
                    a: 0.6,
                },
                lifetime: builtin_interfaces::msg::Duration {
                    sec: 10,
                    nanosec: 0,
                },
                frame_locked: false,
                points: vec![],
                colors: vec![],
                texture_resource: String::new(),
                texture: sensor_msgs::msg::CompressedImage::default(),
                uv_coordinates: vec![],
                text: String::new(),
                mesh_resource: String::new(),
                mesh_file: visualization_msgs::msg::MeshFile::default(),
                mesh_use_embedded_materials: false,
            });
            id += 1;

            // Result pose marker (sphere colored by score)
            let size = if is_best { 0.4 } else { 0.2 };
            markers.push(Marker {
                header: header.clone(),
                ns: "monte_carlo_result".to_string(),
                id,
                type_: 2,  // SPHERE
                action: 0, // ADD
                pose: particle.result_pose.clone(),
                scale: geometry_msgs::msg::Vector3 {
                    x: size,
                    y: size,
                    z: size,
                },
                color: std_msgs::msg::ColorRGBA {
                    // Color gradient: red (low score) -> green (high score)
                    r: (1.0 - normalized_score) as f32,
                    g: normalized_score as f32,
                    b: 0.0,
                    a: if is_best { 1.0 } else { 0.7 },
                },
                lifetime: builtin_interfaces::msg::Duration {
                    sec: 10,
                    nanosec: 0,
                },
                frame_locked: false,
                points: vec![],
                colors: vec![],
                texture_resource: String::new(),
                texture: sensor_msgs::msg::CompressedImage::default(),
                uv_coordinates: vec![],
                text: String::new(),
                mesh_resource: String::new(),
                mesh_file: visualization_msgs::msg::MeshFile::default(),
                mesh_use_embedded_materials: false,
            });
            id += 1;
        }

        MarkerArray { markers }
    }

    /// Publish TF transform from map frame to ndt_base_frame.
    ///
    /// This matches Autoware's `publish_tf()` behavior in ndt_scan_matcher_core.cpp:
    /// - Parent frame: map_frame (typically "map")
    /// - Child frame: ndt_base_frame (typically "ndt_base_link")
    /// - Transform: The NDT result pose
    ///
    /// The TF is published to the `/tf` topic as a TFMessage containing a single
    /// TransformStamped message.
    fn publish_tf(
        tf_pub: &Publisher<TFMessage>,
        stamp: &builtin_interfaces::msg::Time,
        pose: &Pose,
        map_frame: &str,
        ndt_base_frame: &str,
    ) {
        // Convert Pose to Transform
        // Pose uses position/orientation, Transform uses translation/rotation
        let transform = geometry_msgs::msg::Transform {
            translation: geometry_msgs::msg::Vector3 {
                x: pose.position.x,
                y: pose.position.y,
                z: pose.position.z,
            },
            rotation: pose.orientation.clone(),
        };

        // Create TransformStamped message
        let transform_stamped = geometry_msgs::msg::TransformStamped {
            header: Header {
                stamp: stamp.clone(),
                frame_id: map_frame.to_string(),
            },
            child_frame_id: ndt_base_frame.to_string(),
            transform,
        };

        // Create TFMessage with the single transform
        let tf_msg = TFMessage {
            transforms: vec![transform_stamped],
        };

        // Publish to /tf
        if let Err(e) = tf_pub.publish(&tf_msg) {
            log_error!(NODE_NAME, "Failed to publish TF: {e}");
        }
    }

    /// Handle NDT align service request
    /// This service is called by pose_initializer with an initial pose guess.
    /// It performs multi-particle NDT alignment using TPE and returns the best aligned pose.
    /// This matches Autoware's behavior of sampling multiple initial poses to find the best match.
    fn on_ndt_align(
        req: PoseWithCovSrvRequest,
        ndt_manager: &Arc<DualNdtManager>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        params: &NdtParams,
        monte_carlo_pub: &Publisher<MarkerArray>,
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

        // Run multi-particle initial pose estimation using TPE
        // Lock the active manager for the duration of pose estimation
        let mut manager = ndt_manager.lock();
        let score_threshold = params
            .score
            .converged_param_nearest_voxel_transformation_likelihood;

        let result = match initial_pose::estimate_initial_pose(
            &initial_pose,
            &mut manager,
            sensor_points,
            map,
            &params.initial_pose,
            params.ndt.resolution,
            score_threshold,
        ) {
            Ok(r) => r,
            Err(e) => {
                log_error!(NODE_NAME, "Initial pose estimation failed: {e}");
                return PoseWithCovSrvResponse {
                    success: false,
                    reliable: false,
                    pose_with_covariance: initial_pose,
                };
            }
        };

        log_info!(
            NODE_NAME,
            "NDT align complete: converged=true, score={:.3}, reliable={}, particles={}",
            result.score,
            result.reliable,
            result.particles.len()
        );

        // Publish Monte Carlo particle visualization
        let markers =
            Self::create_monte_carlo_markers(&initial_pose.header, &result.particles, result.score);
        if let Err(e) = monte_carlo_pub.publish(&markers) {
            log_debug!(NODE_NAME, "Failed to publish Monte Carlo markers: {e}");
        }

        // Build result with best aligned pose
        let result_pose = PoseWithCovarianceStamped {
            header: initial_pose.header,
            pose: PoseWithCovariance {
                pose: result.pose_with_covariance.pose.pose,
                covariance: params.covariance.output_pose_covariance,
            },
        };

        PoseWithCovSrvResponse {
            success: true,
            reliable: result.reliable,
            pose_with_covariance: result_pose,
        }
    }

    /// Handle map point cloud received
    fn on_map_received(
        msg: PointCloud2,
        map_module: &Arc<MapUpdateModule>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        ndt_manager: &Arc<DualNdtManager>,
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

        // Update NDT target (blocking for initial map load)
        if let Err(e) = ndt_manager.set_target(&points) {
            log_error!(NODE_NAME, "Failed to set NDT target: {e}");
        } else {
            log_info!(NODE_NAME, "NDT target updated with map");
        }
    }

    /// Handle map update service request
    fn on_map_update(
        map_module: &Arc<MapUpdateModule>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        ndt_manager: &Arc<DualNdtManager>,
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

                // Start non-blocking NDT target update
                let started = ndt_manager.start_background_update(filtered_points);
                log_debug!(
                    NODE_NAME,
                    "Map update service: background NDT update started={started}"
                );
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

        // Blocking set for initial map load
        if let Err(e) = self.ndt_manager.set_target(&points) {
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
