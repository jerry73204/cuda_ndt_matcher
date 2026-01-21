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
mod pose_buffer;
mod scan_queue;
mod tf_handler;
mod tpe;
mod visualization;

use anyhow::Result;
use arc_swap::ArcSwap;
use diagnostics::{DiagnosticLevel, DiagnosticsInterface, ScanMatchingDiagnostics};
use dual_ndt_manager::DualNdtManager;
use geometry_msgs::msg::{
    Point, Pose, PoseArray, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped,
};
use map_module::{DynamicMapLoader, MapUpdateModule};
use nalgebra::{Isometry3, Quaternion as NaQuaternion, Translation3, UnitQuaternion, Vector3};
#[cfg(feature = "debug-markers")]
use ndt_cuda::AlignmentDebug;
use params::NdtParams;
use parking_lot::Mutex;
use pose_buffer::SmartPoseBuffer;
use rclrs::{
    log_debug, log_error, log_info, log_warn, Context, CreateBasicExecutor, Node, Publisher,
    QoSHistoryPolicy, QoSProfile, RclrsErrorFilter, Service, SpinOptions, Subscription,
    SubscriptionOptions,
};
use scan_queue::{QueuedScan, ScanQueue, ScanQueueConfig, ScanResult};
use sensor_msgs::msg::PointCloud2;
#[cfg(feature = "debug-output")]
use std::fs::OpenOptions;
#[cfg(feature = "debug-output")]
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std_msgs::msg::Header;
use std_srvs::srv::{SetBool, Trigger};
use tf2_msgs::msg::TFMessage;
use tier4_debug_msgs::msg::{Float32Stamped, Int32Stamped};
use tier4_localization_msgs::srv::PoseWithCovarianceStamped as PoseWithCovSrv;
use visualization::ParticleMarkerConfig;
use visualization_msgs::msg::{Marker, MarkerArray};

// Type aliases
type SetBoolRequest = std_srvs::srv::SetBool_Request;
type SetBoolResponse = std_srvs::srv::SetBool_Response;
type TriggerRequest = std_srvs::srv::Trigger_Request;
type TriggerResponse = std_srvs::srv::Trigger_Response;
type PoseWithCovSrvRequest = tier4_localization_msgs::srv::PoseWithCovarianceStamped_Request;
type PoseWithCovSrvResponse = tier4_localization_msgs::srv::PoseWithCovarianceStamped_Response;

const NODE_NAME: &str = "ndt_scan_matcher";

/// Convert nalgebra Isometry3 to geometry_msgs Pose
fn isometry_to_pose(iso: &Isometry3<f64>) -> Pose {
    let t = iso.translation;
    let q = iso.rotation.quaternion();
    Pose {
        position: Point {
            x: t.x,
            y: t.y,
            z: t.z,
        },
        orientation: geometry_msgs::msg::Quaternion {
            x: q.i,
            y: q.j,
            z: q.k,
            w: q.w,
        },
    }
}

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
    initial_to_result_distance_old_pub: Publisher<Float32Stamped>,
    initial_to_result_distance_new_pub: Publisher<Float32Stamped>,
    initial_to_result_relative_pose_pub: Publisher<PoseStamped>,

    // No-ground scoring (debug)
    no_ground_points_aligned_pub: Publisher<PointCloud2>,
    no_ground_transform_probability_pub: Publisher<Float32Stamped>,
    no_ground_nvtl_pub: Publisher<Float32Stamped>,

    // Per-point score visualization (voxel_score_points with RGB colors)
    voxel_score_points_pub: Publisher<PointCloud2>,

    // MULTI_NDT covariance debug: poses from offset alignments
    multi_ndt_pose_pub: Publisher<PoseArray>,

    // MULTI_NDT covariance debug: initial poses before alignment
    multi_initial_pose_pub: Publisher<PoseArray>,

    // Debug map: currently loaded point cloud map
    debug_loaded_pointcloud_map_pub: Publisher<PointCloud2>,
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
    pose_buffer: Arc<SmartPoseBuffer>,
    latest_sensor_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
    enabled: Arc<AtomicBool>,
    params: Arc<NdtParams>,

    // TF2 handler for sensor frame transforms
    tf_handler: Arc<tf_handler::TfHandler>,

    // Scan queue for batch processing (optional, enabled via params.batch.enabled)
    // Wrapped in Arc so it can be shared with the subscription callback
    scan_queue: Option<Arc<ScanQueue>>,
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

        // Initialize TF2 handler for sensor frame transforms
        let tf_handler = tf_handler::TfHandler::new(node)?;
        log_info!(NODE_NAME, "TF handler initialized for sensor transforms");

        // Shared state
        let map_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>> = Arc::new(ArcSwap::from_pointee(None));
        let pose_buffer = Arc::new(SmartPoseBuffer::new(
            params.validation.initial_pose_timeout_sec,
            params.validation.initial_pose_distance_tolerance_m,
        ));
        let latest_sensor_points: Arc<ArcSwap<Option<Vec<[f32; 3]>>>> =
            Arc::new(ArcSwap::from_pointee(None));
        // Start disabled - wait for trigger_node service to enable
        // This matches Autoware's behavior: pose_initializer refines the initial pose
        // via ndt_align_srv, sends it to EKF, then enables NDT
        let enabled = Arc::new(AtomicBool::new(false));
        // Track consecutive skips due to low score (like Autoware's skipping_publish_num)
        let skip_counter = Arc::new(AtomicI32::new(0));

        // Debug counters for callback tracking
        let callback_count = Arc::new(AtomicI32::new(0));
        let align_count = Arc::new(AtomicI32::new(0));

        // Note: We rely on QoS KeepLast(1) to prevent duplicate message processing,
        // matching Autoware's approach. No explicit timestamp deduplication needed.

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
        // Use actual topic names directly since rclrs doesn't support launch file remappings
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
            initial_to_result_distance_old_pub: node
                .create_publisher("initial_to_result_distance_old")?,
            initial_to_result_distance_new_pub: node
                .create_publisher("initial_to_result_distance_new")?,
            initial_to_result_relative_pose_pub: node
                .create_publisher("initial_to_result_relative_pose")?,
            // No-ground scoring debug
            no_ground_points_aligned_pub: node.create_publisher("points_aligned_no_ground")?,
            no_ground_transform_probability_pub: node
                .create_publisher("no_ground_transform_probability")?,
            no_ground_nvtl_pub: node
                .create_publisher("no_ground_nearest_voxel_transformation_likelihood")?,
            // Per-point score visualization
            voxel_score_points_pub: node.create_publisher("voxel_score_points")?,
            // MULTI_NDT covariance debug
            multi_ndt_pose_pub: node.create_publisher("multi_ndt_pose")?,
            multi_initial_pose_pub: node.create_publisher("multi_initial_pose")?,
            // Debug loaded map
            debug_loaded_pointcloud_map_pub: node
                .create_publisher("debug/loaded_pointcloud_map")?,
        };

        // Create diagnostics interface
        let diagnostics = Arc::new(Mutex::new(DiagnosticsInterface::new(node)?));

        // Initialize scan queue for batch processing (if enabled)
        // Must be created before the subscription so we can pass it to the callback
        let scan_queue: Option<Arc<ScanQueue>> = if params.batch.enabled {
            log_info!(
                NODE_NAME,
                "Batch processing enabled: trigger={}, timeout={}ms, max_depth={}",
                params.batch.batch_trigger,
                params.batch.timeout_ms,
                params.batch.max_queue_depth
            );

            let config = ScanQueueConfig::from_params(&params.batch);

            // Create alignment function that uses the NDT manager
            let align_ndt_manager = Arc::clone(&ndt_manager);
            let align_fn: scan_queue::AlignFn = Arc::new(move |requests| {
                // Get lock on active NDT manager
                let manager = align_ndt_manager.lock();
                // Use batch alignment through the ndt_cuda API
                let results = manager.align_batch_scans(requests)?;
                Ok(results)
            });

            // Create result callback that publishes poses
            let result_pose_pub = pose_pub.clone();
            let result_pose_cov_pub = pose_cov_pub.clone();
            let result_debug_pubs = debug_pubs.clone();
            let result_params = Arc::clone(&params);
            let result_callback: scan_queue::ResultCallback =
                Arc::new(move |results: Vec<ScanResult>| {
                    for result in results {
                        // Only publish if converged
                        if !result.converged {
                            log_debug!(
                                NODE_NAME,
                                "Batch result skipped (not converged): ts_ns={}, score={:.3}",
                                result.timestamp_ns,
                                result.score
                            );
                            continue;
                        }

                        // Convert Isometry3 to Pose
                        let pose = isometry_to_pose(&result.pose);

                        // Publish PoseStamped
                        let pose_msg = PoseStamped {
                            header: result.header.clone(),
                            pose: pose.clone(),
                        };
                        if let Err(e) = result_pose_pub.publish(&pose_msg) {
                            log_error!(NODE_NAME, "Failed to publish batch pose: {e}");
                        }

                        // Publish PoseWithCovarianceStamped with fixed covariance
                        let pose_cov_msg = PoseWithCovarianceStamped {
                            header: result.header.clone(),
                            pose: PoseWithCovariance {
                                pose: pose.clone(),
                                covariance: result_params.covariance.output_pose_covariance,
                            },
                        };
                        if let Err(e) = result_pose_cov_pub.publish(&pose_cov_msg) {
                            log_error!(
                                NODE_NAME,
                                "Failed to publish batch pose with covariance: {e}"
                            );
                        }

                        // Publish TF transform
                        Self::publish_tf(
                            &result_debug_pubs.tf_pub,
                            &result.timestamp,
                            &pose,
                            &result_params.frame.map_frame,
                            &result_params.frame.ndt_base_frame,
                        );

                        log_debug!(
                        NODE_NAME,
                        "Batch result published: ts_ns={}, iter={}, score={:.3}, latency={:.1}ms",
                        result.timestamp_ns,
                        result.iterations,
                        result.score,
                        result.latency_ms
                    );
                    }
                });

            Some(Arc::new(ScanQueue::new(config, align_fn, result_callback)))
        } else {
            None
        };

        // Points subscription
        // Uses QoS KeepLast(1) to ensure we only process the latest message,
        // matching Autoware's approach (no explicit timestamp deduplication needed)
        let points_sub = {
            let ndt_manager = Arc::clone(&ndt_manager);
            let map_module = Arc::clone(&map_module);
            let map_loader = Arc::clone(&map_loader);
            let map_points = Arc::clone(&map_points);
            let pose_buffer = Arc::clone(&pose_buffer);
            let latest_sensor_points = Arc::clone(&latest_sensor_points);
            let enabled = Arc::clone(&enabled);
            let pose_pub = pose_pub.clone();
            let pose_cov_pub = pose_cov_pub.clone();
            let debug_pubs = debug_pubs.clone();
            let diagnostics = Arc::clone(&diagnostics);
            let params = Arc::clone(&params);
            let tf_handler = Arc::clone(&tf_handler);
            let skip_counter = Arc::clone(&skip_counter);
            let callback_count = Arc::clone(&callback_count);
            let align_count = Arc::clone(&align_count);
            let scan_queue = scan_queue.clone();

            let mut opts = SubscriptionOptions::new("points_raw");
            opts.qos = sensor_qos;

            node.create_subscription(opts, move |msg: PointCloud2| {
                Self::on_points(
                    msg,
                    &ndt_manager,
                    &map_module,
                    &map_loader,
                    &map_points,
                    &pose_buffer,
                    &latest_sensor_points,
                    &enabled,
                    &skip_counter,
                    &callback_count,
                    &align_count,
                    &pose_pub,
                    &pose_cov_pub,
                    &debug_pubs,
                    &diagnostics,
                    &params,
                    &tf_handler,
                    &scan_queue,
                );
            })?
        };

        // Initial pose subscription - pushes to pose buffer for interpolation
        // Uses QoS depth 100 (matching Autoware) to buffer messages during node initialization.
        // This prevents losing early EKF messages before spin() starts processing callbacks.
        let initial_pose_sub = {
            let pose_buffer = Arc::clone(&pose_buffer);

            // Use relative topic name - remapping should be handled by rcl layer
            // Launch file remaps: ekf_pose_with_covariance -> /localization/pose_twist_fusion_filter/biased_pose_with_covariance
            let mut opts = SubscriptionOptions::new("ekf_pose_with_covariance");
            // QoS depth 100 matches Autoware's ndt_scan_matcher_core.cpp line 118
            opts.qos = QoSProfile {
                history: QoSHistoryPolicy::KeepLast { depth: 100 },
                ..QoSProfile::default()
            };

            node.create_subscription(opts, move |msg: PoseWithCovarianceStamped| {
                // Debug: log received EKF pose with timestamp (only with debug-output feature)
                #[cfg(feature = "debug-output")]
                {
                    let p = &msg.pose.pose.position;
                    let q = &msg.pose.pose.orientation;
                    let ts = &msg.header.stamp;
                    log_info!(
                        NODE_NAME,
                        "[EKF_IN] ts={}.{:09} pos=({:.3}, {:.3}, {:.3}) quat=({:.6}, {:.6}, {:.6}, {:.6})",
                        ts.sec, ts.nanosec,
                        p.x, p.y, p.z,
                        q.x, q.y, q.z, q.w
                    );
                }
                pose_buffer.push_back(msg);
            })?
        };

        // Log the actual topic name after remapping
        log_debug!(
            NODE_NAME,
            "EKF pose subscription topic (after remapping): {}",
            initial_pose_sub.topic_name()
        );

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
            let debug_pubs = debug_pubs.clone();
            let params = Arc::clone(&params);

            let mut opts = SubscriptionOptions::new("pointcloud_map");
            opts.qos = QoSProfile::default(); // Reliable for map data

            node.create_subscription(opts, move |msg: PointCloud2| {
                Self::on_map_received(
                    msg,
                    &map_module,
                    &map_points,
                    &ndt_manager,
                    &debug_pubs,
                    &params,
                );
            })?
        };

        // Trigger service
        let trigger_srv = {
            let enabled = Arc::clone(&enabled);
            let pose_buffer = Arc::clone(&pose_buffer);

            node.create_service::<SetBool, _>(
                "trigger_node_srv",
                move |req: SetBoolRequest, _info: rclrs::ServiceInfo| {
                    enabled.store(req.data, Ordering::SeqCst);
                    // Clear pose buffer when enabling (matches Autoware behavior)
                    // This ensures we start fresh with EKF poses from after initialization
                    if req.data {
                        pose_buffer.clear();
                        log_info!(NODE_NAME, "NDT scan matcher enabled (pose buffer cleared)");
                    } else {
                        log_info!(NODE_NAME, "NDT scan matcher disabled");
                    }
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
            let pose_buffer = Arc::clone(&pose_buffer);

            node.create_service::<Trigger, _>(
                "map_update_srv",
                move |_req: TriggerRequest, _info: rclrs::ServiceInfo| {
                    Self::on_map_update(&map_module, &map_points, &ndt_manager, &pose_buffer)
                },
            )?
        };

        // Clear debug file at startup (only with debug-output feature)
        #[cfg(feature = "debug-output")]
        {
            let debug_file = std::env::var("NDT_DEBUG_FILE")
                .unwrap_or_else(|_| "/tmp/ndt_cuda_debug.jsonl".to_string());
            // Truncate the file by opening with write-only (not append)
            if let Ok(mut file) = std::fs::File::create(&debug_file) {
                use std::io::Write;
                use std::time::SystemTime;
                let timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let _ = writeln!(
                    file,
                    r#"{{"run_start": true, "unix_timestamp": {timestamp}}}"#
                );
                log_info!(NODE_NAME, "Debug output cleared: {debug_file}");
            }
        }

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
            pose_buffer,
            latest_sensor_points,
            enabled,
            params,
            tf_handler,
            scan_queue,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn on_points(
        msg: PointCloud2,
        ndt_manager: &Arc<DualNdtManager>,
        map_module: &Arc<MapUpdateModule>,
        map_loader: &Arc<DynamicMapLoader>,
        map_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        pose_buffer: &Arc<SmartPoseBuffer>,
        latest_sensor_points: &Arc<ArcSwap<Option<Vec<[f32; 3]>>>>,
        enabled: &Arc<AtomicBool>,
        skip_counter: &Arc<AtomicI32>,
        callback_count: &Arc<AtomicI32>,
        align_count: &Arc<AtomicI32>,
        pose_pub: &Publisher<PoseStamped>,
        pose_cov_pub: &Publisher<PoseWithCovarianceStamped>,
        debug_pubs: &DebugPublishers,
        diagnostics: &Arc<Mutex<DiagnosticsInterface>>,
        params: &NdtParams,
        tf_handler: &Arc<tf_handler::TfHandler>,
        scan_queue: &Option<Arc<ScanQueue>>,
    ) {
        // Track callback invocation
        let _cb_num = callback_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Extract timestamp for debug output
        let timestamp_ns =
            msg.header.stamp.sec as u64 * 1_000_000_000 + msg.header.stamp.nanosec as u64;

        // Note: No explicit deduplication needed - QoS KeepLast(1) ensures we only
        // process the latest message, matching Autoware's approach.

        // Convert sensor points first - needed for align service even before we have initial pose
        let raw_points = match pointcloud::from_pointcloud2(&msg) {
            Ok(pts) => pts,
            Err(e) => {
                log_error!(NODE_NAME, "Failed to convert point cloud: {e}");
                return;
            }
        };

        // Note: Sensor point filtering (distance, z-height, downsampling) is handled
        // upstream by pointcloud_preprocessor. We use default (no-op) filtering here.
        let filter_params = pointcloud::PointFilterParams::default();
        let filter_result = pointcloud::filter_sensor_points(&raw_points, &filter_params);
        let sensor_points = filter_result.points;

        if sensor_points.len() < raw_points.len() {
            let gpu_str = if filter_result.used_gpu { "GPU" } else { "CPU" };
            log_debug!(
                NODE_NAME,
                "Filtered sensor points: {} -> {} (dist:{}, z:{}, downsample:{}) [{}]",
                raw_points.len(),
                sensor_points.len(),
                filter_result.removed_by_distance,
                filter_result.removed_by_z,
                filter_result.removed_by_downsampling,
                gpu_str
            );
        }

        // Transform sensor points from sensor frame to base_link
        // The sensor frame comes from the PointCloud2 header, target is base_frame from params
        let sensor_frame = &msg.header.frame_id;
        let base_frame = &params.frame.base_frame;
        let stamp_ns =
            msg.header.stamp.sec as i64 * 1_000_000_000 + msg.header.stamp.nanosec as i64;

        let sensor_points = if sensor_frame != base_frame {
            match tf_handler.transform_points(
                &sensor_points,
                sensor_frame,
                base_frame,
                Some(stamp_ns),
            ) {
                Some(transformed) => {
                    log_debug!(
                        NODE_NAME,
                        "Transformed {} points: {} -> {}",
                        transformed.len(),
                        sensor_frame,
                        base_frame
                    );
                    transformed
                }
                None => {
                    // TF not available yet - use points as-is with warning
                    log_warn!(
                        NODE_NAME,
                        "TF not available: {} -> {}, using raw sensor frame",
                        sensor_frame,
                        base_frame
                    );
                    sensor_points
                }
            }
        } else {
            // Already in base_frame, no transform needed
            sensor_points
        };

        // Always store sensor points for initial pose estimation service (ndt_align_srv)
        // This must happen before any early returns so the align service can work
        latest_sensor_points.store(Arc::new(Some(sensor_points.clone())));

        // Check if enabled for regular NDT alignment
        if !enabled.load(Ordering::SeqCst) {
            return;
        }

        // Get initial pose via interpolation to match sensor timestamp
        // This implements Autoware's SmartPoseBuffer behavior for better timestamp alignment
        let sensor_time_ns =
            msg.header.stamp.sec as i64 * 1_000_000_000 + msg.header.stamp.nanosec as i64;

        let interpolate_result = pose_buffer.interpolate(sensor_time_ns);
        let initial_pose = match &interpolate_result {
            Some(result) => {
                // Debug: log interpolated pose (only with debug-output feature)
                #[cfg(feature = "debug-output")]
                {
                    let p = &result.interpolated_pose.pose.pose.position;
                    let q = &result.interpolated_pose.pose.pose.orientation;
                    let ts = &result.interpolated_pose.header.stamp;
                    // Convert quaternion to euler angles for easier comparison
                    let quat =
                        UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));
                    let (roll, pitch, yaw) = quat.euler_angles();
                    log_info!(
                        NODE_NAME,
                        "[INTERP] ts={}.{:09} pos=({:.3}, {:.3}, {:.3}) rpy=({:.3}, {:.3}, {:.3}) sensor_ts={}",
                        ts.sec, ts.nanosec,
                        p.x, p.y, p.z,
                        roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees(),
                        sensor_time_ns
                    );
                }
                &result.interpolated_pose
            }
            None => {
                // Interpolation failed - need at least 2 poses, or validation failed
                if pose_buffer.len() < 2 {
                    log_debug!(
                        NODE_NAME,
                        "Waiting for pose buffer to fill (size={}, need 2)",
                        pose_buffer.len()
                    );
                } else {
                    log_warn!(
                        NODE_NAME,
                        "Pose interpolation failed (validation error or timestamp mismatch)"
                    );
                }
                return;
            }
        };

        // Pop old poses to prevent unbounded buffer growth
        pose_buffer.pop_old(sensor_time_ns);

        // Note: Early alignments may have roll=0, pitch=0 (unrefined initial pose)
        // before EKF has fused any NDT output. These alignments may have indefinite
        // Hessians, but the regularization in newton.rs handles this correctly.
        // We process them anyway to bootstrap the EKF with NDT data.

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

            // Publish debug map for visualization
            let debug_map_msg = pointcloud::to_pointcloud2(
                &filtered_map,
                &Header {
                    stamp: msg.header.stamp.clone(),
                    frame_id: params.frame.map_frame.clone(),
                },
            );
            let _ = debug_pubs
                .debug_loaded_pointcloud_map_pub
                .publish(&debug_map_msg);

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

        // ---- Batch Mode: Enqueue scan and return ----
        // If batch processing is enabled, enqueue the scan for parallel GPU processing
        // and return immediately. Results will be published asynchronously by the
        // scan queue's result callback.
        if let Some(queue) = scan_queue {
            // Convert initial pose to Isometry3
            let p = &initial_pose.pose.pose.position;
            let q = &initial_pose.pose.pose.orientation;
            let translation = Translation3::new(p.x, p.y, p.z);
            let quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));
            let initial_isometry = Isometry3::from_parts(translation, quaternion);

            let queued_scan = QueuedScan {
                points: sensor_points.clone(),
                initial_pose: initial_isometry,
                timestamp: msg.header.stamp.clone(),
                timestamp_ns,
                header: msg.header.clone(),
                arrival_time: Instant::now(),
            };

            let enqueued = queue.enqueue(queued_scan);
            if enqueued {
                log_debug!(
                    NODE_NAME,
                    "Scan enqueued for batch processing: ts_ns={}, n_pts={}",
                    timestamp_ns,
                    sensor_points.len()
                );
            }

            // Return early - result will be published by the scan queue callback
            return;
        }

        // ---- Synchronous Mode: Run NDT alignment directly ----

        // Debug: log pose being passed to NDT alignment (only with debug-output feature)
        #[cfg(feature = "debug-output")]
        {
            let p = &initial_pose.pose.pose.position;
            let q = &initial_pose.pose.pose.orientation;
            let quat = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));
            let (roll, pitch, yaw) = quat.euler_angles();
            log_info!(
                NODE_NAME,
                "[NDT_IN] ts_ns={} pos=({:.3}, {:.3}, {:.3}) rpy=({:.3}, {:.3}, {:.3}) n_pts={}",
                timestamp_ns,
                p.x,
                p.y,
                p.z,
                roll.to_degrees(),
                pitch.to_degrees(),
                yaw.to_degrees(),
                sensor_points.len()
            );
        }

        // Get lock on active NDT manager (also checks for pending swap from background update)
        let mut manager = ndt_manager.lock();

        // Compute "before" scores at initial pose (for diagnostics comparison)
        let transform_prob_before = manager
            .evaluate_transform_probability(&sensor_points, &initial_pose.pose.pose)
            .unwrap_or(0.0);
        let nvtl_before = manager
            .evaluate_nvtl(&sensor_points, map, &initial_pose.pose.pose, 0.55)
            .unwrap_or(0.0);

        // Start execution timer here to measure only NDT alignment (matches Autoware's scope)
        let align_start_time = Instant::now();

        // With debug-output feature: collect and write debug data
        #[cfg(feature = "debug-output")]
        let (result, alignment_debug) = match manager.align_with_debug(
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
                (r, Some(debug))
            }
            Err(e) => {
                log_error!(NODE_NAME, "NDT alignment failed: {e}");
                return;
            }
        };

        // Without debug-output feature: just run alignment
        #[cfg(all(not(feature = "debug-output"), feature = "debug-markers"))]
        let (result, alignment_debug): (_, Option<AlignmentDebug>) =
            match manager.align(&sensor_points, map, &initial_pose.pose.pose) {
                Ok(r) => (r, None),
                Err(e) => {
                    log_error!(NODE_NAME, "NDT alignment failed: {e}");
                    return;
                }
            };

        // Without either debug feature: just run alignment
        #[cfg(all(not(feature = "debug-output"), not(feature = "debug-markers")))]
        let (result, _alignment_debug): (_, Option<()>) =
            match manager.align(&sensor_points, map, &initial_pose.pose.pose) {
                Ok(r) => (r, None),
                Err(e) => {
                    log_error!(NODE_NAME, "NDT alignment failed: {e}");
                    return;
                }
            };

        // Calculate execution time immediately after alignment (matches Autoware's scope)
        let exe_time_ms = align_start_time.elapsed().as_secs_f32() * 1000.0;

        // Debug: log NDT result (only with debug-output feature)
        #[cfg(feature = "debug-output")]
        {
            let p = &result.pose.position;
            let q = &result.pose.orientation;
            let quat = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));
            let (roll, pitch, yaw) = quat.euler_angles();
            log_info!(
                NODE_NAME,
                "[NDT_OUT] ts_ns={} pos=({:.3}, {:.3}, {:.3}) rpy=({:.3}, {:.3}, {:.3}) iter={} conv={} osc={}",
                timestamp_ns,
                p.x, p.y, p.z,
                roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees(),
                result.iterations,
                result.converged,
                result.oscillation_count
            );
        }

        // ---- Compute scores for filtering decision ----
        // Like Autoware, we compute NVTL and transform_probability before deciding to publish

        // Compute transform probability (fitness score converted to probability)
        let transform_prob = (-result.score / 10.0).exp();

        // Compute NVTL score
        let nvtl_score = manager
            .evaluate_nvtl(&sensor_points, map, &result.pose, 0.55)
            .unwrap_or(0.0);

        // ---- Convergence gating (matching Autoware's behavior) ----
        // Autoware gates pose publishing on three conditions:
        // 1. is_ok_iteration_num: did NOT hit max iterations (result.converged)
        // 2. is_local_optimal_solution_oscillation: oscillation count <= 10
        // 3. is_ok_score: score above threshold

        // Check 1: Max iterations (result.converged is false when max iterations reached)
        let is_ok_iteration_num = result.converged;

        // Check 2: Oscillation count (Autoware uses threshold of 10)
        const OSCILLATION_THRESHOLD: usize = 10;
        let is_ok_oscillation = result.oscillation_count <= OSCILLATION_THRESHOLD;

        // Check 3: Score threshold
        // converged_param_type: 0 = transform_probability, 1 = NVTL
        let (score_for_check, threshold, score_name) = if params.score.converged_param_type == 0 {
            (
                transform_prob,
                params.score.converged_param_transform_probability,
                "transform_probability",
            )
        } else {
            (
                nvtl_score,
                params
                    .score
                    .converged_param_nearest_voxel_transformation_likelihood,
                "NVTL",
            )
        };
        let is_ok_score = score_for_check >= threshold;

        // Combined convergence check (all three must pass)
        let is_converged = is_ok_iteration_num && is_ok_oscillation && is_ok_score;

        // Track consecutive skips for diagnostics and log reasons
        let skipping_publish_num = if !is_converged {
            let skips = skip_counter.fetch_add(1, Ordering::SeqCst) + 1;

            // Log specific reason(s) for skipping
            if !is_ok_iteration_num {
                log_warn!(
                    NODE_NAME,
                    "Max iterations reached: iter={}, skip_count={skips}",
                    result.iterations
                );
            }
            if !is_ok_oscillation {
                log_warn!(
                    NODE_NAME,
                    "Oscillation detected: count={} > {OSCILLATION_THRESHOLD}, skip_count={skips}",
                    result.oscillation_count
                );
            }
            if !is_ok_score {
                log_warn!(
                    NODE_NAME,
                    "Score below threshold: {score_name}={score_for_check:.3} < {threshold:.3}, skip_count={skips}"
                );
            }
            skips
        } else {
            skip_counter.store(0, Ordering::SeqCst);
            0
        };

        // Create output header (needed for debug publishers even if we skip pose publishing)
        let header = Header {
            stamp: msg.header.stamp.clone(),
            frame_id: params.frame.map_frame.clone(),
        };

        // Only publish pose if all convergence conditions pass
        if is_converged {
            // Estimate covariance based on configured mode
            // For MULTI_NDT modes, we use parallel batch evaluation (Rayon)
            // NOTE: We reuse the manager lock from the alignment - don't try to lock again!
            let covariance_result = covariance::estimate_covariance_full(
                &params.covariance,
                &result.hessian,
                &result.pose,
                Some(&mut *manager), // Reuse existing lock to avoid deadlock
                Some(&sensor_points),
                Some(map),
            );

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

            // Publish MULTI_NDT poses for debug visualization (only for MULTI_NDT modes)
            if let Some(poses) = covariance_result.multi_ndt_poses {
                let pose_array_msg = PoseArray {
                    header: header.clone(),
                    poses,
                };
                let _ = debug_pubs.multi_ndt_pose_pub.publish(&pose_array_msg);
            }

            // Publish MULTI_NDT initial poses for debug visualization
            if let Some(poses) = covariance_result.multi_initial_poses {
                let pose_array_msg = PoseArray {
                    header: header.clone(),
                    poses,
                };
                let _ = debug_pubs.multi_initial_pose_pub.publish(&pose_array_msg);
            }
        }

        // ---- Debug Publishers (always publish for monitoring) ----

        // Track successful alignment
        let align_num = align_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Log periodic summary every 50 alignments
        if align_num % 50 == 0 {
            let total_cb = callback_count.load(Ordering::SeqCst);
            log_info!(
                NODE_NAME,
                "Callback stats: total={total_cb}, aligned={align_num}"
            );
        }

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

        // Publish transform probability
        let transform_prob_msg = Float32Stamped {
            stamp: msg.header.stamp.clone(),
            data: transform_prob as f32,
        };
        let _ = debug_pubs
            .transform_probability_pub
            .publish(&transform_prob_msg);

        // Publish NVTL score
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

        // Calculate distance from old/new interpolation poses to result
        // (interpolate_result is guaranteed to be Some here since we returned early otherwise)
        if let Some(ref interp) = interpolate_result {
            // Distance from old pose (older of the two bracketing poses) to result
            let dx_old = result.pose.position.x - interp.old_pose.pose.pose.position.x;
            let dy_old = result.pose.position.y - interp.old_pose.pose.pose.position.y;
            let dz_old = result.pose.position.z - interp.old_pose.pose.pose.position.z;
            let distance_old = (dx_old * dx_old + dy_old * dy_old + dz_old * dz_old).sqrt();
            let _ = debug_pubs
                .initial_to_result_distance_old_pub
                .publish(&Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: distance_old as f32,
                });

            // Distance from new pose (newer of the two bracketing poses) to result
            let dx_new = result.pose.position.x - interp.new_pose.pose.pose.position.x;
            let dy_new = result.pose.position.y - interp.new_pose.pose.pose.position.y;
            let dz_new = result.pose.position.z - interp.new_pose.pose.pose.position.z;
            let distance_new = (dx_new * dx_new + dy_new * dy_new + dz_new * dz_new).sqrt();
            let _ = debug_pubs
                .initial_to_result_distance_new_pub
                .publish(&Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: distance_new as f32,
                });
        }

        // Publish relative pose (result relative to initial)
        // Compute actual relative transform: relative = result * initial^(-1)
        // This gives the transform that takes you from initial pose to result pose
        let initial_p = &initial_pose.pose.pose.position;
        let initial_q = &initial_pose.pose.pose.orientation;
        let initial_translation = Translation3::new(initial_p.x, initial_p.y, initial_p.z);
        let initial_quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(
            initial_q.w,
            initial_q.x,
            initial_q.y,
            initial_q.z,
        ));
        let initial_isometry: Isometry3<f64> =
            Isometry3::from_parts(initial_translation, initial_quaternion);

        let result_p = &result.pose.position;
        let result_q = &result.pose.orientation;
        let result_translation_rel = Translation3::new(result_p.x, result_p.y, result_p.z);
        let result_quaternion_rel = UnitQuaternion::from_quaternion(NaQuaternion::new(
            result_q.w, result_q.x, result_q.y, result_q.z,
        ));
        let result_isometry_rel: Isometry3<f64> =
            Isometry3::from_parts(result_translation_rel, result_quaternion_rel);

        // Compute relative transform: result * initial^(-1)
        let relative_isometry = result_isometry_rel * initial_isometry.inverse();
        let relative_pose = isometry_to_pose(&relative_isometry);

        let relative_pose_msg = PoseStamped {
            header: header.clone(),
            pose: relative_pose,
        };
        let _ = debug_pubs
            .initial_to_result_relative_pose_pub
            .publish(&relative_pose_msg);

        // Publish NDT marker (pose history visualization)
        // When debug-markers is enabled, publish pose history from debug data.
        // Otherwise, just publish the final pose.
        #[cfg(feature = "debug-markers")]
        let marker_array = if let Some(ref debug) = alignment_debug {
            Self::create_pose_history_markers(&header, debug)
        } else {
            let ndt_marker = Self::create_pose_marker(&header, &result.pose, 0);
            MarkerArray {
                markers: vec![ndt_marker],
            }
        };

        #[cfg(not(feature = "debug-markers"))]
        let marker_array = {
            let ndt_marker = Self::create_pose_marker(&header, &result.pose, 0);
            MarkerArray {
                markers: vec![ndt_marker],
            }
        };

        let _ = debug_pubs.ndt_marker_pub.publish(&marker_array);

        // Publish aligned points (transformed sensor points with proper rotation)
        // Build isometry from result pose for proper point transformation
        let result_translation = Translation3::new(
            result.pose.position.x,
            result.pose.position.y,
            result.pose.position.z,
        );
        let result_quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(
            result.pose.orientation.w,
            result.pose.orientation.x,
            result.pose.orientation.y,
            result.pose.orientation.z,
        ));
        let result_isometry: Isometry3<f64> =
            Isometry3::from_parts(result_translation, result_quaternion);

        let aligned_points: Vec<[f32; 3]> = sensor_points
            .iter()
            .map(|p| {
                // Transform point by result pose (rotation + translation)
                let sensor_pt = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                let map_pt = result_isometry * nalgebra::Point3::from(sensor_pt);
                [map_pt.x as f32, map_pt.y as f32, map_pt.z as f32]
            })
            .collect();
        let aligned_msg = pointcloud::to_pointcloud2(&aligned_points, &header);
        let _ = debug_pubs.points_aligned_pub.publish(&aligned_msg);

        // ---- Per-Point Score Visualization (requires debug-markers feature) ----
        // Compute per-point NDT scores and publish as RGB-colored point cloud.
        // This matches Autoware's voxel_score_points output for debugging.
        #[cfg(feature = "debug-markers")]
        if let Ok((score_points, scores)) =
            manager.compute_per_point_scores_for_visualization(&sensor_points, &result.pose)
        {
            // Convert scores to RGB colors using Autoware's color scheme
            let rgb_values: Vec<u32> = scores
                .iter()
                .map(|&score| {
                    ndt_cuda::scoring::color_to_rgb_packed(&ndt_cuda::scoring::ndt_score_to_color(
                        score,
                        ndt_cuda::scoring::DEFAULT_SCORE_LOWER,
                        ndt_cuda::scoring::DEFAULT_SCORE_UPPER,
                    ))
                })
                .collect();

            let score_cloud_msg =
                pointcloud::to_pointcloud2_with_rgb(&score_points, &rgb_values, &header);
            let _ = debug_pubs.voxel_score_points_pub.publish(&score_cloud_msg);
        }

        // ---- No-Ground Scoring (optional) ----
        // When enabled, filters out ground points and computes scores on the remaining points.
        // Ground is defined as points with transformed_z - base_link_z <= z_margin.
        if params.score.no_ground_points.enable {
            // Build isometry from result pose for transforming points
            let p = &result.pose.position;
            let q = &result.pose.orientation;
            let translation = Translation3::new(p.x, p.y, p.z);
            let quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));
            let pose_isometry: Isometry3<f64> = Isometry3::from_parts(translation, quaternion);
            let base_link_z = p.z;
            let z_threshold = params.score.no_ground_points.z_margin_for_ground_removal as f64;

            // Filter sensor points: keep those whose transformed z is above ground threshold
            let no_ground_points: Vec<[f32; 3]> = sensor_points
                .iter()
                .filter(|pt| {
                    // Transform point to map frame
                    let sensor_pt = Vector3::new(pt[0] as f64, pt[1] as f64, pt[2] as f64);
                    let map_pt = pose_isometry * nalgebra::Point3::from(sensor_pt);
                    // Keep if point_z - base_link_z > threshold
                    map_pt.z - base_link_z > z_threshold
                })
                .copied()
                .collect();

            if !no_ground_points.is_empty() {
                // Compute scores on filtered (non-ground) points
                let no_ground_tp = manager
                    .evaluate_transform_probability(&no_ground_points, &result.pose)
                    .unwrap_or(0.0);
                let no_ground_nvtl = manager
                    .evaluate_nvtl(&no_ground_points, map, &result.pose, 0.55)
                    .unwrap_or(0.0);

                // Publish filtered point cloud (in map frame for visualization)
                let no_ground_aligned: Vec<[f32; 3]> = no_ground_points
                    .iter()
                    .map(|pt| {
                        let sensor_pt = Vector3::new(pt[0] as f64, pt[1] as f64, pt[2] as f64);
                        let map_pt = pose_isometry * nalgebra::Point3::from(sensor_pt);
                        [map_pt.x as f32, map_pt.y as f32, map_pt.z as f32]
                    })
                    .collect();
                let no_ground_cloud_msg = pointcloud::to_pointcloud2(&no_ground_aligned, &header);
                let _ = debug_pubs
                    .no_ground_points_aligned_pub
                    .publish(&no_ground_cloud_msg);

                // Publish no-ground transform probability
                let no_ground_tp_msg = Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: no_ground_tp as f32,
                };
                let _ = debug_pubs
                    .no_ground_transform_probability_pub
                    .publish(&no_ground_tp_msg);

                // Publish no-ground NVTL
                let no_ground_nvtl_msg = Float32Stamped {
                    stamp: msg.header.stamp.clone(),
                    data: no_ground_nvtl as f32,
                };
                let _ = debug_pubs.no_ground_nvtl_pub.publish(&no_ground_nvtl_msg);
            }
        }

        // ---- Diagnostics ----
        // Collect and publish scan matching diagnostics
        let topic_time_stamp = msg.header.stamp.sec as f64 + msg.header.stamp.nanosec as f64 * 1e-9;

        // Extract per-iteration arrays from AlignmentDebug if available
        #[cfg(feature = "debug-output")]
        let (tp_array, nvtl_array) = alignment_debug
            .as_ref()
            .map(|d| {
                // Arrays are only populated when ndt_cuda is built with debug-iteration feature
                #[cfg(feature = "debug-iteration")]
                {
                    let tp = if d.transform_probability_array.is_empty() {
                        None
                    } else {
                        Some(d.transform_probability_array.clone())
                    };
                    let nvtl = if d.nearest_voxel_transformation_likelihood_array.is_empty() {
                        None
                    } else {
                        Some(d.nearest_voxel_transformation_likelihood_array.clone())
                    };
                    (tp, nvtl)
                }
                #[cfg(not(feature = "debug-iteration"))]
                {
                    (None, None)
                }
            })
            .unwrap_or((None, None));

        #[cfg(not(feature = "debug-output"))]
        let (tp_array, nvtl_array): (Option<Vec<f64>>, Option<Vec<f64>>) = (None, None);

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
            transform_probability_before: transform_prob_before,
            nearest_voxel_transformation_likelihood_before: nvtl_before,
            distance_initial_to_result: distance,
            execution_time_ms: exe_time_ms as f64,
            skipping_publish_num,
            transform_probability_array: tp_array,
            nearest_voxel_transformation_likelihood_array: nvtl_array,
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

    /// Create pose history markers from AlignmentDebug data.
    ///
    /// Publishes arrows showing the pose at each iteration of NDT optimization.
    /// Matches Autoware's transformation_array visualization.
    #[cfg(feature = "debug-markers")]
    fn create_pose_history_markers(header: &Header, debug: &AlignmentDebug) -> MarkerArray {
        let mut markers = Vec::new();

        // Convert each 4x4 transformation matrix to a Pose and create a marker
        for (i, matrix_flat) in debug.transformation_array.iter().enumerate() {
            if matrix_flat.len() < 16 {
                continue; // Skip malformed matrices
            }

            // Extract translation from matrix (last column: indices 3, 7, 11)
            let position = Point {
                x: matrix_flat[3],
                y: matrix_flat[7],
                z: matrix_flat[11],
            };

            // Extract rotation matrix (3x3 upper-left block)
            let r00 = matrix_flat[0];
            let r01 = matrix_flat[1];
            let r02 = matrix_flat[2];
            let r10 = matrix_flat[4];
            let r11 = matrix_flat[5];
            let r12 = matrix_flat[6];
            let r20 = matrix_flat[8];
            let r21 = matrix_flat[9];
            let r22 = matrix_flat[10];

            // Convert rotation matrix to quaternion
            let rot_matrix = nalgebra::Matrix3::new(r00, r01, r02, r10, r11, r12, r20, r21, r22);
            let rotation = nalgebra::Rotation3::from_matrix_unchecked(rot_matrix);
            let quat = UnitQuaternion::from_rotation_matrix(&rotation);

            let orientation = geometry_msgs::msg::Quaternion {
                x: quat.i,
                y: quat.j,
                z: quat.k,
                w: quat.w,
            };

            let pose = Pose {
                position,
                orientation,
            };

            // Create marker with gradient color (blue -> cyan -> green)
            // to show progression through iterations
            let progress = i as f32 / debug.transformation_array.len().max(1) as f32;
            let (r, g, b) = if progress < 0.5 {
                // Blue to cyan
                let t = progress * 2.0;
                (0.0, t, 1.0)
            } else {
                // Cyan to green
                let t = (progress - 0.5) * 2.0;
                (0.0, 1.0, 1.0 - t)
            };

            markers.push(Marker {
                header: header.clone(),
                ns: "result_pose_matrix_array".to_string(),
                id: i as i32,
                type_: 0,  // ARROW
                action: 0, // ADD
                pose,
                scale: geometry_msgs::msg::Vector3 {
                    x: 0.3,
                    y: 0.1,
                    z: 0.1,
                },
                color: std_msgs::msg::ColorRGBA { r, g, b, a: 0.8 },
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
            });
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

        // Publish Monte Carlo particle visualization with multiple color schemes
        let markers = visualization::create_monte_carlo_markers_enhanced(
            &initial_pose.header,
            &result.particles,
            result.score,
            &ParticleMarkerConfig::default(),
        );
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
        debug_pubs: &DebugPublishers,
        params: &NdtParams,
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

        // Publish debug map for visualization
        let debug_map_msg = pointcloud::to_pointcloud2(
            &points,
            &Header {
                stamp: msg.header.stamp.clone(),
                frame_id: params.frame.map_frame.clone(),
            },
        );
        let _ = debug_pubs
            .debug_loaded_pointcloud_map_pub
            .publish(&debug_map_msg);

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
        pose_buffer: &Arc<SmartPoseBuffer>,
    ) -> TriggerResponse {
        // Get current position from latest pose in buffer
        let position = match pose_buffer.latest() {
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
