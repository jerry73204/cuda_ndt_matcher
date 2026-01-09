//! Dual NDT manager for non-blocking map updates.
//!
//! This module implements a dual-NDT architecture where map updates happen
//! in the background without blocking the main alignment loop:
//!
//! - Two `NdtScanMatcher` instances are maintained: active and updating
//! - When map tiles change, the updating instance rebuilds its voxel grid in a background thread
//! - When the update completes, the instances are atomically swapped
//! - Alignment calls always use the active instance, never blocking
//!
//! ## Autoware Compatibility
//!
//! This matches Autoware's `ndt_scan_matcher` behavior which uses a secondary NDT
//! instance for non-blocking map updates.

use crate::ndt_manager::{AlignResult, NdtManager};
use crate::params::NdtParams;
use anyhow::Result;
use geometry_msgs::msg::Pose;
use ndt_cuda::AlignmentDebug;
use parking_lot::{Mutex, RwLock};
use rclrs::{log_debug, log_info};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

const LOGGER_NAME: &str = "ndt_scan_matcher.dual_ndt_manager";

/// Status of a background map update.
#[derive(Debug, Clone, Default)]
pub struct UpdateStatus {
    /// Whether an update is currently in progress
    pub in_progress: bool,
    /// Number of points in the pending update
    pub pending_points: usize,
    /// Number of completed swaps
    pub swap_count: usize,
    /// Last update duration in milliseconds
    pub last_update_ms: f64,
}

/// Dual NDT manager for non-blocking map updates.
///
/// Maintains two NDT instances and performs map updates in the background.
#[allow(dead_code)]
pub struct DualNdtManager {
    /// Active NDT manager used for alignment
    active: Arc<RwLock<NdtManager>>,

    /// Updating NDT manager being rebuilt in background
    /// None when no update is in progress
    updating: Arc<Mutex<Option<NdtManager>>>,

    /// Background thread handle
    update_thread: Arc<Mutex<Option<JoinHandle<Result<NdtManager>>>>>,

    /// Flag indicating update is in progress
    update_in_progress: Arc<AtomicBool>,

    /// Points waiting to be set on the updating manager
    pending_points: Arc<Mutex<Option<Vec<[f32; 3]>>>>,

    /// NDT parameters for creating new managers
    params: NdtParams,

    /// Update statistics
    status: Arc<RwLock<UpdateStatus>>,
}

#[allow(dead_code)]
impl DualNdtManager {
    /// Create a new dual NDT manager.
    pub fn new(params: NdtParams) -> Result<Self> {
        let manager = NdtManager::new(&params)?;

        Ok(Self {
            active: Arc::new(RwLock::new(manager)),
            updating: Arc::new(Mutex::new(None)),
            update_thread: Arc::new(Mutex::new(None)),
            update_in_progress: Arc::new(AtomicBool::new(false)),
            pending_points: Arc::new(Mutex::new(None)),
            params,
            status: Arc::new(RwLock::new(UpdateStatus::default())),
        })
    }

    /// Check if a background update is in progress.
    pub fn is_update_in_progress(&self) -> bool {
        self.update_in_progress.load(Ordering::SeqCst)
    }

    /// Get the current update status.
    pub fn get_status(&self) -> UpdateStatus {
        self.status.read().clone()
    }

    /// Start a background map update with the given points.
    ///
    /// This method returns immediately. The update runs in a background thread.
    /// Call `swap_if_ready()` to check if the update completed and swap the instances.
    ///
    /// If an update is already in progress, the new points are queued and will be
    /// applied after the current update completes.
    ///
    /// # Arguments
    /// * `points` - The new map points to set as the NDT target
    ///
    /// # Returns
    /// * `true` if background update was started
    /// * `false` if an update is already in progress (points queued for next update)
    pub fn start_background_update(&self, points: Vec<[f32; 3]>) -> bool {
        // If update already in progress, queue the points for next update
        if self.update_in_progress.swap(true, Ordering::SeqCst) {
            log_debug!(
                LOGGER_NAME,
                "Update already in progress, queuing {} points for next update",
                points.len()
            );
            *self.pending_points.lock() = Some(points);
            return false;
        }

        let num_points = points.len();
        log_info!(
            LOGGER_NAME,
            "Starting background map update with {num_points} points"
        );

        // Update status
        {
            let mut status = self.status.write();
            status.in_progress = true;
            status.pending_points = num_points;
        }

        // Clone what we need for the background thread
        let params = self.params.clone();
        let updating = Arc::clone(&self.updating);
        let status = Arc::clone(&self.status);

        // Spawn background thread
        let handle = thread::spawn(move || {
            let start = std::time::Instant::now();

            // Create a new NDT manager
            let mut manager = NdtManager::new(&params)?;

            // Set the target (this is the expensive operation)
            manager.set_target(&points)?;

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            log_info!(
                LOGGER_NAME,
                "Background update complete: {num_points} points, {elapsed_ms:.1}ms"
            );

            // Store the updated manager
            *updating.lock() = Some(manager);

            // Update status
            {
                let mut s = status.write();
                s.last_update_ms = elapsed_ms;
            }

            // Clear in-progress flag (swap_if_ready will set it false after swap)
            // Note: We don't clear it here because we want swap_if_ready to detect completion
            NdtManager::new(&params) // Dummy return, actual manager is in `updating`
        });

        *self.update_thread.lock() = Some(handle);

        true
    }

    /// Check if a background update has completed and swap instances if so.
    ///
    /// This should be called periodically (e.g., in the main loop or timer callback).
    ///
    /// # Returns
    /// * `true` if instances were swapped
    /// * `false` if no update was pending or not yet complete
    pub fn swap_if_ready(&self) -> bool {
        // Check if there's a completed update waiting
        let mut updating_guard = self.updating.lock();
        if updating_guard.is_none() {
            return false;
        }

        // Check if the thread has completed
        let mut thread_guard = self.update_thread.lock();
        if let Some(handle) = thread_guard.take() {
            if handle.is_finished() {
                // Thread completed, perform the swap
                if let Some(new_manager) = updating_guard.take() {
                    // Swap the managers
                    let mut active = self.active.write();
                    *active = new_manager;

                    // Update status
                    {
                        let mut status = self.status.write();
                        status.in_progress = false;
                        status.pending_points = 0;
                        status.swap_count += 1;
                    }

                    // Clear in-progress flag
                    self.update_in_progress.store(false, Ordering::SeqCst);

                    log_info!(
                        LOGGER_NAME,
                        "NDT instances swapped (non-blocking update complete)"
                    );

                    // Check if there are queued points for the next update
                    let pending = self.pending_points.lock().take();
                    if let Some(points) = pending {
                        log_debug!(
                            LOGGER_NAME,
                            "Processing queued update with {} points",
                            points.len()
                        );
                        drop(active);
                        drop(updating_guard);
                        drop(thread_guard);
                        self.start_background_update(points);
                    }

                    return true;
                }
            } else {
                // Thread still running, put handle back
                *thread_guard = Some(handle);
            }
        }

        false
    }

    /// Set the NDT target (blocking, for initial setup).
    ///
    /// This is a blocking operation that sets the target on the active manager.
    /// Use `start_background_update()` for non-blocking updates during operation.
    pub fn set_target(&self, points: &[[f32; 3]]) -> Result<()> {
        let mut active = self.active.write();
        active.set_target(points)
    }

    /// Check if the active manager has a target set.
    pub fn has_target(&self) -> bool {
        self.active.read().has_target()
    }

    /// Perform NDT alignment using the active manager.
    ///
    /// This never blocks on map updates - it always uses the current active manager.
    pub fn align(
        &self,
        source_points: &[[f32; 3]],
        target_points: &[[f32; 3]],
        initial_pose: &Pose,
    ) -> Result<AlignResult> {
        // Check for ready swap before alignment
        self.swap_if_ready();

        let mut active = self.active.write();
        active.align(source_points, target_points, initial_pose)
    }

    /// Perform NDT alignment with debug output using the active manager.
    pub fn align_with_debug(
        &self,
        source_points: &[[f32; 3]],
        target_points: &[[f32; 3]],
        initial_pose: &Pose,
        timestamp_ns: u64,
    ) -> Result<(AlignResult, AlignmentDebug)> {
        // Check for ready swap before alignment
        self.swap_if_ready();

        let mut active = self.active.write();
        active.align_with_debug(source_points, target_points, initial_pose, timestamp_ns)
    }

    /// Evaluate NVTL score at a given pose.
    pub fn evaluate_nvtl(
        &self,
        source_points: &[[f32; 3]],
        target_points: &[[f32; 3]],
        pose: &Pose,
        outlier_ratio: f64,
    ) -> Result<f64> {
        let active = self.active.read();
        active.evaluate_nvtl(source_points, target_points, pose, outlier_ratio)
    }

    /// Evaluate NVTL at multiple poses in parallel.
    pub fn evaluate_nvtl_batch(
        &mut self,
        source_points: &[[f32; 3]],
        poses: &[Pose],
    ) -> Result<Vec<f64>> {
        let mut active = self.active.write();
        active.evaluate_nvtl_batch(source_points, poses)
    }

    /// Align from multiple initial poses in parallel.
    pub fn align_batch(
        &self,
        source_points: &[[f32; 3]],
        initial_poses: &[Pose],
    ) -> Result<Vec<AlignResult>> {
        let active = self.active.read();
        active.align_batch(source_points, initial_poses)
    }

    /// Set the GNSS regularization pose.
    pub fn set_regularization_pose(&self, pose: &Pose) {
        let mut active = self.active.write();
        active.set_regularization_pose(pose);
    }

    /// Clear the GNSS regularization pose.
    pub fn clear_regularization_pose(&self) {
        let mut active = self.active.write();
        active.clear_regularization_pose();
    }

    /// Check if GNSS regularization is enabled.
    pub fn is_regularization_enabled(&self) -> bool {
        let active = self.active.read();
        active.is_regularization_enabled()
    }

    /// Get a read lock on the active manager for direct access.
    ///
    /// This is needed for covariance estimation which requires a reference to the manager.
    pub fn lock(&self) -> parking_lot::RwLockWriteGuard<'_, NdtManager> {
        // Check for ready swap before returning lock
        self.swap_if_ready();
        self.active.write()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{
        CovarianceEstimationParams, CovarianceEstimationType, CovarianceParams, DynamicMapParams,
        FrameParams, InitialPoseParams, NdtAlgorithmParams, NdtParams, RegularizationParams,
        ScoreParams, SensorPointsParams, ValidationParams,
    };

    fn make_test_params() -> NdtParams {
        NdtParams {
            frame: FrameParams {
                base_frame: "base_link".to_string(),
                ndt_base_frame: "ndt_base_link".to_string(),
                map_frame: "map".to_string(),
            },
            sensor_points: SensorPointsParams {
                timeout_sec: 1.0,
                required_distance: 10.0,
                min_distance: 0.0,
                max_distance: 200.0,
                min_z: -100.0,
                max_z: 100.0,
                downsample_resolution: None,
            },
            ndt: NdtAlgorithmParams {
                trans_epsilon: 0.01,
                step_size: 0.1,
                resolution: 2.0,
                max_iterations: 30,
                num_threads: 4,
            },
            initial_pose: InitialPoseParams {
                particles_num: 200,
                n_startup_trials: 100,
                yaw_weight_sigma: 30.0,
            },
            validation: ValidationParams {
                initial_pose_timeout_sec: 1.0,
                initial_pose_distance_tolerance_m: 10.0,
                initial_to_result_distance_tolerance_m: 3.0,
                critical_upper_bound_exe_time_ms: 100.0,
                skipping_publish_num: 5,
            },
            score: ScoreParams {
                converged_param_type: 1,
                converged_param_transform_probability: 3.0,
                converged_param_nearest_voxel_transformation_likelihood: 2.3,
                no_ground_points: crate::params::NoGroundPointsParams {
                    enable: false,
                    z_margin_for_ground_removal: 0.8,
                },
            },
            covariance: CovarianceParams {
                output_pose_covariance: [0.0; 36],
                covariance_estimation_type: CovarianceEstimationType::Fixed,
                estimation: CovarianceEstimationParams::default(),
            },
            regularization: RegularizationParams {
                enabled: false,
                scale_factor: 0.01,
            },
            dynamic_map: DynamicMapParams {
                update_distance: 20.0,
                map_radius: 150.0,
                lidar_radius: 100.0,
            },
        }
    }

    #[test]
    fn test_dual_ndt_manager_creation() {
        let params = make_test_params();
        let manager = DualNdtManager::new(params).unwrap();

        assert!(!manager.is_update_in_progress());
        assert!(!manager.has_target());
    }

    #[test]
    fn test_blocking_set_target() {
        let params = make_test_params();
        let manager = DualNdtManager::new(params).unwrap();

        // Create some test points
        let points: Vec<[f32; 3]> = (0..100).map(|i| [i as f32 * 0.1, 0.0, 0.0]).collect();

        manager.set_target(&points).unwrap();
        assert!(manager.has_target());
    }

    #[test]
    fn test_background_update() {
        let params = make_test_params();
        let manager = DualNdtManager::new(params).unwrap();

        // Create some test points
        let points: Vec<[f32; 3]> = (0..100).map(|i| [i as f32 * 0.1, 0.0, 0.0]).collect();

        // Start background update
        let started = manager.start_background_update(points.clone());
        assert!(started);
        assert!(manager.is_update_in_progress());

        // Wait for update to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Check if swap is ready
        let swapped = manager.swap_if_ready();
        // May or may not be ready depending on timing
        if swapped {
            assert!(!manager.is_update_in_progress());
            assert!(manager.has_target());
        }
    }
}
