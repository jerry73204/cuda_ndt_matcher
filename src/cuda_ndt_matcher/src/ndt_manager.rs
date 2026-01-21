//! NDT alignment manager using our CubeCL implementation.

// Allow dead_code: NdtManager methods are called from main.rs callbacks.
// Rust doesn't track usage through Arc<Mutex<T>> and closure captures.
#![allow(dead_code)]

use crate::params::NdtParams;
use anyhow::{bail, Result};
use geometry_msgs::msg::{Point, Pose, Quaternion};
use nalgebra::{Isometry3, Matrix6, Quaternion as NaQuaternion, Translation3, UnitQuaternion};
#[cfg(feature = "debug-output")]
pub use ndt_cuda::AlignmentDebug;
use ndt_cuda::NdtScanMatcher;
#[cfg(feature = "debug-voxels")]
use ndt_cuda::VoxelGrid;
use rclrs::{log_debug, log_info, log_warn};
#[cfg(feature = "debug-voxels")]
use std::io::Write;

const LOGGER_NAME: &str = "ndt_scan_matcher.ndt_manager";

/// Extract yaw from quaternion for debug logging
fn quaternion_to_yaw(q: &Quaternion) -> f64 {
    let unit_q = UnitQuaternion::new_normalize(NaQuaternion::new(q.w, q.x, q.y, q.z));
    unit_q.euler_angles().2
}

/// Dump voxel data to JSON file for comparison with Autoware.
/// Only available with `debug-voxels` feature.
/// Output file: NDT_DUMP_VOXELS_FILE env var or /tmp/ndt_cuda_voxels.json
#[cfg(feature = "debug-voxels")]
fn dump_voxel_data(grid: &VoxelGrid) -> Result<()> {
    let output_path = std::env::var("NDT_DUMP_VOXELS_FILE")
        .unwrap_or_else(|_| "/tmp/ndt_cuda_voxels.json".to_string());

    let voxels = grid.voxels();
    log_info!(
        LOGGER_NAME,
        "Dumping {} voxels to {}",
        voxels.len(),
        output_path
    );

    let mut file = std::fs::File::create(&output_path)?;

    // Write JSON header
    writeln!(file, "{{")?;
    writeln!(file, "  \"resolution\": {},", grid.resolution())?;
    writeln!(file, "  \"num_voxels\": {},", voxels.len())?;
    writeln!(file, "  \"voxels\": [")?;

    for (i, voxel) in voxels.iter().enumerate() {
        let mean = &voxel.mean;
        let cov = &voxel.covariance;
        let inv_cov = &voxel.inv_covariance;

        // Format covariance as row-major 3x3 matrix
        let cov_str = format!(
            "[[{:.8},{:.8},{:.8}],[{:.8},{:.8},{:.8}],[{:.8},{:.8},{:.8}]]",
            cov[(0, 0)],
            cov[(0, 1)],
            cov[(0, 2)],
            cov[(1, 0)],
            cov[(1, 1)],
            cov[(1, 2)],
            cov[(2, 0)],
            cov[(2, 1)],
            cov[(2, 2)]
        );

        // Format inv_covariance as row-major 3x3 matrix
        let inv_cov_str = format!(
            "[[{:.8},{:.8},{:.8}],[{:.8},{:.8},{:.8}],[{:.8},{:.8},{:.8}]]",
            inv_cov[(0, 0)],
            inv_cov[(0, 1)],
            inv_cov[(0, 2)],
            inv_cov[(1, 0)],
            inv_cov[(1, 1)],
            inv_cov[(1, 2)],
            inv_cov[(2, 0)],
            inv_cov[(2, 1)],
            inv_cov[(2, 2)]
        );

        let comma = if i < voxels.len() - 1 { "," } else { "" };
        writeln!(
            file,
            "    {{\"mean\": [{:.6},{:.6},{:.6}], \"cov\": {}, \"inv_cov\": {}, \"point_count\": {}}}{}",
            mean[0], mean[1], mean[2], cov_str, inv_cov_str, voxel.point_count, comma
        )?;
    }

    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;

    log_info!(LOGGER_NAME, "Voxel dump complete: {}", output_path);
    Ok(())
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
        // Check NDT_USE_GPU environment variable (default: true for GPU acceleration)
        let use_gpu = std::env::var("NDT_USE_GPU")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let matcher = NdtScanMatcher::builder()
            .resolution(params.ndt.resolution as f32)
            .max_iterations(params.ndt.max_iterations as usize)
            .transformation_epsilon(params.ndt.trans_epsilon)
            .step_size(params.ndt.step_size) // From config (default 0.1)
            .outlier_ratio(0.55) // Autoware default
            .regularization_enabled(params.regularization.enabled)
            .regularization_scale_factor(params.regularization.scale_factor)
            .use_line_search(params.ndt.use_line_search)
            .use_gpu(use_gpu)
            .build()?;

        log_debug!(LOGGER_NAME, "NDT manager created with use_gpu={use_gpu}");

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
        log_info!(LOGGER_NAME, "Setting target with {} points", points.len());
        let result = self.matcher.set_target(points);
        if let Some(grid) = self.matcher.target_grid() {
            log_info!(
                LOGGER_NAME,
                "Target grid created: {} voxels (resolution={})",
                grid.len(),
                grid.resolution()
            );
            // Log actual grid bounding box (min/max of all voxel means)
            let voxels = grid.voxels();
            if !voxels.is_empty() {
                let (min_x, max_x) = voxels
                    .iter()
                    .map(|v| v.mean[0])
                    .fold((f32::MAX, f32::MIN), |(min, max), x| {
                        (min.min(x), max.max(x))
                    });
                let (min_y, max_y) = voxels
                    .iter()
                    .map(|v| v.mean[1])
                    .fold((f32::MAX, f32::MIN), |(min, max), y| {
                        (min.min(y), max.max(y))
                    });
                let (min_z, max_z) = voxels
                    .iter()
                    .map(|v| v.mean[2])
                    .fold((f32::MAX, f32::MIN), |(min, max), z| {
                        (min.min(z), max.max(z))
                    });
                log_info!(
                    LOGGER_NAME,
                    "Grid bounds: X=[{:.1},{:.1}] Y=[{:.1},{:.1}] Z=[{:.1},{:.1}]",
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    min_z,
                    max_z
                );
            }

            // Dump voxel data (only with debug-voxels feature)
            #[cfg(feature = "debug-voxels")]
            if let Err(e) = dump_voxel_data(grid) {
                log_warn!(LOGGER_NAME, "Failed to dump voxel data: {e}");
            }
        } else {
            log_warn!(LOGGER_NAME, "Target grid is None after set_target!");
        }
        result
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
    ///
    /// Only available with the `debug-output` feature.
    #[cfg(feature = "debug-output")]
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

        // Log voxel grid state before alignment
        if let Some(grid) = self.matcher.target_grid() {
            // Log source point cloud bounds
            let (min_x, max_x) = source_points
                .iter()
                .map(|p| p[0])
                .fold((f32::MAX, f32::MIN), |(min, max), x| {
                    (min.min(x), max.max(x))
                });
            let (min_y, max_y) = source_points
                .iter()
                .map(|p| p[1])
                .fold((f32::MAX, f32::MIN), |(min, max), y| {
                    (min.min(y), max.max(y))
                });
            log_info!(
                LOGGER_NAME,
                "align_with_debug: {} pts, {} voxels, source X=[{:.1},{:.1}] Y=[{:.1},{:.1}]",
                source_points.len(),
                grid.len(),
                min_x,
                max_x,
                min_y,
                max_y
            );
        } else {
            log_warn!(LOGGER_NAME, "align_with_debug: target grid is None!");
        }

        // Convert initial pose to isometry
        let initial_guess = pose_to_isometry(initial_pose);

        // Run alignment with debug
        let (result, debug) =
            self.matcher
                .align_with_debug(source_points, initial_guess, timestamp_ns)?;

        // Log result
        log_info!(
            LOGGER_NAME,
            "align: score={:.1}, nvtl={:.2}, iters={}, corr={}, converged={}",
            result.score,
            result.nvtl,
            result.iterations,
            result.num_correspondences,
            result.converged
        );

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

    /// Evaluate transform probability at a given pose
    pub fn evaluate_transform_probability(
        &self,
        source_points: &[[f32; 3]],
        pose: &Pose,
    ) -> Result<f64> {
        let isometry = pose_to_isometry(pose);
        self.matcher
            .evaluate_transform_probability(source_points, &isometry)
    }

    /// Evaluate NVTL at multiple poses in parallel (GPU-accelerated via Rayon).
    ///
    /// This is optimized for multi-NDT covariance estimation where we need
    /// to evaluate NVTL at many offset poses quickly.
    pub fn evaluate_nvtl_batch(
        &mut self,
        source_points: &[[f32; 3]],
        poses: &[Pose],
    ) -> Result<Vec<f64>> {
        let isometries: Vec<Isometry3<f64>> = poses.iter().map(pose_to_isometry).collect();
        self.matcher.evaluate_nvtl_batch(source_points, &isometries)
    }

    /// Align from multiple initial poses and return all results.
    ///
    /// This is useful for multi-NDT covariance estimation where we need
    /// to run alignment from multiple offset poses.
    ///
    /// Prefers GPU batch alignment when available (shares voxel data across
    /// all alignments). Falls back to CPU parallel (Rayon) if GPU fails.
    pub fn align_batch(
        &self,
        source_points: &[[f32; 3]],
        initial_poses: &[Pose],
    ) -> Result<Vec<AlignResult>> {
        let isometries: Vec<Isometry3<f64>> = initial_poses.iter().map(pose_to_isometry).collect();

        // Try GPU batch alignment first (shares voxel data across alignments)
        let results = match self.matcher.align_batch_gpu(source_points, &isometries) {
            Ok(r) => r,
            Err(e) => {
                // GPU failed, fall back to CPU parallel path
                log_debug!(
                    LOGGER_NAME,
                    "GPU batch alignment failed ({}), using CPU fallback",
                    e
                );
                self.matcher.align_batch(source_points, &isometries)?
            }
        };

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

    /// Compute per-point scores for visualization.
    ///
    /// Returns transformed points in map frame and their max NDT scores.
    /// This is used to create colored point cloud visualizations showing
    /// alignment quality at each point.
    pub fn compute_per_point_scores_for_visualization(
        &mut self,
        source_points: &[[f32; 3]],
        pose: &Pose,
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>)> {
        let isometry = pose_to_isometry(pose);
        self.matcher
            .compute_per_point_scores_for_visualization(source_points, &isometry)
    }

    /// Align multiple scans in parallel using GPU batch processing.
    ///
    /// Each scan is a tuple of (source_points, initial_pose). All scans share
    /// the same target voxel grid. This enables processing multiple lidar scans
    /// concurrently to increase GPU utilization and throughput.
    ///
    /// # Arguments
    /// * `scans` - Slice of (source_points, initial_pose) tuples
    ///
    /// # Returns
    /// * Vector of alignment results, one per scan
    pub fn align_batch_scans(
        &self,
        scans: &[(&[[f32; 3]], Isometry3<f64>)],
    ) -> Result<Vec<ndt_cuda::AlignResult>> {
        log_debug!(
            LOGGER_NAME,
            "Batch aligning {} scans in parallel",
            scans.len()
        );

        self.matcher.align_parallel_scans(scans)
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
