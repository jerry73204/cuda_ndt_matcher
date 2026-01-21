//! Initial pose estimation using Monte Carlo sampling with TPE.
//!
//! This module implements the initial pose estimation service that uses
//! Tree-Structured Parzen Estimator (TPE) to efficiently search the 6D
//! pose space and find the best match against the reference map.
//!
//! ## GPU Batch Acceleration
//!
//! The startup phase (first `n_startup_trials` particles) can be batch-processed
//! on GPU when `use_gpu_batch_startup` is enabled. This provides significant
//! speedup since TPE doesn't use trial data during the startup phase anyway.

use crate::ndt_manager::NdtManager;
use crate::params::InitialPoseParams;
use crate::particle::{select_best_particle, Particle};
use crate::tpe::{
    pose_components_to_input, Direction, Input, TreeStructuredParzenEstimator, Trial, ANGLE_X,
    ANGLE_Y, ANGLE_Z, TRANS_X, TRANS_Y, TRANS_Z,
};
use geometry_msgs::msg::{Point, Pose, PoseWithCovariance, PoseWithCovarianceStamped, Quaternion};
use nalgebra::{Quaternion as NaQuaternion, UnitQuaternion};
use rclrs::log_debug;
#[cfg(feature = "debug-output")]
use serde::Serialize;
#[cfg(feature = "debug-output")]
use std::fs::OpenOptions;
#[cfg(feature = "debug-output")]
use std::io::Write;
#[cfg(feature = "debug-output")]
use std::time::Instant;

const LOGGER_NAME: &str = "ndt_scan_matcher.initial_pose";

/// Debug output for pose initialization (only with debug-output feature)
#[cfg(feature = "debug-output")]
#[derive(Debug, Clone, Serialize)]
pub struct InitPoseDebug {
    /// Entry type discriminator for JSONL parsing
    #[serde(rename = "type")]
    pub entry_type: &'static str,
    /// Total pose initialization time in milliseconds
    pub total_time_ms: f64,
    /// Time for random startup phase (first n_startup_trials)
    pub startup_time_ms: f64,
    /// Time for TPE-guided phase
    pub guided_time_ms: f64,
    /// Total particles evaluated
    pub num_particles: usize,
    /// Particles in startup phase
    pub num_startup: usize,
    /// Best score progression (running max as particles are evaluated)
    pub best_score_trajectory: Vec<f64>,
    /// Per-particle alignment time in milliseconds
    pub per_particle_time_ms: Vec<f64>,
    /// Final best particle score (NVTL)
    pub final_score: f64,
    /// Final best particle iteration count
    pub final_iterations: i32,
    /// Whether result is reliable (score >= threshold)
    pub reliable: bool,
}

#[cfg(feature = "debug-output")]
impl InitPoseDebug {
    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Write init debug entry to the debug file (only with debug-output feature)
#[cfg(feature = "debug-output")]
fn write_init_debug(debug: &InitPoseDebug) {
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
}

/// Result of initial pose estimation
#[derive(Debug, Clone)]
pub struct InitialPoseResult {
    /// The estimated pose with covariance
    pub pose_with_covariance: PoseWithCovarianceStamped,
    /// The alignment score
    pub score: f64,
    /// Whether the result is reliable (score above threshold)
    pub reliable: bool,
    /// All evaluated particles (for debugging/visualization)
    pub particles: Vec<Particle>,
}

/// Estimate initial pose using Monte Carlo sampling with TPE
///
/// # Arguments
/// * `initial_pose_with_cov` - Initial guess with covariance
/// * `ndt_manager` - NDT alignment manager
/// * `source_points` - Source point cloud (sensor data)
/// * `target_points` - Target point cloud (map data)
/// * `params` - Initial pose estimation parameters
/// * `resolution` - NDT voxel resolution (for NVTL scoring)
/// * `score_threshold` - Minimum score for reliable result
///
/// # Returns
/// Initial pose estimation result with best pose and all particles
pub fn estimate_initial_pose(
    initial_pose_with_cov: &PoseWithCovarianceStamped,
    ndt_manager: &mut NdtManager,
    source_points: &[[f32; 3]],
    target_points: &[[f32; 3]],
    params: &InitialPoseParams,
    _resolution: f64,
    _score_threshold: f64,
) -> Result<InitialPoseResult, String> {
    // Extract covariance to determine sampling distributions
    let covariance = &initial_pose_with_cov.pose.covariance;

    // Extract standard deviations from diagonal of covariance matrix
    let stddev_x = covariance[0].sqrt().max(0.1); // [0,0]
    let stddev_y = covariance[7].sqrt().max(0.1); // [1,1]
    let stddev_z = covariance[14].sqrt().max(0.1); // [2,2]
    let stddev_roll = covariance[21].sqrt().max(0.01); // [3,3]
    let stddev_pitch = covariance[28].sqrt().max(0.01); // [4,4]
    let stddev_yaw = covariance[35].sqrt().max(0.01); // [5,5]

    // Extract initial pose components
    let initial_pose = &initial_pose_with_cov.pose.pose;
    let (roll, pitch, yaw) = quaternion_to_rpy(&initial_pose.orientation);

    // Log the input pose for debugging
    log_debug!(
        LOGGER_NAME,
        "Input: pos=({:.1}, {:.1}, {:.1}), yaw={:.1}°, stddev_yaw={:.1}°",
        initial_pose.position.x,
        initial_pose.position.y,
        initial_pose.position.z,
        yaw.to_degrees(),
        stddev_yaw.to_degrees()
    );

    // Use GPU NVTL scoring via fast_gicp
    let outlier_ratio = 0.55; // Autoware default

    // Create mean and stddev for TPE
    let mean = pose_components_to_input(
        initial_pose.position.x,
        initial_pose.position.y,
        initial_pose.position.z,
        roll,
        pitch,
        yaw,
    );

    let stddev: Input = [
        stddev_x,
        stddev_y,
        stddev_z,
        stddev_roll,
        stddev_pitch,
        stddev_yaw, // Use yaw stddev from covariance instead of uniform sampling
    ];

    // Initialize TPE
    // Note: We convert fitness_score to likelihood-like score using exp(-fitness_score)
    // This makes it "higher = better" like Autoware's transform_probability
    let mut tpe = TreeStructuredParzenEstimator::new(
        Direction::Maximize,
        params.n_startup_trials as i64,
        mean,
        stddev,
    );

    // Evaluate particles
    let mut particles = Vec::with_capacity(params.particles_num as usize);

    // Debug tracking (only with debug-output feature)
    #[cfg(feature = "debug-output")]
    let total_start = Instant::now();
    #[cfg(feature = "debug-output")]
    let mut per_particle_times: Vec<f64> = Vec::new();
    #[cfg(feature = "debug-output")]
    let mut best_score_trajectory: Vec<f64> = Vec::new();
    #[cfg(feature = "debug-output")]
    let mut running_best_score = f64::NEG_INFINITY;

    // ========================================================================
    // Startup Phase: Batch evaluate first n_startup_trials particles
    // ========================================================================
    // During startup, TPE samples randomly, so we can batch-evaluate all
    // startup particles at once using GPU acceleration.
    let startup_count = (params.n_startup_trials as usize).min(params.particles_num as usize);
    #[cfg(feature = "debug-output")]
    let startup_start = Instant::now();

    if startup_count > 0 {
        log_debug!(
            LOGGER_NAME,
            "Batch evaluating {startup_count} startup particles"
        );

        // Sample all startup candidates from TPE at once
        let startup_inputs: Vec<Input> = (0..startup_count).map(|_| tpe.get_next_input()).collect();
        let startup_poses: Vec<Pose> = startup_inputs.iter().map(input_to_pose).collect();

        // Batch align using GPU (or CPU fallback)
        let batch_results = match ndt_manager.align_batch(source_points, &startup_poses) {
            Ok(results) => results,
            Err(e) => {
                log_debug!(
                    LOGGER_NAME,
                    "Batch alignment failed ({e}), falling back to sequential"
                );
                vec![] // Will fall back to sequential below
            }
        };

        // Process batch results
        // Note: batch alignment time is amortized across all particles (for debug tracking)
        #[cfg(feature = "debug-output")]
        let per_particle_batch_time = {
            let batch_time_ms = startup_start.elapsed().as_secs_f64() * 1000.0;
            batch_time_ms / batch_results.len().max(1) as f64
        };

        for (i, align_result) in batch_results.into_iter().enumerate() {
            let candidate_pose = &startup_poses[i];

            // Compute fitness score for TPE guidance (higher = better)
            let fitness_score = align_result.score;
            let transform_probability = (-fitness_score / 10.0).exp();

            // Also compute NVTL for final particle selection
            let nvtl_score = ndt_manager
                .evaluate_nvtl(
                    source_points,
                    target_points,
                    &align_result.pose,
                    outlier_ratio,
                )
                .unwrap_or(0.0);

            // For final particle selection, prefer NVTL when available
            let selection_score = if nvtl_score > 0.0 {
                nvtl_score
            } else {
                transform_probability * 5.0
            };

            // Create particle with results
            let particle = Particle::new(
                candidate_pose.clone(),
                align_result.pose.clone(),
                selection_score,
                align_result.iterations,
            );
            particles.push(particle);

            // Update TPE with result
            let result_input = pose_to_input(&align_result.pose);
            tpe.add_trial(Trial {
                input: result_input,
                score: transform_probability,
            });

            // Debug tracking (only with debug-output feature)
            #[cfg(feature = "debug-output")]
            {
                per_particle_times.push(per_particle_batch_time);
                if selection_score > running_best_score {
                    running_best_score = selection_score;
                }
                best_score_trajectory.push(running_best_score);
            }
        }
    }
    #[cfg(feature = "debug-output")]
    let startup_time_ms = startup_start.elapsed().as_secs_f64() * 1000.0;

    // ========================================================================
    // Guided Phase: Sequential evaluation for remaining particles
    // ========================================================================
    let remaining = params.particles_num as usize - particles.len();
    #[cfg(feature = "debug-output")]
    let guided_start = Instant::now();
    if remaining > 0 {
        log_debug!(
            LOGGER_NAME,
            "Sequentially evaluating {remaining} guided particles"
        );
    }

    for _ in 0..remaining {
        #[cfg(feature = "debug-output")]
        let particle_start = Instant::now();

        // Get next candidate pose from TPE
        let input = tpe.get_next_input();

        // Convert to geometry_msgs::Pose
        let candidate_pose = input_to_pose(&input);

        // Perform NDT alignment from this candidate pose
        let align_result = match ndt_manager.align(source_points, target_points, &candidate_pose) {
            Ok(result) => result,
            Err(e) => {
                // Skip failed alignments
                log_debug!(LOGGER_NAME, "NDT alignment failed: {e}");
                continue;
            }
        };

        // Compute fitness score for TPE guidance (higher = better)
        // Autoware uses transform_probability which is exp(-fitness_score)
        // This provides gradient signal even when NVTL returns 0 (scan in empty area)
        let fitness_score = align_result.score;
        // Convert to "higher = better" like Autoware's transform_probability
        // Use exp(-score/scale) to map to (0, 1] range - higher is better
        let transform_probability = (-fitness_score / 10.0).exp();

        // Also compute NVTL for final particle selection (not TPE guidance)
        let nvtl_score = ndt_manager
            .evaluate_nvtl(
                source_points,
                target_points,
                &align_result.pose,
                outlier_ratio,
            )
            .unwrap_or(0.0);

        // Get yaw values for debugging
        let candidate_yaw = quaternion_to_rpy(&candidate_pose.orientation).2;
        let result_yaw = quaternion_to_rpy(&align_result.pose.orientation).2;

        // For final particle selection, prefer NVTL when available (non-zero)
        // Fall back to transform_probability when NVTL is 0 (scan in empty area)
        let selection_score = if nvtl_score > 0.0 {
            nvtl_score
        } else {
            // Scale transform_probability to be comparable with NVTL range
            // NVTL is typically 0-5, transform_probability is 0-1
            transform_probability * 5.0
        };

        // Suppress unused variable warnings
        let _ = (
            candidate_yaw,
            result_yaw,
            fitness_score,
            transform_probability,
            nvtl_score,
        );

        // Create particle with results (using selection_score for final selection)
        let particle = Particle::new(
            candidate_pose,
            align_result.pose.clone(),
            selection_score,
            align_result.iterations,
        );
        particles.push(particle.clone());

        // Update TPE with result for next iteration's guidance
        // TPE uses transform_probability (like Autoware) to explore the space (higher = better)
        // This provides gradient signal even when scan lands in empty areas
        let result_input = pose_to_input(&align_result.pose);
        tpe.add_trial(Trial {
            input: result_input,
            score: transform_probability,
        });

        // Debug tracking (only with debug-output feature)
        #[cfg(feature = "debug-output")]
        {
            let particle_time_ms = particle_start.elapsed().as_secs_f64() * 1000.0;
            per_particle_times.push(particle_time_ms);
            if selection_score > running_best_score {
                running_best_score = selection_score;
            }
            best_score_trajectory.push(running_best_score);
        }
    }
    #[cfg(feature = "debug-output")]
    let guided_time_ms = guided_start.elapsed().as_secs_f64() * 1000.0;

    // Select best particle (highest score)
    let best_particle = select_best_particle(&particles)
        .ok_or_else(|| "No particles evaluated successfully".to_string())?;

    // Log selected best particle
    let (_, _, best_yaw) = quaternion_to_rpy(&best_particle.result_pose.orientation);
    log_debug!(
        LOGGER_NAME,
        "Best: pos=({:.1}, {:.1}, {:.1}), yaw={:.1}°, score={:.2}",
        best_particle.result_pose.position.x,
        best_particle.result_pose.position.y,
        best_particle.result_pose.position.z,
        best_yaw.to_degrees(),
        best_particle.score
    );

    // Build result
    // NVTL score is "higher = better" (Autoware threshold is around 2.3)
    let nvtl_threshold = 2.3;
    let reliable = best_particle.score >= nvtl_threshold;

    // Write debug output (only with debug-output feature)
    #[cfg(feature = "debug-output")]
    {
        let total_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let debug = InitPoseDebug {
            entry_type: "init",
            total_time_ms,
            startup_time_ms,
            guided_time_ms,
            num_particles: particles.len(),
            num_startup: startup_count,
            best_score_trajectory,
            per_particle_time_ms: per_particle_times,
            final_score: best_particle.score,
            final_iterations: best_particle.iterations,
            reliable,
        };
        write_init_debug(&debug);
    }

    let result = InitialPoseResult {
        pose_with_covariance: PoseWithCovarianceStamped {
            header: initial_pose_with_cov.header.clone(),
            pose: PoseWithCovariance {
                pose: best_particle.result_pose.clone(),
                // Use input covariance for now (could be refined based on particle spread)
                covariance: initial_pose_with_cov.pose.covariance,
            },
        },
        score: best_particle.score,
        reliable,
        particles,
    };

    Ok(result)
}

/// Convert TPE input to geometry_msgs::Pose
fn input_to_pose(input: &Input) -> Pose {
    let quaternion = rpy_to_quaternion(input[ANGLE_X], input[ANGLE_Y], input[ANGLE_Z]);

    Pose {
        position: Point {
            x: input[TRANS_X],
            y: input[TRANS_Y],
            z: input[TRANS_Z],
        },
        orientation: quaternion,
    }
}

/// Convert geometry_msgs::Pose to TPE input
fn pose_to_input(pose: &Pose) -> Input {
    let (roll, pitch, yaw) = quaternion_to_rpy(&pose.orientation);

    pose_components_to_input(
        pose.position.x,
        pose.position.y,
        pose.position.z,
        roll,
        pitch,
        yaw,
    )
}

/// Convert quaternion to roll-pitch-yaw angles
fn quaternion_to_rpy(q: &Quaternion) -> (f64, f64, f64) {
    let unit_q = UnitQuaternion::new_normalize(NaQuaternion::new(q.w, q.x, q.y, q.z));
    let euler = unit_q.euler_angles();
    (euler.0, euler.1, euler.2)
}

/// Convert roll-pitch-yaw to quaternion
fn rpy_to_quaternion(roll: f64, pitch: f64, yaw: f64) -> Quaternion {
    let unit_q = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
    let q = unit_q.quaternion();

    Quaternion {
        x: q.i,
        y: q.j,
        z: q.k,
        w: q.w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tpe::INPUT_DIM;

    #[test]
    fn test_quaternion_rpy_roundtrip() {
        let roll = 0.1;
        let pitch = 0.2;
        let yaw = 0.3;

        let q = rpy_to_quaternion(roll, pitch, yaw);
        let (r2, p2, y2) = quaternion_to_rpy(&q);

        assert!((roll - r2).abs() < 1e-10);
        assert!((pitch - p2).abs() < 1e-10);
        assert!((yaw - y2).abs() < 1e-10);
    }

    #[test]
    fn test_input_pose_roundtrip() {
        let input: Input = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let pose = input_to_pose(&input);
        let input2 = pose_to_input(&pose);

        for i in 0..INPUT_DIM {
            assert!((input[i] - input2[i]).abs() < 1e-10);
        }
    }
}
