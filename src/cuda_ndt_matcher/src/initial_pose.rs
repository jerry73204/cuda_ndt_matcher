//! Initial pose estimation using Monte Carlo sampling with TPE.
//!
//! This module implements the initial pose estimation service that uses
//! Tree-Structured Parzen Estimator (TPE) to efficiently search the 6D
//! pose space and find the best match against the reference map.

use crate::ndt_manager::NdtManager;
use crate::nvtl::{NvtlConfig, NvtlVoxelGrid};
use crate::params::InitialPoseParams;
use crate::particle::{select_best_particle, Particle};
use crate::tpe::{
    pose_components_to_input, Direction, Input, TreeStructuredParzenEstimator, Trial, ANGLE_X,
    ANGLE_Y, ANGLE_Z, TRANS_X, TRANS_Y, TRANS_Z,
};
use geometry_msgs::msg::{Point, Pose, PoseWithCovariance, PoseWithCovarianceStamped, Quaternion};
use nalgebra::{Quaternion as NaQuaternion, UnitQuaternion};

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
    resolution: f64,
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
                                                        // Yaw stddev not used - yaw is sampled uniformly

    // Extract initial pose components
    let initial_pose = &initial_pose_with_cov.pose.pose;
    let (roll, pitch, yaw) = quaternion_to_rpy(&initial_pose.orientation);

    // Debug: Print initial pose input
    eprintln!("=== Initial Pose Estimation ===");
    eprintln!(
        "INPUT: pos=({:.2}, {:.2}, {:.2}), quat=({:.4}, {:.4}, {:.4}, {:.4})",
        initial_pose.position.x,
        initial_pose.position.y,
        initial_pose.position.z,
        initial_pose.orientation.x,
        initial_pose.orientation.y,
        initial_pose.orientation.z,
        initial_pose.orientation.w
    );
    eprintln!(
        "INPUT RPY: roll={:.2}°, pitch={:.2}°, yaw={:.2}°",
        roll.to_degrees(),
        pitch.to_degrees(),
        yaw.to_degrees()
    );
    eprintln!(
        "STDDEV: x={:.2}, y={:.2}, z={:.2}, roll={:.2}°, pitch={:.2}°",
        stddev_x,
        stddev_y,
        stddev_z,
        stddev_roll.to_degrees(),
        stddev_pitch.to_degrees()
    );
    eprintln!(
        "PARAMS: particles={}, n_startup={}, yaw_weight_sigma={:.1}°",
        params.particles_num, params.n_startup_trials, params.yaw_weight_sigma
    );

    // Build NVTL voxel grid once for efficient scoring of all particles
    let nvtl_config = NvtlConfig {
        resolution,
        outlier_ratio: 0.55, // Autoware default
    };
    let nvtl_grid = NvtlVoxelGrid::new(target_points, &nvtl_config);
    eprintln!("NVTL voxel grid built with resolution={resolution}");

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
        std::f64::consts::PI, // Yaw uses uniform distribution
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

    for _ in 0..params.particles_num {
        // Get next candidate pose from TPE
        let input = tpe.get_next_input();

        // Convert to geometry_msgs::Pose
        let candidate_pose = input_to_pose(&input);

        // Perform NDT alignment from this candidate pose
        let align_result = match ndt_manager.align(source_points, target_points, &candidate_pose) {
            Ok(result) => result,
            Err(e) => {
                // Skip failed alignments
                eprintln!("NDT alignment failed: {e}");
                continue;
            }
        };

        // Compute NVTL score for the result pose (higher = better)
        // This matches Autoware's nearest_voxel_transformation_likelihood
        let fitness_score = align_result.score; // Keep for debug output
        let nvtl_score = nvtl_grid.compute_score_with_pose(source_points, &align_result.pose);

        // Get yaw values for debugging
        let candidate_yaw = quaternion_to_rpy(&candidate_pose.orientation).2;
        let result_yaw = quaternion_to_rpy(&align_result.pose.orientation).2;

        // Use NVTL score directly (no yaw weighting, like Autoware)
        let weighted_score = nvtl_score;

        // Debug: Print every 20th particle or particles with good scores
        let particle_idx = particles.len();
        if particle_idx % 20 == 0 || nvtl_score > 2.0 {
            eprintln!(
                "P[{:3}]: cand_yaw={:7.2}° -> result_yaw={:7.2}°, fitness={:.4}, nvtl={:.4}",
                particle_idx,
                candidate_yaw.to_degrees(),
                result_yaw.to_degrees(),
                fitness_score,
                nvtl_score
            );
        }

        // Create particle with results (using weighted score for selection)
        let particle = Particle::new(
            candidate_pose,
            align_result.pose.clone(),
            weighted_score,
            align_result.iterations,
        );
        particles.push(particle.clone());

        // Update TPE with result for next iteration's guidance
        // TPE uses the NVTL score to explore the space (higher = better)
        let result_input = pose_to_input(&align_result.pose);
        tpe.add_trial(Trial {
            input: result_input,
            score: nvtl_score,
        });
    }

    // Debug: Print top 5 particles by NVTL score
    let mut sorted_particles = particles.clone();
    sorted_particles.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("=== Top 5 Particles (by NVTL score) ===");
    for (i, p) in sorted_particles.iter().take(5).enumerate() {
        let (_, _, result_yaw) = quaternion_to_rpy(&p.result_pose.orientation);
        eprintln!(
            "Top[{}]: pos=({:.2}, {:.2}, {:.2}), yaw={:.2}°, nvtl={:.4}",
            i,
            p.result_pose.position.x,
            p.result_pose.position.y,
            p.result_pose.position.z,
            result_yaw.to_degrees(),
            p.score
        );
    }

    // Select best particle (highest NVTL score)
    let best_particle = select_best_particle(&particles)
        .ok_or_else(|| "No particles evaluated successfully".to_string())?;

    // Debug: Print selected best particle
    let (best_roll, best_pitch, best_yaw) = quaternion_to_rpy(&best_particle.result_pose.orientation);
    eprintln!("=== Selected Best Particle ===");
    eprintln!(
        "BEST: pos=({:.2}, {:.2}, {:.2}), quat=({:.4}, {:.4}, {:.4}, {:.4})",
        best_particle.result_pose.position.x,
        best_particle.result_pose.position.y,
        best_particle.result_pose.position.z,
        best_particle.result_pose.orientation.x,
        best_particle.result_pose.orientation.y,
        best_particle.result_pose.orientation.z,
        best_particle.result_pose.orientation.w
    );
    eprintln!(
        "BEST RPY: roll={:.2}°, pitch={:.2}°, yaw={:.2}°, nvtl={:.4}",
        best_roll.to_degrees(),
        best_pitch.to_degrees(),
        best_yaw.to_degrees(),
        best_particle.score
    );
    eprintln!(
        "YAW DIFF from input: {:.2}° (input_yaw={:.2}°, result_yaw={:.2}°)",
        (best_yaw - yaw).to_degrees(),
        yaw.to_degrees(),
        best_yaw.to_degrees()
    );
    eprintln!("================================");

    // Build result
    // NVTL score is "higher = better" (Autoware threshold is around 2.3)
    let nvtl_threshold = 2.3;
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
        reliable: best_particle.score >= nvtl_threshold,
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
