//! Particle representation for Monte Carlo initial pose estimation.

// Allow dead_code: Particle struct is used in initial_pose.rs for Monte Carlo
// sampling. Rust doesn't track usage across module boundaries with generics.
#![allow(dead_code)]

use geometry_msgs::msg::Pose;

/// A particle represents a candidate pose hypothesis for initial pose estimation.
#[derive(Debug, Clone)]
pub struct Particle {
    /// The initial pose hypothesis (before NDT alignment)
    pub initial_pose: Pose,
    /// The result pose after NDT alignment converged
    pub result_pose: Pose,
    /// NDT alignment score (higher is better)
    pub score: f64,
    /// Number of NDT iterations needed for convergence
    pub iterations: i32,
}

impl Particle {
    /// Create a new particle with alignment results
    pub fn new(initial_pose: Pose, result_pose: Pose, score: f64, iterations: i32) -> Self {
        Self {
            initial_pose,
            result_pose,
            score,
            iterations,
        }
    }
}

/// Find the best particle (highest likelihood score) from a collection
/// Note: likelihood score is "higher = better" (converted from fitness_score)
pub fn select_best_particle(particles: &[Particle]) -> Option<&Particle> {
    particles.iter().max_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry_msgs::msg::{Point, Quaternion};

    fn make_pose(x: f64, y: f64, z: f64) -> Pose {
        Pose {
            position: Point { x, y, z },
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        }
    }

    #[test]
    fn test_select_best_particle() {
        // likelihood score is "higher = better"
        let particles = vec![
            Particle::new(make_pose(0.0, 0.0, 0.0), make_pose(0.1, 0.0, 0.0), 0.5, 10),
            Particle::new(make_pose(1.0, 0.0, 0.0), make_pose(1.1, 0.0, 0.0), 0.9, 8), // best (highest)
            Particle::new(make_pose(2.0, 0.0, 0.0), make_pose(2.1, 0.0, 0.0), 0.3, 12),
        ];

        let best = select_best_particle(&particles).unwrap();
        assert!((best.score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_select_best_empty() {
        let particles: Vec<Particle> = vec![];
        assert!(select_best_particle(&particles).is_none());
    }
}
