//! Debug types for NDT optimization iteration tracking.
//!
//! This module provides structures to capture the internal state of each
//! optimization iteration for comparison with Autoware's implementation.

use nalgebra::{Matrix6, Vector6};
use serde::Serialize;

/// Timing breakdown for a single iteration (only populated when profiling feature is enabled).
#[derive(Debug, Clone, Default, Serialize)]
pub struct IterationTimingDebug {
    /// Total iteration time in milliseconds.
    pub total_ms: f64,
    /// Time to transform source points.
    pub transform_ms: f64,
    /// Time to find correspondences (voxel search).
    pub correspondence_ms: f64,
    /// Time to compute derivatives (Jacobian/Hessian).
    pub derivatives_ms: f64,
    /// Time to solve linear system (Newton step).
    pub solver_ms: f64,
    /// Time for line search (if used).
    pub line_search_ms: f64,
}

/// Debug information captured at each optimization iteration.
#[derive(Debug, Clone, Serialize)]
pub struct IterationDebug {
    /// Iteration number (0-indexed).
    pub iteration: usize,

    /// Pose at the start of this iteration [tx, ty, tz, roll, pitch, yaw].
    pub pose: Vec<f64>,

    /// NDT score at current pose.
    pub score: f64,

    /// Gradient vector (6 elements).
    pub gradient: Vec<f64>,

    /// Hessian matrix (6x6, stored as flat array row-major).
    pub hessian: Vec<f64>,

    /// Newton step before normalization.
    pub newton_step: Vec<f64>,

    /// Newton step norm.
    pub newton_step_norm: f64,

    /// Normalized step direction.
    pub step_direction: Vec<f64>,

    /// Whether step direction was reversed (not an ascent direction).
    pub direction_reversed: bool,

    /// Directional derivative (gradient · step_direction).
    pub directional_derivative: f64,

    /// Step length from line search (or clamped step).
    pub step_length: f64,

    /// Whether line search was used.
    pub used_line_search: bool,

    /// Whether line search converged (if used).
    pub line_search_converged: bool,

    /// Number of correspondences (points with valid voxel matches).
    pub num_correspondences: usize,

    /// Pose after applying the step.
    pub pose_after: Vec<f64>,

    /// Timing breakdown (populated when profiling feature is enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<IterationTimingDebug>,
}

impl IterationDebug {
    /// Create a new iteration debug with default values.
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            pose: vec![0.0; 6],
            score: 0.0,
            gradient: vec![0.0; 6],
            hessian: vec![0.0; 36],
            newton_step: vec![0.0; 6],
            newton_step_norm: 0.0,
            step_direction: vec![0.0; 6],
            direction_reversed: false,
            directional_derivative: 0.0,
            step_length: 0.0,
            used_line_search: false,
            line_search_converged: false,
            num_correspondences: 0,
            pose_after: vec![0.0; 6],
            timing: None,
        }
    }

    /// Set timing from IterationTimingDebug.
    pub fn set_timing(&mut self, timing: IterationTimingDebug) {
        self.timing = Some(timing);
    }

    /// Set pose from array.
    pub fn set_pose(&mut self, pose: &[f64; 6]) {
        self.pose = pose.to_vec();
    }

    /// Set pose_after from array.
    pub fn set_pose_after(&mut self, pose: &[f64; 6]) {
        self.pose_after = pose.to_vec();
    }

    /// Set gradient from nalgebra Vector6.
    pub fn set_gradient(&mut self, g: &Vector6<f64>) {
        self.gradient = (0..6).map(|i| g[i]).collect();
    }

    /// Set Hessian from nalgebra Matrix6.
    pub fn set_hessian(&mut self, h: &Matrix6<f64>) {
        self.hessian = (0..6)
            .flat_map(|i| (0..6).map(move |j| h[(i, j)]))
            .collect();
    }

    /// Set Newton step from nalgebra Vector6.
    pub fn set_newton_step(&mut self, step: &Vector6<f64>) {
        self.newton_step = (0..6).map(|i| step[i]).collect();
        self.newton_step_norm = step.norm();
    }

    /// Set step direction from nalgebra Vector6.
    pub fn set_step_direction(&mut self, dir: &Vector6<f64>) {
        self.step_direction = (0..6).map(|i| dir[i]).collect();
    }

    /// Format as a compact log line for comparison.
    pub fn to_log_line(&self) -> String {
        format!(
            "iter={} pose=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}] score={:.6} step_len={:.6} corr={} rev={}",
            self.iteration,
            self.pose[0], self.pose[1], self.pose[2],
            self.pose[3], self.pose[4], self.pose[5],
            self.score,
            self.step_length,
            self.num_correspondences,
            self.direction_reversed,
        )
    }
}

/// Timing breakdown for the entire alignment (only populated when profiling feature is enabled).
#[derive(Debug, Clone, Default, Serialize)]
pub struct AlignmentTimingDebug {
    /// Total alignment time in milliseconds.
    pub total_ms: f64,
    /// Time to set up source points.
    pub setup_ms: f64,
    /// Total time in derivative computation.
    pub derivatives_total_ms: f64,
    /// Total time in solver.
    pub solver_total_ms: f64,
    /// Total time in line search.
    pub line_search_total_ms: f64,
    /// Time to compute final scores (NVTL, transform probability).
    pub scoring_ms: f64,
}

/// Complete debug history for one NDT alignment call.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AlignmentDebug {
    /// Timestamp in nanoseconds (from ROS header).
    pub timestamp_ns: u64,

    /// Initial pose guess.
    pub initial_pose: Vec<f64>,

    /// Final pose after optimization.
    pub final_pose: Vec<f64>,

    /// Number of source points.
    pub num_source_points: usize,

    /// Iteration history.
    pub iterations: Vec<IterationDebug>,

    /// Final convergence status.
    pub convergence_status: String,

    /// Total iterations performed.
    pub total_iterations: usize,

    /// Final score.
    pub final_score: f64,

    /// Final NVTL.
    pub final_nvtl: f64,

    /// Maximum consecutive oscillation count detected.
    /// Oscillation indicates the optimizer is bouncing between poses.
    pub oscillation_count: usize,

    /// Timing breakdown (populated when profiling feature is enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<AlignmentTimingDebug>,
}

impl AlignmentDebug {
    /// Create a new alignment debug record.
    pub fn new(timestamp_ns: u64) -> Self {
        Self {
            timestamp_ns,
            initial_pose: vec![0.0; 6],
            final_pose: vec![0.0; 6],
            ..Default::default()
        }
    }

    /// Set initial pose from array.
    pub fn set_initial_pose(&mut self, pose: &[f64; 6]) {
        self.initial_pose = pose.to_vec();
    }

    /// Set final pose from array.
    pub fn set_final_pose(&mut self, pose: &[f64; 6]) {
        self.final_pose = pose.to_vec();
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Convert to pretty JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Compute oscillation count from iteration history.
    ///
    /// This analyzes the pose history to detect direction reversals
    /// that indicate the optimizer is oscillating.
    pub fn compute_oscillation(&mut self) {
        if self.iterations.len() < 3 {
            self.oscillation_count = 0;
            return;
        }

        // Convert iteration poses to array format for oscillation detection
        let poses: Vec<[f64; 6]> = self
            .iterations
            .iter()
            .map(|iter| {
                let p = &iter.pose;
                [
                    p.first().copied().unwrap_or(0.0),
                    p.get(1).copied().unwrap_or(0.0),
                    p.get(2).copied().unwrap_or(0.0),
                    p.get(3).copied().unwrap_or(0.0),
                    p.get(4).copied().unwrap_or(0.0),
                    p.get(5).copied().unwrap_or(0.0),
                ]
            })
            .collect();

        let result = super::oscillation::count_oscillation_from_arrays(
            &poses,
            super::oscillation::DEFAULT_OSCILLATION_THRESHOLD,
        );
        self.oscillation_count = result.max_oscillation_count;
    }

    /// Format as compact multi-line log.
    pub fn to_log(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "=== NDT Alignment ts={} points={} status={} iters={} score={:.6} nvtl={:.6} osc={} ===",
            self.timestamp_ns,
            self.num_source_points,
            self.convergence_status,
            self.total_iterations,
            self.final_score,
            self.final_nvtl,
            self.oscillation_count,
        ));
        lines.push(format!(
            "  initial=[{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}]",
            self.initial_pose[0],
            self.initial_pose[1],
            self.initial_pose[2],
            self.initial_pose[3],
            self.initial_pose[4],
            self.initial_pose[5],
        ));
        for iter in &self.iterations {
            lines.push(format!("  {}", iter.to_log_line()));
        }
        lines.push(format!(
            "  final=[{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}]",
            self.final_pose[0],
            self.final_pose[1],
            self.final_pose[2],
            self.final_pose[3],
            self.final_pose[4],
            self.final_pose[5],
        ));
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iteration_debug_new() {
        let debug = IterationDebug::new(0);
        assert_eq!(debug.iteration, 0);
        assert_eq!(debug.score, 0.0);
    }

    #[test]
    fn test_iteration_debug_set_gradient() {
        let mut debug = IterationDebug::new(0);
        let g = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        debug.set_gradient(&g);
        assert_eq!(debug.gradient, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_alignment_debug_to_json() {
        let debug = AlignmentDebug::new(123456789);
        let json = debug.to_json().unwrap();
        assert!(json.contains("123456789"));
    }

    #[test]
    fn test_iteration_to_log_line() {
        let mut debug = IterationDebug::new(5);
        debug.set_pose(&[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        debug.score = 100.5;
        debug.step_length = 0.05;
        debug.num_correspondences = 1000;
        let line = debug.to_log_line();
        assert!(line.contains("iter=5"));
        assert!(line.contains("score=100.5"));
    }
}
