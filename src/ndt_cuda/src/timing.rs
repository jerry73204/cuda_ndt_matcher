//! Detailed timing instrumentation for profiling.
//!
//! This module provides timing collection when the `profiling` feature is enabled.
//! Without the feature, all timing macros compile to no-ops.

use serde::{Deserialize, Serialize};
use std::time::Duration;
#[cfg(feature = "profiling")]
use std::time::Instant;

/// Timing breakdown for a single NDT alignment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlignmentTiming {
    /// Total alignment time
    pub total_ms: f64,
    /// Time to transform source points
    pub transform_source_ms: f64,
    /// Time to find correspondences (voxel search)
    pub correspondence_ms: f64,
    /// Time to compute derivatives (Jacobian/Hessian)
    pub derivatives_ms: f64,
    /// Time to solve linear system
    pub solver_ms: f64,
    /// Time to update pose
    pub pose_update_ms: f64,
    /// Per-iteration timing breakdown
    pub iterations: Vec<IterationTiming>,
}

/// Timing for a single Newton iteration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IterationTiming {
    pub iteration: usize,
    pub total_ms: f64,
    pub transform_ms: f64,
    pub correspondence_ms: f64,
    pub derivatives_ms: f64,
    pub solver_ms: f64,
}

/// Timer that can be enabled/disabled at compile time.
#[cfg(feature = "profiling")]
pub struct Timer {
    start: Instant,
    name: &'static str,
}

#[cfg(feature = "profiling")]
impl Timer {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        Self {
            start: Instant::now(),
            name,
        }
    }

    #[inline]
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// No-op timer when profiling is disabled.
#[cfg(not(feature = "profiling"))]
pub struct Timer;

#[cfg(not(feature = "profiling"))]
impl Timer {
    #[inline(always)]
    pub fn new(_name: &'static str) -> Self {
        Self
    }

    #[inline(always)]
    pub fn elapsed_ms(&self) -> f64 {
        0.0
    }

    #[inline(always)]
    pub fn elapsed(&self) -> Duration {
        Duration::ZERO
    }

    #[inline(always)]
    pub fn name(&self) -> &'static str {
        ""
    }
}

/// Collector for timing data during alignment.
#[cfg(feature = "profiling")]
#[derive(Debug, Default)]
pub struct TimingCollector {
    pub alignment_start: Option<Instant>,
    pub current_iteration: usize,
    pub timing: AlignmentTiming,
    iteration_start: Option<Instant>,
    phase_times: PhaseTimes,
}

#[cfg(feature = "profiling")]
#[derive(Debug, Default)]
struct PhaseTimes {
    transform_ms: f64,
    correspondence_ms: f64,
    derivatives_ms: f64,
    solver_ms: f64,
}

#[cfg(feature = "profiling")]
impl TimingCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start_alignment(&mut self) {
        self.alignment_start = Some(Instant::now());
        self.current_iteration = 0;
        self.timing = AlignmentTiming::default();
    }

    pub fn start_iteration(&mut self) {
        self.iteration_start = Some(Instant::now());
        self.phase_times = PhaseTimes::default();
    }

    pub fn record_phase(&mut self, phase: &str, duration_ms: f64) {
        match phase {
            "transform" => {
                self.phase_times.transform_ms += duration_ms;
                self.timing.transform_source_ms += duration_ms;
            }
            "correspondence" => {
                self.phase_times.correspondence_ms += duration_ms;
                self.timing.correspondence_ms += duration_ms;
            }
            "derivatives" => {
                self.phase_times.derivatives_ms += duration_ms;
                self.timing.derivatives_ms += duration_ms;
            }
            "solver" => {
                self.phase_times.solver_ms += duration_ms;
                self.timing.solver_ms += duration_ms;
            }
            "pose_update" => {
                self.timing.pose_update_ms += duration_ms;
            }
            _ => {}
        }
    }

    pub fn end_iteration(&mut self) {
        let total_ms = self
            .iteration_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        self.timing.iterations.push(IterationTiming {
            iteration: self.current_iteration,
            total_ms,
            transform_ms: self.phase_times.transform_ms,
            correspondence_ms: self.phase_times.correspondence_ms,
            derivatives_ms: self.phase_times.derivatives_ms,
            solver_ms: self.phase_times.solver_ms,
        });

        self.current_iteration += 1;
    }

    pub fn finish(&mut self) -> AlignmentTiming {
        self.timing.total_ms = self
            .alignment_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        std::mem::take(&mut self.timing)
    }
}

/// No-op collector when profiling is disabled.
#[cfg(not(feature = "profiling"))]
#[derive(Debug, Default)]
pub struct TimingCollector;

#[cfg(not(feature = "profiling"))]
impl TimingCollector {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub fn start_alignment(&mut self) {}

    #[inline(always)]
    pub fn start_iteration(&mut self) {}

    #[inline(always)]
    pub fn record_phase(&mut self, _phase: &str, _duration_ms: f64) {}

    #[inline(always)]
    pub fn end_iteration(&mut self) {}

    #[inline(always)]
    pub fn finish(&mut self) -> AlignmentTiming {
        AlignmentTiming::default()
    }
}

/// Convenience macro for timing a block of code.
///
/// Usage:
/// ```ignore
/// let result = time_phase!(collector, "derivatives", {
///     compute_derivatives()
/// });
/// ```
#[macro_export]
macro_rules! time_phase {
    ($collector:expr, $phase:expr, $block:expr) => {{
        #[cfg(feature = "profiling")]
        let _timer = $crate::timing::Timer::new($phase);

        let result = $block;

        #[cfg(feature = "profiling")]
        $collector.record_phase($phase, _timer.elapsed_ms());

        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_collector() {
        let mut collector = TimingCollector::new();

        collector.start_alignment();
        collector.start_iteration();
        collector.record_phase("transform", 1.0);
        collector.record_phase("correspondence", 2.0);
        collector.record_phase("derivatives", 3.0);
        collector.record_phase("solver", 0.5);
        collector.end_iteration();

        let timing = collector.finish();

        #[cfg(feature = "profiling")]
        {
            assert_eq!(timing.iterations.len(), 1);
            assert_eq!(timing.transform_source_ms, 1.0);
            assert_eq!(timing.correspondence_ms, 2.0);
            assert_eq!(timing.derivatives_ms, 3.0);
            assert_eq!(timing.solver_ms, 0.5);
        }

        #[cfg(not(feature = "profiling"))]
        {
            // No-op, timing is zero
            assert_eq!(timing.total_ms, 0.0);
        }
    }
}
