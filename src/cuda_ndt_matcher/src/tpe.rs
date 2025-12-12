//! Tree-Structured Parzen Estimator (TPE) for initial pose estimation.
//!
//! TPE is a sequential model-based optimization algorithm that uses kernel density
//! estimation to model the distribution of good and bad samples, guiding the search
//! toward promising regions of the parameter space.

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

/// Input dimension indices
pub const TRANS_X: usize = 0;
pub const TRANS_Y: usize = 1;
pub const TRANS_Z: usize = 2;
pub const ANGLE_X: usize = 3; // roll
pub const ANGLE_Y: usize = 4; // pitch
pub const ANGLE_Z: usize = 5; // yaw

/// Number of input dimensions (6D pose)
pub const INPUT_DIM: usize = 6;

/// A 6D input vector for pose optimization
pub type Input = [f64; INPUT_DIM];

/// A trial stores an input and its associated score
#[derive(Debug, Clone)]
pub struct Trial {
    pub input: Input,
    pub score: f64,
}

/// Optimization direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Maximize,
    #[allow(dead_code)]
    Minimize,
}

/// Tree-Structured Parzen Estimator for pose optimization
pub struct TreeStructuredParzenEstimator {
    direction: Direction,
    n_startup_trials: i64,
    mean: Input,
    stddev: Input,
    base_stddev: Input,
    trials: Vec<Trial>,
    rng: ThreadRng,
}

impl TreeStructuredParzenEstimator {
    /// Create a new TPE instance
    ///
    /// # Arguments
    /// * `direction` - Whether to maximize or minimize the score
    /// * `n_startup_trials` - Number of random trials before TPE-guided search
    /// * `mean` - Initial mean for sampling distributions
    /// * `stddev` - Initial standard deviation for sampling distributions
    pub fn new(direction: Direction, n_startup_trials: i64, mean: Input, stddev: Input) -> Self {
        // Base standard deviations for stable NDT convergence
        let base_stddev = [
            0.25,                 // TRANS_X: 0.25m
            0.25,                 // TRANS_Y: 0.25m
            0.25,                 // TRANS_Z: 0.25m
            1.0_f64.to_radians(), // ANGLE_X: 1 degree
            1.0_f64.to_radians(), // ANGLE_Y: 1 degree
            2.5_f64.to_radians(), // ANGLE_Z: 2.5 degrees
        ];

        Self {
            direction,
            n_startup_trials,
            mean,
            stddev,
            base_stddev,
            trials: Vec::new(),
            rng: thread_rng(),
        }
    }

    /// Add a trial result
    pub fn add_trial(&mut self, trial: Trial) {
        self.trials.push(trial);
        // Sort trials by score (best first based on direction)
        match self.direction {
            Direction::Maximize => {
                self.trials.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            Direction::Minimize => {
                self.trials.sort_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }

    /// Get the next input to evaluate
    pub fn get_next_input(&mut self) -> Input {
        let above_num = self.compute_above_num();

        // Phase 1: Random sampling during startup
        if (self.trials.len() as i64) < self.n_startup_trials || above_num == 0 {
            return self.generate_random_input();
        }

        // Phase 2: TPE-guided sampling using KDE
        self.generate_tpe_input(above_num)
    }

    /// Generate a random input from the prior distributions
    fn generate_random_input(&mut self) -> Input {
        let mut input = [0.0; INPUT_DIM];

        // Normal distributions for translation and roll/pitch
        for (inp, (&mean, &stddev)) in input[..ANGLE_Z].iter_mut().zip(
            self.mean[..ANGLE_Z]
                .iter()
                .zip(self.stddev[..ANGLE_Z].iter()),
        ) {
            let dist = Normal::new(mean, stddev).unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
            *inp = dist.sample(&mut self.rng);
        }

        // Uniform distribution for yaw [-PI, PI]
        let yaw_dist = Uniform::new(-PI, PI);
        input[ANGLE_Z] = yaw_dist.sample(&mut self.rng);

        input
    }

    /// Generate TPE-guided input using kernel density estimation
    fn generate_tpe_input(&mut self, above_num: usize) -> Input {
        const N_EI_CANDIDATES: usize = 100;

        let mut best_input = self.generate_random_input();
        let mut best_ratio = f64::NEG_INFINITY;

        // Generate candidates and select the one with best log-likelihood ratio
        for _ in 0..N_EI_CANDIDATES {
            let input = self.generate_random_input();
            let ratio = self.compute_log_likelihood_ratio(&input, above_num);
            if ratio > best_ratio {
                best_ratio = ratio;
                best_input = input;
            }
        }

        best_input
    }

    /// Compute the number of "good" samples for KDE
    fn compute_above_num(&self) -> usize {
        let n = self.trials.len();
        if n == 0 {
            return 0;
        }
        // Use clamp(1, 10) on 10% of total as the "good" set size
        (n / 10).clamp(1, 10)
    }

    /// Compute log-likelihood ratio for TPE scoring
    fn compute_log_likelihood_ratio(&self, input: &Input, above_num: usize) -> f64 {
        let n = self.trials.len();
        if n == 0 || above_num == 0 || above_num >= n {
            return 0.0;
        }

        let mut above_logs: Vec<f64> = Vec::with_capacity(above_num);
        let mut below_logs: Vec<f64> = Vec::with_capacity(n - above_num);

        for (i, trial) in self.trials.iter().enumerate() {
            let log_p = self.log_gaussian_pdf(input, &trial.input);

            if i < above_num {
                // Weight by inverse of "good" set size
                let w = 1.0 / above_num as f64;
                above_logs.push(log_p + w.ln());
            } else {
                // Weight by inverse of "bad" set size
                let w = 1.0 / (n - above_num) as f64;
                below_logs.push(log_p + w.ln());
            }
        }

        let above = log_sum_exp(&above_logs);
        let below = log_sum_exp(&below_logs);

        // Penalize candidates near bad samples by factor of 5
        above - below * 5.0
    }

    /// Compute log of Gaussian PDF
    fn log_gaussian_pdf(&self, input: &Input, mu: &Input) -> f64 {
        let mut result = 0.0;

        // Only use X, Y, and yaw for scoring (empirically better)
        for i in [TRANS_X, TRANS_Y, ANGLE_Z] {
            let mut diff = input[i] - mu[i];

            // Handle circular variable for yaw
            if i == ANGLE_Z {
                while diff >= PI {
                    diff -= 2.0 * PI;
                }
                while diff < -PI {
                    diff += 2.0 * PI;
                }
            }

            result += log_gaussian_pdf_1d(diff, self.base_stddev[i]);
        }

        result
    }
}

/// Log of 1D Gaussian PDF (unnormalized)
fn log_gaussian_pdf_1d(diff: f64, sigma: f64) -> f64 {
    -0.5 * (diff / sigma).powi(2) - sigma.ln()
}

/// Numerically stable log-sum-exp
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }

    let sum: f64 = values.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Create input from pose components
pub fn pose_components_to_input(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> Input {
    [x, y, z, roll, pitch, yaw]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpe_startup_phase() {
        let mean = [0.0; INPUT_DIM];
        let stddev = [1.0, 1.0, 0.5, 0.1, 0.1, PI];
        let mut tpe = TreeStructuredParzenEstimator::new(Direction::Maximize, 10, mean, stddev);

        // First 10 trials should be random (startup phase)
        for i in 0..10 {
            let input = tpe.get_next_input();
            assert!(input[ANGLE_Z] >= -PI && input[ANGLE_Z] <= PI);
            tpe.add_trial(Trial {
                input,
                score: i as f64,
            });
        }
    }

    #[test]
    fn test_tpe_guided_phase() {
        let mean = [0.0; INPUT_DIM];
        let stddev = [1.0, 1.0, 0.5, 0.1, 0.1, PI];
        let mut tpe = TreeStructuredParzenEstimator::new(Direction::Maximize, 5, mean, stddev);

        // Add startup trials
        for i in 0..10 {
            let input = tpe.get_next_input();
            tpe.add_trial(Trial {
                input,
                score: i as f64,
            });
        }

        // After startup, should use TPE-guided search
        let _guided_input = tpe.get_next_input();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_log_sum_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);
        // log(e^1 + e^2 + e^3) ≈ 3.407
        assert!((result - 3.407).abs() < 0.01);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        let values: Vec<f64> = vec![];
        let result = log_sum_exp(&values);
        assert!(result.is_infinite() && result < 0.0);
    }
}
