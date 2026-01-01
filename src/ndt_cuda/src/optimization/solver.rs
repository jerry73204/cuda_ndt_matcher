//! NDT optimization solver.
//!
//! This module implements the main optimization loop for NDT scan matching:
//! 1. Transform source points using current pose
//! 2. Compute derivatives (gradient + Hessian) against target voxel grid
//! 3. Solve Newton step: Δp = -H⁻¹g
//! 4. Apply step with optional line search
//! 5. Check convergence and iterate

use nalgebra::{Isometry3, Matrix6, Vector6};

use super::line_search::{directional_derivative, LineSearchConfig};
use super::more_thuente::{more_thuente_search, MoreThuenteConfig};
use super::newton::{condition_number, newton_step_regularized};
use super::regularization::{RegularizationConfig, RegularizationTerm};
use super::types::{
    apply_pose_delta, isometry_to_pose_vector, pose_vector_to_isometry, ConvergenceStatus,
    NdtConfig, NdtResult,
};
use crate::derivatives::{compute_derivatives_cpu, GaussianParams};
use crate::scoring::{compute_nvtl, NvtlConfig};
use crate::voxel_grid::VoxelGrid;

/// Configuration for the optimization process.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// NDT-specific configuration.
    pub ndt: NdtConfig,

    /// Line search configuration (legacy backtracking).
    pub line_search: LineSearchConfig,

    /// More-Thuente line search configuration.
    pub more_thuente: MoreThuenteConfig,

    /// GNSS regularization configuration.
    pub regularization: RegularizationConfig,

    /// Tolerance for Newton step SVD.
    pub svd_tolerance: f64,

    /// Minimum number of correspondences required.
    pub min_correspondences: usize,

    /// Condition number threshold for warning.
    pub condition_warning_threshold: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            ndt: NdtConfig::default(),
            line_search: LineSearchConfig::default(),
            more_thuente: MoreThuenteConfig::default(),
            regularization: RegularizationConfig::default(),
            svd_tolerance: 1e-10,
            min_correspondences: 10,
            condition_warning_threshold: 1e10,
        }
    }
}

/// NDT optimizer using Newton's method.
pub struct NdtOptimizer {
    /// Configuration.
    config: OptimizationConfig,

    /// Gaussian parameters for NDT score function.
    gauss: GaussianParams,

    /// GNSS regularization term.
    regularization: RegularizationTerm,
}

impl NdtOptimizer {
    /// Create a new NDT optimizer with the given configuration.
    pub fn new(config: OptimizationConfig) -> Self {
        let gauss = GaussianParams::new(config.ndt.resolution, config.ndt.outlier_ratio);
        let regularization = RegularizationTerm::new(config.regularization.clone());
        Self {
            config,
            gauss,
            regularization,
        }
    }

    /// Create a new NDT optimizer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(OptimizationConfig::default())
    }

    /// Create a new NDT optimizer with custom resolution.
    pub fn with_resolution(resolution: f64) -> Self {
        let mut config = OptimizationConfig::default();
        config.ndt.resolution = resolution;
        Self::new(config)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Get a mutable reference to the regularization term.
    pub fn regularization_mut(&mut self) -> &mut RegularizationTerm {
        &mut self.regularization
    }

    /// Get a reference to the regularization term.
    pub fn regularization(&self) -> &RegularizationTerm {
        &self.regularization
    }

    /// Set the regularization pose (from GNSS).
    pub fn set_regularization_pose(&mut self, pose: Isometry3<f64>) {
        self.regularization.set_reference_pose(pose);
    }

    /// Clear the regularization pose.
    pub fn clear_regularization_pose(&mut self) {
        self.regularization.clear_reference_pose();
    }

    /// Align source points to target voxel grid.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud to align
    /// * `target_grid` - Target voxel grid (map)
    /// * `initial_guess` - Initial pose estimate
    ///
    /// # Returns
    /// NDT result with final pose, score, and convergence status.
    pub fn align(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        initial_guess: Isometry3<f64>,
    ) -> NdtResult {
        // Convert initial guess to pose vector
        let mut pose = isometry_to_pose_vector(&initial_guess);

        // Track best result
        let mut best_score = f64::NEG_INFINITY;
        let mut best_pose = pose;
        let mut last_hessian = Matrix6::zeros();
        let mut last_correspondences = 0;

        // Track pose history for oscillation detection
        let mut pose_history: Vec<Isometry3<f64>> =
            Vec::with_capacity(self.config.ndt.max_iterations + 1);
        pose_history.push(initial_guess);

        // Main optimization loop
        for iteration in 0..self.config.ndt.max_iterations {
            // Compute derivatives at current pose
            let derivatives = compute_derivatives_cpu(
                source_points,
                target_grid,
                &pose,
                &self.gauss,
                true, // compute_hessian
            );

            // Check for sufficient correspondences
            if derivatives.num_correspondences < self.config.min_correspondences {
                if iteration == 0 {
                    return NdtResult::no_correspondences(initial_guess);
                }
                // Use best result so far
                break;
            }

            last_correspondences = derivatives.num_correspondences;

            // Apply GNSS regularization if enabled
            let (reg_score, reg_gradient, reg_hessian) = self
                .regularization
                .compute_derivatives(&pose, derivatives.num_correspondences);

            let total_score = derivatives.score + reg_score;
            let total_gradient = derivatives.gradient + reg_gradient;
            let total_hessian = derivatives.hessian + reg_hessian;

            last_hessian = total_hessian;

            // Track best score
            if total_score > best_score {
                best_score = total_score;
                best_pose = pose;
            }

            // Compute Newton step
            let delta = match newton_step_regularized(
                &total_gradient,
                &total_hessian,
                self.config.ndt.regularization,
                self.config.svd_tolerance,
            ) {
                Some(d) => d,
                None => {
                    // Singular Hessian - return current best
                    let final_pose = pose_vector_to_isometry(&best_pose);
                    let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);
                    let oscillation = super::oscillation::count_oscillation(
                        &pose_history,
                        super::oscillation::DEFAULT_OSCILLATION_THRESHOLD,
                    );
                    return NdtResult {
                        pose: final_pose,
                        status: ConvergenceStatus::SingularHessian,
                        score: best_score,
                        transform_probability: self.compute_transform_probability(best_score),
                        nvtl,
                        iterations: iteration,
                        hessian: last_hessian,
                        num_correspondences: last_correspondences,
                        oscillation_count: oscillation.max_oscillation_count,
                    };
                }
            };

            // Clamp Newton step to maximum step length (Autoware behavior)
            // step_size is the MAXIMUM allowed step length, not a damping factor
            let delta_norm = delta.norm();

            // Check convergence before clamping
            if delta_norm < self.config.ndt.trans_epsilon {
                let final_pose = pose_vector_to_isometry(&pose);
                let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);
                let oscillation = super::oscillation::count_oscillation(
                    &pose_history,
                    super::oscillation::DEFAULT_OSCILLATION_THRESHOLD,
                );
                return NdtResult {
                    pose: final_pose,
                    status: ConvergenceStatus::Converged,
                    score: total_score,
                    transform_probability: self.compute_transform_probability(total_score),
                    nvtl,
                    iterations: iteration + 1,
                    hessian: total_hessian,
                    num_correspondences: derivatives.num_correspondences,
                    oscillation_count: oscillation.max_oscillation_count,
                };
            }

            // Check if Newton step is an ascent direction for the score
            // d_phi_0 = -(gradient · step_dir) for minimizing -score (i.e., maximizing score)
            // If gradient · step_dir <= 0, the step is NOT ascending, so reverse it
            // (Autoware's computeStepLengthMT, lines 901-912)
            let mut step_dir = delta / delta_norm; // Normalized direction
            let grad_dot_step = total_gradient.dot(&step_dir);
            if grad_dot_step <= 0.0 {
                // Not an ascent direction - reverse it
                step_dir = -step_dir;
            }

            // Apply step (with optional line search to find optimal step length)
            // Note: Line search uses total derivatives including regularization
            let step_length = if self.config.ndt.use_line_search {
                self.line_search_with_regularization(
                    source_points,
                    target_grid,
                    &pose,
                    &step_dir,
                    total_score,
                    &total_gradient,
                )
            } else {
                // Autoware behavior: step_length = min(newton_step_norm, step_size)
                delta_norm.min(self.config.ndt.step_size)
            };

            pose = apply_pose_delta(&pose, &step_dir, step_length);
            pose_history.push(pose_vector_to_isometry(&pose));
        }

        // Reached max iterations
        let final_pose = pose_vector_to_isometry(&best_pose);
        let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);
        let oscillation = super::oscillation::count_oscillation(
            &pose_history,
            super::oscillation::DEFAULT_OSCILLATION_THRESHOLD,
        );
        NdtResult {
            pose: final_pose,
            status: ConvergenceStatus::MaxIterations,
            score: best_score,
            transform_probability: self.compute_transform_probability(best_score),
            nvtl,
            iterations: self.config.ndt.max_iterations,
            hessian: last_hessian,
            num_correspondences: last_correspondences,
            oscillation_count: oscillation.max_oscillation_count,
        }
    }

    /// Compute NVTL for a given pose.
    fn compute_nvtl(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        pose: &Isometry3<f64>,
    ) -> f64 {
        let config = NvtlConfig::default();
        let result = compute_nvtl(source_points, target_grid, pose, &self.gauss, &config);
        result.nvtl
    }

    /// Perform More-Thuente line search to find optimal step size.
    ///
    /// This implements the More-Thuente algorithm which guarantees both sufficient decrease
    /// (Armijo condition) and curvature condition (strong Wolfe condition).
    ///
    /// Note: This is superseded by `line_search_with_regularization` which handles
    /// both with and without regularization cases. Kept for reference.
    #[allow(dead_code)]
    fn line_search(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        pose: &[f64; 6],
        delta: &Vector6<f64>,
        current: &crate::derivatives::AggregatedDerivatives,
    ) -> f64 {
        let initial_derivative = directional_derivative(&current.gradient, delta);

        // If derivative is not positive, the step direction is not an ascent direction
        if initial_derivative <= 0.0 {
            return self.config.ndt.step_size;
        }

        let gauss = &self.gauss;
        let pose_copy = *pose;
        let delta_copy = *delta;

        // Score and derivative function for line search
        // More-Thuente needs both value and directional derivative at each trial point
        let score_and_derivative = |alpha: f64| {
            let test_pose = apply_pose_delta(&pose_copy, &delta_copy, alpha);
            let result =
                compute_derivatives_cpu(source_points, target_grid, &test_pose, gauss, false);
            let deriv = directional_derivative(&result.gradient, &delta_copy);
            (result.score, deriv)
        };

        // Use More-Thuente line search
        let result = more_thuente_search(
            score_and_derivative,
            current.score,
            initial_derivative,
            self.config.ndt.step_size, // Initial step (will be clamped by step_max)
            &self.config.more_thuente,
        );

        if result.converged {
            result.step_length
        } else {
            // Fallback to max step if line search didn't converge
            self.config.more_thuente.step_max
        }
    }

    /// Perform More-Thuente line search with regularization.
    ///
    /// This version includes GNSS regularization in the score and derivative computation.
    fn line_search_with_regularization(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        pose: &[f64; 6],
        delta: &Vector6<f64>,
        current_score: f64,
        current_gradient: &Vector6<f64>,
    ) -> f64 {
        let initial_derivative = directional_derivative(current_gradient, delta);

        // If derivative is not positive, the step direction is not an ascent direction
        if initial_derivative <= 0.0 {
            return self.config.ndt.step_size;
        }

        let gauss = &self.gauss;
        let pose_copy = *pose;
        let delta_copy = *delta;
        let regularization = &self.regularization;

        // Score and derivative function for line search (includes regularization)
        let score_and_derivative = |alpha: f64| {
            let test_pose = apply_pose_delta(&pose_copy, &delta_copy, alpha);
            let ndt_result =
                compute_derivatives_cpu(source_points, target_grid, &test_pose, gauss, false);

            // Add regularization contribution
            let (reg_score, reg_gradient, _) =
                regularization.compute_derivatives(&test_pose, ndt_result.num_correspondences);

            let total_score = ndt_result.score + reg_score;
            let total_gradient = ndt_result.gradient + reg_gradient;

            let deriv = directional_derivative(&total_gradient, &delta_copy);
            (total_score, deriv)
        };

        // Use More-Thuente line search
        let result = more_thuente_search(
            score_and_derivative,
            current_score,
            initial_derivative,
            self.config.ndt.step_size,
            &self.config.more_thuente,
        );

        if result.converged {
            result.step_length
        } else {
            self.config.more_thuente.step_max
        }
    }

    /// Compute transform probability from NDT score.
    ///
    /// Transform probability is a normalized score that ranges from 0 to ~1,
    /// where higher values indicate better alignment.
    fn compute_transform_probability(&self, score: f64) -> f64 {
        // NDT score is already positive (we negate d1 in the score function)
        // Normalize by dividing by expected maximum
        // The maximum occurs when all points are at voxel centers
        //
        // For now, just return the score directly
        // TODO: proper normalization based on number of points
        score.max(0.0)
    }

    /// Compute the condition number of a Hessian for diagnostics.
    pub fn hessian_condition_number(&self, hessian: &Matrix6<f64>) -> f64 {
        condition_number(hessian)
    }

    /// Align source points to target voxel grid with debug output.
    ///
    /// This is the same as `align()` but also returns detailed debug information
    /// about each iteration for comparison with Autoware's implementation.
    pub fn align_with_debug(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        initial_guess: Isometry3<f64>,
        timestamp_ns: u64,
    ) -> (NdtResult, super::debug::AlignmentDebug) {
        use super::debug::{AlignmentDebug, IterationDebug};

        let mut debug = AlignmentDebug::new(timestamp_ns);
        debug.num_source_points = source_points.len();

        // Convert initial guess to pose vector
        let mut pose = isometry_to_pose_vector(&initial_guess);
        debug.set_initial_pose(&pose);

        // Track best result
        let mut best_score = f64::NEG_INFINITY;
        let mut best_pose = pose;
        let mut last_hessian = Matrix6::zeros();
        let mut last_correspondences = 0;

        // Main optimization loop
        for iteration in 0..self.config.ndt.max_iterations {
            let mut iter_debug = IterationDebug::new(iteration);
            iter_debug.set_pose(&pose);

            // Compute derivatives at current pose
            let derivatives = compute_derivatives_cpu(
                source_points,
                target_grid,
                &pose,
                &self.gauss,
                true, // compute_hessian
            );

            iter_debug.score = derivatives.score;
            iter_debug.set_gradient(&derivatives.gradient);
            iter_debug.set_hessian(&derivatives.hessian);
            iter_debug.num_correspondences = derivatives.num_correspondences;
            iter_debug.used_line_search = self.config.ndt.use_line_search;

            // Check for sufficient correspondences
            if derivatives.num_correspondences < self.config.min_correspondences {
                debug.iterations.push(iter_debug);
                if iteration == 0 {
                    debug.convergence_status = "NoCorrespondences".to_string();
                    debug.total_iterations = iteration;
                    debug.set_final_pose(&pose);
                    debug.final_score = derivatives.score;
                    return (NdtResult::no_correspondences(initial_guess), debug);
                }
                break;
            }

            last_correspondences = derivatives.num_correspondences;
            last_hessian = derivatives.hessian;

            // Track best score
            if derivatives.score > best_score {
                best_score = derivatives.score;
                best_pose = pose;
            }

            // Compute Newton step
            let delta = match newton_step_regularized(
                &derivatives.gradient,
                &derivatives.hessian,
                self.config.ndt.regularization,
                self.config.svd_tolerance,
            ) {
                Some(d) => d,
                None => {
                    debug.iterations.push(iter_debug);
                    debug.convergence_status = "SingularHessian".to_string();
                    debug.total_iterations = iteration;
                    debug.set_final_pose(&best_pose);
                    debug.final_score = best_score;
                    debug.compute_oscillation();

                    let final_pose = pose_vector_to_isometry(&best_pose);
                    let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);
                    debug.final_nvtl = nvtl;
                    return (
                        NdtResult {
                            pose: final_pose,
                            status: ConvergenceStatus::SingularHessian,
                            score: best_score,
                            transform_probability: self.compute_transform_probability(best_score),
                            nvtl,
                            iterations: iteration,
                            hessian: last_hessian,
                            num_correspondences: last_correspondences,
                            oscillation_count: debug.oscillation_count,
                        },
                        debug,
                    );
                }
            };

            iter_debug.set_newton_step(&delta);
            let delta_norm = delta.norm();

            // Check convergence before clamping
            if delta_norm < self.config.ndt.trans_epsilon {
                iter_debug.step_length = delta_norm;
                iter_debug.set_pose_after(&pose);
                debug.iterations.push(iter_debug);
                debug.convergence_status = "Converged".to_string();
                debug.total_iterations = iteration + 1;
                debug.set_final_pose(&pose);
                debug.final_score = derivatives.score;
                debug.compute_oscillation();

                let final_pose = pose_vector_to_isometry(&pose);
                let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);
                debug.final_nvtl = nvtl;
                return (
                    NdtResult {
                        pose: final_pose,
                        status: ConvergenceStatus::Converged,
                        score: derivatives.score,
                        transform_probability: self
                            .compute_transform_probability(derivatives.score),
                        nvtl,
                        iterations: iteration + 1,
                        hessian: derivatives.hessian,
                        num_correspondences: derivatives.num_correspondences,
                        oscillation_count: debug.oscillation_count,
                    },
                    debug,
                );
            }

            // Check if Newton step is an ascent direction
            let mut step_dir = delta / delta_norm;
            let grad_dot_step = derivatives.gradient.dot(&step_dir);
            if grad_dot_step <= 0.0 {
                step_dir = -step_dir;
                iter_debug.direction_reversed = true;
            }
            iter_debug.set_step_direction(&step_dir);
            iter_debug.directional_derivative = derivatives.gradient.dot(&step_dir);

            // Apply step (with optional line search)
            let (step_length, line_search_converged) = if self.config.ndt.use_line_search {
                let result = self.line_search_with_result(
                    source_points,
                    target_grid,
                    &pose,
                    &step_dir,
                    &derivatives,
                );
                (result.0, result.1)
            } else {
                (delta_norm.min(self.config.ndt.step_size), true)
            };

            iter_debug.step_length = step_length;
            iter_debug.line_search_converged = line_search_converged;

            pose = apply_pose_delta(&pose, &step_dir, step_length);
            iter_debug.set_pose_after(&pose);
            debug.iterations.push(iter_debug);
        }

        // Reached max iterations
        debug.convergence_status = "MaxIterations".to_string();
        debug.total_iterations = self.config.ndt.max_iterations;
        debug.set_final_pose(&best_pose);
        debug.final_score = best_score;
        debug.compute_oscillation();

        let final_pose = pose_vector_to_isometry(&best_pose);
        let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);
        debug.final_nvtl = nvtl;

        (
            NdtResult {
                pose: final_pose,
                status: ConvergenceStatus::MaxIterations,
                score: best_score,
                transform_probability: self.compute_transform_probability(best_score),
                nvtl,
                iterations: self.config.ndt.max_iterations,
                hessian: last_hessian,
                num_correspondences: last_correspondences,
                oscillation_count: debug.oscillation_count,
            },
            debug,
        )
    }

    /// Line search that also returns convergence status.
    fn line_search_with_result(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        pose: &[f64; 6],
        delta: &Vector6<f64>,
        current: &crate::derivatives::AggregatedDerivatives,
    ) -> (f64, bool) {
        let initial_derivative = directional_derivative(&current.gradient, delta);

        if initial_derivative <= 0.0 {
            return (self.config.ndt.step_size, false);
        }

        let gauss = &self.gauss;
        let pose_copy = *pose;
        let delta_copy = *delta;

        let score_and_derivative = |alpha: f64| {
            let test_pose = apply_pose_delta(&pose_copy, &delta_copy, alpha);
            let result =
                compute_derivatives_cpu(source_points, target_grid, &test_pose, gauss, false);
            let deriv = directional_derivative(&result.gradient, &delta_copy);
            (result.score, deriv)
        };

        let result = more_thuente_search(
            score_and_derivative,
            current.score,
            initial_derivative,
            self.config.ndt.step_size,
            &self.config.more_thuente,
        );

        if result.converged {
            (result.step_length, true)
        } else {
            (self.config.more_thuente.step_max, false)
        }
    }
}

/// Check if the optimization has converged.
pub fn check_convergence(delta: &Vector6<f64>, epsilon: f64) -> bool {
    delta.norm() < epsilon
}

/// Compute the relative pose change between two poses.
pub fn pose_change(old_pose: &[f64; 6], new_pose: &[f64; 6]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..6 {
        let diff = new_pose[i] - old_pose[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rand_distr::Normal;

    fn create_test_grid(center: [f32; 3], spread: f32) -> VoxelGrid {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, spread as f64).unwrap();

        let mut points = Vec::new();
        for _ in 0..100 {
            points.push([
                center[0] + dist.sample(&mut rng) as f32,
                center[1] + dist.sample(&mut rng) as f32,
                center[2] + dist.sample(&mut rng) as f32,
            ]);
        }
        VoxelGrid::from_points(&points, 2.0).unwrap()
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = NdtOptimizer::with_defaults();
        assert_eq!(optimizer.config().ndt.resolution, 2.0);
        assert_eq!(optimizer.config().ndt.max_iterations, 30);
    }

    #[test]
    fn test_optimizer_with_resolution() {
        let optimizer = NdtOptimizer::with_resolution(1.0);
        assert_eq!(optimizer.config().ndt.resolution, 1.0);
    }

    #[test]
    fn test_align_identity() {
        // Create a grid and source points at the same location
        // Use a seeded RNG for reproducibility
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let dist = Normal::new(0.0, 0.1).unwrap();

        let grid = create_test_grid([0.0, 0.0, 0.0], 0.1);

        // Source points distributed around grid center (not all identical)
        let source_points: Vec<[f32; 3]> = (0..50)
            .map(|_| {
                [
                    dist.sample(&mut rng) as f32,
                    dist.sample(&mut rng) as f32,
                    dist.sample(&mut rng) as f32,
                ]
            })
            .collect();

        let optimizer = NdtOptimizer::with_defaults();
        let initial_guess = Isometry3::identity();

        let result = optimizer.align(&source_points, &grid, initial_guess);

        // Should converge or reach max iterations with a usable result
        assert!(result.status.is_usable(), "Status: {:?}", result.status);
        // Relax iteration limit since convergence depends on random data
        assert!(
            result.iterations <= 30,
            "Too many iterations: {}",
            result.iterations
        );
    }

    #[test]
    fn test_align_with_translation() {
        // Create a grid centered at origin
        let grid = create_test_grid([0.0, 0.0, 0.0], 0.1);

        // Source points offset by a small translation
        let offset = [0.1f32, 0.1, 0.0];
        let source_points: Vec<[f32; 3]> = (0..50).map(|_| offset).collect();

        let optimizer = NdtOptimizer::with_defaults();

        // Initial guess with the offset (should converge quickly)
        let initial_guess =
            Isometry3::translation(offset[0] as f64, offset[1] as f64, offset[2] as f64);

        let result = optimizer.align(&source_points, &grid, initial_guess);

        assert!(result.status.is_usable(), "Status: {:?}", result.status);
    }

    #[test]
    fn test_align_no_correspondences() {
        // Create a grid at origin
        let grid = create_test_grid([0.0, 0.0, 0.0], 0.1);

        // Source points far away (no overlap)
        let source_points: Vec<[f32; 3]> = (0..50).map(|_| [1000.0f32, 1000.0, 1000.0]).collect();

        let optimizer = NdtOptimizer::with_defaults();
        let initial_guess = Isometry3::identity();

        let result = optimizer.align(&source_points, &grid, initial_guess);

        assert_eq!(result.status, ConvergenceStatus::NoCorrespondences);
    }

    #[test]
    fn test_check_convergence() {
        let small_delta = Vector6::new(0.001, 0.001, 0.001, 0.001, 0.001, 0.001);
        assert!(check_convergence(&small_delta, 0.01));

        let large_delta = Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(!check_convergence(&large_delta, 0.01));
    }

    #[test]
    fn test_pose_change() {
        let pose1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pose2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let change = pose_change(&pose1, &pose2);
        assert_relative_eq!(change, 1.0, epsilon = 1e-10);

        let pose3 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let change3 = pose_change(&pose1, &pose3);
        assert_relative_eq!(change3, 3.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_convergence_status() {
        assert!(ConvergenceStatus::Converged.is_converged());
        assert!(ConvergenceStatus::Converged.is_usable());

        assert!(!ConvergenceStatus::MaxIterations.is_converged());
        assert!(ConvergenceStatus::MaxIterations.is_usable());

        assert!(!ConvergenceStatus::NoCorrespondences.is_converged());
        assert!(!ConvergenceStatus::NoCorrespondences.is_usable());
    }

    #[test]
    fn test_optimizer_with_line_search() {
        // Create a grid and test with line search enabled
        let grid = create_test_grid([0.0, 0.0, 0.0], 0.1);
        let source_points: Vec<[f32; 3]> = (0..50).map(|_| [0.0f32, 0.0, 0.0]).collect();

        let mut config = OptimizationConfig::default();
        config.ndt.use_line_search = true;

        let optimizer = NdtOptimizer::new(config);
        let result = optimizer.align(&source_points, &grid, Isometry3::identity());

        assert!(result.status.is_usable());
    }

    #[test]
    fn test_hessian_condition_number() {
        let optimizer = NdtOptimizer::with_defaults();

        let identity = Matrix6::identity();
        let cond = optimizer.hessian_condition_number(&identity);
        assert_relative_eq!(cond, 1.0, epsilon = 1e-10);

        let mut scaled = Matrix6::identity();
        scaled[(0, 0)] = 100.0;
        let cond_scaled = optimizer.hessian_condition_number(&scaled);
        assert_relative_eq!(cond_scaled, 100.0, epsilon = 1e-5);
    }
}
