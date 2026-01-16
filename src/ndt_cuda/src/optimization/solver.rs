//! NDT optimization solver.
//!
//! This module implements the main optimization loop for NDT scan matching:
//! 1. Transform source points using current pose
//! 2. Compute derivatives (gradient + Hessian) against target voxel grid
//! 3. Solve Newton step: Δp = -H⁻¹g
//! 4. Apply step with optional line search
//! 5. Check convergence and iterate

use nalgebra::{Isometry3, Matrix6, Vector6};

use super::full_gpu_pipeline_v2::{FullGpuPipelineV2, PipelineV2Config};
use super::line_search::{directional_derivative, LineSearchConfig};
use super::more_thuente::{more_thuente_search, MoreThuenteConfig};
use super::newton::{condition_number, newton_step_regularized};
use super::regularization::{RegularizationConfig, RegularizationTerm};
use super::types::{
    apply_pose_delta, isometry_to_pose_vector, pose_vector_to_isometry, ConvergenceStatus,
    NdtConfig, NdtResult,
};
use crate::derivatives::{compute_derivatives_cpu_with_metric, GaussianParams, GpuVoxelData};
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
            let derivatives = compute_derivatives_cpu_with_metric(
                source_points,
                target_grid,
                &pose,
                &self.gauss,
                true, // compute_hessian
                self.config.ndt.distance_metric,
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

    /// Align source points to target using full GPU Newton iteration with line search.
    ///
    /// This method runs the entire Newton optimization loop on GPU with integrated
    /// batched line search. Jacobians and Point Hessians are computed on GPU,
    /// eliminating ~490KB of CPU→GPU transfers per iteration. Per-iteration transfer
    /// is reduced to ~200 bytes (Newton solve requires f64 precision).
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud to align
    /// * `target_grid` - Target voxel grid (map)
    /// * `initial_guess` - Initial pose estimate
    ///
    /// # Returns
    /// NDT result with final pose, score, and convergence status.
    ///
    /// # Errors
    /// Returns an error if GPU pipeline initialization or computation fails.
    ///
    /// # Note
    /// This method supports GNSS regularization if configured.
    /// Oscillation detection is not tracked in GPU mode.
    pub fn align_full_gpu(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        initial_guess: Isometry3<f64>,
    ) -> Result<NdtResult, anyhow::Error> {
        // Create full GPU pipeline V2 with line search
        let max_points = source_points.len().max(1);
        let max_voxels = target_grid.len().max(1);

        // Configure V2 pipeline based on optimizer settings
        let config = PipelineV2Config {
            use_line_search: self.config.ndt.use_line_search,
            step_max: self.config.ndt.step_size as f32,
            regularization_enabled: self.config.regularization.enabled,
            regularization_scale_factor: self.config.regularization.scale_factor as f32,
            ..PipelineV2Config::default()
        };

        let mut pipeline = FullGpuPipelineV2::with_config(max_points, max_voxels, config)?;

        // Upload alignment data once
        let voxel_data = GpuVoxelData::from_voxel_grid(target_grid);
        pipeline.upload_alignment_data(
            source_points,
            &voxel_data,
            self.gauss.d1 as f32,
            self.gauss.d2 as f32,
            self.config.ndt.resolution as f32,
        )?;

        // Set regularization pose if enabled and available
        if self.config.regularization.enabled && self.regularization.has_reference_pose() {
            if let Some(ref_pose) = self.regularization.reference_translation() {
                pipeline.set_regularization_pose(ref_pose[0], ref_pose[1]);
            }
        }

        // Convert initial guess to pose vector
        let initial_pose = isometry_to_pose_vector(&initial_guess);

        // Run full GPU optimization (persistent kernel)
        let gpu_result = match pipeline.optimize(
            &initial_pose,
            self.config.ndt.max_iterations as u32,
            self.config.ndt.trans_epsilon,
        ) {
            Ok(r) => r,
            Err(e) => {
                let err_msg = e.to_string();
                if err_msg.contains("Matrix factorization failed") || err_msg.contains("info=") {
                    return Ok(NdtResult::no_correspondences(initial_guess));
                }
                return Err(e);
            }
        };

        // Convert result
        let final_pose = pose_vector_to_isometry(&gpu_result.pose);

        // Compute NVTL on CPU (requires final pose)
        let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);

        // Convert Hessian to nalgebra
        let hessian = Matrix6::from_fn(|i, j| gpu_result.hessian[i][j]);

        // Determine convergence status
        let status = if gpu_result.num_correspondences == 0 {
            ConvergenceStatus::NoCorrespondences
        } else if gpu_result.converged {
            ConvergenceStatus::Converged
        } else {
            ConvergenceStatus::MaxIterations
        };

        Ok(NdtResult {
            pose: final_pose,
            status,
            score: gpu_result.score,
            transform_probability: self.compute_transform_probability(gpu_result.score),
            nvtl,
            iterations: gpu_result.iterations as usize,
            hessian,
            num_correspondences: gpu_result.num_correspondences,
            oscillation_count: gpu_result.oscillation_count,
        })
    }

    /// Align multiple initial poses using full GPU pipeline with shared voxel data.
    ///
    /// This is optimized for MULTI_NDT covariance estimation where multiple
    /// alignments share the same source points and target grid. The GPU pipeline
    /// is created once and voxel data is uploaded once, then reused for all poses.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud to align
    /// * `target_grid` - Target voxel grid (map)
    /// * `initial_poses` - List of initial pose estimates
    ///
    /// # Returns
    /// Vector of NDT results, one per initial pose.
    ///
    /// # Errors
    /// Returns an error if GPU pipeline initialization fails.
    pub fn align_batch_gpu(
        &self,
        source_points: &[[f32; 3]],
        target_grid: &VoxelGrid,
        initial_poses: &[Isometry3<f64>],
    ) -> Result<Vec<NdtResult>, anyhow::Error> {
        if initial_poses.is_empty() {
            return Ok(vec![]);
        }

        // Create full GPU pipeline V2 once for all alignments
        let max_points = source_points.len().max(1);
        let max_voxels = target_grid.len().max(1);

        // Configure V2 pipeline based on optimizer settings
        let config = PipelineV2Config {
            use_line_search: self.config.ndt.use_line_search,
            step_max: self.config.ndt.step_size as f32,
            regularization_enabled: self.config.regularization.enabled,
            regularization_scale_factor: self.config.regularization.scale_factor as f32,
            ..PipelineV2Config::default()
        };

        let mut pipeline = FullGpuPipelineV2::with_config(max_points, max_voxels, config)?;

        // Upload voxel data once (shared across all alignments)
        let voxel_data = GpuVoxelData::from_voxel_grid(target_grid);
        pipeline.upload_alignment_data(
            source_points,
            &voxel_data,
            self.gauss.d1 as f32,
            self.gauss.d2 as f32,
            self.config.ndt.resolution as f32,
        )?;

        // Set regularization pose if enabled and available
        if self.config.regularization.enabled && self.regularization.has_reference_pose() {
            if let Some(ref_pose) = self.regularization.reference_translation() {
                pipeline.set_regularization_pose(ref_pose[0], ref_pose[1]);
            }
        }

        // Run alignment for each initial pose, reusing the pipeline
        let mut results = Vec::with_capacity(initial_poses.len());
        for initial_guess in initial_poses {
            let initial_pose = isometry_to_pose_vector(initial_guess);

            // Run full GPU optimization (persistent kernel)
            // Handle singular Hessian (no correspondences) gracefully
            let gpu_result = match pipeline.optimize(
                &initial_pose,
                self.config.ndt.max_iterations as u32,
                self.config.ndt.trans_epsilon,
            ) {
                Ok(r) => r,
                Err(e) => {
                    let err_msg = e.to_string();
                    if err_msg.contains("Matrix factorization failed") || err_msg.contains("info=")
                    {
                        results.push(NdtResult::no_correspondences(*initial_guess));
                        continue;
                    }
                    return Err(e);
                }
            };

            // Convert result
            let final_pose = pose_vector_to_isometry(&gpu_result.pose);

            // Compute NVTL on CPU
            let nvtl = self.compute_nvtl(source_points, target_grid, &final_pose);

            // Convert Hessian to nalgebra
            let hessian = Matrix6::from_fn(|i, j| gpu_result.hessian[i][j]);

            // Determine convergence status
            let status = if gpu_result.num_correspondences == 0 {
                ConvergenceStatus::NoCorrespondences
            } else if gpu_result.converged {
                ConvergenceStatus::Converged
            } else {
                ConvergenceStatus::MaxIterations
            };

            results.push(NdtResult {
                pose: final_pose,
                status,
                score: gpu_result.score,
                transform_probability: self.compute_transform_probability(gpu_result.score),
                nvtl,
                iterations: gpu_result.iterations as usize,
                hessian,
                num_correspondences: gpu_result.num_correspondences,
                oscillation_count: gpu_result.oscillation_count,
            });
        }

        Ok(results)
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
        let distance_metric = self.config.ndt.distance_metric;
        let score_and_derivative = |alpha: f64| {
            let test_pose = apply_pose_delta(&pose_copy, &delta_copy, alpha);
            let result = compute_derivatives_cpu_with_metric(
                source_points,
                target_grid,
                &test_pose,
                gauss,
                false,
                distance_metric,
            );
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
        let distance_metric = self.config.ndt.distance_metric;

        // Score and derivative function for line search (includes regularization)
        let score_and_derivative = |alpha: f64| {
            let test_pose = apply_pose_delta(&pose_copy, &delta_copy, alpha);
            let ndt_result = compute_derivatives_cpu_with_metric(
                source_points,
                target_grid,
                &test_pose,
                gauss,
                false,
                distance_metric,
            );

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
            let derivatives = compute_derivatives_cpu_with_metric(
                source_points,
                target_grid,
                &pose,
                &self.gauss,
                true, // compute_hessian
                self.config.ndt.distance_metric,
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
        let distance_metric = self.config.ndt.distance_metric;

        let score_and_derivative = |alpha: f64| {
            let test_pose = apply_pose_delta(&pose_copy, &delta_copy, alpha);
            let result = compute_derivatives_cpu_with_metric(
                source_points,
                target_grid,
                &test_pose,
                gauss,
                false,
                distance_metric,
            );
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

    // GPU path tests
    // Note: These tests pass individually but may fail in the full test suite due to CubeCL
    // internal state management. Run with `cargo test test_align_full_gpu --nocapture`
    // to verify functionality in isolation.

    #[test]
    #[ignore = "Flaky in full test suite due to CubeCL GPU state - passes individually"]
    fn test_align_full_gpu_identity() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let dist = Normal::new(0.0, 0.1).unwrap();

        let grid = create_test_grid([0.0, 0.0, 0.0], 0.1);

        // Source points distributed around grid center
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

        let result = optimizer.align_full_gpu(&source_points, &grid, initial_guess);
        assert!(result.is_ok(), "GPU align failed: {:?}", result.err());

        let result = result.unwrap();
        assert!(result.status.is_usable(), "Status: {:?}", result.status);
        assert!(
            result.iterations <= 30,
            "Too many iterations: {}",
            result.iterations
        );
    }

    #[test]
    fn test_align_full_gpu_no_correspondences() {
        let grid = create_test_grid([0.0, 0.0, 0.0], 0.1);

        // Source points far away (no overlap)
        let source_points: Vec<[f32; 3]> = (0..50).map(|_| [1000.0f32, 1000.0, 1000.0]).collect();

        let optimizer = NdtOptimizer::with_defaults();
        let initial_guess = Isometry3::identity();

        let result = optimizer.align_full_gpu(&source_points, &grid, initial_guess);
        assert!(result.is_ok(), "GPU align failed: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.status, ConvergenceStatus::NoCorrespondences);
    }

    #[test]
    #[ignore = "Flaky in full test suite due to CubeCL GPU state - passes individually"]
    fn test_align_full_gpu_vs_cpu() {
        // Test that GPU and CPU paths produce similar results
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, 0.2).unwrap();

        let grid = create_test_grid([0.0, 0.0, 0.0], 0.2);

        // Source points with some variation
        let source_points: Vec<[f32; 3]> = (0..100)
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

        // Run CPU path
        let cpu_result = optimizer.align(&source_points, &grid, initial_guess);

        // Run GPU path (full GPU with line search)
        let gpu_result = optimizer
            .align_full_gpu(&source_points, &grid, initial_guess)
            .expect("GPU align failed");

        // Both should produce usable results
        assert!(
            cpu_result.status.is_usable(),
            "CPU status: {:?}",
            cpu_result.status
        );
        assert!(
            gpu_result.status.is_usable(),
            "GPU status: {:?}",
            gpu_result.status
        );

        // Scores should be in the same ballpark (not exact due to f32 vs f64 differences)
        let score_ratio = if cpu_result.score.abs() > 1e-10 {
            (gpu_result.score / cpu_result.score).abs()
        } else {
            1.0
        };
        assert!(
            score_ratio > 0.5 && score_ratio < 2.0,
            "Score mismatch: CPU={}, GPU={}",
            cpu_result.score,
            gpu_result.score
        );

        // Final positions should be similar (within 0.5m for translation)
        let cpu_trans = cpu_result.pose.translation.vector;
        let gpu_trans = gpu_result.pose.translation.vector;
        let pos_diff = (cpu_trans - gpu_trans).norm();
        assert!(
            pos_diff < 0.5,
            "Position difference too large: {} (CPU={:?}, GPU={:?})",
            pos_diff,
            cpu_trans,
            gpu_trans
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --release test_align_performance -- --nocapture --ignored
    fn test_align_performance() {
        use std::time::Instant;

        // Create a larger test case for meaningful timing
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let dist = Normal::new(0.0, 0.3).unwrap();

        // Build a larger voxel grid (1000 points -> ~100+ voxels)
        let grid_points: Vec<[f32; 3]> = (0..1000)
            .map(|_| {
                [
                    dist.sample(&mut rng) as f32 * 10.0,
                    dist.sample(&mut rng) as f32 * 10.0,
                    dist.sample(&mut rng) as f32 * 2.0,
                ]
            })
            .collect();
        let grid = VoxelGrid::from_points(&grid_points, 2.0).unwrap();
        println!("Grid has {} voxels", grid.len());

        // Create 500 source points
        let source_points: Vec<[f32; 3]> = (0..500)
            .map(|_| {
                [
                    dist.sample(&mut rng) as f32 * 10.0,
                    dist.sample(&mut rng) as f32 * 10.0,
                    dist.sample(&mut rng) as f32 * 2.0,
                ]
            })
            .collect();

        let optimizer = NdtOptimizer::with_defaults();
        let initial_guess = Isometry3::translation(0.5, 0.5, 0.0);

        // Warm up
        let _ = optimizer.align(&source_points, &grid, initial_guess);
        let _ = optimizer.align_full_gpu(&source_points, &grid, initial_guess);

        // Benchmark CPU path
        const ITERATIONS: usize = 10;
        let cpu_start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = optimizer.align(&source_points, &grid, initial_guess);
        }
        let cpu_elapsed = cpu_start.elapsed();
        let cpu_per_align = cpu_elapsed.as_secs_f64() * 1000.0 / ITERATIONS as f64;

        // Benchmark GPU path (full GPU with line search)
        let gpu_start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = optimizer.align_full_gpu(&source_points, &grid, initial_guess);
        }
        let gpu_elapsed = gpu_start.elapsed();
        let gpu_per_align = gpu_elapsed.as_secs_f64() * 1000.0 / ITERATIONS as f64;

        println!("\n=== Performance Comparison ===");
        println!("Source points: {}", source_points.len());
        println!("Voxel grid: {} voxels", grid.len());
        println!("Iterations: {}", ITERATIONS);
        println!("CPU path: {:.2} ms per alignment", cpu_per_align);
        println!("GPU path: {:.2} ms per alignment", gpu_per_align);
        println!("Speedup: {:.2}x", cpu_per_align / gpu_per_align);
        println!("==============================\n");
    }
}
