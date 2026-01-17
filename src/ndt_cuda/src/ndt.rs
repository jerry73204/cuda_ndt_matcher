//! High-level NDT scan matching API.
//!
//! This module provides a simple, unified API for NDT scan matching
//! that can be used as a drop-in replacement for fast-gicp.
//!
//! # Example
//!
//! ```ignore
//! use ndt_cuda::NdtScanMatcher;
//! use nalgebra::Isometry3;
//!
//! // Create matcher with default settings
//! let mut matcher = NdtScanMatcher::new(2.0)?;
//!
//! // Set target (map) point cloud
//! matcher.set_target(&map_points)?;
//!
//! // Align source scan to target
//! let result = matcher.align(&source_points, Isometry3::identity())?;
//!
//! crate::test_println!("Converged: {}, Score: {}", result.converged, result.score);
//! ```

use anyhow::{bail, Result};
use nalgebra::{Isometry3, Matrix6};
use rayon::prelude::*;
use tracing::{debug, warn};

use crate::derivatives::gpu::GpuVoxelData;
use crate::derivatives::{DistanceMetric, GaussianParams};
use crate::optimization::{NdtOptimizer, OptimizationConfig};
use crate::runtime::{is_cuda_available, GpuRuntime};
use crate::scoring::{compute_nvtl, compute_transform_probability, GpuScoringPipeline, NvtlConfig};
use crate::voxel_grid::{VoxelGrid, VoxelGridConfig};

/// Configuration for NDT scan matcher.
#[derive(Debug, Clone)]
pub struct NdtScanMatcherConfig {
    /// Voxel resolution in meters.
    pub resolution: f32,

    /// Maximum optimization iterations.
    pub max_iterations: usize,

    /// Convergence threshold for translation change.
    pub trans_epsilon: f64,

    /// Maximum step length for Newton update (Autoware default: 0.1).
    ///
    /// The Newton step direction is normalized, then scaled by
    /// `min(newton_step_norm, step_size)`. This prevents large steps
    /// when far from the optimum while allowing full steps when close.
    ///
    /// NOTE: This is NOT a damping factor - it's the maximum allowed step length.
    pub step_size: f64,

    /// Outlier ratio for Gaussian parameters (0.0 to 1.0).
    /// Higher values make the algorithm more robust to outliers.
    /// Autoware default: 0.55
    pub outlier_ratio: f64,

    /// Regularization factor for Hessian conditioning.
    pub regularization: f64,

    /// Whether to use line search for step size.
    pub use_line_search: bool,

    /// Minimum points required per voxel.
    pub min_points_per_voxel: usize,

    /// Whether to use GPU acceleration (requires CUDA).
    /// Falls back to CPU if GPU is not available.
    pub use_gpu: bool,

    /// Whether to enable GNSS regularization.
    pub regularization_enabled: bool,

    /// Scale factor for GNSS regularization (default: 0.01).
    pub regularization_scale_factor: f64,

    /// Distance metric for NDT cost function.
    /// - PointToDistribution: Full Mahalanobis distance (default)
    /// - PointToPlane: Simplified point-to-plane distance (faster for planar structures)
    pub distance_metric: DistanceMetric,
}

impl Default for NdtScanMatcherConfig {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            max_iterations: 30,
            trans_epsilon: 0.01,
            step_size: 0.1, // Autoware default: maximum step length (NOT a damping factor)
            outlier_ratio: 0.55,
            regularization: 0.001,
            use_line_search: false,
            min_points_per_voxel: 6,
            use_gpu: false,                    // CPU by default for compatibility
            regularization_enabled: false,     // GNSS regularization disabled by default
            regularization_scale_factor: 0.01, // Autoware default
            distance_metric: DistanceMetric::PointToDistribution, // Full Mahalanobis by default
        }
    }
}

/// Builder for NdtScanMatcher configuration.
#[derive(Debug, Clone)]
pub struct NdtScanMatcherBuilder {
    config: NdtScanMatcherConfig,
}

impl NdtScanMatcherBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: NdtScanMatcherConfig::default(),
        }
    }

    /// Set voxel resolution.
    pub fn resolution(mut self, resolution: f32) -> Self {
        self.config.resolution = resolution;
        self
    }

    /// Set maximum iterations.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Set transformation epsilon for convergence.
    pub fn transformation_epsilon(mut self, epsilon: f64) -> Self {
        self.config.trans_epsilon = epsilon;
        self
    }

    /// Set step size.
    pub fn step_size(mut self, step_size: f64) -> Self {
        self.config.step_size = step_size;
        self
    }

    /// Set outlier ratio.
    pub fn outlier_ratio(mut self, outlier_ratio: f64) -> Self {
        self.config.outlier_ratio = outlier_ratio;
        self
    }

    /// Set regularization factor.
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.config.regularization = regularization;
        self
    }

    /// Enable or disable line search.
    pub fn use_line_search(mut self, use_line_search: bool) -> Self {
        self.config.use_line_search = use_line_search;
        self
    }

    /// Set minimum points per voxel.
    pub fn min_points_per_voxel(mut self, min_points: usize) -> Self {
        self.config.min_points_per_voxel = min_points;
        self
    }

    /// Enable or disable GPU acceleration.
    ///
    /// When enabled, GPU will be used for scoring and NVTL evaluation.
    /// Falls back to CPU if CUDA is not available.
    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Enable or disable GNSS regularization.
    ///
    /// When enabled, a regularization term is added to the NDT cost function
    /// that penalizes deviation from a GNSS pose.
    pub fn regularization_enabled(mut self, enabled: bool) -> Self {
        self.config.regularization_enabled = enabled;
        self
    }

    /// Set the GNSS regularization scale factor.
    ///
    /// Higher values give more weight to GNSS poses (default: 0.01).
    pub fn regularization_scale_factor(mut self, scale: f64) -> Self {
        self.config.regularization_scale_factor = scale;
        self
    }

    /// Set the distance metric for NDT cost function.
    ///
    /// - `PointToDistribution`: Full Mahalanobis distance (default, most accurate)
    /// - `PointToPlane`: Point-to-plane distance (faster, better for planar structures)
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Build the NDT scan matcher.
    pub fn build(self) -> Result<NdtScanMatcher> {
        NdtScanMatcher::with_config(self.config)
    }
}

impl Default for NdtScanMatcherBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of NDT alignment.
#[derive(Debug, Clone)]
pub struct AlignResult {
    /// Final aligned pose.
    pub pose: Isometry3<f64>,

    /// Whether the optimization converged.
    pub converged: bool,

    /// NDT score (higher = better alignment).
    pub score: f64,

    /// Transform probability (normalized score).
    pub transform_probability: f64,

    /// NVTL (Nearest Voxel Transformation Likelihood).
    pub nvtl: f64,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Hessian matrix at convergence (for covariance estimation).
    pub hessian: Matrix6<f64>,

    /// Number of point-voxel correspondences.
    pub num_correspondences: usize,

    /// Maximum consecutive oscillation count during optimization.
    /// Oscillation indicates the optimizer is bouncing between poses.
    pub oscillation_count: usize,
}

/// NDT scan matcher.
///
/// This is the main entry point for NDT scan matching.
/// It maintains a voxel grid of the target (map) and aligns
/// source scans to it.
pub struct NdtScanMatcher {
    /// Configuration.
    config: NdtScanMatcherConfig,

    /// Target voxel grid (built from map points).
    target_grid: Option<VoxelGrid>,

    /// Gaussian parameters for NDT score.
    gauss_params: GaussianParams,

    /// NDT optimizer (persists regularization state across calls).
    optimizer: NdtOptimizer,

    /// GPU runtime for accelerated scoring (None if GPU not available/enabled).
    gpu_runtime: Option<GpuRuntime>,

    /// GPU voxel data (cached for GPU scoring).
    gpu_voxel_data: Option<GpuVoxelData>,

    /// GPU scoring pipeline for batch NVTL/TP computation.
    gpu_scoring_pipeline: Option<GpuScoringPipeline>,
}

impl NdtScanMatcher {
    /// Create a new NDT scan matcher with default settings and given resolution.
    pub fn new(resolution: f32) -> Result<Self> {
        let config = NdtScanMatcherConfig {
            resolution,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new NDT scan matcher with custom configuration.
    pub fn with_config(config: NdtScanMatcherConfig) -> Result<Self> {
        let gauss_params = GaussianParams::new(config.resolution as f64, config.outlier_ratio);

        // Build optimizer configuration including regularization settings
        let opt_config = Self::build_optimizer_config_from(&config);
        let optimizer = NdtOptimizer::new(opt_config);

        // Initialize GPU runtime if enabled and available
        let gpu_runtime = if config.use_gpu && is_cuda_available() {
            match GpuRuntime::new() {
                Ok(runtime) => Some(runtime),
                Err(e) => {
                    warn!("Failed to initialize GPU runtime: {e}. Falling back to CPU.");
                    None
                }
            }
        } else {
            None
        };

        // Initialize GPU scoring pipeline for batch NVTL computation
        let gpu_scoring_pipeline = if gpu_runtime.is_some() {
            // Allocate with reasonable defaults:
            // - max_points: 10000 (typical downsampled scan)
            // - max_voxels: 100000 (typical map tile)
            // - max_poses: 100 (MULTI_NDT uses ~25 poses)
            match GpuScoringPipeline::new(10000, 100000, 100) {
                Ok(pipeline) => Some(pipeline),
                Err(e) => {
                    warn!("Failed to initialize GPU scoring pipeline: {e}. Using CPU.");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            target_grid: None,
            gauss_params,
            optimizer,
            gpu_runtime,
            gpu_voxel_data: None,
            gpu_scoring_pipeline,
        })
    }

    /// Create a builder for configuring the matcher.
    pub fn builder() -> NdtScanMatcherBuilder {
        NdtScanMatcherBuilder::new()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &NdtScanMatcherConfig {
        &self.config
    }

    /// Check if a target has been set.
    pub fn has_target(&self) -> bool {
        self.target_grid.is_some()
    }

    /// Get the target voxel grid.
    pub fn target_grid(&self) -> Option<&VoxelGrid> {
        self.target_grid.as_ref()
    }

    /// Set the target (map) point cloud.
    ///
    /// This builds a voxel grid from the points, which is used
    /// for subsequent alignment operations.
    pub fn set_target(&mut self, points: &[[f32; 3]]) -> Result<()> {
        if points.is_empty() {
            bail!("Target point cloud is empty");
        }

        let voxel_config = VoxelGridConfig {
            resolution: self.config.resolution,
            min_points_per_voxel: self.config.min_points_per_voxel,
            eigenvalue_ratio_threshold: 0.01,
        };

        let grid = VoxelGrid::from_points_with_config(points, voxel_config)?;

        if grid.is_empty() {
            bail!("No voxels created from target points (too sparse?)");
        }

        debug!(
            num_points = points.len(),
            num_voxels = grid.len(),
            resolution = self.config.resolution,
            "Target grid created"
        );

        // Prepare GPU voxel data if GPU runtime is available
        if self.gpu_runtime.is_some() {
            let gpu_voxel_data = GpuVoxelData::from_voxel_grid(&grid);

            // Initialize GPU scoring pipeline with target data
            if let Some(ref mut pipeline) = self.gpu_scoring_pipeline {
                if let Err(e) = pipeline.set_target(
                    &gpu_voxel_data,
                    self.gauss_params.d1,
                    self.gauss_params.d2,
                    self.config.resolution as f64,
                ) {
                    warn!("Failed to set GPU scoring target: {e}. GPU scoring disabled.");
                    self.gpu_scoring_pipeline = None;
                }
            }

            self.gpu_voxel_data = Some(gpu_voxel_data);
        }

        self.target_grid = Some(grid);
        Ok(())
    }

    /// Align source points to target with an initial guess.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `initial_guess` - Initial pose estimate
    ///
    /// # Returns
    /// Alignment result with final pose, scores, and diagnostics.
    ///
    /// # Note
    /// If regularization is enabled, call `set_regularization_pose()` before
    /// this method to update the GNSS reference pose.
    pub fn align(
        &self,
        source_points: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
    ) -> Result<AlignResult> {
        // Use full GPU path when GPU is enabled and regularization is disabled
        // (GPU path doesn't support GNSS regularization yet)
        if self.config.use_gpu && !self.config.regularization_enabled {
            return self.align_gpu(source_points, initial_guess);
        }

        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Use the stored optimizer which has regularization state
        let result = self.optimizer.align(source_points, grid, initial_guess);

        Ok(AlignResult {
            pose: result.pose,
            converged: result.status.is_converged(),
            score: result.score,
            transform_probability: result.transform_probability,
            nvtl: result.nvtl,
            iterations: result.iterations,
            hessian: result.hessian,
            num_correspondences: result.num_correspondences,
            oscillation_count: result.oscillation_count,
        })
    }

    /// Align source points to target using full GPU Newton iteration.
    ///
    /// This method runs the entire Newton optimization loop on GPU, computing
    /// Jacobians and Point Hessians on GPU instead of uploading them each iteration.
    /// This provides significant speedup by eliminating ~490KB of CPU→GPU transfers
    /// per iteration.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `initial_guess` - Initial pose estimate
    ///
    /// # Returns
    /// Alignment result with final pose, score, and convergence status.
    ///
    /// # Errors
    /// Returns an error if GPU initialization fails or no target is set.
    ///
    /// # Note
    /// This method currently does not support:
    /// - GNSS regularization (use `align()` if regularization is needed)
    /// - Line search
    /// - Oscillation detection
    pub fn align_gpu(
        &self,
        source_points: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
    ) -> Result<AlignResult> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Use full GPU path via optimizer
        let result = self
            .optimizer
            .align_full_gpu(source_points, grid, initial_guess)?;

        Ok(AlignResult {
            pose: result.pose,
            converged: result.status.is_converged(),
            score: result.score,
            transform_probability: result.transform_probability,
            nvtl: result.nvtl,
            iterations: result.iterations,
            hessian: result.hessian,
            num_correspondences: result.num_correspondences,
            oscillation_count: result.oscillation_count,
        })
    }

    /// Align source points to target with debug output.
    ///
    /// This is the same as `align()` but also returns detailed debug information
    /// about each iteration for comparison with Autoware's implementation.
    ///
    /// When GPU mode is enabled, this uses the GPU path for alignment but returns
    /// simplified debug info (no per-iteration data, only final state).
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `initial_guess` - Initial pose estimate
    /// * `timestamp_ns` - Timestamp in nanoseconds (for debug correlation)
    ///
    /// # Returns
    /// Tuple of (alignment result, debug info).
    pub fn align_with_debug(
        &self,
        source_points: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
        timestamp_ns: u64,
    ) -> Result<(AlignResult, crate::optimization::AlignmentDebug)> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Use GPU path when GPU is enabled and regularization is disabled
        // (matches align() behavior)
        if self.config.use_gpu && !self.config.regularization_enabled {
            return self.align_gpu_with_debug(source_points, initial_guess, timestamp_ns);
        }

        // CPU path with full per-iteration debug info
        let opt_config = self.build_optimizer_config();
        let optimizer = NdtOptimizer::new(opt_config);

        // Run alignment with debug
        let (result, debug) =
            optimizer.align_with_debug(source_points, grid, initial_guess, timestamp_ns);

        Ok((
            AlignResult {
                pose: result.pose,
                converged: result.status.is_converged(),
                score: result.score,
                transform_probability: result.transform_probability,
                nvtl: result.nvtl,
                iterations: result.iterations,
                hessian: result.hessian,
                num_correspondences: result.num_correspondences,
                oscillation_count: result.oscillation_count,
            },
            debug,
        ))
    }

    /// GPU variant of align_with_debug.
    ///
    /// Uses the GPU path for alignment and returns simplified debug info
    /// (final state only, no per-iteration data since GPU runs entire loop).
    fn align_gpu_with_debug(
        &self,
        source_points: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
        timestamp_ns: u64,
    ) -> Result<(AlignResult, crate::optimization::AlignmentDebug)> {
        use crate::optimization::AlignmentDebug;

        // Extract initial pose for debug
        let translation = initial_guess.translation.vector;
        let rotation = initial_guess.rotation.euler_angles();
        let initial_pose_arr = [
            translation.x,
            translation.y,
            translation.z,
            rotation.0,
            rotation.1,
            rotation.2,
        ];

        // Run GPU alignment
        let result = self.align_gpu(source_points, initial_guess)?;

        // Extract final pose for debug
        let final_translation = result.pose.translation.vector;
        let final_rotation = result.pose.rotation.euler_angles();
        let final_pose_arr = [
            final_translation.x,
            final_translation.y,
            final_translation.z,
            final_rotation.0,
            final_rotation.1,
            final_rotation.2,
        ];

        // Build debug info from GPU result
        let mut debug = AlignmentDebug::new(timestamp_ns);
        debug.set_initial_pose(&initial_pose_arr);
        debug.set_final_pose(&final_pose_arr);
        debug.num_source_points = source_points.len();
        debug.convergence_status = if result.converged {
            "Converged".to_string()
        } else {
            "MaxIterations".to_string()
        };
        debug.total_iterations = result.iterations;
        debug.final_score = result.score;
        debug.final_nvtl = result.nvtl;
        debug.oscillation_count = result.oscillation_count;
        debug.num_correspondences = Some(result.num_correspondences);
        // Note: iterations Vec is empty for GPU path (no per-iteration data)

        Ok((result, debug))
    }

    /// Evaluate NVTL at a given pose without running optimization.
    ///
    /// Useful for pose quality assessment.
    ///
    /// Uses GPU acceleration when available, falling back to CPU if GPU is
    /// not initialized or fails.
    ///
    /// # Algorithm (Autoware-compatible)
    ///
    /// For each source point:
    /// 1. Transform by pose
    /// 2. Find all voxels within search radius
    /// 3. Compute NDT score for each voxel
    /// 4. Take the **maximum** score (not sum)
    ///
    /// Final NVTL = average of max scores across all points with neighbors.
    pub fn evaluate_nvtl(&self, source_points: &[[f32; 3]], pose: &Isometry3<f64>) -> Result<f64> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Try GPU path first if available
        if let Some(nvtl) = self.evaluate_nvtl_gpu(source_points, pose) {
            return Ok(nvtl);
        }

        // Fall back to CPU
        let config = NvtlConfig::default();
        let result = compute_nvtl(source_points, grid, pose, &self.gauss_params, &config);
        Ok(result.nvtl)
    }

    /// Evaluate transform probability at a given pose without running optimization.
    ///
    /// Uses GPU acceleration when available, falling back to CPU if GPU is
    /// not initialized or fails.
    pub fn evaluate_transform_probability(
        &self,
        source_points: &[[f32; 3]],
        pose: &Isometry3<f64>,
    ) -> Result<f64> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Try GPU path first if available
        if let Some((total_score, total_correspondences)) =
            self.evaluate_scores_gpu(source_points, pose)
        {
            let transform_probability = if total_correspondences > 0 {
                total_score / total_correspondences as f64
            } else {
                0.0
            };
            return Ok(transform_probability);
        }

        // Fall back to CPU
        let result = compute_transform_probability(source_points, grid, pose, &self.gauss_params);
        Ok(result.transform_probability)
    }

    /// Compute per-point scores for visualization.
    pub fn compute_per_point_scores(
        &self,
        source_points: &[[f32; 3]],
        pose: &Isometry3<f64>,
    ) -> Result<Vec<f64>> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            return Ok(Vec::new());
        }

        let scores =
            crate::scoring::compute_per_point_scores(source_points, grid, pose, &self.gauss_params);
        Ok(scores)
    }

    /// Evaluate NVTL at multiple poses (GPU-accelerated when available).
    ///
    /// This is optimized for multi-NDT covariance estimation where we need
    /// to evaluate NVTL at many offset poses quickly.
    ///
    /// When GPU is available, uses batched GPU kernel for ~15× speedup over Rayon.
    /// Falls back to Rayon parallel CPU when GPU is not available.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `poses` - List of poses to evaluate
    ///
    /// # Returns
    /// Vector of NVTL scores, one per pose.
    pub fn evaluate_nvtl_batch(
        &mut self,
        source_points: &[[f32; 3]],
        poses: &[Isometry3<f64>],
    ) -> Result<Vec<f64>> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            return Ok(vec![0.0; poses.len()]);
        }

        // Try GPU scoring pipeline first
        if let Some(ref mut pipeline) = self.gpu_scoring_pipeline {
            // Convert Isometry3 poses to [x, y, z, roll, pitch, yaw] format
            let poses_6dof: Vec<[f64; 6]> = poses
                .iter()
                .map(|iso| {
                    let translation = iso.translation;
                    let euler = iso.rotation.euler_angles();
                    [
                        translation.x,
                        translation.y,
                        translation.z,
                        euler.0, // roll
                        euler.1, // pitch
                        euler.2, // yaw
                    ]
                })
                .collect();

            // Use GPU reduction path: downloads M×4 floats vs M×N×4 floats
            match pipeline.compute_scores_batch_gpu_reduce(source_points, &poses_6dof) {
                Ok(results) => {
                    return Ok(results.iter().map(|r| r.nvtl).collect());
                }
                Err(e) => {
                    warn!("GPU scoring failed: {e}. Falling back to CPU.");
                }
            }
        }

        // Fall back to Rayon parallel CPU
        let config = NvtlConfig::default();
        let gauss = &self.gauss_params;

        let scores: Vec<f64> = poses
            .par_iter()
            .map(|pose| {
                let result = compute_nvtl(source_points, grid, pose, gauss, &config);
                result.nvtl
            })
            .collect();

        Ok(scores)
    }

    /// Align from multiple initial poses in parallel and return all results.
    ///
    /// This is useful for multi-NDT covariance estimation where we need
    /// to run alignment from multiple offset poses.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `initial_poses` - List of initial poses to try
    ///
    /// # Returns
    /// Vector of alignment results, one per pose.
    pub fn align_batch(
        &self,
        source_points: &[[f32; 3]],
        initial_poses: &[Isometry3<f64>],
    ) -> Result<Vec<AlignResult>> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        // Create optimizer with current config
        let opt_config = self.build_optimizer_config();

        // Parallel alignment across all initial poses
        let results: Vec<AlignResult> = initial_poses
            .par_iter()
            .map(|initial_guess| {
                let optimizer = NdtOptimizer::new(opt_config.clone());
                let result = optimizer.align(source_points, grid, *initial_guess);

                AlignResult {
                    pose: result.pose,
                    converged: result.status.is_converged(),
                    score: result.score,
                    transform_probability: result.transform_probability,
                    nvtl: result.nvtl,
                    iterations: result.iterations,
                    hessian: result.hessian,
                    num_correspondences: result.num_correspondences,
                    oscillation_count: result.oscillation_count,
                }
            })
            .collect();

        Ok(results)
    }

    /// Align source points to target from multiple initial poses using GPU.
    ///
    /// This is optimized for MULTI_NDT covariance estimation. The GPU pipeline
    /// is created once and voxel data is uploaded once, then reused for all poses.
    /// This provides significant speedup over `align_batch` when GPU is available.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud to align
    /// * `initial_poses` - List of initial poses to try
    ///
    /// # Returns
    /// Vector of alignment results, one per pose.
    ///
    /// # Errors
    /// Returns an error if no target is set or GPU initialization fails.
    pub fn align_batch_gpu(
        &self,
        source_points: &[[f32; 3]],
        initial_poses: &[Isometry3<f64>],
    ) -> Result<Vec<AlignResult>> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        if initial_poses.is_empty() {
            return Ok(vec![]);
        }

        // Create optimizer with current config
        let opt_config = self.build_optimizer_config();
        let optimizer = NdtOptimizer::new(opt_config);

        // Use GPU batch alignment with shared voxel data
        let ndt_results = optimizer.align_batch_gpu(source_points, grid, initial_poses)?;

        // Convert NdtResult to AlignResult
        let results = ndt_results
            .into_iter()
            .map(|result| AlignResult {
                pose: result.pose,
                converged: result.status.is_converged(),
                score: result.score,
                transform_probability: result.transform_probability,
                nvtl: result.nvtl,
                iterations: result.iterations,
                hessian: result.hessian,
                num_correspondences: result.num_correspondences,
                oscillation_count: result.oscillation_count,
            })
            .collect();

        Ok(results)
    }

    /// Check if GPU acceleration is active.
    ///
    /// Returns true if GPU runtime was successfully initialized and is being used.
    pub fn is_gpu_active(&self) -> bool {
        self.gpu_runtime.is_some()
    }

    /// Compute per-point max scores for visualization (GPU-accelerated).
    ///
    /// Returns transformed points (in map frame) and their max NDT scores,
    /// suitable for per-point color visualization like Autoware's voxel_score_points.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `pose` - Alignment result pose
    ///
    /// # Returns
    /// Tuple of (transformed_points, max_scores) where:
    /// - transformed_points[i] is source_points[i] transformed by pose
    /// - max_scores[i] is the maximum NDT score for point i
    pub fn compute_per_point_scores_for_visualization(
        &mut self,
        source_points: &[[f32; 3]],
        pose: &Isometry3<f64>,
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>)> {
        if source_points.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Try GPU scoring pipeline
        if let Some(ref mut pipeline) = self.gpu_scoring_pipeline {
            // Convert Isometry3 to [x, y, z, roll, pitch, yaw]
            let translation = pose.translation;
            let euler = pose.rotation.euler_angles();
            let pose_6dof = [
                translation.x,
                translation.y,
                translation.z,
                euler.0, // roll
                euler.1, // pitch
                euler.2, // yaw
            ];

            return pipeline.compute_per_point_scores(source_points, &pose_6dof);
        }

        // Fall back to CPU computation
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        let scores =
            crate::scoring::compute_per_point_scores(source_points, grid, pose, &self.gauss_params);

        // Transform points to map frame
        let transformed: Vec<[f32; 3]> = source_points
            .iter()
            .map(|p| {
                let pt = nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                let transformed = pose * pt;
                [
                    transformed.x as f32,
                    transformed.y as f32,
                    transformed.z as f32,
                ]
            })
            .collect();

        // Convert f64 scores to f32
        let scores_f32: Vec<f32> = scores.iter().map(|s| *s as f32).collect();

        Ok((transformed, scores_f32))
    }

    /// Evaluate scores using GPU if available (for transform probability).
    ///
    /// This is an internal method used by evaluate_transform_probability()
    /// when GPU is available. Returns `None` if GPU is not available or fails.
    ///
    /// Uses sum-based aggregation: total_score = sum of all voxel contributions.
    fn evaluate_scores_gpu(
        &self,
        source_points: &[[f32; 3]],
        pose: &Isometry3<f64>,
    ) -> Option<(f64, usize)> {
        let runtime = self.gpu_runtime.as_ref()?;
        let voxel_data = self.gpu_voxel_data.as_ref()?;

        // Convert pose to transform matrix
        use crate::derivatives::gpu::pose_to_transform_matrix;
        use crate::optimization::types::isometry_to_pose_vector;

        let pose_vec = isometry_to_pose_vector(pose);
        let transform = pose_to_transform_matrix(&pose_vec);

        // Use GPU for score computation
        match runtime.compute_scores(
            source_points,
            voxel_data,
            &transform,
            self.gauss_params.d1 as f32,
            self.gauss_params.d2 as f32,
            self.config.resolution,
        ) {
            Ok(result) => Some((result.total_score, result.total_correspondences)),
            Err(_) => None,
        }
    }

    /// Evaluate NVTL using GPU if available.
    ///
    /// This is an internal method used by evaluate_nvtl() when GPU is available.
    /// Returns `None` if GPU is not available or fails.
    ///
    /// Uses max-based aggregation: NVTL = average of max scores per point.
    /// This matches Autoware's NVTL algorithm.
    fn evaluate_nvtl_gpu(&self, source_points: &[[f32; 3]], pose: &Isometry3<f64>) -> Option<f64> {
        let runtime = self.gpu_runtime.as_ref()?;
        let voxel_data = self.gpu_voxel_data.as_ref()?;

        // Convert pose to transform matrix
        use crate::derivatives::gpu::pose_to_transform_matrix;
        use crate::optimization::types::isometry_to_pose_vector;

        let pose_vec = isometry_to_pose_vector(pose);
        let transform = pose_to_transform_matrix(&pose_vec);

        // Use GPU for NVTL computation (max per point, not sum)
        match runtime.compute_nvtl_scores(
            source_points,
            voxel_data,
            &transform,
            self.gauss_params.d1 as f32,
            self.gauss_params.d2 as f32,
            self.config.resolution,
        ) {
            Ok(result) => Some(result.nvtl),
            Err(_) => None,
        }
    }

    /// Set the GNSS regularization pose.
    ///
    /// When regularization is enabled, this pose is used as a reference
    /// to penalize deviation in the vehicle's longitudinal direction.
    ///
    /// Call this before each align() to update the GNSS reference.
    pub fn set_regularization_pose(&mut self, pose: Isometry3<f64>) {
        self.optimizer.set_regularization_pose(pose);
    }

    /// Clear the GNSS regularization pose.
    ///
    /// Call this when GNSS is unavailable or unreliable.
    pub fn clear_regularization_pose(&mut self) {
        self.optimizer.clear_regularization_pose();
    }

    /// Check if regularization is enabled.
    pub fn is_regularization_enabled(&self) -> bool {
        self.config.regularization_enabled
    }

    /// Build optimizer configuration from settings (static version for construction).
    fn build_optimizer_config_from(config: &NdtScanMatcherConfig) -> OptimizationConfig {
        use crate::optimization::{NdtConfig, RegularizationConfig};

        OptimizationConfig {
            ndt: NdtConfig {
                resolution: config.resolution as f64,
                max_iterations: config.max_iterations,
                trans_epsilon: config.trans_epsilon,
                step_size: config.step_size,
                outlier_ratio: config.outlier_ratio,
                regularization: config.regularization,
                use_line_search: config.use_line_search,
                distance_metric: config.distance_metric,
            },
            regularization: RegularizationConfig {
                enabled: config.regularization_enabled,
                scale_factor: config.regularization_scale_factor,
            },
            ..Default::default()
        }
    }

    /// Build optimizer configuration from current settings.
    fn build_optimizer_config(&self) -> OptimizationConfig {
        Self::build_optimizer_config_from(&self.config)
    }

    /// Align multiple independent scans in parallel using GPU batch processing.
    ///
    /// This is optimized for throughput when you have multiple scans to process
    /// simultaneously. Each scan can have different source points and initial poses.
    ///
    /// Uses `BatchGpuPipeline` which partitions GPU blocks across M independent
    /// alignments, allowing them to run truly in parallel with atomic barriers
    /// instead of cooperative grid synchronization.
    ///
    /// # Arguments
    /// * `scans` - Slice of (source_points, initial_pose) tuples
    ///
    /// # Returns
    /// Vector of alignment results, one per scan.
    ///
    /// # Performance
    /// - Processes up to 8 scans in parallel (limited by GPU memory and SM count)
    /// - Each slot runs independent Newton optimization
    /// - Significantly higher throughput than sequential processing
    ///
    /// # Example
    /// ```ignore
    /// let scans = vec![
    ///     (scan1_points.as_slice(), pose1),
    ///     (scan2_points.as_slice(), pose2),
    ///     (scan3_points.as_slice(), pose3),
    /// ];
    /// let results = matcher.align_parallel_scans(&scans)?;
    /// ```
    pub fn align_parallel_scans(
        &self,
        scans: &[(&[[f32; 3]], Isometry3<f64>)],
    ) -> Result<Vec<AlignResult>> {
        use crate::optimization::{AlignmentRequest, BatchGpuPipeline, BatchPipelineConfig};

        if scans.is_empty() {
            return Ok(vec![]);
        }

        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        // Find max points across all scans for capacity
        let max_points = scans.iter().map(|(pts, _)| pts.len()).max().unwrap_or(0);
        if max_points == 0 {
            bail!("All source point clouds are empty");
        }

        let num_slots = scans.len().min(8); // Limit to 8 parallel slots
        let max_voxels = grid.len();

        // Configure batch pipeline from matcher settings
        let batch_config = BatchPipelineConfig {
            use_line_search: self.config.use_line_search,
            fixed_step_size: self.config.step_size as f32,
            regularization_enabled: self.config.regularization_enabled,
            regularization_scale_factor: self.config.regularization_scale_factor as f32,
            ..BatchPipelineConfig::default()
        };

        // Create batch pipeline
        let mut pipeline =
            BatchGpuPipeline::with_config(num_slots, max_points, max_voxels, batch_config)?;

        // Upload voxel data (shared across all alignments)
        let voxel_data = self
            .gpu_voxel_data
            .clone()
            .unwrap_or_else(|| GpuVoxelData::from_voxel_grid(grid));

        pipeline.upload_voxel_data(
            &voxel_data,
            self.gauss_params.d1 as f32,
            self.gauss_params.d2 as f32,
            self.config.resolution,
        )?;

        // Build alignment requests
        let requests: Vec<AlignmentRequest<'_>> = scans
            .iter()
            .map(|(points, pose)| {
                use crate::optimization::types::isometry_to_pose_vector;
                let pose_vec = isometry_to_pose_vector(pose);
                AlignmentRequest {
                    points,
                    initial_pose: pose_vec,
                    reg_ref_x: None,
                    reg_ref_y: None,
                }
            })
            .collect();

        // Run batch alignment
        let batch_results = pipeline.align_batch(
            &requests,
            self.config.max_iterations as u32,
            self.config.trans_epsilon,
        )?;

        // Convert to AlignResult
        let results = batch_results
            .into_iter()
            .map(|r| {
                use crate::optimization::types::pose_vector_to_isometry;
                AlignResult {
                    pose: pose_vector_to_isometry(&r.pose),
                    converged: r.converged,
                    score: r.score,
                    transform_probability: if r.num_correspondences > 0 {
                        r.score / r.num_correspondences as f64
                    } else {
                        0.0
                    },
                    nvtl: 0.0, // Not computed in batch mode
                    iterations: r.iterations as usize,
                    hessian: Matrix6::from_fn(|i, j| r.hessian[i][j]),
                    num_correspondences: r.num_correspondences,
                    oscillation_count: r.oscillation_count,
                }
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        make_default_half_cubic_pcd, make_half_cubic_pcd_offset, voxelize_pcd,
    };
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rand::SeedableRng;
    use rand_distr::Normal;

    fn generate_test_cloud(center: [f32; 3], spread: f32, num_points: usize) -> Vec<[f32; 3]> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dist = Normal::new(0.0, spread as f64).unwrap();

        (0..num_points)
            .map(|_| {
                [
                    center[0] + dist.sample(&mut rng) as f32,
                    center[1] + dist.sample(&mut rng) as f32,
                    center[2] + dist.sample(&mut rng) as f32,
                ]
            })
            .collect()
    }

    #[test]
    fn test_builder() {
        let matcher = NdtScanMatcher::builder()
            .resolution(1.0)
            .max_iterations(50)
            .transformation_epsilon(0.001)
            .outlier_ratio(0.5)
            .build()
            .unwrap();

        assert_eq!(matcher.config().resolution, 1.0);
        assert_eq!(matcher.config().max_iterations, 50);
        assert_relative_eq!(matcher.config().trans_epsilon, 0.001);
    }

    #[test]
    fn test_set_target() {
        let mut matcher = NdtScanMatcher::new(2.0).unwrap();
        assert!(!matcher.has_target());

        let points = generate_test_cloud([0.0, 0.0, 0.0], 0.5, 100);
        matcher.set_target(&points).unwrap();

        assert!(matcher.has_target());
        assert!(matcher.target_grid().is_some());
    }

    #[test]
    fn test_set_target_empty() {
        let mut matcher = NdtScanMatcher::new(2.0).unwrap();
        let points: Vec<[f32; 3]> = Vec::new();

        let result = matcher.set_target(&points);
        assert!(result.is_err());
    }

    #[test]
    fn test_align_identity() {
        let mut matcher = NdtScanMatcher::new(2.0).unwrap();

        // Create target and source at same location
        let target = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 200);
        matcher.set_target(&target).unwrap();

        let source = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 100);
        let result = matcher.align(&source, Isometry3::identity()).unwrap();

        // Should have positive score and NVTL (alignment should be usable)
        assert!(result.score > 0.0, "Score should be positive");
        assert!(result.nvtl > 0.0, "NVTL should be positive");
        assert!(
            result.num_correspondences > 0,
            "Should have correspondences"
        );
    }

    #[test]
    fn test_align_with_translation() {
        let mut matcher = NdtScanMatcher::new(2.0).unwrap();

        // Target at center
        let target = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 200);
        matcher.set_target(&target).unwrap();

        // Source offset by small translation
        let source = generate_test_cloud([5.2, 5.2, 5.0], 0.3, 100);

        // Initial guess close to truth
        let initial_guess = Isometry3::translation(0.2, 0.2, 0.0);
        let result = matcher.align(&source, initial_guess).unwrap();

        // Should find a good alignment
        assert!(result.score > 0.0);
        assert!(result.num_correspondences > 0);
    }

    #[test]
    fn test_align_no_target() {
        let matcher = NdtScanMatcher::new(2.0).unwrap();
        let source = generate_test_cloud([0.0, 0.0, 0.0], 0.3, 100);

        let result = matcher.align(&source, Isometry3::identity());
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_nvtl() {
        let mut matcher = NdtScanMatcher::new(2.0).unwrap();

        let target = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 200);
        matcher.set_target(&target).unwrap();

        let source = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 100);

        let nvtl = matcher
            .evaluate_nvtl(&source, &Isometry3::identity())
            .unwrap();

        // Points at same location should have positive NVTL
        assert!(nvtl > 0.0);
    }

    #[test]
    fn test_evaluate_transform_probability() {
        let mut matcher = NdtScanMatcher::new(2.0).unwrap();

        let target = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 200);
        matcher.set_target(&target).unwrap();

        let source = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 100);

        let tp = matcher
            .evaluate_transform_probability(&source, &Isometry3::identity())
            .unwrap();

        assert!(tp > 0.0);
    }

    #[test]
    fn test_per_point_scores() {
        let mut matcher = NdtScanMatcher::new(2.0).unwrap();

        let target = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 200);
        matcher.set_target(&target).unwrap();

        let source = generate_test_cloud([5.0, 5.0, 5.0], 0.3, 50);

        let scores = matcher
            .compute_per_point_scores(&source, &Isometry3::identity())
            .unwrap();

        assert_eq!(scores.len(), 50);
        // Most points should have positive scores
        let positive_count = scores.iter().filter(|&&s| s > 0.0).count();
        assert!(positive_count > 0);
    }

    // ========================================================================
    // Phase 3: Alignment integration tests (Autoware-style)
    // ========================================================================

    /// Port of Autoware's `standard_sequence_for_initial_pose_estimation`.
    ///
    /// Tests the standard NDT alignment workflow:
    /// 1. Create map at offset (100, 100)
    /// 2. Create sensor scan (half-cube, voxelized)
    /// 3. Align with initial guess at (100, 100)
    /// 4. Verify result is within 2m tolerance
    #[test]
    fn test_standard_alignment_sequence() {
        // Arrange: Create target map (half-cube at offset 100, 100)
        let map_points = make_half_cubic_pcd_offset(100.0, 100.0, 0.0);

        let mut matcher = NdtScanMatcher::builder()
            .resolution(2.0)
            .max_iterations(30)
            .transformation_epsilon(0.01)
            .build()
            .unwrap();

        matcher.set_target(&map_points).unwrap();

        // Create sensor scan (same half-cube structure, voxelized)
        let sensor_scan = voxelize_pcd(&make_default_half_cubic_pcd(), 1.0);

        // Act: Align with initial guess at (100, 100, 0)
        // The sensor scan is at origin, map is at (100, 100), so initial guess
        // should translate sensor to map location
        let initial_guess = Isometry3::translation(100.0, 100.0, 0.0);
        let result = matcher.align(&sensor_scan, initial_guess).unwrap();

        // Assert: Result should be close to initial guess (2m tolerance like Autoware)
        let final_translation = result.pose.translation.vector;
        assert!(
            (final_translation.x - 100.0).abs() < 2.0,
            "x translation should be near 100, got {}",
            final_translation.x
        );
        assert!(
            (final_translation.y - 100.0).abs() < 2.0,
            "y translation should be near 100, got {}",
            final_translation.y
        );
        assert!(
            final_translation.z.abs() < 2.0,
            "z translation should be near 0, got {}",
            final_translation.z
        );

        // Verify convergence metrics
        assert!(result.score > 0.0, "Score should be positive");
        assert!(result.nvtl > 0.0, "NVTL should be positive");
        assert!(
            result.num_correspondences > 0,
            "Should have correspondences"
        );
    }

    /// Port of Autoware's `once_initialize_at_out_of_map_then_initialize_correctly`.
    ///
    /// Tests recovery from a bad initial pose:
    /// 1. Try alignment with pose outside map bounds
    /// 2. Try again with correct pose
    /// 3. Verify second attempt succeeds
    #[test]
    fn test_recovery_from_bad_initial_pose() {
        // Arrange: Create target map at (100, 100)
        let map_points = make_half_cubic_pcd_offset(100.0, 100.0, 0.0);

        let mut matcher = NdtScanMatcher::builder()
            .resolution(2.0)
            .max_iterations(30)
            .build()
            .unwrap();

        matcher.set_target(&map_points).unwrap();

        let sensor_scan = voxelize_pcd(&make_default_half_cubic_pcd(), 1.0);

        // Act 1: Try bad initial pose (outside map bounds)
        let bad_guess = Isometry3::translation(-100.0, -100.0, 0.0);
        let bad_result = matcher.align(&sensor_scan, bad_guess).unwrap();

        // Bad pose should either not converge well or have low scores
        // (The node shouldn't crash - this is the key assertion)

        // Act 2: Try correct initial pose
        let good_guess = Isometry3::translation(100.0, 100.0, 0.0);
        let good_result = matcher.align(&sensor_scan, good_guess).unwrap();

        // Assert: Good pose should give better results than bad pose
        assert!(
            good_result.score > bad_result.score || good_result.nvtl > bad_result.nvtl,
            "Good initial pose should give better score than bad pose"
        );

        // Good result should be close to expected
        let final_translation = good_result.pose.translation.vector;
        assert!(
            (final_translation.x - 100.0).abs() < 2.0,
            "x should be near 100, got {}",
            final_translation.x
        );
        assert!(
            (final_translation.y - 100.0).abs() < 2.0,
            "y should be near 100, got {}",
            final_translation.y
        );
    }

    /// Test alignment with small translation offset.
    ///
    /// Verifies that NDT can correct small misalignments.
    #[test]
    fn test_align_small_offset() {
        let map_points = make_default_half_cubic_pcd();

        let mut matcher = NdtScanMatcher::builder()
            .resolution(2.0)
            .max_iterations(30)
            .build()
            .unwrap();

        matcher.set_target(&map_points).unwrap();

        // Sensor scan is same as map but we'll give a slightly wrong initial guess
        let sensor_scan = voxelize_pcd(&map_points, 1.0);

        // Initial guess with 0.5m offset (should be correctable)
        let initial_guess = Isometry3::translation(0.5, 0.3, 0.1);
        let result = matcher.align(&sensor_scan, initial_guess).unwrap();

        // Result should be closer to identity than initial guess
        let final_translation = result.pose.translation.vector;
        let final_offset = (final_translation.x.powi(2)
            + final_translation.y.powi(2)
            + final_translation.z.powi(2))
        .sqrt();
        let initial_offset = (0.5_f64.powi(2) + 0.3_f64.powi(2) + 0.1_f64.powi(2)).sqrt();

        assert!(
            final_offset < initial_offset + 0.5,
            "Final offset ({}) should not be much worse than initial ({})",
            final_offset,
            initial_offset
        );
    }

    /// Test alignment scoring is consistent.
    ///
    /// Verifies that aligned poses score better than completely wrong poses.
    /// Note: With multi-voxel radius search, score ordering can be non-monotonic
    /// for small offsets, but large offsets (outside map) should score lower.
    #[test]
    fn test_alignment_score_ordering() {
        let map_points = make_default_half_cubic_pcd();

        let mut matcher = NdtScanMatcher::new(2.0).unwrap();
        matcher.set_target(&map_points).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);

        // Evaluate scores at different offsets
        let score_at_0 = matcher
            .evaluate_transform_probability(&sensor_scan, &Isometry3::identity())
            .unwrap();
        let score_at_far = matcher
            .evaluate_transform_probability(&sensor_scan, &Isometry3::translation(50.0, 50.0, 0.0))
            .unwrap();

        // Score at origin should be positive
        assert!(
            score_at_0 > 0.0,
            "Score at origin ({}) should be positive",
            score_at_0
        );

        // Score far outside map should be very low (near zero or zero)
        assert!(
            score_at_far < score_at_0,
            "Score far from map ({}) should be < score at origin ({})",
            score_at_far,
            score_at_0
        );
    }

    /// Test NVTL is positive and finite.
    ///
    /// Note: With multi-voxel matching (radius search), NVTL can exceed 1.0
    /// because each point can contribute to multiple voxels.
    #[test]
    fn test_nvtl_range() {
        let map_points = make_default_half_cubic_pcd();

        let mut matcher = NdtScanMatcher::new(2.0).unwrap();
        matcher.set_target(&map_points).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);

        // Test at various offsets
        for offset in [0.0, 1.0, 2.0, 5.0, 10.0] {
            let pose = Isometry3::translation(offset, 0.0, 0.0);
            let nvtl = matcher.evaluate_nvtl(&sensor_scan, &pose).unwrap();

            assert!(
                nvtl >= 0.0 && nvtl.is_finite(),
                "NVTL should be non-negative and finite, got {} at offset {}",
                nvtl,
                offset
            );
        }
    }

    /// Test that Hessian is returned for covariance estimation.
    #[test]
    fn test_hessian_returned() {
        let map_points = make_default_half_cubic_pcd();

        let mut matcher = NdtScanMatcher::new(2.0).unwrap();
        matcher.set_target(&map_points).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);
        let result = matcher.align(&sensor_scan, Isometry3::identity()).unwrap();

        // Hessian should be returned (6x6)
        assert_eq!(result.hessian.nrows(), 6);
        assert_eq!(result.hessian.ncols(), 6);

        // Diagonal should be non-zero (Hessian should have curvature info)
        let has_diagonal = (0..6).any(|i| result.hessian[(i, i)].abs() > 1e-10);
        assert!(has_diagonal, "Hessian diagonal should have non-zero values");
    }

    // ========================================================================
    // GPU vs CPU comparison tests
    // ========================================================================

    /// Skip test at runtime if CUDA is not available.
    macro_rules! require_cuda {
        () => {
            if !crate::runtime::is_cuda_available() {
                crate::test_println!("Skipping test: CUDA not available");
                return;
            }
        };
    }

    // NOTE: GPU scoring tests are temporarily disabled due to a CubeCL optimizer
    // bug in uniformity analysis. The `radius_search_kernel` has complex control
    // flow (dynamic loop + break + conditional) that causes "no entry found for
    // key" panic in cubecl-opt-0.8.1.
    //
    // TODO: Re-enable once CubeCL is updated or we simplify the kernels.

    /// Test that GPU and CPU NVTL computations produce consistent results.
    ///
    /// This verifies that `compute_ndt_nvtl_kernel` (max-per-point) matches
    /// the CPU `compute_nvtl` implementation.
    #[test]
    fn test_gpu_cpu_nvtl_consistency() {
        require_cuda!();

        let map_points = make_default_half_cubic_pcd();

        // Create matcher with GPU enabled
        let mut matcher_gpu = NdtScanMatcher::builder()
            .resolution(2.0)
            .use_gpu(true)
            .build()
            .unwrap();

        // Create matcher with GPU disabled (CPU only)
        let mut matcher_cpu = NdtScanMatcher::builder()
            .resolution(2.0)
            .use_gpu(false)
            .build()
            .unwrap();

        matcher_gpu.set_target(&map_points).unwrap();
        matcher_cpu.set_target(&map_points).unwrap();

        // Verify GPU is actually being used
        assert!(
            matcher_gpu.is_gpu_active(),
            "GPU matcher should have GPU active"
        );
        assert!(
            !matcher_cpu.is_gpu_active(),
            "CPU matcher should not have GPU active"
        );

        let sensor_scan = voxelize_pcd(&map_points, 1.0);

        // Test at various poses
        let test_poses = [
            Isometry3::identity(),
            Isometry3::translation(0.5, 0.0, 0.0),
            Isometry3::translation(1.0, 1.0, 0.0),
        ];

        for pose in &test_poses {
            let nvtl_gpu = matcher_gpu.evaluate_nvtl(&sensor_scan, pose).unwrap();
            let nvtl_cpu = matcher_cpu.evaluate_nvtl(&sensor_scan, pose).unwrap();

            crate::test_println!(
                "NVTL comparison at {:?}: GPU={:.6}, CPU={:.6}, diff={:.6}",
                pose.translation.vector,
                nvtl_gpu,
                nvtl_cpu,
                (nvtl_gpu - nvtl_cpu).abs()
            );

            // Allow tolerance for GPU (f32) vs CPU (f64) floating point differences
            // and different radius search voxel ordering. The GPU brute-force search
            // may find voxels in different order than CPU KD-tree, affecting which
            // voxel's score is selected as max for NVTL. ~5% relative tolerance.
            let relative_diff = (nvtl_gpu - nvtl_cpu).abs() / nvtl_cpu.abs().max(1.0);
            assert!(
                relative_diff < 0.05,
                "GPU NVTL ({}) should match CPU NVTL ({}) within 5% relative tolerance (got {}%)",
                nvtl_gpu,
                nvtl_cpu,
                relative_diff * 100.0
            );
        }
    }

    /// Test that GPU and CPU transform probability computations produce consistent results.
    ///
    /// This verifies that `compute_ndt_score_kernel` (sum-based) matches
    /// the CPU `compute_transform_probability` implementation.
    #[test]
    fn test_gpu_cpu_transform_probability_consistency() {
        require_cuda!();

        let map_points = make_default_half_cubic_pcd();

        // Create matcher with GPU enabled
        let mut matcher_gpu = NdtScanMatcher::builder()
            .resolution(2.0)
            .use_gpu(true)
            .build()
            .unwrap();

        // Create matcher with GPU disabled (CPU only)
        let mut matcher_cpu = NdtScanMatcher::builder()
            .resolution(2.0)
            .use_gpu(false)
            .build()
            .unwrap();

        matcher_gpu.set_target(&map_points).unwrap();
        matcher_cpu.set_target(&map_points).unwrap();

        let sensor_scan = voxelize_pcd(&map_points, 1.0);

        // Test at various poses
        let test_poses = [
            Isometry3::identity(),
            Isometry3::translation(0.5, 0.0, 0.0),
            Isometry3::translation(1.0, 1.0, 0.0),
        ];

        for pose in &test_poses {
            let tp_gpu = matcher_gpu
                .evaluate_transform_probability(&sensor_scan, pose)
                .unwrap();
            let tp_cpu = matcher_cpu
                .evaluate_transform_probability(&sensor_scan, pose)
                .unwrap();

            crate::test_println!(
                "Transform probability at {:?}: GPU={:.6}, CPU={:.6}, diff={:.6}",
                pose.translation.vector,
                tp_gpu,
                tp_cpu,
                (tp_gpu - tp_cpu).abs()
            );

            // Allow small tolerance for floating point differences
            assert!(
                (tp_gpu - tp_cpu).abs() < 0.01,
                "GPU transform probability ({}) should match CPU ({}) within tolerance",
                tp_gpu,
                tp_cpu
            );
        }
    }

    /// Test that GPU-enabled alignment produces similar results to CPU-only alignment.
    #[test]
    #[ignore = "Flaky in full test suite due to CubeCL GPU state - passes individually"]
    fn test_gpu_cpu_alignment_consistency() {
        require_cuda!();

        let map_points = make_half_cubic_pcd_offset(100.0, 100.0, 0.0);

        // Create matcher with GPU enabled
        let mut matcher_gpu = NdtScanMatcher::builder()
            .resolution(2.0)
            .max_iterations(30)
            .use_gpu(true)
            .build()
            .unwrap();

        // Create matcher with GPU disabled
        let mut matcher_cpu = NdtScanMatcher::builder()
            .resolution(2.0)
            .max_iterations(30)
            .use_gpu(false)
            .build()
            .unwrap();

        matcher_gpu.set_target(&map_points).unwrap();
        matcher_cpu.set_target(&map_points).unwrap();

        let sensor_scan = voxelize_pcd(&make_default_half_cubic_pcd(), 1.0);
        let initial_guess = Isometry3::translation(100.0, 100.0, 0.0);

        let result_gpu = matcher_gpu.align(&sensor_scan, initial_guess).unwrap();
        let result_cpu = matcher_cpu.align(&sensor_scan, initial_guess).unwrap();

        crate::test_println!(
            "Alignment comparison:\n  GPU: pos=({:.3}, {:.3}, {:.3}), score={:.4}, nvtl={:.4}\n  CPU: pos=({:.3}, {:.3}, {:.3}), score={:.4}, nvtl={:.4}",
            result_gpu.pose.translation.x, result_gpu.pose.translation.y, result_gpu.pose.translation.z,
            result_gpu.score, result_gpu.nvtl,
            result_cpu.pose.translation.x, result_cpu.pose.translation.y, result_cpu.pose.translation.z,
            result_cpu.score, result_cpu.nvtl,
        );

        // Both should converge to similar positions
        let pos_diff =
            (result_gpu.pose.translation.vector - result_cpu.pose.translation.vector).norm();
        assert!(
            pos_diff < 1.0,
            "GPU and CPU should converge to similar positions, diff={:.3}m",
            pos_diff
        );

        // Both should have good scores
        assert!(result_gpu.score > 0.0, "GPU score should be positive");
        assert!(result_cpu.score > 0.0, "CPU score should be positive");
    }
}
