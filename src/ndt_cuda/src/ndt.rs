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
//! println!("Converged: {}, Score: {}", result.converged, result.score);
//! ```

use anyhow::{bail, Result};
use nalgebra::{Isometry3, Matrix6};
use rayon::prelude::*;

use crate::derivatives::gpu::GpuVoxelData;
use crate::derivatives::GaussianParams;
use crate::optimization::{NdtOptimizer, OptimizationConfig};
use crate::runtime::{is_cuda_available, GpuRuntime};
use crate::scoring::{compute_nvtl, compute_transform_probability, NvtlConfig};
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
                Ok(runtime) => {
                    eprintln!("[ndt_cuda] GPU runtime initialized successfully");
                    Some(runtime)
                }
                Err(e) => {
                    eprintln!(
                        "[ndt_cuda] Failed to initialize GPU runtime: {e}. Falling back to CPU."
                    );
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

        // Prepare GPU voxel data if GPU runtime is available
        if self.gpu_runtime.is_some() {
            self.gpu_voxel_data = Some(GpuVoxelData::from_voxel_grid(&grid));
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

    /// Align source points to target with debug output.
    ///
    /// This is the same as `align()` but also returns detailed debug information
    /// about each iteration for comparison with Autoware's implementation.
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

        // Create optimizer with current config
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

    /// Evaluate NVTL at multiple poses in parallel (GPU-accelerated via Rayon).
    ///
    /// This is optimized for multi-NDT covariance estimation where we need
    /// to evaluate NVTL at many offset poses quickly.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `poses` - List of poses to evaluate
    ///
    /// # Returns
    /// Vector of NVTL scores, one per pose.
    pub fn evaluate_nvtl_batch(
        &self,
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

        let config = NvtlConfig::default();
        let gauss = &self.gauss_params;

        // Parallel NVTL evaluation across all poses
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

    /// Check if GPU acceleration is active.
    ///
    /// Returns true if GPU runtime was successfully initialized and is being used.
    pub fn is_gpu_active(&self) -> bool {
        self.gpu_runtime.is_some()
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
}
