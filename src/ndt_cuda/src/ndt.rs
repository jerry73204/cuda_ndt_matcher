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

use crate::derivatives::GaussianParams;
use crate::optimization::{NdtOptimizer, OptimizationConfig};
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

    /// Step size for Newton updates.
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
}

impl Default for NdtScanMatcherConfig {
    fn default() -> Self {
        Self {
            resolution: 2.0,
            max_iterations: 30,
            trans_epsilon: 0.01,
            step_size: 0.1,
            outlier_ratio: 0.55,
            regularization: 0.001,
            use_line_search: false,
            min_points_per_voxel: 6,
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
}

impl NdtScanMatcher {
    /// Create a new NDT scan matcher with default settings and given resolution.
    pub fn new(resolution: f32) -> Result<Self> {
        let mut config = NdtScanMatcherConfig::default();
        config.resolution = resolution;
        Self::with_config(config)
    }

    /// Create a new NDT scan matcher with custom configuration.
    pub fn with_config(config: NdtScanMatcherConfig) -> Result<Self> {
        let gauss_params = GaussianParams::new(config.resolution as f64, config.outlier_ratio);

        Ok(Self {
            config,
            target_grid: None,
            gauss_params,
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

        // Create optimizer with current config
        let opt_config = self.build_optimizer_config();
        let optimizer = NdtOptimizer::new(opt_config);

        // Run alignment
        let result = optimizer.align(source_points, grid, initial_guess);

        Ok(AlignResult {
            pose: result.pose,
            converged: result.status.is_converged(),
            score: result.score,
            transform_probability: result.transform_probability,
            nvtl: result.nvtl,
            iterations: result.iterations,
            hessian: result.hessian,
            num_correspondences: result.num_correspondences,
        })
    }

    /// Evaluate NVTL at a given pose without running optimization.
    ///
    /// Useful for pose quality assessment.
    pub fn evaluate_nvtl(&self, source_points: &[[f32; 3]], pose: &Isometry3<f64>) -> Result<f64> {
        let grid = self
            .target_grid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target set. Call set_target() first."))?;

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        let config = NvtlConfig::default();
        let result = compute_nvtl(source_points, grid, pose, &self.gauss_params, &config);
        Ok(result.nvtl)
    }

    /// Evaluate transform probability at a given pose without running optimization.
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

    /// Build optimizer configuration from current settings.
    fn build_optimizer_config(&self) -> OptimizationConfig {
        use crate::optimization::NdtConfig;

        OptimizationConfig {
            ndt: NdtConfig {
                resolution: self.config.resolution as f64,
                max_iterations: self.config.max_iterations,
                trans_epsilon: self.config.trans_epsilon,
                step_size: self.config.step_size,
                outlier_ratio: self.config.outlier_ratio,
                regularization: self.config.regularization,
                use_line_search: self.config.use_line_search,
            },
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
