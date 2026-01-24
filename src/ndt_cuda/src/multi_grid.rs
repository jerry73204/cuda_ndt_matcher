//! Multi-resolution NDT scan matching for coarse-to-fine alignment.
//!
//! This module implements a multi-grid NDT approach where alignment is performed
//! progressively from coarse to fine resolution voxel grids. This helps with:
//! - Large initial pose errors (coarse grids have larger basins of attraction)
//! - Avoiding local minima (progressive refinement)
//! - Improved convergence rates
//!
//! # Example
//!
//! ```ignore
//! use ndt_cuda::MultiGridNdt;
//! use nalgebra::Isometry3;
//!
//! // Create multi-grid matcher with resolutions: 4.0m -> 2.0m -> 1.0m
//! let mut matcher = MultiGridNdt::new(&[4.0, 2.0, 1.0])?;
//!
//! // Set target (map) point cloud - builds grids at all resolutions
//! matcher.set_target(&map_points)?;
//!
//! // Align source scan - cascades through resolutions
//! let result = matcher.align(&source_points, Isometry3::identity())?;
//! ```

use anyhow::{bail, Result};
use nalgebra::{Isometry3, Matrix6};

use crate::derivatives::DistanceMetric;
use crate::ndt::{AlignResult, NdtScanMatcher, NdtScanMatcherConfig};

/// Configuration for a single resolution level in multi-grid NDT.
#[derive(Debug, Clone)]
pub struct GridLevelConfig {
    /// Voxel resolution in meters.
    pub resolution: f32,

    /// Maximum iterations for this level.
    pub max_iterations: usize,

    /// Convergence threshold for this level.
    pub trans_epsilon: f64,
}

impl GridLevelConfig {
    /// Create a new grid level configuration.
    pub fn new(resolution: f32, max_iterations: usize) -> Self {
        Self {
            resolution,
            max_iterations,
            trans_epsilon: 0.01,
        }
    }

    /// Create with custom convergence threshold.
    pub fn with_epsilon(resolution: f32, max_iterations: usize, trans_epsilon: f64) -> Self {
        Self {
            resolution,
            max_iterations,
            trans_epsilon,
        }
    }
}

/// Configuration for multi-grid NDT.
#[derive(Debug, Clone)]
pub struct MultiGridConfig {
    /// Grid levels from coarse to fine.
    pub levels: Vec<GridLevelConfig>,

    /// Maximum step size for Newton update.
    pub step_size: f64,

    /// Outlier ratio for Gaussian parameters.
    pub outlier_ratio: f64,

    /// Minimum points required per voxel.
    pub min_points_per_voxel: usize,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// Distance metric for NDT cost function.
    pub distance_metric: DistanceMetric,
}

impl Default for MultiGridConfig {
    fn default() -> Self {
        Self {
            // Default: 3 levels - coarse (4m), medium (2m), fine (1m)
            levels: vec![
                GridLevelConfig::new(4.0, 5),  // Coarse: 5 iterations
                GridLevelConfig::new(2.0, 10), // Medium: 10 iterations
                GridLevelConfig::new(1.0, 15), // Fine: 15 iterations
            ],
            step_size: 0.1,
            outlier_ratio: 0.55,
            min_points_per_voxel: 6,
            use_gpu: true,
            distance_metric: DistanceMetric::PointToDistribution,
        }
    }
}

impl MultiGridConfig {
    /// Create a configuration with custom resolutions.
    ///
    /// Iterations are automatically assigned based on resolution:
    /// - Coarsest: 5 iterations
    /// - Middle levels: 10 iterations
    /// - Finest: 15 iterations
    pub fn with_resolutions(resolutions: &[f32]) -> Self {
        let n = resolutions.len();
        let levels = resolutions
            .iter()
            .enumerate()
            .map(|(i, &res)| {
                let max_iter = if i == 0 {
                    5 // Coarsest
                } else if i == n - 1 {
                    15 // Finest
                } else {
                    10 // Middle
                };
                GridLevelConfig::new(res, max_iter)
            })
            .collect();

        Self {
            levels,
            ..Default::default()
        }
    }

    /// Create a two-level configuration (coarse + fine).
    pub fn two_level(coarse: f32, fine: f32) -> Self {
        Self {
            levels: vec![
                GridLevelConfig::new(coarse, 5),
                GridLevelConfig::new(fine, 15),
            ],
            ..Default::default()
        }
    }

    /// Create a three-level configuration (coarse + medium + fine).
    pub fn three_level(coarse: f32, medium: f32, fine: f32) -> Self {
        Self {
            levels: vec![
                GridLevelConfig::new(coarse, 5),
                GridLevelConfig::new(medium, 10),
                GridLevelConfig::new(fine, 15),
            ],
            ..Default::default()
        }
    }
}

/// Builder for MultiGridNdt configuration.
#[derive(Debug, Clone)]
pub struct MultiGridNdtBuilder {
    config: MultiGridConfig,
}

impl MultiGridNdtBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: MultiGridConfig::default(),
        }
    }

    /// Set the grid levels (resolutions and iterations).
    pub fn levels(mut self, levels: Vec<GridLevelConfig>) -> Self {
        self.config.levels = levels;
        self
    }

    /// Set resolutions (iterations auto-assigned).
    pub fn resolutions(mut self, resolutions: &[f32]) -> Self {
        self.config = MultiGridConfig::with_resolutions(resolutions);
        self
    }

    /// Set the maximum step size.
    pub fn step_size(mut self, step_size: f64) -> Self {
        self.config.step_size = step_size;
        self
    }

    /// Set the outlier ratio.
    pub fn outlier_ratio(mut self, ratio: f64) -> Self {
        self.config.outlier_ratio = ratio;
        self
    }

    /// Set minimum points per voxel.
    pub fn min_points_per_voxel(mut self, min_points: usize) -> Self {
        self.config.min_points_per_voxel = min_points;
        self
    }

    /// Enable or disable GPU acceleration.
    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Set the distance metric.
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Build the MultiGridNdt instance.
    pub fn build(self) -> Result<MultiGridNdt> {
        MultiGridNdt::with_config(self.config)
    }
}

impl Default for MultiGridNdtBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of multi-grid alignment with per-level details.
#[derive(Debug, Clone)]
pub struct MultiGridAlignResult {
    /// Final aligned pose.
    pub pose: Isometry3<f64>,

    /// Whether the final level converged.
    pub converged: bool,

    /// Final NDT score.
    pub score: f64,

    /// Transform probability from finest level.
    pub transform_probability: f64,

    /// NVTL from finest level.
    pub nvtl: f64,

    /// Total iterations across all levels.
    pub total_iterations: usize,

    /// Iterations per level.
    pub iterations_per_level: Vec<usize>,

    /// Hessian matrix from finest level.
    pub hessian: Matrix6<f64>,

    /// Number of correspondences from finest level.
    pub num_correspondences: usize,

    /// Maximum oscillation count across all levels.
    pub oscillation_count: usize,

    /// Resolutions used (from coarse to fine).
    pub resolutions: Vec<f32>,
}

impl From<MultiGridAlignResult> for AlignResult {
    fn from(result: MultiGridAlignResult) -> Self {
        Self {
            pose: result.pose,
            converged: result.converged,
            score: result.score,
            transform_probability: result.transform_probability,
            nvtl: result.nvtl,
            iterations: result.total_iterations,
            hessian: result.hessian,
            num_correspondences: result.num_correspondences,
            oscillation_count: result.oscillation_count,
        }
    }
}

/// Multi-resolution NDT scan matcher.
///
/// Performs coarse-to-fine alignment using multiple voxel grid resolutions.
/// This improves convergence for large initial pose errors by first aligning
/// with a coarse grid (large basin of attraction), then refining with finer
/// grids.
pub struct MultiGridNdt {
    /// Configuration.
    config: MultiGridConfig,

    /// NDT matchers for each resolution level (coarse to fine).
    matchers: Vec<NdtScanMatcher>,
}

impl MultiGridNdt {
    /// Create a new multi-grid NDT with given resolutions.
    ///
    /// Resolutions should be provided from coarse to fine (e.g., [4.0, 2.0, 1.0]).
    /// Iterations are automatically assigned.
    pub fn new(resolutions: &[f32]) -> Result<Self> {
        let config = MultiGridConfig::with_resolutions(resolutions);
        Self::with_config(config)
    }

    /// Create a new multi-grid NDT with custom configuration.
    pub fn with_config(config: MultiGridConfig) -> Result<Self> {
        if config.levels.is_empty() {
            bail!("At least one grid level is required");
        }

        // Create matchers for each level
        let matchers = config
            .levels
            .iter()
            .map(|level| {
                let matcher_config = NdtScanMatcherConfig {
                    resolution: level.resolution,
                    max_iterations: level.max_iterations,
                    trans_epsilon: level.trans_epsilon,
                    step_size: config.step_size,
                    outlier_ratio: config.outlier_ratio,
                    min_points_per_voxel: config.min_points_per_voxel,
                    use_gpu: config.use_gpu,
                    distance_metric: config.distance_metric,
                    ..Default::default()
                };
                NdtScanMatcher::with_config(matcher_config)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { config, matchers })
    }

    /// Create a builder for configuring multi-grid NDT.
    pub fn builder() -> MultiGridNdtBuilder {
        MultiGridNdtBuilder::new()
    }

    /// Get the configuration.
    pub fn config(&self) -> &MultiGridConfig {
        &self.config
    }

    /// Get the number of resolution levels.
    pub fn num_levels(&self) -> usize {
        self.matchers.len()
    }

    /// Get resolutions from coarse to fine.
    pub fn resolutions(&self) -> Vec<f32> {
        self.config.levels.iter().map(|l| l.resolution).collect()
    }

    /// Check if all targets have been set.
    pub fn has_target(&self) -> bool {
        self.matchers.iter().all(|m| m.has_target())
    }

    /// Set the target (map) point cloud.
    ///
    /// Builds voxel grids at all resolution levels.
    pub fn set_target(&mut self, points: &[[f32; 3]]) -> Result<()> {
        if points.is_empty() {
            bail!("Target point cloud is empty");
        }

        for (i, matcher) in self.matchers.iter_mut().enumerate() {
            matcher.set_target(points).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to set target for level {} (resolution {}): {}",
                    i,
                    self.config.levels[i].resolution,
                    e
                )
            })?;
        }

        Ok(())
    }

    /// Align source points to target using cascading multi-resolution.
    ///
    /// # Algorithm
    ///
    /// 1. Start with initial guess
    /// 2. Align using coarsest grid (few iterations)
    /// 3. Use result as initial guess for next finer grid
    /// 4. Repeat until finest grid
    /// 5. Return result from finest grid
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud (sensor scan)
    /// * `initial_guess` - Initial pose estimate
    pub fn align(
        &self,
        source_points: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
    ) -> Result<MultiGridAlignResult> {
        if !self.has_target() {
            bail!("No target set. Call set_target() first.");
        }

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        let mut current_pose = initial_guess;
        let mut iterations_per_level = Vec::with_capacity(self.matchers.len());
        let mut total_iterations = 0;
        let mut max_oscillation = 0;
        let mut last_result: Option<AlignResult> = None;

        // Cascade through levels from coarse to fine
        for matcher in &self.matchers {
            let result = matcher.align(source_points, current_pose)?;

            iterations_per_level.push(result.iterations);
            total_iterations += result.iterations;
            max_oscillation = max_oscillation.max(result.oscillation_count);

            // Use this result's pose as initial guess for next level
            current_pose = result.pose;
            last_result = Some(result);
        }

        let final_result = last_result.expect("At least one level should exist");

        Ok(MultiGridAlignResult {
            pose: final_result.pose,
            converged: final_result.converged,
            score: final_result.score,
            transform_probability: final_result.transform_probability,
            nvtl: final_result.nvtl,
            total_iterations,
            iterations_per_level,
            hessian: final_result.hessian,
            num_correspondences: final_result.num_correspondences,
            oscillation_count: max_oscillation,
            resolutions: self.resolutions(),
        })
    }

    /// Align with early termination if coarse level converges well.
    ///
    /// If the NVTL at a coarse level exceeds the threshold, skip remaining
    /// coarse levels and proceed directly to finer levels.
    ///
    /// # Arguments
    /// * `source_points` - Source point cloud
    /// * `initial_guess` - Initial pose estimate
    /// * `nvtl_skip_threshold` - NVTL threshold for skipping levels (e.g., 2.0)
    pub fn align_adaptive(
        &self,
        source_points: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
        nvtl_skip_threshold: f64,
    ) -> Result<MultiGridAlignResult> {
        if !self.has_target() {
            bail!("No target set. Call set_target() first.");
        }

        if source_points.is_empty() {
            bail!("Source point cloud is empty");
        }

        let mut current_pose = initial_guess;
        let mut iterations_per_level = Vec::with_capacity(self.matchers.len());
        let mut total_iterations = 0;
        let mut max_oscillation = 0;
        let mut last_result: Option<AlignResult> = None;

        let n_levels = self.matchers.len();

        for (level_idx, matcher) in self.matchers.iter().enumerate() {
            let result = matcher.align(source_points, current_pose)?;

            iterations_per_level.push(result.iterations);
            total_iterations += result.iterations;
            max_oscillation = max_oscillation.max(result.oscillation_count);
            current_pose = result.pose;

            // Check for early skip (only for non-finest levels)
            let is_finest = level_idx == n_levels - 1;
            if !is_finest && result.nvtl > nvtl_skip_threshold && result.converged {
                // Good enough at this level, skip to finest
                // Fill in skipped levels with 0 iterations
                let skipped_count = (n_levels - 1) - (level_idx + 1);
                iterations_per_level.extend(std::iter::repeat_n(0, skipped_count));
                // Run finest level
                let finest = &self.matchers[n_levels - 1];
                let final_result = finest.align(source_points, current_pose)?;
                iterations_per_level.push(final_result.iterations);
                total_iterations += final_result.iterations;
                max_oscillation = max_oscillation.max(final_result.oscillation_count);
                last_result = Some(final_result);
                break;
            }

            last_result = Some(result);
        }

        let final_result = last_result.expect("At least one level should exist");

        Ok(MultiGridAlignResult {
            pose: final_result.pose,
            converged: final_result.converged,
            score: final_result.score,
            transform_probability: final_result.transform_probability,
            nvtl: final_result.nvtl,
            total_iterations,
            iterations_per_level,
            hessian: final_result.hessian,
            num_correspondences: final_result.num_correspondences,
            oscillation_count: max_oscillation,
            resolutions: self.resolutions(),
        })
    }

    /// Evaluate NVTL at a given pose using the finest grid.
    pub fn evaluate_nvtl(&self, source_points: &[[f32; 3]], pose: &Isometry3<f64>) -> Result<f64> {
        let finest = self
            .matchers
            .last()
            .ok_or_else(|| anyhow::anyhow!("No matchers available"))?;
        finest.evaluate_nvtl(source_points, pose)
    }

    /// Evaluate transform probability at a given pose using the finest grid.
    pub fn evaluate_transform_probability(
        &self,
        source_points: &[[f32; 3]],
        pose: &Isometry3<f64>,
    ) -> Result<f64> {
        let finest = self
            .matchers
            .last()
            .ok_or_else(|| anyhow::anyhow!("No matchers available"))?;
        finest.evaluate_transform_probability(source_points, pose)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::test_utils::make_default_half_cubic_pcd;
    #[test]
    fn test_multi_grid_config_default() {
        let config = MultiGridConfig::default();
        assert_eq!(config.levels.len(), 3);
        assert_eq!(config.levels[0].resolution, 4.0);
        assert_eq!(config.levels[1].resolution, 2.0);
        assert_eq!(config.levels[2].resolution, 1.0);
    }
    #[test]
    fn test_multi_grid_config_with_resolutions() {
        let config = MultiGridConfig::with_resolutions(&[8.0, 4.0, 2.0, 1.0]);
        assert_eq!(config.levels.len(), 4);
        assert_eq!(config.levels[0].resolution, 8.0);
        assert_eq!(config.levels[0].max_iterations, 5); // Coarsest
        assert_eq!(config.levels[3].resolution, 1.0);
        assert_eq!(config.levels[3].max_iterations, 15); // Finest
    }
    #[test]
    fn test_multi_grid_config_two_level() {
        let config = MultiGridConfig::two_level(4.0, 2.0);
        assert_eq!(config.levels.len(), 2);
        assert_eq!(config.levels[0].resolution, 4.0);
        assert_eq!(config.levels[1].resolution, 2.0);
    }
    #[test]
    fn test_multi_grid_ndt_creation() {
        let ndt = MultiGridNdt::new(&[4.0, 2.0]).unwrap();
        assert_eq!(ndt.num_levels(), 2);
        assert_eq!(ndt.resolutions(), vec![4.0, 2.0]);
        assert!(!ndt.has_target());
    }
    #[test]
    fn test_multi_grid_ndt_builder() {
        let ndt = MultiGridNdt::builder()
            .resolutions(&[4.0, 2.0, 1.0])
            .use_gpu(false)
            .step_size(0.05)
            .build()
            .unwrap();

        assert_eq!(ndt.num_levels(), 3);
    }
    #[test]
    fn test_multi_grid_set_target() {
        let mut ndt = MultiGridNdt::new(&[4.0, 2.0]).unwrap();
        let points = make_default_half_cubic_pcd();

        ndt.set_target(&points).unwrap();
        assert!(ndt.has_target());
    }
    #[test]
    fn test_multi_grid_align_identity() {
        let mut ndt = MultiGridNdt::builder()
            .resolutions(&[4.0, 2.0])
            .use_gpu(false)
            .build()
            .unwrap();

        let points = make_default_half_cubic_pcd();
        ndt.set_target(&points).unwrap();

        let result = ndt.align(&points, Isometry3::identity()).unwrap();

        // Should converge near identity for self-alignment
        assert!(result.total_iterations > 0);
        assert_eq!(result.iterations_per_level.len(), 2);

        let translation = result.pose.translation.vector;
        assert!(
            translation.norm() < 1.0,
            "Translation should be small: {:?}",
            translation
        );
    }
    #[test]
    fn test_multi_grid_align_with_offset() {
        let mut ndt = MultiGridNdt::builder()
            .resolutions(&[4.0, 2.0])
            .use_gpu(false)
            .build()
            .unwrap();

        let points = make_default_half_cubic_pcd();
        ndt.set_target(&points).unwrap();

        // Start with an offset
        let initial = Isometry3::translation(0.5, 0.5, 0.0);
        let result = ndt.align(&points, initial).unwrap();

        // Should recover close to identity
        let translation = result.pose.translation.vector;
        assert!(
            translation.norm() < 1.0,
            "Should recover from offset: {:?}",
            translation
        );
    }
    #[test]
    fn test_multi_grid_result_conversion() {
        let multi_result = MultiGridAlignResult {
            pose: Isometry3::identity(),
            converged: true,
            score: -10.0,
            transform_probability: 0.9,
            nvtl: 2.5,
            total_iterations: 15,
            iterations_per_level: vec![5, 10],
            hessian: Matrix6::identity(),
            num_correspondences: 1000,
            oscillation_count: 0,
            resolutions: vec![4.0, 2.0],
        };

        let align_result: AlignResult = multi_result.into();
        assert_eq!(align_result.iterations, 15);
        assert!(align_result.converged);
    }
    #[test]
    fn test_multi_grid_empty_target_error() {
        let mut ndt = MultiGridNdt::new(&[4.0, 2.0]).unwrap();
        let result = ndt.set_target(&[]);
        assert!(result.is_err());
    }
    #[test]
    fn test_multi_grid_no_target_error() {
        let ndt = MultiGridNdt::new(&[4.0, 2.0]).unwrap();
        let result = ndt.align(&[[0.0, 0.0, 0.0]], Isometry3::identity());
        assert!(result.is_err());
    }
}
