//! Scoring and quality metrics for NDT scan matching.
//!
//! This module provides metrics to evaluate alignment quality:
//! - **Transform Probability**: Average NDT score normalized per correspondence
//! - **NVTL (Nearest Voxel Transformation Likelihood)**: Max score per point, averaged
//! - **Per-point scores**: Individual scores for visualization/debugging
//!
//! Based on Autoware's NDT implementation and Magnusson 2009.
//!
//! # GPU Acceleration
//!
//! The `pipeline` module provides GPU-accelerated batch scoring for multiple poses,
//! used by MULTI_NDT_SCORE covariance estimation. See [`GpuScoringPipeline`].

pub mod colors;
pub mod gpu;
pub mod metrics;
pub mod nvtl;
pub mod pipeline;

pub use colors::{
    color_to_rgb_packed, ndt_score_to_color, score_to_color, ColorRGBA, DEFAULT_SCORE_LOWER,
    DEFAULT_SCORE_UPPER,
};
pub use metrics::{compute_per_point_scores, compute_transform_probability, ScoringResult};
pub use nvtl::{compute_nvtl, NvtlConfig, NvtlResult};
pub use pipeline::{BatchScoringResult, GpuScoringPipeline};
