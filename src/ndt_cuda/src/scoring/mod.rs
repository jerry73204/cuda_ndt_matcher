//! Scoring and quality metrics for NDT scan matching.
//!
//! This module provides metrics to evaluate alignment quality:
//! - **Transform Probability**: Average NDT score normalized per correspondence
//! - **NVTL (Nearest Voxel Transformation Likelihood)**: Max score per point, averaged
//! - **Per-point scores**: Individual scores for visualization/debugging
//!
//! Based on Autoware's NDT implementation and Magnusson 2009.

pub mod metrics;
pub mod nvtl;

pub use metrics::{compute_per_point_scores, compute_transform_probability, ScoringResult};
pub use nvtl::{compute_nvtl, NvtlConfig, NvtlResult};
