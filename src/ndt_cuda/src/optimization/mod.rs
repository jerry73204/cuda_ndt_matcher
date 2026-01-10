//! Newton-based optimization for NDT scan matching.
//!
//! This module implements the optimization loop for NDT:
//! 1. Transform source points using current pose
//! 2. Compute derivatives (gradient + Hessian)
//! 3. Solve Newton step: Δp = -H⁻¹g
//! 4. Update pose and check convergence
//!
//! Based on Magnusson 2009, Chapter 6.

pub mod debug;
pub mod line_search;
pub mod more_thuente;
pub mod newton;
pub mod oscillation;
pub mod regularization;
pub mod solver;
pub mod types;

pub use debug::{AlignmentDebug, AlignmentTimingDebug, IterationDebug, IterationTimingDebug};
pub use line_search::{LineSearchConfig, LineSearchResult};
pub use more_thuente::{more_thuente_search, MoreThuenteConfig, MoreThuenteResult};
pub use newton::{newton_step, newton_step_regularized};
pub use oscillation::{
    count_oscillation, count_oscillation_from_arrays, OscillationResult,
    DEFAULT_OSCILLATION_THRESHOLD,
};
pub use regularization::{RegularizationConfig, RegularizationTerm};
pub use solver::{NdtOptimizer, OptimizationConfig};
pub use types::{ConvergenceStatus, NdtConfig, NdtResult};
