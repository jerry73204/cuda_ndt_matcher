//! Newton-based optimization for NDT scan matching.
//!
//! This module implements the optimization loop for NDT:
//! 1. Transform source points using current pose
//! 2. Compute derivatives (gradient + Hessian)
//! 3. Solve Newton step: Δp = -H⁻¹g
//! 4. Update pose and check convergence
//!
//! Based on Magnusson 2009, Chapter 6.

pub mod batch_pipeline;
pub mod debug;
pub mod full_gpu_pipeline_v2;
pub mod gpu_initial_pose;
pub mod gpu_newton;
pub mod gpu_pipeline_kernels;
pub mod line_search;
pub mod more_thuente;
pub mod newton;
pub mod oscillation;
pub mod regularization;
pub mod solver;
pub mod types;

pub use debug::{AlignmentDebug, AlignmentTimingDebug, IterationDebug, IterationTimingDebug};
pub use full_gpu_pipeline_v2::{FullGpuOptimizationResultV2, FullGpuPipelineV2, PipelineV2Config};
pub use gpu_initial_pose::{
    BatchedNdtResult, GpuInitialPoseConfig, GpuInitialPosePipeline, PipelineMemoryRequirements,
};
pub use gpu_newton::{GpuNewtonError, GpuNewtonSolver};
pub use gpu_pipeline_kernels::{
    batch_score_gradient_kernel, batch_transform_kernel, check_convergence_kernel,
    compute_transform_from_sincos_kernel, dot_product_6_kernel, generate_candidates_kernel,
    more_thuente_kernel, update_pose_kernel, DEFAULT_NUM_CANDIDATES, MAX_NEIGHBORS,
};
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

// Batch processing pipeline for parallel multi-scan alignment
pub use batch_pipeline::{
    AlignmentRequest, BatchAlignmentResult, BatchGpuPipeline, BatchPipelineConfig,
};
