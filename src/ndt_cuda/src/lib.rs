//! CubeCL-based CUDA NDT scan matching library.
//!
//! This library provides GPU-accelerated Normal Distributions Transform (NDT)
//! scan matching using CubeCL for CUDA/ROCm/WebGPU backends.
//!
//! # Architecture
//!
//! The NDT algorithm is split into phases:
//! - Phase 1: Voxel grid construction from map point cloud
//! - Phase 2: Derivative computation (gradient + Hessian)
//! - Phase 3: Newton optimization
//! - Phase 4: Scoring (transform probability, NVTL)
//!
//! # Usage
//!
//! ```ignore
//! use ndt_cuda::NdtScanMatcher;
//! use nalgebra::Isometry3;
//!
//! // Create NDT scan matcher
//! let mut matcher = NdtScanMatcher::new(2.0)?;
//!
//! // Set target (map) point cloud
//! let map_points: Vec<[f32; 3]> = load_pcd_file("map.pcd");
//! matcher.set_target(&map_points)?;
//!
//! // Align source scan to target
//! let source_points: Vec<[f32; 3]> = get_lidar_scan();
//! let initial_guess = Isometry3::identity();
//!
//! let result = matcher.align(&source_points, initial_guess)?;
//! println!("Converged: {}, NVTL: {}", result.converged, result.nvtl);
//! ```

/// Print to stdout only when the `test-verbose` feature is enabled.
///
/// Use this macro in tests for debug output that is normally too verbose.
/// Enable with: `cargo test --features test-verbose`
#[macro_export]
macro_rules! test_println {
    ($($arg:tt)*) => {
        #[cfg(feature = "test-verbose")]
        println!($($arg)*);
    };
}

pub mod derivatives;
pub mod filtering;
pub mod multi_grid;
pub mod ndt;
pub mod optimization;
pub mod runtime;
pub mod scoring;
pub mod test_utils;
pub mod timing;
pub mod voxel_grid;

pub use derivatives::{
    AggregatedDerivatives, AngularDerivatives, DerivativeResult, DistanceMetric, GaussianParams,
    PointDerivatives,
};
pub use optimization::{
    AlignmentDebug, AlignmentRequest, BatchAlignmentResult, BatchGpuPipeline, BatchPipelineConfig,
    ConvergenceStatus, IterationDebug, LineSearchConfig, LineSearchResult, NdtConfig, NdtOptimizer,
    NdtResult, OptimizationConfig,
};
pub use scoring::{
    compute_nvtl, compute_per_point_scores, compute_transform_probability, NvtlConfig, NvtlResult,
    ScoringResult,
};
pub use timing::{AlignmentTiming, IterationTiming, Timer, TimingCollector};
pub use voxel_grid::{Voxel, VoxelGrid, VoxelGridConfig};

// GPU-accelerated voxel grid construction
#[cfg(feature = "cuda")]
pub use voxel_grid::GpuVoxelGridBuilder;

// High-level API (recommended for most users)
pub use ndt::{AlignResult, NdtScanMatcher, NdtScanMatcherBuilder, NdtScanMatcherConfig};

// Multi-grid NDT for coarse-to-fine alignment
pub use multi_grid::{
    GridLevelConfig, MultiGridAlignResult, MultiGridConfig, MultiGridNdt, MultiGridNdtBuilder,
};

// GPU runtime (optional, for direct GPU access)
pub use runtime::{is_cuda_available, GpuDerivativeResult, GpuRuntime, GpuScoreResult};

// GPU-accelerated point cloud filtering
#[cfg(feature = "cuda")]
pub use filtering::GpuPointFilter;
pub use filtering::{CpuPointFilter, FilterParams, FilterResult};
