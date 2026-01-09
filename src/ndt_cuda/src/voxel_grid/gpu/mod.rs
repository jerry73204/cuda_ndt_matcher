//! GPU-accelerated voxel grid construction using CubeCL.
//!
//! This module provides GPU-accelerated voxel grid construction:
//! 1. Morton code computation for spatial indexing
//! 2. Radix sort for grouping points by voxel
//! 3. Segment detection to find voxel boundaries
//! 4. Parallel prefix sum (scan) for various reductions
//! 5. Voxel statistics (mean, covariance, inverse covariance)
//! 6. Morton-based radius search for spatial queries
//! 7. Integration into `GpuVoxelGrid` struct
//!
//! # Pipeline
//!
//! ```text
//! Points → Morton Codes → Radix Sort → Segments → Statistics → VoxelGrid
//!                                                      ↓
//!                                              Radius Search Index
//! ```
//!
//! # Current Status
//!
//! - GPU kernels are defined using CubeCL's `#[cube]` macro
//! - CPU reference implementations are provided for testing and fallback
//! - GPU launch infrastructure will be added when integrated with the main pipeline

pub mod autoware_comparison;
pub mod grid;
pub mod morton;
pub mod pipeline;
pub mod radius_search;
pub mod radix_sort;
pub mod scan;
pub mod segments;
pub mod statistics;

// Re-export the main GpuVoxelGrid struct
pub use grid::{GpuVoxelGrid, GpuVoxelGridConfig, ValidVoxel};

// Re-export CPU reference implementations (GPU launchers to be added)
pub use morton::{compute_morton_codes_cpu, morton_decode_3d, MortonCodeResult};
pub use radius_search::{
    radius_search_brute_force_cpu, radius_search_cpu, RadiusSearchConfig, RadiusSearchResult,
};
pub use radix_sort::{
    radix_sort_by_key, radix_sort_by_key_cpu, radix_sort_by_key_gpu, RadixSortResult,
};
pub use scan::{exclusive_scan_cpu, inclusive_scan_cpu};
pub use segments::{
    detect_segments_cpu, detect_segments_with_lengths_cpu, SegmentResult, SegmentResultWithLengths,
};
pub use statistics::{compute_voxel_statistics_cpu, VoxelStatistics, VoxelSums};

// Pipeline for zero-copy GPU execution
pub use pipeline::{GpuPipelineBuffers, PipelineResult};
