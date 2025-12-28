//! Derivative computation for NDT optimization.
//!
//! This module implements the gradient and Hessian computation described in
//! Magnusson 2009, Chapter 6. The key equations are:
//!
//! - Score function (Eq. 6.9): `p(x) = -d1 * exp(-d2/2 * (x-μ)ᵀΣ⁻¹(x-μ))`
//! - Gradient (Eq. 6.12): `∂p/∂pᵢ = d1*d2 * exp(...) * (x-μ)ᵀΣ⁻¹ * ∂T(x)/∂pᵢ`
//! - Hessian (Eq. 6.13): Second derivatives including point Hessian terms
//!
//! The pose vector p = [tx, ty, tz, roll, pitch, yaw] uses Euler angles (XYZ order).
//!
//! # GPU Acceleration
//!
//! The `gpu` submodule provides CUDA-accelerated derivative computation using CubeCL.
//! Key kernels:
//! - `radius_search_kernel`: Find neighboring voxels for each transformed point
//! - `compute_ndt_score_kernel`: Compute per-point scores
//! - `compute_ndt_gradient_kernel`: Compute per-point gradients

pub mod angular;
pub mod cpu;
pub mod gpu;
pub mod types;

pub use angular::AngularDerivatives;
pub use cpu::compute_derivatives_cpu;
pub use gpu::{
    compute_point_jacobians_cpu, pose_to_transform_matrix, GpuDerivativeResult, GpuDerivatives,
    GpuVoxelData, MAX_NEIGHBORS,
};
pub use types::{AggregatedDerivatives, DerivativeResult, GaussianParams, PointDerivatives};
