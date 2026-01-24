//! GPU-accelerated point cloud filtering.
//!
//! This module provides GPU kernels for filtering sensor points:
//! - Distance-based filtering (min/max distance from origin)
//! - Z-height filtering (min/max z value)
//! - Voxel grid downsampling
//!
//! # Example
//!
//! ```ignore
//! use ndt_cuda::filtering::{FilterParams, GpuPointFilter};
//!
//! let filter = GpuPointFilter::new()?;
//! let params = FilterParams {
//!     min_distance: 1.0,
//!     max_distance: 100.0,
//!     min_z: -2.0,
//!     max_z: 10.0,
//!     downsample_resolution: Some(0.5),
//! };
//! let filtered = filter.filter(&points, &params)?;
//! ```

mod cpu;
mod kernels;

pub use cpu::{filter_points_cpu, voxel_downsample_cpu};
pub use kernels::{
    compact_points_kernel, compute_filter_mask_kernel, compute_voxel_centroids_kernel,
    prefix_sum_kernel,
};

use anyhow::Result;

/// Parameters for point cloud filtering.
#[derive(Clone, Debug)]
pub struct FilterParams {
    /// Minimum distance from sensor origin
    pub min_distance: f32,
    /// Maximum distance from sensor origin
    pub max_distance: f32,
    /// Minimum z value (ground filtering)
    pub min_z: f32,
    /// Maximum z value (ceiling filtering)
    pub max_z: f32,
    /// Voxel grid downsampling resolution (None = no downsampling)
    pub downsample_resolution: Option<f32>,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            min_distance: 0.0,
            max_distance: f32::MAX,
            min_z: f32::MIN,
            max_z: f32::MAX,
            downsample_resolution: None,
        }
    }
}

/// Result of point filtering operation.
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Filtered points
    pub points: Vec<[f32; 3]>,
    /// Number of points removed by distance filter
    pub removed_by_distance: usize,
    /// Number of points removed by z filter
    pub removed_by_z: usize,
    /// Number of points removed by downsampling
    pub removed_by_downsampling: usize,
    /// Whether GPU was used
    pub used_gpu: bool,
}

/// GPU-accelerated point cloud filter.
#[cfg(feature = "cuda")]
pub struct GpuPointFilter {
    client: cubecl::client::ComputeClient<
        <cubecl::cuda::CudaRuntime as cubecl::prelude::Runtime>::Server,
    >,
}

#[cfg(feature = "cuda")]
impl GpuPointFilter {
    /// Create a new GPU point filter.
    pub fn new() -> Result<Self> {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        use cubecl::prelude::Runtime;

        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        Ok(Self { client })
    }

    /// Filter points using GPU acceleration.
    pub fn filter(&self, points: &[[f32; 3]], params: &FilterParams) -> Result<FilterResult> {
        if points.is_empty() {
            return Ok(FilterResult {
                points: Vec::new(),
                removed_by_distance: 0,
                removed_by_z: 0,
                removed_by_downsampling: 0,
                used_gpu: true,
            });
        }

        // For small point clouds, CPU is faster due to kernel launch overhead
        if points.len() < 10000 {
            let mut result = filter_points_cpu(points, params);
            result.used_gpu = false;
            return Ok(result);
        }

        // GPU filtering pipeline:
        // 1. Compute filter mask (which points pass distance/z filter)
        // 2. Prefix sum to compute output indices
        // 3. Compact valid points to output array
        // 4. Optional: voxel downsampling

        let original_count = points.len();
        let num_points = points.len() as u32;

        // Flatten points for GPU
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();

        // Upload points to GPU
        use cubecl::prelude::*;
        let points_gpu = self.client.create(f32::as_bytes(&points_flat));

        // Create mask buffer (1 = keep, 0 = discard)
        let mask_gpu = self.client.empty(num_points as usize * size_of::<u32>());

        // Compute filter mask
        let min_dist_sq = params.min_distance * params.min_distance;
        let max_dist_sq = params.max_distance * params.max_distance;

        let cube_count = num_points.div_ceil(256);
        unsafe {
            compute_filter_mask_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, points_flat.len(), 1),
                ScalarArg::new(min_dist_sq),
                ScalarArg::new(max_dist_sq),
                ScalarArg::new(params.min_z),
                ScalarArg::new(params.max_z),
                ScalarArg::new(num_points),
                ArrayArg::from_raw_parts::<u32>(&mask_gpu, num_points as usize, 1),
            );
        }

        // Download mask to count valid points and compute prefix sum on CPU
        // (GPU prefix sum is complex; for moderate sizes, CPU is fine)
        let mask_bytes = self.client.read_one(mask_gpu);
        let mask: Vec<u32> = u32::from_bytes(&mask_bytes).to_vec();

        // Count removed points
        let valid_count: usize = mask.iter().map(|&m| m as usize).sum();
        let removed_by_filter = original_count - valid_count;

        // Compute prefix sum (exclusive scan) on CPU
        let mut prefix_sum = Vec::with_capacity(mask.len() + 1);
        prefix_sum.push(0u32);
        let mut sum = 0u32;
        for &m in &mask {
            sum += m;
            prefix_sum.push(sum);
        }

        // Upload prefix sum
        let prefix_gpu = self.client.create(u32::as_bytes(&prefix_sum));

        // Allocate output buffer
        let output_gpu = self.client.empty(valid_count * 3 * size_of::<f32>());

        // Compact points
        unsafe {
            compact_points_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, points_flat.len(), 1),
                ArrayArg::from_raw_parts::<u32>(
                    &self.client.create(u32::as_bytes(&mask)),
                    mask.len(),
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&prefix_gpu, prefix_sum.len(), 1),
                ScalarArg::new(num_points),
                ArrayArg::from_raw_parts::<f32>(&output_gpu, valid_count * 3, 1),
            );
        }

        // Download compacted points
        let output_bytes = self.client.read_one(output_gpu);
        let output_flat: Vec<f32> = f32::from_bytes(&output_bytes).to_vec();

        let mut filtered_points: Vec<[f32; 3]> =
            output_flat.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

        // Estimate how many were removed by each filter (approximate)
        let removed_by_z = points
            .iter()
            .filter(|p| p[2] < params.min_z || p[2] > params.max_z)
            .count();
        let removed_by_distance = removed_by_filter.saturating_sub(removed_by_z);

        // Apply voxel downsampling if requested (on CPU for now, can be GPU later)
        let removed_by_downsampling = if let Some(resolution) = params.downsample_resolution {
            let before = filtered_points.len();
            filtered_points = voxel_downsample_cpu(&filtered_points, resolution);
            before - filtered_points.len()
        } else {
            0
        };

        Ok(FilterResult {
            points: filtered_points,
            removed_by_distance,
            removed_by_z,
            removed_by_downsampling,
            used_gpu: true,
        })
    }
}

/// CPU-only point filter (fallback when CUDA not available).
pub struct CpuPointFilter;

impl CpuPointFilter {
    /// Create a new CPU point filter.
    pub fn new() -> Self {
        Self
    }

    /// Filter points using CPU.
    pub fn filter(&self, points: &[[f32; 3]], params: &FilterParams) -> FilterResult {
        filter_points_cpu(points, params)
    }
}

impl Default for CpuPointFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_filter_params_default() {
        let params = FilterParams::default();
        assert_eq!(params.min_distance, 0.0);
        assert_eq!(params.max_distance, f32::MAX);
        assert!(params.downsample_resolution.is_none());
    }
    #[test]
    fn test_cpu_filter_distance() {
        let points = vec![
            [1.0, 0.0, 0.0],  // distance = 1.0
            [5.0, 0.0, 0.0],  // distance = 5.0
            [10.0, 0.0, 0.0], // distance = 10.0
        ];

        let params = FilterParams {
            min_distance: 3.0,
            max_distance: 8.0,
            ..Default::default()
        };

        let result = filter_points_cpu(&points, &params);
        assert_eq!(result.points.len(), 1);
        assert_eq!(result.points[0], [5.0, 0.0, 0.0]);
    }
    #[test]
    fn test_cpu_filter_z() {
        let points = vec![
            [1.0, 0.0, -5.0], // below min_z
            [1.0, 0.0, 0.0],  // within range
            [1.0, 0.0, 15.0], // above max_z
        ];

        let params = FilterParams {
            min_z: -2.0,
            max_z: 10.0,
            ..Default::default()
        };

        let result = filter_points_cpu(&points, &params);
        assert_eq!(result.points.len(), 1);
    }
    #[test]
    fn test_cpu_voxel_downsample() {
        let points = vec![
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [5.0, 5.0, 5.0], // different voxel
        ];

        let result = voxel_downsample_cpu(&points, 1.0);
        assert_eq!(result.len(), 2); // Two voxels

        // Find the centroid of the first voxel
        let centroid: Vec<_> = result.iter().filter(|p| p[0] < 1.0).collect();
        assert_eq!(centroid.len(), 1);
        let c = centroid[0];
        assert!((c[0] - 0.2).abs() < 0.01);
        assert!((c[1] - 0.2).abs() < 0.01);
        assert!((c[2] - 0.2).abs() < 0.01);
    }
}
