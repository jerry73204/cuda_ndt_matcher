//! GPU Zero-Copy Pipeline for voxel grid construction.
//!
//! This module provides a zero-copy GPU pipeline that keeps data on the GPU
//! between consecutive operations, eliminating unnecessary CPU-GPU transfers.
//!
//! # Architecture
//!
//! ```text
//! Points (CPU)
//!     │ UPLOAD (once)
//!     ▼
//! ┌─────────────────────────────────┐
//! │  GPU Memory (CubeCL-managed)    │
//! │                                 │
//! │  Morton Codes ──► Radix Sort    │
//! │       (CubeCL)    (cuda_ffi)    │
//! │                       │         │
//! │                       ▼         │
//! │              Segment Detect     │
//! │                (cuda_ffi)       │
//! │                       │         │
//! │                       ▼         │
//! │              Statistics         │
//! │                (CubeCL)         │
//! └─────────────────────────────────┘
//!     │ DOWNLOAD (once)
//!     ▼
//! Results (CPU)
//! ```
//!
//! # Interoperability
//!
//! CubeCL buffers expose raw CUDA device pointers via `get_resource()`,
//! which can be passed to cuda_ffi functions for CUB operations.

use anyhow::{Context, Result};
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::morton::compute_morton_codes_kernel;
use super::statistics::{
    accumulate_segment_covariances_kernel, accumulate_segment_sums_kernel, compute_means_kernel,
    finalize_voxels_cpu,
};

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// Pre-allocated GPU buffers for the voxel construction pipeline.
///
/// All buffers are owned by CubeCL's memory manager, ensuring proper
/// lifetime management and enabling raw pointer extraction for cuda_ffi.
pub struct GpuPipelineBuffers {
    client: CudaClient,

    // Capacity tracking
    max_points: usize,
    max_segments: usize,

    // Morton code stage (u64 stored as two u32s for CubeCL compatibility)
    morton_codes_low: Handle,  // [u32; max_points]
    morton_codes_high: Handle, // [u32; max_points]
    point_indices: Handle,     // [u32; max_points]

    // Radix sort output
    sorted_codes: Handle,   // [u64; max_points] (8 bytes per element)
    sorted_indices: Handle, // [u32; max_points]

    // Segment detection scratch/output
    boundaries: Handle,      // [u32; max_points]
    segment_ids: Handle,     // [u32; max_points]
    indices_scratch: Handle, // [u32; max_points]
    segment_starts: Handle,  // [u32; max_points] (max size)
    num_selected: Handle,    // [i32; 1]

    // Statistics buffers
    position_sums: Handle, // [f32; max_segments * 3]
    counts: Handle,        // [u32; max_segments]
    means: Handle,         // [f32; max_segments * 3]
    cov_sums: Handle,      // [f32; max_segments * 9]

    // CUB temporary storage
    sort_temp: Handle,
    sum_temp: Handle,
    select_temp: Handle,

    // Temp storage sizes
    sort_temp_bytes: usize,
    sum_temp_bytes: usize,
    select_temp_bytes: usize,

    // Grid bounds (computed during Morton code stage)
    grid_min: [f32; 3],
    grid_max: [f32; 3],
}

impl GpuPipelineBuffers {
    /// Create pipeline buffers with given capacity.
    ///
    /// # Arguments
    /// * `max_points` - Maximum number of points to support
    /// * `max_segments` - Maximum number of segments (voxels) to support
    pub fn new(max_points: usize, max_segments: usize) -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        // Query CUB temp storage sizes
        let sort_temp_bytes =
            cuda_ffi::radix_sort_temp_size(max_points).context("Failed to query sort temp size")?;
        let (sum_temp_bytes, select_temp_bytes) = cuda_ffi::segment_detect_temp_sizes(max_points)
            .context("Failed to query segment temp sizes")?;

        // Allocate all buffers
        let morton_codes_low = client.empty(max_points * std::mem::size_of::<u32>());
        let morton_codes_high = client.empty(max_points * std::mem::size_of::<u32>());
        let point_indices = client.empty(max_points * std::mem::size_of::<u32>());

        let sorted_codes = client.empty(max_points * std::mem::size_of::<u64>());
        let sorted_indices = client.empty(max_points * std::mem::size_of::<u32>());

        let boundaries = client.empty(max_points * std::mem::size_of::<u32>());
        let segment_ids = client.empty(max_points * std::mem::size_of::<u32>());
        let indices_scratch = client.empty(max_points * std::mem::size_of::<u32>());
        let segment_starts = client.empty(max_points * std::mem::size_of::<u32>());
        let num_selected = client.empty(std::mem::size_of::<i32>());

        let position_sums = client.empty(max_segments * 3 * std::mem::size_of::<f32>());
        let counts = client.empty(max_segments * std::mem::size_of::<u32>());
        let means = client.empty(max_segments * 3 * std::mem::size_of::<f32>());
        let cov_sums = client.empty(max_segments * 9 * std::mem::size_of::<f32>());

        let sort_temp = client.empty(sort_temp_bytes);
        let sum_temp = client.empty(sum_temp_bytes);
        let select_temp = client.empty(select_temp_bytes);

        Ok(Self {
            client,
            max_points,
            max_segments,
            morton_codes_low,
            morton_codes_high,
            point_indices,
            sorted_codes,
            sorted_indices,
            boundaries,
            segment_ids,
            indices_scratch,
            segment_starts,
            num_selected,
            position_sums,
            counts,
            means,
            cov_sums,
            sort_temp,
            sum_temp,
            select_temp,
            sort_temp_bytes,
            sum_temp_bytes,
            select_temp_bytes,
            grid_min: [0.0; 3],
            grid_max: [0.0; 3],
        })
    }

    /// Get the maximum points capacity.
    pub fn max_points(&self) -> usize {
        self.max_points
    }

    /// Get the maximum segments capacity.
    pub fn max_segments(&self) -> usize {
        self.max_segments
    }

    /// Extract raw CUDA device pointer from a CubeCL handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let binding = handle.clone().binding();
        let resource = self.client.get_resource(binding);
        resource.resource().ptr
    }

    /// Compute Morton codes on GPU.
    ///
    /// Uploads points, computes Morton codes, and stores results in
    /// `morton_codes_low`, `morton_codes_high`, and `point_indices`.
    fn compute_morton_codes(&mut self, points_flat: &[f32], resolution: f32) -> Result<usize> {
        let num_points = points_flat.len() / 3;

        if num_points == 0 {
            return Ok(0);
        }

        if num_points > self.max_points {
            anyhow::bail!("Too many points: {} > max {}", num_points, self.max_points);
        }

        // Compute bounds on CPU
        let (grid_min, grid_max) = compute_bounds_flat(points_flat);
        self.grid_min = grid_min;
        self.grid_max = grid_max;
        let inv_resolution = 1.0 / resolution;

        // Upload points
        let points_gpu = self.client.create(f32::as_bytes(points_flat));
        let min_bound_gpu = self.client.create(f32::as_bytes(&grid_min));

        // Launch Morton codes kernel
        let cube_count = num_points.div_ceil(256) as u32;
        unsafe {
            compute_morton_codes_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<f32>(&min_bound_gpu, 3, 1),
                ScalarArg::new(inv_resolution),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<u32>(&self.morton_codes_low, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&self.morton_codes_high, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&self.point_indices, num_points, 1),
            );
        }

        Ok(num_points)
    }

    /// Run radix sort using cuda_ffi in-place API.
    ///
    /// Sorts Morton codes in `sorted_codes` with corresponding indices.
    fn radix_sort_inplace(&self, num_points: usize) -> Result<()> {
        if num_points == 0 {
            return Ok(());
        }

        // We need input and output buffers for sort
        // Input: morton_codes (packed), point_indices
        // Output: sorted_codes, sorted_indices

        // The challenge is that morton codes are in split format (low/high u32)
        // and we need packed u64 for cuda_ffi

        // For now, download, pack, upload, sort, which defeats zero-copy
        // TODO: Add a CubeCL kernel to pack u64 on GPU

        let codes_low_bytes = self.client.read_one(self.morton_codes_low.clone());
        let codes_high_bytes = self.client.read_one(self.morton_codes_high.clone());
        let indices_bytes = self.client.read_one(self.point_indices.clone());

        let codes_low = u32::from_bytes(&codes_low_bytes);
        let codes_high = u32::from_bytes(&codes_high_bytes);
        let indices = u32::from_bytes(&indices_bytes);

        // Pack Morton codes
        let mut packed_codes: Vec<u64> = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let code = (codes_high[i] as u64) << 32 | codes_low[i] as u64;
            packed_codes.push(code);
        }

        // Upload packed codes and indices to temporary input position
        // We'll use boundaries and indices_scratch as input buffers
        let input_codes_bytes: Vec<u8> =
            packed_codes.iter().flat_map(|c| c.to_le_bytes()).collect();
        let input_indices_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();

        // Create temporary input buffers
        let input_codes = self.client.create(&input_codes_bytes);
        let input_indices = self.client.create(&input_indices_bytes);

        // Sync CubeCL before cuda_ffi
        cubecl::future::block_on(self.client.sync());

        // Call cuda_ffi sort
        unsafe {
            cuda_ffi::sort_pairs_inplace(
                self.raw_ptr(&self.sort_temp),
                self.sort_temp_bytes,
                self.raw_ptr(&input_codes),
                self.raw_ptr(&self.sorted_codes),
                self.raw_ptr(&input_indices),
                self.raw_ptr(&self.sorted_indices),
                num_points,
            )
            .context("Radix sort failed")?;
        }

        Ok(())
    }

    /// Run segment detection using cuda_ffi in-place API.
    fn detect_segments_inplace(&self, num_points: usize) -> Result<u32> {
        if num_points == 0 {
            return Ok(0);
        }

        // Sync CubeCL before cuda_ffi
        cubecl::future::block_on(self.client.sync());

        let counts = unsafe {
            cuda_ffi::detect_segments_inplace(
                self.raw_ptr(&self.sorted_codes),
                self.raw_ptr(&self.boundaries),
                self.raw_ptr(&self.segment_ids),
                self.raw_ptr(&self.indices_scratch),
                self.raw_ptr(&self.segment_starts),
                self.raw_ptr(&self.num_selected),
                self.raw_ptr(&self.sum_temp),
                self.sum_temp_bytes,
                self.raw_ptr(&self.select_temp),
                self.select_temp_bytes,
                num_points,
            )
            .context("Segment detection failed")?
        };

        Ok(counts.num_segments)
    }

    /// Build a voxel grid using the zero-copy pipeline.
    ///
    /// This method runs the full pipeline:
    /// 1. Upload points (single CPU→GPU transfer)
    /// 2. Compute Morton codes (GPU)
    /// 3. Radix sort (GPU via cuda_ffi)
    /// 4. Segment detection (GPU via cuda_ffi)
    /// 5. Statistics computation (GPU)
    /// 6. Download results (single GPU→CPU transfer)
    /// 7. CPU eigendecomposition
    pub fn build(
        &mut self,
        points: &[[f32; 3]],
        resolution: f32,
        min_points_per_voxel: usize,
    ) -> Result<PipelineResult> {
        let points_flat: Vec<f32> = points.iter().flat_map(|p| p.iter().copied()).collect();

        // Step 1 & 2: Upload and compute Morton codes
        let num_points = self.compute_morton_codes(&points_flat, resolution)?;

        if num_points == 0 {
            return Ok(PipelineResult::empty());
        }

        // Step 3: Radix sort
        self.radix_sort_inplace(num_points)?;

        // Step 4: Segment detection
        let num_segments = self.detect_segments_inplace(num_points)?;

        if num_segments == 0 {
            return Ok(PipelineResult::empty());
        }

        if num_segments as usize > self.max_segments {
            anyhow::bail!(
                "Too many segments: {} > max {}",
                num_segments,
                self.max_segments
            );
        }

        // Step 5 & 6: Statistics (need points on GPU)
        // Upload points for statistics
        let points_gpu = self.client.create(f32::as_bytes(&points_flat));

        // Build proper segment_starts array: [0, boundary1, boundary2, ...]
        // detect_segments_inplace stores boundary indices (where codes change),
        // but statistics kernels need segment_starts[i] = start of segment i
        let boundary_starts_bytes = self.client.read_one(self.segment_starts.clone());
        let num_boundaries = (num_segments - 1) as usize;
        let boundary_starts: Vec<u32> = boundary_starts_bytes
            .chunks(4)
            .take(num_boundaries)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Prepend 0 for the first segment
        let mut full_segment_starts = Vec::with_capacity(num_segments as usize);
        full_segment_starts.push(0u32);
        full_segment_starts.extend(boundary_starts);

        // Upload the proper segment_starts
        let segment_starts_bytes: Vec<u8> = full_segment_starts
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let segment_starts_gpu = self.client.create(&segment_starts_bytes);

        // Launch statistics kernels
        let cube_count = (num_segments as usize).div_ceil(256) as u32;

        unsafe {
            accumulate_segment_sums_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&self.sorted_indices, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&segment_starts_gpu, num_segments as usize, 1),
                ScalarArg::new(num_segments),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.position_sums, num_segments as usize * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&self.counts, num_segments as usize, 1),
            );

            compute_means_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&self.position_sums, num_segments as usize * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&self.counts, num_segments as usize, 1),
                ScalarArg::new(num_segments),
                ArrayArg::from_raw_parts::<f32>(&self.means, num_segments as usize * 3, 1),
            );

            accumulate_segment_covariances_kernel::launch_unchecked::<f32, CudaRuntime>(
                &self.client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&points_gpu, num_points * 3, 1),
                ArrayArg::from_raw_parts::<u32>(&self.sorted_indices, num_points, 1),
                ArrayArg::from_raw_parts::<u32>(&segment_starts_gpu, num_segments as usize, 1),
                ArrayArg::from_raw_parts::<f32>(&self.means, num_segments as usize * 3, 1),
                ScalarArg::new(num_segments),
                ScalarArg::new(num_points as u32),
                ArrayArg::from_raw_parts::<f32>(&self.cov_sums, num_segments as usize * 9, 1),
            );
        }

        // Step 6: Download results
        let means_bytes = self.client.read_one(self.means.clone());
        let means = f32::from_bytes(&means_bytes).to_vec();

        let cov_sums_bytes = self.client.read_one(self.cov_sums.clone());
        let cov_sums = f32::from_bytes(&cov_sums_bytes).to_vec();

        let counts_bytes = self.client.read_one(self.counts.clone());
        let counts = u32::from_bytes(&counts_bytes).to_vec();

        // Download segment codes for coordinate decoding
        let sorted_codes_bytes = self.client.read_one(self.sorted_codes.clone());
        let sorted_codes: Vec<u64> = sorted_codes_bytes
            .chunks(8)
            .take(num_points)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();

        // Build segment codes (code at each segment start)
        // full_segment_starts already has [0, boundary1, boundary2, ...]
        let segment_codes: Vec<u64> = full_segment_starts
            .iter()
            .map(|&start| sorted_codes[start as usize])
            .collect();

        // Step 7: CPU finalization
        let stats = finalize_voxels_cpu(
            means[..num_segments as usize * 3].to_vec(),
            cov_sums[..num_segments as usize * 9].to_vec(),
            counts[..num_segments as usize].to_vec(),
            min_points_per_voxel as u32,
        );

        Ok(PipelineResult {
            stats,
            segment_codes,
            grid_min: self.grid_min,
            num_segments: num_segments as usize,
        })
    }
}

/// Result of the GPU pipeline.
pub struct PipelineResult {
    /// Finalized voxel statistics.
    pub stats: super::statistics::VoxelStatistics,
    /// Morton code for each segment.
    pub segment_codes: Vec<u64>,
    /// Grid minimum bounds.
    pub grid_min: [f32; 3],
    /// Number of segments.
    pub num_segments: usize,
}

impl PipelineResult {
    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            stats: super::statistics::VoxelStatistics {
                means: Vec::new(),
                covariances: Vec::new(),
                inv_covariances: Vec::new(),
                principal_axes: Vec::new(),
                point_counts: Vec::new(),
                valid: Vec::new(),
                num_voxels: 0,
                min_points: 0,
            },
            segment_codes: Vec::new(),
            grid_min: [0.0; 3],
            num_segments: 0,
        }
    }
}

/// Compute min/max bounds of a flattened point cloud.
fn compute_bounds_flat(points: &[f32]) -> ([f32; 3], [f32; 3]) {
    let num_points = points.len() / 3;
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for i in 0..num_points {
        for j in 0..3 {
            let v = points[i * 3 + j];
            min[j] = min[j].min(v);
            max[j] = max[j].max(v);
        }
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = GpuPipelineBuffers::new(10000, 5000);
        assert!(pipeline.is_ok());
        let p = pipeline.unwrap();
        assert_eq!(p.max_points(), 10000);
        assert_eq!(p.max_segments(), 5000);
    }

    #[test]
    fn test_pipeline_empty_input() {
        let mut pipeline = GpuPipelineBuffers::new(1000, 500).unwrap();
        let result = pipeline.build(&[], 2.0, 5);
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.num_segments, 0);
    }

    #[test]
    fn test_pipeline_small_point_cloud() {
        let mut pipeline = GpuPipelineBuffers::new(1000, 500).unwrap();

        // Create a small point cloud with points in different voxels
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [5.0, 0.0, 0.0],
            [5.1, 0.1, 0.1],
            [10.0, 10.0, 10.0],
        ];

        let result = pipeline.build(&points, 2.0, 1);
        assert!(result.is_ok());
        let r = result.unwrap();
        // With resolution 2.0, we should have a few voxels
        assert!(r.num_segments > 0);
    }
}
