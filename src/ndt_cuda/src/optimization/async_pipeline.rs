//! Async Batch Pipeline for double-buffered NDT alignment.
//!
//! This module implements Phase 23.1: Async Streams + Double Buffering.
//! It provides a pipeline that overlaps H2D/D2H transfers with kernel execution
//! using CUDA streams and pinned memory.
//!
//! # Architecture
//!
//! ```text
//! Stream 0: [H2D batch 0]──[Batch Kernel 0]──[D2H batch 0]
//! Stream 1:      [H2D batch 1]──[Batch Kernel 1]──[D2H batch 1]
//!                      ▲ overlap ▲         ▲ overlap ▲
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ndt_cuda::optimization::AsyncBatchPipeline;
//!
//! // Create pipeline (double-buffered)
//! let mut pipeline = AsyncBatchPipeline::new(4, 2000, 10000)?;
//! pipeline.upload_voxel_data(&voxel_data, gauss_d1, gauss_d2, resolution)?;
//!
//! // Submit batch (returns immediately)
//! pipeline.submit(&requests, max_iterations, epsilon)?;
//!
//! // Do other work while GPU computes...
//!
//! // Poll for results (non-blocking)
//! if let Some(results) = pipeline.poll()? {
//!     // Process results
//! }
//!
//! // Or wait for results (blocking)
//! let results = pipeline.wait()?;
//! ```

use anyhow::{Context, Result};
use cuda_ffi::{
    batch_ndt_blocks_per_slot, batch_persistent_ndt_init_barriers_async_raw,
    batch_persistent_ndt_launch_async_raw, AsyncDeviceBuffer, CudaEvent, CudaStream, PinnedBuffer,
};

use crate::derivatives::gpu::GpuVoxelData;

use super::batch_pipeline::{AlignmentRequest, BatchAlignmentResult, BatchPipelineConfig};

/// Buffer set for one side of the double buffer.
struct BufferSet {
    // Host pinned memory (for async transfers)
    h_source_points: PinnedBuffer<f32>,
    h_initial_poses: PinnedBuffer<f32>,
    h_points_per_slot: PinnedBuffer<i32>,
    h_reg_ref_x: PinnedBuffer<f32>,
    h_reg_ref_y: PinnedBuffer<f32>,

    // Host output buffers (pinned)
    h_out_poses: PinnedBuffer<f32>,
    h_out_iterations: PinnedBuffer<i32>,
    h_out_converged: PinnedBuffer<u32>,
    h_out_scores: PinnedBuffer<f32>,
    h_out_hessians: PinnedBuffer<f32>,
    h_out_correspondences: PinnedBuffer<u32>,
    h_out_oscillations: PinnedBuffer<u32>,
    h_out_alpha_sums: PinnedBuffer<f32>,

    // Device buffers
    d_source_points: AsyncDeviceBuffer,
    d_initial_poses: AsyncDeviceBuffer,
    d_points_per_slot: AsyncDeviceBuffer,
    d_reduce_buffers: AsyncDeviceBuffer,
    d_barrier_counters: AsyncDeviceBuffer,
    d_barrier_senses: AsyncDeviceBuffer,
    d_reg_ref_x: AsyncDeviceBuffer,
    d_reg_ref_y: AsyncDeviceBuffer,

    // Device output buffers
    d_out_poses: AsyncDeviceBuffer,
    d_out_iterations: AsyncDeviceBuffer,
    d_out_converged: AsyncDeviceBuffer,
    d_out_scores: AsyncDeviceBuffer,
    d_out_hessians: AsyncDeviceBuffer,
    d_out_correspondences: AsyncDeviceBuffer,
    d_out_oscillations: AsyncDeviceBuffer,
    d_out_alpha_sums: AsyncDeviceBuffer,

    // Stream and event for this buffer set
    stream: CudaStream,
    event: CudaEvent,

    // Tracking
    num_requests: usize,
    max_points: usize,
}

impl BufferSet {
    fn new(num_slots: usize, max_points_per_slot: usize) -> Result<Self> {
        let points_size = num_slots * max_points_per_slot * 3;

        Ok(Self {
            // Host pinned input buffers
            h_source_points: PinnedBuffer::new(points_size)?,
            h_initial_poses: PinnedBuffer::new(num_slots * 6)?,
            h_points_per_slot: PinnedBuffer::new(num_slots)?,
            h_reg_ref_x: PinnedBuffer::new(num_slots)?,
            h_reg_ref_y: PinnedBuffer::new(num_slots)?,

            // Host pinned output buffers
            h_out_poses: PinnedBuffer::new(num_slots * 6)?,
            h_out_iterations: PinnedBuffer::new(num_slots)?,
            h_out_converged: PinnedBuffer::new(num_slots)?,
            h_out_scores: PinnedBuffer::new(num_slots)?,
            h_out_hessians: PinnedBuffer::new(num_slots * 36)?,
            h_out_correspondences: PinnedBuffer::new(num_slots)?,
            h_out_oscillations: PinnedBuffer::new(num_slots)?,
            h_out_alpha_sums: PinnedBuffer::new(num_slots)?,

            // Device input buffers
            d_source_points: AsyncDeviceBuffer::new(points_size * std::mem::size_of::<f32>())?,
            d_initial_poses: AsyncDeviceBuffer::new(num_slots * 6 * std::mem::size_of::<f32>())?,
            d_points_per_slot: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<i32>())?,
            d_reduce_buffers: AsyncDeviceBuffer::new(
                num_slots * cuda_ffi::batch_reduce_buffer_size(),
            )?,
            d_barrier_counters: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<i32>())?,
            d_barrier_senses: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<i32>())?,
            d_reg_ref_x: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<f32>())?,
            d_reg_ref_y: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<f32>())?,

            // Device output buffers
            d_out_poses: AsyncDeviceBuffer::new(num_slots * 6 * std::mem::size_of::<f32>())?,
            d_out_iterations: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<i32>())?,
            d_out_converged: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<u32>())?,
            d_out_scores: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<f32>())?,
            d_out_hessians: AsyncDeviceBuffer::new(num_slots * 36 * std::mem::size_of::<f32>())?,
            d_out_correspondences: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<u32>())?,
            d_out_oscillations: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<u32>())?,
            d_out_alpha_sums: AsyncDeviceBuffer::new(num_slots * std::mem::size_of::<f32>())?,

            // Stream and event
            stream: CudaStream::new_non_blocking()?,
            event: CudaEvent::new_disable_timing()?,

            num_requests: 0,
            max_points: 0,
        })
    }
}

/// Async batch pipeline with double buffering for overlapped execution.
///
/// This pipeline uses two buffer sets and CUDA streams to overlap:
/// - H2D transfer of next batch with kernel execution of current batch
/// - D2H transfer of previous results with kernel execution of current batch
pub struct AsyncBatchPipeline {
    // ========================================================================
    // Double buffer
    // ========================================================================
    buffers: [BufferSet; 2],
    current: usize,         // Index of buffer to use for next submit
    pending: Option<usize>, // Index of buffer with pending results

    // ========================================================================
    // Capacity
    // ========================================================================
    num_slots: usize,
    max_points_per_slot: usize,

    // ========================================================================
    // Shared voxel data (read-only, shared across all batches)
    // ========================================================================
    d_voxel_means: AsyncDeviceBuffer,
    d_voxel_inv_covs: AsyncDeviceBuffer,
    d_hash_table: AsyncDeviceBuffer,
    hash_capacity: u32,
    gauss_d1: f32,
    gauss_d2: f32,
    resolution: f32,
    voxel_data_uploaded: bool,

    // ========================================================================
    // Configuration
    // ========================================================================
    config: BatchPipelineConfig,
    max_iterations: i32,
    epsilon: f32,
}

impl AsyncBatchPipeline {
    /// Create a new async batch pipeline with double buffering.
    ///
    /// # Arguments
    ///
    /// * `num_slots` - Maximum number of parallel alignments per batch
    /// * `max_points_per_slot` - Maximum points per alignment
    /// * `max_voxels` - Maximum voxels in the map
    pub fn new(num_slots: usize, max_points_per_slot: usize, max_voxels: usize) -> Result<Self> {
        Self::with_config(
            num_slots,
            max_points_per_slot,
            max_voxels,
            BatchPipelineConfig::default(),
        )
    }

    /// Create a new async batch pipeline with custom configuration.
    pub fn with_config(
        num_slots: usize,
        max_points_per_slot: usize,
        max_voxels: usize,
        config: BatchPipelineConfig,
    ) -> Result<Self> {
        // Create double buffer
        let buffer0 = BufferSet::new(num_slots, max_points_per_slot)
            .context("Failed to create buffer set 0")?;
        let buffer1 = BufferSet::new(num_slots, max_points_per_slot)
            .context("Failed to create buffer set 1")?;

        // Allocate shared voxel data buffers
        let d_voxel_means = AsyncDeviceBuffer::new(max_voxels * 3 * std::mem::size_of::<f32>())?;
        let d_voxel_inv_covs = AsyncDeviceBuffer::new(max_voxels * 9 * std::mem::size_of::<f32>())?;

        let hash_capacity = cuda_ffi::hash_table_capacity(max_voxels)?;
        let hash_table_bytes = cuda_ffi::hash_table_size(hash_capacity)?;
        let d_hash_table = AsyncDeviceBuffer::new(hash_table_bytes)?;

        Ok(Self {
            buffers: [buffer0, buffer1],
            current: 0,
            pending: None,
            num_slots,
            max_points_per_slot,
            d_voxel_means,
            d_voxel_inv_covs,
            d_hash_table,
            hash_capacity,
            gauss_d1: 0.0,
            gauss_d2: 0.0,
            resolution: 0.0,
            voxel_data_uploaded: false,
            config,
            max_iterations: 30,
            epsilon: 0.01,
        })
    }

    /// Upload shared voxel data to GPU.
    ///
    /// This is called once per map load. The voxel data is shared across
    /// all alignments in all batches.
    pub fn upload_voxel_data(
        &mut self,
        voxel_data: &GpuVoxelData,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
    ) -> Result<()> {
        let num_voxels = voxel_data.num_voxels;

        self.gauss_d1 = gauss_d1;
        self.gauss_d2 = gauss_d2;
        self.resolution = resolution;

        // Create pinned host buffers for upload
        let mut h_means = PinnedBuffer::<f32>::new(voxel_data.means.len())?;
        let mut h_inv_covs = PinnedBuffer::<f32>::new(voxel_data.inv_covariances.len())?;
        let mut h_valid = PinnedBuffer::<u32>::new(voxel_data.valid.len())?;

        h_means.copy_from_slice(&voxel_data.means);
        h_inv_covs.copy_from_slice(&voxel_data.inv_covariances);
        h_valid.copy_from_slice(&voxel_data.valid);

        // Reallocate device buffers if needed
        let means_size = voxel_data.means.len() * std::mem::size_of::<f32>();
        let inv_covs_size = voxel_data.inv_covariances.len() * std::mem::size_of::<f32>();

        self.d_voxel_means = AsyncDeviceBuffer::new(means_size)?;
        self.d_voxel_inv_covs = AsyncDeviceBuffer::new(inv_covs_size)?;

        // Use a stream for uploads
        let stream = CudaStream::new()?;

        // Async uploads
        unsafe {
            stream.memcpy_h2d_async(
                self.d_voxel_means.as_mut_ptr(),
                h_means.as_void_ptr(),
                means_size,
            )?;
            stream.memcpy_h2d_async(
                self.d_voxel_inv_covs.as_mut_ptr(),
                h_inv_covs.as_void_ptr(),
                inv_covs_size,
            )?;
        }

        stream.synchronize()?;

        // Build hash table
        let mut d_valid =
            AsyncDeviceBuffer::new(voxel_data.valid.len() * std::mem::size_of::<u32>())?;
        unsafe {
            stream.memcpy_h2d_async(
                d_valid.as_mut_ptr(),
                h_valid.as_void_ptr(),
                voxel_data.valid.len() * std::mem::size_of::<u32>(),
            )?;
        }
        stream.synchronize()?;

        unsafe {
            cuda_ffi::hash_table_init(self.d_hash_table.as_u64(), self.hash_capacity)?;
            cuda_ffi::hash_table_build(
                self.d_voxel_means.as_u64(),
                d_valid.as_u64(),
                num_voxels,
                self.resolution,
                self.d_hash_table.as_u64(),
                self.hash_capacity,
            )?;
        }

        cuda_ffi::cuda_device_synchronize()?;

        self.voxel_data_uploaded = true;

        Ok(())
    }

    /// Set optimization parameters.
    pub fn set_optimization_params(&mut self, max_iterations: i32, epsilon: f32) {
        self.max_iterations = max_iterations;
        self.epsilon = epsilon;
    }

    /// Submit a batch for async processing.
    ///
    /// This function returns immediately after enqueueing the operations.
    /// Use `poll()` or `wait()` to retrieve results.
    ///
    /// # Arguments
    ///
    /// * `requests` - Alignment requests (up to num_slots)
    ///
    /// # Returns
    ///
    /// Ok(()) if submission successful, error otherwise.
    pub fn submit(&mut self, requests: &[AlignmentRequest<'_>]) -> Result<()> {
        if !self.voxel_data_uploaded {
            anyhow::bail!("Voxel data not uploaded. Call upload_voxel_data() first.");
        }

        let num_requests = requests.len();

        if num_requests == 0 {
            return Ok(());
        }

        if num_requests > self.num_slots {
            anyhow::bail!(
                "Too many requests: {num_requests} > {} slots",
                self.num_slots
            );
        }

        // Wait for previous use of this buffer to complete
        let buf = &mut self.buffers[self.current];
        buf.stream.synchronize()?;

        // Track max points for blocks_per_slot calculation
        let max_points = requests.iter().map(|r| r.points.len()).max().unwrap_or(0);

        if max_points > self.max_points_per_slot {
            anyhow::bail!(
                "Too many points in request: {max_points} > {}",
                self.max_points_per_slot
            );
        }

        buf.num_requests = num_requests;
        buf.max_points = max_points;

        // Stage input data to pinned host memory
        let h_points = buf.h_source_points.as_mut_slice();
        let h_poses = buf.h_initial_poses.as_mut_slice();
        let h_counts = buf.h_points_per_slot.as_mut_slice();
        let h_reg_x = buf.h_reg_ref_x.as_mut_slice();
        let h_reg_y = buf.h_reg_ref_y.as_mut_slice();

        for (slot, req) in requests.iter().enumerate() {
            h_counts[slot] = req.points.len() as i32;

            // Copy points (with padding)
            let base = slot * self.max_points_per_slot * 3;
            for (i, p) in req.points.iter().enumerate() {
                h_points[base + i * 3] = p[0];
                h_points[base + i * 3 + 1] = p[1];
                h_points[base + i * 3 + 2] = p[2];
            }

            // Copy initial pose
            let pose_base = slot * 6;
            for i in 0..6 {
                h_poses[pose_base + i] = req.initial_pose[i] as f32;
            }

            // Copy regularization refs
            h_reg_x[slot] = req.reg_ref_x.unwrap_or(0.0);
            h_reg_y[slot] = req.reg_ref_y.unwrap_or(0.0);
        }

        // Async H2D transfers
        let stream = &buf.stream;
        let points_bytes = num_requests * self.max_points_per_slot * 3 * std::mem::size_of::<f32>();
        let poses_bytes = num_requests * 6 * std::mem::size_of::<f32>();
        let counts_bytes = num_requests * std::mem::size_of::<i32>();
        let reg_bytes = num_requests * std::mem::size_of::<f32>();

        unsafe {
            stream.memcpy_h2d_async(
                buf.d_source_points.as_mut_ptr(),
                buf.h_source_points.as_void_ptr(),
                points_bytes,
            )?;
            stream.memcpy_h2d_async(
                buf.d_initial_poses.as_mut_ptr(),
                buf.h_initial_poses.as_void_ptr(),
                poses_bytes,
            )?;
            stream.memcpy_h2d_async(
                buf.d_points_per_slot.as_mut_ptr(),
                buf.h_points_per_slot.as_void_ptr(),
                counts_bytes,
            )?;
            stream.memcpy_h2d_async(
                buf.d_reg_ref_x.as_mut_ptr(),
                buf.h_reg_ref_x.as_void_ptr(),
                reg_bytes,
            )?;
            stream.memcpy_h2d_async(
                buf.d_reg_ref_y.as_mut_ptr(),
                buf.h_reg_ref_y.as_void_ptr(),
                reg_bytes,
            )?;

            // Clear reduce buffers
            stream.memset_async(
                buf.d_reduce_buffers.as_mut_ptr(),
                0,
                num_requests * cuda_ffi::batch_reduce_buffer_size(),
            )?;

            // Initialize barriers
            batch_persistent_ndt_init_barriers_async_raw(
                buf.d_barrier_counters.as_u64(),
                buf.d_barrier_senses.as_u64(),
                num_requests,
                stream.as_raw(),
            )?;
        }

        // Launch kernel
        let blocks_per_slot = batch_ndt_blocks_per_slot(max_points);

        unsafe {
            batch_persistent_ndt_launch_async_raw(
                self.d_voxel_means.as_u64(),
                self.d_voxel_inv_covs.as_u64(),
                self.d_hash_table.as_u64(),
                self.hash_capacity,
                self.gauss_d1,
                self.gauss_d2,
                self.resolution,
                buf.d_source_points.as_u64(),
                buf.d_initial_poses.as_u64(),
                buf.d_points_per_slot.as_u64(),
                buf.d_reduce_buffers.as_u64(),
                buf.d_barrier_counters.as_u64(),
                buf.d_barrier_senses.as_u64(),
                buf.d_out_poses.as_u64(),
                buf.d_out_iterations.as_u64(),
                buf.d_out_converged.as_u64(),
                buf.d_out_scores.as_u64(),
                buf.d_out_hessians.as_u64(),
                buf.d_out_correspondences.as_u64(),
                buf.d_out_oscillations.as_u64(),
                buf.d_out_alpha_sums.as_u64(),
                num_requests,
                blocks_per_slot,
                self.max_points_per_slot,
                self.max_iterations,
                self.epsilon,
                self.config.use_line_search,
                self.config.num_candidates as i32,
                self.config.armijo_mu,
                self.config.wolfe_nu,
                self.config.fixed_step_size,
                if self.config.regularization_enabled {
                    buf.d_reg_ref_x.as_u64()
                } else {
                    0
                },
                if self.config.regularization_enabled {
                    buf.d_reg_ref_y.as_u64()
                } else {
                    0
                },
                self.config.regularization_scale_factor,
                self.config.regularization_enabled,
                stream.as_raw(),
            )?;
        }

        // Async D2H transfers for outputs
        let out_poses_bytes = num_requests * 6 * std::mem::size_of::<f32>();
        let out_iter_bytes = num_requests * std::mem::size_of::<i32>();
        let out_u32_bytes = num_requests * std::mem::size_of::<u32>();
        let out_f32_bytes = num_requests * std::mem::size_of::<f32>();
        let out_hess_bytes = num_requests * 36 * std::mem::size_of::<f32>();

        unsafe {
            stream.memcpy_d2h_async(
                buf.h_out_poses.as_mut_void_ptr(),
                buf.d_out_poses.as_ptr(),
                out_poses_bytes,
            )?;
            stream.memcpy_d2h_async(
                buf.h_out_iterations.as_mut_void_ptr(),
                buf.d_out_iterations.as_ptr(),
                out_iter_bytes,
            )?;
            stream.memcpy_d2h_async(
                buf.h_out_converged.as_mut_void_ptr(),
                buf.d_out_converged.as_ptr(),
                out_u32_bytes,
            )?;
            stream.memcpy_d2h_async(
                buf.h_out_scores.as_mut_void_ptr(),
                buf.d_out_scores.as_ptr(),
                out_f32_bytes,
            )?;
            stream.memcpy_d2h_async(
                buf.h_out_hessians.as_mut_void_ptr(),
                buf.d_out_hessians.as_ptr(),
                out_hess_bytes,
            )?;
            stream.memcpy_d2h_async(
                buf.h_out_correspondences.as_mut_void_ptr(),
                buf.d_out_correspondences.as_ptr(),
                out_u32_bytes,
            )?;
            stream.memcpy_d2h_async(
                buf.h_out_oscillations.as_mut_void_ptr(),
                buf.d_out_oscillations.as_ptr(),
                out_u32_bytes,
            )?;
            stream.memcpy_d2h_async(
                buf.h_out_alpha_sums.as_mut_void_ptr(),
                buf.d_out_alpha_sums.as_ptr(),
                out_f32_bytes,
            )?;
        }

        // Record completion event
        buf.event.record(stream)?;

        // Update state
        self.pending = Some(self.current);
        self.current = 1 - self.current; // Swap buffers

        Ok(())
    }

    /// Poll for results (non-blocking).
    ///
    /// Returns `Some(results)` if a batch has completed, `None` if still running.
    pub fn poll(&mut self) -> Result<Option<Vec<BatchAlignmentResult>>> {
        let pending_idx = match self.pending {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let buf = &self.buffers[pending_idx];

        // Non-blocking check
        if !buf.event.is_complete() {
            return Ok(None);
        }

        // Results are ready, extract them
        let results = self.extract_results(pending_idx)?;
        self.pending = None;

        Ok(Some(results))
    }

    /// Wait for results (blocking).
    ///
    /// Returns results from the most recently submitted batch.
    pub fn wait(&mut self) -> Result<Option<Vec<BatchAlignmentResult>>> {
        let pending_idx = match self.pending {
            Some(idx) => idx,
            None => return Ok(None),
        };

        // Block until complete
        self.buffers[pending_idx].event.synchronize()?;

        // Extract results
        let results = self.extract_results(pending_idx)?;
        self.pending = None;

        Ok(Some(results))
    }

    /// Check if there are pending results.
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Get the number of slots in this pipeline.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Get the maximum points per slot.
    pub fn max_points_per_slot(&self) -> usize {
        self.max_points_per_slot
    }

    /// Extract results from a completed buffer set.
    fn extract_results(&self, buf_idx: usize) -> Result<Vec<BatchAlignmentResult>> {
        let buf = &self.buffers[buf_idx];
        let num_requests = buf.num_requests;

        if num_requests == 0 {
            return Ok(vec![]);
        }

        let poses = buf.h_out_poses.as_slice();
        let iterations = buf.h_out_iterations.as_slice();
        let converged = buf.h_out_converged.as_slice();
        let scores = buf.h_out_scores.as_slice();
        let hessians = buf.h_out_hessians.as_slice();
        let correspondences = buf.h_out_correspondences.as_slice();
        let oscillations = buf.h_out_oscillations.as_slice();
        let alpha_sums = buf.h_out_alpha_sums.as_slice();

        let mut results = Vec::with_capacity(num_requests);

        for slot in 0..num_requests {
            let pose: [f64; 6] = std::array::from_fn(|i| poses[slot * 6 + i] as f64);

            let mut hessian = [[0.0f64; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    hessian[i][j] = hessians[slot * 36 + i * 6 + j] as f64;
                }
            }

            results.push(BatchAlignmentResult {
                pose,
                score: scores[slot] as f64,
                converged: converged[slot] != 0,
                iterations: iterations[slot] as u32,
                hessian,
                num_correspondences: correspondences[slot] as usize,
                oscillation_count: oscillations[slot] as usize,
                alpha_sum: alpha_sums[slot] as f64,
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_async_pipeline_creation() {
        let pipeline = AsyncBatchPipeline::new(4, 2000, 10000);
        assert!(pipeline.is_ok());

        let pipeline = pipeline.unwrap();
        assert_eq!(pipeline.num_slots(), 4);
        assert_eq!(pipeline.max_points_per_slot(), 2000);
        assert!(!pipeline.has_pending());
    }
    #[test]
    fn test_async_pipeline_empty_batch() {
        let mut pipeline = AsyncBatchPipeline::new(4, 2000, 10000).unwrap();

        // Upload minimal voxel data
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0],
            inv_covariances: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            principal_axes: vec![0.0, 0.0, 1.0],
            valid: vec![1],
            num_voxels: 1,
        };
        pipeline
            .upload_voxel_data(&voxel_data, 0.55, 0.4, 2.0)
            .unwrap();

        // Submit empty batch
        pipeline.submit(&[]).unwrap();

        // No pending since empty
        assert!(!pipeline.has_pending());
    }
    #[test]
    fn test_async_pipeline_single_alignment() {
        let mut pipeline = AsyncBatchPipeline::new(4, 2000, 10000).unwrap();

        // Create voxels
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
            principal_axes: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            valid: vec![1, 1, 1],
            num_voxels: 3,
        };
        pipeline
            .upload_voxel_data(&voxel_data, 0.55, 0.4, 2.0)
            .unwrap();
        pipeline.set_optimization_params(30, 0.01);

        // Create source points
        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];

        let requests = vec![AlignmentRequest {
            points: &source_points,
            initial_pose: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            reg_ref_x: None,
            reg_ref_y: None,
        }];

        // Submit
        pipeline.submit(&requests).unwrap();
        assert!(pipeline.has_pending());

        // Wait for results
        let results = pipeline.wait().unwrap();
        assert!(results.is_some());

        let results = results.unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
        crate::test_println!(
            "Async single alignment: {} iterations, converged={}, score={:.4}",
            result.iterations,
            result.converged,
            result.score
        );
        assert!(result.iterations > 0);
    }
    #[test]
    fn test_async_pipeline_double_buffer() {
        let mut pipeline = AsyncBatchPipeline::new(4, 2000, 10000).unwrap();

        // Create voxels
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            inv_covariances: vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
            principal_axes: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            valid: vec![1, 1, 1],
            num_voxels: 3,
        };
        pipeline
            .upload_voxel_data(&voxel_data, 0.55, 0.4, 2.0)
            .unwrap();
        pipeline.set_optimization_params(30, 0.01);

        let source_points = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];

        // Submit batch 0
        let requests0 = vec![AlignmentRequest {
            points: &source_points,
            initial_pose: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            reg_ref_x: None,
            reg_ref_y: None,
        }];
        pipeline.submit(&requests0).unwrap();

        // Wait for batch 0
        let results0 = pipeline.wait().unwrap().unwrap();
        assert_eq!(results0.len(), 1);

        // Submit batch 1 (uses different buffer)
        let requests1 = vec![AlignmentRequest {
            points: &source_points,
            initial_pose: [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            reg_ref_x: None,
            reg_ref_y: None,
        }];
        pipeline.submit(&requests1).unwrap();

        // Wait for batch 1
        let results1 = pipeline.wait().unwrap().unwrap();
        assert_eq!(results1.len(), 1);

        crate::test_println!(
            "Double buffer test: batch0={} iters, batch1={} iters",
            results0[0].iterations,
            results1[0].iterations
        );
    }
    #[test]
    fn test_async_pipeline_poll() {
        let mut pipeline = AsyncBatchPipeline::new(4, 2000, 10000).unwrap();

        // Create voxels
        let voxel_data = GpuVoxelData {
            means: vec![0.0, 0.0, 0.0],
            inv_covariances: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            principal_axes: vec![0.0, 0.0, 1.0],
            valid: vec![1],
            num_voxels: 1,
        };
        pipeline
            .upload_voxel_data(&voxel_data, 0.55, 0.4, 2.0)
            .unwrap();
        pipeline.set_optimization_params(30, 0.01);

        let source_points = vec![[0.0f32, 0.0, 0.0]];

        let requests = vec![AlignmentRequest {
            points: &source_points,
            initial_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            reg_ref_x: None,
            reg_ref_y: None,
        }];

        // Submit
        pipeline.submit(&requests).unwrap();

        // Poll until complete
        let mut poll_count = 0;
        loop {
            match pipeline.poll().unwrap() {
                Some(results) => {
                    assert_eq!(results.len(), 1);
                    crate::test_println!("Poll completed after {} polls", poll_count);
                    break;
                }
                None => {
                    poll_count += 1;
                    if poll_count > 10000 {
                        panic!("Poll timeout");
                    }
                    std::hint::spin_loop();
                }
            }
        }
    }
}
