//! Batch GPU Pipeline for parallel NDT alignments.
//!
//! This module implements Phase 22: Batch Multi-Alignment with Non-Cooperative Kernel.
//! It processes M alignments in parallel by partitioning GPU blocks into independent
//! slots, each running a complete Newton optimization using atomic barriers.
//!
//! # Architecture
//!
//! ```text
//! GPU with 80 SMs (example with M=4 slots):
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Slot 0: Blocks 0-19   (20 blocks, handles alignment 0)     │
//! │  Slot 1: Blocks 20-39  (20 blocks, handles alignment 1)     │
//! │  Slot 2: Blocks 40-59  (20 blocks, handles alignment 2)     │
//! │  Slot 3: Blocks 60-79  (20 blocks, handles alignment 3)     │
//! └─────────────────────────────────────────────────────────────┘
//!
//! Memory Layout:
//! Shared (read-only):
//! ├── voxel_means[V × 3]
//! ├── voxel_inv_covs[V × 9]
//! └── hash_table[capacity]
//!
//! Per-Slot (M copies):
//! ├── source_points[M][max_points × 3]
//! ├── reduce_buffer[M][160]
//! ├── barrier_state[M][2]
//! ├── initial_pose[M][6]
//! └── outputs[M]
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ndt_cuda::optimization::BatchGpuPipeline;
//!
//! // Create pipeline for 4 slots, 2000 points max per slot
//! let mut pipeline = BatchGpuPipeline::new(4, 2000, 10000)?;
//!
//! // Upload shared voxel data (once per map load)
//! pipeline.upload_voxel_data(&voxel_data, gauss_d1, gauss_d2, resolution)?;
//!
//! // Submit batch of alignments
//! let results = pipeline.align_batch(&[
//!     AlignmentRequest { points: &scan0, initial_pose: pose0 },
//!     AlignmentRequest { points: &scan1, initial_pose: pose1 },
//!     AlignmentRequest { points: &scan2, initial_pose: pose2 },
//!     AlignmentRequest { points: &scan3, initial_pose: pose3 },
//! ], max_iterations, epsilon)?;
//! ```

use anyhow::Result;
use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::derivatives::gpu::GpuVoxelData;

/// Type alias for CUDA compute client.
type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

/// Result of a single alignment within a batch.
#[derive(Debug, Clone)]
pub struct BatchAlignmentResult {
    /// Final pose [tx, ty, tz, roll, pitch, yaw]
    pub pose: [f64; 6],
    /// Final NDT score (negative log-likelihood)
    pub score: f64,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: u32,
    /// Final Hessian matrix (6x6, row-major)
    pub hessian: [[f64; 6]; 6],
    /// Number of point-voxel correspondences
    pub num_correspondences: usize,
    /// Maximum consecutive oscillation count
    pub oscillation_count: usize,
    /// Accumulated step sizes
    pub alpha_sum: f64,
}

/// Configuration for the batch pipeline.
#[derive(Debug, Clone)]
pub struct BatchPipelineConfig {
    /// Number of line search candidates (K)
    pub num_candidates: u32,
    /// Whether to enable line search
    pub use_line_search: bool,
    /// Armijo constant for line search (mu)
    pub armijo_mu: f32,
    /// Curvature constant for Strong Wolfe condition (nu)
    pub wolfe_nu: f32,
    /// Fixed step size when line search is disabled
    pub fixed_step_size: f32,
    /// Whether GNSS regularization is enabled
    pub regularization_enabled: bool,
    /// Scale factor for regularization term
    pub regularization_scale_factor: f32,
}

impl Default for BatchPipelineConfig {
    fn default() -> Self {
        Self {
            num_candidates: 8,
            use_line_search: false,
            armijo_mu: 1e-4,
            wolfe_nu: 0.9,
            fixed_step_size: 0.1,
            regularization_enabled: false,
            regularization_scale_factor: 0.01,
        }
    }
}

/// Request for a single alignment within a batch.
#[derive(Debug, Clone)]
pub struct AlignmentRequest<'a> {
    /// Source points for this alignment
    pub points: &'a [[f32; 3]],
    /// Initial pose estimate [tx, ty, tz, roll, pitch, yaw]
    pub initial_pose: [f64; 6],
    /// Regularization reference X (from GNSS), if any
    pub reg_ref_x: Option<f32>,
    /// Regularization reference Y (from GNSS), if any
    pub reg_ref_y: Option<f32>,
}

/// Batch GPU Pipeline for parallel NDT alignments.
///
/// Processes M alignments in parallel using a single kernel launch with
/// atomic barriers for per-slot synchronization.
pub struct BatchGpuPipeline {
    client: CudaClient,
    #[allow(dead_code)]
    device: CudaDevice,

    // ========================================================================
    // Capacity
    // ========================================================================
    num_slots: usize,
    max_points_per_slot: usize,
    max_voxels: usize,

    // ========================================================================
    // Shared voxel data (uploaded once per map)
    // ========================================================================
    voxel_means: Handle,    // [V × 3]
    voxel_inv_covs: Handle, // [V × 9]
    hash_table: Handle,     // [capacity × 16]
    hash_capacity: u32,
    num_voxels: usize,
    resolution: f32,
    gauss_d1: f32,
    gauss_d2: f32,

    // ========================================================================
    // Per-slot buffers
    // ========================================================================
    /// Source points for all slots [num_slots × max_points_per_slot × 3]
    all_source_points: Handle,
    /// Initial poses for all slots [num_slots × 6]
    all_initial_poses: Handle,
    /// Point counts per slot [num_slots]
    points_per_slot: Handle,
    /// Reduce buffers for all slots [num_slots × 160]
    all_reduce_buffers: Handle,
    /// Barrier counters [num_slots]
    barrier_counters: Handle,
    /// Barrier senses [num_slots]
    barrier_senses: Handle,
    /// Regularization X refs [num_slots]
    reg_ref_x: Handle,
    /// Regularization Y refs [num_slots]
    reg_ref_y: Handle,

    // ========================================================================
    // Per-slot outputs
    // ========================================================================
    /// Output poses [num_slots × 6]
    all_out_poses: Handle,
    /// Output iteration counts [num_slots]
    all_out_iterations: Handle,
    /// Output convergence flags [num_slots]
    all_out_converged: Handle,
    /// Output scores [num_slots]
    all_out_scores: Handle,
    /// Output Hessians [num_slots × 36]
    all_out_hessians: Handle,
    /// Output correspondence counts [num_slots]
    all_out_correspondences: Handle,
    /// Output oscillation counts [num_slots]
    all_out_oscillations: Handle,
    /// Output alpha sums [num_slots]
    all_out_alpha_sums: Handle,

    // ========================================================================
    // Configuration
    // ========================================================================
    config: BatchPipelineConfig,
}

impl BatchGpuPipeline {
    /// Get raw CUDA device pointer from CubeCL handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let binding = handle.clone().binding();
        let resource = self.client.get_resource(binding);
        resource.resource().ptr
    }

    /// Create a new batch pipeline.
    ///
    /// # Arguments
    ///
    /// * `num_slots` - Number of parallel alignment slots (M)
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

    /// Create a new batch pipeline with custom configuration.
    pub fn with_config(
        num_slots: usize,
        max_points_per_slot: usize,
        max_voxels: usize,
        config: BatchPipelineConfig,
    ) -> Result<Self> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        // Sync device before allocating
        cubecl::future::block_on(client.sync());

        // ====================================================================
        // Shared voxel data
        // ====================================================================
        let voxel_means = client.empty(max_voxels * 3 * std::mem::size_of::<f32>());
        let voxel_inv_covs = client.empty(max_voxels * 9 * std::mem::size_of::<f32>());

        let hash_capacity = cuda_ffi::hash_table_capacity(max_voxels)?;
        let hash_table_bytes = cuda_ffi::hash_table_size(hash_capacity)?;
        let hash_table = client.empty(hash_table_bytes);

        // ====================================================================
        // Per-slot buffers
        // ====================================================================
        let all_source_points =
            client.empty(num_slots * max_points_per_slot * 3 * std::mem::size_of::<f32>());
        let all_initial_poses = client.empty(num_slots * 6 * std::mem::size_of::<f32>());
        let points_per_slot = client.empty(num_slots * std::mem::size_of::<i32>());
        let all_reduce_buffers = client.empty(num_slots * cuda_ffi::batch_reduce_buffer_size());
        let barrier_counters = client.empty(num_slots * std::mem::size_of::<i32>());
        let barrier_senses = client.empty(num_slots * std::mem::size_of::<i32>());
        let reg_ref_x = client.empty(num_slots * std::mem::size_of::<f32>());
        let reg_ref_y = client.empty(num_slots * std::mem::size_of::<f32>());

        // ====================================================================
        // Per-slot outputs
        // ====================================================================
        let all_out_poses = client.empty(num_slots * 6 * std::mem::size_of::<f32>());
        let all_out_iterations = client.empty(num_slots * std::mem::size_of::<i32>());
        let all_out_converged = client.empty(num_slots * std::mem::size_of::<u32>());
        let all_out_scores = client.empty(num_slots * std::mem::size_of::<f32>());
        let all_out_hessians = client.empty(num_slots * 36 * std::mem::size_of::<f32>());
        let all_out_correspondences = client.empty(num_slots * std::mem::size_of::<u32>());
        let all_out_oscillations = client.empty(num_slots * std::mem::size_of::<u32>());
        let all_out_alpha_sums = client.empty(num_slots * std::mem::size_of::<f32>());

        Ok(Self {
            client,
            device,
            num_slots,
            max_points_per_slot,
            max_voxels,
            voxel_means,
            voxel_inv_covs,
            hash_table,
            hash_capacity,
            num_voxels: 0,
            resolution: 0.0,
            gauss_d1: 0.0,
            gauss_d2: 0.0,
            all_source_points,
            all_initial_poses,
            points_per_slot,
            all_reduce_buffers,
            barrier_counters,
            barrier_senses,
            reg_ref_x,
            reg_ref_y,
            all_out_poses,
            all_out_iterations,
            all_out_converged,
            all_out_scores,
            all_out_hessians,
            all_out_correspondences,
            all_out_oscillations,
            all_out_alpha_sums,
            config,
        })
    }

    /// Upload shared voxel data to GPU.
    ///
    /// This is called once per map load. The voxel data is shared across
    /// all alignments in the batch.
    pub fn upload_voxel_data(
        &mut self,
        voxel_data: &GpuVoxelData,
        gauss_d1: f32,
        gauss_d2: f32,
        resolution: f32,
    ) -> Result<()> {
        let num_voxels = voxel_data.num_voxels;

        if num_voxels > self.max_voxels {
            anyhow::bail!("Too many voxels: {num_voxels} > {}", self.max_voxels);
        }

        self.num_voxels = num_voxels;
        self.resolution = resolution;
        self.gauss_d1 = gauss_d1;
        self.gauss_d2 = gauss_d2;

        // Upload voxel data
        self.voxel_means = self.client.create(f32::as_bytes(&voxel_data.means));
        self.voxel_inv_covs = self
            .client
            .create(f32::as_bytes(&voxel_data.inv_covariances));

        // Sync before FFI calls
        cubecl::future::block_on(self.client.sync());

        // Build hash table
        let voxel_means_ptr = self.raw_ptr(&self.voxel_means);
        let voxel_valid_ptr = {
            let voxel_valid = self.client.create(u32::as_bytes(&voxel_data.valid));
            self.raw_ptr(&voxel_valid)
        };
        let hash_table_ptr = self.raw_ptr(&self.hash_table);

        unsafe {
            cuda_ffi::hash_table_init(hash_table_ptr, self.hash_capacity)?;
            cuda_ffi::hash_table_build(
                voxel_means_ptr,
                voxel_valid_ptr,
                num_voxels,
                self.resolution,
                hash_table_ptr,
                self.hash_capacity,
            )?;
        }

        cuda_ffi::cuda_device_synchronize()?;

        Ok(())
    }

    /// Align a batch of scans in parallel.
    ///
    /// # Arguments
    ///
    /// * `requests` - Slice of alignment requests (up to num_slots)
    /// * `max_iterations` - Maximum Newton iterations per alignment
    /// * `epsilon` - Convergence threshold for pose delta norm
    ///
    /// # Returns
    ///
    /// Vector of alignment results (one per request)
    pub fn align_batch(
        &mut self,
        requests: &[AlignmentRequest<'_>],
        max_iterations: u32,
        epsilon: f64,
    ) -> Result<Vec<BatchAlignmentResult>> {
        let num_requests = requests.len();

        if num_requests == 0 {
            return Ok(vec![]);
        }

        if num_requests > self.num_slots {
            anyhow::bail!(
                "Too many requests: {num_requests} > {} slots",
                self.num_slots
            );
        }

        // Find max points across all requests (for blocks_per_slot calculation)
        let max_points = requests.iter().map(|r| r.points.len()).max().unwrap_or(0);

        if max_points > self.max_points_per_slot {
            anyhow::bail!(
                "Too many points in request: {max_points} > {}",
                self.max_points_per_slot
            );
        }

        // ====================================================================
        // Prepare input buffers
        // ====================================================================

        // Source points: flatten and pad to max_points_per_slot
        let mut all_points_flat = vec![0.0f32; num_requests * self.max_points_per_slot * 3];
        let mut point_counts = vec![0i32; num_requests];
        let mut initial_poses_flat = vec![0.0f32; num_requests * 6];
        let mut reg_x_flat = vec![0.0f32; num_requests];
        let mut reg_y_flat = vec![0.0f32; num_requests];

        for (slot, req) in requests.iter().enumerate() {
            point_counts[slot] = req.points.len() as i32;

            // Copy points
            let base = slot * self.max_points_per_slot * 3;
            for (i, p) in req.points.iter().enumerate() {
                all_points_flat[base + i * 3] = p[0];
                all_points_flat[base + i * 3 + 1] = p[1];
                all_points_flat[base + i * 3 + 2] = p[2];
            }

            // Copy initial pose
            let pose_base = slot * 6;
            for i in 0..6 {
                initial_poses_flat[pose_base + i] = req.initial_pose[i] as f32;
            }

            // Copy regularization refs
            reg_x_flat[slot] = req.reg_ref_x.unwrap_or(0.0);
            reg_y_flat[slot] = req.reg_ref_y.unwrap_or(0.0);
        }

        // Upload to GPU
        self.all_source_points = self.client.create(f32::as_bytes(&all_points_flat));
        self.all_initial_poses = self.client.create(f32::as_bytes(&initial_poses_flat));
        self.points_per_slot = self.client.create(i32::as_bytes(&point_counts));
        self.reg_ref_x = self.client.create(f32::as_bytes(&reg_x_flat));
        self.reg_ref_y = self.client.create(f32::as_bytes(&reg_y_flat));

        // Clear reduce buffers
        let reduce_zeros = vec![0.0f32; num_requests * 160];
        self.all_reduce_buffers = self.client.create(f32::as_bytes(&reduce_zeros));

        // Clear output buffers
        let zeros_pose = vec![0.0f32; num_requests * 6];
        let zeros_iter = vec![0i32; num_requests];
        let zeros_u32 = vec![0u32; num_requests];
        let zeros_f32 = vec![0.0f32; num_requests];
        let zeros_hess = vec![0.0f32; num_requests * 36];

        self.all_out_poses = self.client.create(f32::as_bytes(&zeros_pose));
        self.all_out_iterations = self.client.create(i32::as_bytes(&zeros_iter));
        self.all_out_converged = self.client.create(u32::as_bytes(&zeros_u32));
        self.all_out_scores = self.client.create(f32::as_bytes(&zeros_f32));
        self.all_out_hessians = self.client.create(f32::as_bytes(&zeros_hess));
        self.all_out_correspondences = self.client.create(u32::as_bytes(&zeros_u32));
        self.all_out_oscillations = self.client.create(u32::as_bytes(&zeros_u32));
        self.all_out_alpha_sums = self.client.create(f32::as_bytes(&zeros_f32));

        // Sync before kernel launch
        cubecl::future::block_on(self.client.sync());

        // Initialize barriers
        unsafe {
            cuda_ffi::batch_persistent_ndt_init_barriers_raw(
                self.raw_ptr(&self.barrier_counters),
                self.raw_ptr(&self.barrier_senses),
                num_requests,
            )?;
        }

        cuda_ffi::cuda_device_synchronize()?;

        // ====================================================================
        // Calculate grid configuration
        // ====================================================================

        let blocks_per_slot = cuda_ffi::batch_ndt_blocks_per_slot(max_points);

        // ====================================================================
        // Launch kernel
        // ====================================================================

        unsafe {
            cuda_ffi::batch_persistent_ndt_launch_raw(
                self.raw_ptr(&self.voxel_means),
                self.raw_ptr(&self.voxel_inv_covs),
                self.raw_ptr(&self.hash_table),
                self.hash_capacity,
                self.gauss_d1,
                self.gauss_d2,
                self.resolution,
                self.raw_ptr(&self.all_source_points),
                self.raw_ptr(&self.all_initial_poses),
                self.raw_ptr(&self.points_per_slot),
                self.raw_ptr(&self.all_reduce_buffers),
                self.raw_ptr(&self.barrier_counters),
                self.raw_ptr(&self.barrier_senses),
                self.raw_ptr(&self.all_out_poses),
                self.raw_ptr(&self.all_out_iterations),
                self.raw_ptr(&self.all_out_converged),
                self.raw_ptr(&self.all_out_scores),
                self.raw_ptr(&self.all_out_hessians),
                self.raw_ptr(&self.all_out_correspondences),
                self.raw_ptr(&self.all_out_oscillations),
                self.raw_ptr(&self.all_out_alpha_sums),
                num_requests,
                blocks_per_slot,
                self.max_points_per_slot,
                max_iterations as i32,
                epsilon as f32,
                self.config.use_line_search,
                self.config.num_candidates as i32,
                self.config.armijo_mu,
                self.config.wolfe_nu,
                self.config.fixed_step_size,
                if self.config.regularization_enabled {
                    self.raw_ptr(&self.reg_ref_x)
                } else {
                    0
                },
                if self.config.regularization_enabled {
                    self.raw_ptr(&self.reg_ref_y)
                } else {
                    0
                },
                self.config.regularization_scale_factor,
                self.config.regularization_enabled,
            )?;
        }

        cuda_ffi::batch_persistent_ndt_sync_raw()?;

        // ====================================================================
        // Download results
        // ====================================================================

        let poses_bytes = self.client.read_one(self.all_out_poses.clone());
        let poses_f32 = f32::from_bytes(&poses_bytes);

        let iter_bytes = self.client.read_one(self.all_out_iterations.clone());
        let iterations = i32::from_bytes(&iter_bytes);

        let converged_bytes = self.client.read_one(self.all_out_converged.clone());
        let converged = u32::from_bytes(&converged_bytes);

        let score_bytes = self.client.read_one(self.all_out_scores.clone());
        let scores = f32::from_bytes(&score_bytes);

        let hess_bytes = self.client.read_one(self.all_out_hessians.clone());
        let hessians_flat = f32::from_bytes(&hess_bytes);

        let corr_bytes = self.client.read_one(self.all_out_correspondences.clone());
        let correspondences = u32::from_bytes(&corr_bytes);

        let osc_bytes = self.client.read_one(self.all_out_oscillations.clone());
        let oscillations = u32::from_bytes(&osc_bytes);

        let alpha_bytes = self.client.read_one(self.all_out_alpha_sums.clone());
        let alpha_sums = f32::from_bytes(&alpha_bytes);

        // ====================================================================
        // Build results
        // ====================================================================

        let mut results = Vec::with_capacity(num_requests);

        for slot in 0..num_requests {
            let pose: [f64; 6] = std::array::from_fn(|i| poses_f32[slot * 6 + i] as f64);

            let mut hessian = [[0.0f64; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    hessian[i][j] = hessians_flat[slot * 36 + i * 6 + j] as f64;
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

    /// Get the number of slots in this pipeline.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Get the maximum points per slot.
    pub fn max_points_per_slot(&self) -> usize {
        self.max_points_per_slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_pipeline_creation() {
        let pipeline = BatchGpuPipeline::new(4, 2000, 10000);
        assert!(pipeline.is_ok());

        let pipeline = pipeline.unwrap();
        assert_eq!(pipeline.num_slots(), 4);
        assert_eq!(pipeline.max_points_per_slot(), 2000);
    }

    #[test]
    fn test_batch_pipeline_empty_batch() {
        let mut pipeline = BatchGpuPipeline::new(4, 2000, 10000).unwrap();

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

        // Align empty batch
        let results = pipeline.align_batch(&[], 30, 0.01).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_pipeline_single_alignment() {
        let mut pipeline = BatchGpuPipeline::new(4, 2000, 10000).unwrap();

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

        let results = pipeline.align_batch(&requests, 30, 0.01).unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
        crate::test_println!(
            "Single alignment: {} iterations, converged={}, score={:.4}",
            result.iterations,
            result.converged,
            result.score
        );
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_batch_pipeline_multiple_alignments() {
        let mut pipeline = BatchGpuPipeline::new(4, 2000, 10000).unwrap();

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

        // Create source points for 4 alignments
        let points0 = vec![
            [1.0f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let points1 = vec![
            [0.5f32, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ];
        let points2 = vec![[0.0f32, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]];
        let points3 = vec![
            [2.0f32, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, -2.0, 0.0],
        ];

        let requests = vec![
            AlignmentRequest {
                points: &points0,
                initial_pose: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                reg_ref_x: None,
                reg_ref_y: None,
            },
            AlignmentRequest {
                points: &points1,
                initial_pose: [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                reg_ref_x: None,
                reg_ref_y: None,
            },
            AlignmentRequest {
                points: &points2,
                initial_pose: [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                reg_ref_x: None,
                reg_ref_y: None,
            },
            AlignmentRequest {
                points: &points3,
                initial_pose: [0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
                reg_ref_x: None,
                reg_ref_y: None,
            },
        ];

        let results = pipeline.align_batch(&requests, 30, 0.01).unwrap();
        assert_eq!(results.len(), 4);

        #[allow(unused_variables)]
        for (i, result) in results.iter().enumerate() {
            crate::test_println!(
                "Alignment {}: {} iterations, converged={}, score={:.4}",
                i,
                result.iterations,
                result.converged,
                result.score
            );
            assert!(result.iterations > 0);
        }
    }
}
