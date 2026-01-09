# Phase 11: GPU Zero-Copy Pipeline

## Overview

Eliminate unnecessary CPU-GPU memory transfers in the voxel grid construction pipeline by keeping data on GPU between consecutive operations.

## Problem Statement

The current GPU pipeline has 6 CPU-GPU transfers where only 2 are necessary:

```
CURRENT PIPELINE (6 transfers):

Points (CPU)
    │ UPLOAD #1
    ▼
Morton Codes (CubeCL GPU)
    │ DOWNLOAD #1
    ▼
    │ UPLOAD #2
Radix Sort (cuda_ffi GPU)
    │ DOWNLOAD #2
    ▼
    │ UPLOAD #3
Segment Detect (cuda_ffi GPU)
    │ DOWNLOAD #3
    ▼
    │ UPLOAD #4
Statistics (CubeCL GPU)
    │ DOWNLOAD #4
    ▼
Results (CPU)
```

Each transfer crosses the PCIe bus, adding latency and consuming bandwidth.

## Target Architecture

```
TARGET PIPELINE (2 transfers):

Points (CPU)
    │ UPLOAD (once)
    ▼
┌─────────────────────────────────┐
│  GPU Memory (CubeCL-managed)    │
│                                 │
│  Morton Codes ──► Radix Sort    │
│       (CubeCL)    (cuda_ffi)    │
│                       │         │
│                       ▼         │
│              Segment Detect     │
│                (cuda_ffi)       │
│                       │         │
│                       ▼         │
│              Statistics         │
│                (CubeCL)         │
└─────────────────────────────────┘
    │ DOWNLOAD (once)
    ▼
Results (CPU)
```

## Technical Approach

### CubeCL-cuda_ffi Interoperability

CubeCL's CUDA backend stores raw device pointers that can be extracted:

```rust
// CubeCL buffer management
pub struct GpuResource {
    pub ptr: u64,              // Raw CUdeviceptr
    pub binding: *mut c_void,
    pub size: u64,
}

// Extract raw pointer from CubeCL handle
let binding_resource = client.get_resource(handle.binding());
let gpu_resource = binding_resource.resource();
let raw_cuda_ptr: u64 = gpu_resource.ptr;
```

### Strategy

1. **CubeCL owns all GPU memory** - Single memory manager, no fragmentation
2. **cuda_ffi accepts raw pointers** - No internal allocation in CUB wrappers
3. **Pre-allocate buffers** - Reuse across multiple voxel grid constructions

## Implementation Phases

### Phase 11.1: cuda_ffi In-Place API

Add functions that operate on pre-allocated GPU memory:

**File: `cuda_ffi/src/radix_sort.rs`**
```rust
/// Sort key-value pairs using pre-allocated GPU buffers.
///
/// # Arguments
/// * `d_keys_in` - Device pointer to input keys (CUdeviceptr)
/// * `d_keys_out` - Device pointer to output keys
/// * `d_values_in` - Device pointer to input values
/// * `d_values_out` - Device pointer to output values
/// * `num_items` - Number of elements to sort
///
/// # Safety
/// All pointers must be valid CUDA device pointers with sufficient size.
pub unsafe fn sort_pairs_inplace(
    d_keys_in: u64,
    d_keys_out: u64,
    d_values_in: u64,
    d_values_out: u64,
    num_items: usize,
) -> Result<(), CudaError>;
```

**File: `cuda_ffi/src/segment_detect.rs`**
```rust
/// Segment counts returned by in-place detection.
pub struct SegmentCounts {
    pub num_segments: u32,
    pub num_boundaries: u32,
}

/// Detect segments using pre-allocated GPU buffers.
///
/// # Arguments
/// * `d_sorted_codes` - Device pointer to sorted Morton codes
/// * `d_boundaries` - Device pointer for boundary flags (scratch buffer)
/// * `d_segment_ids` - Device pointer for output segment IDs
/// * `d_segment_starts` - Device pointer for output segment starts
/// * `d_num_selected` - Device pointer for output count (single u32)
/// * `num_items` - Number of input codes
///
/// # Safety
/// All pointers must be valid CUDA device pointers with sufficient size.
pub unsafe fn detect_segments_inplace(
    d_sorted_codes: u64,
    d_boundaries: u64,
    d_segment_ids: u64,
    d_segment_starts: u64,
    d_num_selected: u64,
    num_items: usize,
) -> Result<SegmentCounts, CudaError>;
```

**File: `cuda_ffi/csrc/radix_sort.cu`**
```cuda
extern "C" CudaError cub_radix_sort_pairs_u64_u32_inplace(
    void* d_temp_storage,
    size_t temp_storage_bytes,
    const uint64_t* d_keys_in,
    uint64_t* d_keys_out,
    const uint32_t* d_values_in,
    uint32_t* d_values_out,
    int num_items,
    int begin_bit,
    int end_bit,
    cudaStream_t stream
);
```

**File: `cuda_ffi/csrc/segment_detect.cu`**
```cuda
extern "C" CudaError cub_detect_segments_inplace(
    const uint64_t* d_sorted_codes,
    uint32_t* d_boundaries,
    uint32_t* d_segment_ids,
    uint32_t* d_segment_starts,
    int* d_num_selected,
    int num_items,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    cudaStream_t stream
);
```

### Phase 11.2: GPU Pipeline Manager

New module to manage GPU buffers across operations:

**File: `ndt_cuda/src/voxel_grid/gpu/pipeline.rs`**
```rust
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;

/// Pre-allocated GPU buffers for the voxel construction pipeline.
pub struct GpuPipelineBuffers {
    client: ComputeClient<CudaServer>,

    // Input/output
    points: Handle,           // [f32; num_points * 3]

    // Morton code stage
    morton_codes: Handle,     // [u64; num_points]
    point_indices: Handle,    // [u32; num_points]

    // Radix sort output
    sorted_codes: Handle,     // [u64; num_points]
    sorted_indices: Handle,   // [u32; num_points]

    // Segment detection
    boundaries: Handle,       // [u32; num_points] (scratch)
    segment_ids: Handle,      // [u32; num_points]
    segment_starts: Handle,   // [u32; num_points] (max size)
    num_selected: Handle,     // [u32; 1]

    // Statistics
    position_sums: Handle,    // [f32; max_segments * 3]
    counts: Handle,           // [u32; max_segments]
    means: Handle,            // [f32; max_segments * 3]
    cov_sums: Handle,         // [f32; max_segments * 9]

    // CUB temporary storage
    sort_temp: Handle,
    scan_temp: Handle,
    select_temp: Handle,

    // Capacity
    max_points: usize,
    max_segments: usize,
}

impl GpuPipelineBuffers {
    /// Create pipeline buffers for given capacity.
    pub fn new(max_points: usize) -> Result<Self, anyhow::Error>;

    /// Resize buffers if needed.
    pub fn ensure_capacity(&mut self, num_points: usize) -> Result<(), anyhow::Error>;

    /// Get raw CUDA device pointer for a handle.
    fn raw_ptr(&self, handle: &Handle) -> u64 {
        let res = self.client.get_resource(handle.binding());
        res.resource().ptr
    }

    /// Execute the full pipeline, keeping data on GPU.
    pub fn build_voxel_grid(
        &mut self,
        points: &[[f32; 3]],
        config: &VoxelGridConfig,
    ) -> Result<VoxelGrid, anyhow::Error>;
}
```

### Phase 11.3: Pipeline Integration

**File: `ndt_cuda/src/voxel_grid/gpu_builder.rs`**

Update `GpuVoxelGridBuilder` to use the zero-copy pipeline:

```rust
impl GpuVoxelGridBuilder {
    /// Build using zero-copy GPU pipeline.
    pub fn build_zero_copy(
        &mut self,
        points: &[[f32; 3]],
        config: &VoxelGridConfig,
    ) -> Result<VoxelGrid> {
        // Ensure buffers are allocated
        self.pipeline.ensure_capacity(points.len())?;

        // Single upload
        self.upload_points(points)?;

        // GPU operations (no transfers)
        self.compute_morton_codes()?;
        self.radix_sort_inplace()?;
        self.detect_segments_inplace()?;
        self.compute_statistics()?;

        // Single download
        let stats = self.download_statistics()?;

        // CPU finalization
        finalize_voxels_cpu(stats, config)
    }
}
```

### Phase 11.4: Stream Synchronization

Ensure CubeCL and cuda_ffi operations are properly ordered:

```rust
// Option A: Default stream (simplest)
// Both CubeCL and cuda_ffi use stream 0 by default

// Option B: Explicit synchronization
impl GpuPipelineBuffers {
    fn sync_before_cuda_ffi(&self) {
        // Ensure CubeCL operations complete before cuda_ffi
        cubecl::future::block_on(self.client.sync());
    }

    fn sync_after_cuda_ffi(&self) {
        // cudaDeviceSynchronize() called in cuda_ffi
    }
}
```

## Files to Modify

| File | Changes |
|------|---------|
| `cuda_ffi/src/radix_sort.rs` | Add `sort_pairs_inplace()` |
| `cuda_ffi/src/segment_detect.rs` | Add `detect_segments_inplace()`, `SegmentCounts` |
| `cuda_ffi/csrc/radix_sort.cu` | Add `cub_radix_sort_pairs_u64_u32_inplace()` |
| `cuda_ffi/csrc/segment_detect.cu` | Add `cub_detect_segments_inplace()` |
| `ndt_cuda/src/voxel_grid/gpu/pipeline.rs` | NEW: `GpuPipelineBuffers` |
| `ndt_cuda/src/voxel_grid/gpu/mod.rs` | Export pipeline module |
| `ndt_cuda/src/voxel_grid/gpu_builder.rs` | Add `build_zero_copy()` |

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_sort_pairs_inplace() {
    // Allocate via CubeCL
    let client = CudaRuntime::client(&CudaDevice::new(0));
    let keys_in = client.create(&[5u64, 3, 1, 4, 2]);
    let keys_out = client.empty(5 * 8);
    // ... extract ptrs, call inplace, verify
}

#[test]
fn test_zero_copy_pipeline_matches_original() {
    let points = make_test_point_cloud();
    let config = VoxelGridConfig::default();

    let builder = GpuVoxelGridBuilder::new()?;
    let grid_original = builder.build_segmented(&points, &config)?;
    let grid_zero_copy = builder.build_zero_copy(&points, &config)?;

    assert_eq!(grid_original.len(), grid_zero_copy.len());
    // Compare voxel statistics...
}
```

### Performance Benchmarks

```rust
#[bench]
fn bench_build_segmented(b: &mut Bencher) {
    let points = make_large_point_cloud(100_000);
    let builder = GpuVoxelGridBuilder::new().unwrap();
    b.iter(|| builder.build_segmented(&points, &config));
}

#[bench]
fn bench_build_zero_copy(b: &mut Bencher) {
    let points = make_large_point_cloud(100_000);
    let mut builder = GpuVoxelGridBuilder::new().unwrap();
    b.iter(|| builder.build_zero_copy(&points, &config));
}
```

## Expected Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| CPU↔GPU transfers | 6 | 2 | 3x fewer |
| Memory allocations per call | 6+ | 0 (pre-allocated) | Eliminated |
| PCIe bandwidth usage | ~6N bytes | ~2N bytes | 3x reduction |
| Allocation overhead | ~100μs per alloc | 0 | Eliminated |

For a typical 100K point cloud:
- Before: ~600KB transferred (6 × 100KB)
- After: ~200KB transferred (2 × 100KB)
- Savings: ~400KB per voxel grid construction

## Dependencies

- CubeCL v0.8.x with CUDA backend
- cuda_ffi crate (local)
- CUDA Toolkit with CUB headers

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Stream synchronization issues | Use default stream, add explicit sync points |
| Memory lifetime bugs | CubeCL owns all memory, cuda_ffi borrows |
| Buffer size mismatches | Pre-allocate with headroom, resize on demand |
| Pointer alignment issues | CubeCL already aligns for GPU (256-byte) |

## Success Criteria

1. All existing tests pass with zero-copy pipeline
2. No memory leaks (verified with cuda-memcheck)
3. Measurable performance improvement in benchmarks
4. No regression in voxel grid accuracy

## Future Enhancements

1. **Async pipeline**: Overlap CPU eigendecomposition with GPU work
2. **Multi-stream**: Use separate streams for independent operations
3. **Buffer pooling**: Reuse buffers across multiple NDT alignments
4. **GPU-resident VoxelGrid**: Keep voxel data on GPU for correspondence search
