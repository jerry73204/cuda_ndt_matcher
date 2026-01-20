# Phase 23: GPU Utilization Improvements

## Overview

This phase focuses on improving GPU utilization and reducing per-iteration latency. Current profiling shows CUDA per-iteration is 2.2x slower than Autoware's OpenMP implementation despite using GPU acceleration. This phase addresses memory bandwidth bottlenecks, synchronization overhead, and underutilized compute resources.

## Current State Analysis

### Performance Gap

| Metric | Autoware (OpenMP) | CUDA | Gap |
|--------|-------------------|------|-----|
| Per-iteration time | ~2.3ms | ~5.1ms | 2.2x slower |
| Iterations | 3.76 | 3.63 | Similar |
| Total alignment | 8.83ms | 18.45ms | 2.1x slower |

### Root Causes

1. **Memory Bandwidth Bound**
   - ~300-500 bytes memory traffic per point
   - ~500 FLOPs compute per point
   - Arithmetic intensity: ~1-1.7 FLOP/byte (memory bound regime)

2. **Hash Table Random Access**
   - Scattered global memory reads for voxel lookup
   - Poor cache utilization across warps

3. **Serial Newton Solve**
   - Only thread 0 of block 0 executes 6x6 solve
   - All other threads idle during Phase C

4. **Excessive Synchronization**
   - 9+ `grid.sync()` calls per iteration
   - Cooperative kernel limits occupancy

## Architecture

### Phase 23.1: Async Streams + Double Buffering

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Double-Buffered Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stream 0: [H2D batch 0]──[Batch Kernel 0]──[D2H batch 0]               │
│  Stream 1:      [H2D batch 1]──[Batch Kernel 1]──[D2H batch 1]          │
│                       ▲ overlap ▲         ▲ overlap ▲                    │
│                                                                          │
│  Pinned Memory Buffers (×2):                                            │
│  ├── h_points_pinned[max_batch × max_points × 3]                        │
│  ├── h_poses_pinned[max_batch × 6]                                      │
│  └── h_results_pinned[max_batch × result_size]                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 23.2: Texture Memory for Voxel Data

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Memory Hierarchy                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Before:                           After:                                │
│  ┌──────────────┐                  ┌──────────────┐                     │
│  │ Global Memory │ ◄── voxel_means │ Texture Cache │ ◄── voxel_means    │
│  │ (uncached)    │ ◄── voxel_covs  │ (separate L1) │ ◄── voxel_covs     │
│  └──────────────┘                  └──────────────┘                     │
│        │                                  │                              │
│        ▼                                  ▼                              │
│  Cache miss likely              Hardware-accelerated                     │
│  for scattered reads            interpolation + caching                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 23.3: Warp-Level Reduction

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Reduction Comparison                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Before (Block-level):                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Thread → Shared Memory → __syncthreads() → Tree Reduce → Global │    │
│  │          (29 values)      (log N times)                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  After (Warp-level):                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Thread → Registers → __shfl_down_sync() → Lane 0 → Atomic Global│    │
│  │          (29 values)   (5 iterations)      (1 write per warp)   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Benefits: No shared memory, no __syncthreads(), 32x fewer atomics      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 23.4: Warp-Cooperative Newton Solve

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Newton Solve Parallelization                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Before: Single thread (threadIdx.x == 0 && blockIdx.x == 0)            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Thread 0: Load H[36] → Jacobi SVD (serial) → Store delta[6]     │    │
│  │ Threads 1-255: IDLE                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  After: Warp-parallel (32 threads collaborate)                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Lane 0-5: Each owns one column of L (Cholesky)                  │    │
│  │ All lanes: Parallel dot products via warp shuffle               │    │
│  │ Lane 0: Broadcasts result                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 23.1: Async Streams + Double Buffering

**Status:** Not Started

**Files:**
- `src/cuda_ffi/src/async_batch.rs` (new)
- `src/cuda_ffi/csrc/async_utils.cu` (new)
- `src/ndt_cuda/src/optimization/async_pipeline.rs` (new)

**Tasks:**

1. **Pinned Memory Allocation**
   ```rust
   // FFI bindings for cudaMallocHost / cudaFreeHost
   pub fn cuda_malloc_host<T>(count: usize) -> Result<*mut T, CudaError>;
   pub fn cuda_free_host<T>(ptr: *mut T) -> Result<(), CudaError>;
   ```

2. **Stream Management**
   ```rust
   pub struct CudaStream {
       handle: cudaStream_t,
   }

   impl CudaStream {
       pub fn new() -> Result<Self, CudaError>;
       pub fn synchronize(&self) -> Result<(), CudaError>;
       pub fn as_raw(&self) -> cudaStream_t;
   }
   ```

3. **Event Management**
   ```rust
   pub struct CudaEvent {
       handle: cudaEvent_t,
   }

   impl CudaEvent {
       pub fn new() -> Result<Self, CudaError>;
       pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError>;
       pub fn is_complete(&self) -> bool;
       pub fn synchronize(&self) -> Result<(), CudaError>;
   }
   ```

4. **Async Memory Operations**
   ```rust
   pub fn cuda_memcpy_async_h2d<T>(
       dst: *mut T,
       src: *const T,
       count: usize,
       stream: &CudaStream,
   ) -> Result<(), CudaError>;

   pub fn cuda_memcpy_async_d2h<T>(
       dst: *mut T,
       src: *const T,
       count: usize,
       stream: &CudaStream,
   ) -> Result<(), CudaError>;
   ```

5. **Double-Buffered Pipeline**
   ```rust
   pub struct AsyncBatchPipeline {
       streams: [CudaStream; 2],
       events: [CudaEvent; 2],
       buffers: [BatchBuffers; 2],
       current: usize,
       pending: Option<usize>,
   }

   impl AsyncBatchPipeline {
       pub fn submit(&mut self, scans: &[ScanData]) -> Result<(), CudaError>;
       pub fn poll(&mut self) -> Option<Vec<AlignResult>>;
       pub fn wait(&mut self) -> Option<Vec<AlignResult>>;
   }
   ```

6. **Update batch_persistent_ndt_launch to accept stream parameter**

**Validation:**
- Unit test: async H2D/D2H correctness
- Unit test: double buffer overlap (stream 1 starts before stream 0 finishes)
- Benchmark: measure overlap efficiency

**Expected Impact:** 1.2x throughput improvement

---

### Phase 23.2: Texture Memory for Voxel Data

**Status:** Not Started

**Files:**
- `src/cuda_ffi/csrc/texture_voxels.cu` (new)
- `src/cuda_ffi/csrc/persistent_ndt.cu` (update)
- `src/cuda_ffi/csrc/batch_persistent_ndt.cu` (update)
- `src/cuda_ffi/src/texture.rs` (new)

**Tasks:**

1. **Create Texture Objects for Voxel Arrays**
   ```cpp
   // In texture_voxels.cu
   cudaTextureObject_t create_voxel_texture(
       const float* voxel_data,
       size_t num_voxels,
       int components  // 3 for means, 9 for inv_covs
   );

   void destroy_voxel_texture(cudaTextureObject_t tex);
   ```

2. **Texture Descriptor Configuration**
   ```cpp
   cudaTextureDesc texDesc = {};
   texDesc.addressMode[0] = cudaAddressModeClamp;
   texDesc.filterMode = cudaFilterModePoint;  // No interpolation
   texDesc.readMode = cudaReadModeElementType;
   texDesc.normalizedCoords = 0;
   ```

3. **Update Kernel to Use Texture Reads**
   ```cpp
   // Before:
   float vx = voxel_means[vidx * 3 + 0];

   // After:
   float vx = tex1Dfetch<float>(tex_means, vidx * 3 + 0);
   ```

4. **Rust FFI Bindings**
   ```rust
   pub struct VoxelTextures {
       means: cudaTextureObject_t,
       inv_covs: cudaTextureObject_t,
   }

   impl VoxelTextures {
       pub fn new(means: &CudaBuffer<f32>, inv_covs: &CudaBuffer<f32>, num_voxels: usize) -> Result<Self, CudaError>;
   }

   impl Drop for VoxelTextures {
       fn drop(&mut self) { /* destroy textures */ }
   }
   ```

5. **Update VoxelGrid to Create Textures on GPU Upload**

**Validation:**
- Unit test: texture read correctness (compare with global memory reads)
- Benchmark: memory bandwidth improvement
- Rosbag test: results unchanged

**Expected Impact:** 1.3-1.5x per-iteration speedup

---

### Phase 23.3: Warp-Level Reduction

**Status:** Not Started

**Files:**
- `src/cuda_ffi/csrc/warp_reduce.cuh` (new)
- `src/cuda_ffi/csrc/persistent_ndt.cu` (update)
- `src/cuda_ffi/csrc/batch_persistent_ndt.cu` (update)

**Tasks:**

1. **Warp Shuffle Reduction Primitives**
   ```cpp
   // warp_reduce.cuh
   __device__ __forceinline__ float warp_reduce_sum(float val) {
       #pragma unroll
       for (int offset = 16; offset > 0; offset /= 2) {
           val += __shfl_down_sync(0xffffffff, val, offset);
       }
       return val;
   }

   __device__ __forceinline__ void warp_reduce_sum_array(
       float* vals,
       int count
   ) {
       #pragma unroll
       for (int i = 0; i < count; i++) {
           vals[i] = warp_reduce_sum(vals[i]);
       }
   }
   ```

2. **Replace Block Reduction in Phase B**
   ```cpp
   // Before: shared memory tree reduction
   partial_sums[threadIdx.x * REDUCE_SIZE + i] = my_val;
   __syncthreads();
   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { ... }

   // After: warp-level reduction + cross-warp atomic
   float warp_sum = warp_reduce_sum(my_val);
   int lane = threadIdx.x % 32;
   int warp_id = threadIdx.x / 32;

   if (lane == 0) {
       atomicAdd(&reduce_buffer[i], warp_sum);
   }
   ```

3. **Remove Shared Memory for Reduction**
   - Keep shared memory only for line search (smaller footprint)
   - Reduces shared memory pressure, increases occupancy

4. **Update Shared Memory Calculation**
   ```cpp
   // Before: BLOCK_SIZE * REDUCE_SIZE * sizeof(float) = 256 * 29 * 4 = 29.7 KB
   // After: LS_BATCH_SIZE * LS_VALUES_PER_CAND * sizeof(float) = 4 * 8 * 4 = 128 bytes
   ```

**Validation:**
- Unit test: warp reduction correctness
- Unit test: cross-warp atomic accumulation
- Benchmark: reduction phase speedup
- Verify occupancy improvement (fewer shared memory)

**Expected Impact:** 1.3x reduction phase speedup, higher occupancy

---

### Phase 23.4: Warp-Cooperative Newton Solve

**Status:** Not Started

**Files:**
- `src/cuda_ffi/csrc/warp_cholesky.cuh` (new)
- `src/cuda_ffi/csrc/persistent_ndt.cu` (update)
- `src/cuda_ffi/csrc/batch_persistent_ndt.cu` (update)

**Tasks:**

1. **Warp-Parallel Cholesky Decomposition**
   ```cpp
   // warp_cholesky.cuh
   __device__ void warp_cholesky_6x6(
       const float* H,   // [36] input Hessian (symmetric)
       float* L,         // [36] output lower triangular
       int lane          // threadIdx.x % 32
   ) {
       // Each of lanes 0-5 owns one column of L
       // Parallel dot products using warp shuffles
       // Lane 0 broadcasts diagonal elements
   }
   ```

2. **Warp-Parallel Forward/Back Substitution**
   ```cpp
   __device__ void warp_solve_6x6(
       const float* L,   // [36] Cholesky factor
       const float* b,   // [6] right-hand side
       float* x,         // [6] solution
       int lane
   ) {
       // Forward substitution: L y = b
       // Backward substitution: L^T x = y
       // Parallel via warp shuffles
   }
   ```

3. **Replace Single-Thread Jacobi SVD**
   ```cpp
   // Before (Phase C):
   if (threadIdx.x == 0 && blockIdx.x == 0) {
       jacobi_svd_solve_6x6_f64(H, g, delta, &solve_success);
   }

   // After:
   if (blockIdx.x == 0) {
       int lane = threadIdx.x % 32;
       if (threadIdx.x < 32) {  // First warp only
           warp_cholesky_6x6(H, L, lane);
           __syncwarp();
           warp_solve_6x6(L, g, delta, lane);
       }
   }
   ```

4. **Fallback for Indefinite Hessians**
   - Cholesky fails for indefinite matrices
   - Detect via negative diagonal in L
   - Fall back to steepest descent with small step

**Validation:**
- Unit test: warp Cholesky correctness (compare with CPU)
- Unit test: indefinite matrix detection
- Benchmark: Newton solve speedup
- Numerical stability test: 10,000 random Hessians

**Expected Impact:** 1.2x overall speedup (Newton solve parallelized)

---

### Phase 23.5: Shared Memory Voxel Cache (Optional)

**Status:** Not Started

**Files:**
- `src/cuda_ffi/csrc/persistent_ndt.cu` (update)
- `src/cuda_ffi/csrc/batch_persistent_ndt.cu` (update)

**Tasks:**

1. **Identify Hot Voxels per Block**
   - Points in same block likely query same voxels
   - Use warp vote to find most common voxel indices

2. **Collaborative Cache Loading**
   ```cpp
   __shared__ float cached_means[CACHE_SIZE][3];
   __shared__ float cached_inv_covs[CACHE_SIZE][9];
   __shared__ int cached_ids[CACHE_SIZE];

   // First N threads load most popular voxels
   if (threadIdx.x < CACHE_SIZE) {
       int voxel_id = get_hot_voxel(threadIdx.x, ...);
       cached_ids[threadIdx.x] = voxel_id;
       // Load voxel data...
   }
   __syncthreads();
   ```

3. **Cache Lookup Before Global Memory**
   ```cpp
   int cache_slot = find_in_cache(voxel_idx, cached_ids, CACHE_SIZE);
   if (cache_slot >= 0) {
       // Use cached data
       voxel_mean[0] = cached_means[cache_slot][0];
       // ...
   } else {
       // Fall back to global/texture memory
   }
   ```

**Validation:**
- Benchmark: cache hit rate
- Benchmark: speedup vs texture-only

**Expected Impact:** 1.1-1.2x additional speedup (if cache hit rate > 50%)

---

### Phase 23.6: Hierarchical Early-Exit (Optional)

**Status:** Not Started

**Files:**
- `src/cuda_ffi/csrc/adaptive_ndt.cu` (new)

**Tasks:**

1. **Track Per-Point Score Contribution**
   - After iteration 0, store each point's score contribution
   - Points with negligible contribution can be skipped

2. **Warp-Level Point Compaction**
   ```cpp
   bool active = (my_score_contrib > threshold);
   uint32_t mask = __ballot_sync(0xffffffff, active);
   int active_count = __popc(mask);

   if (!active) return;  // Early exit
   ```

3. **Adaptive Threshold**
   - Start with low threshold (include most points)
   - Increase threshold as iterations progress
   - Near convergence, only process high-impact points

**Validation:**
- Unit test: early-exit correctness (final result unchanged)
- Benchmark: iteration speedup vs point reduction
- Verify convergence not affected

**Expected Impact:** 1.3x speedup in later iterations (30-50% fewer points)

## Performance Expectations

### Per-Phase Impact

| Phase | Component | Expected Speedup | Cumulative |
|-------|-----------|------------------|------------|
| 23.1 | Async streams | 1.2x throughput | 1.2x |
| 23.2 | Texture memory | 1.3-1.5x | 1.5-1.8x |
| 23.3 | Warp reduction | 1.3x (reduction) | 1.6-2.0x |
| 23.4 | Warp Newton | 1.2x | 1.9-2.4x |
| 23.5 | Voxel cache | 1.1x | 2.0-2.6x |
| 23.6 | Early-exit | 1.3x (late iters) | 2.2-2.8x |

### Target Performance

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Per-iteration time | 5.1ms | 2.0-2.5ms | 2-2.5x |
| Total alignment | 18.45ms | 7-9ms | 2-2.5x |
| vs Autoware | 2.1x slower | Parity or faster | - |

## Memory Overhead

### Phase 23.1 (Async Streams)

| Buffer | Size | Notes |
|--------|------|-------|
| Pinned host (×2) | 2 × 500KB | Points + poses + results |
| Device (×2) | 2 × 500KB | Double buffering |
| Streams + Events | ~1KB | Handle overhead |
| **Total** | ~2MB | Negligible |

### Phase 23.2 (Texture Memory)

| Resource | Size | Notes |
|----------|------|-------|
| Texture objects | ~100 bytes | Handles only |
| Texture cache | Hardware | No additional allocation |
| **Total** | ~100 bytes | Negligible |

### Phase 23.5 (Voxel Cache)

| Buffer | Size | Notes |
|--------|------|-------|
| Shared memory | ~3KB per block | 64 voxels × (12 + 36 + 4) bytes |
| **Total** | ~3KB | Reduces available shared mem |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Texture cache thrashing | Reduced benefit | Benchmark with real workloads; fall back if no improvement |
| Warp divergence in reduction | Performance loss | Ensure uniform control flow; use predication |
| Cholesky instability | Incorrect results | Fallback to Jacobi SVD for indefinite cases |
| Shared memory pressure | Lower occupancy | Phase 23.3 reduces shared mem usage; monitor occupancy |
| Async complexity | Race conditions | Thorough testing; use events for synchronization |

## Success Criteria

1. **Performance**: 2x speedup in per-iteration time (5.1ms → 2.5ms)
2. **Correctness**: Results match within 1e-6 of current implementation
3. **Stability**: No crashes or hangs in 10,000+ alignments
4. **Throughput**: Async pipeline achieves >80% overlap efficiency

## Dependencies

- CUDA 11.0+ (warp shuffle intrinsics)
- Compute capability 7.0+ (cooperative groups, if retained)
- No external library dependencies

## Future Extensions

### Phase 23.7: CUDA Graphs (Optional)

Capture iteration kernel sequence as a graph:
- Further reduce launch overhead
- Enable driver-level optimization
- Requires decoupled kernel architecture

### Phase 23.8: Multi-GPU Support (Optional)

Split point cloud across multiple GPUs:
- Each GPU processes subset of points
- Final reduction across GPUs
- Requires multi-device memory management
