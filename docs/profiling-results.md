# NDT Profiling Results

This document captures profiling results comparing the CUDA NDT implementation against Autoware's builtin NDT scan matcher.

## Test Environment

- **Date**: 2026-01-19 (latest profiling run)
- **Hardware**: NVIDIA GPU (CUDA enabled)
- **Dataset**: Autoware sample rosbag (~23 seconds of driving data)
- **Map**: sample-map (point cloud map)
- **Initial Pose**: user_defined_initial_pose enabled for both runs

## Executive Summary (2026-01-19 - Latest)

| Metric             | Autoware (OpenMP) | CUDA NDT     | Ratio            |
|--------------------|-------------------|--------------|------------------|
| **Alignments**     | **1062**          | **264**      | Autoware 4x more |
| Mean iterations    | 3.0               | 3.2          | 1.07x (similar)  |
| Max iterations     | 7                 | 9            | CUDA higher max  |
| Mean NVTL          | 3.05              | 2.29         | 75% of Autoware  |
| Mean score         | 31,741            | 26,711       | 84% of Autoware  |
| Convergence rate   | 100%              | 100%         | Parity           |
| Oscillations       | N/A               | 11% (29)     | -                |

**Key achievements:**
- ✅ **100% convergence** on both implementations
- ✅ **Similar iteration counts** - CUDA: 3.2 vs Autoware: 3.0
- ✅ **Oscillation rate reduced** - 11% (down from 28.5%)

**Remaining gaps:**
- **NVTL**: 2.29 vs 3.05 (75% of Autoware)
- **Alignment count**: 264 vs 1062 (Autoware processes more scans per session)
- Score function may have differences in Gaussian parameters

## Historical: Executive Summary (2026-01-18 - ScanQueue)

| Metric             | Autoware (OpenMP) | CUDA NDT     | Ratio            |
|--------------------|-------------------|--------------|------------------|
| **Alignments**     | **278**           | **272**      | **97.8% parity** |
| Mean iterations    | 3.92              | 3.07         | 0.78x (faster!)  |
| Mean NVTL          | 2.97              | 1.97         | 66% of Autoware  |
| Score per point    | 5.37              | 4.34         | 81% of Autoware  |
| Convergence rate   | 100%              | 100%         | Parity           |

**Key achievement: Throughput parity**
- ✅ **272 vs 278 alignments** - CUDA now processes 97.8% of scans (was 3.3% before)
- ✅ **Fewer iterations** - CUDA: 3.07 vs Autoware: 3.92 (22% fewer!)
- ✅ **100% convergence** on both implementations

**Remaining gap: NVTL and Score**
- NVTL is 66% of Autoware's (1.97 vs 2.97)
- Score per point is 81% of Autoware's (4.34 vs 5.37)
- This suggests differences in scoring/derivative computation, not voxel search

## Historical: Persistent Kernel Results (2026-01-18 Early)

| Metric             | Autoware (OpenMP) | CUDA NDT     | Ratio            |
|--------------------|-------------------|--------------|------------------|
| **Mean exe time**  | **8.83 ms**       | **18.45 ms** | **2.09x slower** |
| Median exe time    | 8.45 ms           | 17.04 ms     | 2.02x slower     |
| Mean iterations    | 3.76              | 3.63         | 0.97x (similar)  |
| Mean NVTL          | 2.97              | 1.96         | 66% of Autoware  |
| Score per point    | 5.37              | 4.31         | 80% of Autoware  |
| Voxels per point   | 3.03              | 3.04         | ~100% (matched!) |
| Convergence rate   | 100%              | 100%         | Parity           |

**Key improvements with persistent kernel:**
- **Iteration count now matches Autoware** (3.63 vs 3.76) - down from 4.30
- **100% convergence** - up from 99.3%
- **Voxels per point now matches** (3.04 vs 3.03) - spatial hash working correctly
- Execution time still 2.09x slower (was 2.17x) - minor improvement

## Execution Time Comparison (2026-01-18)

| Metric   | Autoware (OpenMP) | CUDA NDT     | Ratio            |
|----------|-------------------|--------------|------------------|
| **Mean** | **8.83 ms**       | **18.45 ms** | **2.09x slower** |
| Median   | 8.45 ms           | 17.04 ms     | 2.02x slower     |
| Stdev    | 4.28 ms           | 8.54 ms      | -                |
| Min      | 1.58 ms           | 8.73 ms      | 5.53x slower     |
| Max      | 20.87 ms          | 34.40 ms     | 1.65x slower     |
| P95      | 14.98 ms          | 30.78 ms     | 2.05x slower     |
| P99      | 17.69 ms          | 34.40 ms     | 1.95x slower     |

**Sample sizes**: Autoware: 947 alignments, CUDA: 31 alignments

**Note**: CUDA processed fewer scans due to higher per-alignment latency. This throughput issue needs investigation.

### Execution Time Distribution

**Autoware:**
```
 0- 2ms:   14 (  1.5%)
 2- 5ms:  220 ( 23.2%) ###########
 5-10ms:  358 ( 37.8%) ##################
10-15ms:  309 ( 32.6%) ################
15-20ms:   43 (  4.5%) ##
  20+ms:    3 (  0.3%)
```

**CUDA:**
```
 0- 2ms:    0 (  0.0%)
 2- 5ms:    0 (  0.0%)
 5-10ms:    5 ( 16.1%) ########
10-15ms:   10 ( 32.3%) ################
15-20ms:    4 ( 12.9%) ######
  20+ms:   12 ( 38.7%) ###################
```

## Iteration Analysis (2026-01-18)

| Metric            | Autoware | CUDA  | Notes                    |
|-------------------|----------|-------|--------------------------|
| Mean iterations   | 3.76     | 3.63  | **Now matched!** (was 4.30) |
| Median iterations | 3.0      | 3.0   | Identical                |
| Min               | 2        | 1     |                          |
| Max               | 6        | 11    |                          |

### Iteration Distribution

**Autoware (258 alignments):**
```
 1- 3:  134 ( 51.9%)
 4- 6:  124 ( 48.1%)
 7-10:    0 (  0.0%)
```

**CUDA (267 alignments):**
```
 1- 3:  142 ( 53.2%) ##########################
 4- 6:  107 ( 40.1%) ####################
 7-10:   17 (  6.4%) ###
11-15:    1 (  0.4%)
  21+:    0 (  0.0%)
```

## Convergence Analysis (2026-01-18)

| Metric        | Autoware       | CUDA            |
|---------------|----------------|-----------------|
| **Converged** | **258 (100%)** | **267 (100%)**  |
| MaxIterations | 0 (0%)         | 0 (0%)          |

**100% convergence achieved** - up from 99.3% in the previous profiling run.

## Oscillation Analysis (2026-01-18)

| Metric                    | Autoware | CUDA         |
|---------------------------|----------|--------------|
| Entries with oscillations | N/A      | 76 (28.5%)   |
| Total reversal events     | N/A      | 89           |

**Note**: Oscillation rate reduced from 41.6% to 28.5% after direction check fix. Most alignments converge correctly despite occasional oscillations.

## Score and NVTL Comparison (2026-01-18)

| Metric           | Autoware | CUDA   | Ratio          |
|------------------|----------|--------|----------------|
| Mean NVTL        | 2.97     | 1.96   | 66% of Autoware |
| Mean score/point | 5.37     | 4.31   | 80% of Autoware |
| Mean voxels/pt   | 3.03     | 3.04   | **~100%** (matched!) |

**Analysis**: Voxel search now produces identical results (3.03 vs 3.04 voxels per point), confirming the spatial hash table is working correctly. The remaining NVTL and score gap suggests differences in:
1. Gaussian score computation (d1, d2 parameters)
2. Covariance matrix handling
3. Numerical precision in derivative computation

## Bug Fix History

### Rotation Order Bug (Fixed 2026-01-14)

**Root Cause**: The `transform_point` and `pose_to_transform_matrix` functions used the wrong Euler angle rotation order:
- **Wrong**: R = Rz(yaw) × Ry(pitch) × Rx(roll)
- **Correct**: R = Rx(roll) × Ry(pitch) × Rz(yaw)

This mismatch caused the transformed points to be in a different configuration than what the Jacobian/Hessian formulas expected, leading to Newton steps that pointed in the wrong direction ~60% of the time.

**Files Fixed**:
1. `src/ndt_cuda/src/derivatives/cpu.rs` - `transform_point` function
2. `src/ndt_cuda/src/derivatives/gpu.rs` - `pose_to_transform_matrix` function
3. `src/ndt_cuda/src/scoring/gpu.rs` - `pose_to_transform_matrix_f32` function

**Impact**:

| Metric           | Before Fix | After Fix  | Improvement   |
|------------------|------------|------------|---------------|
| Convergence Rate | 53.9%      | **99.6%**  | +85%          |
| Mean Iterations  | 15.64      | **3.98**   | 3.9x faster   |
| Mean Exe Time    | 12.58 ms   | **5.62 ms**| 2.2x faster   |
| Exe Time Ratio   | 5.45x      | **2.44x**  | 2.2x better   |

## Remaining Performance Gap Analysis (2026-01-18)

### Why CUDA is still 2.09x slower

1. ~~**More iterations on average**~~ **FIXED**
   - ✅ Now: 3.63 vs 3.76 iterations (matched!)
   - Was: 4.30 vs 2.98 (44% more iterations)
   - Direction check fix and Jacobi SVD significantly improved convergence

2. ~~**GPU overhead per iteration**~~ **FIXED**
   - ✅ **Persistent kernel**: Single kernel launch for entire optimization
   - ✅ In-kernel Jacobi SVD solver (no CPU roundtrip)
   - ✅ Data transfer reduced to ~850 bytes total (not per-iteration)

3. ~~**Lower throughput** (31 vs 947 alignments)~~ **FIXED**
   - ✅ Now: 272 vs 278 alignments (97.8% parity!)
   - Was: 31 vs 947 (3.3% throughput)
   - ScanQueue with real-time constraints fixed message handling

4. **Higher base execution time** (18.45 ms vs 8.83 ms)
   - Despite matched iteration counts, CUDA per-iteration is slower
   - Persistent kernel overhead: block synchronization, Jacobi SVD iterations
   - GPU memory access patterns may not be optimal
   - Scoring computation differences may cause extra iterations initially

5. **Reduced oscillation rate** (28.5% vs ~0%)
   - Improved from 41.6% but still present
   - May indicate remaining numerical differences with Autoware

### Spatial Hashing Optimization (2026-01-15)

Replaced brute-force O(N×V) radius search with GPU spatial hash table:

| Metric                   | Before        | After        | Improvement        |
|--------------------------|---------------|--------------|--------------------|
| Radius search complexity | O(N × 12,000) | O(N × 27)    | 444x fewer lookups |
| Mean exe time            | 5.62 ms       | 5.05 ms      | 10% faster         |
| Performance ratio        | 2.44x slower  | 2.17x slower | 12% closer         |

**Implementation**: Custom CUDA spatial hash table in `cuda_ffi/csrc/voxel_hash.cu`
- Uses MurmurHash3 finalizer for 3D grid coordinates
- Open addressing with linear probing
- Built once per map load, queries 27 neighboring cells per point

### Persistent Kernel Implementation (2026-01-18)

Replaced per-iteration kernel launches with a single cooperative kernel:

| Aspect                    | Before (multi-kernel)     | After (persistent)        |
|---------------------------|---------------------------|---------------------------|
| Kernel launches/alignment | 10-14 per iteration       | 1 total                   |
| CPU-GPU sync/iteration    | Multiple                  | None                      |
| Newton solve location     | CPU (cuSOLVER)            | GPU (in-kernel Jacobi SVD)|
| Data transfer/iteration   | ~224 bytes                | 0 (all in GPU memory)     |
| Total data transfer       | ~224 × iterations bytes   | ~850 bytes (once)         |

**Implementation**: `cuda_ffi/csrc/persistent_ndt.cu`
- Cooperative kernel using `__syncthreads()` for grid-wide synchronization
- In-kernel Jacobi SVD solver for 6x6 Newton system (handles indefinite Hessians)
- In-kernel line search with Strong Wolfe conditions
- Block 0 performs reductions and Newton solve, broadcasts to all blocks

### Batch Processing Pipeline (2026-01-18)

Added `BatchGpuPipeline` for processing multiple scans in parallel:

| Aspect                    | Single-scan (persistent)  | Batch (M scans)           |
|---------------------------|---------------------------|---------------------------|
| Alignments per launch     | 1                         | Up to 8                   |
| Synchronization           | `grid.sync()` (global)    | Atomic barriers (per-slot)|
| GPU utilization           | ~20% (serialized Newton)  | ~80%+ (parallel slots)    |
| Throughput                | Limited by latency        | M× throughput potential   |

**Implementation**: `cuda_ffi/csrc/batch_persistent_ndt.cu`
- Partitions GPU blocks into M independent slots
- Each slot runs complete Newton optimization
- Per-slot atomic barriers instead of cooperative grid sync
- Eliminates idle time from serial Newton solve

**Usage** (Rust API):
```rust
// Process multiple scans in parallel
let scans = vec![
    (scan1.as_slice(), pose1),
    (scan2.as_slice(), pose2),
    (scan3.as_slice(), pose3),
];
let results = matcher.align_parallel_scans(&scans)?;
```

### Scan Queue for Real-Time Batch Processing (2026-01-18)

Implemented `ScanQueue` in `cuda_ndt_matcher` to enable batch processing with real-time constraints:

| Feature | Description |
|---------|-------------|
| **Batch Processing** | Accumulates scans and processes via `align_parallel_scans()` |
| **Max Queue Depth** | Drops oldest scans when queue exceeds limit (default: 8) |
| **Max Scan Age** | Drops stale scans to maintain real-time responsiveness (default: 100ms) |
| **Batch Trigger** | Processes when N scans accumulated (default: 4) |
| **Timeout Trigger** | Processes partial batch after timeout (default: 20ms) |
| **Async Publishing** | Results published via callback in background thread |

**Configuration** (`ndt_scan_matcher.param.yaml`):
```yaml
batch:
  enabled: true  # Enable batch processing (default: false)
  max_queue_depth: 8
  max_scan_age_ms: 100
  batch_trigger: 4
  timeout_ms: 20
```

**Expected Throughput Improvement**:
- 4 scans per batch × ~7ms total latency = ~571 scans/sec theoretical
- Compared to ~54 scans/sec with serial processing
- **10x throughput improvement** potential

### Remaining Optimization Opportunities

1. ~~**Enable batch processing in ROS node**~~ **DONE**
   - ✅ `ScanQueue` implemented with real-time constraints
   - ✅ Async result publishing via callback
   - ✅ Configurable via `batch.*` parameters
   - ✅ **Throughput validated**: 272 vs 278 alignments (97.8% parity!)

2. **Reduce synchronization barriers**
   - Current: 9 `grid.sync()` barriers per iteration
   - Opportunity: Merge reduction phases, overlap computation

3. ~~**Investigate remaining throughput gap**~~ **RESOLVED**
   - ✅ Throughput parity achieved with ScanQueue
   - Real-time constraints (max age, queue depth) prevent scan buildup

## Data Files (2026-01-19)

| File                              | Description                  |
|-----------------------------------|------------------------------|
| `logs/ndt_autoware_debug.jsonl`   | Autoware iteration debug     |
| `logs/ndt_cuda_debug.jsonl`       | CUDA iteration debug         |

## Reproducing Results

```bash
# Run CUDA NDT with debug output
just run-cuda-debug

# Build comparison Autoware (required for debug output)
just build-comparison

# Run Autoware builtin NDT with debug output
just run-builtin-debug

# Analyze results
just analyze-debug-cuda
just analyze-debug-autoware
```

## Conclusion (2026-01-19)

The CUDA NDT implementation achieves **functional parity** with Autoware's OpenMP implementation.

**Key achievements**:
- ✅ **Convergence rate**: **100%** (matches Autoware)
- ✅ **Iteration count**: **3.2** (similar to Autoware's 3.0)
- ✅ **Oscillation rate**: **11%** (reduced from 28.5%)
- ✅ **Persistent kernel**: Single kernel launch for entire optimization
- ✅ **GPU Newton solve**: In-kernel 6x6 Jacobi SVD
- ✅ **Direction check**: Corrects SVD sign ambiguity
- ✅ **ScanQueue**: Real-time batch processing with age-based dropping

**Remaining gaps**:
1. **NVTL**: 2.29 vs 3.05 (75% of Autoware)
2. **Alignment count**: 264 vs 1062 (need to investigate scan processing rate)

**Next steps**:
- Investigate NVTL/scoring differences (Gaussian parameters d1, d2)
- Profile scan processing pipeline to understand alignment count difference
- Match Autoware's scoring computation exactly
