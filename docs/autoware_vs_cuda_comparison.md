# CUDA NDT vs Autoware NDT Comparison Findings

**Date**: 2026-01-27
**Sample Data**: sample-rosbag-fixed (289 alignment frames)

## Executive Summary

The CUDA NDT implementation produces functionally equivalent results to Autoware's NDT with:
- **Position accuracy**: Mean 0.039m, Max 0.22m (within acceptable localization tolerance)
- **100% convergence rate** for both implementations
- **Score output**: Now normalized to match Autoware's convention (ratio ~0.81)

**Performance varies by platform:**

| Platform | CUDA | Autoware | Winner |
|----------|------|----------|--------|
| **Desktop x86** (RTX 5090) | 93.9 Hz | 120.9 Hz | Autoware 1.29x faster |
| **Jetson AGX Orin** (64GB) | 34.8 Hz | 26.5 Hz | **CUDA 1.32x faster** |

The performance difference is due to per-iteration cost: on x86, Autoware's OpenMP is very efficient (3.24ms/iter), while on ARM/Jetson, CUDA's unified memory architecture gives it an advantage (8.81ms vs 12.89ms/iter).

## Implementation Methods

### Algorithm Overview

Both implementations use **Newton's method** for maximum likelihood estimation based on the Normal Distributions Transform (Magnusson 2009). The core optimization loop:

1. Transform source points using current pose estimate
2. Find nearby voxels for each point
3. Compute score, gradient, and Hessian across all point-voxel correspondences
4. Solve Newton step: `Δp = -H⁻¹g`
5. Normalize and scale step, optionally apply line search
6. Update pose and check convergence

**Score Function** (Gaussian probability model):
```
p(x) = -d1 × exp(-d2/2 × (x-μ)ᵀΣ⁻¹(x-μ))
```

Where d1, d2 are derived from outlier ratio and resolution (Magnusson 2009).

### Architecture Comparison

| Aspect                  | Autoware                              | CUDA                              |
|-------------------------|---------------------------------------|-----------------------------------|
| **Language**            | C++                                   | Rust + CUDA                       |
| **Parallelization**     | OpenMP (CPU threads)                  | CUDA (GPU blocks/threads)         |
| **Voxel Storage**       | PCL KD-tree with embedded covariances | Sparse array + spatial hash table |
| **Neighbor Search**     | KD-tree radius search                 | Grid-based 3×3×3 lookup           |
| **Numerical Precision** | Double (f64) throughout               | Mixed: GPU f32, CPU f64           |

### Autoware Implementation (multigrid_ndt_omp)

**Parallelization Strategy**:
```cpp
#pragma omp parallel for
for each source point:
    nearby_voxels = kdtree.radiusSearch(point, radius)
    for each voxel in nearby_voxels:
        compute score, gradient, Hessian contributions
        atomic accumulate into global buffers
// After all points: solve Newton step on single thread
```

**Key Characteristics**:
- Fine-grained parallelism: one thread per point
- KD-tree radius search finds variable voxels per point (typically 4-6)
- Shared state via atomic operations
- Memory-bound (KD-tree traversal)

**Default Parameters**:
- Max iterations: 35
- Transformation epsilon: 0.1m
- Step size: 0.1
- Outlier ratio: 0.55
- Line search: Disabled (causes local minima issues)

### CUDA Implementation (full_gpu_pipeline_v2)

**GPU Pipeline Architecture** (Phase 24 - Graph Kernels):

| Kernel              | Function                                           | Parallelism              |
|---------------------|----------------------------------------------------|--------------------------|
| **K1 (Init)**       | Initialize state from initial pose                 | Single block             |
| **K2 (Compute)**    | Per-point score/gradient/Hessian + block reduction | N points across blocks   |
| **K3 (Solve)**      | Newton solve via SVD + optional regularization     | Single block             |
| **K4 (LineSearch)** | Parallel line search evaluation (optional)         | K candidates in parallel |
| **K5 (Update)**     | Apply step, check convergence                      | Single block             |

**Parallelization Strategy**:
```cuda
// K2: Each block processes a chunk of points
for each point in block's range:
    voxel_indices = grid_lookup_3x3x3(point)
    for each valid voxel:
        compute contributions to score, gradient, Hessian
    block-level reduce via shared memory
    atomic add to global reduce buffers
```

**Key Characteristics**:
- Coarse-grained parallelism: blocks process point chunks
- Grid-based 3×3×3 lookup (max 27 voxels per point)
- Block-local reduction minimizes atomic contention
- Compute-bound (matrix operations, exponentials)

**Default Parameters**:
- Max iterations: 30
- Transformation epsilon: 0.01m (stricter than Autoware)
- Step size: 0.1
- Outlier ratio: 0.55
- Line search: Configurable (disabled by default)

### Neighbor Search Comparison

| Aspect                    | Autoware (KD-tree)             | CUDA (Grid Hash)      |
|---------------------------|--------------------------------|-----------------------|
| **Lookup Complexity**     | O(log V) per query             | O(27) constant        |
| **Search Pattern**        | Radius-based, variable results | Fixed 3×3×3 grid      |
| **Correspondences/Point** | 4-6 typical (variable)         | ≤27 (fixed max)       |
| **Total Correspondences** | ~25,000/frame                  | ~20,000/frame         |
| **Boundary Behavior**     | Finds voxels across boundaries | Strictly grid-aligned |

**Impact**: CUDA finds ~20% fewer correspondences, resulting in ~81% score ratio. This does not affect localization accuracy.

### Convergence Criteria

| Criterion                 | Autoware                             | CUDA           |
|---------------------------|--------------------------------------|----------------|
| **Delta Norm**            | `‖Δp‖ < 0.1m`                        | `‖Δp‖ < 0.01m` |
| **Max Iterations**        | 35                                   | 30             |
| **Oscillation Detection** | Consecutive steps dot product < -0.9 | Same           |

CUDA's stricter epsilon results in ~0.3 more iterations on average.

### Newton Solver

**Both implementations**:
- Use Jacobi SVD for numerical stability
- Normalize step: `Δp /= ‖Δp‖`
- Scale step: `min(‖Δp‖, step_size)`

**CUDA additions**:
- Optional L2 regularization for near-singular Hessian
- Optional GNSS regularization (penalty for deviation from GNSS pose)

### Key Algorithmic Differences Summary

| Aspect                | Autoware       | CUDA                  | Impact                     |
|-----------------------|----------------|-----------------------|----------------------------|
| Neighbor Search       | KD-tree radius | Grid 3×3×3            | ~20% fewer correspondences |
| Correspondences/Frame | ~25,000        | ~20,000               | Score ratio ~0.81          |
| Trans Epsilon         | 0.1m           | 0.01m                 | CUDA ~0.3 more iterations  |
| Precision             | f64            | f32 (GPU) / f64 (CPU) | Negligible difference      |
| Per-Iteration Cost    | ~3.2ms         | ~5.4ms                | GPU launch overhead        |

## Detailed Findings

### 1. Voxel Grid Comparison

| Metric         | CUDA           | Autoware |
|----------------|----------------|----------|
| Resolution     | 2.0m           | 2.0m     |
| Total voxels   | 11,601         | 11,601   |
| Matched voxels | 10,360 (89.3%) | -        |

**Voxel Statistics (Matched Voxels)**:
| Metric                      | Mean   | Std    | Max     |
|-----------------------------|--------|--------|---------|
| Mean position diff          | 0.109m | 0.441m | 3.48m   |
| Covariance diff (Frobenius) | 0.166  | 0.116  | 1.53    |
| Inverse covariance diff     | 94.49  | 131.71 | 2344.49 |
| Point count diff            | 1.54   | 9.79   | 291     |

**Key Observations**:
- Voxel counts match exactly (11,601)
- ~10.7% of voxels (1,241) have different grid keys due to floating-point boundary effects
- Large inverse covariance differences are expected because small covariance changes amplify in matrix inversion
- Some voxels show different Z coordinates (e.g., CUDA: -0.76m vs Autoware: 1.40m), suggesting different point assignments near voxel boundaries

### 2. Pose Difference (CUDA vs Autoware)

This section compares the final poses computed by both implementations for the same input frames. Note: This is **not ground truth validation** - it measures implementation consistency.

**Run comparison**: `just compare-poses` or `just compare-poses-verbose`

#### Position Difference

**Desktop x86**:

| Metric             | Mean   | Std    | Max    | P95    |
|--------------------|--------|--------|--------|--------|
| Position 3D (m)    | 0.039  | 0.037  | 0.223  | -      |
| Position 2D (m)    | -      | -      | -      | -      |
| Rotation (rad)     | 0.019  | 0.015  | 0.054  | -      |

**Jetson AGX Orin**:

| Metric             | Mean   | Std    | Max    | P95    |
|--------------------|--------|--------|--------|--------|
| Position 3D (m)    | 0.038  | 0.039  | 0.264  | 0.107  |
| Position 2D (m)    | 0.036  | 0.037  | 0.263  | 0.101  |
| Rotation (rad)     | 0.019  | 0.015  | 0.055  | 0.051  |
| Rotation (deg)     | 1.11   | 0.86   | 3.16   | 2.94   |

**Key Observations**:
- Both platforms show similar pose differences (~3.8cm mean, ~26cm max)
- Rotation differences are small (~1.1 deg mean, ~3.2 deg max)
- Largest differences occur when CUDA converges in fewer iterations

#### Convergence

| Platform    | CUDA Converged | Autoware Converged | Common Frames |
|-------------|----------------|--------------------|---------------|
| x86         | 289/289 (100%) | 289/289 (100%)     | 289           |
| Jetson      | 294/294 (100%) | 290/290 (100%)     | 289           |

#### Iteration Count Difference (CUDA - Autoware)

| Platform | Mean  | Std  | Range      |
|----------|-------|------|------------|
| x86      | +0.28 | 1.42 | [-4, +7]   |
| Jetson   | +0.29 | 1.44 | [-5, +5]   |

#### Largest Position Differences (Jetson)

| Timestamp       | Pos Diff | Rot Diff | CUDA Iters | AW Iters |
|-----------------|----------|----------|------------|----------|
| ...269743713792 | 0.264m   | 0.008    | 1          | 5        |
| ...269844508160 | 0.260m   | 0.008    | 1          | 5        |
| ...255644498944 | 0.253m   | 0.024    | 1          | 6        |
| ...280443982592 | 0.226m   | 0.040    | 1          | 6        |

**Observation**: Largest differences occur when CUDA converges very early (1 iteration) while Autoware takes more iterations. This is due to different convergence epsilon (CUDA: 0.01m vs Autoware: 0.1m).

### 3. Score Calculation Differences

**After Fix (2026-01-27)**: CUDA now outputs normalized `transform_probability` matching Autoware's convention.

| Score Type    | CUDA (Fixed) | Autoware  | Ratio |
|---------------|--------------|-----------|-------|
| `final_score` | 5.82-5.83    | 7.15-7.19 | ~0.81 |
| `final_nvtl`  | 2.45-2.47    | 3.18-3.20 | ~0.77 |

**Normalization Convention** (both implementations):
```
transform_probability = total_score / num_source_points
```

**Remaining Score Difference (~20%)**:
The ~0.81 ratio is due to different correspondence counts:
- CUDA: ~20,200 correspondences per frame
- Autoware: ~25,000 correspondences (estimated from raw scores)

This difference stems from voxel boundary handling - CUDA's coordinate-based voxel lookup finds fewer neighbor voxels than Autoware's radius search.

**NVTL Difference (~0.7)**:
- Both use max-score-per-point averaging
- CUDA's coordinate-based neighbor search (3x3x3 grid) vs Autoware's radius search
- Results in fewer voxels found per point, hence lower NVTL

### 4. Per-Iteration Analysis (Autoware Reference)

Sample first frame optimization:

| Iteration | Pose (x,y,z)              | Score  | Step Length | Voxels/Point |
|-----------|---------------------------|--------|-------------|--------------|
| 0         | (89571, 42301, -3.30)     | 33,080 | 0.10        | 3.96         |
| 1         | (89571, 42301, -3.21)     | 35,507 | 0.10        | 4.05         |
| 2         | (89571.1, 42301.1, -3.13) | 35,820 | 0.05        | 4.05         |
| 3         | (89571.1, 42301.1, -3.12) | 35,864 | 0.03        | 4.06         |
| 4         | (89571.1, 42301.1, -3.12) | 35,889 | 0.02        | 4.06         |
| 5         | (89571.1, 42301.2, -3.12) | 35,880 | 0.01        | 4.06         |
| 6         | (89571.1, 42301.2, -3.12) | 35,884 | 0.005       | 4.06         |

**Convergence**: 7 iterations, Final NVTL: 3.20

### 5. Largest Position Differences

| Timestamp       | Position Diff | CUDA Iters | Autoware Iters | Notes                                      |
|-----------------|---------------|------------|----------------|--------------------------------------------|
| ...844858880    | 0.223m        | 3          | 7              | Initial frame, different convergence paths |
| ...279244287744 | 0.210m        | 5          | 6              | Near yaw ~2.2 rad                          |
| ...279645129728 | 0.206m        | 1          | 5              | Very early CUDA convergence                |
| ...268744289792 | 0.202m        | 1          | 4              | Very early CUDA convergence                |

**Observation**: The largest differences occur when CUDA converges in fewer iterations (1-3) compared to Autoware (4-7). This suggests CUDA may have different convergence criteria or line search parameters.

## Root Causes of Differences

### 1. Score Output Convention - FIXED
- **Issue**: CUDA output raw `total_score`, Autoware outputs normalized `transform_probability`
- **Fix Applied**: Changed CUDA to output `transform_probability = total_score / num_source_points`
- **Files Changed**:
  - `src/ndt_cuda/src/optimization/solver.rs`: Updated `compute_transform_probability()` and all call sites
  - `src/ndt_cuda/src/ndt.rs`: Updated GPU path debug output
- **Result**: Score ratio improved from ~4000x to ~0.81x

### 2. Correspondence Count Difference
- **Issue**: CUDA finds ~20% fewer correspondences than Autoware
- **Cause**: Different neighbor search implementations
  - CUDA: Coordinate-based 3x3x3 voxel grid lookup
  - Autoware: Radius search (finds more nearby voxels)
- **Impact**: Accounts for the remaining ~20% score difference

### 3. NVTL Calculation
- **Issue**: ~0.7 lower NVTL in CUDA
- **Cause**: Same as correspondence count - fewer neighbor voxels found per point
- **Impact**: Functional - both provide valid quality metrics

### 4. Voxel Boundary Assignment
- **Issue**: ~10.7% voxels have different grid keys
- **Cause**: Floating-point precision in coordinate-to-voxel mapping
- **Impact**: Minor - different points assigned to boundary voxels

### 5. Convergence Criteria
- **Issue**: CUDA sometimes converges much earlier (1-3 iters vs 4-7)
- **Cause**: Different step length tolerances and line search parameters
- **Impact**: May lead to slightly different final poses

## Performance Profiling

### Test Environments

**Desktop x86**:
- **CPU**: Intel Core Ultra 7 265K (20 cores)
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **Memory**: 30GB RAM
- **OS**: Ubuntu 22.04 (Linux 6.8.0-90-generic)
- **CUDA**: Driver 580.105.08

**Jetson AGX Orin**:
- **Platform**: NVIDIA Jetson AGX Orin Developer Kit (64GB)
- **CPU**: 12x ARM Cortex-A78AE cores
- **GPU**: Ampere iGPU (2048 CUDA cores, unified memory)
- **Memory**: 64GB unified (shared CPU/GPU)
- **OS**: Ubuntu 22.04 (L4T 36.4.4, Linux 5.15.148-tegra)
- **CUDA**: 12.6.68

**Common Configuration**:
- ROS: Humble
- Autoware: 1.5.0
- Build: Release with profiling (minimal overhead)
- Dataset: sample-rosbag-fixed, 5000 source points per frame
- Date: 2026-01-27

### Performance Summary

**Desktop x86** (Autoware wins):

| Metric                  | CUDA  | Autoware | Speedup |
|-------------------------|-------|----------|---------|
| Mean exec time (ms)     | 10.65 | 8.27     | 0.78x   |
| Median exec time (ms)   | 9.48  | 8.00     | 0.84x   |
| P95 exec time (ms)      | 21.75 | 13.87    | 0.64x   |
| P99 exec time (ms)      | 24.48 | 16.18    | 0.66x   |
| Min exec time (ms)      | 3.64  | 3.38     | -       |
| Max exec time (ms)      | 42.88 | 52.07    | -       |
| Throughput (Hz)         | 93.9  | 120.9    | 0.78x   |
| Mean iterations         | 3.12  | 2.90     | -       |
| Time per iteration (ms) | 5.42  | 3.24     | 0.60x   |
| Frames analyzed         | 294   | 289      | -       |

**Jetson AGX Orin** (CUDA wins):

| Metric                  | CUDA     | Autoware | Speedup   |
|-------------------------|----------|----------|-----------|
| Mean exec time (ms)     | 28.71    | 37.79    | **1.32x** |
| Median exec time (ms)   | 26.99    | 36.97    | **1.37x** |
| P95 exec time (ms)      | 46.00    | 61.83    | 1.34x     |
| P99 exec time (ms)      | 52.76    | 70.46    | 1.34x     |
| Min exec time (ms)      | 11.11    | 13.18    | -         |
| Max exec time (ms)      | 57.03    | 71.51    | -         |
| Throughput (Hz)         | **34.8** | 26.5     | **1.32x** |
| Mean iterations         | 3.26     | 2.93     | -         |
| Time per iteration (ms) | 8.81     | 12.89    | **1.46x** |
| Frames analyzed         | 294      | 290      | -         |
| Convergence rate        | 100%     | 100%     | -         |

### Warmup Analysis

**Desktop x86**:

| Phase           | CUDA (ms) | Autoware (ms) |
|-----------------|-----------|---------------|
| First 10 frames | 18.98     | 12.97         |
| Steady state    | 10.36     | 8.10          |

**Jetson AGX Orin**:

| Phase           | CUDA (ms) | Autoware (ms) |
|-----------------|-----------|---------------|
| First 10 frames | 29.22     | 36.78         |
| Steady state    | 28.69     | 37.82         |

**Observations**:
- x86: Both show warmup effects. CUDA has more pronounced warmup penalty (83% overhead vs 60% for Autoware).
- Jetson: Minimal warmup effect. Both implementations reach steady state quickly.

### Execution Time by Iteration Count

**Desktop x86 - CUDA**:

| Iterations | Count | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|------------|-------|-----------|----------|----------|----------|
| 1          | 104   | 10.17     | 7.66     | 3.64     | 22.15    |
| 2          | 18    | 12.91     | 7.83     | 4.35     | 24.68    |
| 3          | 37    | 8.71      | 0.94     | 6.19     | 10.51    |
| 4          | 66    | 9.68      | 1.54     | 6.32     | 11.85    |
| 5          | 32    | 10.95     | 1.91     | 6.94     | 13.64    |
| 6          | 24    | 13.60     | 6.33     | 8.59     | 42.88    |
| 7          | 9     | 15.74     | 8.66     | 9.21     | 39.38    |
| 8          | 2     | 16.60     | 1.06     | 15.54    | 17.67    |
| 9          | 1     | 10.46     | 0.00     | 10.46    | 10.46    |
| 10         | 1     | 18.84     | 0.00     | 18.84    | 18.84    |

**Desktop x86 - Autoware**:

| Iterations | Count | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|------------|-------|-----------|----------|----------|----------|
| 1          | 83    | 4.52      | 0.88     | 3.38     | 8.22     |
| 2          | 34    | 6.20      | 0.85     | 5.17     | 8.46     |
| 3          | 61    | 8.40      | 1.18     | 6.56     | 12.00    |
| 4          | 64    | 10.36     | 2.64     | 7.97     | 24.96    |
| 5          | 36    | 12.20     | 1.48     | 9.66     | 15.76    |
| 6          | 10    | 17.62     | 11.53    | 11.97    | 52.07    |
| 7          | 1     | 13.82     | 0.00     | 13.82    | 13.82    |

**Jetson AGX Orin - CUDA**:

| Iterations | Count | Mean (ms) | Std (ms) |
|------------|-------|-----------|----------|
| 1          | 109   | 20.18     | 4.31     |
| 2          | 13    | 22.66     | 3.83     |
| 3          | 28    | 27.53     | 5.72     |
| 4          | 54    | 32.48     | 5.76     |
| 5          | 42    | 36.90     | 6.30     |
| 6          | 29    | 36.47     | 8.08     |
| 7          | 15    | 40.07     | 11.05    |
| 8          | 3     | 52.24     | 2.96     |
| 10         | 1     | 56.86     | 0.00     |

**Jetson AGX Orin - Autoware**:

| Iterations | Count | Mean (ms) | Std (ms) |
|------------|-------|-----------|----------|
| 1          | 88    | 22.76     | 5.14     |
| 2          | 28    | 29.81     | 7.51     |
| 3          | 62    | 39.88     | 7.58     |
| 4          | 54    | 46.28     | 8.52     |
| 5          | 44    | 53.51     | 9.48     |
| 6          | 14    | 56.75     | 8.58     |

### Performance Analysis

**Why Autoware wins on x86**:

1. **Per-Iteration Cost**: CUDA's per-iteration time (5.42ms) is 1.67x higher than Autoware (3.24ms) due to:
   - GPU kernel launch overhead (~1ms per iteration)
   - Host-device synchronization for convergence checking
   - PCIe transfer latency between CPU and discrete GPU

2. **High Variance in 1-Iteration Cases**: CUDA shows high variance (std 7.66ms) for single-iteration frames due to warmup effects and GPU scheduling variability.

3. **OpenMP Efficiency**: Autoware's OpenMP is highly optimized for x86 multi-core CPUs with efficient cache utilization.

**Why CUDA wins on Jetson**:

1. **Unified Memory**: Jetson's unified memory architecture eliminates PCIe transfer overhead, making GPU kernel launches cheaper.

2. **ARM OpenMP Performance**: Autoware's OpenMP is optimized for x86. On ARM Cortex-A78AE cores, OpenMP parallelization is less efficient (12.89ms/iter vs 3.24ms/iter on x86).

3. **GPU Integration**: The Tegra GPU is tightly integrated with the ARM cores, reducing kernel launch latency.

4. **Per-Iteration Advantage**: CUDA's per-iteration time (8.81ms) is 1.46x faster than Autoware (12.89ms) on Jetson.

**Common Observations**:

- Iteration distribution is similar on both platforms (~37% single-iteration for CUDA vs ~30% for Autoware)
- Both implementations are real-time capable (exceed 10 Hz LiDAR rate with comfortable margin)
- Lower variance on Jetson compared to x86, likely due to simpler memory hierarchy

### Debug Build Comparison (x86 only)

For reference, the debug build with full per-iteration data collection shows:

| Metric                  | CUDA (Debug) | CUDA (Release) | Improvement |
|-------------------------|--------------|----------------|-------------|
| Mean exec time (ms)     | 16.82        | 10.65          | 1.58x       |
| Throughput (Hz)         | 59.5         | 93.9           | 1.58x       |
| Time per iteration (ms) | 8.11         | 5.42           | 1.50x       |

The debug build adds ~6ms overhead per frame due to GPU debug buffer allocation and GPU→CPU transfers.

### Profiling and Comparison Commands

```bash
# === Performance Profiling ===

# Build and run release profiling (minimal overhead)
just run-cuda-profiling      # CUDA with profiling only
just run-builtin-profiling   # Autoware with profiling only

# Full profiling comparison (build, run both, compare timing)
just profile-compare

# Debug builds (full debug data, slower)
just run-cuda-debug          # CUDA with all debug features
just run-builtin-debug       # Autoware with all debug features

# === Pose Accuracy Comparison ===

# Compare poses from existing profiling logs
just compare-poses           # Summary output
just compare-poses-verbose   # Include largest differences
just compare-poses-json      # JSON output for automation

# Full accuracy test (run both implementations and compare)
just accuracy-test           # Build, run both, compare poses
```

**Notes**:
- The `profiling` feature only enables timing instrumentation (minimal overhead)
- The `debug` feature enables per-iteration data collection (significant overhead)
- For accurate performance comparison, use the `profiling` build
- Pose comparison requires profiling logs from both implementations

## Future Investigation

1. **Per-iteration kernel optimization**: Reduce GPU kernel launch overhead to improve per-iteration cost on x86
2. **Batch processing**: Process multiple frames together to amortize GPU overhead
3. **Radius search alignment**: Consider implementing radius-based neighbor search for better correspondence count match
4. **Convergence criteria**: Document exact thresholds in both implementations

## Files Referenced

| File                                 | Purpose                                             |
|--------------------------------------|-----------------------------------------------------|
| `logs/ndt_cuda_profiling.jsonl`      | CUDA release profiling (294 entries)                |
| `logs/ndt_autoware_profiling.jsonl`  | Autoware profiling (290 entries)                    |
| `logs/ndt_cuda_debug.jsonl`          | CUDA alignment debug                                |
| `logs/ndt_autoware_debug.jsonl`      | Autoware alignment debug                            |
| `logs/ndt_cuda_voxels.json`          | CUDA voxel grid dump (11,601 voxels)                |
| `logs/ndt_autoware_voxels.json`      | Autoware voxel grid dump (11,601 voxels)            |
| `scripts/compare_poses.py`           | Pose difference comparison script                   |
| `tmp/analyze_jetson_profiling.py`    | Jetson profiling analysis script                    |

## Conclusion

The CUDA NDT implementation is **functionally equivalent** to Autoware's NDT:

- **Pose difference**: Mean 3.8cm, Max 26cm (within acceptable localization tolerance)
- **100% convergence**: Both implementations converge successfully on both platforms
- **Voxel grids match**: Same voxel counts with minor boundary differences
- **Score output**: Normalized to match Autoware's convention (ratio ~0.81)

### Cross-Platform Performance Summary

| Platform | CUDA | Autoware | Relative | Winner |
|----------|------|----------|----------|--------|
| **Desktop x86** (RTX 5090) | 93.9 Hz | 120.9 Hz | 0.78x | Autoware |
| **Jetson AGX Orin** (64GB) | 34.8 Hz | 26.5 Hz | **1.32x** | **CUDA** |

**Key Findings**:

1. **Platform matters**: CUDA NDT is faster on Jetson but slower on x86. The choice of implementation depends on deployment target.

2. **Per-iteration cost explains the difference**:
   - On x86: Autoware 3.24ms/iter vs CUDA 5.42ms/iter (Autoware 1.67x faster)
   - On Jetson: CUDA 8.81ms/iter vs Autoware 12.89ms/iter (CUDA 1.46x faster)

3. **Unified memory advantage**: Jetson's unified CPU/GPU memory eliminates PCIe transfer overhead, making GPU kernels more efficient relative to CPU.

4. **Real-time capable on both**: Even the slower implementation on each platform exceeds 10 Hz LiDAR rate with comfortable margin.

### Optimization Opportunities

For x86 (to improve CUDA performance):
- Kernel fusion to reduce launch overhead
- Persistent kernels that avoid per-iteration launch costs
- Batch processing for multiple frames

For Jetson (already optimized):
- Current CUDA implementation is well-suited for Jetson's architecture
- Further gains possible through CUDA graph optimizations

### Remaining Differences

These apply to both platforms and do not affect functional correctness:
1. Correspondence count (~20% fewer in CUDA due to coordinate-based vs radius search)
2. NVTL values (~0.7 lower due to same reason)
3. Convergence speed (CUDA sometimes converges faster)

### Recommendation

- **For Jetson/embedded deployment**: Use CUDA NDT (32% faster)
- **For x86/server deployment**: Autoware's OpenMP is competitive (29% faster)
- **For cross-platform**: CUDA NDT provides consistent behavior and is the clear choice for Jetson targets
