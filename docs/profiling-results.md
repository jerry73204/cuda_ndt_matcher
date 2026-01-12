# NDT Profiling Results

This document captures profiling results comparing the CUDA NDT implementation against Autoware's builtin NDT scan matcher.

## Test Environment

- **Date**: 2026-01-10
- **Hardware**: NVIDIA GPU (CUDA enabled)
- **Dataset**: Autoware sample rosbag (~23 seconds of driving data)
- **Map**: sample-map-rosbag (point cloud map)

## Execution Time Comparison

| Metric   | Autoware (OpenMP) | CUDA GPU     | Ratio           |
|----------|-------------------|--------------|-----------------|
| **Mean** | **2.48 ms**       | **13.07 ms** | **5.3x slower** |
| Median   | 2.42 ms           | 15.86 ms     | 6.5x slower     |
| Stdev    | 1.20 ms           | 8.24 ms      | -               |
| Min      | 1.08 ms           | 2.00 ms      | 1.9x slower     |
| Max      | 16.46 ms          | 27.02 ms     | 1.6x slower     |
| P95      | 3.57 ms           | 24.46 ms     | 6.9x slower     |
| P99      | 3.98 ms           | 26.34 ms     | 6.6x slower     |

**Sample sizes**: Autoware: 221 alignments, CUDA: 225 alignments (from rosbag recording)

## Root Cause Analysis

### ~~Issue 1: Duplicate Message Processing (Critical)~~ - RESOLVED

**Update (2026-01-10)**: This was a false alarm caused by analyzing accumulated debug data.

The initial analysis showed:
| Metric                     | Value        |
|----------------------------|--------------|
| Total alignments in debug file | 4,668    |
| Unique timestamps          | 449          |
| Apparent duplicate ratio   | ~10x         |

**Actual cause**: The debug file (`/tmp/ndt_cuda_debug.jsonl`) was using `append` mode, accumulating data from 19 separate test runs. Each individual run processes timestamps correctly:

| Run | Entries | Unique Timestamps | Dedup Working? |
|-----|---------|-------------------|----------------|
| 1   | 181     | 181               | Yes            |
| 2   | 260     | 260               | Yes            |
| ... | ...     | ...               | Yes            |
| 19  | 268     | 268               | Yes            |

**Resolution**:
- The deduplication logic IS working correctly (HashSet-based timestamp tracking)
- Fixed the debug file to truncate at startup instead of appending
- Added callback counters for future debugging

**Verified**: Within a single run, each timestamp is processed exactly once. The rosbag has ~260 unique scans, matching the alignment count per run.

### Issue 1: Low Convergence Rate (Primary Performance Issue)

| Metric            | Autoware | CUDA       |
|-------------------|----------|------------|
| Convergence rate  | ~100%    | 51.9%      |
| Mean iterations   | 3-5      | 15.2       |
| Median iterations | 3-4      | 8          |
| Max iterations    | ~10      | 30 (limit) |
| Hit max iterations| ~0%      | 48.1%      |

**Impact**: Non-converging alignments run for all 30 iterations, significantly increasing execution time.

**Execution time distribution** (from latest run):
| Category    | Count | Percentage | Notes |
|-------------|-------|------------|-------|
| Fast (<5ms) | 66    | 29.3%      | Converged quickly (1-3 iter) |
| Medium (5-15ms) | 37 | 16.4%   | Converged with more iterations |
| Slow (15-25ms) | 117 | 52.0%   | Hit max iterations (30) |
| Very slow (>=25ms) | 5 | 2.2%  | Worst cases |

**Possible causes**:
- Derivative computation differences
- Step size or line search behavior
- Initial pose quality differences
- Voxel search radius differences

### Issue 2: GPU Overhead

Even without the above issues, GPU-based NDT has inherent overhead:
- Memory transfer (CPU → GPU → CPU)
- Kernel launch latency
- Small batch sizes (single scan) don't fully utilize GPU parallelism

## CUDA NDT Detailed Statistics

From `/tmp/ndt_cuda_debug.jsonl` analysis:

| Metric                       | Value                                |
|------------------------------|--------------------------------------|
| Mean correspondences         | 3,447 points                         |
| Mean score                   | 5,363                                |
| Mean NVTL                    | 2.16                                 |
| Convergence status breakdown | 56.8% Converged, 43.2% MaxIterations |

## Recommendations

### High Priority

1. **Investigate convergence issues** (Primary focus)
   - Compare derivative values between CUDA and Autoware at each iteration
   - Verify Hessian computation matches Autoware's implementation
   - Check if step size clamping behavior differs
   - Analyze why 48% of alignments hit max iterations

2. **Analyze fast vs slow trajectories**
   - First ~87 entries are fast (<5ms), then majority become slow
   - Investigate what changes in the trajectory at that point
   - Could be map loading, entering challenging area, or initial pose drift

### Medium Priority

3. **Optimize GPU memory transfers**
   - Keep voxel grid on GPU between alignments
   - Use pinned memory for faster transfers
   - Consider async transfers with CUDA streams

4. **Batch processing**
   - Process multiple initial pose hypotheses in parallel on GPU
   - This is where GPU excels over CPU

### Low Priority

5. **Profile individual phases**
   - Enable `profiling` feature: `cargo build --features profiling`
   - Identify which phase (voxel search, derivatives, solver) is slowest

## Data Files

| File                                           | Description                   |
|------------------------------------------------|-------------------------------|
| `rosbag/builtin_20260110_023936/`              | Autoware NDT recorded output  |
| `rosbag/builtin_20260110_023936/exe_times.txt` | Extracted execution times     |
| `rosbag/cuda_20260110_121409/`                 | CUDA NDT recorded output      |
| `rosbag/cuda_20260110_121409/exe_times.txt`    | Extracted execution times     |
| `/tmp/ndt_cuda_debug.jsonl`                    | Detailed CUDA iteration debug |

## Reproducing Results

```bash
# Run Autoware builtin NDT
just run-builtin

# Run CUDA NDT
just run-cuda

# Extract execution times
python3 tmp/extract_exe_times.py rosbag/builtin_*/
python3 tmp/extract_exe_times.py rosbag/cuda_*/

# Run comparison analysis
python3 tmp/analyze_comparison.py
```

## Conclusion

The CUDA NDT implementation is currently **5.3x slower** than Autoware's OpenMP implementation. The primary cause is:

1. **Low convergence rate** (51.9% vs ~100%) - 48% of alignments hit the 30-iteration limit
2. **GPU overhead** - inherent but can be optimized

**Note**: The previously reported "duplicate message processing" issue was a false alarm caused by analyzing accumulated debug data from multiple test runs. The deduplication logic is working correctly.

**Key insight**: When the CUDA implementation converges quickly (entries 0-86), execution times are 2-3ms (comparable to or faster than Autoware). The performance gap is caused by the 52% of alignments that hit max iterations.

Focus should be on understanding why convergence degrades partway through the trajectory. Potential causes:
- Map tile transitions affecting voxel quality
- Initial pose drift accumulating errors
- Entering areas with less distinctive map features
