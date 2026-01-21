# Debug Infrastructure

This document describes the debug data categories in CUDA NDT and the feature-gating implementation.

## Implementation: Feature-Gated Debug System ✅

The debug system uses compile-time feature flags for zero-overhead production builds.

**Environment variables** (retained for output paths only):
| Variable               | Purpose                                                       | Feature Required |
|------------------------|---------------------------------------------------------------|------------------|
| `NDT_DEBUG_FILE`       | JSONL output path (default: `/tmp/ndt_cuda_debug.jsonl`)      | `debug-output`   |
| `NDT_DUMP_VOXELS_FILE` | Voxel dump output path (default: `/tmp/ndt_cuda_voxels.json`) | `debug-voxels`   |

**Benefits**:
- Debug code completely eliminated when features disabled (zero overhead)
- Selective activation: enable only needed debug categories
- Compile-time guarantees: no accidental debug paths in production

## Debug Data Categories

### Category 1: `debug-iteration` — Per-Iteration Optimization Data

**Purpose**: Compare optimization trajectory with Autoware for algorithm validation.

**Data Collected**:
- Pose at each iteration (6 values)
- Score, gradient (6), Hessian (36)
- Newton step, step length, direction reversal
- Number of correspondences

**Files**:
- `ndt_cuda/src/optimization/debug.rs` — `IterationDebug`, `AlignmentDebug` structs
- `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` — GPU debug buffer allocation and parsing
- `ndt_cuda/src/optimization/solver.rs` — Debug data population

**Overhead**:
- GPU: Extra buffer allocation (~1KB per iteration × max_iterations)
- GPU: Debug data written to buffer in kernel
- CPU: Parsing debug buffer after alignment (~0.1ms)

**Current Gate**: `#[cfg(feature = "debug-iteration")]` (compile-time) ✅

---

### Category 2: `debug-output` — JSONL File Output

**Purpose**: Write debug data to files for post-hoc analysis.

**Data Written**:
- Alignment debug records (per scan callback)
- Initial pose debug records (per init request)
- Run start markers

**Files**:
- `cuda_ndt_matcher/src/main.rs:564-582` — Debug file initialization
- `cuda_ndt_matcher/src/main.rs:927-937` — Alignment JSON output
- `cuda_ndt_matcher/src/initial_pose.rs:66-81` — Init pose JSON output

**Overhead**:
- JSON serialization (~0.5ms per alignment)
- File I/O (append to JSONL)

**Current Gate**: `#[cfg(feature = "debug-output")]` (compile-time) ✅

---

### Category 3: `debug-voxels` — Voxel Grid Dump

**Purpose**: Dump voxel grid to JSON for comparison with Autoware.

**Data Written**:
- Resolution, voxel count
- Per-voxel: mean, covariance, inverse covariance, point count

**Files**:
- `cuda_ndt_matcher/src/ndt_manager.rs:24-94` — `dump_voxel_data()` function
- `cuda_ndt_matcher/src/ndt_manager.rs:197` — Trigger check

**Overhead**:
- JSON serialization of entire voxel grid (~50-500ms for large grids)
- File I/O

**Current Gate**: `#[cfg(feature = "debug-voxels")]` (compile-time) ✅

---

### Category 4: `debug-cov` — GPU vs CPU Covariance Comparison

**Purpose**: Validate GPU covariance computation against CPU reference.

**Data Computed**:
- Downloads GPU covariance sums
- Computes CPU reference covariance
- Compares trace ratios and max differences

**Files**:
- `ndt_cuda/src/voxel_grid/gpu/pipeline.rs:465-536` — Comparison logic

**Overhead**:
- GPU→CPU transfer of covariance data (~1ms)
- CPU covariance computation (~5-50ms depending on grid size)
- When disabled: Zero (code not compiled)

**Current Gate**: `#[cfg(feature = "debug-cov")]` (compile-time) ✅

---

### Category 5: `debug-vpp` — Voxel-Per-Point Distribution

**Purpose**: Track how many voxels each source point matches.

**Data Computed**:
- Count of points with 0, 1, 2, 3+ voxel matches

**Files**:
- `ndt_cuda/src/derivatives/cpu.rs:432-444` — Distribution tracking

**Overhead**:
- Counter increments per point (~negligible)
- Only compiled in debug builds (`#[cfg(debug_assertions)]`)

**Current Gate**: `#[cfg(feature = "debug-vpp")]` (compile-time) ✅

---

### Category 6: `debug-markers` — Visualization Features

**Purpose**: Publish ROS markers and colored point clouds for visual debugging.

**Data Published**:
- Pose history markers (trajectory visualization with color gradient)
- Voxel score point clouds (per-point NDT scores with RGB coloring)

**Files**:
- `cuda_ndt_matcher/src/main.rs:1243-1261` — Pose history markers
- `cuda_ndt_matcher/src/main.rs:1282-1303` — Voxel score visualization
- `cuda_ndt_matcher/src/visualization.rs` — Visualization utilities

**Overhead**:
- When disabled: Zero (code not compiled)
- When enabled: Per-point score computation (~1-2ms) + marker generation

**Current Gate**: `#[cfg(feature = "debug-markers")]` (compile-time) ✅

---

### Category 7: `profiling` — Detailed Timing (Already Feature-Gated)

**Purpose**: Measure time spent in each phase of alignment.

**Data Collected**:
- Per-iteration timing (transform, correspondence, derivatives, solver)
- Total alignment timing breakdown

**Files**:
- `ndt_cuda/src/timing.rs` — `Timer`, `TimingCollector`
- `ndt_cuda/src/optimization/debug.rs` — `IterationTimingDebug`, `AlignmentTimingDebug`

**Overhead**:
- When disabled: Zero (compiles to no-ops)
- When enabled: ~1μs per timer call

**Current Gate**: `#[cfg(feature = "profiling")]` (compile-time) ✅

---

## Implemented Feature Gates ✅

All debug categories are now feature-gated for zero-overhead when disabled:

```toml
# ndt_cuda/Cargo.toml
[features]
default = ["cuda"]
cuda = []
profiling = []              # Detailed timing instrumentation
debug-iteration = []        # Per-iteration data (pose, score, gradient, Hessian)
debug-cov = []              # GPU vs CPU covariance comparison
debug-vpp = []              # Voxel-per-point distribution tracking
debug = ["debug-iteration", "debug-cov", "debug-vpp", "profiling"]
```

```toml
# cuda_ndt_matcher/Cargo.toml
[features]
default = []
debug-output = ["ndt_cuda/debug-iteration"]  # JSONL output for alignment + pose init
debug-voxels = []                            # Voxel grid JSON dump
debug-markers = ["ndt_cuda/debug-iteration"] # Pose markers + voxel score visualization
debug = ["debug-output", "debug-voxels", "debug-markers", "ndt_cuda/debug"]
profiling = ["ndt_cuda/profiling"]
```

### Build Commands

```bash
just build-cuda-debug       # Build with all debug features
just build-cuda-profiling   # Build with profiling only
just run-cuda-debug         # Run with full debug output
```

### Benefits

1. **Zero overhead**: Debug code not compiled in release builds
2. **Smaller binaries**: No unused debug infrastructure
3. **Clearer intent**: Feature flags are explicit in Cargo.toml
4. **Better optimization**: Compiler can eliminate dead code paths
5. **CI integration**: Can have separate debug builds for testing

### Remaining Environment Variables

Environment variables are still used for output paths (when features are enabled):

| Variable               | Description                                                    |
|------------------------|----------------------------------------------------------------|
| `NDT_DEBUG_FILE`       | Debug JSONL output path (default: `/tmp/ndt_cuda_debug.jsonl`) |
| `NDT_DUMP_VOXELS_FILE` | Voxel dump output path (default: `/tmp/ndt_cuda_voxels.json`)  |

## File Summary

| File                                                | Category | Gate                              |                   |
|-----------------------------------------------------|----------|-----------------------------------|-------------------|
| `ndt_cuda/src/optimization/debug.rs`                | 1, 6     | `debug-iteration`, `profiling` ✅ |                   |
| `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` | 1        | `debug-iteration` ✅              |                   |
| `ndt_cuda/src/optimization/solver.rs`               | 1        | `debug-iteration` ✅              |                   |
| `ndt_cuda/src/ndt.rs`                               | 1, 2     | `debug-iteration` ✅              |                   |
| `ndt_cuda/src/timing.rs`                            | 6        | `profiling` ✅                    |                   |
| `ndt_cuda/src/voxel_grid/gpu/pipeline.rs`           | 4        | `debug-cov` ✅                    |                   |
| `ndt_cuda/src/derivatives/cpu.rs`                   | 5        | `debug-vpp` ✅                    |                   |
| `cuda_ndt_matcher/src/main.rs`                      | 2        | `debug-output` ✅                 |                   |
| `cuda_ndt_matcher/src/initial_pose.rs`              | 2        | `debug-output` ✅                 |                   |
| `cuda_ndt_matcher/src/ndt_manager.rs`               | 2, 3     | `debug-output`, `debug-voxels` ✅ |                   |
| `cuda_ndt_matcher/src/dual_ndt_manager.rs`          | 2        | `debug-output` ✅                 |                   |
| `cuda_ndt_matcher/src/initial_pose.rs`              | 2        | 30-81, 180, 397-413               | `NDT_DEBUG`       |
| `cuda_ndt_matcher/src/ndt_manager.rs`               | 3        | 24-94, 197                        | `NDT_DUMP_VOXELS` |
