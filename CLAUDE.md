# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

CUDA/Rust re-implementation of Autoware's `ndt_scan_matcher` using CubeCL for GPU compute.

**Target**: Autoware 1.5.0

**Reference implementation**: `tests/comparison/autoware_core/localization/autoware_ndt_scan_matcher/`

**Documentation**:
- `docs/autoware-comparison.md` - Feature comparison and GPU acceleration status
- `docs/autoware_vs_cuda_comparison.md` - Detailed CUDA vs Autoware comparison findings
- `docs/roadmap/` - Implementation phases and status
- `docs/profiling-results.md` - Performance analysis
- `docs/optimization-approaches.md` - Potential optimizations for iteration reduction

## Build Commands

**Always use justfile** (never run colcon directly):

```bash
just build    # colcon build with --release
just clean    # rm -rf build install log target
just lint     # Format check + clippy (requires build first)
just test     # Run tests (requires build first)
```

**Running cargo directly** (for specific tests):
```bash
cargo --config build/ros2_cargo_config.toml test -p ndt_cuda --lib test_name
```

## Running

```bash
# Demo mode with logging
just run-cuda      # CUDA NDT
just run-builtin   # Autoware NDT (baseline)
```

See `docs/rosbag-replay-guide.md` for custom rosbag testing.

## Project Structure

```
src/
├── ndt_cuda/           # Core NDT library (CubeCL GPU kernels)
├── cuda_ffi/           # CUDA FFI bindings (CUB primitives)
├── cuda_ndt_matcher/   # ROS 2 node
└── cuda_ndt_matcher_launch/  # Launch files and config
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NDT_USE_GPU=0` | Force CPU mode (default: 1 for GPU) |
| `NDT_DEBUG=1` | Enable debug JSONL output |
| `NDT_DEBUG_VPP=1` | Log voxel-per-point distribution |
| `NDT_DEBUG_COV=1` | Compare GPU vs CPU covariance (output via tracing::debug) |
| `NDT_DUMP_VOXELS=1` | Dump voxel data to JSON for comparison |
| `NDT_DUMP_VOXELS_FILE` | Output path (default: `/tmp/ndt_cuda_voxels.json`) |
| `CUDA_ARCH` | CUDA compute capability (default: 87 for Jetson Orin) |

**Pipeline config**: `PipelineV2Config::enable_debug = true` collects per-iteration debug data (score, gradient, Hessian, step size) from the graph kernels with zero overhead when disabled.

## Cargo Features

**ndt_cuda crate**:
| Feature | Description |
|---------|-------------|
| `cuda` | Enable CUDA backend (default) |
| `profiling` | Enable timing instrumentation (minimal overhead) |
| `debug-iteration` | Enable per-iteration data collection (adds overhead) |
| `debug-cov` | Enable GPU vs CPU covariance comparison |
| `debug-vpp` | Enable voxel-per-point distribution logging |
| `debug` | All debug features combined (`debug-iteration` + `debug-cov` + `debug-vpp` + `profiling`) |
| `test-verbose` | Enable verbose println output in tests |

**cuda_ffi crate**:
| Feature | Description |
|---------|-------------|
| `test-verbose` | Enable verbose println output in tests |

Enable features with: `cargo test --features test-verbose` or `cargo build --features profiling`

**Important**: Use `profiling` for performance measurement. The `debug` feature adds significant overhead from per-iteration data collection.

## ROS 2 Integration Notes

**EKF Subscription QoS**: Uses depth 100 (matching Autoware) to buffer messages during node initialization. With depth 1, early EKF messages are lost before `spin()` starts processing callbacks.

**Initial Pose**: Demo scripts always enable `user_defined_initial_pose` for reproducible testing. Without this, the EKF initializes to an unknown state. The default pose is set in `ndt_replay_simulation.launch.xml`.

**SmartPoseBuffer**: Rejects interpolation when target timestamp is before first pose (matches Autoware behavior). Does NOT use fallback to first pose.

## Coding Conventions

- **Logging**: Use `rclrs::log_*!` in `cuda_ndt_matcher`, `tracing::*!` in `ndt_cuda`
- **Transforms**: Use nalgebra for all rotation/quaternion math
- **Format strings**: Use named parameters: `println!("{e}")` not `println!("{}", e)`

## CubeCL Limitations

1. **No dynamic array indexing**: Use fully unrolled loops instead of `arr[i as usize]`
2. **Parameter count limit**: Kernels with >12 parameters fail; combine buffers
3. **No `as usize`**: Use explicit indices

## Jetson Platform Notes

**cudarc CUDA version feature**: The `cudarc` crate requires a CUDA version feature that matches the symbols available in Jetson's Tegra CUDA libraries. Jetson's libcuda.so and libcusolver.so are missing some symbols that desktop CUDA has:

| Symbol | Required by | Available on Jetson |
|--------|-------------|---------------------|
| `cuEventElapsedTime_v2` | `cuda-12080`+ | ❌ No |
| `cusolverDnXgeev` | `cuda-12060`+ | ❌ No |

**Solution**: Use `cuda-12050` feature for cudarc to avoid missing symbol panics:
```toml
cudarc = { version = "0.18", features = ["cuda-12050", ...] }
```

Despite Jetson having CUDA 12.6 installed, `cuda-12050` is the highest compatible feature because Tegra drivers have a limited API surface.

## Key Files

| File | Purpose |
|------|---------|
| `ndt_cuda/src/optimization/full_gpu_pipeline_v2.rs` | Full GPU Newton with line search (graph kernels) |
| `ndt_cuda/src/optimization/debug.rs` | Per-iteration debug data structures |
| `cuda_ffi/csrc/ndt_graph_kernels.cu` | CUDA graph kernels (K1-K5) - Phase 24 |
| `cuda_ffi/csrc/ndt_graph_common.cuh` | Buffer layouts and configuration for graph kernels |
| `cuda_ffi/src/graph_ndt.rs` | Rust FFI bindings for graph kernels |
| `cuda_ffi/csrc/persistent_ndt.cu` | Legacy: CUDA persistent kernel with cooperative groups |
| `ndt_cuda/src/voxel_grid/gpu/pipeline.rs` | Zero-copy voxel grid construction |
| `cuda_ndt_matcher/src/main.rs` | ROS node entry point |

## Claude Code Practices

- Use `timeout` parameter on Bash tool instead of `timeout` command
- Use `run_in_background: true` for long-running processes
- Create temp files in `$project/tmp/` not `/tmp/`
- Always use Write/Edit tools to create files, not `cat << EOF` heredoc patterns in Bash
- **Do NOT modify files in `external/autoware_repo`** - copy to `src/` first

### Process Cleanup

When stopping `play_launch` or multi-process ROS systems, **kill the entire process group** to avoid orphan child processes that may interfere with topics:

```bash
# Get the PGID of play_launch and kill the whole group
PGID=$(ps -o pgid= -p $(pgrep -f play_launch) | tr -d ' ')
kill -9 -$PGID

# Or use pkill with -g flag to kill process group
pkill -9 -g $PGID
```

**Never** use `pkill -9 -f play_launch` alone as it leaves orphaned child processes (component containers, ros2 nodes) that hold topics and prevent clean restarts.

**Common orphan processes to kill:**
```bash
pkill -9 -f "component_container"      # ROS 2 composable node containers
pkill -9 -f "component_container_mt"   # Multi-threaded containers
pkill -9 -f "robot_state_publisher"    # TF publisher
pkill -9 -f "ros2-daemon"              # ROS 2 CLI daemon
```

## Profiling

**Release profiling** (minimal overhead, for accurate performance comparison):
```bash
just profile-compare           # Full workflow: build, run both, compare
just run-cuda-profiling        # CUDA with timing data only
just run-builtin-profiling     # Autoware with timing data only
just compare-profiling         # Analyze results
```

**Debug profiling** (full debug data, adds overhead):
```bash
just run-cuda-debug            # CUDA with all debug features
just run-builtin-debug         # Autoware with all debug features
```

**Build differences**:
| Build | Feature | Overhead | Use Case |
|-------|---------|----------|----------|
| `build-cuda-profiling` | `profiling` | Minimal | Performance measurement |
| `build-cuda-debug` | `debug` | Significant | Debug data collection |

Output files:
- `logs/ndt_cuda_profiling.jsonl` - CUDA timing data
- `logs/ndt_autoware_profiling.jsonl` - Autoware timing data
- `logs/ndt_cuda_debug.jsonl` - CUDA full debug data
- `logs/ndt_autoware_debug.jsonl` - Autoware full debug data

Analysis scripts:
- `tmp/profile_comparison.py` - Compare CUDA vs Autoware performance
- `scripts/analyze_profile.py` - Analyze profile directory structure

## Comparison Testing

The `tests/comparison/` directory contains a fork of `autoware_ndt_scan_matcher` with debug patches for comparison testing. Builtin NDT recipes delegate to `tests/comparison/justfile`.

**Setup:**
```bash
# Initialize submodule (if not already done)
git submodule update --init tests/comparison/autoware_universe

# Build patched Autoware NDT
just build-comparison
```

**Usage:**
```bash
# Run Autoware (unpatched, no debug)
just run-builtin

# Run Autoware with debug output (requires build-comparison first)
just run-builtin-debug

# Dump voxel data for comparison
just dump-voxels-autoware
just dump-voxels-cuda
just compare-voxels
```

**Architecture:**
- `tests/comparison/autoware_universe/` - git submodule with debug patches
- `tests/comparison/justfile` - builds and runs patched Autoware
- Main justfile delegates: `run-builtin`, `run-builtin-debug`, `dump-voxels-autoware`, `analyze-debug-autoware`
- `run_ndt_simulation.sh` overlays `tests/comparison/install/` when available

**Patches included:**
- Per-iteration debug output (score, gradient, Hessian)
- Voxel grid dump for covariance comparison
- Convergence status logging

## Validation Status

### Covariance Formula Bug Fixed (2026-01-19)

**Root cause found and fixed**: The CPU voxel grid construction in `types.rs` used an incorrect covariance formula.

**Bug location**: `src/ndt_cuda/src/voxel_grid/types.rs:69-82`

**Wrong formula** (caused ~73% score):
```rust
cov = (sum_sq/n - mean*mean^T) * (n-1)/n  // WRONG
```

**Correct formula** (matches Autoware):
```rust
cov = (sum_sq - n*mean*mean^T) / (n-1)    // Standard sample covariance
```

**Impact**: The ratio between wrong/correct formulas is `(n-1)²/n²`:
| Points (n) | Formula ratio | Observed in dumps |
|------------|---------------|-------------------|
| 6          | 0.69          | 0.55 (matched)    |
| 10         | 0.81          | 0.63 (matched)    |
| 100        | 0.98          | 0.94 (matched)    |

**Verification**:
- GPU vs CPU cov_sums comparison: ratio = 1.000000 (exact match)
- All 417 unit tests pass (351 ndt_cuda + 66 cuda_ffi)
- All 7 Autoware comparison tests pass

**Note**: The GPU pipeline in `statistics.rs` was already correct (accumulates centered deviations and divides by n-1). Only the CPU path in `types.rs::from_statistics` had the bug.

**Investigation tools** (for future debugging):
```bash
# Generate voxel dumps
NDT_DUMP_VOXELS=1 just run-cuda
NDT_DUMP_VOXELS=1 just run-builtin

# Compare voxels
python3 tmp/compare_matching_voxels.py
python3 tmp/analyze_by_point_count.py

# Debug GPU vs CPU cov_sums
NDT_DEBUG_COV=1 cargo test -p ndt_cuda -- voxel --nocapture
```
