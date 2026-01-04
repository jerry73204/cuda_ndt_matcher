# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This project re-implements Autoware's `ndt_scan_matcher` in CUDA and Rust. The reference implementation is at `external/autoware_core/localization/autoware_ndt_scan_matcher/`.

NDT (Normal Distributions Transform) scan matching is used for position estimation in autonomous driving. Key features:
- Estimate position by scan matching against a point cloud map
- Initial position estimation via Monte Carlo method
- Optional regularization using GNSS

## Implementation Status

### CubeCL NDT Implementation

The project uses a pure-Rust NDT implementation built with [CubeCL](https://github.com/tracel-ai/cubecl). See `docs/cubecl-ndt-roadmap.md` for implementation details.

| Phase | Component | Status |
|-------|-----------|--------|
| Phase 1 | Voxel Grid Construction | ✅ Complete |
| Phase 2 | Derivative Computation | ✅ Complete |
| Phase 3 | Newton Optimization | ✅ Complete |
| Phase 4 | Scoring & NVTL | ✅ Complete |
| Phase 5 | Integration | ✅ Complete |
| Phase 6 | Validation | ⚠️ In Progress |
| Phase 7.1 | TF Broadcast | ✅ Complete |
| Phase 7.2 | Dynamic Map Loading | ✅ Complete |
| Phase 7.3 | GNSS Regularization | ✅ Complete |
| Phase 7.4 | Multi-NDT Covariance | ✅ Complete |
| Phase 7.5 | Diagnostics Interface | ✅ Complete |
| Phase 7.6 | Oscillation Detection | ✅ Complete |
| Phase 9.1 | GPU Infrastructure (CubeCL) | ✅ Complete |
| Phase 9.2 | GPU Voxel Grid | ✅ Complete |
| Phase 9.3 | GPU Derivatives (Hessian) | ✅ Complete |

### Validation Status

Integration tests run but convergence is poor compared to Autoware baseline:
- **311 unit tests pass** (45 cuda_ndt_matcher + 266 ndt_cuda)
- **Trajectory RMSE: ~12m** (target: <0.3m)
- **Convergence rate: ~54%** (target: >80%)

**Derivative formulas verified correct:**
- Jacobian formulas (∂T/∂pose) match `AngularDerivatives` reference exactly
- Point Hessian formulas (∂²T/∂pose²) match `AngularDerivatives` reference exactly
- CPU vs GPU derivative computation matches (score, gradient, Hessian diagonal signs)
- All 15 h_ang terms (a2, a3, b2, b3, c2, c3, d1-d3, e1-e3, f1-f3) verified

Root cause investigation ongoing. KD-tree radius search infrastructure is complete (matching Autoware's algorithm), but numerical convergence differs.

### Core Components

| Component | Status | Notes |
|-----------|--------|-------|
| ndt_cuda library | ✅ | CubeCL-based NDT with Newton optimizer |
| PointCloud2 conversion | ✅ | Efficient point cloud handling |
| ROS subscriptions/publishers | ✅ | Full Autoware integration |
| trigger_node_srv service | ✅ | Enable/disable NDT matching |
| Launch files | ✅ | Drop-in replacement for Autoware NDT |
| Covariance estimation | ✅ | FIXED, LAPLACE, and MULTI_NDT modes |
| TF broadcast | ✅ | Publishes map → ndt_base_link transform |
| Dynamic map loading | ✅ | Position-based map tile updates |
| GNSS regularization | ✅ | Penalizes longitudinal drift in open areas |
| Oscillation detection | ✅ | Detects optimizer direction reversals |
| Diagnostics | ✅ | Publishes to /diagnostics topic |

## Build System

The project uses `colcon-cargo-ros2` for Rust ROS 2 integration.

### Prerequisites

```bash
# Source ROS 2 (required before any build)
source /opt/ros/humble/setup.bash

# Or use direnv (recommended)
direnv allow  # Sources .envrc which handles ROS setup
```

### Build Commands

**IMPORTANT:** Always use the justfile for build/clean operations. Never run colcon directly.

```bash
just build    # colcon build with --release
just clean    # rm -rf build install log target
just format   # Format code with rustfmt
just lint     # Run format check and clippy (requires build first)
just test     # Run tests (requires build first)
just quality  # Run lint + test
```

**Important:** Always source ROS before building. The `.envrc` file handles this automatically if using direnv.

### Running

```bash
# Source the workspace after building
source install/setup.bash

# Run the node directly
ros2 run cuda_ndt_matcher cuda_ndt_matcher

# Or use the launch file
ros2 launch cuda_ndt_matcher_launch cuda_ndt_scan_matcher.launch.xml
```

### Replay Simulation Testing

Test the NDT implementation using rosbag replay simulation:

```bash
# Download sample map and rosbag data (one-time setup)
just download-data

# Terminal 1: Start Autoware NDT as systemd service (baseline)
just start-ndt-autoware

# Or start CUDA NDT (our implementation)
just start-ndt-cuda

# Terminal 2: Play the rosbag (after Autoware finishes loading)
just start-rosbag

# Terminal 3: Enable NDT matching
just enable-ndt

# Terminal 3: Monitor NDT output
just monitor-ndt

# Stop the service when done
just stop-ndt-autoware  # or stop-ndt-cuda
```

**Service management:**
- `just start-ndt-{autoware,cuda}` - Start as systemd user service
- `just stop-ndt-{autoware,cuda}` - Stop the service
- `just restart-ndt-{autoware,cuda}` - Restart the service
- `just status-ndt-{autoware,cuda}` - Show service status
- `just log-ndt-{autoware,cuda}` - Follow service logs

**Important:** The `just start-*` recipes start background systemd services. Do NOT append `&` to these commands - they return immediately after starting the service. Use `just log-*` to monitor service output.

**Note:** The sample rosbag may require manual pose initialization in RViz using "2D Pose Estimate" tool.

For running with custom rosbags, see `docs/rosbag-replay-guide.md`.

### Demo Mode with play_launch

The `just run-builtin` and `just run-cuda` recipes use `play_launch` to run the full demo with logging:

```bash
# Run Autoware builtin NDT
just run-builtin

# Run CUDA NDT (our implementation)
just run-cuda
```

**Reading logs:** Logs are written to `play_log/<timestamp>/` with a `latest` symlink:

```bash
# Node logs are in play_log/latest/node/<node_name>/
cat play_log/latest/node/ndt_scan_matcher/err   # stderr output
cat play_log/latest/node/ndt_scan_matcher/out   # stdout output

# Check node parameters
cat play_log/latest/node/ndt_scan_matcher/params_files/0.yaml
```

## Project Structure

```
cuda_ndt_matcher/
├── docs/                        # Design documentation
│   ├── overview.md              # Project goals and strategy
│   ├── architecture.md          # System architecture, ROS interface
│   ├── integration.md           # ROS2 integration details
│   ├── roadmap.md               # Phased work items with tests
│   ├── rosbag-replay-guide.md   # Guide for custom rosbag replay simulation
│   └── cubecl-ndt-roadmap.md    # CubeCL NDT implementation plan
├── data/                        # Test data (downloaded via just download-data)
│   ├── sample-map-rosbag/       # PCD map and lanelet2_map.osm
│   └── sample-rosbag/           # Sample rosbag for replay simulation
├── external/
│   ├── autoware_repo/           # Autoware installation (install/setup.bash)
│   └── autoware_core/.../autoware_ndt_scan_matcher/  # C++ reference
├── Cargo.toml                   # Cargo workspace root
├── target/                      # Cargo build output
├── src/
│   ├── ndt_cuda/                # CubeCL NDT library
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── voxel_grid/      # Voxelization kernels
│   │   │   ├── derivatives/     # Jacobian/Hessian computation
│   │   │   ├── optimization/    # Newton solver
│   │   │   └── scoring/         # Transform probability, NVTL
│   │   └── Cargo.toml
│   ├── cuda_ndt_matcher/        # Main ROS package (Rust)
│   │   ├── src/
│   │   │   ├── main.rs          # Node entry point, subscriptions, publishers
│   │   │   ├── ndt_manager.rs   # NDTCuda wrapper with nalgebra transforms
│   │   │   ├── params.rs        # ROS parameters from config
│   │   │   ├── pointcloud.rs    # PointCloud2 <-> Vec<[f32;3]> conversion
│   │   │   ├── covariance.rs    # Covariance estimation (FIXED, LAPLACE, MULTI_NDT)
│   │   │   ├── diagnostics.rs   # ROS diagnostics interface (/diagnostics topic)
│   │   │   ├── map_module.rs    # Dynamic map loading and filtering
│   │   │   └── initial_pose.rs  # Initial pose estimation with TPE
│   │   ├── Cargo.toml
│   │   └── package.xml
│   ├── cuda_ndt_matcher_launch/ # Launch package (CMake)
│   │   ├── launch/
│   │   │   ├── cuda_ndt_scan_matcher.launch.xml   # Drop-in replacement for Autoware NDT
│   │   │   └── ndt_replay_simulation.launch.xml   # Replay simulation with use_cuda arg
│   │   ├── config/ndt_scan_matcher.param.yaml
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   ├── rosbag_sensor_kit_launch/      # Custom sensor kit for sample rosbag (3 LiDARs)
│   │   ├── launch/
│   │   │   ├── lidar.launch.xml       # top, left, right (no rear)
│   │   │   ├── sensing.launch.xml
│   │   │   └── pointcloud_preprocessor.launch.py
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   ├── rosbag_sensor_kit_description/ # Sensor URDF/calibration for rosbag_sensor_kit
│   │   ├── config/
│   │   │   ├── sensor_kit_calibration.yaml
│   │   │   └── sensors_calibration.yaml
│   │   ├── urdf/
│   │   │   ├── sensor_kit.xacro
│   │   │   └── sensors.xacro
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   └── individual_params/             # Vehicle/sensor calibration parameters
│       ├── config/default/rosbag_sensor_kit/
│       │   ├── sensors_calibration.yaml
│       │   ├── sensor_kit_calibration.yaml
│       │   └── imu_corrector.param.yaml
│       ├── CMakeLists.txt
│       └── package.xml
├── scripts/                     # Helper scripts
│   ├── download_sample_data.sh  # Downloads sample map and rosbag
│   └── clock_from_sensor.py     # Republish /clock from sensor timestamps (timestamp fix)
├── build/                       # Generated by colcon
├── .envrc                       # direnv config (sources ROS)
└── justfile
```

### Custom Sensor Kit (rosbag_sensor_kit)

The Autoware sample rosbag contains only 3 LiDARs (top, left, right) but `sample_sensor_kit` expects 4 (including rear). We created `rosbag_sensor_kit_launch` and `rosbag_sensor_kit_description` to match the actual rosbag configuration.

Key differences from `sample_sensor_kit`:
- `lidar.launch.xml`: Removed rear LiDAR definition
- `pointcloud_preprocessor.launch.py`: Concatenates 3 topics instead of 4
- IMU/GNSS: Delegates to `sample_sensor_kit_launch` (unchanged)

## COLCON_IGNORE Files

The `target/` directory is at project root and excluded automatically since colcon uses `--base-paths src`.

## Reference Implementation

The Autoware `ndt_scan_matcher` at `external/autoware_core/localization/autoware_ndt_scan_matcher/` includes:

- `ndt_scan_matcher_core.hpp/cpp`: Main ROS node implementation
- `ndt_omp/`: OpenMP-parallelized NDT algorithm
- `map_update_module.hpp/cpp`: Dynamic map loading
- `particle.hpp/cpp`: Particle representation for Monte Carlo
- `hyper_parameters.hpp`: Configuration parameters

## ndt_cuda Library

The `ndt_cuda` crate provides a pure-Rust NDT implementation using [CubeCL](https://github.com/tracel-ai/cubecl) for GPU compute.

### Why CubeCL?

- **Pure Rust**: No C++/CUDA FFI complexity
- **Multi-platform**: Same code for CUDA, ROCm, WebGPU
- **Type-safe**: Rust's guarantees for GPU code

### CubeCL Limitations

When writing GPU kernels, be aware of these CubeCL constraints:

1. **No `as usize` for array indexing**: Dynamic array indexing with `arr[i as usize]` is not supported. Use fully unrolled loops with explicit indices instead.

2. **Parameter count limit**: Kernels with >12 parameters hit Rust's tuple trait limits (`Eq`, `Hash`, `Debug` not implemented for large tuples). Workaround: combine multiple buffers into single arrays.

3. **No dynamic control flow in some cases**: Some operations require compile-time constants.

Example of working around the indexing limitation:
```rust
// BAD: Dynamic indexing not supported
for i in 0..6 {
    result += arr[i as usize];
}

// GOOD: Fully unrolled
let v0 = arr[0]; let v1 = arr[1]; let v2 = arr[2];
let v3 = arr[3]; let v4 = arr[4]; let v5 = arr[5];
result = v0 + v1 + v2 + v3 + v4 + v5;
```

### API

```rust
use ndt_cuda::NdtScanMatcher;

let mut matcher = NdtScanMatcher::builder()
    .resolution(2.0)
    .max_iterations(30)
    .transformation_epsilon(0.01)
    .build()?;

// Set target (map) point cloud
matcher.set_target(&map_points)?;

// Align source to target with initial guess
let result = matcher.align(&source_points, initial_pose)?;
// result.converged, result.score, result.pose, result.hessian
```

### Key Algorithm (Magnusson 2009)

The NDT algorithm we're implementing:

1. **Voxelization**: Build Gaussian voxel grid from map (mean + covariance per voxel)
2. **Correspondence**: Find voxels using KD-tree radius search on centroids (matches Autoware's `radiusSearch`)
3. **Derivatives**: Compute gradient (6x1) and Hessian (6x6) using:
   - Score: `p(x) = -d1 * exp(-d2/2 * (x-μ)ᵀΣ⁻¹(x-μ))` (Eq. 6.9)
   - Gradient: Eq. 6.12
   - Hessian: Eq. 6.13
4. **Newton step**: Solve `Δp = -H⁻¹g` (6x6 linear system)
5. **Iterate**: Until convergence (typically 5-10 iterations)

See `docs/cubecl-ndt-roadmap.md` for full implementation details.

### Derivative Verification Tests

The derivative computations are verified against the reference `AngularDerivatives` implementation:

- `test_jacobians_match_angular_derivatives` - Verifies all 8 Jacobian rotation terms
- `test_point_hessians_match_angular_derivatives` - Verifies all 15 point Hessian terms
- `test_cpu_vs_gpu_derivatives` - End-to-end CPU/GPU comparison
- `test_cpu_vs_gpu_single_point_single_voxel` - Simplified single-point verification

Run with: `cargo --config build/ros2_cargo_config.toml test -p ndt_cuda --lib "test_jacobians\|test_hessians\|test_cpu_vs_gpu" -- --nocapture`

### Voxel Search Infrastructure

The `VoxelSearch` struct (in `voxel_grid/search.rs`) provides KD-tree based radius search matching Autoware's implementation:

```rust
// Build KD-tree from voxel centroids (using kiddo crate)
let search = VoxelSearch::from_voxels(&voxels);

// Find all voxels within radius (matches Autoware's radiusSearch)
let nearby_indices = search.within(&query_point, resolution);

// Accumulate contributions from ALL nearby voxels
for idx in nearby_indices {
    let voxel = &voxels[idx];
    // Compute derivatives...
}
```

Key details:
- Uses `kiddo` crate for high-performance KD-tree
- Search radius = voxel resolution (2.0m default), matching Autoware
- Returns multiple voxels per query point for smoother gradients

## Development Notes

### ROS 2 Rust Patterns (from LCTK reference)

**Service callbacks:**
```rust
node.create_service::<SetBool, _>("service_name", move |req: Request, _info: ServiceInfo| {
    Response { success: true, message: "OK".to_string() }
})?
```

**Parameters:**
```rust
let value: String = node
    .declare_parameter::<Arc<str>>("param.name")
    .default("default".into())
    .mandatory()?
    .get()
    .to_string();
```

**Executor:**
```rust
use rclrs::{Context, CreateBasicExecutor};  // Must import trait
let mut executor = Context::default_from_env()?.create_basic_executor();
```

### Transform Conversions

Use nalgebra for rotation conversions (not manual quaternion math):
```rust
use nalgebra::{Isometry3, UnitQuaternion, Translation3};

// Pose to transform
let translation = Translation3::new(p.x, p.y, p.z);
let quaternion = UnitQuaternion::from_quaternion(NaQuaternion::new(q.w, q.x, q.y, q.z));
let isometry = Isometry3::from_parts(translation, quaternion);
let matrix = isometry.to_homogeneous();

// Transform to quaternion
let rotation = nalgebra::Rotation3::from_matrix_unchecked(rotation_matrix);
let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
```

### Build Gotchas

- `ament_cargo` does NOT install data files (launch, config) - use separate CMake package
- Cargo workspace is at project root `Cargo.toml`, with members in `src/`
- Use `colcon build --base-paths src` to avoid scanning external/ and target/ directories
- Add COLCON_IGNORE to directories containing unwanted CMakeLists.txt

### rclrs Limitations

The Rust ROS 2 library (`rclrs`) has some limitations:

- **Topic remapping from launch files doesn't work**: Publishers must use the final topic name directly, not rely on `<remap from="X" to="Y"/>` in launch XML. For example, use `node.create_publisher("pose")` not `node.create_publisher("ndt_pose")` with a remap.

### Running Cargo Commands Directly

When running cargo commands outside of `just` recipes (e.g., for testing a specific test), pass the ROS 2 cargo config:

```bash
cargo --config build/ros2_cargo_config.toml test -p ndt_cuda --lib test_name
cargo --config build/ros2_cargo_config.toml build -p ndt_cuda
```

This config file is generated by colcon and provides the correct paths for ROS 2 message dependencies.

## Coding Conventions

- Use named parameters in format strings: `println!("{e}")` not `println!("{}", e)`
- Use nalgebra for all rotation/transform math
- Prefer `Arc<ArcSwap<T>>` over `Arc<Mutex<T>>` for read-heavy concurrent access
- Clone variables in a local scope before moving into closures
- Use type aliases for ROS service types: `type SetBoolRequest = std_srvs::srv::SetBool_Request;`

## Claude Code Practices

- When running long commands (ROS2 service calls, etc.), set `timeout` parameter on the Bash tool instead of using `timeout` command prefix
- Example: Use `Bash(command: "ros2 service call ...", timeout: 60000)` instead of `Bash(command: "timeout 60 ros2 service call ...")`
- For long-running background processes (like `just run-cuda`), use the Bash tool with `run_in_background: true` and set an appropriate `timeout`
- Example: `Bash(command: "just run-cuda", timeout: 300000, run_in_background: true)`
- Create temporary scripts and files in `$project/tmp/` directory instead of `/tmp/` or using heredocs
- Create files using Write/Edit tools instead of `cat << EOF` heredoc syntax in Bash
- **Do NOT modify files in `external/autoware_repo`** - this is a symlink to the main Autoware workspace
- If you need to modify Autoware components, copy them to `src/` first (e.g., `src/autoware_ndt_scan_matcher/`)

## NDT Topic Mappings

When running in Autoware's localization stack, the CUDA NDT node uses these topic mappings:

| Topic | Type | Description |
|-------|------|-------------|
| `/localization/util/downsample/pointcloud` | PointCloud2 | Input: downsampled LiDAR scan |
| `/localization/pose_twist_fusion_filter/biased_pose_with_covariance` | PoseWithCovarianceStamped | Input: EKF pose for initial guess |
| `/sensing/gnss/pose_with_covariance` | PoseWithCovarianceStamped | Input: GNSS pose for regularization |
| `/localization/pose_estimator/pose` | PoseStamped | Output: NDT pose |
| `/localization/pose_estimator/pose_with_covariance` | PoseWithCovarianceStamped | Output: NDT pose with covariance |
| `/tf` | TFMessage | Output: map → ndt_base_link transform |
| `/diagnostics` | DiagnosticArray | Output: scan matching diagnostics |
| `/localization/pose_estimator/trigger_node` | SetBool service | Enable/disable NDT matching |
| `/map/get_differential_pointcloud_map` | GetDifferentialPointCloudMap service | Dynamic map loading |

## Launch File Architecture

The replay simulation uses Autoware's `logging_simulator.launch.xml` as the base:

- **Autoware mode** (`use_cuda:=false`): Uses `logging_simulator` with `localization:=true`
- **CUDA mode** (`use_cuda:=true`): Uses `logging_simulator` with `localization:=false`, then manually includes:
  - EKF localizer (`pose_twist_fusion_filter`)
  - Gyro odometer (`twist_estimator`)
  - Pose initializer and pointcloud downsampling (`util`)
  - Our CUDA NDT (`cuda_ndt_scan_matcher.launch.xml`)
