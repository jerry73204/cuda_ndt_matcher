# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This project re-implements Autoware's `ndt_scan_matcher` in CUDA and Rust. The reference implementation is at `external/autoware_core/localization/autoware_ndt_scan_matcher/`.

NDT (Normal Distributions Transform) scan matching is used for position estimation in autonomous driving. Key features:
- Estimate position by scan matching against a point cloud map
- Initial position estimation via Monte Carlo method
- Optional regularization using GNSS

## Implementation Status

**Phase 1 + 2 + 3 + 5: COMPLETE**

| Component | Status |
|-----------|--------|
| fast-gicp integration | ✅ |
| PointCloud2 conversion | ✅ |
| NDT alignment (NDTCuda) | ✅ |
| ROS subscriptions (points_raw, ekf_pose, regularization_pose) | ✅ |
| ROS publishers (ndt_pose, ndt_pose_with_covariance) | ✅ |
| trigger_node_srv service | ✅ |
| All parameters from config | ✅ |
| Launch file with remappings | ✅ |
| Covariance estimation (FIXED, LAPLACE, MULTI_NDT, MULTI_NDT_SCORE) | ✅ |
| Hessian/cost evaluation FFI bindings | ✅ |
| Dynamic map loading service client | ✅ |
| Replay simulation launch files | ✅ |

**Remaining phases:** Initial Pose Estimation, Validation & Diagnostics, Performance Optimization

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

```bash
just build    # colcon build with --release
just clean    # rm -rf build install log src/target
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
just play-rosbag

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
- `just logs-ndt-{autoware,cuda}` - Follow service logs

**Note:** The sample rosbag may require manual pose initialization in RViz using "2D Pose Estimate" tool.

For running with custom rosbags, see `docs/rosbag-replay-guide.md`.

## Project Structure

```
cuda_ndt_matcher/
├── docs/                        # Design documentation
│   ├── overview.md              # Project goals and strategy
│   ├── architecture.md          # System architecture, ROS interface
│   ├── integration.md           # fast_gicp_rust + cubecl usage
│   ├── roadmap.md               # Phased work items with tests
│   └── rosbag-replay-guide.md   # Guide for custom rosbag replay simulation
├── data/                        # Test data (downloaded via just download-data)
│   ├── sample-map-rosbag/       # PCD map and lanelet2_map.osm
│   └── sample-rosbag/           # Sample rosbag for replay simulation
├── external/
│   ├── autoware_repo/           # Autoware installation (install/setup.bash)
│   ├── autoware_core/.../autoware_ndt_scan_matcher/  # C++ reference
│   └── fast_gicp_rust/          # Rust CUDA NDT bindings
│       ├── fast-gicp/           # High-level Rust API
│       ├── fast-gicp-sys/       # C++ FFI layer
│       └── fast_gicp/           # Upstream C++ (git submodule, has COLCON_IGNORE)
├── src/
│   ├── Cargo.toml               # Workspace root
│   ├── target/                  # Cargo build output (has COLCON_IGNORE)
│   ├── cuda_ndt_matcher/        # Main ROS package (Rust)
│   │   ├── src/
│   │   │   ├── main.rs          # Node entry point, subscriptions, publishers
│   │   │   ├── ndt_manager.rs   # NDTCuda wrapper with nalgebra transforms
│   │   │   ├── params.rs        # ROS parameters from config
│   │   │   └── pointcloud.rs    # PointCloud2 <-> Vec<[f32;3]> conversion
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

These prevent colcon from picking up unwanted CMake packages:
- `external/fast_gicp_rust/fast_gicp/COLCON_IGNORE` - Upstream C++ library
- `src/target/COLCON_IGNORE` - Cargo build artifacts (contains vendored C++)

## Reference Implementation

The Autoware `ndt_scan_matcher` at `external/autoware_core/localization/autoware_ndt_scan_matcher/` includes:

- `ndt_scan_matcher_core.hpp/cpp`: Main ROS node implementation
- `ndt_omp/`: OpenMP-parallelized NDT algorithm
- `map_update_module.hpp/cpp`: Dynamic map loading
- `particle.hpp/cpp`: Particle representation for Monte Carlo
- `hyper_parameters.hpp`: Configuration parameters

## fast_gicp_rust

`external/fast_gicp_rust/` provides Rust bindings for CUDA-accelerated point cloud registration.

**Key types:**
- `NDTCuda` - GPU NDT via `NDTCuda::builder().resolution().build()`
- `PointCloudXYZ` - Point cloud via `PointCloudXYZ::from_points(&[[f32;3]])`
- `Transform3f` - 4x4 transform matrix
- `NdtDistanceMode` - P2D (point-to-distribution) or D2D
- `NeighborSearchMethod` - Direct1, Direct7, Direct27

**API pattern:**
```rust
let ndt = NDTCuda::builder()
    .resolution(2.0)
    .max_iterations(30)
    .transformation_epsilon(0.01)
    .distance_mode(NdtDistanceMode::P2D)
    .neighbor_search_method(NeighborSearchMethod::Direct7)
    .build()?;

let result = ndt.align_with_guess(&source, &target, Some(&initial_transform))?;
// result.has_converged, result.fitness_score, result.final_transformation
```

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
- Cargo workspace must be at `src/Cargo.toml`, not project root
- `build/ros2_cargo_config.toml` generated by colcon, needed for standalone cargo commands
- Git submodules in external dependencies need `git submodule update --init --recursive`
- Add COLCON_IGNORE to directories containing unwanted CMakeLists.txt

## Coding Conventions

- Use named parameters in format strings: `println!("{e}")` not `println!("{}", e)`
- Use nalgebra for all rotation/transform math
- Prefer `Arc<ArcSwap<T>>` over `Arc<Mutex<T>>` for read-heavy concurrent access
- Clone variables in a local scope before moving into closures
- Use type aliases for ROS service types: `type SetBoolRequest = std_srvs::srv::SetBool_Request;`

## NDT Topic Mappings

When running in Autoware's localization stack, the CUDA NDT node uses these topic mappings:

| Topic | Type | Description |
|-------|------|-------------|
| `/localization/util/downsample/pointcloud` | PointCloud2 | Input: downsampled LiDAR scan |
| `/localization/pose_twist_fusion_filter/biased_pose_with_covariance` | PoseWithCovarianceStamped | Input: EKF pose for initial guess |
| `/sensing/gnss/pose_with_covariance` | PoseWithCovarianceStamped | Input: GNSS pose for regularization |
| `/localization/pose_estimator/pose` | PoseStamped | Output: NDT pose |
| `/localization/pose_estimator/pose_with_covariance` | PoseWithCovarianceStamped | Output: NDT pose with covariance |
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
