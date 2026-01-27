# Phase 25: Code Restructure (cuda_ndt_matcher)

**Status**: Planned
**Date**: 2026-01-28

## Motivation

The `cuda_ndt_matcher` crate has grown organically and needs restructuring:

1. **main.rs is too large** (1,934 lines) - difficult to navigate and maintain
2. **Flat module structure** - unclear relationships between modules (e.g., `tpe.rs` only used by `initial_pose.rs`)
3. **CPU/GPU paths not explicit** - hard to identify which code runs on GPU vs CPU

## Goals

- Split `main.rs` into manageable modules
- Create hierarchical module structure reflecting actual dependencies
- Make CPU/GPU implementations explicit where both exist
- Improve code discoverability and maintainability

## Current Structure

```
src/cuda_ndt_matcher/src/
├── main.rs              (1,934 lines) ← TOO BIG
├── covariance.rs        (703 lines)
├── diagnostics.rs       (446 lines)
├── dual_ndt_manager.rs  (468 lines)
├── initial_pose.rs      (527 lines)
├── map_module.rs        (836 lines)   ← Contains two distinct components
├── ndt_manager.rs       (554 lines)
├── nvtl.rs              (413 lines)
├── params.rs            (440 lines)
├── particle.rs          (79 lines)    ← Only used by initial_pose
├── pointcloud.rs        (419 lines)   ← Has both CPU and GPU paths
├── pose_buffer.rs       (464 lines)
├── scan_queue.rs        (457 lines)
├── tf_handler.rs        (396 lines)
├── tpe.rs               (303 lines)   ← Only used by initial_pose
└── visualization.rs     (709 lines)

Total: 9,148 lines across 16 files
```

## Target Structure

```
src/cuda_ndt_matcher/src/
├── main.rs                    (~50 lines)   - Entry point only
├── lib.rs                     (~30 lines)   - Crate re-exports
│
├── node/                      - ROS node components
│   ├── mod.rs                 (~30 lines)
│   ├── state.rs               (~150 lines) - NdtScanMatcherNode struct
│   ├── init.rs                (~450 lines) - new() initialization
│   ├── callbacks.rs           (~550 lines) - on_points() callback
│   ├── services.rs            (~250 lines) - Service handlers
│   ├── publishers.rs          (~200 lines) - Debug publishers, TF
│   └── processing.rs          (~300 lines) - Alignment processing logic
│
├── alignment/                 - NDT alignment (GPU path)
│   ├── mod.rs
│   ├── manager.rs             (554 lines)  ← ndt_manager.rs
│   ├── dual_manager.rs        (468 lines)  ← dual_ndt_manager.rs
│   ├── covariance.rs          (703 lines)  ← covariance.rs
│   └── batch.rs               (457 lines)  ← scan_queue.rs
│
├── initial_pose/              - Initial pose estimation
│   ├── mod.rs
│   ├── estimator.rs           (527 lines)  ← initial_pose.rs
│   ├── tpe.rs                 (303 lines)  ← tpe.rs (PUBLIC)
│   └── particle.rs            (79 lines)   ← particle.rs
│
├── map/                       - Map management (CPU)
│   ├── mod.rs
│   ├── tiles.rs               (~500 lines) ← map_module.rs (MapUpdateModule)
│   └── loader.rs              (~340 lines) ← map_module.rs (DynamicMapLoader)
│
├── transform/                 - Spatial transforms (CPU)
│   ├── mod.rs
│   ├── tf_handler.rs          (396 lines)  ← tf_handler.rs
│   └── pose_buffer.rs         (464 lines)  ← pose_buffer.rs
│
├── scoring/                   - Scoring reference (CPU)
│   ├── mod.rs
│   └── nvtl.rs                (413 lines)  ← nvtl.rs
│
├── io/                        - I/O utilities
│   ├── mod.rs
│   ├── pointcloud/            - Explicit CPU/GPU split
│   │   ├── mod.rs
│   │   ├── cpu.rs             ← CPU conversion/filtering
│   │   └── gpu.rs             ← GPU-accelerated filtering
│   ├── params.rs              (440 lines)  ← params.rs
│   └── diagnostics.rs         (446 lines)  ← diagnostics.rs
│
└── visualization/             - Debug visualization (CPU)
    ├── mod.rs
    └── markers.rs             (709 lines)  ← visualization.rs
```

## Implementation Phases

### Phase 1: Split main.rs

**Priority**: First

Split the 1,934-line `main.rs` into the `node/` module hierarchy.

#### 1.1 Create node/state.rs

Extract `NdtScanMatcherNode` struct definition (lines 129-167):

```rust
// node/state.rs
pub struct NdtScanMatcherNode {
    // All fields from current NdtScanMatcherNode
}
```

#### 1.2 Create node/publishers.rs

Extract (lines 84-123, 1507-1678):
- `DebugPublishers` struct
- `publish_tf()` function
- Marker creation helpers

#### 1.3 Create node/services.rs

Extract service handlers (lines 1684-1903):
- `on_ndt_align()` - Initial pose alignment service
- `on_map_update()` - Map update service
- `on_map_received()` - Map subscription callback

#### 1.4 Create node/processing.rs

Extract core alignment logic from `on_points()`:
- Pose interpolation and validation
- Alignment execution (sync/batch)
- Convergence gating
- Covariance estimation
- Utility: `isometry_to_pose()` (lines 61-83)

#### 1.5 Create node/callbacks.rs

Remaining `on_points()` structure:
- Point cloud conversion and filtering
- Sensor frame transformation
- Calls to processing functions
- Debug publishing

#### 1.6 Create node/init.rs

Extract `NdtScanMatcherNode::new()` (lines 170-617):
- Parameter loading
- Publisher creation
- Subscription creation
- Service creation

#### 1.7 Slim main.rs

Final `main.rs` (~50 lines):

```rust
mod node;

fn main() {
    // Initialize logging
    // Create node
    // Spin
}
```

### Phase 2: Reorganize Module Structure

Move files to new directory structure:

| Current | New Location |
|---------|--------------|
| `ndt_manager.rs` | `alignment/manager.rs` |
| `dual_ndt_manager.rs` | `alignment/dual_manager.rs` |
| `covariance.rs` | `alignment/covariance.rs` |
| `scan_queue.rs` | `alignment/batch.rs` |
| `initial_pose.rs` | `initial_pose/estimator.rs` |
| `tpe.rs` | `initial_pose/tpe.rs` |
| `particle.rs` | `initial_pose/particle.rs` |
| `map_module.rs` | `map/tiles.rs` + `map/loader.rs` |
| `tf_handler.rs` | `transform/tf_handler.rs` |
| `pose_buffer.rs` | `transform/pose_buffer.rs` |
| `nvtl.rs` | `scoring/nvtl.rs` |
| `pointcloud.rs` | `io/pointcloud/` |
| `params.rs` | `io/params.rs` |
| `diagnostics.rs` | `io/diagnostics.rs` |
| `visualization.rs` | `visualization/markers.rs` |

### Phase 3: Split Dual CPU/GPU Implementations

#### 3.1 Split pointcloud.rs

Current `pointcloud.rs` has both CPU and GPU filtering paths:

```rust
// io/pointcloud/mod.rs
pub mod cpu;
pub mod gpu;

// Re-exports
pub use cpu::{from_pointcloud2, to_pointcloud2, to_pointcloud2_with_rgb};

/// Auto-select CPU or GPU based on point count
pub fn filter_sensor_points(
    points: &[[f32; 3]],
    params: &PointFilterParams
) -> FilterResult {
    if points.len() > 10_000 && gpu::is_available() {
        gpu::filter_sensor_points(points, params)
    } else {
        cpu::filter_sensor_points(points, params)
    }
}
```

```rust
// io/pointcloud/cpu.rs
pub fn from_pointcloud2(msg: &PointCloud2) -> Result<Vec<[f32; 3]>, Error> { ... }
pub fn to_pointcloud2(points: &[[f32; 3]], frame_id: &str) -> PointCloud2 { ... }
pub fn to_pointcloud2_with_rgb(...) -> PointCloud2 { ... }
pub fn filter_sensor_points(...) -> FilterResult { ... }
```

```rust
// io/pointcloud/gpu.rs
pub fn is_available() -> bool { ... }
pub fn filter_sensor_points(...) -> FilterResult { ... }
```

#### 3.2 Split map_module.rs

Split into two focused modules:

```rust
// map/tiles.rs (~500 lines)
pub struct MapUpdateModule { ... }
pub struct MapTile { ... }
pub struct MapUpdateResult { ... }
pub struct MapStats { ... }
```

```rust
// map/loader.rs (~340 lines)
pub struct DynamicMapLoader { ... }
pub struct MapLoaderStatus { ... }
```

### Phase 4: Module Visibility

```rust
// lib.rs
pub mod alignment;      // GPU alignment path
pub mod initial_pose;   // Initial pose estimation
pub mod map;            // CPU map management
pub mod transform;      // CPU transforms
pub mod scoring;        // CPU scoring reference
pub mod io;             // I/O with explicit CPU/GPU
pub mod visualization;  // CPU visualization

// node/ is internal (ROS-specific)
```

```rust
// initial_pose/mod.rs
pub mod tpe;            // PUBLIC as requested
pub mod particle;       // PUBLIC (used by visualization)
mod estimator;

pub use estimator::estimate_initial_pose;
pub use tpe::TreeStructuredParzenEstimator;
pub use particle::{Particle, select_best_particle};
```

## Module Classification

### GPU Path (alignment/)

| Module | Purpose |
|--------|---------|
| `alignment/manager.rs` | NDT alignment via ndt_cuda |
| `alignment/dual_manager.rs` | Non-blocking dual NDT |
| `alignment/covariance.rs` | Covariance estimation (orchestrates GPU batch) |
| `alignment/batch.rs` | Scan queue for batch GPU processing |

### CPU Modules

| Module | Purpose |
|--------|---------|
| `map/` | Tile management, map loading |
| `transform/` | TF buffer, pose interpolation |
| `scoring/` | NVTL reference implementation |
| `io/params.rs` | Parameter loading |
| `io/diagnostics.rs` | ROS diagnostics |
| `visualization/` | Debug markers and clouds |

### Mixed CPU/GPU

| Module | Purpose |
|--------|---------|
| `initial_pose/` | CPU orchestrator for GPU batch alignment |
| `io/pointcloud/` | Explicit CPU/GPU filtering |
| `node/` | ROS orchestration |

## Migration Strategy

1. **Preserve git history**: Use `git mv` for file moves
2. **Incremental changes**: One phase at a time, verify builds
3. **Update imports**: Fix all `use` statements after each move
4. **Run tests**: Ensure all tests pass after each phase

## Verification

After each phase:

```bash
just build
just test
just lint
```

## Risks

1. **Import breakage**: Many files import from main.rs indirectly
2. **Circular dependencies**: Must carefully order module declarations
3. **Feature flags**: Some code is behind `#[cfg(feature = "...")]`

## Dependencies

- No external dependencies
- Internal refactoring only
- No API changes to ndt_cuda crate

## Success Criteria

- [ ] `main.rs` reduced to ~50 lines
- [ ] All modules in logical hierarchical structure
- [ ] CPU/GPU paths clearly identifiable
- [ ] All tests passing
- [ ] No functionality changes
