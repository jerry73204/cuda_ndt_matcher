# Phase 8: Missing Features (Autoware Parity)

**Status**: ✅ Complete

## Goal

Implement features present in Autoware's `ndt_scan_matcher` that are missing or incomplete in our implementation.

## 8.1 Fix Pose Output Publishing ✅ COMPLETE

**Problem**: Node computed alignments but didn't publish poses to ROS topics.

**Root Cause**: Topic name mismatch between code and launch file remappings.

The launch file expected:
```xml
<remap from="ndt_pose" to="$(var output_pose_topic)"/>
<remap from="ndt_pose_with_covariance" to="$(var output_pose_with_covariance_topic)"/>
```

But the code created publishers with wrong names:
```rust
// WRONG:
let pose_pub = node.create_publisher("pose")?;
let pose_cov_pub = node.create_publisher("pose_with_covariance")?;
```

**Fix**: Changed topic names to match launch file remappings:
```rust
// CORRECT:
let pose_pub = node.create_publisher("ndt_pose")?;
let pose_cov_pub = node.create_publisher("ndt_pose_with_covariance")?;
```

**File Modified**: `src/cuda_ndt_matcher/src/main.rs:165-167`

## 8.2 Sensor Point Filtering ✅ COMPLETE

**Implemented Features**:
- Distance-based filtering (min/max distance from sensor)
- Z-height filtering (min/max z value for ground/ceiling)
- Voxel grid downsampling
- GPU-accelerated filtering with CPU fallback

**Implementation** in `src/ndt_cuda/src/filtering/`:
```rust
pub struct FilterParams {
    pub min_distance: f32,      // ✅ Implemented
    pub max_distance: f32,      // ✅ Implemented
    pub min_z: f32,             // ✅ Implemented
    pub max_z: f32,             // ✅ Implemented
    pub downsample_resolution: Option<f32>, // ✅ Implemented
}

// GPU filter with automatic CPU fallback for small point clouds
pub struct GpuPointFilter { ... }
pub struct CpuPointFilter { ... }
```

**Files**:
- `src/ndt_cuda/src/filtering/mod.rs` - GpuPointFilter, CpuPointFilter, FilterParams
- `src/ndt_cuda/src/filtering/cpu.rs` - CPU implementation
- `src/ndt_cuda/src/filtering/kernels.rs` - GPU kernels

## 8.3 Non-Blocking Map Updates ✅ COMPLETE

**Implemented**: Dual-NDT architecture for non-blocking map updates.

**Implementation** in `src/cuda_ndt_matcher/src/dual_ndt_manager.rs`:
```rust
pub struct DualNdtManager {
    /// Active NDT manager used for alignment
    active: Arc<RwLock<NdtManager>>,
    /// Updating NDT manager being rebuilt in background
    updating: Arc<Mutex<Option<NdtManager>>>,
    /// Background thread handle
    update_thread: Arc<Mutex<Option<JoinHandle<Result<NdtManager>>>>>,
    /// Flag indicating update is in progress
    update_in_progress: Arc<AtomicBool>,
}

impl DualNdtManager {
    pub fn start_background_update(&self, points: Vec<[f32; 3]>);
    pub fn try_swap(&self) -> bool;
    pub fn get_status(&self) -> UpdateStatus;
}
```

**Features**:
- Background thread rebuilds voxel grid without blocking alignment
- Atomic swap when update completes
- Status tracking (in_progress, pending_points, swap_count, last_update_ms)

## 8.4 TF2 Transform Listener ✅ COMPLETE

**Implemented**: TF2 buffer subscribing to /tf and /tf_static with transform lookups.

**Implementation** in `src/cuda_ndt_matcher/src/tf_handler.rs`:
```rust
pub struct TfHandler {
    buffer: Arc<RwLock<TransformBuffer>>,
    tf_sub: Subscription<TFMessage>,
    tf_static_sub: Subscription<TFMessage>,
}

impl TfHandler {
    pub fn new(node: &Node) -> Result<Arc<Self>>;
    pub fn lookup_transform(
        &self,
        source_frame: &str,
        target_frame: &str,
        time_ns: Option<i64>,
    ) -> Option<Isometry3<f64>>;
}
```

**Features**:
- Subscribes to /tf and /tf_static topics
- Maintains timestamped transform buffer
- Supports time-based lookup with interpolation
- Stale transform detection (>10s warning)

**Use Cases**:
- Transform sensor points from LiDAR frame to base_link
- Handle multi-LiDAR setups with different sensor origins

**Note**: Rust `tf2_ros` bindings may not be mature. Consider using service-based lookup as fallback.

## 8.5 Point2Plane Metric ✅ COMPLETE

**Autoware Feature**: Alternative distance metric using plane-to-point distance instead of full Mahalanobis.

**Implementation**:
```rust
pub enum DistanceMetric {
    /// Full Mahalanobis distance (current implementation)
    PointToDistribution,
    /// Simplified plane-to-point distance
    PointToPlane,
}

// In derivative computation:
match metric {
    DistanceMetric::PointToDistribution => {
        // Current: (x - μ)ᵀ Σ⁻¹ (x - μ)
    }
    DistanceMetric::PointToPlane => {
        // Simplified: ((x - μ) · n)² where n is principal axis
    }
}
```

**Files Modified**:
- `src/ndt_cuda/src/derivatives/cpu.rs`
- `src/ndt_cuda/src/voxel_grid/types.rs` - Store principal axis per voxel

## 8.6 Multi-Grid NDT ✅ COMPLETE

**Autoware Feature**: Experimental multi-resolution voxel grids for coarse-to-fine alignment.

**Implementation Approach**:
```rust
pub struct MultiGridNdt {
    /// Coarse grid (e.g., 4.0m resolution)
    coarse: VoxelGrid,
    /// Fine grid (e.g., 2.0m resolution)
    fine: VoxelGrid,
    /// Optional ultra-fine grid (e.g., 1.0m resolution)
    ultra_fine: Option<VoxelGrid>,
}

impl MultiGridNdt {
    pub fn align(&self, source: &[[f32; 3]], initial: Isometry3<f64>) -> NdtResult {
        // 1. Coarse alignment (few iterations)
        let coarse_result = self.coarse.align(source, initial, max_iter=3);

        // 2. Fine alignment (more iterations)
        let fine_result = self.fine.align(source, coarse_result.pose, max_iter=10);

        fine_result
    }
}
```

## Tests

- [x] Pose publishing with correct topic names (ndt_pose, ndt_pose_with_covariance)
- [x] Sensor point filtering reduces point count appropriately (unit tests)
- [x] Non-blocking map updates with DualNdtManager
- [x] TF lookup works for sensor→base_link (TfHandler)
- [x] Point2Plane metric produces reasonable results (unit tests)
- [x] Multi-grid improves convergence for large initial errors
