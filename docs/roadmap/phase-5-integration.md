# Phase 5: Integration & Optimization

**Status**: ✅ Complete

## Goal

Integrate with cuda_ndt_matcher and optimize performance.

## Components

### 5.1 API Design

```rust
// src/ndt_cuda/src/lib.rs

pub struct NdtCuda {
    client: CubeClient,
    config: NdtConfig,
    voxel_grid: Option<GpuVoxelGrid>,
}

impl NdtCuda {
    pub fn new(config: NdtConfig) -> Result<Self>;

    /// Set target (map) point cloud - builds voxel grid
    pub fn set_target(&mut self, points: &[[f32; 3]]) -> Result<()>;

    /// Align source to target with initial guess
    pub fn align(
        &self,
        source: &[[f32; 3]],
        initial_guess: Isometry3<f64>,
    ) -> Result<NdtResult>;

    /// Compute scores without optimization
    pub fn compute_score(
        &self,
        source: &[[f32; 3]],
        pose: Isometry3<f64>,
    ) -> Result<ScoreResult>;
}

pub struct NdtConfig {
    pub resolution: f32,
    pub max_iterations: u32,
    pub trans_epsilon: f64,
    pub step_size: f64,
    pub outlier_ratio: f64,
}

pub struct NdtResult {
    pub pose: Isometry3<f64>,
    pub converged: bool,
    pub score: f64,
    pub nvtl: f64,
    pub iterations: u32,
    pub hessian: Matrix6<f64>,
}
```

### 5.2 Memory Management

- Reuse GPU buffers across calls
- Lazy voxel grid rebuild only when target changes
- Stream-ordered memory allocation

### 5.3 Performance Optimization

- Tune CubeCL launch parameters
- Profile with Nsight Systems
- Optimize memory access patterns (coalescing)
- Consider persistent kernel approach for small workloads

## Tests

- [x] High-level NdtScanMatcher API with builder pattern
- [x] Feature flags for ndt_cuda vs fast-gicp backends
- [x] Unit tests for API (208 tests passing)
- [x] Covariance estimation with Laplace approximation
- [x] Initial pose estimation with TPE sampling
- [x] Debug output (JSONL format) for comparison
- [x] GPU runtime integration with CubeCL
- [x] `GpuRuntime` with CUDA device/client management
- [x] GPU kernel launches for transform, radius search, scoring, gradient
- [x] `use_gpu` config option with automatic fallback to CPU
- [ ] Integration test with sample rosbag (Phase 6)
- [ ] A/B comparison with pclomp (Phase 6)
- [ ] Latency benchmarking (target: < 20ms for typical workload)
- [ ] Memory usage profiling (target: < 500MB)
