# CubeCL NDT Implementation Roadmap

This document outlines the plan to implement custom CUDA kernels for NDT scan matching using [CubeCL](https://github.com/tracel-ai/cubecl), phasing out the fast-gicp dependency.

## Current Status (2026-01-21)

| Phase                             | Status       | Notes                                                    |
|-----------------------------------|--------------|----------------------------------------------------------|
| Phase 1: Voxel Grid               | ✅ Complete  | CPU + GPU hybrid implementation with KD-tree search      |
| Phase 2: Derivatives              | ✅ Complete  | CPU multi-voxel matching, GPU kernels defined            |
| Phase 3: Newton Solver            | ✅ Complete  | More-Thuente line search implemented                     |
| Phase 4: Scoring                  | ✅ Complete  | NVTL and transform probability                           |
| Phase 5: Integration              | ✅ Complete  | API complete, GPU runtime implemented                    |
| Phase 6: Validation               | ⚠️ Partial    | Algorithm verified, rosbag testing pending               |
| Phase 7: ROS Features             | ✅ Complete  | TF, map loading, multi-NDT, Monte Carlo viz, GPU scoring |
| Phase 8: Missing Features         | ✅ Complete  | All sub-phases complete including 8.6 multi-grid         |
| Phase 9: Full GPU Acceleration    | ⚠️ Partial    | 9.1 workaround, 9.2 GPU voxel grid complete              |
| Phase 10: SmartPoseBuffer         | ✅ Complete  | Pose interpolation for timestamp-aligned initial guess   |
| Phase 11: GPU Zero-Copy Voxel     | ✅ Complete  | CubeCL-cuda_ffi interop, radix sort + segment on GPU     |
| Phase 12: GPU Derivative Pipeline | ⚠️ Superseded | Replaced by FullGpuPipelineV2 (Phase 15)                 |
| Phase 13: GPU Scoring Pipeline    | ✅ Complete  | Batched NVTL/TP scoring for MULTI_NDT_SCORE              |
| Phase 14: Full GPU Newton         | ✅ Complete  | GPU Jacobians, cuSOLVER Newton (superseded by Phase 15)  |
| Phase 15: Full GPU + Line Search  | ✅ Complete  | ~200 bytes/iter + batched More-Thuente (K=8 candidates)  |
| Phase 16: GPU Initial Pose        | ✅ Complete  | Batch kernels + pipeline + initial_pose.rs integration   |
| Phase 17: Kernel Optimization     | ✅ Complete  | Persistent kernel with cooperative groups                |
| Phase 18: Persistent Full Features| ✅ Complete  | All 5 features complete including line search (Option A) |
| Phase 19: Cleanup & Enhancements  | ✅ Complete  | Struct cleanup, alpha tracking, per-iteration debug      |
| Phase 22: Batch Multi-Alignment   | ✅ Complete  | All sub-phases complete including 22.5 ROS integration   |
| Phase 23: GPU Utilization         | ⚠️ Partial   | 23.1 complete (async streams), texture/warp pending      |
| Phase 24: CUDA Graphs Pipeline    | ✅ Complete  | All sub-phases (24.1-24.5) complete                      |

**Core NDT algorithm is fully implemented on CPU and matches Autoware's pclomp.**
**GPU runtime uses persistent kernel (single launch) for all optimization.**
**417 tests pass (351 ndt_cuda + 66 cuda_ffi), 6 ignored.**

### Phase 17 Note

The persistent kernel (single kernel launch for entire Newton iteration loop) is now fully functional.
The previous hash table lookup issue was caused by memory visibility problems between CubeCL buffers
and CUDA FFI operations. Fixed by adding `cuda_device_synchronize()` calls to ensure all CubeCL
buffer writes are complete before the cooperative kernel reads from them.

## Phase Documents

- [Phase 1: Voxel Grid Construction](phase-1-voxel-grid.md)
- [Phase 2: Derivative Computation](phase-2-derivatives.md)
- [Phase 3: Newton Optimization](phase-3-newton.md)
- [Phase 4: Scoring & NVTL](phase-4-scoring.md)
- [Phase 5: Integration & Optimization](phase-5-integration.md)
- [Phase 6: Validation & Production](phase-6-validation.md)
- [Phase 7: ROS Integration & Production Features](phase-7-ros-features.md)
- [Phase 8: Missing Features (Autoware Parity)](phase-8-autoware-parity.md)
- [Phase 9: Full GPU Acceleration](phase-9-gpu-acceleration.md)
- [Phase 10: SmartPoseBuffer](phase-10-smart-pose-buffer.md)
- [Phase 11: GPU Zero-Copy Voxel Pipeline](phase-11-gpu-zero-copy-pipeline.md) ✅
- [Phase 12: GPU Derivative Pipeline](phase-12-gpu-derivative-pipeline.md) ✅
- [Phase 13: GPU Scoring Pipeline](phase-13-gpu-scoring-pipeline.md) ✅
- [Phase 14: Full GPU Newton](phase-14-iteration-optimization.md) ✅
- [Phase 15: Full GPU + Line Search](phase-15-gpu-line-search.md) ✅
- [Phase 16: GPU Initial Pose Pipeline](phase-16-gpu-initial-pose-pipeline.md) ✅
- [Phase 17: Kernel Optimization](phase-17-kernel-optimization.md) ✅ - Persistent kernel with cooperative groups
- [Phase 18: Persistent Kernel Features](phase-18-persistent-kernel-features.md) ✅ - All 5 features: Hessian, regularization, correspondences, oscillation, line search
- [Phase 19: Cleanup & Enhancements](phase-19-cleanup.md) ✅ - Struct cleanup, alpha tracking, per-iteration debug
- [Phase 22: Batch Multi-Alignment](phase-22-batch-alignment.md) ✅ - Non-cooperative kernel for parallel multi-scan alignment
- [Phase 23: GPU Utilization](phase-23-gpu-utilization.md) ⚠️ - 23.1 complete, texture/warp pending
- [Phase 24: CUDA Graphs Pipeline](phase-24-cuda-graphs-pipeline.md) ✅ - Replace cooperative kernel for Jetson/small GPU compatibility
- [Implementation Notes](implementation-notes.md) - Dependencies, risks, references

## Background

### Why Replace fast-gicp?

The fast-gicp NDTCuda implementation has fundamental issues:

| Issue                                  | Impact                                                   |
|----------------------------------------|----------------------------------------------------------|
| Uses Levenberg-Marquardt optimizer     | Never converges properly (hits 30 iterations every time) |
| Different cost function (Mahalanobis)  | Different optimization landscape vs pclomp               |
| SO(3) exponential map parameterization | Different Jacobian structure than Euler angles           |
| No exposed iteration count             | Cannot diagnose convergence                              |

### Why CubeCL?

- **Pure Rust**: No C++/CUDA FFI complexity
- **Multi-platform**: CUDA, ROCm, WebGPU from same codebase
- **Type-safe**: Rust's type system for GPU code
- **Automatic vectorization**: `Line<T>` handles SIMD
- **Autotuning**: Runtime optimization of kernel parameters

## Architecture Overview

```
cuda_ndt_matcher/
├── src/
│   ├── cuda_ndt_matcher/           # ROS node (existing)
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── ndt_manager.rs      # Will use new NDT engine
│   │   │   └── ...
│   │   └── Cargo.toml
│   │
│   └── ndt_cuda/                   # NEW: CubeCL NDT library
│       ├── src/
│       │   ├── lib.rs
│       │   ├── voxel_grid/         # Phase 1: Voxelization
│       │   │   ├── mod.rs
│       │   │   ├── kernels.rs      # CubeCL kernels
│       │   │   └── types.rs
│       │   ├── derivatives/        # Phase 2: Derivative computation
│       │   │   ├── mod.rs
│       │   │   ├── jacobian.rs
│       │   │   ├── hessian.rs
│       │   │   └── kernels.rs
│       │   ├── optimization/       # Phase 3: Newton solver
│       │   │   ├── mod.rs
│       │   │   ├── newton.rs
│       │   │   └── line_search.rs
│       │   ├── scoring/            # Phase 4: Probability scoring
│       │   │   ├── mod.rs
│       │   │   └── kernels.rs
│       │   └── ndt.rs              # High-level API
│       ├── benches/
│       └── Cargo.toml
```

## Timeline Summary

### Completed Work

| Phase | Status | Actual Duration |
|-------|--------|-----------------|
| Phase 1: Voxel Grid | ✅ Complete | ~2 weeks |
| Phase 2: Derivatives | ✅ Complete | ~3 weeks |
| Phase 3: Newton Solver | ✅ Complete | ~1.5 weeks |
| Phase 4: Scoring | ✅ Complete | ~1 week |
| Phase 5: Integration (API) | ✅ Complete | ~1.5 weeks |

### Remaining Work

| Phase                                | Estimated Duration | Priority     | Status         |
|--------------------------------------|--------------------|--------------|----------------|
| Phase 6: Validation                  | 1-2 weeks          | High         | ⚠️ Partial     |
| **Total Remaining**                  | **~1-2 weeks**     |              |                |

### Priority Order

1. **Phase 6: Validation** - Run rosbag comparison to verify algorithm correctness
