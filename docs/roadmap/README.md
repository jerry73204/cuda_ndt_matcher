# CubeCL NDT Implementation Roadmap

This document outlines the plan to implement custom CUDA kernels for NDT scan matching using [CubeCL](https://github.com/tracel-ai/cubecl), phasing out the fast-gicp dependency.

## Current Status (2026-01-05)

| Phase                          | Status         | Notes                                                    |
|--------------------------------|----------------|----------------------------------------------------------|
| Phase 1: Voxel Grid            | ✅ Complete    | CPU + GPU hybrid implementation with KD-tree search      |
| Phase 2: Derivatives           | ✅ Complete    | CPU multi-voxel matching, GPU kernels defined            |
| Phase 3: Newton Solver         | ✅ Complete    | More-Thuente line search implemented                     |
| Phase 4: Scoring               | ✅ Complete    | NVTL and transform probability                           |
| Phase 5: Integration           | ✅ Complete    | API complete, GPU runtime implemented                    |
| Phase 6: Validation            | ⚠️ Partial      | Algorithm verified, rosbag testing pending               |
| Phase 7: ROS Features          | ✅ Complete    | TF, map loading, multi-NDT, Monte Carlo viz, GPU scoring |
| Phase 8: Missing Features      | ✅ Complete    | All sub-phases complete including 8.6 multi-grid         |
| Phase 9: Full GPU Acceleration | ⚠️ Partial     | 9.1 workaround, 9.2 GPU voxel grid complete              |
| Phase 10: SmartPoseBuffer      | ✅ Complete    | Pose interpolation for timestamp-aligned initial guess   |
| Phase 11: GPU Zero-Copy Pipeline | 🔲 Planned   | Eliminate CPU-GPU transfers between pipeline stages      |

**Core NDT algorithm is fully implemented on CPU and matches Autoware's pclomp.**
**GPU runtime is implemented with CubeCL for accelerated scoring and voxel grid construction.**
**351 tests pass (282 ndt_cuda + 56 cuda_ndt_matcher + 13 cuda_ffi). All GPU tests enabled and passing.**

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
- [Phase 11: GPU Zero-Copy Pipeline](phase-11-gpu-zero-copy-pipeline.md)
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
| Phase 6: Validation                  | 1-2 weeks          | High         | Pending        |
| Phase 9.3: GPU Derivatives           | 1-2 weeks          | Medium       | 🔲 Not started |
| Phase 9.4: GPU Memory Pooling        | 3-4 days           | Low          | 🔲 Not started |
| Phase 9.5: Async GPU Execution       | 1 week             | Low          | 🔲 Not started |
| Phase 11: GPU Zero-Copy Pipeline     | 1-2 weeks          | Medium       | 🔲 Not started |
| **Total Remaining**                  | **~3-4 weeks**     |              | 6, 9.3-9.5, 11 |

### Priority Order

1. **Phase 6: Validation** - Run rosbag comparison to verify algorithm correctness
2. **Phase 9.3: GPU Derivatives** - Performance improvement for optimization loop
3. **Phase 11: GPU Zero-Copy Pipeline** - Eliminate CPU-GPU transfers (3x fewer transfers)
4. **Phase 9.4-9.5: GPU Optimization** - Memory pooling and async execution
