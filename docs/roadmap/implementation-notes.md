# Implementation Notes

This document covers dependencies, risks, known issues, and references for the CubeCL NDT implementation.

## Code TODOs

Outstanding TODO comments in the codebase that represent integration or improvement work:

| Location | Description | Priority |
|----------|-------------|----------|
| `src/ndt_cuda/src/optimization/solver.rs:431` | Proper score normalization based on number of points | Low |

### Integration Tasks

These are the key integration items needed for full Autoware compatibility:

1. ~~**GPU Scoring Path** (`ndt.rs`)~~ ✅ Complete
   - `evaluate_transform_probability()` uses GPU (sum-based aggregation)
   - `evaluate_nvtl()` uses GPU (max-per-point, Autoware-compatible)

2. **Score Normalization** (`solver.rs:431`)
   - Currently: Raw score returned without normalization
   - Target: Normalize by number of correspondences for consistent comparison
   - Note: May affect convergence thresholds

3. ~~**Test Map Loading** (`main.rs:990`)~~ ✅ Complete
   - Dynamic map loading with `pcd_loader_service` implemented
   - `should_update()` triggers initial map load when no tiles loaded
   - Diagnostics for `is_succeed_call_pcd_loader` added
   - Target: Add option to load map from file for standalone testing
   - Benefit: Easier debugging without full Autoware stack

## Dependencies

### Rust Crates

```toml
[dependencies]
cubecl = { version = "0.4", features = ["cuda"] }
cubecl-cuda = "0.4"
nalgebra = "0.33"
```

### Build Requirements

- CUDA Toolkit 12.x
- Rust nightly (for CubeCL proc macros)
- NVIDIA GPU (compute capability 7.0+)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CubeCL alpha stability | Pin version, contribute fixes upstream |
| Eigendecomposition complexity | Start with simplified regularization |
| Hash table performance | Profile early, consider alternatives |
| Numerical precision | Use f64 for accumulation, f32 for storage |

## Success Criteria

### Algorithm (Verified ✅)

1. **Core Algorithm**: Multi-voxel radius search matches Autoware's pclomp ✅
2. **Gaussian Parameters**: d1, d2, d3 match Magnusson 2009 exactly ✅
3. **Score/Gradient/Hessian**: Match Autoware's equations ✅
4. **Convergence**: < 10 iterations for typical scenarios ✅
5. **Stability**: Handles edge cases (singular Hessian, no correspondences) ✅

### Integration (Pending)

1. **Pose Accuracy**: Final pose within 1cm / 0.1° of pclomp (rosbag validation)
2. **ROS Interface**: TF broadcast, diagnostics, full topic compatibility
3. **Large Maps**: Dynamic tile-based loading with pcd_loader service
4. **Reliability**: No crashes during continuous operation

### Performance (Pending GPU)

1. **Latency**: < 20ms for typical workload (currently ~50ms on CPU)
2. **Memory**: < 500MB GPU memory usage
3. **Throughput**: 10+ Hz alignment rate

## Known Issues & Workarounds

### CubeCL Optimizer Bug (cubecl-opt-0.8.1)

**Issue**: CubeCL's uniformity analysis in `cubecl-opt-0.8.1` panics with "no entry found for key" when compiling kernels with complex control flow patterns.

**Trigger Pattern**:
```rust
// This pattern triggers the bug:
for v in 0..num_voxels {  // Dynamic runtime loop bound
    if count >= MAX_NEIGHBORS {
        break;            // Early exit
    }
    // ... conditional logic
}
```

**Workaround**: Avoid `break` statements in loops with dynamic bounds. Use a conditional flag instead:
```rust
// This works:
for v in 0..num_voxels {
    let should_process = count < MAX_NEIGHBORS;
    if should_process {
        // ... conditional logic
    }
}
```

**Applied in**: `src/ndt_cuda/src/derivatives/gpu.rs` - `radius_search_kernel`

**Status**: Workaround implemented. Tests pass. Bug not yet fixed upstream in CubeCL.

## References

1. Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform. PhD Thesis.
2. [CubeCL Documentation](https://github.com/tracel-ai/cubecl)
3. [Autoware NDT Implementation](https://github.com/autowarefoundation/autoware.universe)
4. [More-Thuente Line Search](https://www.ii.uib.no/~lennMDL/talks/MT-paper.pdf)

## Sources

- [CubeCL GitHub](https://github.com/tracel-ai/cubecl)
- [Rust-CUDA Project](https://github.com/Rust-GPU/Rust-CUDA)
- [Burn Deep Learning Framework](https://burn.dev/blog/going-big-and-small-for-2025/)
- [CubeCL Architecture Overview](https://gist.github.com/nihalpasham/570d4fe01b403985e1eaf620b6613774)
