# Phase 6: Validation & Production

**Status**: ⚠️ Partial - Algorithm verified, rosbag testing pending

## Goal

Validate against pclomp and prepare for production.

## Components

### 6.1 Numerical Validation

- Compare every intermediate value with pclomp
- Log divergence points
- Create regression test suite

### 6.2 Edge Cases

- Empty point clouds
- Single-voxel scenes
- Degenerate covariances
- Large initial pose errors

### 6.3 Documentation

- API documentation
- Performance tuning guide
- Troubleshooting guide

## Validation Status

**Core algorithm verified correct** - When map coverage is good, our scores match Autoware's:
- Score per point: 4-5 (matching Autoware's 5.2)
- Convergence: 95%+ in good map coverage areas
- Multi-voxel radius search: ~2.4 voxels/point

**Known issue: Voxels-per-point ratio**

With real rosbag data, we observe lower voxels-per-point than Autoware:

| Metric | CUDA | Autoware | Notes |
|--------|------|----------|-------|
| Voxels per point | ~1.2 | ~3.4 | 2.8x difference |
| Hessian magnitude | ~1e5 | ~1e7 | 100x smaller |
| Hessian eigenvalues | Mixed | All negative | Causes direction reversal |
| Convergence | 30 iters (max) | 3 iters | Non-convergent |

Root cause under investigation - likely related to:
- Different source point counts (756 vs 1459)
- Possible map loading or filtering differences

Use `NDT_DEBUG_VPP=1` to diagnose voxel-per-point distribution.

**Remaining issues:**
- Lower convergence in sparse map areas or during map transitions
- Need better filtering to skip NDT when conditions are poor

**Derivative formulas verified correct:**
- Jacobian formulas (∂T/∂pose) match `AngularDerivatives` reference exactly
- Point Hessian formulas (∂²T/∂pose²) match `AngularDerivatives` reference exactly
- CPU vs GPU derivative computation matches (score, gradient, Hessian diagonal signs)
- All 15 h_ang terms (a2, a3, b2, b3, c2, c3, d1-d3, e1-e3, f1-f3) verified

**Bug fixes applied:**
- Fixed duplicate message processing (was ~17x duplicates due to out-of-order delivery)
- Fixed `use_line_search` default (was true, should be false like Autoware)
- Fixed Hessian regularization (was 1e-6, should be 0 like Autoware)
