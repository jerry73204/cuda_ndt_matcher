# Phase 1: Voxel Grid Construction

**Status**: ✅ Complete

## Goal

Build GPU-accelerated voxel grid from point cloud map with covariance computation.

## Components

### 1.1 Voxel ID Computation

```rust
#[cube(launch_unchecked)]
fn compute_voxel_ids<F: Float>(
    points: &Array<Line<F>>,      // [N, 3] point cloud
    voxel_ids: &mut Array<u32>,   // [N] output voxel IDs
    min_bound: &Array<F>,         // [3] bounding box min
    resolution: F,
    grid_dims: &Array<u32>,       // [3] grid dimensions
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = (points[idx * 3 + 0] - min_bound[0]) / resolution;
    let y = (points[idx * 3 + 1] - min_bound[1]) / resolution;
    let z = (points[idx * 3 + 2] - min_bound[2]) / resolution;

    let ix = Line::cast_from(x);
    let iy = Line::cast_from(y);
    let iz = Line::cast_from(z);

    voxel_ids[idx] = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1];
}
```

### 1.2 Point Accumulation (Parallel Histogram)

- Sort points by voxel ID (use `cub::DeviceRadixSort` via cuBLAS or implement in CubeCL)
- Segment reduce to compute per-voxel statistics

### 1.3 Covariance Computation

```rust
#[cube(launch_unchecked)]
fn compute_covariance<F: Float>(
    // Per-voxel accumulated statistics
    point_sums: &Array<F>,        // [V, 3] sum of points
    point_sq_sums: &Array<F>,     // [V, 6] sum of x*x^T (upper triangle)
    point_counts: &Array<u32>,    // [V] count per voxel

    // Output
    means: &mut Array<F>,         // [V, 3] voxel centroids
    covariances: &mut Array<F>,   // [V, 9] 3x3 covariance matrices
    inv_covariances: &mut Array<F>, // [V, 9] inverse covariances
) {
    let voxel_idx = ABSOLUTE_POS;
    let count = point_counts[voxel_idx];

    if count < 6 {  // Minimum points for valid covariance
        return;
    }

    // Compute mean
    let mean_x = point_sums[voxel_idx * 3 + 0] / F::cast_from(count);
    // ... mean_y, mean_z

    // Compute covariance using single-pass formula:
    // cov = (sum_sq - sum * mean^T) / (n - 1)
    // ...

    // Regularize via eigendecomposition (or simplified approach)
    // ...

    // Compute inverse
    // ...
}
```

### 1.4 Eigenvalue Regularization

For singular covariances, inflate small eigenvalues:
- Option A: Use cuSOLVER for batched eigendecomposition
- Option B: Implement power iteration in CubeCL (simpler, may be sufficient)
- Option C: Use Jacobi eigenvalue algorithm (pure CubeCL)

## Data Structures

```rust
/// GPU voxel grid
pub struct GpuVoxelGrid {
    /// Voxel centroids [V, 3]
    pub means: CubeBuffer<f32>,
    /// Inverse covariances [V, 9] (row-major 3x3)
    pub inv_covariances: CubeBuffer<f32>,
    /// Point counts per voxel
    pub counts: CubeBuffer<u32>,
    /// Voxel coordinates for spatial lookup
    pub coords: CubeBuffer<i32>,  // [V, 3]
    /// Hash table for O(1) voxel lookup
    pub hash_table: GpuHashTable,
    /// Grid parameters
    pub resolution: f32,
    pub min_bound: [f32; 3],
}
```

## Tests

- [x] Voxel ID matches CPU implementation
- [x] Mean/covariance matches within floating-point tolerance
- [x] Inverse covariance is valid (no NaN/Inf)
- [x] KD-tree radius search returns correct voxels
- [x] Multi-voxel correspondences match Autoware behavior
- [x] GPU voxel grid construction (test_gpu_voxel_grid_construction)
- [x] GPU/CPU consistency verified (test_gpu_cpu_consistency)
- [ ] GPU kernel performance: < 10ms for 100K point cloud (benchmark pending)
