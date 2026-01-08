# Phase 4: Scoring & NVTL

**Status**: ✅ Complete

## Goal

Compute transform probability and NVTL for quality assessment.

## Components

### 4.1 Transform Probability

```rust
#[cube(launch_unchecked)]
fn compute_transform_probability<F: Float>(
    transformed_points: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    voxel_hash: &Array<u32>,
    gauss_d1: F,
    gauss_d2: F,
    scores: &mut Array<F>,  // Per-point scores
) {
    // Similar to derivative computation but only score
    // ...
}
```

### 4.2 NVTL (Nearest Voxel Transformation Likelihood)

```rust
#[cube(launch_unchecked)]
fn compute_nvtl<F: Float>(
    transformed_points: &Array<F>,
    voxel_means: &Array<F>,
    voxel_inv_covs: &Array<F>,
    voxel_hash: &Array<u32>,
    gauss_d1: F,
    gauss_d2: F,
    nearest_scores: &mut Array<F>,  // Max score per point
) {
    let idx = ABSOLUTE_POS;

    let x_trans = load_vec3(transformed_points, idx);
    let mut max_score = F::new(0.0);

    for offset in NEIGHBOR_OFFSETS {
        let voxel_idx = find_voxel(x_trans, offset, voxel_hash);
        if voxel_idx >= 0 {
            let score = compute_point_score(x_trans, voxel_idx, ...);
            if score > max_score {
                max_score = score;
            }
        }
    }

    nearest_scores[idx] = max_score;
}
```

## Tests

- [x] Transform probability matches CPU implementation
- [x] Per-point scores computed correctly
- [x] NVTL neighbor search finds all relevant voxels (radius search)
- [x] NVTL vs transform probability comparison
- [x] Scoring functions match Autoware's algorithm
- [ ] GPU kernel performance: < 2ms for scoring
