# Phase 3: Newton Optimization

**Status**: ✅ Complete

## Goal

Implement Newton's method with optional More-Thuente line search.

## Components

### 3.1 Newton Step (CPU)

The 6x6 linear solve is too small for GPU benefit:

```rust
pub fn newton_step(
    gradient: &Vector6<f64>,
    hessian: &Matrix6<f64>,
) -> Vector6<f64> {
    // SVD solve: delta = -H^{-1} * g
    let svd = hessian.svd(true, true);
    svd.solve(gradient, 1e-10).unwrap() * -1.0
}
```

### 3.2 Transformation Update (GPU)

Apply delta to transformation and transform all points:

```rust
#[cube(launch_unchecked)]
fn transform_points<F: Float>(
    points: &Array<F>,           // [N, 3] source points
    transform: &Array<F>,        // [16] 4x4 transformation matrix
    output: &mut Array<F>,       // [N, 3] transformed points
) {
    let idx = ABSOLUTE_POS;
    if idx >= points.len() / 3 {
        return;
    }

    let x = points[idx * 3 + 0];
    let y = points[idx * 3 + 1];
    let z = points[idx * 3 + 2];

    // T * [x, y, z, 1]^T
    output[idx * 3 + 0] = transform[0]*x + transform[1]*y + transform[2]*z + transform[3];
    output[idx * 3 + 1] = transform[4]*x + transform[5]*y + transform[6]*z + transform[7];
    output[idx * 3 + 2] = transform[8]*x + transform[9]*y + transform[10]*z + transform[11];
}
```

### 3.3 Convergence Check

```rust
pub fn check_convergence(
    delta: &Vector6<f64>,
    trans_epsilon: f64,
    iteration: usize,
    max_iterations: usize,
) -> bool {
    iteration >= max_iterations || delta.norm() < trans_epsilon
}
```

### 3.4 Line Search (Optional)

More-Thuente line search - currently disabled in Autoware due to local minima issues.
Implement as optional feature for experimentation.

## Main Loop

```rust
pub fn align(
    &mut self,
    source: &GpuPointCloud,
    initial_guess: Isometry3<f64>,
) -> NdtResult {
    let mut transform = initial_guess;

    for iteration in 0..self.max_iterations {
        // 1. Transform source points (GPU)
        self.transform_points(source, &transform);

        // 2. Compute angular derivatives (CPU, tiny)
        let ang_deriv = compute_angular_derivatives(&transform);

        // 3. Compute point Jacobians (GPU)
        self.compute_point_jacobians(source, &ang_deriv);

        // 4. Compute gradient & Hessian (GPU)
        let (score, gradient, hessian) = self.compute_derivatives();

        // 5. Newton step (CPU, 6x6 solve)
        let delta = newton_step(&gradient, &hessian);

        // 6. Update transform
        transform = apply_delta(transform, delta);

        // 7. Check convergence
        if check_convergence(&delta, self.trans_epsilon, iteration, self.max_iterations) {
            break;
        }
    }

    NdtResult { transform, score, iterations, hessian }
}
```

## Tests

- [x] Convergence within 10 iterations for good initial guess
- [x] More-Thuente line search implemented and tested
- [x] Step size clamping matches Autoware behavior
- [x] Handles edge cases (no correspondences, singular Hessian)
- [ ] Final pose matches pclomp within 1cm / 0.1 degree (rosbag validation pending)
