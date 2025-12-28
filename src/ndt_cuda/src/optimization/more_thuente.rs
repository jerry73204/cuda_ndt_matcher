//! More-Thuente line search algorithm.
//!
//! This module implements the More-Thuente line search algorithm for finding
//! step lengths that satisfy the strong Wolfe conditions. This is the algorithm
//! used by Autoware's NDT implementation.
//!
//! # References
//!
//! - More, J.J. & Thuente, D.J. (1994). "Line search algorithms with guaranteed
//!   sufficient decrease". ACM Transactions on Mathematical Software.
//! - Sun, W. & Yuan, Y. (2006). "Optimization Theory and Methods: Nonlinear Programming".

/// Configuration for More-Thuente line search.
#[derive(Debug, Clone)]
pub struct MoreThuenteConfig {
    /// Maximum step length (Autoware default: 0.1)
    pub step_max: f64,

    /// Minimum step length (Autoware default: 1e-9)
    pub step_min: f64,

    /// Sufficient decrease parameter (mu/c1 in Wolfe conditions)
    /// Default: 1e-4 (Autoware)
    pub mu: f64,

    /// Curvature condition parameter (nu/c2 in Wolfe conditions)
    /// Default: 0.9 (Autoware)
    pub nu: f64,

    /// Maximum line search iterations
    /// Default: 10 (Autoware)
    pub max_iterations: usize,
}

impl Default for MoreThuenteConfig {
    fn default() -> Self {
        Self {
            step_max: 0.1,
            step_min: 1e-9,
            mu: 1e-4,
            nu: 0.9,
            max_iterations: 10,
        }
    }
}

/// Result of More-Thuente line search.
#[derive(Debug, Clone)]
pub struct MoreThuenteResult {
    /// Final step length
    pub step_length: f64,

    /// Whether the search converged (satisfies Wolfe conditions)
    pub converged: bool,

    /// Number of iterations performed
    pub iterations: usize,

    /// Final function value (phi(alpha))
    pub final_value: f64,

    /// Final derivative (phi'(alpha))
    pub final_derivative: f64,
}

/// Auxiliary function psi used in More-Thuente algorithm.
///
/// psi(alpha) = phi(alpha) - phi(0) - mu * phi'(0) * alpha
///
/// Equation 1.6 in More-Thuente (1994)
#[inline]
fn auxiliary_psi(alpha: f64, phi_alpha: f64, phi_0: f64, dphi_0: f64, mu: f64) -> f64 {
    phi_alpha - phi_0 - mu * dphi_0 * alpha
}

/// Derivative of auxiliary function psi.
///
/// psi'(alpha) = phi'(alpha) - mu * phi'(0)
#[inline]
fn auxiliary_dpsi(dphi_alpha: f64, dphi_0: f64, mu: f64) -> f64 {
    dphi_alpha - mu * dphi_0
}

/// Select trial value based on cubic/quadratic interpolation.
///
/// Implements the trial value selection algorithm from More-Thuente (1994),
/// Section 4, with interpolation formulas from Sun & Yuan (2006).
fn trial_value_selection(
    a_l: f64,
    f_l: f64,
    g_l: f64,
    a_u: f64,
    f_u: f64,
    g_u: f64,
    a_t: f64,
    f_t: f64,
    g_t: f64,
) -> f64 {
    // Case 1: f_t > f_l
    // The minimizer is bracketed between a_l and a_t
    if f_t > f_l {
        // Cubic interpolant minimizer (Eq. 2.4.52 Sun & Yuan)
        let z = 3.0 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
        let w_sq = z * z - g_t * g_l;
        if w_sq < 0.0 {
            // Fallback to quadratic
            return a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));
        }
        let w = w_sq.sqrt();
        let a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2.0 * w);

        // Quadratic interpolant minimizer (Eq. 2.4.2 Sun & Yuan)
        let denom = g_l - (f_l - f_t) / (a_l - a_t);
        if denom.abs() < 1e-15 {
            return a_c;
        }
        let a_q = a_l - 0.5 * (a_l - a_t) * g_l / denom;

        // Choose the one closer to a_l
        if (a_c - a_l).abs() < (a_q - a_l).abs() {
            return a_c;
        }
        return 0.5 * (a_q + a_c);
    }

    // Case 2: f_t <= f_l and g_t * g_l < 0
    // The minimizer is bracketed between a_l and a_t
    if g_t * g_l < 0.0 {
        // Cubic interpolant minimizer
        let z = 3.0 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
        let w_sq = z * z - g_t * g_l;
        if w_sq < 0.0 {
            // Fallback to secant
            return a_l - (a_l - a_t) / (g_l - g_t) * g_l;
        }
        let w = w_sq.sqrt();
        let a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2.0 * w);

        // Secant step (Eq. 2.4.5 Sun & Yuan)
        let a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

        // Choose the one further from a_t
        if (a_c - a_t).abs() >= (a_s - a_t).abs() {
            return a_c;
        }
        return a_s;
    }

    // Case 3: f_t <= f_l and g_t * g_l >= 0 and |g_t| <= |g_l|
    if g_t.abs() <= g_l.abs() {
        // Cubic interpolant minimizer
        let z = 3.0 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
        let w_sq = z * z - g_t * g_l;

        let a_c = if w_sq >= 0.0 {
            let w = w_sq.sqrt();
            a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2.0 * w)
        } else {
            // Extrapolate
            if a_t > a_l {
                a_t + 0.66 * (a_u - a_t)
            } else {
                a_t - 0.66 * (a_t - a_u)
            }
        };

        // Secant step
        let a_s = if (g_l - g_t).abs() > 1e-15 {
            a_l - (a_l - a_t) / (g_l - g_t) * g_l
        } else {
            a_c
        };

        let a_t_next = if (a_c - a_t).abs() < (a_s - a_t).abs() {
            a_c
        } else {
            a_s
        };

        // Ensure we stay within bounds
        if a_t > a_l {
            return (a_t + 0.66 * (a_u - a_t)).min(a_t_next);
        }
        return (a_t + 0.66 * (a_u - a_t)).max(a_t_next);
    }

    // Case 4: f_t <= f_l and g_t * g_l >= 0 and |g_t| > |g_l|
    // Cubic interpolant using a_u
    let z = 3.0 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
    let w_sq = z * z - g_t * g_u;
    if w_sq < 0.0 {
        // Fallback
        return 0.5 * (a_l + a_u);
    }
    let w = w_sq.sqrt();
    a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2.0 * w)
}

/// Update interval endpoints according to More-Thuente algorithm.
///
/// Returns true if the interval has converged (degenerated to a point).
fn update_interval(
    a_l: &mut f64,
    f_l: &mut f64,
    g_l: &mut f64,
    a_u: &mut f64,
    f_u: &mut f64,
    g_u: &mut f64,
    a_t: f64,
    f_t: f64,
    g_t: f64,
) -> bool {
    // Case U1: f_t > f_l
    // The minimizer lies in [a_l, a_t]
    if f_t > *f_l {
        *a_u = a_t;
        *f_u = f_t;
        *g_u = g_t;
        return false;
    }

    // Case U2: f_t <= f_l and g_t * (a_l - a_t) > 0
    // The minimizer lies in [a_t, a_u]
    if g_t * (*a_l - a_t) > 0.0 {
        *a_l = a_t;
        *f_l = f_t;
        *g_l = g_t;
        return false;
    }

    // Case U3: f_t <= f_l and g_t * (a_l - a_t) < 0
    // The minimizer lies in [a_t, a_l]
    if g_t * (*a_l - a_t) < 0.0 {
        *a_u = *a_l;
        *f_u = *f_l;
        *g_u = *g_l;

        *a_l = a_t;
        *f_l = f_t;
        *g_l = g_t;
        return false;
    }

    // Interval converged (g_t * (a_l - a_t) == 0)
    true
}

/// Perform More-Thuente line search.
///
/// This function finds a step length `alpha` that satisfies the strong Wolfe conditions:
/// 1. Sufficient decrease: phi(alpha) <= phi(0) + mu * alpha * phi'(0)
/// 2. Curvature condition: |phi'(alpha)| <= nu * |phi'(0)|
///
/// # Arguments
///
/// * `score_and_derivative` - Function that computes (score, directional_derivative) at a step length.
///                           The score should be MAXIMIZED (NDT convention).
/// * `initial_score` - Score at step length 0 (phi(0) = -score for minimization)
/// * `initial_derivative` - Directional derivative at step length 0 (should be positive for ascent)
/// * `initial_step` - Initial step length to try
/// * `config` - Line search configuration
///
/// # Returns
///
/// Result containing the final step length and convergence status.
pub fn more_thuente_search<F>(
    score_and_derivative: F,
    initial_score: f64,
    initial_derivative: f64,
    initial_step: f64,
    config: &MoreThuenteConfig,
) -> MoreThuenteResult
where
    F: Fn(f64) -> (f64, f64), // (score, directional_derivative)
{
    // For NDT, we MAXIMIZE score, but More-Thuente is for minimization.
    // Convert: phi = -score, so minimizing phi = maximizing score
    // phi'(0) = -d(score)/d(alpha) = -initial_derivative
    //
    // For descent direction, we need phi'(0) < 0, i.e., initial_derivative > 0

    // If initial_derivative <= 0, not an ascent direction
    if initial_derivative <= 0.0 {
        return MoreThuenteResult {
            step_length: 0.0,
            converged: false,
            iterations: 0,
            final_value: initial_score,
            final_derivative: initial_derivative,
        };
    }

    // Convert to minimization convention (phi = -score)
    let phi_0 = -initial_score;
    let dphi_0 = -initial_derivative; // This is negative (descent direction)

    // Initial step
    let mut a_t = initial_step.clamp(config.step_min, config.step_max);

    // Evaluate at initial step
    let (score_t, deriv_t) = score_and_derivative(a_t);
    let mut phi_t = -score_t;
    let mut dphi_t = -deriv_t;

    // Check if we can skip line search (step_min >= step_max)
    if config.step_max - config.step_min < 0.0 {
        return MoreThuenteResult {
            step_length: a_t,
            converged: true,
            iterations: 0,
            final_value: score_t,
            final_derivative: deriv_t,
        };
    }

    // Initialize interval endpoints
    let mut a_l = 0.0;
    let mut a_u = 0.0;

    // Using auxiliary function psi initially (open interval)
    let mut f_l = auxiliary_psi(a_l, phi_0, phi_0, dphi_0, config.mu);
    let mut g_l = auxiliary_dpsi(dphi_0, dphi_0, config.mu);
    let mut f_u = auxiliary_psi(a_u, phi_0, phi_0, dphi_0, config.mu);
    let mut g_u = auxiliary_dpsi(dphi_0, dphi_0, config.mu);

    let mut open_interval = true;
    let mut iterations = 0;

    // Calculate psi at current trial point
    let mut psi_t = auxiliary_psi(a_t, phi_t, phi_0, dphi_0, config.mu);
    let mut dpsi_t = auxiliary_dpsi(dphi_t, dphi_0, config.mu);

    // Main loop
    while iterations < config.max_iterations {
        // Check Wolfe conditions (in terms of psi/phi):
        // Sufficient decrease: psi(a_t) <= 0
        // Curvature: |phi'(a_t)| <= nu * |phi'(0)|
        let sufficient_decrease = psi_t <= 0.0;
        let curvature = dphi_t.abs() <= config.nu * dphi_0.abs();

        if sufficient_decrease && curvature {
            // Wolfe conditions satisfied
            return MoreThuenteResult {
                step_length: a_t,
                converged: true,
                iterations,
                final_value: -phi_t, // Convert back to score
                final_derivative: -dphi_t,
            };
        }

        // Select new trial value
        if open_interval {
            a_t = trial_value_selection(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, dpsi_t);
        } else {
            a_t = trial_value_selection(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, dphi_t);
        }

        // Clamp to bounds
        a_t = a_t.clamp(config.step_min, config.step_max);

        // Evaluate at new trial point
        let (score_t_new, deriv_t_new) = score_and_derivative(a_t);
        phi_t = -score_t_new;
        dphi_t = -deriv_t_new;

        // Update psi values
        psi_t = auxiliary_psi(a_t, phi_t, phi_0, dphi_0, config.mu);
        dpsi_t = auxiliary_dpsi(dphi_t, dphi_0, config.mu);

        // Check if interval should become closed
        if open_interval && psi_t <= 0.0 && dpsi_t >= 0.0 {
            open_interval = false;

            // Convert f_l, g_l, f_u, g_u from psi to phi
            f_l = f_l + phi_0 - config.mu * dphi_0 * a_l;
            g_l = g_l + config.mu * dphi_0;
            f_u = f_u + phi_0 - config.mu * dphi_0 * a_u;
            g_u = g_u + config.mu * dphi_0;
        }

        // Update interval
        let interval_converged = if open_interval {
            update_interval(&mut a_l, &mut f_l, &mut g_l, &mut a_u, &mut f_u, &mut g_u, a_t, psi_t, dpsi_t)
        } else {
            update_interval(&mut a_l, &mut f_l, &mut g_l, &mut a_u, &mut f_u, &mut g_u, a_t, phi_t, dphi_t)
        };

        if interval_converged {
            break;
        }

        iterations += 1;
    }

    // Return best result found
    MoreThuenteResult {
        step_length: a_t,
        converged: false,
        iterations,
        final_value: -phi_t,
        final_derivative: -dphi_t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_config_default() {
        let config = MoreThuenteConfig::default();
        assert_eq!(config.step_max, 0.1);
        assert_eq!(config.mu, 1e-4);
        assert_eq!(config.nu, 0.9);
    }

    #[test]
    fn test_auxiliary_psi() {
        // psi(0) should be 0
        let psi = auxiliary_psi(0.0, 1.0, 1.0, -0.5, 1e-4);
        assert_relative_eq!(psi, 0.0, epsilon = 1e-10);

        // psi(alpha) = phi(alpha) - phi(0) - mu * phi'(0) * alpha
        let psi = auxiliary_psi(1.0, 0.5, 1.0, -0.5, 1e-4);
        let expected = 0.5 - 1.0 - 1e-4 * (-0.5) * 1.0;
        assert_relative_eq!(psi, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_minimization() {
        // Minimize f(x) = (x - 2)^2, equivalently maximize -f(x)
        // f'(x) = 2(x - 2)
        // Starting at x = 0, search direction = +1 (toward minimum)
        // f(0) = 4, f'(0) = -4
        // Score = -f, so score(0) = -4, d(score)/dx = -f'(x) = 4

        let config = MoreThuenteConfig {
            step_max: 5.0,
            step_min: 0.0,
            mu: 1e-4,
            nu: 0.9,
            max_iterations: 20,
        };

        let score_and_deriv = |alpha: f64| {
            let x = alpha; // Starting at 0, direction +1
            let f = (x - 2.0).powi(2);
            let df = 2.0 * (x - 2.0);
            let score = -f;
            let dscore = -df; // d(score)/d(alpha) = -f'(x) * dx/d(alpha) = -f'(x) * 1
            (score, dscore)
        };

        let initial_score = -4.0; // -f(0)
        let initial_derivative = 4.0; // -f'(0) = 4

        let result = more_thuente_search(
            score_and_deriv,
            initial_score,
            initial_derivative,
            1.0,
            &config,
        );

        // Wolfe conditions are satisfied at step=1.0:
        // - Sufficient decrease: psi(1) = phi(1) - phi(0) - mu*dphi(0)*1 = 1 - 4 + 0.0004 < 0 ✓
        // - Curvature: |dphi(1)| = 2 <= 0.9 * 4 = 3.6 ✓
        // The algorithm correctly accepts step=1 even though optimum is at step=2
        assert!(result.converged, "should converge");
        assert!(result.step_length >= 1.0, "step = {}", result.step_length);
        assert!(result.step_length <= 2.5, "step = {}", result.step_length);
        // Score improved: score(0) = -4, score(1) = -1
        assert!(result.final_value > initial_score);
    }

    #[test]
    fn test_no_improvement_direction() {
        // If initial_derivative <= 0, should return immediately
        let config = MoreThuenteConfig::default();

        let score_and_deriv = |_alpha: f64| (0.0, 0.0);

        let result = more_thuente_search(score_and_deriv, 1.0, 0.0, 0.1, &config);

        assert_eq!(result.step_length, 0.0);
        assert!(!result.converged);
    }

    #[test]
    fn test_negative_derivative() {
        // Negative derivative means we're going downhill (wrong direction)
        let config = MoreThuenteConfig::default();

        let score_and_deriv = |_alpha: f64| (0.0, 0.0);

        let result = more_thuente_search(score_and_deriv, 1.0, -1.0, 0.1, &config);

        assert_eq!(result.step_length, 0.0);
        assert!(!result.converged);
    }

    #[test]
    fn test_simple_line_search() {
        // Simple test: score increases linearly, then decreases
        // score(alpha) = 10 * alpha - alpha^2 for alpha in [0, 10]
        // score'(alpha) = 10 - 2*alpha
        // Maximum at alpha = 5

        let config = MoreThuenteConfig {
            step_max: 10.0,
            step_min: 0.0,
            mu: 1e-4,
            nu: 0.9,
            max_iterations: 20,
        };

        let score_and_deriv = |alpha: f64| {
            let score = 10.0 * alpha - alpha * alpha;
            let dscore = 10.0 - 2.0 * alpha;
            (score, dscore)
        };

        let initial_score = 0.0;
        let initial_derivative = 10.0;

        let result = more_thuente_search(
            score_and_deriv,
            initial_score,
            initial_derivative,
            1.0,
            &config,
        );

        // Should find a step that improves the score
        assert!(result.step_length > 0.0);
        assert!(result.final_value > initial_score);
    }
}
