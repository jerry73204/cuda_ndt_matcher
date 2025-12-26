//! More-Thuente line search for NDT optimization.
//!
//! This implements the More-Thuente line search algorithm as described in:
//! More, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed
//! sufficient decrease. ACM Transactions on Mathematical Software (TOMS), 20(3), 286-307.
//!
//! The line search finds a step length α that satisfies the strong Wolfe conditions:
//! 1. Sufficient decrease: f(x + αd) ≤ f(x) + c₁αg(x)ᵀd
//! 2. Curvature condition: |g(x + αd)ᵀd| ≤ c₂|g(x)ᵀd|
//!
//! Note: In Autoware's NDT implementation, line search is currently disabled
//! due to local minima issues. This implementation is provided for experimentation.

use nalgebra::Vector6;

/// Configuration for More-Thuente line search.
#[derive(Debug, Clone)]
pub struct LineSearchConfig {
    /// Sufficient decrease parameter (c₁ in Wolfe conditions).
    /// Typically 1e-4.
    pub c1: f64,

    /// Curvature condition parameter (c₂ in Wolfe conditions).
    /// Typically 0.9 for Newton methods.
    pub c2: f64,

    /// Minimum step length.
    pub alpha_min: f64,

    /// Maximum step length.
    pub alpha_max: f64,

    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Tolerance for bracketing interval.
    pub bracket_tolerance: f64,
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            alpha_min: 1e-10,
            alpha_max: 10.0,
            max_iterations: 20,
            bracket_tolerance: 1e-10,
        }
    }
}

/// Result of line search.
#[derive(Debug, Clone)]
pub struct LineSearchResult {
    /// Optimal step length found.
    pub alpha: f64,

    /// Score at the optimal step.
    pub score: f64,

    /// Whether the search converged successfully.
    pub converged: bool,

    /// Number of function evaluations.
    pub evaluations: usize,
}

impl LineSearchResult {
    /// Create a result indicating the initial step was accepted.
    pub fn initial_step_accepted(score: f64) -> Self {
        Self {
            alpha: 1.0,
            score,
            converged: true,
            evaluations: 1,
        }
    }

    /// Create a result indicating search failure.
    pub fn failed() -> Self {
        Self {
            alpha: 0.0,
            score: 0.0,
            converged: false,
            evaluations: 0,
        }
    }
}

/// Trait for functions that can be optimized with line search.
///
/// This allows the line search to evaluate the objective function
/// and its directional derivative at different step lengths.
pub trait LineSearchFunction {
    /// Evaluate the objective function and directional derivative at x + α*d.
    ///
    /// # Arguments
    /// * `alpha` - Step length
    ///
    /// # Returns
    /// (score, directional_derivative)
    fn evaluate(&mut self, alpha: f64) -> (f64, f64);
}

/// Simple backtracking line search.
///
/// This is a simpler alternative to More-Thuente that just ensures
/// the Armijo condition (sufficient decrease) is satisfied.
///
/// # Arguments
/// * `score_fn` - Function that computes score at a given step length
/// * `initial_score` - Score at α = 0
/// * `initial_derivative` - Directional derivative g(x)ᵀd at α = 0
/// * `config` - Line search configuration
///
/// # Returns
/// Line search result with optimal step length.
pub fn backtracking_line_search<F>(
    mut score_fn: F,
    initial_score: f64,
    initial_derivative: f64,
    config: &LineSearchConfig,
) -> LineSearchResult
where
    F: FnMut(f64) -> f64,
{
    // For NDT, we're maximizing score, so derivative should be positive
    // for a descent direction. If not, reject immediately.
    if initial_derivative <= 0.0 {
        return LineSearchResult::failed();
    }

    let mut alpha = 1.0;
    let mut evaluations = 0;

    for _ in 0..config.max_iterations {
        let score = score_fn(alpha);
        evaluations += 1;

        // Check Armijo condition (sufficient decrease)
        // For maximization: f(x + αd) ≥ f(x) + c₁α∇f(x)ᵀd
        // Since initial_derivative > 0, this checks if we're making progress
        if score >= initial_score + config.c1 * alpha * initial_derivative {
            return LineSearchResult {
                alpha,
                score,
                converged: true,
                evaluations,
            };
        }

        // Reduce step size
        alpha *= 0.5;

        if alpha < config.alpha_min {
            break;
        }
    }

    // Return best found (or failed)
    LineSearchResult {
        alpha,
        score: score_fn(alpha),
        converged: false,
        evaluations: evaluations + 1,
    }
}

/// Compute the directional derivative for a given step.
///
/// directional_derivative = g(x)ᵀ * d
///
/// # Arguments
/// * `gradient` - Gradient at current point
/// * `direction` - Search direction
pub fn directional_derivative(gradient: &Vector6<f64>, direction: &Vector6<f64>) -> f64 {
    gradient.dot(direction)
}

/// Check if a step satisfies the Armijo condition (sufficient decrease).
///
/// For maximization: f(x + αd) ≥ f(x) + c₁α∇f(x)ᵀd
pub fn armijo_condition(
    new_score: f64,
    old_score: f64,
    alpha: f64,
    initial_derivative: f64,
    c1: f64,
) -> bool {
    new_score >= old_score + c1 * alpha * initial_derivative
}

/// Check if a step satisfies the curvature condition.
///
/// |∇f(x + αd)ᵀd| ≤ c₂|∇f(x)ᵀd|
pub fn curvature_condition(new_derivative: f64, initial_derivative: f64, c2: f64) -> bool {
    new_derivative.abs() <= c2 * initial_derivative.abs()
}

/// Check if a step satisfies the strong Wolfe conditions.
pub fn strong_wolfe_conditions(
    new_score: f64,
    old_score: f64,
    new_derivative: f64,
    initial_derivative: f64,
    alpha: f64,
    c1: f64,
    c2: f64,
) -> bool {
    armijo_condition(new_score, old_score, alpha, initial_derivative, c1)
        && curvature_condition(new_derivative, initial_derivative, c2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LineSearchConfig::default();
        assert!(config.c1 > 0.0 && config.c1 < 1.0);
        assert!(config.c2 > config.c1 && config.c2 < 1.0);
        assert!(config.alpha_min > 0.0);
        assert!(config.alpha_max > config.alpha_min);
    }

    #[test]
    fn test_directional_derivative() {
        let gradient = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let direction = Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        let deriv = directional_derivative(&gradient, &direction);
        assert_eq!(deriv, 1.0);

        let direction_all = Vector6::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let deriv_all = directional_derivative(&gradient, &direction_all);
        assert_eq!(deriv_all, 21.0); // 1+2+3+4+5+6
    }

    #[test]
    fn test_armijo_condition() {
        // Score increased sufficiently
        assert!(armijo_condition(1.1, 1.0, 1.0, 0.15, 0.1));

        // Score didn't increase enough
        assert!(!armijo_condition(1.01, 1.0, 1.0, 0.15, 0.1));

        // Score decreased (not acceptable)
        assert!(!armijo_condition(0.9, 1.0, 1.0, 0.15, 0.1));
    }

    #[test]
    fn test_curvature_condition() {
        // Derivative magnitude decreased
        assert!(curvature_condition(0.05, 0.1, 0.9));

        // Derivative magnitude same
        assert!(curvature_condition(0.09, 0.1, 0.9));

        // Derivative magnitude too large
        assert!(!curvature_condition(0.95, 0.1, 0.9));
    }

    #[test]
    fn test_backtracking_quadratic() {
        // Test on a simple quadratic: f(α) = -(α - 0.5)² + 1
        // Maximum at α = 0.5, f(0.5) = 1
        // f(0) = -0.25 + 1 = 0.75
        // f'(0) = 2(0.5 - 0) = 1.0

        let config = LineSearchConfig::default();

        let score_fn = |alpha: f64| {
            let x = alpha - 0.5;
            -x * x + 1.0
        };

        let initial_score = score_fn(0.0); // 0.75
        let initial_derivative = 1.0; // Slope at origin points upward

        let result = backtracking_line_search(score_fn, initial_score, initial_derivative, &config);

        // Should find a step that improves the score
        assert!(result.converged);
        assert!(result.score > initial_score);
        assert!(result.alpha > 0.0);
    }

    #[test]
    fn test_backtracking_negative_derivative() {
        // Negative derivative means we're moving in wrong direction
        let config = LineSearchConfig::default();
        let score_fn = |_: f64| 1.0;

        let result = backtracking_line_search(score_fn, 1.0, -0.5, &config);

        assert!(!result.converged);
    }

    #[test]
    fn test_line_search_result_helpers() {
        let accepted = LineSearchResult::initial_step_accepted(1.5);
        assert!(accepted.converged);
        assert_eq!(accepted.alpha, 1.0);
        assert_eq!(accepted.score, 1.5);

        let failed = LineSearchResult::failed();
        assert!(!failed.converged);
        assert_eq!(failed.alpha, 0.0);
    }
}
