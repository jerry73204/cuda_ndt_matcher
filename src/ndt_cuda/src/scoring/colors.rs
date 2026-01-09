//! Score-to-color mapping for NDT visualization.
//!
//! This module provides functions to convert NDT scores to RGB colors
//! for per-point visualization, matching Autoware's color scheme.

/// RGBA color with values in [0, 1].
#[derive(Debug, Clone, Copy)]
pub struct ColorRGBA {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// Convert a normalized score [0, 1] to an RGBA color.
///
/// This implements Autoware's `exchange_color_crc` function, which produces
/// a smooth gradient from blue (low) → cyan → green → yellow → red (high).
///
/// The algorithm uses a 4-quadrant sine-based color ramp:
/// - [0.0, 0.25]: Blue → Cyan (G increases)
/// - [0.25, 0.5]: Cyan → Green (B decreases)
/// - [0.5, 0.75]: Green → Yellow (R increases)
/// - [0.75, 1.0]: Yellow → Red (G decreases)
pub fn score_to_color(normalized: f32) -> ColorRGBA {
    // Clamp to [0, 0.9999] like Autoware
    let value = normalized.clamp(0.0, 0.9999);

    let (r, g, b) = if value < 0.25 {
        // Blue → Cyan: increase G
        let t = value * 4.0; // [0, 1] over this range
        (0.0, (t * std::f32::consts::FRAC_PI_2).sin(), 1.0)
    } else if value < 0.5 {
        // Cyan → Green: decrease B
        let t = (value - 0.25) * 4.0;
        (0.0, 1.0, (t * std::f32::consts::FRAC_PI_2).cos())
    } else if value < 0.75 {
        // Green → Yellow: increase R
        let t = (value - 0.5) * 4.0;
        ((t * std::f32::consts::FRAC_PI_2).sin(), 1.0, 0.0)
    } else {
        // Yellow → Red: decrease G
        let t = (value - 0.75) * 4.0;
        (1.0, (t * std::f32::consts::FRAC_PI_2).cos(), 0.0)
    };

    ColorRGBA { r, g, b, a: 0.999 }
}

/// Convert an NDT score to a color using the specified score range.
///
/// # Arguments
/// * `score` - The raw NDT score (typically negative, e.g., -0.55 to 0)
/// * `lower` - Lower bound of the score range (default: 1.0 in Autoware)
/// * `upper` - Upper bound of the score range (default: 3.5 in Autoware)
///
/// # Notes
/// Autoware uses positive scores in range [1.0, 3.5] for visualization.
/// Our NDT scores may be negative (when d1 < 0), so we negate them.
pub fn ndt_score_to_color(score: f32, lower: f32, upper: f32) -> ColorRGBA {
    // NDT score = -d1 * exp(...), with d1 typically -0.55
    // So score is positive. Higher = better match.
    let range = upper - lower;
    let normalized = if range > 0.0 {
        (score - lower) / range
    } else {
        0.5
    };
    score_to_color(normalized)
}

/// Default score range for visualization (matches Autoware).
pub const DEFAULT_SCORE_LOWER: f32 = 1.0;
pub const DEFAULT_SCORE_UPPER: f32 = 3.5;

/// Convert a color to RGB bytes [0, 255].
pub fn color_to_rgb_bytes(color: &ColorRGBA) -> [u8; 3] {
    [
        (color.r * 255.0) as u8,
        (color.g * 255.0) as u8,
        (color.b * 255.0) as u8,
    ]
}

/// Pack RGB into a single u32 (for PointCloud2 RGB field).
/// Format: 0x00RRGGBB
pub fn color_to_rgb_packed(color: &ColorRGBA) -> u32 {
    let r = (color.r * 255.0) as u32;
    let g = (color.g * 255.0) as u32;
    let b = (color.b * 255.0) as u32;
    (r << 16) | (g << 8) | b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_to_color_range() {
        // Test endpoints
        let blue = score_to_color(0.0);
        assert!(blue.b > 0.9, "Low score should be blue");
        assert!(blue.r < 0.1, "Low score should have no red");

        let red = score_to_color(0.99);
        assert!(red.r > 0.9, "High score should be red");
        assert!(red.b < 0.1, "High score should have no blue");

        // Test midpoint (should be greenish)
        let mid = score_to_color(0.5);
        assert!(mid.g > 0.9, "Mid score should have high green");
    }

    #[test]
    fn test_ndt_score_to_color() {
        // Score at lower bound -> blue
        let low = ndt_score_to_color(1.0, 1.0, 3.5);
        assert!(low.b > 0.9);

        // Score at upper bound -> red
        let high = ndt_score_to_color(3.5, 1.0, 3.5);
        assert!(high.r > 0.9);

        // Score in middle -> greenish
        let mid = ndt_score_to_color(2.25, 1.0, 3.5);
        assert!(mid.g > 0.9);
    }

    #[test]
    fn test_color_packing() {
        let color = ColorRGBA {
            r: 1.0,
            g: 0.5,
            b: 0.0,
            a: 1.0,
        };
        let packed = color_to_rgb_packed(&color);
        // R=255, G=127, B=0 -> 0x00FF7F00
        assert_eq!(packed >> 16, 255); // R
        assert!((packed >> 8) & 0xFF >= 127); // G (approximately)
        assert_eq!(packed & 0xFF, 0); // B
    }
}
