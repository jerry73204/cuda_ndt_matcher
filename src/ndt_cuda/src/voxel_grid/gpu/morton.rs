//! Morton code (Z-order curve) computation for GPU spatial indexing.
//!
//! Morton codes interleave the bits of 3D coordinates to create a 1D index
//! that preserves spatial locality. Points with similar Morton codes are
//! spatially close, making sorted Morton codes ideal for GPU processing.
//!
//! # Algorithm
//!
//! For a 3D point (x, y, z):
//! 1. Normalize to grid coordinates: `gx = (x - min_x) / resolution`
//! 2. Quantize to integer: `ix = floor(gx) & 0x1FFFFF` (21 bits per axis)
//! 3. Interleave bits: Morton code has z-bits at 0,3,6..., y at 1,4,7..., x at 2,5,8...
//!
//! This gives a 63-bit Morton code (21 bits × 3 axes) stored in u64.
//!
//! # GPU Implementation
//!
//! The GPU implementation uses two passes:
//! 1. Bounds reduction: Find min/max coordinates (parallel reduction)
//! 2. Morton encoding: Compute Morton code for each point (embarrassingly parallel)

use cubecl::prelude::*;

// ============================================================================
// GPU Kernels for Morton Code Computation
// ============================================================================

/// GPU kernel: Compute Morton codes for a batch of points.
///
/// Each thread processes one point, computing its Morton code from
/// pre-computed bounds.
///
/// # Inputs
/// - `points`: [N * 3] point coordinates (x, y, z) flattened
/// - `min_bound`: [3] minimum bounds of the point cloud
/// - `inv_resolution`: 1.0 / resolution
/// - `num_points`: number of points
///
/// # Outputs
/// - `codes_low`: [N] lower 32 bits of Morton codes
/// - `codes_high`: [N] upper 32 bits of Morton codes
/// - `indices`: [N] original point indices (identity mapping)
///
/// Note: We split u64 into two u32 arrays because CubeCL doesn't support u64 directly.
#[cube(launch_unchecked)]
pub fn compute_morton_codes_kernel<F: Float>(
    points: &Array<F>,
    min_bound: &Array<F>,
    inv_resolution: F,
    num_points: u32,
    codes_low: &mut Array<u32>,
    codes_high: &mut Array<u32>,
    indices: &mut Array<u32>,
) {
    let idx = ABSOLUTE_POS;

    if idx >= num_points {
        terminate!();
    }

    // Load point coordinates
    let base = idx * 3;
    let px = points[base];
    let py = points[base + 1];
    let pz = points[base + 2];

    // Load min bounds
    let min_x = min_bound[0];
    let min_y = min_bound[1];
    let min_z = min_bound[2];

    // Compute grid coordinates
    let gx = (px - min_x) * inv_resolution;
    let gy = (py - min_y) * inv_resolution;
    let gz = (pz - min_z) * inv_resolution;

    // Quantize to 21-bit integers (max 0x1FFFFF = 2097151)
    let max_coord = 0x1FFFFFu32;
    let ix_raw = u32::cast_from(F::floor(F::abs(gx)));
    let iy_raw = u32::cast_from(F::floor(F::abs(gy)));
    let iz_raw = u32::cast_from(F::floor(F::abs(gz)));

    // Clamp to max_coord using select
    let ix = select(ix_raw > max_coord, max_coord, ix_raw);
    let iy = select(iy_raw > max_coord, max_coord, iy_raw);
    let iz = select(iz_raw > max_coord, max_coord, iz_raw);

    // Morton encode using bit interleaving
    // We compute the 63-bit Morton code as two 32-bit parts
    let (low, high) = morton_encode_split(ix, iy, iz);

    codes_low[idx] = low;
    codes_high[idx] = high;
    indices[idx] = idx;
}

/// Expand bits for Morton encoding (GPU version).
/// Spreads 21 bits across 63 bits with 2-bit gaps.
#[cube]
fn expand_bits_21(x: u32) -> (u32, u32) {
    // We need to spread 21 bits across 63 bit positions
    // Lower 11 bits go to positions 0,3,6,...,30 (low word)
    // Upper 10 bits go to positions 33,36,...,60 (high word)

    let x_low = x & 0x7FF; // Lower 11 bits
    let x_high = (x >> 11) & 0x3FF; // Upper 10 bits

    // Expand lower 11 bits to positions 0,3,6,...,30 in low word
    let mut lo = x_low;
    lo = (lo | (lo << 16)) & 0x030000FF;
    lo = (lo | (lo << 8)) & 0x0300F00F;
    lo = (lo | (lo << 4)) & 0x030C30C3;
    lo = (lo | (lo << 2)) & 0x09249249;

    // Expand upper 10 bits to positions 0,3,6,...,27 in high word
    let mut hi = x_high;
    hi = (hi | (hi << 16)) & 0x030000FF;
    hi = (hi | (hi << 8)) & 0x0300F00F;
    hi = (hi | (hi << 4)) & 0x030C30C3;
    hi = (hi | (hi << 2)) & 0x09249249;

    (lo, hi)
}

/// Morton encode 3 coordinates into split 32-bit words.
#[cube]
fn morton_encode_split(x: u32, y: u32, z: u32) -> (u32, u32) {
    let (x_lo, x_hi) = expand_bits_21(x);
    let (y_lo, y_hi) = expand_bits_21(y);
    let (z_lo, z_hi) = expand_bits_21(z);

    // Interleave: x at bit positions 2,5,8,..., y at 1,4,7,..., z at 0,3,6,...
    let low = (x_lo << 2) | (y_lo << 1) | z_lo;
    let high = (x_hi << 2) | (y_hi << 1) | z_hi;

    (low, high)
}

// ============================================================================
// CPU Reference Implementations
// ============================================================================

/// Result of Morton code computation.
#[derive(Debug)]
pub struct MortonCodeResult {
    /// Morton codes for each point (raw bytes, interpret as u64).
    pub codes: Vec<u8>,
    /// Original point indices (raw bytes, interpret as u32).
    pub indices: Vec<u8>,
    /// Number of points.
    pub num_points: u32,
    /// Grid minimum bounds used for normalization.
    pub grid_min: [f32; 3],
    /// Grid maximum bounds.
    pub grid_max: [f32; 3],
}

/// Compute Morton codes for a point cloud (CPU reference implementation).
///
/// # Arguments
/// * `points` - Point cloud as flat array [x0, y0, z0, x1, y1, z1, ...]
/// * `resolution` - Voxel resolution for grid quantization
///
/// # Returns
/// Morton codes and original indices for each point.
pub fn compute_morton_codes_cpu(points: &[f32], resolution: f32) -> MortonCodeResult {
    let num_points = points.len() / 3;

    if num_points == 0 {
        return MortonCodeResult {
            codes: Vec::new(),
            indices: Vec::new(),
            num_points: 0,
            grid_min: [0.0; 3],
            grid_max: [0.0; 3],
        };
    }

    // Compute bounds
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut max_z = f32::MIN;

    for i in 0..num_points {
        let px = points[i * 3];
        let py = points[i * 3 + 1];
        let pz = points[i * 3 + 2];

        min_x = min_x.min(px);
        min_y = min_y.min(py);
        min_z = min_z.min(pz);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
        max_z = max_z.max(pz);
    }

    let grid_min = [min_x, min_y, min_z];
    let grid_max = [max_x, max_y, max_z];
    let inv_resolution = 1.0 / resolution;

    // Compute Morton codes
    let mut codes = Vec::with_capacity(num_points * 8); // u64 = 8 bytes
    let mut indices = Vec::with_capacity(num_points * 4); // u32 = 4 bytes

    for i in 0..num_points {
        let px = points[i * 3];
        let py = points[i * 3 + 1];
        let pz = points[i * 3 + 2];

        let gx = (px - min_x) * inv_resolution;
        let gy = (py - min_y) * inv_resolution;
        let gz = (pz - min_z) * inv_resolution;

        let ix = if gx >= 0.0 {
            Ord::min(gx as u32, 0x1FFFFF)
        } else {
            0
        };
        let iy = if gy >= 0.0 {
            Ord::min(gy as u32, 0x1FFFFF)
        } else {
            0
        };
        let iz = if gz >= 0.0 {
            Ord::min(gz as u32, 0x1FFFFF)
        } else {
            0
        };

        let code = morton_encode_cpu(ix, iy, iz);
        codes.extend_from_slice(&code.to_le_bytes());
        indices.extend_from_slice(&(i as u32).to_le_bytes());
    }

    MortonCodeResult {
        codes,
        indices,
        num_points: num_points as u32,
        grid_min,
        grid_max,
    }
}

/// CPU version of Morton encoding.
fn morton_encode_cpu(x: u32, y: u32, z: u32) -> u64 {
    fn expand_bits_cpu(mut x: u64) -> u64 {
        x &= 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFF_u64;
        x = (x | (x << 16)) & 0x1F0000FF0000FF_u64;
        x = (x | (x << 8)) & 0x100F00F00F00F00F_u64;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3_u64;
        x = (x | (x << 2)) & 0x1249249249249249_u64;
        x
    }

    let xx = expand_bits_cpu(x as u64);
    let yy = expand_bits_cpu(y as u64);
    let zz = expand_bits_cpu(z as u64);
    (xx << 2) | (yy << 1) | zz
}

/// Decode a Morton code back to 3D grid coordinates.
/// Useful for debugging and validation.
pub fn morton_decode_3d(code: u64) -> (u32, u32, u32) {
    fn compact_bits(mut x: u64) -> u32 {
        // Reverse of expand_bits: extract every 3rd bit
        x &= 0x1249249249249249;
        x = (x | (x >> 2)) & 0x10C30C30C30C30C3;
        x = (x | (x >> 4)) & 0x100F00F00F00F00F;
        x = (x | (x >> 8)) & 0x1F0000FF0000FF;
        x = (x | (x >> 16)) & 0x1F00000000FFFF;
        x = (x | (x >> 32)) & 0x1FFFFF;
        x as u32
    }

    let x = compact_bits(code >> 2);
    let y = compact_bits(code >> 1);
    let z = compact_bits(code);

    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_encode_decode_roundtrip() {
        // Test various coordinates
        let test_cases = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
            (100, 200, 300),
            (0x1FFFFF, 0x1FFFFF, 0x1FFFFF), // Max values
        ];

        for (x, y, z) in test_cases {
            let code = morton_encode_cpu(x, y, z);
            let (dx, dy, dz) = morton_decode_3d(code);
            assert_eq!(
                (x, y, z),
                (dx, dy, dz),
                "Roundtrip failed for ({x}, {y}, {z})"
            );
        }
    }

    #[test]
    fn test_morton_ordering() {
        // Points close in space should have similar Morton codes
        let c1 = morton_encode_cpu(0, 0, 0);
        let c2 = morton_encode_cpu(1, 0, 0);
        let c3 = morton_encode_cpu(0, 1, 0);
        let c4 = morton_encode_cpu(100, 100, 100);

        // c1, c2, c3 should all be close (differ by just a few bits)
        assert!(c2 - c1 < 10, "Adjacent cells should have similar codes");
        assert!(c3 - c1 < 10, "Adjacent cells should have similar codes");

        // c4 should be much larger
        assert!(
            c4 > c1 + 1000,
            "Distant cells should have very different codes"
        );
    }

    #[test]
    fn test_compute_morton_codes_cpu() {
        let points = vec![
            0.0, 0.0, 0.0, // Point 0
            1.0, 0.0, 0.0, // Point 1
            0.0, 1.0, 0.0, // Point 2
        ];

        let result = compute_morton_codes_cpu(&points, 0.5);
        assert_eq!(result.num_points, 3);
        assert_eq!(result.codes.len(), 3 * 8); // 3 u64s
        assert_eq!(result.indices.len(), 3 * 4); // 3 u32s
    }
}
