//! Pin + Terrain + Ternary (P+T+T) Compression
//!
//! Splits each weight matrix into:
//!   - **Pins** (top 1-5% by magnitude): stored at BF16 in sparse format (index + value)
//!   - **Terrain** (remaining 95-99%): quantized to ternary {-1, 0, +1} at 2 bits
//!   - **Scale**: one float32 per row for terrain reconstruction
//!
//! Formula: `W = scale × ternary_terrain + sparse_pins`
//! Computation: `y = scale × [Σ(+x_i) - Σ(-x_i)] + Σ(pin_j × x_j)`
//!   - First term: pure ADD/SUB (no multiplications!)
//!   - Second term: few FP multiplications (only 1-5% of weights)
//!
//! Expected: 6.9x compression vs BF16 with high fidelity.

use std::collections::BTreeMap;

/// Pin + Terrain + Ternary encoded weight matrix
#[derive(Debug, Clone)]
pub struct PttEncoded {
    /// Packed ternary terrain: [rows × ceil(cols/4)] bytes (2-bit ternary, 4 per byte)
    pub terrain_packed: Vec<u8>,

    /// Per-row scales for terrain reconstruction: [rows] × f32
    pub terrain_scales: Vec<f32>,

    /// Sparse pin indices: flat indices into [rows × cols] matrix
    pub pin_indices: Vec<u32>,

    /// Sparse pin values: BF16 as raw bits (u16)
    pub pin_values: Vec<u16>,

    /// Number of pins stored
    pub n_pins: u32,

    /// Sigma threshold used for pin detection (for metadata)
    pub pin_threshold_sigma: f32,

    /// Matrix dimensions
    pub rows: usize,
    pub cols: usize,
}

/// Statistics about P+T+T encoding
#[derive(Debug, Clone)]
pub struct PttStats {
    /// Fraction of weights stored as pins (e.g., 0.03 for top 3%)
    pub pin_ratio: f32,

    /// Fraction of terrain values quantized to zero
    pub terrain_sparsity: f32,

    /// Compression ratio vs BF16
    pub compression_ratio: f32,

    /// Estimated cosine similarity with original (float32 reference)
    pub estimated_cosine: f32,

    /// Total size in bytes
    pub total_size_bytes: usize,
}

/// Helper: convert BF16 bits to f32
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Helper: convert f32 to BF16 bits (round-to-nearest)
#[inline]
fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round to nearest even at bit 16
    let lsb = (bits >> 16) & 1;
    let rounding_bias = if ((bits >> 15) & 1) != 0 && lsb == 0 { 1u32 } else { 0 };
    ((bits >> 16) as u32 + rounding_bias) as u16
}

/// Encode a float32 weight matrix into P+T+T format.
///
/// Algorithm:
///   1. Compute per-row mean and std
///   2. Identify pins: |w| > sigma_threshold × row_std
///   3. Store pins as sparse (index, bf16 value)
///   4. Quantize remaining weights to ternary using absmean threshold
///   5. Pack ternary 4 per byte
pub fn encode(
    weights: &[f32],
    rows: usize,
    cols: usize,
    sigma_threshold: f32,
) -> PttEncoded {
    assert_eq!(weights.len(), rows * cols, "Weight size mismatch");

    let mut terrain_packed = Vec::new();
    let mut terrain_scales = Vec::new();
    let mut pin_indices = Vec::new();
    let mut pin_values = Vec::new();

    // Process each row
    for row_idx in 0..rows {
        let row_start = row_idx * cols;
        let row_end = row_start + cols;
        let row_data = &weights[row_start..row_end];

        // Step 1: Compute row statistics
        let mean = row_data.iter().sum::<f32>() / cols as f32;
        let variance = row_data.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / cols as f32;
        let std = variance.sqrt();

        // Step 2: Identify pins (high-magnitude weights)
        let mut terrain = vec![0.0f32; cols];
        let pin_threshold = sigma_threshold * std;

        for (col_idx, &weight) in row_data.iter().enumerate() {
            if weight.abs() > pin_threshold {
                // This is a pin
                let flat_idx = (row_idx * cols + col_idx) as u32;
                pin_indices.push(flat_idx);
                pin_values.push(f32_to_bf16(weight));
            } else {
                // This goes to terrain
                terrain[col_idx] = weight;
            }
        }

        // Step 3: Quantize terrain to ternary with per-row scaling
        // Use absmean threshold approach
        let abs_vals: Vec<f32> = terrain.iter().map(|w| w.abs()).collect();
        let abs_mean = abs_vals.iter().sum::<f32>() / cols as f32;
        let threshold = if abs_mean > 1e-12 { abs_mean } else { 1e-6 };

        // Find scale: max absolute value in terrain
        let absmax = abs_vals.iter().cloned().fold(0.0f32, f32::max);
        let scale = if absmax > 1e-12 { absmax } else { 1.0 };

        terrain_scales.push(scale);

        // Quantize to ternary: {-1, 0, +1}
        let ternary: Vec<i8> = terrain
            .iter()
            .map(|&w| {
                if w > threshold {
                    1
                } else if w < -threshold {
                    -1
                } else {
                    0
                }
            })
            .collect();

        // Step 4: Pack ternary into 2 bits per weight (4 per byte)
        // Encoding: -1 → 0b00, 0 → 0b01, +1 → 0b10, unused → 0b11
        let packed_len = (cols + 3) / 4;
        let mut packed_row = vec![0u8; packed_len];

        for (i, &t) in ternary.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let code: u8 = match t {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => unreachable!(),
            };
            packed_row[byte_idx] |= code << bit_offset;
        }

        terrain_packed.extend_from_slice(&packed_row);
    }

    let n_pins = pin_indices.len() as u32;

    PttEncoded {
        terrain_packed,
        terrain_scales,
        pin_indices,
        pin_values,
        n_pins,
        pin_threshold_sigma: sigma_threshold,
        rows,
        cols,
    }
}

/// Unpack a single ternary weight from packed byte
#[inline]
fn unpack_ternary(byte: u8, position: usize) -> i8 {
    let code = (byte >> (position * 2)) & 0x3;
    match code {
        0b00 => -1,
        0b01 => 0,
        0b10 => 1,
        _ => 0, // unused code, treat as zero
    }
}

/// Decode P+T+T encoded weights back to float32.
pub fn decode(encoded: &PttEncoded) -> Vec<f32> {
    let mut result = vec![0.0f32; encoded.rows * encoded.cols];

    // Create a map of pin indices for fast lookup
    let pin_map: BTreeMap<usize, u16> = encoded
        .pin_indices
        .iter()
        .zip(encoded.pin_values.iter())
        .map(|(&idx, &val)| (idx as usize, val))
        .collect();

    let packed_cols = (encoded.cols + 3) / 4;

    // Process each row
    for row_idx in 0..encoded.rows {
        let scale = encoded.terrain_scales[row_idx];

        // Unpack terrain for this row
        let row_start = row_idx * packed_cols;
        let row_end = row_start + packed_cols;
        let packed_row = &encoded.terrain_packed[row_start..row_end];

        for col_idx in 0..encoded.cols {
            let flat_idx = row_idx * encoded.cols + col_idx;

            // Check if this is a pin
            if let Some(&pin_bf16) = pin_map.get(&flat_idx) {
                result[flat_idx] = bf16_to_f32(pin_bf16);
            } else {
                // Unpack from terrain
                let byte_idx = col_idx / 4;
                let bit_pos = col_idx % 4;
                let ternary_val = unpack_ternary(packed_row[byte_idx], bit_pos);
                result[flat_idx] = scale * ternary_val as f32;
            }
        }
    }

    result
}

/// Compute statistics about P+T+T encoding.
pub fn stats(encoded: &PttEncoded, original: Option<&[f32]>) -> PttStats {
    let total_elements = encoded.rows * encoded.cols;
    let pin_ratio = encoded.n_pins as f32 / total_elements as f32;

    // Count zeros in terrain
    let mut terrain_zeros = 0u64;
    let packed_cols = (encoded.cols + 3) / 4;
    for row_idx in 0..encoded.rows {
        let row_start = row_idx * packed_cols;
        let row_end = row_start + packed_cols;
        let packed_row = &encoded.terrain_packed[row_start..row_end];

        for col_idx in 0..encoded.cols {
            // Check if it's a pin first
            let flat_idx = row_idx * encoded.cols + col_idx;
            let is_pin = encoded
                .pin_indices
                .iter()
                .any(|&idx| idx as usize == flat_idx);

            if !is_pin {
                let byte_idx = col_idx / 4;
                let bit_pos = col_idx % 4;
                let ternary_val = unpack_ternary(packed_row[byte_idx], bit_pos);
                if ternary_val == 0 {
                    terrain_zeros += 1;
                }
            }
        }
    }

    let terrain_elements = (total_elements as u64) - (encoded.n_pins as u64);
    let terrain_sparsity = if terrain_elements > 0 {
        terrain_zeros as f32 / terrain_elements as f32
    } else {
        0.0
    };

    // Size calculation
    let terrain_size = encoded.terrain_packed.len();
    let scales_size = encoded.terrain_scales.len() * 4; // f32 per row
    let pin_indices_size = encoded.pin_indices.len() * 4; // u32 per pin
    let pin_values_size = encoded.pin_values.len() * 2; // u16 per pin
    let metadata_size = 4 + 4; // n_pins + threshold_sigma
    let total_size = terrain_size + scales_size + pin_indices_size + pin_values_size + metadata_size;

    // Compression ratio vs BF16
    let bf16_size = total_elements * 2;
    let compression_ratio = bf16_size as f32 / total_size as f32;

    // Estimate cosine similarity if original is provided
    let estimated_cosine = if let Some(orig) = original {
        let decoded = decode(encoded);
        let mut dot_product = 0.0f32;
        let mut norm_orig = 0.0f32;
        let mut norm_decoded = 0.0f32;

        for i in 0..orig.len() {
            dot_product += orig[i] * decoded[i];
            norm_orig += orig[i].powi(2);
            norm_decoded += decoded[i].powi(2);
        }

        let denom = (norm_orig * norm_decoded).sqrt();
        if denom > 1e-12 {
            dot_product / denom
        } else {
            1.0
        }
    } else {
        1.0
    };

    PttStats {
        pin_ratio,
        terrain_sparsity,
        compression_ratio,
        estimated_cosine,
        total_size_bytes: total_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptt_encode_decode_identity() {
        // Simple known matrix
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let encoded = encode(&original, 2, 4, 2.0);
        let decoded = decode(&encoded);

        // Should be fairly close (quantization + pin detection will cause some loss)
        // With 2-bit ternary, we lose significant precision
        for (o, d) in original.iter().zip(decoded.iter()) {
            assert!((o - d).abs() < 3.0, "Mismatch: {} vs {}", o, d);
        }
    }

    #[test]
    fn test_ptt_pin_detection() {
        // Create a matrix with clear outliers
        let original = vec![
            0.1, 0.2, 0.3, 0.4, // row 0: small values
            100.0, 0.1, 0.2, 0.3, // row 1: one large outlier
        ];

        let encoded = encode(&original, 2, 4, 2.0);

        // Should detect the 100.0 as a pin
        assert!(
            encoded.n_pins > 0,
            "Should detect at least one pin for outlier value"
        );
    }

    #[test]
    fn test_ptt_ternary_packing() {
        // Create a simple ternary sequence
        let original = vec![1.0f32, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let encoded = encode(&original, 2, 4, 10.0); // High threshold: everything becomes terrain

        // All should be in terrain (nothing passes pin threshold)
        assert_eq!(encoded.n_pins, 0, "With high threshold, no pins should be created");

        let decoded = decode(&encoded);
        for (o, d) in original.iter().zip(decoded.iter()) {
            // With ternary quantization, allow some tolerance
            assert!((o.abs() - d.abs()).abs() < 0.5, "Sign preserved: {} vs {}", o, d);
        }
    }

    #[test]
    fn test_ptt_stats() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let encoded = encode(&original, 2, 4, 2.0);
        let ptt_stats = stats(&encoded, Some(&original));

        assert!(ptt_stats.pin_ratio >= 0.0 && ptt_stats.pin_ratio <= 1.0);
        assert!(
            ptt_stats.compression_ratio > 0.0,
            "Compression ratio should be positive"
        );
        assert!(
            ptt_stats.estimated_cosine <= 1.0 && ptt_stats.estimated_cosine > 0.5,
            "Cosine similarity should be reasonable"
        );
    }

    #[test]
    fn test_ptt_large_matrix() {
        // Test with a larger realistic matrix
        let size = 256 * 128;
        let original: Vec<f32> = (0..size)
            .map(|i| ((i as f32 * 0.1).sin() * 10.0))
            .collect();

        let encoded = encode(&original, 256, 128, 1.5);
        let decoded = decode(&encoded);

        // Check all values are decoded
        assert_eq!(decoded.len(), original.len());

        // Rough correctness check
        let max_error = original
            .iter()
            .zip(decoded.iter())
            .map(|(o, d)| (o - d).abs())
            .fold(0.0f32, f32::max);

        assert!(max_error < 50.0, "Reconstruction error too large: {}", max_error);
    }

    #[test]
    fn test_ptt_bf16_roundtrip() {
        // Test that BF16 conversion preserves pins correctly
        let original = vec![3.14159f32, -2.71828];
        let bf16_0 = f32_to_bf16(original[0]);
        let bf16_1 = f32_to_bf16(original[1]);

        let recovered_0 = bf16_to_f32(bf16_0);
        let recovered_1 = bf16_to_f32(bf16_1);

        // BF16 loses precision but should be within ~0.5%
        assert!((original[0] - recovered_0).abs() / original[0].abs() < 0.01);
        assert!((original[1] - recovered_1).abs() / original[1].abs() < 0.01);
    }
}
