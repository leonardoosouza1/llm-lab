//! BitNet 1.58-bit ternary quantization.
//!
//! Weights are quantized to {-1, 0, +1} using absmean thresholding.
//! Packed as 2 bits per weight (4 weights per byte).
//! GEMV becomes pure addition/subtraction — zero multiplications.
//!
//! Encoding: -1 = 0b00, 0 = 0b01, +1 = 0b10, unused = 0b11
//! Pack order: [w0:bits0-1, w1:bits2-3, w2:bits4-5, w3:bits6-7]

/// Ternary quantized weight matrix
#[derive(Debug, Clone)]
pub struct TernaryWeight {
    pub packed: Vec<u8>,       // 2 bits per weight, 4 per byte
    pub scale: f32,            // Per-tensor scale (absmax of original)
    pub shape: [usize; 2],    // [out_features, in_features]
    pub sparsity: f32,         // Fraction of zeros (for stats)
}

/// BF16 → f32 helper
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Quantize BF16 weights to ternary {-1, 0, +1}
/// Algorithm: absmean thresholding (BitNet b1.58)
///   1. Compute scale = max(|w|)
///   2. Normalize: w_norm = w / scale
///   3. threshold = mean(|w_norm|)
///   4. Quantize: w > threshold → +1, w < -threshold → -1, else → 0
pub fn quantize_to_ternary(bf16_data: &[u16], shape: [usize; 2]) -> TernaryWeight {
    let [rows, cols] = shape;
    let n = bf16_data.len();
    assert_eq!(n, rows * cols);

    // Convert to f32
    let f32_data: Vec<f32> = bf16_data.iter().map(|&b| bf16_to_f32(b)).collect();

    // Compute absmax (scale)
    let absmax = f32_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if absmax > 1e-12 { absmax } else { 1.0 };

    // Normalize and compute threshold
    let normalized: Vec<f32> = f32_data.iter().map(|&v| v / scale).collect();
    let abs_mean: f32 = normalized.iter().map(|v| v.abs()).sum::<f32>() / n as f32;

    // Quantize to ternary
    let mut n_zeros = 0u64;
    let ternary: Vec<i8> = normalized.iter().map(|&v| {
        if v > abs_mean { 1i8 }
        else if v < -abs_mean { -1i8 }
        else { n_zeros += 1; 0i8 }
    }).collect();

    // Pack: 4 weights per byte
    // Encoding: -1 → 0b00, 0 → 0b01, +1 → 0b10
    let packed_len = (n + 3) / 4;
    let mut packed = vec![0u8; packed_len];

    for i in 0..n {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let code: u8 = match ternary[i] {
            -1 => 0b00,
             0 => 0b01,
             1 => 0b10,
             _ => unreachable!(),
        };
        packed[byte_idx] |= code << bit_offset;
    }

    let sparsity = n_zeros as f32 / n as f32;

    TernaryWeight { packed, scale, shape, sparsity }
}

/// Unpack a single weight from packed byte
#[inline]
fn unpack_weight(byte: u8, position: usize) -> i8 {
    let code = (byte >> (position * 2)) & 0x3;
    match code {
        0b00 => -1,
        0b01 =>  0,
        0b10 =>  1,
        _    =>  0,  // unused code, treat as zero
    }
}

/// Ternary GEMV: y = W_ternary @ x * scale
/// Pure addition and subtraction — zero multiplications!
///
/// For each output row: y[i] = scale * (sum(x[j] where w==+1) - sum(x[j] where w==-1))
pub fn ternary_gemv(weight: &TernaryWeight, x: &[f32]) -> Vec<f32> {
    let [rows, cols] = weight.shape;
    assert_eq!(x.len(), cols, "Input dimension mismatch");

    let packed_cols = (cols + 3) / 4;
    let mut output = vec![0.0f32; rows];

    for row in 0..rows {
        let mut sum = 0.0f32;
        let row_start = row * packed_cols;

        for pk in 0..packed_cols {
            let byte = weight.packed[row_start + pk];

            // Unroll 4 weights per byte
            for sub in 0..4 {
                let col = pk * 4 + sub;
                if col >= cols { break; }

                let w = unpack_weight(byte, sub);
                match w {
                    1  => sum += x[col],   // Addition only
                    -1 => sum -= x[col],   // Subtraction only
                    _  => {}               // Zero: skip entirely
                }
            }
        }

        output[row] = sum * weight.scale;
    }

    output
}

/// Ternary GEMV with output accumulation: y += alpha * W_ternary @ x * scale
pub fn ternary_gemv_accumulate(weight: &TernaryWeight, x: &[f32], y: &mut [f32], alpha: f32) {
    let [rows, cols] = weight.shape;
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);

    let packed_cols = (cols + 3) / 4;

    for row in 0..rows {
        let mut sum = 0.0f32;
        let row_start = row * packed_cols;

        for pk in 0..packed_cols {
            let byte = weight.packed[row_start + pk];
            for sub in 0..4 {
                let col = pk * 4 + sub;
                if col >= cols { break; }
                match unpack_weight(byte, sub) {
                    1  => sum += x[col],
                    -1 => sum -= x[col],
                    _  => {}
                }
            }
        }

        y[row] += alpha * sum * weight.scale;
    }
}

/// Statistics about ternary quantization
#[derive(Debug, Clone)]
pub struct TernaryStats {
    pub shape: [usize; 2],
    pub scale: f32,
    pub sparsity: f32,           // Fraction of zeros
    pub size_bytes: usize,       // Packed size
    pub compression_ratio: f32,  // vs BF16
    pub n_positive: usize,
    pub n_negative: usize,
    pub n_zero: usize,
}

pub fn ternary_stats(weight: &TernaryWeight) -> TernaryStats {
    let [rows, cols] = weight.shape;
    let total = rows * cols;
    let packed_cols = (cols + 3) / 4;

    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    let mut n_zero = 0usize;

    for row in 0..rows {
        for pk in 0..packed_cols {
            let byte = weight.packed[row * packed_cols + pk];
            for sub in 0..4 {
                if row * cols + pk * 4 + sub >= total { break; }
                match unpack_weight(byte, sub) {
                    1  => n_pos += 1,
                    -1 => n_neg += 1,
                    _  => n_zero += 1,
                }
            }
        }
    }

    let bf16_size = total * 2;
    let ternary_size = weight.packed.len() + 4; // packed + scale

    TernaryStats {
        shape: weight.shape,
        scale: weight.scale,
        sparsity: n_zero as f32 / total as f32,
        size_bytes: ternary_size,
        compression_ratio: bf16_size as f32 / ternary_size as f32,
        n_positive: n_pos,
        n_negative: n_neg,
        n_zero,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        // Test that pack/unpack is consistent
        let vals: Vec<i8> = vec![-1, 0, 1, 0, 1, -1, 0, 1];
        let mut packed = vec![0u8; 2];  // 8 values, 4 per byte

        for i in 0..8 {
            let byte_idx = i / 4;
            let pos = i % 4;
            let code: u8 = match vals[i] {
                -1 => 0b00,
                 0 => 0b01,
                 1 => 0b10,
                 _ => unreachable!(),
            };
            packed[byte_idx] |= code << (pos * 2);
        }

        for i in 0..8 {
            let byte_idx = i / 4;
            let pos = i % 4;
            assert_eq!(unpack_weight(packed[byte_idx], pos), vals[i],
                "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ternary_gemv_simple() {
        // Manual test: W = [[1, -1, 0, 1]], x = [1, 2, 3, 4]
        // Expected: y = scale * (1 - 2 + 0 + 4) = scale * 3
        let bf16_data: Vec<u16> = [1.0f32, -1.0, 0.0, 1.0].iter()
            .map(|&v| (v.to_bits() >> 16) as u16)
            .collect();

        let weight = quantize_to_ternary(&bf16_data, [1, 4]);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = ternary_gemv(&weight, &x);

        assert_eq!(y.len(), 1);
        // Result should be approximately 3.0 * scale
        let expected = 3.0 * weight.scale;
        assert!((y[0] - expected).abs() < 0.5,
            "Expected ~{}, got {}", expected, y[0]);
    }

    #[test]
    fn test_ternary_quantize_sparsity() {
        // Random-ish BF16 data — should have ~33% zeros with absmean threshold
        let f32_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin() * 10.0).collect();
        let bf16_data: Vec<u16> = f32_data.iter()
            .map(|&v| (v.to_bits() >> 16) as u16)
            .collect();

        let weight = quantize_to_ternary(&bf16_data, [16, 16]);
        let stats = ternary_stats(&weight);

        // With absmean threshold, expect ~30-40% sparsity
        assert!(stats.sparsity > 0.1, "Sparsity too low: {}", stats.sparsity);
        assert!(stats.sparsity < 0.8, "Sparsity too high: {}", stats.sparsity);

        // Compression should be ~8x vs BF16 (2 bits vs 16 bits)
        assert!(stats.compression_ratio > 5.0,
            "Compression ratio too low: {}", stats.compression_ratio);
    }

    #[test]
    fn test_ternary_gemv_larger() {
        // Test with a bigger matrix
        let n = 128;
        let m = 64;
        let f32_data: Vec<f32> = (0..n*m).map(|i| (i as f32 * 0.1).sin()).collect();
        let bf16_data: Vec<u16> = f32_data.iter()
            .map(|&v| (v.to_bits() >> 16) as u16)
            .collect();

        let weight = quantize_to_ternary(&bf16_data, [m, n]);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
        let y = ternary_gemv(&weight, &x);

        assert_eq!(y.len(), m);
        // Just verify it doesn't crash and produces finite values
        for &val in &y {
            assert!(val.is_finite(), "Non-finite output: {}", val);
        }
    }

    #[test]
    fn test_ternary_accumulate() {
        let bf16_data: Vec<u16> = [1.0f32, -1.0, 1.0, 0.0].iter()
            .map(|&v| (v.to_bits() >> 16) as u16)
            .collect();
        let weight = quantize_to_ternary(&bf16_data, [1, 4]);
        let x = vec![1.0, 1.0, 1.0, 1.0];

        let mut y = vec![10.0]; // Start with non-zero
        ternary_gemv_accumulate(&weight, &x, &mut y, 1.0);

        // Should be 10.0 + scale * (1 - 1 + 1 + 0) = 10.0 + scale * 1
        assert!(y[0] > 10.0, "Accumulation failed: {}", y[0]);
    }
}
