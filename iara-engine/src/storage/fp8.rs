//! FP8 E4M3 quantization for weight compression.
//! Stores weights in 1 byte (vs 2 bytes BF16) with per-row scaling.
//! Dequantization to F32 happens on-the-fly before GEMM.

/// Convert f32 to FP8 E4M3 (u8)
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() { return 0x7F; }
    let clamped = val.clamp(-448.0, 448.0);
    if clamped == 0.0 { return if clamped.is_sign_negative() { 0x80 } else { 0x00 }; }

    let bits = clamped.to_bits();
    let sign = (bits >> 31) & 1;
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x7FFFFF;

    // FP8 E4M3: bias = 7, exp range [0, 15], mant = 3 bits.
    // Round-to-nearest on the kept mantissa bits (add the round bit at position
    // 19 = just below the 3 retained bits, then carry into the exponent on
    // overflow). Plain truncation biases every weight toward zero.
    let rounded = f32_mant + (1 << 19);
    let (f32_mant, f32_exp) = if rounded & 0x0080_0000 != 0 {
        (0u32, f32_exp + 1) // mantissa carried out of 23 bits → bump exponent
    } else {
        (rounded, f32_exp)
    };
    let fp8_exp = (f32_exp + 7).clamp(0, 15) as u8;
    let fp8_mant = ((f32_mant >> 20) & 0x7) as u8; // Top 3 mantissa bits

    // Handle subnormals and overflow
    if fp8_exp == 0 { return (sign as u8) << 7; }
    if fp8_exp >= 15 { return ((sign as u8) << 7) | 0x7E; } // Max normal, not NaN

    ((sign as u8) << 7) | (fp8_exp << 3) | fp8_mant
}

/// Convert FP8 E4M3 (u8) to f32
pub fn fp8_e4m3_to_f32(val: u8) -> f32 {
    if val == 0x7F || val == 0xFF { return f32::NAN; }

    let sign = (val >> 7) & 1;
    let exp = ((val >> 3) & 0xF) as i32;
    let mant = (val & 0x7) as u32;

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    if exp == 0 {
        // Subnormal: value = (-1)^sign * 2^(-6) * (0.mant)
        let frac = mant as f32 / 8.0;
        let val = frac * 2.0f32.powi(-6);
        return if sign == 1 { -val } else { val };
    }

    // Normal: reconstruct as f32
    let f32_exp = ((exp - 7 + 127) as u32) & 0xFF;
    let f32_mant = mant << 20;
    let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mant;
    f32::from_bits(f32_bits)
}

/// BF16 helper: convert u16 bf16 bits to f32
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// FP8 quantized weight tensor with per-row scales
#[derive(Debug, Clone)]
pub struct Fp8Quantized {
    pub data: Vec<u8>,        // FP8 E4M3 values [rows * cols]
    pub scales: Vec<f32>,     // Per-row absmax scales [rows]
    pub shape: [usize; 2],    // [out_features, in_features]
}

/// Quantize BF16 weights to FP8 E4M3 with per-row scaling
pub fn quantize_bf16_to_fp8(bf16_data: &[u16], shape: [usize; 2]) -> Fp8Quantized {
    let [rows, cols] = shape;
    assert_eq!(bf16_data.len(), rows * cols);

    let mut fp8_data = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows);

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;

        // Find absmax of row
        let mut absmax: f32 = 0.0;
        for i in start..end {
            let val = bf16_to_f32(bf16_data[i]);
            absmax = absmax.max(val.abs());
        }

        // Scale: map absmax to 448.0 (FP8 E4M3 max representable)
        let scale = if absmax > 1e-12 { absmax / 448.0 } else { 1.0 };
        scales.push(scale);

        // Quantize each value
        for i in start..end {
            let val = bf16_to_f32(bf16_data[i]) / scale;
            fp8_data.push(f32_to_fp8_e4m3(val));
        }
    }

    Fp8Quantized { data: fp8_data, scales, shape }
}

/// Dequantize FP8 row to F32 (CPU fallback)
pub fn dequantize_fp8_row(fp8_data: &[u8], scale: f32) -> Vec<f32> {
    fp8_data.iter().map(|&v| fp8_e4m3_to_f32(v) * scale).collect()
}

/// Dequantize entire FP8 tensor to F32 (CPU fallback)
pub fn dequantize_fp8_to_f32(quantized: &Fp8Quantized) -> Vec<f32> {
    let [rows, cols] = quantized.shape;
    let mut output = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let scale = quantized.scales[row];
        for i in start..end {
            output.push(fp8_e4m3_to_f32(quantized.data[i]) * scale);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_roundtrip_zero() {
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(0.0)), 0.0);
    }

    #[test]
    fn test_fp8_roundtrip_one() {
        let encoded = f32_to_fp8_e4m3(1.0);
        let decoded = fp8_e4m3_to_f32(encoded);
        assert!((decoded - 1.0).abs() < 0.1, "1.0 roundtrip: got {}", decoded);
    }

    #[test]
    fn test_fp8_roundtrip_negative() {
        let encoded = f32_to_fp8_e4m3(-2.5);
        let decoded = fp8_e4m3_to_f32(encoded);
        assert!((decoded - (-2.5)).abs() < 0.5, "-2.5 roundtrip: got {}", decoded);
    }

    #[test]
    fn test_fp8_clamp_overflow() {
        let encoded = f32_to_fp8_e4m3(1000.0);
        let decoded = fp8_e4m3_to_f32(encoded);
        assert!(decoded <= 448.0, "overflow should clamp: got {}", decoded);
    }

    #[test]
    fn test_fp8_nan() {
        let encoded = f32_to_fp8_e4m3(f32::NAN);
        assert_eq!(encoded, 0x7F);
        assert!(fp8_e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn test_quantize_bf16_roundtrip() {
        // Create some BF16 test data (small values)
        let f32_values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let bf16_values: Vec<u16> = f32_values.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();

        let quantized = quantize_bf16_to_fp8(&bf16_values, [16, 16]);
        let dequantized = dequantize_fp8_to_f32(&quantized);

        // Check NRMSE < 5%
        let mse: f32 = f32_values.iter().zip(&dequantized)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / f32_values.len() as f32;
        let rmse = mse.sqrt();
        let max_val = f32_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = f32_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let range = max_val - min_val;
        let nrmse = if range > 0.0 { rmse / range } else { 0.0 };

        assert!(nrmse < 0.05, "NRMSE too high: {:.4}", nrmse);
    }
}
