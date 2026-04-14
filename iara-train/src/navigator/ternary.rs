//! Operações de quantização ternária e matmul sparse.
//!
//! Representa pesos como {-1, 0, +1} empacotados em 2 bits por elemento.
//! O mapa ternário serve como "navegador" — indica quais neurônios ativar,
//! enquanto os pesos fp16/fp32 originais fazem a computação precisa.

/// Quantiza uma matriz F32 para ternário {-1, 0, +1} com escala por-matriz.
/// Retorna (ternary_vals, scale) onde ternary_vals ∈ {-1.0, 0.0, 1.0}
pub fn quantize_ternary(weights: &[f32]) -> (Vec<f32>, f32) {
    let scale = weights.iter().map(|v| v.abs()).sum::<f32>() / weights.len() as f32;
    let scale = if scale < 1e-8 { 1e-8 } else { scale };
    let t: Vec<f32> = weights.iter()
        .map(|&v| (v / scale).round().clamp(-1.0, 1.0))
        .collect();
    (t, scale)
}

/// Empacota valores ternários {-1,0,+1} em 2 bits por elemento.
/// Codificação: 0b00=0, 0b01=+1, 0b10=-1 (INVALID: 0b11)
pub fn pack_ternary(vals: &[f32]) -> Vec<u8> {
    let n_bytes = (vals.len() * 2 + 7) / 8;
    let mut packed = vec![0u8; n_bytes];
    for (i, &v) in vals.iter().enumerate() {
        let code: u8 = match v as i8 {
            0  => 0b00,
            1  => 0b01,
            -1 => 0b10,
            _  => 0b00,
        };
        let bit_pos = i * 2;
        let byte_idx = bit_pos / 8;
        let bit_offset = (bit_pos % 8) as u8;
        packed[byte_idx] |= code << bit_offset;
    }
    packed
}

/// Desempacota 2 bits → f32 {-1.0, 0.0, 1.0}
pub fn unpack_ternary(packed: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let bit_pos = i * 2;
        let byte_idx = bit_pos / 8;
        let bit_offset = (bit_pos % 8) as u8;
        let code = (packed[byte_idx] >> bit_offset) & 0b11;
        out.push(match code {
            0b01 =>  1.0,
            0b10 => -1.0,
            _    =>  0.0,
        });
    }
    out
}

/// Métricas de compressão e qualidade.
pub struct TernaryMetrics {
    pub compression_ratio: f32,
    pub sparsity: f32,       // fração de zeros
    pub scale: f32,
    pub rmse: f32,
    pub cosine_sim: f32,
}

impl TernaryMetrics {
    /// Computa métricas comparando original com reconstrução ternária.
    pub fn compute(original: &[f32], ternary: &[f32], scale: f32) -> Self {
        let n = original.len();
        let zeros = ternary.iter().filter(|&&v| v == 0.0).count();

        // RMSE entre original e reconstrução ternária (t * scale)
        let rmse = (original.iter().zip(ternary.iter())
            .map(|(o, t)| (o - t * scale).powi(2))
            .sum::<f32>() / n as f32)
            .sqrt();

        // Cosine similarity
        let dot: f32 = original.iter().zip(ternary.iter())
            .map(|(o, t)| o * t * scale).sum();
        let no = original.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nt = ternary.iter().map(|v| (v * scale) * (v * scale)).sum::<f32>().sqrt();
        let cosine_sim = dot / (no * nt + 1e-8);

        // Compressão: original f32 (4B/elem) vs packed 2-bit (0.25B/elem) + scale (4B)
        let orig_bytes = n * 4;
        let packed_bytes = (n * 2 + 7) / 8 + 4;
        let compression_ratio = orig_bytes as f32 / packed_bytes as f32;

        TernaryMetrics { compression_ratio, sparsity: zeros as f32 / n as f32, scale, rmse, cosine_sim }
    }
}

/// Matmul ternário sparse: y = x @ W_ternary * scale
/// Skip explícito de neurônios com peso 0 (não contribuem para output).
/// Entrada: x [batch, in_dim], W [out_dim, in_dim] (row per neuron)
/// Saída: y [batch, out_dim]
pub fn ternary_matmul_sparse(
    x: &[f32],
    ternary: &[f32],
    scale: f32,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * out_dim];

    for b in 0..batch {
        let x_row = &x[b * in_dim..(b + 1) * in_dim];
        for i in 0..out_dim {
            let w_row = &ternary[i * in_dim..(i + 1) * in_dim];
            // Soma sparse: só elementos não-zero do peso ternário
            let dot: f32 = w_row.iter().zip(x_row.iter())
                .filter(|(&w, _)| w != 0.0)  // skip zeros
                .map(|(&w, &x)| if w > 0.0 { x } else { -x }) // +1 ou -1: só sign
                .sum();
            y[b * out_dim + i] = dot * scale;
        }
    }
    y
}

/// Versão com threshold: só computa saída onde |y_fast| > threshold.
/// Retorna (y, active_mask) onde active_mask[i] = true se neurônio i foi calculado.
pub fn ternary_matmul_routed(
    x: &[f32],
    ternary: &[f32],
    scale: f32,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    threshold: f32,
) -> (Vec<f32>, Vec<bool>) {
    let y_fast = ternary_matmul_sparse(x, ternary, scale, batch, out_dim, in_dim);
    let active: Vec<bool> = y_fast.iter().map(|&v| v.abs() > threshold).collect();
    (y_fast, active)
}

/// Pack ternary values into u32 (16 elements per u32), row-major layout.
/// packed[row * packed_cols + k_block]
pub fn pack_ternary_u32_row_major(ternary: &[f32], out_dim: usize, in_dim: usize) -> Vec<u32> {
    let packed_cols = in_dim / 16;
    let mut out = vec![0u32; out_dim * packed_cols];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let c: u32 = match ternary[row * in_dim + col] as i32 { 1 => 1, -1 => 2, _ => 0 };
            out[row * packed_cols + col / 16] |= c << ((col % 16) * 2);
        }
    }
    out
}

/// Pack ternary values into u32 (16 elements per u32), column-major layout.
/// packed[k_block * out_dim + row]
/// Adjacent output rows are adjacent in memory → coalesced GPU reads.
pub fn pack_ternary_u32_col_major(ternary: &[f32], out_dim: usize, in_dim: usize) -> Vec<u32> {
    let packed_cols = in_dim / 16;
    let mut out = vec![0u32; packed_cols * out_dim];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let c: u32 = match ternary[row * in_dim + col] as i32 { 1 => 1, -1 => 2, _ => 0 };
            out[(col / 16) * out_dim + row] |= c << ((col % 16) * 2);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let vals: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0];
        let packed = pack_ternary(&vals);
        let unpacked = unpack_ternary(&packed, vals.len());
        for (a, b) in vals.iter().zip(unpacked.iter()) {
            assert!((a - b).abs() < 1e-6, "pack/unpack mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_quantize_scale() {
        let w = vec![0.5f32, -0.3, 0.1, -0.8, 0.0, 0.4];
        let (t, scale) = quantize_ternary(&w);
        // Todos devem ser {-1, 0, 1}
        for v in &t {
            assert!(*v == -1.0 || *v == 0.0 || *v == 1.0, "non-ternary: {v}");
        }
        assert!(scale > 0.0);
    }

    #[test]
    fn test_sparse_matmul_correctness() {
        // 2×3 matrix: [[1,0,-1], [0,1,0]]
        let w: Vec<f32> = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let x: Vec<f32> = vec![2.0, 3.0, 4.0];
        let scale = 1.0f32;
        let y = ternary_matmul_sparse(&x, &w, scale, 1, 2, 3);
        // neuron 0: 1*2 + 0*3 + (-1)*4 = -2
        // neuron 1: 0*2 + 1*3 + 0*4 = 3
        assert!((y[0] - (-2.0)).abs() < 1e-5, "expected -2.0, got {}", y[0]);
        assert!((y[1] - 3.0).abs() < 1e-5, "expected 3.0, got {}", y[1]);
    }
}
