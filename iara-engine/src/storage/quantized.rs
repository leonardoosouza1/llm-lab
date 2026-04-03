// ============================================================================
// storage/quantized.rs — Carregamento e quantização INT4
// ============================================================================
// Suporta:
//   1. Quantização on-load: BF16 → INT4 (grupo-sábio na CPU)
//   2. Formato .iara: pré-quantizado para carga rápida
//   3. Upload GPU: INT4 packed + scales + zeros
// ============================================================================

use half::bf16;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use compute_core::{ComputeBackend, AnyTensor};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Erros ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum QuantizationError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Quantization error: {0}")]
    Quantize(String),

    #[error("Invalid IARA format: {0}")]
    InvalidFormat(String),
}

pub type QuantizationResult<T> = std::result::Result<T, QuantizationError>;

// ── Tensor quantizado ────────────────────────────────────────────────────────

/// Tensor INT4 quantizado com metadados de grupo.
/// Todos os tensores (packed, scales, zeros) residem na GPU via AnyTensor.
pub struct QuantizedTensor {
    /// [rows, cols/2] INT4 comprimido na GPU (2 valores por byte)
    pub packed: AnyTensor,

    /// [rows, n_groups] F32 na GPU — escalas por grupo
    pub scales: AnyTensor,

    /// [rows, n_groups] F32 na GPU — zero-points por grupo
    pub zeros: AnyTensor,

    /// Shape original [rows, cols]
    pub shape: [usize; 2],

    /// Tamanho do grupo (tipicamente 128)
    pub group_size: usize,
}

// ── Resultado de quantização (CPU-side) ──────────────────────────────────────

/// Resultado intermediário da quantização INT4 na CPU.
/// Pronto para upload via ComputeBackend.
#[derive(Clone)]
pub struct QuantizationCpuResult {
    /// Bytes comprimidos INT4: [rows * cols / 2]
    pub packed: Vec<u8>,

    /// Escalas F32: [rows * n_groups]
    pub scales: Vec<f32>,

    /// Zero-points F32: [rows * n_groups]
    pub zeros: Vec<f32>,

    /// Número de grupos por linha
    pub n_groups_per_row: usize,
}

// ── Quantização CPU: BF16 → INT4 ────────────────────────────────────────────

/// Quantiza tensor BF16 para INT4 na CPU.
///
/// Algoritmo por grupo de `group_size` pesos:
///   1. Converter BF16 → F32
///   2. Encontrar min/max do grupo
///   3. scale = (max - min) / 15.0
///   4. zero = min
///   5. q = clamp(round((val - zero) / scale), 0, 15)
///   6. Pack 2 valores INT4 por byte (low nibble + high nibble)
///
/// # Arguments
/// * `bf16_data` — slice de bits BF16 (como u16)
/// * `shape` — [rows, cols]
/// * `group_size` — tipicamente 128; deve dividir cols
///
/// # Returns
/// Erro se cols não é divisível por group_size ou shape está vazio
pub fn quantize_bf16_to_q4(
    bf16_data: &[u16],
    shape: &[usize],
    group_size: usize,
) -> QuantizationResult<QuantizationCpuResult> {
    if shape.len() != 2 {
        return Err(QuantizationError::Quantize(
            format!("shape deve ter 2 dimensões, got {}", shape.len()),
        ));
    }

    let rows = shape[0];
    let cols = shape[1];

    if cols % group_size != 0 {
        return Err(QuantizationError::Quantize(
            format!(
                "cols {} não é divisível por group_size {}",
                cols, group_size
            ),
        ));
    }

    if bf16_data.len() != rows * cols {
        return Err(QuantizationError::Quantize(
            format!(
                "data length {} ≠ rows*cols {}*{}={}",
                bf16_data.len(),
                rows,
                cols,
                rows * cols
            ),
        ));
    }

    let n_groups_per_row = cols / group_size;
    let packed_size = rows * cols / 2; // 2 valores INT4 por byte
    let mut packed = vec![0u8; packed_size];
    let mut scales = Vec::with_capacity(rows * n_groups_per_row);
    let mut zeros = Vec::with_capacity(rows * n_groups_per_row);

    // Quantizar por linha
    for row_idx in 0..rows {
        let row_start = row_idx * cols;
        let row_data = &bf16_data[row_start..row_start + cols];

        // Quantizar cada grupo
        for group_idx in 0..n_groups_per_row {
            let group_start = group_idx * group_size;
            let group_end = group_start + group_size;
            let group_data = &row_data[group_start..group_end];

            // Converter BF16 → F32 e encontrar min/max
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let f32_data: Vec<f32> = group_data
                .iter()
                .map(|&bits| {
                    let f = bf16::from_bits(bits).to_f32();
                    min_val = min_val.min(f);
                    max_val = max_val.max(f);
                    f
                })
                .collect();

            // Evitar quantização trivial (todos zeros)
            let scale = if (max_val - min_val).abs() < 1e-10 {
                1.0
            } else {
                (max_val - min_val) / 15.0
            };

            let zero = min_val;
            scales.push(scale);
            zeros.push(zero);

            // Quantizar valores do grupo
            let q_data: Vec<u8> = f32_data
                .iter()
                .map(|&val| {
                    let normalized = if scale.abs() > 1e-10 {
                        (val - zero) / scale
                    } else {
                        0.0
                    };
                    let quantized = normalized.round() as i32;
                    quantized.clamp(0, 15) as u8
                })
                .collect();

            // Pack 2 valores INT4 por byte: low nibble + high nibble
            for i in 0..group_size / 2 {
                let lo = q_data[2 * i];
                let hi = q_data[2 * i + 1];
                let packed_byte = (hi << 4) | lo;

                let packed_idx = row_idx * (cols / 2) + group_idx * (group_size / 2) + i;
                packed[packed_idx] = packed_byte;
            }
        }
    }

    log::info!(
        "📊 Quantização Q4: [{}, {}] BF16 → packed {} B + scales {} + zeros {}",
        rows,
        cols,
        packed.len(),
        scales.len(),
        zeros.len()
    );

    Ok(QuantizationCpuResult {
        packed,
        scales,
        zeros,
        n_groups_per_row,
    })
}

// ── Upload para GPU ──────────────────────────────────────────────────────────

/// Upload de QuantizationCpuResult para GPU via backend.
#[allow(dead_code)]
pub fn upload_quantized(
    result: &QuantizationCpuResult,
    shape: &[usize],
    group_size: usize,
    backend: &Arc<dyn ComputeBackend>,
) -> QuantizationResult<QuantizedTensor> {
    if shape.len() != 2 {
        return Err(QuantizationError::Quantize(
            format!("shape deve ter 2 dimensões, got {}", shape.len()),
        ));
    }

    let rows = shape[0];
    let cols = shape[1];

    // Upload packed INT4: [rows, cols/2]
    let packed = backend.upload_q4(&result.packed, &[rows, cols / 2]);

    // Upload scales F32: [rows, n_groups]
    let scales = backend.upload_f32(&result.scales, &[rows, result.n_groups_per_row]);

    // Upload zeros F32: [rows, n_groups]
    let zeros = backend.upload_f32(&result.zeros, &[rows, result.n_groups_per_row]);

    log::info!(
        "✅ Quantizado upload: packed {} B, scales [{}, {}], zeros [{}, {}]",
        result.packed.len(),
        rows,
        result.n_groups_per_row,
        rows,
        result.n_groups_per_row
    );

    Ok(QuantizedTensor {
        packed,
        scales,
        zeros,
        shape: [rows, cols],
        group_size,
    })
}

// ── Formato .iara (pré-quantizado) ───────────────────────────────────────────

/// Metadados de um tensor dentro de um arquivo .iara
#[derive(Clone, Serialize, Deserialize)]
pub struct IaraTensorMeta {
    /// Shape original [rows, cols]
    pub shape: [usize; 2],

    /// Offset no arquivo de dados (absoluto)
    pub packed_offset: usize,
    pub packed_len: usize,

    pub scales_offset: usize,
    pub scales_len: usize,

    pub zeros_offset: usize,
    pub zeros_len: usize,
}

/// Magic bytes do formato .iara
const IARA_MAGIC: &[u8; 4] = b"IARA";
const IARA_VERSION: u32 = 1;

/// Carrega tensor quantizado de memmap .iara.
#[allow(dead_code)]
pub fn load_quantized_tensor_from_iara(
    mmap: &[u8],
    tensor_name: &str,
    meta: &IaraTensorMeta,
    header_start: usize,
    backend: &Arc<dyn ComputeBackend>,
    group_size: usize,
) -> QuantizationResult<QuantizedTensor> {
    let rows = meta.shape[0];
    let cols = meta.shape[1];
    let n_groups = cols / group_size;

    // Offsets absolutos no mmap (a partir do início dos dados)
    let data_start = header_start;
    let packed_start = data_start + meta.packed_offset;
    let scales_start = data_start + meta.scales_offset;
    let zeros_start = data_start + meta.zeros_offset;

    // Validar bounds
    if packed_start + meta.packed_len > mmap.len()
        || scales_start + meta.scales_len > mmap.len()
        || zeros_start + meta.zeros_len > mmap.len()
    {
        return Err(QuantizationError::InvalidFormat(
            "Offsets de tensor excedem tamanho do arquivo".into(),
        ));
    }

    // Extrair slices
    let packed_bytes = &mmap[packed_start..packed_start + meta.packed_len];
    let scales_bytes = &mmap[scales_start..scales_start + meta.scales_len];
    let zeros_bytes = &mmap[zeros_start..zeros_start + meta.zeros_len];

    // Parsear F32 (escalas e zeros)
    let scales: Vec<f32> = scales_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let zeros: Vec<f32> = zeros_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    if scales.len() != rows * n_groups || zeros.len() != rows * n_groups {
        return Err(QuantizationError::Quantize(
            format!(
                "Tamanho de scales/zeros inconsistente: {} vs {} (esperado {})",
                scales.len(),
                zeros.len(),
                rows * n_groups
            ),
        ));
    }

    // Upload para GPU
    let packed = backend.upload_q4(packed_bytes, &[rows, cols / 2]);
    let scales = backend.upload_f32(&scales, &[rows, n_groups]);
    let zeros = backend.upload_f32(&zeros, &[rows, n_groups]);

    log::info!(
        "📥 Tensor quantizado carregado: {} | shape [{}, {}]",
        tensor_name,
        rows,
        cols
    );

    Ok(QuantizedTensor {
        packed,
        scales,
        zeros,
        shape: [rows, cols],
        group_size,
    })
}

// ── Sistema de cache pré-quantizado ──────────────────────────────────────────

/// Salva mapa de tensores quantizados em cache .iara.
///
/// Layout do arquivo:
/// ```
/// [4 bytes]   magic "IARA"
/// [4 bytes]   version (u32 LE)
/// [4 bytes]   group_size (u32 LE)
/// [4 bytes]   num_tensors (u32 LE)
/// [...]       Tensor entries:
///   Para cada tensor:
///   [4 bytes]   name_len (u32 LE)
///   [name_len]  name (UTF-8)
///   [4 bytes]   rows (u32 LE)
///   [4 bytes]   cols (u32 LE)
///   [4 bytes]   packed_len (u32 LE)
///   [packed_len] packed_data (INT4)
///   [4 bytes]   scales_len (u32 LE)
///   [scales_len] scales_data (f32 LE)
///   [4 bytes]   zeros_len (u32 LE)
///   [zeros_len]  zeros_data (f32 LE)
/// ```
pub fn save_quantized_cache(
    model_dir: &Path,
    tensors: &HashMap<String, QuantizationCpuResult>,
    shapes: &HashMap<String, [usize; 2]>,
    group_size: usize,
) -> QuantizationResult<()> {
    let cache_path = model_dir.join("quantized_cache.iara");
    let mut file = File::create(&cache_path)?;

    // Magic + version + group_size + tensor count
    file.write_all(IARA_MAGIC)?;
    file.write_all(&IARA_VERSION.to_le_bytes())?;
    file.write_all(&(group_size as u32).to_le_bytes())?;
    file.write_all(&(tensors.len() as u32).to_le_bytes())?;

    // Salvar cada tensor
    for (name, cpu_result) in tensors {
        let shape = shapes
            .get(name)
            .ok_or_else(|| QuantizationError::Quantize(
                format!("Shape não encontrado para tensor '{}'", name),
            ))?;

        let rows = shape[0] as u32;
        let cols = shape[1] as u32;

        // Nome do tensor
        let name_bytes = name.as_bytes();
        file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        file.write_all(name_bytes)?;

        // Shape
        file.write_all(&rows.to_le_bytes())?;
        file.write_all(&cols.to_le_bytes())?;

        // Packed INT4
        file.write_all(&(cpu_result.packed.len() as u32).to_le_bytes())?;
        file.write_all(&cpu_result.packed)?;

        // Scales F32
        file.write_all(&(cpu_result.scales.len() as u32).to_le_bytes())?;
        for &s in &cpu_result.scales {
            file.write_all(&s.to_le_bytes())?;
        }

        // Zeros F32
        file.write_all(&(cpu_result.zeros.len() as u32).to_le_bytes())?;
        for &z in &cpu_result.zeros {
            file.write_all(&z.to_le_bytes())?;
        }
    }

    log::info!(
        "💾 Cache quantizado salvo: {} ({} tensores)",
        cache_path.display(),
        tensors.len()
    );

    Ok(())
}

/// Carrega cache quantizado pré-salvo de disco.
/// Retorna HashMap<tensor_name, QuantizationCpuResult> se encontrado e válido.
pub fn load_quantized_cache(
    model_dir: &Path,
) -> QuantizationResult<Option<(HashMap<String, QuantizationCpuResult>, HashMap<String, [usize; 2]>, usize)>> {
    let cache_path = model_dir.join("quantized_cache.iara");

    if !cache_path.exists() {
        log::debug!("⏭️  Cache quantizado não encontrado: {}", cache_path.display());
        return Ok(None);
    }

    let data = std::fs::read(&cache_path)?;

    if data.len() < 16 {
        return Err(QuantizationError::InvalidFormat(
            "Arquivo cache muito pequeno: < 16 bytes".into(),
        ));
    }

    // Magic
    if &data[0..4] != IARA_MAGIC {
        return Err(QuantizationError::InvalidFormat(
            format!(
                "Magic inválido no cache: {:?} (esperado IARA)",
                &data[0..4]
            ),
        ));
    }

    // Version
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != IARA_VERSION {
        return Err(QuantizationError::InvalidFormat(
            format!(
                "Version incompatível no cache: {} (esperado {})",
                version, IARA_VERSION
            ),
        ));
    }

    // Group size
    let group_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

    // Tensor count
    let num_tensors = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

    let mut tensors = HashMap::new();
    let mut shapes = HashMap::new();
    let mut pos = 16usize;

    for tensor_idx in 0..num_tensors {
        // Nome
        if pos + 4 > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler tensor {}", tensor_idx),
            ));
        }
        let name_len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + name_len > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler nome tensor {}", tensor_idx),
            ));
        }
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|e| QuantizationError::InvalidFormat(format!("UTF-8 inválido no nome: {}", e)))?
            .to_string();
        pos += name_len;

        // Shape
        if pos + 8 > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler shape tensor {}", tensor_idx),
            ));
        }
        let rows = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let cols = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        pos += 8;

        // Packed INT4
        if pos + 4 > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler packed_len tensor '{}'", name),
            ));
        }
        let packed_len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + packed_len > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler packed_data tensor '{}'", name),
            ));
        }
        let packed = data[pos..pos + packed_len].to_vec();
        pos += packed_len;

        // Scales F32
        if pos + 4 > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler scales_len tensor '{}'", name),
            ));
        }
        let scales_len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + scales_len * 4 > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler scales_data tensor '{}'", name),
            ));
        }
        let mut scales = Vec::with_capacity(scales_len);
        for i in 0..scales_len {
            let offset = pos + i * 4;
            let scale = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            scales.push(scale);
        }
        pos += scales_len * 4;

        // Zeros F32
        if pos + 4 > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler zeros_len tensor '{}'", name),
            ));
        }
        let zeros_len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + zeros_len * 4 > data.len() {
            return Err(QuantizationError::InvalidFormat(
                format!("Arquivo truncado ao ler zeros_data tensor '{}'", name),
            ));
        }
        let mut zeros = Vec::with_capacity(zeros_len);
        for i in 0..zeros_len {
            let offset = pos + i * 4;
            let zero = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            zeros.push(zero);
        }
        pos += zeros_len * 4;

        // Validar consistência de scales/zeros
        if scales.len() != (rows * cols / group_size) {
            return Err(QuantizationError::Quantize(
                format!(
                    "Tamanho scales inconsistente para '{}': {} vs esperado {}",
                    name,
                    scales.len(),
                    rows * cols / group_size
                ),
            ));
        }
        if zeros.len() != (rows * cols / group_size) {
            return Err(QuantizationError::Quantize(
                format!(
                    "Tamanho zeros inconsistente para '{}': {} vs esperado {}",
                    name,
                    zeros.len(),
                    rows * cols / group_size
                ),
            ));
        }

        tensors.insert(
            name.clone(),
            QuantizationCpuResult {
                packed,
                scales,
                zeros,
                n_groups_per_row: cols / group_size,
            },
        );
        shapes.insert(name, [rows, cols]);
    }

    log::info!(
        "✅ Cache quantizado carregado: {} tensores, group_size={}",
        tensors.len(),
        group_size
    );

    Ok(Some((tensors, shapes, group_size)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_bf16_to_q4() {
        // Teste simples: tensores pequenos com valores conhecidos
        let rows = 2;
        let cols = 4;
        let group_size = 2;

        // Criar dados BF16: [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let bf16_data: Vec<u16> = values
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect();

        let result = quantize_bf16_to_q4(&bf16_data, &[rows, cols], group_size)
            .expect("quantização falhou");

        // Verificações básicas
        assert_eq!(result.packed.len(), rows * cols / 2);
        assert_eq!(result.scales.len(), rows * (cols / group_size));
        assert_eq!(result.zeros.len(), rows * (cols / group_size));
    }
}
