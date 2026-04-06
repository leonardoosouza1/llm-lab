// ============================================================================
// storage/ — Leitura de pesos safetensors, decoder, embedding e quantização
// ============================================================================
pub mod decoder;
pub mod embedding;
pub mod quantized;
pub mod bitnet;
pub mod fp8;
pub mod iara_format;
pub mod surgery;

// ── Re-exports para fácil acesso ──
#[allow(unused_imports)]
pub use fp8::{
    f32_to_fp8_e4m3, fp8_e4m3_to_f32, Fp8Quantized,
    quantize_bf16_to_fp8, dequantize_fp8_to_f32,
};


#[allow(unused_imports)]
pub use quantized::{
    QuantizedTensor, QuantizationError, QuantizationResult, QuantizationCpuResult,
    IaraTensorMeta,
    quantize_bf16_to_q4, upload_quantized,
    load_quantized_tensor_from_iara,
    save_quantized_cache, load_quantized_cache,
};

// ============================================================================
// safetensors — Leitura zero-copy de pesos via memmap2
// ============================================================================
// Formato safetensors:
//   [8 bytes: header_len (u64 LE)]
//   [header_len bytes: JSON header com metadados dos tensores]
//   [tensor data: bytes contíguos dos tensores]
//
// O JSON header contém para cada tensor:
//   { "name": { "dtype": "BF16", "shape": [rows, cols], "data_offsets": [start, end] } }
//   Os offsets são relativos ao INÍCIO dos dados (após o header).
// ============================================================================

use serde::Deserialize;
use std::collections::HashMap;

/// Metadados de um tensor individual dentro de um arquivo safetensors
#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: Vec<usize>,
}

/// Mapa de nome do tensor → arquivo shard que o contém
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SafetensorsIndex {
    pub metadata: serde_json::Value,
    pub weight_map: HashMap<String, String>,
}

/// Resultado da leitura de um tensor: floats convertidos para f32
#[allow(dead_code)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

