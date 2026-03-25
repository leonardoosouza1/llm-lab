// ============================================================================
// safetensors_reader.rs — Leitura zero-copy de pesos via memmap2
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

use half::bf16;
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Metadados de um tensor individual dentro de um arquivo safetensors
#[derive(Debug, Deserialize, Clone)]
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

/// Carrega o index.json do modelo safetensors
pub fn load_index(index_path: &Path) -> Result<SafetensorsIndex, Box<dyn std::error::Error>> {
    let file = File::open(index_path)?;
    let index: SafetensorsIndex = serde_json::from_reader(file)?;
    log::info!(
        "📋 Index carregado: {} tensores mapeados",
        index.weight_map.len()
    );
    Ok(index)
}

/// Lista todos os nomes de tensores disponíveis no modelo
pub fn list_tensor_names(index: &SafetensorsIndex) -> Vec<String> {
    let mut names: Vec<String> = index.weight_map.keys().cloned().collect();
    names.sort();
    names
}

/// Parseia o header de um arquivo safetensors mapeado em memória
fn parse_shard_header(
    mmap: &Mmap,
) -> Result<(HashMap<String, TensorMeta>, usize), Box<dyn std::error::Error>> {
    // Primeiros 8 bytes = tamanho do header (u64 little-endian)
    if mmap.len() < 8 {
        return Err("Arquivo safetensors muito pequeno".into());
    }

    let header_len = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;
    let header_end = 8 + header_len;

    if mmap.len() < header_end {
        return Err(format!(
            "Arquivo truncado: esperava {} bytes de header, arquivo tem {} bytes",
            header_end,
            mmap.len()
        )
        .into());
    }

    let header_json = std::str::from_utf8(&mmap[8..header_end])?;

    // O header pode conter "__metadata__" que não é um tensor
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)?;

    let mut tensors = HashMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let meta: TensorMeta = serde_json::from_value(value)?;
        tensors.insert(name, meta);
    }

    Ok((tensors, header_end))
}

/// Lê um tensor específico de um shard safetensors via memmap.
/// Converte BF16 → F32 automaticamente.
pub fn read_tensor(
    model_dir: &Path,
    index: &SafetensorsIndex,
    tensor_name: &str,
) -> Result<TensorData, Box<dyn std::error::Error>> {
    // 1. Descobrir em qual shard está o tensor
    let shard_name = index
        .weight_map
        .get(tensor_name)
        .ok_or_else(|| format!("Tensor '{}' não encontrado no index", tensor_name))?;

    let shard_path: PathBuf = model_dir.join(shard_name);
    log::info!("📂 Abrindo shard: {}", shard_path.display());

    // 2. Memory-map do arquivo (zero-copy)
    let file = File::open(&shard_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    log::info!("🗺️  Shard mapeado: {:.2} GB", mmap.len() as f64 / 1e9);

    // 3. Parsear header do shard
    let (shard_tensors, data_start) = parse_shard_header(&mmap)?;

    // 4. Encontrar o tensor no header do shard
    let meta = shard_tensors
        .get(tensor_name)
        .ok_or_else(|| format!("Tensor '{}' não encontrado no header do shard", tensor_name))?;

    let offset_begin = data_start + meta.data_offsets[0];
    let offset_end = data_start + meta.data_offsets[1];
    let raw_bytes = &mmap[offset_begin..offset_end];

    log::info!(
        "🧬 Tensor: {} | dtype: {} | shape: {:?} | bytes: {}",
        tensor_name,
        meta.dtype,
        meta.shape,
        raw_bytes.len()
    );

    // 5. Converter para f32
    let floats = match meta.dtype.as_str() {
        "BF16" => {
            let num_elements = raw_bytes.len() / 2;
            let mut result = Vec::with_capacity(num_elements);
            for chunk in raw_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = bf16::from_bits(bits).to_f32();
                result.push(val);
            }
            result
        }
        "F16" => {
            let num_elements = raw_bytes.len() / 2;
            let mut result = Vec::with_capacity(num_elements);
            for chunk in raw_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = half::f16::from_bits(bits).to_f32();
                result.push(val);
            }
            result
        }
        "F32" => {
            let num_elements = raw_bytes.len() / 4;
            let mut result = Vec::with_capacity(num_elements);
            for chunk in raw_bytes.chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result.push(val);
            }
            result
        }
        other => {
            return Err(format!("dtype não suportado: {}", other).into());
        }
    };

    // Stats
    let min = floats.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = floats.iter().sum::<f32>() / floats.len() as f32;
    log::info!(
        "📊 Stats: {} floats | min={:.6} max={:.6} mean={:.6}",
        floats.len(),
        min,
        max,
        mean
    );

    Ok(TensorData {
        name: tensor_name.to_string(),
        shape: meta.shape.clone(),
        data: floats,
    })
}
