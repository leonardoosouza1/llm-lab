// ============================================================================
// embedding.rs — Leitura de embeddings do modelo via mmap
// ============================================================================
// Extrai linhas específicas de embed_tokens.weight [vocab_size, hidden_dim]
// diretamente do safetensors, sem carregar a matriz toda na RAM.
//
// Cada token ID mapeia para uma linha de 4096 floats (hidden_dim do Qwen 2.5).
// ============================================================================

use crate::safetensors_reader::{SafetensorsIndex, TensorMeta};
use half::bf16;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Resultado da leitura de embeddings: vetor de linhas [hidden_dim]
pub struct EmbeddingResult {
    pub hidden_dim: usize,
    pub vectors: Vec<Vec<f32>>, // Uma linha por token
}

/// Parseia header de um shard e retorna metadados dos tensores + offset dos dados
fn parse_shard_header(
    mmap: &Mmap,
) -> Result<(HashMap<String, TensorMeta>, usize), Box<dyn std::error::Error>> {
    let header_len = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;
    let header_end = 8 + header_len;
    let header_json = std::str::from_utf8(&mmap[8..header_end])?;
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

/// Lê linhas específicas de um tensor 2D (ex: embed_tokens.weight)
///
/// Retorna apenas as linhas solicitadas, sem carregar a matriz completa.
/// Ideal para lookup de embeddings por token ID.
pub fn read_embedding_rows(
    model_dir: &Path,
    index: &SafetensorsIndex,
    tensor_name: &str,
    row_indices: &[u32],
) -> Result<EmbeddingResult, Box<dyn std::error::Error>> {
    // Descobrir shard
    let shard_name = index
        .weight_map
        .get(tensor_name)
        .ok_or_else(|| format!("Tensor '{}' não encontrado", tensor_name))?;

    let shard_path = model_dir.join(shard_name);
    let file = File::open(&shard_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Parsear header
    let (shard_tensors, data_start) = parse_shard_header(&mmap)?;
    let meta = shard_tensors
        .get(tensor_name)
        .ok_or_else(|| format!("Tensor '{}' não no header do shard", tensor_name))?;

    // Validar shape [vocab_size, hidden_dim]
    if meta.shape.len() != 2 {
        return Err(format!(
            "Tensor '{}' não é 2D: shape={:?}",
            tensor_name, meta.shape
        )
        .into());
    }

    let vocab_size = meta.shape[0];
    let hidden_dim = meta.shape[1];
    let bytes_per_element = match meta.dtype.as_str() {
        "BF16" | "F16" => 2,
        "F32" => 4,
        other => return Err(format!("dtype não suportado: {}", other).into()),
    };
    let row_bytes = hidden_dim * bytes_per_element;

    let tensor_data_start = data_start + meta.data_offsets[0];

    log::info!(
        "📐 Embedding: {} shape=[{}, {}] dtype={} — extraindo {} linhas",
        tensor_name,
        vocab_size,
        hidden_dim,
        meta.dtype,
        row_indices.len()
    );

    let mut vectors = Vec::with_capacity(row_indices.len());

    for &row_idx in row_indices {
        if row_idx as usize >= vocab_size {
            // Retornar vetor zerado para IDs fora do range
            vectors.push(vec![0.0f32; hidden_dim]);
            continue;
        }

        let row_start = tensor_data_start + (row_idx as usize * row_bytes);
        let row_end = row_start + row_bytes;
        let row_data = &mmap[row_start..row_end];

        let row_floats: Vec<f32> = match meta.dtype.as_str() {
            "BF16" => row_data
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F16" => row_data
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F32" => row_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            _ => unreachable!(),
        };

        vectors.push(row_floats);
    }

    Ok(EmbeddingResult {
        hidden_dim,
        vectors,
    })
}

/// Retorna o embedding do **último token** do prompt.
///
/// Para predição de próximo token em modelos autoregressivos (Qwen/Llama),
/// o estado oculto relevante é o do último token — é a partir dele que o
/// modelo prediz o que vem a seguir. Usar a média de todos os tokens destrói
/// a estrutura posicional e produz um vetor sem significado direcional no
/// espaço do lm_head.
///
/// O motor de ressonância ainda recebe TODOS os embeddings como pulso físico
/// (via `pipeline.rs → think_tx`), garantindo que todos os tokens influenciem
/// a LiquidNetwork neuromórfica.
pub fn combine_embeddings(embeddings: &EmbeddingResult) -> Vec<f32> {
    if embeddings.vectors.is_empty() {
        return vec![];
    }

    // Último token: ponto de partida para predição autoregressiva
    let last = embeddings.vectors.last().unwrap().clone();

    let norm: f32 = last.iter().map(|v| v * v).sum::<f32>().sqrt();
    log::info!(
        "🧮 Embedding do último token: dim={}, norm={:.4}, total_tokens={}",
        embeddings.hidden_dim, norm, embeddings.vectors.len()
    );

    last
}
