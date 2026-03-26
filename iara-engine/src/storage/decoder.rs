// ============================================================================
// decoder.rs — Decodificação: ativação final → tokens via lm_head.weight
// ============================================================================
// Computa similaridade cosseno entre o vetor de ativação final [hidden_dim]
// e cada linha de lm_head.weight [vocab_size, hidden_dim] via mmap streaming.
//
// Retorna os top-k tokens mais similares com seus scores.
// ============================================================================

use crate::storage::SafetensorsIndex;
use half::bf16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

/// Resultado de um token decodificado
#[derive(Debug, Clone)]
pub struct DecodedToken {
    pub token_id: u32,
    pub score: f32,
    pub text: String,
}

/// Decodifica o vetor de ativação final contra lm_head.weight via logits brutos.
///
/// Usa mmap para ler lm_head.weight linha por linha sem carregar ~2.5GB na RAM.
///
/// **Importante:** usa produto escalar PURO (não cosine similarity), pois é
/// exatamente o que `logits = hidden @ lm_head.T` faz no Qwen original.
/// Cosine similarity normaliza os dois lados e destrói o ranking treinado.
pub fn decode_activation(
    model_dir: &Path,
    index: &SafetensorsIndex,
    activation: &[f32],
    top_k: usize,
) -> Result<Vec<DecodedToken>, Box<dyn std::error::Error>> {
    let tensor_name = "lm_head.weight";

    let shard_name = index
        .weight_map
        .get(tensor_name)
        .ok_or_else(|| format!("Tensor '{}' não encontrado", tensor_name))?;

    let shard_path = model_dir.join(shard_name);
    let file = File::open(&shard_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Parsear header
    let header_len = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;
    let header_end = 8 + header_len;
    let header_json = std::str::from_utf8(&mmap[8..header_end])?;
    let raw: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(header_json)?;

    let meta_value = raw
        .get(tensor_name)
        .ok_or_else(|| format!("{} não encontrado no header", tensor_name))?;
    let meta: crate::storage::TensorMeta = serde_json::from_value(meta_value.clone())?;

    if meta.shape.len() != 2 {
        return Err(format!("lm_head não é 2D: {:?}", meta.shape).into());
    }

    let vocab_size = meta.shape[0];
    let hidden_dim = meta.shape[1];

    if activation.len() != hidden_dim {
        return Err(format!(
            "Dimensão incompatível: activation={} vs hidden_dim={}",
            activation.len(),
            hidden_dim
        )
        .into());
    }

    let bytes_per_element: usize = match meta.dtype.as_str() {
        "BF16" | "F16" => 2,
        "F32" => 4,
        other => return Err(format!("dtype não suportado: {}", other).into()),
    };
    let row_bytes = hidden_dim * bytes_per_element;
    let tensor_data_start = header_end + meta.data_offsets[0];

    log::info!(
        "🔍 Decodificando: lm_head [{}, {}] — {} comparações",
        vocab_size,
        hidden_dim,
        vocab_size
    );

    // Produto escalar bruto: logits = hidden @ lm_head.weight.T
    // NÃO normalizar — o modelo foi treinado com logits absolutos.
    // Aplicamos softmax numérico estável no final para converter em probabilidades.
    let block_size = 4096;
    let num_blocks = (vocab_size + block_size - 1) / block_size;

    let all_scores: Vec<(u32, f32)> = (0..num_blocks)
        .into_par_iter()
        .flat_map(|block_idx| {
            let start_row = block_idx * block_size;
            let end_row = (start_row + block_size).min(vocab_size);
            let mut block_scores = Vec::with_capacity(end_row - start_row);

            for row_idx in start_row..end_row {
                let row_start = tensor_data_start + row_idx * row_bytes;
                let row_end = row_start + row_bytes;
                if row_end > mmap.len() { break; }
                let row_data = &mmap[row_start..row_end];

                // Logit = dot(hidden, lm_head_row)  — produto escalar puro
                let logit: f32 = match meta.dtype.as_str() {
                    "BF16" => row_data
                        .chunks_exact(2)
                        .zip(activation.iter())
                        .map(|(c, &a)| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32() * a)
                        .sum(),
                    "F16" => row_data
                        .chunks_exact(2)
                        .zip(activation.iter())
                        .map(|(c, &a)| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32() * a)
                        .sum(),
                    "F32" => row_data
                        .chunks_exact(4)
                        .zip(activation.iter())
                        .map(|(c, &a)| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * a)
                        .sum(),
                    _ => 0.0,
                };

                block_scores.push((row_idx as u32, logit));
            }
            block_scores
        })
        .collect();

    // Softmax numérico estável: P(token) = exp(logit - max) / sum(exp(logit - max))
    // Operamos só sobre o top-k para não alocar 150k floats
    // (ranquear por logit bruto primeiro, depois converter top-k para probs)
    let all_scores = {
        let mut s = all_scores;
        s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Softmax sobre os top-k logits
        let top = s.iter().take(top_k).cloned().collect::<Vec<_>>();
        let max_logit = top.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|(_, l)| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        top.into_iter().zip(exps.iter())
            .map(|((id, _), &e)| (id, e / (sum_exp + 1e-8)))
            .collect::<Vec<_>>()
    };

    // Encontrar top-k
    let mut sorted = all_scores;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(top_k);

    let results: Vec<DecodedToken> = sorted
        .into_iter()
        .map(|(id, score)| DecodedToken {
            token_id: id,
            score,
            text: crate::tokenizer::token_id_to_text(id),
        })
        .collect();

    log::info!(
        "🏆 Top-{}: {:?}",
        top_k,
        results
            .iter()
            .map(|t| format!("{}[{}]: {:.4}", t.text, t.token_id, t.score))
            .collect::<Vec<_>>()
    );

    Ok(results)
}
