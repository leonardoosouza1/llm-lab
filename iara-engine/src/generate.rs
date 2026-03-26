// ============================================================================
// generate.rs — Geração autoregressiva com streaming e telemetria
// ============================================================================

use crate::tokenizer;
use crate::model::QwenModel;
use crate::model::ModelResult;
use tokio::sync::broadcast;
use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct GenerationUpdate {
    pub token: String,
    pub full_text: String,
    pub is_final: bool,
    /// Logits brutos do token gerado (top-K para observabilidade)
    pub top_logits: Vec<(u32, f32)>,
}

/// Gera resposta com streaming e telemetria de logits.
pub async fn generate_response(
    prompt: &str,
    model: &mut QwenModel,
    frame_tx: &broadcast::Sender<String>,
    max_new_tokens: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut tokens: Vec<u32> = tokenizer::tokenize(prompt)
        .iter().map(|t| t.id).collect();
    let mut generated = String::new();

    log::info!("🚀 Geração nativa HIP: '{}' ({} tokens)", prompt, tokens.len());
    model.reset_cache();

    for i in 0..max_new_tokens {
        let input = if i == 0 { tokens.as_slice() } else { &tokens[tokens.len() - 1..] };
        let logits = model.forward(input)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        // Top-10 logits para telemetria (Fase 2: mapa de calor de ativações)
        let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k: Vec<(u32, f32)> = indexed[..10.min(indexed.len())].to_vec();

        let next_id = top_k[0].0;
        let token_text = tokenizer::token_id_to_text(next_id);

        if token_text == "<|endoftext|>" || token_text == "<|im_end|>" { break; }

        generated.push_str(&token_text);
        tokens.push(next_id);

        let update = GenerationUpdate {
            token: token_text.clone(),
            full_text: generated.clone(),
            is_final: false,
            top_logits: top_k,
        };

        let _ = frame_tx.send(serde_json::to_string(&serde_json::json!({
            "type": "generation_update",
            "data": update
        }))?);

        log::debug!("Token {}: {}", i, token_text);
    }

    Ok(generated)
}
