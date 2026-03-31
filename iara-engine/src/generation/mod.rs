// ============================================================================
// generation/mod.rs — Loop autoregressivo com ESN (Fase 1)
// ============================================================================
//
// Fluxo por token:
//   1. model.forward(input)  → logits [vocab_size]
//   2. esn.step(logits, pos) → estado do reservatório evolui (observação)
//   3. params = esn.modulate(params) → ajusta temperatura (no-op se não treinado)
//   4. sample_token(logits, params)  → next_id
//   5. broadcast: GenerationUpdate + EsnTelemetry para frontend
//
// A ESN é reinicializada no início de cada geração via esn.reset().
// O reservatório W_res e W_in são fixos (gerados uma vez na criação).
// Apenas W_out é treinável — começa em zero (Fase 1a).
// ============================================================================

use crate::lnn::EsnObserver;
use crate::tokenizer;
use crate::model::QwenModel;
use crate::inference::{SamplingParams, sample_token};
use tokio::sync::broadcast;
use serde::Serialize;

/// ESN seed fixo — garante reservatório reprodutível entre reinicializações.
const ESN_SEED: u64 = 0xDEAD_BEEF_1337_4242;

#[derive(Debug, Serialize, Clone)]
pub struct GenerationUpdate {
    pub token: String,
    pub full_text: String,
    pub is_final: bool,
    /// Logits brutos do token gerado (top-K para observabilidade)
    pub top_logits: Vec<(u32, f32)>,
}

/// Gera resposta com streaming, telemetria de logits e ESN fase 1.
pub async fn generate_response(
    prompt: &str,
    model: &mut QwenModel,
    frame_tx: &broadcast::Sender<String>,
    max_new_tokens: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut tokens: Vec<u32> = tokenizer::tokenize(prompt)
        .iter().map(|t| t.id).collect();
    let mut generated = String::new();

    log::info!("Geração nativa: '{}' ({} tokens)", prompt, tokens.len());
    model.reset_cache();

    // ── Inicializa ESN ───────────────────────────────────────────────────
    let mut esn = EsnObserver::new(ESN_SEED);
    esn.reset();

    let mut sampling = SamplingParams::default();

    for i in 0..max_new_tokens {
        let input = if i == 0 { tokens.as_slice() } else { &tokens[tokens.len() - 1..] };
        let logits = model.forward(input)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        // ── ESN: observa logits do token atual ───────────────────────────
        esn.step(&logits, i, max_new_tokens);

        // ── ESN: modula parâmetros de sampling do próximo token ──────────
        // Com W_out = 0 (fase 1a), esta chamada é no-op.
        sampling = esn.modulate(&sampling);

        // ── Amostra próximo token ────────────────────────────────────────
        let next_id = sample_token(&logits, &sampling);

        // ── Top-K logits para telemetria ─────────────────────────────────
        let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k: Vec<(u32, f32)> = indexed[..10.min(indexed.len())].to_vec();

        let token_text = tokenizer::token_id_to_text(next_id);
        if token_text == "<|endoftext|>" || token_text == "<|im_end|>" {
            break;
        }

        generated.push_str(&token_text);
        tokens.push(next_id);

        // ── Telemetria ESN ───────────────────────────────────────────────
        let esn_tel = esn.telemetry();

        // ── Broadcast frame ──────────────────────────────────────────────
        let update = GenerationUpdate {
            token: token_text.clone(),
            full_text: generated.clone(),
            is_final: false,
            top_logits: top_k,
        };

        let _ = frame_tx.send(serde_json::to_string(&serde_json::json!({
            "type": "generation_update",
            "data": update,
            "esn": esn_tel,
        }))?);

        log::debug!("Token {}: {} | esn.entropy={:.3} top1={:.3} state_norm={:.3}",
            i, token_text,
            esn_tel.entropy, esn_tel.top1_prob, esn_tel.state_norm);
    }

    Ok(generated)
}
