// ============================================================================
// generate.rs — Atenção multi-token + geração autoregressiva Qwen3
// ============================================================================

use crate::tokenizer;
use crate::model::QwenModel;
use tokio::sync::broadcast;
use serde::Serialize;
use candle_core::{Tensor, IndexOp};

#[derive(Debug, Serialize)]
pub struct GenerationUpdate {
    pub token: String,
    pub full_text: String,
    pub is_final: bool,
}

/// Gera uma resposta completa do modelo usando tensores reais e KV-Cache.
pub async fn generate_response(
    prompt: &str,
    model: &mut QwenModel,
    frame_tx: &broadcast::Sender<String>,
    max_new_tokens: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut tokens = tokenizer::tokenize(prompt).iter().map(|t| t.id).collect::<Vec<u32>>();
    let mut generated_text = String::new();
    let device = model.device.clone();

    log::info!("🚀 Geração Otimizada (KV-Cache): '{}'", prompt);
    
    // 1. Resetar cache para novo contexto
    model.reset_cache();

    for i in 0..max_new_tokens {
        // 2. No KV-Cache, se i > 0, processamos apenas o ÚLTIMO token
        let input_ids = if i == 0 {
            tokens.as_slice() // No primeiro passo, processamos o prompt inteiro (prefill)
        } else {
            &tokens[tokens.len() - 1..] // Nos passos seguintes, apenas o último token (decode)
        };

        let input_tensor = Tensor::new(input_ids, &device)?.unsqueeze(0)?;

        // 3. Forward Pass com consciência de posição
        let mut logits = model.forward(&input_tensor, tokens.len() - input_ids.len())?;
        
        // ── Interceptação de Logits (Filtro de Intuição) ──
        // Aqui o Rust atua "entre o tensor e a resposta".
        // Podemos manipular os logits brutos ANTES da seleção do token.
        // Exemplo: Penalizar repetições ou forçar uma "personalidade" via ressonância.
        {
            // logits = apply_intuition_filter(logits, &liquid_net);
        }

        // 4. Seleção de Token (Greedy)
        let last_logits = logits.i((0, input_ids.len() - 1))?;
        let next_token_id = last_logits.argmax(0)?.to_scalar::<u32>()?;

        // 5. Decodificar e Feedback
        let token_text = tokenizer::token_id_to_text(next_token_id);
        if token_text == "<|endoftext|>" || token_text == "<|im_end|>" {
            break;
        }

        generated_text.push_str(&token_text);
        tokens.push(next_token_id);

        // 6. Stream para o Dashboard
        let update = GenerationUpdate {
            token: token_text,
            full_text: generated_text.clone(),
            is_final: false,
        };
        let _ = frame_tx.send(serde_json::to_string(&serde_json::json!({
            "type": "generation_update",
            "data": update
        }))?);
        
        log::info!("📝 Token {}: {}", i, update.token);
    }

    Ok(generated_text)
}
