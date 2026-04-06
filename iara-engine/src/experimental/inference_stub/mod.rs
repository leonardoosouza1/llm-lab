// ============================================================================
// inference/mod.rs — Pipeline de Inferência Nativo (sem candle)
// ============================================================================

use crate::tokenizer;
use crate::model::{CausalLM, ModelResult};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct InferenceEngine {
    pub model: Arc<Mutex<CausalLM>>,
}

impl InferenceEngine {
    pub fn new(model: Arc<Mutex<CausalLM>>) -> Self {
        Self { model }
    }

    /// Greedy decoding — máxima precisão, sem temperatura.
    pub async fn answer(&self, prompt: &str, max_new_tokens: usize) -> ModelResult<String> {
        let mut tokens = tokenizer::tokenize(prompt)
            .iter().map(|t| t.id).collect::<Vec<u32>>();

        let mut generated = String::new();
        let mut model = self.model.lock().await;
        model.reset_cache();

        for i in 0..max_new_tokens {
            // Prefill no primeiro passo, decode a partir do segundo
            let input = if i == 0 { tokens.as_slice() } else { &tokens[tokens.len() - 1..] };
            let (logits, _traces) = model.forward(input, None)?;

            // Greedy argmax
            let next_id = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            let text = tokenizer::token_id_to_text(next_id);
            if text == "<|endoftext|>" || text == "<|im_end|>" { break; }

            generated.push_str(&text);
            tokens.push(next_id);

            // Streaming para stdout
            use std::io::Write;
            print!("{}", text);
            std::io::stdout().flush().ok();
        }
        println!();

        Ok(generated)
    }
}
