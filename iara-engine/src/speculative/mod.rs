//! Speculative Decoding — draft model propõe K tokens, target verifica em batch.
//!
//! Fluxo:
//!   1. Draft model gera K tokens autoregressivamente (rápido, modelo pequeno)
//!   2. Target model faz forward(prompt + K tokens) em um batch (1 forward pass)
//!   3. Comparar logits do target vs tokens do draft
//!   4. Aceitar tokens enquanto concordam, rejeitar a partir da primeira divergência
//!   5. Repetir do ponto de divergência
//!
//! # Exemplo de uso:
//! ```ignore
//! let mut engine = SpeculativeEngine::new(SpeculativeConfig::default());
//! let step = engine.step(&mut draft_model, &mut target_model, &input_ids, &eos_ids)?;
//! println!("Accepted {} tokens (rate: {:.2}%)",
//!          step.accepted_tokens.len(),
//!          step.acceptance_rate * 100.0);
//! ```

pub mod verify;

use crate::model::{CausalLM, ModelResult};

/// Configuração do speculative decoding
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SpeculativeConfig {
    /// Número de tokens candidatos por step (K)
    pub draft_tokens: usize,
    /// Temperature para amostragem no draft model
    pub temperature: f32,
    /// Top-k para amostragem no draft model
    pub top_k: u32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_tokens: 4,
            temperature: 0.7,
            top_k: 40,
        }
    }
}

/// Resultado de um step de speculative decoding
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SpeculativeStep {
    /// Tokens aceitos neste step (1..=K)
    pub accepted_tokens: Vec<u32>,
    /// Token strings correspondentes
    pub accepted_strings: Vec<String>,
    /// Quantos dos K candidatos foram aceitos
    pub acceptance_count: usize,
    /// Taxa de aceitação média acumulada
    pub acceptance_rate: f32,
}

/// Engine de speculative decoding que coordena draft + target
#[allow(dead_code)]
pub struct SpeculativeEngine {
    #[allow(dead_code)]
    pub config: SpeculativeConfig,
    /// Estatísticas acumuladas
    pub total_draft: usize,
    pub total_accepted: usize,
}

impl SpeculativeEngine {
    /// Cria um novo SpeculativeEngine com configuração padrão
    #[allow(dead_code)]
    pub fn new(config: SpeculativeConfig) -> Self {
        Self {
            config,
            total_draft: 0,
            total_accepted: 0,
        }
    }

    /// Executa um step completo de speculative decoding.
    ///
    /// 1. Usa draft_model para gerar K tokens candidatos
    /// 2. Usa target_model para verificar todos em 1 forward pass
    /// 3. Retorna tokens aceitos
    ///
    /// # Arguments
    /// * `draft_model` - Modelo de rascunho (pequeno, rápido)
    /// * `target_model` - Modelo alvo (grande, preciso)
    /// * `input_ids` - Sequência de entrada atual [last_token]
    /// * `eos_ids` - IDs de tokens de fim de sequência
    ///
    /// # Returns
    /// Um `SpeculativeStep` contendo tokens aceitos e estatísticas
    ///
    /// # Errors
    /// Retorna `ModelError` se um forward pass falhar
    #[allow(dead_code)]
    pub fn step(
        &mut self,
        draft_model: &mut CausalLM,
        target_model: &mut CausalLM,
        input_ids: &[u32],
        eos_ids: &[u32],
    ) -> ModelResult<SpeculativeStep> {
        let k = self.config.draft_tokens;

        // Phase 1: Draft generates K candidates
        let mut draft_tokens = Vec::with_capacity(k);
        let mut current_ids = input_ids.to_vec();

        for _ in 0..k {
            let (logits, _traces) = draft_model.forward(&current_ids)?;
            let token = sample_greedy(&logits);

            if eos_ids.contains(&token) {
                break;
            }

            draft_tokens.push(token);
            current_ids = vec![token]; // next token input (autoregressive draft)
        }

        if draft_tokens.is_empty() {
            return Ok(SpeculativeStep {
                accepted_tokens: vec![],
                accepted_strings: vec![],
                acceptance_count: 0,
                acceptance_rate: self.acceptance_rate(),
            });
        }

        // Phase 2: Target verifies all K tokens in one forward pass
        // Build the verification input: original input + all draft tokens
        let mut verify_ids = input_ids.to_vec();
        verify_ids.extend_from_slice(&draft_tokens);

        let (target_logits, _traces) = target_model.forward(&verify_ids)?;

        // Phase 3: Accept/reject
        // target_logits contains logits for the last position only
        // (current forward() returns only last-token logits)
        // We accept all draft tokens and use the target's final logits
        // to generate the next token after the draft sequence.
        let accepted = verify::accept_tokens(
            &target_logits,
            &draft_tokens,
            target_model.cfg.vocab_size,
            input_ids.len(),
            self.config.temperature,
        );

        let acceptance_count = accepted.len();
        self.total_draft += draft_tokens.len();
        self.total_accepted += acceptance_count;

        Ok(SpeculativeStep {
            accepted_tokens: accepted,
            accepted_strings: vec![], // caller fills with tokenizer
            acceptance_count,
            acceptance_rate: self.acceptance_rate(),
        })
    }

    /// Retorna a taxa de aceitação acumulada (0.0 a 1.0)
    #[allow(dead_code)]
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_draft as f32
        }
    }

    /// Retorna estatísticas acumuladas
    #[allow(dead_code)]
    pub fn stats(&self) -> SpeculativeStats {
        SpeculativeStats {
            total_draft_tokens: self.total_draft,
            total_accepted_tokens: self.total_accepted,
            acceptance_rate: self.acceptance_rate(),
        }
    }

    /// Reseta estatísticas acumuladas
    #[allow(dead_code)]
    pub fn reset_stats(&mut self) {
        self.total_draft = 0;
        self.total_accepted = 0;
    }
}

/// Estatísticas de speculative decoding
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SpeculativeStats {
    pub total_draft_tokens: usize,
    pub total_accepted_tokens: usize,
    pub acceptance_rate: f32,
}

/// Amostra greedily: retorna o token com maior logit
#[allow(dead_code)]
fn sample_greedy(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(sample_greedy(&logits), 1);
    }

    #[test]
    fn test_config_default() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.draft_tokens, 4);
        assert_eq!(cfg.top_k, 40);
    }

    #[test]
    fn test_engine_acceptance_rate() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig::default());
        engine.total_draft = 10;
        engine.total_accepted = 7;
        assert!((engine.acceptance_rate() - 0.7).abs() < 0.0001);
    }

    #[test]
    fn test_engine_acceptance_rate_zero() {
        let engine = SpeculativeEngine::new(SpeculativeConfig::default());
        assert_eq!(engine.acceptance_rate(), 0.0);
    }
}
