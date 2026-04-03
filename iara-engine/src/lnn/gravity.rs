// ============================================================================
// lnn/gravity.rs — Gravidade Semântica: importância por token
// ============================================================================
//
// Gravity Score: mede a "massa semântica" de cada token no contexto.
// gravity(t) = mean_over_heads( attention_weight(h, t) × ||V_h(t)||₂ )
//
// Tokens com alta gravidade recebem correção pela CfC layer.
// Tokens com baixa gravidade passam direto (sem custo extra).
//
// Integração:
//   1. Após attention softmax: backend.gravity_score(attn_weights, v_norms)
//   2. Router: gravity > threshold → CfC correction
//   3. threshold adaptativo via ESN state_norm
// ============================================================================

use std::sync::Arc;
use compute_core::{ComputeBackend, AnyTensor, GravityScoreParams};

/// Configuração do sistema de gravidade semântica.
#[derive(Debug, Clone)]
pub struct GravityConfig {
    /// Threshold base de gravidade (tokens acima são corrigidos).
    /// Ajustado adaptativamente pelo ESN.
    pub base_threshold: f32,
    /// Fração máxima de tokens a corrigir por step (controle de custo).
    pub max_correction_ratio: f32,
    /// Se true, calcula gravity mas não aplica correção (modo observação).
    #[allow(dead_code)]
    pub observe_only: bool,
}

impl Default for GravityConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.5,
            max_correction_ratio: 0.3,  // máximo 30% dos tokens corrigidos
            observe_only: false,
        }
    }
}

/// Resultado do cálculo de gravidade para um step de geração.
#[derive(Debug, Clone)]
pub struct GravityResult {
    /// Scores de gravidade por token [seq_len] (na CPU após download).
    #[allow(dead_code)]
    pub scores: Vec<f32>,
    /// Índices dos tokens que excedem o threshold (candidatos a correção).
    pub high_gravity_tokens: Vec<usize>,
    /// Threshold efetivo usado neste step.
    pub effective_threshold: f32,
    /// Média de gravidade (telemetria).
    pub mean_gravity: f32,
    /// Máximo de gravidade (telemetria).
    pub max_gravity: f32,
}

/// Sistema de Gravidade Semântica.
pub struct GravitySystem {
    pub config: GravityConfig,
    /// Buffer GPU para scores de gravidade [max_seq_len].
    pub(crate) gravity_buf: AnyTensor,
    /// Buffer GPU para normas dos value vectors [n_kv_heads, max_seq_len].
    pub(crate) vnorm_buf: AnyTensor,
    /// Histórico de gravidade média (para adaptação do threshold).
    pub(crate) history: Vec<f32>,
    #[allow(dead_code)]
    pub(crate) max_seq_len: usize,
    pub(crate) n_kv_heads: usize,
    head_dim: usize,
}

impl GravitySystem {
    /// Cria o sistema de gravidade. Aloca buffers GPU.
    ///
    /// `backend` — backend de compute (ROCm, OpenCL, etc.)
    /// `max_seq_len` — tamanho máximo da sequência (para alocação de buffer)
    /// `n_kv_heads` — número de cabeçotes de atenção com chave-valor
    /// `head_dim` — dimensão de cada cabeçote
    /// `config` — configuração de gravidade (threshold, max_correction_ratio, etc.)
    pub fn new(
        backend: &Arc<dyn ComputeBackend>,
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        config: GravityConfig,
    ) -> Self {
        let gravity_buf = backend.alloc_f32(&[max_seq_len]);
        let vnorm_buf = backend.alloc_f32(&[n_kv_heads, max_seq_len]);
        Self {
            config,
            gravity_buf,
            vnorm_buf,
            history: Vec::new(),
            max_seq_len,
            n_kv_heads,
            head_dim,
        }
    }

    /// Calcula gravity scores via GPU.
    ///
    /// Chamado APÓS softmax da atenção, ANTES da multiplicação por V.
    /// Usa o kernel `gravity_score` do backend para computar:
    ///   gravity[t] = mean_over_h(attn_weight[h, t] × ||v[h, t]||₂)
    ///
    /// `attn_weights` — [n_heads, seq_len] pesos de atenção pós-softmax
    /// `v_cache` — cache de valores [seq_len, kv_dim]
    /// `seq_len` — tokens válidos no cache
    /// `n_heads` — número total de cabeçotes (incluindo repetições em MQA)
    ///
    /// Retorna `GravityResult` com scores, tokens acima do threshold, e estatísticas.
    pub fn compute(
        &mut self,
        backend: &Arc<dyn ComputeBackend>,
        attn_weights: &AnyTensor,
        v_cache: &AnyTensor,
        seq_len: usize,
        n_heads: usize,
    ) -> Result<GravityResult, compute_core::BackendError> {
        // Calcula pesos de gravidade via backend (kernel specializado)
        // O kernel computa:
        //   1. value_norms[h, t] = ||V[h, t]||₂  (via value_norms kernel)
        //   2. gravity_out[t] = Σ_h (attn_weights[h, t] * value_norms[h, t]) / n_heads
        let params = GravityScoreParams {
            attn_weights,
            v_cache,
            value_norms: &self.vnorm_buf,
            gravity_out: &self.gravity_buf,
            n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            seq_len,
        };
        backend.gravity_score(&params)?;

        // Download scores para CPU
        let scores = backend.download_f32(&self.gravity_buf);
        let scores = scores[..seq_len].to_vec();

        // Estatísticas
        let mean_gravity = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };
        let max_gravity = scores.iter().cloned().fold(0f32, f32::max);

        // Threshold adaptativo: base + ajuste histórico
        let effective_threshold = self.adaptive_threshold(mean_gravity);

        // Tokens acima do threshold
        let mut high_gravity_tokens: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, &g)| g > effective_threshold)
            .map(|(i, _)| i)
            .collect();

        // Limita a max_correction_ratio para controlar custo computacional
        let max_correctable = (seq_len as f32 * self.config.max_correction_ratio) as usize;
        if high_gravity_tokens.len() > max_correctable {
            // Mantém os top-N por gravity score (mais importantes primeiro)
            high_gravity_tokens.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            high_gravity_tokens.truncate(max_correctable);
        }

        self.history.push(mean_gravity);

        Ok(GravityResult {
            scores,
            high_gravity_tokens,
            effective_threshold,
            mean_gravity,
            max_gravity,
        })
    }

    /// Threshold adaptativo baseado no histórico de gravidade.
    ///
    /// Se a gravidade média atual está acima da histórica:
    ///   - Eleva threshold (mais seletivo: menos tokens corrigidos)
    ///   - Economiza custo, mantém maior precisão
    ///
    /// Se a gravidade atual está abaixo da histórica:
    ///   - Reduz threshold (mais permissivo: mais tokens corrigidos)
    ///   - Aplica correção mais agressiva quando há mais ruído
    fn adaptive_threshold(&self, current_mean: f32) -> f32 {
        if self.history.len() < 3 {
            return self.config.base_threshold;
        }

        // Média móvel das últimas 10 observações
        let window_size = self.history.len().min(10);
        let window: Vec<&f32> = self.history.iter().rev().take(window_size).collect();
        let historical_mean = window.iter().copied().sum::<f32>() / window.len() as f32;

        // Razão da média atual pela histórica
        let ratio = current_mean / (historical_mean + 1e-6);

        // Ajusta threshold por essa razão
        (self.config.base_threshold * ratio).clamp(0.1, 0.9)
    }

    /// Número de observações no histórico.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Threshold efetivo (baseado na última média, ou base_threshold se vazio).
    pub fn current_threshold(&self) -> f32 {
        if self.history.len() < 3 {
            self.config.base_threshold
        } else {
            let window_size = self.history.len().min(10);
            let historical_mean: f32 = self.history.iter().rev()
                .take(window_size).sum::<f32>() / window_size as f32;
            (self.config.base_threshold * (historical_mean / (historical_mean + 1e-6)))
                .clamp(0.1, 0.9)
        }
    }

    /// Reset para nova geração.
    ///
    /// Deve ser chamado no início de cada geração de tokens,
    /// para limpar o histórico de gravidade.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

