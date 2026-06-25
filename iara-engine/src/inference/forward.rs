// ============================================================================
// inference/forward.rs — Forward pass GPU-only com telemetria básica
// ============================================================================
// Delega o compute inteiro para model.forward() (hipBLAS + kernels HIP).
// Telemetria por camada é lightweight: timing do forward total distribuído
// uniformemente entre camadas. Ativações detalhadas por camada requerem
// kernels de redução HIP (fase futura).
// ============================================================================

use crate::model::{CausalLM, ModelResult};
use crate::inference::sampling::{SamplingParams, sample_token, sample_token_buffered, SamplingBuffer};
use crate::tokenizer;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ── Telemetria por camada ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTrace {
    pub layer_idx: usize,
    /// Energia de atenção (placeholder: será populado por kernels de redução futuros)
    pub attention_energy: f32,
    /// Energia do gate MLP (placeholder)
    pub mlp_energy: f32,
    /// Norma residual (placeholder)
    pub residual_norm: f32,
    /// Máximo absoluto de ativação (placeholder)
    pub max_activation: f32,
    /// Top-10 neurônios mais ativos (índices, ordenados por magnitude desc)
    pub hot_neurons: Vec<usize>,
    /// Magnitude |ativação| de cada hot neuron, paralelo a `hot_neurons`.
    #[serde(default)]
    pub hot_neuron_values: Vec<f32>,
    /// Microsegundos estimados nesta camada (forward total / num_layers)
    pub time_us: u128,
    /// Kuramoto phase coherence R ∈ [0,1] computed from hidden state
    pub phase_coherence: f32,
    /// Fraction of MLP neurons with |activation| > threshold (0.0-100.0)
    pub mlp_activation_rate: f32,
}

/// Resultado completo de um step de geração com telemetria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStep {
    pub token_id: u32,
    pub token_str: String,
    /// Top-40 logits (id, valor)
    pub top_logits: Vec<(u32, f32)>,
    /// Trace por camada
    pub layer_traces: Vec<LayerTrace>,
    /// Camadas acima da média de ativação
    pub active_layers: Vec<usize>,
    /// ms total do forward pass
    pub time_ms: u128,
}

// ── Forward com tracing ───────────────────────────────────────────────────────

/// Forward pass GPU-only instrumentado (with reusable sampling buffer).
/// Toda computação é executada em GPU via hipBLAS + kernels HIP.
/// O único to_host() é dos logits finais para sampling.
pub fn forward_traced_buffered(
    model: &mut CausalLM,
    token_ids: &[u32],
    sampling: &SamplingParams,
    layer_budget: Option<crate::lnn::esn::LayerBudget>,
    sampling_buf: &mut SamplingBuffer,
) -> ModelResult<GenerationStep> {
    let total_start = Instant::now();
    let num_layers  = model.cfg.num_layers;

    // GPU forward: embedding → transformer blocks → lm_head → logits [vocab]
    // Returns both logits and per-layer traces with real telemetry
    let (logits, mut layer_traces) = model.forward(token_ids, layer_budget)?;

    let elapsed_us = total_start.elapsed().as_micros();
    let us_per_layer = elapsed_us / num_layers.max(1) as u128;

    // Fill in timing information for all layers
    for trace in &mut layer_traces {
        trace.time_us = us_per_layer;
    }

    // Top-40 logits (telemetria — pré-sampling)
    // Usa select_nth_unstable para O(n) ao invés de O(n log n) sort completo.
    let top_k = 40.min(logits.len());
    let mut idx_logits: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    idx_logits.select_nth_unstable_by(top_k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top_slice = idx_logits[..top_k].to_vec();
    top_slice.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_logits = top_slice;

    // Token amostrado (zero-alloc via reusable buffer)
    let token_id  = sample_token_buffered(&logits, sampling, sampling_buf);
    let token_str = tokenizer::token_id_to_text(token_id);

    let active_layers: Vec<usize> = (0..num_layers).collect(); // todos ativos

    Ok(GenerationStep {
        token_id,
        token_str,
        top_logits,
        layer_traces,
        active_layers,
        time_ms: total_start.elapsed().as_millis(),
    })
}

/// Fast decode step — skips top_logits and layer_traces for maximum throughput.
/// Returns only (token_id, token_str). Avoids 152K Vec allocation per token.
/// When temperature < 0.001 (greedy), uses GPU argmax to skip CPU sampling entirely.
pub fn forward_decode_fast(
    model: &mut CausalLM,
    token_ids: &[u32],
    sampling: &SamplingParams,
    sampling_buf: &mut SamplingBuffer,
) -> ModelResult<(u32, String)> {
    // Greedy mode: use GPU argmax (skip expensive logits download + CPU sampling)
    if sampling.temperature < 0.001 {
        let token_id = model.forward_greedy(token_ids)?;
        let token_str = tokenizer::token_id_to_text(token_id);
        return Ok((token_id, token_str));
    }

    let (logits, _layer_traces) = model.forward(token_ids, None)?;
    let token_id  = sample_token_buffered(&logits, sampling, sampling_buf);
    let token_str = tokenizer::token_id_to_text(token_id);
    Ok((token_id, token_str))
}

/// Legacy wrapper — allocates a SamplingBuffer per call.
/// Use `forward_traced_buffered` for hot decode loops.
pub fn forward_traced(
    model: &mut CausalLM,
    token_ids: &[u32],
    sampling: &SamplingParams,
    layer_budget: Option<crate::lnn::esn::LayerBudget>,
) -> ModelResult<GenerationStep> {
    let mut buf = SamplingBuffer::new();
    forward_traced_buffered(model, token_ids, sampling, layer_budget, &mut buf)
}

