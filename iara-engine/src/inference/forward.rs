// ============================================================================
// inference/forward.rs — Forward pass GPU-only com telemetria básica
// ============================================================================
// Delega o compute inteiro para model.forward() (hipBLAS + kernels HIP).
// Telemetria por camada é lightweight: timing do forward total distribuído
// uniformemente entre camadas. Ativações detalhadas por camada requerem
// kernels de redução HIP (fase futura).
// ============================================================================

use crate::model::{QwenModel, ModelResult};
use crate::inference::sampling::{SamplingParams, sample_token};
use crate::tokenizer;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ── Telemetria por camada ─────────────────────────────────────────────────────

/// Alias para compatibilidade com debug/brain.rs e api/handlers.rs
pub type LayerActivationTrace = LayerTrace;

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
    /// Top-10 neurônios mais ativos (placeholder)
    pub hot_neurons: Vec<usize>,
    /// Microsegundos estimados nesta camada (forward total / num_layers)
    pub time_us: u128,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActivationPattern {
    pub peak_energy: f32,
    pub mean_energy: f32,
    pub early_focus: bool,
    pub mid_focus: bool,
    pub late_focus: bool,
    pub total_us: u128,
}

// ── Forward com tracing ───────────────────────────────────────────────────────

/// Forward pass GPU-only instrumentado.
/// Toda computação é executada em GPU via hipBLAS + kernels HIP.
/// O único to_host() é dos logits finais para sampling.
pub fn forward_traced(
    model: &mut QwenModel,
    token_ids: &[u32],
    sampling: &SamplingParams,
) -> ModelResult<GenerationStep> {
    let total_start = Instant::now();
    let num_layers  = model.cfg.num_layers;

    // GPU forward: embedding → transformer blocks → lm_head → logits [vocab]
    let logits = model.forward(token_ids)?;

    let elapsed_us = total_start.elapsed().as_micros();
    let us_per_layer = elapsed_us / num_layers.max(1) as u128;

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

    // Token amostrado
    let token_id  = sample_token(&logits, sampling);
    let token_str = tokenizer::token_id_to_text(token_id);

    // Telemetria de camadas: distribui o tempo uniformemente.
    // Energias serão preenchidas por kernels de redução na próxima fase.
    let layer_traces: Vec<LayerTrace> = (0..num_layers).map(|i| LayerTrace {
        layer_idx:        i,
        attention_energy: 0.0,
        mlp_energy:       0.0,
        residual_norm:    0.0,
        max_activation:   0.0,
        hot_neurons:      vec![],
        time_us:          us_per_layer,
    }).collect();

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

// ── Análise de padrões ────────────────────────────────────────────────────────

pub fn analyze_pattern(traces: &[LayerTrace]) -> ActivationPattern {
    if traces.is_empty() { return ActivationPattern::default(); }
    let energies: Vec<f32> = traces.iter().map(|t| t.max_activation).collect();
    let mean = energies.iter().sum::<f32>() / energies.len() as f32;
    let peak = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let n = traces.len();
    let third = n / 3;
    let e = if third > 0 { energies[..third].iter().sum::<f32>() / third as f32 } else { 0.0 };
    let m = if third > 0 { energies[third..2*third].iter().sum::<f32>() / third as f32 } else { 0.0 };
    let l = if n > 2*third { energies[2*third..].iter().sum::<f32>() / (n - 2*third) as f32 } else { 0.0 };
    ActivationPattern {
        peak_energy: peak,
        mean_energy: mean,
        early_focus: e > m && e > l,
        mid_focus:   m > e && m > l,
        late_focus:  l > e && l > m,
        total_us: traces.iter().map(|t| t.time_us).sum(),
    }
}

pub fn layer_importance(traces: &[LayerTrace]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = traces.iter()
        .map(|t| (t.layer_idx, t.max_activation)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}
