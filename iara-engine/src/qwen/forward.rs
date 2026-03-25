// ============================================================================
// qwen/forward.rs — Forward Pass with Layer Activation Tracing
// ============================================================================
// Extends QwenModel to track activations at each layer
// Returns GenerationStep with full telemetry for BrainTrace
// ============================================================================

use candle_core::{Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Telemetry Types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationTrace {
    /// Index of the layer (0-35 for Qwen 8B)
    pub layer_idx: usize,

    /// Sum of attention weights (indicates attention intensity)
    pub attention_energy: f32,

    /// Gate output magnitude from MLP
    pub mlp_energy: f32,

    /// Norm of output after residual connection
    pub residual_norm: f32,

    /// Peak activation value in the layer
    pub max_activation: f32,

    /// Time spent in this layer (microseconds)
    pub computation_time_us: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStep {
    /// The generated token ID
    pub token_id: u32,

    /// The generated token as string
    pub token_str: String,

    /// Top-K logits (sparse representation)
    pub logits: Vec<f32>,

    /// Attention weights from last layer
    pub attention_weights: Vec<f32>,

    /// Per-layer activation traces
    pub layer_traces: Vec<LayerActivationTrace>,

    /// Total time for forward pass (milliseconds)
    pub generation_time_ms: u128,

    /// Which layers had significant activation (above threshold)
    pub active_layers: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core Function: Forward with Tracing
// ─────────────────────────────────────────────────────────────────────────────

pub async fn forward_and_trace(
    model: &crate::model::QwenModel,
    token_id: u32,
    position: usize,
) -> Result<GenerationStep, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let device = &model.device;

    // 1. Embed token
    let mut hidden = model
        .embed_tokens
        .get(token_id as usize)?
        .unsqueeze(0)?; // [1, 2048]

    let mut layer_traces = Vec::new();
    let mut last_attention = None;

    // 2. Forward through layers with tracing
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let layer_start = Instant::now();

        // Normalize input
        let hidden_norm = layer.input_layernorm.forward(&hidden)?;

        // ──────────────────────────────────────────────────────────────────
        // ATTENTION
        // ──────────────────────────────────────────────────────────────────
        let attn_q = layer.attention_wq.forward(&hidden_norm)?;
        let attn_k = layer.attention_wk.forward(&hidden_norm)?;
        let attn_v = layer.attention_wv.forward(&hidden_norm)?;

        // Compute attention energy (sum of attention weights)
        let attention_scores = (attn_q.matmul(&attn_k.transpose(1, 2)?)?)?;
        let attention_energy = attention_scores
            .pow(&Tensor::new(&[2.0], device)?)?
            .sum_all()?
            .to_scalar::<f32>()?;

        // Apply softmax and aggregate attention
        let attention_weights = candle_nn::ops::softmax(&attention_scores, 2)?;
        last_attention = Some(attention_weights.clone());

        let attn_out = attention_weights.matmul(&attn_v)?;
        let attn_out = layer.attention_wo.forward(&attn_out)?;

        // ──────────────────────────────────────────────────────────────────
        // MLP
        // ──────────────────────────────────────────────────────────────────
        let mlp_norm = layer.post_attention_layernorm.forward(&hidden)?;

        let gate = layer.mlp_gate.forward(&mlp_norm)?;
        let up = layer.mlp_up.forward(&mlp_norm)?;

        // Gate magnitude (measure of MLP intensity)
        let mlp_energy = gate
            .pow(&Tensor::new(&[2.0], device)?)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let mlp_out = (gate.silu()? * up)?;
        let mlp_out = layer.mlp_down.forward(&mlp_out)?;

        // ──────────────────────────────────────────────────────────────────
        // RESIDUAL + NORM
        // ──────────────────────────────────────────────────────────────────
        hidden = (hidden + attn_out + mlp_out)?;

        // Residual norm (measure of information flow)
        let residual_norm = hidden
            .pow(&Tensor::new(&[2.0], device)?)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let max_activation = attention_energy.max(mlp_energy).max(residual_norm);

        layer_traces.push(LayerActivationTrace {
            layer_idx,
            attention_energy,
            mlp_energy,
            residual_norm,
            max_activation,
            computation_time_us: layer_start.elapsed().as_micros(),
        });
    }

    // 3. Final normalization
    let hidden_final = model.norm.forward(&hidden)?;

    // 4. Project to logits
    let logits_tensor = model.lm_head.forward(&hidden_final)?;
    let logits_full: Vec<f32> = logits_tensor.squeeze(0)?.to_vec1()?;

    // 5. Extract top-K logits (sparse representation)
    let mut logits_indexed: Vec<(usize, f32)> = logits_full
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    logits_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k_logits: Vec<f32> = logits_indexed
        .iter()
        .take(40)
        .map(|(_, v)| *v)
        .collect();

    // 6. Get most likely token
    let token_id = logits_indexed[0].0 as u32;
    let token_str = model.tokenizer.decode(&[token_id])?;

    // 7. Identify "active" layers (above threshold)
    let threshold = 0.5 * layer_traces.iter().map(|t| t.max_activation).sum::<f32>()
        / layer_traces.len() as f32;
    let active_layers: Vec<usize> = layer_traces
        .iter()
        .filter(|t| t.max_activation > threshold)
        .map(|t| t.layer_idx)
        .collect();

    let generation_time_ms = start.elapsed().as_millis();

    Ok(GenerationStep {
        token_id,
        token_str,
        logits: top_k_logits,
        attention_weights: last_attention
            .map(|t| t.to_vec1().unwrap_or_default())
            .unwrap_or_default(),
        layer_traces,
        generation_time_ms,
        active_layers,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Analyze Layer Activation Pattern
// ─────────────────────────────────────────────────────────────────────────────

pub fn analyze_activation_pattern(traces: &[LayerActivationTrace]) -> ActivationPattern {
    if traces.is_empty() {
        return ActivationPattern::default();
    }

    let max_energy = traces.iter().map(|t| t.max_activation).fold(f32::NEG_INFINITY, f32::max);
    let mean_energy = traces.iter().map(|t| t.max_activation).sum::<f32>() / traces.len() as f32;

    let early_layers = &traces[..traces.len() / 3];
    let mid_layers = &traces[traces.len() / 3..2 * traces.len() / 3];
    let late_layers = &traces[2 * traces.len() / 3..];

    let early_energy = early_layers.iter().map(|t| t.max_activation).sum::<f32>() / early_layers.len() as f32;
    let mid_energy = mid_layers.iter().map(|t| t.max_activation).sum::<f32>() / mid_layers.len() as f32;
    let late_energy = late_layers.iter().map(|t| t.max_activation).sum::<f32>() / late_layers.len() as f32;

    ActivationPattern {
        peak_energy: max_energy,
        mean_energy,
        early_focus: early_energy > mid_energy && early_energy > late_energy,
        mid_focus: mid_energy > early_energy && mid_energy > late_energy,
        late_focus: late_energy > early_energy && late_energy > mid_energy,
        total_computation_us: traces.iter().map(|t| t.computation_time_us).sum(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ActivationPattern {
    pub peak_energy: f32,
    pub mean_energy: f32,
    pub early_focus: bool,
    pub mid_focus: bool,
    pub late_focus: bool,
    pub total_computation_us: u128,
}

impl Default for ActivationPattern {
    fn default() -> Self {
        Self {
            peak_energy: 0.0,
            mean_energy: 0.0,
            early_focus: false,
            mid_focus: false,
            late_focus: false,
            total_computation_us: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Compute Layer Importance
// ─────────────────────────────────────────────────────────────────────────────

pub fn layer_importance_ranking(
    traces: &[LayerActivationTrace],
) -> Vec<(usize, f32)> {
    let mut ranked: Vec<_> = traces
        .iter()
        .map(|t| (t.layer_idx, t.max_activation))
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}
