// ============================================================================
// qwen/forward.rs — Forward Pass com Tracing completo (Fase 2: Lab de Dissecção)
// ============================================================================
// Cada forward pass captura telemetria completa por camada:
//   - Energia de atenção (mapa de calor)
//   - Energia do MLP (gate intensity)
//   - Norma residual por camada
//   - Ativação máxima (neurônios mais ativos)
//   - Top-K logits por step
//   - Tempo por camada (profiling)
// ============================================================================

use crate::model::{QwenModel, ModelResult, ModelError};
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
    /// Energia de atenção: soma(Q·K^T)² — intensidade do padrão de atenção
    pub attention_energy: f32,
    /// Energia do gate MLP: soma(gate_proj²)
    pub mlp_energy: f32,
    /// Norma L2 do hidden state após residual
    pub residual_norm: f32,
    /// Máximo absoluto de ativação na camada
    pub max_activation: f32,
    /// Top-10 índices mais ativos no hidden state
    pub hot_neurons: Vec<usize>,
    /// Microsegundos gastos nesta camada
    pub time_us: u128,
}

/// Resultado completo de um step de geração com telemetria total.
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

/// Padrão de ativação agregado (para visualização)
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

/// Forward pass instrumentado: retorna geração + telemetria completa.
/// Usado pelo laboratório de dissecção (Fase 2).
pub fn forward_traced(
    model: &mut QwenModel,
    token_ids: &[u32],
    sampling: &SamplingParams,
) -> ModelResult<GenerationStep> {
    let total_start = Instant::now();
    let h = model.cfg.hidden_size;
    let n = token_ids.len();

    // 1. Embedding lookup
    let embed = model.embed.to_host()?;
    let mut x: Vec<f32> = token_ids.iter().flat_map(|&id| {
        let off = id as usize * h;
        embed[off..off + h].to_vec()
    }).collect();

    let mut layer_traces: Vec<LayerTrace> = Vec::with_capacity(model.cfg.num_layers);

    // 2. Transformer blocks com tracing
    for (li, layer) in model.layers.iter_mut().enumerate() {
        let layer_start = Instant::now();

        // RMSNorm
        let normed = rms_norm_host(&x, &layer.input_norm_w, model.cfg.rms_eps, n, h);

        // Projections
        let wq = layer.wq.to_host()?;
        let wk = layer.wk.to_host()?;
        let wv = layer.wv.to_host()?;
        let wo = layer.wo.to_host()?;
        let q_dim = layer.wq.shape[0];
        let kv_dim = layer.wk.shape[0];

        let mut q = gemm_host(&normed, &wq, n, h, q_dim);
        let mut k = gemm_host(&normed, &wk, n, h, kv_dim);
        let v = gemm_host(&normed, &wv, n, h, kv_dim);

        // RoPE: posição absoluta do primeiro token = tokens já em cache
        let pos_start = layer.kv_k.seq_len();
        crate::model::apply_rope(&mut q, n, model.cfg.num_heads,    model.cfg.head_dim, pos_start, model.cfg.rope_theta);
        crate::model::apply_rope(&mut k, n, model.cfg.num_kv_heads, model.cfg.head_dim, pos_start, model.cfg.rope_theta);

        // Atenção — captura energia (no espaço rotacionado)
        let attn_energy = q.iter().map(|&v| v * v).sum::<f32>();

        layer.kv_k.append(&k);
        layer.kv_v.append(&v);
        let seq = layer.kv_k.seq_len();

        let head_dim = model.cfg.head_dim;
        let n_heads  = model.cfg.num_heads;
        let n_kv     = model.cfg.num_kv_heads;
        let scale    = (head_dim as f32).sqrt();
        let mut attn = vec![0.0f32; n * q_dim];
        let kv_k_s = layer.kv_k.as_slice();
        let kv_v_s = layer.kv_v.as_slice();

        for h_i in 0..n_heads {
            let kv_h = h_i * n_kv / n_heads;
            let q_off = h_i * head_dim;
            let k_off = kv_h * head_dim;
            for qi in 0..n {
                let mut scores = vec![0.0f32; seq];
                for ki in 0..seq {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qi * q_dim + q_off + d]
                             * kv_k_s[ki * kv_dim + k_off + d];
                    }
                    scores[ki] = dot / scale;
                }
                softmax_inplace(&mut scores);
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ki in 0..seq { acc += scores[ki] * kv_v_s[ki * kv_dim + k_off + d]; }
                    attn[qi * q_dim + q_off + d] = acc;
                }
            }
        }

        let attn_proj = gemm_host(&attn, &wo, n, q_dim, h);
        for i in 0..n * h { x[i] += attn_proj[i]; }

        // MLP — captura gate energy
        let normed2 = rms_norm_host(&x, &layer.post_attn_norm_w, model.cfg.rms_eps, n, h);
        let wg = layer.wgate.to_host()?;
        let wu = layer.wup.to_host()?;
        let wd = layer.wdown.to_host()?;
        let inter = layer.wgate.shape[0];

        let gate = gemm_host(&normed2, &wg, n, h, inter);
        let up   = gemm_host(&normed2, &wu, n, h, inter);
        let mlp_energy = gate.iter().map(|&v| v * v).sum::<f32>();

        let swiglu: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        let mlp = gemm_host(&swiglu, &wd, n, inter, h);
        for i in 0..n * h { x[i] += mlp[i]; }

        // Métricas do hidden state pós-residual
        let residual_norm = (x.iter().map(|&v| v * v).sum::<f32>() / (n * h) as f32).sqrt();
        let max_act = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max).abs();

        // Top-10 neurônios mais ativos (índices no último token)
        let last_off = (n - 1) * h;
        let mut hot: Vec<(usize, f32)> = x[last_off..last_off + h]
            .iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
        hot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let hot_neurons: Vec<usize> = hot[..10.min(hot.len())].iter().map(|&(i, _)| i).collect();

        layer_traces.push(LayerTrace {
            layer_idx: li,
            attention_energy: attn_energy,
            mlp_energy,
            residual_norm,
            max_activation: max_act,
            hot_neurons,
            time_us: layer_start.elapsed().as_micros(),
        });
    }

    // 3. Final norm + lm_head
    let x_norm = rms_norm_host(&x, &model.norm_w, model.cfg.rms_eps, n, h);
    let last = &x_norm[(n - 1) * h..];
    let lm = model.lm_head.to_host()?;
    let vocab = model.cfg.vocab_size;
    let logits = gemm_host(last, &lm, 1, h, vocab);

    // Top-40 logits (telemetria — sempre dos logits brutos, pré-amostragem)
    let mut idx_logits: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    idx_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_logits: Vec<(u32, f32)> = idx_logits[..40.min(idx_logits.len())].to_vec();

    // Token amostrado — temperatura / top-k / top-p aplicados aqui
    let token_id = sample_token(&logits, sampling);
    let token_str = tokenizer::token_id_to_text(token_id);

    // Camadas acima da média de ativação
    let mean_act = layer_traces.iter().map(|t| t.max_activation).sum::<f32>()
        / layer_traces.len() as f32;
    let active_layers: Vec<usize> = layer_traces.iter()
        .filter(|t| t.max_activation > mean_act)
        .map(|t| t.layer_idx)
        .collect();

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
    let e = energies[..n/3].iter().sum::<f32>() / (n/3) as f32;
    let m = energies[n/3..2*n/3].iter().sum::<f32>() / (n/3) as f32;
    let l = energies[2*n/3..].iter().sum::<f32>() / (n - 2*n/3) as f32;
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
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

// ── Primitivas host ───────────────────────────────────────────────────────────

fn rms_norm_host(x: &[f32], w: &[f32], eps: f32, n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let rms = (row.iter().map(|&v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        for j in 0..d { out[i * d + j] = row[j] / rms * w[j]; }
    }
    out
}

fn gemm_host(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k { acc += a[i * k + p] * b[j * k + p]; }
            c[i * n + j] = acc;
        }
    }
    c
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in x.iter_mut() { *v /= sum; }
}

#[inline(always)]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
