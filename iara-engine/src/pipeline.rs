// ============================================================================
// pipeline.rs — Pipeline Qwen3 correto: embed → RMSNorm → Attention → MLP → decode
// ============================================================================
//
// Fluxo correto por layer (arquitetura Qwen/Llama pré-norm):
//
//   normed_attn  = RMSNorm(activation, input_layernorm.weight)
//   attn_delta   = Attention(normed_attn)               ← Q/K/V/O proj + RoPE
//   activation   = activation + attn_delta              ← residual aditivo
//
//   normed_mlp   = RMSNorm(activation, post_attn_layernorm.weight)
//   mlp_delta    = SiLU_MLP(normed_mlp)                 ← gate*up+silu+down
//   activation   = activation + mlp_delta               ← residual aditivo
//
// Os três bugs originais corrigidos:
//   1. RMSNorm agora é puro (sem CfC nos layernorm weights)
//   2. Residual é soma aditiva (não weighted blend)
//   3. MLP retorna apenas delta (sem CfC extra e sem RMSNorm pós-MLP)
// ============================================================================

use crate::decoder;
use crate::embedding;
use crate::safetensors_reader;
use crate::tokenizer;
use rayon::prelude::*;
use serde::Serialize;
use std::path::Path;
use tokio::sync::{broadcast, mpsc};


/// Resultado do pipeline completo
#[derive(Debug, Clone, Serialize)]
pub struct PipelineResult {
    pub prompt: String,
    pub input_tokens: Vec<String>,
    pub layers_processed: usize,
    pub top_tokens: Vec<TokenResult>,
    pub total_time_ms: u128,
}

/// Um token decodificado com score
#[derive(Debug, Clone, Serialize)]
pub struct TokenResult {
    pub token_id: u32,
    pub score: f32,
    pub text: String,
}

/// Frame de progresso enviado para o frontend durante a pipeline
#[derive(Debug, Serialize)]
pub struct PipelineProgress {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub layer_idx: usize,
    pub total_layers: usize,
    pub sublayer: String,
    pub grid_size: usize,
    pub data: Vec<f32>,
    pub energy: f32,
    pub spikes: usize,
    pub phase: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// § Pipeline Principal
// ─────────────────────────────────────────────────────────────────────────────

/// Executa o pipeline completo de inferência Qwen3
pub async fn run_pipeline(
    prompt: &str,
    model_dir: &Path,
    index: &safetensors_reader::SafetensorsIndex,
    frame_tx: &broadcast::Sender<String>,
    stimulus_tx: Option<&mpsc::Sender<(usize, Vec<f32>)>>,
    grid_size: usize,
    _device: &candle_core::Device,
) -> Result<PipelineResult, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = std::time::Instant::now();

    // ── 1. Tokenizar ──────────────────────────────────────────────────────────
    let tokens = tokenizer::tokenize(prompt);
    let token_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let token_texts: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
    send_status(frame_tx, &format!("🔤 Tokenizado: {} tokens", tokens.len()));

    // ── 2. Embeddings ─────────────────────────────────────────────────────────
    send_status(frame_tx, "📐 Carregando embeddings...");
    let embed_result = embedding::read_embedding_rows(
        model_dir, index, "model.embed_tokens.weight", &token_ids,
    )
    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    })?;

    // mean embedding (sem L2 norm — preserva magnitude)
    let mut activation = embedding::combine_embeddings(&embed_result);
    let hidden_dim = embed_result.hidden_dim;

    log::info!(
        "📐 Embedding: dim={}, primeiros valores: [{:.4}, {:.4}, {:.4}...]",
        hidden_dim,
        activation.get(0).copied().unwrap_or(0.0),
        activation.get(1).copied().unwrap_or(0.0),
        activation.get(2).copied().unwrap_or(0.0),
    );

    // ── 3. Detectar número de layers ──────────────────────────────────────────
    let mut num_layers = 0;
    while index.weight_map.contains_key(
        &format!("model.layers.{}.input_layernorm.weight", num_layers)
    ) {
        num_layers += 1;
    }
    if num_layers == 0 {
        return Err("Nenhuma camada detectada no index".into());
    }
    log::info!("🧬 Detectadas {} camadas", num_layers);
    send_status(frame_tx, &format!("🧬 Propagando por {} camadas...", num_layers));

    // ── 4. Loop de layers ─────────────────────────────────────────────────────
    for layer_idx in 0..num_layers {
        // 4a. PRÉ-ATENÇÃO: RMSNorm puro
        let ln_in = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let normed_attn = apply_rmsnorm(model_dir, index, &ln_in, &activation)
            .map_err(io_err)?;

        // 4b. Injetar normed na Fauna Digital (Interceptação por Camada)
        if let Some(st_tx) = stimulus_tx {
            // Enviamos um par (layer_idx, activation) para o orquestrador
            // Para simplicidade, usaremos uma mensagem especial se necessário, 
            // mas aqui apenas enviamos o clone para o canal mpsc.
            // O main.rs precisará saber qual layer é qual.
            let _ = st_tx.send((layer_idx, normed_attn.clone())).await;
        }

        // 4c. ATENÇÃO: retorna apenas delta (sem residual interno)
        let attn_delta = compute_attention_delta(
            model_dir, index, layer_idx, &normed_attn,
            frame_tx, grid_size,
        )
        .await
        .map_err(io_err)?;

        // 4d. Residual aditivo (verdadeiro)
        for (a, d) in activation.iter_mut().zip(attn_delta.iter()) {
            *a += d;
        }

        // 4e. PRÉ-MLP: RMSNorm puro
        let ln_post = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        let normed_mlp = apply_rmsnorm(model_dir, index, &ln_post, &activation)
            .map_err(io_err)?;

        // 4f. MLP: retorna apenas delta
        let mlp_delta = compute_mlp_delta(
            model_dir, index, layer_idx, &normed_mlp,
            frame_tx, grid_size,
        )
        .await
        .map_err(io_err)?;

        // 4g. Residual aditivo
        for (a, d) in activation.iter_mut().zip(mlp_delta.iter()) {
            *a += d;
        }

        // 4h. Injetar ativação pós-layer na Fauna (Interceptação Real)
        if let Some(st_tx) = stimulus_tx {
            let _ = st_tx.send((layer_idx, activation.clone())).await;
        }

        if layer_idx % 4 == 0 || layer_idx == num_layers - 1 {
            let norm: f32 = activation.iter().map(|v| v * v).sum::<f32>().sqrt();
            log::info!("📊 Layer {}/{}: activation norm={:.4}", layer_idx + 1, num_layers, norm);
        }
    }

    // ── 5. Final RMSNorm ──────────────────────────────────────────────────────
    send_status(frame_tx, "📐 Aplicando norma final...");
    activation = apply_rmsnorm(model_dir, index, "model.norm.weight", &activation)
        .map_err(io_err)?;

    // ── 6. Decodificar via lm_head ────────────────────────────────────────────
    send_status(frame_tx, "🔍 Decodificando via lm_head...");
    let decoded = decoder::decode_activation(model_dir, index, &activation, 50)
        .map_err(io_err)?;

    let top_tokens: Vec<TokenResult> = decoded.into_iter().map(|d| TokenResult {
        token_id: d.token_id, score: d.score, text: d.text,
    }).collect();

    let total_time = start_time.elapsed().as_millis();
    log::info!(
        "✅ Pipeline completo em {}ms — top token: {} (score={:.4})",
        total_time,
        top_tokens.first().map(|t| t.text.as_str()).unwrap_or("?"),
        top_tokens.first().map(|t| t.score).unwrap_or(0.0),
    );

    Ok(PipelineResult {
        prompt: prompt.to_string(),
        input_tokens: token_texts,
        layers_processed: num_layers,
        top_tokens,
        total_time_ms: total_time,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// § RMSNorm puro (sem CfC)
// ─────────────────────────────────────────────────────────────────────────────

/// Aplica RMSNorm correto: y[i] = (x[i] / rms(x)) * gamma[i]
///
/// Esta é a operação exata que Qwen/Llama usa em input_layernorm e
/// post_attention_layernorm. Sem CfC, sem residual, sem nada além da norma.
fn apply_rmsnorm(
    model_dir: &Path,
    index: &safetensors_reader::SafetensorsIndex,
    tensor_name: &str,
    input: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let td = safetensors_reader::read_tensor(model_dir, index, tensor_name)?;
    let gamma = &td.data;

    if gamma.len() != input.len() {
        return Err(format!(
            "RMSNorm dim mismatch: gamma={} vs input={}",
            gamma.len(), input.len()
        ).into());
    }

    let eps = 1e-6_f32;
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let rrms = 1.0 / (mean_sq + eps).sqrt();

    let out: Vec<f32> = input.iter().zip(gamma.iter())
        .map(|(&x, &g)| x * rrms * g)
        .collect();

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// § Atenção: retorna delta (sem residual interno)
// ─────────────────────────────────────────────────────────────────────────────

/// Calcula o delta de atenção (o_proj output) para um único token.
///
/// Para input de um único token, a atenção colapsa trivialmente:
///   - softmax sobre uma única posição = [1.0] por head
///   - attn_out[head] = V[kv_head]    ← sem ponderação complexa
///   - output = O_proj(expand_kv(V))
///
/// Retorna APENAS o delta para ser somado residualmente pelo chamador.
async fn compute_attention_delta(
    model_dir: &Path,
    index: &safetensors_reader::SafetensorsIndex,
    layer_idx: usize,
    normed_input: &[f32],
    frame_tx: &broadcast::Sender<String>,
    grid_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let q_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
    let k_name = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
    let v_name = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx);
    let o_name = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx);

    let q_tensor = safetensors_reader::read_tensor(model_dir, index, &q_name)?;
    let k_tensor = safetensors_reader::read_tensor(model_dir, index, &k_name)?;
    let v_tensor = safetensors_reader::read_tensor(model_dir, index, &v_name)?;
    let o_tensor = safetensors_reader::read_tensor(model_dir, index, &o_name)?;

    let hidden_dim = normed_input.len();
    let out_q = q_tensor.data.len() / hidden_dim; // num_heads * head_dim = 4096
    let out_k = k_tensor.data.len() / hidden_dim; // num_kv_heads * head_dim = 1024
    let out_v = v_tensor.data.len() / hidden_dim; // num_kv_heads * head_dim = 1024

    let num_heads: usize = 32;
    let head_dim: usize = 128;
    let num_kv_heads: usize = 8;
    let rope_theta: f32 = 1_000_000.0;
    let token_pos: usize = 0;

    // ── Projeções Q, K, V ───────────────────────────────────────────────────
    let mut q = vec![0.0f32; out_q];
    let mut k = vec![0.0f32; out_k];
    let mut v = vec![0.0f32; out_v];

    q.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row = i * hidden_dim;
        for j in 0..hidden_dim { *val += q_tensor.data[row + j] * normed_input[j]; }
    });
    k.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row = i * hidden_dim;
        for j in 0..hidden_dim { *val += k_tensor.data[row + j] * normed_input[j]; }
    });
    v.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row = i * hidden_dim;
        for j in 0..hidden_dim { *val += v_tensor.data[row + j] * normed_input[j]; }
    });

    // ── RoPE ────────────────────────────────────────────────────────────────
    apply_rope(&mut q, token_pos, head_dim, rope_theta);
    apply_rope(&mut k, token_pos, head_dim, rope_theta);

    // ── Atenção para single-token (GQA) ─────────────────────────────────────
    // Com uma única posição na sequência, softmax([score]) = [1.0] por head.
    // Logo: attn_out[h] = V[kv_head(h)] — a atenção é identidade sobre V.
    // Expandimos cada KV head para os Q heads correspondentes.
    let ratio = num_heads / num_kv_heads; // = 4
    let mut attn_out = vec![0.0f32; out_q]; // [num_heads * head_dim]

    for h in 0..num_heads {
        let kv_h = h / ratio;
        let src = kv_h * head_dim;
        let dst = h * head_dim;
        attn_out[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
    }

    // ── O_proj ──────────────────────────────────────────────────────────────
    // o_proj.weight shape: [hidden_dim, out_q]
    let mut output = vec![0.0f32; hidden_dim];
    output.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row = i * out_q;
        let end = (row + out_q).min(o_tensor.data.len());
        let len = end - row;
        if len > 0 {
            *val = o_tensor.data[row..end]
                .iter()
                .zip(attn_out[..len].iter())
                .map(|(&w, &a)| w * a)
                .sum();
        }
    });

    // ── Frame de progresso ───────────────────────────────────────────────────
    let energy: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
    send_progress(frame_tx, layer_idx, 0, "Self-Attention", grid_size, &output, energy,
        &format!("Layer {} — Attention delta", layer_idx + 1));

    tokio::task::yield_now().await;
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// § MLP: retorna delta (sem CfC, sem RMSNorm extra)
// ─────────────────────────────────────────────────────────────────────────────

/// Calcula o delta MLP: SiLU(gate_proj(x)) * up_proj(x) projetado por down_proj.
///
/// Esta é a operação SwiGLU exata usada pelo Qwen3.
/// Retorna APENAS o delta para ser somado residualmente pelo chamador.
async fn compute_mlp_delta(
    model_dir: &Path,
    index: &safetensors_reader::SafetensorsIndex,
    layer_idx: usize,
    normed_input: &[f32],
    frame_tx: &broadcast::Sender<String>,
    grid_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let gate_name = format!("model.layers.{}.mlp.gate_proj.weight", layer_idx);
    let up_name   = format!("model.layers.{}.mlp.up_proj.weight",   layer_idx);
    let down_name = format!("model.layers.{}.mlp.down_proj.weight", layer_idx);

    let gate_tensor = safetensors_reader::read_tensor(model_dir, index, &gate_name)?;
    let up_tensor   = safetensors_reader::read_tensor(model_dir, index, &up_name)?;

    let hidden_dim = normed_input.len();
    let intermediate_size = gate_tensor.data.len() / hidden_dim;

    // ── Gate + Up projections + SiLU ────────────────────────────────────────
    // SwiGLU: intermediate[i] = SiLU(gate[i]) * up[i]
    // SiLU(x) = x * sigmoid(x)
    let mut intermediate = vec![0.0f32; intermediate_size];
    intermediate.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row = i * hidden_dim;
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for j in 0..hidden_dim {
            let w = row + j;
            if w < gate_tensor.data.len() {
                let a = normed_input[j];
                g += gate_tensor.data[w] * a;
                u += up_tensor.data[w] * a;
            }
        }
        let silu_g = g * (1.0 / (1.0 + (-g).exp())); // SiLU(g)
        *val = silu_g * u;
    });

    drop(gate_tensor);
    drop(up_tensor);

    // ── Down projection ──────────────────────────────────────────────────────
    // down_proj.weight shape: [hidden_dim, intermediate_size]
    let down_tensor = safetensors_reader::read_tensor(model_dir, index, &down_name)?;
    let mut output = vec![0.0f32; hidden_dim];
    output.par_iter_mut().enumerate().for_each(|(j, val)| {
        let row = j * intermediate_size;
        let end = (row + intermediate_size).min(down_tensor.data.len());
        if end > row {
            *val = down_tensor.data[row..end]
                .iter()
                .zip(intermediate.iter())
                .map(|(&w, &x)| w * x)
                .sum();
        }
    });

    // ── Frame de progresso ───────────────────────────────────────────────────
    let energy: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
    send_progress(frame_tx, layer_idx, 0, "MLP SwiGLU", grid_size, &output, energy,
        &format!("Layer {} — MLP delta", layer_idx + 1));

    tokio::task::yield_now().await;
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// § Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Aplica RoPE (Rotary Positional Embeddings) ao vetor.
fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let num_heads = vec.len() / head_dim;
    for h in 0..num_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        for i in (0..head_dim).step_by(2) {
            if i + 1 >= head_dim { break; }
            let freq = pos as f32 / theta.powf(i as f32 / head_dim as f32);
            let (sin, cos) = freq.sin_cos();
            let v0 = head[i];
            let v1 = head[i + 1];
            head[i]     = v0 * cos - v1 * sin;
            head[i + 1] = v0 * sin + v1 * cos;
        }
    }
}

/// Converte qualquer Display em Box<dyn Error + Send + Sync>.
fn io_err<E: std::fmt::Display>(e: E) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
}

/// Envia um frame de progresso para o frontend.
fn send_progress(
    frame_tx: &broadcast::Sender<String>,
    layer_idx: usize,
    total_layers: usize,
    sublayer: &str,
    grid_size: usize,
    data: &[f32],
    energy: f32,
    phase: &str,
) {
    let grid = downsample_for_grid(data, grid_size);
    let progress = PipelineProgress {
        msg_type: "pipeline_layer".to_string(),
        layer_idx,
        total_layers,
        sublayer: sublayer.to_string(),
        grid_size,
        data: grid,
        energy,
        spikes: 0,
        phase: phase.to_string(),
    };
    if let Ok(json) = serde_json::to_string(&progress) {
        let _ = frame_tx.send(json);
    }
}

/// Envia mensagem de status textual para o frontend.
fn send_status(frame_tx: &broadcast::Sender<String>, message: &str) {
    let msg = serde_json::json!({ "type": "status", "message": message });
    let _ = frame_tx.send(msg.to_string());
}

/// Faz downsample de um vetor para um grid NxN (para visualização).
fn downsample_for_grid(data: &[f32], grid_size: usize) -> Vec<f32> {
    let total = grid_size * grid_size;
    let n = data.len();
    if n == 0 { return vec![0.0; total]; }
    let mut grid = vec![0.0f32; total];
    let ratio = n as f64 / total as f64;
    if n >= total {
        for c in 0..total {
            let s = (c as f64 * ratio) as usize;
            let e = (((c + 1) as f64 * ratio) as usize).min(n);
            if s < n && e > s {
                let sum: f32 = data[s..e].iter().sum();
                grid[c] = sum / (e - s) as f32;
            }
        }
    } else {
        for c in 0..total {
            let idx = ((c as f64 * ratio) as usize).min(n - 1);
            grid[c] = data[idx];
        }
    }
    grid
}
