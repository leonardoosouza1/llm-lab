//! Diagnose NaN: run each sub-operation of the first decoder layer independently.
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use std::path::Path;

use iara_train::model::config::Qwen2Config;

type B = Wgpu;

fn check<const D: usize>(label: &str, t: &Tensor<B, D>) {
    let data = t.clone().into_data();
    let vals = data.as_slice::<f32>().unwrap();
    let has_nan = vals.iter().any(|v| v.is_nan());
    let has_inf = vals.iter().any(|v| v.is_infinite());
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let n = vals.len();
    println!("{label}: shape={:?} nan={has_nan} inf={has_inf} range=[{min:.4},{max:.4}] n={n}", t.shape().dims::<D>());
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let device = burn::backend::wgpu::WgpuDevice::default();

    let model_dir = Path::new("models/qwen2.5-1.5b");
    let config = Qwen2Config::from_json(&model_dir.join("config.json")).unwrap();
    let model = config
        .load_from_safetensors::<B>(&model_dir.join("model.safetensors"), &device)
        .unwrap();

    let input_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![785i64, 6722, 315, 9625, 374, 12095], [1, 6]),
        &device,
    );

    // === Step 1: Embedding ===
    let hidden = model.model.embed_tokens.forward(input_ids);
    check("embed_out", &hidden);

    // === Step 2: First layer norm ===
    let normed = model.model.layers[0].input_layernorm.forward(hidden.clone());
    check("layer0_norm1_out", &normed);

    // === Step 3: Q projection ===
    let q = model.model.layers[0].self_attn.q_proj.forward(normed.clone());
    check("layer0_q_proj_out", &q);

    let k = model.model.layers[0].self_attn.k_proj.forward(normed.clone());
    check("layer0_k_proj_out", &k);

    let v = model.model.layers[0].self_attn.v_proj.forward(normed.clone());
    check("layer0_v_proj_out", &v);

    // === Step 4: Reshape Q/K/V ===
    let [batch, seq_len, _] = q.dims();
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;

    let q4 = q.reshape([batch, seq_len, num_heads, head_dim]).swap_dims(1, 2);
    check("layer0_q_4d", &q4);

    let k4 = k.reshape([batch, seq_len, num_kv_heads, head_dim]).swap_dims(1, 2);
    check("layer0_k_4d", &k4);

    // === Step 5: RoPE on Q ===
    let q_rope = model.model.layers[0].self_attn.rotary.forward(q4.clone());
    check("layer0_q_rope", &q_rope);

    let k_rope = model.model.layers[0].self_attn.rotary.forward(k4.clone());
    check("layer0_k_rope", &k_rope);

    // === Step 6: GQA repeat K ===
    let n_rep = num_heads / num_kv_heads;
    let k_exp = k_rope
        .unsqueeze_dim::<5>(2)
        .expand([batch, num_kv_heads, n_rep, seq_len, head_dim])
        .reshape([batch, num_heads, seq_len, head_dim]);
    check("layer0_k_expanded", &k_exp);

    // === Step 7: Attention weights ===
    let scale = (head_dim as f64).sqrt();
    let attn_w = q_rope.matmul(k_exp.swap_dims(2, 3)).div_scalar(scale);
    check("layer0_attn_weights_raw", &attn_w);

    // === Step 8: Causal mask ===
    let causal_mask = Tensor::<B, 2, Bool>::triu_mask([seq_len, seq_len], 1, &device);
    let attn_w_masked = attn_w.mask_fill(causal_mask.unsqueeze_dims::<4>(&[0, 1]), f32::NEG_INFINITY);
    check("layer0_attn_weights_masked", &attn_w_masked);

    // === Step 9: Softmax ===
    let attn_soft = burn::tensor::activation::softmax(attn_w_masked, 3);
    check("layer0_attn_softmax", &attn_soft);

    println!("\nDone — NaN first appears after the step above that shows nan=true");
}
