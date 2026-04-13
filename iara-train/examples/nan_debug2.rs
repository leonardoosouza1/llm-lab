//! Detailed NaN analysis — test softmax with large-neg vs -inf mask.
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{Int, TensorData, activation::softmax};
use std::path::Path;

use iara_train::model::config::Qwen2Config;

type B = Wgpu;

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

    let seq_len = 6;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;

    let hidden = model.model.embed_tokens.forward(input_ids);
    let normed = model.model.layers[0].input_layernorm.forward(hidden.clone());
    let q = model.model.layers[0].self_attn.q_proj.forward(normed.clone());
    let k = model.model.layers[0].self_attn.k_proj.forward(normed.clone());
    let [batch, _, _] = q.dims();

    let q4 = q.reshape([batch, seq_len, num_heads, head_dim]).swap_dims(1, 2);
    let k4 = k.reshape([batch, seq_len, num_kv_heads, head_dim]).swap_dims(1, 2);

    let q_rope = model.model.layers[0].self_attn.rotary.forward(q4);
    let k_rope = model.model.layers[0].self_attn.rotary.forward(k4);

    let n_rep = num_heads / num_kv_heads;
    let k_exp = k_rope
        .unsqueeze_dim::<5>(2)
        .expand([batch, num_kv_heads, n_rep, seq_len, head_dim])
        .reshape([batch, num_heads, seq_len, head_dim]);

    let scale = (head_dim as f64).sqrt();
    let attn_w = q_rope.matmul(k_exp.swap_dims(2, 3)).div_scalar(scale);

    // Raw attention weights
    let raw_data = attn_w.clone().into_data();
    let raw_vals = raw_data.as_slice::<f32>().unwrap();
    let has_nan_raw = raw_vals.iter().any(|v| v.is_nan());
    let has_inf_raw = raw_vals.iter().any(|v| v.is_infinite());
    println!("attn_w_raw: nan={has_nan_raw} inf={has_inf_raw} max={:.3} min={:.3}",
        raw_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        raw_vals.iter().cloned().fold(f32::INFINITY, f32::min));

    let causal_mask = Tensor::<B, 2, Bool>::triu_mask([seq_len, seq_len], 1, &device);

    // Test with large negative value (avoids NaN from -inf in softmax)
    let attn_large = attn_w.clone().mask_fill(
        causal_mask.clone().unsqueeze_dims::<4>(&[0, 1]),
        -1e4f32,
    );
    let soft_large = softmax(attn_large, 3);
    let large_data = soft_large.into_data();
    let large_vals = large_data.as_slice::<f32>().unwrap();
    let has_nan_large = large_vals.iter().any(|v| v.is_nan());
    println!("softmax with -1e4 mask: nan={has_nan_large}");
    println!("  row [head=0,q=0]: {:?}", &large_vals[0..seq_len]);  // first head, first query

    // Test with -inf
    let attn_inf = attn_w.mask_fill(
        causal_mask.unsqueeze_dims::<4>(&[0, 1]),
        f32::NEG_INFINITY,
    );
    let soft_inf = softmax(attn_inf, 3);
    let inf_data = soft_inf.into_data();
    let inf_vals = inf_data.as_slice::<f32>().unwrap();
    let has_nan_inf = inf_vals.iter().any(|v| v.is_nan());
    let nan_count = inf_vals.iter().filter(|v| v.is_nan()).count();
    println!("softmax with -inf mask: nan={has_nan_inf} count={nan_count}/{}", inf_vals.len());
    println!("  row [head=0,q=0]: {:?}", &inf_vals[0..seq_len]);

    // Test softmax on a simple known-good input with -inf
    let test_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(vec![5.0f32, f32::NEG_INFINITY, f32::NEG_INFINITY], [1, 3]),
        &device,
    );
    let test_soft = softmax(test_tensor, 1);
    let test_data = test_soft.into_data();
    let test_vals = test_data.as_slice::<f32>().unwrap();
    println!("\nSimple softmax([5, -inf, -inf]): {:?}", test_vals);
    println!("  (expected: [1.0, 0.0, 0.0], got nan: {})", test_vals.iter().any(|v| v.is_nan()));

    // Also test all -inf
    let test_allinf = Tensor::<B, 2>::from_data(
        TensorData::new(vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY], [1, 3]),
        &device,
    );
    let test_allinf_soft = softmax(test_allinf, 1);
    let allinf_data = test_allinf_soft.into_data();
    let allinf_vals = allinf_data.as_slice::<f32>().unwrap();
    println!("Simple softmax([-inf, -inf, -inf]): {:?}", allinf_vals);
    println!("  (expected: NaN, got nan: {})", allinf_vals.iter().any(|v| v.is_nan()));
}
