//! Quick test: load model, run forward, check logits are finite.
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use std::path::Path;

use iara_train::model::config::Qwen2Config;

type B = Wgpu;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Load config + weights
    let model_dir = Path::new("models/qwen2.5-1.5b");
    let config = Qwen2Config::from_json(&model_dir.join("config.json")).unwrap();
    let model = config
        .load_from_safetensors::<B>(&model_dir.join("model.safetensors"), &device)
        .unwrap();

    // Create input: just a few token IDs
    let input_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![785i64, 6722, 315, 9625, 374, 12095], [1, 6]),
        &device,
    );
    println!("Input: [1, 6] tokens");

    // Forward pass
    let output = model.forward(input_ids, false);
    let logits = output.logits;
    let [batch, seq, vocab] = logits.dims();
    println!("Logits shape: [{batch}, {seq}, {vocab}]");

    // Check first few logits
    let first_logits = logits.clone().slice([0..1, 0..1, 0..10]);
    let data = first_logits.into_data();
    let vals = data.as_slice::<f32>().unwrap();
    println!("First 10 logits at pos 0: {:?}", vals);

    // Check if finite
    let has_nan = vals.iter().any(|v| v.is_nan());
    let has_inf = vals.iter().any(|v| v.is_infinite());
    println!("Has NaN: {has_nan}, Has Inf: {has_inf}");

    // Argmax of last position (what token would the model predict?)
    let last_logits = logits.slice([0..1, (seq-1)..seq, 0..vocab]);
    let last_logits_2d = last_logits.reshape([1, vocab]);
    let pred_token = last_logits_2d.argmax(1);
    let pred_data = pred_token.into_data();
    // argmax returns i32 on WGPU
    let pred_vals = pred_data.as_slice::<i32>().unwrap();
    println!("Predicted next token ID: {}", pred_vals[0]);

    if has_nan {
        println!("\nERROR: Forward pass produces NaN!");
    } else {
        println!("\nForward pass OK — logits are finite.");
    }
}
