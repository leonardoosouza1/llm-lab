//! Fase 1: Block Influence (BI) Analysis
//!
//! Mede a "influência" de cada layer do Qwen2.5-1.5B calculando:
//!   BI[i] = 1 - cos_sim(layer_i_input, layer_i_output)
//!
//! BI próximo de 0 → layer quase não modifica o hidden state → candidato a merge/remoção.
//! BI próximo de 1 → layer transforma significativamente → deve ser mantido.
//!
//! Referência: ShortGPT (2024) — https://arxiv.org/abs/2403.03853
//!
//! Execução:
//!   cargo run -p iara-train --example bi_analysis --features wgpu

use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use std::path::Path;
use std::time::Instant;

use iara_train::model::config::Qwen2Config;

type B = Wgpu;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Cosine similarity entre dois tensors 3D (batch, seq, hidden) — flattened.
fn cosine_sim_3d(a: Tensor<B, 3>, b: Tensor<B, 3>) -> f32 {
    let a = a.flatten::<1>(0, 2);
    let b = b.flatten::<1>(0, 2);
    let dot = a.clone().mul(b.clone()).sum().into_data();
    let dot = dot.as_slice::<f32>().unwrap()[0];
    let na = a.clone().mul(a).sum().sqrt().into_data();
    let na = na.as_slice::<f32>().unwrap()[0];
    let nb = b.clone().mul(b).sum().sqrt().into_data();
    let nb = nb.as_slice::<f32>().unwrap()[0];
    dot / (na * nb + 1e-8)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let device = burn::backend::wgpu::WgpuDevice::default();

    println!("\n┌─ IARA Fase 1: Block Influence Analysis ────────────────────────────────────────────┐");
    println!("│ BI[i] = 1 - cos_sim(layer_i_input, layer_i_output)                               │");
    println!("│ BI≈0 → layer redundante (candidato a merge)   BI≈1 → layer crítica              │");
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘\n");

    let model_dir = Path::new("models/qwen2.5-1.5b");
    let config = Qwen2Config::from_json(&model_dir.join("config.json")).unwrap();

    println!("Carregando modelo Qwen2.5-1.5B ({} layers)...", config.num_hidden_layers);
    let t_load = Instant::now();
    let model = config
        .load_from_safetensors::<B>(&model_dir.join("model.safetensors"), &device)
        .expect("failed to load model");
    println!("  ✓ carregado em {:.1}s\n", t_load.elapsed().as_secs_f32());

    // Sample inputs: 3 prompts cobrindo domínios diferentes
    // Token IDs fixos para reprodutibilidade (sem tokenizer necessário)
    let prompts: Vec<(&str, Vec<i64>)> = vec![
        ("código",     vec![755i64, 623, 5761, 320, 77, 2599, 1858, 374, 4756]),
        ("matemática", vec![3838, 374, 279, 32726, 315, 865, 6381, 220, 17]),
        ("linguagem",  vec![9887, 757, 279, 3374, 315, 9625, 304, 13435]),
    ];

    let num_layers = config.num_hidden_layers;
    let mut bi_sum = vec![0.0f32; num_layers];

    for (domain, token_ids) in &prompts {
        let seq_len = token_ids.len();
        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(token_ids.clone(), [1, seq_len]),
            &device,
        );

        // Forward com captura de todos os hidden states.
        // Após o patch em qwen2.rs, all_hidden_states tem 29 entradas:
        //   [0] = embedding output (antes de qualquer decoder layer)
        //   [i+1] = saída da decoder layer i  (para i in 0..28)
        let output = model.forward(input_ids, true);
        let states = output
            .hidden_states
            .expect("hidden_states vazio — certifique-se do patch em qwen2.rs");

        assert_eq!(
            states.len(), num_layers + 1,
            "Esperava {} estados (embed + {} layers), obteve {}",
            num_layers + 1, num_layers, states.len()
        );

        for i in 0..num_layers {
            let before = states[i].clone();     // estado antes da layer i
            let after  = states[i + 1].clone(); // estado depois da layer i
            let cos = cosine_sim_3d(before, after);
            let bi = (1.0 - cos).max(0.0); // nunca negativo por erro numérico
            bi_sum[i] += bi;
        }

        println!("  ✓ domínio '{domain}' ({seq_len} tokens) processado");
    }

    let n_prompts = prompts.len() as f32;
    let bi_avg: Vec<f32> = bi_sum.iter().map(|s| s / n_prompts).collect();

    // ─── Tabela de resultados ────────────────────────────────────────────────

    let bi_max = bi_avg.iter().cloned().fold(0.0f32, f32::max);
    let bi_min = bi_avg.iter().cloned().fold(f32::MAX, f32::min);
    let merge_threshold = bi_min + (bi_max - bi_min) * 0.30;

    let mut merge_candidates: Vec<usize> = Vec::new();

    println!("\n{}", "─".repeat(80));
    println!("{:>5}  {:>8}  {:>7}  {:>6}  {}",
        "Layer", "BI Score", "Relat%", "Prio", "Visualização");
    println!("{}", "─".repeat(80));

    for (i, &bi) in bi_avg.iter().enumerate() {
        let relative = if bi_max > 0.0 { bi / bi_max } else { 0.0 };
        let bar_len = (relative * 40.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(40 - bar_len);

        let tag = if bi < merge_threshold {
            merge_candidates.push(i);
            "MERGE"
        } else if relative > 0.70 {
            "KEEP!"
        } else {
            "keep "
        };

        println!("{:>5}  {:>8.5}  {:>6.1}%  {:>5}  {}",
            i, bi, relative * 100.0, tag, bar);
    }
    println!("{}", "─".repeat(80));

    // ─── Sugestão de grupos ──────────────────────────────────────────────────

    println!("\n┌─ Sugestão de 4 Super Layers ───────────────────────────────────────────────────────┐");
    let group_boundaries = [0usize, 7, 14, 21, 28];
    let sl_names = ["SL0 Compreensão", "SL1 Raciocínio", "SL2 Conhecimento", "SL3 Geração"];
    for g in 0..4 {
        let s = group_boundaries[g];
        let e = group_boundaries[g + 1];
        let slice = &bi_avg[s..e];
        let avg = slice.iter().sum::<f32>() / slice.len() as f32;
        let cands: Vec<usize> = slice.iter().enumerate()
            .filter(|(_, &b)| b < merge_threshold)
            .map(|(j, _)| s + j)
            .collect();
        println!("│  {:>18}: layers {:>2}-{:>2}  avgBI={:.5}  candidates_to_merge={:?}",
            sl_names[g], s, e - 1, avg, cands);
    }
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘\n");

    // ─── Sumário ─────────────────────────────────────────────────────────────

    let max_layer = bi_avg.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i);
    let min_layer = bi_avg.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i);

    println!("Sumário:");
    println!("  BI max: {:.5} (layer {:?})", bi_max, max_layer);
    println!("  BI min: {:.5} (layer {:?})", bi_min, min_layer);
    println!("  Threshold merge (<30% do max): {:.5}", merge_threshold);
    println!("  Candidatos a merge: {}/{} = {:.0}% do modelo",
        merge_candidates.len(), num_layers,
        merge_candidates.len() as f32 / num_layers as f32 * 100.0);

    // Salvar JSON para as próximas fases
    std::fs::create_dir_all("data").ok();
    let json = serde_json::json!({
        "bi_scores": bi_avg,
        "bi_max": bi_max,
        "bi_min": bi_min,
        "merge_threshold": merge_threshold,
        "merge_candidates": merge_candidates,
        "super_layer_groups": {
            "SL0": [0, 7],
            "SL1": [7, 14],
            "SL2": [14, 21],
            "SL3": [21, 28]
        }
    });
    std::fs::write(
        "data/bi_scores.json",
        serde_json::to_string_pretty(&json).unwrap(),
    ).expect("failed to write bi_scores.json");

    println!("\n✓ Scores salvos → data/bi_scores.json");
    println!("→ Próximo: cargo run -p iara-train --example lsh_lookup --features wgpu");
}
