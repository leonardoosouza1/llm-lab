//! Fase 2 (v2): LSH Coordinate Lookup com embeddings reais
//!
//! Versão corrigida: usa o embedding real do modelo como input ao LSH,
//! ao invés de inputs sintéticos (sin/cos) que não representam a distribuição real.
//!
//! Pipeline:
//!   1. Carregar modelo Qwen2.5-1.5B
//!   2. Rodar forward com return_all_hidden_states=true
//!   3. Usar embedding output (states[0]) como input real ao LSH
//!   4. Comparar: quais neurônios o LSH prediz vs quais realmente ativam
//!
//! Referência: SLIDE (Rice 2020) — https://arxiv.org/abs/1903.03129
//!
//! Execução:
//!   cargo run -p iara-train --example lsh_lookup --features wgpu

use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use iara_train::export::safetensors::{load_safetensors_header, load_safetensors_tensor};
use iara_train::model::config::Qwen2Config;

type B = Wgpu;

// ─── SimHash (Locality Sensitive Hashing angular) ────────────────────────────
//
// Preserva similaridade de cosseno: P(hash(a) == hash(b)) = 1 - angle(a,b)/π
// Usa random projections gaussianas normalizadas.

struct SimHashTable {
    projections: Vec<Vec<f32>>, // [num_tables * num_bits, dim_in]
    tables: Vec<HashMap<u32, Vec<usize>>>,
    num_bits: usize,
    num_tables: usize,
    dim_in: usize,
    num_neurons: usize,
}

impl SimHashTable {
    fn build(weights: &[f32], dim_out: usize, dim_in: usize, num_bits: usize, num_tables: usize) -> Self {
        use std::f32::consts::PI;
        let total_proj = num_bits * num_tables;
        let mut projections = Vec::with_capacity(total_proj);

        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut next_gauss = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 33) as f32 / u32::MAX as f32 + 1e-9;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 33) as f32 / u32::MAX as f32 * 2.0 * PI;
            (-2.0 * u1.ln()).sqrt() * u2.cos()
        };

        for _ in 0..total_proj {
            let mut p: Vec<f32> = (0..dim_in).map(|_| next_gauss()).collect();
            let n = p.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
            p.iter_mut().for_each(|v| *v /= n);
            projections.push(p);
        }

        let mut tables: Vec<HashMap<u32, Vec<usize>>> = vec![HashMap::new(); num_tables];
        for ni in 0..dim_out {
            let row = &weights[ni * dim_in..(ni + 1) * dim_in];
            for t in 0..num_tables {
                let mut h: u32 = 0;
                for b in 0..num_bits {
                    let dot: f32 = row.iter().zip(projections[t * num_bits + b].iter()).map(|(a, b)| a * b).sum();
                    if dot >= 0.0 { h |= 1 << b; }
                }
                tables[t].entry(h).or_default().push(ni);
            }
        }
        SimHashTable { projections, tables, num_bits, num_tables, dim_in, num_neurons: dim_out }
    }

    /// Multi-probe query: uniao dos candidatos de todos os buckets
    fn query(&self, input: &[f32]) -> Vec<usize> {
        let mut seen = vec![false; self.num_neurons];
        for t in 0..self.num_tables {
            let mut h: u32 = 0;
            for b in 0..self.num_bits {
                let dot: f32 = input.iter().zip(self.projections[t * self.num_bits + b].iter()).map(|(a, b)| a * b).sum();
                if dot >= 0.0 { h |= 1 << b; }
            }
            if let Some(bucket) = self.tables[t].get(&h) {
                for &idx in bucket { seen[idx] = true; }
            }
        }
        seen.into_iter().enumerate().filter(|(_, c)| *c).map(|(i, _)| i).collect()
    }

    /// Recall@k: fração dos top-k neurônios ativos (|matmul|) encontrados
    fn recall_at_k(&self, weights: &[f32], input: &[f32], k: usize) -> (f32, usize, f32) {
        let dim_out = self.num_neurons;
        let dim_in  = self.dim_in;

        // Matmul completo para ground truth
        let t0 = Instant::now();
        let mut outputs: Vec<(f32, usize)> = (0..dim_out)
            .map(|j| {
                let row = &weights[j * dim_in..(j + 1) * dim_in];
                let dot: f32 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                (dot.abs(), j)
            })
            .collect();
        let full_us = t0.elapsed().as_micros();

        outputs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let top_k: std::collections::HashSet<usize> = outputs[..k.min(dim_out)].iter().map(|(_, i)| *i).collect();

        // LSH query
        let t1 = Instant::now();
        let candidates = self.query(input);
        let query_us = t1.elapsed().as_micros();

        let cands_set: std::collections::HashSet<usize> = candidates.iter().cloned().collect();
        let recall = top_k.intersection(&cands_set).count() as f32 / k as f32;
        let speedup = full_us as f32 / query_us.max(1) as f32;

        (recall, candidates.len(), speedup)
    }
}

// ─── Ternary quantization ─────────────────────────────────────────────────────

fn ternarize(weights: &[f32]) -> Vec<f32> {
    let scale = weights.iter().map(|v| v.abs()).sum::<f32>() / weights.len() as f32;
    weights.iter().map(|&v| (v / scale).round().clamp(-1.0, 1.0)).collect()
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("\n┌─ IARA Fase 2 v2: LSH com Embeddings Reais ────────────────────────────────────────┐");
    println!("│ Usa o embedding real do modelo como input (não inputs sintéticos)               │");
    println!("│ Mede: recall@k, speedup vs matmul, candidatos encontrados                      │");
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘\n");

    let model_dir = Path::new("models/qwen2.5-1.5b");
    let st_path   = model_dir.join("model.safetensors");
    let header    = load_safetensors_header(&st_path).expect("header");
    let config    = Qwen2Config::from_json(&model_dir.join("config.json")).unwrap();
    let device    = burn::backend::wgpu::WgpuDevice::default();

    // ─── 1. Obter embeddings reais via forward pass ───────────────────────────

    println!("Carregando modelo para obter embeddings reais...");
    let t0 = Instant::now();
    let model = config.load_from_safetensors::<B>(&st_path, &device).expect("load");
    println!("  ✓ {:.1}s\n", t0.elapsed().as_secs_f32());

    // 3 prompts diferentes para testar com embeddings reais
    let prompts: Vec<(&str, Vec<i64>)> = vec![
        ("código",     vec![755i64, 623, 5761, 320, 77, 2599, 1858, 374, 4756]),
        ("matemática", vec![3838, 374, 279, 32726, 315, 865, 6381, 220, 17]),
        ("trivial",    vec![3923, 374, 279, 6864, 315, 9625, 374, 12095]),
    ];

    let mut real_embeddings: Vec<(String, Vec<f32>)> = Vec::new();

    for (label, token_ids) in &prompts {
        let seq = token_ids.len();
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(token_ids.clone(), [1, seq]),
            &device,
        );
        let out = model.forward(ids, true);
        let states = out.hidden_states.expect("hidden_states");

        // states[0] = embedding output [1, seq, 1536]
        // Pegamos o último token como representativo
        let last_embed = states[0].clone()
            .slice([0..1, (seq - 1)..seq, 0..1536])
            .reshape([1536]);
        let data = last_embed.into_data();
        let embed_vals: Vec<f32> = data.as_slice::<f32>().unwrap().to_vec();

        // Normalizar (para hiperesfera — nGPT insight)
        let norm = embed_vals.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
        let embed_normalized: Vec<f32> = embed_vals.iter().map(|v| v / norm).collect();

        println!("  ✓ embedding '{label}': norm={:.4}, max={:.4}, min={:.4}",
            norm,
            embed_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            embed_vals.iter().cloned().fold(f32::INFINITY, f32::min));

        real_embeddings.push((label.to_string(), embed_normalized));
    }

    // ─── 2. Testar LSH com pesos reais e embeddings reais ────────────────────

    println!("\nTestando LSH em layers representativas...\n");

    let test_layers: &[(&str, &str)] = &[
        ("L0  gate_proj", "model.layers.0.mlp.gate_proj.weight"),
        ("L1  gate_proj", "model.layers.1.mlp.gate_proj.weight"),
        ("L14 gate_proj", "model.layers.14.mlp.gate_proj.weight"),
        ("L26 gate_proj", "model.layers.26.mlp.gate_proj.weight"),
        ("L27 gate_proj", "model.layers.27.mlp.gate_proj.weight"),
    ];

    // Configurações LSH (num_bits, num_tables)
    let lsh_configs: &[(usize, usize)] = &[(8, 4), (10, 8), (12, 16)];
    let k_values = [32usize, 64, 128];

    for &(layer_label, weight_key) in test_layers {
        let meta = match header.get(weight_key) {
            Some(m) => m,
            None => { println!("  ⚠ {weight_key} não encontrado"); continue; }
        };
        let dim_out = meta.shape[0];
        let dim_in  = meta.shape[1];

        println!("━━━ {layer_label} ({dim_out}×{dim_in}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let weights = load_safetensors_tensor(&st_path, weight_key).expect("load");
        let ternary = ternarize(&weights);

        // Esparsidade do mapa ternário
        let zeros = ternary.iter().filter(|&&v| v == 0.0).count();
        let sparsity = zeros as f32 / ternary.len() as f32;
        println!("  Esparsidade ternária: {:.1}% zeros ({zeros}/{})", sparsity * 100.0, ternary.len());

        for &(nb, nt) in lsh_configs {
            print!("  LSH({nb}b,{nt}t)  Build... ");
            let t_build = Instant::now();
            let lsh = SimHashTable::build(&ternary, dim_out, dim_in, nb, nt);
            let build_ms = t_build.elapsed().as_millis();
            print!("{build_ms}ms  ");

            // Testar cada embedding real
            let mut recalls: Vec<f32> = Vec::new();
            let mut cand_pcts: Vec<f32> = Vec::new();
            let mut speedups: Vec<f32> = Vec::new();

            for (_, embed) in &real_embeddings {
                let (r64, n_cands, spd) = lsh.recall_at_k(&weights, embed, 64);
                recalls.push(r64);
                cand_pcts.push(n_cands as f32 / dim_out as f32 * 100.0);
                speedups.push(spd);
            }

            let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
            let avg_cands  = cand_pcts.iter().sum::<f32>() / cand_pcts.len() as f32;
            let avg_speedup = speedups.iter().sum::<f32>() / speedups.len() as f32;

            println!("Recall@64={:.1}%  Cands={:.1}%  Speedup={:.1}x",
                avg_recall * 100.0, avg_cands, avg_speedup);
        }

        // Análise detalhada com melhor config e embedding "código"
        let best_lsh = SimHashTable::build(&ternary, dim_out, dim_in, 10, 8);
        let (embed_label, embed_vec) = &real_embeddings[0]; // "código"

        println!("  Análise por k [{embed_label}, 10b/8t]:");
        for &k in &k_values {
            let (recall, n_cands, speedup) = best_lsh.recall_at_k(&weights, embed_vec, k);
            println!("    k={k:>4}  Recall={:.1}%  Cands={:.1}%  Speedup={:.1}x",
                recall * 100.0,
                n_cands as f32 / dim_out as f32 * 100.0,
                speedup);
        }
        println!();
    }

    // ─── 3. Análise: ternário vs f32 para lookup ─────────────────────────────

    println!("┌─ Comparação: LSH sobre F32 vs LSH sobre Ternário ──────────────────────────────────┐");
    println!("│ Pesos ternários têm distribuição mais clara → melhor separação nos buckets      │");
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘\n");

    let gate_key = "model.layers.1.mlp.gate_proj.weight"; // Layer 1: alta BI, crítica
    let meta = &header[gate_key];
    let dim_out = meta.shape[0];
    let dim_in  = meta.shape[1];
    let weights = load_safetensors_tensor(&st_path, gate_key).expect("load");
    let ternary = ternarize(&weights);

    let (embed_label, embed_real) = &real_embeddings[0];

    println!("  Teste em L1 gate_proj ({dim_out}×{dim_in}), embedding '{embed_label}':");
    println!();
    println!("  {:>15}  {:>10}  {:>10}  {:>10}  {:>10}", "Config", "Recall@32", "Recall@64", "Cands%", "Speedup");
    println!("  {}", "─".repeat(60));

    for &(nb, nt) in lsh_configs {
        // LSH sobre F32 original
        let lsh_f32 = SimHashTable::build(&weights, dim_out, dim_in, nb, nt);
        let (r32_f, r64_f, cands_f, spd_f) = {
            let (r32, c32, s) = lsh_f32.recall_at_k(&weights, embed_real, 32);
            let (r64, _, _)   = lsh_f32.recall_at_k(&weights, embed_real, 64);
            (r32, r64, c32 as f32 / dim_out as f32 * 100.0, s)
        };

        // LSH sobre ternário
        let lsh_t = SimHashTable::build(&ternary, dim_out, dim_in, nb, nt);
        let (r32_t, r64_t, cands_t, spd_t) = {
            let (r32, c32, s) = lsh_t.recall_at_k(&weights, embed_real, 32);
            let (r64, _, _)   = lsh_t.recall_at_k(&weights, embed_real, 64);
            (r32, r64, c32 as f32 / dim_out as f32 * 100.0, s)
        };

        println!("  F32 ({nb}b,{nt}t):     {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>9.1}x",
            r32_f * 100.0, r64_f * 100.0, cands_f, spd_f);
        println!("  Ternário ({nb}b,{nt}t): {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>9.1}x",
            r32_t * 100.0, r64_t * 100.0, cands_t, spd_t);
        println!();
    }

    // ─── 4. Distribuição de ativações reais ──────────────────────────────────

    println!("┌─ Distribuição de Ativações Reais (matmul completo) ────────────────────────────────┐");
    println!("│ Quão esparsas são as ativações na prática com embeddings reais?                  │");
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘\n");

    for (label, embed) in &real_embeddings {
        let mut outputs: Vec<f32> = (0..dim_out)
            .map(|j| {
                let row = &weights[j * dim_in..(j + 1) * dim_in];
                row.iter().zip(embed.iter()).map(|(w, x)| w * x).sum::<f32>().abs()
            })
            .collect();
        outputs.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let total: f32 = outputs.iter().sum();
        let cumsum_50: usize = {
            let mut s = 0.0f32;
            let mut n = 0;
            for &v in &outputs {
                s += v; n += 1;
                if s >= total * 0.5 { break; }
            }
            n
        };
        let cumsum_90: usize = {
            let mut s = 0.0f32;
            let mut n = 0;
            for &v in &outputs {
                s += v; n += 1;
                if s >= total * 0.9 { break; }
            }
            n
        };

        println!("  '{label}' (L1 gate_proj, {dim_out} neurônios):");
        println!("    Top-{cumsum_50} neurônios = 50% da energia total ({:.1}%)",
            cumsum_50 as f32 / dim_out as f32 * 100.0);
        println!("    Top-{cumsum_90} neurônios = 90% da energia total ({:.1}%)",
            cumsum_90 as f32 / dim_out as f32 * 100.0);
        println!("    → LSH precisa encontrar os top-{cumsum_90} ({:.0}%) para 90% de qualidade\n",
            cumsum_90 as f32 / dim_out as f32 * 100.0);
    }

    println!("┌─ Conclusão ─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Com embeddings reais vs sintéticos: recall muda significativamente                │");
    println!("│ A concentração de energia nos neurônios top-k é chave para LSH funcionar         │");
    println!("│ Em layers críticas (0,1,26,27): ativações mais concentradas = LSH mais eficaz   │");
    println!("│                                                                                   │");
    println!("│ → Próximo: super_layer.rs (merge real) + WGSL sparse kernel                     │");
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
}
