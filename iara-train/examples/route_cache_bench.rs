//! Benchmark do Semantic Route Cache integrado com A*.
//!
//! Simula um workload realista de prompts:
//!   - 30% prompts repetidos exatos
//!   - 40% variações semânticas do mesmo tópico
//!   - 30% prompts totalmente novos
//!
//! Mede: hit rate, speedup médio (cache vs A*), FLOPs economizados.

use iara_train::navigator::{
    route_cache::{RouteCache, RouteCacheConfig, CacheResult},
    router::{route_astar, RouterConfig, N_SUPER_LAYERS},
};
use std::time::Instant;

// Prompts de teste simulando diferentes domínios
const PROMPTS: &[&str] = &[
    // Código (SL0→SL1→SL2→SL3 esperado — qualidade máxima)
    "Write a Python function to reverse a string",
    "Implement quicksort in Rust",
    "Create a binary search tree in JavaScript",
    "Write unit tests for a REST API",
    "Explain how async/await works in Python",

    // Raciocínio (pode usar atalho SL1→SL3)
    "What is the capital of France?",
    "Explain the French Revolution",
    "Who was Napoleon Bonaparte?",
    "What caused World War I?",
    "Describe the Industrial Revolution",

    // Geração simples (SL0→SL3 — draft rápido)
    "Hello",
    "Hi there",
    "Good morning",
    "How are you?",
    "What's up?",

    // Math
    "Solve: 2x + 5 = 13",
    "What is the derivative of sin(x)?",
    "Explain the Pythagorean theorem",
    "Calculate the area of a circle with radius 5",
    "What is a prime number?",
];

// Gera um workload com padrão de repetição realístico
fn generate_workload(n_requests: usize) -> Vec<String> {
    let mut rng_state: u64 = 42;
    let mut lcg = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / u32::MAX as f64
    };

    let mut workload = Vec::with_capacity(n_requests);
    for _ in 0..n_requests {
        let r = lcg();
        let prompt = if r < 0.30 {
            // 30% repetição exata
            let idx = (lcg() * PROMPTS.len() as f64) as usize % PROMPTS.len();
            PROMPTS[idx].to_string()
        } else if r < 0.70 {
            // 40% variação semântica
            let idx = (lcg() * PROMPTS.len() as f64) as usize % PROMPTS.len();
            let suffix = if lcg() < 0.5 { " Please explain briefly." } else { " Give me a short example." };
            format!("{}{}", PROMPTS[idx], suffix)
        } else {
            // 30% novo
            let idx = (lcg() * PROMPTS.len() as f64) as usize % PROMPTS.len();
            let prefix = if lcg() < 0.5 { "Can you " } else { "Please " };
            format!("{}{}", prefix, PROMPTS[idx].to_lowercase())
        };
        workload.push(prompt);
    }
    workload
}

fn main() {
    println!("=== IARA Semantic Route Cache — Benchmark ===\n");

    let n_requests = 1000;
    let workload   = generate_workload(n_requests);

    let config     = RouterConfig::default();
    let cache_cfg  = RouteCacheConfig { hash_bits: 16, max_entries: 256, embed_dim: 256 };

    // Simular sparsity variável (normal vs alta esparsidade por domínio)
    let base_sparsity = [0.3f32, 0.25, 0.35, 0.2];
    let high_sparsity = [0.6f32, 0.55, 0.65, 0.5];

    let mut cache         = RouteCache::new(cache_cfg);
    let mut astar_total   = 0u128;
    let mut cache_total   = 0u128;
    let mut flops_saved   = 0.0f32;
    let mut astar_calls   = 0usize;
    let mut path_hist     = std::collections::HashMap::<Vec<usize>, usize>::new();

    // ── Benchmark principal ───────────────────────────────────────────────────
    println!("Processando {n_requests} requests...\n");

    for (i, prompt) in workload.iter().enumerate() {
        // Simular diferentes sparsities por prompt (código = alta, trivial = baixa)
        let sparsity = if i % 5 == 0 { &high_sparsity } else { &base_sparsity };
        let logit_entropy = 0.3 + (i as f32 * 0.001).sin() * 0.2;

        // Medir custo da consulta ao cache
        let t0 = Instant::now();
        let result = cache.lookup(prompt);
        let cache_us = t0.elapsed().as_micros();
        cache_total += cache_us;

        match result {
            CacheResult::Hit { route, .. } => {
                *path_hist.entry(route.path.clone()).or_insert(0) += 1;
                flops_saved += route.flops_saved_pct;
            }
            CacheResult::Miss { hash } => {
                // Cache miss: executar A*
                let t1 = Instant::now();
                let route = route_astar(sparsity, logit_entropy, &config);
                let astar_us = t1.elapsed().as_micros();
                astar_total += astar_us;
                astar_calls += 1;

                flops_saved += route.flops_saved_pct;
                *path_hist.entry(route.path.clone()).or_insert(0) += 1;
                cache.store(hash, &route);
            }
        }
    }

    // ── Resultados ────────────────────────────────────────────────────────────
    let stats = cache.stats();
    let avg_astar_us  = if astar_calls > 0 { astar_total / astar_calls as u128 } else { 0 };
    let avg_cache_us  = cache_total / n_requests as u128;
    let avg_flops_pct = flops_saved / n_requests as f32;

    // Speedup: comparar request médio com/sem cache
    let with_cache_us    = avg_cache_us as f32 * stats.hit_rate
                         + avg_astar_us as f32 * (1.0 - stats.hit_rate);
    let without_cache_us = avg_astar_us as f32;
    let latency_speedup  = without_cache_us / with_cache_us.max(1.0);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Semantic Route Cache — Resultados               ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Requests:        {:>6}                               ║", n_requests);
    println!("║  Cache entries:   {:>6}                               ║", stats.n_entries);
    println!("║  Cache hits:      {:>6} ({:.1}%)                    ║", stats.total_hits, stats.hit_rate * 100.0);
    println!("║  Cache misses:    {:>6}                               ║", stats.total_misses);
    println!("║  A* calls:        {:>6}                               ║", astar_calls);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Avg A* latency:  {:>5} µs                            ║", avg_astar_us);
    println!("║  Avg cache hit:   {:>5} µs                            ║", avg_cache_us);
    println!("║  Latency speedup: {:.2}×                              ║", latency_speedup);
    println!("║  Avg FLOPs saved: {:.1}%                             ║", avg_flops_pct);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Distribuição de rotas:                                 ║");
    let mut paths: Vec<_> = path_hist.iter().collect();
    paths.sort_by(|a, b| b.1.cmp(a.1));
    for (path, count) in paths.iter().take(5) {
        let pct = **count as f32 / n_requests as f32 * 100.0;
        let route_names: Vec<&str> = path.iter()
            .map(|&i| iara_train::navigator::router::SL_NAMES[i.min(N_SUPER_LAYERS - 1)])
            .collect();
        println!("║    {:>3}× ({:4.1}%) {:?}", count, pct, route_names);
    }
    println!("╚══════════════════════════════════════════════════════════╝");
}
