// ============================================================================
// IARA Engine Stress Test — HTTP End-to-End Benchmark
// ============================================================================
// Requires a running IARA server with a model loaded.
//
// Usage:
//   Terminal 1:  ./run_iara.sh release
//   Terminal 2:  curl -X POST localhost:3000/api/load-model -H 'Content-Type: application/json' -d '{"name":"qwen2.5-1.5b"}'
//   Terminal 2:  cargo run --release -p iara-engine --bin stress_test
// ============================================================================

use std::time::Instant;
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "http://localhost:3000";

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
}

#[derive(Deserialize, Debug)]
struct GenerateResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,       // milliseconds
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    total_duration: u64,      // milliseconds
}

#[derive(Deserialize, Debug)]
struct HealthResponse {
    #[serde(default)]
    status: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    model_vram_gb: f64,
    #[serde(default)]
    vram_free_gb: Option<f64>,
    #[serde(default)]
    vram_total_gb: Option<f64>,
}

struct BenchResult {
    name: String,
    prompt_tokens: u64,
    gen_tokens: u64,
    latency_ms: f64,
    tok_per_sec: f64,
    error: Option<String>,
}

fn print_bench_table(results: &[BenchResult]) {
    println!("\n{}", "=".repeat(100));
    println!("{:<35} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "Test", "Prompt", "Gen", "Latency", "tok/s", "Status");
    println!("{}", "-".repeat(100));
    for r in results {
        let status = r.error.as_deref().unwrap_or("OK");
        println!("{:<35} {:>8} {:>8} {:>10.0}ms {:>10.1} {:>12}",
            r.name, r.prompt_tokens, r.gen_tokens, r.latency_ms, r.tok_per_sec, status);
    }
    println!("{}", "=".repeat(100));
}

async fn check_health(client: &reqwest::Client) -> Result<HealthResponse, String> {
    client.get(format!("{}/health", BASE_URL))
        .send().await.map_err(|e| format!("Connection failed: {}", e))?
        .json::<HealthResponse>().await.map_err(|e| format!("Parse failed: {}", e))
}

async fn generate(
    client: &reqwest::Client,
    prompt: &str,
    max_tokens: usize,
    model: &str,
) -> Result<(GenerateResponse, f64), String> {
    let req = GenerateRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.9),
        max_tokens: Some(max_tokens),
    };

    let start = Instant::now();
    let resp = client.post(format!("{}/api/generate", BASE_URL))
        .json(&req)
        .timeout(std::time::Duration::from_secs(300))
        .send().await.map_err(|e| format!("Request failed: {}", e))?;

    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {}: {}", status, &body[..body.len().min(200)]));
    }

    let gen: GenerateResponse = resp.json().await
        .map_err(|e| format!("JSON parse: {}", e))?;
    Ok((gen, wall_ms))
}

#[tokio::main]
async fn main() {
    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "IARA ENGINE STRESS TEST");
    println!("{}", "=".repeat(80));

    let client = reqwest::Client::new();

    // ── 1. Health check ──────────────────────────────────────────────────
    println!("\n[1] Checking server health...");
    let health = match check_health(&client).await {
        Ok(h) => h,
        Err(e) => {
            println!("FATAL: Server not reachable at {}", BASE_URL);
            println!("Error: {}", e);
            println!("\nStart the server first:");
            println!("  Terminal 1: ./run_iara.sh release");
            println!("  Terminal 2: curl -X POST localhost:3000/api/load-model \\");
            println!("    -H 'Content-Type: application/json' -d '{{\"name\":\"qwen2.5-1.5b\"}}'");
            std::process::exit(1);
        }
    };

    println!("  Status:     {}", health.status);
    println!("  Model:      {}", health.model);
    println!("  VRAM Model: {:.2} GB", health.model_vram_gb);
    println!("  VRAM Free:  {:.2} GB", health.vram_free_gb.unwrap_or(0.0));
    println!("  VRAM Total: {:.2} GB", health.vram_total_gb.unwrap_or(0.0));

    if health.model.is_empty() || health.model == "none" {
        println!("\nFATAL: No model loaded. Load one first:");
        println!("  curl -X POST localhost:3000/api/load-model \\");
        println!("    -H 'Content-Type: application/json' -d '{{\"name\":\"qwen2.5-1.5b\"}}'");
        std::process::exit(1);
    }

    let model_name = health.model.clone();
    let mut results: Vec<BenchResult> = Vec::new();

    // ── 2. Warmup ────────────────────────────────────────────────────────
    println!("\n[2] Warmup (1 token)...");
    match generate(&client, "Hello", 1, &model_name).await {
        Ok((_, ms)) => println!("  Warmup done in {:.0}ms", ms),
        Err(e) => println!("  Warmup failed: {}", e),
    }

    // ── 3. Single token latency (time to first token) ────────────────────
    println!("\n[3] Single-token latency (3 runs)...");
    let mut ttft_times = Vec::new();
    for i in 0..3 {
        let prompt = format!("The meaning of life is");
        match generate(&client, &prompt, 1, &model_name).await {
            Ok((gen, wall_ms)) => {
                ttft_times.push(wall_ms);
                println!("  Run {}: {:.0}ms", i + 1, wall_ms);
                results.push(BenchResult {
                    name: format!("TTFT run {}", i + 1),
                    prompt_tokens: gen.prompt_eval_count,
                    gen_tokens: gen.eval_count,
                    latency_ms: wall_ms,
                    tok_per_sec: if gen.eval_duration > 0 {
                        gen.eval_count as f64 / (gen.eval_duration as f64 / 1000.0)
                    } else { 0.0 },
                    error: None,
                });
            }
            Err(e) => {
                println!("  Run {} FAILED: {}", i + 1, e);
                results.push(BenchResult {
                    name: format!("TTFT run {}", i + 1),
                    prompt_tokens: 0, gen_tokens: 0, latency_ms: 0.0, tok_per_sec: 0.0,
                    error: Some(e),
                });
            }
        }
    }

    // ── 4. Throughput sweep (varying generation length) ──────────────────
    println!("\n[4] Throughput sweep...");
    let gen_lengths = [10, 50, 100];
    for &max_tok in &gen_lengths {
        let prompt = "Explain quantum computing in simple terms. Be detailed and thorough.";
        match generate(&client, prompt, max_tok, &model_name).await {
            Ok((gen, wall_ms)) => {
                let tok_s = if gen.eval_duration > 0 {
                    gen.eval_count as f64 / (gen.eval_duration as f64 / 1000.0)
                } else { 0.0 };
                println!("  max_tokens={:<4} -> {} tokens in {:.0}ms ({:.1} tok/s)",
                    max_tok, gen.eval_count, wall_ms, tok_s);
                results.push(BenchResult {
                    name: format!("Throughput max={}", max_tok),
                    prompt_tokens: gen.prompt_eval_count,
                    gen_tokens: gen.eval_count,
                    latency_ms: wall_ms,
                    tok_per_sec: tok_s,
                    error: None,
                });
            }
            Err(e) => {
                println!("  max_tokens={} FAILED: {}", max_tok, e);
                results.push(BenchResult {
                    name: format!("Throughput max={}", max_tok),
                    prompt_tokens: 0, gen_tokens: 0, latency_ms: 0.0, tok_per_sec: 0.0,
                    error: Some(e),
                });
            }
        }
    }

    // ── 5. Prompt length sweep ───────────────────────────────────────────
    println!("\n[5] Prompt length sweep...");
    let prompts = [
        ("Short (~10 tok)", "What is AI?"),
        ("Medium (~50 tok)", "Explain the difference between supervised learning, unsupervised learning, and reinforcement learning in machine learning. Give examples of each."),
        ("Long (~100 tok)", "Write a comprehensive analysis of the following topics: neural network architectures including transformers, attention mechanisms, convolutional neural networks, and recurrent neural networks. Compare their strengths and weaknesses for different types of tasks including natural language processing, computer vision, and time series analysis. Also discuss recent developments."),
    ];
    for (label, prompt) in &prompts {
        match generate(&client, prompt, 20, &model_name).await {
            Ok((gen, wall_ms)) => {
                let tok_s = if gen.eval_duration > 0 {
                    gen.eval_count as f64 / (gen.eval_duration as f64 / 1000.0)
                } else { 0.0 };
                println!("  {:<20} prompt_tokens={} gen={} {:.0}ms ({:.1} tok/s)",
                    label, gen.prompt_eval_count, gen.eval_count, wall_ms, tok_s);
                results.push(BenchResult {
                    name: format!("Prompt {}", label),
                    prompt_tokens: gen.prompt_eval_count,
                    gen_tokens: gen.eval_count,
                    latency_ms: wall_ms,
                    tok_per_sec: tok_s,
                    error: None,
                });
            }
            Err(e) => {
                println!("  {} FAILED: {}", label, e);
                results.push(BenchResult {
                    name: format!("Prompt {}", label),
                    prompt_tokens: 0, gen_tokens: 0, latency_ms: 0.0, tok_per_sec: 0.0,
                    error: Some(e),
                });
            }
        }
    }

    // ── 6. Burst test (concurrent requests) ──────────────────────────────
    println!("\n[6] Burst test (5 concurrent requests)...");
    let burst_start = Instant::now();
    let mut handles = Vec::new();
    for i in 0..5 {
        let c = client.clone();
        let m = model_name.clone();
        let p = format!("Count from {} to {}", i * 10, i * 10 + 5);
        handles.push(tokio::spawn(async move {
            generate(&c, &p, 10, &m).await
        }));
    }

    let mut burst_ok = 0;
    let mut burst_fail = 0;
    let mut burst_tok_total = 0u64;
    for h in handles {
        match h.await {
            Ok(Ok((gen, _))) => { burst_ok += 1; burst_tok_total += gen.eval_count; }
            _ => { burst_fail += 1; }
        }
    }
    let burst_ms = burst_start.elapsed().as_secs_f64() * 1000.0;
    let burst_tps = burst_tok_total as f64 / (burst_ms / 1000.0);
    println!("  OK: {}, Failed: {}, Total tokens: {}, Wall time: {:.0}ms, Aggregate: {:.1} tok/s",
        burst_ok, burst_fail, burst_tok_total, burst_ms, burst_tps);

    results.push(BenchResult {
        name: "Burst 5x concurrent".into(),
        prompt_tokens: 0,
        gen_tokens: burst_tok_total,
        latency_ms: burst_ms,
        tok_per_sec: burst_tps,
        error: if burst_fail > 0 { Some(format!("{} failed", burst_fail)) } else { None },
    });

    // ── 7. Final VRAM check ──────────────────────────────────────────────
    println!("\n[7] Post-test VRAM check...");
    if let Ok(h) = check_health(&client).await {
        println!("  VRAM Used: {:.2} GB / {:.2} GB ({:.0}%)",
            h.vram_total_gb.unwrap_or(0.0) - h.vram_free_gb.unwrap_or(0.0),
            h.vram_total_gb.unwrap_or(0.0),
            (1.0 - h.vram_free_gb.unwrap_or(0.0) / h.vram_total_gb.unwrap_or(0.01).max(0.01)) * 100.0);
    }

    // ── Summary ──────────────────────────────────────────────────────────
    print_bench_table(&results);

    // Average TTFT
    if !ttft_times.is_empty() {
        let avg_ttft = ttft_times.iter().sum::<f64>() / ttft_times.len() as f64;
        println!("\nAverage TTFT (time to first token): {:.0}ms", avg_ttft);
    }

    // Average tok/s (excluding TTFT and burst)
    let throughput_results: Vec<f64> = results.iter()
        .filter(|r| r.name.starts_with("Throughput") && r.error.is_none() && r.tok_per_sec > 0.0)
        .map(|r| r.tok_per_sec)
        .collect();
    if !throughput_results.is_empty() {
        let avg_tps = throughput_results.iter().sum::<f64>() / throughput_results.len() as f64;
        println!("Average throughput: {:.1} tok/s", avg_tps);
    }

    println!("\nStress test complete!");
}
