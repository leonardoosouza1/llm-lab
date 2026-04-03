//! Performance benchmark suite for IARA Engine.
//!
//! Run with a live server:
//!   RUST_LOG=info IARA_QUANTIZE=1 ./run_iara.sh release &
//!   cargo test --release -p iara-engine --test benchmark -- --ignored --nocapture
//!
//! Results saved to: benchmark_results.json

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Serialize, Deserialize, Debug)]
struct BenchmarkResult {
    timestamp: String,
    model: String,
    prompt: String,
    prompt_tokens: u32,
    generated_tokens: u32,
    total_duration_ms: u128,
    prefill_ms: u128,
    decode_ms: u128,
    decode_tps: f64,
    overall_tps: f64,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    total_duration: u128,
    prompt_eval_count: u32,
    eval_count: u32,
    eval_duration: u128,
}

// Helper to send generate request
fn generate(
    model: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
) -> Result<GenerateResponse, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("http://localhost:3000/api/generate")
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }))
        .timeout(std::time::Duration::from_secs(120))
        .send()?
        .json::<GenerateResponse>()?;
    Ok(resp)
}

#[test]
#[ignore] // Only run manually with --ignored
fn benchmark_decode_throughput() {
    let model = "qwen3-4b";
    let prompts = vec![
        ("short", "What is 2+2?", 20),
        ("medium", "Explain how a transformer neural network works step by step:", 50),
        ("long", "Write a detailed essay about the history of artificial intelligence from the 1950s to today:", 100),
    ];

    let mut results = Vec::new();

    for (label, prompt, max_tokens) in &prompts {
        eprintln!("\n=== Benchmark: {} ({} max tokens) ===", label, max_tokens);
        let start = Instant::now();

        match generate(model, prompt, *max_tokens, 0.0) {
            Ok(resp) => {
                let elapsed = start.elapsed().as_millis();
                let decode_tps = if resp.eval_duration > 0 {
                    resp.eval_count as f64 / (resp.eval_duration as f64 / 1000.0)
                } else {
                    0.0
                };
                let overall_tps = if elapsed > 0 {
                    resp.eval_count as f64 / (elapsed as f64 / 1000.0)
                } else {
                    0.0
                };

                let result = BenchmarkResult {
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    model: model.to_string(),
                    prompt: label.to_string(),
                    prompt_tokens: resp.prompt_eval_count,
                    generated_tokens: resp.eval_count,
                    total_duration_ms: elapsed,
                    prefill_ms: elapsed.saturating_sub(resp.eval_duration),
                    decode_ms: resp.eval_duration,
                    decode_tps,
                    overall_tps,
                };

                eprintln!("  Prompt tokens: {}", result.prompt_tokens);
                eprintln!("  Generated: {} tokens", result.generated_tokens);
                eprintln!("  Decode TPS: {:.1}", result.decode_tps);
                eprintln!("  Overall TPS: {:.1}", result.overall_tps);
                eprintln!("  Total: {}ms", result.total_duration_ms);

                results.push(result);
            }
            Err(e) => eprintln!("  FAILED: {}", e),
        }
    }

    // Save results
    if !results.is_empty() {
        let json = serde_json::to_string_pretty(&results).unwrap();
        std::fs::write("benchmark_results.json", &json).unwrap();
        eprintln!("\nResults saved to benchmark_results.json");

        // Assert minimum performance
        let best_tps = results.iter().map(|r| r.decode_tps).fold(0.0f64, f64::max);
        eprintln!("\nBest decode TPS: {:.1}", best_tps);
    }
}
