//! Correctness validation: ensures greedy (temp=0) output is deterministic.
//!
//! First run creates baseline. Subsequent runs verify output matches.
//!
//! Run: cargo test --release -p iara-engine --test correctness -- --ignored --nocapture

use serde::{Deserialize, Serialize};
use std::path::Path;

const BASELINE_FILE: &str = "correctness_baseline.json";

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct BaselineEntry {
    prompt: String,
    model: String,
    max_tokens: u32,
    response: String,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

fn generate_greedy(
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("http://localhost:3000/api/generate")
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }))
        .timeout(std::time::Duration::from_secs(120))
        .send()?
        .json::<GenerateResponse>()?;
    Ok(resp.response)
}

#[test]
#[ignore]
fn correctness_greedy_deterministic() {
    let model = "qwen3-4b";
    let test_cases = vec![
        ("What is the capital of France?", 10),
        ("2 + 2 =", 5),
        ("The color of the sky is", 8),
        ("fn main() {", 15),
        ("Hello, my name is", 10),
    ];

    // Generate current outputs
    let mut current: Vec<BaselineEntry> = Vec::new();
    for (prompt, max_tokens) in &test_cases {
        eprint!("  Testing: '{}' ... ", prompt);
        match generate_greedy(model, prompt, *max_tokens) {
            Ok(response) => {
                eprintln!("OK: '{}'", response.chars().take(50).collect::<String>());
                current.push(BaselineEntry {
                    prompt: prompt.to_string(),
                    model: model.to_string(),
                    max_tokens: *max_tokens,
                    response,
                });
            }
            Err(e) => {
                eprintln!("FAILED: {}", e);
                panic!("Generation failed for prompt: {}", prompt);
            }
        }
    }

    // Compare or create baseline
    let baseline_path = Path::new(BASELINE_FILE);
    if baseline_path.exists() {
        let baseline_json = std::fs::read_to_string(baseline_path).unwrap();
        let baseline: Vec<BaselineEntry> = serde_json::from_str(&baseline_json).unwrap();

        eprintln!("\n=== Comparing against baseline ===");
        let mut all_match = true;
        for (base, curr) in baseline.iter().zip(current.iter()) {
            if base.response != curr.response {
                eprintln!(
                    "MISMATCH for '{}': baseline='{}' current='{}'",
                    base.prompt, base.response, curr.response
                );
                all_match = false;
            } else {
                eprintln!("  MATCH: '{}'", base.prompt);
            }
        }
        assert!(
            all_match,
            "Correctness regression detected! Output differs from baseline."
        );
        eprintln!("\n✅ All outputs match baseline.");
    } else {
        let json = serde_json::to_string_pretty(&current).unwrap();
        std::fs::write(baseline_path, &json).unwrap();
        eprintln!("\n📝 Baseline created: {}", BASELINE_FILE);
        eprintln!("   Run again to verify determinism.");
    }
}
