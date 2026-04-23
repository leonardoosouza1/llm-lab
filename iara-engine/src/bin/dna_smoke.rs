//! dna_smoke — load a .dna model via CausalLM::load_dna and run a forward pass.
//!
//! Usage:
//!   dna_smoke --dna models/qwen2.5-1.5b.dna --config-dir models/qwen2.5-1.5b
//!
//! Prints VRAM used, forward latency, top-5 logits. Verifies finite output.
//! Sanity check before plugging .dna into the HTTP server.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use compute_core::ComputeBackend;

use iara_engine::model::architecture::ModelArchitecture;
use iara_engine::model::{CausalLM, ModelCfg};

#[derive(Parser, Debug)]
#[command(about = "Load a .dna model and run a forward pass smoke test")]
struct Cli {
    #[arg(long)]
    dna: PathBuf,

    /// Directory containing config.json (typically the original safetensors dir).
    #[arg(long)]
    config_dir: PathBuf,

    /// Token IDs to feed as input (space-separated).
    #[arg(long, default_value = "1")]
    tokens: String,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    let backend: Arc<dyn ComputeBackend> = {
        #[cfg(feature = "rocm")]
        {
            let be = rocm_backend::RocmBackend::probe()
                .map_err(|e| anyhow::anyhow!("ROCm probe failed: {e}"))?;
            Arc::new(be)
        }
        #[cfg(all(feature = "opencl", not(feature = "rocm")))]
        {
            let be = opencl_backend::OpenClBackend::probe()
                .map_err(|e| anyhow::anyhow!("OpenCL probe failed: {e}"))?;
            Arc::new(be)
        }
        #[cfg(not(any(feature = "rocm", feature = "opencl")))]
        {
            anyhow::bail!("No GPU backend compiled in. Enable rocm or opencl features.");
        }
    };

    let config_path = cli.config_dir.join("config.json");
    let cfg = ModelCfg::from_config_json(&config_path)
        .map_err(|e| anyhow::anyhow!("config parse: {e}"))?;
    let arch = ModelArchitecture::from_config_json(&config_path)
        .map_err(|e| anyhow::anyhow!("arch parse: {e}"))?;

    println!("Loading .dna: {}", cli.dna.display());
    let t_load = Instant::now();
    let mut model = CausalLM::load_dna(&cli.dna, cfg.clone(), arch, backend)
        .map_err(|e| anyhow::anyhow!("load_dna: {e}"))?;
    println!("Loaded in {:.2}s ({} layers, hidden={}, vocab={})",
        t_load.elapsed().as_secs_f32(), cfg.num_layers, cfg.hidden_size, cfg.vocab_size);

    let tokens: Vec<u32> = cli.tokens.split_whitespace()
        .map(|s| s.parse::<u32>().unwrap_or(1))
        .collect();
    println!("Input tokens: {:?}", tokens);

    model.reset_cache();
    let t_fwd = Instant::now();
    let (logits, _traces) = model.forward(&tokens, None)
        .map_err(|e| anyhow::anyhow!("forward: {e}"))?;
    let dt_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;

    let finite = logits.iter().all(|x| x.is_finite());
    let (mn, mx) = logits.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
        |(lo, hi), &v| (lo.min(v), hi.max(v)));

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<_> = indexed.iter().take(5).collect();

    println!("Forward pass: {:.2} ms", dt_ms);
    println!("Logits: len={}, finite={}, min={:.4}, max={:.4}", logits.len(), finite, mn, mx);
    println!("Top 5 tokens (by logit):");
    for (rank, (tok, lg)) in top5.iter().enumerate() {
        println!("  #{}: token_id={} logit={:.4}", rank + 1, tok, lg);
    }

    if !finite {
        anyhow::bail!("logits contain NaN/Inf — model broken");
    }
    println!("\n✅ .dna forward smoke test PASSED");
    Ok(())
}
