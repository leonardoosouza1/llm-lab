//! IARA Training Engine -- Native HIP LoRA fine-tuning
//!
//! Zero frameworks. Pure Rust + HIP kernels.
//!
//! Usage:
//!   cargo run -p iara-train --release -- \
//!     --model-dir models/qwen2.5-1.5b \
//!     --data data/train.bin \
//!     --steps 100 --lr 1e-4 --lora-rank 32

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "iara-train")]
struct Args {
    #[arg(long, default_value = "models/qwen2.5-1.5b")]
    model_dir: String,
    #[arg(long, default_value = "data/train.bin")]
    data: String,
    #[arg(long, default_value_t = 32)]
    lora_rank: usize,
    #[arg(long, default_value_t = 64.0)]
    lora_alpha: f64,
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 128)]
    seq_len: usize,
    #[arg(long, default_value = "checkpoints")]
    output_dir: String,
    /// Enable full observability (writes to observe/ directory)
    #[arg(long)]
    observe: bool,
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args = Args::parse();

    println!("================================================");
    println!("  IARA Training Engine v0.3.0");
    println!("  Native Rust + HIP kernels on AMD GPU");
    println!("  Zero frameworks. Zero WGSL.");
    println!("================================================");

    // Enable observability if requested
    if args.observe {
        iara_observe::GLOBAL_COLLECTOR.enable("observe");
        log::info!("Observability enabled -- recording to observe/");
    }

    // Initialize native ROCm backend
    // TODO: hip_ffi::hip_set_device(0)
    // TODO: Load weights, init LoRA, train

    log::info!("Native training skeleton ready -- implement forward/backward HIP kernels next");
}
