//! IARA Training Engine -- Native CPU LoRA fine-tuning
//!
//! Zero frameworks. Pure Rust + CPU f32.
//!
//! Usage:
//!   # Run a forward pass test (top-5 next tokens for a token sequence):
//!   cargo run -p iara-train -- forward --model-dir models/qwen2.5-1.5b --tokens 9707,1879,28
//!
//!   # Train with LoRA:
//!   cargo run -p iara-train -- train \
//!     --model-dir models/qwen2.5-1.5b \
//!     --data data/train.bin \
//!     --steps 100 --lr 2e-4 --lora-rank 16 --output lora_output/

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "iara-train", about = "IARA LoRA Training Engine")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a forward pass and print top-k next tokens.
    Forward {
        #[arg(long, default_value = "models/qwen2.5-1.5b")]
        model_dir: PathBuf,
        /// Comma-separated token IDs to feed as input (e.g. "9707,1879,28").
        #[arg(long, default_value = "9707")]
        tokens: String,
        /// Number of top tokens to display.
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Fine-tune with LoRA.
    Train {
        #[arg(long, default_value = "models/qwen2.5-1.5b")]
        model_dir: PathBuf,
        /// Path to pre-tokenized .bin file (flat little-endian u32 token IDs).
        #[arg(long, default_value = "data/train.bin")]
        data: PathBuf,
        #[arg(long, default_value_t = 16)]
        lora_rank: usize,
        #[arg(long, default_value_t = 32.0)]
        lora_alpha: f32,
        #[arg(long, default_value_t = 2e-4)]
        lr: f32,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value = "lora_output")]
        output: PathBuf,
        #[arg(long, default_value_t = 50)]
        checkpoint_every: usize,
        #[arg(long, default_value_t = 10)]
        log_every: usize,
    },
    /// Quick sanity-check: load model, run one forward pass, print loss.
    Sanity {
        #[arg(long, default_value = "models/qwen2.5-1.5b")]
        model_dir: PathBuf,
        /// Path to pre-tokenized .bin file for loss evaluation.
        #[arg(long, default_value = "data/train.bin")]
        data: PathBuf,
        #[arg(long, default_value_t = 32)]
        seq_len: usize,
    },
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    println!("================================================");
    println!("  IARA Training Engine v0.3.0");
    println!("  Native Rust CPU LoRA fine-tuning");
    println!("  Zero frameworks. Zero GPU (training).");
    println!("================================================");

    let args = Args::parse();

    match args.command {
        Command::Forward { model_dir, tokens, top_k } => {
            cmd_forward(&model_dir, &tokens, top_k);
        }
        Command::Train {
            model_dir, data, lora_rank, lora_alpha, lr, steps,
            batch_size, seq_len, output, checkpoint_every, log_every,
        } => {
            cmd_train(
                &model_dir, &data, lora_rank, lora_alpha, lr, steps,
                batch_size, seq_len, &output, checkpoint_every, log_every,
            );
        }
        Command::Sanity { model_dir, data, seq_len } => {
            cmd_sanity(&model_dir, &data, seq_len);
        }
    }
}

// ── Commands ─────────────────────────────────────────────────────────────────

fn cmd_forward(model_dir: &PathBuf, tokens_str: &str, top_k: usize) {
    log::info!("Loading model from {}", model_dir.display());
    let t0 = std::time::Instant::now();

    let weights = iara_train::ModelWeights::from_dir(model_dir)
        .unwrap_or_else(|e| panic!("Failed to load model: {e}"));
    log::info!("Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let input_ids: Vec<u32> = tokens_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if input_ids.is_empty() {
        eprintln!("No valid token IDs in: {}", tokens_str);
        std::process::exit(1);
    }

    log::info!("Running forward pass on {} tokens...", input_ids.len());
    let t_fwd = std::time::Instant::now();
    let result = iara_train::forward(&weights, &input_ids, false);
    let elapsed = t_fwd.elapsed().as_secs_f32() * 1000.0;

    let top = iara_train::model::forward::top_k_last_token(
        &result.logits, weights.config.vocab_size, top_k
    );

    println!("\nForward pass: {:.1}ms", elapsed);
    println!("Top-{} next tokens:", top_k);
    for (rank, (token_id, prob)) in top.iter().enumerate() {
        println!("  #{}: token_id={:6}  prob={:.4}", rank + 1, token_id, prob);
    }
}

fn cmd_train(
    model_dir: &PathBuf,
    data_path: &PathBuf,
    lora_rank: usize,
    lora_alpha: f32,
    lr: f32,
    steps: usize,
    batch_size: usize,
    seq_len: usize,
    output: &PathBuf,
    checkpoint_every: usize,
    log_every: usize,
) {
    log::info!("Loading model from {}", model_dir.display());
    let t0 = std::time::Instant::now();

    let weights = iara_train::ModelWeights::from_dir(model_dir)
        .unwrap_or_else(|e| panic!("Failed to load model: {e}"));
    log::info!("Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut lora = iara_train::LoraAdapter::new_for_qwen2(
        lora_rank, lora_alpha, &weights.config
    );

    let dataset = iara_train::TextDataset::from_token_file(data_path, seq_len);
    log::info!("Dataset: {} samples, seq_len={}", dataset.num_samples(), seq_len);

    let config = iara_train::TrainConfig {
        lr,
        num_steps: steps,
        batch_size,
        seq_len,
        checkpoint_every,
        output_dir: output.clone(),
        lora_rank,
        lora_alpha,
        log_every,
        ..Default::default()
    };

    let trainer = iara_train::Trainer::new(config);
    trainer.train(&weights, &mut lora, &dataset)
        .unwrap_or_else(|e| panic!("Training failed: {e}"));
}

fn cmd_sanity(model_dir: &PathBuf, data_path: &PathBuf, seq_len: usize) {
    log::info!("Loading model from {}", model_dir.display());
    let t0 = std::time::Instant::now();

    let weights = iara_train::ModelWeights::from_dir(model_dir)
        .unwrap_or_else(|e| panic!("Failed to load model: {e}"));
    log::info!("Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let dataset = iara_train::TextDataset::from_token_file(data_path, seq_len);

    let t_fwd = std::time::Instant::now();
    let loss = iara_train::train::trainer::Trainer::sanity_check(&weights, &dataset)
        .unwrap_or_else(|e| panic!("Sanity check failed: {e}"));
    let elapsed = t_fwd.elapsed().as_secs_f32();

    println!("\nSanity check complete:");
    println!("  Forward pass time: {:.2}s", elapsed);
    println!("  CE loss on first batch: {:.4}", loss);
    println!("  (Expected ~10-12 for random tokens on vocab_size=151936)");
    println!("  (Expected ~5-8 for real text on a pretrained model)");
}
