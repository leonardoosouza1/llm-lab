//! IARA Training Engine — LoRA fine-tuning + knowledge distillation
//!
//! Usage:
//!   cargo run -p iara-train --release -- \
//!     --model-dir models/qwen2.5-1.5b \
//!     --data data/train.bin \
//!     --steps 100 --lr 1e-4 --lora-rank 32

use clap::Parser;
use std::path::{Path, PathBuf};

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::tensor::{Int, Tensor};

use iara_train::model::config::Qwen2Config;
use iara_train::model::qwen2::Qwen2ForCausalLM;
use iara_train::train::trainer::{TrainConfig, Trainer};
use iara_train::train::dataset::TextDataset;

// Backend type aliases
type TrainBackend = Autodiff<Wgpu>;
type InnerBackend = Wgpu;

#[derive(Parser, Debug)]
#[command(name = "iara-train", about = "IARA Training Engine — LoRA fine-tuning + distillation")]
struct Args {
    /// Path to model directory (must contain config.json + model.safetensors)
    #[arg(long, default_value = "models/qwen2.5-1.5b")]
    model_dir: String,

    /// Pre-tokenized training data (.bin file of u32 LE token IDs)
    #[arg(long, default_value = "data/train.bin")]
    data: String,

    /// LoRA rank (0 = no LoRA, full fine-tune — NOT recommended for 1.5B)
    #[arg(long, default_value_t = 32)]
    lora_rank: usize,

    /// LoRA alpha scaling
    #[arg(long, default_value_t = 64.0)]
    lora_alpha: f64,

    /// Learning rate
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,

    /// Number of training steps
    #[arg(long, default_value_t = 100)]
    steps: usize,

    /// Batch size
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Max sequence length per training sample
    #[arg(long, default_value_t = 128)]
    seq_len: usize,

    /// Output directory for checkpoints
    #[arg(long, default_value = "checkpoints/qwen2.5-1.5b")]
    output_dir: String,

    /// Teacher GGUF model for distillation (optional)
    #[arg(long)]
    teacher: Option<String>,

    /// Distillation temperature
    #[arg(long, default_value_t = 2.0)]
    distill_temp: f64,

    /// Distillation alpha (1.0 = pure CE, 0.0 = pure KL)
    #[arg(long, default_value_t = 0.5)]
    distill_alpha: f64,
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args = Args::parse();

    println!("================================================");
    println!("  IARA Training Engine v0.1.0");
    println!("  Pure Rust + burn-wgpu (Vulkan) on AMD GPU");
    println!("================================================");
    println!("  Model:    {}", args.model_dir);
    println!("  Data:     {}", args.data);
    println!("  LoRA:     rank={}, alpha={}", args.lora_rank, args.lora_alpha);
    println!("  Training: steps={}, lr={}, batch={}, seq_len={}",
             args.steps, args.lr, args.batch_size, args.seq_len);
    println!("  Output:   {}", args.output_dir);
    println!();

    // Initialize WGPU device (Vulkan backend on AMD)
    let device = WgpuDevice::default();

    // Load model config
    let model_dir = Path::new(&args.model_dir);
    let config = Qwen2Config::from_json(&model_dir.join("config.json"))
        .expect("Failed to load config.json");

    log::info!(
        "Model config: {} layers, hidden={}, intermediate={}, vocab={}",
        config.num_hidden_layers, config.hidden_size,
        config.intermediate_size, config.vocab_size,
    );

    // Load pre-trained weights from safetensors
    let st_path = model_dir.join("model.safetensors");
    let model: Qwen2ForCausalLM<TrainBackend> = config
        .load_from_safetensors(&st_path, &device)
        .expect("Failed to load model weights");

    log::info!("Model loaded successfully. Param count: ~{}M",
        (config.hidden_size * config.vocab_size +  // embed + lm_head
         config.num_hidden_layers * (
             4 * config.hidden_size * config.hidden_size +  // QKV+O (approx)
             3 * config.hidden_size * config.intermediate_size +  // gate+up+down
             2 * config.hidden_size  // norms
         )) / 1_000_000
    );

    // TODO: Apply LoRA to attention layers
    // For now, train the full model (will OOM on large models)

    // Training config
    let train_config = TrainConfig {
        lr: args.lr,
        num_steps: args.steps,
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        checkpoint_every: 50,
        output_dir: PathBuf::from(&args.output_dir),
        distill_alpha: args.distill_alpha,
        distill_temperature: args.distill_temp,
        lora_rank: args.lora_rank,
        ..Default::default()
    };

    // Load dataset
    let data_path = Path::new(&args.data);
    let dataset = TextDataset::from_token_file(data_path, args.seq_len);

    // Forward function closure
    let forward_fn = |model: &Qwen2ForCausalLM<TrainBackend>,
                      input_ids: Tensor<TrainBackend, 2, Int>|
        -> Tensor<TrainBackend, 3> {
        model.forward(input_ids, false).logits
    };

    // Run training
    let trainer = Trainer::<TrainBackend>::new(train_config, device);
    let trained_model = trainer.train(
        model,
        dataset,
        forward_fn,
        None::<fn(&Tensor<TrainBackend, 2, Int>) -> Tensor<TrainBackend, 3>>,
    );

    log::info!("Training complete!");
    // TODO: Export trained weights to safetensors → GGUF
}
