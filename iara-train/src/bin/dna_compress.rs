//! dna_compress — convert a safetensors model to `.dna` V4 via Thin SVD.
//!
//! Parallel worker pool (rayon) computes factors for independent tensors
//! concurrently. A dedicated writer thread serializes output so the .dna
//! file is written in a safe, deterministic order.
//!
//! Usage:
//!   dna_compress --input model.safetensors --output model.dna --rank 64 \
//!                [--polish-steps 50] [--polish-lr 0.01] [--threads N]
//!
//! Per-tensor rank selection:
//!   - Default rank from --rank
//!   - Suffix overrides: embed/lm_head = max(rank, 256); q/k/v_proj = max(rank, 128)
//!   - Small tensors (< 2D or any dim < 32) stored as TRAW (no SVD)
//!   - embed_tokens / lm_head always Full (SVD breaks row-select semantics)

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use clap::Parser;
use compute_core::dna_format::{quantize_q4_0, DnaWriter, Q4_0_BLOCK};
use compute_core::ComputeBackend;
use half::bf16;
use log::{info, warn};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;

use iara_train::distill::dna::{
    compress_tensor, compress_tensor_randomized, compress_tensor_rsvd_gpu_raw,
    default_rank_policy, polish_svd, HostBf16Matrix, RankPolicy,
};

/// Aggregates one or more safetensors shards under a unified tensor-name API.
///
/// Single file: opens directly. Sharded directory (with `model.safetensors.index.json`):
/// mmaps every shard, leaks them to `'static`, and routes `tensor(name)` to the right shard.
struct ShardedSt {
    names: Vec<String>,
    by_name: HashMap<String, &'static SafeTensors<'static>>,
}

impl ShardedSt {
    fn open(input: &std::path::Path) -> anyhow::Result<Self> {
        // Decide single vs sharded.
        let (shard_paths, parent): (Vec<PathBuf>, PathBuf) = if input.is_dir() {
            let idx = input.join("model.safetensors.index.json");
            let s = fs::read_to_string(&idx)
                .map_err(|e| anyhow::anyhow!("read {}: {}", idx.display(), e))?;
            let v: serde_json::Value = serde_json::from_str(&s)?;
            let wm = v["weight_map"].as_object()
                .ok_or_else(|| anyhow::anyhow!("index.json missing weight_map"))?;
            let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for (_, val) in wm.iter() {
                if let Some(f) = val.as_str() { set.insert(f.to_string()); }
            }
            (set.into_iter().map(|f| input.join(f)).collect(), input.to_path_buf())
        } else if input.extension().and_then(|e| e.to_str()) == Some("json") {
            let s = fs::read_to_string(input)?;
            let v: serde_json::Value = serde_json::from_str(&s)?;
            let wm = v["weight_map"].as_object()
                .ok_or_else(|| anyhow::anyhow!("index.json missing weight_map"))?;
            let par = input.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();
            let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for (_, val) in wm.iter() {
                if let Some(f) = val.as_str() { set.insert(f.to_string()); }
            }
            (set.into_iter().map(|f| par.join(f)).collect(), par)
        } else {
            (vec![input.to_path_buf()], input.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf())
        };

        let _ = parent;
        info!("Opening {} safetensors shard(s)", shard_paths.len());

        let mut names: Vec<String> = Vec::new();
        let mut by_name: HashMap<String, &'static SafeTensors<'static>> = HashMap::new();
        for p in &shard_paths {
            let file = fs::File::open(p)
                .map_err(|e| anyhow::anyhow!("open {}: {}", p.display(), e))?;
            let mmap = unsafe { Mmap::map(&file)? };
            // Leak mmap to 'static so SafeTensors can borrow with that lifetime.
            let mmap_static: &'static Mmap = Box::leak(Box::new(mmap));
            let st = SafeTensors::deserialize(&mmap_static[..])
                .map_err(|e| anyhow::anyhow!("safetensors {}: {}", p.display(), e))?;
            let st_static: &'static SafeTensors<'static> = Box::leak(Box::new(st));
            for n in st_static.names() {
                if by_name.insert(n.to_string(), st_static).is_none() {
                    names.push(n.to_string());
                }
            }
        }
        Ok(Self { names, by_name })
    }

    fn names(&self) -> &[String] { &self.names }

    fn tensor(&self, name: &str) -> anyhow::Result<TensorView<'static>> {
        let st = self.by_name.get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in any shard", name))?;
        st.tensor(name).map_err(|e| anyhow::anyhow!("tensor '{}': {}", name, e))
    }
}

#[derive(Parser, Debug)]
#[command(about = "Compress safetensors → DNA V4 (.dna)")]
struct Cli {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

    /// Default rank for SVD factorization.
    #[arg(long, default_value_t = 64)]
    rank: usize,

    /// Polish steps per tensor (0 = skip polish).
    #[arg(long, default_value_t = 0)]
    polish_steps: usize,

    /// Polish learning rate.
    #[arg(long, default_value_t = 0.01)]
    polish_lr: f32,

    /// Skip tensors smaller than this many elements (store as TRAW).
    #[arg(long, default_value_t = 1024)]
    min_numel_for_svd: usize,

    /// Use randomized SVD (Halko) for tensors above this numel threshold.
    #[arg(long, default_value_t = 1_000_000)]
    randomized_threshold: usize,

    /// Randomized SVD oversampling.
    #[arg(long, default_value_t = 10)]
    rsvd_oversample: usize,

    /// Power iterations for randomized SVD.
    #[arg(long, default_value_t = 2)]
    rsvd_power_iters: usize,

    /// Force full (deterministic) SVD for all tensors regardless of size.
    #[arg(long)]
    force_full_svd: bool,

    /// Worker threads for parallel compression (0 = rayon default = num CPUs).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Disable GPU probing and force CPU-only compression.
    #[arg(long)]
    cpu_only: bool,

    /// Numel threshold above which to prefer GPU rsvd (when available).
    /// Small tensors stay on CPU (rayon-parallel) to amortize GPU dispatch cost.
    #[arg(long, default_value_t = 100_000)]
    gpu_threshold: usize,

    /// Storage dtype for SVD factors. "bf16" (default) or "q4_0".
    /// q4_0: block=32, fp16 scale + 16 nibbles → 4.5 bpw (~3.5× smaller than bf16).
    #[arg(long, default_value = "bf16")]
    dtype: String,
}

/// Message sent from compute workers to the writer thread.
enum WriteMsg {
    Tsvd {
        name: String,
        rows: u32,
        cols: u32,
        rank: u32,
        a_bf16: Vec<u16>,
        b_bf16: Vec<u16>,
    },
    TsvdQ4 {
        name: String,
        rows: u32,
        cols: u32,
        rank: u32,
        a_q4: Vec<u8>,
        b_q4: Vec<u8>,
    },
    TrawBf16 {
        name: String,
        shape: Vec<u32>,
        data: Vec<u16>,
    },
    TrawF32 {
        name: String,
        shape: Vec<u32>,
        data: Vec<f32>,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok(); // may already be initialized — ignore
    }
    let nthreads = rayon::current_num_threads();
    info!("Parallel compression with {} worker threads", nthreads);

    let use_q4 = match cli.dtype.as_str() {
        "bf16" => false,
        "q4_0" | "q4" | "Q4_0" => true,
        other => anyhow::bail!("--dtype must be 'bf16' or 'q4_0', got '{other}'"),
    };
    if use_q4 {
        info!("Storage dtype: Q4_0 (block=32, fp16 scale + 16 nibbles = 4.5 bpw)");
    }

    // Probe GPU backend for full-GPU rsvd path. Falls back to CPU rayon if absent.
    // why: hipBLAS/hipSOLVER handles are single-queue per context. Wrap in a mutex
    //      so rayon workers serialize the GPU section while still parallelizing
    //      the bf16→f32 conversion and I/O build before/after.
    let gpu_backend: Option<std::sync::Arc<std::sync::Mutex<rocm_backend::RocmBackend>>> =
        if cli.cpu_only {
            info!("--cpu-only set; skipping GPU probe");
            None
        } else {
            match rocm_backend::RocmBackend::probe() {
                Ok(b) => {
                    info!("ROCm GPU detected — using full-GPU rsvd for tensors with numel ≥ {}", cli.gpu_threshold);
                    Some(std::sync::Arc::new(std::sync::Mutex::new(b)))
                }
                Err(e) => {
                    warn!("No ROCm GPU ({e}). Falling back to CPU rsvd.");
                    None
                }
            }
        };

    let st = ShardedSt::open(&cli.input)?;

    // Detect tied word embeddings from sibling config.json. If tied and the
    // safetensors lacks an explicit lm_head, duplicate embed bytes under
    // "lm_head.weight" in the .dna so the engine loader doesn't need a tie
    // resolver. Disk cost: one extra TRAW for a ~500MB embed.
    let config_dir: PathBuf = if cli.input.is_dir() {
        cli.input.clone()
    } else {
        cli.input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
    };
    let config_path = config_dir.join("config.json");
    let tied_embeddings = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("tie_word_embeddings").and_then(|b| b.as_bool()))
        .unwrap_or(false);
    let safetensors_has_lm_head = st.names().iter().any(|n|
        *n == "lm_head.weight" || n.ends_with(".lm_head.weight"));
    let synth_lm_head = tied_embeddings && !safetensors_has_lm_head;
    if synth_lm_head {
        info!("Tied embeddings detected; will write lm_head.weight as a duplicate of embed_tokens.");
    }
    let policy = default_rank_policy(cli.rank);

    let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    let total = names.len();
    info!("Found {} tensors in {}", total, cli.input.display());

    // ── Writer thread: single producer of file writes ─────────────────────
    let output_path = cli.output.clone();
    let (tx, rx) = mpsc::channel::<WriteMsg>();
    let writer_handle = std::thread::spawn(move || -> anyhow::Result<(usize, usize, u64)> {
        let mut writer = DnaWriter::create(&output_path)?;
        let (mut n_svd, mut n_raw, mut bytes_out) = (0usize, 0usize, 0u64);
        while let Ok(msg) = rx.recv() {
            match msg {
                WriteMsg::Tsvd { name, rows, cols, rank, a_bf16, b_bf16 } => {
                    bytes_out += 16 + (a_bf16.len() as u64 + b_bf16.len() as u64) * 2;
                    writer.write_tsvd(&name, rows, cols, rank, &a_bf16, &b_bf16)?;
                    n_svd += 1;
                }
                WriteMsg::TsvdQ4 { name, rows, cols, rank, a_q4, b_q4 } => {
                    bytes_out += 16 + a_q4.len() as u64 + b_q4.len() as u64;
                    writer.write_tsvd_q4(&name, rows, cols, rank, &a_q4, &b_q4)?;
                    n_svd += 1;
                }
                WriteMsg::TrawBf16 { name, shape, data } => {
                    bytes_out += 12 + data.len() as u64 * 2;
                    writer.write_traw_bf16(&name, &shape, &data)?;
                    n_raw += 1;
                }
                WriteMsg::TrawF32 { name, shape, data } => {
                    bytes_out += 12 + data.len() as u64 * 4;
                    writer.write_traw_f32(&name, &shape, &data)?;
                    n_raw += 1;
                }
            }
        }
        writer.finalize()?;
        Ok((n_svd, n_raw, bytes_out))
    });

    // ── Parallel compute: factor each tensor on a worker ──────────────────
    let t0 = Instant::now();
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let bytes_in = std::sync::atomic::AtomicU64::new(0);

    // Closure per tensor: parse → factor (GPU if available) → send to writer.
    // Raw/non-SVD tensors go through rayon (CPU-bound byte shuffling).
    // SVD candidates go through sequential GPU loop below (single hipBLAS/hipSOLVER queue).
    let process_raw = |name: &String, tx: &mpsc::Sender<WriteMsg>| -> Option<(usize, usize, usize)> {
        let view = match st.tensor(name) {
            Ok(v) => v,
            Err(e) => { warn!("tensor '{name}' parse error: {e}"); return None; }
        };
        let shape: Vec<u32> = view.shape().iter().map(|&d| d as u32).collect();
        let numel: usize = view.shape().iter().product();
        bytes_in.fetch_add(view.data().len() as u64, std::sync::atomic::Ordering::Relaxed);

        let is_2d = view.shape().len() == 2;
        let small = numel < cli.min_numel_for_svd || view.shape().iter().any(|&d| d < 32);
        // Force raw storage for tensors where SVD truncation breaks inference quality.
        // - embed_tokens/lm_head: SVD breaks row-select semantics (any rank loses info)
        // - attention v_proj: low-magnitude weights without bias (Qwen3 has no qkv bias)
        //   are dominated by SVD residual → garbage logits even at rank=256.
        //   Tested: V proj rank=256 yielded 42% L2 drift on Qwen3-4B layer 0; with TRAW
        //   the drift drops to 0 (BF16 round-trip only).
        // For Qwen2.5 with q/k/v_bias, the bias absorbed the SVD residual, so SVD was
        // viable there. Qwen3 removed bias → must store raw.
        let force_full = name.ends_with(".embed_tokens.weight")
                       || name.ends_with("lm_head.weight")
                       || name == "embed_tokens.weight"
                       // Attention projections: SVD residual accumulates across
                       // 36 layers + propagates through residual stream. Without QKV
                       // bias (Qwen3 has attention_bias=false), residual error grows
                       // exponentially. Tested: v_proj alone fixed L0 attn_out exactly
                       // but pre_final_norm still 7× smaller than safetensors due to
                       // q/k/o residuals through MLP path. Storing all attention raw
                       // recovers correctness at modest disk cost.
                       || name.ends_with(".q_proj.weight")
                       || name.ends_with(".k_proj.weight")
                       || name.ends_with(".v_proj.weight")
                       || name.ends_with(".o_proj.weight")
                       // MLP gate/up/down: SVD truncation accumulates across 36 layers
                       // → repetition collapse (e.g. "ooksooks...") even at rank 512.
                       // Until polish-based or lossless quantization arrives, store raw.
                       || name.ends_with(".gate_proj.weight")
                       || name.ends_with(".up_proj.weight")
                       || name.ends_with(".down_proj.weight");
        let is_raw = !is_2d || small || force_full;

        if is_raw {
            if let Some(msg) = build_raw_msg(name, &shape, &view) { tx.send(msg).ok(); }
            let done = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % 25 == 0 || done == total {
                info!("progress {done}/{total} ({:.0}s elapsed)", t0.elapsed().as_secs_f32());
            }
            return None;
        }
        Some((view.shape()[0], view.shape()[1], numel))
    };

    // Partition: raw in rayon (parallel), SVD candidates collected for sequential GPU pass.
    let svd_candidates: Vec<&String> = names.par_iter().filter_map({
        let tx = tx.clone();
        move |name| {
            match process_raw(name, &tx) {
                None => None,
                Some(_) => Some(name),
            }
        }
    }).collect();
    info!("Raw tensors done; {} SVD candidates pending GPU pass", svd_candidates.len());

    let backend = gpu_backend.as_ref()
        .ok_or_else(|| anyhow::anyhow!("GPU required for SVD pass — pass --cpu-only to disable SVD entirely"))?
        .clone();

    for name in svd_candidates {
        let view = st.tensor(name).map_err(|e| anyhow::anyhow!("tensor parse {name}: {e}"))?;
        let rows = view.shape()[0];
        let cols = view.shape()[1];
        let numel = rows * cols;
        let rank = policy.rank_for(name).min(rows).min(cols);

        let seed = name.bytes().fold(0u64, |h, b| h.wrapping_mul(1315423911).wrapping_add(b as u64));
        let t_tensor = Instant::now();
        let use_gpu = numel >= cli.gpu_threshold && !cli.force_full_svd
                     && view.dtype() == safetensors::Dtype::BF16;

        let mut factors = if use_gpu {
            // Zero-copy path: reinterpret mmap bytes as &[u16] and upload directly.
            // bf16 is little-endian on all supported targets; bytes align to 2.
            let bytes = view.data();
            debug_assert_eq!(bytes.len(), numel * 2);
            let bf16_slice: &[u16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, numel)
            };
            let guard = backend.lock().expect("gpu backend mutex");
            let r = compress_tensor_rsvd_gpu_raw(&*guard, bf16_slice, rows, cols, rank, cli.rsvd_oversample, seed)
                .map_err(|e| anyhow::anyhow!("{name}: GPU rsvd failed: {e:?}"))?;
            guard.sync().ok();
            r
        } else {
            let f32_data = view_to_f32(&view);
            let w = HostBf16Matrix::from_f32(&f32_data, rows, cols);
            drop(f32_data);
            if cli.force_full_svd {
                compress_tensor(&w, rank)
            } else {
                compress_tensor_randomized(&w, rank, cli.rsvd_oversample, cli.rsvd_power_iters, seed)
            }
        };
        let method = if use_gpu { "gpu-rsvd" } else if cli.force_full_svd { "full" } else { "rsvd" };
        log::debug!("{name} {rows}x{cols} rank={rank} [{method}] rmse={:.4e} in {:.2}s",
            factors.residual_rmse, t_tensor.elapsed().as_secs_f32());

        if cli.polish_steps > 0 {
            let f32_data = view_to_f32(&view);
            let w = HostBf16Matrix::from_f32(&f32_data, rows, cols);
            drop(f32_data);
            polish_svd(&mut factors, &w, cli.polish_steps, cli.polish_lr);
        }

        if use_q4 {
            // bf16 factors → f32 → q4_0. Q4 block=32; enforce elem counts divide 32.
            let a_numel = rows * rank;
            let b_numel = rank * cols;
            if a_numel % Q4_0_BLOCK != 0 || b_numel % Q4_0_BLOCK != 0 {
                warn!("{name}: elem counts not multiples of 32 (rows*rank={a_numel}, rank*cols={b_numel}); \
                       falling back to bf16 for this tensor");
                tx.send(WriteMsg::Tsvd {
                    name: name.to_string(),
                    rows: rows as u32, cols: cols as u32, rank: rank as u32,
                    a_bf16: factors.a_bf16, b_bf16: factors.b_bf16,
                }).ok();
            } else {
                let a_f32: Vec<f32> = factors.a_bf16.iter()
                    .map(|&b| bf16::from_bits(b).to_f32()).collect();
                let b_f32: Vec<f32> = factors.b_bf16.iter()
                    .map(|&b| bf16::from_bits(b).to_f32()).collect();
                let a_q4 = quantize_q4_0(&a_f32);
                let b_q4 = quantize_q4_0(&b_f32);
                tx.send(WriteMsg::TsvdQ4 {
                    name: name.to_string(),
                    rows: rows as u32, cols: cols as u32, rank: rank as u32,
                    a_q4, b_q4,
                }).ok();
            }
        } else {
            tx.send(WriteMsg::Tsvd {
                name: name.to_string(),
                rows: rows as u32, cols: cols as u32, rank: rank as u32,
                a_bf16: factors.a_bf16, b_bf16: factors.b_bf16,
            }).ok();
        }

        let done = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if done % 25 == 0 || done == total {
            info!("progress {done}/{total} ({:.0}s elapsed)", t0.elapsed().as_secs_f32());
        }
    }

    // Synthetic lm_head duplicate for tied embeddings.
    if synth_lm_head {
        let embed_name = names.iter().find(|n|
            n.ends_with(".embed_tokens.weight") || *n == "embed_tokens.weight"
        ).cloned();
        if let Some(en) = embed_name {
            let view = st.tensor(&en).map_err(|e| anyhow::anyhow!("re-read embed for lm_head: {e}"))?;
            let shape: Vec<u32> = view.shape().iter().map(|&d| d as u32).collect();
            if let Some(msg) = build_raw_msg("lm_head.weight", &shape, &view) {
                tx.send(msg).ok();
                info!("Wrote synthetic lm_head.weight (tied to {en})");
            }
        } else {
            warn!("tie_word_embeddings=true but no embed_tokens found in safetensors");
        }
    }

    // Drop remaining tx so the writer thread's recv loop exits.
    drop(tx);
    let (n_svd, n_raw, bytes_out) = writer_handle.join()
        .map_err(|_| anyhow::anyhow!("writer thread panicked"))??;

    let bytes_in = bytes_in.load(std::sync::atomic::Ordering::Relaxed);
    let meta = fs::metadata(&cli.output)?;
    let ratio = (bytes_out as f64) / (bytes_in.max(1) as f64);
    info!(
        "wrote {} ({} tensors: {} SVD + {} raw) in {:.1}s. compression: {:.1}% of original ({:.2} MB → {:.2} MB)",
        cli.output.display(),
        n_svd + n_raw, n_svd, n_raw,
        t0.elapsed().as_secs_f32(),
        ratio * 100.0,
        bytes_in as f64 / 1024.0 / 1024.0,
        meta.len() as f64 / 1024.0 / 1024.0,
    );
    Ok(())
}

fn build_raw_msg(name: &str, shape: &[u32], view: &TensorView) -> Option<WriteMsg> {
    let data = view.data();
    // why: the inference engine loads 1D tensors (norms, biases) as F32 on device
    //      and panics with "expected F32 DeviceBuffer" if we hand it BF16. Widen
    //      here; they're tiny (hidden-size scale), so cost is negligible.
    let widen_to_f32 = shape.len() == 1;
    match view.dtype() {
        safetensors::Dtype::BF16 | safetensors::Dtype::F16 if widen_to_f32 => {
            let is_bf16 = view.dtype() == safetensors::Dtype::BF16;
            let f32s: Vec<f32> = data.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                if is_bf16 { f32::from_bits((bits as u32) << 16) }
                else { half::f16::from_bits(bits).to_f32() }
            }).collect();
            Some(WriteMsg::TrawF32 { name: name.into(), shape: shape.to_vec(), data: f32s })
        }
        safetensors::Dtype::BF16 | safetensors::Dtype::F16 => {
            let u16s: Vec<u16> = data.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Some(WriteMsg::TrawBf16 { name: name.into(), shape: shape.to_vec(), data: u16s })
        }
        safetensors::Dtype::F32 => {
            let f32s: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            Some(WriteMsg::TrawF32 { name: name.into(), shape: shape.to_vec(), data: f32s })
        }
        other => {
            warn!("{name}: unsupported dtype {:?}, storing as raw BF16 zeros (placeholder)", other);
            let numel: usize = view.shape().iter().product();
            let zeros = vec![0u16; numel];
            Some(WriteMsg::TrawBf16 { name: name.into(), shape: shape.to_vec(), data: zeros })
        }
    }
}

fn view_to_f32(view: &TensorView) -> Vec<f32> {
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        safetensors::Dtype::BF16 => data.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        safetensors::Dtype::F16 => data.chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        other => panic!("unsupported dtype for SVD: {:?}", other),
    }
}

#[allow(dead_code)]
fn _ensure_policy_api_exposed() { let _ = std::any::type_name::<RankPolicy>(); }
