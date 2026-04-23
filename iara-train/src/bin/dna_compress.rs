//! dna_compress — convert a safetensors model to `.dna` V4 via Thin SVD.
//!
//! Usage:
//!   dna_compress --input model.safetensors --output model.dna --rank 64 \
//!                [--polish-steps 50] [--polish-lr 0.01]
//!
//! Per-tensor rank selection:
//!   - Default rank from --rank
//!   - Suffix overrides: embed/lm_head = max(rank, 256); q/k/v_proj = max(rank, 128)
//!   - Small tensors (< 2D or any dim < 32) stored as TRAW (no SVD)

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use compute_core::dna_format::DnaWriter;
use half::bf16;
use log::{info, warn};
use memmap2::Mmap;
use safetensors::{tensor::TensorView, SafeTensors};

use iara_train::distill::dna::{
    compress_tensor, default_rank_policy, polish_svd, HostBf16Matrix, RankPolicy,
};

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
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    // Zero-copy mmap input: safetensors views point into mmap'd pages on demand,
    // so peak RAM is driven by the SVD on a single tensor, not the whole model.
    let file = fs::File::open(&cli.input)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap).map_err(|e| anyhow::anyhow!("safetensors: {e}"))?;
    let policy = default_rank_policy(cli.rank);

    let mut writer = DnaWriter::create(&cli.output)?;
    let (mut n_svd, mut n_raw) = (0usize, 0usize);
    let (mut bytes_in, mut bytes_out) = (0u64, 0u64);

    // Iterate tensor names only — materialize each view inside the loop so intermediate
    // buffers (f32_data, HostBf16Matrix, nalgebra SVD scratch, factors) drop before next iter.
    let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    let total = names.len();

    for (idx, name) in names.iter().enumerate() {
        let view = st.tensor(name).map_err(|e| anyhow::anyhow!("tensor {name}: {e}"))?;
        let numel: usize = view.shape().iter().product();
        bytes_in += view.data().len() as u64;

        let is_2d = view.shape().len() == 2;
        let small = numel < cli.min_numel_for_svd || view.shape().iter().any(|&d| d < 32);

        if !is_2d || small {
            write_raw(&mut writer, name, &view)?;
            n_raw += 1;
            continue;
        }

        let rows = view.shape()[0];
        let cols = view.shape()[1];
        let rank = policy.rank_for(name).min(rows).min(cols);

        // Per-tensor peak RAM (F32): W_f32=rows*cols*4 + SVD workspace ≈ ~3-4x.
        let est_mb = (rows * cols * 16) as f64 / 1e6;
        info!("[{}/{}] {name} {rows}x{cols} rank={rank} (~{:.0} MB peak)", idx + 1, total, est_mb);

        let f32_data = view_to_f32(&view);
        let w = HostBf16Matrix::from_f32(&f32_data, rows, cols);
        drop(f32_data);

        let mut factors = compress_tensor(&w, rank);
        info!("  rmse={:.4e}", factors.residual_rmse);

        if cli.polish_steps > 0 {
            let before = factors.residual_rmse;
            let after = polish_svd(&mut factors, &w, cli.polish_steps, cli.polish_lr);
            info!("  polish: {:.4e} → {:.4e}", before, after);
        }

        writer.write_tsvd(
            name,
            rows as u32,
            cols as u32,
            rank as u32,
            &factors.a_bf16,
            &factors.b_bf16,
        )?;
        n_svd += 1;
        bytes_out += 16 + (factors.a_bf16.len() as u64 + factors.b_bf16.len() as u64) * 2;
        // factors, w, view drop here — memory reclaimed before next iter.
    }

    writer.finalize()?;

    let meta = fs::metadata(&cli.output)?;
    let ratio = (bytes_out as f64) / (bytes_in.max(1) as f64);
    info!(
        "wrote {} ({} tensors: {} SVD + {} raw). compression: {:.1}% of original ({:.2} MB → {:.2} MB)",
        cli.output.display(),
        n_svd + n_raw,
        n_svd,
        n_raw,
        ratio * 100.0,
        bytes_in as f64 / 1024.0 / 1024.0,
        meta.len() as f64 / 1024.0 / 1024.0,
    );
    Ok(())
}

fn write_raw(w: &mut DnaWriter, name: &str, view: &TensorView) -> anyhow::Result<()> {
    let shape: Vec<u32> = view.shape().iter().map(|&d| d as u32).collect();
    let dtype = view.dtype();
    let data = view.data();
    match dtype {
        safetensors::Dtype::BF16 | safetensors::Dtype::F16 => {
            let u16s: Vec<u16> = data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            w.write_traw_bf16(name, &shape, &u16s)?;
        }
        safetensors::Dtype::F32 => {
            let f32s: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            w.write_traw_f32(name, &shape, &f32s)?;
        }
        other => {
            warn!("{name}: unsupported dtype {:?}, storing as raw BF16 zeros (placeholder)", other);
            let numel: usize = view.shape().iter().product();
            let zeros = vec![0u16; numel];
            w.write_traw_bf16(name, &shape, &zeros)?;
        }
    }
    Ok(())
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

// Suppress unused-import warning on `RankPolicy` — kept for API clarity.
#[allow(dead_code)]
fn _ensure_policy_api_exposed() { let _ = std::any::type_name::<RankPolicy>(); }
