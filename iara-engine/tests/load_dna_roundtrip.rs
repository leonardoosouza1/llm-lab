//! DNA V4 load roundtrip: write a minimal 1-layer .dna via DnaWriter,
//! then load it via CausalLM::load_dna and verify tensor resolution + layer count.
//!
//! Requires an actual compute backend (rocm or opencl). Gated by feature flags
//! so the test compiles only when a backend is available.

#![cfg(any(feature = "rocm", feature = "opencl"))]

use std::path::PathBuf;
use std::sync::Arc;

use compute_core::ComputeBackend;
use iara_engine::model::architecture::{
    Activation, MlpType, ModelArchitecture, NormType, TensorNameMap,
};
use iara_engine::model::{CausalLM, LayerWeight, ModelCfg};
use iara_engine::storage::dna::DnaWriter;

fn make_backend() -> Arc<dyn ComputeBackend> {
    #[cfg(feature = "rocm")]
    {
        let be = rocm_backend::RocmBackend::probe()
            .expect("ROCm backend probe failed; ensure /dev/kfd is accessible");
        return Arc::new(be);
    }
    #[cfg(all(feature = "opencl", not(feature = "rocm")))]
    {
        let be = opencl_backend::OpenClBackend::probe()
            .expect("OpenCL backend probe failed");
        return Arc::new(be);
    }
}

/// Synthetic 1-layer architecture for tests. All dims tiny.
fn tiny_cfg() -> ModelCfg {
    ModelCfg {
        hidden_size: 32,
        num_layers: 1,
        intermediate_size: 64,
        vocab_size: 16,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 8,
        rms_eps: 1e-6,
        rope_theta: 10000.0,
        max_seq_len: 32,
    }
}

fn tiny_arch() -> ModelArchitecture {
    ModelArchitecture {
        family: "test-tiny".into(),
        norm_type: NormType::RMSNorm,
        activation: Activation::SiLU,
        mlp_type: MlpType::GatedProjection,
        has_qk_norm: false,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_ids: vec![0],
        tensor_map: TensorNameMap::standard(),
    }
}

/// Write a minimal 1-layer .dna with 32x32 projections (rank=8 for attn, rank=8 for MLP)
/// plus TRAW embed, lm_head, norms.
fn write_tiny_dna(path: &std::path::Path) {
    let cfg = tiny_cfg();
    let h = cfg.hidden_size as u32;
    let inter = cfg.intermediate_size as u32;
    let vocab = cfg.vocab_size as u32;
    let q_dim = (cfg.num_heads * cfg.head_dim) as u32;
    let kv_dim = (cfg.num_kv_heads * cfg.head_dim) as u32;
    let rank_attn: u32 = 8;
    let rank_mlp: u32 = 8;

    // Helper: bf16 zeros of size N
    let zeros = |n: usize| -> Vec<u16> { vec![0u16; n] };

    let mut w = DnaWriter::create(path).expect("create .dna");

    // embed_tokens [vocab, hidden] as TRAW BF16
    w.write_traw_bf16(
        "model.embed_tokens.weight",
        &[vocab, h],
        &zeros((vocab * h) as usize),
    ).unwrap();

    // Layer 0 projections — all TSVD
    // wq: [q_dim, h]
    w.write_tsvd(
        "model.layers.0.self_attn.q_proj.weight",
        q_dim, h, rank_attn,
        &zeros((q_dim * rank_attn) as usize),
        &zeros((rank_attn * h) as usize),
    ).unwrap();
    w.write_tsvd(
        "model.layers.0.self_attn.k_proj.weight",
        kv_dim, h, rank_attn,
        &zeros((kv_dim * rank_attn) as usize),
        &zeros((rank_attn * h) as usize),
    ).unwrap();
    w.write_tsvd(
        "model.layers.0.self_attn.v_proj.weight",
        kv_dim, h, rank_attn,
        &zeros((kv_dim * rank_attn) as usize),
        &zeros((rank_attn * h) as usize),
    ).unwrap();
    w.write_tsvd(
        "model.layers.0.self_attn.o_proj.weight",
        h, q_dim, rank_attn,
        &zeros((h * rank_attn) as usize),
        &zeros((rank_attn * q_dim) as usize),
    ).unwrap();
    w.write_tsvd(
        "model.layers.0.mlp.gate_proj.weight",
        inter, h, rank_mlp,
        &zeros((inter * rank_mlp) as usize),
        &zeros((rank_mlp * h) as usize),
    ).unwrap();
    w.write_tsvd(
        "model.layers.0.mlp.up_proj.weight",
        inter, h, rank_mlp,
        &zeros((inter * rank_mlp) as usize),
        &zeros((rank_mlp * h) as usize),
    ).unwrap();
    w.write_tsvd(
        "model.layers.0.mlp.down_proj.weight",
        h, inter, rank_mlp,
        &zeros((h * rank_mlp) as usize),
        &zeros((rank_mlp * inter) as usize),
    ).unwrap();

    // Norms — TRAW BF16 (loader accepts BF16 from take_full; final_norm requires AnyTensor)
    w.write_traw_bf16("model.layers.0.input_layernorm.weight", &[h], &zeros(h as usize)).unwrap();
    w.write_traw_bf16("model.layers.0.post_attention_layernorm.weight", &[h], &zeros(h as usize)).unwrap();
    w.write_traw_bf16("model.norm.weight", &[h], &zeros(h as usize)).unwrap();

    // lm_head [vocab, h] TRAW BF16 (not tied)
    w.write_traw_bf16("lm_head.weight", &[vocab, h], &zeros((vocab * h) as usize)).unwrap();

    w.finalize().unwrap();
}

#[test]
fn load_dna_roundtrip_tiny() {
    let _ = env_logger::builder().is_test(true).try_init();

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let dna_path: PathBuf = tmp.path().to_path_buf();
    write_tiny_dna(&dna_path);

    let backend = make_backend();
    let cfg = tiny_cfg();
    let arch = tiny_arch();

    let model = CausalLM::load_dna(&dna_path, cfg.clone(), arch, backend)
        .expect("load_dna should succeed");

    // Structural checks
    assert_eq!(model.layers.len(), cfg.num_layers, "layer count");
    let l0 = &model.layers[0];

    // All 7 projections must be SVD
    for (name, lw) in [
        ("wq", &l0.wq), ("wk", &l0.wk), ("wv", &l0.wv), ("wo", &l0.wo),
        ("wgate", &l0.wgate), ("wup", &l0.wup), ("wdown", &l0.wdown),
    ] {
        match lw {
            LayerWeight::Svd { rank, .. } => assert_eq!(*rank, 8, "{name} rank"),
            other => panic!("{name} expected Svd, got {:?}", std::mem::discriminant(other)),
        }
    }

    // Norms exist and have expected numel
    assert_eq!(l0.input_norm_w.numel(), cfg.hidden_size);
    assert_eq!(l0.post_attn_norm_w.numel(), cfg.hidden_size);
    assert_eq!(model.norm_w.numel(), cfg.hidden_size);

    // lm_head resolved
    match &model.lm_head {
        LayerWeight::Bf16(t) => assert_eq!(t.numel(), cfg.vocab_size * cfg.hidden_size),
        _ => panic!("lm_head should be Bf16 (Full)"),
    }
}
