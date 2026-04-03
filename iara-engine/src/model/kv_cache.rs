// ============================================================================
// kv_cache.rs — KV-Cache com janela deslizante (bounded sliding window)
// ============================================================================
// GpuKvCache: buffer pré-alocado via ComputeBackend (AnyTensor).
// BoundedKvCache: cache CPU-only, inalterado.
// ============================================================================

use std::sync::Arc;
use compute_core::{AnyTensor, ComputeBackend};

/// Modo de armazenamento do KV-cache
#[derive(Debug, Clone, Copy)]
pub enum KvCacheMode {
    /// Precisão total: [max_seq, kv_dim] F32
    F32,
    /// Quantizado INT8: reduz memória 4x com per-head scales
    Int8,
}

impl KvCacheMode {
    /// Detecta modo a partir de variável de ambiente IARA_KV_INT8
    pub fn detect() -> Self {
        if std::env::var("IARA_KV_INT8").is_ok() {
            Self::Int8
        } else {
            Self::F32
        }
    }
}

/// Cache de K ou V em GPU — buffer pré-alocado via ComputeBackend.
/// Suporta dois modos: F32 completo ou INT8 quantizado (4x menor).
pub struct GpuKvCache {
    pub mode: KvCacheMode,
    /// Buffer de dados: F32 ou INT8 dependendo do modo
    pub data: AnyTensor,
    /// Scales para modo INT8: [max_seq_len, n_kv_heads] F32
    pub scales: Option<AnyTensor>,
    #[allow(dead_code)]
    pub max_seq_len: usize,
    #[allow(dead_code)]
    pub kv_dim: usize,
    /// Número de heads (necessário para INT8 quantize/dequant)
    #[allow(dead_code)]
    pub n_kv_heads: usize,
    /// Dimensão por head (necessário para INT8 quantize/dequant)
    #[allow(dead_code)]
    pub head_dim: usize,
    /// Linhas válidas (tokens já escritos no cache).
    pub current_len: usize,
}

impl GpuKvCache {
    pub fn new(
        backend: &Arc<dyn ComputeBackend>,
        max_seq_len: usize,
        kv_dim: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let mode = KvCacheMode::detect();

        let data = match mode {
            KvCacheMode::F32 => backend.alloc_f32(&[max_seq_len, kv_dim]),
            // INT8: armazenar como u8 (1 byte por elemento)
            KvCacheMode::Int8 => backend.alloc_q4(&[max_seq_len, kv_dim]),
        };

        // Scales: [max_seq_len, n_kv_heads] F32 para modo INT8
        let scales = match mode {
            KvCacheMode::F32 => None,
            KvCacheMode::Int8 => Some(backend.alloc_f32(&[max_seq_len, n_kv_heads])),
        };

        Self {
            mode,
            data,
            scales,
            max_seq_len,
            kv_dim,
            n_kv_heads,
            head_dim,
            current_len: 0,
        }
    }

    pub fn reset(&mut self) { self.current_len = 0; }
}

