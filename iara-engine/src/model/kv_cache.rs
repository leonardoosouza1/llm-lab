// ============================================================================
// kv_cache.rs — KV-Cache com janela deslizante (bounded sliding window)
// ============================================================================
// GpuKvCache: buffer pré-alocado via ComputeBackend (AnyTensor).
// BoundedKvCache: cache CPU-only, inalterado.
// ============================================================================

use std::sync::Arc;
use compute_core::{AnyTensor, ComputeBackend};

/// Cache de K ou V em GPU — buffer pré-alocado via ComputeBackend.
/// Layout: [max_seq_len, kv_dim] F32.
pub struct GpuKvCache {
    /// Buffer no device: [max_seq_len, kv_dim] F32.
    pub data: AnyTensor,
    #[allow(dead_code)]
    pub max_seq_len: usize,
    #[allow(dead_code)]
    pub kv_dim: usize,
    /// Linhas válidas (tokens já escritos no cache).
    pub current_len: usize,
}

impl GpuKvCache {
    pub fn new(
        backend: &Arc<dyn ComputeBackend>,
        max_seq_len: usize,
        kv_dim: usize,
    ) -> Self {
        let data = backend.alloc_f32(&[max_seq_len, kv_dim]);
        Self { data, max_seq_len, kv_dim, current_len: 0 }
    }

    pub fn reset(&mut self) { self.current_len = 0; }
}

