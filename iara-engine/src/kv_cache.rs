// ============================================================================
// kv_cache.rs — KV-Cache com janela deslizante (bounded sliding window)
// ============================================================================
// Mantém os últimos `max_seq_len` tokens de K ou V para uma camada.
// Memória alocada UMA vez na carga do modelo — zero alocação durante geração.
// Quando o cache está cheio, o token mais antigo é descartado via copy_within.
// ============================================================================

/// Cache de K ou V para uma camada, com capacidade fixa.
///
/// Layout: buffer plano row-major `[seq, kv_dim]`.
/// Linha i = `data[i*kv_dim .. (i+1)*kv_dim]`.
#[derive(Debug, Clone)]
pub struct BoundedKvCache {
    /// Buffer pré-alocado — nunca cresce após `new()`.
    data: Vec<f32>,
    /// Máximo de linhas (tokens) que o cache pode conter.
    pub max_seq_len: usize,
    /// Floats por linha (n_kv_heads × head_dim).
    pub kv_dim: usize,
    /// Linhas válidas atualmente (0..=max_seq_len).
    pub current_len: usize,
}

impl BoundedKvCache {
    /// Cria um cache com capacidade `max_seq_len` tokens.
    /// Pré-aloca `max_seq_len * kv_dim * 4` bytes.
    pub fn new(max_seq_len: usize, kv_dim: usize) -> Self {
        Self {
            data: vec![0.0f32; max_seq_len * kv_dim],
            max_seq_len,
            kv_dim,
            current_len: 0,
        }
    }

    /// Adiciona um ou mais tokens ao cache.
    ///
    /// `new_data` deve ter comprimento `n_tokens * kv_dim`.
    /// Se o append ultrapassar `max_seq_len`, os tokens mais antigos são
    /// descartados (sliding window): mantém os últimos `max_seq_len - 1`.
    pub fn append(&mut self, new_data: &[f32]) {
        debug_assert_eq!(
            new_data.len() % self.kv_dim, 0,
            "new_data deve ser múltiplo de kv_dim"
        );

        for chunk in new_data.chunks_exact(self.kv_dim) {
            if self.current_len < self.max_seq_len {
                // Ainda há espaço: escreve na próxima linha vazia.
                let dst = self.current_len * self.kv_dim;
                self.data[dst..dst + self.kv_dim].copy_from_slice(chunk);
                self.current_len += 1;
            } else {
                // Cache cheio: descarta o token mais antigo com memmove.
                // copy_within é um memmove seguro (overlapping ok).
                // TODO: ring buffer para evitar O(max_seq_len) por token.
                self.data.copy_within(self.kv_dim.., 0);
                let dst = (self.max_seq_len - 1) * self.kv_dim;
                self.data[dst..dst + self.kv_dim].copy_from_slice(chunk);
                // current_len permanece em max_seq_len
            }
        }
    }

    /// Retorna uma slice sobre todas as linhas válidas: shape `[current_len, kv_dim]`.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.current_len * self.kv_dim]
    }

    /// Número de tokens válidos no cache.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    /// Zera o contador sem desalocar memória.
    /// O buffer pré-alocado é reutilizado na próxima geração.
    #[inline]
    pub fn reset(&mut self) {
        self.current_len = 0;
    }
}
