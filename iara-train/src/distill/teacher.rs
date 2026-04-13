//! Teacher model wrapper for knowledge distillation via llama.cpp FFI.
//!
//! Loads a GGUF model through the llama-sys bindings and runs inference to
//! produce per-position logits. These logits serve as soft targets for the
//! student model during distillation training.
//!
//! # Usage
//!
//! ```ignore
//! let mut teacher = TeacherModel::load("models/teacher-7b.gguf", 2048)?;
//! let logits: Vec<Vec<f32>> = teacher.get_logits(&[1, 2, 3, 4])?;
//! // logits[i] has vocab_size entries for position i
//! ```
//!
//! The teacher runs entirely through llama.cpp (possibly GPU-offloaded),
//! independent of the burn compute graph. Its logits are detached constants
//! fed into the KL-divergence loss.

use llama_sys::*;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// One-time backend initialization
// ---------------------------------------------------------------------------

static BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

fn ensure_backend_init() {
    if !BACKEND_INITIALIZED.swap(true, Ordering::SeqCst) {
        unsafe { llama_backend_init() };
        log::info!("[teacher] llama.cpp backend initialized");
    }
}

// ---------------------------------------------------------------------------
// TeacherModel
// ---------------------------------------------------------------------------

/// A GGUF model loaded via llama.cpp, used as teacher for distillation.
///
/// The teacher produces logits for every input position in a single forward
/// pass (prefill). These logits are then used as soft targets in the KL
/// divergence loss.
pub struct TeacherModel {
    model: *mut llama_model,
    ctx: *mut llama_context,
    #[allow(dead_code)]
    vocab: *const llama_vocab,
    n_vocab: usize,
    n_ctx: u32,
}

// Safety: llama.cpp models/contexts are not thread-safe by default.
// We ensure single-threaded access through &mut self on get_logits.
unsafe impl Send for TeacherModel {}

impl Drop for TeacherModel {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        }
        log::info!("[teacher] model resources freed");
    }
}

impl TeacherModel {
    /// Load a teacher model from a GGUF file.
    ///
    /// # Arguments
    ///
    /// * `gguf_path` — Path to the GGUF model file.
    /// * `n_ctx` — Maximum context length. Should be >= the training seq_len.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the model file cannot be loaded or the context
    /// cannot be created.
    pub fn load(gguf_path: &str, n_ctx: u32) -> Result<Self, String> {
        ensure_backend_init();

        let c_path = CString::new(gguf_path).map_err(|e| format!("invalid path: {e}"))?;

        log::info!("[teacher] loading model from {gguf_path}");
        let t0 = std::time::Instant::now();

        // Model params: offload all layers to GPU if available
        let mut model_params = unsafe { llama_model_default_params() };
        model_params.n_gpu_layers = 99;

        let model = unsafe { llama_model_load_from_file(c_path.as_ptr(), model_params) };
        if model.is_null() {
            return Err(format!("failed to load teacher model: {gguf_path}"));
        }

        // Context params: we need logits for ALL positions (not just last)
        let mut ctx_params = unsafe { llama_context_default_params() };
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx; // process entire sequence in one batch
        ctx_params.n_ubatch = n_ctx;
        ctx_params.n_seq_max = 1;
        // We need logits at every position for distillation, not just the last
        // This is handled by the batch setup below (logits=true per position)

        let ctx = unsafe { llama_init_from_model(model, ctx_params) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            return Err("failed to create teacher context".into());
        }

        let vocab = unsafe { llama_model_get_vocab(model) };
        let n_vocab = unsafe { llama_vocab_n_tokens(vocab) } as usize;

        let load_ms = t0.elapsed().as_millis();
        let n_params = unsafe { llama_model_n_params(model) };
        log::info!(
            "[teacher] loaded in {}ms: {}M params, vocab={}, n_ctx={}",
            load_ms, n_params / 1_000_000, n_vocab, n_ctx,
        );

        Ok(Self {
            model,
            ctx,
            vocab,
            n_vocab,
            n_ctx,
        })
    }

    /// Run the teacher on a sequence of tokens and return logits for each position.
    ///
    /// # Arguments
    ///
    /// * `tokens` — Input token IDs (length must be <= n_ctx).
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<f32>>` of shape `[seq_len][vocab_size]`, where `result[i]`
    /// contains the logits after processing `tokens[0..=i]`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the sequence is too long or if llama.cpp decode fails.
    pub fn get_logits(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Ok(vec![]);
        }
        if seq_len as u32 > self.n_ctx {
            return Err(format!(
                "sequence length {seq_len} exceeds teacher n_ctx {}",
                self.n_ctx
            ));
        }

        // Clear KV cache before each forward pass
        let mem = unsafe { llama_get_memory(self.ctx) };
        unsafe { llama_memory_clear(mem, false) };

        // Build a batch with logits=true for every position
        let mut batch = unsafe { llama_batch_init(seq_len as i32, 0, 1) };

        unsafe {
            for (i, &tok) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok as llama_token;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                *(*batch.seq_id.add(i)) = 0;
                *batch.logits.add(i) = 1; // request logits at every position
            }
            batch.n_tokens = seq_len as i32;
        }

        // Decode the entire sequence in one call
        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret != 0 {
            unsafe { llama_batch_free(batch) };
            return Err(format!("teacher decode failed with code {ret}"));
        }

        // Extract logits for each position
        let mut all_logits = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, i as i32) };
            if logits_ptr.is_null() {
                unsafe { llama_batch_free(batch) };
                return Err(format!("null logits at position {i}"));
            }

            let logits_slice = unsafe { std::slice::from_raw_parts(logits_ptr, self.n_vocab) };
            all_logits.push(logits_slice.to_vec());
        }

        unsafe { llama_batch_free(batch) };

        Ok(all_logits)
    }

    /// Get the vocabulary size of the teacher model.
    pub fn vocab_size(&self) -> usize {
        self.n_vocab
    }

    /// Get the maximum context length.
    pub fn context_length(&self) -> u32 {
        self.n_ctx
    }

    /// Pre-compute teacher logits for an entire dataset and write them to a
    /// binary file. Each position's logits are stored as `vocab_size` f32
    /// values in little-endian order.
    ///
    /// The output file layout is:
    /// ```text
    /// [header]   u32 n_tokens, u32 vocab_size
    /// [data]     n_tokens * vocab_size * f32 (little-endian)
    /// ```
    ///
    /// This allows the training loop to memory-map teacher logits instead of
    /// running inference on-the-fly.
    pub fn precompute_to_file(
        &mut self,
        tokens: &[u32],
        output_path: &std::path::Path,
        chunk_size: usize,
    ) -> Result<(), String> {
        use std::io::Write;

        let n_tokens = tokens.len();
        let vocab_size = self.n_vocab;

        let mut file = std::fs::File::create(output_path)
            .map_err(|e| format!("cannot create {}: {e}", output_path.display()))?;

        // Write header
        file.write_all(&(n_tokens as u32).to_le_bytes())
            .map_err(|e| format!("write error: {e}"))?;
        file.write_all(&(vocab_size as u32).to_le_bytes())
            .map_err(|e| format!("write error: {e}"))?;

        log::info!(
            "[teacher] precomputing logits for {} tokens in chunks of {} -> {}",
            n_tokens, chunk_size, output_path.display(),
        );

        // Process in chunks to fit within n_ctx
        let effective_chunk = chunk_size.min(self.n_ctx as usize);
        let mut processed = 0usize;

        while processed < n_tokens {
            let end = (processed + effective_chunk).min(n_tokens);
            let chunk_tokens = &tokens[processed..end];

            let chunk_logits = self.get_logits(chunk_tokens)?;

            for position_logits in &chunk_logits {
                for &val in position_logits {
                    file.write_all(&val.to_le_bytes())
                        .map_err(|e| format!("write error: {e}"))?;
                }
            }

            processed = end;

            if processed % 1000 < effective_chunk {
                log::info!(
                    "[teacher] precomputed {}/{} positions ({:.1}%)",
                    processed, n_tokens, processed as f64 / n_tokens as f64 * 100.0,
                );
            }
        }

        file.flush().map_err(|e| format!("flush error: {e}"))?;

        let file_size = std::fs::metadata(output_path)
            .map(|m| m.len())
            .unwrap_or(0);
        log::info!(
            "[teacher] precomputation done: {} positions, {:.1} MB",
            n_tokens, file_size as f64 / 1_048_576.0,
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pre-computed teacher logits loader (memory-mapped)
// ---------------------------------------------------------------------------

/// Memory-mapped reader for pre-computed teacher logits.
///
/// Created by [`TeacherModel::precompute_to_file`], this struct provides
/// zero-copy access to teacher logits during training.
pub struct PrecomputedTeacherLogits {
    _mmap: memmap2::Mmap,
    data_ptr: *const f32,
    n_tokens: usize,
    vocab_size: usize,
}

// Safety: the mmap is read-only and the pointer is derived from it.
unsafe impl Send for PrecomputedTeacherLogits {}
unsafe impl Sync for PrecomputedTeacherLogits {}

impl PrecomputedTeacherLogits {
    /// Open a pre-computed logits file.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| format!("mmap failed on {}: {e}", path.display()))?
        };

        if mmap.len() < 8 {
            return Err("precomputed logits file too small (no header)".into());
        }

        let n_tokens = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
        let vocab_size = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;

        let expected_size = 8 + n_tokens * vocab_size * 4;
        if mmap.len() < expected_size {
            return Err(format!(
                "precomputed logits file too small: expected {} bytes for {}x{}, got {}",
                expected_size, n_tokens, vocab_size, mmap.len(),
            ));
        }

        let data_ptr = unsafe { mmap.as_ptr().add(8) as *const f32 };

        log::info!(
            "[teacher] loaded precomputed logits: {} tokens, vocab_size={}, {:.1} MB",
            n_tokens, vocab_size, mmap.len() as f64 / 1_048_576.0,
        );

        Ok(Self {
            _mmap: mmap,
            data_ptr,
            n_tokens,
            vocab_size,
        })
    }

    /// Get logits for a single position.
    ///
    /// Returns a slice of `vocab_size` f32 values.
    pub fn get_position(&self, pos: usize) -> Option<&[f32]> {
        if pos >= self.n_tokens {
            return None;
        }
        let offset = pos * self.vocab_size;
        Some(unsafe { std::slice::from_raw_parts(self.data_ptr.add(offset), self.vocab_size) })
    }

    /// Get logits for a range of positions.
    ///
    /// Returns a flat slice of `(end - start) * vocab_size` f32 values.
    pub fn get_range(&self, start: usize, end: usize) -> Option<&[f32]> {
        if start >= self.n_tokens || end > self.n_tokens || start >= end {
            return None;
        }
        let offset = start * self.vocab_size;
        let len = (end - start) * self.vocab_size;
        Some(unsafe { std::slice::from_raw_parts(self.data_ptr.add(offset), len) })
    }

    /// Total number of positions in the file.
    pub fn num_tokens(&self) -> usize {
        self.n_tokens
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
