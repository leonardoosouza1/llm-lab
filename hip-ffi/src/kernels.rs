//! Safe wrappers for HIP LLM inference kernels.
//!
//! Kernels are compiled from `kernels/llm_kernels.hip` by `build.rs` using
//!   hipcc --offload-arch=gfx1032 -O3 -o $OUT_DIR/llm_kernels.hsaco kernels/llm_kernels.hip
//!
//! The resulting `.hsaco` (HIP Shader Archive Code Object) is embedded at
//! compile-time via `include_bytes!`.  The embedding will only succeed after
//! `build.rs` has run, which is always the case in a normal `cargo build`.
//!
//! # Usage
//! ```no_run
//! let k = GpuKernels::new().unwrap();
//! unsafe { k.embed_lookup(embed_gpu, ids_gpu, out_gpu, n_tokens, hidden).unwrap(); }
//! ```

#![allow(clippy::too_many_arguments)]

use crate::raw::{hipFunction_t, hipModule_t};
use crate::safe::{
    hip_launch_kernel, hip_module_get_function, hip_module_load_data, hip_module_unload, HipError,
};
use std::ptr;

// ---------------------------------------------------------------------------
// Embedded kernel binary
// ---------------------------------------------------------------------------

/// The compiled kernel object, embedded at build time.
/// `build.rs` is responsible for producing `$OUT_DIR/llm_kernels.hsaco`.
static KERNEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/llm_kernels.hsaco"));

// ---------------------------------------------------------------------------
// GpuKernels — loaded module with cached function handles
// ---------------------------------------------------------------------------

/// Holds the loaded HIP module and one `hipFunction_t` handle per kernel.
/// Construct once with [`GpuKernels::new`] and reuse across inference calls.
///
/// # Thread safety
/// The struct is `Send + Sync` because HIP function handles are valid to use
/// from any thread after the module is loaded, provided the caller serialises
/// concurrent launches (e.g. via a Mutex or by using separate streams).
pub struct GpuKernels {
    module: hipModule_t,
    fn_embed_lookup: hipFunction_t,
    fn_rms_norm: hipFunction_t,
    fn_add_bias: hipFunction_t,
    fn_rope_inplace: hipFunction_t,
    fn_silu_mul: hipFunction_t,
    fn_softmax_rows: hipFunction_t,
    fn_copy_kv_cache: hipFunction_t,
    fn_f32_to_bf16: hipFunction_t,
    fn_dequant_gemv_q4: hipFunction_t,
    fn_dequant_gemv_q4_naive: hipFunction_t,
    use_naive_gemv: bool,
    fn_gravity_score: hipFunction_t,
    fn_cfc_gate_update: hipFunction_t,
    fn_value_norms: hipFunction_t,
    fn_kv_cache_quantize: hipFunction_t,
    fn_add_tensors: hipFunction_t,
    fn_fused_norm_bias_rope: hipFunction_t,
    fn_kv_cache_int8_quantize_append: hipFunction_t,
    fn_kv_cache_int8_dequantize: hipFunction_t,
    fn_gather_rows_bf16: hipFunction_t,
    fn_gather_cols_bf16: hipFunction_t,
    fn_kuramoto_coherence: hipFunction_t,
    fn_topk_select: hipFunction_t,
    fn_fused_attention_decode: hipFunction_t,
}

// SAFETY: hipModule_t / hipFunction_t are opaque pointers; HIP guarantees
// they are valid across threads once loaded.
unsafe impl Send for GpuKernels {}
unsafe impl Sync for GpuKernels {}

impl GpuKernels {
    /// Load the embedded kernel binary and resolve all function handles.
    ///
    /// Returns `Err(HipError)` if the module fails to load (e.g. no GPU
    /// present, wrong GFX arch) or any function name is not found.
    pub fn new() -> Result<Self, HipError> {
        let module = hip_module_load_data(KERNEL_BYTES)?;

        macro_rules! get_fn {
            ($name:literal) => {
                hip_module_get_function(module, $name)?
            };
        }

        let fn_embed_lookup = get_fn!("embed_lookup");
        let fn_rms_norm = get_fn!("rms_norm");
        let fn_add_bias = get_fn!("add_bias_inplace");
        let fn_rope_inplace = get_fn!("rope_inplace");
        let fn_silu_mul = get_fn!("silu_mul_inplace");
        let fn_softmax_rows = get_fn!("softmax_rows");
        let fn_copy_kv_cache = get_fn!("copy_to_kv_cache");
        let fn_f32_to_bf16 = get_fn!("f32_to_bf16");
        let fn_dequant_gemv_q4 = get_fn!("dequant_gemv_q4");
        let fn_dequant_gemv_q4_naive = get_fn!("dequant_gemv_q4_naive");
        let use_naive_gemv = std::env::var("IARA_NAIVE_GEMV").map(|v| v == "1").unwrap_or(false);
        if use_naive_gemv {
            eprintln!("⚠️  IARA_NAIVE_GEMV=1: usando kernel GEMV Q4 naive (debug mode)");
        }
        let fn_gravity_score = get_fn!("gravity_score");
        let fn_cfc_gate_update = get_fn!("cfc_gate_update");
        let fn_value_norms = get_fn!("value_norms");
        let fn_kv_cache_quantize = get_fn!("kv_cache_quantize_append");
        let fn_add_tensors = get_fn!("add_tensors_inplace");
        let fn_fused_norm_bias_rope = get_fn!("fused_norm_bias_rope");
        let fn_kv_cache_int8_quantize_append = get_fn!("kv_cache_int8_quantize_append");
        let fn_kv_cache_int8_dequantize = get_fn!("kv_cache_int8_dequantize");
        let fn_gather_rows_bf16 = get_fn!("gather_rows_bf16");
        let fn_gather_cols_bf16 = get_fn!("gather_cols_bf16");
        let fn_kuramoto_coherence = get_fn!("kuramoto_coherence");
        let fn_topk_select = get_fn!("topk_select");
        let fn_fused_attention_decode = get_fn!("fused_attention_decode");

        Ok(Self {
            module,
            fn_embed_lookup,
            fn_rms_norm,
            fn_add_bias,
            fn_rope_inplace,
            fn_silu_mul,
            fn_softmax_rows,
            fn_copy_kv_cache,
            fn_f32_to_bf16,
            fn_dequant_gemv_q4,
            fn_dequant_gemv_q4_naive,
            use_naive_gemv,
            fn_gravity_score,
            fn_cfc_gate_update,
            fn_value_norms,
            fn_kv_cache_quantize,
            fn_add_tensors,
            fn_fused_norm_bias_rope,
            fn_kv_cache_int8_quantize_append,
            fn_kv_cache_int8_dequantize,
            fn_gather_rows_bf16,
            fn_gather_cols_bf16,
            fn_kuramoto_coherence,
            fn_topk_select,
            fn_fused_attention_decode,
        })
    }
}

impl Drop for GpuKernels {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // Unloading may fail only if the module was already unloaded, which
            // cannot happen with this RAII wrapper.  Ignore the result in Drop.
            unsafe {
                let _ = hip_module_unload(self.module);
            }
            self.module = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build a kernel-argument pointer array from references.
//
// ---------------------------------------------------------------------------
// Inline helpers
// ---------------------------------------------------------------------------

/// Round up integer division.
#[inline(always)]
fn ceil_div(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// Clamp a dimension to the nearest power-of-two block size ≤ max_block.
/// Falls back to `min_block` if `dim` is smaller than it.
#[inline(always)]
fn block_size(dim: u32, max_block: u32) -> u32 {
    let mut b = 64u32; // RDNA2 wavefront; minimum sensible block
    while b * 2 <= dim.min(max_block) {
        b *= 2;
    }
    b
}

// ---------------------------------------------------------------------------
// Kernel wrappers
// ---------------------------------------------------------------------------

impl GpuKernels {
    // -------------------------------------------------------------------------
    // Kernel 1: embed_lookup
    //
    // Copy embedding rows for a batch of token IDs.
    //   embed  : [vocab, hidden]  BF16 (stored as u16) on GPU
    //   token_ids: [n_tokens]     u32 on GPU
    //   out    : [n_tokens, hidden] f32 on GPU (output)
    //
    // Grid : (n_tokens, 1, 1)
    // Block: (min(hidden, 256), 1, 1)
    // -------------------------------------------------------------------------

    /// Copy BF16 embedding rows for `n_tokens` tokens into an F32 output buffer.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn embed_lookup(
        &self,
        embed_bf16: *const u16, // [vocab, hidden] BF16 on GPU
        token_ids: *const u32,  // [n_tokens] on GPU
        out: *mut f32,          // [n_tokens, hidden] on GPU
        n_tokens: i32,
        hidden: i32,
    ) -> Result<(), HipError> {
        let bsz = block_size(hidden as u32, 256);
        let grid = (n_tokens as u32, 1, 1);
        let block = (bsz, 1, 1);

        // Mutable copies so we can take &mut for the args array.
        let mut p_embed = embed_bf16;
        let mut p_token_ids = token_ids;
        let mut p_out = out;
        let mut p_hidden = hidden;

        hip_launch_kernel(
            self.fn_embed_lookup,
            grid,
            block,
            0,               // no dynamic shared memory
            ptr::null_mut(), // default stream
            &mut [
                &mut p_embed as *mut *const u16 as *mut std::ffi::c_void,
                &mut p_token_ids as *mut *const u32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_hidden as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 2: rms_norm
    //
    // RMSNorm: out[tok, i] = (x[tok, i] / rms(x[tok])) * w[i]
    //
    // Grid : (n_tokens, 1, 1)
    // Block: (min(hidden, 1024), 1, 1)  — must be power of 2
    // Shared: block_size * sizeof(f32) bytes
    // -------------------------------------------------------------------------

    /// Apply RMSNorm: `out = (x / rms(x)) * w`.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn rms_norm(
        &self,
        out: *mut f32, // [n_tokens, hidden] on GPU
        x: *const f32, // [n_tokens, hidden] on GPU
        w: *const f32, // [hidden] on GPU
        n_tokens: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<(), HipError> {
        let bsz = block_size(hidden as u32, 1024);
        let shared_bytes = bsz * std::mem::size_of::<f32>() as u32;

        let grid = (n_tokens as u32, 1, 1);
        let block = (bsz, 1, 1);

        let mut p_out = out;
        let mut p_x = x;
        let mut p_w = w;
        let mut p_hidden = hidden;
        let mut p_eps = eps;

        hip_launch_kernel(
            self.fn_rms_norm,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_x as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_w as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_hidden as *mut i32 as *mut std::ffi::c_void,
                &mut p_eps as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 3: add_bias_inplace
    //
    // x[tok, d] += bias[d]
    //
    // Grid : (n_tokens, 1, 1)
    // Block: (min(dim, 1024), 1, 1)
    // -------------------------------------------------------------------------

    /// Add a bias vector to every row of `x` in-place.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn add_bias_inplace(
        &self,
        x: *mut f32,      // [n_tokens, dim] on GPU
        bias: *const f32, // [dim] on GPU
        n_tokens: i32,
        dim: i32,
    ) -> Result<(), HipError> {
        let bsz = block_size(dim as u32, 1024);
        let grid = (n_tokens as u32, 1, 1);
        let block = (bsz, 1, 1);

        let mut p_x = x;
        let mut p_bias = bias;
        let mut p_dim = dim;

        hip_launch_kernel(
            self.fn_add_bias,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_x as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_bias as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_dim as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 4: rope_inplace
    //
    // Apply RoPE in-place using Qwen "halves" convention.
    //   x : [n_tokens, n_heads, head_dim]
    //
    // Grid : (n_tokens, n_heads, 1)
    // Block: (head_dim / 2, 1, 1)  — max 128 for head_dim=256
    // -------------------------------------------------------------------------

    /// Apply Rotary Position Embeddings in-place to Q or K.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn rope_inplace(
        &self,
        x: *mut f32, // [n_tokens, n_heads, head_dim] on GPU
        n_tokens: i32,
        n_heads: i32,
        head_dim: i32,
        pos_start: i32,
        theta: f32,
    ) -> Result<(), HipError> {
        let half = (head_dim / 2) as u32;
        let grid = (n_tokens as u32, n_heads as u32, 1);
        let block = (half, 1, 1);

        let mut p_x = x;
        let mut p_n_heads = n_heads;
        let mut p_head_dim = head_dim;
        let mut p_pos_start = pos_start;
        let mut p_theta = theta;

        hip_launch_kernel(
            self.fn_rope_inplace,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_x as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_n_heads as *mut i32 as *mut std::ffi::c_void,
                &mut p_head_dim as *mut i32 as *mut std::ffi::c_void,
                &mut p_pos_start as *mut i32 as *mut std::ffi::c_void,
                &mut p_theta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 5: silu_mul_inplace
    //
    // SwiGLU: gate[i] = silu(gate[i]) * up[i]
    //   gate, up : [n_tokens, inter]
    //
    // Grid : (ceil(n_tokens * inter / 256), 1, 1)
    // Block: (256, 1, 1)
    // -------------------------------------------------------------------------

    /// Apply SwiGLU activation: `gate[i] = silu(gate[i]) * up[i]`.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn silu_mul_inplace(
        &self,
        gate: *mut f32, // [n_tokens, inter] on GPU — modified in-place
        up: *const f32, // [n_tokens, inter] on GPU
        n_tokens: i32,
        inter: i32,
    ) -> Result<(), HipError> {
        let total = (n_tokens * inter) as u32;
        let block_x: u32 = 256;
        let grid = (ceil_div(total, block_x), 1, 1);
        let block = (block_x, 1, 1);

        let mut p_gate = gate;
        let mut p_up = up;
        let mut p_total = (n_tokens * inter) as i32;

        hip_launch_kernel(
            self.fn_silu_mul,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_gate as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_up as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_total as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 6: softmax_rows
    //
    // Numerically-stable softmax over each row of [n_rows, seq_len].
    // One block per row (one attention head × one query token).
    //
    // Grid : (n_rows, 1, 1)
    // Block: (min(next_pow2(seq_len), 1024), 1, 1)
    // Shared: block_size * sizeof(f32) bytes
    // -------------------------------------------------------------------------

    /// Apply row-wise softmax to `scores[n_rows, seq_len]` in-place.
    ///
    /// # Safety
    /// The `scores` pointer must point to a valid GPU allocation of
    /// `n_rows * seq_len` f32 elements.
    pub unsafe fn softmax_rows(
        &self,
        scores: *mut f32, // [n_rows, seq_len] on GPU
        n_rows: i32,
        seq_len: i32,
    ) -> Result<(), HipError> {
        // Choose block size = next power-of-two >= seq_len, capped at 1024.
        let bsz = {
            let mut b = 64u32;
            while b < (seq_len as u32).min(1024) {
                b <<= 1;
            }
            b.min(1024)
        };
        let shared_bytes = bsz * std::mem::size_of::<f32>() as u32;

        let grid = (n_rows as u32, 1, 1);
        let block = (bsz, 1, 1);

        let mut p_scores = scores;
        let mut p_seq_len = seq_len;

        hip_launch_kernel(
            self.fn_softmax_rows,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_scores as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq_len as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 7: copy_to_kv_cache
    //
    // Append new_kv rows to cache starting at `current_len`.
    //   cache   : [max_seq, kv_dim]
    //   new_kv  : [n_tokens, kv_dim]
    //
    // Grid : (1, 1, 1)  — one block; strided loop inside kernel
    // Block: (min(n_tokens * kv_dim, 1024), 1, 1)
    // -------------------------------------------------------------------------

    /// Append `n_tokens` rows from `new_kv` into the GPU KV-cache at
    /// row offset `current_len`.
    ///
    /// # Safety
    /// `cache` must have capacity for at least `current_len + n_tokens` rows
    /// of `kv_dim` elements.  `new_kv` must contain exactly `n_tokens * kv_dim`
    /// f32 elements.
    pub unsafe fn copy_to_kv_cache(
        &self,
        cache: *mut f32,    // [max_seq, kv_dim] on GPU
        new_kv: *const f32, // [n_tokens, kv_dim] on GPU
        n_tokens: i32,
        kv_dim: i32,
        current_len: i32,
    ) -> Result<(), HipError> {
        let total = (n_tokens * kv_dim) as u32;
        let block_x = total.min(1024);
        let grid = (1, 1, 1);
        let block = (block_x, 1, 1);

        let mut p_cache = cache;
        let mut p_new_kv = new_kv;
        let mut p_n_tokens = n_tokens;
        let mut p_kv_dim = kv_dim;
        let mut p_current_len = current_len;

        hip_launch_kernel(
            self.fn_copy_kv_cache,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_cache as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_new_kv as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_n_tokens as *mut i32 as *mut std::ffi::c_void,
                &mut p_kv_dim as *mut i32 as *mut std::ffi::c_void,
                &mut p_current_len as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    /// Convert F32 to BF16 (stored as u16)
    pub unsafe fn f32_to_bf16(
        &self,
        src: *const f32,
        dst: *mut u16,
        n: i32,
    ) -> Result<(), HipError> {
        let block_x = block_size(n as u32, 256);
        let grid_x = ceil_div(n as u32, block_x);

        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_n = n;

        hip_launch_kernel(
            self.fn_f32_to_bf16,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            ptr::null_mut(),
            &mut [
                &mut p_src as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_dst as *mut *mut u16 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kernel 8: dequant_gemv_q4
    //
    // GEMV com dequantização INT4 on-the-fly.
    // Pesos comprimidos: 2 valores por byte (4-bit cada).
    // Escalas e zeros-points por grupo para cada linha de saída.
    // Grid : (n, 1, 1)  — um bloco por linha de saída
    // Block: (min(k, 1024), 1, 1)
    // ─────────────────────────────────────────────────────────────────────────

    /// Dequantized GEMV: Q4 packed weights × F32 activations → F32 output.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn dequant_gemv_q4(
        &self,
        packed: *const u8,  // [n, k/2] Q4 comprimido
        scales: *const f32, // [n, k/group_size] F32 escalas
        zeros: *const f32,  // [n, k/group_size] F32 zero-points
        x: *const f32,      // [m, k] F32 ativações
        out: *mut f32,      // [m, n] F32 saída
        m: i32,
        n: i32,
        k: i32,
        group_size: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let bsz = block_size(k as u32, 256);  // max 256 (kernel __launch_bounds__)
        let grid = (n as u32, m.max(1) as u32, 1);  // (N rows, M tokens)
        let block = (bsz, 1, 1);

        // Select kernel and shared memory based on mode
        let (kernel_fn, shared_mem) = if self.use_naive_gemv {
            // Naive kernel: 256 floats for tree reduction
            (self.fn_dequant_gemv_q4_naive, (256 * std::mem::size_of::<f32>()) as u32)
        } else {
            // Optimized kernel: only 8 floats for cross-warp reduction (256/32 = 8 warps)
            (self.fn_dequant_gemv_q4, (8 * std::mem::size_of::<f32>()) as u32)
        };

        let mut p_packed = packed;
        let mut p_scales = scales;
        let mut p_zeros = zeros;
        let mut p_x = x;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_group_size = group_size;
        let mut p_beta = beta;

        hip_launch_kernel(
            kernel_fn,
            grid,
            block,
            shared_mem,
            ptr::null_mut(),
            &mut [
                &mut p_packed as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_scales as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_zeros as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_x as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_group_size as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kernel 9: gravity_score
    //
    // Calcula gravity score por token: média(attn_weights[h,i] * value_norm[h,i])
    // Grid : (seq_len, 1, 1)
    // Block: (n_heads, 1, 1)  — redução dentro do bloco
    // Shared: n_heads * sizeof(f32) bytes
    // ─────────────────────────────────────────────────────────────────────────

    /// Gravity score: semantic importance per token via attention × value norms.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn gravity_score(
        &self,
        attn_weights: *const f32, // [n_heads, seq_len] F32 após softmax
        value_norms: *const f32,  // [n_heads, seq_len] F32 normas L2
        gravity_out: *mut f32,    // [seq_len] F32 saída
        n_heads: i32,
        seq_len: i32,
    ) -> Result<(), HipError> {
        let shared_bytes = (n_heads as u32) * std::mem::size_of::<f32>() as u32;

        let grid = (seq_len as u32, 1, 1);
        let block = (n_heads as u32, 1, 1);

        let mut p_attn_weights = attn_weights;
        let mut p_value_norms = value_norms;
        let mut p_gravity_out = gravity_out;
        let mut p_n_heads = n_heads;
        let mut p_seq_len = seq_len;

        hip_launch_kernel(
            self.fn_gravity_score,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_attn_weights as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_value_norms as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_gravity_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_n_heads as *mut i32 as *mut std::ffi::c_void,
                &mut p_seq_len as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kernel 10: value_norms
    //
    // Calcula norma L2 dos value vectors por token/head: ||v[h,i,:]||₂
    // Grid : (n_kv_heads, seq_len, 1)  — um bloco por (head, token)
    // Block: (min(head_dim, 256), 1, 1)
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute L2 norms of value vectors for gravity scoring.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn value_norms(
        &self,
        values: *const f32,  // [seq_len, n_kv_heads * head_dim] F32
        norms_out: *mut f32, // [n_kv_heads, seq_len] F32 saída
        seq_len: i32,
        n_kv_heads: i32,
        head_dim: i32,
    ) -> Result<(), HipError> {
        let block_x: u32 = (head_dim as u32).min(256);
        let grid = (n_kv_heads as u32, seq_len as u32, 1);
        let block = (block_x, 1, 1);
        let shared_bytes = block_x * std::mem::size_of::<f32>() as u32;

        let mut p_values = values;
        let mut p_norms_out = norms_out;
        let mut p_seq_len = seq_len;
        let mut p_n_kv_heads = n_kv_heads;
        let mut p_head_dim = head_dim;

        hip_launch_kernel(
            self.fn_value_norms,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_values as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_norms_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq_len as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_kv_heads as *mut i32 as *mut std::ffi::c_void,
                &mut p_head_dim as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kernel 11: cfc_gate_update
    //
    // CfC gate update: h_new = sigmoid(f) * h_prev + (1 - sigmoid(f)) * tanh(g)
    // Grid : (1, 1, 1)  — loop interno
    // Block: (256, 1, 1)
    // ─────────────────────────────────────────────────────────────────────────

    /// CfC gated update kernel (in-place on h_out).
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn cfc_gate_update(
        &self,
        f_proj: *const f32, // [cfc_dim] F32 — forget gate
        g_proj: *const f32, // [cfc_dim] F32 — candidate gate
        h_prev: *const f32, // [cfc_dim] F32 — anterior estado
        h_out: *mut f32,    // [cfc_dim] F32 — novo estado (escrita)
        cfc_dim: i32,
    ) -> Result<(), HipError> {
        let block_x = block_size(cfc_dim as u32, 256);
        let grid_x = ceil_div(cfc_dim as u32, block_x);

        let mut p_f_proj = f_proj;
        let mut p_g_proj = g_proj;
        let mut p_h_prev = h_prev;
        let mut p_h_out = h_out;
        let mut p_cfc_dim = cfc_dim;

        hip_launch_kernel(
            self.fn_cfc_gate_update,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            ptr::null_mut(),
            &mut [
                &mut p_f_proj as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_g_proj as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_h_prev as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_h_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_cfc_dim as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kernel 12: kv_cache_quantize_append
    //
    // Append de KV cache com quantização on-the-fly para INT4.
    // Grid : (1, 1, 1)  — loop interno
    // Block: (256, 1, 1)
    // ─────────────────────────────────────────────────────────────────────────

    /// Append quantized KV to cache with on-the-fly INT4 encoding.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn kv_cache_quantize_append(
        &self,
        cache: *mut u8,     // [max_seq, kv_dim/2] Q4 comprimido
        scales: *mut f32,   // [max_seq, kv_dim/group_size] F32
        zeros: *mut f32,    // [max_seq, kv_dim/group_size] F32
        new_kv: *const f32, // [n_new, kv_dim] F32
        insert_pos: i32,
        n_new_tokens: i32,
        kv_dim: i32,
        max_seq: i32,
        group_size: i32,
    ) -> Result<(), HipError> {
        let block_x: u32 = 256;
        let grid = (1, 1, 1);
        let block = (block_x, 1, 1);

        let mut p_cache = cache;
        let mut p_scales = scales;
        let mut p_zeros = zeros;
        let mut p_new_kv = new_kv;
        let mut p_insert_pos = insert_pos;
        let mut p_n_new_tokens = n_new_tokens;
        let mut p_kv_dim = kv_dim;
        let mut p_max_seq = max_seq;
        let mut p_group_size = group_size;

        hip_launch_kernel(
            self.fn_kv_cache_quantize,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_cache as *mut *mut u8 as *mut std::ffi::c_void,
                &mut p_scales as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_zeros as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_new_kv as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_insert_pos as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_new_tokens as *mut i32 as *mut std::ffi::c_void,
                &mut p_kv_dim as *mut i32 as *mut std::ffi::c_void,
                &mut p_max_seq as *mut i32 as *mut std::ffi::c_void,
                &mut p_group_size as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    /// Element-wise in-place addition: a[i] += alpha * b[i] for F32 tensors.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn add_tensors_inplace(
        &self,
        a: *mut f32,      // [n] F32 on GPU (modified in-place)
        b: *const f32,    // [n] F32 on GPU
        n: i32,
        alpha: f32,
    ) -> Result<(), HipError> {
        let block_x: u32 = 256;
        let grid_x = ceil_div(n as u32, block_x);
        let grid = (grid_x, 1, 1);
        let block = (block_x, 1, 1);

        let mut p_a = a;
        let mut p_b = b;
        let mut p_n = n;
        let mut p_alpha = alpha;

        hip_launch_kernel(
            self.fn_add_tensors,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_a as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_b as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_alpha as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 15: fused_norm_bias_rope
    //
    // Fuse RMSNorm + optional bias + RoPE into a single kernel launch.
    //   output[tok, i] = rope( (input[tok, i] / rms + bias[i]) * weight[i] )
    //
    // Grid : (n_tokens, 1, 1)
    // Block: (256, 1, 1)
    // Shared: 256 * sizeof(f32) bytes (for variance reduction)
    // -------------------------------------------------------------------------

    /// Apply RMSNorm + optional bias + RoPE in a single fused kernel.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    /// If bias is None, has_bias must be 0 and bias_ptr can be null.
    pub unsafe fn fused_norm_bias_rope(
        &self,
        output: *mut f32,          // [n_tokens, hidden_size] on GPU
        input: *const f32,         // [n_tokens, hidden_size] on GPU
        norm_weight: *const f32,   // [hidden_size] on GPU
        bias: *const f32,          // [hidden_size] on GPU (may be null if has_bias==0)
        n_tokens: i32,
        hidden_size: i32,
        n_heads: i32,
        pos_start: i32,            // absolute position for RoPE
        theta: f32,                // RoPE base
        eps: f32,                  // epsilon for RMSNorm
        has_bias: i32,             // 1 if bias is valid, 0 otherwise
    ) -> Result<(), HipError> {
        let shared_bytes = 256 * std::mem::size_of::<f32>() as u32;
        let grid = (n_tokens as u32, 1, 1);
        let block = (256u32, 1, 1);

        let mut p_output = output;
        let mut p_input = input;
        let mut p_norm_weight = norm_weight;
        let mut p_bias = bias;
        let mut p_hidden_size = hidden_size;
        let mut p_n_heads = n_heads;
        let mut p_pos_start = pos_start;
        let mut p_theta = theta;
        let mut p_eps = eps;
        let mut p_has_bias = has_bias;

        hip_launch_kernel(
            self.fn_fused_norm_bias_rope,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_output as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_input as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_norm_weight as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_bias as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_hidden_size as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_heads as *mut i32 as *mut std::ffi::c_void,
                &mut p_pos_start as *mut i32 as *mut std::ffi::c_void,
                &mut p_theta as *mut f32 as *mut std::ffi::c_void,
                &mut p_eps as *mut f32 as *mut std::ffi::c_void,
                &mut p_has_bias as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 16: kv_cache_int8_quantize_append
    //
    // Quantize F32 K/V values to INT8 with per-head absolute-max scaling.
    // Appends quantized values to the INT8 KV cache at insert_pos.
    //
    // Grid : (n_new_tokens * n_kv_heads, 1, 1)
    // Block: (256, 1, 1)
    // Shared: 256 * sizeof(f32) bytes (for parallel reduction)
    // -------------------------------------------------------------------------

    /// Quantize and append F32 K/V values to INT8 KV cache with per-head scales.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn kv_cache_int8_quantize_append(
        &self,
        cache_q: *mut i8,          // [max_seq, kv_dim] INT8 on GPU
        cache_scales: *mut f32,    // [max_seq, n_kv_heads] F32 on GPU
        new_kv: *const f32,        // [n_new_tokens, kv_dim] F32 on GPU
        insert_pos: i32,
        n_new_tokens: i32,
        kv_dim: i32,
        head_dim: i32,
        n_kv_heads: i32,
        max_seq: i32,
    ) -> Result<(), HipError> {
        let num_blocks = (n_new_tokens as u32) * (n_kv_heads as u32);
        let shared_bytes = 256 * std::mem::size_of::<f32>() as u32;

        let grid = (num_blocks, 1, 1);
        let block = (256u32, 1, 1);

        let mut p_cache_q = cache_q;
        let mut p_cache_scales = cache_scales;
        let mut p_new_kv = new_kv;
        let mut p_insert_pos = insert_pos;
        let mut p_n_new_tokens = n_new_tokens;
        let mut p_kv_dim = kv_dim;
        let mut p_head_dim = head_dim;
        let mut p_n_kv_heads = n_kv_heads;
        let mut p_max_seq = max_seq;

        hip_launch_kernel(
            self.fn_kv_cache_int8_quantize_append,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_cache_q as *mut *mut i8 as *mut std::ffi::c_void,
                &mut p_cache_scales as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_new_kv as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_insert_pos as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_new_tokens as *mut i32 as *mut std::ffi::c_void,
                &mut p_kv_dim as *mut i32 as *mut std::ffi::c_void,
                &mut p_head_dim as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_kv_heads as *mut i32 as *mut std::ffi::c_void,
                &mut p_max_seq as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kernel 17: kv_cache_int8_dequantize
    //
    // Dequantize INT8 KV cache back to F32 for attention computation.
    //
    // Grid : (seq_len, n_kv_heads, 1)
    // Block: (256, 1, 1)
    // -------------------------------------------------------------------------

    /// Dequantize INT8 KV cache to F32 for attention computation.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn kv_cache_int8_dequantize(
        &self,
        cache_q: *const i8,        // [max_seq, kv_dim] INT8 on GPU
        cache_scales: *const f32,  // [max_seq, n_kv_heads] F32 on GPU
        output: *mut f32,          // [seq_len, kv_dim] F32 on GPU
        seq_len: i32,
        kv_dim: i32,
        head_dim: i32,
        n_kv_heads: i32,
    ) -> Result<(), HipError> {
        let grid = (seq_len as u32, n_kv_heads as u32, 1);
        let block = (256u32, 1, 1);

        let mut p_cache_q = cache_q;
        let mut p_cache_scales = cache_scales;
        let mut p_output = output;
        let mut p_seq_len = seq_len;
        let mut p_kv_dim = kv_dim;
        let mut p_head_dim = head_dim;
        let mut p_n_kv_heads = n_kv_heads;

        hip_launch_kernel(
            self.fn_kv_cache_int8_dequantize,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_cache_q as *mut *const i8 as *mut std::ffi::c_void,
                &mut p_cache_scales as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_output as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq_len as *mut i32 as *mut std::ffi::c_void,
                &mut p_kv_dim as *mut i32 as *mut std::ffi::c_void,
                &mut p_head_dim as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_kv_heads as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Sparse MLP: gather_rows_bf16
    //
    // Gathers selected rows from BF16 weight matrix [total_rows, cols]
    // into contiguous output [n_indices, cols].
    //
    // Grid : (n_indices, 1, 1)
    // Block: (256, 1, 1)
    // -------------------------------------------------------------------------
    pub unsafe fn gather_rows_bf16(
        &self,
        src: *const u16,       // [total_rows, cols] BF16 on GPU
        dst: *mut u16,         // [n_indices, cols] BF16 on GPU
        indices: *const i32,   // [n_indices] on GPU
        n_indices: i32,
        cols: i32,
    ) -> Result<(), HipError> {
        let grid = (n_indices as u32, 1, 1);
        let block = (256u32, 1, 1);

        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_indices = indices;
        let mut p_n_indices = n_indices;
        let mut p_cols = cols;

        hip_launch_kernel(
            self.fn_gather_rows_bf16,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_src as *mut *const u16 as *mut std::ffi::c_void,
                &mut p_dst as *mut *mut u16 as *mut std::ffi::c_void,
                &mut p_indices as *mut *const i32 as *mut std::ffi::c_void,
                &mut p_n_indices as *mut i32 as *mut std::ffi::c_void,
                &mut p_cols as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Sparse MLP: gather_cols_bf16
    //
    // Gathers selected columns from BF16 weight matrix [rows, total_cols]
    // into contiguous output [rows, n_indices].
    //
    // Grid : (rows, 1, 1)
    // Block: (256, 1, 1)
    // -------------------------------------------------------------------------
    pub unsafe fn gather_cols_bf16(
        &self,
        src: *const u16,       // [rows, total_cols] BF16 on GPU
        dst: *mut u16,         // [rows, n_indices] BF16 on GPU
        indices: *const i32,   // [n_indices] on GPU
        rows: i32,
        total_cols: i32,
        n_indices: i32,
    ) -> Result<(), HipError> {
        let grid = (rows as u32, 1, 1);
        let block = (256u32, 1, 1);

        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_indices = indices;
        let mut p_rows = rows;
        let mut p_total_cols = total_cols;
        let mut p_n_indices = n_indices;

        hip_launch_kernel(
            self.fn_gather_cols_bf16,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_src as *mut *const u16 as *mut std::ffi::c_void,
                &mut p_dst as *mut *mut u16 as *mut std::ffi::c_void,
                &mut p_indices as *mut *const i32 as *mut std::ffi::c_void,
                &mut p_rows as *mut i32 as *mut std::ffi::c_void,
                &mut p_total_cols as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_indices as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kuramoto GPU-side prediction: coherence computation
    //
    // Grid : (n_neurons, 1, 1) — one block per neuron
    // Block: (256, 1, 1)
    // -------------------------------------------------------------------------
    pub unsafe fn kuramoto_coherence(
        &self,
        hidden: *const f32,          // [hidden_dim] F32 on GPU
        tuning_forks: *const f32,    // [n_neurons, phase_dim] F32 on GPU
        coherences: *mut f32,        // [n_neurons] F32 on GPU
        n_neurons: i32,
        hidden_dim: i32,
        phase_dim: i32,
    ) -> Result<(), HipError> {
        let grid = (n_neurons as u32, 1, 1);
        let block = (256u32, 1, 1);

        let mut p_hidden = hidden;
        let mut p_forks = tuning_forks;
        let mut p_coh = coherences;
        let mut p_nn = n_neurons;
        let mut p_hd = hidden_dim;
        let mut p_pd = phase_dim;

        hip_launch_kernel(
            self.fn_kuramoto_coherence,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_hidden as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_forks as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_coh as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_nn as *mut i32 as *mut std::ffi::c_void,
                &mut p_hd as *mut i32 as *mut std::ffi::c_void,
                &mut p_pd as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Kuramoto GPU-side prediction: top-K selection
    //
    // Grid : (1, 1, 1) — single block
    // Block: (256, 1, 1)
    // -------------------------------------------------------------------------
    pub unsafe fn topk_select(
        &self,
        coherences: *const f32,  // [n_neurons] F32 on GPU
        indices: *mut i32,       // [top_k] I32 on GPU — output
        n_neurons: i32,
        top_k: i32,
    ) -> Result<(), HipError> {
        let grid = (1u32, 1, 1);
        let block = (256u32, 1, 1);

        let mut p_coh = coherences;
        let mut p_idx = indices;
        let mut p_nn = n_neurons;
        let mut p_tk = top_k;

        hip_launch_kernel(
            self.fn_topk_select,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_coh as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_idx as *mut *mut i32 as *mut std::ffi::c_void,
                &mut p_nn as *mut i32 as *mut std::ffi::c_void,
                &mut p_tk as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ── Fused Attention Decode ─────────────────────────────────────────────
    // Fuses Q×K^T → softmax → ×V into a single kernel for n=1 decode.
    // Eliminates the scores tensor and 3 separate kernel launches per layer.
    // Grid : (n_heads, 1, 1)
    // Block: (256, 1, 1)
    // Shared: (256 + head_dim) * sizeof(float)
    // -----------------------------------------------------------------------

    /// Fused single-token attention (decode only, n=1).
    ///
    /// # Safety
    /// All pointers must point to valid GPU allocations.
    pub unsafe fn fused_attention_decode(
        &self,
        q: *const f32,          // [1, n_heads * head_dim]
        k_cache: *const f32,    // [max_seq, n_kv_heads * head_dim]
        v_cache: *const f32,    // [max_seq, n_kv_heads * head_dim]
        out: *mut f32,          // [1, n_heads * head_dim]
        seq_len: i32,
        head_dim: i32,
        n_heads: i32,
        n_kv_heads: i32,
        kv_dim: i32,
        scale: f32,
    ) -> Result<(), HipError> {
        let grid = (n_heads as u32, 1, 1);
        let block = (256u32, 1, 1);
        let shared_bytes = (256 + head_dim as u32) * 4; // floats

        let mut p_q = q;
        let mut p_k = k_cache;
        let mut p_v = v_cache;
        let mut p_out = out;
        let mut p_seq = seq_len;
        let mut p_hd = head_dim;
        let mut p_nh = n_heads;
        let mut p_nk = n_kv_heads;
        let mut p_kd = kv_dim;
        let mut p_sc = scale;

        hip_launch_kernel(
            self.fn_fused_attention_decode,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_q as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_k as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_v as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq as *mut i32 as *mut std::ffi::c_void,
                &mut p_hd as *mut i32 as *mut std::ffi::c_void,
                &mut p_nh as *mut i32 as *mut std::ffi::c_void,
                &mut p_nk as *mut i32 as *mut std::ffi::c_void,
                &mut p_kd as *mut i32 as *mut std::ffi::c_void,
                &mut p_sc as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }
}
