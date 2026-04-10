//! Safe wrappers for HIP LLM inference kernels.
//!
//! Kernels are compiled from `kernels/llm_kernels.hip` by `build.rs` using
//!   hipcc --offload-arch=gfx1031 -O3 -mwavefrontsize32 -ffast-math --genco -o $OUT_DIR/llm_kernels.hsaco kernels/llm_kernels.hip
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

use crate::raw::{hipFunction_t, hipModule_t, hipStream_t};
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

/// Wave64 GEMV kernels — compiled separately with wave64 for 2.26x better
/// memory bandwidth on RDNA2. Used for the decode hot path.
static KERNEL_W64_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_w64.hsaco"));

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
    stream: hipStream_t, // dedicated stream for all kernel launches
    fn_embed_lookup: hipFunction_t,
    fn_rms_norm: hipFunction_t,
    fn_add_bias: hipFunction_t,
    fn_rope_inplace: hipFunction_t,
    fn_silu_mul: hipFunction_t,
    fn_softmax_rows: hipFunction_t,
    fn_softmax_rows_causal: hipFunction_t,
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
    fn_dequant_gemv_q4k: hipFunction_t,
    fn_dequant_gemv_q6k: hipFunction_t,
    fn_quantize_f32_q8_1: hipFunction_t,
    fn_gemv_q4k_q8: hipFunction_t,
    fn_gemv_q6k_q8: hipFunction_t,
    fn_fused_gate_up_silu_q4k: hipFunction_t,
    fn_fused_gate_up_silu_q6k: hipFunction_t,
    fn_gemv_q4k_q8_multi: hipFunction_t,
    fn_gemv_q6k_q8_multi: hipFunction_t,
    fn_fused_qkv_q4k: hipFunction_t,
    fn_fused_rms_norm_quantize: hipFunction_t,
    fn_fused_rope_kv_copy: hipFunction_t,
    fn_fused_gate_up_silu_q4k_multi: hipFunction_t,
    fn_gemv_q4k_f32_multi: hipFunction_t,
    fn_gemv_bf16_f32_multi: hipFunction_t,
    fn_fused_rms_qkv_q4k_f32: hipFunction_t,
    fn_fused_rms_gate_up_silu_q4k_f32: hipFunction_t,
    fn_fused_rms_qkv_q4_f32: hipFunction_t,
    fn_fused_rms_gate_up_silu_q4_f32: hipFunction_t,
    fn_argmax_f32: hipFunction_t,
    // Wave64 GEMV kernels (2.26x faster BW on RDNA2)
    module_w64: hipModule_t,
    fn_gemv_q4k_q8_w64: hipFunction_t,
    fn_fused_gate_up_silu_q4k_w64: hipFunction_t,
    fn_fused_qkv_q4k_w64: hipFunction_t,
    fn_gemv_q6k_q8_w64: hipFunction_t,
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
        let fn_softmax_rows_causal = get_fn!("softmax_rows_causal");
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
        let fn_dequant_gemv_q4k = get_fn!("dequant_gemv_q4k");
        let fn_dequant_gemv_q6k = get_fn!("dequant_gemv_q6k");
        let fn_quantize_f32_q8_1 = get_fn!("quantize_f32_q8_1");
        let fn_gemv_q4k_q8 = get_fn!("gemv_q4k_q8");
        let fn_gemv_q6k_q8 = get_fn!("gemv_q6k_q8");
        let fn_fused_gate_up_silu_q4k = get_fn!("fused_gate_up_silu_q4k");
        let fn_fused_gate_up_silu_q6k = get_fn!("fused_gate_up_silu_q6k");
        let fn_gemv_q4k_q8_multi = get_fn!("gemv_q4k_q8_multi");
        let fn_gemv_q6k_q8_multi = get_fn!("gemv_q6k_q8_multi");
        let fn_fused_qkv_q4k = get_fn!("fused_qkv_q4k");
        let fn_fused_rms_norm_quantize = get_fn!("fused_rms_norm_quantize");
        let fn_fused_rope_kv_copy = get_fn!("fused_rope_kv_copy");
        let fn_fused_gate_up_silu_q4k_multi = get_fn!("fused_gate_up_silu_q4k_multi");
        let fn_gemv_q4k_f32_multi = get_fn!("gemv_q4k_f32_multi");
        let fn_gemv_bf16_f32_multi = get_fn!("gemv_bf16_f32_multi");
        let fn_fused_rms_qkv_q4k_f32 = get_fn!("fused_rms_qkv_q4k_f32");
        let fn_fused_rms_gate_up_silu_q4k_f32 = get_fn!("fused_rms_gate_up_silu_q4k_f32");
        let fn_fused_rms_qkv_q4_f32 = get_fn!("fused_rms_qkv_q4_f32");
        let fn_fused_rms_gate_up_silu_q4_f32 = get_fn!("fused_rms_gate_up_silu_q4_f32");
        let fn_argmax_f32 = get_fn!("argmax_f32");

        // Load wave64 GEMV kernels
        let module_w64 = hip_module_load_data(KERNEL_W64_BYTES)?;
        macro_rules! get_w64_fn {
            ($name:literal) => {
                hip_module_get_function(module_w64, $name)?
            };
        }
        let fn_gemv_q4k_q8_w64 = get_w64_fn!("gemv_q4k_q8_w64");
        let fn_fused_gate_up_silu_q4k_w64 = get_w64_fn!("fused_gate_up_silu_q4k_w64");
        let fn_fused_qkv_q4k_w64 = get_w64_fn!("fused_qkv_q4k_w64");
        let fn_gemv_q6k_q8_w64 = get_w64_fn!("gemv_q6k_q8_w64");

        Ok(Self {
            module,
            stream: ptr::null_mut(), // default stream; call set_stream() to change
            fn_embed_lookup,
            fn_rms_norm,
            fn_add_bias,
            fn_rope_inplace,
            fn_silu_mul,
            fn_softmax_rows,
            fn_softmax_rows_causal,
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
            fn_dequant_gemv_q4k,
            fn_dequant_gemv_q6k,
            fn_quantize_f32_q8_1,
            fn_gemv_q4k_q8,
            fn_gemv_q6k_q8,
            fn_fused_gate_up_silu_q4k,
            fn_fused_gate_up_silu_q6k,
            fn_gemv_q4k_q8_multi,
            fn_gemv_q6k_q8_multi,
            fn_fused_qkv_q4k,
            fn_fused_rms_norm_quantize,
            fn_fused_rope_kv_copy,
            fn_fused_gate_up_silu_q4k_multi,
            fn_gemv_q4k_f32_multi,
            fn_gemv_bf16_f32_multi,
            fn_fused_rms_qkv_q4k_f32,
            fn_fused_rms_gate_up_silu_q4k_f32,
            fn_fused_rms_qkv_q4_f32,
            fn_fused_rms_gate_up_silu_q4_f32,
            fn_argmax_f32,
            module_w64,
            fn_gemv_q4k_q8_w64,
            fn_fused_gate_up_silu_q4k_w64,
            fn_fused_qkv_q4k_w64,
            fn_gemv_q6k_q8_w64,
        })
    }

    /// Set the HIP stream for all subsequent kernel launches.
    pub fn set_stream(&mut self, stream: hipStream_t) {
        self.stream = stream;
    }

    /// Get the current stream.
    pub fn get_stream(&self) -> hipStream_t {
        self.stream
    }
}

impl Drop for GpuKernels {
    fn drop(&mut self) {
        unsafe {
            if !self.module.is_null() {
                let _ = hip_module_unload(self.module);
                self.module = ptr::null_mut();
            }
            if !self.module_w64.is_null() {
                let _ = hip_module_unload(self.module_w64);
                self.module_w64 = ptr::null_mut();
            }
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
            &mut [
                &mut p_scores as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq_len as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    /// Launch `softmax_rows_causal` — causal masked softmax for prefill attention.
    ///
    /// # Safety
    /// `scores` must point to at least `n_rows * seq_len` contiguous f32 on device.
    pub unsafe fn softmax_rows_causal(
        &self,
        scores: *mut f32,
        n_rows: i32,
        seq_len: i32,
        n_tokens: i32,
        query_offset: i32,
    ) -> Result<(), HipError> {
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
        let mut p_n_tokens = n_tokens;
        let mut p_query_offset = query_offset;

        hip_launch_kernel(
            self.fn_softmax_rows_causal,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_scores as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq_len as *mut i32 as *mut std::ffi::c_void,
                &mut p_n_tokens as *mut i32 as *mut std::ffi::c_void,
                &mut p_query_offset as *mut i32 as *mut std::ffi::c_void,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
    // Native GGML Q4_K GEMV — raw Q4_K blocks, dequant on-the-fly in F32
    // Grid : (N, M, 1)  — one block per output row per token
    // Block: (256, 1, 1) — threads stride over qs bytes
    // Shared: 8 floats = 32 bytes
    // ─────────────────────────────────────────────────────────────────────────

    /// GGML Q4_K GEMV: operates directly on raw Q4_K blocks on GPU.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn dequant_gemv_q4k(
        &self,
        data: *const u8,   // [n * (k/256) * 144] raw Q4_K blocks
        x: *const f32,     // [m, k] F32 activations
        out: *mut f32,     // [m, n] F32 output
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, m.max(1) as u32, 1);
        let block = (256u32, 1, 1);
        let shared_mem = (8 * std::mem::size_of::<f32>()) as u32;

        let mut p_data = data;
        let mut p_x = x;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_dequant_gemv_q4k,
            grid,
            block,
            shared_mem,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Native GGML Q6_K GEMV — raw Q6_K blocks, dequant on-the-fly in F32
    // ─────────────────────────────────────────────────────────────────────────

    /// GGML Q6_K GEMV: operates directly on raw Q6_K blocks on GPU.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations of the correct size.
    pub unsafe fn dequant_gemv_q6k(
        &self,
        data: *const u8,   // [n * (k/256) * 210] raw Q6_K blocks
        x: *const f32,     // [m, k] F32 activations
        out: *mut f32,     // [m, n] F32 output
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, m.max(1) as u32, 1);
        let block = (256u32, 1, 1);
        let shared_mem = (8 * std::mem::size_of::<f32>()) as u32;

        let mut p_data = data;
        let mut p_x = x;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_dequant_gemv_q6k,
            grid,
            block,
            shared_mem,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // quantize_f32_q8_1: Quantize F32 activations → Q8_1 blocks
    // ─────────────────────────────────────────────────────────────────────────

    /// Quantize F32 activations to Q8_1 format for fast integer GEMV.
    ///
    /// # Safety
    /// `x` must point to `k` f32 values. `out` must have space for `(k/32)*36` bytes.
    pub unsafe fn quantize_f32_q8_1(
        &self,
        x: *const f32,      // [K] F32 activations
        out: *mut u8,        // [K/32 * 36] Q8_1 blocks
        k: i32,
    ) -> Result<(), HipError> {
        let n_blocks = k / 32;
        // 8 warps per block (256 threads), each warp handles one Q8_1 block
        let grid = ((n_blocks as u32 + 7) / 8, 1, 1);
        let block = (256u32, 1, 1);

        let mut p_x = x;
        let mut p_out = out;
        let mut p_k = k;

        hip_launch_kernel(
            self.fn_quantize_f32_q8_1,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_x as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut u8 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // gemv_q4k_q8: Q4_K × Q8_1 GEMV with dp4a
    // ─────────────────────────────────────────────────────────────────────────

    /// Q4_K × Q8_1 GEMV with dp4a integer dot product.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations.
    pub unsafe fn gemv_q4k_q8(
        &self,
        data: *const u8,     // [N * (K/256) * 144] raw Q4_K blocks
        x_q8: *const u8,     // [M * (K/32) * 36] Q8_1 blocks
        out: *mut f32,       // [M, N] output
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, m.max(1) as u32, 1);
        let block = (32u32, 1, 1);  // single warp (wave32)

        let mut p_data = data;
        let mut p_x_q8 = x_q8;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_gemv_q4k_q8,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x_q8 as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // gemv_q4k_q8_multi: Multi-row Q4_K × Q8_1 GEMV (8 rows per block)
    // ─────────────────────────────────────────────────────────────────────────

    /// Multi-row Q4_K × Q8_1 GEMV. 8 rows per block, sharing Q8 input via LDS.
    /// Better for large N (lm_head with N=151936).
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations.
    pub unsafe fn gemv_q4k_q8_multi(
        &self,
        data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (((n + 7) / 8) as u32, m.max(1) as u32, 1);
        let block = (32u32, 8, 1);  // 8 warps per block
        let nq8 = k / 32;
        let shared_bytes = (nq8 * 36) as u32;  // Q8_1 blocks in LDS

        let mut p_data = data;
        let mut p_x_q8 = x_q8;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_gemv_q4k_q8_multi,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x_q8 as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // gemv_q6k_q8: Q6_K × Q8_1 GEMV with integer multiply
    // ─────────────────────────────────────────────────────────────────────────

    /// Q6_K × Q8_1 GEMV with integer multiply and warp reduce.
    ///
    /// # Safety
    /// All GPU pointers must be valid device allocations.
    pub unsafe fn gemv_q6k_q8(
        &self,
        data: *const u8,     // [N * (K/256) * 210] raw Q6_K blocks
        x_q8: *const u8,     // [M * (K/32) * 36] Q8_1 blocks
        out: *mut f32,       // [M, N] output
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, m.max(1) as u32, 1);
        let block = (32u32, 1, 1);

        let mut p_data = data;
        let mut p_x_q8 = x_q8;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_gemv_q6k_q8,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x_q8 as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // gemv_q6k_q8_multi: Multi-row Q6_K × Q8_1 GEMV (8 rows per block)
    // ─────────────────────────────────────────────────────────────────────────

    pub unsafe fn gemv_q6k_q8_multi(
        &self,
        data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (((n + 7) / 8) as u32, m.max(1) as u32, 1);
        let block = (32u32, 8, 1);

        let mut p_data = data;
        let mut p_x_q8 = x_q8;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_gemv_q6k_q8_multi,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x_q8 as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fused_gate_up_silu_q4k: Fused MLP gate+up+SiLU for Q4_K weights
    // ─────────────────────────────────────────────────────────────────────────

    pub unsafe fn fused_gate_up_silu_q4k(
        &self,
        gate_data: *const u8,
        up_data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        n: i32,
        k: i32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, 1, 1);
        let block = (32u32, 1, 1);

        let mut p_gate = gate_data;
        let mut p_up = up_data;
        let mut p_x = x_q8;
        let mut p_out = out;
        let mut p_n = n;
        let mut p_k = k;

        hip_launch_kernel(
            self.fn_fused_gate_up_silu_q4k,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_gate as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_up   as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x    as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out  as *mut *mut f32  as *mut std::ffi::c_void,
                &mut p_n    as *mut i32       as *mut std::ffi::c_void,
                &mut p_k    as *mut i32       as *mut std::ffi::c_void,
            ],
        )
    }

    pub unsafe fn fused_gate_up_silu_q6k(
        &self,
        gate_data: *const u8,
        up_data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        n: i32,
        k: i32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, 1, 1);
        let block = (32u32, 1, 1);

        let mut p_gate = gate_data;
        let mut p_up = up_data;
        let mut p_x = x_q8;
        let mut p_out = out;
        let mut p_n = n;
        let mut p_k = k;

        hip_launch_kernel(
            self.fn_fused_gate_up_silu_q6k,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_gate as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_up   as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x    as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out  as *mut *mut f32  as *mut std::ffi::c_void,
                &mut p_n    as *mut i32       as *mut std::ffi::c_void,
                &mut p_k    as *mut i32       as *mut std::ffi::c_void,
            ],
        )
    }

    pub unsafe fn fused_gate_up_silu_q4k_multi(
        &self,
        gate_data: *const u8,
        up_data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        n: i32,
        k: i32,
    ) -> Result<(), HipError> {
        let grid = (((n + 7) / 8) as u32, 1, 1);
        let block = (32u32, 8, 1);
        let nq8 = k / 32;
        let shared_bytes = (nq8 * 36) as u32;

        let mut p_gate = gate_data;
        let mut p_up = up_data;
        let mut p_x = x_q8;
        let mut p_out = out;
        let mut p_n = n;
        let mut p_k = k;

        hip_launch_kernel(
            self.fn_fused_gate_up_silu_q4k_multi,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_gate as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_up   as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x    as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out  as *mut *mut f32  as *mut std::ffi::c_void,
                &mut p_n    as *mut i32       as *mut std::ffi::c_void,
                &mut p_k    as *mut i32       as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // gemv_q4k_f32_multi — Q4_K × F32 GEMV (no intermediate Q8 quantize)
    // Eliminates the quantize step between gate+up output and down projection.
    // Grid : (ceil(N/8), M, 1)  Block: (32, 8, 1)
    // Shared: K * sizeof(float) bytes
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemv_q4k_f32_multi(
        &self,
        w_data: *const u8,  // [N * (K/256) * 144] raw Q4_K blocks
        x: *const f32,      // [K] F32 activations
        out: *mut f32,      // [N] F32 output
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (((n + 7) / 8) as u32, m.max(1) as u32, 1);
        let block = (32u32, 8, 1);
        let shared_bytes = (k * 4) as u32;  // K floats in shared memory

        let mut p_data = w_data;
        let mut p_x    = x;
        let mut p_out  = out;
        let mut p_m    = m;
        let mut p_n    = n;
        let mut p_k    = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_gemv_q4k_f32_multi,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x    as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_m    as *mut i32        as *mut std::ffi::c_void,
                &mut p_n    as *mut i32        as *mut std::ffi::c_void,
                &mut p_k    as *mut i32        as *mut std::ffi::c_void,
                &mut p_beta as *mut f32        as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // gemv_bf16_f32_multi — BF16 weights × F32 activations GEMV
    // Replaces hipBLAS SGEMM for lm_head decode (M=1, N=vocab, K=hidden).
    // Grid : (ceil(N/8), 1, 1)  Block: (32, 8, 1)
    // Shared: K * sizeof(float) bytes
    // ─────────────────────────────────────────────────────────────────────────
    pub unsafe fn gemv_bf16_f32_multi(
        &self,
        w: *const u16,   // [N, K] BF16 weights
        x: *const f32,   // [K] F32 activations
        out: *mut f32,   // [N] F32 output
        n: i32,
        k: i32,
    ) -> Result<(), HipError> {
        let grid = (((n + 7) / 8) as u32, 1, 1);
        let block = (32u32, 8, 1);
        let shared_bytes = (k * 4) as u32;  // K floats in shared memory

        let mut p_w   = w;
        let mut p_x   = x;
        let mut p_out = out;
        let mut p_n   = n;
        let mut p_k   = k;

        hip_launch_kernel(
            self.fn_gemv_bf16_f32_multi,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_w   as *mut *const u16 as *mut std::ffi::c_void,
                &mut p_x   as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_n   as *mut i32        as *mut std::ffi::c_void,
                &mut p_k   as *mut i32        as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fused_rms_qkv_q4k_f32 — RMSNorm + Q/K/V projections in one kernel
    // Eliminates Q8 intermediate: rms_norm_quantize + fused_qkv → single launch
    // Grid : (ceil((Nq+Nk+Nv)/8), 1, 1)
    // Block: (32, 8, 1) — 8 warps, shared = K floats
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fused_rms_qkv_q4k_f32(
        &self,
        x: *const f32,
        norm_w: *const f32,
        wq_data: *const u8,
        wk_data: *const u8,
        wv_data: *const u8,
        out_q: *mut f32,
        out_k: *mut f32,
        out_v: *mut f32,
        bias_q: *const f32,
        bias_k: *const f32,
        bias_v: *const f32,
        nq: i32, nk: i32, nv: i32, k: i32, eps: f32,
    ) -> Result<(), HipError> {
        let total = nq + nk + nv;
        let grid = (((total + 7) / 8) as u32, 1, 1);
        let block = (32u32, 8, 1);
        let shared_bytes = (k * 4) as u32;  // K floats in shared

        let mut p_x      = x;
        let mut p_nw     = norm_w;
        let mut p_wq     = wq_data;
        let mut p_wk     = wk_data;
        let mut p_wv     = wv_data;
        let mut p_oq     = out_q;
        let mut p_ok     = out_k;
        let mut p_ov     = out_v;
        let mut p_bq     = bias_q;
        let mut p_bk     = bias_k;
        let mut p_bv     = bias_v;
        let mut p_nq     = nq;
        let mut p_nk     = nk;
        let mut p_nv     = nv;
        let mut p_k      = k;
        let mut p_eps    = eps;

        hip_launch_kernel(
            self.fn_fused_rms_qkv_q4k_f32,
            grid, block, shared_bytes, self.stream,
            &mut [
                &mut p_x   as *mut *const f32  as *mut std::ffi::c_void,
                &mut p_nw  as *mut *const f32  as *mut std::ffi::c_void,
                &mut p_wq  as *mut *const u8   as *mut std::ffi::c_void,
                &mut p_wk  as *mut *const u8   as *mut std::ffi::c_void,
                &mut p_wv  as *mut *const u8   as *mut std::ffi::c_void,
                &mut p_oq  as *mut *mut f32    as *mut std::ffi::c_void,
                &mut p_ok  as *mut *mut f32    as *mut std::ffi::c_void,
                &mut p_ov  as *mut *mut f32    as *mut std::ffi::c_void,
                &mut p_bq  as *mut *const f32  as *mut std::ffi::c_void,
                &mut p_bk  as *mut *const f32  as *mut std::ffi::c_void,
                &mut p_bv  as *mut *const f32  as *mut std::ffi::c_void,
                &mut p_nq  as *mut i32         as *mut std::ffi::c_void,
                &mut p_nk  as *mut i32         as *mut std::ffi::c_void,
                &mut p_nv  as *mut i32         as *mut std::ffi::c_void,
                &mut p_k   as *mut i32         as *mut std::ffi::c_void,
                &mut p_eps as *mut f32         as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fused_rms_gate_up_silu_q4k_f32 — RMSNorm + gate+up+SiLU in one kernel
    // Eliminates Q8 intermediate for MLP input
    // Grid : (ceil(N/8), 1, 1)
    // Block: (32, 8, 1) — 8 warps, shared = K floats
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fused_rms_gate_up_silu_q4k_f32(
        &self,
        x: *const f32,
        norm_w: *const f32,
        gate_data: *const u8,
        up_data: *const u8,
        out: *mut f32,
        n: i32, k: i32, eps: f32,
    ) -> Result<(), HipError> {
        let grid = (((n + 7) / 8) as u32, 1, 1);
        let block = (32u32, 8, 1);
        let shared_bytes = (k * 4) as u32;

        let mut p_x    = x;
        let mut p_nw   = norm_w;
        let mut p_gate = gate_data;
        let mut p_up   = up_data;
        let mut p_out  = out;
        let mut p_n    = n;
        let mut p_k    = k;
        let mut p_eps  = eps;

        hip_launch_kernel(
            self.fn_fused_rms_gate_up_silu_q4k_f32,
            grid, block, shared_bytes, self.stream,
            &mut [
                &mut p_x    as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_nw   as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_gate as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_up   as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_out  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_n    as *mut i32        as *mut std::ffi::c_void,
                &mut p_k    as *mut i32        as *mut std::ffi::c_void,
                &mut p_eps  as *mut f32        as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fused_rms_qkv_q4_f32 — RMSNorm + Q/K/V for IARA native Q4 format
    // Grid : (ceil((Nq+Nk+Nv)/8), 1, 1)
    // Block: (32, 8, 1) — shared = K*4 + 8*4 bytes
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fused_rms_qkv_q4_f32(
        &self,
        x: *const f32, norm_w: *const f32,
        wq_packed: *const u8, wq_scales: *const f32, wq_zeros: *const f32,
        wk_packed: *const u8, wk_scales: *const f32, wk_zeros: *const f32,
        wv_packed: *const u8, wv_scales: *const f32, wv_zeros: *const f32,
        out_q: *mut f32, out_k: *mut f32, out_v: *mut f32,
        bias_q: *const f32, bias_k: *const f32, bias_v: *const f32,
        nq: i32, nk: i32, nv: i32, k: i32, group_size: i32, eps: f32,
    ) -> Result<(), HipError> {
        let total = nq + nk + nv;
        let grid = (((total + 7) / 8) as u32, 1, 1);
        let block = (32u32, 8, 1);
        let shared_bytes = (k * 4 + 8 * 4) as u32;

        let mut p_x   = x;     let mut p_nw  = norm_w;
        let mut p_wqp = wq_packed; let mut p_wqs = wq_scales; let mut p_wqz = wq_zeros;
        let mut p_wkp = wk_packed; let mut p_wks = wk_scales; let mut p_wkz = wk_zeros;
        let mut p_wvp = wv_packed; let mut p_wvs = wv_scales; let mut p_wvz = wv_zeros;
        let mut p_oq  = out_q;  let mut p_ok  = out_k;  let mut p_ov  = out_v;
        let mut p_bq  = bias_q; let mut p_bk  = bias_k; let mut p_bv  = bias_v;
        let mut p_nq  = nq;     let mut p_nk  = nk;     let mut p_nv  = nv;
        let mut p_k   = k;      let mut p_gs  = group_size; let mut p_eps = eps;

        hip_launch_kernel(
            self.fn_fused_rms_qkv_q4_f32,
            grid, block, shared_bytes, self.stream,
            &mut [
                &mut p_x   as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_nw  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_wqp as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_wqs as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_wqz as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_wkp as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_wks as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_wkz as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_wvp as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_wvs as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_wvz as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_oq  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_ok  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_ov  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_bq  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_bk  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_bv  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_nq  as *mut i32        as *mut std::ffi::c_void,
                &mut p_nk  as *mut i32        as *mut std::ffi::c_void,
                &mut p_nv  as *mut i32        as *mut std::ffi::c_void,
                &mut p_k   as *mut i32        as *mut std::ffi::c_void,
                &mut p_gs  as *mut i32        as *mut std::ffi::c_void,
                &mut p_eps as *mut f32        as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fused_rms_gate_up_silu_q4_f32 — RMSNorm + gate+up+SiLU for IARA Q4
    // Grid : (ceil(N/8), 1, 1)
    // Block: (32, 8, 1) — shared = K*4 + 8*4 bytes
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fused_rms_gate_up_silu_q4_f32(
        &self,
        x: *const f32, norm_w: *const f32,
        gate_packed: *const u8, gate_scales: *const f32, gate_zeros: *const f32,
        up_packed: *const u8, up_scales: *const f32, up_zeros: *const f32,
        out: *mut f32,
        n: i32, k: i32, group_size: i32, eps: f32,
    ) -> Result<(), HipError> {
        let grid = (((n + 7) / 8) as u32, 1, 1);
        let block = (32u32, 8, 1);
        let shared_bytes = (k * 4 + 8 * 4) as u32;

        let mut p_x   = x;           let mut p_nw  = norm_w;
        let mut p_gp  = gate_packed; let mut p_gs  = gate_scales; let mut p_gz  = gate_zeros;
        let mut p_up  = up_packed;   let mut p_us  = up_scales;   let mut p_uz  = up_zeros;
        let mut p_out = out;
        let mut p_n   = n;           let mut p_k   = k;
        let mut p_grs = group_size;  let mut p_eps = eps;

        hip_launch_kernel(
            self.fn_fused_rms_gate_up_silu_q4_f32,
            grid, block, shared_bytes, self.stream,
            &mut [
                &mut p_x   as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_nw  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_gp  as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_gs  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_gz  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_up  as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_us  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_uz  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_n   as *mut i32        as *mut std::ffi::c_void,
                &mut p_k   as *mut i32        as *mut std::ffi::c_void,
                &mut p_grs as *mut i32        as *mut std::ffi::c_void,
                &mut p_eps as *mut f32        as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fused QKV Q4K — Q+K+V projections in a single launch
    // Grid : (ceil((Nq+Nk+Nv)/8), 1, 1)
    // Block: (32, 8, 1) — 8 warps sharing Q8 via LDS
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fused_qkv_q4k(
        &self,
        wq_data: *const u8,
        wk_data: *const u8,
        wv_data: *const u8,
        x_q8: *const u8,
        out_q: *mut f32,
        out_k: *mut f32,
        out_v: *mut f32,
        bias_q: *const f32,  // nullptr = no bias
        bias_k: *const f32,
        bias_v: *const f32,
        nq: i32, nk: i32, nv: i32, k: i32,
    ) -> Result<(), HipError> {
        let total = nq + nk + nv;
        let grid = (((total + 7) / 8) as u32, 1, 1);
        let block = (32u32, 8, 1);
        let nq8 = k / 32;
        let shared_bytes = (nq8 * 36) as u32;

        let mut p_wq = wq_data;
        let mut p_wk = wk_data;
        let mut p_wv = wv_data;
        let mut p_x = x_q8;
        let mut p_oq = out_q;
        let mut p_ok = out_k;
        let mut p_ov = out_v;
        let mut p_bq = bias_q;
        let mut p_bk = bias_k;
        let mut p_bv = bias_v;
        let mut p_nq = nq;
        let mut p_nk = nk;
        let mut p_nv = nv;
        let mut p_k = k;

        hip_launch_kernel(
            self.fn_fused_qkv_q4k,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_wq as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_wk as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_wv as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_x  as *mut *const u8  as *mut std::ffi::c_void,
                &mut p_oq as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_ok as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_ov as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_bq as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_bk as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_bv as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_nq as *mut i32        as *mut std::ffi::c_void,
                &mut p_nk as *mut i32        as *mut std::ffi::c_void,
                &mut p_nv as *mut i32        as *mut std::ffi::c_void,
                &mut p_k  as *mut i32        as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fused RMSNorm + Quantize Q8_1
    // Grid : (1, 1, 1) — single block for 1 token
    // Block: (1024, 1, 1)
    // Shared: (1024 + hidden) * 4 bytes
    // ─────────────────────────────────────────────────────────────────────────
    pub unsafe fn fused_rms_norm_quantize(
        &self,
        x: *const f32,
        w: *const f32,
        out_f32: *mut f32,
        out_q8: *mut u8,
        hidden: i32,
        eps: f32,
    ) -> Result<(), HipError> {
        let block_size = 1024i32.min((hidden as u32).next_power_of_two() as i32);
        let grid = (1u32, 1, 1);
        let block = (block_size as u32, 1, 1);
        let shared_bytes = ((block_size + hidden) as u32) * 4; // reduction + normed buf

        let mut p_x = x;
        let mut p_w = w;
        let mut p_of = out_f32;
        let mut p_oq = out_q8;
        let mut p_h = hidden;
        let mut p_e = eps;

        hip_launch_kernel(
            self.fn_fused_rms_norm_quantize,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_x  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_w  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_of as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_oq as *mut *mut u8    as *mut std::ffi::c_void,
                &mut p_h  as *mut i32        as *mut std::ffi::c_void,
                &mut p_e  as *mut f32        as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fused RoPE + KV cache copy (4 launches → 1)
    // Grid : (n_heads + n_kv_heads, 1, 1)
    // Block: (head_dim, 1, 1)
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fused_rope_kv_copy(
        &self,
        q: *mut f32, k: *mut f32, v: *const f32,
        kv_k_cache: *mut f32, kv_v_cache: *mut f32,
        n_heads: i32, n_kv_heads: i32, head_dim: i32,
        pos_start: i32, theta: f32,
        kv_dim: i32, max_seq: i32,
    ) -> Result<(), HipError> {
        let grid = ((n_heads + n_kv_heads) as u32, 1, 1);
        let block = (head_dim as u32, 1, 1);

        let mut p_q = q;
        let mut p_k = k;
        let mut p_v = v;
        let mut p_kk = kv_k_cache;
        let mut p_kv = kv_v_cache;
        let mut p_nh = n_heads;
        let mut p_nk = n_kv_heads;
        let mut p_hd = head_dim;
        let mut p_ps = pos_start;
        let mut p_th = theta;
        let mut p_kd = kv_dim;
        let mut p_ms = max_seq;

        hip_launch_kernel(
            self.fn_fused_rope_kv_copy,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_q  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_k  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_v  as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_kk as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_kv as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_nh as *mut i32        as *mut std::ffi::c_void,
                &mut p_nk as *mut i32        as *mut std::ffi::c_void,
                &mut p_hd as *mut i32        as *mut std::ffi::c_void,
                &mut p_ps as *mut i32        as *mut std::ffi::c_void,
                &mut p_th as *mut f32        as *mut std::ffi::c_void,
                &mut p_kd as *mut i32        as *mut std::ffi::c_void,
                &mut p_ms as *mut i32        as *mut std::ffi::c_void,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
            self.stream,
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
        q: *const f32,          // [n_heads * head_dim]
        k_cache: *const f32,    // [seq_len, n_kv_heads * head_dim]
        v_cache: *const f32,    // [seq_len, n_kv_heads * head_dim]
        out: *mut f32,          // [n_heads * head_dim]
        seq_len: i32,
        head_dim: i32,
        n_heads: i32,
        n_kv_heads: i32,
        scale: f32,
    ) -> Result<(), HipError> {
        let grid = (n_heads as u32, 1, 1);
        let block = (256u32, 1, 1);
        // Dynamic shared memory: seq_len floats for attention scores
        let shared_bytes = (seq_len as u32) * 4;

        let mut p_q = q;
        let mut p_k = k_cache;
        let mut p_v = v_cache;
        let mut p_out = out;
        let mut p_seq = seq_len;
        let mut p_hd = head_dim;
        let mut p_nh = n_heads;
        let mut p_nk = n_kv_heads;
        let mut p_sc = scale;

        hip_launch_kernel(
            self.fn_fused_attention_decode,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_q as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_k as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_v as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq as *mut i32 as *mut std::ffi::c_void,
                &mut p_hd as *mut i32 as *mut std::ffi::c_void,
                &mut p_nh as *mut i32 as *mut std::ffi::c_void,
                &mut p_nk as *mut i32 as *mut std::ffi::c_void,
                &mut p_sc as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // argmax_f32: GPU-side argmax over F32 logits
    //
    // Finds the index of the maximum value. Eliminates 608KB logits download.
    // Grid: (1, 1, 1), Block: (1024, 1, 1)
    // ─────────────────────────────────────────────────────────────────────────

    /// GPU-side argmax: returns index of maximum value in `x[0..ncols]`.
    ///
    /// # Safety
    /// `x` must point to at least `ncols` f32 values on device.
    /// `dst` must point to at least 1 i32 on device.
    /// `dst_val` can be null or point to 1 f32 on device.
    pub unsafe fn argmax_f32(
        &self,
        x: *const f32,         // [ncols] logits on device
        dst: *mut i32,         // [1] output argmax index
        dst_val: *mut f32,     // [1] output max value (nullable)
        ncols: i32,
    ) -> Result<(), HipError> {
        let grid = (1u32, 1, 1);
        let block = (1024u32, 1, 1);

        let mut p_x = x;
        let mut p_dst = dst;
        let mut p_dst_val = dst_val;
        let mut p_ncols = ncols;

        hip_launch_kernel(
            self.fn_argmax_f32,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_x as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_dst as *mut *mut i32 as *mut std::ffi::c_void,
                &mut p_dst_val as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_ncols as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Wave64 GEMV kernels (2.26x faster bandwidth on RDNA2)
    // ─────────────────────────────────────────────────────────────────────────

    /// Q4_K × Q8_1 GEMV using wave64 (64 threads per row).
    /// 2.26x better bandwidth than wave32 version.
    pub unsafe fn gemv_q4k_q8_w64(
        &self,
        data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, m.max(1) as u32, 1);
        let block = (64u32, 1, 1);  // wave64: one full wavefront per row

        let mut p_data = data;
        let mut p_x_q8 = x_q8;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_gemv_q4k_q8_w64,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x_q8 as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }

    /// Fused gate+up+SiLU GEMV using wave64 (64 threads per row).
    pub unsafe fn fused_gate_up_silu_q4k_w64(
        &self,
        gate_data: *const u8,
        up_data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        n: i32,
        k: i32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, 1, 1);
        let block = (64u32, 1, 1);

        let mut p_gate = gate_data;
        let mut p_up = up_data;
        let mut p_x = x_q8;
        let mut p_out = out;
        let mut p_n = n;
        let mut p_k = k;

        hip_launch_kernel(
            self.fn_fused_gate_up_silu_q4k_w64,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_gate as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_up as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    /// Fused Q/K/V projection using wave64 (64 threads per row, 4 rows per block).
    pub unsafe fn fused_qkv_q4k_w64(
        &self,
        wq_data: *const u8, wk_data: *const u8, wv_data: *const u8,
        x_q8: *const u8,
        out_q: *mut f32, out_k: *mut f32, out_v: *mut f32,
        bias_q: *const f32, bias_k: *const f32, bias_v: *const f32,
        nq: i32, nk: i32, nv: i32, k: i32,
    ) -> Result<(), HipError> {
        let total_n = (nq + nk + nv) as u32;
        let rows_per_block = 4u32;
        let grid = (ceil_div(total_n, rows_per_block), 1, 1);
        let block = (64u32, rows_per_block, 1);
        let nq8 = k / 32;
        // +32 bytes: rows_per_block(4) × 2 halves × 4 bytes for LDS cross-half accumulator slots
        let shared_bytes = (nq8 * 36 + 32) as u32;

        let mut p_wq = wq_data; let mut p_wk = wk_data; let mut p_wv = wv_data;
        let mut p_x = x_q8;
        let mut p_oq = out_q; let mut p_ok = out_k; let mut p_ov = out_v;
        let mut p_bq = bias_q; let mut p_bk = bias_k; let mut p_bv = bias_v;
        let mut p_nq = nq; let mut p_nk = nk; let mut p_nv = nv; let mut p_k = k;

        hip_launch_kernel(
            self.fn_fused_qkv_q4k_w64,
            grid,
            block,
            shared_bytes,
            self.stream,
            &mut [
                &mut p_wq as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_wk as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_wv as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_oq as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_ok as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_ov as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_bq as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_bk as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_bv as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_nq as *mut i32 as *mut std::ffi::c_void,
                &mut p_nk as *mut i32 as *mut std::ffi::c_void,
                &mut p_nv as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
            ],
        )
    }

    /// Q6_K × Q8_1 GEMV using wave64.
    pub unsafe fn gemv_q6k_q8_w64(
        &self,
        data: *const u8,
        x_q8: *const u8,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        beta: f32,
    ) -> Result<(), HipError> {
        let grid = (n as u32, m.max(1) as u32, 1);
        let block = (64u32, 1, 1);

        let mut p_data = data;
        let mut p_x_q8 = x_q8;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_beta = beta;

        hip_launch_kernel(
            self.fn_gemv_q6k_q8_w64,
            grid,
            block,
            0,
            self.stream,
            &mut [
                &mut p_data as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_x_q8 as *mut *const u8 as *mut std::ffi::c_void,
                &mut p_out as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_m as *mut i32 as *mut std::ffi::c_void,
                &mut p_n as *mut i32 as *mut std::ffi::c_void,
                &mut p_k as *mut i32 as *mut std::ffi::c_void,
                &mut p_beta as *mut f32 as *mut std::ffi::c_void,
            ],
        )
    }
}
