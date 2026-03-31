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
use crate::safe::{hip_launch_kernel, hip_module_get_function, hip_module_load_data,
                  hip_module_unload, HipError};
use std::ptr;

// ---------------------------------------------------------------------------
// Embedded kernel binary
// ---------------------------------------------------------------------------

/// The compiled kernel object, embedded at build time.
/// `build.rs` is responsible for producing `$OUT_DIR/llm_kernels.hsaco`.
static KERNEL_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/llm_kernels.hsaco"));

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
    module:            hipModule_t,
    fn_embed_lookup:   hipFunction_t,
    fn_rms_norm:       hipFunction_t,
    fn_add_bias:       hipFunction_t,
    fn_rope_inplace:   hipFunction_t,
    fn_silu_mul:       hipFunction_t,
    fn_softmax_rows:   hipFunction_t,
    fn_copy_kv_cache:  hipFunction_t,
    fn_f32_to_bf16:    hipFunction_t,
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

        let fn_embed_lookup  = get_fn!("embed_lookup");
        let fn_rms_norm      = get_fn!("rms_norm");
        let fn_add_bias      = get_fn!("add_bias_inplace");
        let fn_rope_inplace  = get_fn!("rope_inplace");
        let fn_silu_mul      = get_fn!("silu_mul_inplace");
        let fn_softmax_rows  = get_fn!("softmax_rows");
        let fn_copy_kv_cache = get_fn!("copy_to_kv_cache");
        let fn_f32_to_bf16   = get_fn!("f32_to_bf16");

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
        })
    }
}

impl Drop for GpuKernels {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // Unloading may fail only if the module was already unloaded, which
            // cannot happen with this RAII wrapper.  Ignore the result in Drop.
            unsafe { let _ = hip_module_unload(self.module); }
            self.module = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build a kernel-argument pointer array from references.
//
// `hip_launch_kernel` (via hipModuleLaunchKernel) expects `void**` where each
// element is a pointer TO the kernel argument (i.e. &arg, not arg itself).
//
// The macro collects each expression as a local variable, takes its address,
// and casts to *mut c_void, building a temporary slice on the stack.
// ---------------------------------------------------------------------------
macro_rules! kargs {
    ( $( $arg:expr ),* $(,)? ) => {{
        // Bind each argument to a named local so we can take a stable address.
        // The variables are alive for the duration of the surrounding expression.
        #[allow(unused_mut)]
        let mut args: &mut [*mut std::ffi::c_void] = &mut [
            $( &mut { $arg } as *mut _ as *mut std::ffi::c_void ),*
        ];
        args
    }};
}

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
    let mut b = 64u32;          // RDNA2 wavefront; minimum sensible block
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
        embed_bf16: *const u16,  // [vocab, hidden] BF16 on GPU
        token_ids:  *const u32,  // [n_tokens] on GPU
        out:        *mut f32,    // [n_tokens, hidden] on GPU
        n_tokens:   i32,
        hidden:     i32,
    ) -> Result<(), HipError> {
        let bsz = block_size(hidden as u32, 256);
        let grid  = (n_tokens as u32, 1, 1);
        let block = (bsz, 1, 1);

        // Mutable copies so we can take &mut for the args array.
        let mut p_embed     = embed_bf16;
        let mut p_token_ids = token_ids;
        let mut p_out       = out;
        let mut p_hidden    = hidden;

        hip_launch_kernel(
            self.fn_embed_lookup,
            grid,
            block,
            0,              // no dynamic shared memory
            ptr::null_mut(), // default stream
            &mut [
                &mut p_embed     as *mut *const u16 as *mut std::ffi::c_void,
                &mut p_token_ids as *mut *const u32 as *mut std::ffi::c_void,
                &mut p_out       as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_hidden    as *mut i32        as *mut std::ffi::c_void,
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
        out:      *mut f32,    // [n_tokens, hidden] on GPU
        x:        *const f32,  // [n_tokens, hidden] on GPU
        w:        *const f32,  // [hidden] on GPU
        n_tokens: i32,
        hidden:   i32,
        eps:      f32,
    ) -> Result<(), HipError> {
        let bsz = block_size(hidden as u32, 1024);
        let shared_bytes = bsz * std::mem::size_of::<f32>() as u32;

        let grid  = (n_tokens as u32, 1, 1);
        let block = (bsz, 1, 1);

        let mut p_out    = out;
        let mut p_x      = x;
        let mut p_w      = w;
        let mut p_hidden = hidden;
        let mut p_eps    = eps;

        hip_launch_kernel(
            self.fn_rms_norm,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_out    as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_x      as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_w      as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_hidden as *mut i32        as *mut std::ffi::c_void,
                &mut p_eps    as *mut f32        as *mut std::ffi::c_void,
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
        x:        *mut f32,    // [n_tokens, dim] on GPU
        bias:     *const f32,  // [dim] on GPU
        n_tokens: i32,
        dim:      i32,
    ) -> Result<(), HipError> {
        let bsz = block_size(dim as u32, 1024);
        let grid  = (n_tokens as u32, 1, 1);
        let block = (bsz, 1, 1);

        let mut p_x    = x;
        let mut p_bias = bias;
        let mut p_dim  = dim;

        hip_launch_kernel(
            self.fn_add_bias,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_x    as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_bias as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_dim  as *mut i32        as *mut std::ffi::c_void,
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
        x:         *mut f32,  // [n_tokens, n_heads, head_dim] on GPU
        n_tokens:  i32,
        n_heads:   i32,
        head_dim:  i32,
        pos_start: i32,
        theta:     f32,
    ) -> Result<(), HipError> {
        let half = (head_dim / 2) as u32;
        let grid  = (n_tokens as u32, n_heads as u32, 1);
        let block = (half, 1, 1);

        let mut p_x         = x;
        let mut p_n_heads   = n_heads;
        let mut p_head_dim  = head_dim;
        let mut p_pos_start = pos_start;
        let mut p_theta     = theta;

        hip_launch_kernel(
            self.fn_rope_inplace,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_x         as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_n_heads   as *mut i32      as *mut std::ffi::c_void,
                &mut p_head_dim  as *mut i32      as *mut std::ffi::c_void,
                &mut p_pos_start as *mut i32      as *mut std::ffi::c_void,
                &mut p_theta     as *mut f32      as *mut std::ffi::c_void,
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
        gate:     *mut f32,   // [n_tokens, inter] on GPU — modified in-place
        up:       *const f32, // [n_tokens, inter] on GPU
        n_tokens: i32,
        inter:    i32,
    ) -> Result<(), HipError> {
        let total       = (n_tokens * inter) as u32;
        let block_x: u32 = 256;
        let grid  = (ceil_div(total, block_x), 1, 1);
        let block = (block_x, 1, 1);

        let mut p_gate  = gate;
        let mut p_up    = up;
        let mut p_total = (n_tokens * inter) as i32;

        hip_launch_kernel(
            self.fn_silu_mul,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_gate  as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_up    as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_total as *mut i32        as *mut std::ffi::c_void,
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
        scores:   *mut f32,  // [n_rows, seq_len] on GPU
        n_rows:   i32,
        seq_len:  i32,
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

        let grid  = (n_rows as u32, 1, 1);
        let block = (bsz, 1, 1);

        let mut p_scores  = scores;
        let mut p_seq_len = seq_len;

        hip_launch_kernel(
            self.fn_softmax_rows,
            grid,
            block,
            shared_bytes,
            ptr::null_mut(),
            &mut [
                &mut p_scores  as *mut *mut f32 as *mut std::ffi::c_void,
                &mut p_seq_len as *mut i32      as *mut std::ffi::c_void,
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
        cache:       *mut f32,    // [max_seq, kv_dim] on GPU
        new_kv:      *const f32,  // [n_tokens, kv_dim] on GPU
        n_tokens:    i32,
        kv_dim:      i32,
        current_len: i32,
    ) -> Result<(), HipError> {
        let total      = (n_tokens * kv_dim) as u32;
        let block_x    = total.min(1024);
        let grid  = (1, 1, 1);
        let block = (block_x, 1, 1);

        let mut p_cache       = cache;
        let mut p_new_kv      = new_kv;
        let mut p_n_tokens    = n_tokens;
        let mut p_kv_dim      = kv_dim;
        let mut p_current_len = current_len;

        hip_launch_kernel(
            self.fn_copy_kv_cache,
            grid,
            block,
            0,
            ptr::null_mut(),
            &mut [
                &mut p_cache       as *mut *mut f32   as *mut std::ffi::c_void,
                &mut p_new_kv      as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_n_tokens    as *mut i32        as *mut std::ffi::c_void,
                &mut p_kv_dim      as *mut i32        as *mut std::ffi::c_void,
                &mut p_current_len as *mut i32        as *mut std::ffi::c_void,
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
        let mut p_n   = n;

        hip_launch_kernel(
            self.fn_f32_to_bf16,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            ptr::null_mut(),
            &mut [
                &mut p_src as *mut *const f32 as *mut std::ffi::c_void,
                &mut p_dst as *mut *mut u16   as *mut std::ffi::c_void,
                &mut p_n   as *mut i32        as *mut std::ffi::c_void,
            ],
        )
    }
}
