// ============================================================================
// backend.rs — OpenCL ComputeBackend com suporte a UMA (memória unificada)
// ============================================================================
//
// Dois caminhos de alocação/transferência:
//
//   GPU discreta (is_uma = false):
//     • CL_MEM_READ_WRITE — buffer vive na VRAM
//     • upload_f32 → enqueue_write_buffer (cópia PCIe)
//     • download_f32 → enqueue_read_buffer (cópia PCIe)
//
//   Intel iGPU UMA (is_uma = true):
//     • CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR — buffer em RAM
//       compartilhada por CPU e GPU (sem barramento no caminho)
//     • upload_f32 → mapa o buffer (ponteiro direto), memcpy, desmapa
//       O driver Intel sabe que não há cópia real necessária.
//     • download_f32 → queue.finish() para garantir visibilidade, depois
//       mapa para leitura, memcpy, desmapa.
//       O "copy" é apenas um memcpy dentro da mesma RAM (ns, não µs).
//
// O pipeline de inferência (kernels, GEMMs) é idêntico para ambos os caminhos.
// ============================================================================

use std::any::Any;
use std::sync::Arc;

use opencl3::memory::{Buffer, ClMem, CL_MAP_READ, CL_MAP_WRITE, CL_MEM_ALLOC_HOST_PTR, CL_MEM_READ_WRITE};
use opencl3::types::{cl_float, cl_int, cl_mem, cl_uint, cl_ushort, CL_BLOCKING};

use compute_core::{
    AddBiasParams, AnyTensor, BackendError, ComputeBackend, DType,
    EmbedParams, GemmBf16Params, KvCacheParams, RmsNormParams,
    RopeParams, SgemmParams, SiluMulParams, SoftmaxParams,
};

use crate::context::OclContext;
use crate::kernels::OclKernels;

// ── Buffer wrappers ────────────────────────────────────────────────────────

/// Wrapper type-safe para Buffer OpenCL F32.
/// Na UMA: `buf` foi alocado com `CL_MEM_ALLOC_HOST_PTR` — vive em RAM física
/// acessível tanto pela CPU quanto pela GPU Intel.
pub struct OclBufferF32 {
    pub buf: Buffer<cl_float>,
    pub len: usize,
}
// SAFETY: Buffer OpenCL é representado por um handle opaco (cl_mem).
// O acesso concorrente é serializado pelo CommandQueue.
unsafe impl Send for OclBufferF32 {}
unsafe impl Sync for OclBufferF32 {}

/// Wrapper type-safe para Buffer OpenCL U16 (BF16 bits).
/// Na UMA: alocado com `CL_MEM_ALLOC_HOST_PTR`.
pub struct OclBufferU16 {
    pub buf: Buffer<cl_ushort>,
    pub len: usize,
}
unsafe impl Send for OclBufferU16 {}
unsafe impl Sync for OclBufferU16 {}

// ── Backend ────────────────────────────────────────────────────────────────

pub struct OpenClBackend {
    ctx:    OclContext,
    kernels: OclKernels,
    name:   String,
    /// Copia de ctx.is_uma para acesso rápido nos métodos de transferência.
    is_uma: bool,
}

// SAFETY: OclContext e OclKernels contêm handles OpenCL que podem ser usados
// de múltiplas threads desde que o acesso à queue seja serializado.
unsafe impl Send for OpenClBackend {}
unsafe impl Sync for OpenClBackend {}

impl OpenClBackend {
    /// Detecta e inicializa o backend OpenCL (prefere Intel iGPU).
    pub fn probe() -> Result<Self, BackendError> {
        let ctx = OclContext::detect()?;
        let is_uma = ctx.is_uma;
        let name = format!(
            "OpenCL {} ({})",
            ctx.device.name().unwrap_or_else(|_| "Unknown".into()),
            if is_uma { "UMA/zero-copy" } else { "discreta" }
        );
        let kernels = OclKernels::build(&ctx.context)?;
        Ok(Self { ctx, kernels, name, is_uma })
    }

    // ── Helpers de downcast ───────────────────────────────────────────────

    fn f32_buf(t: &AnyTensor) -> &OclBufferF32 {
        t.downcast_ref::<OclBufferF32>()
            .expect("OpenClBackend: AnyTensor interno não é OclBufferF32")
    }

    fn u16_buf(t: &AnyTensor) -> &OclBufferU16 {
        t.downcast_ref::<OclBufferU16>()
            .expect("OpenClBackend: AnyTensor interno não é OclBufferU16")
    }

    // ── Helpers de alocação ───────────────────────────────────────────────

    fn alloc_f32_buf(&self, len: usize) -> Result<OclBufferF32, BackendError> {
        // UMA: CL_MEM_ALLOC_HOST_PTR diz ao driver para alocar em RAM física
        // acessível pela CPU e pela GPU sem cópia entre barramentos.
        // GPU discreta: CL_MEM_READ_WRITE (VRAM padrão).
        let flags = if self.is_uma {
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
        } else {
            CL_MEM_READ_WRITE
        };
        let buf = unsafe {
            Buffer::<cl_float>::create(
                &self.ctx.context,
                flags,
                len,
                std::ptr::null_mut(),
            )
            .map_err(|e| BackendError::Alloc(format!("OpenCL alloc F32 ({len}): {e}")))?
        };
        Ok(OclBufferF32 { buf, len })
    }

    fn alloc_u16_buf(&self, len: usize) -> Result<OclBufferU16, BackendError> {
        let flags = if self.is_uma {
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
        } else {
            CL_MEM_READ_WRITE
        };
        let buf = unsafe {
            Buffer::<cl_ushort>::create(
                &self.ctx.context,
                flags,
                len,
                std::ptr::null_mut(),
            )
            .map_err(|e| BackendError::Alloc(format!("OpenCL alloc U16 ({len}): {e}")))?
        };
        Ok(OclBufferU16 { buf, len })
    }

    // ── Upload UMA (zero-copy via map/unmap) ──────────────────────────────
    //
    // Na UMA, mapear um buffer CL_MEM_ALLOC_HOST_PTR retorna um ponteiro
    // direto para a RAM onde o buffer já vive — sem mover bytes entre
    // espaços de endereçamento. O memcpy abaixo é RAM→RAM (ns, não µs).
    //
    // Na GPU discreta (is_uma = false), usamos enqueue_write_buffer que
    // aciona a cópia via PCIe.
    //

    fn upload_f32_into_buf(&self, data: &[f32], buf: &mut OclBufferF32) {
        if self.is_uma {
            // Mapa o buffer para escrita (não invalida cache — MAP_WRITE)
            // Na UMA, map retorna o ponteiro da RAM diretamente.
            let byte_size = data.len() * std::mem::size_of::<f32>();
            let mut mapped_ptr: cl_mem = std::ptr::null_mut();
            let _ev = unsafe {
                self.ctx.queue.enqueue_map_buffer(
                    &buf.buf, CL_BLOCKING,
                    CL_MAP_WRITE,
                    0, byte_size,
                    &mut mapped_ptr,
                    &[],
                ).expect("UMA: falha ao mapear buffer F32 para escrita")
            };
            // memcpy dentro da mesma RAM — ordem de ns
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    mapped_ptr as *mut f32,
                    data.len(),
                );
            }
            // Desmapa → sinaliza ao driver que a CPU terminou de escrever
            let _ = unsafe {
                self.ctx.queue.enqueue_unmap_mem_object(buf.buf.get(), mapped_ptr, &[])
            };
        } else {
            let ev = unsafe {
                self.ctx.queue
                    .enqueue_write_buffer(&mut buf.buf, CL_BLOCKING, 0, data, &[])
                    .expect("OpenCL: falha ao escrever buffer F32")
            };
            ev.wait().expect("OpenCL: wait upload F32");
        }
    }

    fn upload_u16_into_buf(&self, data: &[u16], buf: &mut OclBufferU16) {
        if self.is_uma {
            let byte_size = data.len() * std::mem::size_of::<u16>();
            let mut mapped_ptr: cl_mem = std::ptr::null_mut();
            let _ev = unsafe {
                self.ctx.queue.enqueue_map_buffer(
                    &buf.buf, CL_BLOCKING,
                    CL_MAP_WRITE,
                    0, byte_size,
                    &mut mapped_ptr,
                    &[],
                ).expect("UMA: falha ao mapear buffer U16 para escrita")
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    mapped_ptr as *mut u16,
                    data.len(),
                );
            }
            let _ = unsafe {
                self.ctx.queue.enqueue_unmap_mem_object(buf.buf.get(), mapped_ptr, &[])
            };
        } else {
            let ev = unsafe {
                self.ctx.queue
                    .enqueue_write_buffer(&mut buf.buf, CL_BLOCKING, 0, data, &[])
                    .expect("OpenCL: falha ao escrever buffer U16")
            };
            ev.wait().expect("OpenCL: wait upload U16");
        }
    }
}

impl ComputeBackend for OpenClBackend {
    fn name(&self) -> &str {
        &self.name
    }

    // ── Gerenciamento de memória ──────────────────────────────────────────

    fn alloc_f32(&self, shape: &[usize]) -> AnyTensor {
        let len: usize = shape.iter().product();
        // alloc_f32_buf só falha em OOM — propagar via panic é aceitável aqui
        // pois a API do trait não retorna Result
        let buf = self.alloc_f32_buf(len).expect("OpenCL: falha ao alocar F32");
        AnyTensor::new(
            Arc::new(buf) as Arc<dyn Any + Send + Sync>,
            shape.to_vec(),
            DType::F32,
        )
    }

    fn alloc_bf16(&self, shape: &[usize]) -> AnyTensor {
        let len: usize = shape.iter().product();
        let buf = self.alloc_u16_buf(len).expect("OpenCL: falha ao alocar BF16");
        AnyTensor::new(
            Arc::new(buf) as Arc<dyn Any + Send + Sync>,
            shape.to_vec(),
            DType::BF16,
        )
    }

    fn upload_f32(&self, data: &[f32], shape: &[usize]) -> AnyTensor {
        let mut buf = self
            .alloc_f32_buf(data.len())
            .expect("OpenCL: falha ao alocar F32 para upload");
        self.upload_f32_into_buf(data, &mut buf);
        AnyTensor::new(
            Arc::new(buf) as Arc<dyn Any + Send + Sync>,
            shape.to_vec(),
            DType::F32,
        )
    }

    fn upload_bf16(&self, data: &[u16], shape: &[usize]) -> AnyTensor {
        let mut buf = self
            .alloc_u16_buf(data.len())
            .expect("OpenCL: falha ao alocar U16 para upload");
        self.upload_u16_into_buf(data, &mut buf);
        AnyTensor::new(
            Arc::new(buf) as Arc<dyn Any + Send + Sync>,
            shape.to_vec(),
            DType::BF16,
        )
    }

    fn download_f32(&self, t: &AnyTensor) -> Vec<f32> {
        let buf = Self::f32_buf(t);
        let mut result = vec![0f32; buf.len];

        if self.is_uma {
            // UMA: garante que todos os kernels terminaram (memory barrier)
            // Depois mapa para leitura — na UMA retorna ponteiro direto para RAM.
            // O memcpy abaixo é RAM→RAM — sem PCIe, sem latência de cópia.
            self.ctx.queue.finish().expect("UMA: falha ao sincronizar queue");

            let byte_size = buf.len * std::mem::size_of::<f32>();
            let mut mapped_ptr: cl_mem = std::ptr::null_mut();
            let _ev = unsafe {
                self.ctx.queue.enqueue_map_buffer(
                    &buf.buf, CL_BLOCKING,
                    CL_MAP_READ,
                    0, byte_size,
                    &mut mapped_ptr,
                    &[],
                ).expect("UMA: falha ao mapear buffer F32 para leitura")
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped_ptr as *const f32,
                    result.as_mut_ptr(),
                    buf.len,
                );
            }
            let _ = unsafe {
                self.ctx.queue.enqueue_unmap_mem_object(buf.buf.get(), mapped_ptr, &[])
            };
        } else {
            // GPU discreta: cópia via PCIe (enqueue_read_buffer)
            let ev = unsafe {
                self.ctx
                    .queue
                    .enqueue_read_buffer(&buf.buf, CL_BLOCKING, 0, &mut result, &[])
                    .expect("OpenCL: falha ao ler buffer F32")
            };
            ev.wait().expect("OpenCL: wait download F32");
        }

        result
    }

    // ── GEMM ─────────────────────────────────────────────────────────────

    fn gemm_bf16_f32(&self, p: &GemmBf16Params) -> Result<(), BackendError> {
        let a_buf = Self::f32_buf(p.a);
        let b_buf = Self::u16_buf(p.b);
        let c_buf = Self::f32_buf(p.c);

        let m = p.m as cl_int;
        let n = p.n as cl_int;
        let k = p.k as cl_int;
        let alpha: cl_float = p.alpha;
        let beta: cl_float = p.beta;

        let kernel = &self.kernels.gemm_bf16;
        unsafe {
            kernel
                .set_arg(0, &a_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &b_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &c_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(3, &m)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(4, &n)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(5, &k)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(6, &alpha)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(7, &beta)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            let a_off = p.a_offset as cl_int;
            kernel
                .set_arg(8, &a_off)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            let gws = [p.m, p.n];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    2,
                    std::ptr::null(),
                    gws.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| BackendError::Gemm(e.to_string()))?;
            // in-order queue: nenhuma sincronização necessária entre kernels
        }
        Ok(())
    }

    fn sgemm(&self, p: &SgemmParams) -> Result<(), BackendError> {
        let a_buf = Self::f32_buf(p.a);
        let b_buf = Self::f32_buf(p.b);
        let c_buf = Self::f32_buf(p.c);

        let m = p.m as cl_int;
        let n = p.n as cl_int;
        let k = p.k as cl_int;
        let trans_a = p.trans_a as cl_int;
        let trans_b = p.trans_b as cl_int;
        let lda = p.lda as cl_int;
        let ldb = p.ldb as cl_int;
        let ldc = p.ldc as cl_int;
        let alpha: cl_float = p.alpha;
        let beta: cl_float = p.beta;

        let kernel = &self.kernels.sgemm;
        unsafe {
            kernel
                .set_arg(0, &a_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &b_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &c_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(3, &m)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(4, &n)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(5, &k)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(6, &trans_a)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(7, &trans_b)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(8, &lda)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(9, &ldb)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(10, &ldc)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(11, &alpha)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(12, &beta)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            let a_off = p.a_offset as cl_int;
            let b_off = p.b_offset as cl_int;
            let c_off = p.c_offset as cl_int;
            kernel
                .set_arg(13, &a_off)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(14, &b_off)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(15, &c_off)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            let gws = [p.m, p.n];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    2,
                    std::ptr::null(),
                    gws.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| BackendError::Gemm(e.to_string()))?;
            // in-order queue: nenhuma sincronização necessária entre kernels
        }
        Ok(())
    }

    // ── Kernels customizados ──────────────────────────────────────────────

    fn embed_lookup(&self, p: &EmbedParams) -> Result<(), BackendError> {
        // Upload temporário de token_ids para o device
        let mut ids_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.ctx.context,
                CL_MEM_READ_WRITE,
                p.n_tokens,
                std::ptr::null_mut(),
            )
            .map_err(|e| BackendError::Alloc(e.to_string()))?
        };
        // CL_BLOCKING: chamada já bloqueia até o upload concluir
        let _ev = unsafe {
            self.ctx
                .queue
                .enqueue_write_buffer(&mut ids_buf, CL_BLOCKING, 0, p.token_ids, &[])
                .map_err(|e| BackendError::Transfer(e.to_string()))?
        };

        let vocab_buf = Self::u16_buf(p.vocab);
        let out_buf = Self::f32_buf(p.out);
        let hidden = p.hidden_size as cl_int;

        // local_size para iteração sobre hidden (dimensão 1 do grid)
        let local_size = 64usize.min(p.hidden_size);

        let kernel = &self.kernels.embed_lookup;
        unsafe {
            kernel
                .set_arg(0, &vocab_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &ids_buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &out_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(3, &hidden)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            // Grid 2D: dim0 = n_tokens (1 work-item por token),
            //          dim1 = local_size (threads que cobrem hidden em paralelo)
            let gws = [p.n_tokens, local_size];
            let lws = [1usize, local_size];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    2,
                    std::ptr::null(),
                    gws.as_ptr(),
                    lws.as_ptr(),
                    &[],
                )
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
        }
        Ok(())
    }

    fn rms_norm(&self, p: &RmsNormParams) -> Result<(), BackendError> {
        let x_buf = Self::f32_buf(p.input);
        let w_buf = Self::f32_buf(p.weight);
        let o_buf = Self::f32_buf(p.output);
        let hidden = p.hidden_size as cl_int;
        let eps: cl_float = p.eps;

        // local_size: maior potência de 2 <= min(hidden_size, 256)
        let mut local_size = 1usize;
        while local_size * 2 <= p.hidden_size.min(256) {
            local_size *= 2;
        }

        let kernel = &self.kernels.rms_norm;
        unsafe {
            kernel
                .set_arg(0, &o_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &x_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &w_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(3, &hidden)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(4, &eps)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            // arg 5: __local scratch — local_size * sizeof(f32) bytes
            kernel
                .set_arg_local_buffer(5, local_size * std::mem::size_of::<f32>())
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            // Grid 1D: n_tokens grupos de local_size threads cada
            let gws = [p.n_tokens * local_size];
            let lws = [local_size];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    1,
                    std::ptr::null(),
                    gws.as_ptr(),
                    lws.as_ptr(),
                    &[],
                )
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
        }
        Ok(())
    }

    fn rope_inplace(&self, p: &RopeParams) -> Result<(), BackendError> {
        let t_buf = Self::f32_buf(p.tensor);
        let n_heads = p.n_heads as cl_int;
        let head_dim = p.head_dim as cl_int;
        let pos = p.pos_offset as cl_int;
        let theta: cl_float = p.theta;
        // Cada work-item processa um par (i, i+half) em um (tok, head)
        let half = (p.head_dim / 2).max(1);

        let kernel = &self.kernels.rope_inplace;
        unsafe {
            kernel
                .set_arg(0, &t_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &n_heads)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &head_dim)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(3, &pos)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(4, &theta)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            // Grid 2D: dim0 = n_tokens * half (tok decodificado dentro do kernel)
            //          dim1 = n_heads
            let gws = [p.n_tokens * half, p.n_heads];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    2,
                    std::ptr::null(),
                    gws.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
        }
        Ok(())
    }

    fn softmax_rows(&self, p: &SoftmaxParams) -> Result<(), BackendError> {
        let s_buf = Self::f32_buf(p.scores);
        let row_len = p.row_len as cl_int;

        // local_size: maior potência de 2 <= min(row_len, 256)
        let mut local_size = 1usize;
        while local_size * 2 <= p.row_len.min(256) {
            local_size *= 2;
        }

        let kernel = &self.kernels.softmax_rows;
        unsafe {
            kernel
                .set_arg(0, &s_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &row_len)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            // arg 2: __local scratch
            kernel
                .set_arg_local_buffer(2, local_size * std::mem::size_of::<f32>())
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            // arg 3: score_offset (para atenção multi-cabeça)
            let score_off = p.offset_elems as cl_int;
            kernel
                .set_arg(3, &score_off)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            // Grid 1D: n_rows grupos de local_size threads cada
            let gws = [p.n_rows * local_size];
            let lws = [local_size];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    1,
                    std::ptr::null(),
                    gws.as_ptr(),
                    lws.as_ptr(),
                    &[],
                )
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
        }
        Ok(())
    }

    fn silu_mul_inplace(&self, p: &SiluMulParams) -> Result<(), BackendError> {
        let g_buf = Self::f32_buf(p.gate);
        let u_buf = Self::f32_buf(p.up);
        let total_elems = p.n_tokens * p.intermediate_size;
        let total = total_elems as cl_int;

        // Alinha o grid a múltiplos de 256
        let global = ((total_elems + 255) / 256) * 256;

        let kernel = &self.kernels.silu_mul;
        unsafe {
            kernel
                .set_arg(0, &g_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &u_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &total)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            let gws = [global];
            let lws = [256usize];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    1,
                    std::ptr::null(),
                    gws.as_ptr(),
                    lws.as_ptr(),
                    &[],
                )
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
        }
        Ok(())
    }

    fn copy_to_kv_cache(&self, p: &KvCacheParams) -> Result<(), BackendError> {
        let c_buf = Self::f32_buf(p.cache);
        let k_buf = Self::f32_buf(p.new_kv);
        let n_tokens = p.n_new_tokens as cl_int;
        let kv_dim = p.kv_dim as cl_int;
        let insert_pos = p.insert_pos as cl_int;

        let total_elems = p.n_new_tokens * p.kv_dim;
        let global = ((total_elems + 255) / 256) * 256;

        let kernel = &self.kernels.copy_kv_cache;
        unsafe {
            kernel
                .set_arg(0, &c_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &k_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &n_tokens)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(3, &kv_dim)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(4, &insert_pos)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            let gws = [global];
            let lws = [256usize];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    1,
                    std::ptr::null(),
                    gws.as_ptr(),
                    lws.as_ptr(),
                    &[],
                )
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
        }
        Ok(())
    }

    fn add_bias_inplace(&self, p: &AddBiasParams) -> Result<(), BackendError> {
        let t_buf = Self::f32_buf(p.tensor);
        let b_buf = Self::f32_buf(p.bias);
        let dim = p.dim as cl_int;

        let kernel = &self.kernels.add_bias;
        unsafe {
            kernel
                .set_arg(0, &t_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(1, &b_buf.buf)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
            kernel
                .set_arg(2, &dim)
                .map_err(|e| BackendError::Kernel(e.to_string()))?;

            // Grid 2D: (n_tokens, dim) — um work-item por (token, dimensão)
            let gws = [p.n_tokens, p.dim];
            let _ev = self
                .ctx
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    2,
                    std::ptr::null(),
                    gws.as_ptr(),
                    std::ptr::null(),
                    &[],
                )
                .map_err(|e| BackendError::Kernel(e.to_string()))?;
        }
        Ok(())
    }

    // ── Sync e informações ────────────────────────────────────────────────

    fn sync(&self) -> Result<(), BackendError> {
        self.ctx
            .queue
            .finish()
            .map_err(|e| BackendError::Kernel(format!("OpenCL finish: {e}")))
    }

    fn mem_info(&self) -> Option<(usize, usize)> {
        // OpenCL não expõe memória livre de forma portável; retorna None
        None
    }
}
