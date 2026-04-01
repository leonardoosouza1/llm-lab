// ============================================================================
// lnn/cfc.rs — CfC (Closed-form Continuous) Liquid Correction Layer
// ============================================================================
//
// Camada líquida que corrige erros de quantização nos hidden states.
// Inspirada em Liquid Time-Constant Networks, mas com solução fechada (CfC)
// — sem solver ODE, custo computacional mínimo.
//
// Equação do neurônio líquido CfC:
//   f = sigmoid(x @ Wf + h_prev @ Wf2 + bf)      ← gate: quanto muda
//   g = tanh(x @ Wg + h_prev @ Wg2 + bg)         ← candidate: novo estado
//   h_new = f * h_prev + (1-f) * g                ← update adaptativo
//   correction = h_new @ Wout + bout               ← projeção para hidden dim
//
// O "τ adaptativo" (time-constant) é implícito no gate f:
//   - Tokens pouco informativos: f ≈ 1 → h muda pouco
//   - Tokens críticos: f ≈ 0 → h absorve o candidate g (mudança rápida)
//
// Integração com Gravity:
//   - gravity > threshold → aplica CfC correction ao hidden state
//   - gravity ≤ threshold → bypass (custo zero, sem correção)
//
// Parâmetros:
//   - cfc_dim=512 → ~6M parâmetros (0.07% de um modelo 8B)
//   - Mantidos em F32 no device
//   - Inicializados com Xavier uniform × 0.01 (correção começa quase zero)
// ============================================================================

use std::sync::Arc;
use compute_core::{ComputeBackend, AnyTensor};
use serde::Serialize;

/// Configuração da CfC layer.
#[derive(Debug, Clone)]
pub struct CfcConfig {
    /// Dimensão do estado interno CfC (tipicamente hidden/16 ou hidden/8).
    pub cfc_dim: usize,
    /// Dimensão dos hidden states do transformer.
    pub hidden_dim: usize,
    /// Learning rate para treinamento online (fase futura, não usado ainda).
    pub lr: f32,
}

impl CfcConfig {
    /// Cria configuração com cfc_dim = hidden_dim / divisor.
    pub fn new(hidden_dim: usize, cfc_divisor: usize) -> Self {
        let cfc_dim = (hidden_dim / cfc_divisor).max(64);
        Self {
            cfc_dim,
            hidden_dim,
            lr: 1e-4,
        }
    }
}

/// Pesos da CfC layer (todos F32, mantidos em GPU).
pub struct CfcWeights {
    // Gate f: f = sigmoid(x @ Wf + h_prev @ Wf2 + bf)
    /// [cfc_dim, hidden_dim] — projeção do input para a gate
    pub wf: AnyTensor,
    /// [cfc_dim, cfc_dim] — recorrência da gate
    pub wf2: AnyTensor,
    /// [cfc_dim] — bias da gate
    pub bf: AnyTensor,

    // Candidate g: g = tanh(x @ Wg + h_prev @ Wg2 + bg)
    /// [cfc_dim, hidden_dim] — projeção do input para o candidate
    pub wg: AnyTensor,
    /// [cfc_dim, cfc_dim] — recorrência do candidate
    pub wg2: AnyTensor,
    /// [cfc_dim] — bias do candidate
    pub bg: AnyTensor,

    // Output projection: correction = h_new @ Wout + bout
    /// [hidden_dim, cfc_dim] — projeção do estado final para o hidden dim
    pub wout: AnyTensor,
    /// [hidden_dim] — bias de saída
    pub bout: AnyTensor,
}

/// Estado da CfC layer.
pub struct CfcState {
    /// Estado recorrente h [cfc_dim] F32 no device.
    /// Mantém a memória de curto prazo da CfC.
    pub h: AnyTensor,
}

/// CfC Liquid Correction Layer.
///
/// Fornece correção adaptativa de erros de quantização através de uma
/// rede recorrente de dimensão reduzida. O estado persiste durante uma
/// sequência de geração.
pub struct CfcLayer {
    pub config: CfcConfig,
    pub weights: CfcWeights,
    pub state: CfcState,
    /// Buffer temporário para correction output [1, hidden_dim]
    pub(crate) correction_buf: AnyTensor,
    /// Buffer temporário para f_proj [cfc_dim]
    f_proj_buf: AnyTensor,
    /// Buffer temporário para g_proj [cfc_dim]
    g_proj_buf: AnyTensor,
}

impl CfcLayer {
    /// Cria uma nova CfC layer com pesos inicializados com Xavier uniform × 0.01.
    ///
    /// Os pesos são inicialmente muito pequenos → correção começa próxima a zero
    /// e aumenta gradualmente conforme o treinamento online (futuro).
    ///
    /// `backend` — backend de compute (ROCm, OpenCL, etc.)
    /// `config` — configuração (dimensões, learning rate)
    ///
    /// Aloca todos os buffers necessários no device.
    pub fn new(backend: &Arc<dyn ComputeBackend>, config: CfcConfig) -> Self {
        let cd = config.cfc_dim;
        let hd = config.hidden_dim;

        // ── Inicialização Xavier uniform ───────────────────────────────────
        // range = sqrt(6 / (fan_in + fan_out))
        // Com fator × 0.01 para correção começar quase zero.
        let init_weights = |fan_in: usize, fan_out: usize| -> Vec<f32> {
            let range = (6.0 / (fan_in + fan_out) as f32).sqrt();
            let mut rng = 0x1234_5678u64;
            (0..fan_in * fan_out)
                .map(|_| {
                    // LCG simples para reprodutibilidade
                    rng = rng
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let u = (rng >> 33) as f32 / u32::MAX as f32;
                    (u * 2.0 - 1.0) * range * 0.01 // Escala extra pequena
                })
                .collect()
        };

        let weights = CfcWeights {
            wf: backend.upload_f32(&init_weights(cd, hd), &[cd, hd]),
            wf2: backend.upload_f32(&init_weights(cd, cd), &[cd, cd]),
            bf: backend.upload_f32(&vec![0.0f32; cd], &[cd]),
            wg: backend.upload_f32(&init_weights(cd, hd), &[cd, hd]),
            wg2: backend.upload_f32(&init_weights(cd, cd), &[cd, cd]),
            bg: backend.upload_f32(&vec![0.0f32; cd], &[cd]),
            wout: backend.upload_f32(&init_weights(hd, cd), &[hd, cd]),
            bout: backend.upload_f32(&vec![0.0f32; hd], &[hd]),
        };

        let state = CfcState {
            h: backend.upload_f32(&vec![0.0f32; cd], &[cd]),
        };

        let correction_buf = backend.alloc_f32(&[1, hd]);
        let f_proj_buf = backend.alloc_f32(&[cd]);
        let g_proj_buf = backend.alloc_f32(&[cd]);

        Self {
            config,
            weights,
            state,
            correction_buf,
            f_proj_buf,
            g_proj_buf,
        }
    }

    /// Forward da CfC: recebe hidden state, retorna correção.
    ///
    /// Executa a atualização do neurônio líquido:
    ///   1. f_proj = x @ Wf^T + h_prev @ Wf2^T + bf
    ///   2. g_proj = x @ Wg^T + h_prev @ Wg2^T + bg
    ///   3. h_new = sigmoid(f) * h_prev + (1-sigmoid(f)) * tanh(g)
    ///   4. correction = h_new @ Wout^T + bout
    ///
    /// `x` — [1, hidden_dim] F32, hidden state do último token (já em F32)
    /// `backend` — backend de compute
    ///
    /// Retorna referência ao buffer de correção [1, hidden_dim].
    /// A correção deve ser SOMADA ao hidden state:
    ///   hidden_corrigido = hidden_quantizado + correction
    ///
    /// O estado interno h é atualizado in-place durante o forward.
    pub fn forward(
        &mut self,
        backend: &Arc<dyn ComputeBackend>,
        x: &AnyTensor,
    ) -> Result<&AnyTensor, compute_core::BackendError> {
        let cd = self.config.cfc_dim;
        let hd = self.config.hidden_dim;

        // ── Step 1: f_proj = x @ Wf^T + h_prev @ Wf2^T + bf ───────────────

        // x @ Wf^T: [1, hd] × [cd, hd]^T = [1, cd]
        backend.sgemm(&compute_core::SgemmParams {
            trans_a: false,
            trans_b: true,
            m: 1,
            n: cd,
            k: hd,
            alpha: 1.0,
            beta: 0.0,
            a: x,
            lda: hd,
            a_offset: 0,
            b: &self.weights.wf,
            ldb: hd,
            b_offset: 0,
            c: &self.f_proj_buf,
            ldc: cd,
            c_offset: 0,
        })?;

        // + h_prev @ Wf2^T: acumula (beta=1.0)
        backend.sgemm(&compute_core::SgemmParams {
            trans_a: false,
            trans_b: true,
            m: 1,
            n: cd,
            k: cd,
            alpha: 1.0,
            beta: 1.0, // acumula
            a: &self.state.h,
            lda: cd,
            a_offset: 0,
            b: &self.weights.wf2,
            ldb: cd,
            b_offset: 0,
            c: &self.f_proj_buf,
            ldc: cd,
            c_offset: 0,
        })?;

        // + bf via add_bias_inplace
        backend.add_bias_inplace(&compute_core::AddBiasParams {
            tensor: &self.f_proj_buf,
            bias: &self.weights.bf,
            n_tokens: 1,
            dim: cd,
        })?;

        // ── Step 2: g_proj = x @ Wg^T + h_prev @ Wg2^T + bg ───────────────

        // x @ Wg^T
        backend.sgemm(&compute_core::SgemmParams {
            trans_a: false,
            trans_b: true,
            m: 1,
            n: cd,
            k: hd,
            alpha: 1.0,
            beta: 0.0,
            a: x,
            lda: hd,
            a_offset: 0,
            b: &self.weights.wg,
            ldb: hd,
            b_offset: 0,
            c: &self.g_proj_buf,
            ldc: cd,
            c_offset: 0,
        })?;

        // + h_prev @ Wg2^T
        backend.sgemm(&compute_core::SgemmParams {
            trans_a: false,
            trans_b: true,
            m: 1,
            n: cd,
            k: cd,
            alpha: 1.0,
            beta: 1.0,
            a: &self.state.h,
            lda: cd,
            a_offset: 0,
            b: &self.weights.wg2,
            ldb: cd,
            b_offset: 0,
            c: &self.g_proj_buf,
            ldc: cd,
            c_offset: 0,
        })?;

        // + bg
        backend.add_bias_inplace(&compute_core::AddBiasParams {
            tensor: &self.g_proj_buf,
            bias: &self.weights.bg,
            n_tokens: 1,
            dim: cd,
        })?;

        // ── Step 3: CfC gate update (kernel GPU) ──────────────────────────
        // h_new = sigmoid(f_proj) * h_prev + (1-sigmoid(f_proj)) * tanh(g_proj)
        // O kernel cfc_forward:
        //   1. Computa f = sigmoid(f_proj)
        //   2. Computa g_tanh = tanh(g_proj)
        //   3. Atualiza h_prev ← f * h_prev + (1-f) * g_tanh (in-place)
        //   4. Computa correction = h_new @ Wout^T + bout

        backend.cfc_forward(&compute_core::CfcForwardParams {
            x,
            h_prev: &self.state.h,
            wf: &self.weights.wf,
            wf2: &self.weights.wf2,
            bf: &self.weights.bf,
            wg: &self.weights.wg,
            wg2: &self.weights.wg2,
            bg: &self.weights.bg,
            wout: &self.weights.wout,
            bout: &self.weights.bout,
            h_out: &self.state.h,     // in-place state update
            correction: &self.correction_buf,
            n_tokens: 1,
            hidden: hd,
            cfc_dim: cd,
        })?;

        // Se o kernel cfc_forward não estiver implementado ainda,
        // fazer fallback com projeções manuais (comentado para agora):
        //
        // // Step 3a: f = sigmoid(f_proj)
        // backend.sigmoid_inplace(&self.f_proj_buf)?;
        //
        // // Step 3b: g_tanh = tanh(g_proj)
        // backend.tanh_inplace(&self.g_proj_buf)?;
        //
        // // Step 3c: h_new = f * h_prev + (1-f) * g_tanh
        // // [Seria necessário kernel customizado ou múltiplas ops]
        //
        // // Step 4: correction = h_new @ Wout^T + bout
        // backend.sgemm(&compute_core::SgemmParams {
        //     trans_a: false, trans_b: true,
        //     m: 1, n: hd, k: cd,
        //     alpha: 1.0, beta: 0.0,
        //     a: &self.state.h, lda: cd, a_offset: 0,
        //     b: &self.weights.wout, ldb: cd, b_offset: 0,
        //     c: &self.correction_buf, ldc: hd, c_offset: 0,
        // })?;
        // backend.add_bias_inplace(&compute_core::AddBiasParams {
        //     tensor: &self.correction_buf,
        //     bias: &self.weights.bout,
        //     n_tokens: 1,
        //     dim: hd,
        // })?;

        Ok(&self.correction_buf)
    }

    /// Reset do estado CfC (no início de cada geração).
    ///
    /// Deve ser chamado quando iniciando uma nova sequência de tokens,
    /// para "esquecer" o contexto anterior.
    pub fn reset(&mut self, backend: &Arc<dyn ComputeBackend>) {
        let zeros = vec![0.0f32; self.config.cfc_dim];
        self.state.h = backend.upload_f32(&zeros, &[self.config.cfc_dim]);
    }

    /// Número de parâmetros treináveis.
    ///
    /// Retorna o total de elementos float32 que formam os pesos.
    pub fn param_count(&self) -> usize {
        let cd = self.config.cfc_dim;
        let hd = self.config.hidden_dim;
        // wf + wf2 + bf + wg + wg2 + bg + wout + bout
        cd * hd + cd * cd + cd + cd * hd + cd * cd + cd + hd * cd + hd
    }

    /// VRAM estimada em bytes (todos os pesos em F32).
    pub fn vram_bytes(&self) -> usize {
        self.param_count() * 4 // F32 = 4 bytes
    }

    /// Retorna o estado interno (para inspecção ou debug).
    pub fn state_ref(&self) -> &AnyTensor {
        &self.state.h
    }

    /// Retorna referência mutável ao estado (para injeção de estado de fora).
    pub fn state_mut(&mut self) -> &mut AnyTensor {
        &mut self.state.h
    }
}

/// Telemetria da CfC para broadcast WebSocket.
///
/// Contém estatísticas instantâneas sobre a CfC layer:
/// ativação, tamanho da correção, uso de VRAM, etc.
#[derive(Debug, Clone, Serialize)]
pub struct CfcTelemetry {
    /// Norma L2 do estado interno h [cfc_dim].
    /// Indica a intensidade de ativação da memória líquida.
    pub state_norm: f32,
    /// Norma L2 da correção [hidden_dim].
    /// Indica o quanto a CfC está ajustando os hidden states.
    pub correction_norm: f32,
    /// Número total de parâmetros (para informação).
    pub param_count: usize,
    /// VRAM em megabytes (estimada).
    pub vram_mb: f32,
}

impl CfcTelemetry {
    /// Cria telemetria a partir de uma CfC layer.
    ///
    /// Requer backend para fazer downloads dos buffers.
    pub fn from_layer(
        layer: &CfcLayer,
        _backend: &Arc<dyn ComputeBackend>,
    ) -> Self {
        let param_count = layer.param_count();
        let vram_bytes = layer.vram_bytes();

        // Normas (para a versão actual, usar 0 como placeholder se sem acesso a download)
        // Versão futura: fazer download dos buffers e calcular normas reais
        let state_norm = 0.0f32; // TODO: download state e calcular
        let correction_norm = 0.0f32; // TODO: download correction_buf e calcular

        Self {
            state_norm,
            correction_norm,
            param_count,
            vram_mb: vram_bytes as f32 / (1024.0 * 1024.0),
        }
    }
}
