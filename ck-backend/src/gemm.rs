//! CK GEMM wrapper — fused GEMM + activation (SiLU, GELU, PassThrough)

use std::fmt;

/// Activation function fused into GEMM epilogue
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CkActivation {
    None,       // PassThrough — for attention Q@K.T
    SiLU,       // gate_proj path (Qwen, Llama, Mistral)
    GELU,       // Phi, Gemma path
}

impl fmt::Display for CkActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CkActivation::None => write!(f, "PassThrough"),
            CkActivation::SiLU => write!(f, "SiLU"),
            CkActivation::GELU => write!(f, "GELU"),
        }
    }
}

/// Configuration for a CK GEMM operation
#[derive(Debug, Clone)]
pub struct CkGemmConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub activation: CkActivation,
    pub input_type: CkDataType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CkDataType {
    BF16,
    FP16,
    FP32,
}

/// Result of CK GEMM operation
#[derive(Debug)]
pub enum CkError {
    UnsupportedShape { m: usize, n: usize, k: usize },
    KernelLaunchFailed(i32),
    NotAvailable,
}

impl fmt::Display for CkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CkError::UnsupportedShape { m, n, k } => {
                write!(f, "CK: unsupported shape {}x{}x{}", m, n, k)
            }
            CkError::KernelLaunchFailed(code) => write!(f, "CK kernel failed: {}", code),
            CkError::NotAvailable => write!(f, "CK backend not compiled"),
        }
    }
}

impl std::error::Error for CkError {}

// ── CPU fallback for testing without GPU ──

/// CPU reference GEMM with optional activation (for testing/fallback)
pub fn cpu_gemm_with_activation(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    activation: CkActivation,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = match activation {
                CkActivation::None => sum,
                CkActivation::SiLU => sum * sigmoid(sum),
                CkActivation::GELU => {
                    sum * 0.5
                        * (1.0
                            + ((2.0f32 / std::f32::consts::PI).sqrt()
                                * (sum + 0.044715 * sum.powi(3)))
                            .tanh())
                }
            };
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Check if CK backend is available at runtime
pub fn is_ck_available() -> bool {
    #[cfg(feature = "composable-kernel")]
    {
        true
    }
    #[cfg(not(feature = "composable-kernel"))]
    {
        false
    }
}

/// Attempt fused GEMM using CK. Returns Err if CK not available or shape unsupported.
pub fn ck_gemm_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    config: &CkGemmConfig,
) -> Result<(), CkError> {
    if !is_ck_available() {
        return Err(CkError::NotAvailable);
    }

    // When CK feature is enabled, this would call the actual GPU kernels
    // For now, use CPU fallback
    cpu_gemm_with_activation(a, b, c, config.m, config.n, config.k, config.activation);
    Ok(())
}

/// MLP forward with fused GEMM+SiLU for gate projection
/// gate = SiLU(x @ W_gate)  — 1 fused operation with CK
/// up   = x @ W_up          — standard GEMM
/// out  = gate * up          — elementwise
/// down = out @ W_down       — standard GEMM
pub fn fused_mlp_forward(
    x: &[f32],          // [batch, in_dim]
    w_gate: &[f32],     // [in_dim, intermediate]
    w_up: &[f32],       // [in_dim, intermediate]
    w_down: &[f32],     // [intermediate, out_dim]
    batch: usize,
    in_dim: usize,
    intermediate: usize,
    out_dim: usize,
) -> Vec<f32> {
    // 1. gate = SiLU(x @ W_gate) — FUSED
    let mut gate = vec![0.0f32; batch * intermediate];
    let gate_config = CkGemmConfig {
        m: batch,
        n: intermediate,
        k: in_dim,
        activation: CkActivation::SiLU,
        input_type: CkDataType::FP32,
    };

    match ck_gemm_f32(x, w_gate, &mut gate, &gate_config) {
        Ok(()) => {}
        Err(_) => {
            // Fallback to CPU
            cpu_gemm_with_activation(
                x,
                w_gate,
                &mut gate,
                batch,
                intermediate,
                in_dim,
                CkActivation::SiLU,
            );
        }
    }

    // 2. up = x @ W_up — standard GEMM
    let mut up = vec![0.0f32; batch * intermediate];
    cpu_gemm_with_activation(
        x,
        w_up,
        &mut up,
        batch,
        intermediate,
        in_dim,
        CkActivation::None,
    );

    // 3. out = gate * up — elementwise
    let out: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();

    // 4. down = out @ W_down — standard GEMM
    let mut result = vec![0.0f32; batch * out_dim];
    cpu_gemm_with_activation(
        &out,
        w_down,
        &mut result,
        batch,
        out_dim,
        intermediate,
        CkActivation::None,
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_cpu_gemm_passthrough() {
        // Simple 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let mut c = vec![0.0; 4];

        cpu_gemm_with_activation(&a, &b, &mut c, 2, 2, 3, CkActivation::None);

        // a[0,0]=1, a[0,1]=2, a[0,2]=3
        // b[0,0]=1, b[1,0]=3, b[2,0]=5
        // c[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        assert!((c[0] - 22.0).abs() < 1e-5);

        // a[0,0]=1, a[0,1]=2, a[0,2]=3
        // b[0,1]=2, b[1,1]=4, b[2,1]=6
        // c[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        assert!((c[1] - 28.0).abs() < 1e-5);

        // a[1,0]=4, a[1,1]=5, a[1,2]=6
        // b[0,0]=1, b[1,0]=3, b[2,0]=5
        // c[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        assert!((c[2] - 49.0).abs() < 1e-5);

        // a[1,0]=4, a[1,1]=5, a[1,2]=6
        // b[0,1]=2, b[1,1]=4, b[2,1]=6
        // c[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        assert!((c[3] - 64.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_gemm_silu() {
        // Single element: 1x1 @ 1x1 = 1x1
        let a = vec![2.0]; // 1x1
        let b = vec![3.0]; // 1x1
        let mut c = vec![0.0; 1];

        cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::SiLU);

        // result = 2*3 = 6
        // silu(6) = 6 * sigmoid(6)
        let expected = 6.0 * sigmoid(6.0);
        assert!((c[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_gemm_gelu() {
        // Single element: 1x1 @ 1x1 = 1x1
        let a = vec![1.0]; // 1x1
        let b = vec![1.0]; // 1x1
        let mut c = vec![0.0; 1];

        cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::GELU);

        // result = 1*1 = 1
        let x = 1.0f32;
        let expected =
            x * 0.5 * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
        assert!((c[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_ck_activation_display() {
        assert_eq!(CkActivation::None.to_string(), "PassThrough");
        assert_eq!(CkActivation::SiLU.to_string(), "SiLU");
        assert_eq!(CkActivation::GELU.to_string(), "GELU");
    }

    #[test]
    fn test_is_ck_available() {
        // Should be false by default (without composable-kernel feature)
        let available = is_ck_available();
        #[cfg(feature = "composable-kernel")]
        assert!(available);
        #[cfg(not(feature = "composable-kernel"))]
        assert!(!available);
    }

    #[test]
    fn test_ck_gemm_fallback() {
        // Without feature, should return NotAvailable
        #[cfg(not(feature = "composable-kernel"))]
        {
            let a = vec![1.0];
            let b = vec![2.0];
            let mut c = vec![0.0];
            let config = CkGemmConfig {
                m: 1,
                n: 1,
                k: 1,
                activation: CkActivation::None,
                input_type: CkDataType::FP32,
            };
            let result = ck_gemm_f32(&a, &b, &mut c, &config);
            assert!(result.is_err());
            assert!(matches!(result, Err(CkError::NotAvailable)));
        }
    }

    #[test]
    fn test_ck_error_display() {
        let err1 = CkError::UnsupportedShape {
            m: 10,
            n: 20,
            k: 30,
        };
        assert_eq!(err1.to_string(), "CK: unsupported shape 10x20x30");

        let err2 = CkError::KernelLaunchFailed(42);
        assert_eq!(err2.to_string(), "CK kernel failed: 42");

        let err3 = CkError::NotAvailable;
        assert_eq!(err3.to_string(), "CK backend not compiled");
    }
}
