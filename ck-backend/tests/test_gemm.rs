use ck_backend::gemm::*;

#[test]
fn test_passthrough_gemm_small() {
    // 2x3 @ 3x2 = 2x2 matrix multiply
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
    let mut c = vec![0.0; 4];

    cpu_gemm_with_activation(&a, &b, &mut c, 2, 2, 3, CkActivation::None);

    // Expected results from manual calculation
    // c[0,0] = 1*1 + 2*3 + 3*5 = 22
    // c[0,1] = 1*2 + 2*4 + 3*6 = 28
    // c[1,0] = 4*1 + 5*3 + 6*5 = 49
    // c[1,1] = 4*2 + 5*4 + 6*6 = 64
    assert!((c[0] - 22.0).abs() < 1e-5);
    assert!((c[1] - 28.0).abs() < 1e-5);
    assert!((c[2] - 49.0).abs() < 1e-5);
    assert!((c[3] - 64.0).abs() < 1e-5);
}

#[test]
fn test_passthrough_gemm_identity() {
    // Identity-like: 3x3 @ 3x3 = 3x3
    let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // identity
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // random
    let mut c = vec![0.0; 9];

    cpu_gemm_with_activation(&a, &b, &mut c, 3, 3, 3, CkActivation::None);

    // Identity @ B = B
    for i in 0..9 {
        assert!((c[i] - b[i]).abs() < 1e-5);
    }
}

#[test]
fn test_silu_activation() {
    // Simple 1x1 @ 1x1 to test SiLU
    let a = vec![2.0];
    let b = vec![3.0];
    let mut c = vec![0.0];

    cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::SiLU);

    // Raw result = 2*3 = 6
    // SiLU(6) = 6 * sigmoid(6)
    let sigmoid_6 = 1.0 / (1.0 + (-6.0_f32).exp());
    let expected = 6.0 * sigmoid_6;

    assert!((c[0] - expected).abs() < 1e-5);
}

#[test]
fn test_silu_zero() {
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let a = vec![1.0];
    let b = vec![0.0];
    let mut c = vec![0.0];

    cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::SiLU);

    assert!(c[0].abs() < 1e-5);
}

#[test]
fn test_silu_negative() {
    // SiLU(-2) = -2 * sigmoid(-2)
    let a = vec![1.0];
    let b = vec![-2.0];
    let mut c = vec![0.0];

    cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::SiLU);

    let sigmoid_neg2 = 1.0 / (1.0 + (2.0_f32).exp());
    let expected = -2.0 * sigmoid_neg2;

    assert!((c[0] - expected).abs() < 1e-5);
}

#[test]
fn test_gelu_activation() {
    // Simple 1x1 @ 1x1 to test GELU
    let a = vec![1.0];
    let b = vec![1.0];
    let mut c = vec![0.0];

    cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::GELU);

    // Raw result = 1*1 = 1
    // GELU approximation
    let x = 1.0_f32;
    let cdf = 0.5
        * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
    let expected = x * cdf;

    assert!((c[0] - expected).abs() < 1e-5);
}

#[test]
fn test_gelu_zero() {
    // GELU(0) = 0
    let a = vec![1.0];
    let b = vec![0.0];
    let mut c = vec![0.0];

    cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::GELU);

    assert!(c[0].abs() < 1e-5);
}

#[test]
fn test_gelu_negative() {
    // GELU should handle negative values
    let a = vec![1.0];
    let b = vec![-1.0];
    let mut c = vec![0.0];

    cpu_gemm_with_activation(&a, &b, &mut c, 1, 1, 1, CkActivation::GELU);

    let x = -1.0_f32;
    let cdf = 0.5
        * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
    let expected = x * cdf;

    assert!((c[0] - expected).abs() < 1e-5);
}

#[test]
fn test_gemm_batch_silu() {
    // 3x2 @ 2x2 = 3x2 with SiLU
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
    let b = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let mut c = vec![0.0; 6];

    cpu_gemm_with_activation(&a, &b, &mut c, 3, 2, 2, CkActivation::SiLU);

    // Verify shapes and that SiLU was applied
    assert_eq!(c.len(), 6);
    // First element: (1*1 + 2*3) = 7, SiLU(7) = 7 * sigmoid(7)
    let sigmoid_7 = 1.0 / (1.0 + (-7.0_f32).exp());
    let expected_c0 = 7.0 * sigmoid_7;
    assert!((c[0] - expected_c0).abs() < 1e-5);
}

#[test]
fn test_activation_enum_equality() {
    assert_eq!(CkActivation::None, CkActivation::None);
    assert_eq!(CkActivation::SiLU, CkActivation::SiLU);
    assert_eq!(CkActivation::GELU, CkActivation::GELU);
    assert_ne!(CkActivation::None, CkActivation::SiLU);
    assert_ne!(CkActivation::SiLU, CkActivation::GELU);
}

#[test]
fn test_fused_mlp_forward_basic() {
    // Simple MLP: batch=1, in_dim=2, intermediate=2, out_dim=1
    let x = vec![1.0, 2.0]; // [1, 2]
    let w_gate = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2] identity-like
    let w_up = vec![1.0, 1.0, 1.0, 1.0]; // [2, 2] all ones
    let w_down = vec![1.0, 1.0]; // [2, 1] all ones

    let result = fused_mlp_forward(&x, &w_gate, &w_up, &w_down, 1, 2, 2, 1);

    // Check output shape
    assert_eq!(result.len(), 1);

    // 1. gate = SiLU(x @ w_gate)
    // x @ w_gate: [1, 2] @ [[1, 0], [0, 1]] = [1*1+2*0, 1*0+2*1] = [1, 2]
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let gate_0 = 1.0 * sigmoid(1.0);
    let gate_1 = 2.0 * sigmoid(2.0);

    // 2. up = x @ w_up
    // x @ w_up: [1, 2] @ [[1, 1], [1, 1]] = [1*1+2*1, 1*1+2*1] = [3, 3]
    let up_0 = 3.0;
    let up_1 = 3.0;

    // 3. out = gate * up — elementwise
    let out_0 = gate_0 * up_0;
    let out_1 = gate_1 * up_1;

    // 4. result = out @ w_down
    // out @ w_down: [out_0, out_1] @ [[1], [1]] = [out_0*1 + out_1*1] = [out_0 + out_1]
    let expected = out_0 + out_1;

    assert!((result[0] - expected).abs() < 1e-3);
}

#[test]
fn test_fused_mlp_forward_shape() {
    let batch = 2;
    let in_dim = 4;
    let intermediate = 8;
    let out_dim = 4;

    let x = vec![1.0; batch * in_dim];
    let w_gate = vec![0.5; in_dim * intermediate];
    let w_up = vec![0.5; in_dim * intermediate];
    let w_down = vec![0.5; intermediate * out_dim];

    let result = fused_mlp_forward(&x, &w_gate, &w_up, &w_down, batch, in_dim, intermediate, out_dim);

    // Verify output shape
    assert_eq!(result.len(), batch * out_dim);
}

#[test]
fn test_fused_mlp_forward_zeros() {
    // All-zero inputs should produce all-zero outputs
    let batch = 2;
    let in_dim = 3;
    let intermediate = 3;
    let out_dim = 2;

    let x = vec![0.0; batch * in_dim];
    let w_gate = vec![1.0; in_dim * intermediate];
    let w_up = vec![1.0; in_dim * intermediate];
    let w_down = vec![1.0; intermediate * out_dim];

    let result = fused_mlp_forward(&x, &w_gate, &w_up, &w_down, batch, in_dim, intermediate, out_dim);

    for val in result {
        assert!(val.abs() < 1e-6);
    }
}

#[test]
fn test_ck_datatype_equality() {
    assert_eq!(CkDataType::BF16, CkDataType::BF16);
    assert_eq!(CkDataType::FP16, CkDataType::FP16);
    assert_eq!(CkDataType::FP32, CkDataType::FP32);
    assert_ne!(CkDataType::BF16, CkDataType::FP16);
    assert_ne!(CkDataType::FP16, CkDataType::FP32);
}

#[test]
fn test_ck_gemm_config_clone() {
    let config1 = CkGemmConfig {
        m: 10,
        n: 20,
        k: 30,
        activation: CkActivation::SiLU,
        input_type: CkDataType::FP32,
    };
    let config2 = config1.clone();

    assert_eq!(config1.m, config2.m);
    assert_eq!(config1.n, config2.n);
    assert_eq!(config1.k, config2.k);
    assert_eq!(config1.activation, config2.activation);
    assert_eq!(config1.input_type, config2.input_type);
}

#[test]
fn test_ck_error_types() {
    let err1 = CkError::UnsupportedShape {
        m: 1,
        n: 2,
        k: 3,
    };
    assert!(matches!(
        err1,
        CkError::UnsupportedShape {
            m: 1,
            n: 2,
            k: 3
        }
    ));

    let err2 = CkError::KernelLaunchFailed(-1);
    assert!(matches!(err2, CkError::KernelLaunchFailed(-1)));

    let err3 = CkError::NotAvailable;
    assert!(matches!(err3, CkError::NotAvailable));
}

#[test]
fn test_large_matrix_multiply() {
    // Test a larger matrix: 100x50 @ 50x30 = 100x30
    let m = 100;
    let n = 30;
    let k = 50;

    let a = vec![0.5; m * k];
    let b = vec![2.0; k * n];
    let mut c = vec![0.0; m * n];

    cpu_gemm_with_activation(&a, &b, &mut c, m, n, k, CkActivation::None);

    // All elements should be 0.5 * 2.0 * 50 = 50
    for val in &c {
        assert!((val - 50.0).abs() < 1e-4);
    }
}

#[test]
fn test_activation_type_debug() {
    let act1 = CkActivation::None;
    let act2 = CkActivation::SiLU;
    let act3 = CkActivation::GELU;

    // Just verify Debug trait works
    let s1 = format!("{:?}", act1);
    let s2 = format!("{:?}", act2);
    let s3 = format!("{:?}", act3);

    assert!(s1.contains("None"));
    assert!(s2.contains("SiLU"));
    assert!(s3.contains("GELU"));
}
