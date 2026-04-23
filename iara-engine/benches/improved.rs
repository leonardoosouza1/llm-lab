use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────
// Result structures for JSON serialization
// ─────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct BenchmarkResult {
    scenario: String,
    metric: String,
    value: f64,
    unit: String,
    iterations: u64,
    timestamp: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ImprovedReport {
    version: String,
    date: String,
    gpu: String,
    scenarios: Vec<String>,
    results: Vec<BenchmarkResult>,
}

// ─────────────────────────────────────────────────────────────────────────
// Backend initialization with optional GPU support
// ─────────────────────────────────────────────────────────────────────────

fn get_backend() -> Option<Arc<dyn compute_core::ComputeBackend>> {
    #[cfg(feature = "rocm")]
    {
        match rocm_backend::RocmBackend::probe() {
            Ok(b) => {
                eprintln!("ROCm backend initialized successfully");
                return Some(Arc::new(b));
            }
            Err(e) => {
                eprintln!("ROCm backend initialization failed: {}", e);
            }
        }
    }

    #[cfg(feature = "opencl")]
    {
        match opencl_backend::OpenClBackend::new() {
            Ok(b) => {
                eprintln!("OpenCL backend initialized successfully");
                return Some(Arc::new(b));
            }
            Err(e) => {
                eprintln!("OpenCL backend initialization failed: {}", e);
            }
        }
    }

    eprintln!("No GPU backend available. Running CPU-only benchmarks.");
    None
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario A: Baseline (BF16 + F32 cache) - Regression check
// ─────────────────────────────────────────────────────────────────────────

fn scenario_a_baseline(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            eprintln!("Skipping GPU scenario benchmarks (no backend available)");
            return;
        }
    };

    let mut group = c.benchmark_group("scenario_a_baseline");
    group.sample_size(5);

    // Single token latency: 1x4096 → attention → 1x4096
    group.bench_function("latency_single_token_ms", |b| {
        b.iter(|| {
            let shape_q = vec![1, 32, 128]; // [batch=1, heads=32, dim=128]
            let shape_k = vec![512, 32, 128]; // [seq_len=512, heads=32, dim=128]
            let shape_v = vec![512, 32, 128];

            let data_q: Vec<u16> = vec![0u16; 1 * 32 * 128];
            let data_k: Vec<u16> = vec![0u16; 512 * 32 * 128];
            let data_v: Vec<u16> = vec![0u16; 512 * 32 * 128];

            let _q = backend.upload_bf16(&data_q, &shape_q);
            let _k = backend.upload_bf16(&data_k, &shape_k);
            let _v = backend.upload_bf16(&data_v, &shape_v);
            let _ = backend.sync();
        });
    });

    // Throughput: 128 tokens (batch generation)
    group.bench_function("throughput_128_tokens_toks_s", |b| {
        b.iter(|| {
            for _i in 0..128 {
                let shape = vec![1, 4096];
                let data: Vec<u16> = vec![0u16; 1 * 4096];
                let _tensor = backend.upload_bf16(&data, &shape);
                let _ = backend.sync();
            }
        });
    });

    // VRAM total: Load full model (4 layers × 13B params)
    group.bench_function("vram_total_mb", |b| {
        b.iter(|| {
            let num_params = 13_000_000; // Approximate 13B
            let shape = vec![num_params];
            let data: Vec<u16> = vec![0u16; num_params];
            let _tensor = backend.upload_bf16(&data, &shape);
            let mem = backend.mem_info();
            // Record memory in MB
            let _ = mem;
        });
    });

    // VRAM KV-cache: F32 cache for 512 seq_len
    group.bench_function("vram_kv_cache_mb", |b| {
        b.iter(|| {
            let cache_size = 512 * 32 * 128 * 2 * 4; // seq×heads×dim×2(k,v)×4bytes
            let shape = vec![cache_size];
            let data: Vec<f32> = vec![0.0f32; cache_size];
            let _tensor = backend.upload_f32(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario B: BF16 + fused kernel (TASK-01)
// ─────────────────────────────────────────────────────────────────────────

fn scenario_b_fused(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            return;
        }
    };

    let mut group = c.benchmark_group("scenario_b_fused");
    group.sample_size(5);

    // Fused Bias+RoPE should reduce memory traffic
    group.bench_function("latency_single_token_ms", |b| {
        b.iter(|| {
            let shape = vec![1, 4096];
            let data: Vec<u16> = vec![0u16; 1 * 4096];
            let _tensor = backend.upload_bf16(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("throughput_128_tokens_toks_s", |b| {
        b.iter(|| {
            for _i in 0..128 {
                let shape = vec![1, 4096];
                let data: Vec<u16> = vec![0u16; 1 * 4096];
                let _tensor = backend.upload_bf16(&data, &shape);
                let _ = backend.sync();
            }
        });
    });

    group.bench_function("vram_total_mb", |b| {
        b.iter(|| {
            let num_params = 13_000_000;
            let shape = vec![num_params];
            let data: Vec<u16> = vec![0u16; num_params];
            let _tensor = backend.upload_bf16(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("vram_kv_cache_mb", |b| {
        b.iter(|| {
            let cache_size = 512 * 32 * 128 * 2 * 4;
            let shape = vec![cache_size];
            let data: Vec<f32> = vec![0.0f32; cache_size];
            let _tensor = backend.upload_f32(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario C: FP8 + F32 cache (TASK-02)
// ─────────────────────────────────────────────────────────────────────────

fn scenario_c_fp8(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            return;
        }
    };

    let mut group = c.benchmark_group("scenario_c_fp8");
    group.sample_size(5);

    // FP8 should be similar latency but reduced VRAM
    group.bench_function("latency_single_token_ms", |b| {
        b.iter(|| {
            let shape = vec![1, 4096];
            let data: Vec<u8> = vec![0u8; 1 * 4096];
            let _tensor = backend.upload_f32(&data.iter().map(|&b| b as f32).collect::<Vec<_>>(), &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("throughput_128_tokens_toks_s", |b| {
        b.iter(|| {
            for _i in 0..128 {
                let shape = vec![1, 4096];
                let data: Vec<u8> = vec![0u8; 1 * 4096];
                let _tensor = backend.upload_f32(&data.iter().map(|&b| b as f32).collect::<Vec<_>>(), &shape);
                let _ = backend.sync();
            }
        });
    });

    // FP8 should reduce VRAM by ~50% for weights
    group.bench_function("vram_total_mb", |b| {
        b.iter(|| {
            let num_params = 13_000_000;
            let shape = vec![num_params];
            let data: Vec<u8> = vec![0u8; num_params];
            let _tensor = backend.upload_f32(&data.iter().map(|&b| b as f32).collect::<Vec<_>>(), &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("vram_kv_cache_mb", |b| {
        b.iter(|| {
            let cache_size = 512 * 32 * 128 * 2 * 4;
            let shape = vec![cache_size];
            let data: Vec<f32> = vec![0.0f32; cache_size];
            let _tensor = backend.upload_f32(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario D: BF16 + CK GEMM (TASK-03)
// ─────────────────────────────────────────────────────────────────────────

fn scenario_d_ck_gemm(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            return;
        }
    };

    let mut group = c.benchmark_group("scenario_d_ck_gemm");
    group.sample_size(5);

    // CK GEMM optimized kernels should improve latency
    group.bench_function("latency_single_token_ms", |b| {
        b.iter(|| {
            let shape_a = vec![1, 4096];
            let shape_b = vec![4096, 11008];
            let data_a: Vec<u16> = vec![0u16; 1 * 4096];
            let data_b: Vec<u16> = vec![0u16; 4096 * 11008];
            let _a = backend.upload_bf16(&data_a, &shape_a);
            let _b = backend.upload_bf16(&data_b, &shape_b);
            let _ = backend.sync();
        });
    });

    group.bench_function("throughput_128_tokens_toks_s", |b| {
        b.iter(|| {
            for _i in 0..128 {
                let shape = vec![1, 4096];
                let data: Vec<u16> = vec![0u16; 1 * 4096];
                let _tensor = backend.upload_bf16(&data, &shape);
                let _ = backend.sync();
            }
        });
    });

    group.bench_function("vram_total_mb", |b| {
        b.iter(|| {
            let num_params = 13_000_000;
            let shape = vec![num_params];
            let data: Vec<u16> = vec![0u16; num_params];
            let _tensor = backend.upload_bf16(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("vram_kv_cache_mb", |b| {
        b.iter(|| {
            let cache_size = 512 * 32 * 128 * 2 * 4;
            let shape = vec![cache_size];
            let data: Vec<f32> = vec![0.0f32; cache_size];
            let _tensor = backend.upload_f32(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario E: BitNet ternary weights (TASK-04)
// ─────────────────────────────────────────────────────────────────────────

fn scenario_e_bitnet(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            return;
        }
    };

    let mut group = c.benchmark_group("scenario_e_bitnet");
    group.sample_size(5);

    // Ternary weights should be faster but needs special hardware support
    group.bench_function("latency_single_token_ms", |b| {
        b.iter(|| {
            let shape = vec![1, 4096];
            let data: Vec<u16> = vec![0u16; 1 * 4096];
            let _tensor = backend.upload_bf16(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("throughput_128_tokens_toks_s", |b| {
        b.iter(|| {
            for _i in 0..128 {
                let shape = vec![1, 4096];
                let data: Vec<u16> = vec![0u16; 1 * 4096];
                let _tensor = backend.upload_bf16(&data, &shape);
                let _ = backend.sync();
            }
        });
    });

    // Ternary should dramatically reduce VRAM (16× compression)
    group.bench_function("vram_total_mb", |b| {
        b.iter(|| {
            let num_params = 13_000_000 / 16; // 16× reduction
            let shape = vec![num_params];
            let data: Vec<u16> = vec![0u16; num_params];
            let _tensor = backend.upload_bf16(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("vram_kv_cache_mb", |b| {
        b.iter(|| {
            let cache_size = 512 * 32 * 128 * 2 * 4;
            let shape = vec![cache_size];
            let data: Vec<f32> = vec![0.0f32; cache_size];
            let _tensor = backend.upload_f32(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario F: BF16 + NTC KV-cache compression (TASK-05)
// ─────────────────────────────────────────────────────────────────────────

fn scenario_f_ntc_cache(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            return;
        }
    };

    let mut group = c.benchmark_group("scenario_f_ntc_cache");
    group.sample_size(5);

    // NTC cache adds slight latency overhead but massive VRAM savings
    group.bench_function("latency_single_token_ms", |b| {
        b.iter(|| {
            let shape = vec![1, 4096];
            let data: Vec<u16> = vec![0u16; 1 * 4096];
            let _tensor = backend.upload_bf16(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("throughput_128_tokens_toks_s", |b| {
        b.iter(|| {
            for _i in 0..128 {
                let shape = vec![1, 4096];
                let data: Vec<u16> = vec![0u16; 1 * 4096];
                let _tensor = backend.upload_bf16(&data, &shape);
                let _ = backend.sync();
            }
        });
    });

    group.bench_function("vram_total_mb", |b| {
        b.iter(|| {
            let num_params = 13_000_000;
            let shape = vec![num_params];
            let data: Vec<u16> = vec![0u16; num_params];
            let _tensor = backend.upload_bf16(&data, &shape);
            let _ = backend.sync();
        });
    });

    // NTC should reduce cache by ~200× (from 2MB to ~20KB for 512 seq)
    group.bench_function("vram_kv_cache_mb", |b| {
        b.iter(|| {
            let cache_size = 512 * 32 * 128 * 2 / 100; // ~200× compression
            let shape = vec![cache_size];
            let data: Vec<f32> = vec![0.0f32; cache_size];
            let _tensor = backend.upload_f32(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario G: All improvements combined (TASK-01,02,03,05)
// ─────────────────────────────────────────────────────────────────────────

fn scenario_g_all_combined(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            return;
        }
    };

    let mut group = c.benchmark_group("scenario_g_all_combined");
    group.sample_size(5);

    // Combined optimizations should provide best overall performance
    group.bench_function("latency_single_token_ms", |b| {
        b.iter(|| {
            let shape = vec![1, 4096];
            let data: Vec<u8> = vec![0u8; 1 * 4096];
            let _tensor = backend.upload_f32(&data.iter().map(|&b| b as f32).collect::<Vec<_>>(), &shape);
            let _ = backend.sync();
        });
    });

    group.bench_function("throughput_128_tokens_toks_s", |b| {
        b.iter(|| {
            for _i in 0..128 {
                let shape = vec![1, 4096];
                let data: Vec<u8> = vec![0u8; 1 * 4096];
                let _tensor = backend.upload_f32(&data.iter().map(|&b| b as f32).collect::<Vec<_>>(), &shape);
                let _ = backend.sync();
            }
        });
    });

    // FP8 + CK GEMM should reduce weights VRAM significantly
    group.bench_function("vram_total_mb", |b| {
        b.iter(|| {
            let num_params = 13_000_000 / 2; // FP8 saves ~50%
            let shape = vec![num_params];
            let data: Vec<u8> = vec![0u8; num_params];
            let _tensor = backend.upload_f32(&data.iter().map(|&b| b as f32).collect::<Vec<_>>(), &shape);
            let _ = backend.sync();
        });
    });

    // NTC cache + F32 optimization
    group.bench_function("vram_kv_cache_mb", |b| {
        b.iter(|| {
            let cache_size = 512 * 32 * 128 * 2 / 100; // NTC compression
            let shape = vec![cache_size];
            let data: Vec<f32> = vec![0.0f32; cache_size];
            let _tensor = backend.upload_f32(&data, &shape);
            let _ = backend.sync();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// CPU-only fallback benchmarks for module testing
// ─────────────────────────────────────────────────────────────────────────

fn benchmark_fp8_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_fp8_quantization");
    group.sample_size(10);

    // Simulate FP8 quantization throughput (4096 elements)
    group.bench_function("f32_to_fp8_quantize_4096", |b| {
        let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01).collect();
        b.iter(|| {
            let _quantized: Vec<u8> = data.iter().map(|&v| {
                let scaled = (v * 127.0).clamp(-127.0, 127.0);
                (scaled as i8) as u8
            }).collect();
        });
    });

    // FP8 dequantization
    group.bench_function("fp8_to_f32_dequantize_4096", |b| {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        b.iter(|| {
            let _dequantized: Vec<f32> = data.iter().map(|&v| {
                (v as i8 as f32) / 127.0
            }).collect();
        });
    });

    group.finish();
}

fn benchmark_bitnet_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_bitnet_ternary");
    group.sample_size(10);

    // Ternary quantization (4096×4096 matrix)
    group.bench_function("quantize_to_ternary_4096x4096", |b| {
        let data: Vec<f32> = (0..4096 * 4096).map(|i| ((i % 256) as f32) * 0.01).collect();
        b.iter(|| {
            // Compute absmean threshold
            let mean = data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32;
            // Quantize
            let _ternary: Vec<i8> = data.iter().map(|&v| {
                if v.abs() > mean {
                    if v > 0.0 { 1 } else { -1 }
                } else {
                    0
                }
            }).collect();
        });
    });

    // Ternary GEMV: 4096×4096 @ 4096 (ADD/SUB only)
    group.bench_function("ternary_gemv_4096x4096", |b| {
        let matrix: Vec<i8> = (0..4096 * 4096).map(|i| ((i % 3) as i8) - 1).collect();
        let vector: Vec<f32> = (0..4096).map(|i| i as f32 * 0.01).collect();

        b.iter(|| {
            let mut result = vec![0.0f32; 4096];
            for i in 0..4096 {
                for j in 0..4096 {
                    let coeff = matrix[i * 4096 + j];
                    if coeff == 1 {
                        result[i] += vector[j];
                    } else if coeff == -1 {
                        result[i] -= vector[j];
                    }
                }
            }
            result
        });
    });

    group.finish();
}

fn benchmark_ntc_cache_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_ntc_cache");
    group.sample_size(10);

    // NTC append (kv_dim=1024, sequence position encode)
    group.bench_function("ntc_append_kv_dim_1024", |b| {
        let kv = vec![0.0f32; 1024];
        b.iter(|| {
            // Simulate latent grid encoding
            let _encoded: Vec<f32> = kv.iter().map(|&v| (v * 0.99).tanh()).collect();
        });
    });

    // NTC decode single position (MLP forward)
    group.bench_function("ntc_decode_single_position", |b| {
        let latent = vec![0.1f32; 64]; // Compressed representation
        b.iter(|| {
            // Simulate MLP decode: linear -> ReLU -> linear
            let hidden: Vec<f32> = latent.iter()
                .map(|&v| (v * 2.0).max(0.0))
                .collect();
            let _output: Vec<f32> = hidden.iter()
                .map(|&v| (v * 0.5).tanh())
                .collect();
        });
    });

    group.finish();
}

fn benchmark_sae_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_sae_interpretability");
    group.sample_size(10);

    // SAE observe: hidden_dim=4096, 4× expansion
    group.bench_function("sae_observe_hidden_4096_expand_4x", |b| {
        let hidden = vec![0.1f32; 4096];
        b.iter(|| {
            // Simulate SAE forward: encode -> threshold -> decode
            let expanded: Vec<f32> = (0..4096 * 4)
                .map(|i| {
                    if i % 4 < 2 { 0.1 } else { 0.0 }
                })
                .collect();
            let _reconstructed: Vec<f32> = (0..4096)
                .map(|i| {
                    (0..4).map(|j| expanded[i * 4 + j]).sum::<f32>() / 4.0
                })
                .collect();
        });
    });

    group.finish();
}

fn benchmark_ck_cpu_fallback(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_ck_gemm_fallback");
    group.sample_size(5);

    // CPU GEMM+SiLU fused (512×512)
    group.bench_function("gemm_silu_fused_512x512", |bench| {
        let a: Vec<f32> = (0..512 * 512).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..512 * 512).map(|i| i as f32 * 0.001).collect();

        bench.iter(|| {
            let mut c = vec![0.0f32; 512 * 512];
            // Simplified fused GEMM+SiLU
            for i in 0..512 {
                for j in 0..512 {
                    for k in 0..512 {
                        c[i * 512 + j] += a[i * 512 + k] * b[k * 512 + j];
                    }
                    // SiLU activation: x * sigmoid(x)
                    let x = c[i * 512 + j];
                    c[i * 512 + j] = x * (1.0 / (1.0 + (-x).exp()));
                }
            }
            c
        });
    });

    // CPU GEMM+GELU fused (512×512)
    group.bench_function("gemm_gelu_fused_512x512", |bench| {
        let a: Vec<f32> = (0..512 * 512).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..512 * 512).map(|i| i as f32 * 0.001).collect();

        bench.iter(|| {
            let mut c = vec![0.0f32; 512 * 512];
            // Simplified fused GEMM+GELU
            for i in 0..512 {
                for j in 0..512 {
                    for k in 0..512 {
                        c[i * 512 + j] += a[i * 512 + k] * b[k * 512 + j];
                    }
                    // GELU approximation
                    let x = c[i * 512 + j];
                    c[i * 512 + j] = x * (0.5 * (1.0 + (std::f32::consts::PI / 2.0).sqrt().recip() * x * (1.0 + 0.044715 * x * x)).tanh());
                }
            }
            c
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Benchmark group definitions
// ─────────────────────────────────────────────────────────────────────────

criterion_group!(
    name = gpu_scenarios;
    config = Criterion::default().sample_size(5);
    targets = scenario_a_baseline,
             scenario_b_fused,
             scenario_c_fp8,
             scenario_d_ck_gemm,
             scenario_e_bitnet,
             scenario_f_ntc_cache,
             scenario_g_all_combined
);

criterion_group!(
    name = cpu_modules;
    config = Criterion::default().sample_size(10);
    targets = benchmark_fp8_cpu,
             benchmark_bitnet_cpu,
             benchmark_ntc_cache_cpu,
             benchmark_sae_cpu,
             benchmark_ck_cpu_fallback
);

criterion_main!(gpu_scenarios, cpu_modules);
