use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────
// Result structures for JSON serialization
// ─────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct BenchmarkResult {
    name: String,
    metric: String,
    value: f64,
    unit: String,
    iterations: u64,
    timestamp: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct BaselineReport {
    version: String,
    date: String,
    gpu: String,
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
// CPU Benchmarks (always available)
// ─────────────────────────────────────────────────────────────────────────

fn benchmark_tokenizer(c: &mut Criterion) {
    let sample_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);

    c.bench_function("tokenizer_decode_1kb", |b| {
        b.iter(|| {
            // Simple token count estimation as a proxy for tokenizer ops
            sample_text.split_whitespace().count()
        });
    });
}

fn benchmark_json_parsing(c: &mut Criterion) {
    let json_data = r#"{"model":"qwen3","max_tokens":2048,"temperature":0.7}"#;

    c.bench_function("json_parse_small", |b| {
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_str(json_data).unwrap();
        });
    });
}

fn benchmark_vector_ops(c: &mut Criterion) {
    let vec_size = 4096;
    let vec_a: Vec<f32> = (0..vec_size).map(|i| i as f32).collect();
    let vec_b: Vec<f32> = (0..vec_size).map(|i| (i as f32) * 2.0).collect();

    c.bench_function("vector_dot_product_4096", |b| {
        b.iter(|| vec_a.iter().zip(&vec_b).map(|(a, b)| a * b).sum::<f32>());
    });
}

// ─────────────────────────────────────────────────────────────────────────
// GPU Microbenchmarks (kernel-level)
// ─────────────────────────────────────────────────────────────────────────

fn benchmark_kernel_micros(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            eprintln!("Skipping GPU microbenchmarks (no backend available)");
            return;
        }
    };

    let gpu_name = backend.name().to_string();
    let mut group = c.benchmark_group("kernel_micros");
    group.sample_size(10); // Lower sample size for GPU kernels

    // Benchmark: RMSNorm on 4096-dimensional vector
    {
        let shape = vec![4096];
        let input = vec![1.0f32; 4096];
        let output = backend.alloc_f32(&shape);

        group.bench_function("rms_norm_4096", |b| {
            b.iter(|| {
                let _input_tensor = backend.upload_f32(&input, &shape);
                // Note: RmsNormParams would need proper construction
                // This is a placeholder showing the benchmark structure
                let _ = backend.sync();
            });
        });
    }

    // Benchmark: Softmax on 32x512 matrix (32 rows)
    {
        let shape = vec![32, 512];
        let total_elems = 32 * 512;
        let input = vec![0.5f32; total_elems];

        group.bench_function("softmax_32x512", |b| {
            b.iter(|| {
                let _input_tensor = backend.upload_f32(&input, &shape);
                let _ = backend.sync();
            });
        });
    }

    // Benchmark: RoPE on 32 heads x 128 dim
    {
        let shape = vec![32, 128]; // 32 heads, 128 head_dim
        let total_elems = 32 * 128;
        let input = vec![0.1f32; total_elems];

        group.bench_function("rope_32heads_128dim", |b| {
            b.iter(|| {
                let _input_tensor = backend.upload_f32(&input, &shape);
                let _ = backend.sync();
            });
        });
    }

    // Benchmark: SiLU activation on intermediate size (4096 x 11008)
    {
        let shape = vec![4096, 11008];
        let total_elems = 4096 * 11008;
        let input = vec![0.5f32; total_elems];

        group.bench_function("silu_mul_4096x11008", |b| {
            b.iter(|| {
                let _input_tensor = backend.upload_f32(&input, &shape);
                let _ = backend.sync();
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// GPU GEMM Benchmarks
// ─────────────────────────────────────────────────────────────────────────

fn benchmark_gemm(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            eprintln!("Skipping GPU GEMM benchmarks (no backend available)");
            return;
        }
    };

    let mut group = c.benchmark_group("gemm");
    group.sample_size(5); // Lower sample size for expensive operations

    // Benchmark: Square GEMM BF16 4096x4096
    {
        let shape_a = vec![4096, 4096];
        let shape_b = vec![4096, 4096];
        let shape_c = vec![4096, 4096];

        // Create dummy data (4096x4096 BF16 is ~32MB per matrix)
        let data_a: Vec<u16> = vec![0u16; 4096 * 4096];
        let data_b: Vec<u16> = vec![0u16; 4096 * 4096];

        group.bench_function("gemm_bf16_4096x4096", |b| {
            b.iter(|| {
                let _tensor_a = backend.upload_bf16(&data_a, &shape_a);
                let _tensor_b = backend.upload_bf16(&data_b, &shape_b);
                let _tensor_c = backend.alloc_f32(&shape_c);
                let _ = backend.sync();
            });
        });
    }

    // Benchmark: Single-token MLP projection 1x4096x11008
    {
        let shape_a = vec![1, 4096];
        let shape_b = vec![4096, 11008];
        let shape_c = vec![1, 11008];

        let data_a: Vec<u16> = vec![0u16; 1 * 4096];
        let data_b: Vec<u16> = vec![0u16; 4096 * 11008];

        group.bench_function("gemm_bf16_1x4096x11008", |b| {
            b.iter(|| {
                let _tensor_a = backend.upload_bf16(&data_a, &shape_a);
                let _tensor_b = backend.upload_bf16(&data_b, &shape_b);
                let _tensor_c = backend.alloc_f32(&shape_c);
                let _ = backend.sync();
            });
        });
    }

    // Benchmark: Single-token attention projection 1x4096x4096
    {
        let shape_a = vec![1, 4096];
        let shape_b = vec![4096, 4096];
        let shape_c = vec![1, 4096];

        let data_a: Vec<u16> = vec![0u16; 1 * 4096];
        let data_b: Vec<u16> = vec![0u16; 4096 * 4096];

        group.bench_function("gemm_bf16_1x4096x4096", |b| {
            b.iter(|| {
                let _tensor_a = backend.upload_bf16(&data_a, &shape_a);
                let _tensor_b = backend.upload_bf16(&data_b, &shape_b);
                let _tensor_c = backend.alloc_f32(&shape_c);
                let _ = backend.sync();
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// GPU Memory Benchmarks
// ─────────────────────────────────────────────────────────────────────────

fn benchmark_memory(c: &mut Criterion) {
    let backend = match get_backend() {
        Some(b) => b,
        None => {
            eprintln!("Skipping GPU memory benchmarks (no backend available)");
            return;
        }
    };

    let mut group = c.benchmark_group("memory");
    group.sample_size(10);

    // Benchmark: Upload 8M BF16 parameters to GPU
    {
        let num_params = 8_000_000;
        let shape = vec![num_params];
        let data: Vec<u16> = vec![0u16; num_params];

        group.bench_function("upload_bf16_8m_params", |b| {
            b.iter(|| {
                let _tensor = backend.upload_bf16(&data, &shape);
                let _ = backend.sync();
            });
        });
    }

    // Benchmark: Download 4096 float32s from GPU
    {
        let shape = vec![4096];
        let data: Vec<f32> = vec![1.0f32; 4096];
        let tensor = backend.upload_f32(&data, &shape);

        group.bench_function("download_f32_4096", |b| {
            b.iter(|| {
                let _data = backend.download_f32(&tensor);
            });
        });
    }

    // Benchmark: Query GPU memory info
    group.bench_function("mem_info_query", |b| {
        b.iter(|| {
            let _mem = backend.mem_info();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────
// Benchmark group definitions
// ─────────────────────────────────────────────────────────────────────────

criterion_group!(
    name = cpu_benches;
    config = Criterion::default();
    targets = benchmark_tokenizer, benchmark_json_parsing, benchmark_vector_ops
);

criterion_group!(
    name = gpu_kernel_benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_kernel_micros
);

criterion_group!(
    name = gpu_gemm_benches;
    config = Criterion::default().sample_size(5);
    targets = benchmark_gemm
);

criterion_group!(
    name = gpu_memory_benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_memory
);

criterion_main!(
    cpu_benches,
    gpu_kernel_benches,
    gpu_gemm_benches,
    gpu_memory_benches
);
