//! WGSL Sparse Ternary MatMul — GPU compute kernel via wgpu.
//!
//! Implementa y = x @ W_ternary * scale em compute shader WGSL:
//!   - Pesos ternários {-1,0,+1} empacotados em 2 bits (16 elems por u32)
//!   - Cada workgroup (256 threads) computa um neurônio de saída
//!   - Redução paralela na workgroup shared memory
//!   - Zeros não contribuem (branch no shader, sem multiply)
//!
//! Comparação: CPU (ternary_matmul_sparse) vs GPU (WGSL kernel)
//! Matriz: [8960 × 1536] — MLP up-projection do Qwen2.5-1.5B
//!
//! Referências:
//!   - SLIDE (Rice 2020): sub-linear matmul via LSH + sparse activations
//!   - BitNet (MSRA 2024): ternary weights para inferência eficiente

use bytemuck;
use iara_train::navigator::ternary::{quantize_ternary, ternary_matmul_sparse};
use std::time::Instant;
use wgpu::util::DeviceExt;

// ─── WGSL Compute Shader ────────────────────────────────────────────────────

const SHADER_SRC: &str = r#"
// Ternary Sparse MatMul — WGSL Compute Shader
// Formato packed: 2 bits por elemento, 16 elementos por u32
//   0b00 = 0   (zero — sem contribuição, branch taken)
//   0b01 = +1  (soma x_j)
//   0b10 = -1  (subtrai x_j)

struct Uniforms {
    out_dim : u32,
    in_dim  : u32,
    scale   : f32,
    _pad    : u32,
};

@group(0) @binding(0) var<storage, read>       packed  : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in    : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out   : array<f32>;
@group(0) @binding(3) var<uniform>             u       : Uniforms;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id)        wg_id    : vec3<u32>,
) {
    let out_idx = wg_id.x;
    if out_idx >= u.out_dim { return; }

    let in_dim = u.in_dim;
    let tid    = local_id.x;

    var acc: f32 = 0.0;

    // Cada thread cobre elementos em stride de 256 (coalesced memory)
    var j: u32 = tid;
    while j < in_dim {
        // Desempacotar elemento ternário: neuron=out_idx, dim=j
        let elem_idx = out_idx * in_dim + j;
        let u32_idx  = elem_idx / 16u;         // 16 elementos por u32
        let bit_off  = (elem_idx % 16u) * 2u;  // offset em bits (2 por elem)
        let code     = (packed[u32_idx] >> bit_off) & 3u;

        let x_val = x_in[j];
        if code == 1u {
            acc += x_val;        // peso +1
        } else if code == 2u {
            acc -= x_val;        // peso -1
        }
        // code == 0: contribuição zero → skip

        j += 256u;
    }

    partial[tid] = acc;
    workgroupBarrier();

    // Redução paralela (log2 passagens)
    if tid < 128u { partial[tid] += partial[tid + 128u]; } workgroupBarrier();
    if tid <  64u { partial[tid] += partial[tid +  64u]; } workgroupBarrier();
    if tid <  32u { partial[tid] += partial[tid +  32u]; } workgroupBarrier();
    if tid <  16u { partial[tid] += partial[tid +  16u]; } workgroupBarrier();
    if tid <   8u { partial[tid] += partial[tid +   8u]; } workgroupBarrier();
    if tid <   4u { partial[tid] += partial[tid +   4u]; } workgroupBarrier();
    if tid <   2u { partial[tid] += partial[tid +   2u]; } workgroupBarrier();
    if tid <   1u { partial[tid] += partial[tid +   1u]; } workgroupBarrier();

    if tid == 0u {
        y_out[out_idx] = partial[0u] * u.scale;
    }
}
"#;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Empacota Vec<f32> ternário {-1,0,+1} → Vec<u32> (16 elems / u32).
fn pack_ternary_u32(vals: &[f32]) -> Vec<u32> {
    let n_u32 = (vals.len() + 15) / 16;
    let mut out = vec![0u32; n_u32];
    for (i, &v) in vals.iter().enumerate() {
        let code: u32 = match v as i32 { 1 => 0b01, -1 => 0b10, _ => 0b00 };
        out[i / 16] |= code << ((i % 16) * 2);
    }
    out
}

// ─── GPU Engine ─────────────────────────────────────────────────────────────

struct GpuTernaryMatmul {
    device   : wgpu::Device,
    queue    : wgpu::Queue,
    pipeline : wgpu::ComputePipeline,
    bg_layout: wgpu::BindGroupLayout,
}

impl GpuTernaryMatmul {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found");

        println!("Adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label:             Some("iara-ternary"),
                required_features: wgpu::Features::empty(),
                required_limits:   wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to request device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("ternary_sparse"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("ternary_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("ternary_pl"),
            bind_group_layouts: &[Some(&bg_layout)],
            immediate_size:     0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("ternary_pipeline"),
            layout:      Some(&pipeline_layout),
            module:      &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { device, queue, pipeline, bg_layout }
    }

    /// Executa y = x @ W_ternary * scale na GPU.
    async fn compute(
        &self,
        packed_weights: &[u32],
        x: &[f32],
        scale: f32,
        out_dim: usize,
        in_dim: usize,
    ) -> (Vec<f32>, u128) {
        let dev = &self.device;
        let q   = &self.queue;

        let buf_packed = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("packed"),
            contents: bytemuck::cast_slice(packed_weights),
            usage:    wgpu::BufferUsages::STORAGE,
        });
        let buf_x = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("x_in"),
            contents: bytemuck::cast_slice(x),
            usage:    wgpu::BufferUsages::STORAGE,
        });
        let buf_y = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("y_out"),
            size:               (out_dim * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Uniforms: [out_dim, in_dim, scale_as_bits, pad]
        let uniforms: [u32; 4] = [out_dim as u32, in_dim as u32, scale.to_bits(), 0];
        let buf_uni = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("ternary_bg"),
            layout:  &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_packed.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_uni.as_entire_binding() },
            ],
        });

        let buf_staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("staging"),
            size:               (out_dim * 4) as u64,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let t0 = Instant::now();

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ternary_enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ternary_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(out_dim as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&buf_y, 0, &buf_staging, 0, (out_dim * 4) as u64);
        q.submit(std::iter::once(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        buf_staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("GPU poll failed");
        rx.recv().unwrap().expect("map_async failed");
        let gpu_us = t0.elapsed().as_micros();

        let mapped = buf_staging.slice(..).get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        buf_staging.unmap();

        (result, gpu_us)
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    pollster::block_on(run());
}

async fn run() {
    println!("=== IARA Ternary Sparse MatMul — WGSL Kernel ===\n");

    let out_dim = 8960usize;  // MLP up-projection Qwen2.5-1.5B
    let in_dim  = 1536usize;
    let batch   = 1usize;

    println!("Matriz: [{out_dim} × {in_dim}] (MLP up-projection Qwen2.5-1.5B)");

    // ── 1. Gerar pesos sintéticos ─────────────────────────────────────────────
    println!("\n--- Fase 1: Gerando pesos e quantizando ---");
    let n_weights = out_dim * in_dim;
    let weights_f32: Vec<f32> = (0..n_weights)
        .map(|i| (i as f32 * 0.000137).sin() * 0.3 + (i as f32 * 0.000073).cos() * 0.2)
        .collect();

    let (ternary, scale) = quantize_ternary(&weights_f32);
    let sparsity = ternary.iter().filter(|&&v| v == 0.0).count() as f32 / n_weights as f32;
    println!("  Scale:    {:.6}", scale);
    println!("  Sparsity: {:.1}% zeros", sparsity * 100.0);
    println!("  Compressão: {} MB → {:.1} MB (2-bit)",
        n_weights * 4 / 1024 / 1024,
        n_weights as f32 * 2.0 / 8.0 / 1024.0 / 1024.0
    );

    // ── 2. CPU reference ──────────────────────────────────────────────────────
    println!("\n--- Fase 2: CPU reference ---");
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    let n_warmup = 3;
    let n_runs   = 10;
    let mut y_cpu = Vec::new();

    // Warmup
    for _ in 0..n_warmup {
        y_cpu = ternary_matmul_sparse(&x, &ternary, scale, batch, out_dim, in_dim);
    }
    let t_cpu = Instant::now();
    for _ in 0..n_runs {
        y_cpu = ternary_matmul_sparse(&x, &ternary, scale, batch, out_dim, in_dim);
    }
    let cpu_us = t_cpu.elapsed().as_micros() / n_runs as u128;
    println!("  CPU avg ({n_runs} runs): {cpu_us} µs");

    // ── 3. GPU setup ──────────────────────────────────────────────────────────
    println!("\n--- Fase 3: Inicializando GPU ---");
    let gpu = GpuTernaryMatmul::new().await;
    let packed_u32 = pack_ternary_u32(&ternary);
    println!("  Packed buffer: {} KB", packed_u32.len() * 4 / 1024);

    // ── 4. GPU benchmark ─────────────────────────────────────────────────────
    println!("\n--- Fase 4: GPU compute (WGSL) ---");
    for _ in 0..n_warmup {
        gpu.compute(&packed_u32, &x, scale, out_dim, in_dim).await;
    }
    let mut y_gpu   = Vec::new();
    let mut gpu_us_acc = 0u128;
    for _ in 0..n_runs {
        let (y, us) = gpu.compute(&packed_u32, &x, scale, out_dim, in_dim).await;
        y_gpu = y;
        gpu_us_acc += us;
    }
    let gpu_us = gpu_us_acc / n_runs as u128;
    println!("  GPU avg ({n_runs} runs): {gpu_us} µs (incl. readback)");

    // ── 5. Verificação de correctude ──────────────────────────────────────────
    println!("\n--- Fase 5: Verificação ---");
    let max_diff = y_cpu.iter().zip(y_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let dot: f32 = y_cpu.iter().zip(y_gpu.iter()).map(|(a, b)| a * b).sum();
    let na  = y_cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb  = y_gpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos_sim = dot / (na * nb + 1e-8);

    println!("  CPU[:5]: {:?}", &y_cpu[..5]);
    println!("  GPU[:5]: {:?}", &y_gpu[..5]);
    println!("  Max |diff|: {:.2e}", max_diff);
    println!("  Cosine sim: {:.6}", cos_sim);

    let correct = max_diff < 1e-3 && cos_sim > 0.9999;

    // ── 6. Resumo ─────────────────────────────────────────────────────────────
    let speedup = cpu_us as f32 / gpu_us as f32;
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║       IARA WGSL Ternary Kernel — Resumo             ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  Matriz:      [{out_dim} × {in_dim}]                    ║");
    println!("║  Sparsity:    {:.1}% zeros                          ║", sparsity * 100.0);
    println!("║  CPU time:    {:>6} µs                             ║", cpu_us);
    println!("║  GPU time:    {:>6} µs (incl. readback)            ║", gpu_us);
    println!("║  Speedup:     {:.2}×                               ║", speedup);
    println!("║  Cosine sim:  {:.6}                             ║", cos_sim);
    println!("║  Correto:     {}                                ║", if correct { "SIM ✓" } else { "NÃO ✗" });
    println!("╚══════════════════════════════════════════════════════╝");

    if !correct {
        eprintln!("\nERRO: resultado GPU diverge do CPU reference.");
        std::process::exit(1);
    }
}
