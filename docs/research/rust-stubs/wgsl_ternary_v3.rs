//! WGSL Ternary Kernel v3 — Buffers pré-alocados + medição de overhead.
//!
//! Isola os componentes de latência:
//!   A) Alocação de buffers GPU
//!   B) Dispatch puro (encoder + submit + fence)
//!   C) Readback D2H (map_async + copy)
//!   D) Total (A+B+C)
//!
//! Em produção: buffers são alocados 1x, output fica na GPU (sem readback),
//! dispatch é batch para todas as layers → latência real ≈ B only.

use bytemuck;
use iara_train::navigator::ternary::{quantize_ternary, ternary_matmul_sparse};
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Uniforms { out_dim: u32, in_dim: u32, scale: f32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out  : array<f32>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;

var<workgroup> partial: array<f32, 256>;

fn tsign(code: u32) -> f32 {
    let nz = f32(min(code, 1u));
    return nz * (1.0 - 2.0 * f32(code >> 1u));
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id)        wid : vec3<u32>,
) {
    let out_idx = wid.x;
    if out_idx >= u.out_dim { return; }
    let tid = lid.x;
    var acc: f32 = 0.0;

    var j: u32 = tid * 4u;
    while j + 3u < u.in_dim {
        let base = out_idx * u.in_dim + j;
        let u0 = base / 16u; let b0 = (base % 16u) * 2u;
        var s0: f32; var s1: f32; var s2: f32; var s3: f32;
        if b0 <= 24u {
            let w = packed[u0];
            s0 = tsign((w >> b0) & 3u);
            s1 = tsign((w >> (b0 + 2u)) & 3u);
            s2 = tsign((w >> (b0 + 4u)) & 3u);
            s3 = tsign((w >> (b0 + 6u)) & 3u);
        } else {
            let e0=base; let e1=base+1u; let e2=base+2u; let e3=base+3u;
            s0 = tsign((packed[e0/16u] >> ((e0%16u)*2u)) & 3u);
            s1 = tsign((packed[e1/16u] >> ((e1%16u)*2u)) & 3u);
            s2 = tsign((packed[e2/16u] >> ((e2%16u)*2u)) & 3u);
            s3 = tsign((packed[e3/16u] >> ((e3%16u)*2u)) & 3u);
        }
        acc += s0 * x_in[j] + s1 * x_in[j+1u] + s2 * x_in[j+2u] + s3 * x_in[j+3u];
        j += 1024u;
    }
    var r = j;
    while r < u.in_dim {
        let e = out_idx * u.in_dim + r;
        acc += tsign((packed[e/16u] >> ((e%16u)*2u)) & 3u) * x_in[r];
        r += 1024u;
    }

    partial[tid] = acc;
    workgroupBarrier();
    if tid < 128u { partial[tid] += partial[tid + 128u]; } workgroupBarrier();
    if tid <  64u { partial[tid] += partial[tid +  64u]; } workgroupBarrier();
    if tid <  32u { partial[tid] += partial[tid +  32u]; } workgroupBarrier();
    if tid <  16u { partial[tid] += partial[tid +  16u]; } workgroupBarrier();
    if tid <   8u { partial[tid] += partial[tid +   8u]; } workgroupBarrier();
    if tid <   4u { partial[tid] += partial[tid +   4u]; } workgroupBarrier();
    if tid <   2u { partial[tid] += partial[tid +   2u]; } workgroupBarrier();
    if tid <   1u { partial[tid] += partial[tid +   1u]; } workgroupBarrier();
    if tid == 0u { y_out[out_idx] = partial[0u] * u.scale; }
}
"#;

fn pack_ternary_u32(vals: &[f32]) -> Vec<u32> {
    let n = (vals.len() + 15) / 16;
    let mut out = vec![0u32; n];
    for (i, &v) in vals.iter().enumerate() {
        let c: u32 = match v as i32 { 1 => 1, -1 => 2, _ => 0 };
        out[i / 16] |= c << ((i % 16) * 2);
    }
    out
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v*v).sum::<f32>().sqrt();
    let nb = b.iter().map(|v| v*v).sum::<f32>().sqrt();
    d / (na * nb + 1e-8)
}

/// Infra com buffers persistentes (como seria em produção)
struct PersistentKernel {
    device:    wgpu::Device,
    queue:     wgpu::Queue,
    pipeline:  wgpu::ComputePipeline,
    bg:        wgpu::BindGroup,
    buf_y:     wgpu::Buffer,
    staging:   wgpu::Buffer,
    out_dim:   usize,
}

impl PersistentKernel {
    async fn new(packed: &[u32], x: &[f32], scale: f32, out_dim: usize, in_dim: usize) -> Self {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, force_fallback_adapter: false,
        }).await.unwrap();

        println!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("v3"), ..Default::default()
        }).await.unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        // Buffers persistentes (alocados 1x!)
        let buf_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(packed), usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (out_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniforms: [u32; 4] = [out_dim as u32, in_dim as u32, scale.to_bits(), 0];
        let buf_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&uniforms), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_u.as_entire_binding() },
            ],
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (out_dim * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { device, queue, pipeline, bg, buf_y, staging, out_dim }
    }

    /// Dispatch puro (sem alocação, sem readback) — mede só compute.
    fn dispatch_only(&self) {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
          p.set_pipeline(&self.pipeline);
          p.set_bind_group(0, &self.bg, &[]);
          p.dispatch_workgroups(self.out_dim as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// Dispatch + fence (mede compute real — espera GPU terminar).
    fn dispatch_and_wait(&self) {
        self.dispatch_only();
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    /// Dispatch + readback completo (mede end-to-end com D2H).
    fn dispatch_and_read(&self) -> Vec<f32> {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
          p.set_pipeline(&self.pipeline);
          p.set_bind_group(0, &self.bg, &[]);
          p.dispatch_workgroups(self.out_dim as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_y, 0, &self.staging, 0, (self.out_dim * 4) as u64);
        self.queue.submit(std::iter::once(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let mapped = self.staging.slice(..).get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        result
    }
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    println!("=== IARA WGSL Kernel v3 — Análise de Overhead ===\n");

    let out_dim = 8960;
    let in_dim  = 1536;
    let n = out_dim * in_dim;

    let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.000137).sin() * 0.3 + (i as f32 * 0.000073).cos() * 0.2).collect();
    let (ternary, scale) = quantize_ternary(&weights);
    let packed = pack_ternary_u32(&ternary);
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let sparsity = ternary.iter().filter(|&&v| v == 0.0).count() as f32 / n as f32;

    println!("Matriz: [{out_dim} × {in_dim}], sparsity: {:.1}%", sparsity * 100.0);

    // CPU reference
    let n_warmup = 3;
    let n_runs   = 30;

    let mut y_cpu = Vec::new();
    for _ in 0..n_warmup { y_cpu = ternary_matmul_sparse(&x, &ternary, scale, 1, out_dim, in_dim); }
    let t0 = Instant::now();
    for _ in 0..n_runs { y_cpu = ternary_matmul_sparse(&x, &ternary, scale, 1, out_dim, in_dim); }
    let cpu_us = t0.elapsed().as_micros() / n_runs as u128;

    // GPU persistent kernel
    let gpu = PersistentKernel::new(&packed, &x, scale, out_dim, in_dim).await;

    // Warmup
    for _ in 0..n_warmup { gpu.dispatch_and_wait(); }

    // A) Dispatch + fence (compute real, sem readback, sem alloc)
    let t1 = Instant::now();
    for _ in 0..n_runs { gpu.dispatch_and_wait(); }
    let compute_us = t1.elapsed().as_micros() / n_runs as u128;

    // B) Dispatch + readback (com D2H copy)
    for _ in 0..n_warmup { gpu.dispatch_and_read(); }
    let t2 = Instant::now();
    let mut y_gpu = Vec::new();
    for _ in 0..n_runs { y_gpu = gpu.dispatch_and_read(); }
    let total_us = t2.elapsed().as_micros() / n_runs as u128;

    let readback_us = total_us.saturating_sub(compute_us);

    // C) Latência batch: 28 layers em sequência (simula full model forward)
    for _ in 0..n_warmup { for _ in 0..28 { gpu.dispatch_only(); } gpu.dispatch_and_wait(); }
    let t3 = Instant::now();
    for _ in 0..n_runs {
        for _ in 0..28 { gpu.dispatch_only(); }
        gpu.dispatch_and_wait(); // fence apenas no final
    }
    let batch28_us = t3.elapsed().as_micros() / n_runs as u128;
    let per_layer_batch_us = batch28_us / 28;

    // D) Latência batch com IARA skip (14 layers)
    for _ in 0..n_warmup { for _ in 0..14 { gpu.dispatch_only(); } gpu.dispatch_and_wait(); }
    let t4 = Instant::now();
    for _ in 0..n_runs {
        for _ in 0..14 { gpu.dispatch_only(); }
        gpu.dispatch_and_wait();
    }
    let batch14_us = t4.elapsed().as_micros() / n_runs as u128;

    // Verificação
    let cos = cosine_sim(&y_cpu, &y_gpu);
    let max_diff: f32 = y_cpu.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let ok = cos > 0.9999 && max_diff < 1e-2;

    // Throughput em GFLOPS (considerando 2 ops por multiply-add)
    let flops_per_matmul = out_dim as f64 * in_dim as f64 * 2.0; // não-zeros * 2
    let gflops_compute = flops_per_matmul / (compute_us as f64 * 1e3);
    let gflops_batch   = flops_per_matmul / (per_layer_batch_us as f64 * 1e3);

    // Bandwidth em GB/s (packed buffer + x_in read per dispatch)
    let bytes_read = (packed.len() * 4 + in_dim * 4) as f64;
    let bw_gbs = bytes_read / (compute_us as f64 * 1e3);

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║         IARA WGSL Kernel v3 — Análise de Overhead              ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  CPU single-core:    {:>8} µs                              ║", cpu_us);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  GPU compute (fence):     {:>5} µs  ← kernel real             ║", compute_us);
    println!("║  GPU + D2H readback:      {:>5} µs  ← com readback            ║", total_us);
    println!("║  Readback overhead:       {:>5} µs  ({:.0}% do total)         ║",
        readback_us, readback_us as f32 / total_us.max(1) as f32 * 100.0);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  28 layers batch (fence): {:>5} µs  ({:.0} µs/layer)          ║",
        batch28_us, per_layer_batch_us);
    println!("║  14 layers IARA skip:     {:>5} µs  (2.0× vs 28 layers)      ║", batch14_us);
    println!("║  Speedup IARA batch:      {:.2}×                               ║",
        batch28_us as f32 / batch14_us.max(1) as f32);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Throughput compute: {:.1} GFLOPS                           ║", gflops_compute);
    println!("║  Throughput batch:   {:.1} GFLOPS/layer                     ║", gflops_batch);
    println!("║  Memory bandwidth:   {:.1} GB/s (effective)                 ║", bw_gbs);
    println!("║  RX 6750 XT peak:    288 GB/s → {:.0}% utilization            ║",
        bw_gbs / 288.0 * 100.0);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Correctness:  CosSim={:.6}  max|diff|={:.2e}  {}          ║", cos, max_diff, if ok { "✓" } else { "✗" });
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Projeção de latência de inferência real
    println!("\n--- Projeção: Inferência real por token ---");
    println!("  Ternary MLP (gate+up+down) per layer:");
    println!("    3 × {compute_us} µs = {} µs (compute only)", compute_us * 3);
    let attn_est = compute_us * 2; // Q/K/V projections (estimativa grosseira)
    println!("  Attn (Q+K+V+O proj) per layer: ~{attn_est} µs");
    let layer_total = compute_us * 3 + attn_est;
    println!("  Total per layer: ~{layer_total} µs");
    let base_28 = layer_total * 28;
    let iara_14 = layer_total * 14;
    println!("  Baseline (28 layers): ~{base_28} µs = {:.1} ms", base_28 as f32 / 1000.0);
    println!("  IARA skip (14 layers): ~{iara_14} µs = {:.1} ms", iara_14 as f32 / 1000.0);
    println!("  Tokens/s baseline: ~{:.0}", 1_000_000.0 / base_28 as f64);
    println!("  Tokens/s IARA:     ~{:.0}", 1_000_000.0 / iara_14 as f64);

    if !ok { std::process::exit(1); }
}
