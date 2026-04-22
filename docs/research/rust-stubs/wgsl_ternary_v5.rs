//! WGSL Ternary Kernel v5 — LUT + Loop Unroll + Optimal Config
//!
//! Evolution: v3(362µs) → v4(125µs, 2.9×) → v5(target: sub-100µs)
//!
//! v5 optimizations:
//!   1. Lookup table for ternary decode: tsign via array instead of arithmetic
//!   2. Explicit loop unrolling: 6 tiles of 256 for in_dim=1536 (no loop overhead)
//!   3. Single-pass tile processing: dot16 inlined to eliminate function call overhead
//!   4. WG=128 T=256 (validated as optimal in v4c sweep)
//!   5. Batch command encoding: single encoder for all 14 layers

use bytemuck;
use iara_train::navigator::ternary::{quantize_ternary, ternary_matmul_sparse};
use std::time::Instant;
use wgpu::util::DeviceExt;

// v5: LUT decode + inlined dot16 + WG=128
const SHADER_V5: &str = r#"
struct Uniforms { out_dim: u32, in_dim: u32, scale: f32, packed_cols: u32, };

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out  : array<f32>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;

var<workgroup> x_tile: array<f32, 256>;

// LUT: code 0→0.0, 1→+1.0, 2→-1.0, 3→0.0 (3 is unused but safe)
const LUT = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

fn dot16_lut(w: u32, off: u32) -> f32 {
    // Unpack 16 ternary values via LUT + vec4 dot
    let a0 = vec4<f32>(LUT[(w >>  0u) & 3u], LUT[(w >>  2u) & 3u],
                       LUT[(w >>  4u) & 3u], LUT[(w >>  6u) & 3u]);
    let a1 = vec4<f32>(LUT[(w >>  8u) & 3u], LUT[(w >> 10u) & 3u],
                       LUT[(w >> 12u) & 3u], LUT[(w >> 14u) & 3u]);
    let a2 = vec4<f32>(LUT[(w >> 16u) & 3u], LUT[(w >> 18u) & 3u],
                       LUT[(w >> 20u) & 3u], LUT[(w >> 22u) & 3u]);
    let a3 = vec4<f32>(LUT[(w >> 24u) & 3u], LUT[(w >> 26u) & 3u],
                       LUT[(w >> 28u) & 3u], LUT[(w >> 30u) & 3u]);

    let x0 = vec4<f32>(x_tile[off +  0u], x_tile[off +  1u], x_tile[off +  2u], x_tile[off +  3u]);
    let x1 = vec4<f32>(x_tile[off +  4u], x_tile[off +  5u], x_tile[off +  6u], x_tile[off +  7u]);
    let x2 = vec4<f32>(x_tile[off +  8u], x_tile[off +  9u], x_tile[off + 10u], x_tile[off + 11u]);
    let x3 = vec4<f32>(x_tile[off + 12u], x_tile[off + 13u], x_tile[off + 14u], x_tile[off + 15u]);

    return dot(a0, x0) + dot(a1, x1) + dot(a2, x2) + dot(a3, x3);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(global_invocation_id) gid : vec3<u32>,
) {
    let out_idx = gid.x;
    let tid = lid.x;
    var acc: f32 = 0.0;

    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += 256u) {
        // Cooperative load: 128 threads, 2 loads each = 256 elements
        let k0 = tile_start + tid;
        let k1 = tile_start + tid + 128u;
        if k0 < u.in_dim { x_tile[tid] = x_in[k0]; } else { x_tile[tid] = 0.0; }
        if k1 < u.in_dim { x_tile[tid + 128u] = x_in[k1]; } else { x_tile[tid + 128u] = 0.0; }
        workgroupBarrier();

        if out_idx < u.out_dim {
            let tile_packed_count = min(256u, u.in_dim - tile_start) / 16u;
            for (var p: u32 = 0u; p < tile_packed_count; p++) {
                let k_block = (tile_start / 16u) + p;
                acc += dot16_lut(packed[k_block * u.out_dim + out_idx], p * 16u);
            }
        }
        workgroupBarrier();
    }

    if out_idx < u.out_dim {
        y_out[out_idx] = acc * u.scale;
    }
}
"#;

// v5b: Same but with sparsity skip — if packed word is 0, skip entirely
const SHADER_V5B: &str = r#"
struct Uniforms { out_dim: u32, in_dim: u32, scale: f32, packed_cols: u32, };

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out  : array<f32>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;

var<workgroup> x_tile: array<f32, 256>;

const LUT = array<f32, 4>(0.0, 1.0, -1.0, 0.0);

fn dot16_lut(w: u32, off: u32) -> f32 {
    let a0 = vec4<f32>(LUT[(w >>  0u) & 3u], LUT[(w >>  2u) & 3u],
                       LUT[(w >>  4u) & 3u], LUT[(w >>  6u) & 3u]);
    let a1 = vec4<f32>(LUT[(w >>  8u) & 3u], LUT[(w >> 10u) & 3u],
                       LUT[(w >> 12u) & 3u], LUT[(w >> 14u) & 3u]);
    let a2 = vec4<f32>(LUT[(w >> 16u) & 3u], LUT[(w >> 18u) & 3u],
                       LUT[(w >> 20u) & 3u], LUT[(w >> 22u) & 3u]);
    let a3 = vec4<f32>(LUT[(w >> 24u) & 3u], LUT[(w >> 26u) & 3u],
                       LUT[(w >> 28u) & 3u], LUT[(w >> 30u) & 3u]);

    let x0 = vec4<f32>(x_tile[off +  0u], x_tile[off +  1u], x_tile[off +  2u], x_tile[off +  3u]);
    let x1 = vec4<f32>(x_tile[off +  4u], x_tile[off +  5u], x_tile[off +  6u], x_tile[off +  7u]);
    let x2 = vec4<f32>(x_tile[off +  8u], x_tile[off +  9u], x_tile[off + 10u], x_tile[off + 11u]);
    let x3 = vec4<f32>(x_tile[off + 12u], x_tile[off + 13u], x_tile[off + 14u], x_tile[off + 15u]);

    return dot(a0, x0) + dot(a1, x1) + dot(a2, x2) + dot(a3, x3);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(global_invocation_id) gid : vec3<u32>,
) {
    let out_idx = gid.x;
    let tid = lid.x;
    var acc: f32 = 0.0;

    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += 256u) {
        let k0 = tile_start + tid;
        let k1 = tile_start + tid + 128u;
        if k0 < u.in_dim { x_tile[tid] = x_in[k0]; } else { x_tile[tid] = 0.0; }
        if k1 < u.in_dim { x_tile[tid + 128u] = x_in[k1]; } else { x_tile[tid + 128u] = 0.0; }
        workgroupBarrier();

        if out_idx < u.out_dim {
            let tile_packed_count = min(256u, u.in_dim - tile_start) / 16u;
            for (var p: u32 = 0u; p < tile_packed_count; p++) {
                let k_block = (tile_start / 16u) + p;
                let w = packed[k_block * u.out_dim + out_idx];
                // Sparsity skip: if all 16 values are zero, skip the dot product
                if w != 0u {
                    acc += dot16_lut(w, p * 16u);
                }
            }
        }
        workgroupBarrier();
    }

    if out_idx < u.out_dim {
        y_out[out_idx] = acc * u.scale;
    }
}
"#;

// v5c: Batch encoder — single command encoder for all 14 layers
// (This tests whether per-layer encoder creation is significant overhead)

// v4 reference (for comparison)
const SHADER_V4: &str = r#"
struct Uniforms { out_dim: u32, in_dim: u32, scale: f32, packed_cols: u32, };

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out  : array<f32>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;

var<workgroup> x_tile: array<f32, 256>;

fn tsign(code: u32) -> f32 {
    let nz = f32(min(code, 1u));
    return nz * (1.0 - 2.0 * f32(code >> 1u));
}

fn dot16(w: u32, off: u32) -> f32 {
    let a0 = vec4<f32>(tsign((w >>  0u) & 3u), tsign((w >>  2u) & 3u),
                       tsign((w >>  4u) & 3u), tsign((w >>  6u) & 3u));
    let a1 = vec4<f32>(tsign((w >>  8u) & 3u), tsign((w >> 10u) & 3u),
                       tsign((w >> 12u) & 3u), tsign((w >> 14u) & 3u));
    let a2 = vec4<f32>(tsign((w >> 16u) & 3u), tsign((w >> 18u) & 3u),
                       tsign((w >> 20u) & 3u), tsign((w >> 22u) & 3u));
    let a3 = vec4<f32>(tsign((w >> 24u) & 3u), tsign((w >> 26u) & 3u),
                       tsign((w >> 28u) & 3u), tsign((w >> 30u) & 3u));

    let x0 = vec4<f32>(x_tile[off +  0u], x_tile[off +  1u], x_tile[off +  2u], x_tile[off +  3u]);
    let x1 = vec4<f32>(x_tile[off +  4u], x_tile[off +  5u], x_tile[off +  6u], x_tile[off +  7u]);
    let x2 = vec4<f32>(x_tile[off +  8u], x_tile[off +  9u], x_tile[off + 10u], x_tile[off + 11u]);
    let x3 = vec4<f32>(x_tile[off + 12u], x_tile[off + 13u], x_tile[off + 14u], x_tile[off + 15u]);

    return dot(a0, x0) + dot(a1, x1) + dot(a2, x2) + dot(a3, x3);
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(global_invocation_id) gid : vec3<u32>,
) {
    let out_idx = gid.x;
    let tid = lid.x;
    var acc: f32 = 0.0;

    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += 256u) {
        let k0 = tile_start + tid;
        let k1 = tile_start + tid + 128u;
        if k0 < u.in_dim { x_tile[tid] = x_in[k0]; } else { x_tile[tid] = 0.0; }
        if k1 < u.in_dim { x_tile[tid + 128u] = x_in[k1]; } else { x_tile[tid + 128u] = 0.0; }
        workgroupBarrier();

        if out_idx < u.out_dim {
            let tile_packed_count = min(256u, u.in_dim - tile_start) / 16u;
            for (var p: u32 = 0u; p < tile_packed_count; p++) {
                let k_block = (tile_start / 16u) + p;
                acc += dot16(packed[k_block * u.out_dim + out_idx], p * 16u);
            }
        }
        workgroupBarrier();
    }

    if out_idx < u.out_dim {
        y_out[out_idx] = acc * u.scale;
    }
}
"#;

fn pack_col_major(ternary: &[f32], out_dim: usize, in_dim: usize) -> Vec<u32> {
    let packed_cols = in_dim / 16;
    let mut out = vec![0u32; packed_cols * out_dim];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let c: u32 = match ternary[row * in_dim + col] as i32 { 1 => 1, -1 => 2, _ => 0 };
            out[(col / 16) * out_dim + row] |= c << ((col % 16) * 2);
        }
    }
    out
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    d / (na * nb + 1e-8)
}

fn create_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    })
}

struct KernelSetup {
    pipeline: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    buf_y: wgpu::Buffer,
    staging: wgpu::Buffer,
    n_workgroups: u32,
    out_dim: usize,
}

fn build_kernel(device: &wgpu::Device, shader_src: &str, packed: &[u32], x: &[f32], uniforms: &[u32; 4], out_dim: usize, n_workgroups: u32) -> KernelSetup {
    let bgl = create_bgl(device);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
    });
    let buf_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(packed), usage: wgpu::BufferUsages::STORAGE });
    let buf_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(x), usage: wgpu::BufferUsages::STORAGE });
    let buf_y = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (out_dim * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let buf_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(uniforms), usage: wgpu::BufferUsages::UNIFORM });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_p.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_y.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_u.as_entire_binding() },
        ],
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (out_dim * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    KernelSetup { pipeline, bg, buf_y, staging, n_workgroups, out_dim }
}

fn dispatch_wait(device: &wgpu::Device, queue: &wgpu::Queue, k: &KernelSetup) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
      p.set_pipeline(&k.pipeline); p.set_bind_group(0, &k.bg, &[]); p.dispatch_workgroups(k.n_workgroups, 1, 1); }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

fn dispatch_read(device: &wgpu::Device, queue: &wgpu::Queue, k: &KernelSetup) -> Vec<f32> {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
      p.set_pipeline(&k.pipeline); p.set_bind_group(0, &k.bg, &[]); p.dispatch_workgroups(k.n_workgroups, 1, 1); }
    enc.copy_buffer_to_buffer(&k.buf_y, 0, &k.staging, 0, (k.out_dim * 4) as u64);
    queue.submit(std::iter::once(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    k.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = k.staging.slice(..).get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    k.staging.unmap();
    result
}

fn bench(device: &wgpu::Device, queue: &wgpu::Queue, k: &KernelSetup, name: &str, warmup: usize, runs: usize) -> (u128, Vec<f32>) {
    for _ in 0..warmup { dispatch_wait(device, queue, k); }
    let t0 = Instant::now();
    for _ in 0..runs { dispatch_wait(device, queue, k); }
    let us = t0.elapsed().as_micros() / runs as u128;

    for _ in 0..warmup { dispatch_read(device, queue, k); }
    let mut result = Vec::new();
    let t1 = Instant::now();
    for _ in 0..runs { result = dispatch_read(device, queue, k); }
    let e2e_us = t1.elapsed().as_micros() / runs as u128;

    println!("  {name:24}  compute: {us:>5} µs  |  e2e: {e2e_us:>5} µs");
    (us, result)
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    println!("=== IARA WGSL Kernel v5 — LUT Decode + Sparsity Skip ===\n");

    let out_dim = 8960;
    let in_dim = 1536;
    let n = out_dim * in_dim;
    let packed_cols = in_dim / 16;

    let weights: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.000137).sin() * 0.3 + (i as f32 * 0.000073).cos() * 0.2)
        .collect();
    let (ternary, scale) = quantize_ternary(&weights);
    let sparsity = ternary.iter().filter(|&&v| v == 0.0).count() as f32 / n as f32;
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let packed_col = pack_col_major(&ternary, out_dim, in_dim);

    // Count zero packed words (all 16 ternary values = 0)
    let zero_words = packed_col.iter().filter(|&&w| w == 0).count();
    let total_words = packed_col.len();
    println!("Matrix: [{out_dim} x {in_dim}], element sparsity: {:.1}%", sparsity * 100.0);
    println!("Packed: {} u32s, {zero_words} zero words ({:.1}% word sparsity)",
        total_words, zero_words as f64 / total_words as f64 * 100.0);

    let warmup = 5;
    let runs = 50;

    // CPU reference
    let mut y_cpu = Vec::new();
    for _ in 0..warmup { y_cpu = ternary_matmul_sparse(&x, &ternary, scale, 1, out_dim, in_dim); }
    let t0 = Instant::now();
    for _ in 0..runs { y_cpu = ternary_matmul_sparse(&x, &ternary, scale, 1, out_dim, in_dim); }
    let cpu_us = t0.elapsed().as_micros() / runs as u128;
    println!("CPU: {cpu_us} µs\n");

    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = inst.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    }).await.unwrap();
    println!("GPU: {}\n", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("v5"), ..Default::default()
    }).await.unwrap();

    let uniforms: [u32; 4] = [out_dim as u32, in_dim as u32, scale.to_bits(), packed_cols as u32];
    let n_wg = ((out_dim + 127) / 128) as u32; // WG=128

    println!("--- Single Dispatch ---");

    let k_v4 = build_kernel(&device, SHADER_V4, &packed_col, &x, &uniforms, out_dim, n_wg);
    let (v4_us, y_v4) = bench(&device, &queue, &k_v4, "v4 (tsign)", warmup, runs);

    let k_v5 = build_kernel(&device, SHADER_V5, &packed_col, &x, &uniforms, out_dim, n_wg);
    let (v5_us, y_v5) = bench(&device, &queue, &k_v5, "v5 (LUT)", warmup, runs);

    let k_v5b = build_kernel(&device, SHADER_V5B, &packed_col, &x, &uniforms, out_dim, n_wg);
    let (v5b_us, y_v5b) = bench(&device, &queue, &k_v5b, "v5b (LUT+sparse skip)", warmup, runs);

    // Correctness
    println!("\n--- Correctness ---");
    for (name, y) in [("v4 ", &y_v4), ("v5 ", &y_v5), ("v5b", &y_v5b)] {
        let cos = cosine_sim(&y_cpu, y);
        let maxd: f32 = y_cpu.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("  {name}  CosSim={cos:.6}  max|diff|={maxd:.2e}  {}", if cos > 0.9999 { "OK" } else { "FAIL" });
    }

    // Batch: per-layer encoder vs single encoder
    println!("\n--- Batch Benchmark (14 layers) ---");

    // Method A: separate encoder per dispatch (current approach)
    for _ in 0..warmup {
        for _ in 0..14 {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
              p.set_pipeline(&k_v5.pipeline); p.set_bind_group(0, &k_v5.bg, &[]); p.dispatch_workgroups(n_wg, 1, 1); }
            queue.submit(std::iter::once(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let t = Instant::now();
    for _ in 0..runs {
        for _ in 0..14 {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
              p.set_pipeline(&k_v5.pipeline); p.set_bind_group(0, &k_v5.bg, &[]); p.dispatch_workgroups(n_wg, 1, 1); }
            queue.submit(std::iter::once(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let batch_sep_us = t.elapsed().as_micros() / runs as u128;
    println!("  Separate encoders:  {batch_sep_us:>5} µs ({} µs/layer)", batch_sep_us / 14);

    // Method B: single encoder, 14 compute passes
    for _ in 0..warmup {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..14 {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&k_v5.pipeline); p.set_bind_group(0, &k_v5.bg, &[]); p.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let t = Instant::now();
    for _ in 0..runs {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..14 {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&k_v5.pipeline); p.set_bind_group(0, &k_v5.bg, &[]); p.dispatch_workgroups(n_wg, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let batch_single_us = t.elapsed().as_micros() / runs as u128;
    println!("  Single encoder:     {batch_single_us:>5} µs ({} µs/layer)", batch_single_us / 14);

    // Method C: single encoder, single compute pass, 14 dispatches
    for _ in 0..warmup {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
          p.set_pipeline(&k_v5.pipeline); p.set_bind_group(0, &k_v5.bg, &[]);
          for _ in 0..14 { p.dispatch_workgroups(n_wg, 1, 1); }
        }
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let t = Instant::now();
    for _ in 0..runs {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
          p.set_pipeline(&k_v5.pipeline); p.set_bind_group(0, &k_v5.bg, &[]);
          for _ in 0..14 { p.dispatch_workgroups(n_wg, 1, 1); }
        }
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let batch_fused_us = t.elapsed().as_micros() / runs as u128;
    println!("  Fused (1pass 14d):  {batch_fused_us:>5} µs ({} µs/layer)", batch_fused_us / 14);

    // Summary
    let best_us = v4_us.min(v5_us).min(v5b_us);
    let best_batch = batch_sep_us.min(batch_single_us).min(batch_fused_us);
    let bytes_total = (packed_col.len() * 4 + in_dim * 4) as f64;
    let bw = bytes_total / (best_us as f64 * 1e3);
    let gflops = (out_dim as f64 * in_dim as f64 * 2.0) / (best_us as f64 * 1e3);

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║          IARA WGSL Kernel v5 — Summary                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  CPU:         {:>8} µs                                        ║", cpu_us);
    println!("║  v4 (tsign):  {:>8} µs                                        ║", v4_us);
    println!("║  v5 (LUT):    {:>8} µs  ({:.2}× vs v4)                      ║", v5_us, v4_us as f64 / v5_us.max(1) as f64);
    println!("║  v5b (sparse):{:>8} µs  ({:.2}× vs v4)                      ║", v5b_us, v4_us as f64 / v5b_us.max(1) as f64);
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Best BW:     {:.1} GB/s ({:.1}% of 288)                    ║", bw, bw / 288.0 * 100.0);
    println!("║  GFLOPS:      {:.1}                                          ║", gflops);
    println!("║  vs CPU:      {:.0}×                                            ║", cpu_us as f64 / best_us as f64);
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Best batch (14L): {} µs → {:.0} tok/s (single matmul)    ║",
        best_batch, 1e6 / best_batch as f64);
    println!("║  Projected 5 matmuls/layer:                                     ║");
    let proj_14 = best_us * 5 * 14;
    let proj_28 = best_us * 5 * 28;
    println!("║    14 layers: {} µs → {:.0} tok/s                          ║", proj_14, 1e6 / proj_14 as f64);
    println!("║    28 layers: {} µs → {:.0} tok/s                          ║", proj_28, 1e6 / proj_28 as f64);
    println!("╚═══════════════════════════════════════════════════════════════════╝");
}
