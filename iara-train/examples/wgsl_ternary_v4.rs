//! WGSL Ternary Kernel v4 — Tiled Shared Memory + No-Reduction Architecture
//!
//! Eliminates the #1 bottleneck from v3: the 8-barrier reduction tree.
//!
//! Key optimizations vs v3:
//!   1. One thread per output element (no workgroup reduction at all)
//!   2. x_in loaded cooperatively into shared memory (amortized across 256 outputs)
//!   3. Column-major packed layout for coalesced global reads
//!   4. Full u32 unpack: 16 ternary → 4× vec4 dot products (hardware FMA)
//!   5. v4b: Register tiling (2 rows/thread) to double arithmetic per shared mem load
//!
//! Memory layout change:
//!   v3: packed[row * packed_cols + k_block]     (row-major, strided reads)
//!   v4: packed[k_block * out_dim + row]          (col-major, coalesced reads)

use bytemuck;
use iara_train::navigator::ternary::{quantize_ternary, ternary_matmul_sparse};
use std::time::Instant;
use wgpu::util::DeviceExt;

// ─── v3 baseline (reduction-based, row-major) ──────────────────────────────

const SHADER_V3: &str = r#"
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
        if b0 <= 24u {
            let w = packed[u0];
            acc += tsign((w >> b0) & 3u) * x_in[j]
                 + tsign((w >> (b0 + 2u)) & 3u) * x_in[j+1u]
                 + tsign((w >> (b0 + 4u)) & 3u) * x_in[j+2u]
                 + tsign((w >> (b0 + 6u)) & 3u) * x_in[j+3u];
        } else {
            let e0=base; let e1=base+1u; let e2=base+2u; let e3=base+3u;
            acc += tsign((packed[e0/16u] >> ((e0%16u)*2u)) & 3u) * x_in[j]
                 + tsign((packed[e1/16u] >> ((e1%16u)*2u)) & 3u) * x_in[j+1u]
                 + tsign((packed[e2/16u] >> ((e2%16u)*2u)) & 3u) * x_in[j+2u]
                 + tsign((packed[e3/16u] >> ((e3%16u)*2u)) & 3u) * x_in[j+3u];
        }
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

// ─── v4: No-reduction + shared x_in + column-major packed ──────────────────

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

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(global_invocation_id) gid : vec3<u32>,
) {
    let out_idx = gid.x;
    let tid = lid.x;
    var acc: f32 = 0.0;

    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += 256u) {
        // Cooperative load: 256 threads, 1 float each → shared memory
        let k_global = tile_start + tid;
        if k_global < u.in_dim {
            x_tile[tid] = x_in[k_global];
        } else {
            x_tile[tid] = 0.0;
        }
        workgroupBarrier();

        // Each thread processes its own output row using shared x_in
        if out_idx < u.out_dim {
            let tile_packed_count = min(256u, u.in_dim - tile_start) / 16u;
            for (var p: u32 = 0u; p < tile_packed_count; p++) {
                let k_block = (tile_start / 16u) + p;
                // Column-major: adjacent threads → adjacent addresses
                let w = packed[k_block * u.out_dim + out_idx];
                acc += dot16(w, p * 16u);
            }
        }
        workgroupBarrier();
    }

    if out_idx < u.out_dim {
        y_out[out_idx] = acc * u.scale;
    }
}
"#;

// ─── v4b: Register tiling — 2 output rows per thread ───────────────────────

const SHADER_V4B: &str = r#"
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

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(workgroup_id)         wid : vec3<u32>,
) {
    let tid = lid.x;
    let out_idx_0 = wid.x * 512u + tid;
    let out_idx_1 = out_idx_0 + 256u;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;

    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += 256u) {
        let k_global = tile_start + tid;
        if k_global < u.in_dim {
            x_tile[tid] = x_in[k_global];
        } else {
            x_tile[tid] = 0.0;
        }
        workgroupBarrier();

        let tile_packed_count = min(256u, u.in_dim - tile_start) / 16u;
        for (var p: u32 = 0u; p < tile_packed_count; p++) {
            let k_block = (tile_start / 16u) + p;
            let col_base = k_block * u.out_dim;
            let off = p * 16u;

            if out_idx_0 < u.out_dim {
                acc0 += dot16(packed[col_base + out_idx_0], off);
            }
            if out_idx_1 < u.out_dim {
                acc1 += dot16(packed[col_base + out_idx_1], off);
            }
        }
        workgroupBarrier();
    }

    if out_idx_0 < u.out_dim { y_out[out_idx_0] = acc0 * u.scale; }
    if out_idx_1 < u.out_dim { y_out[out_idx_1] = acc1 * u.scale; }
}
"#;

// ─── Packing ────────────────────────────────────────────────────────────────

/// Row-major: packed[row * packed_cols + k_block]
fn pack_row_major(ternary: &[f32], out_dim: usize, in_dim: usize) -> Vec<u32> {
    let packed_cols = in_dim / 16;
    let mut out = vec![0u32; out_dim * packed_cols];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let c: u32 = match ternary[row * in_dim + col] as i32 { 1 => 1, -1 => 2, _ => 0 };
            out[row * packed_cols + col / 16] |= c << ((col % 16) * 2);
        }
    }
    out
}

/// Column-major: packed[k_block * out_dim + row]
/// Adjacent threads (consecutive rows) read consecutive addresses → coalesced.
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

// ─── GPU infrastructure ────────────────────────────────────────────────────

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

fn build_kernel(
    device: &wgpu::Device,
    shader_src: &str,
    packed: &[u32],
    x: &[f32],
    uniforms: &[u32; 4],
    out_dim: usize,
    n_workgroups: u32,
) -> KernelSetup {
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
    let compute_us = t0.elapsed().as_micros() / runs as u128;

    for _ in 0..warmup { dispatch_read(device, queue, k); }
    let mut result = Vec::new();
    let t1 = Instant::now();
    for _ in 0..runs { result = dispatch_read(device, queue, k); }
    let total_us = t1.elapsed().as_micros() / runs as u128;

    println!("  {name:24}  compute: {compute_us:>5} µs  |  end-to-end: {total_us:>5} µs");
    (compute_us, result)
}

fn bench_batch(device: &wgpu::Device, queue: &wgpu::Queue, k: &KernelSetup, name: &str, n_layers: u32, warmup: usize, runs: usize) -> u128 {
    for _ in 0..warmup {
        for _ in 0..n_layers {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
              p.set_pipeline(&k.pipeline); p.set_bind_group(0, &k.bg, &[]); p.dispatch_workgroups(k.n_workgroups, 1, 1); }
            queue.submit(std::iter::once(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let t = Instant::now();
    for _ in 0..runs {
        for _ in 0..n_layers {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
              p.set_pipeline(&k.pipeline); p.set_bind_group(0, &k.bg, &[]); p.dispatch_workgroups(k.n_workgroups, 1, 1); }
            queue.submit(std::iter::once(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let batch_us = t.elapsed().as_micros() / runs as u128;
    let per_layer = batch_us / n_layers as u128;
    let toks = 1_000_000.0 / batch_us as f64;
    println!("  {name:24}  {n_layers}-layer: {batch_us:>5} µs ({per_layer} µs/layer) → {toks:.0} tok/s");
    batch_us
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    println!("=== IARA WGSL Kernel v4 — Tiled + No-Reduction Architecture ===\n");

    let out_dim = 8960;
    let in_dim = 1536;
    let n = out_dim * in_dim;
    let packed_cols = in_dim / 16; // 96

    let weights: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.000137).sin() * 0.3 + (i as f32 * 0.000073).cos() * 0.2)
        .collect();
    let (ternary, scale) = quantize_ternary(&weights);
    let sparsity = ternary.iter().filter(|&&v| v == 0.0).count() as f32 / n as f32;
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    println!("Matrix: [{out_dim} x {in_dim}], sparsity: {:.1}%, scale: {scale:.6}", sparsity * 100.0);
    println!("Packed size: {:.2} MB ({} u32s)", out_dim as f64 * packed_cols as f64 * 4.0 / 1e6, out_dim * packed_cols);

    let packed_row = pack_row_major(&ternary, out_dim, in_dim);
    let packed_col = pack_col_major(&ternary, out_dim, in_dim);

    // CPU reference
    let warmup = 3;
    let runs = 30;
    let mut y_cpu = Vec::new();
    for _ in 0..warmup { y_cpu = ternary_matmul_sparse(&x, &ternary, scale, 1, out_dim, in_dim); }
    let t0 = Instant::now();
    for _ in 0..runs { y_cpu = ternary_matmul_sparse(&x, &ternary, scale, 1, out_dim, in_dim); }
    let cpu_us = t0.elapsed().as_micros() / runs as u128;
    println!("CPU reference: {cpu_us} µs\n");

    // GPU init
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = inst.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    }).await.unwrap();
    println!("GPU: {}\n", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("v4"), ..Default::default()
    }).await.unwrap();

    // ─── Build all kernels ──────────────────────────────────────────────────

    let u_v3: [u32; 4] = [out_dim as u32, in_dim as u32, scale.to_bits(), 0];
    let u_v4: [u32; 4] = [out_dim as u32, in_dim as u32, scale.to_bits(), packed_cols as u32];

    println!("--- Single Dispatch Comparison ---");

    let k_v3 = build_kernel(&device, SHADER_V3, &packed_row, &x, &u_v3, out_dim, out_dim as u32);
    let (v3_us, y_v3) = bench(&device, &queue, &k_v3, "v3 (reduction)", warmup, runs);

    let n_wg_v4 = ((out_dim + 255) / 256) as u32; // 35 workgroups
    let k_v4 = build_kernel(&device, SHADER_V4, &packed_col, &x, &u_v4, out_dim, n_wg_v4);
    let (v4_us, y_v4) = bench(&device, &queue, &k_v4, "v4 (tiled+colmaj)", warmup, runs);

    let n_wg_v4b = ((out_dim + 511) / 512) as u32; // 18 workgroups
    let k_v4b = build_kernel(&device, SHADER_V4B, &packed_col, &x, &u_v4, out_dim, n_wg_v4b);
    let (v4b_us, y_v4b) = bench(&device, &queue, &k_v4b, "v4b (2row/thread)", warmup, runs);

    // ─── Batch benchmark (14-layer IARA skip) ──────────────────────────────

    println!("\n--- Batch Latency (14 layers = IARA skip path) ---");
    bench_batch(&device, &queue, &k_v3, "v3 (reduction)", 14, warmup, runs);
    bench_batch(&device, &queue, &k_v4, "v4 (tiled+colmaj)", 14, warmup, runs);
    bench_batch(&device, &queue, &k_v4b, "v4b (2row/thread)", 14, warmup, runs);

    // ─── Correctness ────────────────────────────────────────────────────────

    println!("\n--- Correctness (vs CPU reference) ---");
    for (name, y_gpu) in [("v3 ", &y_v3), ("v4 ", &y_v4), ("v4b", &y_v4b)] {
        let cos = cosine_sim(&y_cpu, y_gpu);
        let maxd: f32 = y_cpu.iter().zip(y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let ok = cos > 0.9999;
        println!("  {name}  CosSim={cos:.6}  max|diff|={maxd:.2e}  {}", if ok { "OK" } else { "FAIL" });
        if !ok {
            // Debug: show first few values
            println!("       CPU[0..5]: {:?}", &y_cpu[..5]);
            println!("       GPU[0..5]: {:?}", &y_gpu[..5]);
        }
    }

    // ─── Summary ────────────────────────────────────────────────────────────

    let best_us = v4_us.min(v4b_us);
    let bytes_total = (packed_col.len() * 4 + in_dim * 4) as f64;
    let bw_gbs = bytes_total / (best_us as f64 * 1e3);
    let gflops = (out_dim as f64 * in_dim as f64 * 2.0) / (best_us as f64 * 1e3);

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║          IARA WGSL Kernel v4 — Summary                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  CPU single-core:    {:>8} µs                                  ║", cpu_us);
    println!("║  v3 (reduction):     {:>8} µs  ← baseline                      ║", v3_us);
    println!("║  v4 (tiled+colmaj):  {:>8} µs  ({:.1}× vs v3)                ║", v4_us, v3_us as f64 / v4_us.max(1) as f64);
    println!("║  v4b (2row/thread):  {:>8} µs  ({:.1}× vs v3)                ║", v4b_us, v3_us as f64 / v4b_us.max(1) as f64);
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Best BW:  {:.1} GB/s  ({:.1}% of 288 GB/s)                  ║", bw_gbs, bw_gbs / 288.0 * 100.0);
    println!("║  GFLOPS:   {:.1}                                             ║", gflops);
    println!("║  vs CPU:   {:.0}×                                               ║", cpu_us as f64 / best_us as f64);
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Projection
    println!("\n--- Projection ---");
    let matmuls_per_layer = 5u128; // gate+up+down+qkv+o
    let layer_us = best_us * matmuls_per_layer;
    let iara_14 = layer_us * 14;
    println!("  {matmuls_per_layer} matmuls/layer × {best_us} µs = {layer_us} µs/layer");
    println!("  28 layers: {} µs → {:.0} tok/s", layer_us * 28, 1e6 / (layer_us as f64 * 28.0));
    println!("  14 layers (IARA skip): {iara_14} µs → {:.0} tok/s", 1e6 / iara_14 as f64);

    // Fail if correctness is bad
    let cos_v4 = cosine_sim(&y_cpu, &y_v4);
    let cos_v4b = cosine_sim(&y_cpu, &y_v4b);
    if cos_v4 < 0.999 || cos_v4b < 0.999 {
        eprintln!("\nFAIL: Correctness check failed!");
        std::process::exit(1);
    }
}
