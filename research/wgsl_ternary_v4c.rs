//! WGSL Ternary Kernel v4c — Occupancy-tuned register tiling
//!
//! v4 winner: tiled shared mem + col-major + no-reduction = 125µs (2.9× vs v3)
//! v4b lost: 2row/thread with 256-thread WG → only 18 WGs for 32 CUs → underutilized
//!
//! v4c strategy: test multiple (workgroup_size, rows_per_thread) combos to find
//! the sweet spot between register tiling benefit and occupancy.
//!
//! Variants:
//!   A) WG=128, 1row/thread → 70 WGs (2.2 per CU)
//!   B) WG=64,  1row/thread → 140 WGs (4.4 per CU)
//!   C) WG=64,  2row/thread → 70 WGs (2.2 per CU), 2× arith per x_tile load
//!   D) WG=128, 2row/thread → 35 WGs (1.1 per CU)
//!   E) WG=64,  4row/thread → 35 WGs (1.1 per CU), 4× arith per x_tile load

use bytemuck;
use iara_train::navigator::ternary::{quantize_ternary, ternary_matmul_sparse};
use std::time::Instant;
use wgpu::util::DeviceExt;

fn make_shader_with_tile(wg_size: u32, rows_per_thread: u32, tile_k: u32) -> String {
    let total_rows = wg_size * rows_per_thread;
    let loads_per_thread = tile_k / wg_size; // how many x_in elements each thread loads

    let mut acc_decls = String::new();
    let mut acc_inits = String::new();
    let mut acc_compute = String::new();
    let mut acc_store = String::new();

    for r in 0..rows_per_thread {
        let offset = r * wg_size;
        acc_decls.push_str(&format!("    var acc{r}: f32 = 0.0;\n"));
        acc_inits.push_str(&format!("    let out_idx_{r} = wid.x * {total_rows}u + tid + {offset}u;\n"));
        acc_compute.push_str(&format!(
            "            if out_idx_{r} < u.out_dim {{\n                acc{r} += dot16(packed[col_base + out_idx_{r}], off);\n            }}\n"
        ));
        acc_store.push_str(&format!(
            "    if out_idx_{r} < u.out_dim {{ y_out[out_idx_{r}] = acc{r} * u.scale; }}\n"
        ));
    }

    format!(r#"
struct Uniforms {{ out_dim: u32, in_dim: u32, scale: f32, packed_cols: u32, }};

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       x_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out  : array<f32>;
@group(0) @binding(3) var<uniform>             u      : Uniforms;

var<workgroup> x_tile: array<f32, {tile_k}>;

fn tsign(code: u32) -> f32 {{
    let nz = f32(min(code, 1u));
    return nz * (1.0 - 2.0 * f32(code >> 1u));
}}

fn dot16(w: u32, off: u32) -> f32 {{
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
}}

@compute @workgroup_size({wg_size}, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(workgroup_id)         wid : vec3<u32>,
) {{
    let tid = lid.x;
{acc_inits}
{acc_decls}
    for (var tile_start: u32 = 0u; tile_start < u.in_dim; tile_start += {tile_k}u) {{
        // Cooperative load: each thread loads {loads_per_thread} elements
        for (var ld: u32 = 0u; ld < {loads_per_thread}u; ld++) {{
            let local_idx = ld * {wg_size}u + tid;
            let k_global = tile_start + local_idx;
            if k_global < u.in_dim && local_idx < {tile_k}u {{
                x_tile[local_idx] = x_in[k_global];
            }} else if local_idx < {tile_k}u {{
                x_tile[local_idx] = 0.0;
            }}
        }}
        workgroupBarrier();

        let tile_packed_count = min({tile_k}u, u.in_dim - tile_start) / 16u;
        for (var p: u32 = 0u; p < tile_packed_count; p++) {{
            let k_block = (tile_start / 16u) + p;
            let col_base = k_block * u.out_dim;
            let off = p * 16u;
{acc_compute}
        }}
        workgroupBarrier();
    }}

{acc_store}
}}
"#)
}

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

fn main() {
    pollster::block_on(run());
}

async fn run() {
    println!("=== IARA WGSL Kernel v4c — Occupancy Sweep ===\n");

    let out_dim = 8960;
    let in_dim = 1536;
    let n = out_dim * in_dim;
    let packed_cols = in_dim / 16;

    let weights: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.000137).sin() * 0.3 + (i as f32 * 0.000073).cos() * 0.2)
        .collect();
    let (ternary, scale) = quantize_ternary(&weights);
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let packed_col = pack_col_major(&ternary, out_dim, in_dim);

    // CPU reference
    let warmup = 3;
    let runs = 30;
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
        label: Some("v4c"), ..Default::default()
    }).await.unwrap();

    let uniforms: [u32; 4] = [out_dim as u32, in_dim as u32, scale.to_bits(), packed_cols as u32];

    // (wg_size, rows_per_thread, tile_k, description)
    let configs: Vec<(u32, u32, u32, &str)> = vec![
        (256, 1, 256, "WG256 1r T256"),
        (128, 1, 256, "WG128 1r T256"),
        (128, 2, 256, "WG128 2r T256"),
        (64,  1, 256, "WG64  1r T256"),
        (64,  2, 256, "WG64  2r T256"),
        (64,  4, 256, "WG64  4r T256"),
        (64,  1, 64,  "WG64  1r T64 "),
        (64,  2, 64,  "WG64  2r T64 "),
        (32,  1, 256, "WG32  1r T256"),
        (32,  2, 256, "WG32  2r T256"),
        (32,  4, 256, "WG32  4r T256"),
        (32,  8, 256, "WG32  8r T256"),
    ];

    println!("{:20} {:>6} {:>6} {:>5} {:>5}   {:>8} {:>8}",
        "Config", "WGs", "Thds", "µs", "CosSim", "GB/s", "GFLOPS");
    println!("{:-<80}", "");

    let bytes_total = (packed_col.len() * 4 + in_dim * 4) as f64;
    let flops_total = out_dim as f64 * in_dim as f64 * 2.0;

    let mut best_us = u128::MAX;
    let mut best_name = "";

    for (wg_size, rows_per_thread, tile_k, name) in &configs {
        let total_rows_per_wg = wg_size * rows_per_thread;
        let n_wgs = ((out_dim as u32 + total_rows_per_wg - 1) / total_rows_per_wg) as u32;
        let total_threads = n_wgs * wg_size;

        let shader_src = make_shader_with_tile(*wg_size, *rows_per_thread, *tile_k);
        let k = build_kernel(&device, &shader_src, &packed_col, &x, &uniforms, out_dim, n_wgs);

        // Warmup
        for _ in 0..warmup { dispatch_wait(&device, &queue, &k); }

        // Benchmark
        let t = Instant::now();
        for _ in 0..runs { dispatch_wait(&device, &queue, &k); }
        let us = t.elapsed().as_micros() / runs as u128;

        // Correctness
        let y_gpu = dispatch_read(&device, &queue, &k);
        let cos = cosine_sim(&y_cpu, &y_gpu);

        let bw = bytes_total / (us as f64 * 1e3);
        let gf = flops_total / (us as f64 * 1e3);

        let marker = if us < best_us { best_us = us; best_name = name; " *" } else { "  " };
        println!("{name:20} {:>6} {:>6} {:>5} {cos:.4}   {:>7.1} {:>7.1}{marker}",
            n_wgs, total_threads, us, bw, gf);
    }

    println!("\nBest: {best_name} at {best_us} µs ({:.1}× vs CPU)", cpu_us as f64 / best_us as f64);

    // Batch for best config
    println!("\n--- Batch (14-layer IARA skip) with best config ---");
    let (wg_size, rows_per_thread, tile_k, _) = configs.iter().find(|(_, _, _, n)| *n == best_name).unwrap();
    let total_rows_per_wg = wg_size * rows_per_thread;
    let n_wgs = ((out_dim as u32 + total_rows_per_wg - 1) / total_rows_per_wg) as u32;
    let shader_src = make_shader_with_tile(*wg_size, *rows_per_thread, *tile_k);
    let k_best = build_kernel(&device, &shader_src, &packed_col, &x, &uniforms, out_dim, n_wgs);

    for _ in 0..warmup {
        for _ in 0..14 {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
              p.set_pipeline(&k_best.pipeline); p.set_bind_group(0, &k_best.bg, &[]); p.dispatch_workgroups(n_wgs, 1, 1); }
            queue.submit(std::iter::once(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let t = Instant::now();
    for _ in 0..runs {
        for _ in 0..14 {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            { let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
              p.set_pipeline(&k_best.pipeline); p.set_bind_group(0, &k_best.bg, &[]); p.dispatch_workgroups(n_wgs, 1, 1); }
            queue.submit(std::iter::once(enc.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let batch_us = t.elapsed().as_micros() / runs as u128;
    println!("  14-layer: {batch_us} µs ({} µs/layer) → {:.0} tok/s (single matmul)",
        batch_us / 14, 1e6 / batch_us as f64);

    let matmuls_per_layer = 5u128;
    let layer_us = best_us * matmuls_per_layer;
    let iara_14 = layer_us * 14;
    println!("  Projected full layer ({matmuls_per_layer} matmuls): {layer_us} µs/layer");
    println!("  Projected 14-layer: {iara_14} µs → {:.0} tok/s", 1e6 / iara_14 as f64);
    println!("  Projected 28-layer: {} µs → {:.0} tok/s", layer_us * 28, 1e6 / (layer_us as f64 * 28.0));
}
