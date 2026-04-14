//! HIP Ternary GEMV Benchmark — Direct kernel launch on AMD RX 6750 XT
//!
//! Loads real Qwen2.5-1.5B MLP weights, quantizes to ternary {-1,0,+1},
//! packs in HIP format (row-major bytes, -1=0b00, 0=0b01, +1=0b10),
//! and benchmarks gemv_ternary_f32_w32 directly via hip-ffi.
//!
//! This measures raw kernel throughput with ZERO framework overhead.
//!
//! Run: cargo run -p iara-train --example hip_ternary_bench --release
//!
//! Expected (RX 6750 XT, 288 GB/s peak):
//!   - gate_proj [8960×1536]: ~30-60µs (ternary = 2 bits/weight, 2× less data than Q4)
//!   - 14-layer full MLP pass: <1ms → 300+ tok/s for decode

use std::path::Path;
use std::time::Instant;

// ── Safetensors loading (same as iara_e2e_gpu.rs) ──────────────────────────

fn parse_safetensors_header(mmap: &memmap2::Mmap) -> serde_json::Value {
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header_json = std::str::from_utf8(&mmap[8..8 + header_size]).unwrap();
    serde_json::from_str(header_json).unwrap()
}

fn load_weight_f32(
    safetensors: &memmap2::Mmap,
    header: &serde_json::Value,
    tensor_name: &str,
) -> Option<(Vec<f32>, [usize; 2])> {
    let info = header.get(tensor_name)?;
    let dtype = info["dtype"].as_str()?;
    let shape: Vec<usize> = info["data_offsets"].as_array().map(|_| {
        info["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
    })?;
    let offsets = info["data_offsets"].as_array()?;
    let start = offsets[0].as_u64()? as usize;
    let end = offsets[1].as_u64()? as usize;

    if shape.len() != 2 { return None; }
    let rows = shape[0];
    let cols = shape[1];

    let header_size = u64::from_le_bytes(safetensors[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_size + start;
    let data_end = 8 + header_size + end;
    let data = &safetensors[data_start..data_end];

    let values: Vec<f32> = match dtype {
        "BF16" => {
            data.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            }).collect()
        }
        "F16" => {
            data.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                let sign = ((bits >> 15) & 1) as u32;
                let exp = ((bits >> 10) & 0x1F) as u32;
                let mant = (bits & 0x3FF) as u32;
                if exp == 0 {
                    if mant == 0 { return f32::from_bits(sign << 31); }
                    let mut e = 1u32;
                    let mut m = mant;
                    while (m & 0x400) == 0 { m <<= 1; e += 1; }
                    m &= 0x3FF;
                    f32::from_bits((sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13))
                } else if exp == 31 {
                    f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
                } else {
                    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
                }
            }).collect()
        }
        "F32" => {
            let (prefix, aligned, suffix) = unsafe { data.align_to::<f32>() };
            assert!(prefix.is_empty() && suffix.is_empty());
            aligned.to_vec()
        }
        _ => return None,
    };
    Some((values, [rows, cols]))
}

// ── Ternary quantization ────────────────────────────────────────────────────

fn quantize_ternary(weights: &[f32]) -> (Vec<f32>, f32) {
    let scale = weights.iter().map(|v| v.abs()).sum::<f32>() / weights.len() as f32;
    let scale = if scale < 1e-8 { 1e-8 } else { scale };
    let t: Vec<f32> = weights.iter()
        .map(|&v| (v / scale).round().clamp(-1.0, 1.0))
        .collect();
    (t, scale)
}

/// Pack ternary values into HIP format: row-major bytes, 4 weights per byte.
/// Encoding: -1=0b00, 0=0b01, +1=0b10
/// packed[row * packed_cols + k/4] bits[(k%4)*2 +: 2]
fn pack_ternary_hip(ternary: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let packed_cols = (in_dim + 3) / 4;
    let mut packed = vec![0u8; out_dim * packed_cols];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let val = ternary[row * in_dim + col] as i32;
            // HIP encoding: -1=0b00, 0=0b01, +1=0b10
            let code: u8 = match val {
                -1 => 0b00,
                 0 => 0b01,
                 1 => 0b10,
                 _ => 0b01, // treat as zero
            };
            let byte_idx = row * packed_cols + col / 4;
            let bit_offset = (col % 4) * 2;
            packed[byte_idx] |= code << bit_offset;
        }
    }
    packed
}

/// CPU reference: ternary matmul y = W @ x * scale
fn ternary_matmul_cpu(ternary: &[f32], x: &[f32], scale: f32, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut acc = 0.0f32;
        for k in 0..in_dim {
            let w = ternary[i * in_dim + k];
            if w > 0.0 { acc += x[k]; }
            else if w < 0.0 { acc -= x[k]; }
        }
        y[i] = acc * scale;
    }
    y
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    d / (na * nb + 1e-8)
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  IARA HIP Ternary GEMV Benchmark — AMD RX 6750 XT         ║");
    println!("║  Direct kernel launch, zero framework overhead             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── 1. Initialize HIP ───────────────────────────────────────────────
    println!("▸ Initializing HIP...");
    hip_ffi::hip_set_device(0).expect("Failed to set HIP device 0");

    let stream = hip_ffi::hip_stream_create().expect("Failed to create HIP stream");
    let mut kernels = hip_ffi::GpuKernels::new().expect("Failed to load HIP kernels");
    kernels.set_stream(stream);

    println!("  ✓ HIP device 0 ready, kernels loaded\n");

    // ── 2. Load Qwen2.5-1.5B weights ───────────────────────────────────
    let model_path = Path::new("/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors");
    println!("▸ Loading weights from {:?}", model_path);

    let file = std::fs::File::open(model_path).expect("model.safetensors not found");
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let header = parse_safetensors_header(&mmap);

    // Qwen2.5-1.5B MLP matrices per layer:
    //   gate_proj: [8960, 1536]  (up-project)
    //   up_proj:   [8960, 1536]  (gate)
    //   down_proj: [1536, 8960]  (down-project)
    let layer_count = 28;
    let mlp_names = ["gate_proj", "up_proj", "down_proj"];

    // ── 3. Benchmark single matrices first ──────────────────────────────
    println!("\n═══ Phase 1: Single matrix kernel latency ═══\n");

    // Test with layer 0 gate_proj [8960, 1536]
    let test_name = "model.layers.0.mlp.gate_proj.weight";
    let (weights, shape) = load_weight_f32(&mmap, &header, test_name)
        .expect("Failed to load gate_proj");
    let (out_dim, in_dim) = (shape[0], shape[1]);
    println!("  Loaded {test_name}: [{out_dim} × {in_dim}]");

    // Quantize
    let (ternary, scale) = quantize_ternary(&weights);
    let sparsity = ternary.iter().filter(|&&v| v == 0.0).count() as f32 / ternary.len() as f32;
    println!("  Ternary scale={scale:.4}, sparsity={:.1}%", sparsity * 100.0);

    // Pack HIP format
    let packed = pack_ternary_hip(&ternary, out_dim, in_dim);
    let packed_size_kb = packed.len() as f32 / 1024.0;
    println!("  Packed: {} bytes ({:.1} KB vs {:.1} KB fp32)",
             packed.len(), packed_size_kb, (out_dim * in_dim * 4) as f32 / 1024.0);

    // Random input
    let x_host: Vec<f32> = (0..in_dim).map(|i| ((i as f32 * 0.01).sin() * 0.5)).collect();

    // CPU reference
    let y_cpu = ternary_matmul_cpu(&ternary, &x_host, scale, out_dim, in_dim);

    // Upload to GPU
    let d_packed = hip_ffi::DeviceBuffer::<u8>::from_slice(&packed).expect("upload packed");
    let d_x = hip_ffi::DeviceBuffer::<f32>::from_slice(&x_host).expect("upload x");
    let mut d_out = hip_ffi::DeviceBuffer::<f32>::alloc(out_dim).expect("alloc output");

    // Warmup
    for _ in 0..5 {
        unsafe {
            kernels.gemv_ternary_f32(
                d_packed.as_ptr(), d_x.as_ptr(), d_out.as_mut_ptr(),
                1, out_dim as i32, in_dim as i32, scale, 0.0,
            ).expect("warmup failed");
        }
    }
    hip_ffi::hip_stream_synchronize(stream).unwrap();

    // Benchmark: 100 iterations
    let n_iters = 100;
    hip_ffi::hip_stream_synchronize(stream).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_iters {
        unsafe {
            kernels.gemv_ternary_f32(
                d_packed.as_ptr(), d_x.as_ptr(), d_out.as_mut_ptr(),
                1, out_dim as i32, in_dim as i32, scale, 0.0,
            ).unwrap();
        }
    }
    hip_ffi::hip_stream_synchronize(stream).unwrap();
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

    // Verify correctness
    let y_gpu = d_out.to_host().unwrap();
    let cos = cosine_sim(&y_cpu, &y_gpu);
    let max_err = max_abs_err(&y_cpu, &y_gpu);

    let weight_bytes = packed.len() as f64;
    let bw_gbs = weight_bytes / (per_call_us * 1e-6) / 1e9;

    println!("\n  gate_proj [{out_dim}×{in_dim}]:");
    println!("    Latency: {per_call_us:.1} µs ({n_iters} iters)");
    println!("    Bandwidth: {bw_gbs:.1} GB/s (peak=288 GB/s → {:.1}% util)",
             bw_gbs / 288.0 * 100.0);
    println!("    CosSim: {cos:.6}  MaxAbsErr: {max_err:.6}");

    // ── 4. Benchmark all 3 MLP matrices ─────────────────────────────────
    println!("\n═══ Phase 2: All MLP projections (layer 0) ═══\n");

    struct MatResult {
        name: String,
        shape: [usize; 2],
        latency_us: f64,
        bw_gbs: f64,
        cos: f32,
    }
    let mut results = Vec::new();

    for proj_name in &mlp_names {
        let tensor_name = format!("model.layers.0.mlp.{proj_name}.weight");
        let (w, s) = load_weight_f32(&mmap, &header, &tensor_name).unwrap();
        let (od, id) = (s[0], s[1]);
        let (t, sc) = quantize_ternary(&w);
        let pk = pack_ternary_hip(&t, od, id);

        let x_h: Vec<f32> = (0..id).map(|i| ((i as f32 * 0.01).sin() * 0.5)).collect();
        let y_ref = ternary_matmul_cpu(&t, &x_h, sc, od, id);

        let dp = hip_ffi::DeviceBuffer::<u8>::from_slice(&pk).unwrap();
        let dx = hip_ffi::DeviceBuffer::<f32>::from_slice(&x_h).unwrap();
        let mut dy = hip_ffi::DeviceBuffer::<f32>::alloc(od).unwrap();

        // Warmup
        for _ in 0..5 {
            unsafe { kernels.gemv_ternary_f32(dp.as_ptr(), dx.as_ptr(), dy.as_mut_ptr(),
                1, od as i32, id as i32, sc, 0.0).unwrap(); }
        }
        hip_ffi::hip_stream_synchronize(stream).unwrap();

        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe { kernels.gemv_ternary_f32(dp.as_ptr(), dx.as_ptr(), dy.as_mut_ptr(),
                1, od as i32, id as i32, sc, 0.0).unwrap(); }
        }
        hip_ffi::hip_stream_synchronize(stream).unwrap();
        let elapsed = t0.elapsed();
        let us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

        let y_g = dy.to_host().unwrap();
        let c = cosine_sim(&y_ref, &y_g);
        let bw = pk.len() as f64 / (us * 1e-6) / 1e9;

        println!("  {proj_name:10} [{od:5}×{id:4}]: {us:6.1} µs  {bw:5.1} GB/s  cos={c:.6}");
        results.push(MatResult { name: proj_name.to_string(), shape: [od, id], latency_us: us, bw_gbs: bw, cos: c });
    }

    let total_mlp_us: f64 = results.iter().map(|r| r.latency_us).sum();
    println!("\n  Total MLP (3 matrices): {total_mlp_us:.1} µs");

    // ── 5. Full 28-layer (or 14-layer with skip) simulation ─────────────
    println!("\n═══ Phase 3: Full inference simulation ═══\n");

    // Pre-load and pack all 28 layers' MLP weights
    struct LayerMLP {
        gate_packed: hip_ffi::DeviceBuffer<u8>,
        up_packed: hip_ffi::DeviceBuffer<u8>,
        down_packed: hip_ffi::DeviceBuffer<u8>,
        gate_scale: f32,
        up_scale: f32,
        down_scale: f32,
        gate_shape: [usize; 2],
        up_shape: [usize; 2],
        down_shape: [usize; 2],
    }

    println!("  Loading all {layer_count} layers...");
    let load_t = Instant::now();
    let mut layers: Vec<LayerMLP> = Vec::with_capacity(layer_count);

    for l in 0..layer_count {
        let mut packed_bufs = Vec::new();
        let mut scales = Vec::new();
        let mut shapes = Vec::new();

        for proj in &mlp_names {
            let name = format!("model.layers.{l}.mlp.{proj}.weight");
            let (w, s) = load_weight_f32(&mmap, &header, &name)
                .unwrap_or_else(|| panic!("Missing {name}"));
            let (t, sc) = quantize_ternary(&w);
            let pk = pack_ternary_hip(&t, s[0], s[1]);
            let db = hip_ffi::DeviceBuffer::<u8>::from_slice(&pk).unwrap();
            packed_bufs.push(db);
            scales.push(sc);
            shapes.push(s);
        }

        layers.push(LayerMLP {
            gate_packed: packed_bufs.remove(0),
            up_packed: packed_bufs.remove(0),
            down_packed: packed_bufs.remove(0),
            gate_scale: scales[0],
            up_scale: scales[1],
            down_scale: scales[2],
            gate_shape: shapes[0],
            up_shape: shapes[1],
            down_shape: shapes[2],
        });
    }
    println!("  ✓ All layers loaded in {:.1}s", load_t.elapsed().as_secs_f64());

    // GPU buffers for intermediate activations
    let hidden = 1536;
    let intermediate = 8960;
    let d_hidden = hip_ffi::DeviceBuffer::<f32>::from_slice(&x_host[..hidden]).unwrap();
    let mut d_gate = hip_ffi::DeviceBuffer::<f32>::alloc(intermediate).unwrap();
    let mut d_up = hip_ffi::DeviceBuffer::<f32>::alloc(intermediate).unwrap();
    let mut d_down = hip_ffi::DeviceBuffer::<f32>::alloc(hidden).unwrap();

    // ── 5a. All 28 layers ───────────────────────────────────────────────
    // Warmup
    for l in 0..layer_count {
        let ly = &layers[l];
        unsafe {
            kernels.gemv_ternary_f32(ly.gate_packed.as_ptr(), d_hidden.as_ptr(), d_gate.as_mut_ptr(),
                1, ly.gate_shape[0] as i32, ly.gate_shape[1] as i32, ly.gate_scale, 0.0).unwrap();
            kernels.gemv_ternary_f32(ly.up_packed.as_ptr(), d_hidden.as_ptr(), d_up.as_mut_ptr(),
                1, ly.up_shape[0] as i32, ly.up_shape[1] as i32, ly.up_scale, 0.0).unwrap();
            // SiLU(gate) * up would happen here — skip for pure GEMV bench
            kernels.gemv_ternary_f32(ly.down_packed.as_ptr(), d_gate.as_ptr(), d_down.as_mut_ptr(),
                1, ly.down_shape[0] as i32, ly.down_shape[1] as i32, ly.down_scale, 0.0).unwrap();
        }
    }
    hip_ffi::hip_stream_synchronize(stream).unwrap();

    let n_passes = 20;
    hip_ffi::hip_stream_synchronize(stream).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_passes {
        for l in 0..layer_count {
            let ly = &layers[l];
            unsafe {
                kernels.gemv_ternary_f32(ly.gate_packed.as_ptr(), d_hidden.as_ptr(), d_gate.as_mut_ptr(),
                    1, ly.gate_shape[0] as i32, ly.gate_shape[1] as i32, ly.gate_scale, 0.0).unwrap();
                kernels.gemv_ternary_f32(ly.up_packed.as_ptr(), d_hidden.as_ptr(), d_up.as_mut_ptr(),
                    1, ly.up_shape[0] as i32, ly.up_shape[1] as i32, ly.up_scale, 0.0).unwrap();
                kernels.gemv_ternary_f32(ly.down_packed.as_ptr(), d_gate.as_ptr(), d_down.as_mut_ptr(),
                    1, ly.down_shape[0] as i32, ly.down_shape[1] as i32, ly.down_scale, 0.0).unwrap();
            }
        }
    }
    hip_ffi::hip_stream_synchronize(stream).unwrap();
    let total_28 = t0.elapsed();
    let per_pass_28_us = total_28.as_secs_f64() * 1e6 / n_passes as f64;
    let per_layer_28_us = per_pass_28_us / layer_count as f64;
    let tok_s_28_mlp = 1e6 / per_pass_28_us;

    println!("  28-layer MLP-only (3 GEMV/layer):");
    println!("    Total: {per_pass_28_us:.0} µs/pass  ({per_layer_28_us:.0} µs/layer)");
    println!("    MLP-only tok/s: {tok_s_28_mlp:.0}");

    // ── 5b. 14 layers (BI skip: layers 7-20 have BI≈0) ─────────────────
    let active_layers: Vec<usize> = (0..28).filter(|&l| l < 7 || l > 20).collect();
    let n_active = active_layers.len();

    // Warmup
    for &l in &active_layers {
        let ly = &layers[l];
        unsafe {
            kernels.gemv_ternary_f32(ly.gate_packed.as_ptr(), d_hidden.as_ptr(), d_gate.as_mut_ptr(),
                1, ly.gate_shape[0] as i32, ly.gate_shape[1] as i32, ly.gate_scale, 0.0).unwrap();
            kernels.gemv_ternary_f32(ly.up_packed.as_ptr(), d_hidden.as_ptr(), d_up.as_mut_ptr(),
                1, ly.up_shape[0] as i32, ly.up_shape[1] as i32, ly.up_scale, 0.0).unwrap();
            kernels.gemv_ternary_f32(ly.down_packed.as_ptr(), d_gate.as_ptr(), d_down.as_mut_ptr(),
                1, ly.down_shape[0] as i32, ly.down_shape[1] as i32, ly.down_scale, 0.0).unwrap();
        }
    }
    hip_ffi::hip_stream_synchronize(stream).unwrap();

    hip_ffi::hip_stream_synchronize(stream).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_passes {
        for &l in &active_layers {
            let ly = &layers[l];
            unsafe {
                kernels.gemv_ternary_f32(ly.gate_packed.as_ptr(), d_hidden.as_ptr(), d_gate.as_mut_ptr(),
                    1, ly.gate_shape[0] as i32, ly.gate_shape[1] as i32, ly.gate_scale, 0.0).unwrap();
                kernels.gemv_ternary_f32(ly.up_packed.as_ptr(), d_hidden.as_ptr(), d_up.as_mut_ptr(),
                    1, ly.up_shape[0] as i32, ly.up_shape[1] as i32, ly.up_scale, 0.0).unwrap();
                kernels.gemv_ternary_f32(ly.down_packed.as_ptr(), d_gate.as_ptr(), d_down.as_mut_ptr(),
                    1, ly.down_shape[0] as i32, ly.down_shape[1] as i32, ly.down_scale, 0.0).unwrap();
            }
        }
    }
    hip_ffi::hip_stream_synchronize(stream).unwrap();
    let total_14 = t0.elapsed();
    let per_pass_14_us = total_14.as_secs_f64() * 1e6 / n_passes as f64;
    let per_layer_14_us = per_pass_14_us / n_active as f64;
    let tok_s_14_mlp = 1e6 / per_pass_14_us;

    println!("\n  {n_active}-layer MLP-only (BI skip layers 7-20):");
    println!("    Total: {per_pass_14_us:.0} µs/pass  ({per_layer_14_us:.0} µs/layer)");
    println!("    MLP-only tok/s: {tok_s_14_mlp:.0}");
    println!("    Speedup vs 28: {:.2}×", per_pass_28_us / per_pass_14_us);

    // ── 6. Estimate full decode token budget ────────────────────────────
    println!("\n═══ Phase 4: Token budget estimate ═══\n");

    // Full decode = attention + MLP per layer.
    // Attention GEMV: Q/K/V projections [1536→1536] + output proj [1536→1536]
    // That's 4 × [1536,1536] ternary GEMV per layer.
    // Attention softmax + RoPE + KV cache are lightweight in comparison.

    // Benchmark [1536,1536] GEMV (attention projections)
    let attn_name = "model.layers.0.self_attn.q_proj.weight";
    if let Some((w, s)) = load_weight_f32(&mmap, &header, attn_name) {
        let (od, id) = (s[0], s[1]);
        let (t, sc) = quantize_ternary(&w);
        let pk = pack_ternary_hip(&t, od, id);
        let dp = hip_ffi::DeviceBuffer::<u8>::from_slice(&pk).unwrap();
        let x_h: Vec<f32> = (0..id).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let dx = hip_ffi::DeviceBuffer::<f32>::from_slice(&x_h).unwrap();
        let mut dy = hip_ffi::DeviceBuffer::<f32>::alloc(od).unwrap();

        for _ in 0..5 {
            unsafe { kernels.gemv_ternary_f32(dp.as_ptr(), dx.as_ptr(), dy.as_mut_ptr(),
                1, od as i32, id as i32, sc, 0.0).unwrap(); }
        }
        hip_ffi::hip_stream_synchronize(stream).unwrap();

        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe { kernels.gemv_ternary_f32(dp.as_ptr(), dx.as_ptr(), dy.as_mut_ptr(),
                1, od as i32, id as i32, sc, 0.0).unwrap(); }
        }
        hip_ffi::hip_stream_synchronize(stream).unwrap();
        let attn_proj_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

        // Per-layer cost: 4 attn proj + 3 MLP proj + norm + rope + softmax (est ~5µs)
        let attn_per_layer_us = attn_proj_us * 4.0;
        let overhead_per_layer_us = 5.0; // RMSNorm + RoPE + softmax estimate

        let full_28_us = (total_mlp_us + attn_per_layer_us + overhead_per_layer_us) * 28.0;
        let full_14_us = (total_mlp_us + attn_per_layer_us + overhead_per_layer_us) * n_active as f64;

        // lm_head: [vocab_size=151936, hidden=1536] — huge but runs once per token
        let lm_head_est_us = attn_proj_us * (151936.0 / od as f64); // scale by size ratio

        println!("  Attention projection [{od}×{id}]: {attn_proj_us:.1} µs");
        println!("  Per-layer budget:");
        println!("    4× attn proj: {attn_per_layer_us:.1} µs");
        println!("    3× MLP proj:  {total_mlp_us:.1} µs");
        println!("    overhead:     ~{overhead_per_layer_us:.0} µs");
        println!("    total/layer:  {:.1} µs", attn_per_layer_us + total_mlp_us + overhead_per_layer_us);

        let total_28_est = (attn_per_layer_us + total_mlp_us + overhead_per_layer_us) * 28.0 + lm_head_est_us;
        let total_14_est = (attn_per_layer_us + total_mlp_us + overhead_per_layer_us) * n_active as f64 + lm_head_est_us;

        println!("\n  Estimated full decode token:");
        println!("    28 layers: {total_28_est:.0} µs → {:.0} tok/s", 1e6 / total_28_est);
        println!("    {n_active} layers (BI skip): {total_14_est:.0} µs → {:.0} tok/s", 1e6 / total_14_est);
        println!("    lm_head est: {lm_head_est_us:.0} µs");

        // What we need for 300 tok/s
        let target_budget_us = 1e6 / 300.0;
        println!("\n  Target 300 tok/s = {target_budget_us:.0} µs budget");
        println!("    28-layer gap: {:.0} µs over budget", total_28_est - target_budget_us);
        if total_14_est < target_budget_us {
            println!("    {n_active}-layer: ✓ {:.0} µs UNDER budget!", target_budget_us - total_14_est);
        } else {
            println!("    {n_active}-layer gap: {:.0} µs over budget", total_14_est - target_budget_us);
        }
    }

    // ── 7. Per-layer breakdown ──────────────────────────────────────────
    println!("\n═══ Phase 5: Per-layer GEMV breakdown ═══\n");
    println!("  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}", "Layer", "gate(µs)", "up(µs)", "down(µs)", "total(µs)");
    println!("  {}", "─".repeat(55));

    for l in 0..layer_count.min(6) {
        let ly = &layers[l];
        let mut times = [0.0f64; 3];

        for (i, (packed, shape, sc)) in [
            (&ly.gate_packed, ly.gate_shape, ly.gate_scale),
            (&ly.up_packed, ly.up_shape, ly.up_scale),
            (&ly.down_packed, ly.down_shape, ly.down_scale),
        ].iter().enumerate() {
            // Use correct input: gate/up use hidden, down uses intermediate
            let input_ptr = if i < 2 { d_hidden.as_ptr() } else { d_gate.as_ptr() };
            let out_ptr = if i < 2 {
                if i == 0 { d_gate.as_mut_ptr() } else { d_up.as_mut_ptr() }
            } else {
                d_down.as_mut_ptr()
            };

            for _ in 0..3 {
                unsafe { kernels.gemv_ternary_f32(packed.as_ptr(), input_ptr, out_ptr,
                    1, shape[0] as i32, shape[1] as i32, *sc, 0.0).unwrap(); }
            }
            hip_ffi::hip_stream_synchronize(stream).unwrap();

            let t0 = Instant::now();
            for _ in 0..50 {
                unsafe { kernels.gemv_ternary_f32(packed.as_ptr(), input_ptr, out_ptr,
                    1, shape[0] as i32, shape[1] as i32, *sc, 0.0).unwrap(); }
            }
            hip_ffi::hip_stream_synchronize(stream).unwrap();
            times[i] = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
        }

        let total = times[0] + times[1] + times[2];
        let marker = if l >= 7 && l <= 20 { " (skip)" } else { "" };
        println!("  L{l:02}{marker:7}  {:.1}  {:>10.1}  {:>10.1}  {:>10.1}", times[0], times[1], times[2], total);
    }
    println!("  ...");

    println!("\n═══ Summary ═══\n");
    println!("  HIP ternary GEMV: raw kernel performance measured.");
    println!("  Next: integrate into iara-engine forward pass with layer skip.");
    println!("  Target: 300 tok/s on Qwen2.5-1.5B with ternary weights + BI skip.\n");
}
