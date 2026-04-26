//! Smoke test: load llm_kernels.hsaco via HSA, dispatch `embed_lookup` directly,
//! compare timing vs HIP path.
//!
//! Run: cargo run --release -p hip-ffi --example hsa_smoke

use hip_ffi::hsa_dispatch::{HsaContext, KernelInfo};
use std::time::Instant;

const KERNEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/llm_kernels.hsaco"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("HSA AQL smoke test — loading {} bytes of hsaco", KERNEL_BYTES.len());

    let ctx = HsaContext::init()?;
    println!("agent ok queue ok kernarg_pool ok");

    let module = ctx.load_module(KERNEL_BYTES)?;
    println!("module loaded");

    // Pick a tiny kernel for smoke. `f32_to_bf16` has signature (src, dst, n).
    // It's small + cheap so dispatch overhead dominates measurement.
    let k_info = ctx.get_kernel(&module, "f32_to_bf16")?;
    println!("kernel f32_to_bf16: kernel_object=0x{:x} kernarg_size={} group={} private={}",
        k_info.kernel_object, k_info.kernarg_size, k_info.group_segment_size, k_info.private_segment_size);

    // Allocate a tiny 1-element src/dst (we only care about dispatch latency, not result).
    use hip_ffi::raw::{hipMalloc, hipFree};
    let mut src: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut dst: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        hipMalloc(&mut src, 4);
        hipMalloc(&mut dst, 2);
    }

    // kernarg layout: ptr, ptr, i32 = 8+8+4 = 20 bytes (rounded to 16-align = 24).
    let karg = ctx.alloc_kernarg(k_info.kernarg_size as usize)?;
    unsafe {
        // Zero whole buffer (hidden args follow explicit args; many must be 0/null).
        std::ptr::write_bytes(karg, 0, k_info.kernarg_size as usize);
        let p = karg as *mut u64;
        *p.offset(0) = src as u64;
        *p.offset(1) = dst as u64;
        let np = karg.add(16) as *mut i32;
        *np = 1;
    }

    // Single dispatch warmup.
    ctx.dispatch_sync(&k_info, karg, (1, 1, 1), (1, 1, 1), 0)?;

    let n = 10_000usize;
    let t0 = Instant::now();
    for _ in 0..n {
        ctx.dispatch_sync(&k_info, karg, (1, 1, 1), (1, 1, 1), 0)?;
    }
    let elapsed = t0.elapsed();
    let per_us = elapsed.as_nanos() as f64 / n as f64 / 1000.0;
    println!("HSA sync dispatch: {} runs in {:?} = {:.2}µs/launch", n, elapsed, per_us);

    // ASYNC batch: submit 1000 in one go, barrier, sync once. Measures pure submit overhead.
    let batch = 1000usize;
    let t0 = Instant::now();
    for _ in 0..n / batch {
        for _ in 0..batch {
            ctx.dispatch_async(&k_info, karg, (1, 1, 1), (1, 1, 1), 0)?;
        }
        ctx.submit_barrier()?;
        ctx.wait_completion();
    }
    let elapsed = t0.elapsed();
    let per_us = elapsed.as_nanos() as f64 / n as f64 / 1000.0;
    println!("HSA async batch={}: {} runs in {:?} = {:.2}µs/launch", batch, n, elapsed, per_us);

    // Compare HIP path. Need to set up GpuKernels.
    {
        use hip_ffi::raw::*;
        let mut module_h = std::ptr::null_mut();
        unsafe {
            hipModuleLoadData(&mut module_h, KERNEL_BYTES.as_ptr() as *const _);
        }
        let cn = std::ffi::CString::new("f32_to_bf16").unwrap();
        let mut func_h: hipFunction_t = std::ptr::null_mut();
        unsafe {
            hipModuleGetFunction(&mut func_h, module_h, cn.as_ptr());
        }

        let mut sp = src;
        let mut dp = dst;
        let mut np = 1i32;
        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut sp as *mut _ as *mut _,
            &mut dp as *mut _ as *mut _,
            &mut np as *mut _ as *mut _,
        ];

        // Warmup
        unsafe {
            hipModuleLaunchKernel(func_h, 1, 1, 1, 1, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut());
            hipDeviceSynchronize();
        }

        let t0 = Instant::now();
        for _ in 0..n {
            unsafe {
                hipModuleLaunchKernel(func_h, 1, 1, 1, 1, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut());
                hipDeviceSynchronize();
            }
        }
        let elapsed = t0.elapsed();
        let per_us = elapsed.as_nanos() as f64 / n as f64 / 1000.0;
        println!("HIP dispatch: {} runs in {:?} = {:.2}µs/launch", n, elapsed, per_us);
    }

    unsafe {
        hipFree(src);
        hipFree(dst);
    }

    Ok(())
}
