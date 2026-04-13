//! Quick test: does burn-wgpu actually use the AMD GPU?
use burn::prelude::*;
use burn::backend::Wgpu;

type B = Wgpu;

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    println!("Device: {:?}", device);

    // Simple GEMM test
    let a: Tensor<B, 2> = Tensor::random([1024, 1024], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let b: Tensor<B, 2> = Tensor::random([1024, 1024], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

    let start = std::time::Instant::now();
    let mut c = a.clone().matmul(b.clone());
    for _ in 0..100 {
        c = c.matmul(b.clone());
    }
    // Force sync by reading a value
    let val = c.slice([0..1, 0..1]).into_data();
    let elapsed = start.elapsed();
    println!("100x matmul 1024x1024: {:.1}ms", elapsed.as_millis());
    println!("First element: {:?}", val);

    // Test larger GEMM (similar to MLP forward)
    let x: Tensor<B, 2> = Tensor::random([512, 1536], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let w: Tensor<B, 2> = Tensor::random([8960, 1536], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

    let start2 = std::time::Instant::now();
    let _y = x.matmul(w.transpose());
    let elapsed2 = start2.elapsed();
    println!("MLP-sized GEMM [512,1536]x[1536,8960]: {:.1}ms", elapsed2.as_millis());

    println!("\nGPU test passed!");
}
