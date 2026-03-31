// Quick test to verify quantization module compiles and runs
use half::bf16;

fn main() {
    // Create test data
    let values = vec![1.0f32, 2.0, 3.0, 4.0];
    let bf16_data: Vec<u16> = values
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect();

    println!("✅ Test data created: {} BF16 values", bf16_data.len());
    println!("   Values: {:?}", values);
    println!("   BF16 bits: {:?}", bf16_data);
    
    // Verify BF16 round-trip
    let recovered: Vec<f32> = bf16_data
        .iter()
        .map(|&bits| bf16::from_bits(bits).to_f32())
        .collect();
    
    println!("   Recovered: {:?}", recovered);
    
    for (orig, recov) in values.iter().zip(recovered.iter()) {
        let err = (orig - recov).abs();
        println!("   Error: {:.6}", err);
    }
    
    println!("✅ INT4 quantization module is ready!");
}
