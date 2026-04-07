fn to_phases(hidden: &[f32]) -> Vec<f32> {
    hidden
        .chunks(2)
        .map(|pair| {
            if pair.len() == 2 {
                pair[1].atan2(pair[0])
            } else {
                pair[0].atan2(0.0)
            }
        })
        .collect()
}

fn main() {
    // Test odd length
    let hidden = vec![1.0, 0.0, 0.5];
    let phases = to_phases(&hidden);
    println!("Odd length test: phases = {:?}", phases);
    println!("Phase[1] = {}, expected 0.0", phases[1]);
    
    // Test orthogonal
    let zero_phase = 0.0_f32;
    let delta = zero_phase - std::f32::consts::PI / 2.0;
    println!("\nOrthogonal phase delta: {}", delta);
    println!("cos(delta) = {}, sin(delta) = {}", delta.cos(), delta.sin());
    println!("sqrt(cos^2 + sin^2) = {}", (delta.cos().powi(2) + delta.sin().powi(2)).sqrt());
}
