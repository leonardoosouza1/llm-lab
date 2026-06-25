// ============================================================================
// MetaVerse Solver: Use physics universes to solve real problems
// ============================================================================
// Demonstrates: feeding LLM tensor data into themed universes and extracting
// solutions through evolutionary physics simulation.
//
// Usage: cargo run --release --bin solve -- <universe> [grid_size] [organisms] [ticks]
//   universe: fluid | photonic | crystal | plasma | hybrid
// ============================================================================

use metaverse_core::universes::*;
use rand::{Rng, SeedableRng};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let universe_type = args.get(1).map(|s| s.as_str()).unwrap_or("fluid");
    let grid_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_organisms: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500);
    let n_ticks: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    println!("=== MetaVerse Solver ===");
    println!("Universe: {} | Grid: {}x{} | Organisms: {} | Ticks: {}",
        universe_type, grid_size, grid_size, n_organisms, n_ticks);
    println!("{:-<70}", "");

    // Generate synthetic "model weights" as the problem to solve
    // In production, this would come from actual .safetensors / .iara files
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let problem_size = grid_size * grid_size;
    let problem: Vec<f32> = (0..problem_size)
        .map(|i| {
            // Synthetic weight matrix with structure:
            // - Sparse (70% near-zero)
            // - A few "hot" regions (important neurons)
            let x = (i % grid_size) as f32 / grid_size as f32;
            let y = (i / grid_size) as f32 / grid_size as f32;
            let base = (x * 6.28).sin() * (y * 9.42).cos() * 0.3;
            let hot = if (x - 0.3).abs() < 0.1 && (y - 0.7).abs() < 0.1 { 2.0 }
                else if (x - 0.8).abs() < 0.05 && (y - 0.2).abs() < 0.15 { 1.5 }
                else { 0.0 };
            let noise: f32 = rng.gen::<f32>() * 0.1 - 0.05;
            base + hot + noise
        })
        .collect();

    // Count "hot" neurons (|weight| > 0.5) — ground truth
    let hot_count = problem.iter().filter(|&&w| w.abs() > 0.5).count();
    println!("Problem: {} weights, {} hot neurons ({:.1}% sparsity)",
        problem_size, hot_count, (1.0 - hot_count as f64 / problem_size as f64) * 100.0);

    // Create universe and inject problem
    let mut universe: Box<dyn UniversePhysics> = match universe_type {
        "fluid" => Box::new(FluidUniverse::new(0.1)),
        "photonic" => Box::new(PhotonicUniverse::new()),
        "crystal" => Box::new(CrystalUniverse::new(2.0, 1.0)),
        "plasma" => Box::new(PlasmaUniverse::new()),
        "hybrid" => {
            let mut h = HybridUniverse::new();
            h.add_physics(Box::new(FluidUniverse::new(0.1)), 0.5);
            h.add_physics(Box::new(PlasmaUniverse::new()), 0.5);
            Box::new(h)
        }
        _ => {
            eprintln!("Unknown universe: {}. Options: fluid, photonic, crystal, plasma, hybrid", universe_type);
            std::process::exit(1);
        }
    };

    universe.inject_problem(&problem);

    // Spawn organisms uniformly across the grid
    let mut organisms: Vec<UniverseOrganism> = (0..n_organisms)
        .map(|i| {
            let x = rng.gen::<f32>() * grid_size as f32;
            let y = rng.gen::<f32>() * grid_size as f32;
            let mut org = UniverseOrganism::new(i as u64, x, y);
            org.energy = 1.0;
            org.charge = rng.gen::<f32>() * 2.0 - 1.0;
            org.spin = rng.gen::<f32>() * 2.0 - 1.0;
            org.wavelength = 0.5 + rng.gen::<f32>() * 2.0;
            org
        })
        .collect();

    let mut grid = UniverseGrid::new(grid_size, grid_size);

    // Run simulation
    let mut alive_history = Vec::new();
    let mut avg_fitness_history = Vec::new();

    for tick in 0..n_ticks {
        universe.tick(&mut organisms, &mut grid);

        let alive = organisms.iter().filter(|o| o.alive).count();
        let avg_fit: f32 = if alive > 0 {
            organisms.iter().filter(|o| o.alive).map(|o| o.fitness).sum::<f32>() / alive as f32
        } else { 0.0 };

        alive_history.push(alive);
        avg_fitness_history.push(avg_fit);

        if tick % 20 == 0 || tick == n_ticks - 1 {
            let avg_energy: f32 = if alive > 0 {
                organisms.iter().filter(|o| o.alive).map(|o| o.energy).sum::<f32>() / alive as f32
            } else { 0.0 };

            println!("t={:>4} | alive={:>4}/{} | fitness={:.4} | energy={:.3}",
                tick, alive, n_organisms, avg_fit, avg_energy);
        }
    }

    // Extract solution
    let solution = universe.extract_solution();
    println!("{:-<70}", "");
    println!("=== Solution Analysis ===");
    println!("Solution vector length: {}", solution.len());

    if !solution.is_empty() {
        let sol_mean: f32 = solution.iter().sum::<f32>() / solution.len() as f32;
        let sol_max: f32 = solution.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sol_min: f32 = solution.iter().cloned().fold(f32::INFINITY, f32::min);
        let sol_nonzero = solution.iter().filter(|&&v| v.abs() > 0.01).count();

        println!("Mean: {:.4} | Min: {:.4} | Max: {:.4}", sol_mean, sol_min, sol_max);
        println!("Non-zero elements: {} ({:.1}%)", sol_nonzero, sol_nonzero as f64 / solution.len() as f64 * 100.0);

        // Compare solution to ground truth "hot" regions
        if solution.len() == problem.len() {
            // Threshold solution to find "discovered hot neurons"
            let sol_threshold = sol_mean + (sol_max - sol_mean) * 0.5;
            let discovered_hot: Vec<usize> = solution.iter()
                .enumerate()
                .filter(|(_, &v)| v > sol_threshold)
                .map(|(i, _)| i)
                .collect();

            let true_hot: Vec<usize> = problem.iter()
                .enumerate()
                .filter(|(_, &w)| w.abs() > 0.5)
                .map(|(i, _)| i)
                .collect();

            let true_hot_set: std::collections::HashSet<_> = true_hot.iter().collect();
            let discovered_set: std::collections::HashSet<_> = discovered_hot.iter().collect();
            let overlap = true_hot_set.intersection(&discovered_set).count();

            let precision = if discovered_hot.is_empty() { 0.0 } else { overlap as f64 / discovered_hot.len() as f64 };
            let recall = if true_hot.is_empty() { 0.0 } else { overlap as f64 / true_hot.len() as f64 };
            let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

            println!("\n=== Hot Neuron Discovery ===");
            println!("Ground truth hot neurons: {}", true_hot.len());
            println!("Discovered hot neurons: {}", discovered_hot.len());
            println!("Overlap: {}", overlap);
            println!("Precision: {:.2}% | Recall: {:.2}% | F1: {:.4}", precision*100.0, recall*100.0, f1);
        }
    }

    // Show survivor map (where organisms ended up)
    let alive_orgs: Vec<_> = organisms.iter().filter(|o| o.alive).collect();
    println!("\n=== Survivor Map ===");
    println!("Alive: {} / {}", alive_orgs.len(), n_organisms);
    if !alive_orgs.is_empty() {
        let avg_x: f32 = alive_orgs.iter().map(|o| o.position.x).sum::<f32>() / alive_orgs.len() as f32;
        let avg_y: f32 = alive_orgs.iter().map(|o| o.position.y).sum::<f32>() / alive_orgs.len() as f32;
        println!("Centroid: ({:.1}, {:.1})", avg_x, avg_y);

        // Density map: which grid cells have most survivors?
        let mut density = vec![0u32; grid_size * grid_size];
        for o in &alive_orgs {
            let gx = (o.position.x as usize).min(grid_size - 1);
            let gy = (o.position.y as usize).min(grid_size - 1);
            density[gy * grid_size + gx] += 1;
        }
        let max_density = *density.iter().max().unwrap_or(&1);
        let hot_cells = density.iter().filter(|&&d| d > max_density / 3).count();
        println!("Hot cells (>33% max density): {} ({:.1}% of grid)",
            hot_cells, hot_cells as f64 / (grid_size * grid_size) as f64 * 100.0);
    }
}
