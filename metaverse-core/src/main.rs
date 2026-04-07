use metaverse_core::simulation::{Simulation, SimulationConfig};

fn main() {
    env_logger::init();
    
    let config = SimulationConfig::default();
    println!("World {}x{}", config.world_width, config.world_height);
    println!("Producers: {}, Consumers: {}", config.initial_producers, config.initial_consumers);
    
    let mut sim = Simulation::new(config);
    
    // Run simulation for 10000 ticks, print stats every 100
    for _epoch in 0..100 {
        let reports = sim.run(100);
        let last = reports.last().unwrap();
        let stats = sim.current_stats();
        println!(
            "Tick {:>6} | Alive: {:>4} | Births: {:>3} | Deaths: {:>3} | Producers: {:>3} | Consumers: {:>3} | Diversity: {:.2} | AvgGen: {:.1}",
            last.tick, last.alive_count, 
            reports.iter().map(|r| r.births).sum::<usize>(),
            reports.iter().map(|r| r.deaths).sum::<usize>(),
            stats.producers, stats.consumers,
            stats.shannon_diversity, stats.avg_generation,
        );
    }
    
    println!("\nSimulation complete at tick {}", sim.tick);
    let final_stats = sim.current_stats();
    println!("Final population: {}", final_stats.total_organisms);
    println!("Species: {}", final_stats.species_count);
    println!("Max generation: {:.0}", final_stats.max_generation);
    println!("Shannon diversity: {:.3}", final_stats.shannon_diversity);
}
