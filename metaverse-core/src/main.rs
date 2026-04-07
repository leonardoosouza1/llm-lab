use metaverse_core::simulation::{Simulation, SimulationConfig};

fn main() {
    env_logger::init();

    let config = SimulationConfig::default();
    println!("=== MetaVerse: Mundo Metafísico para Organismos Digitais ===");
    println!("World {}x{} | Producers: {} | Consumers: {}",
        config.world_width, config.world_height,
        config.initial_producers, config.initial_consumers);
    println!("{:-<90}", "");

    let mut sim = Simulation::new(config);
    let mut peak_population = 0usize;
    let mut total_births = 0usize;
    let mut total_deaths = 0usize;
    let mut max_gen_seen = 0u32;

    for _epoch in 0..50 {
        let reports = sim.run(100);
        let last = reports.last().unwrap();
        let stats = sim.current_stats();

        let epoch_births: usize = reports.iter().map(|r| r.births).sum();
        let epoch_deaths: usize = reports.iter().map(|r| r.deaths).sum();
        total_births += epoch_births;
        total_deaths += epoch_deaths;

        if last.alive_count > peak_population {
            peak_population = last.alive_count;
        }
        if stats.max_generation > max_gen_seen {
            max_gen_seen = stats.max_generation;
        }

        println!(
            "Tick {:>5} | Pop: {:>4} | +{:<3} -{:<3} | P:{:>3} C:{:>3} | H={:.2} | Gen={:.1} | E={:.1}",
            last.tick, last.alive_count,
            epoch_births, epoch_deaths,
            stats.producers, stats.consumers,
            stats.shannon_diversity, stats.avg_generation,
            last.avg_energy,
        );

        // Early exit if ecosystem collapses and stays dead for 500 ticks
        if last.alive_count == 0 && sim.tick > 500 {
            println!("\n--- Ecosystem collapsed at tick {} ---", sim.tick);
            break;
        }
    }

    println!("{:-<90}", "");
    println!("=== Simulation Summary ===");
    println!("Duration: {} ticks", sim.tick);
    println!("Peak population: {}", peak_population);
    println!("Total births: {} | Total deaths: {}", total_births, total_deaths);
    println!("Max generation reached: {}", max_gen_seen);

    let final_stats = sim.current_stats();
    println!("Final population: {} (P:{} C:{})",
        final_stats.total_organisms, final_stats.producers, final_stats.consumers);
    println!("Shannon diversity: {:.3}", final_stats.shannon_diversity);
}
