// ============================================================================
// MetaVerse Export Binary: Generate JSON frames for frontend visualization
// ============================================================================
// Runs the simulation for 5000 ticks and exports simulation frames as JSON
// that a frontend can consume for visualization and analysis.
// ============================================================================

use metaverse_core::simulation::{Simulation, SimulationConfig};
use serde::{Serialize, Deserialize};
use std::io::Write;

/// Minimal organism representation for export
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportOrganism {
    id: u64,
    x: f32,
    y: f32,
    energy: f32,
    is_producer: bool,
    generation: u32,
    alive: bool,
}

/// World statistics for export
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportWorldStats {
    total_pop: usize,
    producers: usize,
    consumers: usize,
    births: usize,
    deaths: usize,
    avg_energy: f32,
    shannon_diversity: f32,
}

/// Single frame snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportFrame {
    tick: u64,
    light_intensity: f32,
    season: String,
    organisms: Vec<ExportOrganism>,
    stats: ExportWorldStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    topography: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nutrient_map: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    water_map: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    toxin_map: Option<Vec<f32>>,
}

/// Root export structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportData {
    width: usize,
    height: usize,
    topography: Vec<f32>,
    frames: Vec<ExportFrame>,
}

fn season_name(season: &metaverse_core::world::Season) -> String {
    match season {
        metaverse_core::world::Season::Spring => "Spring".to_string(),
        metaverse_core::world::Season::Summer => "Summer".to_string(),
        metaverse_core::world::Season::Autumn => "Autumn".to_string(),
        metaverse_core::world::Season::Winter => "Winter".to_string(),
    }
}

fn compute_shannon_diversity(organisms: &[ExportOrganism], total_pop: usize) -> f32 {
    if total_pop == 0 {
        return 0.0;
    }

    let producers = organisms.iter().filter(|o| o.is_producer).count() as f32;
    let consumers = organisms.iter().filter(|o| !o.is_producer).count() as f32;
    let total = total_pop as f32;

    let mut h = 0.0;
    if producers > 0.0 {
        let p = producers / total;
        h -= p * p.ln();
    }
    if consumers > 0.0 {
        let p = consumers / total;
        h -= p * p.ln();
    }

    h
}

fn extract_resource_map(world: &metaverse_core::world::World, resource_type: &str) -> Vec<f32> {
    world
        .cells
        .iter()
        .map(|cell| match resource_type {
            "nutrient" => cell.nutrient_a + cell.nutrient_b + cell.nutrient_c,
            "water" => cell.water,
            "toxin" => cell.toxin,
            _ => 0.0,
        })
        .collect()
}

fn main() {
    // Parse command-line arguments for number of ticks and seed
    let args: Vec<String> = std::env::args().collect();
    let num_ticks: u64 = if args.len() > 1 {
        args[1].parse().unwrap_or(5000)
    } else {
        5000
    };

    let seed: u64 = if args.len() > 2 {
        args[2].parse().unwrap_or(42)
    } else {
        42
    };

    eprintln!("Starting MetaVerse export: {} ticks with seed {}", num_ticks, seed);

    // Create simulation with 128x128 world
    let config = SimulationConfig {
        world_width: 128,
        world_height: 128,
        initial_producers: 200,
        initial_consumers: 50,
        max_organisms: 2000,
        seed,
        catastrophe_probability: 0.0001,
        ticks_per_day: 100,
        days_per_season: 25,
    };

    let mut sim = Simulation::new(config);
    let mut frames = Vec::new();

    // Extract initial topography (only exported once at frame 0)
    let initial_topography = sim.world.topography.clone();
    let width = sim.world.config.width;
    let height = sim.world.config.height;

    // Run simulation for N ticks, capturing frames every tick
    for tick_num in 0..num_ticks {
        let tick_report = sim.tick();

        // Prepare frame data
        let organisms: Vec<ExportOrganism> = sim
            .organisms
            .iter()
            .map(|org| ExportOrganism {
                id: org.id,
                x: org.position.x,
                y: org.position.y,
                energy: org.energy,
                is_producer: org.is_producer,
                generation: org.generation,
                alive: org.alive,
            })
            .collect();

        let total_pop = organisms.len();
        let producers = organisms.iter().filter(|o| o.is_producer).count();
        let consumers = organisms.iter().filter(|o| !o.is_producer).count();
        let shannon_diversity = compute_shannon_diversity(&organisms, total_pop);

        let stats = ExportWorldStats {
            total_pop,
            producers,
            consumers,
            births: tick_report.births,
            deaths: tick_report.deaths,
            avg_energy: tick_report.avg_energy,
            shannon_diversity,
        };

        // Topography only on frame 0
        let topography = if tick_num == 0 {
            Some(initial_topography.clone())
        } else {
            None
        };

        // Resource maps every 50 ticks
        let (nutrient_map, water_map, toxin_map) = if tick_num % 50 == 0 {
            (
                Some(extract_resource_map(&sim.world, "nutrient")),
                Some(extract_resource_map(&sim.world, "water")),
                Some(extract_resource_map(&sim.world, "toxin")),
            )
        } else {
            (None, None, None)
        };

        let frame = ExportFrame {
            tick: tick_num,
            light_intensity: sim.world.current_light(),
            season: season_name(&sim.world.season),
            organisms,
            stats,
            topography,
            nutrient_map,
            water_map,
            toxin_map,
        };

        frames.push(frame);

        // Status update every 500 ticks
        if (tick_num + 1) % 500 == 0 {
            eprintln!("Processed {} ticks...", tick_num + 1);
        }
    }

    // Create final export data
    let export = ExportData {
        width,
        height,
        topography: initial_topography,
        frames,
    };

    // Serialize to JSON (compact for resource maps)
    let json = serde_json::to_string(&export).expect("Failed to serialize to JSON");

    // Write to stdout
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(json.as_bytes()).expect("Failed to write to stdout");
    handle.write_all(b"\n").expect("Failed to write newline");

    eprintln!("Export complete: {} frames written to stdout", num_ticks);
}
