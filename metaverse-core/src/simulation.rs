// ============================================================================
// Simulation Module: Main Loop for MetaVerse
// ============================================================================
// The simulation orchestrates all subsystems: world physics, organism life cycles,
// metabolism, brain decision-making, reproduction, and evolution.
// ============================================================================

use crate::{
    chemistry::Metabolism,
    world::{World, WorldConfig},
    Vec2, OrganismId, Tick,
    fep::FreeEnergyMinimizer,
    dream::{DreamEngine, Experience},
};
use crate::ecology::{EcosystemStats, RedQueenTracker};
use rand::{SeedableRng, Rng};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Simulation configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub world_width: usize,
    pub world_height: usize,
    pub initial_producers: usize,     // starting autotrophs
    pub initial_consumers: usize,     // starting heterotrophs
    pub max_organisms: usize,         // population cap
    pub seed: u64,
    pub catastrophe_probability: f32, // per tick
    pub ticks_per_day: u32,
    pub days_per_season: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            world_width: 128,
            world_height: 128,
            initial_producers: 200,
            initial_consumers: 50,
            max_organisms: 2000,
            seed: 42,
            catastrophe_probability: 0.0001,
            ticks_per_day: 100,
            days_per_season: 25,
        }
    }
}

/// A simplified organism representation for the simulation
#[derive(Debug, Clone)]
pub struct SimulationOrganism {
    pub id: OrganismId,
    pub position: Vec2,
    pub energy: f32,
    pub generation: u32,
    pub alive: bool,
    pub is_producer: bool,           // autotroph vs heterotroph
    pub complexity: f32,
    pub metabolism: Metabolism,
    pub age: u32,
    pub fep: FreeEnergyMinimizer,
    pub dream: DreamEngine,
}

impl SimulationOrganism {
    pub fn new(id: OrganismId, position: Vec2, is_producer: bool, complexity: f32) -> Self {
        Self {
            id,
            position,
            energy: 100.0,
            generation: 0,
            alive: true,
            is_producer,
            complexity,
            metabolism: Metabolism::new(complexity),
            age: 0,
            fep: FreeEnergyMinimizer::new(6),
            dream: DreamEngine::new(0.5, 0.02),
        }
    }

    pub fn can_reproduce(&self) -> bool {
        self.alive && self.energy > 60.0
    }

    pub fn take_damage(&mut self, damage: f32) {
        self.energy = (self.energy - damage).max(0.0);
        if self.energy <= 0.0 {
            self.alive = false;
        }
    }
}

/// Report from a single simulation tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickReport {
    pub tick: Tick,
    pub alive_count: usize,
    pub births: usize,
    pub deaths: usize,
    pub avg_energy: f32,
    pub catastrophe: Option<String>,
}

/// Main simulation state
pub struct Simulation {
    pub config: SimulationConfig,
    pub world: World,
    pub organisms: Vec<SimulationOrganism>,
    pub stats_history: Vec<EcosystemStats>,
    pub red_queen: RedQueenTracker,
    next_organism_id: OrganismId,
    rng: rand_chacha::ChaCha8Rng,
    pub tick: Tick,
}

impl Simulation {
    /// Create a new simulation with initial organisms
    pub fn new(config: SimulationConfig) -> Self {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(config.seed);

        // Create world
        let world_config = WorldConfig {
            width: config.world_width,
            height: config.world_height,
            wrap_around: true,
            ticks_per_day: config.ticks_per_day,
            days_per_season: config.days_per_season,
            base_light_intensity: 20.0,
            base_temperature: 20.0,
            resource_diffusion_rates: Default::default(),
            pheromone_decay_rate: 0.02,
            catastrophe_probability: config.catastrophe_probability,
        };

        let world = World::new(world_config, config.seed);
        let mut organisms = Vec::new();
        let mut next_id = 1u64;

        // Spawn initial producers (autotrophs)
        for _ in 0..config.initial_producers {
            let x = rng.gen_range(0..config.world_width) as f32;
            let y = rng.gen_range(0..config.world_height) as f32;
            let pos = Vec2::new(x, y);

            let organism = SimulationOrganism::new(next_id, pos, true, 2.0);
            organisms.push(organism);
            next_id += 1;
        }

        // Spawn initial consumers (heterotrophs)
        for _ in 0..config.initial_consumers {
            let x = rng.gen_range(0..config.world_width) as f32;
            let y = rng.gen_range(0..config.world_height) as f32;
            let pos = Vec2::new(x, y);

            let mut organism = SimulationOrganism::new(next_id, pos, false, 3.0);
            organism.metabolism.has_aerobic = true;
            organisms.push(organism);
            next_id += 1;
        }

        let red_queen = RedQueenTracker::new(100);

        Self {
            config,
            world,
            organisms,
            stats_history: Vec::new(),
            red_queen,
            next_organism_id: next_id,
            rng,
            tick: 0,
        }
    }

    /// Run one simulation tick through all phases
    pub fn tick(&mut self) -> TickReport {
        let mut births = 0;
        let mut deaths = 0;

        // PHASE 1: WORLD UPDATE
        let _light_before = self.world.current_light();
        self.world.tick();
        let light_after = self.world.current_light();

        // PHASE 2-3: PERCEPTION & DECISION
        let mut organism_actions: HashMap<OrganismId, (Vec2, bool)> = HashMap::new();
        for org in self.organisms.iter_mut() {
            if !org.alive {
                continue;
            }

            // Create sensory input vector: [energy/100, light_after/20, is_producer_f32, x/W, y/H, age/5000]
            let is_producer_f32 = if org.is_producer { 1.0 } else { 0.0 };
            let sensory_input = vec![
                org.energy / 100.0,
                light_after / 20.0,
                is_producer_f32,
                org.position.x / self.config.world_width as f32,
                org.position.y / self.config.world_height as f32,
                org.age as f32 / 5000.0,
            ];

            // Update beliefs and compute free energy
            org.fep.update_beliefs(&sensory_input);
            let free_energy = org.fep.compute_free_energy(&sensory_input);

            let move_dir = if org.is_producer && light_after > 0.1 {
                Vec2::new(
                    (self.rng.gen::<f32>() - 0.5) * 2.0,
                    (self.rng.gen::<f32>() - 0.5) * 2.0,
                ).normalized()
            } else {
                Vec2::new(
                    (self.rng.gen::<f32>() - 0.5) * 2.0,
                    (self.rng.gen::<f32>() - 0.5) * 2.0,
                ).normalized()
            };

            // FEP influence: if free energy is high, organism explores more
            let speed = if org.is_producer { 0.3 } else { 0.5 };
            let speed_multiplier = if free_energy > 2.0 { 1.5 } else { 1.0 };
            let move_vec = move_dir * speed * speed_multiplier;
            let can_reproduce = org.can_reproduce();
            organism_actions.insert(org.id, (move_vec, can_reproduce));
        }

        // PHASE 4: ACTION (Movement)
        for org in self.organisms.iter_mut() {
            if !org.alive || org.dream.is_sleeping {
                continue;
            }
            if let Some((move_vec, _)) = organism_actions.get(&org.id) {
                org.position = org.position + *move_vec;

                // Wrap around world
                org.position.x = org.position.x % (self.config.world_width as f32);
                org.position.y = org.position.y % (self.config.world_height as f32);
                if org.position.x < 0.0 {
                    org.position.x += self.config.world_width as f32;
                }
                if org.position.y < 0.0 {
                    org.position.y += self.config.world_height as f32;
                }

                // Movement costs energy
                org.energy -= 0.1;
            }
        }

        // PHASE 5: METABOLISM
        let org_positions: Vec<(usize, usize)> = self.organisms
            .iter()
            .filter(|o| o.alive && !o.dream.is_sleeping)
            .map(|o| {
                let x = (o.position.x as usize) % self.config.world_width;
                let y = (o.position.y as usize) % self.config.world_height;
                (x, y)
            })
            .collect();

        let mut sleep_skip_map: Vec<bool> = vec![false; self.organisms.len()];
        for (i, org) in self.organisms.iter_mut().enumerate() {
            if !org.alive || org.dream.is_sleeping {
                sleep_skip_map[i] = true;
                continue;
            }

            if i < org_positions.len() {
                let (x, y) = org_positions[i];
                let cell = self.world.cell_at_mut(x, y);

                // Producers photosynthesis
                if org.is_producer {
                    org.metabolism.has_photosynthesis = true;
                }

                let report = org.metabolism.tick(cell);
                org.energy += report.energy_gained;
                org.energy -= report.energy_spent;
            }
        }

        // DREAM PHASE (between Metabolism and Interactions)
        for (i, org) in self.organisms.iter_mut().enumerate() {
            if !org.alive {
                continue;
            }

            // Record experience before sleep cycle
            if !sleep_skip_map.get(i).copied().unwrap_or(false) {
                // Sensory input for experience
                let is_producer_f32 = if org.is_producer { 1.0 } else { 0.0 };
                let sensory_input = vec![
                    org.energy / 100.0,
                    light_after / 20.0,
                    is_producer_f32,
                    org.position.x / self.config.world_width as f32,
                    org.position.y / self.config.world_height as f32,
                    org.age as f32 / 5000.0,
                ];

                // Get move direction from previous action (approximate)
                let action = if let Some((move_vec, _)) = organism_actions.get(&org.id) {
                    vec![move_vec.x, move_vec.y]
                } else {
                    vec![0.0, 0.0]
                };

                let reward = org.energy / 100.0; // Energy as reward signal
                let surprise = org.fep.get_surprise();

                let experience = Experience::new(
                    sensory_input,
                    action,
                    reward,
                    surprise,
                    self.tick,
                );
                org.dream.experience_buffer.record(experience);
            }

            // Sleep/wake cycle management
            if org.dream.is_sleeping {
                org.dream.tick_sleep(light_after);
            } else {
                org.dream.tick_wakefulness();
                if org.dream.should_sleep(light_after) {
                    org.dream.is_sleeping = true;
                }
            }
        }

        // PHASE 6-7: INTERACTIONS & LIFECYCLE
        let max_organisms = self.config.max_organisms;
        let current_len = self.organisms.len();
        let mut reproduction_candidates = Vec::new();

        // First pass: age organisms, check deaths, collect reproduction candidates
        for org in self.organisms.iter_mut() {
            if !org.alive {
                continue;
            }

            org.age += 1;

            // Starvation
            if org.energy <= 0.0 {
                org.alive = false;
                deaths += 1;
                continue;
            }

            // Senescence (age limit)
            if org.age > 5000 {
                org.alive = false;
                deaths += 1;
                continue;
            }

            // Collect reproduction candidates
            if org.can_reproduce() && current_len < max_organisms {
                reproduction_candidates.push((org.id, org.position, org.is_producer, org.complexity, org.generation));
            }
        }

        // Second pass: attempt reproduction
        let mut to_spawn = Vec::new();
        for (parent_id, parent_pos, is_producer, parent_complexity, parent_gen) in reproduction_candidates {
            let reproduction_prob = if is_producer { 0.05 } else { 0.02 };
            if self.rng.gen::<f32>() < reproduction_prob {
                // Find parent and modify it
                if let Some(parent) = self.organisms.iter_mut().find(|o| o.id == parent_id && o.alive) {
                    let mut offspring = SimulationOrganism::new(
                        self.next_organism_id,
                        parent_pos + Vec2::new(
                            (self.rng.gen::<f32>() - 0.5) * 2.0,
                            (self.rng.gen::<f32>() - 0.5) * 2.0,
                        ),
                        is_producer,
                        parent_complexity * (0.95 + self.rng.gen::<f32>() * 0.1),
                    );

                    offspring.generation = parent_gen + 1;
                    offspring.energy = 30.0;

                    // Parent invests energy
                    parent.energy -= 20.0;

                    to_spawn.push(offspring);
                    self.next_organism_id += 1;
                    births += 1;
                }
            }
        }

        self.organisms.extend(to_spawn);

        // PHASE 8: EVOLUTION
        let alive_organisms = self.organisms.iter().filter(|o| o.alive).count();

        // Record fitness for Red Queen
        let avg_fitness = self.organisms.iter()
            .filter(|o| o.alive)
            .map(|o| o.energy)
            .sum::<f32>() / (alive_organisms.max(1) as f32);

        self.red_queen.record(avg_fitness);

        // Compute ecosystem stats every 10 ticks
        if self.tick % 10 == 0 {
            let stats = self.current_stats();
            self.stats_history.push(stats);
        }

        // PHASE 9: CLEANUP
        self.organisms.retain(|o| o.alive);

        let avg_energy = if self.organisms.is_empty() {
            0.0
        } else {
            self.organisms.iter().map(|o| o.energy).sum::<f32>() / self.organisms.len() as f32
        };

        self.tick += 1;

        TickReport {
            tick: self.tick,
            alive_count: self.organisms.len(),
            births,
            deaths,
            avg_energy,
            catastrophe: None,
        }
    }

    /// Run N ticks and return all reports
    pub fn run(&mut self, n_ticks: u64) -> Vec<TickReport> {
        let mut reports = Vec::new();
        for _ in 0..n_ticks {
            let report = self.tick();
            reports.push(report);
        }
        reports
    }

    /// Get current ecosystem statistics
    pub fn current_stats(&self) -> EcosystemStats {
        let alive_organisms = self.organisms.iter().filter(|o| o.alive).count();
        let producers = self.organisms.iter().filter(|o| o.alive && o.is_producer).count();
        let consumers = self.organisms.iter().filter(|o| o.alive && !o.is_producer).count();

        let avg_generation = if alive_organisms > 0 {
            self.organisms
                .iter()
                .filter(|o| o.alive)
                .map(|o| o.generation as f32)
                .sum::<f32>()
                / alive_organisms as f32
        } else {
            0.0
        };

        let avg_energy = if alive_organisms > 0 {
            self.organisms
                .iter()
                .filter(|o| o.alive)
                .map(|o| o.energy)
                .sum::<f32>()
                / alive_organisms as f32
        } else {
            0.0
        };

        // Simple Shannon diversity using producer/consumer ratio
        let shannon_diversity = if alive_organisms > 0 {
            let p = producers as f32 / alive_organisms as f32;
            let c = consumers as f32 / alive_organisms as f32;
            let mut h = 0.0;
            if p > 0.0 { h -= p * p.log2(); }
            if c > 0.0 { h -= c * c.log2(); }
            h
        } else {
            0.0
        };

        EcosystemStats {
            tick: self.tick,
            total_organisms: alive_organisms,
            producers,
            consumers,
            species_count: if alive_organisms > 0 { 2 } else { 0 },
            avg_generation,
            max_generation: self.organisms.iter().map(|o| o.generation).max().unwrap_or(0),
            avg_energy,
            shannon_diversity,
        }
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.stats_history).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_new() {
        let config = SimulationConfig::default();
        let sim = Simulation::new(config);
        assert_eq!(sim.organisms.len(), 250);
        assert_eq!(sim.tick, 0);
    }

    #[test]
    fn test_simulation_tick() {
        let mut config = SimulationConfig::default();
        config.initial_producers = 50;
        config.initial_consumers = 10;
        let mut sim = Simulation::new(config);

        let report = sim.tick();
        assert_eq!(report.tick, 1);
        assert!(report.alive_count > 0);
    }

    #[test]
    fn test_simulation_organisms_survive() {
        let mut config = SimulationConfig::default();
        config.initial_producers = 100;
        config.initial_consumers = 20;
        let mut sim = Simulation::new(config);

        for _ in 0..100 {
            sim.tick();
        }

        assert!(sim.organisms.len() > 0,
            "Population went extinct after 100 ticks!");
    }
}
