// ============================================================================
// World Module: Grid-Based Environment for Digital Organisms
// ============================================================================

use crate::chemistry::{ResourceCell, ResourceType};
use crate::physics::{GrayScottField, GrayScottParams, diffuse_scalar_field};
use crate::{Vec2, Tick};
use rand::SeedableRng;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Seasons of the world
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

impl Season {
    pub fn next(&self) -> Self {
        match self {
            Season::Spring => Season::Summer,
            Season::Summer => Season::Autumn,
            Season::Autumn => Season::Winter,
            Season::Winter => Season::Spring,
        }
    }
}

/// World configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    pub width: usize,
    pub height: usize,
    pub wrap_around: bool,
    pub ticks_per_day: u32,
    pub days_per_season: u32,
    pub base_light_intensity: f32,
    pub base_temperature: f32,
    pub resource_diffusion_rates: ResourceDiffusionRates,
    pub pheromone_decay_rate: f32,
    pub catastrophe_probability: f32,
}

/// Diffusion rates for different resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDiffusionRates {
    pub water: f32,
    pub nutrient_a: f32,
    pub nutrient_b: f32,
    pub nutrient_c: f32,
    pub oxygen: f32,
    pub toxin: f32,
    pub mineral: f32,
    pub pheromone: f32,
}

impl Default for ResourceDiffusionRates {
    fn default() -> Self {
        Self {
            water: 0.1,
            nutrient_a: 0.05,
            nutrient_b: 0.04,
            nutrient_c: 0.03,
            oxygen: 0.15,
            toxin: 0.08,
            mineral: 0.02,
            pheromone: 0.12,
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            wrap_around: true,
            ticks_per_day: 100,
            days_per_season: 25,
            base_light_intensity: 20.0,
            base_temperature: 20.0,
            resource_diffusion_rates: ResourceDiffusionRates::default(),
            pheromone_decay_rate: 0.02,
            catastrophe_probability: 0.0001,
        }
    }
}

/// Catastrophic events that affect the world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Catastrophe {
    Drought {
        center_x: usize,
        center_y: usize,
        radius: usize,
    },
    Flood {
        center_x: usize,
        center_y: usize,
        radius: usize,
    },
    Glaciation {
        temperature_drop: f32,
    },
    Eruption {
        center_x: usize,
        center_y: usize,
        radius: usize,
    },
    Meteor {
        center_x: usize,
        center_y: usize,
        radius: usize,
    },
}

/// World statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldStats {
    pub total_energy: f32,
    pub avg_temperature: f32,
    pub total_nutrients: f32,
    pub total_water: f32,
    pub total_toxins: f32,
    pub season: Season,
    pub day: u32,
    pub tick: Tick,
}

/// The world grid containing all resources and organisms
pub struct World {
    pub config: WorldConfig,
    pub cells: Vec<ResourceCell>,
    pub topography: Vec<f32>,
    pub tick: Tick,
    pub day: u32,
    pub season: Season,
    rng: rand_chacha::ChaCha8Rng,
}

impl World {
    /// Create a new world with Turing pattern topography
    pub fn new(config: WorldConfig, seed: u64) -> Self {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        // Generate topography using Gray-Scott reaction-diffusion
        let mut gs_field = GrayScottField::new(config.width, config.height);
        let gs_params = GrayScottParams::default();

        // Run 500 steps to develop Turing patterns
        for _ in 0..500 {
            gs_field.step(&gs_params);
        }

        let topography = gs_field.topography();

        // Check if topography is non-trivial
        let min_topo = topography.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_topo = topography.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let is_trivial = (max_topo - min_topo).abs() < 0.01;

        // If trivial, generate random topography
        let topography = if is_trivial {
            let mut topo = vec![0.5; config.width * config.height];
            for val in &mut topo {
                *val = rng.gen::<f32>();
            }
            topo
        } else {
            topography
        };

        // Initialize cells based on topography
        let cells: Vec<ResourceCell> = topography
            .iter()
            .map(|&h| {
                let mut cell = ResourceCell::empty();
                // High terrain: less water, more minerals
                // Low terrain: more water, more nutrients
                if h > 0.7 {
                    cell.water = 3.0;
                    cell.mineral = 5.0;
                } else if h > 0.4 {
                    cell.water = 8.0;
                    cell.nutrient_a = 5.0;
                } else {
                    cell.water = 12.0;
                    cell.nutrient_a = 8.0;
                }
                cell.oxygen = 10.0;
                cell.temperature = 20.0;
                cell
            })
            .collect();

        // Scatter nutrient deposits on high-ground clusters
        let mut world = Self {
            config,
            cells,
            topography,
            tick: 0,
            day: 0,
            season: Season::Spring,
            rng,
        };

        world.scatter_nutrient_deposits();

        world
    }

    /// Scatter initial nutrient deposits
    fn scatter_nutrient_deposits(&mut self) {
        let num_deposits = (self.config.width / 16).max(2);

        for _ in 0..num_deposits {
            let x = self.rng.gen_range(0..self.config.width);
            let y = self.rng.gen_range(0..self.config.height);
            let radius = self.rng.gen_range(3..8);

            for dy in 0..=radius {
                for dx in 0..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        let positions = vec![
                            (x + dx, y + dy),
                            ((x as i32 - dx as i32).max(0) as usize, y + dy),
                            (x + dx, (y as i32 - dy as i32).max(0) as usize),
                            ((x as i32 - dx as i32).max(0) as usize, (y as i32 - dy as i32).max(0) as usize),
                        ];

                        for (px, py) in positions {
                            if px < self.config.width && py < self.config.height {
                                let cell = self.cell_at_mut(px, py);
                                cell.nutrient_a = (cell.nutrient_a + 3.0).min(20.0);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Advance the world by one tick
    pub fn tick(&mut self) {
        // Update day/night cycle
        let light_intensity = self.current_light();

        // Update season if needed
        self.update_season();

        // Diffuse all resources
        self.diffuse_resources();

        // Decay pheromones
        for cell in &mut self.cells {
            cell.pheromone[0] *= 1.0 - self.config.pheromone_decay_rate;
            cell.pheromone[1] *= 1.0 - self.config.pheromone_decay_rate;
            cell.pheromone[2] *= 1.0 - self.config.pheromone_decay_rate;
        }

        // Regenerate resources based on light and topography
        self.regenerate_resources(light_intensity);

        // Attempt catastrophic event
        if self.rng.gen::<f32>() < self.config.catastrophe_probability {
            self.maybe_catastrophe();
        }

        // Increment tick counter
        self.tick += 1;
    }

    /// Get cell at (x, y) with wrap-around handling
    pub fn cell_at(&self, x: usize, y: usize) -> &ResourceCell {
        let (wx, wy) = self.wrap(x as i32, y as i32);
        &self.cells[wy * self.config.width + wx]
    }

    /// Get mutable cell at (x, y) with wrap-around handling
    pub fn cell_at_mut(&mut self, x: usize, y: usize) -> &mut ResourceCell {
        let (wx, wy) = self.wrap(x as i32, y as i32);
        &mut self.cells[wy * self.config.width + wx]
    }

    /// Get current light intensity (day/night cycle)
    pub fn current_light(&self) -> f32 {
        let time_of_day = (self.tick % self.config.ticks_per_day as u64) as f32
            / self.config.ticks_per_day as f32;
        let light = (time_of_day * std::f32::consts::TAU).sin().max(0.0);
        light * self.config.base_light_intensity * self.season_light_modifier()
    }

    /// Get light intensity modifier based on season
    fn season_light_modifier(&self) -> f32 {
        match self.season {
            Season::Spring => 0.9,
            Season::Summer => 1.1,
            Season::Autumn => 0.85,
            Season::Winter => 0.6,
        }
    }

    /// Get temperature modifier based on season
    fn season_temperature_modifier(&self) -> f32 {
        match self.season {
            Season::Spring => 0.0,
            Season::Summer => 10.0,
            Season::Autumn => -5.0,
            Season::Winter => -15.0,
        }
    }

    /// Update season based on tick count
    fn update_season(&mut self) {
        let ticks_per_season =
            self.config.ticks_per_day as u64 * self.config.days_per_season as u64;
        let season_index = (self.tick / ticks_per_season) as usize % 4;

        let new_season = match season_index {
            0 => Season::Spring,
            1 => Season::Summer,
            2 => Season::Autumn,
            _ => Season::Winter,
        };

        if new_season != self.season {
            self.season = new_season;
        }

        // Update day counter
        self.day = ((self.tick / self.config.ticks_per_day as u64) % (self.config.days_per_season as u64 * 4)) as u32;
    }

    /// Diffuse all resource fields
    fn diffuse_resources(&mut self) {
        let width = self.config.width;
        let height = self.config.height;

        // Extract and diffuse water field
        let mut water_field: Vec<f32> = self.cells.iter().map(|c| c.water).collect();
        diffuse_scalar_field(
            &mut water_field,
            width,
            height,
            self.config.resource_diffusion_rates.water,
        );

        // Extract and diffuse nutrient_a field
        let mut nutrient_a_field: Vec<f32> = self.cells.iter().map(|c| c.nutrient_a).collect();
        diffuse_scalar_field(
            &mut nutrient_a_field,
            width,
            height,
            self.config.resource_diffusion_rates.nutrient_a,
        );

        // Extract and diffuse nutrient_b field
        let mut nutrient_b_field: Vec<f32> = self.cells.iter().map(|c| c.nutrient_b).collect();
        diffuse_scalar_field(
            &mut nutrient_b_field,
            width,
            height,
            self.config.resource_diffusion_rates.nutrient_b,
        );

        // Extract and diffuse nutrient_c field
        let mut nutrient_c_field: Vec<f32> = self.cells.iter().map(|c| c.nutrient_c).collect();
        diffuse_scalar_field(
            &mut nutrient_c_field,
            width,
            height,
            self.config.resource_diffusion_rates.nutrient_c,
        );

        // Extract and diffuse oxygen field
        let mut oxygen_field: Vec<f32> = self.cells.iter().map(|c| c.oxygen).collect();
        diffuse_scalar_field(
            &mut oxygen_field,
            width,
            height,
            self.config.resource_diffusion_rates.oxygen,
        );

        // Extract and diffuse toxin field
        let mut toxin_field: Vec<f32> = self.cells.iter().map(|c| c.toxin).collect();
        diffuse_scalar_field(
            &mut toxin_field,
            width,
            height,
            self.config.resource_diffusion_rates.toxin,
        );

        // Extract and diffuse mineral field
        let mut mineral_field: Vec<f32> = self.cells.iter().map(|c| c.mineral).collect();
        diffuse_scalar_field(
            &mut mineral_field,
            width,
            height,
            self.config.resource_diffusion_rates.mineral,
        );

        // Update cells with diffused values
        for (i, cell) in self.cells.iter_mut().enumerate() {
            cell.water = water_field[i];
            cell.nutrient_a = nutrient_a_field[i];
            cell.nutrient_b = nutrient_b_field[i];
            cell.nutrient_c = nutrient_c_field[i];
            cell.oxygen = oxygen_field[i];
            cell.toxin = toxin_field[i];
            cell.mineral = mineral_field[i];
        }
    }

    /// Regenerate resources based on light and topography
    fn regenerate_resources(&mut self, light_intensity: f32) {
        let base_temp = self.config.base_temperature;
        let temp_mod = self.season_temperature_modifier();

        for (i, cell) in self.cells.iter_mut().enumerate() {
            let height = self.topography[i];

            // Light → nutrient conversion for photosynthesis
            let light_gen = light_intensity * 0.05;
            cell.light = light_intensity;
            cell.nutrient_a = (cell.nutrient_a + light_gen).min(20.0);

            // Height affects water availability
            if height < 0.5 {
                cell.water = (cell.water + 0.2).min(20.0);
            }

            // Oxygen regeneration
            if light_intensity > 0.2 && height < 0.8 {
                cell.oxygen = (cell.oxygen + 0.1).min(15.0);
            }

            // Temperature fluctuation with season
            cell.temperature = base_temp + temp_mod;
        }
    }

    /// Attempt catastrophic event
    fn maybe_catastrophe(&mut self) {
        let event_type = self.rng.gen_range(0..5);
        let event = match event_type {
            0 => {
                let cx = self.rng.gen_range(0..self.config.width);
                let cy = self.rng.gen_range(0..self.config.height);
                let r = self.rng.gen_range(10..40);
                Catastrophe::Drought {
                    center_x: cx,
                    center_y: cy,
                    radius: r,
                }
            }
            1 => {
                let cx = self.rng.gen_range(0..self.config.width);
                let cy = self.rng.gen_range(0..self.config.height);
                let r = self.rng.gen_range(10..40);
                Catastrophe::Flood {
                    center_x: cx,
                    center_y: cy,
                    radius: r,
                }
            }
            2 => Catastrophe::Glaciation {
                temperature_drop: self.rng.gen_range(5.0..15.0),
            },
            3 => {
                let cx = self.rng.gen_range(0..self.config.width);
                let cy = self.rng.gen_range(0..self.config.height);
                let r = self.rng.gen_range(5..20);
                Catastrophe::Eruption {
                    center_x: cx,
                    center_y: cy,
                    radius: r,
                }
            }
            _ => {
                let cx = self.rng.gen_range(0..self.config.width);
                let cy = self.rng.gen_range(0..self.config.height);
                let r = self.rng.gen_range(10..30);
                Catastrophe::Meteor {
                    center_x: cx,
                    center_y: cy,
                    radius: r,
                }
            }
        };

        self.apply_catastrophe(&event);
    }

    /// Apply a catastrophic event to the world
    pub fn apply_catastrophe(&mut self, event: &Catastrophe) {
        match event {
            Catastrophe::Drought {
                center_x,
                center_y,
                radius,
            } => {
                for y in 0..self.config.height {
                    for x in 0..self.config.width {
                        let dx = (x as i32 - *center_x as i32).abs() as f32;
                        let dy = (y as i32 - *center_y as i32).abs() as f32;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dist < *radius as f32 {
                            let cell = self.cell_at_mut(x, y);
                            cell.water *= 0.5;
                            cell.nutrient_a *= 0.7;
                        }
                    }
                }
            }
            Catastrophe::Flood {
                center_x,
                center_y,
                radius,
            } => {
                for y in 0..self.config.height {
                    for x in 0..self.config.width {
                        let dx = (x as i32 - *center_x as i32).abs() as f32;
                        let dy = (y as i32 - *center_y as i32).abs() as f32;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dist < *radius as f32 {
                            let cell = self.cell_at_mut(x, y);
                            cell.water = 1.0;
                            cell.nutrient_a *= 0.3;
                            cell.toxin = (cell.toxin + 0.2).min(1.0);
                        }
                    }
                }
            }
            Catastrophe::Glaciation { temperature_drop: _ } => {
                for cell in &mut self.cells {
                    cell.water = (cell.water * 0.8).max(0.2);
                    cell.oxygen = (cell.oxygen + 0.1).min(1.0);
                    cell.nutrient_a *= 0.6;
                }
            }
            Catastrophe::Eruption {
                center_x,
                center_y,
                radius,
            } => {
                for y in 0..self.config.height {
                    for x in 0..self.config.width {
                        let dx = (x as i32 - *center_x as i32).abs() as f32;
                        let dy = (y as i32 - *center_y as i32).abs() as f32;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dist < *radius as f32 {
                            let cell = self.cell_at_mut(x, y);
                            cell.water = 0.0;
                            cell.nutrient_a = 0.0;
                            cell.nutrient_b = 0.0;
                            cell.nutrient_c = 0.0;
                            cell.oxygen = 0.0;
                            cell.mineral = 0.0;
                            cell.toxin = 0.8;
                        }
                    }
                }
            }
            Catastrophe::Meteor {
                center_x,
                center_y,
                radius,
            } => {
                for y in 0..self.config.height {
                    for x in 0..self.config.width {
                        let dx = (x as i32 - *center_x as i32).abs() as f32;
                        let dy = (y as i32 - *center_y as i32).abs() as f32;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dist < *radius as f32 {
                            let cell = self.cell_at_mut(x, y);
                            cell.water = 0.0;
                            cell.nutrient_a = 0.0;
                            cell.nutrient_b = 0.0;
                            cell.nutrient_c = 0.0;
                            cell.oxygen = 0.0;
                            cell.mineral = 0.0;
                            cell.toxin = 0.9;
                        } else if dist < *radius as f32 * 1.5 {
                            let cell = self.cell_at_mut(x, y);
                            cell.toxin = (cell.toxin + 0.3).min(1.0);
                        }
                    }
                }
            }
        }
    }

    /// Get neighboring cells (8-directional)
    pub fn neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = self.wrap(x as i32 + dx, y as i32 + dy);
                result.push((nx, ny));
            }
        }
        result
    }

    /// Wrap coordinates for toroidal topology
    pub fn wrap(&self, x: i32, y: i32) -> (usize, usize) {
        let width = self.config.width as i32;
        let height = self.config.height as i32;

        let wx = if self.config.wrap_around {
            ((x % width) + width) % width
        } else {
            x.max(0).min(width - 1)
        } as usize;

        let wy = if self.config.wrap_around {
            ((y % height) + height) % height
        } else {
            y.max(0).min(height - 1)
        } as usize;

        (wx, wy)
    }

    /// Get resource gradient at position (for chemotaxis)
    pub fn resource_gradient(&self, x: usize, y: usize, resource: ResourceType) -> Vec2 {
        let mut gx = 0.0;
        let mut gy = 0.0;

        let neighbors = self.neighbors(x, y);
        for (nx, ny) in neighbors {
            let neighbor_val = self.cell_at(nx, ny).get(resource);
            let center_val = self.cell_at(x, y).get(resource);
            let diff = neighbor_val - center_val;

            gx += diff * (nx as f32 - x as f32);
            gy += diff * (ny as f32 - y as f32);
        }

        Vec2::new(gx, gy).normalized()
    }

    /// Get world statistics
    pub fn stats(&self) -> WorldStats {
        let mut total_energy = 0.0;
        let mut total_nutrients = 0.0;
        let mut total_water = 0.0;
        let mut total_toxins = 0.0;

        for cell in &self.cells {
            total_energy += (cell.nutrient_a + cell.nutrient_b + cell.nutrient_c + cell.oxygen) * 0.25;
            total_nutrients += cell.nutrient_a + cell.nutrient_b + cell.nutrient_c;
            total_water += cell.water;
            total_toxins += cell.toxin;
        }

        let avg_temperature = self.config.base_temperature + self.season_temperature_modifier();

        WorldStats {
            total_energy,
            avg_temperature,
            total_nutrients,
            total_water,
            total_toxins,
            season: self.season,
            day: self.day,
            tick: self.tick,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_creation_generates_nontrivial_topography() {
        let config = WorldConfig {
            width: 64,
            height: 64,
            ..Default::default()
        };
        let world = World::new(config, 42);

        let min_topo = world.topography.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_topo = world.topography.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Ensure topography is non-trivial
        assert!((max_topo - min_topo).abs() > 0.01, "Topography too uniform");
    }

    #[test]
    fn test_day_night_cycle() {
        let config = WorldConfig {
            width: 64,
            height: 64,
            ticks_per_day: 100,
            base_light_intensity: 1.0,
            ..Default::default()
        };
        let mut world = World::new(config, 42);

        // At tick 0, should be near midnight (light ≈ 0)
        let light_midnight = world.current_light();
        assert!(light_midnight < 0.1, "Light should be ~0 at midnight");

        // At quarter period, should be near midday (light > 0.5)
        world.tick = 25;
        let light_midday = world.current_light();
        assert!(light_midday > 0.5, "Light should be high at midday");

        // At half period, should be near midnight again
        world.tick = 50;
        let light_midnight_again = world.current_light();
        assert!(light_midnight_again < 0.1, "Light should be ~0 again at midnight");
    }

    #[test]
    fn test_season_cycling() {
        let config = WorldConfig {
            width: 64,
            height: 64,
            ticks_per_day: 100,
            days_per_season: 25,
            ..Default::default()
        };
        let mut world = World::new(config, 42);

        let ticks_per_season = 100u64 * 25u64; // 2500 ticks

        assert_eq!(world.season, Season::Spring);

        // Advance to summer
        world.tick = ticks_per_season;
        world.update_season();
        assert_eq!(world.season, Season::Summer);

        // Advance to autumn
        world.tick = ticks_per_season * 2;
        world.update_season();
        assert_eq!(world.season, Season::Autumn);

        // Advance to winter
        world.tick = ticks_per_season * 3;
        world.update_season();
        assert_eq!(world.season, Season::Winter);

        // Back to spring
        world.tick = ticks_per_season * 4;
        world.update_season();
        assert_eq!(world.season, Season::Spring);
    }

    #[test]
    fn test_wrap_coordinates() {
        let config = WorldConfig {
            width: 256,
            height: 256,
            wrap_around: true,
            ..Default::default()
        };
        let world = World::new(config, 42);

        // Test wrapping
        let (x, y) = world.wrap(-1, 0);
        assert_eq!(x, 255);
        assert_eq!(y, 0);

        let (x, y) = world.wrap(256, 256);
        assert_eq!(x, 0);
        assert_eq!(y, 0);

        let (x, y) = world.wrap(100, 100);
        assert_eq!(x, 100);
        assert_eq!(y, 100);
    }

    #[test]
    fn test_catastrophe_drought() {
        let config = WorldConfig {
            width: 128,
            height: 128,
            ..Default::default()
        };
        let mut world = World::new(config, 42);

        // Record initial water at center
        let center_before = world.cell_at(64, 64).water;

        // Apply drought
        let event = Catastrophe::Drought {
            center_x: 64,
            center_y: 64,
            radius: 20,
        };
        world.apply_catastrophe(&event);

        let center_after = world.cell_at(64, 64).water;

        // Water should decrease
        assert!(center_after < center_before, "Drought should reduce water");
    }

    #[test]
    fn test_diffusion_conservation() {
        let config = WorldConfig {
            width: 128,
            height: 128,
            wrap_around: true,
            ..Default::default()
        };
        let mut world = World::new(config, 42);

        let total_before: f32 = world.cells.iter().map(|c| c.water).sum();

        world.diffuse_resources();

        let total_after: f32 = world.cells.iter().map(|c| c.water).sum();

        // Diffusion should conserve total mass (within floating point precision)
        assert!((total_before - total_after).abs() < 1e-3, "Diffusion should conserve mass");
    }
}
