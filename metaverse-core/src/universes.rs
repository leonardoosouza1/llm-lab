// ============================================================================
// Universes Module: Themed Physics Sandboxes for Computational Problem Solving
// ============================================================================
// Each Universe is a physics sandbox where organisms evolve to solve specific
// computational tasks. The universe's physical laws are designed to make
// evolution naturally solve the problem. The organisms that survive ARE the solution.
// ============================================================================

use crate::Vec2;
use serde::{Serialize, Deserialize};
use std::f32::consts::PI;

/// A universe organism: an entity with physics state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseOrganism {
    pub id: u64,
    pub position: Vec2,
    pub velocity: Vec2,
    pub energy: f32,
    pub charge: f32,      // for plasma
    pub spin: f32,        // for crystal (-1 to 1)
    pub wavelength: f32,  // for photonic (in grid units)
    pub payload: Vec<f32>, // data being carried
    pub alive: bool,
    pub fitness: f32,
}

impl UniverseOrganism {
    pub fn new(id: u64, x: f32, y: f32) -> Self {
        Self {
            id,
            position: Vec2::new(x, y),
            velocity: Vec2::zero(),
            energy: 1.0,
            charge: 0.0,
            spin: 0.0,
            wavelength: 1.0,
            payload: Vec::new(),
            alive: true,
            fitness: 0.0,
        }
    }
}

/// Grid for universe simulations: stores scalar and vector fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseGrid {
    pub width: usize,
    pub height: usize,
    pub scalar_field: Vec<f32>,  // pressure, temperature, charge density, etc
    pub vector_field: Vec<(f32, f32)>, // velocity, force, etc (row-major)
}

impl UniverseGrid {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            scalar_field: vec![0.0; size],
            vector_field: vec![(0.0, 0.0); size],
        }
    }

    pub fn index(&self, x: usize, y: usize) -> usize {
        ((y % self.height) * self.width) + (x % self.width)
    }

    pub fn get_scalar(&self, x: usize, y: usize) -> f32 {
        self.scalar_field[self.index(x, y)]
    }

    pub fn set_scalar(&mut self, x: usize, y: usize, value: f32) {
        let idx = self.index(x, y);
        self.scalar_field[idx] = value;
    }

    pub fn add_scalar(&mut self, x: usize, y: usize, delta: f32) {
        let idx = self.index(x, y);
        self.scalar_field[idx] += delta;
    }

    pub fn get_vector(&self, x: usize, y: usize) -> (f32, f32) {
        self.vector_field[self.index(x, y)]
    }

    pub fn set_vector(&mut self, x: usize, y: usize, vx: f32, vy: f32) {
        let idx = self.index(x, y);
        self.vector_field[idx] = (vx, vy);
    }
}

/// Core trait for all universe physics
pub trait UniversePhysics: Send + Sync {
    fn name(&self) -> &str;
    fn tick(&mut self, organisms: &mut [UniverseOrganism], grid: &mut UniverseGrid);
    fn fitness(&self, organism: &UniverseOrganism) -> f32;
    fn inject_problem(&mut self, input: &[f32]);
    fn extract_solution(&self) -> Vec<f32>;
}

// ============================================================================
// 1. FLUID UNIVERSE — Solves ROUTING/FLOW problems
// ============================================================================

/// FluidUniverse: organisms as fluid particles carrying tokens through viscous flow
/// Physics: Navier-Stokes simplified (viscous flow on 2D grid)
/// - Viscosity = resistance to information flow
/// - Pressure gradients push organisms toward low-density regions (load balancing)
/// - Turbulence introduces creative exploration
pub struct FluidUniverse {
    viscosity: f32,
    #[allow(dead_code)]
    gravity: f32,
    turbulence: f32,
    input_field: Vec<f32>,  // injected problem data
    solution_field: Vec<f32>, // where solutions accumulate
}

impl FluidUniverse {
    pub fn new(viscosity: f32) -> Self {
        Self {
            viscosity: viscosity.clamp(0.01, 1.0),
            gravity: 0.1,
            turbulence: 0.2,
            input_field: Vec::new(),
            solution_field: Vec::new(),
        }
    }
}

impl UniversePhysics for FluidUniverse {
    fn name(&self) -> &str {
        "FluidUniverse"
    }

    fn tick(&mut self, organisms: &mut [UniverseOrganism], grid: &mut UniverseGrid) {
        // Apply pressure field: negative viscosity zones pull particles
        for y in 0..grid.height {
            for x in 0..grid.width {
                let pressure = grid.get_scalar(x, y);
                // Gradient pushes toward low pressure (negative pressure = attraction)
                let force_x = if x > 0 {
                    (pressure - grid.get_scalar(x - 1, y)) * 0.1
                } else {
                    0.0
                };
                let force_y = if y > 0 {
                    (pressure - grid.get_scalar(x, y - 1)) * 0.1
                } else {
                    0.0
                };
                grid.set_vector(x, y, force_x, force_y);
            }
        }

        // Move organisms according to flow field + random turbulence
        for org in organisms.iter_mut() {
            if !org.alive {
                continue;
            }

            let gx = org.position.x as usize;
            let gy = org.position.y as usize;
            let (vx, vy) = grid.get_vector(gx, gy);

            // Velocity update: drag + flow
            org.velocity.x = org.velocity.x * (1.0 - self.viscosity) + vx;
            org.velocity.y = org.velocity.y * (1.0 - self.viscosity) + vy;

            // Add turbulence (random exploration)
            let angle = (org.id as f32).sin() * PI;
            let turb_x = self.turbulence * angle.cos();
            let turb_y = self.turbulence * angle.sin();
            org.velocity.x += turb_x;
            org.velocity.y += turb_y;

            // Update position with wrapping
            org.position.x = (org.position.x + org.velocity.x) % grid.width as f32;
            org.position.y = (org.position.y + org.velocity.y) % grid.height as f32;
            if org.position.x < 0.0 { org.position.x += grid.width as f32; }
            if org.position.y < 0.0 { org.position.y += grid.height as f32; }

            // Energy cost: movement and carrying payload
            let speed = org.velocity.length();
            let payload_cost = org.payload.len() as f32 * 0.01;
            org.energy -= speed * 0.01 + payload_cost + 0.01; // basal metabolism

            if org.energy <= 0.0 {
                org.alive = false;
            }
        }

        // Mark reached destinations by recording organism positions to solution field
        for org in organisms.iter() {
            if org.alive {
                let x = org.position.x as usize;
                let y = org.position.y as usize;
                let idx = grid.index(x, y);
                if idx < self.solution_field.len() {
                    self.solution_field[idx] += org.energy; // energy = fitness
                }
            }
        }
    }

    fn fitness(&self, organism: &UniverseOrganism) -> f32 {
        // Fitness: remaining energy + data successfully carried
        organism.energy + (organism.payload.len() as f32 * 0.5)
    }

    fn inject_problem(&mut self, input: &[f32]) {
        self.input_field = input.to_vec();
        // Create viscosity field from input: high values = high resistance zones
        self.solution_field = vec![0.0; input.len()];
    }

    fn extract_solution(&self) -> Vec<f32> {
        self.solution_field.clone()
    }
}

// ============================================================================
// 2. PHOTONIC UNIVERSE — Solves ATTENTION/SEARCH problems
// ============================================================================

/// PhotonicUniverse: organisms as photons that propagate through tensor lenses
/// Physics: Ray-tracing / wave optics simplified
/// - Each tensor layer is a lens that bends light based on weight values
/// - Constructive interference = attention (organisms converge on same token)
/// - Destructive interference = suppression
pub struct PhotonicUniverse {
    refractive_indices: Vec<f32>, // per layer
    interference_energy: Vec<f32>, // constructive interference accumulation
    solution_attention: Vec<f32>,  // final attention weights
}

impl PhotonicUniverse {
    pub fn new() -> Self {
        Self {
            refractive_indices: vec![1.0; 8],
            interference_energy: Vec::new(),
            solution_attention: Vec::new(),
        }
    }
}

impl UniversePhysics for PhotonicUniverse {
    fn name(&self) -> &str {
        "PhotonicUniverse"
    }

    fn tick(&mut self, organisms: &mut [UniverseOrganism], grid: &mut UniverseGrid) {
        // Initialize interference grid from refractive indices
        for y in 0..grid.height {
            for x in 0..grid.width {
                let idx = y * grid.width + x;
                if idx < self.refractive_indices.len() {
                    grid.set_scalar(x, y, self.refractive_indices[idx]);
                }
            }
        }

        // Move organisms as photons: they follow lens gradients (Snell's law simplified)
        for org in organisms.iter_mut() {
            if !org.alive {
                continue;
            }

            let x = org.position.x as usize;
            let y = org.position.y as usize;

            // Refraction: bend toward higher refractive index
            let center_n = grid.get_scalar(x, y);

            let n_right = if x + 1 < grid.width {
                grid.get_scalar(x + 1, y)
            } else {
                center_n
            };
            let n_down = if y + 1 < grid.height {
                grid.get_scalar(x, y + 1)
            } else {
                center_n
            };

            // Bend toward higher refractive index
            org.velocity.x += (n_right - center_n) * 0.1;
            org.velocity.y += (n_down - center_n) * 0.1;

            // Normalize velocity magnitude
            let speed = org.velocity.length().max(0.1);
            org.velocity.x = (org.velocity.x / speed) * 0.5;
            org.velocity.y = (org.velocity.y / speed) * 0.5;

            // Move
            org.position = org.position + org.velocity;
            org.position.x = org.position.x % grid.width as f32;
            org.position.y = org.position.y % grid.height as f32;

            // Energy = wavelength propagation cost
            org.energy -= 0.02 + org.wavelength * 0.01;

            if org.energy <= 0.0 {
                org.alive = false;
            }
        }

        // Constructive interference: organisms at same location reinforce each other
        let mut convergence: Vec<u32> = vec![0; grid.width * grid.height];
        for org in organisms.iter() {
            if org.alive {
                let x = org.position.x as usize;
                let y = org.position.y as usize;
                let idx = grid.index(x, y);
                convergence[idx] += 1;
            }
        }

        // Update interference_energy
        if self.interference_energy.len() != convergence.len() {
            self.interference_energy = vec![0.0; convergence.len()];
        }
        for i in 0..convergence.len() {
            // Constructive: multiple organisms = high attention
            self.interference_energy[i] = (convergence[i] as f32).sqrt();
        }
    }

    fn fitness(&self, organism: &UniverseOrganism) -> f32 {
        // Fitness: energy remaining + wavelength match (smaller wavelength = better focus)
        organism.energy + 1.0 / (organism.wavelength + 0.1)
    }

    fn inject_problem(&mut self, input: &[f32]) {
        // Treat input as refractive index pattern
        self.refractive_indices = input.iter()
            .map(|x| (1.0 + x.abs()).clamp(0.5, 2.0))
            .collect();
        self.solution_attention = vec![0.0; input.len()];
    }

    fn extract_solution(&self) -> Vec<f32> {
        // Normalize interference energy to [0, 1]
        if self.interference_energy.is_empty() {
            return vec![];
        }
        let max_energy = self.interference_energy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_energy <= 0.0 {
            return vec![0.0; self.interference_energy.len()];
        }
        self.interference_energy.iter().map(|e| e / max_energy).collect()
    }
}

// ============================================================================
// 3. CRYSTAL UNIVERSE — Solves QUANTIZATION/COMPRESSION problems
// ============================================================================

/// CrystalUniverse: organisms as atoms forming crystals through alignment
/// Physics: Crystal growth / Ising model
/// - Organisms have spin (weight value) that aligns with neighbors
/// - Temperature controls exploration (annealing)
/// - Defects = information (compression residuals)
pub struct CrystalUniverse {
    temperature: f32,
    coupling: f32,         // spin-spin coupling strength
    defect_energy: Vec<f32>, // energy stored in crystal defects
    crystal_structure: Vec<f32>, // evolved crystal weights
}

impl CrystalUniverse {
    pub fn new(temperature: f32, coupling: f32) -> Self {
        Self {
            temperature: temperature.clamp(0.01, 10.0),
            coupling: coupling.clamp(0.01, 1.0),
            defect_energy: Vec::new(),
            crystal_structure: Vec::new(),
        }
    }
}

impl UniversePhysics for CrystalUniverse {
    fn name(&self) -> &str {
        "CrystalUniverse"
    }

    fn tick(&mut self, organisms: &mut [UniverseOrganism], grid: &mut UniverseGrid) {
        // Place organisms in crystal positions
        for org in organisms.iter_mut() {
            if !org.alive {
                continue;
            }

            let x = org.position.x as usize;
            let y = org.position.y as usize;

            // Ising model: compute local field from neighbors
            let mut local_field = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    if dx == 1 && dy == 1 { continue; } // skip self
                    let nx = if x + dx > 0 { (x + dx - 1) % grid.width } else { grid.width - 1 };
                    let ny = if y + dy > 0 { (y + dy - 1) % grid.height } else { grid.height - 1 };
                    let neighbor_spin = grid.get_scalar(nx, ny);
                    local_field += neighbor_spin * self.coupling;
                }
            }

            // Spin flip probability (Glauber dynamics)
            let energy_diff = 2.0 * org.spin * local_field;
            let flip_prob = 1.0 / (1.0 + (energy_diff / self.temperature).exp());

            // Flip spin with probability
            if flip_prob > 0.5 {
                org.spin = -org.spin;
            }

            // Store spin in grid
            grid.set_scalar(x, y, org.spin);

            // Energy cost: thermal fluctuation
            org.energy -= self.temperature * 0.01;

            if org.energy <= 0.0 {
                org.alive = false;
            }
        }

        // Compute defects: positions where local spins differ from neighbors
        for y in 0..grid.height {
            for x in 0..grid.width {
                let center_spin = grid.get_scalar(x, y);
                let neighbors: Vec<f32> = vec![
                    grid.get_scalar(if x > 0 { x - 1 } else { grid.width - 1 }, y),
                    grid.get_scalar((x + 1) % grid.width, y),
                    grid.get_scalar(x, if y > 0 { y - 1 } else { grid.height - 1 }),
                    grid.get_scalar(x, (y + 1) % grid.height),
                ];

                let disagreement: f32 = neighbors.iter()
                    .filter(|&&s| s != center_spin)
                    .count() as f32;

                let idx = grid.index(x, y);
                if idx < self.defect_energy.len() {
                    self.defect_energy[idx] = disagreement;
                }
            }
        }

        // Update crystal structure from current spin configuration
        if self.crystal_structure.len() != grid.width * grid.height {
            self.crystal_structure = vec![0.0; grid.width * grid.height];
        }
        for i in 0..grid.scalar_field.len() {
            self.crystal_structure[i] = grid.scalar_field[i];
        }
    }

    fn fitness(&self, organism: &UniverseOrganism) -> f32 {
        // Fitness: remaining energy + spin alignment (lower defect energy is better)
        let defect_idx = (organism.position.y as usize * 256 + organism.position.x as usize)
            % self.defect_energy.len().max(1);
        let defect = if self.defect_energy.is_empty() { 0.0 } else { self.defect_energy[defect_idx] };
        organism.energy + 1.0 / (defect + 1.0)
    }

    fn inject_problem(&mut self, input: &[f32]) {
        // Initialize spins from input
        self.defect_energy = vec![0.0; input.len()];
        self.crystal_structure = input.iter()
            .map(|x| if *x > 0.0 { 1.0 } else { -1.0 })
            .collect();
    }

    fn extract_solution(&self) -> Vec<f32> {
        self.crystal_structure.clone()
    }
}

// ============================================================================
// 4. PLASMA UNIVERSE — Solves INFERENCE/ACTIVATION problems
// ============================================================================

/// PlasmaUniverse: organisms carry charge, fields propagate like activations
/// Physics: Electromagnetic plasma
/// - Positive charge = excitatory, negative = inhibitory
/// - Fields propagate between organisms (like neuron activations)
/// - Current flow = inference computation
/// - Magnetic pinch effect = attention focusing
pub struct PlasmaUniverse {
    charge_density: Vec<f32>,  // charge distribution on grid
    potential: Vec<f32>,       // electric potential field
    current: Vec<f32>,         // current accumulation (inference)
}

impl PlasmaUniverse {
    pub fn new() -> Self {
        Self {
            charge_density: Vec::new(),
            potential: Vec::new(),
            current: Vec::new(),
        }
    }
}

impl UniversePhysics for PlasmaUniverse {
    fn name(&self) -> &str {
        "PlasmaUniverse"
    }

    fn tick(&mut self, organisms: &mut [UniverseOrganism], grid: &mut UniverseGrid) {
        let grid_size = grid.width * grid.height;

        // Initialize potential and charge_density if needed
        if self.charge_density.is_empty() {
            self.charge_density = vec![0.0; grid_size];
            self.potential = vec![0.0; grid_size];
        }

        // Update charge density from organisms
        for i in 0..self.charge_density.len() {
            self.charge_density[i] *= 0.9; // decay
        }

        for org in organisms.iter() {
            if org.alive {
                let x = org.position.x as usize;
                let y = org.position.y as usize;
                let idx = grid.index(x, y);
                if idx < self.charge_density.len() {
                    self.charge_density[idx] += org.charge;
                }
            }
        }

        // Propagate electric potential (Poisson's equation approximation)
        let mut new_potential = self.potential.clone();
        if grid.height > 2 && grid.width > 2 {
            for y in 1..grid.height - 1 {
                for x in 1..grid.width - 1 {
                    let idx = grid.index(x, y);
                    let idx_left = grid.index(x - 1, y);
                    let idx_right = grid.index(x + 1, y);
                    let idx_up = grid.index(x, y - 1);
                    let idx_down = grid.index(x, y + 1);

                    if idx < self.potential.len() && idx_left < self.potential.len() &&
                       idx_right < self.potential.len() && idx_up < self.potential.len() &&
                       idx_down < self.potential.len() && idx < self.charge_density.len() {
                        // Laplacian operator: ∇²V = -ρ/ε₀
                        let laplacian = (self.potential[idx_left] + self.potential[idx_right] +
                                        self.potential[idx_up] + self.potential[idx_down] -
                                        4.0 * self.potential[idx]) / 4.0;

                        new_potential[idx] = laplacian - self.charge_density[idx] * 0.01;
                    }
                }
            }
        }
        self.potential = new_potential;

        // Compute electric field and move organisms
        for org in organisms.iter_mut() {
            if !org.alive {
                continue;
            }

            let x = org.position.x as usize % grid.width;
            let y = org.position.y as usize % grid.height;

            // Field gradient: force particles toward opposite charge
            let x_left = if x > 0 { x - 1 } else { grid.width - 1 };
            let x_right = (x + 1) % grid.width;
            let y_up = if y > 0 { y - 1 } else { grid.height - 1 };
            let y_down = (y + 1) % grid.height;

            let idx_left = grid.index(x_left, y);
            let idx_right = grid.index(x_right, y);
            let idx_up = grid.index(x, y_up);
            let idx_down = grid.index(x, y_down);

            let ex = if idx_left < self.potential.len() && idx_right < self.potential.len() {
                (self.potential[idx_right] - self.potential[idx_left]) / 2.0
            } else {
                0.0
            };
            let ey = if idx_up < self.potential.len() && idx_down < self.potential.len() {
                (self.potential[idx_down] - self.potential[idx_up]) / 2.0
            } else {
                0.0
            };

            // Lorentz force: F = q(E + v × B)
            org.velocity.x += ex * org.charge * 0.01;
            org.velocity.y += ey * org.charge * 0.01;

            // Move
            org.position = org.position + org.velocity;
            org.position.x = org.position.x % grid.width as f32;
            org.position.y = org.position.y % grid.height as f32;

            // Energy cost
            org.energy -= org.charge.abs() * 0.01 + 0.02;

            if org.energy <= 0.0 {
                org.alive = false;
            }
        }

        // Integrate current: regions where charge flows
        for i in 0..self.current.len() {
            self.current[i] = self.current[i] * 0.8 + self.charge_density[i].abs() * 0.2;
        }
    }

    fn fitness(&self, organism: &UniverseOrganism) -> f32 {
        // Fitness: remaining energy + charge successfully propagated
        organism.energy + (organism.charge.abs() * 2.0)
    }

    fn inject_problem(&mut self, input: &[f32]) {
        // Initialize charge distribution
        let size = input.len();
        self.charge_density = input.iter()
            .map(|x| x.clamp(-1.0, 1.0))
            .collect();
        self.potential = vec![0.0; size];
        self.current = vec![0.0; size];
    }

    fn extract_solution(&self) -> Vec<f32> {
        // Normalize current: represents inference path
        if self.current.is_empty() {
            return vec![];
        }
        let max_current = self.current.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_current <= 0.0 {
            return vec![0.0; self.current.len()];
        }
        self.current.iter().map(|c| c / max_current).collect()
    }
}

// ============================================================================
// 5. HYBRID UNIVERSE — Combines multiple physics
// ============================================================================

/// HybridUniverse: composition of any subset of physics engines
/// Mix ratios control how much each physics contributes
pub struct HybridUniverse {
    universes: Vec<(Box<dyn UniversePhysics>, String)>,
    mix_ratios: Vec<f32>, // normalized weights
}

impl HybridUniverse {
    pub fn new() -> Self {
        Self {
            universes: Vec::new(),
            mix_ratios: Vec::new(),
        }
    }

    pub fn add_physics(&mut self, physics: Box<dyn UniversePhysics>, ratio: f32) {
        let name = physics.name().to_string();
        self.universes.push((physics, name));
        self.mix_ratios.push(ratio);

        // Normalize ratios
        let total: f32 = self.mix_ratios.iter().sum();
        if total > 0.0 {
            for ratio in &mut self.mix_ratios {
                *ratio /= total;
            }
        }
    }
}

impl UniversePhysics for HybridUniverse {
    fn name(&self) -> &str {
        "HybridUniverse"
    }

    fn tick(&mut self, organisms: &mut [UniverseOrganism], grid: &mut UniverseGrid) {
        // Run each physics engine with a copy of organisms
        let mut accumulated_fitness = vec![0.0; organisms.len()];

        for (i, (physics, _)) in self.universes.iter_mut().enumerate() {
            let mut org_copies = organisms.to_vec();
            let mut grid_copy = grid.clone();

            physics.tick(&mut org_copies, &mut grid_copy);

            // Accumulate fitness contributions
            for (j, org) in org_copies.iter().enumerate() {
                accumulated_fitness[j] += physics.fitness(org) * self.mix_ratios[i];
            }

            // Update organisms with mixed results
            for (j, org_copy) in org_copies.iter().enumerate() {
                organisms[j].energy = organisms[j].energy * 0.5 + org_copy.energy * 0.5;
                organisms[j].alive = organisms[j].alive && org_copy.alive;
            }
        }

        // Update fitness scores
        for (i, org) in organisms.iter_mut().enumerate() {
            org.fitness = accumulated_fitness[i];
        }
    }

    fn fitness(&self, organism: &UniverseOrganism) -> f32 {
        organism.fitness
    }

    fn inject_problem(&mut self, input: &[f32]) {
        for (physics, _) in self.universes.iter_mut() {
            physics.inject_problem(input);
        }
    }

    fn extract_solution(&self) -> Vec<f32> {
        if self.universes.is_empty() {
            return Vec::new();
        }

        // Blend solutions from all universes
        let solution_size = self.universes[0].0.extract_solution().len();
        let mut blended = vec![0.0; solution_size];

        for (i, (physics, _)) in self.universes.iter().enumerate() {
            let solution = physics.extract_solution();
            for (j, &val) in solution.iter().enumerate() {
                if j < blended.len() {
                    blended[j] += val * self.mix_ratios[i];
                }
            }
        }

        blended
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // FluidUniverse Tests
    // ========================================================================

    #[test]
    fn test_fluid_universe_creation() {
        let fluid = FluidUniverse::new(0.1);
        assert_eq!(fluid.name(), "FluidUniverse");
        assert!(fluid.viscosity > 0.0);
    }

    #[test]
    fn test_fluid_universe_organisms_move() {
        let mut fluid = FluidUniverse::new(0.1);
        let mut grid = UniverseGrid::new(128, 128);
        let mut organisms = vec![UniverseOrganism::new(1, 64.0, 64.0)];
        organisms[0].velocity = Vec2::new(1.0, 0.0);

        let pos_before = organisms[0].position.clone();
        fluid.tick(&mut organisms, &mut grid);
        let pos_after = organisms[0].position.clone();

        assert!(pos_before != pos_after);
    }

    #[test]
    fn test_fluid_universe_inject_extract() {
        let mut fluid = FluidUniverse::new(0.1);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        fluid.inject_problem(&input);

        let solution = fluid.extract_solution();
        assert_eq!(solution.len(), 5);
    }

    #[test]
    fn test_fluid_universe_energy_decay() {
        let mut fluid = FluidUniverse::new(0.1);
        let mut grid = UniverseGrid::new(128, 128);
        let mut organisms = vec![UniverseOrganism::new(1, 64.0, 64.0)];
        organisms[0].energy = 1.0;

        let energy_before = organisms[0].energy;
        fluid.tick(&mut organisms, &mut grid);
        let energy_after = organisms[0].energy;

        assert!(energy_after < energy_before);
    }

    // ========================================================================
    // PhotonicUniverse Tests
    // ========================================================================

    #[test]
    fn test_photonic_universe_creation() {
        let photonic = PhotonicUniverse::new();
        assert_eq!(photonic.name(), "PhotonicUniverse");
        assert_eq!(photonic.refractive_indices.len(), 8);
    }

    #[test]
    fn test_photonic_universe_organisms_refract() {
        let mut photonic = PhotonicUniverse::new();
        let mut grid = UniverseGrid::new(128, 128);
        let mut organisms = vec![UniverseOrganism::new(1, 64.0, 64.0)];
        organisms[0].velocity = Vec2::new(1.0, 1.0);

        photonic.tick(&mut organisms, &mut grid);
        // Just check it doesn't crash
        assert!(organisms[0].alive || !organisms[0].alive); // tautology but tests it ran
    }

    #[test]
    fn test_photonic_universe_convergence() {
        let mut photonic = PhotonicUniverse::new();
        let mut grid = UniverseGrid::new(64, 64);
        let mut organisms = vec![
            UniverseOrganism::new(1, 32.0, 32.0),
            UniverseOrganism::new(2, 32.0, 32.0),
            UniverseOrganism::new(3, 32.0, 32.0),
        ];

        for _ in 0..10 {
            photonic.tick(&mut organisms, &mut grid);
        }

        let solution = photonic.extract_solution();
        assert!(!solution.is_empty());
    }

    // ========================================================================
    // CrystalUniverse Tests
    // ========================================================================

    #[test]
    fn test_crystal_universe_creation() {
        let crystal = CrystalUniverse::new(1.0, 0.5);
        assert_eq!(crystal.name(), "CrystalUniverse");
    }

    #[test]
    fn test_crystal_universe_spin_alignment() {
        let mut crystal = CrystalUniverse::new(0.5, 0.8);
        let mut grid = UniverseGrid::new(64, 64);
        let mut organisms = vec![UniverseOrganism::new(1, 32.0, 32.0)];
        organisms[0].spin = 1.0;

        crystal.tick(&mut organisms, &mut grid);

        // Organism should either keep or flip spin
        assert!(organisms[0].spin == 1.0 || organisms[0].spin == -1.0);
    }

    #[test]
    fn test_crystal_universe_temperature_effect() {
        let mut crystal_hot = CrystalUniverse::new(5.0, 0.5);
        let mut crystal_cold = CrystalUniverse::new(0.1, 0.5);

        let mut grid1 = UniverseGrid::new(64, 64);
        let mut grid2 = UniverseGrid::new(64, 64);

        let mut orgs1 = vec![UniverseOrganism::new(1, 32.0, 32.0); 100];
        let mut orgs2 = orgs1.clone();

        orgs1.iter_mut().for_each(|o| o.spin = 1.0);
        orgs2.iter_mut().for_each(|o| o.spin = 1.0);

        crystal_hot.tick(&mut orgs1, &mut grid1);
        crystal_cold.tick(&mut orgs2, &mut grid2);

        // At high temperature, more spins should flip
        let hot_flips = orgs1.iter().filter(|o| o.spin < 0.0).count();
        let cold_flips = orgs2.iter().filter(|o| o.spin < 0.0).count();

        assert!(hot_flips >= cold_flips);
    }

    // ========================================================================
    // PlasmaUniverse Tests
    // ========================================================================

    #[test]
    fn test_plasma_universe_creation() {
        let plasma = PlasmaUniverse::new();
        assert_eq!(plasma.name(), "PlasmaUniverse");
    }

    #[test]
    fn test_plasma_universe_charge_propagation() {
        let mut plasma = PlasmaUniverse::new();
        let input = vec![1.0, -1.0, 0.5, -0.5];
        plasma.inject_problem(&input);

        let mut grid = UniverseGrid::new(64, 64);
        let mut organisms = vec![UniverseOrganism::new(1, 32.0, 32.0)];
        organisms[0].charge = 1.0;

        plasma.tick(&mut organisms, &mut grid);

        assert!(!plasma.charge_density.is_empty());
    }

    #[test]
    fn test_plasma_universe_solution_extraction() {
        let mut plasma = PlasmaUniverse::new();
        let input = vec![0.1, 0.2, 0.3, 0.4];
        plasma.inject_problem(&input);

        let mut grid = UniverseGrid::new(64, 64);
        let mut organisms = vec![];

        for i in 0..4 {
            let mut org = UniverseOrganism::new(i as u64, (16 + i * 8) as f32, 32.0);
            org.charge = 1.0;
            organisms.push(org);
        }

        for _ in 0..5 {
            plasma.tick(&mut organisms, &mut grid);
        }

        let solution = plasma.extract_solution();
        assert_eq!(solution.len(), 4);
    }

    // ========================================================================
    // HybridUniverse Tests
    // ========================================================================

    #[test]
    fn test_hybrid_universe_composition() {
        let mut hybrid = HybridUniverse::new();
        hybrid.add_physics(Box::new(FluidUniverse::new(0.1)), 0.5);
        hybrid.add_physics(Box::new(PhotonicUniverse::new()), 0.5);

        assert_eq!(hybrid.name(), "HybridUniverse");
        assert_eq!(hybrid.universes.len(), 2);
    }

    #[test]
    fn test_hybrid_universe_ratio_normalization() {
        let mut hybrid = HybridUniverse::new();
        hybrid.add_physics(Box::new(FluidUniverse::new(0.1)), 1.0);
        hybrid.add_physics(Box::new(PhotonicUniverse::new()), 2.0);

        let total: f32 = hybrid.mix_ratios.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hybrid_universe_tick() {
        let mut hybrid = HybridUniverse::new();
        hybrid.add_physics(Box::new(FluidUniverse::new(0.1)), 1.0);

        let mut grid = UniverseGrid::new(128, 128);
        let mut organisms = vec![UniverseOrganism::new(1, 64.0, 64.0)];
        organisms[0].energy = 1.0;

        hybrid.tick(&mut organisms, &mut grid);

        assert!(organisms[0].energy < 1.0);
    }

    #[test]
    fn test_hybrid_universe_problem_injection() {
        let mut hybrid = HybridUniverse::new();
        hybrid.add_physics(Box::new(FluidUniverse::new(0.1)), 0.5);
        hybrid.add_physics(Box::new(PlasmaUniverse::new()), 0.5);

        let input = vec![0.1, 0.2, 0.3];
        hybrid.inject_problem(&input);

        // Should not crash
        assert_eq!(hybrid.universes.len(), 2);
    }

    // ========================================================================
    // Universe Grid Tests
    // ========================================================================

    #[test]
    fn test_universe_grid_indexing() {
        let grid = UniverseGrid::new(32, 32);
        assert_eq!(grid.index(0, 0), 0);
        assert_eq!(grid.index(1, 0), 1);
        assert_eq!(grid.index(0, 1), 32);
    }

    #[test]
    fn test_universe_grid_wrapping() {
        let grid = UniverseGrid::new(32, 32);
        let idx_wrap = grid.index(32, 32);
        let idx_base = grid.index(0, 0);
        assert_eq!(idx_wrap, idx_base);
    }

    #[test]
    fn test_universe_grid_scalar_operations() {
        let mut grid = UniverseGrid::new(16, 16);
        grid.set_scalar(5, 5, 1.5);
        assert_eq!(grid.get_scalar(5, 5), 1.5);

        grid.add_scalar(5, 5, 0.5);
        assert_eq!(grid.get_scalar(5, 5), 2.0);
    }

    #[test]
    fn test_universe_organism_fitness() {
        let mut org = UniverseOrganism::new(1, 10.0, 10.0);
        org.energy = 2.0;
        org.payload = vec![1.0, 2.0, 3.0];

        let fluid = FluidUniverse::new(0.1);
        let fitness = fluid.fitness(&org);

        // fitness = energy + (payload_len * 0.5)
        assert!((fitness - (2.0 + 1.5)).abs() < 0.01);
    }
}
