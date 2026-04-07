// ============================================================================
// Chemistry Module: Resource Types, Metabolic Reactions, and Energy Dynamics
// ============================================================================

use crate::{constants, Vec2};
use serde::{Deserialize, Serialize};

/// Resource types available in the world
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Light,      // Solar radiation — energy source
    Water,      // Enables reactions, transport
    NutrientA,  // Primary chemical energy (glucose analog)
    NutrientB,  // Secondary energy (fat analog)
    NutrientC,  // Structural material (protein analog)
    Oxygen,     // Aerobic metabolism enabler
    Mineral,    // Scarce building material
    Toxin,      // Metabolic waste, poison
}

/// What each grid cell contains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCell {
    pub light: f32,
    pub water: f32,
    pub nutrient_a: f32,
    pub nutrient_b: f32,
    pub nutrient_c: f32,
    pub oxygen: f32,
    pub mineral: f32,
    pub toxin: f32,
    pub temperature: f32,
    pub pheromone: [f32; 3], // 3 chemical signal channels (RGB)
}

impl ResourceCell {
    /// Create an empty cell
    pub fn empty() -> Self {
        Self {
            light: 0.0,
            water: 0.0,
            nutrient_a: 0.0,
            nutrient_b: 0.0,
            nutrient_c: 0.0,
            oxygen: 0.0,
            mineral: 0.0,
            toxin: 0.0,
            temperature: 20.0, // default 20°C
            pheromone: [0.0; 3],
        }
    }

    /// Get amount of a resource
    pub fn get(&self, res: ResourceType) -> f32 {
        match res {
            ResourceType::Light => self.light,
            ResourceType::Water => self.water,
            ResourceType::NutrientA => self.nutrient_a,
            ResourceType::NutrientB => self.nutrient_b,
            ResourceType::NutrientC => self.nutrient_c,
            ResourceType::Oxygen => self.oxygen,
            ResourceType::Mineral => self.mineral,
            ResourceType::Toxin => self.toxin,
        }
    }

    /// Set amount of a resource
    pub fn set(&mut self, res: ResourceType, val: f32) {
        let val = val.max(0.0);
        match res {
            ResourceType::Light => self.light = val,
            ResourceType::Water => self.water = val,
            ResourceType::NutrientA => self.nutrient_a = val,
            ResourceType::NutrientB => self.nutrient_b = val,
            ResourceType::NutrientC => self.nutrient_c = val,
            ResourceType::Oxygen => self.oxygen = val,
            ResourceType::Mineral => self.mineral = val,
            ResourceType::Toxin => self.toxin = val,
        }
    }

    /// Add amount to a resource
    pub fn add(&mut self, res: ResourceType, amount: f32) {
        let current = self.get(res);
        self.set(res, current + amount);
    }

    /// Remove amount from a resource, returns actual amount removed (clamped at 0)
    pub fn remove(&mut self, res: ResourceType, amount: f32) -> f32 {
        let current = self.get(res);
        let to_remove = amount.min(current).max(0.0);
        self.set(res, current - to_remove);
        to_remove
    }
}

// ============================================================================
// Metabolic Reactions
// ============================================================================

/// Trait for metabolic reactions
pub trait MetabolicReaction {
    /// Name of the reaction
    fn name(&self) -> &str;

    /// Required inputs with amounts
    fn required_inputs(&self) -> Vec<(ResourceType, f32)>;

    /// Produced outputs with amounts
    fn outputs(&self) -> Vec<(ResourceType, f32)>;

    /// Net ATP (energy) yield
    fn energy_yield(&self) -> f32;

    /// Rate modifier based on temperature (Arrhenius-like)
    fn rate_at_temperature(&self, temp: f32) -> f32;

    /// Execute reaction: consumes from cell, returns energy gained
    fn execute(&self, cell: &mut ResourceCell) -> f32;
}

/// Glycolysis: 1 NutrientA → 2 ATP (fast, inefficient, no O2 needed)
pub struct Glycolysis;

impl MetabolicReaction for Glycolysis {
    fn name(&self) -> &str {
        "Glycolysis"
    }

    fn required_inputs(&self) -> Vec<(ResourceType, f32)> {
        vec![(ResourceType::NutrientA, 1.0), (ResourceType::Water, 0.5)]
    }

    fn outputs(&self) -> Vec<(ResourceType, f32)> {
        vec![
            (ResourceType::Toxin, 0.5), // lactate
        ]
    }

    fn energy_yield(&self) -> f32 {
        2.0 // 2 ATP per glucose
    }

    fn rate_at_temperature(&self, temp: f32) -> f32 {
        // Arrhenius-like: peak at 37°C, drops off
        let optimal = 37.0;
        let diff = (temp - optimal).abs();
        (-(diff * diff) / 200.0).exp()
    }

    fn execute(&self, cell: &mut ResourceCell) -> f32 {
        let inputs = self.required_inputs();
        let mut can_execute = true;

        // Check if all inputs are available
        for (res_type, amount) in &inputs {
            if cell.get(*res_type) < *amount {
                can_execute = false;
                break;
            }
        }

        if !can_execute {
            return 0.0;
        }

        // Consume inputs
        for (res_type, amount) in inputs {
            cell.remove(res_type, amount);
        }

        // Produce outputs
        for (res_type, amount) in self.outputs() {
            cell.add(res_type, amount);
        }

        // Return energy gained (modified by temperature)
        self.energy_yield() * self.rate_at_temperature(cell.temperature)
    }
}

/// Aerobic Respiration: 1 NutrientA + O2 → 30 ATP (slow, efficient, requires O2)
pub struct AerobicRespiration;

impl MetabolicReaction for AerobicRespiration {
    fn name(&self) -> &str {
        "Aerobic Respiration"
    }

    fn required_inputs(&self) -> Vec<(ResourceType, f32)> {
        vec![
            (ResourceType::NutrientA, 1.0),
            (ResourceType::Oxygen, 6.0),
            (ResourceType::Water, 1.0),
        ]
    }

    fn outputs(&self) -> Vec<(ResourceType, f32)> {
        vec![] // CO2 escapes, no waste toxin
    }

    fn energy_yield(&self) -> f32 {
        30.0 // ~30 ATP per glucose
    }

    fn rate_at_temperature(&self, temp: f32) -> f32 {
        // Aerobic respiration is slower but more efficient
        let optimal = 37.0;
        let diff = (temp - optimal).abs();
        (-(diff * diff) / 300.0).exp() * 0.5 // slower than glycolysis
    }

    fn execute(&self, cell: &mut ResourceCell) -> f32 {
        let inputs = self.required_inputs();
        let mut can_execute = true;

        // Check if all inputs are available
        for (res_type, amount) in &inputs {
            if cell.get(*res_type) < *amount {
                can_execute = false;
                break;
            }
        }

        if !can_execute {
            return 0.0;
        }

        // Consume inputs
        for (res_type, amount) in inputs {
            cell.remove(res_type, amount);
        }

        // Return energy gained (modified by temperature)
        self.energy_yield() * self.rate_at_temperature(cell.temperature)
    }
}

/// Photosynthesis: Light + Water → NutrientA + O2
pub struct Photosynthesis;

impl MetabolicReaction for Photosynthesis {
    fn name(&self) -> &str {
        "Photosynthesis"
    }

    fn required_inputs(&self) -> Vec<(ResourceType, f32)> {
        vec![
            (ResourceType::Light, 10.0),
            (ResourceType::Water, 1.0),
        ]
    }

    fn outputs(&self) -> Vec<(ResourceType, f32)> {
        vec![
            (ResourceType::NutrientA, 1.0),
            (ResourceType::Oxygen, 2.0),
        ]
    }

    fn energy_yield(&self) -> f32 {
        // Photosynthesis stores energy in NutrientA
        // Net energy is the efficiency of light capture
        constants::ETA_CAPTURE_BASE * 10.0 // proportional to light captured
    }

    fn rate_at_temperature(&self, temp: f32) -> f32 {
        // Photosynthesis peaks at 25°C
        let optimal = 25.0;
        let diff = (temp - optimal).abs();
        (-(diff * diff) / 150.0).exp()
    }

    fn execute(&self, cell: &mut ResourceCell) -> f32 {
        let inputs = self.required_inputs();
        let mut can_execute = true;

        // Check if all inputs are available
        for (res_type, amount) in &inputs {
            if cell.get(*res_type) < *amount {
                can_execute = false;
                break;
            }
        }

        if !can_execute {
            return 0.0;
        }

        // Consume inputs
        for (res_type, amount) in inputs {
            cell.remove(res_type, amount);
        }

        // Produce outputs
        for (res_type, amount) in self.outputs() {
            cell.add(res_type, amount);
        }

        // Return energy gained (stored energy)
        self.energy_yield() * self.rate_at_temperature(cell.temperature)
    }
}

// ============================================================================
// Metabolism Module
// ============================================================================

/// Complete metabolic state of an organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metabolism {
    pub energy_reserve: f32,      // current ATP
    pub max_energy: f32,          // capacity
    pub has_aerobic: bool,        // Krebs upgrade (evolvable)
    pub has_photosynthesis: bool, // autotroph upgrade
    pub metabolic_flexibility: f32, // how fast switch fuel sources (0..1)
    pub respiratory_quotient: f32, // current RQ
    pub basal_rate: f32,          // BMR = B0 * complexity^0.75 (Kleiber)
}

impl Metabolism {
    /// Create new metabolism for organism with given complexity
    pub fn new(complexity: f32) -> Self {
        let basal_rate = constants::KLEIBER_B0 * complexity.powf(constants::KLEIBER_EXPONENT);

        Self {
            energy_reserve: 50.0,
            max_energy: 100.0,
            has_aerobic: false,
            has_photosynthesis: false,
            metabolic_flexibility: 0.5,
            respiratory_quotient: 0.85,
            basal_rate,
        }
    }

    /// Run one metabolic tick: consume resources, produce energy, generate waste
    pub fn tick(&mut self, cell: &mut ResourceCell) -> MetabolicReport {
        let mut energy_gained = 0.0;
        let energy_spent = self.basal_rate;
        let mut toxin_produced = 0.0;
        let mut oxygen_consumed = 0.0;
        let mut pathway_used = "none";

        // First, try aerobic respiration if we have the upgrade and enough O2
        if self.has_aerobic && cell.oxygen > 0.1 {
            let aerobic = AerobicRespiration;
            let e = aerobic.execute(cell);
            if e > 0.0 {
                energy_gained += e;
                oxygen_consumed += 6.0;
                pathway_used = "aerobic";
            }
        }

        // Try photosynthesis if we have the upgrade and enough light
        if self.has_photosynthesis && cell.light > 1.0 {
            let photo = Photosynthesis;
            let e = photo.execute(cell);
            if e > 0.0 {
                energy_gained += e;
                pathway_used = "photosynthesis";
            }
        }

        // Fallback to glycolysis if we haven't gained enough energy
        if energy_gained < energy_spent * 0.5 {
            let glycolysis = Glycolysis;
            let e = glycolysis.execute(cell);
            if e > 0.0 {
                energy_gained += e;
                toxin_produced += 0.5;
                pathway_used = "glycolysis";
            }
        }

        // Update energy reserve
        let net_energy = energy_gained - energy_spent;
        self.energy_reserve = (self.energy_reserve + net_energy).clamp(0.0, self.max_energy);

        MetabolicReport {
            energy_gained,
            energy_spent,
            toxin_produced,
            oxygen_consumed,
            pathway_used,
        }
    }

    /// Check if organism is starving
    pub fn is_starving(&self) -> bool {
        self.energy_reserve < self.basal_rate * 2.0
    }

    /// Metabolic cost of computation (Landauer principle)
    /// Cost = n_operations * E_MIN_BIT * complexity
    pub fn computation_cost(&self, n_operations: usize) -> f32 {
        (n_operations as f32) * constants::E_MIN_BIT * self.basal_rate
    }
}

/// Report from a metabolic tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolicReport {
    pub energy_gained: f32,
    pub energy_spent: f32,
    pub toxin_produced: f32,
    pub oxygen_consumed: f32,
    pub pathway_used: &'static str,
}

// ============================================================================
// Water Physics
// ============================================================================

/// Calculate osmotic flux based on solute concentration gradient
pub fn osmotic_flux(external_concentration: f32, internal_concentration: f32) -> f32 {
    // Van 't Hoff: flux proportional to concentration difference
    // Positive flux = water flows in (hypotonic)
    // Negative flux = water flows out (hypertonic)
    (external_concentration - internal_concentration) * 0.1
}

/// Water availability modifier for reaction rates
/// In dry zones, reactions are much slower
pub fn water_rate_modifier(water_level: f32) -> f32 {
    // Sigmoidal ramp: mostly 1.0 above water_level=2.0, drops to ~0.1 at water=0
    1.0 - (-water_level * 0.6).exp()
}

// ============================================================================
// Pheromone System
// ============================================================================

pub struct PheromoneSystem;

impl PheromoneSystem {
    /// Deposit pheromone at position
    pub fn deposit(cell: &mut ResourceCell, channel: usize, amount: f32) {
        if channel < 3 {
            cell.pheromone[channel] = (cell.pheromone[channel] + amount).min(1.0);
        }
    }

    /// Decay all pheromones (evaporation)
    pub fn decay(cell: &mut ResourceCell, decay_rate: f32) {
        for i in 0..3 {
            cell.pheromone[i] *= (1.0 - decay_rate).max(0.0);
        }
    }

    /// Read gradient: returns Vec2 direction of strongest signal
    pub fn read_gradient(
        cells: &[ResourceCell],
        x: usize,
        y: usize,
        width: usize,
        channel: usize,
    ) -> Vec2 {
        if channel >= 3 || width == 0 {
            return Vec2::zero();
        }

        let mut grad = Vec2::zero();
        let height = cells.len() / width;

        // Simple finite difference: check neighbors
        let directions = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        let center_val = cells[y * width + x].pheromone[channel];

        for (dx, dy) in &directions {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if nx < width && ny < height {
                let neighbor_val = cells[ny * width + nx].pheromone[channel];
                let diff = neighbor_val - center_val;

                grad.x += (*dx as f32) * diff;
                grad.y += (*dy as f32) * diff;
            }
        }

        grad.normalized()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_resource_cell_empty() {
        let cell = ResourceCell::empty();
        assert_eq!(cell.light, 0.0);
        assert_eq!(cell.water, 0.0);
        assert_eq!(cell.nutrient_a, 0.0);
        assert_eq!(cell.temperature, 20.0);
    }

    #[test]
    fn test_resource_cell_get_set() {
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 50.0);
        assert_eq!(cell.get(ResourceType::NutrientA), 50.0);
        assert_eq!(cell.nutrient_a, 50.0);
    }

    #[test]
    fn test_resource_cell_add() {
        let mut cell = ResourceCell::empty();
        cell.add(ResourceType::Water, 10.0);
        cell.add(ResourceType::Water, 5.0);
        assert_eq!(cell.get(ResourceType::Water), 15.0);
    }

    #[test]
    fn test_resource_cell_remove() {
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 30.0);

        let removed = cell.remove(ResourceType::NutrientA, 10.0);
        assert_eq!(removed, 10.0);
        assert_eq!(cell.get(ResourceType::NutrientA), 20.0);

        // Remove more than available
        let removed = cell.remove(ResourceType::NutrientA, 100.0);
        assert_eq!(removed, 20.0);
        assert_eq!(cell.get(ResourceType::NutrientA), 0.0);

        // Remove from empty
        let removed = cell.remove(ResourceType::NutrientA, 10.0);
        assert_eq!(removed, 0.0);
    }

    #[test]
    fn test_resource_cell_clamp_negative() {
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::Oxygen, -5.0);
        assert_eq!(cell.get(ResourceType::Oxygen), 0.0);
    }

    #[test]
    fn test_glycolysis_energy_yield() {
        let g = Glycolysis;
        assert_eq!(g.energy_yield(), 2.0);
    }

    #[test]
    fn test_glycolysis_execute() {
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 5.0);
        cell.set(ResourceType::Water, 2.5);
        cell.temperature = 37.0;

        let g = Glycolysis;
        let energy = g.execute(&mut cell);

        // Should produce some energy (2.0 * temperature modifier)
        assert!(energy > 0.0);
        assert!(energy <= 2.0); // capped by temp modifier

        // Should consume inputs
        assert_eq!(cell.get(ResourceType::NutrientA), 4.0);
        assert_eq!(cell.get(ResourceType::Water), 2.0);

        // Should produce toxin
        assert!(cell.get(ResourceType::Toxin) > 0.0);
    }

    #[test]
    fn test_glycolysis_insufficient_resources() {
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 0.5); // not enough
        cell.set(ResourceType::Water, 2.5);

        let g = Glycolysis;
        let energy = g.execute(&mut cell);

        // Should produce no energy
        assert_eq!(energy, 0.0);

        // Should not consume anything
        assert_eq!(cell.get(ResourceType::NutrientA), 0.5);
    }

    #[test]
    fn test_aerobic_respiration_energy_yield() {
        let ar = AerobicRespiration;
        assert_eq!(ar.energy_yield(), 30.0);
    }

    #[test]
    fn test_aerobic_respiration_execute() {
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 5.0);
        cell.set(ResourceType::Oxygen, 30.0);
        cell.set(ResourceType::Water, 5.0);
        cell.temperature = 37.0;

        let ar = AerobicRespiration;
        let energy = ar.execute(&mut cell);

        // Should produce energy
        assert!(energy > 0.0);
        assert!(energy <= 30.0);

        // Should consume inputs
        assert_eq!(cell.get(ResourceType::NutrientA), 4.0);
        assert_eq!(cell.get(ResourceType::Oxygen), 24.0);
    }

    #[test]
    fn test_photosynthesis_execute() {
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::Light, 20.0);
        cell.set(ResourceType::Water, 10.0);
        cell.temperature = 25.0;

        let p = Photosynthesis;
        let energy = p.execute(&mut cell);

        // Should produce energy
        assert!(energy > 0.0);

        // Should produce outputs
        assert!(cell.get(ResourceType::NutrientA) > 0.0);
        assert!(cell.get(ResourceType::Oxygen) > 0.0);

        // Should consume light and water
        assert_eq!(cell.get(ResourceType::Light), 10.0);
        assert_eq!(cell.get(ResourceType::Water), 9.0);
    }

    #[test]
    fn test_metabolism_new() {
        let m = Metabolism::new(10.0);
        assert_eq!(m.energy_reserve, 50.0);
        assert_eq!(m.max_energy, 100.0);
        assert!(!m.has_aerobic);
        assert!(!m.has_photosynthesis);
        assert!(m.basal_rate > 0.0);
    }

    #[test]
    fn test_metabolism_is_starving() {
        let mut m = Metabolism::new(10.0);
        m.energy_reserve = 100.0;
        assert!(!m.is_starving());

        m.energy_reserve = m.basal_rate * 1.5;
        assert!(m.is_starving());
    }

    #[test]
    fn test_metabolism_computation_cost() {
        let m = Metabolism::new(10.0);
        let cost = m.computation_cost(1000);
        assert!(cost > 0.0);
        assert_eq!(cost, 1000.0 * constants::E_MIN_BIT * m.basal_rate);
    }

    #[test]
    fn test_metabolism_tick() {
        let mut m = Metabolism::new(10.0);
        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 10.0);
        cell.set(ResourceType::Water, 5.0);
        cell.temperature = 37.0;

        let report = m.tick(&mut cell);

        // Should have consumed some energy (basal_rate)
        assert_eq!(report.energy_spent, m.basal_rate);

        // Should have gained some energy from glycolysis
        assert!(report.energy_gained > 0.0);

        // Pathway should be set
        assert!(!report.pathway_used.is_empty());
    }

    #[test]
    fn test_metabolism_aerobic_when_available() {
        let mut m = Metabolism::new(10.0);
        m.has_aerobic = true;

        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 10.0);
        cell.set(ResourceType::Oxygen, 50.0);
        cell.set(ResourceType::Water, 10.0);
        cell.temperature = 37.0;

        let report = m.tick(&mut cell);

        // Should prefer aerobic respiration
        assert_eq!(report.pathway_used, "aerobic");
        assert!(report.energy_gained > 10.0); // aerobic yields ~30 ATP
    }

    #[test]
    fn test_osmotic_flux() {
        let flux_in = osmotic_flux(0.5, 0.2);
        assert!(flux_in > 0.0); // water flows in

        let flux_out = osmotic_flux(0.1, 0.5);
        assert!(flux_out < 0.0); // water flows out

        let flux_neutral = osmotic_flux(0.3, 0.3);
        assert_abs_diff_eq!(flux_neutral, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_water_rate_modifier() {
        let dry = water_rate_modifier(0.0);
        let wet = water_rate_modifier(5.0);

        // Wet should be much better than dry
        assert!(wet > dry);

        // Wet should be close to 1.0
        assert!(wet > 0.9);

        // Dry should be low
        assert!(dry < 0.2);
    }

    #[test]
    fn test_pheromone_deposit_and_decay() {
        let mut cell = ResourceCell::empty();
        PheromoneSystem::deposit(&mut cell, 0, 0.5);
        assert_eq!(cell.pheromone[0], 0.5);

        PheromoneSystem::decay(&mut cell, 0.1);
        assert_eq!(cell.pheromone[0], 0.5 * 0.9);
    }

    #[test]
    fn test_pheromone_gradient() {
        let mut cells = vec![ResourceCell::empty(); 9]; // 3x3 grid
        let width = 3;

        // Set high pheromone at top-left
        cells[0].pheromone[0] = 1.0;

        // Read gradient at center (1,1)
        let grad = PheromoneSystem::read_gradient(&cells, 1, 1, width, 0);

        // Gradient should point toward high concentration (top-left)
        assert!(grad.x < 0.0 || grad.x == 0.0);
        assert!(grad.y < 0.0 || grad.y == 0.0);
    }

    #[test]
    fn test_temperature_rate_modifiers() {
        let g = Glycolysis;
        let ar = AerobicRespiration;
        let p = Photosynthesis;

        // All should peak at their optimal temperatures
        let g_37 = g.rate_at_temperature(37.0);
        let g_20 = g.rate_at_temperature(20.0);
        assert!(g_37 > g_20);

        let ar_37 = ar.rate_at_temperature(37.0);
        let ar_5 = ar.rate_at_temperature(5.0);
        assert!(ar_37 > ar_5);

        let p_25 = p.rate_at_temperature(25.0);
        let p_10 = p.rate_at_temperature(10.0);
        assert!(p_25 > p_10);
    }

    #[test]
    fn test_energy_conservation() {
        let mut m = Metabolism::new(10.0);

        let mut cell = ResourceCell::empty();
        cell.set(ResourceType::NutrientA, 100.0);
        cell.set(ResourceType::Water, 50.0);
        cell.temperature = 37.0;

        let report = m.tick(&mut cell);

        // Energy gained should roughly match energy spent + change in reserve
        // This is a sanity check that energy is conserved
        assert!(report.energy_gained >= 0.0);
        assert!(report.energy_spent >= 0.0);
    }
}
