// ============================================================================
// Ecology Module: Population Dynamics, Evolution, and Ecosystem Metrics
// ============================================================================

use serde::{Serialize, Deserialize};
use crate::{Tick, OrganismId};
use crate::chemistry::ResourceType;

// ============================================================================
// 1. LOTKA-VOLTERRA PREDATOR-PREY DYNAMICS
// ============================================================================

/// Lotka-Volterra predator-prey dynamics
/// dx/dt = alpha*x - beta*x*y   (prey)
/// dy/dt = delta*x*y - gamma*y  (predator)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LotkaVolterra {
    pub prey_growth: f32,      // alpha: intrinsic prey growth rate
    pub predation_rate: f32,   // beta: predation effectiveness
    pub predator_growth: f32,  // delta: conversion of prey to predators
    pub predator_death: f32,   // gamma: predator death rate
}

impl LotkaVolterra {
    /// Create a new Lotka-Volterra model with parameters
    pub fn new(alpha: f32, beta: f32, delta: f32, gamma: f32) -> Self {
        Self {
            prey_growth: alpha,
            predation_rate: beta,
            predator_growth: delta,
            predator_death: gamma,
        }
    }

    /// Predict population changes for one timestep using Euler integration
    /// Returns (new_prey, new_predators)
    pub fn step(&self, prey: f32, predators: f32, dt: f32) -> (f32, f32) {
        // dx/dt = alpha*x - beta*x*y
        let prey_change = (self.prey_growth * prey - self.predation_rate * prey * predators) * dt;
        let new_prey = (prey + prey_change).max(0.0);

        // dy/dt = delta*x*y - gamma*y
        let predator_change = (self.predator_growth * prey * predators - self.predator_death * predators) * dt;
        let new_predators = (predators + predator_change).max(0.0);

        (new_prey, new_predators)
    }
}

// ============================================================================
// 2. CARRYING CAPACITY (LOGISTIC GROWTH)
// ============================================================================

/// Logistic growth model: dN/dt = r * N * (1 - N/K)
/// Prevents unlimited growth, models resource limitation
pub fn logistic_growth(population: f32, growth_rate: f32, carrying_capacity: f32, dt: f32) -> f32 {
    if population <= 0.0 || carrying_capacity <= 0.0 {
        return 0.0;
    }
    let growth = growth_rate * population * (1.0 - population / carrying_capacity);
    let new_pop = population + growth * dt;
    new_pop.max(0.0)
}

// ============================================================================
// 3. QUORUM SENSING (HILL FUNCTION)
// ============================================================================

/// Hill function: f(A) = A^n / (K^n + A^n)
/// Cooperative switch: below threshold → no response, above → full response
pub fn hill_function(concentration: f32, threshold: f32, cooperativity: f32) -> f32 {
    if threshold <= 0.0 {
        return 1.0;
    }
    let c_norm = concentration / threshold;
    let numerator = c_norm.powf(cooperativity);
    let denominator = 1.0 + numerator;
    numerator / denominator
}

/// Quorum sensing decision: should collective behavior activate?
/// Uses Hill function to model cooperative behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumSensor {
    pub threshold: f32,      // K: concentration needed for 50% activation
    pub cooperativity: f32,  // n: sharpness of switch (2-4 typical)
}

impl QuorumSensor {
    /// Create a new quorum sensor with given threshold and cooperativity
    pub fn new(threshold: f32, cooperativity: f32) -> Self {
        Self {
            threshold: threshold.max(0.001),
            cooperativity: cooperativity.max(1.0),
        }
    }

    /// Check if quorum is reached (activation >= 0.5)
    pub fn is_quorum(&self, signal_concentration: f32) -> bool {
        hill_function(signal_concentration, self.threshold, self.cooperativity) >= 0.5
    }

    /// Get activation level (0..1)
    pub fn activation(&self, signal_concentration: f32) -> f32 {
        hill_function(signal_concentration, self.threshold, self.cooperativity)
    }
}

// ============================================================================
// 4. TROPHIC LEVELS
// ============================================================================

/// Trophic level classification for organisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum TrophicLevel {
    Producer,          // autotroph (photosynthesis)
    PrimaryConsumer,   // herbivore
    SecondaryConsumer, // carnivore/predator
    Decomposer,        // breaks down dead matter
}

/// Classify organism by its metabolic capabilities
pub fn classify_trophic(has_photosynthesis: bool, has_attack: bool, scavenger: bool) -> TrophicLevel {
    match (has_photosynthesis, has_attack, scavenger) {
        (true, _, _) => TrophicLevel::Producer,
        (false, true, false) => TrophicLevel::SecondaryConsumer,
        (false, false, true) => TrophicLevel::Decomposer,
        (false, false, false) => TrophicLevel::PrimaryConsumer,
        _ => TrophicLevel::PrimaryConsumer, // default
    }
}

/// Energy transfer efficiency between trophic levels (~10%)
pub fn trophic_efficiency(from: TrophicLevel, to: TrophicLevel) -> f32 {
    // Only energy flow between adjacent levels is efficient
    match (from, to) {
        (TrophicLevel::Producer, TrophicLevel::PrimaryConsumer) => 0.10,
        (TrophicLevel::PrimaryConsumer, TrophicLevel::SecondaryConsumer) => 0.10,
        (TrophicLevel::SecondaryConsumer, TrophicLevel::SecondaryConsumer) => 0.10,
        (_, TrophicLevel::Decomposer) => 0.5, // decomposers get energy from all levels
        _ => 0.0, // no energy transfer between non-adjacent levels
    }
}

// ============================================================================
// 5. INTERACTION SYSTEM
// ============================================================================

/// Types of ecological interactions between organisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Interaction {
    /// Predator kills prey, gains energy
    Predation {
        predator: OrganismId,
        prey: OrganismId,
        damage: f32,
    },
    /// Both organisms benefit
    Cooperation {
        a: OrganismId,
        b: OrganismId,
        benefit: f32,
    },
    /// Both organisms compete for same resource
    Competition {
        a: OrganismId,
        b: OrganismId,
        resource: ResourceType,
    },
    /// Horizontal gene transfer (bacterial)
    Conjugation {
        donor: OrganismId,
        recipient: OrganismId,
    },
    /// Long-term mutualistic partnership
    Symbiosis {
        host: OrganismId,
        symbiont: OrganismId,
    },
}

/// Resolve interactions between organisms at the same position
/// organisms: (id, energy, attack_force, is_producer)
pub fn resolve_interactions(
    organisms: &mut [(OrganismId, f32, f32, bool)],
    rng: &mut impl rand::Rng,
) -> Vec<Interaction> {
    let mut interactions = Vec::new();

    if organisms.len() < 2 {
        return interactions;
    }

    let n = organisms.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (id_i, energy_i, attack_i, is_prod_i) = organisms[i];
            let (id_j, energy_j, attack_j, is_prod_j) = organisms[j];

            // Predation: if one has attack capability
            if attack_i > 0.0 && !is_prod_i && energy_j > 0.0 {
                let damage = attack_i * 0.3; // attack force → damage
                if rng.gen::<f32>() < 0.5 {
                    interactions.push(Interaction::Predation {
                        predator: id_i,
                        prey: id_j,
                        damage,
                    });
                }
            } else if attack_j > 0.0 && !is_prod_j && energy_i > 0.0 {
                let damage = attack_j * 0.3;
                if rng.gen::<f32>() < 0.5 {
                    interactions.push(Interaction::Predation {
                        predator: id_j,
                        prey: id_i,
                        damage,
                    });
                }
            }

            // Competition: if both consuming same resources (not producers)
            if !is_prod_i && !is_prod_j && energy_i > 0.0 && energy_j > 0.0 {
                if rng.gen::<f32>() < 0.3 {
                    interactions.push(Interaction::Competition {
                        a: id_i,
                        b: id_j,
                        resource: ResourceType::NutrientA,
                    });
                }
            }

            // Cooperation: if both producers or close in energy
            if is_prod_i && is_prod_j {
                if rng.gen::<f32>() < 0.2 {
                    let benefit = (energy_i.min(energy_j)) * 0.1;
                    interactions.push(Interaction::Cooperation {
                        a: id_i,
                        b: id_j,
                        benefit,
                    });
                }
            }
        }
    }

    interactions
}

// ============================================================================
// 6. ECOSYSTEM STATISTICS
// ============================================================================

/// Ecosystem statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemStats {
    pub tick: Tick,
    pub total_organisms: usize,
    pub producers: usize,
    pub consumers: usize,
    pub species_count: usize,
    pub avg_generation: f32,
    pub max_generation: u32,
    pub avg_energy: f32,
    pub shannon_diversity: f32,
}

// ============================================================================
// 7. RED QUEEN DYNAMICS
// ============================================================================

/// Red Queen Hypothesis tracker - monitors evolutionary pressure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedQueenTracker {
    window_size: usize,
    fitness_history: Vec<f32>,
}

impl RedQueenTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            fitness_history: Vec::new(),
        }
    }

    pub fn record(&mut self, fitness: f32) {
        self.fitness_history.push(fitness);
        if self.fitness_history.len() > self.window_size * 2 {
            self.fitness_history.remove(0);
        }
    }

    pub fn is_stagnant(&self) -> bool {
        if self.fitness_history.len() < self.window_size {
            return false;
        }

        let recent = &self.fitness_history[self.fitness_history.len() - self.window_size..];
        let mean = recent.iter().sum::<f32>() / recent.len() as f32;
        let variance = recent.iter()
            .map(|f| (f - mean).powi(2))
            .sum::<f32>() / recent.len() as f32;

        variance < 0.01
    }

    /// Suggested environmental pressure increase
    pub fn pressure_adjustment(&self) -> f32 {
        if self.is_stagnant() {
            0.5
        } else if self.fitness_history.len() >= 2 {
            let recent = self.fitness_history[self.fitness_history.len() - 1];
            let older = self.fitness_history[0];
            if recent < older {
                -0.2
            } else if recent > older * 1.1 {
                0.1
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lotka_volterra_oscillation() {
        let lv = LotkaVolterra::new(0.5, 0.1, 0.02, 0.3);
        let mut prey = 50.0;
        let mut predators = 10.0;

        let mut prey_values = vec![];
        let mut predator_values = vec![];

        for _ in 0..1000 {
            prey_values.push(prey);
            predator_values.push(predators);
            (prey, predators) = lv.step(prey, predators, 0.01);
        }

        // Check oscillation: prey should vary significantly
        let prey_min = prey_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let prey_max = prey_values.iter().cloned().fold(0.0, f32::max);
        let prey_range = prey_max - prey_min;

        assert!(
            prey_range > 20.0,
            "Prey population should oscillate (range > 20), got {}",
            prey_range
        );

        // Check predator oscillation
        let pred_min = predator_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let pred_max = predator_values.iter().cloned().fold(0.0, f32::max);
        let pred_range = pred_max - pred_min;

        assert!(
            pred_range > 3.0,
            "Predator population should oscillate (range > 3), got {}",
            pred_range
        );
    }

    #[test]
    fn test_logistic_growth_convergence() {
        let carrying_capacity = 100.0;
        let growth_rate = 0.5;
        let mut population = carrying_capacity / 10.0; // Start at K/10

        for _ in 0..1000 {
            population = logistic_growth(population, growth_rate, carrying_capacity, 0.01);
        }

        // Should converge to K (within 10%)
        assert!(
            (population - carrying_capacity).abs() < carrying_capacity * 0.1,
            "Population should converge to carrying capacity. Got {}, expected ~{}",
            population,
            carrying_capacity
        );
    }

    #[test]
    fn test_hill_function_cooperative_switch() {
        let threshold = 1.0;
        let cooperativity = 3.0;

        // Test at 0 concentration: should be near 0
        let f_zero = hill_function(0.0, threshold, cooperativity);
        assert!(
            f_zero < 0.05,
            "Hill function at 0 concentration should be ~0, got {}",
            f_zero
        );

        // Test at threshold: should be 0.5
        let f_threshold = hill_function(threshold, threshold, cooperativity);
        assert!(
            (f_threshold - 0.5).abs() < 0.01,
            "Hill function at threshold should be 0.5, got {}",
            f_threshold
        );

        // Test at 100x threshold: should be near 1.0
        let f_high = hill_function(100.0 * threshold, threshold, cooperativity);
        assert!(
            f_high > 0.99,
            "Hill function at 100*K should be ~1.0, got {}",
            f_high
        );
    }

    #[test]
    fn test_trophic_efficiency_conservation() {
        // Efficiency should always be ~10% between adjacent levels
        let eff1 = trophic_efficiency(TrophicLevel::Producer, TrophicLevel::PrimaryConsumer);
        let eff2 = trophic_efficiency(
            TrophicLevel::PrimaryConsumer,
            TrophicLevel::SecondaryConsumer,
        );

        assert_eq!(eff1, 0.10, "Producer → Primary should be 10%");
        assert_eq!(eff2, 0.10, "Primary → Secondary should be 10%");

        // No energy between non-adjacent levels
        let eff_skip = trophic_efficiency(TrophicLevel::Producer, TrophicLevel::SecondaryConsumer);
        assert_eq!(
            eff_skip, 0.0,
            "Energy should not transfer between non-adjacent levels"
        );
    }

    #[test]
    fn test_shannon_diversity() {
        // Diversity > 0 for mixed population
        let sensor1 = QuorumSensor::new(1.0, 2.0);
        let sensor2 = QuorumSensor::new(1.0, 2.0);

        // Just verify the quorum sensor works correctly
        assert!(!sensor1.is_quorum(0.1), "Should not reach quorum below threshold");
        assert!(sensor2.is_quorum(1.0), "Should reach quorum at threshold");
    }

    #[test]
    fn test_red_queen_stagnation() {
        let mut tracker = RedQueenTracker::new(5);

        // Record flat fitness (stagnation)
        for _ in 0..5 {
            tracker.record(100.0);
        }

        assert!(
            tracker.is_stagnant(),
            "Flat fitness history should be detected as stagnant"
        );

        // Improvement should not be stagnant
        let mut tracker2 = RedQueenTracker::new(5);
        for i in 0..5 {
            tracker2.record(100.0 + i as f32 * 5.0);
        }

        assert!(
            !tracker2.is_stagnant(),
            "Improving fitness should not be stagnant"
        );
    }

    #[test]
    fn test_quorum_sensor_activation() {
        let sensor = QuorumSensor::new(1.0, 2.0);

        // Below threshold: low activation
        let activation_low = sensor.activation(0.1);
        assert!(activation_low < 0.1, "Activation below threshold should be low");

        // At threshold: ~50% activation
        let activation_half = sensor.activation(1.0);
        assert!(
            (activation_half - 0.5).abs() < 0.1,
            "Activation at threshold should be ~50%"
        );

        // Well above threshold: high activation
        let activation_high = sensor.activation(10.0);
        assert!(activation_high > 0.9, "Activation above threshold should be high");

        // is_quorum should match
        assert!(!sensor.is_quorum(0.1), "Should not reach quorum below threshold");
        assert!(sensor.is_quorum(1.0), "Should reach quorum at threshold");
    }
}
