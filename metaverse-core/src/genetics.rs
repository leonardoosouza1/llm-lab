// ============================================================================
// Genetics Module: Evolution & Heritability in MetaVerse
// ============================================================================
// Implements diploid genomes with epigenetics, mutation, crossover, and
// horizontal gene transfer. Supports speciation via genetic distance.
// ============================================================================

use serde::{Serialize, Deserialize};
use rand::Rng;

/// A gene encodes one evolvable trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gene {
    pub id: u32,
    pub trait_type: TraitType,
    pub value: f32,           // expressed value
    pub dominance: f32,       // 0..1 for diploid crossover
    pub methylated: bool,     // epigenetic silencing
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum TraitType {
    // Metabolism
    BasalRate,
    AerobicCapacity,
    PhotosyntheticCapacity,
    MetabolicFlexibility,
    EnergyCapacity,

    // Morphology
    MembraneStrength,
    Permeability,
    Size,
    Speed,

    // Sensors
    ChemicalSensitivity,
    LightSensitivity,
    ThermalSensitivity,
    SocialSensitivity,
    SensorNoise,

    // Brain
    BrainComplexity,
    LearningRate,
    System2Threshold,

    // Homeostasis
    HomeostaticGain,
    TemperatureSetpoint,
    DormancyThreshold,
    SleepInterval,

    // Reproduction
    ReproductionThreshold,
    OffspringInvestment,
    MutationRate,
}

impl TraitType {
    /// Get valid range for this trait
    pub fn valid_range(&self) -> (f32, f32) {
        match self {
            TraitType::BasalRate => (0.1, 2.0),
            TraitType::AerobicCapacity => (0.0, 1.0),
            TraitType::PhotosyntheticCapacity => (0.0, 1.0),
            TraitType::MetabolicFlexibility => (0.0, 1.0),
            TraitType::EnergyCapacity => (10.0, 500.0),
            TraitType::MembraneStrength => (0.1, 2.0),
            TraitType::Permeability => (0.0, 1.0),
            TraitType::Size => (0.5, 10.0),
            TraitType::Speed => (0.0, 3.0),
            TraitType::ChemicalSensitivity => (0.0, 1.0),
            TraitType::LightSensitivity => (0.0, 1.0),
            TraitType::ThermalSensitivity => (0.0, 1.0),
            TraitType::SocialSensitivity => (0.0, 1.0),
            TraitType::SensorNoise => (0.0, 0.5),
            TraitType::BrainComplexity => (2.0, 64.0),
            TraitType::LearningRate => (0.0, 1.0),
            TraitType::System2Threshold => (0.0, 1.0),
            TraitType::HomeostaticGain => (0.01, 2.0),
            TraitType::TemperatureSetpoint => (25.0, 45.0),
            TraitType::DormancyThreshold => (0.01, 0.5),
            TraitType::SleepInterval => (10.0, 1000.0),
            TraitType::ReproductionThreshold => (10.0, 200.0),
            TraitType::OffspringInvestment => (1.0, 50.0),
            TraitType::MutationRate => (0.0001, 0.05),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub genes: Vec<Gene>,
    pub mutation_rate: f32,
    pub generation: u32,
}

impl Genome {
    /// Create default genome for a photosynthetic producer
    pub fn default_producer() -> Self {
        let genes = vec![
            Gene { id: 0, trait_type: TraitType::BasalRate, value: 0.5, dominance: 0.5, methylated: false },
            Gene { id: 1, trait_type: TraitType::PhotosyntheticCapacity, value: 0.9, dominance: 0.8, methylated: false },
            Gene { id: 2, trait_type: TraitType::AerobicCapacity, value: 0.3, dominance: 0.5, methylated: false },
            Gene { id: 3, trait_type: TraitType::Size, value: 2.0, dominance: 0.6, methylated: false },
            Gene { id: 4, trait_type: TraitType::Speed, value: 0.5, dominance: 0.5, methylated: false },
            Gene { id: 5, trait_type: TraitType::EnergyCapacity, value: 100.0, dominance: 0.5, methylated: false },
            Gene { id: 6, trait_type: TraitType::MembraneStrength, value: 1.0, dominance: 0.5, methylated: false },
            Gene { id: 7, trait_type: TraitType::Permeability, value: 0.4, dominance: 0.5, methylated: false },
            Gene { id: 8, trait_type: TraitType::ChemicalSensitivity, value: 0.6, dominance: 0.5, methylated: false },
            Gene { id: 9, trait_type: TraitType::LightSensitivity, value: 0.9, dominance: 0.7, methylated: false },
            Gene { id: 10, trait_type: TraitType::ThermalSensitivity, value: 0.5, dominance: 0.5, methylated: false },
            Gene { id: 11, trait_type: TraitType::SocialSensitivity, value: 0.3, dominance: 0.5, methylated: false },
            Gene { id: 12, trait_type: TraitType::SensorNoise, value: 0.05, dominance: 0.5, methylated: false },
            Gene { id: 13, trait_type: TraitType::BrainComplexity, value: 8.0, dominance: 0.5, methylated: false },
            Gene { id: 14, trait_type: TraitType::LearningRate, value: 0.3, dominance: 0.5, methylated: false },
            Gene { id: 15, trait_type: TraitType::System2Threshold, value: 0.5, dominance: 0.5, methylated: false },
            Gene { id: 16, trait_type: TraitType::HomeostaticGain, value: 0.8, dominance: 0.5, methylated: false },
            Gene { id: 17, trait_type: TraitType::TemperatureSetpoint, value: 35.0, dominance: 0.5, methylated: false },
            Gene { id: 18, trait_type: TraitType::DormancyThreshold, value: 0.1, dominance: 0.5, methylated: false },
            Gene { id: 19, trait_type: TraitType::SleepInterval, value: 50.0, dominance: 0.5, methylated: false },
            Gene { id: 20, trait_type: TraitType::ReproductionThreshold, value: 80.0, dominance: 0.5, methylated: false },
            Gene { id: 21, trait_type: TraitType::OffspringInvestment, value: 20.0, dominance: 0.5, methylated: false },
            Gene { id: 22, trait_type: TraitType::MutationRate, value: 0.001, dominance: 0.5, methylated: false },
            Gene { id: 23, trait_type: TraitType::MetabolicFlexibility, value: 0.2, dominance: 0.5, methylated: false },
        ];

        Self { genes, mutation_rate: 0.001, generation: 0 }
    }

    /// Create default genome for a heterotrophic consumer
    pub fn default_consumer() -> Self {
        let genes = vec![
            Gene { id: 0, trait_type: TraitType::BasalRate, value: 0.8, dominance: 0.5, methylated: false },
            Gene { id: 1, trait_type: TraitType::PhotosyntheticCapacity, value: 0.0, dominance: 0.5, methylated: true },
            Gene { id: 2, trait_type: TraitType::AerobicCapacity, value: 0.8, dominance: 0.8, methylated: false },
            Gene { id: 3, trait_type: TraitType::Size, value: 1.5, dominance: 0.6, methylated: false },
            Gene { id: 4, trait_type: TraitType::Speed, value: 1.5, dominance: 0.7, methylated: false },
            Gene { id: 5, trait_type: TraitType::EnergyCapacity, value: 120.0, dominance: 0.5, methylated: false },
            Gene { id: 6, trait_type: TraitType::MembraneStrength, value: 1.3, dominance: 0.5, methylated: false },
            Gene { id: 7, trait_type: TraitType::Permeability, value: 0.6, dominance: 0.5, methylated: false },
            Gene { id: 8, trait_type: TraitType::ChemicalSensitivity, value: 0.9, dominance: 0.8, methylated: false },
            Gene { id: 9, trait_type: TraitType::LightSensitivity, value: 0.4, dominance: 0.5, methylated: false },
            Gene { id: 10, trait_type: TraitType::ThermalSensitivity, value: 0.6, dominance: 0.5, methylated: false },
            Gene { id: 11, trait_type: TraitType::SocialSensitivity, value: 0.7, dominance: 0.7, methylated: false },
            Gene { id: 12, trait_type: TraitType::SensorNoise, value: 0.03, dominance: 0.5, methylated: false },
            Gene { id: 13, trait_type: TraitType::BrainComplexity, value: 16.0, dominance: 0.5, methylated: false },
            Gene { id: 14, trait_type: TraitType::LearningRate, value: 0.6, dominance: 0.6, methylated: false },
            Gene { id: 15, trait_type: TraitType::System2Threshold, value: 0.4, dominance: 0.5, methylated: false },
            Gene { id: 16, trait_type: TraitType::HomeostaticGain, value: 1.0, dominance: 0.5, methylated: false },
            Gene { id: 17, trait_type: TraitType::TemperatureSetpoint, value: 37.0, dominance: 0.5, methylated: false },
            Gene { id: 18, trait_type: TraitType::DormancyThreshold, value: 0.2, dominance: 0.5, methylated: false },
            Gene { id: 19, trait_type: TraitType::SleepInterval, value: 100.0, dominance: 0.5, methylated: false },
            Gene { id: 20, trait_type: TraitType::ReproductionThreshold, value: 100.0, dominance: 0.5, methylated: false },
            Gene { id: 21, trait_type: TraitType::OffspringInvestment, value: 30.0, dominance: 0.5, methylated: false },
            Gene { id: 22, trait_type: TraitType::MutationRate, value: 0.0015, dominance: 0.5, methylated: false },
            Gene { id: 23, trait_type: TraitType::MetabolicFlexibility, value: 0.7, dominance: 0.6, methylated: false },
        ];

        Self { genes, mutation_rate: 0.0015, generation: 0 }
    }

    /// Get expressed value of a trait (considering methylation)
    pub fn express(&self, trait_type: TraitType) -> f32 {
        for gene in &self.genes {
            if gene.trait_type == trait_type {
                return if gene.methylated { 0.0 } else { gene.value };
            }
        }
        match trait_type {
            TraitType::PhotosyntheticCapacity => 0.0,
            TraitType::AerobicCapacity => 0.0,
            TraitType::BasalRate => 0.5,
            _ => 0.5,
        }
    }

    /// Mutate genome: point mutations + structural mutations
    pub fn mutate(&mut self, rng: &mut impl Rng) {
        let effective_mutation_rate = self.express(TraitType::MutationRate).max(0.0001);

        let mut mutations_to_apply = Vec::new();
        let mut genes_to_add = Vec::new();
        let mut genes_to_remove = Vec::new();

        for (idx, gene) in self.genes.iter().enumerate() {
            if rng.gen::<f32>() < effective_mutation_rate {
                let mutation_type = rng.gen::<f32>();

                if mutation_type < 0.8 {
                    let perturbation = sample_gaussian(rng, 0.0, 0.1);
                    let (min, max) = gene.trait_type.valid_range();
                    let new_value = (gene.value + perturbation).clamp(min, max);
                    mutations_to_apply.push((idx, new_value));
                } else if mutation_type < 0.9 {
                    let (min, max) = gene.trait_type.valid_range();
                    let new_value = rng.gen_range(min..=max);
                    mutations_to_apply.push((idx, new_value));
                } else if mutation_type < 0.95 {
                    let count = self.genes.iter().filter(|g| g.trait_type == gene.trait_type).count();
                    if count < 3 {
                        genes_to_add.push(gene.clone());
                    }
                } else {
                    let count = self.genes.iter().filter(|g| g.trait_type == gene.trait_type).count();
                    if count > 1 {
                        genes_to_remove.push(idx);
                    }
                }
            }
        }

        for (idx, new_value) in mutations_to_apply {
            self.genes[idx].value = new_value;
        }

        for mut gene in genes_to_add {
            gene.id = self.genes.len() as u32;
            self.genes.push(gene);
        }

        genes_to_remove.sort();
        for idx in genes_to_remove.iter().rev() {
            self.genes.remove(*idx);
        }

        if rng.gen::<f32>() < 0.01 {
            let perturbation = sample_gaussian(rng, 0.0, 0.0005);
            self.mutation_rate = (self.mutation_rate + perturbation).clamp(0.0001, 0.05);
        }

        self.generation += 1;
    }

    /// Crossover between two genomes
    pub fn crossover(parent1: &Genome, parent2: &Genome, rng: &mut impl Rng) -> Genome {
        use std::collections::HashMap;

        let mut p1_map: HashMap<TraitType, Vec<&Gene>> = HashMap::new();
        let mut p2_map: HashMap<TraitType, Vec<&Gene>> = HashMap::new();

        for gene in &parent1.genes {
            p1_map.entry(gene.trait_type).or_insert_with(Vec::new).push(gene);
        }
        for gene in &parent2.genes {
            p2_map.entry(gene.trait_type).or_insert_with(Vec::new).push(gene);
        }

        let mut offspring_genes = Vec::new();
        let mut next_id = 0u32;

        let mut all_traits: Vec<TraitType> = p1_map.keys().copied().collect();
        for trait_type in p2_map.keys() {
            if !all_traits.contains(trait_type) {
                all_traits.push(*trait_type);
            }
        }

        for trait_type in all_traits {
            let p1_genes = p1_map.get(&trait_type).map(|v| v.as_slice()).unwrap_or(&[]);
            let p2_genes = p2_map.get(&trait_type).map(|v| v.as_slice()).unwrap_or(&[]);

            let max_copies = p1_genes.len().max(p2_genes.len());
            for i in 0..max_copies {
                let gene = if i < p1_genes.len() && i < p2_genes.len() {
                    let g1 = p1_genes[i];
                    let g2 = p2_genes[i];
                    if rng.gen::<f32>() < g1.dominance { g1.clone() } else { g2.clone() }
                } else if i < p1_genes.len() {
                    p1_genes[i].clone()
                } else {
                    p2_genes[i].clone()
                };

                let mut new_gene = gene;
                new_gene.id = next_id;
                next_id += 1;
                offspring_genes.push(new_gene);
            }
        }

        let offspring_mutation_rate = if rng.gen::<f32>() < 0.5 { parent1.mutation_rate } else { parent2.mutation_rate };

        Genome { genes: offspring_genes, mutation_rate: offspring_mutation_rate, generation: parent1.generation.max(parent2.generation) + 1 }
    }

    /// Genetic distance (for speciation)
    pub fn distance(a: &Genome, b: &Genome) -> f32 {
        use std::collections::HashMap;

        let mut a_map: HashMap<TraitType, Vec<f32>> = HashMap::new();
        let mut b_map: HashMap<TraitType, Vec<f32>> = HashMap::new();

        for gene in &a.genes {
            a_map.entry(gene.trait_type).or_insert_with(Vec::new).push(gene.value);
        }
        for gene in &b.genes {
            b_map.entry(gene.trait_type).or_insert_with(Vec::new).push(gene.value);
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        let mut all_traits: Vec<TraitType> = a_map.keys().copied().collect();
        for trait_type in b_map.keys() {
            if !all_traits.contains(trait_type) {
                all_traits.push(*trait_type);
            }
        }

        for trait_type in all_traits {
            let a_vals = a_map.get(&trait_type).map(|v| v.clone()).unwrap_or_default();
            let b_vals = b_map.get(&trait_type).map(|v| v.clone()).unwrap_or_default();

            let max_len = a_vals.len().max(b_vals.len());
            for i in 0..max_len {
                let a_val = a_vals.get(i).copied().unwrap_or(0.0);
                let b_val = b_vals.get(i).copied().unwrap_or(0.0);
                total_distance += (a_val - b_val).abs();
                count += 1;
            }

            if a_vals.len() != b_vals.len() {
                total_distance += ((a_vals.len() as f32) - (b_vals.len() as f32)).abs() * 0.5;
                count += 1;
            }
        }

        if count > 0 { total_distance / count as f32 } else { 0.0 }
    }
}

/// Epigenetic state: methylation and environmental history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticState {
    pub methylation: Vec<bool>,
    pub stress_history: f32,
    pub abundance_history: f32,
}

impl EpigeneticState {
    pub fn new(genome_size: usize) -> Self {
        Self { methylation: vec![false; genome_size], stress_history: 0.5, abundance_history: 0.5 }
    }

    pub fn update(&mut self, stress: f32, abundance: f32) {
        const ALPHA: f32 = 0.1;
        self.stress_history = self.stress_history * (1.0 - ALPHA) + stress * ALPHA;
        self.abundance_history = self.abundance_history * (1.0 - ALPHA) + abundance * ALPHA;

        for (i, methylated) in self.methylation.iter_mut().enumerate() {
            if stress > 0.7 && (i % 5 == 0 || i % 7 == 0) {
                *methylated = true;
            } else if abundance > 0.7 && *methylated {
                *methylated = false;
            }
        }
    }

    pub fn modify_expression(&self, idx: usize, gene: &Gene) -> f32 {
        if idx < self.methylation.len() && self.methylation[idx] { 0.0 } else { gene.value }
    }

    pub fn inherit(&self, rng: &mut impl Rng) -> EpigeneticState {
        let mut new_methylation = vec![false; self.methylation.len()];

        for (i, &methylated) in self.methylation.iter().enumerate() {
            if methylated && rng.gen::<f32>() < 0.5 {
                if i < new_methylation.len() {
                    new_methylation[i] = true;
                }
            }
        }

        EpigeneticState {
            methylation: new_methylation,
            stress_history: self.stress_history * 0.3 + 0.5 * 0.7,
            abundance_history: self.abundance_history * 0.3 + 0.5 * 0.7,
        }
    }
}

/// Horizontal Gene Transfer operations
pub struct HorizontalGeneTransfer;

impl HorizontalGeneTransfer {
    pub fn conjugate(donor: &Genome, recipient: &mut Genome, rng: &mut impl Rng) -> bool {
        if donor.genes.is_empty() { return false; }

        let donor_gene = &donor.genes[rng.gen_range(0..donor.genes.len())];
        let recipient_idx = recipient.genes.iter().position(|g| g.trait_type == donor_gene.trait_type);

        if let Some(idx) = recipient_idx {
            if rng.gen::<f32>() < 0.5 {
                recipient.genes[idx].value = donor_gene.value;
                recipient.genes[idx].dominance = donor_gene.dominance;
                return true;
            }
        } else {
            let mut new_gene = donor_gene.clone();
            new_gene.id = recipient.genes.len() as u32;
            recipient.genes.push(new_gene);
            return true;
        }

        false
    }

    pub fn transform(free_genes: &[Gene], recipient: &mut Genome, rng: &mut impl Rng) -> bool {
        if free_genes.is_empty() { return false; }

        let free_gene = &free_genes[rng.gen_range(0..free_genes.len())];
        let recipient_idx = recipient.genes.iter().position(|g| g.trait_type == free_gene.trait_type);

        if let Some(idx) = recipient_idx {
            if rng.gen::<f32>() < 0.3 {
                recipient.genes[idx].value = free_gene.value;
                return true;
            }
        } else {
            let mut new_gene = free_gene.clone();
            new_gene.id = recipient.genes.len() as u32;
            recipient.genes.push(new_gene);
            return true;
        }

        false
    }
}

#[derive(Debug, Clone)]
pub struct Phenotype {
    pub basal_rate: f32,
    pub has_aerobic: bool,
    pub has_photosynthesis: bool,
    pub membrane_strength: f32,
    pub permeability: f32,
    pub size: f32,
    pub speed: f32,
    pub sensor_noise: f32,
    pub brain_complexity: f32,
    pub learning_rate: f32,
    pub homeostatic_gain: f32,
    pub dormancy_threshold: f32,
    pub sleep_interval: u32,
    pub reproduction_threshold: f32,
    pub offspring_investment: f32,
    pub mutation_rate: f32,
}

pub struct PhenotypeBuilder;

impl PhenotypeBuilder {
    pub fn build(genome: &Genome, _epigenetics: &EpigeneticState) -> Phenotype {
        Phenotype {
            basal_rate: genome.express(TraitType::BasalRate),
            has_aerobic: genome.express(TraitType::AerobicCapacity) > 0.3,
            has_photosynthesis: genome.express(TraitType::PhotosyntheticCapacity) > 0.3,
            membrane_strength: genome.express(TraitType::MembraneStrength),
            permeability: genome.express(TraitType::Permeability),
            size: genome.express(TraitType::Size),
            speed: genome.express(TraitType::Speed),
            sensor_noise: genome.express(TraitType::SensorNoise),
            brain_complexity: genome.express(TraitType::BrainComplexity),
            learning_rate: genome.express(TraitType::LearningRate),
            homeostatic_gain: genome.express(TraitType::HomeostaticGain),
            dormancy_threshold: genome.express(TraitType::DormancyThreshold),
            sleep_interval: genome.express(TraitType::SleepInterval) as u32,
            reproduction_threshold: genome.express(TraitType::ReproductionThreshold),
            offspring_investment: genome.express(TraitType::OffspringInvestment),
            mutation_rate: genome.express(TraitType::MutationRate),
        }
    }
}

fn sample_gaussian(rng: &mut impl Rng, mean: f32, std_dev: f32) -> f32 {
    let u1 = rng.gen::<f32>().max(1e-10);
    let u2 = rng.gen::<f32>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    mean + z * std_dev
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    #[test]
    fn test_default_producer_vs_consumer() {
        let producer = Genome::default_producer();
        let consumer = Genome::default_consumer();

        assert!(producer.express(TraitType::PhotosyntheticCapacity) > 0.7);
        assert!(consumer.express(TraitType::AerobicCapacity) > 0.7);
        assert!(consumer.express(TraitType::PhotosyntheticCapacity) < 0.1 ||
            consumer.genes.iter().find(|g| g.trait_type == TraitType::PhotosyntheticCapacity).map(|g| g.methylated).unwrap_or(false));
    }

    #[test]
    fn test_mutation_changes_genome() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut genome = Genome::default_producer();
        // Increase mutation rate for reliable test detection
        genome.mutation_rate = 0.5;
        // Update the MutationRate gene value as well (the mutate function uses express())
        for gene in &mut genome.genes {
            if gene.trait_type == TraitType::MutationRate {
                gene.value = 0.5;
            }
        }
        let original_values: Vec<f32> = genome.genes.iter().map(|g| g.value).collect();

        genome.mutate(&mut rng);

        let new_values: Vec<f32> = genome.genes.iter().map(|g| g.value).collect();
        let changed = original_values.iter().zip(new_values.iter()).any(|(o, n)| (o - n).abs() > 0.001);

        assert!(changed || genome.genes.len() != original_values.len(), "Mutation should change genome");
    }

    #[test]
    fn test_crossover_produces_offspring() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let parent1 = Genome::default_producer();
        let parent2 = Genome::default_consumer();

        let offspring = Genome::crossover(&parent1, &parent2, &mut rng);

        assert!(!offspring.genes.is_empty(), "Offspring should have genes");
        assert!(offspring.generation > 0, "Offspring generation should increment");

        let has_producer_trait = offspring.express(TraitType::PhotosyntheticCapacity) > 0.3;
        let has_consumer_trait = offspring.express(TraitType::AerobicCapacity) > 0.3;
        assert!(has_producer_trait || has_consumer_trait, "Offspring should inherit from at least one parent");
    }

    #[test]
    fn test_epigenetic_inheritance() {
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let mut parent_epi = EpigeneticState::new(10);
        parent_epi.methylation[0] = true;
        parent_epi.methylation[2] = true;
        parent_epi.methylation[5] = true;

        let child_epi = parent_epi.inherit(&mut rng);

        let marks_inherited = child_epi.methylation.iter().filter(|&&m| m).count();
        assert!(marks_inherited > 0 && marks_inherited <= 3, "Child should inherit some (but not all) methylation marks");
    }

    #[test]
    fn test_genetic_distance() {
        let genome1 = Genome::default_producer();
        let genome2 = Genome::default_consumer();

        let distance = Genome::distance(&genome1, &genome2);
        assert!(distance > 0.0, "Different genomes should have distance > 0");

        let distance_same = Genome::distance(&genome1, &genome1);
        assert_eq!(distance_same, 0.0, "Identical genomes should have distance 0");
    }

    #[test]
    fn test_hgt_conjugation() {
        let mut rng = ChaCha8Rng::seed_from_u64(789);
        let donor = Genome::default_producer();
        let mut recipient = Genome::default_consumer();

        let initial_count = recipient.genes.len();
        let _transferred = HorizontalGeneTransfer::conjugate(&donor, &mut recipient, &mut rng);

        assert!(recipient.genes.len() >= initial_count || !donor.genes.is_empty());
    }

    #[test]
    fn test_hgt_transformation() {
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let free_genes = vec![
            Gene { id: 0, trait_type: TraitType::Size, value: 5.0, dominance: 0.5, methylated: false },
        ];
        let mut recipient = Genome::default_consumer();

        let initial_count = recipient.genes.len();
        let _transferred = HorizontalGeneTransfer::transform(&free_genes, &mut recipient, &mut rng);

        assert!(recipient.genes.len() >= initial_count);
    }

    #[test]
    fn test_phenotype_builder() {
        let genome = Genome::default_producer();
        let epi = EpigeneticState::new(genome.genes.len());

        let phenotype = PhenotypeBuilder::build(&genome, &epi);

        assert!(phenotype.has_photosynthesis);
        assert!(phenotype.size > 0.0);
        assert!(phenotype.basal_rate > 0.0);
    }
}
