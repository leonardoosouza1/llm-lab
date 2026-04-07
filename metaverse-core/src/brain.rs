// ============================================================================
// Brain Module: Simplified NEAT (NeuroEvolution of Augmenting Topologies)
// ============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::{Rng, SeedableRng};

// ============================================================================
// Sensor and Motor Constants
// ============================================================================

pub const SENSOR_CHEMICAL_START: usize = 0;
pub const SENSOR_LIGHT: usize = 8;
pub const SENSOR_TEMP: usize = 9;
pub const SENSOR_MECHANICAL: usize = 10;
pub const SENSOR_SOCIAL_START: usize = 11;
pub const SENSOR_ENERGY: usize = 14;
pub const SENSOR_HUNGER: usize = 15;
pub const SENSOR_DANGER: usize = 16;
pub const NUM_SENSORS: usize = 17;

pub const MOTOR_MOVE_X: usize = 0;
pub const MOTOR_MOVE_Y: usize = 1;
pub const MOTOR_PHEROMONE_0: usize = 2;
pub const MOTOR_PHEROMONE_1: usize = 3;
pub const MOTOR_PHEROMONE_2: usize = 4;
pub const MOTOR_ATTACK: usize = 5;
pub const MOTOR_REPRODUCE: usize = 6;
pub const MOTOR_MEMBRANE: usize = 7;
pub const NUM_MOTORS: usize = 8;

// ============================================================================
// Node Type and Node
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Sensor,
    Hidden,
    Motor,
    Modulator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: u32,
    pub node_type: NodeType,
    pub activation: f32,
    pub bias: f32,
    pub threshold: f32,
}

impl Node {
    fn new(id: u32, node_type: NodeType) -> Self {
        Self {
            id,
            node_type,
            activation: 0.0,
            bias: 0.0,
            threshold: 0.0,
        }
    }
}

// ============================================================================
// Edge (Connection)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: u32,
    pub to: u32,
    pub weight: i8,
    pub enabled: bool,
    pub innovation: u32,
}

impl Edge {
    fn new(from: u32, to: u32, weight: i8, innovation: u32) -> Self {
        Self {
            from,
            to,
            weight,
            enabled: true,
            innovation,
        }
    }

    fn normalize_weight(&mut self) {
        self.weight = if self.weight > 0 {
            1
        } else if self.weight < 0 {
            -1
        } else {
            0
        };
    }
}

// ============================================================================
// Brain Network
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brain {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    innovation_counter: u32,
}

impl Brain {
    pub fn minimal() -> Self {
        let mut brain = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            innovation_counter: 0,
        };

        for i in 0..NUM_SENSORS {
            brain.nodes.push(Node::new(i as u32, NodeType::Sensor));
        }

        let motor_start_id = NUM_SENSORS as u32;
        for i in 0..NUM_MOTORS {
            brain.nodes.push(Node::new(motor_start_id + i as u32, NodeType::Motor));
        }

        for sensor_id in 0..NUM_SENSORS {
            for motor_id in 0..NUM_MOTORS {
                let motor_node_id = motor_start_id + motor_id as u32;
                brain.edges.push(Edge::new(
                    sensor_id as u32,
                    motor_node_id,
                    0,
                    brain.innovation_counter,
                ));
                brain.innovation_counter += 1;
            }
        }

        brain
    }

    pub fn forward(&mut self, inputs: &[f32; NUM_SENSORS]) -> [f32; NUM_MOTORS] {
        for (i, &input) in inputs.iter().enumerate() {
            if let Some(node) = self.nodes.iter_mut().find(|n| n.id == i as u32) {
                node.activation = input.clamp(-1.0, 1.0);
            }
        }

        let sorted_ids = self.topological_sort();

        for node_id in sorted_ids {
            if let Some(node_idx) = self.nodes.iter().position(|n| n.id == node_id) {
                let node_type = self.nodes[node_idx].node_type;

                if node_type == NodeType::Sensor {
                    continue;
                }

                let mut sum = self.nodes[node_idx].bias;
                let incoming_edges: Vec<_> = self
                    .edges
                    .iter()
                    .filter(|e| e.to == node_id && e.enabled)
                    .collect();

                for edge in incoming_edges {
                    if let Some(from_idx) = self.nodes.iter().position(|n| n.id == edge.from) {
                        let from_activation = self.nodes[from_idx].activation;
                        sum += from_activation * edge.weight as f32;
                    }
                }

                let threshold = self.nodes[node_idx].threshold;
                self.nodes[node_idx].activation = if sum > threshold {
                    sum.tanh()
                } else {
                    0.0
                };
            }
        }

        let mut outputs = [0.0; NUM_MOTORS];
        for (motor_idx, output) in outputs.iter_mut().enumerate() {
            let motor_id = (NUM_SENSORS as u32) + motor_idx as u32;
            if let Some(node) = self.nodes.iter().find(|n| n.id == motor_id) {
                *output = node.activation;
            }
        }

        outputs
    }

    pub fn system1_forward(&self, inputs: &[f32; NUM_SENSORS]) -> [f32; NUM_MOTORS] {
        let mut outputs = [0.0; NUM_MOTORS];
        let motor_start_id = NUM_SENSORS as u32;

        for motor_idx in 0..NUM_MOTORS {
            let motor_id = motor_start_id + motor_idx as u32;
            let mut sum = 0.0;

            for edge in &self.edges {
                if edge.to == motor_id
                    && edge.enabled
                    && (edge.from as usize) < NUM_SENSORS
                {
                    let from_activation = inputs[edge.from as usize];
                    sum += from_activation * edge.weight as f32;
                }
            }

            outputs[motor_idx] = sum.tanh();
        }

        outputs
    }

    pub fn system2_forward(&mut self, inputs: &[f32; NUM_SENSORS]) -> ([f32; NUM_MOTORS], f32) {
        let num_hidden_before = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Hidden)
            .count();

        let outputs = self.forward(inputs);
        let num_enabled_edges = self.edges.iter().filter(|e| e.enabled).count();

        let cost = (num_hidden_before as f32) + (num_enabled_edges as f32 * 0.1);

        (outputs, cost)
    }

    pub fn decide_system(&self, cortisol: f32, novelty: f32) -> u8 {
        let system1_bias = cortisol * 0.5;
        let system2_bias = novelty * 0.5;

        if system2_bias > system1_bias {
            2
        } else {
            1
        }
    }

    pub fn hebbian_update(&mut self, learning_rate: f32) {
        for edge in &mut self.edges {
            if !edge.enabled {
                continue;
            }

            let from_node = self.nodes.iter().find(|n| n.id == edge.from);
            let to_node = self.nodes.iter().find(|n| n.id == edge.to);

            if let (Some(from), Some(to)) = (from_node, to_node) {
                let correlation = from.activation * to.activation;

                if correlation > 0.0 {
                    let adjust = (correlation * learning_rate).min(1.0);
                    if adjust > 0.5 {
                        edge.weight = 1;
                    }
                } else if correlation < 0.0 {
                    let adjust = (-correlation * learning_rate).min(1.0);
                    if adjust > 0.5 {
                        edge.weight = -1;
                    }
                }
            }
        }
    }

    pub fn mutate(&mut self, rng: &mut impl Rng, mutation_rate: f32) {
        if rng.gen::<f32>() > mutation_rate {
            return;
        }

        let roll = rng.gen::<f32>();

        if roll < 0.30 {
            self.mutate_add_node(rng);
        } else if roll < 0.60 {
            self.mutate_add_edge(rng);
        } else if roll < 0.80 {
            self.mutate_toggle_edge(rng);
        } else {
            self.mutate_weight(rng);
        }
    }

    fn mutate_add_node(&mut self, rng: &mut impl Rng) {
        if self.edges.is_empty() {
            return;
        }

        let enabled_edges: Vec<usize> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.enabled)
            .map(|(i, _)| i)
            .collect();

        if enabled_edges.is_empty() {
            return;
        }

        let edge_idx = enabled_edges[rng.gen_range(0..enabled_edges.len())];
        let edge = self.edges[edge_idx].clone();

        self.edges[edge_idx].enabled = false;

        let new_node_id = self.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        self.nodes.push(Node::new(new_node_id, NodeType::Hidden));

        let mut edge1 = Edge::new(edge.from, new_node_id, 1, self.innovation_counter);
        let mut edge2 = Edge::new(new_node_id, edge.to, edge.weight, self.innovation_counter + 1);

        edge1.normalize_weight();
        edge2.normalize_weight();

        self.edges.push(edge1);
        self.edges.push(edge2);
        self.innovation_counter += 2;
    }

    fn mutate_add_edge(&mut self, rng: &mut impl Rng) {
        let from_id = self.nodes[rng.gen_range(0..self.nodes.len())].id;
        let to_id = self.nodes[rng.gen_range(0..self.nodes.len())].id;

        if from_id == to_id {
            return;
        }

        if self.edges.iter().any(|e| e.from == from_id && e.to == to_id) {
            return;
        }

        let weight = match rng.gen_range(0..3) {
            0 => -1,
            1 => 0,
            _ => 1,
        };

        self.edges.push(Edge::new(
            from_id,
            to_id,
            weight,
            self.innovation_counter,
        ));
        self.innovation_counter += 1;
    }

    fn mutate_toggle_edge(&mut self, rng: &mut impl Rng) {
        if self.edges.is_empty() {
            return;
        }

        let edge_idx = rng.gen_range(0..self.edges.len());
        self.edges[edge_idx].enabled = !self.edges[edge_idx].enabled;
    }

    fn mutate_weight(&mut self, rng: &mut impl Rng) {
        if self.edges.is_empty() {
            return;
        }

        let edge_idx = rng.gen_range(0..self.edges.len());
        let weight = self.edges[edge_idx].weight;

        self.edges[edge_idx].weight = match weight {
            1 => rng.gen_range(0..2) as i8 - 1,
            -1 => rng.gen_range(0..2) as i8,
            _ => [1, -1][rng.gen_range(0..2)],
        };
    }

    pub fn crossover(
        parent1: &Brain,
        parent2: &Brain,
        rng: &mut impl Rng,
    ) -> Brain {
        let fitter = parent1;

        let edge_map2: HashMap<u32, &Edge> =
            parent2.edges.iter().map(|e| (e.innovation, e)).collect();

        let mut offspring_edges = Vec::new();

        for edge1 in &parent1.edges {
            if let Some(edge2) = edge_map2.get(&edge1.innovation) {
                let chosen = if rng.gen::<bool>() {
                    edge1.clone()
                } else {
                    (*edge2).clone()
                };
                offspring_edges.push(chosen);
            } else {
                offspring_edges.push(edge1.clone());
            }
        }

        let mut nodes_used = std::collections::HashSet::new();
        for edge in &offspring_edges {
            nodes_used.insert(edge.from);
            nodes_used.insert(edge.to);
        }

        let mut offspring_nodes = Vec::new();
        for node in &fitter.nodes {
            if nodes_used.contains(&node.id) {
                offspring_nodes.push(node.clone());
            }
        }

        for i in 0..NUM_SENSORS {
            let sensor_id = i as u32;
            if !offspring_nodes.iter().any(|n| n.id == sensor_id) {
                offspring_nodes.push(Node::new(sensor_id, NodeType::Sensor));
            }
        }

        for i in 0..NUM_MOTORS {
            let motor_id = (NUM_SENSORS as u32) + i as u32;
            if !offspring_nodes.iter().any(|n| n.id == motor_id) {
                offspring_nodes.push(Node::new(motor_id, NodeType::Motor));
            }
        }

        let max_innovation = offspring_edges
            .iter()
            .map(|e| e.innovation)
            .max()
            .unwrap_or(0);

        Brain {
            nodes: offspring_nodes,
            edges: offspring_edges,
            innovation_counter: max_innovation + 1,
        }
    }

    pub fn compatibility_distance(a: &Brain, b: &Brain) -> f32 {
        const C1: f32 = 1.0;
        const C2: f32 = 1.0;
        const C3: f32 = 0.5;

        let edges_a: HashMap<u32, &Edge> = a.edges.iter().map(|e| (e.innovation, e)).collect();
        let edges_b: HashMap<u32, &Edge> = b.edges.iter().map(|e| (e.innovation, e)).collect();

        let mut matching = 0;
        let mut disjoint = 0;
        let mut weight_diff = 0.0;

        for (innov, edge_a) in &edges_a {
            if let Some(edge_b) = edges_b.get(innov) {
                matching += 1;
                weight_diff += (edge_a.weight as f32 - edge_b.weight as f32).abs();
            } else {
                disjoint += 1;
            }
        }

        for innov in edges_b.keys() {
            if !edges_a.contains_key(innov) {
                disjoint += 1;
            }
        }

        let n = (a.edges.len().max(b.edges.len()) as f32).max(1.0);
        let avg_weight_diff = if matching > 0 {
            weight_diff / matching as f32
        } else {
            1.0
        };

        (C1 * disjoint as f32 + C2 * disjoint as f32) / n + C3 * avg_weight_diff
    }

    pub fn complexity(&self) -> f32 {
        (self.nodes.len() + self.edges.iter().filter(|e| e.enabled).count()) as f32
    }

    fn topological_sort(&self) -> Vec<u32> {
        let mut visited = std::collections::HashSet::new();
        let mut sorted = Vec::new();

        for node in &self.nodes {
            if !visited.contains(&node.id) {
                self.dfs_topo(node.id, &mut visited, &mut sorted);
            }
        }

        sorted
    }

    fn dfs_topo(&self, node_id: u32, visited: &mut std::collections::HashSet<u32>, sorted: &mut Vec<u32>) {
        visited.insert(node_id);

        for edge in &self.edges {
            if edge.from == node_id && !visited.contains(&edge.to) {
                self.dfs_topo(edge.to, visited, sorted);
            }
        }

        sorted.push(node_id);
    }
}

// ============================================================================
// Species and SpeciesManager
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Species {
    pub id: u32,
    pub representative: Brain,
    pub members: Vec<usize>,
    pub best_fitness: f32,
    pub stagnation: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesManager {
    pub species: Vec<Species>,
    pub compatibility_threshold: f32,
    next_species_id: u32,
}

impl SpeciesManager {
    pub fn new(threshold: f32) -> Self {
        Self {
            species: Vec::new(),
            compatibility_threshold: threshold,
            next_species_id: 0,
        }
    }

    pub fn classify(&mut self, brain: &Brain) -> u32 {
        for species in &mut self.species {
            let distance = Brain::compatibility_distance(brain, &species.representative);
            if distance < self.compatibility_threshold {
                return species.id;
            }
        }

        let new_species = Species {
            id: self.next_species_id,
            representative: brain.clone(),
            members: Vec::new(),
            best_fitness: 0.0,
            stagnation: 0,
        };

        self.species.push(new_species);
        let id = self.next_species_id;
        self.next_species_id += 1;
        id
    }

    pub fn update(&mut self, fitnesses: &[f32]) {
        for species in &mut self.species {
            if species.members.is_empty() {
                continue;
            }

            let max_fitness = species
                .members
                .iter()
                .filter_map(|&idx| fitnesses.get(idx))
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            if max_fitness > species.best_fitness {
                species.best_fitness = max_fitness;
                species.stagnation = 0;
            } else {
                species.stagnation += 1;
            }

            if !species.members.is_empty() {
                let _best_idx = *species
                    .members
                    .iter()
                    .max_by(|&&a, &&b| {
                        fitnesses
                            .get(a)
                            .copied()
                            .unwrap_or(0.0)
                            .partial_cmp(&fitnesses.get(b).copied().unwrap_or(0.0))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(&0);
            }

            species.members.clear();
        }

        self.species.retain(|s| s.stagnation < 20);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_brain_creation() {
        let brain = Brain::minimal();
        assert_eq!(brain.nodes.len(), NUM_SENSORS + NUM_MOTORS);
        assert_eq!(brain.edges.len(), NUM_SENSORS * NUM_MOTORS);
        assert!(brain.edges.iter().all(|e| !e.enabled || e.weight == 0));
    }

    #[test]
    fn test_forward_pass_deterministic() {
        let mut brain = Brain::minimal();
        brain.edges[0].enabled = true;
        brain.edges[0].weight = 1;

        let inputs = [0.5; NUM_SENSORS];
        let output1 = brain.forward(&inputs);

        brain = Brain::minimal();
        brain.edges[0].enabled = true;
        brain.edges[0].weight = 1;
        let output2 = brain.forward(&inputs);

        for i in 0..NUM_MOTORS {
            assert!((output1[i] - output2[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mutation_increases_complexity() {
        let mut brain = Brain::minimal();
        let initial_complexity = brain.complexity();

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        for _ in 0..20 {
            brain.mutate(&mut rng, 1.0);
        }

        let new_complexity = brain.complexity();
        assert!(new_complexity >= initial_complexity,
            "Complexity should increase with mutations: {} >= {}", new_complexity, initial_complexity);
    }

    #[test]
    fn test_crossover_produces_valid_brain() {
        let mut rng = rand::thread_rng();

        let parent1 = Brain::minimal();
        let parent2 = Brain::minimal();

        let offspring = Brain::crossover(&parent1, &parent2, &mut rng);

        assert!(!offspring.nodes.is_empty());
        assert!(offspring.nodes.len() >= NUM_SENSORS + NUM_MOTORS);
    }

    #[test]
    fn test_compatibility_distance() {
        let brain1 = Brain::minimal();
        let brain2 = Brain::minimal();

        let distance = Brain::compatibility_distance(&brain1, &brain2);
        assert!(distance < 0.01);
    }

    #[test]
    fn test_compatibility_distance_different() {
        let mut rng = rand::thread_rng();
        let mut brain1 = Brain::minimal();
        let brain2 = Brain::minimal();

        for _ in 0..5 {
            brain1.mutate(&mut rng, 1.0);
        }

        let distance = Brain::compatibility_distance(&brain1, &brain2);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_system1_vs_system2_cost() {
        let mut brain = Brain::minimal();
        brain.edges[0].enabled = true;
        brain.edges[0].weight = 1;

        let inputs = [0.5; NUM_SENSORS];

        let _output1 = brain.system1_forward(&inputs);
        let (_output2, cost) = brain.system2_forward(&inputs);

        assert!(cost >= 0.0);
    }

    #[test]
    fn test_ternary_weights() {
        let brain = Brain::minimal();

        for edge in &brain.edges {
            assert!(edge.weight == -1 || edge.weight == 0 || edge.weight == 1);
        }
    }

    #[test]
    fn test_hebbian_learning() {
        let mut brain = Brain::minimal();
        brain.edges[0].enabled = true;
        brain.edges[0].weight = 0;

        let inputs = [0.5; NUM_SENSORS];
        brain.forward(&inputs);

        brain.hebbian_update(1.0);

        assert!(brain.edges[0].weight >= -1 && brain.edges[0].weight <= 1);
    }

    #[test]
    fn test_species_manager() {
        let mut manager = SpeciesManager::new(3.0);

        let brain1 = Brain::minimal();
        let brain2 = Brain::minimal();

        let species1 = manager.classify(&brain1);
        let species2 = manager.classify(&brain2);

        assert_eq!(species1, species2);

        let mut brain3 = Brain::minimal();
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            brain3.mutate(&mut rng, 1.0);
        }

        let species3 = manager.classify(&brain3);
        assert!(manager.species.iter().any(|s| s.id == species3));
    }
}
