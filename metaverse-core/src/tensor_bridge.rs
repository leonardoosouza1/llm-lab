// ============================================================================
// Tensor Bridge Module: LLM Tensor Integration for MetaVerse
// ============================================================================
// Bridge between real neural network weights (LLM tensors) and the
// virtual organism world. Converts tensor properties into genetic traits
// and neural topologies suitable for organism evolution.
// ============================================================================

use std::io::{Read, Write, BufWriter, BufReader};
use std::fs::File;
use crate::brain::Brain;
use crate::genetics::{Genome, Gene, TraitType};

// ============================================================================
// TensorSlice: Core data structure for flat tensors
// ============================================================================

#[derive(Debug, Clone)]
pub struct TensorSlice {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub name: String,
}

impl TensorSlice {
    /// Create a new tensor slice
    pub fn new(name: String, shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert!(
            shape.iter().product::<usize>() == data.len(),
            "Shape product must equal data length"
        );
        Self { data, shape, name }
    }

    /// Get element at indices
    pub fn get(&self, indices: &[usize]) -> f32 {
        assert_eq!(indices.len(), self.shape.len(), "Index dimension mismatch");
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            idx += indices[i] * stride;
            stride *= self.shape[i];
        }
        self.data[idx]
    }

    /// Compute mean of all elements
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Compute standard deviation
    pub fn std(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.data.len() as f32;
        variance.sqrt()
    }

    /// Compute L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.data.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    }

    /// Get activation pattern: which elements exceed threshold
    pub fn activation_pattern(&self, threshold: f32) -> Vec<bool> {
        self.data.iter()
            .map(|x| x.abs() > threshold)
            .collect()
    }

    /// Compute sparsity: fraction of near-zero elements
    pub fn sparsity(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let near_zero = self.data.iter()
            .filter(|x| x.abs() < 1e-6)
            .count();
        near_zero as f32 / self.data.len() as f32
    }

    /// Total number of elements
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// WeightDistribution: Statistical properties of a tensor
// ============================================================================

#[derive(Debug, Clone)]
pub struct WeightDistribution {
    pub mean: f32,
    pub std: f32,
    pub kurtosis: f32,
    pub skewness: f32,
    pub sparsity: f32,
    pub histogram: Vec<u32>,
}

impl WeightDistribution {
    /// Create from a tensor slice
    pub fn from_tensor(tensor: &TensorSlice) -> Self {
        let mean = tensor.mean();
        let std = tensor.std();
        let sparsity = tensor.sparsity();

        // Compute kurtosis and skewness
        let n = tensor.data.len() as f32;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;

        for &val in &tensor.data {
            let d = val - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
        }

        m2 /= n;
        m3 /= n;
        m4 /= n;

        let var = m2;
        let skewness = if var > 1e-10 { m3 / var.sqrt().powi(3) } else { 0.0 };
        let kurtosis = if var > 1e-10 { m4 / (var * var) - 3.0 } else { 0.0 };

        // Build histogram (20 bins from -2.0 to 2.0)
        let mut histogram = vec![0u32; 20];
        for &val in &tensor.data {
            let normalized = (val - mean) / (std + 1e-10);
            let clamped = normalized.clamp(-2.0, 2.0);
            let bin = ((clamped + 2.0) / 4.0 * 20.0) as usize;
            if bin < 20 {
                histogram[bin] += 1;
            }
        }

        Self {
            mean,
            std,
            kurtosis,
            skewness,
            sparsity,
            histogram,
        }
    }
}

// ============================================================================
// PulseFormat: Lightweight binary format for organism-scale networks
// ============================================================================

pub struct PulseWriter {
    path: String,
}

impl PulseWriter {
    /// Create a new PulseWriter
    pub fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }

    /// Write tensors to file in Pulse format
    pub fn write_all(&self, tensors: &[TensorSlice]) -> std::io::Result<()> {
        let file = File::create(&self.path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(b"PULS")?;
        writer.write_all(&[1u8])?; // version
        writer.write_all(&(tensors.len() as u32).to_le_bytes())?;

        // Write each tensor
        for tensor in tensors {
            self.write_tensor(&mut writer, tensor)?;
        }

        writer.flush()?;
        Ok(())
    }

    fn write_tensor(&self, writer: &mut BufWriter<File>, tensor: &TensorSlice) -> std::io::Result<()> {
        // Write name
        let name_bytes = tensor.name.as_bytes();
        writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        writer.write_all(name_bytes)?;

        // Write shape
        writer.write_all(&[tensor.shape.len() as u8])?;
        for &dim in &tensor.shape {
            writer.write_all(&(dim as u32).to_le_bytes())?;
        }

        // Write data
        for &val in &tensor.data {
            writer.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }
}

pub struct PulseReader {
    path: String,
}

impl PulseReader {
    /// Create a new PulseReader
    pub fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }

    /// Read all tensors from file
    pub fn read_all(&self) -> std::io::Result<Vec<TensorSlice>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"PULS" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid Pulse format magic bytes",
            ));
        }

        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;

        let mut num_tensors_bytes = [0u8; 4];
        reader.read_exact(&mut num_tensors_bytes)?;
        let num_tensors = u32::from_le_bytes(num_tensors_bytes) as usize;

        let mut tensors = Vec::new();
        for _ in 0..num_tensors {
            let tensor = self.read_tensor(&mut reader)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Read a specific tensor by name
    pub fn read_tensor_by_name(&self, target_name: &str) -> std::io::Result<Option<TensorSlice>> {
        let tensors = self.read_all()?;
        Ok(tensors.into_iter().find(|t| t.name == target_name))
    }

    fn read_tensor(&self, reader: &mut BufReader<File>) -> std::io::Result<TensorSlice> {
        // Read name
        let mut name_len_bytes = [0u8; 2];
        reader.read_exact(&mut name_len_bytes)?;
        let name_len = u16::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|_| std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid UTF-8 in tensor name",
            ))?;

        // Read shape
        let mut ndims_bytes = [0u8; 1];
        reader.read_exact(&mut ndims_bytes)?;
        let ndims = ndims_bytes[0] as usize;

        let mut shape = Vec::new();
        for _ in 0..ndims {
            let mut dim_bytes = [0u8; 4];
            reader.read_exact(&mut dim_bytes)?;
            shape.push(u32::from_le_bytes(dim_bytes) as usize);
        }

        // Read data
        let num_elements: usize = shape.iter().product();
        let mut data = vec![0.0f32; num_elements];
        for val in &mut data {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            *val = f32::from_le_bytes(bytes);
        }

        Ok(TensorSlice::new(name, shape, data))
    }
}

// ============================================================================
// TensorAnalyzer: Statistical analysis and compatibility
// ============================================================================

pub struct TensorAnalyzer;

impl TensorAnalyzer {
    /// Compute spectral entropy from singular values (approximated)
    pub fn spectral_entropy(tensor: &TensorSlice) -> f32 {
        if tensor.data.is_empty() {
            return 0.0;
        }

        // Use power iteration to estimate the dominant singular value
        let _dominant = Self::power_iteration_max_singular_value(tensor);

        // Estimate spectral entropy as a normalized measure
        // High entropy = more uniform singular value distribution
        let mean = tensor.mean().abs().max(1e-10);
        let std = tensor.std().max(1e-10);

        // Simple entropy approximation based on distribution shape
        let normalized_std = std / mean;
        (normalized_std / (1.0 + normalized_std)).max(0.0).min(1.0)
    }

    /// Power iteration to estimate largest singular value
    fn power_iteration_max_singular_value(tensor: &TensorSlice) -> f32 {
        if tensor.data.len() < 2 {
            return tensor.data.get(0).copied().unwrap_or(0.0).abs();
        }

        // For 1D: just return max absolute value
        if tensor.shape.len() == 1 {
            return tensor.data.iter().map(|x| x.abs()).fold(f32::NEG_INFINITY, f32::max);
        }

        // For multi-dimensional, use matrix power iteration on flattened version
        let mut v: Vec<f32> = vec![1.0 / (tensor.data.len() as f32).sqrt(); tensor.data.len()];

        for _ in 0..5 {
            // Matrix-vector multiply (A^T * A * v)
            let mut av = vec![0.0; tensor.data.len()];
            for (i, ai) in av.iter_mut().enumerate() {
                *ai = tensor.data.iter().enumerate()
                    .map(|(j, &x)| if i == j { x * x } else { x * x * 0.1 })
                    .sum();
            }

            let norm = av.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                break;
            }

            v = av.into_iter().map(|x| x / norm).collect();
        }

        // Return approximation of largest singular value
        v.iter().map(|x| x.abs()).fold(f32::NEG_INFINITY, f32::max).max(0.0)
    }

    /// Compute weight distribution statistics
    pub fn weight_distribution(tensor: &TensorSlice) -> WeightDistribution {
        WeightDistribution::from_tensor(tensor)
    }

    /// Compute cosine similarity between two tensors
    pub fn layer_similarity(a: &TensorSlice, b: &TensorSlice) -> f32 {
        if a.data.is_empty() || b.data.is_empty() {
            return 0.0;
        }

        // Pad shorter tensor with zeros
        let max_len = a.data.len().max(b.data.len());
        let mut a_padded = a.data.clone();
        let mut b_padded = b.data.clone();
        a_padded.resize(max_len, 0.0);
        b_padded.resize(max_len, 0.0);

        let dot: f32 = a_padded.iter().zip(&b_padded)
            .map(|(x, y)| x * y)
            .sum();
        let norm_a: f32 = a_padded.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b_padded.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Estimate effective rank of tensor
    pub fn effective_rank(tensor: &TensorSlice) -> f32 {
        if tensor.data.is_empty() {
            return 0.0;
        }

        // Estimate rank from distribution shape
        // Sparse tensors have lower effective rank
        let sparsity = tensor.sparsity();
        let std = tensor.std();
        let mean = tensor.mean().abs().max(1e-10);

        // Effective rank ~ (1 - sparsity) * sqrt(variance/mean)
        let shape_ratio = (std / mean).min(10.0);
        let rank_estimate = (1.0 - sparsity) * shape_ratio.sqrt();

        rank_estimate.max(0.0).min(tensor.num_elements() as f32)
    }
}

// ============================================================================
// TensorToOrganism: Convert LLM tensors to organism genetics & brains
// ============================================================================

pub struct TensorToOrganism;

impl TensorToOrganism {
    /// Convert tensor statistics to genome
    pub fn tensor_to_genome(tensors: &[TensorSlice]) -> Genome {
        let mut genes = Vec::new();
        let mut gene_id = 0u32;

        if tensors.is_empty() {
            return Genome::default_producer();
        }

        // Extract features from each tensor layer
        for tensor in tensors {
            let mean = tensor.mean();
            let std = tensor.std();
            let sparsity = tensor.sparsity();
            let l2_norm = tensor.l2_norm();

            // Map tensor features to organism traits

            // Layer mean -> Size (morphology)
            let size = (mean.abs() * 10.0).clamp(0.5, 10.0);
            genes.push(Gene {
                id: gene_id,
                trait_type: TraitType::Size,
                value: size,
                dominance: 0.5,
                methylated: false,
            });
            gene_id += 1;

            // Layer std -> Aggression-like trait (Energy for now)
            let energy = (std * 200.0).clamp(10.0, 500.0);
            genes.push(Gene {
                id: gene_id,
                trait_type: TraitType::EnergyCapacity,
                value: energy,
                dominance: 0.5,
                methylated: false,
            });
            gene_id += 1;

            // Sparsity -> Efficiency (Metabolic flexibility)
            let efficiency = sparsity.clamp(0.0, 1.0);
            genes.push(Gene {
                id: gene_id,
                trait_type: TraitType::MetabolicFlexibility,
                value: efficiency,
                dominance: 0.5,
                methylated: false,
            });
            gene_id += 1;

            // L2 norm -> Speed (activity level)
            let speed = (l2_norm / 100.0).clamp(0.0, 3.0);
            genes.push(Gene {
                id: gene_id,
                trait_type: TraitType::Speed,
                value: speed,
                dominance: 0.5,
                methylated: false,
            });
            gene_id += 1;
        }

        // Number of layers -> Complexity
        let complexity = (tensors.len() as f32).clamp(2.0, 64.0);
        genes.push(Gene {
            id: gene_id,
            trait_type: TraitType::BrainComplexity,
            value: complexity,
            dominance: 0.5,
            methylated: false,
        });
        gene_id += 1;

        // Total parameters -> Learning capacity
        let total_params = tensors.iter().map(|t| t.num_elements()).sum::<usize>();
        let learning_rate = (total_params as f32 / 1000.0).clamp(0.0, 1.0);
        genes.push(Gene {
            id: gene_id,
            trait_type: TraitType::LearningRate,
            value: learning_rate,
            dominance: 0.5,
            methylated: false,
        });

        // Add essential traits if missing
        let has_basal = genes.iter().any(|g| g.trait_type == TraitType::BasalRate);
        if !has_basal {
            genes.push(Gene {
                id: gene_id,
                trait_type: TraitType::BasalRate,
                value: 0.5,
                dominance: 0.5,
                methylated: false,
            });
        }

        Genome {
            genes,
            mutation_rate: 0.001,
            generation: 0,
        }
    }

    /// Convert tensors to a NeatBrain
    pub fn tensor_to_brain(
        tensors: &[TensorSlice],
        _max_nodes: usize,
        max_connections: usize,
    ) -> Brain {
        let mut brain = Brain::minimal();

        if tensors.is_empty() {
            return brain;
        }

        // Find the largest tensor for sampling connections
        let largest = tensors.iter()
            .max_by_key(|t| t.num_elements())
            .unwrap();

        // Sample connections from largest tensor weights
        let num_samples = max_connections.min(largest.num_elements());

        for i in 0..num_samples {
            let idx = (i * largest.num_elements()) / num_samples;
            if idx < largest.data.len() {
                let weight = largest.data[idx];

                // Quantize to ternary
                let ternary_weight = if weight.abs() > 0.5 {
                    if weight > 0.0 { 1 } else { -1 }
                } else {
                    0
                };

                // Add or modify edge
                if i < brain.edges.len() {
                    brain.edges[i].weight = ternary_weight;
                    brain.edges[i].enabled = ternary_weight != 0;
                }
            }
        }

        brain
    }

    /// Export brain weights as a tensor
    pub fn brain_to_tensor(brain: &Brain) -> TensorSlice {
        let data: Vec<f32> = brain.edges.iter()
            .map(|e| e.weight as f32)
            .collect();

        TensorSlice::new(
            "brain.weights".to_string(),
            vec![data.len()],
            data,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_slice_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = TensorSlice::new("test".to_string(), shape, data);

        assert_eq!(tensor.num_elements(), 4);
        assert_eq!(tensor.name, "test");
    }

    #[test]
    fn test_tensor_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = TensorSlice::new(
            "test".to_string(),
            vec![5],
            data,
        );

        let mean = tensor.mean();
        assert!((mean - 3.0).abs() < 1e-6);

        let l2 = tensor.l2_norm();
        assert!((l2 - 7.416).abs() < 0.01);
    }

    #[test]
    fn test_tensor_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = TensorSlice::new(
            "test".to_string(),
            vec![5],
            data,
        );

        let std = tensor.std();
        assert!(std > 1.0 && std < 2.0);
    }

    #[test]
    fn test_tensor_sparsity() {
        let data = vec![0.0000001, 0.0000002, 1.0, 0.0, 0.0000003, -0.5];
        let tensor = TensorSlice::new(
            "test".to_string(),
            vec![6],
            data,
        );

        let sparsity = tensor.sparsity();
        assert!(sparsity > 0.4 && sparsity < 0.8);
    }

    #[test]
    fn test_activation_pattern() {
        let data = vec![0.1, 1.0, 0.05, 2.0, 0.01];
        let tensor = TensorSlice::new(
            "test".to_string(),
            vec![5],
            data,
        );

        let pattern = tensor.activation_pattern(0.5);
        assert_eq!(pattern[0], false);
        assert_eq!(pattern[1], true);
        assert_eq!(pattern[2], false);
        assert_eq!(pattern[3], true);
        assert_eq!(pattern[4], false);
    }

    #[test]
    fn test_pulse_write_read_roundtrip() {
        let tempdir = std::env::temp_dir();
        let path = tempdir.join("test_pulse.bin").to_string_lossy().to_string();

        let tensors = vec![
            TensorSlice::new(
                "layer1".to_string(),
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            TensorSlice::new(
                "layer2".to_string(),
                vec![3],
                vec![0.5, 1.5, 2.5],
            ),
        ];

        {
            let writer = PulseWriter::new(&path);
            writer.write_all(&tensors).expect("Failed to write");
        }

        {
            let reader = PulseReader::new(&path);
            let loaded = reader.read_all().expect("Failed to read");
            assert_eq!(loaded.len(), 2);
            assert_eq!(loaded[0].name, "layer1");
            assert_eq!(loaded[0].shape, vec![2, 3]);
            assert_eq!(loaded[0].data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_pulse_multiple_tensors() {
        let tempdir = std::env::temp_dir();
        let path = tempdir.join("test_multi.bin").to_string_lossy().to_string();

        let tensors = vec![
            TensorSlice::new("a".to_string(), vec![2], vec![1.0, 2.0]),
            TensorSlice::new("b".to_string(), vec![3], vec![3.0, 4.0, 5.0]),
            TensorSlice::new("c".to_string(), vec![1, 2], vec![0.5, 1.5]),
        ];

        {
            let writer = PulseWriter::new(&path);
            writer.write_all(&tensors).expect("Failed to write");
        }

        {
            let reader = PulseReader::new(&path);
            let loaded = reader.read_all().expect("Failed to read");
            assert_eq!(loaded.len(), 3);
            assert_eq!(loaded[1].name, "b");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tensor_to_genome_produces_valid_genome() {
        let tensors = vec![
            TensorSlice::new(
                "layer1".to_string(),
                vec![4],
                vec![0.5, 1.0, -0.5, 0.2],
            ),
        ];

        let genome = TensorToOrganism::tensor_to_genome(&tensors);

        assert!(!genome.genes.is_empty());
        assert!(genome.genes.len() >= 3);

        // Check that size trait was set
        let has_size = genome.genes.iter()
            .any(|g| g.trait_type == TraitType::Size);
        assert!(has_size);
    }

    #[test]
    fn test_tensor_to_brain_produces_valid_brain() {
        let tensors = vec![
            TensorSlice::new(
                "weights".to_string(),
                vec![10],
                vec![0.5, 1.0, -0.5, 0.2, 0.8, -0.1, 0.3, 0.6, -0.4, 0.1],
            ),
        ];

        let brain = TensorToOrganism::tensor_to_brain(&tensors, 32, 50);

        assert!(!brain.edges.is_empty());
        assert!(!brain.nodes.is_empty());

        // All weights should be ternary
        for edge in &brain.edges {
            assert!(edge.weight == -1 || edge.weight == 0 || edge.weight == 1);
        }
    }

    #[test]
    fn test_brain_to_tensor_roundtrip() {
        let brain = Brain::minimal();
        let tensor = TensorToOrganism::brain_to_tensor(&brain);

        assert_eq!(tensor.num_elements(), brain.edges.len());
        assert_eq!(tensor.name, "brain.weights");
    }

    #[test]
    fn test_weight_distribution_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = TensorSlice::new(
            "test".to_string(),
            vec![5],
            data,
        );

        let dist = WeightDistribution::from_tensor(&tensor);

        assert!(dist.mean > 0.0);
        assert!(dist.std > 0.0);
        assert_eq!(dist.histogram.len(), 20);
    }

    #[test]
    fn test_layer_similarity_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = TensorSlice::new("a".to_string(), vec![4], data.clone());
        let tensor2 = TensorSlice::new("b".to_string(), vec![4], data);

        let similarity = TensorAnalyzer::layer_similarity(&tensor1, &tensor2);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_similarity_orthogonal() {
        let tensor1 = TensorSlice::new(
            "a".to_string(),
            vec![2],
            vec![1.0, 0.0],
        );
        let tensor2 = TensorSlice::new(
            "b".to_string(),
            vec![2],
            vec![0.0, 1.0],
        );

        let similarity = TensorAnalyzer::layer_similarity(&tensor1, &tensor2);
        assert!(similarity.abs() < 0.01);
    }

    #[test]
    fn test_spectral_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let tensor = TensorSlice::new(
            "uniform".to_string(),
            vec![5],
            data,
        );

        let entropy = TensorAnalyzer::spectral_entropy(&tensor);
        assert!(entropy >= 0.0 && entropy <= 1.0);
    }

    #[test]
    fn test_effective_rank() {
        let dense_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dense_tensor = TensorSlice::new(
            "dense".to_string(),
            vec![6],
            dense_data,
        );

        let sparse_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0];
        let sparse_tensor = TensorSlice::new(
            "sparse".to_string(),
            vec![6],
            sparse_data,
        );

        let dense_rank = TensorAnalyzer::effective_rank(&dense_tensor);
        let sparse_rank = TensorAnalyzer::effective_rank(&sparse_tensor);

        assert!(dense_rank > sparse_rank);
    }
}
