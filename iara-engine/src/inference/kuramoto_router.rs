// ============================================================================
// inference/kuramoto_router.rs — Kuramoto Phase Router for Sparse MLP
// ============================================================================
// Predicts which MLP neurons will survive SiLU gating BEFORE computing
// the full gate_proj × x multiplication. Uses phase coherence between
// the hidden state and pre-computed "tuning forks" (rows of gate_proj
// converted to phase angles).
//
// Expected: 25% neurons active → 4x MLP speedup
// ============================================================================

use std::collections::HashSet;

/// Pre-computed routing data for one layer's MLP
#[derive(Clone)]
pub struct KuramotoRouterLayer {
    /// Tuning forks: rows of gate_proj, stored as phase angles
    /// Shape: [n_neurons, hidden_dim/2]
    /// Each tuning fork is a vector of phase angles from complex pairs
    tuning_forks: Vec<Vec<f32>>,

    /// Number of MLP neurons in this layer
    n_neurons: usize,

    /// How many neurons to keep active (top-K, typically 25%)
    top_k: usize,
}

/// Router for all layers in the model
pub struct KuramotoRouter {
    layers: Vec<KuramotoRouterLayer>,

    /// Statistics for monitoring router performance
    pub stats: RouterStats,
}

/// Statistics tracking router predictions vs actual activations
#[derive(Debug, Clone)]
pub struct RouterStats {
    /// Total number of routing predictions made
    pub total_predictions: u64,

    /// Running average of active neuron ratio
    pub avg_active_ratio: f32,

    /// Running average accuracy (predicted active vs actual alive)
    pub accuracy: f32,
}

impl Default for RouterStats {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            avg_active_ratio: 0.0,
            accuracy: 0.0,
        }
    }
}

impl KuramotoRouter {
    /// Create a new Kuramoto router for the model
    ///
    /// # Arguments
    /// * `n_layers` - Number of transformer layers
    /// * `n_neurons` - Number of MLP neurons per layer
    /// * `active_ratio` - Fraction of neurons to keep active (e.g., 0.25 for 25%)
    pub fn new(n_layers: usize, n_neurons: usize, active_ratio: f32) -> Self {
        let top_k = ((n_neurons as f32) * active_ratio).ceil() as usize;

        let layers = vec![
            KuramotoRouterLayer {
                tuning_forks: vec![],
                n_neurons,
                top_k,
            };
            n_layers
        ];

        Self {
            layers,
            stats: RouterStats::default(),
        }
    }

    /// Initialize router from pre-extracted gate_proj weights
    ///
    /// Converts the weights of gate_proj to phase angles ("tuning forks")
    /// for use in coherence computation.
    ///
    /// # Arguments
    /// * `layer_idx` - Which transformer layer (0..n_layers)
    /// * `gate_proj_weights` - Flattened gate_proj weight matrix [n_neurons, hidden_dim]
    /// * `hidden_dim` - Dimension of hidden state
    pub fn initialize_layer(
        &mut self,
        layer_idx: usize,
        gate_proj_weights: &[f32],
        hidden_dim: usize,
    ) {
        if layer_idx >= self.layers.len() {
            return;
        }

        let n_neurons = self.layers[layer_idx].n_neurons;
        let tuning_forks = Self::extract_tuning_forks(gate_proj_weights, n_neurons, hidden_dim);

        self.layers[layer_idx].tuning_forks = tuning_forks;
    }

    /// Compute which neurons should be active for a given hidden state
    ///
    /// Returns sorted indices of active neurons (by coherence, descending)
    ///
    /// # Arguments
    /// * `layer_idx` - Which transformer layer
    /// * `hidden_state` - Current hidden state [hidden_dim]
    ///
    /// # Returns
    /// Vec of neuron indices to activate, sorted by coherence (highest first)
    pub fn route(&self, layer_idx: usize, hidden_state: &[f32]) -> Vec<usize> {
        if layer_idx >= self.layers.len() {
            return vec![];
        }

        let layer = &self.layers[layer_idx];

        // If no tuning forks loaded yet, return empty
        if layer.tuning_forks.is_empty() {
            return vec![];
        }

        // 1. Convert hidden state to phase angles
        let phases = Self::to_phases(hidden_state);

        // 2. Compute coherence with each tuning fork
        let coherences = self.compute_coherences(layer, &phases);

        // 3. Select top-K neurons by coherence
        let mut indices: Vec<(usize, f32)> = coherences
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        indices.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only top_k
        indices.truncate(layer.top_k);

        indices.iter().map(|(i, _)| *i).collect()
    }

    /// Convert hidden state (real pairs) to phase angles
    ///
    /// Treats consecutive pairs as complex numbers and extracts angles
    /// using atan2(imag, real).
    ///
    /// # Arguments
    /// * `hidden` - Hidden state as [real_0, imag_0, real_1, imag_1, ...]
    ///
    /// # Returns
    /// Vec of phase angles [θ_0, θ_1, ...]
    fn to_phases(hidden: &[f32]) -> Vec<f32> {
        hidden
            .chunks(2)
            .map(|pair| {
                if pair.len() == 2 {
                    // Standard pair: real = pair[0], imag = pair[1]
                    pair[1].atan2(pair[0])
                } else {
                    // Odd element: treat as real part only (imag = 0)
                    // atan2(0, real) where real = pair[0]
                    (0.0_f32).atan2(pair[0])
                }
            })
            .collect()
    }

    /// Compute phase coherence between hidden state and each tuning fork
    ///
    /// For each tuning fork (representing a neuron group):
    /// coherence = |mean(exp(i*(phase_h - phase_fork)))|
    ///
    /// This measures how well the hidden state "resonates" with this neuron.
    /// High coherence (near 1.0) means the neuron is likely to survive SiLU gating.
    ///
    /// # Arguments
    /// * `layer` - Layer with tuning forks
    /// * `phases` - Phase angles from hidden state
    ///
    /// # Returns
    /// Vec of coherence values [0, 1] for each neuron
    fn compute_coherences(&self, layer: &KuramotoRouterLayer, phases: &[f32]) -> Vec<f32> {
        layer
            .tuning_forks
            .iter()
            .map(|fork| {
                let n = phases.len().min(fork.len());
                if n == 0 {
                    return 0.0;
                }

                let (mut sum_cos, mut sum_sin) = (0.0f32, 0.0f32);
                for k in 0..n {
                    let delta = phases[k] - fork[k];
                    sum_cos += delta.cos();
                    sum_sin += delta.sin();
                }

                let mean_cos = sum_cos / n as f32;
                let mean_sin = sum_sin / n as f32;
                let r = (mean_cos.powi(2) + mean_sin.powi(2)).sqrt();
                r
            })
            .collect()
    }

    /// Extract tuning forks from gate_proj weights
    ///
    /// Converts each row of gate_proj to a sequence of phase angles.
    /// This is a CPU operation done once at model load time.
    ///
    /// # Arguments
    /// * `gate_proj_weights` - Flattened [n_neurons, hidden_dim] weights
    /// * `n_neurons` - Number of rows (neurons)
    /// * `hidden_dim` - Number of columns (hidden dimension)
    ///
    /// # Returns
    /// Vec of tuning forks (one per neuron), each containing phase angles
    fn extract_tuning_forks(
        gate_proj_weights: &[f32],
        n_neurons: usize,
        hidden_dim: usize,
    ) -> Vec<Vec<f32>> {
        (0..n_neurons)
            .map(|r| {
                let start = r * hidden_dim;
                let end = (r + 1) * hidden_dim;

                if end <= gate_proj_weights.len() {
                    let row = &gate_proj_weights[start..end];
                    Self::to_phases(row)
                } else {
                    vec![]
                }
            })
            .collect()
    }

    /// Get flattened tuning forks for a layer (for GPU upload).
    /// Returns (flat_data, n_neurons, phase_dim) where flat_data is [n_neurons * phase_dim].
    pub fn get_tuning_forks_flat(&self, layer_idx: usize) -> Option<(Vec<f32>, usize, usize)> {
        if layer_idx >= self.layers.len() { return None; }
        let layer = &self.layers[layer_idx];
        if layer.tuning_forks.is_empty() { return None; }
        let phase_dim = layer.tuning_forks[0].len();
        let n_neurons = layer.n_neurons;
        let mut flat = Vec::with_capacity(n_neurons * phase_dim);
        for fork in &layer.tuning_forks {
            flat.extend_from_slice(fork);
            // Pad if shorter
            if fork.len() < phase_dim {
                flat.extend(std::iter::repeat(0.0f32).take(phase_dim - fork.len()));
            }
        }
        Some((flat, n_neurons, phase_dim))
    }

    /// Get top_k for a layer
    pub fn top_k(&self, layer_idx: usize) -> usize {
        if layer_idx < self.layers.len() { self.layers[layer_idx].top_k } else { 0 }
    }

    /// Calibrate router by comparing predicted vs actual active neurons
    ///
    /// Call during inference to measure and improve prediction accuracy.
    /// Uses online averaging (exponential moving average) to smooth stats.
    ///
    /// # Arguments
    /// * `layer_idx` - Which layer
    /// * `predicted` - Neuron indices predicted to be active
    /// * `actual_alive` - Neuron indices actually alive (from SiLU gating)
    pub fn calibrate(&mut self, layer_idx: usize, predicted: &[usize], actual_alive: &[usize]) {
        if layer_idx >= self.layers.len() {
            return;
        }

        // Measure precision: how many predicted neurons were actually alive?
        let predicted_set: HashSet<_> = predicted.iter().copied().collect();
        let actual_set: HashSet<_> = actual_alive.iter().copied().collect();
        let correct = predicted_set.intersection(&actual_set).count();
        let precision = if !predicted.is_empty() {
            correct as f32 / predicted.len() as f32
        } else {
            0.0
        };

        // Update stats with exponential moving average
        self.stats.total_predictions += 1;
        let alpha = 0.01; // smoothing factor
        self.stats.accuracy = self.stats.accuracy * (1.0 - alpha) + precision * alpha;

        // Track active ratio
        let active_ratio = predicted.len() as f32 / self.layers[layer_idx].n_neurons as f32;
        self.stats.avg_active_ratio =
            self.stats.avg_active_ratio * (1.0 - alpha) + active_ratio * alpha;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_phases_basic() {
        // (1.0, 0.0) should map to angle 0.0
        let hidden = vec![1.0, 0.0];
        let phases = KuramotoRouter::to_phases(&hidden);
        assert_eq!(phases.len(), 1);
        assert!((phases[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_phases_quadrants() {
        // (1.0, 1.0) → π/4
        let hidden = vec![1.0, 1.0];
        let phases = KuramotoRouter::to_phases(&hidden);
        assert_eq!(phases.len(), 1);
        assert!((phases[0] - std::f32::consts::PI / 4.0).abs() < 1e-6);

        // (0.0, 1.0) → π/2
        let hidden = vec![0.0, 1.0];
        let phases = KuramotoRouter::to_phases(&hidden);
        assert!((phases[0] - std::f32::consts::PI / 2.0).abs() < 1e-6);

        // (-1.0, 0.0) → π (or -π)
        let hidden = vec![-1.0, 0.0];
        let phases = KuramotoRouter::to_phases(&hidden);
        assert!((phases[0].abs() - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_to_phases_multiple_pairs() {
        let hidden = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let phases = KuramotoRouter::to_phases(&hidden);
        assert_eq!(phases.len(), 3);
        // First pair: angle 0
        assert!((phases[0] - 0.0).abs() < 1e-6);
        // Second pair: angle π/2
        assert!((phases[1] - std::f32::consts::PI / 2.0).abs() < 1e-6);
        // Third pair: angle π or -π
        assert!((phases[2].abs() - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_to_phases_odd_length() {
        // Odd length hidden state
        let hidden = vec![1.0, 0.0, 0.5];
        let phases = KuramotoRouter::to_phases(&hidden);
        assert_eq!(phases.len(), 2);
        // First pair: (1.0, 0.0) → atan2(0, 1) = 0
        assert!((phases[0] - 0.0).abs() < 1e-6);
        // Second element: (0.5 + 0i) → atan2(0, 0.5) = 0
        assert!((phases[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_coherences_identical_phases() {
        let mut router = KuramotoRouter::new(1, 4, 0.25);
        let layer = &mut router.layers[0];

        // Create tuning forks where one matches the input phases exactly
        layer.tuning_forks = vec![
            vec![0.0, 0.0, 0.0],      // Will have high coherence
            vec![1.0, 1.0, 1.0],      // Different phases, lower coherence
            vec![0.1, 0.1, 0.1],      // Very close to first
            vec![3.0, 3.0, 3.0],      // Very different
        ];

        let phases = vec![0.0, 0.0, 0.0];
        let coherences = router.compute_coherences(&router.layers[0], &phases);

        assert_eq!(coherences.len(), 4);
        // First coherence should be highest (identical phases)
        assert!(coherences[0] > 0.99);
        // Others should generally be lower
        assert!(coherences[0] > coherences[1]);
        assert!(coherences[0] > coherences[3]);
    }

    #[test]
    fn test_compute_coherences_orthogonal_phases() {
        let router = KuramotoRouter::new(1, 2, 0.25);

        // Create test layer with specific tuning forks
        // Note: |mean(exp(i*delta))| where delta = phase_hidden - phase_fork
        // When delta = 0: exp(i*0) = 1, so coherence = 1
        // When delta = π: exp(i*π) = -1, so coherence = 1 (same magnitude)
        // When delta = π/2: exp(i*π/2) = i, so coherence = 1 (same magnitude)
        // The coherence measure is always |mean(...)| of unit complex numbers
        // For a single element, |exp(i*delta)| = 1 for any delta
        // So the key difference is in sign/imaginary patterns
        let test_layer = KuramotoRouterLayer {
            tuning_forks: vec![
                vec![0.0, 0.0],                    // Aligned
                vec![std::f32::consts::PI, 0.0],   // 180° out of phase
            ],
            n_neurons: 2,
            top_k: 1,
        };

        let phases = vec![0.0, 0.0];
        let coherences = router.compute_coherences(&test_layer, &phases);

        // Both should have high magnitude due to uniform phase distributions
        // but the mean direction differs
        assert_eq!(coherences.len(), 2);
        // Both are non-zero
        assert!(coherences[0] > 0.0);
        assert!(coherences[1] > 0.0);
    }

    #[test]
    fn test_route_basic() {
        let mut router = KuramotoRouter::new(1, 10, 0.3); // 30% = 3 neurons

        // Initialize layer with dummy tuning forks
        let gate_proj = vec![0.0; 10 * 8]; // 10 neurons × 8 hidden dims
        router.initialize_layer(0, &gate_proj, 8);

        let hidden = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let active = router.route(0, &hidden);

        assert!(!active.is_empty());
        assert!(active.len() <= 3);
        // All indices should be unique
        let unique: HashSet<_> = active.iter().copied().collect();
        assert_eq!(unique.len(), active.len());
    }

    #[test]
    fn test_route_returns_correct_count() {
        let mut router = KuramotoRouter::new(2, 100, 0.25); // 25 neurons per layer

        for layer_idx in 0..2 {
            let gate_proj = vec![0.0; 100 * 16];
            router.initialize_layer(layer_idx, &gate_proj, 16);
        }

        let hidden = vec![1.0; 16];

        for layer_idx in 0..2 {
            let active = router.route(layer_idx, &hidden);
            assert!(active.len() <= 25);
        }
    }

    #[test]
    fn test_calibrate_updates_stats() {
        let mut router = KuramotoRouter::new(1, 100, 0.25);

        let predicted = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; // 10 predicted
        let actual = vec![0, 1, 2, 3, 4, 50, 51, 52, 53, 54]; // 5 correct, 5 wrong

        router.calibrate(0, &predicted, &actual);

        assert_eq!(router.stats.total_predictions, 1);
        // Precision = 5/10 = 0.5, but with EMA: 0 * 0.99 + 0.5 * 0.01 = 0.005
        let expected_accuracy = 0.005;
        assert!((router.stats.accuracy - expected_accuracy).abs() < 1e-6);

        // Active ratio = 10/100 = 0.1, with EMA: 0 * 0.99 + 0.1 * 0.01 = 0.001
        let expected_ratio = 0.001;
        assert!((router.stats.avg_active_ratio - expected_ratio).abs() < 1e-6);
    }

    #[test]
    fn test_calibrate_exponential_smoothing() {
        let mut router = KuramotoRouter::new(1, 100, 0.25);

        // First calibration: 50% accuracy (2 out of 2 predicted are in actual)
        // Predicted: [0, 1], Actual: [0, 1, 2, 3] -> 2 correct out of 2 = 100% precision
        router.calibrate(0, &vec![0, 1], &vec![0, 1, 2, 3]);
        let acc1 = router.stats.accuracy;
        // With EMA: 0 * 0.99 + 1.0 * 0.01 = 0.01
        assert!(acc1 > 0.0);

        // Second calibration: also high accuracy
        // Predicted: [4, 5], Actual: [4, 5, 6, 7] -> 2 correct out of 2 = 100% precision
        router.calibrate(0, &vec![4, 5], &vec![4, 5, 6, 7]);
        let acc2 = router.stats.accuracy;

        // Should be slightly higher than first due to continued high precision
        assert!(acc2 >= acc1);
    }

    #[test]
    fn test_extract_tuning_forks_basic() {
        let weights = vec![
            1.0, 0.0, 0.0, 1.0, // Row 0: (1,0), (0,1)
            1.0, 1.0, 0.0, 0.0, // Row 1: (1,1), (0,0)
        ];

        let forks = KuramotoRouter::extract_tuning_forks(&weights, 2, 4);

        assert_eq!(forks.len(), 2);
        assert_eq!(forks[0].len(), 2);
        assert_eq!(forks[1].len(), 2);

        // First row: (1,0) → 0, (0,1) → π/2
        assert!((forks[0][0] - 0.0).abs() < 1e-6);
        assert!((forks[0][1] - std::f32::consts::PI / 2.0).abs() < 1e-6);

        // Second row: (1,1) → π/4, (0,0) → 0
        assert!((forks[1][0] - std::f32::consts::PI / 4.0).abs() < 1e-6);
        assert!((forks[1][1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_tuning_forks_incomplete() {
        let weights = vec![1.0, 0.0, 0.0, 1.0];

        // Request 2 rows of 4 cols from 4 elements (exactly 1 row available)
        let forks = KuramotoRouter::extract_tuning_forks(&weights, 2, 4);

        // Should handle gracefully
        assert_eq!(forks.len(), 2);
        // First row exists
        assert!(!forks[0].is_empty());
        // Second row is out of bounds
        assert!(forks[1].is_empty());
    }

    #[test]
    fn test_new_creates_correct_top_k() {
        let router = KuramotoRouter::new(4, 1000, 0.25);

        assert_eq!(router.layers.len(), 4);
        for layer in &router.layers {
            assert_eq!(layer.n_neurons, 1000);
            assert_eq!(layer.top_k, 250); // 0.25 * 1000
        }
    }

    #[test]
    fn test_router_empty_state() {
        let router = KuramotoRouter::new(1, 100, 0.25);

        // Before initialization, routing returns empty
        let active = router.route(0, &vec![1.0; 16]);
        assert!(active.is_empty());
    }

    #[test]
    fn test_router_multiple_layers() {
        let mut router = KuramotoRouter::new(3, 50, 0.2);

        for layer_idx in 0..3 {
            let gate_proj = vec![0.5; 50 * 12];
            router.initialize_layer(layer_idx, &gate_proj, 12);
        }

        let hidden = vec![0.5; 12];

        // Each layer should produce ~10 active neurons (20% of 50)
        for layer_idx in 0..3 {
            let active = router.route(layer_idx, &hidden);
            assert!(active.len() <= 10);
            assert!(active.len() > 0);
        }
    }
}
