// ============================================================================
// inference/importance.rs — Tensor Importance Instrumentation
// ============================================================================
//
// Lightweight tracking of tensor importance during inference using:
//   - Welford's algorithm for online mean/variance (no memory overhead)
//   - RMS activation magnitude tracking
//   - Usage frequency heuristic (how often output > threshold)
//   - Categorization into Alive/Dormant/Dead for pruning recommendations
//
// ============================================================================

use std::collections::HashMap;

/// Per-tensor importance statistics accumulated during inference
#[derive(Debug, Clone)]
pub struct ImportanceScore {
    /// Count of forward passes processed
    pub count: usize,

    /// Running mean of activation RMS magnitude (Welford)
    pub mean_activation_rms: f32,

    /// Running variance of activation RMS (Welford)
    pub variance_activation_rms: f32,

    /// M2 accumulator for Welford variance (internal use)
    pub m2_activation_rms: f32,

    /// Maximum activation seen in this tensor
    pub max_activation: f32,

    /// Count of times activation exceeded threshold
    pub activation_count_above_threshold: usize,

    /// Usage frequency: (activations above threshold) / (total forward passes)
    pub usage_frequency: f32,

    /// Estimated contribution to loss (simple magnitude-based heuristic)
    pub contribution_to_loss: f32,

    /// Tensor category
    pub category: TensorCategory,
}

impl ImportanceScore {
    /// Create a new empty score
    pub fn new() -> Self {
        Self {
            count: 0,
            mean_activation_rms: 0.0,
            variance_activation_rms: 0.0,
            m2_activation_rms: 0.0,
            max_activation: 0.0,
            activation_count_above_threshold: 0,
            usage_frequency: 0.0,
            contribution_to_loss: 0.0,
            category: TensorCategory::Alive,
        }
    }

    /// Update with new activation data using Welford's algorithm
    pub fn record_activation(&mut self, activation_rms: f32, max_activation: f32) {
        self.count += 1;
        self.max_activation = self.max_activation.max(max_activation);

        // Welford's online mean and variance
        let delta = activation_rms - self.mean_activation_rms;
        self.mean_activation_rms += delta / self.count as f32;

        let delta2 = activation_rms - self.mean_activation_rms;
        self.m2_activation_rms += delta * delta2;

        // Update variance
        if self.count > 1 {
            self.variance_activation_rms = self.m2_activation_rms / (self.count - 1) as f32;
        }

        // Track frequency of high activation
        const ACTIVATION_THRESHOLD: f32 = 0.01; // tunable
        if activation_rms > ACTIVATION_THRESHOLD {
            self.activation_count_above_threshold += 1;
        }

        // Update usage frequency
        self.usage_frequency = if self.count > 0 {
            self.activation_count_above_threshold as f32 / self.count as f32
        } else {
            0.0
        };

        // Estimated contribution to loss: product of mean and variance
        self.contribution_to_loss = self.mean_activation_rms * (1.0 + self.variance_activation_rms.sqrt());
    }
}

/// Classification of tensor importance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCategory {
    /// High importance: keep at current precision or higher
    Alive,

    /// Moderate importance: could be merged/combined with neighbors
    Dormant,

    /// Negligible importance: safe to prune
    Dead,
}

/// Main struct for tracking importance across all tensors
pub struct TensorImportance {
    /// Per-tensor importance scores
    pub scores: HashMap<String, ImportanceScore>,

    /// Total tokens processed
    pub total_tokens: usize,
}

impl TensorImportance {
    /// Initialize empty importance tracker
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            total_tokens: 0,
        }
    }

    /// Record activation data for a tensor in a single forward pass
    pub fn record_layer(
        &mut self,
        tensor_name: &str,
        activation_rms: f32,
        max_activation: f32,
    ) {
        self.scores
            .entry(tensor_name.to_string())
            .or_insert_with(ImportanceScore::new)
            .record_activation(activation_rms, max_activation);
    }

    /// Increment token counter
    pub fn increment_tokens(&mut self, count: usize) {
        self.total_tokens += count;
    }

    /// Categorize all tensors based on accumulated statistics
    pub fn categorize(&mut self) {
        if self.scores.is_empty() {
            return;
        }

        // Compute percentiles for categorization
        let mut contributions: Vec<f32> = self
            .scores
            .values()
            .map(|s| s.contribution_to_loss)
            .collect();
        contributions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = contributions.len();
        let p25 = if len > 0 {
            contributions[len / 4]
        } else {
            0.0
        };
        let p75 = if len > 0 {
            contributions[(3 * len) / 4]
        } else {
            0.0
        };

        // Apply thresholds
        for score in self.scores.values_mut() {
            score.category = if score.contribution_to_loss < p25 {
                TensorCategory::Dead
            } else if score.contribution_to_loss < p75 {
                TensorCategory::Dormant
            } else {
                TensorCategory::Alive
            };
        }
    }

    /// Get tensors classified as dead (prunable)
    pub fn prunable_tensors(&self) -> Vec<String> {
        self.scores
            .iter()
            .filter(|(_, score)| score.category == TensorCategory::Dead)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get tensors classified as dormant (mergeable)
    pub fn dormant_tensors(&self) -> Vec<String> {
        self.scores
            .iter()
            .filter(|(_, score)| score.category == TensorCategory::Dormant)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get tensors classified as alive (critical)
    pub fn alive_tensors(&self) -> Vec<String> {
        self.scores
            .iter()
            .filter(|(_, score)| score.category == TensorCategory::Alive)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Generate a summary report
    pub fn report(&self) -> ImportanceReport {
        let total_tensors = self.scores.len();
        let alive = self.scores.values().filter(|s| s.category == TensorCategory::Alive).count();
        let dormant = self.scores.values().filter(|s| s.category == TensorCategory::Dormant).count();
        let dead = self.scores.values().filter(|s| s.category == TensorCategory::Dead).count();

        let avg_contribution = if !self.scores.is_empty() {
            self.scores.values().map(|s| s.contribution_to_loss).sum::<f32>() / total_tensors as f32
        } else {
            0.0
        };

        ImportanceReport {
            total_tensors,
            alive,
            dormant,
            dead,
            total_tokens: self.total_tokens,
            avg_contribution,
        }
    }
}

impl Default for TensorImportance {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary report of tensor importance
#[derive(Debug, Clone)]
pub struct ImportanceReport {
    pub total_tensors: usize,
    pub alive: usize,
    pub dormant: usize,
    pub dead: usize,
    pub total_tokens: usize,
    pub avg_contribution: f32,
}

impl ImportanceReport {
    /// Pretty-print the report
    pub fn display(&self) -> String {
        vec![
            format!("Tensor Importance Analysis"),
            format!("─────────────────────────────────"),
            format!("Total tensors: {}", self.total_tensors),
            format!("  Alive:   {} ({:.1}%)", self.alive, 100.0 * self.alive as f32 / self.total_tensors.max(1) as f32),
            format!("  Dormant: {} ({:.1}%)", self.dormant, 100.0 * self.dormant as f32 / self.total_tensors.max(1) as f32),
            format!("  Dead:    {} ({:.1}%)", self.dead, 100.0 * self.dead as f32 / self.total_tensors.max(1) as f32),
            format!(""),
            format!("Tokens processed: {}", self.total_tokens),
            format!("Avg contribution to loss: {:.6}", self.avg_contribution),
            format!(""),
            format!("Recommendations:"),
            format!("  - Prune {} dead tensors for ~{:.1}% size reduction",
                self.dead,
                20.0 * self.dead as f32 / self.total_tensors.max(1) as f32),
            format!("  - Consider merging {} dormant tensors", self.dormant),
        ].join("\n")
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_score_welford() {
        let mut score = ImportanceScore::new();

        // Record activations: [1.0, 2.0, 3.0]
        score.record_activation(1.0, 1.0);
        assert_eq!(score.count, 1);
        assert!((score.mean_activation_rms - 1.0).abs() < 1e-6);

        score.record_activation(2.0, 2.0);
        assert_eq!(score.count, 2);
        assert!((score.mean_activation_rms - 1.5).abs() < 1e-6);

        score.record_activation(3.0, 3.0);
        assert_eq!(score.count, 3);
        assert!((score.mean_activation_rms - 2.0).abs() < 1e-6);

        // Variance of [1, 2, 3] is 1.0
        assert!((score.variance_activation_rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_importance_max_activation() {
        let mut score = ImportanceScore::new();

        score.record_activation(0.5, 1.0);
        score.record_activation(0.3, 5.0);
        score.record_activation(0.4, 2.0);

        assert_eq!(score.max_activation, 5.0);
    }

    #[test]
    fn test_importance_usage_frequency() {
        let mut score = ImportanceScore::new();

        const THRESHOLD: f32 = 0.01;

        // High activation (above threshold)
        score.record_activation(0.1, 0.1); // count_above = 1

        // Low activation
        score.record_activation(0.001, 0.001); // count_above = 1

        // High activation
        score.record_activation(0.05, 0.05); // count_above = 2

        assert_eq!(score.count, 3);
        assert_eq!(score.activation_count_above_threshold, 2);
        assert!((score.usage_frequency - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_importance_categorization() {
        let mut imp = TensorImportance::new();

        // Create tensors with different importance levels
        imp.record_layer("tensor_high_1", 0.5, 1.0);
        imp.record_layer("tensor_high_2", 0.4, 0.8);
        imp.record_layer("tensor_med_1", 0.1, 0.2);
        imp.record_layer("tensor_low_1", 0.01, 0.02);
        imp.record_layer("tensor_low_2", 0.005, 0.01);

        imp.categorize();

        let alive = imp.alive_tensors();
        let dormant = imp.dormant_tensors();
        let dead = imp.prunable_tensors();

        assert!(alive.contains(&"tensor_high_1".to_string()));
        assert!(alive.contains(&"tensor_high_2".to_string()));
        assert!(dead.contains(&"tensor_low_1".to_string()) || dead.contains(&"tensor_low_2".to_string()));

        let report = imp.report();
        assert_eq!(report.total_tensors, 5);
        assert!(report.alive > 0);
    }

    #[test]
    fn test_importance_report() {
        let mut imp = TensorImportance::new();

        for i in 0..10 {
            imp.record_layer(&format!("tensor_{}", i), 0.1 * i as f32, 0.2 * i as f32);
        }
        imp.increment_tokens(100);

        imp.categorize();
        let report = imp.report();

        assert_eq!(report.total_tensors, 10);
        assert_eq!(report.total_tokens, 100);
        assert!(report.alive > 0);
        assert!(report.dead > 0);

        let display = report.display();
        assert!(display.contains("Total tensors: 10"));
        assert!(display.contains("Tokens processed: 100"));
    }
}
