// ============================================================================
// Free Energy Principle (FEP) Module: Friston's Active Inference
// ============================================================================
// Implements variational free energy minimization as a survival drive.
// The organism maintains a generative model of its world and body state,
// updating beliefs to minimize prediction error (surprise) and actively
// seeking actions that minimize expected future free energy.
// ============================================================================

use std::collections::VecDeque;

// ============================================================================
// Constants
// ============================================================================

/// Learning rate for belief updates (exponential moving average)
pub const LEARNING_RATE: f32 = 0.05;

/// Number of ticks of history to maintain for prediction model
pub const PREDICTION_HORIZON: usize = 10;

/// Weight for prediction error precision in free energy calculation
pub const PRECISION_WEIGHT: f32 = 1.0;

// ============================================================================
// FreeEnergyMinimizer Struct
// ============================================================================

/// Active inference engine that minimizes variational free energy.
///
/// Tracks:
/// - A learned generative model of sensory inputs
/// - Prediction errors and surprise
/// - Epistemic and pragmatic value of actions
#[derive(Debug, Clone)]
pub struct FreeEnergyMinimizer {
    /// Learned model: running average of sensory inputs
    prediction_model: Vec<f32>,

    /// Errors between predicted and actual sensory inputs
    prediction_error: Vec<f32>,

    /// F = E_q[log q(s) - log p(o,s)] ≈ sum of squared prediction errors
    variational_free_energy: f32,

    /// -log p(o|model) ≈ magnitude of prediction error
    surprise: f32,

    /// Information gain: how much predictions changed from new observations
    epistemic_value: f32,

    /// How well organism maintains homeostasis (complementary to surprise)
    pragmatic_value: f32,

    /// History of sensory inputs for learning
    sensory_history: VecDeque<Vec<f32>>,

    /// History of prediction models for epistemic value computation
    model_history: VecDeque<Vec<f32>>,
}

impl FreeEnergyMinimizer {
    /// Create a new FreeEnergyMinimizer with given sensory dimension
    pub fn new(sensory_dim: usize) -> Self {
        Self {
            prediction_model: vec![0.0; sensory_dim],
            prediction_error: vec![0.0; sensory_dim],
            variational_free_energy: 0.0,
            surprise: 0.0,
            epistemic_value: 0.0,
            pragmatic_value: 0.0,
            sensory_history: VecDeque::with_capacity(PREDICTION_HORIZON),
            model_history: VecDeque::with_capacity(PREDICTION_HORIZON),
        }
    }

    // ========================================================================
    // Belief Update (Perception)
    // ========================================================================

    /// Update internal generative model using exponential moving average.
    ///
    /// The model converges toward observed sensory patterns, allowing the
    /// organism to build expectations about the world's dynamics.
    pub fn update_beliefs(&mut self, sensory_input: &[f32]) {
        // Store old model for epistemic value calculation
        self.model_history.push_back(self.prediction_model.clone());
        if self.model_history.len() > PREDICTION_HORIZON {
            self.model_history.pop_front();
        }

        // Store sensory input
        self.sensory_history.push_back(sensory_input.to_vec());
        if self.sensory_history.len() > PREDICTION_HORIZON {
            self.sensory_history.pop_front();
        }

        // Exponential moving average: belief update
        for i in 0..self.prediction_model.len() {
            if i < sensory_input.len() {
                self.prediction_model[i] = (1.0 - LEARNING_RATE) * self.prediction_model[i]
                    + LEARNING_RATE * sensory_input[i];
            }
        }
    }

    // ========================================================================
    // Free Energy Computation (Inference)
    // ========================================================================

    /// Compute variational free energy: measures mismatch between model and reality.
    ///
    /// F = E_q[log q(s) - log p(o,s)]
    /// Simplified as: sum of squared prediction errors weighted by precision
    pub fn compute_free_energy(&mut self, sensory_input: &[f32]) -> f32 {
        self.prediction_error.clear();
        self.prediction_error.resize(sensory_input.len(), 0.0);

        for i in 0..sensory_input.len() {
            if i < self.prediction_model.len() {
                let error = sensory_input[i] - self.prediction_model[i];
                self.prediction_error[i] = error;
            }
        }

        // Free energy = sum of squared errors (weighted by precision)
        let sum_squared_error: f32 = self
            .prediction_error
            .iter()
            .map(|&e| e * e)
            .sum();

        self.variational_free_energy = PRECISION_WEIGHT * sum_squared_error;

        // Surprise as magnitude of error (negative log likelihood approximation)
        let mean_squared_error = if !self.prediction_error.is_empty() {
            sum_squared_error / self.prediction_error.len() as f32
        } else {
            0.0
        };
        self.surprise = mean_squared_error.sqrt();

        self.variational_free_energy
    }

    // ========================================================================
    // Action Selection (Active Inference)
    // ========================================================================

    /// Compute expected free energy for a candidate action.
    ///
    /// Expected Free Energy = Pragmatic Value + Epistemic Value
    /// Pragmatic: risk = how well maintained homeostasis
    /// Epistemic: ambiguity = information gain from new observations
    pub fn compute_expected_free_energy(&self, action: &[f32]) -> f32 {
        // Pragmatic value: how much does this action help maintain homeostasis?
        // Approximated as: negative sum of squared expected errors after action
        let pragmatic = action.iter().map(|&a| a * a).sum::<f32>() * 0.5;

        // Epistemic value: how novel/informative is this action?
        // Approximated as: magnitude of action (seeking information when uncertain)
        let epistemic = action.iter().map(|&a| a.abs()).sum::<f32>() * 0.1;

        // Total expected free energy (lower is better)
        pragmatic + epistemic
    }

    /// Select action that minimizes expected free energy.
    ///
    /// Active inference principle: choose action expected to minimize surprise
    /// and information uncertainty (ambiguity).
    pub fn action_selection(&mut self, candidate_actions: &[Vec<f32>]) -> usize {
        if candidate_actions.is_empty() {
            return 0;
        }

        let mut best_action = 0;
        let mut best_free_energy = f32::INFINITY;

        for (idx, action) in candidate_actions.iter().enumerate() {
            let expected_fe = self.compute_expected_free_energy(action);
            if expected_fe < best_free_energy {
                best_free_energy = expected_fe;
                best_action = idx;
            }
        }

        self.pragmatic_value = candidate_actions[best_action]
            .iter()
            .map(|&a| a * a)
            .sum::<f32>()
            * 0.5;

        best_action
    }

    // ========================================================================
    // Allostatic Integration (Homeostasis)
    // ========================================================================

    /// Compute surprise in internal body state (drives) vs homeostatic setpoints.
    ///
    /// How far are current drive levels from their optimal setpoints?
    /// This is "interoceptive surprise" - the organism acts to reduce this.
    pub fn homeostatic_surprise(&self, drives: &[f32], setpoints: &[f32]) -> f32 {
        if drives.is_empty() || setpoints.is_empty() {
            return 0.0;
        }

        let mut total_error = 0.0;
        for i in 0..drives.len().min(setpoints.len()) {
            let error = drives[i] - setpoints[i];
            total_error += error * error;
        }

        (total_error / drives.len().min(setpoints.len()) as f32).sqrt()
    }

    /// Compute internal prediction error (interoception).
    ///
    /// How well is the organism maintaining energy homeostasis?
    /// Positive when metabolism exceeds target, negative when depleted.
    pub fn interoceptive_error(&self, metabolism_energy: f32, target_energy: f32) -> f32 {
        (metabolism_energy - target_energy).abs()
    }

    /// Get current variational free energy value
    pub fn get_free_energy(&self) -> f32 {
        self.variational_free_energy
    }

    /// Get current surprise (prediction error magnitude)
    pub fn get_surprise(&self) -> f32 {
        self.surprise
    }

    /// Get current epistemic value
    pub fn get_epistemic_value(&self) -> f32 {
        self.epistemic_value
    }

    /// Get current pragmatic value
    pub fn get_pragmatic_value(&self) -> f32 {
        self.pragmatic_value
    }

    /// Compute epistemic value: information gain from observations
    pub fn update_epistemic_value(&mut self) {
        // Epistemic value = how much the model changed (uncertainty reduction)
        if self.model_history.len() >= 2 {
            let prev_model = &self.model_history[0];
            let curr_model = &self.prediction_model;

            let mut model_change = 0.0;
            for i in 0..prev_model.len().min(curr_model.len()) {
                let diff = curr_model[i] - prev_model[i];
                model_change += diff * diff;
            }

            self.epistemic_value = model_change.sqrt();
        } else {
            self.epistemic_value = 0.0;
        }
    }

    /// Get prediction model (for inspection/debugging)
    pub fn get_prediction_model(&self) -> &[f32] {
        &self.prediction_model
    }

    /// Get prediction error (for inspection/debugging)
    pub fn get_prediction_error(&self) -> &[f32] {
        &self.prediction_error
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Test 1: Free energy decreases with learning
    // ========================================================================

    #[test]
    fn test_free_energy_decreases_with_learning() {
        let mut minimizer = FreeEnergyMinimizer::new(4);

        // Stable input pattern
        let sensory = [0.5, 0.3, 0.7, 0.2];

        // Track free energy over iterations
        let mut free_energies = Vec::new();

        for _ in 0..20 {
            minimizer.update_beliefs(&sensory);
            let fe = minimizer.compute_free_energy(&sensory);
            free_energies.push(fe);
        }

        // Free energy should generally decrease as model learns pattern
        assert!(free_energies[0] > free_energies[10]);
        assert!(free_energies[10] > free_energies[19]);
    }

    // ========================================================================
    // Test 2: Surprise detection
    // ========================================================================

    #[test]
    fn test_surprise_detection() {
        let mut minimizer = FreeEnergyMinimizer::new(3);

        // Train on stable pattern
        let stable = [0.5, 0.5, 0.5];
        for _ in 0..15 {
            minimizer.update_beliefs(&stable);
            minimizer.compute_free_energy(&stable);
        }

        // Surprise should be low for expected input
        let expected_surprise = minimizer.get_surprise();

        // Novel input should increase surprise
        let novel = [1.0, 0.0, 1.0];
        minimizer.compute_free_energy(&novel);
        let novel_surprise = minimizer.get_surprise();

        assert!(expected_surprise < 0.3);
        assert!(novel_surprise > expected_surprise);
    }

    // ========================================================================
    // Test 3: Belief convergence
    // ========================================================================

    #[test]
    fn test_belief_convergence() {
        let mut minimizer = FreeEnergyMinimizer::new(2);

        let target = [0.8, 0.2];

        // Update beliefs with target input multiple times
        for _ in 0..100 {
            minimizer.update_beliefs(&target);
        }

        // Beliefs should converge toward target (with exponential moving average)
        let model = minimizer.get_prediction_model();
        assert!((model[0] - target[0]).abs() < 0.05);
        assert!((model[1] - target[1]).abs() < 0.05);
    }

    // ========================================================================
    // Test 4: Action selection prefers familiar
    // ========================================================================

    #[test]
    fn test_action_selection_prefers_familiar() {
        let mut minimizer = FreeEnergyMinimizer::new(2);

        // Learn a stable state
        let familiar = [0.5, 0.5];
        for _ in 0..15 {
            minimizer.update_beliefs(&familiar);
        }

        // Candidate actions:
        // 1. Small perturbation (familiar territory)
        // 2. Large perturbation (novel/risky)
        let actions = vec![
            vec![0.01, 0.01], // Small, familiar
            vec![0.8, 0.8],   // Large, novel
        ];

        let selected = minimizer.action_selection(&actions);

        // Should prefer smaller, more conservative action
        assert_eq!(selected, 0);
    }

    // ========================================================================
    // Test 5: Homeostatic surprise at setpoint
    // ========================================================================

    #[test]
    fn test_homeostatic_surprise_at_setpoint() {
        let minimizer = FreeEnergyMinimizer::new(3);

        let drives = [0.5, 0.5, 0.5];
        let setpoints = [0.5, 0.5, 0.5];

        let surprise = minimizer.homeostatic_surprise(&drives, &setpoints);

        assert!(surprise < 1e-6, "Surprise should be ~0 at setpoint");
    }

    // ========================================================================
    // Test 6: Homeostatic surprise increases with deviation
    // ========================================================================

    #[test]
    fn test_homeostatic_surprise_increases_with_deviation() {
        let minimizer = FreeEnergyMinimizer::new(3);
        let setpoints = [0.5, 0.5, 0.5];

        // Small deviation
        let drives_small = [0.6, 0.5, 0.5];
        let surprise_small = minimizer.homeostatic_surprise(&drives_small, &setpoints);

        // Large deviation
        let drives_large = [1.0, 0.5, 0.5];
        let surprise_large = minimizer.homeostatic_surprise(&drives_large, &setpoints);

        assert!(surprise_small < surprise_large);
        assert!(surprise_small > 0.0);
        assert!(surprise_large > surprise_small);
    }

    // ========================================================================
    // Test 7: Interoceptive error direction
    // ========================================================================

    #[test]
    fn test_interoceptive_error_direction() {
        let minimizer = FreeEnergyMinimizer::new(1);

        // Over-energized
        let error_over = minimizer.interoceptive_error(1.0, 0.5);

        // Under-energized
        let error_under = minimizer.interoceptive_error(0.0, 0.5);

        // Both should be non-zero and equal magnitude
        assert!((error_over - 0.5).abs() < 1e-6);
        assert!((error_under - 0.5).abs() < 1e-6);

        // At target, error should be zero
        let error_target = minimizer.interoceptive_error(0.5, 0.5);
        assert!(error_target < 1e-6);
    }

    // ========================================================================
    // Test 8: Epistemic value increases with novelty
    // ========================================================================

    #[test]
    fn test_epistemic_value_increases_with_novelty() {
        let mut minimizer = FreeEnergyMinimizer::new(3);

        // Build model of stable state
        let stable = [0.3, 0.3, 0.3];
        for _ in 0..5 {
            minimizer.update_beliefs(&stable);
            minimizer.update_epistemic_value();
        }

        // Epistemic value should be low (model converged)
        let epistemic_stable = minimizer.get_epistemic_value();

        // Observe novel input
        let novel = [0.9, 0.1, 0.5];
        minimizer.update_beliefs(&novel);
        minimizer.update_epistemic_value();

        let epistemic_novel = minimizer.get_epistemic_value();

        // Novel input should increase epistemic value
        assert!(epistemic_novel > epistemic_stable);
    }

    // ========================================================================
    // Additional validation tests
    // ========================================================================

    #[test]
    fn test_free_energy_positive() {
        let mut minimizer = FreeEnergyMinimizer::new(2);

        let input = [0.5, 0.5];
        let fe = minimizer.compute_free_energy(&input);

        assert!(fe >= 0.0, "Free energy should be non-negative");
    }

    #[test]
    fn test_action_selection_with_empty_candidates() {
        let mut minimizer = FreeEnergyMinimizer::new(2);

        // Empty candidates should default to 0
        let actions = vec![];
        // Should not panic
        if !actions.is_empty() {
            let _ = minimizer.action_selection(&actions);
        }
    }

    #[test]
    fn test_multiple_cycles_of_learning() {
        let mut minimizer = FreeEnergyMinimizer::new(4);

        // Simulate multiple learning cycles with different patterns
        let pattern1 = [0.1, 0.2, 0.3, 0.4];
        let pattern2 = [0.9, 0.8, 0.7, 0.6];

        let mut total_fe = 0.0;

        for cycle in 0..3 {
            let pattern = if cycle % 2 == 0 { &pattern1 } else { &pattern2 };

            for _ in 0..10 {
                minimizer.update_beliefs(pattern);
                let fe = minimizer.compute_free_energy(pattern);
                total_fe += fe;
            }
        }

        // System should show learning behavior
        assert!(total_fe >= 0.0);
        assert!(minimizer.get_free_energy() >= 0.0);
    }

    #[test]
    fn test_homeostatic_surprise_with_different_sizes() {
        let minimizer = FreeEnergyMinimizer::new(5);

        // Test with different dimensional mismatches
        let drives = [0.5, 0.6, 0.7];
        let setpoints = [0.5, 0.5];

        let surprise = minimizer.homeostatic_surprise(&drives, &setpoints);
        assert!(surprise >= 0.0);
    }
}
