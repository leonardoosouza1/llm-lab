// ============================================================================
// debug/brain.rs — BrainTrace: Complete Execution Timeline
// ============================================================================
// Captures everything that happened during generation
// Allows retroactive analysis of what the model did and why
// ============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::qwen::{GenerationStep, LayerActivationTrace};

// ─────────────────────────────────────────────────────────────────────────────
// Core Type: BrainTrace
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainTrace {
    /// Original input prompt
    pub prompt: String,

    /// Full generated response
    pub response: String,

    /// All generation steps (token by token)
    pub steps: Vec<GenerationStep>,

    /// All IARA simulation snapshots (parallel to steps)
    pub iara_snapshots: Vec<SimulationSnapshot>,

    /// For each step, which layers fired (indices)
    pub layer_activation_path: Vec<Vec<usize>>,

    /// Energy (sum of activations) over time
    pub energy_trajectory: Vec<f32>,

    /// Heatmap of attention patterns across all steps
    pub attention_heatmap: Vec<Vec<f32>>,

    /// Timestamp when trace was finalized
    pub finalized_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSnapshot {
    /// Number of active nodes in IARA
    pub nodes_active: usize,

    /// Mean energy across all nodes
    pub mean_energy: f32,

    /// Phase coherence (0-1, higher = more synchronized)
    pub phase_coherence: f32,

    /// Time when snapshot was taken (ms from start)
    pub timestamp_ms: u128,
}

impl BrainTrace {
    /// Create new trace for a prompt
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            response: String::new(),
            steps: Vec::new(),
            iara_snapshots: Vec::new(),
            layer_activation_path: Vec::new(),
            energy_trajectory: Vec::new(),
            attention_heatmap: Vec::new(),
            finalized_at: None,
        }
    }

    /// Add a generation step
    pub fn add_step(&mut self, step: GenerationStep) {
        // Accumulate response
        self.response.push_str(&step.token_str);

        // Track which layers fired
        let active_layers = step.active_layers.clone();
        self.layer_activation_path.push(active_layers);

        // Track energy
        let step_energy: f32 = step.layer_traces.iter().map(|t| t.max_activation).sum();
        self.energy_trajectory.push(step_energy);

        // Store step
        self.steps.push(step);
    }

    /// Add IARA simulation snapshot
    pub fn add_snapshot(&mut self, snapshot: SimulationSnapshot) {
        self.iara_snapshots.push(snapshot);
    }

    /// Finalize trace (mark as complete, compute analysis)
    pub fn finalize(&mut self) {
        self.finalized_at = Some(chrono::Utc::now().to_rfc3339());
    }

    /// Analyze the trace and return insights
    pub fn analyze(&self) -> BrainAnalysis {
        BrainAnalysis {
            most_active_layers: self.most_active_layers(),
            energy_progression: self.energy_progression(),
            attention_flow: self.attention_flow(),
            iara_resonance_score: self.iara_resonance_score(),
            layer_timeline: self.layer_timeline(),
            anomalies: self.detect_anomalies(),
            summary: self.generate_summary(),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Analysis Methods
    // ────────────────────────────────────────────────────────────────────

    /// Find most active layers across all steps
    fn most_active_layers(&self) -> Vec<(usize, f32)> {
        let mut layer_energy: HashMap<usize, f32> = HashMap::new();

        for step in &self.steps {
            for trace in &step.layer_traces {
                *layer_energy.entry(trace.layer_idx).or_insert(0.0) +=
                    trace.max_activation;
            }
        }

        let mut ranked: Vec<_> = layer_energy.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ranked.into_iter().take(10).collect()
    }

    /// Energy progression (mean energy per step)
    fn energy_progression(&self) -> Vec<f32> {
        self.steps
            .iter()
            .map(|step| {
                if step.layer_traces.is_empty() {
                    0.0
                } else {
                    step.layer_traces.iter().map(|t| t.max_activation).sum::<f32>()
                        / step.layer_traces.len() as f32
                }
            })
            .collect()
    }

    /// How attention "flows" from beginning to end
    fn attention_flow(&self) -> AttentionFlow {
        let early_focus = if !self.attention_heatmap.is_empty() {
            self.attention_heatmap[0].clone()
        } else {
            vec![]
        };

        let late_focus = if !self.attention_heatmap.is_empty() {
            self.attention_heatmap[self.attention_heatmap.len() - 1].clone()
        } else {
            vec![]
        };

        let transition = Self::compute_attention_shift(&early_focus, &late_focus);

        AttentionFlow {
            early_focus,
            late_focus,
            transition,
        }
    }

    /// How well IARA "resonated" with the generation
    /// High score = coherent simulation + good model output
    fn iara_resonance_score(&self) -> f32 {
        if self.iara_snapshots.is_empty() {
            0.0
        } else {
            let mean_coherence = self.iara_snapshots
                .iter()
                .map(|s| s.phase_coherence)
                .sum::<f32>()
                / self.iara_snapshots.len() as f32;

            // Normalize to 0-1
            (mean_coherence * 100.0).min(1.0)
        }
    }

    /// Timeline: which layers were active at each step
    fn layer_timeline(&self) -> Vec<StepLayerInfo> {
        self.steps
            .iter()
            .enumerate()
            .map(|(idx, step)| StepLayerInfo {
                step_num: idx,
                token: step.token_str.clone(),
                active_layers: step.active_layers.clone(),
                total_energy: step
                    .layer_traces
                    .iter()
                    .map(|t| t.max_activation)
                    .sum(),
            })
            .collect()
    }

    /// Detect anomalies (unexpected activation patterns)
    fn detect_anomalies(&self) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // Check for energy spikes
        let mean_energy = self.energy_trajectory.iter().sum::<f32>()
            / self.energy_trajectory.len().max(1) as f32;
        let std_dev = (self.energy_trajectory
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / self.energy_trajectory.len().max(1) as f32)
            .sqrt();

        for (idx, &energy) in self.energy_trajectory.iter().enumerate() {
            if energy > mean_energy + 2.0 * std_dev {
                anomalies.push(Anomaly {
                    step: idx,
                    kind: "energy_spike".to_string(),
                    severity: ((energy - mean_energy) / std_dev).min(5.0),
                    description: format!(
                        "Unusual energy spike at token {}: {:.2} (mean {:.2})",
                        idx, energy, mean_energy
                    ),
                });
            }
        }

        // Check for layer dropout (layers that suddenly stop activating)
        for layer_idx in 0..36 {
            let activations: Vec<bool> = self.steps
                .iter()
                .map(|s| s.active_layers.contains(&layer_idx))
                .collect();

            // Find sudden drops
            for window in activations.windows(3) {
                if window[0] && !window[1] && !window[2] {
                    anomalies.push(Anomaly {
                        step: self.steps.len() / 2,
                        kind: "layer_dropout".to_string(),
                        severity: 1.0,
                        description: format!("Layer {} stopped activating", layer_idx),
                    });
                    break;
                }
            }
        }

        anomalies
    }

    /// Generate human-readable summary
    fn generate_summary(&self) -> String {
        let top_layers = self.most_active_layers();
        let mean_energy = self.energy_trajectory.iter().sum::<f32>()
            / self.energy_trajectory.len().max(1) as f32;

        format!(
            "Generated {} tokens from prompt '{}'. \
             Top active layers: {:?}. \
             Mean energy: {:.2}. \
             IARA resonance: {:.2}.",
            self.steps.len(),
            &self.prompt[..self.prompt.len().min(30)],
            top_layers.iter().map(|(l, _)| l).take(3).collect::<Vec<_>>(),
            mean_energy,
            self.iara_resonance_score()
        )
    }

    fn compute_attention_shift(early: &[f32], late: &[f32]) -> f32 {
        if early.is_empty() || late.is_empty() {
            0.0
        } else {
            // Compute cosine distance between attention patterns
            let dot = early.iter().zip(late.iter()).map(|(a, b)| a * b).sum::<f32>();
            let norm_early = early.iter().map(|a| a * a).sum::<f32>().sqrt();
            let norm_late = late.iter().map(|a| a * a).sum::<f32>().sqrt();

            if norm_early > 0.0 && norm_late > 0.0 {
                dot / (norm_early * norm_late)
            } else {
                0.0
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis Output Types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct BrainAnalysis {
    pub most_active_layers: Vec<(usize, f32)>,
    pub energy_progression: Vec<f32>,
    pub attention_flow: AttentionFlow,
    pub iara_resonance_score: f32,
    pub layer_timeline: Vec<StepLayerInfo>,
    pub anomalies: Vec<Anomaly>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AttentionFlow {
    pub early_focus: Vec<f32>,
    pub late_focus: Vec<f32>,
    pub transition: f32, // Cosine similarity (0-1)
}

#[derive(Debug, Clone, Serialize)]
pub struct StepLayerInfo {
    pub step_num: usize,
    pub token: String,
    pub active_layers: Vec<usize>,
    pub total_energy: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Anomaly {
    pub step: usize,
    pub kind: String,
    pub severity: f32,
    pub description: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Export Formats
// ─────────────────────────────────────────────────────────────────────────────

impl BrainTrace {
    /// Export to JSON for external analysis
    pub fn to_json(&self) -> Result<String, serde_json::error::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export analysis to JSON
    pub fn analysis_to_json(&self) -> Result<String, serde_json::error::Error> {
        let analysis = self.analyze();
        serde_json::to_string_pretty(&analysis)
    }

    /// Export as CSV for spreadsheet analysis
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("step,token,energy,active_layers_count\n");

        for (idx, step) in self.steps.iter().enumerate() {
            let energy: f32 = step.layer_traces.iter().map(|t| t.max_activation).sum();
            csv.push_str(&format!(
                "{},{},{:.2},{}\n",
                idx,
                step.token_str,
                energy,
                step.active_layers.len()
            ));
        }

        csv
    }
}
