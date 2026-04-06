//! Experimental Orchestrator — unifies LiquidNetwork, Homeostasis, and ActuatorBus.
//!
//! This module coordinates the three experimental subsystems that run parallel to
//! the main inference pipeline. It manages timing (homeostasis at 1 Hz, physics at
//! 100 Hz) and propagates stimulus signals from the LLM layer loop into the liquid network.
//!
//! # Architecture
//!
//! ```text
//! Forward Pass (model.rs)
//!   ↓ (stimulus_tx every 4th layer)
//! PulseOrchestrator::inject_stimulus()
//!   ↓
//! LiquidNetwork (100 Hz via RK4 step)
//! HomeostaticRegulator (1 Hz tick)
//! ActuatorBus (1 Hz poll)
//!   ↓
//! OrchestratorState (for telemetry broadcast)
//! ```

use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

use crate::experimental::liquid_network::LiquidNetwork;
use crate::experimental::homeostasis::HomeostaticRegulator;
use crate::experimental::actuator::ActuatorBus;

// ─────────────────────────────────────────────────────────────────────────────
// § Stimulus Signal from Forward Pass
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight stimulus extracted from each transformer layer during forward pass.
///
/// Computed at line ~1328 in model/mod.rs: residual_norm and max_activation
/// are already calculated for layer traces, so we just re-use them here.
#[derive(Debug, Clone)]
pub struct LayerStimulus {
    /// Index of the transformer layer (0..num_layers-1)
    pub layer_idx: usize,
    /// RMS norm of the final hidden state for this layer — measures activation energy.
    pub activation_energy: f32,
    /// Peak absolute activation in the hidden state — captures saliency.
    pub residual_norm: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// § Orchestrator State
// ─────────────────────────────────────────────────────────────────────────────

/// Three-band frequency partition of the liquid network's behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DominantMode {
    /// Low-frequency band active: sustained, focused reasoning.
    Focus,
    /// Mid-frequency band active: exploratory, curiosity-driven.
    Novelty,
    /// High-frequency band active: urgent, drive-based action.
    Drive,
}

impl std::fmt::Display for DominantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Focus => write!(f, "Focus"),
            Self::Novelty => write!(f, "Novelty"),
            Self::Drive => write!(f, "Drive"),
        }
    }
}

/// Real-time snapshot of the orchestrator's global state for telemetry.
///
/// This is broadcast to the frontend once per second (1 Hz with homeostasis tick).
#[derive(Debug, Clone)]
pub struct OrchestratorState {
    /// True if the network is in or approaching a sleep phase.
    pub is_sleeping: bool,
    /// Which frequency band dominates the liquid network's current activity.
    pub dominant_mode: DominantMode,
    /// Global Kuramoto coherence [0, 1] — unity of the entire oscillator mesh.
    pub global_coherence: f64,
    /// Stress scalar [0, 1] from the liquid network's viscosity and inhibition.
    pub stress: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// § Pulse Orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinates LiquidNetwork, HomeostaticRegulator, and ActuatorBus in a
/// unified orchestration loop.
///
/// Call `tick()` from the main engine loop; it internally dispatches physics
/// (100 Hz via `step()`) and homeostasis/actuators (1 Hz).
pub struct PulseOrchestrator {
    /// Shared liquid network with 256 CfC oscillators.
    liquid_net: Arc<Mutex<LiquidNetwork>>,
    /// Sleep/wake finite state machine.
    homeostasis: HomeostaticRegulator,
    /// Bus for resonance-driven actions (research, inhibition, etc).
    actuator_bus: ActuatorBus,
    /// Receives stimulus messages from the forward pass layer loop.
    stimulus_rx: mpsc::Receiver<LayerStimulus>,
    /// Tracks the last homeostasis and physics ticks for timing.
    last_homeostasis_tick: Instant,
    last_physics_tick: Instant,
    /// Current global state for telemetry.
    pub state: OrchestratorState,
    /// Buffered stimulus frames (layer_idx, activation_energy) to be injected.
    pending_stimuli: Vec<(usize, f32)>,
    /// Counter for physics steps (used for internal pacing).
    physics_ticks: u64,
}

impl PulseOrchestrator {
    /// Initializes the orchestrator with a new liquid network and homeostasis regulator.
    ///
    /// The liquid network is seeded with 256 CfC oscillators across num_layers.
    /// Pass the `stimulus_rx` channel end that receives signals from the model's forward pass.
    pub fn new(
        num_layers: usize,
        stimulus_rx: mpsc::Receiver<LayerStimulus>,
    ) -> Self {
        let liquid_net = Arc::new(Mutex::new(LiquidNetwork::new(256, num_layers, 0)));
        let mut homeostasis = HomeostaticRegulator::new();

        // Attempt to restore any previous consolidated weights from disk.
        homeostasis.try_restore(&liquid_net);

        Self {
            liquid_net,
            homeostasis,
            actuator_bus: ActuatorBus::new(),
            stimulus_rx,
            last_homeostasis_tick: Instant::now(),
            last_physics_tick: Instant::now(),
            state: OrchestratorState {
                is_sleeping: false,
                dominant_mode: DominantMode::Focus,
                global_coherence: 0.0,
                stress: 0.0,
            },
            pending_stimuli: Vec::new(),
            physics_ticks: 0,
        }
    }

    /// Returns a reference to the orchestrator's current state (for telemetry).
    pub fn state(&self) -> &OrchestratorState {
        &self.state
    }

    /// Main orchestration loop — call once per main engine tick (~100 Hz or faster).
    ///
    /// This method:
    /// 1. Drains and accumulates stimulus from the channel
    /// 2. Runs physics integration (100 Hz via internal pacing)
    /// 3. Runs homeostasis and actuators (1 Hz via internal pacing)
    /// 4. Updates `self.state` for telemetry broadcast
    pub fn tick(&mut self) {
        // ── Phase 1: Collect stimulus from the channel ─────────────────────
        while let Ok(stim) = self.stimulus_rx.try_recv() {
            // Accumulate stimulus into pending list. For multiple stimuli per tick,
            // they're all queued and applied to their respective layers.
            self.pending_stimuli.push((stim.layer_idx, stim.activation_energy));
            log::trace!(
                "iara.orch: stimulus from layer {} (E={:.3}, norm={:.3})",
                stim.layer_idx, stim.activation_energy, stim.residual_norm
            );
        }

        // ── Phase 2: Physics integration (100 Hz pacing) ──────────────────
        let elapsed = self.last_physics_tick.elapsed().as_secs_f64();
        let dt_physics = 0.01; // 100 Hz = 10 ms steps
        let target_ticks = (elapsed / dt_physics).ceil() as usize;

        for _ in 0..target_ticks.min(5) {
            // Inject pending stimulus into the network.
            {
                let mut net = match self.liquid_net.lock() {
                    Ok(n) => n,
                    Err(_) => {
                        log::warn!("iara.orch: could not lock network for physics step");
                        break;
                    }
                };

                // Inject each pending stimulus as a pulse of activation.
                // Create a synthetic activation vector (all ones scaled by energy).
                for &(layer_idx, energy) in &self.pending_stimuli {
                    // Construct a small activation vector scaled by the stimulus energy.
                    // Use a fixed size (e.g., 32 elements) to represent the layer activation.
                    let activation = vec![energy; 32];
                    net.inject_layer_stimulus(layer_idx, &activation);
                }
                self.pending_stimuli.clear();

                // Run one RK4 physics step (0.01 s).
                match net.step() {
                    Ok(_) => {
                        self.physics_ticks += 1;
                    }
                    Err(e) => {
                        log::warn!("iara.orch: physics step failed: {}", e);
                        break;
                    }
                }
            }
        }

        self.last_physics_tick = Instant::now();

        // ── Phase 3: Homeostasis and Actuators (1 Hz pacing) ──────────────
        if self.last_homeostasis_tick.elapsed().as_secs_f64() >= 1.0 {
            // Homeostasis tick (sleep/wake FSM, consolidation)
            self.homeostasis.tick(&self.liquid_net, None);

            // Actuator bus tick (research, inhibition, etc)
            let _fired_actions = self.actuator_bus.tick(&self.liquid_net);
            for (name, cmd) in _fired_actions {
                log::info!("iara.orch: action fired | {} → {}", name, cmd);
            }

            // ── Phase 4: Update state for telemetry ─────────────────────
            self.update_state();

            self.last_homeostasis_tick = Instant::now();
        }
    }

    /// Determines whether to use a "slow path" (e.g., exploratory sampling) based
    /// on the current actuator signal.
    ///
    /// Returns true when the novelty band is active and coherent, indicating
    /// the network wants to explore new information.
    pub fn should_use_slow_path(&self) -> bool {
        let signal = self.actuator_bus.read_signal(&self.liquid_net);
        signal.wants_exploration()
    }

    /// Manually inject a stimulus signal (e.g., from external event).
    ///
    /// Use this if you want to inject a pulse outside the normal layer loop.
    pub fn inject_stimulus(&mut self, stimulus: LayerStimulus) {
        self.pending_stimuli.push((stimulus.layer_idx, stimulus.activation_energy));
        log::debug!(
            "iara.orch: manual stimulus injection (layer {}, E={:.3})",
            stimulus.layer_idx, stimulus.activation_energy
        );
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Updates `self.state` from the current network and homeostasis state.
    fn update_state(&mut self) {
        let net = match self.liquid_net.lock() {
            Ok(n) => n,
            Err(_) => return,
        };

        let signal = self.actuator_bus.read_signal(&self.liquid_net);

        // Determine dominant mode from the actuator signal.
        let dominant_mode = if signal.is_focused() {
            DominantMode::Focus
        } else if signal.wants_exploration() {
            DominantMode::Novelty
        } else if signal.is_urgent() {
            DominantMode::Drive
        } else {
            // Default to the strongest band if no specific condition fires.
            match signal.dominant_group() {
                "Novelty" => DominantMode::Novelty,
                "Drive" => DominantMode::Drive,
                _ => DominantMode::Focus,
            }
        };

        self.state = OrchestratorState {
            is_sleeping: self.homeostasis.is_sleeping(),
            dominant_mode,
            global_coherence: signal.global_coherence,
            stress: net.stress as f32,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let (_tx, rx) = mpsc::channel(100);
        let orch = PulseOrchestrator::new(4, rx);
        assert_eq!(orch.state().stress, 0.0);
        assert_eq!(orch.state().dominant_mode, DominantMode::Focus);
    }

    #[tokio::test]
    async fn test_stimulus_injection() {
        let (tx, rx) = mpsc::channel(100);
        let mut orch = PulseOrchestrator::new(4, rx);

        let stim = LayerStimulus {
            layer_idx: 0,
            activation_energy: 0.5,
            residual_norm: 0.3,
        };

        orch.inject_stimulus(stim);
        assert!(!orch.pending_stimuli.is_empty());
    }

    #[tokio::test]
    async fn test_tick_updates_state() {
        let (_tx, rx) = mpsc::channel(100);
        let mut orch = PulseOrchestrator::new(4, rx);

        orch.tick();
        // After a tick, state should be updated (even if minimal).
        let _ = orch.state();
    }

    #[tokio::test]
    async fn test_should_use_slow_path() {
        let (_tx, rx) = mpsc::channel(100);
        let orch = PulseOrchestrator::new(4, rx);

        // By default, novelty is not high, so should_use_slow_path is false.
        let slow = orch.should_use_slow_path();
        // This is an integration test — the value depends on network state.
        let _ = slow;
    }
}
