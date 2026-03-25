//! Homeostatic regulation and circadian cycle management for IARA.
//!
//! This module drives the **digital sleep/wake cycle** of the liquid network.
//! It monitors rolling entropy, counts idle time, and transitions the network
//! through its circadian phases — triggering consolidation during deep sleep.
//!
//! # State Machine
//!
//! ```text
//! Awake ──(idle ≥ 300 s)──→ Drowsy ──(idle ≥ 600 s)──→ LightSleep
//!   ↑                            │                           │
//!   └── Waking ←── DeepSleep ←──┴──── (idle ≥ 900 s) ──────┘
//!         ↑              │
//!         └── (stimulus) ┘
//! ```
//!
//! # Consolidation Routine (Deep Sleep)
//!
//! 1. **Synaptic pruning** — zero weak connections (|W| < 0.02)
//! 2. **Homeostatic re-normalisation** — restore L1-norm balance post-pruning
//! 3. **Disk snapshot** — persist the refined weight matrix to `/tmp/iara_neural_state.bin`
//! 4. **Activation reset** — clear transient working memory; long-term memory stays intact
//!
//! # EWC State Commitment
//!
//! After the first deep-sleep save, the current weight matrix is committed as the
//! EWC anchor via `LiquidNetwork::commit_anchor()`. This prevents future Oja/STDP
//! updates from catastrophically overwriting the consolidated semantic knowledge.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::liquid_network::{CircadianPhase, LiquidNetwork, LlmTelemetry};

// ─────────────────────────────────────────────────────────────────────────────
// § Transition thresholds (seconds without external stimulus)
// ─────────────────────────────────────────────────────────────────────────────

/// Idle seconds before Awake → Drowsy.
const IDLE_DROWSY_S: u64 = 300;
/// Idle seconds before Drowsy → LightSleep.
const IDLE_LIGHT_S: u64 = 600;
/// Idle seconds before LightSleep → DeepSleep.
const IDLE_DEEP_S: u64 = 900;

/// Prune connections below this absolute magnitude during DeepSleep GC.
const PRUNE_THRESHOLD: f64 = 0.02;

/// Target mean activation for intrinsic plasticity (homeostatic firing rate).
const TARGET_ACTIVATION: f64 = 0.2;

/// Entropy below this (during idle) accelerates sleep onset.
const LOW_ENTROPY_THRESHOLD: f64 = 0.20;

/// Rolling entropy window size (samples ≈ seconds at 1 Hz tick rate).
const ENTROPY_WINDOW: usize = 60;

/// Path for disk snapshots of the neural weight matrix.
const STATE_PATH: &str = "/tmp/iara_neural_state.bin";

// ─────────────────────────────────────────────────────────────────────────────
// § HomeostaticRegulator
// ─────────────────────────────────────────────────────────────────────────────

/// Drives the circadian state machine and triggers sleep consolidation.
///
/// Call [`HomeostaticRegulator::tick`] at **1 Hz** (every second), passing
/// the most recent telemetry if one arrived since the last call.
///
/// On startup, call [`HomeostaticRegulator::try_restore`] to reload any
/// previously consolidated weight matrix from disk.
pub struct HomeostaticRegulator {
    /// Rolling window of entropy samples (one per tick).
    entropy_window: VecDeque<f64>,
    /// Monotonic clock of the last external stimulus.
    last_stimulus: Instant,
    /// Guards against running consolidation twice per sleep cycle.
    consolidation_done: bool,
    /// EWC anchor committed: guards against re-committing on every wake.
    anchor_committed: bool,
    /// Path for persisting neural state snapshots.
    state_path: PathBuf,
}

impl HomeostaticRegulator {
    /// Creates a fresh regulator. Call [`try_restore`] immediately after.
    pub fn new() -> Self {
        Self {
            entropy_window: VecDeque::with_capacity(ENTROPY_WINDOW),
            last_stimulus: Instant::now(),
            consolidation_done: false,
            anchor_committed: false,
            state_path: PathBuf::from(STATE_PATH),
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Main loop body — call once per second.
    ///
    /// Accepts `telemetry: Option<&LlmTelemetry>` — `Some` means a token
    /// arrived from the LLM since the last tick; `None` means the network
    /// was idle.
    pub fn tick(
        &mut self,
        network: &Arc<Mutex<LiquidNetwork>>,
        telemetry: Option<&LlmTelemetry>,
    ) {
        // ── Record stimulus ────────────────────────────────────────────────
        if let Some(tel) = telemetry {
            self.push_entropy(tel.entropy_chaos);
            self.last_stimulus = Instant::now();
            self.consolidation_done = false; // new waking period resets flag
        }

        let idle_s = self.last_stimulus.elapsed().as_secs();
        let mean_h = self.mean_entropy();

        let mut net = match network.lock() {
            Ok(n) => n,
            Err(_) => return,
        };

        match net.circadian_state {
            CircadianPhase::Awake => {
                // Slip into Drowsy on extended silence OR sustained low entropy.
                if idle_s >= IDLE_DROWSY_S || (idle_s > 60 && mean_h < LOW_ENTROPY_THRESHOLD) {
                    log::info!("Circadian: Awake → Drowsy");
                    net.circadian_state = CircadianPhase::Drowsy;
                }
            }

            CircadianPhase::Drowsy => {
                // Intrinsic plasticity nudges per-node frequencies toward
                // target firing rate, preventing chronic over/under-excitation.
                net.intrinsic_plasticity(TARGET_ACTIVATION);

                if idle_s >= IDLE_LIGHT_S {
                    log::info!("Circadian: Drowsy → LightSleep");
                    net.circadian_state = CircadianPhase::LightSleep;
                } else if telemetry.is_some() {
                    Self::snap_awake(&mut net);
                }
            }

            CircadianPhase::LightSleep => {
                if idle_s >= IDLE_DEEP_S {
                    log::info!("Circadian: LightSleep → DeepSleep");
                    net.circadian_state = CircadianPhase::DeepSleep;
                } else if telemetry.is_some() {
                    Self::snap_awake(&mut net);
                }
            }

            CircadianPhase::DeepSleep => {
                // Run consolidation exactly once per sleep episode.
                if !self.consolidation_done {
                    self.run_consolidation(&mut net);

                    // Commit the now-consolidated weights as the EWC anchor
                    // so future plasticity cannot catastrophically overwrite them.
                    if !self.anchor_committed {
                        net.commit_anchor();
                        self.anchor_committed = true;
                        log::info!("EWC: consolidated weights locked as semantic anchor");
                    }
                }
                if telemetry.is_some() {
                    Self::snap_awake(&mut net);
                }
            }

            CircadianPhase::Waking => {
                // Waking is transient — one tick to let the network "stretch",
                // then immediately advance to Awake.
                log::info!("Circadian: Waking → Awake");
                net.circadian_state = CircadianPhase::Awake;
            }
        }
    }

    /// Attempts to restore the last consolidated neural state from disk.
    ///
    /// Should be called once at startup before the physics loop begins.
    pub fn try_restore(&mut self, network: &Arc<Mutex<LiquidNetwork>>) {
        if !self.state_path.exists() {
            log::info!("iara.homeostasis: no prior snapshot found — starting fresh");
            return;
        }
        let mut net = match network.lock() {
            Ok(n) => n,
            Err(_) => return,
        };
        match net.load_state(&self.state_path) {
            Ok(_) => {
                net.commit_anchor();
                self.anchor_committed = true;
                log::info!("iara.homeostasis: neural state restored — EWC anchor set ({})", self.state_path.display());
            }
            Err(e) => log::warn!("iara.homeostasis: could not restore neural state: {}", e),
        }
    }

    // ── Observability ─────────────────────────────────────────────────────

    /// Mean entropy over the rolling window. Returns 0.5 when empty.
    pub fn mean_entropy(&self) -> f64 {
        if self.entropy_window.is_empty() {
            return 0.5;
        }
        self.entropy_window.iter().sum::<f64>() / self.entropy_window.len() as f64
    }

    /// Entropy variance — low variance + low mean = stable, sleepy state.
    pub fn entropy_variance(&self) -> f64 {
        let n = self.entropy_window.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean_entropy();
        self.entropy_window.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / n as f64
    }

    /// True when idle long enough to be in or approaching a sleep phase.
    pub fn is_sleeping(&self) -> bool {
        self.last_stimulus.elapsed().as_secs() >= IDLE_LIGHT_S
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Runs the full deep-sleep neural garbage collection routine.
    fn run_consolidation(&mut self, net: &mut LiquidNetwork) {
        log::info!("Deep sleep: neural GC starting");

        // 1 — Synaptic pruning: remove connections too weak to matter.
        let pruned = net.prune_weak_connections(PRUNE_THRESHOLD);
        log::info!("iara.consolidation: {} weak synapses pruned", pruned);

        // 2 — Re-normalise L1 norms after pruning may have shifted balance.
        net.homeostasis(1.0);

        // 3 — Persist the consolidated weight matrix (long-term memory).
        match net.save_state(&self.state_path) {
            Ok(_) => log::info!("iara.consolidation: snapshot saved to {}", self.state_path.display()),
            Err(e) => log::warn!("iara.consolidation: snapshot failed: {}", e),
        }

        // 4 — Clear residual activations: wipe transient working memory.
        //     The weight matrix (long-term memory) is deliberately preserved.
        net.reset_activations();

        self.consolidation_done = true;
        log::info!("Deep sleep consolidation complete — episodic memory committed to disk");
    }

    /// Immediate Waking transition on stimulus receipt.
    fn snap_awake(net: &mut LiquidNetwork) {
        log::info!("Stimulus received — Waking");
        net.circadian_state = CircadianPhase::Waking;
    }

    /// Pushes a new entropy sample into the capped rolling window.
    fn push_entropy(&mut self, entropy: f64) {
        if self.entropy_window.len() >= ENTROPY_WINDOW {
            self.entropy_window.pop_front();
        }
        self.entropy_window.push_back(entropy.clamp(0.0, 1.0));
    }
}

impl Default for HomeostaticRegulator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_network() -> Arc<Mutex<LiquidNetwork>> {
        Arc::new(Mutex::new(LiquidNetwork::new(8)))
    }

    fn dummy_tel(entropy: f64) -> LlmTelemetry {
        LlmTelemetry {
            token: "t".into(),
            entropy_chaos: entropy,
            context_energy: 0.8,
            hardware_stress: 0.1,
        }
    }

    #[test]
    fn mean_entropy_empty_returns_half() {
        let r = HomeostaticRegulator::new();
        assert!((r.mean_entropy() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn entropy_window_tracks_samples() {
        let mut r = HomeostaticRegulator::new();
        for _ in 0..10 {
            r.push_entropy(0.3);
        }
        assert!((r.mean_entropy() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn tick_with_stimulus_resets_idle() {
        let net = make_network();
        let mut r = HomeostaticRegulator::new();
        let tel = dummy_tel(0.4);
        // Tick with stimulus — should record timestamp.
        r.tick(&net, Some(&tel));
        assert!(r.last_stimulus.elapsed().as_millis() < 100);
    }

    #[cfg(feature = "circadian")]
    #[test]
    fn deep_sleep_triggers_consolidation() {
        let net = make_network();
        {
            let mut n = net.lock().unwrap();
            let tel = dummy_tel(0.2);
            n.inject_telemetry(&tel);
            for _ in 0..50 {
                n.step().unwrap();
            }
            // Manually drive to DeepSleep.
            n.circadian_state = CircadianPhase::DeepSleep;
        }
        let mut r = HomeostaticRegulator::new();
        // Override state path to a tmpdir.
        r.state_path = std::env::temp_dir().join("iara_test_consol.bin");
        r.tick(&net, None);
        assert!(r.consolidation_done, "consolidation should have run");
        // Cleanup.
        let _ = std::fs::remove_file(&r.state_path);
    }

    #[cfg(feature = "circadian")]
    #[test]
    fn stimulus_during_sleep_wakes_network() {
        let net = make_network();
        {
            let mut n = net.lock().unwrap();
            n.circadian_state = CircadianPhase::LightSleep;
        }
        let mut r = HomeostaticRegulator::new();
        r.state_path = std::env::temp_dir().join("iara_test_wake.bin");
        let tel = dummy_tel(0.5);
        r.tick(&net, Some(&tel));
        let phase = net.lock().unwrap().circadian_state;
        assert_eq!(
            phase,
            CircadianPhase::Waking,
            "stimulus should snap network to Waking"
        );
    }
}
