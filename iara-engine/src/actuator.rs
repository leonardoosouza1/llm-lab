//! Continuous Actuation — mapping liquid network resonance to logical actions.
//!
//! Instead of forcing IARA's wave-form internal state into discrete tokens,
//! this module reads the **topology of resonance** directly as control signals.
//!
//! # Functional Groups
//!
//! The 256 oscillators are partitioned into three **frequency bands** by natural
//! frequency rank (golden-angle seeding guarantees a deterministic partition):
//!
//! | Band | Nodes (approx.) | Role |
//! |------|-----------------|------|
//! | 0 — Focus | low-ω (~85) | Sustained attention, working memory |
//! | 1 — Novelty | mid-ω (~85) | Exploration, divergent association |
//! | 2 — Drive | high-ω (~86) | Urgency, action initiation |
//!
//! Each band exposes two scalars:
//! - **Energy** `E_b = mean(x_i²)` — intensity of the group.
//! - **Coherence** `r_b = |mean(e^{iφ_j})|` — unity / synchrony of the group.
//!
//! These 6 values (+ global Kuramoto `r`) form an [`ActuatorSignal`] that
//! any [`Actuator`] implementation consumes to decide whether to fire.
//!
//! # Example Flow
//!
//! ```text
//! LiquidNetwork (100 Hz physics)
//!     ↓  spectral_band_energy() + sub-group Kuramoto coherence
//! ActuatorBus::tick()  [1 Hz]
//!     ↓  ActuatorSignal { focus_*, novelty_*, drive_*, global_coherence }
//! ResearchActuator  → fires when novelty_energy > 0.3 ∧ novelty_coherence > 0.55
//! InhibitionActuator→ fires when drive_energy > 0.7 ∧ drive_coherence > 0.6
//! FocusActuator     → fires on entry/exit of deep focus state
//! ```

use std::sync::{Arc, Mutex};

use crate::liquid_network::LiquidNetwork;

// ─────────────────────────────────────────────────────────────────────────────
// § ActuatorSignal
// ─────────────────────────────────────────────────────────────────────────────

/// A 1-Hz snapshot of the liquid network's resonance topology.
///
/// All scalars are normalised to `[0, 1]` and safe to threshold.
#[derive(Debug, Clone)]
pub struct ActuatorSignal {
    /// Mean energy in the low-frequency (Focus) band — sustained reasoning.
    pub focus_energy: f64,
    /// Mean energy in the mid-frequency (Novelty) band — curiosity / exploration.
    pub novelty_energy: f64,
    /// Mean energy in the high-frequency (Drive) band — urgency / stress.
    pub drive_energy: f64,
    /// Kuramoto sub-coherence of the Focus group: unity of attention.
    pub focus_coherence: f64,
    /// Kuramoto sub-coherence of the Novelty group: unity of curiosity.
    pub novelty_coherence: f64,
    /// Kuramoto sub-coherence of the Drive group: unity of urgency.
    pub drive_coherence: f64,
    /// Global Kuramoto order parameter `r` across all nodes.
    pub global_coherence: f64,
}

impl ActuatorSignal {
    /// A fully silent signal with no activation in any group.
    pub fn zero() -> Self {
        Self {
            focus_energy: 0.0,
            novelty_energy: 0.0,
            drive_energy: 0.0,
            focus_coherence: 0.0,
            novelty_coherence: 0.0,
            drive_coherence: 0.0,
            global_coherence: 0.0,
        }
    }

    /// True when the novelty group is active and coherent — network wants to
    /// explore new information.
    pub fn wants_exploration(&self) -> bool {
        self.novelty_energy > 0.30 && self.novelty_coherence > 0.55
    }

    /// True when the focus band dominates both energy and coherence — deep
    /// single-topic reasoning.
    pub fn is_focused(&self) -> bool {
        self.focus_energy > self.novelty_energy
            && self.focus_energy > self.drive_energy
            && self.focus_coherence > 0.60
    }

    /// True when the drive band signals unsustainable urgency (stress overload).
    pub fn is_urgent(&self) -> bool {
        self.drive_energy > 0.70
    }

    /// Dominant group label for logging.
    pub fn dominant_group(&self) -> &'static str {
        let f = self.focus_energy;
        let n = self.novelty_energy;
        let d = self.drive_energy;
        if f >= n && f >= d {
            "Focus"
        } else if n >= f && n >= d {
            "Novelty"
        } else {
            "Drive"
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § Actuator Trait
// ─────────────────────────────────────────────────────────────────────────────

/// A component that reacts to the liquid network's continuous resonance signal.
///
/// Implement this trait to connect IARA's wave-form state to any downstream
/// system: research schedulers, speech synthesisers, agent routers, or hardware.
///
/// The bus calls `poll` at ~1 Hz. Return `Some(command)` to emit an action or
/// `None` to stay silent this tick.
pub trait Actuator: Send + Sync {
    /// Human-readable name used in structured log output.
    fn name(&self) -> &str;

    /// Inspects the current signal and optionally emits a command string.
    fn poll(&mut self, signal: &ActuatorSignal) -> Option<String>;
}

// ─────────────────────────────────────────────────────────────────────────────
// § ActuatorBus
// ─────────────────────────────────────────────────────────────────────────────

/// Reads the network's resonance state at 1 Hz and dispatches to all registered actuators.
///
/// # Example
///
/// ```rust,no_run
/// use jarvis::actuator::{ActuatorBus, ResearchActuator, InhibitionActuator};
/// let mut bus = ActuatorBus::new();
/// bus.register(ResearchActuator::new(120));
/// bus.register(InhibitionActuator::new());
/// ```
pub struct ActuatorBus {
    actuators: Vec<Box<dyn Actuator>>,
}

impl ActuatorBus {
    pub fn new() -> Self {
        Self { actuators: Vec::new() }
    }

    /// Adds an actuator to the bus.
    pub fn register(&mut self, actuator: impl Actuator + 'static) {
        self.actuators.push(Box::new(actuator));
    }

    /// Reads the current signal and polls all registered actuators.
    ///
    /// Returns `(actuator_name, command)` pairs for every action that fired.
    pub fn tick(&mut self, network: &Arc<Mutex<LiquidNetwork>>) -> Vec<(String, String)> {
        let signal = self.read_signal(network);
        let mut fired = Vec::new();

        for act in &mut self.actuators {
            if let Some(cmd) = act.poll(&signal) {
                log::info!(
                    "iara.actuator: {} fired | dominant={} global_r={:.3}",
                    act.name(), signal.dominant_group(), signal.global_coherence
                );
                fired.push((act.name().to_string(), cmd));
            }
        }

        fired
    }

    /// Computes an [`ActuatorSignal`] from the current network state.
    ///
    /// Uses [`LiquidNetwork::spectral_band_energy`] for the three energy bands
    /// and computes per-band Kuramoto sub-coherence from raw node phases.
    pub fn read_signal(&self, network: &Arc<Mutex<LiquidNetwork>>) -> ActuatorSignal {
        let net = match network.lock() {
            Ok(n) => n,
            Err(_) => return ActuatorSignal::zero(),
        };

        let bands = net.spectral_band_energy();
        let (global_r, _) = net.phase_coherence();
        let (focus_r, novelty_r, drive_r) = sub_group_coherence(&net);

        // Soft-saturation normalisation: tanh(4·E) maps [0, ∞) → [0, 1).
        let norm = |e: f64| (e * 4.0).tanh();

        ActuatorSignal {
            focus_energy: norm(bands[0]),
            novelty_energy: norm(bands[1]),
            drive_energy: norm(bands[2]),
            focus_coherence: focus_r,
            novelty_coherence: novelty_r,
            drive_coherence: drive_r,
            global_coherence: global_r,
        }
    }
}

impl Default for ActuatorBus {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § Sub-group Kuramoto coherence (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Computes per-band Kuramoto order parameters by partitioning nodes into
/// three equal frequency bands (low / mid / high) based on natural-frequency rank.
///
/// Returns `(focus_r, novelty_r, drive_r)` each in `[0, 1]`.
fn sub_group_coherence(net: &LiquidNetwork) -> (f64, f64, f64) {
    let n = net.nodes.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let third = n / 3;

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| {
        net.nodes[a]
            .frequency
            .partial_cmp(&net.nodes[b].frequency)
            .unwrap()
    });

    let mut re = [0.0_f64; 3];
    let mut im = [0.0_f64; 3];
    let mut cnt = [0usize; 3];

    for (rank, &idx) in sorted.iter().enumerate() {
        let band = if rank < third { 0 } else if rank < 2 * third { 1 } else { 2 };
        let phi = net.nodes[idx].phase;
        re[band] += phi.cos();
        im[band] += phi.sin();
        cnt[band] += 1;
    }

    let r = |b: usize| -> f64 {
        let c = cnt[b] as f64;
        if c == 0.0 {
            return 0.0;
        }
        let rx = re[b] / c;
        let ix = im[b] / c;
        (rx * rx + ix * ix).sqrt()
    };

    (r(0), r(1), r(2))
}

// ─────────────────────────────────────────────────────────────────────────────
// § Built-in Actuators
// ─────────────────────────────────────────────────────────────────────────────

/// Triggers autonomous research when the novelty group reaches resonance.
///
/// The intuition: when the mid-frequency "curiosity" band lights up coherently,
/// IARA is in an exploratory state and should seek new information seeds.
pub struct ResearchActuator {
    cooldown: u32,
    ticks_since_last: u32,
    novelty_threshold: f64,
}

impl ResearchActuator {
    /// Creates a research actuator with the given cooldown period (seconds at 1 Hz).
    pub fn new(cooldown_secs: u32) -> Self {
        Self {
            cooldown: cooldown_secs,
            ticks_since_last: cooldown_secs, // start in ready state
            novelty_threshold: 0.55,
        }
    }
}

impl Actuator for ResearchActuator {
    fn name(&self) -> &str {
        "ResearchActuator"
    }

    fn poll(&mut self, signal: &ActuatorSignal) -> Option<String> {
        self.ticks_since_last += 1;
        if self.ticks_since_last >= self.cooldown
            && signal.novelty_energy > 0.30
            && signal.novelty_coherence > self.novelty_threshold
        {
            self.ticks_since_last = 0;
            Some(format!(
                "research:trigger | n_energy={:.3} n_coherence={:.3}",
                signal.novelty_energy, signal.novelty_coherence
            ))
        } else {
            None
        }
    }
}

/// Engages cognitive suppression when the drive band saturates.
///
/// High drive energy + coherence = cognitive overload → inhibit new inference
/// tasks until the stress dissipates.
pub struct InhibitionActuator {
    was_suppressed: bool,
}

impl InhibitionActuator {
    pub fn new() -> Self {
        Self { was_suppressed: false }
    }
}

impl Default for InhibitionActuator {
    fn default() -> Self {
        Self::new()
    }
}

impl Actuator for InhibitionActuator {
    fn name(&self) -> &str {
        "InhibitionActuator"
    }

    fn poll(&mut self, signal: &ActuatorSignal) -> Option<String> {
        let overloaded = signal.is_urgent() && signal.drive_coherence > 0.60;
        if overloaded && !self.was_suppressed {
            self.was_suppressed = true;
            Some(format!(
                "inhibit:engage | drive_e={:.3} drive_r={:.3}",
                signal.drive_energy, signal.drive_coherence
            ))
        } else if !overloaded && self.was_suppressed {
            self.was_suppressed = false;
            Some("inhibit:disengage | stress cleared — resuming full inference".to_string())
        } else {
            None
        }
    }
}

/// Emits state-change notifications when the network enters or exits deep focus.
pub struct FocusActuator {
    was_focused: bool,
}

impl FocusActuator {
    pub fn new() -> Self {
        Self { was_focused: false }
    }
}

impl Default for FocusActuator {
    fn default() -> Self {
        Self::new()
    }
}

impl Actuator for FocusActuator {
    fn name(&self) -> &str {
        "FocusActuator"
    }

    fn poll(&mut self, signal: &ActuatorSignal) -> Option<String> {
        if signal.is_focused() && !self.was_focused {
            self.was_focused = true;
            Some(format!(
                "focus:enter | focus_e={:.3} focus_r={:.3} global_r={:.3}",
                signal.focus_energy, signal.focus_coherence, signal.global_coherence
            ))
        } else if !signal.is_focused() && self.was_focused {
            self.was_focused = false;
            Some("focus:exit | returning to exploratory mode".to_string())
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liquid_network::{LiquidNetwork, LlmTelemetry};
    use std::sync::{Arc, Mutex};

    fn make_net() -> Arc<Mutex<LiquidNetwork>> {
        Arc::new(Mutex::new(LiquidNetwork::new(12, 2, 0)))
    }

    #[test]
    fn zero_network_no_inhibition() {
        let mut act = InhibitionActuator::new();
        let result = act.poll(&ActuatorSignal::zero());
        assert!(result.is_none());
    }

    #[test]
    fn inhibition_engages_on_high_drive() {
        let mut act = InhibitionActuator::new();
        let mut sig = ActuatorSignal::zero();
        sig.drive_energy = 0.9;
        sig.drive_coherence = 0.8;
        let r = act.poll(&sig);
        assert!(r.is_some());
        assert!(r.unwrap().contains("inhibit:engage"));
    }

    #[test]
    fn inhibition_disengages_when_drive_drops() {
        let mut act = InhibitionActuator::new();
        let mut sig = ActuatorSignal::zero();
        sig.drive_energy = 0.9;
        sig.drive_coherence = 0.8;
        act.poll(&sig); // engage

        sig.drive_energy = 0.1;
        sig.drive_coherence = 0.1;
        let r = act.poll(&sig);
        assert!(r.is_some());
        assert!(r.unwrap().contains("inhibit:disengage"));
    }

    #[test]
    fn research_actuator_respects_cooldown() {
        let mut act = ResearchActuator::new(10); // 10 tick cooldown
        act.ticks_since_last = 0; // force not ready
        let mut sig = ActuatorSignal::zero();
        sig.novelty_energy = 0.5;
        sig.novelty_coherence = 0.8;
        // Should not fire — on cooldown.
        let r = act.poll(&sig);
        assert!(r.is_none());
    }

    #[test]
    fn bus_reads_non_zero_signal_after_stimulation() {
        let bus = ActuatorBus::new();
        let net = make_net();
        {
            let mut n = net.lock().unwrap();
            n.inject_telemetry(&LlmTelemetry {
                token: "x".into(),
                entropy_chaos: 0.2,
                context_energy: 0.9,
                hardware_stress: 0.05,
            });
            for _ in 0..200 {
                n.step().unwrap();
            }
        }
        let sig = bus.read_signal(&net);
        let total = sig.focus_energy + sig.novelty_energy + sig.drive_energy;
        assert!(total > 0.0, "stimulated network should have non-zero energy");
    }

    #[test]
    fn dominant_group_returns_correct_label() {
        let mut sig = ActuatorSignal::zero();
        sig.focus_energy = 0.8;
        assert_eq!(sig.dominant_group(), "Focus");
        sig.novelty_energy = 0.9;
        assert_eq!(sig.dominant_group(), "Novelty");
        sig.drive_energy = 1.0;
        assert_eq!(sig.dominant_group(), "Drive");
    }
}
