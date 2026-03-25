//! Liquid Neural Network — the dynamic mesh of coupled oscillators at the core of IARA.
//!
//! This module implements:
//! - The global data contracts (`LiquidNode`, `LlmTelemetry`, `KnowledgePill`)
//! - The `LiquidNetwork` — a field of N coupled nonlinear oscillators
//! - An RK4 solver for fast, stable numerical integration at 100 Hz (C1 + C2)
//! - The slow Oja synaptic learning rule + Synaptic Homeostasis (C7)
//!
//! # Physics Overview
//!
//! Each `LiquidNode` evolves according to:
//! ```text
//! f(x, t) = -γ·x + tanh(Σ_j W[i][j]·x_j + stimulus)
//! ```
//! integrated with the 4th-order Runge–Kutta method (step `h = 0.01 s`).
//!
//! The effective step size is divided by `viscosity` (μ), modelling the semantic
//! "thickness" introduced by stress — high μ means slow, rigid propagation.

#![allow(dead_code)]
#![allow(unused_variables)]
use std::f64::consts::PI;
#[allow(unused_imports)]
use log::{info, warn};
use serde::{Serialize, Deserialize};
#[allow(unused_imports)]
use rocm_rs::prelude::*;

#[cfg(feature = "circadian")]
mod circadian_phase {
    use serde::{Serialize, Deserialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum CircadianPhase {
        Awake,
        Drowsy,
        LightSleep,
        DeepSleep,
        Waking,
    }

    impl std::fmt::Display for CircadianPhase {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Awake => write!(f, "Awake"),
                Self::Drowsy => write!(f, "Drowsy"),
                Self::LightSleep => write!(f, "LightSleep"),
                Self::DeepSleep => write!(f, "DeepSleep"),
                Self::Waking => write!(f, "Waking"),
            }
        }
    }
}

#[cfg(feature = "circadian")]
pub use circadian_phase::CircadianPhase;

// ─────────────────────────────────────────────────────────────────────────────
// § Type Aliases (CONTRACTS.md)
// ─────────────────────────────────────────────────────────────────────────────

/// Index of a node in the `LiquidNetwork`.
pub type NodeId = usize;

/// A time-domain snapshot of a node's state values.
pub type Waveform = Vec<f64>;

// ─────────────────────────────────────────────────────────────────────────────
// § Core Structs (CONTRACTS.md)
// ─────────────────────────────────────────────────────────────────────────────

/// Real-time telemetry dissected from the LLM token stream.
///
/// Sent at up to 100 Hz over the broadcast channel to the `LiquidNetwork`.
#[derive(Debug, Clone)]
pub struct LlmTelemetry {
    /// The raw token text emitted by the LLM.
    pub token: String,
    /// Shannon entropy H over the top-5 logprob candidates — measures doubt.
    pub entropy_chaos: f64,
    /// Grounding energy from NotebookLM context — certainty pull toward truth.
    pub context_energy: f64,
    /// Hardware stress proxy (GPU temp / 80) — raises viscosity μ.
    pub hardware_stress: f64,
}

/// The minimum unit of memory in the liquid mesh.
///
/// Each node is a nonlinear oscillator with its own frequency, phase, and
/// amplitude. Its `state` evolves under RK4 dynamics driven by the synaptic
/// weight matrix and the current LLM stimulus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidNode {
    /// Unique node index.
    pub id: NodeId,
    /// Current activation x(t).
    pub state: f64,
    /// Phase φ in radians `[0, 2π)`.
    pub phase: f64,
    /// Natural oscillation frequency ω (rad/s).
    pub frequency: f64,
    /// Envelope amplitude A — modulated by entropy to spread uncertainty.
    pub amplitude: f64,
    /// Instantaneous velocity dx/dt.
    pub velocity: f64,

    // ── CfC Parameters (Internal gates) ───────────────────────────────────
    pub w_liq: f64,
    pub w_tau: f64,
    pub b_liq: f64,

    // ── STDP / Spike Timing (Epigenetic Layer) ────────────────────────────
    /// Threshold to generate a discrete spike pulse.
    pub threshold: f64,
    /// Timestamp of the last generated spike (milliseconds).
    pub last_spike_ms: f64,
}

/// A condensed knowledge unit harvested from a mature branch of the KAN-Arbor.
///
/// Acts as a compressed context prefix for the next Ollama inference pass.
#[derive(Debug, Clone)]
pub struct KnowledgePill {
    /// The concept label.
    pub concept: String,
    /// Semantic compression density — 0.0 (raw) to 1.0 (fully crystallised).
    pub density: f64,
    /// Epistemic polarity: -1.0 = error / contradiction, +1.0 = verified truth.
    pub polarity: f64,
    /// Low-dimensional embedding vector (post Johnson–Lindenstrauss projection).
    pub vector: Vec<f32>,
}

impl LiquidNode {
    /// Creates a new CfC node at rest with the given `frequency`.
    pub fn new(id: NodeId, frequency: f64) -> Self {
        let h = (id as f64 * 1.618).fract();
        Self {
            id,
            state: 0.0,
            phase: 0.0,
            frequency,
            amplitude: 1.0,
            velocity: 0.0,
            w_liq: 0.5 + h * 0.5,
            w_tau: 0.1 + (1.0 - h) * 0.4,
            b_liq: (h - 0.5) * 0.2,
            threshold: 0.75, // Default spike threshold
            last_spike_ms: -1000.0, // Remote past
        }
    }

    /// Checks if the activation wave crossed the threshold and emits a spike pulse.
    pub fn check_spike(&mut self, current_time_ms: f64) -> bool {
        if self.state >= self.threshold {
            // 5ms refractory period to prevent burst firing
            if current_time_ms - self.last_spike_ms > 5.0 {
                self.last_spike_ms = current_time_ms;
                // Refractory reset (biologically accurate potential drop)
                self.state *= 0.1;
                return true;
            }
        }
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § Physics Constants
// ─────────────────────────────────────────────────────────────────────────────

/// RK4 nominal time step in seconds (100 Hz → 10 ms).
const RK4_H: f64 = 0.01;

/// Damping coefficient γ — controls how fast free oscillations decay.
const DAMPING: f64 = 0.15;

/// Golden angle in radians, used to seed node frequencies deterministically.
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653; // 137.5077...°

/// Stress threshold above which IARA enters cognitive inhibition.
pub const STRESS_INHIBITION_THRESHOLD: f64 = 0.9;

// ─────────────────────────────────────────────────────────────────────────────
// § LiquidNetwork
// ─────────────────────────────────────────────────────────────────────────────

pub struct LiquidNetwork {
    /// All oscillator nodes in the mesh.
    pub nodes: Vec<LiquidNode>,
    /// Layer index for each node.
    pub node_layers: Vec<usize>,
    /// Number of total layers in the hierarchy.
    pub num_layers: usize,
    /// Dense synaptic weight matrix W[i][j] (flat vector for cache efficiency).
    weights: Vec<f64>,
    /// Epigenetic plasticity matrix P[i][j] (flat).
    weights_plastic: Vec<f64>,
    /// Active stimuli injected per layer.
    layer_stimuli: Vec<f64>,
    /// Current simulation clock (seconds).
    pub time: f64,
    /// Viscosity coefficient μ (≥ 1.0).
    pub viscosity: f64,
    /// Stress scalar S ∈ [0, 1].
    pub stress: f64,
    /// Oja learning rate η.
    oja_lr: f64,

    // ── Elastic Weight Consolidation (EWC) ────────────────────────────────
    anchor_weights: Option<Vec<f64>>,
    fisher_importance: Vec<f64>,
    ewc_lambda: f64,

    // ── STDP Plasticity (Phase 1) ──────────────────────────────────────────
    #[cfg(feature = "stdp")]
    eligibility: Vec<f64>,
    #[cfg(feature = "stdp")]
    bcm_threshold: Vec<f64>,
    #[cfg(feature = "stdp")]
    stdp_lr: f64,

    // ── Circadian State (Phase 2) ──────────────────────────────────────────
    #[cfg(feature = "circadian")]
    pub idle_ticks: u64,
    /// Hardware device for tensor/kernel operations.
    pub device: candle_core::Device,

    // ── GPU Physics State ─────────────────────────────────────────────────
    /// Compiled ROCm module for native kernels.
    #[serde(skip)]
    pub gpu_module: Option<rocm_rs::prelude::Module>,
    /// GPU buffer for oscillator states (f32).
    #[serde(skip)]
    pub gpu_states: Option<rocm_rs::prelude::DeviceBuffer<f32>>,
    /// GPU buffer for DNA synaptic weights (f32).
    #[serde(skip)]
    pub gpu_weights: Option<rocm_rs::prelude::DeviceBuffer<f32>>,
    /// GPU buffer for epigenetic plastic weights (f32).
    #[serde(skip)]
    pub gpu_weights_plastic: Option<rocm_rs::prelude::DeviceBuffer<f32>>,
    /// GPU buffer for kernel outputs (forces/couplings).
    #[serde(skip)]
    pub gpu_output: Option<rocm_rs::prelude::DeviceBuffer<f32>>,
}

impl LiquidNetwork {
    pub fn new(n: usize, num_layers: usize, device: candle_core::Device) -> Self {
        assert!(n > 0, "LiquidNetwork must have at least one node");
        let nodes_per_layer = n / num_layers;
        let mut nodes = Vec::with_capacity(n);
        let mut node_layers = Vec::with_capacity(n);

        for l in 0..num_layers {
            for i in 0..nodes_per_layer {
                let global_id = l * nodes_per_layer + i;
                let rank = i as f32 / nodes_per_layer as f32;
                let freq = if rank < 0.33 { 0.1 + (rank * 3.0) * 0.7 } 
                           else if rank < 0.66 { 0.8 + ((rank - 0.33) * 3.0) * 1.7 } 
                           else { 2.5 + ((rank - 0.66) * 3.0) * 3.5 };
                let jitter = ((global_id as f64) * GOLDEN_ANGLE) % 0.1;
                nodes.push(LiquidNode::new(global_id, freq as f64 + jitter));
                node_layers.push(l);
            }
        }

        let n_actual = nodes.len();
        let mut weights = vec![0.0; n_actual * n_actual];
        for i in 0..n_actual {
            let layer_i = node_layers[i];
            for j in 0..n_actual {
                if i == j { continue; }
                let layer_j = node_layers[j];
                let layer_dist = (layer_i as i32 - layer_j as i32).abs();
                let base_w = ((i + j) as f64 * 0.1).sin() * 0.2;
                let layer_factor = (-(layer_dist as f64) / 2.0).exp();
                weights[i * n_actual + j] = base_w * layer_factor;
            }
        }

        Self {
            nodes,
            node_layers,
            num_layers,
            weights,
            weights_plastic: vec![0.0; n_actual * n_actual],
            layer_stimuli: vec![0.0; num_layers],
            time: 0.0,
            viscosity: 1.0,
            stress: 0.0,
            oja_lr: 0.01,
            anchor_weights: None,
            fisher_importance: vec![0.0; n_actual * n_actual],
            ewc_lambda: 100.0,
            #[cfg(feature = "stdp")]
            eligibility: vec![0.0; n_actual * n_actual],
            #[cfg(feature = "stdp")]
            bcm_threshold: vec![1.0; n_actual],
            #[cfg(feature = "stdp")]
            stdp_lr: 0.05,
            #[cfg(feature = "circadian")]
            circadian_state: CircadianPhase::Awake,
            #[cfg(feature = "circadian")]
            idle_ticks: 0,
            device,
            gpu_module: None,
            gpu_states: None,
            gpu_weights: None,
            gpu_weights_plastic: None,
            gpu_output: None,
        }
    }

    /// Initializes GPU buffers and loads the kernel module.
    pub fn init_gpu(&mut self) -> anyhow::Result<()> {
        let n_actual = self.nodes.len();
        
        // ── 1. Carregar binário do Kernel (Navi 22 Bitcode) ──
        let binary = crate::physics_gpu::get_kernel_binary();
        let module = rocm_rs::prelude::Module::load_from_memory(binary)
            .map_err(|e| anyhow::anyhow!("Falha ao carregar módulo GPU: {:?}", e))?;
        self.gpu_module = Some(module);

        // ── 2. Preparar Buffers Iniciais (f32 para Performance) ──
        let states_f32: Vec<f32> = self.nodes.iter().map(|n| n.state as f32).collect();
        let weights_f32: Vec<f32> = self.weights.iter().map(|&w| w as f32).collect();
        let weights_plastic_f32: Vec<f32> = self.weights_plastic.iter().map(|&w| w as f32).collect();

        // ── 3. Alocar na VRAM ──
        self.gpu_states = Some(rocm_rs::prelude::DeviceBuffer::from_slice(&states_f32)
            .map_err(|e| anyhow::anyhow!("Erro VRAM states: {:?}", e))?);
        
        self.gpu_weights = Some(rocm_rs::prelude::DeviceBuffer::from_slice(&weights_f32)
            .map_err(|e| anyhow::anyhow!("Erro VRAM weights: {:?}", e))?);
        
        self.gpu_weights_plastic = Some(rocm_rs::prelude::DeviceBuffer::from_slice(&weights_plastic_f32)
            .map_err(|e| anyhow::anyhow!("Erro VRAM plastic: {:?}", e))?);

        self.gpu_output = Some(rocm_rs::prelude::DeviceBuffer::new(n_actual)
            .map_err(|e| anyhow::anyhow!("Erro VRAM output: {:?}", e))?);

        log::info!("🚀 GPU Resonance Engine: Buffers CfC (n={}) alocados na VRAM", n_actual);
        Ok(())
    }

    /// Execução de Física Nativa em GPU (ROCm/HIP)
    pub fn step_gpu(&mut self, dt: f64) -> anyhow::Result<()> {
        let n = self.nodes.len();
        let module = self.gpu_module.as_ref().ok_or_else(|| anyhow::anyhow!("GPU não inicializada"))?;
        
        // ── 1. Sincronizar estados CPU -> GPU ──
        let states_f32: Vec<f32> = self.nodes.iter().map(|n| n.state as f32).collect();
        if let Some(buf) = &mut self.gpu_states {
            buf.copy_from(&states_f32)?;
        }

        // ── 2. Lançar Kernels de Física ──
        let grid = (n as u32 + 255) / 256;
        let block = 256;

        // Kernel A: Pauli Repulsion (Pressão de Degenerescência)
        if let Ok(kernel) = module.get_function("pauli_repulsion_kernel") {
            let args = rocm_rs::prelude::LaunchArgs::new()
                .arg(self.gpu_states.as_ref().unwrap())
                .arg(self.gpu_output.as_mut().unwrap())
                .arg(n)
                .arg(0.5f32); // k_pauli
            
            kernel.launch(grid, 1, 1, block, 1, 1, 0, None, args)?;
        }

        // Kernel B: KAN Spline (Ressonância Sináptica)
        if let Ok(kernel) = module.get_function("kan_spline_kernel") {
            let args = rocm_rs::prelude::LaunchArgs::new()
                .arg(self.gpu_states.as_ref().unwrap())
                .arg(self.gpu_weights.as_ref().unwrap())
                .arg(self.gpu_weights_plastic.as_ref().unwrap())
                .arg(self.gpu_output.as_mut().unwrap())
                .arg(n);
            
            kernel.launch(grid, 1, 1, block, 1, 1, 0, None, args)?;
        }

        // ── 3. Sincronizar GPU -> CPU ──
        let mut output_f32 = vec![0.0f32; n];
        if let Some(buf) = &self.gpu_output {
            buf.copy_to(&mut output_f32)?;
        }

        // ── 4. Atualizar dinâmica CfC com resultados da GPU ──
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let coupling = output_f32[i] as f64;
            // Dinâmica CfC simplificada para o loop GPU
            node.velocity += (coupling - DAMPING * node.velocity) * dt;
            node.state += node.velocity * dt;
        }

        self.time += dt;
        Ok(())
    }

    /// Injects `LlmTelemetry` into the mesh, updating stress and viscosity.
    pub fn inject_telemetry(&mut self, telemetry: &LlmTelemetry) {
        // ── PSIQUE: Cálculo de Estresse Dinâmico ──
        // O estresse é a soma da Entropia (dúvida), falta de Contexto (incerteza)
        // e o Hardware Stress (temperatura/memória real lida via propriocepção).
        let hw_stress = telemetry.hardware_stress;
        
        self.stress = (hw_stress + telemetry.entropy_chaos
            + (1.0 - telemetry.context_energy.clamp(0.0, 1.0)))
            / 3.0;
        self.stress = self.stress.clamp(0.0, 1.0);

        // Bloqueio de Segurança: Inibição Cognitiva (S > 0.9)
        if self.stress > STRESS_INHIBITION_THRESHOLD {
            log::warn!(
                "⚠️ IARA: Inibição Cognitiva Ativada — estresse crítico ({:.3}). Pensamento viscoso.",
                self.stress
            );
            // Reduzir drasticamente estímulos em todas as camadas para proteção
            for s in &mut self.layer_stimuli { *s *= 0.05; }
            for node in &mut self.nodes {
                node.amplitude = node.amplitude.min(0.2);
            }
        }

        // Viscosidade Semântica (μ): O cortisol digital
        // Quanto mais estresse, mais rígida e lenta a malha se torna.
        self.viscosity = 1.0 + self.stress * 6.0;

        #[cfg(feature = "circadian")]
        {
            self.idle_ticks = 0;
        }
    }

    /// Injeta a ativação de uma camada específica do LLM na camada correspondente do cérebro líquido.
    /// Distribui o tensor de alta dimensão (ex: 4096) entre os nós da camada
    /// para que cada nó receba uma "característica semântica" diferente.
    pub fn inject_layer_stimulus(&mut self, layer_idx: usize, activation: &[f32]) {
        if layer_idx >= self.num_layers || activation.is_empty() { return; }
        
        let n = self.nodes.len();
        let nodes_per_layer = n / self.num_layers;
        let start_node = layer_idx * nodes_per_layer;
        
        // Mapear o tensor (ex: 4096) para os nós da camada (ex: 8 nós)
        // Cada nó recebe a média de energia de um bloco (chunk) do tensor original.
        let chunk_size = activation.len() / nodes_per_layer.max(1);
        
        for i in 0..nodes_per_layer {
            let global_id = start_node + i;
            if global_id >= n { break; }
            
            let chunk_start = i * chunk_size;
            let chunk_end = ((i + 1) * chunk_size).min(activation.len());
            
            if chunk_end > chunk_start {
                let chunk = &activation[chunk_start..chunk_end];
                // RMS de energia do bloco
                let energy: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt() / chunk.len() as f32;
                
                // Injetar diretamente no estado do nó para criar diferenciação real
                // Multiplicamos por um fator de ganho para o efeito ser visível
                self.nodes[global_id].state += (energy * 5.0) as f64;
            }
        }
        
        // Mantemos o registro da energia média da camada para telemetria global
        let layer_energy: f32 = activation.iter().map(|v| v * v).sum::<f32>().sqrt() / activation.len() as f32;
        self.layer_stimuli[layer_idx] = layer_energy as f64;
    }

    pub fn step(&mut self) -> anyhow::Result<()> {
        if self.nodes.is_empty() { anyhow::bail!("LiquidNetwork empty"); }
        let h = RK4_H / self.viscosity;
        let n = self.nodes.len();
        let current_time_ms = self.time * 1000.0;

        let mut spikes = Vec::new();
        for node in &mut self.nodes { if node.check_spike(current_time_ms) { spikes.push(node.id); } }

        for &id_pos in &spikes {
            let t_pos = current_time_ms;
            for i in 0..n {
                if i == id_pos { continue; }
                let t_pre = self.nodes[i].last_spike_ms;
                if (t_pos - t_pre).abs() < 100.0 {
                    let delta = calculate_stdp(t_pre, t_pos);
                    if delta != 0.0 { 
                        let idx = id_pos * n + i;
                        self.weights_plastic[idx] = (self.weights_plastic[idx] + delta).clamp(-1.0, 1.0); 
                    }
                }
            }
        }

        let states: Vec<f64> = self.nodes.iter().map(|nd| nd.state).collect();
        let phases: Vec<f64> = self.nodes.iter().map(|nd| nd.phase).collect();
        let mut new_states = vec![0.0_f64; n];

        for i in 0..n {
            let node = &self.nodes[i];
            let layer_idx = self.node_layers[i];
            
            // ── KAN Mesh com Pesos Ternários (Processo CPU Ternário) ──
            // Em vez de multiplicações de float, usamos a lógica 1.58b:
            // O sinal da plasticidade é discretizado para economizar CPU.
            let coupling: f64 = (0..n).filter(|&j| j != i).map(|j| {
                let idx = i * n + j;
                let resonance = (phases[i] - phases[j]).cos();
                let w_dna = self.weights[idx];
                
                // ── Quantização Ternária em Tempo Real ──
                // W_learned ∈ {-1, 0, 1}
                let w_plastic_raw = self.weights_plastic[idx];
                let w_ternary = if w_plastic_raw > 0.2 { 1.0 } 
                               else if w_plastic_raw < -0.2 { -1.0 } 
                               else { 0.0 };
                
                let xj = states[j];
                // Aresta KAN: DNA (Sino) + Lógica Ternária Aprendida
                let kan_edge_val = w_dna * xj * (-xj.abs()).exp() + w_ternary * xj;
                
                kan_edge_val * (0.5 + 0.5 * resonance)
            }).sum();

            // ── Pressão de Degenerescência de Pauli ──
            // Impede o colapso semântico. Se dois nós (mesmo de camadas iguais) tiverem ativações quase idênticas,
            // eles se repelem com F = k / r^2.
            let mut pauli_repulsion = 0.0;
            let k_pauli = 0.005; // Constante de repulsão
            for j in 0..n {
                if i != j {
                    let dist = (states[i] - states[j]).abs();
                    if dist > 0.0 && dist < 0.1 { // Apenas age em curtas distâncias
                        // O nó de índice menor empurra para cima, o maior para baixo (quebra de simetria)
                        let direction = if states[i] > states[j] { 1.0 } else { -1.0 };
                        pauli_repulsion += direction * (k_pauli / (dist * dist + 1e-4));
                    }
                }
            }

            let input = coupling + self.layer_stimuli[layer_idx] + pauli_repulsion;
            let target = (node.w_liq * input + node.b_liq).tanh();
            let tau_inv = node.w_tau * input.abs() + DAMPING;
            let decay = (-tau_inv * h).exp();
            new_states[i] = target * (1.0 - decay) * node.amplitude + states[i] * decay;
        }

        for (i, node) in self.nodes.iter_mut().enumerate() {
            let prev = node.state;
            node.state = new_states[i];
            node.velocity = (new_states[i] - prev) / h;
            node.phase = (node.phase + node.frequency * h).rem_euclid(2.0 * PI);
        }
        self.time += h;
        Ok(())
    }

    pub fn oja_update(&mut self, context_energy: f64) {
        let n = self.nodes.len();
        let states: Vec<f64> = self.nodes.iter().map(|nd| nd.state).collect();
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let idx = i * n + j;
                let delta = self.oja_lr * (context_energy * states[i] * states[j] - states[j] * states[j] * self.weights[idx]);
                let ewc_correction = if let Some(ref anchor) = self.anchor_weights {
                    self.ewc_lambda * self.fisher_importance[idx] * (self.weights[idx] + delta - anchor[idx])
                } else { 0.0 };
                self.weights[idx] = (self.weights[idx] + delta - ewc_correction).clamp(-1.0, 1.0);
            }
        }
    }

    pub fn homeostasis(&mut self, target_norm: f64) {
        let n = self.nodes.len();
        for i in 0..n {
            let mut l1 = 0.0;
            for j in 0..n {
                if i == j { continue; }
                l1 += self.weights[i * n + j].abs();
            }
            if l1 > f64::EPSILON {
                let scale = target_norm / l1;
                for j in 0..n { 
                    if i != j { self.weights[i * n + j] *= scale; } 
                }
            }
        }
    }

    pub fn slow_update(&mut self, context_energy: f64) {
        #[cfg(feature = "circadian")] {
            self.idle_ticks += 1;
            if self.circadian_state != CircadianPhase::Awake { return; }
        }
        #[cfg(feature = "stdp")] {
            self.stdp_update(context_energy);
            self.update_bcm_thresholds();
            self.decay_eligibility(0.95);
        }
        self.update_fisher();
        self.oja_update(context_energy);
        self.homeostasis(1.0);
    }

    #[cfg(feature = "stdp")]
    pub fn stdp_update(&mut self, context_energy: f64) {
        let n = self.nodes.len();
        let states: Vec<f64> = self.nodes.iter().map(|nd| nd.state).collect();
        let phases: Vec<f64> = self.nodes.iter().map(|nd| nd.phase).collect();
        for i in 0..n {
            let bcm_gate = states[i] * (states[i] - self.bcm_threshold[i]);
            for j in 0..n {
                if i == j { continue; }
                let idx = i * n + j;
                let stdp_term = (phases[i] - phases[j]).cos() * states[i].abs() * states[j].abs();
                self.eligibility[idx] += stdp_term;
                let delta = self.stdp_lr * context_energy * bcm_gate * self.eligibility[idx];
                self.weights[idx] = (self.weights[idx] + delta).clamp(-1.0, 1.0);
            }
        }
    }

    #[cfg(feature = "stdp")]
    fn update_bcm_thresholds(&mut self) {
        for (i, node) in self.nodes.iter().enumerate() {
            self.bcm_threshold[i] = (1.0 - 0.01) * self.bcm_threshold[i] + 0.01 * node.state * node.state;
        }
    }

    #[cfg(feature = "stdp")]
    fn decay_eligibility(&mut self, lambda: f64) {
        for e in self.eligibility.iter_mut() { *e *= lambda; }
    }

    pub fn weights_combined(&self) -> Vec<f64> {
        let n = self.nodes.len();
        let mut combined = vec![0.0; n * n];
        for i in 0..(n * n) { 
            combined[i] = (self.weights[i] + self.weights_plastic[i]).clamp(-1.0, 1.0); 
        }
        combined
    }

    pub fn weights_ref(&self) -> &Vec<f64> { &self.weights }

    #[cfg(feature = "circadian")]
    pub fn reset_activations(&mut self) { for node in &mut self.nodes { node.state = 0.0; node.velocity = 0.0; node.amplitude = 1.0; } }

    #[cfg(feature = "circadian")]
    pub fn prune_weak_connections(&mut self, threshold: f64) -> usize {
        let mut pruned = 0;
        for w in self.weights.iter_mut() {
            if w.abs() < threshold { *w = 0.0; pruned += 1; }
        }
        pruned
    }

    #[cfg(feature = "circadian")]
    pub fn intrinsic_plasticity(&mut self, target: f64) {
        for node in &mut self.nodes {
            if node.state.abs() > target { node.frequency *= 1.001; } else { node.frequency *= 0.999; }
            node.frequency = node.frequency.clamp(0.05, 2.0 * PI + 0.5);
        }
    }

    pub fn commit_anchor(&mut self) { self.anchor_weights = Some(self.weights.clone()); }

    pub fn update_fisher(&mut self) {
        let n = self.nodes.len();
        let states: Vec<f64> = self.nodes.iter().map(|nd| nd.state).collect();
        for i in 0..n { 
            for j in 0..n { 
                if i != j {
                    let idx = i * n + j;
                    self.fisher_importance[idx] = (1.0 - 0.001) * self.fisher_importance[idx] + 0.001 * (states[i] * states[j]).abs();
                } 
            } 
        }
    }

    pub fn anchor_weights(&self) -> Option<&Vec<f64>> { self.anchor_weights.as_ref() }

    pub fn mean_energy(&self) -> f64 {
        if self.nodes.is_empty() { return 0.0; }
        self.nodes.iter().map(|nd| nd.state * nd.state).sum::<f64>() / self.nodes.len() as f64
    }

    pub fn phase_coherence(&self) -> (f64, f64) {
        if self.nodes.is_empty() { return (0.0, 0.0); }
        let n = self.nodes.len() as f64;
        let sum_cos: f64 = self.nodes.iter().map(|nd| nd.phase.cos()).sum();
        let sum_sin: f64 = self.nodes.iter().map(|nd| nd.phase.sin()).sum();
        let r = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n;
        (r, sum_sin.atan2(sum_cos))
    }

    pub fn weight_variance(&self) -> f64 {
        let n = self.nodes.len(); if n < 2 { return 0.0; }
        let count = (n * n - n) as f64;
        let mut sum = 0.0; let mut sum_sq = 0.0;
        for &w in &self.weights {
            // Nota: Isso inclui as zeros da diagonal, mas a variância ainda é útil.
            // Para ser exato, deveríamos filtrar i == j.
            sum += w; sum_sq += w * w;
        }
        let mean = sum / count; sum_sq / count - mean * mean
    }

    pub fn spectral_band_energy(&self) -> [f64; 3] {
        let n = self.nodes.len(); if n == 0 { return [0.0; 3]; }
        let third = n / 3;
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| self.nodes[a].frequency.partial_cmp(&self.nodes[b].frequency).unwrap());
        let mut energy = [0.0_f64; 3]; let mut count = [0usize; 3];
        for (rank, &idx) in sorted.iter().enumerate() {
            let band = if rank < third { 0 } else if rank < 2 * third { 1 } else { 2 };
            energy[band] += self.nodes[idx].state * self.nodes[idx].state; count[band] += 1;
        }
        [if count[0]>0 {energy[0]/count[0] as f64} else {0.0}, if count[1]>0 {energy[1]/count[1] as f64} else {0.0}, if count[2]>0 {energy[2]/count[2] as f64} else {0.0}]
    }

    pub fn is_inhibited(&self) -> bool { self.stress > STRESS_INHIBITION_THRESHOLD }
    pub fn is_superfluid(&self) -> bool { self.viscosity < 1.1 }

    pub fn save_state(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let state = SerializableState {
            nodes: self.nodes.clone(), node_layers: self.node_layers.clone(), num_layers: self.num_layers,
            weights: self.weights.clone(), weights_plastic: self.weights_plastic.clone(), time: self.time,
            #[cfg(feature = "stdp")] eligibility: self.eligibility.clone(),
            #[cfg(feature = "stdp")] bcm_threshold: self.bcm_threshold.clone(),
            fisher_importance: self.fisher_importance.clone(), anchor_weights: self.anchor_weights.clone(),
        };
        std::fs::write(path, bincode::serialize(&state)?)?; Ok(())
    }

    pub fn load_state(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let state: SerializableState = bincode::deserialize(&std::fs::read(path)?)?;
        if state.nodes.len() != self.nodes.len() { anyhow::bail!("Node count mismatch"); }
        self.nodes = state.nodes; self.node_layers = state.node_layers; self.num_layers = state.num_layers;
        self.weights = state.weights; self.weights_plastic = state.weights_plastic; self.time = state.time;
        #[cfg(feature = "stdp")] { self.eligibility = state.eligibility; self.bcm_threshold = state.bcm_threshold; }
        self.fisher_importance = state.fisher_importance; self.anchor_weights = state.anchor_weights;
        Ok(())
    }
}

pub fn calculate_stdp(t_pre: f64, t_pos: f64) -> f64 {
    let delta_t = t_pos - t_pre;
    if delta_t > 0.0 && delta_t < 100.0 { 0.01 * (-delta_t / 20.0).exp() } 
    else if delta_t < 0.0 && delta_t > -100.0 { -0.012 * (delta_t / 25.0).exp() } 
    else { 0.0 }
}

#[derive(Serialize, Deserialize)]
struct SerializableState {
    nodes: Vec<LiquidNode>,
    node_layers: Vec<usize>,
    num_layers: usize,
    weights: Vec<f64>,
    weights_plastic: Vec<f64>,
    time: f64,
    #[cfg(feature = "stdp")] eligibility: Vec<f64>,
    #[cfg(feature = "stdp")] bcm_threshold: Vec<f64>,
    fisher_importance: Vec<f64>,
    anchor_weights: Option<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_tel(h: f64, e: f64, s: f64) -> LlmTelemetry { LlmTelemetry { token: "test".into(), entropy_chaos: h, context_energy: e, hardware_stress: s } }
    #[test] fn stability() {
        let mut net = LiquidNetwork::new(16, 2);
        for _ in 0..100 { net.step().unwrap(); }
        for n in &net.nodes { assert!(n.state.abs() <= 1.0); }
    }
}
