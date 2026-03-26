// ============================================================================
// liquid_brain.rs — Snapshot visual do cérebro líquido (CfC LNN)
// ============================================================================
//
// Produz um frame JSON "brain_map" que o frontend renderiza como um cérebro
// 2D com três lobos de ressonância:
//
//   Lobo Focus   (freq baixa)  — esquerda  — azul/ciano
//   Lobo Novelty (freq média)  — topo      — verde/amarelo
//   Lobo Drive   (freq alta)   — direita   — laranja/vermelho
//
// Os nós pulsam na frequência da sua CfC. As sinapses mais fortes aparecem
// como linhas luminosas conectando os lobos.
// ============================================================================

use crate::experimental::liquid_network::LiquidNetwork;
use serde::Serialize;
use std::f32::consts::TAU;

// ─────────────────────────────────────────────────────────────────────────────
// § Structs de snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Posição e estado visual de um nó no mapa cerebral.
#[derive(Debug, Clone, Serialize)]
pub struct NodeViz {
    pub id: usize,
    /// Índice da camada LLM (0..31).
    pub layer: usize,
    /// Posição X normalizada [0, 1] no canvas.
    pub x: f32,
    /// Posição Y normalizada [0, 1] no canvas.
    pub y: f32,
    /// Posição Z (profundidade de camada).
    pub z: f32,
    /// Ativação atual, normalizada para [-1, 1].
    pub activation: f32,
    /// Fase atual em radianos [0, 2π).
    pub phase: f32,
    /// Amplitude do envelope.
    pub amplitude: f32,
    /// Banda espectral: 0 = focus, 1 = novelty, 2 = drive.
    pub band: u8,
}

/// Sinapse forte entre dois nós.
#[derive(Debug, Clone, Serialize)]
pub struct SynapseViz {
    pub from: usize,
    pub to: usize,
    /// Força normalizada [0, 1].
    pub strength: f32,
    /// true = excitatório, false = inibitório.
    pub excitatory: bool,
}

/// Métricas de banda espectral.
#[derive(Debug, Clone, Serialize)]
pub struct BandStats {
    pub energy: f32,
    pub coherence: f32,
}

/// Frame completo do cérebro líquido — enviado como "brain_map" via WebSocket.
#[derive(Debug, Serialize)]
pub struct BrainSnapshot {
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Nós com posição e estado.
    pub nodes: Vec<NodeViz>,
    /// Top-N sinapses mais fortes.
    pub synapses: Vec<SynapseViz>,
    /// Estatísticas por banda: [focus, novelty, drive].
    pub bands: [BandStats; 3],
    /// Coerência global de Kuramoto r ∈ [0, 1].
    pub global_r: f32,
    /// Fase circadiana atual.
    pub circadian: String,
    /// Energia média da rede.
    pub mean_energy: f32,
    /// Número total de camadas.
    pub num_layers: usize,
    /// Tick da simulação.
    pub tick: u64,
    }


// ─────────────────────────────────────────────────────────────────────────────
// § Layout dos nós: três lobos
// ─────────────────────────────────────────────────────────────────────────────

/// Calcula a posição 2D de um nó com base na banda e no índice dentro do lobe.
///
/// Os três lobos têm centros em:
///   Focus   → (0.25, 0.55)
///   Novelty → (0.50, 0.25)
///   Drive   → (0.75, 0.55)
///
/// Dentro de cada lobo, os nós são distribuídos em espiral áurea.
fn lobe_position(band: u8, idx_in_band: usize, band_size: usize) -> (f32, f32) {
    let (cx, cy, radius) = match band {
        0 => (0.25f32, 0.58f32, 0.18f32), // Focus — esquerda
        1 => (0.50f32, 0.22f32, 0.18f32), // Novelty — topo
        _ => (0.75f32, 0.58f32, 0.18f32), // Drive — direita
    };

    // Espiral de Fibonacci para layout uniforme no círculo
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt()); // ~137.5°
    let r = radius * (idx_in_band as f32 / band_size as f32).sqrt();
    let theta = idx_in_band as f32 * golden;

    (cx + r * theta.cos(), cy + r * theta.sin())
}

// ─────────────────────────────────────────────────────────────────────────────
// § Produção do snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Gera um `BrainSnapshot` a partir do estado atual da `LiquidNetwork`.
pub fn snapshot(net: &LiquidNetwork, tick: u64) -> BrainSnapshot {
    let n = net.nodes.len();
    if n == 0 {
        return empty_snapshot(tick);
    }

    // ── Classificar nós em bandas por frequência natural ─────────────────────
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| {
        net.nodes[a]
            .frequency
            .partial_cmp(&net.nodes[b].frequency)
            .unwrap()
    });
    let third = n / 3;
    let mut band_of = vec![0u8; n];
    let mut idx_in_band = vec![0usize; n];
    let mut band_counts = [0usize; 3];
    for (rank, &node_id) in sorted.iter().enumerate() {
        let b = if rank < third {
            0u8
        } else if rank < 2 * third {
            1
        } else {
            2
        };
        band_of[node_id] = b;
        idx_in_band[node_id] = band_counts[b as usize];
        band_counts[b as usize] += 1;
    }

    // ── Nós com posição e estado ──────────────────────────────────────────────
    let num_layers = net.num_layers;
    let nodes: Vec<NodeViz> = net
        .nodes
        .iter()
        .map(|nd| {
            let b = band_of[nd.id];
            let idx = idx_in_band[nd.id];
            let sz = band_counts[b as usize].max(1);
            let (x, y) = lobe_position(b, idx, sz);
            let layer = net.node_layers[nd.id];
            let z = layer as f32 / num_layers.max(1) as f32;

            NodeViz {
                id: nd.id,
                layer,
                x,
                y,
                z,
                activation: nd.state.clamp(-1.0, 1.0) as f32,
                phase: (nd.phase % TAU as f64) as f32,
                amplitude: nd.amplitude.clamp(0.0, 2.0) as f32,
                band: b,
            }
        })
        .collect();

    // ── Top sinapses (excluir zeros/diagonais) ───────────────────────────────
    let synapses = top_synapses(net, &nodes, 120);

    // ── Energia por banda ─────────────────────────────────────────────────────
    let mut band_energy = [0.0f64; 3];
    let mut band_sq_sum = [[0.0f64; 2]; 3]; // [sum_cos, sum_sin] para coerência
    let mut band_n = [0usize; 3];
    for nd in &net.nodes {
        let b = band_of[nd.id] as usize;
        band_energy[b] += nd.state * nd.state;
        band_sq_sum[b][0] += nd.phase.cos();
        band_sq_sum[b][1] += nd.phase.sin();
        band_n[b] += 1;
    }

    let bands = std::array::from_fn(|b| {
        let cnt = band_n[b].max(1) as f64;
        let energy = (band_energy[b] / cnt).sqrt() as f32;
        let rc = band_sq_sum[b][0] / cnt;
        let rs = band_sq_sum[b][1] / cnt;
        let coherence = ((rc * rc + rs * rs).sqrt()) as f32;
        BandStats { energy, coherence }
    });

    // ── Coerência global de Kuramoto ──────────────────────────────────────────
    let (r, _) = net.phase_coherence();

    // ── Fase circadiana ───────────────────────────────────────────────────────
    #[cfg(feature = "circadian")]
    let circadian = net.circadian_state.to_string();
    #[cfg(not(feature = "circadian"))]
    let circadian = "Awake".to_string();

    BrainSnapshot {
        msg_type: "brain_map".to_string(),
        nodes,
        synapses,
        bands,
        global_r: r as f32,
        circadian,
        mean_energy: net.mean_energy() as f32,
        num_layers,
        tick,
    }
}

/// Extrai as N sinapses mais fortes da rede (em valor absoluto de peso).
fn top_synapses(net: &LiquidNetwork, _nodes: &[NodeViz], top_n: usize) -> Vec<SynapseViz> {
    // Coletar pares (i, j, |w|, w) para os nós mostrados
    // Usamos as posições dos nós para filtrar arestas muito longas (reduz ruído)
    let n = net.nodes.len();
    let weights = net.weights_combined();

    let mut pairs: Vec<(f32, usize, usize, bool)> = Vec::with_capacity(top_n * 4);

    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let w = weights[i * n + j] as f32;
            let aw = w.abs();
            if aw < 0.05 { continue; } // ignorar conexões fracas

            // Limitar a top_n × 4 candidatos para memória
            if pairs.len() < top_n * 4 || aw > pairs.last().map(|p| p.0).unwrap_or(0.0) {
                pairs.push((aw, i, j, w > 0.0));
            }
        }
    }

    // Ordenar por força e truncar
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_n);

    // Normalizar força [0, 1]
    let max_w = pairs.first().map(|p| p.0).unwrap_or(1.0).max(1e-6);

    pairs
        .into_iter()
        .map(|(aw, i, j, exc)| SynapseViz {
            from: i,
            to: j,
            strength: aw / max_w,
            excitatory: exc,
        })
        .collect()
}

/// Snapshot vazio (rede ainda não inicializada).
fn empty_snapshot(tick: u64) -> BrainSnapshot {
    BrainSnapshot {
        msg_type: "brain_map".to_string(),
        nodes: vec![],
        synapses: vec![],
        bands: std::array::from_fn(|_| BandStats { energy: 0.0, coherence: 0.0 }),
        global_r: 0.0,
        circadian: "Awake".to_string(),
        mean_energy: 0.0,
        num_layers: 0,
        tick,
    }
}
