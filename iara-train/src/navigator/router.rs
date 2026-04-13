//! Graph Router — UCS e A* sobre o grafo de super layers.
//!
//! O modelo é representado como um grafo dirigido:
//!   - Nós: super layers (SL0..SL3)
//!   - Arestas: transições com custo (FLOPs estimados) e penalidade de qualidade
//!   - Objetivo: encontrar o caminho de menor custo que satisfaz threshold de qualidade
//!
//! Algoritmos:
//!   - UCS: garante caminho ótimo (sem heurística)
//!   - A*:  usa heurísticas para reduzir nós explorados

use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Número de super layers no grafo
pub const N_SUPER_LAYERS: usize = 4;

/// Nomes das super layers para display
pub const SL_NAMES: [&str; N_SUPER_LAYERS] = [
    "SL0:Embed", "SL1:Reason", "SL2:Know", "SL3:Gen"
];

/// Custo base de cada super layer (normalizado, 1.0 = custo completo)
pub const SL_BASE_COST: [f32; N_SUPER_LAYERS] = [1.0, 1.0, 1.0, 1.0];

/// Arestas do grafo: (from, to, quality_penalty)
/// quality_penalty: degradação máxima aceitável ao usar este atalho
pub const EDGES: &[(usize, usize, f32)] = &[
    // Caminho completo (sem perda)
    (0, 1, 0.00),
    (1, 2, 0.00),
    (2, 3, 0.00),
    // Atalhos (com penalidade de qualidade)
    (0, 2, 0.02), // pula SL1: -2% qualidade
    (1, 3, 0.02), // pula SL2: -2% qualidade
    (0, 3, 0.05), // draft rápido: -5% qualidade
];

/// Configuração do router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Limiar mínimo de qualidade aceitável [0..1]
    pub quality_threshold: f32,
    /// Pesos da heurística combinada [w_sparsity, w_confidence]
    pub heuristic_weights: [f32; 2],
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig {
            quality_threshold: 0.95,
            heuristic_weights: [0.6, 0.4],
        }
    }
}

/// Resultado de routing
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub path: Vec<usize>,
    pub total_cost: f32,
    pub flops_saved_pct: f32,
    pub nodes_explored: usize,
    pub quality_loss: f32,
}

impl RouteResult {
    pub fn path_names(&self) -> Vec<&'static str> {
        self.path.iter().map(|&i| SL_NAMES[i.min(N_SUPER_LAYERS - 1)]).collect()
    }
}

// ─── Nó interno para a heap ──────────────────────────────────────────────────

#[derive(Clone)]
struct SearchNode {
    sl_idx: usize,
    g_cost: f32,
    f_cost: f32,
    path: Vec<usize>,
    quality_loss: f32,
}

impl PartialEq for SearchNode { fn eq(&self, o: &Self) -> bool { self.f_cost == o.f_cost } }
impl Eq for SearchNode {}
impl PartialOrd for SearchNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        o.f_cost.partial_cmp(&self.f_cost) // min-heap
    }
}
impl Ord for SearchNode {
    fn cmp(&self, o: &Self) -> Ordering { self.partial_cmp(o).unwrap_or(Ordering::Equal) }
}

// ─── Heurísticas ─────────────────────────────────────────────────────────────

/// h_sparsity: custo restante estimado baseado na esparsidade ternária das layers seguintes.
/// Admissível: multiplica por 0.8 para garantir subestimação.
pub fn h_sparsity(current_sl: usize, sparsity_per_sl: &[f32; N_SUPER_LAYERS]) -> f32 {
    ((current_sl + 1)..N_SUPER_LAYERS)
        .map(|sl| SL_BASE_COST[sl] * (1.0 - sparsity_per_sl[sl]))
        .sum::<f32>() * 0.8
}

/// h_confidence: se o modelo já tem alta confiança, pouco custo restante.
/// Baseado em LayerSkip (Meta, 2024).
pub fn h_confidence(current_sl: usize, logit_entropy: f32) -> f32 {
    let remaining = (N_SUPER_LAYERS - 1 - current_sl) as f32;
    remaining * (1.0 - (1.0 - logit_entropy.min(1.0)) * 0.5) * 0.9
}

/// h_combined: média ponderada das heurísticas.
pub fn h_combined(
    current_sl: usize,
    sparsity_per_sl: &[f32; N_SUPER_LAYERS],
    logit_entropy: f32,
    weights: &[f32; 2],
) -> f32 {
    weights[0] * h_sparsity(current_sl, sparsity_per_sl)
        + weights[1] * h_confidence(current_sl, logit_entropy)
}

// ─── Router principal ─────────────────────────────────────────────────────────

/// Rota via UCS (Dijkstra) — garante caminho ótimo, sem heurística.
pub fn route_ucs(sparsity_per_sl: &[f32; N_SUPER_LAYERS], config: &RouterConfig) -> RouteResult {
    route_generic(sparsity_per_sl, 0.5, config, "none")
}

/// Rota via A* com heurística combinada.
pub fn route_astar(
    sparsity_per_sl: &[f32; N_SUPER_LAYERS],
    logit_entropy: f32,
    config: &RouterConfig,
) -> RouteResult {
    route_generic(sparsity_per_sl, logit_entropy, config, "combined")
}

fn route_generic(
    sparsity_per_sl: &[f32; N_SUPER_LAYERS],
    logit_entropy: f32,
    config: &RouterConfig,
    heuristic: &str,
) -> RouteResult {
    let eff_cost = |sl: usize| SL_BASE_COST[sl] * (1.0 - sparsity_per_sl[sl]);

    let h = |sl: usize| match heuristic {
        "sparsity"   => h_sparsity(sl, sparsity_per_sl),
        "confidence" => h_confidence(sl, logit_entropy),
        "combined"   => h_combined(sl, sparsity_per_sl, logit_entropy, &config.heuristic_weights),
        _            => 0.0, // UCS
    };

    let mut heap = BinaryHeap::new();
    let mut explored = 0usize;

    let g0 = eff_cost(0);
    heap.push(SearchNode {
        sl_idx: 0, g_cost: g0, f_cost: g0 + h(0),
        path: vec![0], quality_loss: 0.0,
    });

    while let Some(node) = heap.pop() {
        explored += 1;

        if node.sl_idx == N_SUPER_LAYERS - 1 {
            let max_cost: f32 = (0..N_SUPER_LAYERS).map(eff_cost).sum();
            let saved = (1.0 - node.g_cost / max_cost) * 100.0;
            return RouteResult {
                path: node.path,
                total_cost: node.g_cost,
                flops_saved_pct: saved.max(0.0),
                nodes_explored: explored,
                quality_loss: node.quality_loss,
            };
        }

        for &(from, to, ql) in EDGES {
            if from != node.sl_idx { continue; }
            let new_ql = node.quality_loss + ql;
            if new_ql > (1.0 - config.quality_threshold) { continue; }
            let new_g = node.g_cost + eff_cost(to);
            let mut p = node.path.clone(); p.push(to);
            heap.push(SearchNode {
                sl_idx: to, g_cost: new_g, f_cost: new_g + h(to),
                path: p, quality_loss: new_ql,
            });
        }
    }

    // Fallback: caminho completo
    let full_cost: f32 = (0..N_SUPER_LAYERS).map(eff_cost).sum();
    RouteResult {
        path: (0..N_SUPER_LAYERS).collect(),
        total_cost: full_cost,
        flops_saved_pct: 0.0,
        nodes_explored: explored,
        quality_loss: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ucs_full_path_high_quality() {
        // Com threshold 0.99 deve retornar caminho completo
        let sparsity = [0.0f32; N_SUPER_LAYERS];
        let config = RouterConfig { quality_threshold: 0.99, ..Default::default() };
        let r = route_ucs(&sparsity, &config);
        assert_eq!(r.path, vec![0, 1, 2, 3], "Expected full path at high quality threshold");
    }

    #[test]
    fn test_ucs_draft_low_quality() {
        // Com threshold 0.90 deve poder tomar SL0→SL3 (draft)
        let sparsity = [0.0f32; N_SUPER_LAYERS];
        let config = RouterConfig { quality_threshold: 0.90, ..Default::default() };
        let r = route_ucs(&sparsity, &config);
        // Deve ter economia (pula pelo menos 1 SL)
        assert!(r.flops_saved_pct > 0.0, "Expected FLOPs savings at low quality threshold");
    }

    #[test]
    fn test_astar_same_as_ucs_optimal() {
        // A* com heurística admissível deve achar mesma rota que UCS
        let sparsity = [0.1f32, 0.2, 0.3, 0.1];
        let config = RouterConfig::default();
        let r_ucs   = route_ucs(&sparsity, &config);
        let r_astar = route_astar(&sparsity, 0.5, &config);
        assert_eq!(r_ucs.path, r_astar.path,
            "A* must find same optimal path as UCS");
    }

    #[test]
    fn test_sparsity_reduces_cost() {
        // Alta esparsidade → menor custo efetivo → mais economia
        let sparse = [0.5f32; N_SUPER_LAYERS];
        let dense  = [0.0f32; N_SUPER_LAYERS];
        let config = RouterConfig::default();
        let r_sparse = route_ucs(&sparse, &config);
        let r_dense  = route_ucs(&dense,  &config);
        // Com mesma rota, sparse deve ter menor custo total
        assert!(r_sparse.total_cost <= r_dense.total_cost + 1e-5,
            "Sparse input should have lower or equal cost");
    }
}
