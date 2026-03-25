// ============================================================================
// simulation.rs — Gerenciamento da rede de neurônios CfC (MIT Architecture)
// ============================================================================
// Gerencia o array de neurônios CfC, aplica estímulos e processa a 
// propagação de sinal entre vizinhos (sinapses).
// ============================================================================

use crate::physics::NeuronioCfC;
use rayon::prelude::*;

/// Estado da simulação: array de neurônios CfC + metadata
pub struct Simulation {
    pub nodes: Vec<NeuronioCfC>,
    pub layer_name: String,
    pub tick_count: u64,
    pub grid_size: usize,
}

impl Simulation {
    /// Instancia neurônios CfC a partir de um array de pesos f32
    pub fn from_weights(weights: &[f32], layer_name: &str) -> Self {
        log::info!(
            "🧪 Instanciando {} neurônios CfC (LTC) para '{}'",
            weights.len(),
            layer_name
        );

        let nodes: Vec<NeuronioCfC> = weights
            .par_iter()
            .map(|&w| NeuronioCfC::from_weight(w))
            .collect();

        log::info!("✅ {} neurônios criados com sucesso", nodes.len());

        Simulation {
            nodes,
            layer_name: layer_name.to_string(),
            tick_count: 0,
            grid_size: 128,
        }
    }

    /// Aplica um pulso de energia nos neurônios (Impulso de estado)
    pub fn apply_stimulus(&mut self, energy: &[f32]) {
        if energy.is_empty() {
            return;
        }

        log::info!(
            "💥 Estimulando {} neurônios CfC com pulso externo",
            self.nodes.len()
        );

        for (i, node) in self.nodes.iter_mut().enumerate() {
            let stimulus = energy[i % energy.len()];
            // Injeção direta no estado x (impulso delta)
            node.x += stimulus;
        }
    }

    /// Atualiza todos os neurônios e processa propagação de sinal CfC + Kuramoto (GABA)
    pub fn tick(&mut self, dt: f32) {
        // 1. Atualizar dinâmica interna (Closed-form) em paralelo
        self.nodes.par_iter_mut().for_each(|node| {
            node.tick(0.0, dt);
        });

        let num_nodes = self.nodes.len();
        let grid_w = (num_nodes as f64).sqrt() as usize;

        if grid_w > 0 {
            // 2. Acoplamento de Kuramoto com Inibição Lateral (K_ij Semântico)
            let mut adjustments = vec![0.0f32; num_nodes];
            let base_k = 0.1; // Aumentado para resposta mais rápida

            for idx in 0..num_nodes {
                let x = idx % grid_w;
                let y = idx / grid_w;
                let w_i = self.nodes[idx].weight;

                let neighbors = [
                    (x as i32 + 1, y as i32), (x as i32 - 1, y as i32),
                    (x as i32, y as i32 + 1), (x as i32, y as i32 - 1)
                ];

                let mut coupling_sum = 0.0;
                for (nx, ny) in neighbors {
                    if nx >= 0 && nx < grid_w as i32 && ny >= 0 && ny < grid_w as i32 {
                        let n_idx = (ny as usize * grid_w) + nx as usize;
                        if n_idx < num_nodes {
                            let w_j = self.nodes[n_idx].weight;
                            // K_ij: Atração se similar, Repulsão se diferente
                            let distance = (w_j - w_i).abs().min(2.0) / 2.0;
                            let k_ij = base_k * (1.0 - 2.8 * distance); // Fator de inibição forte
                            
                            coupling_sum += k_ij * (self.nodes[n_idx].x - self.nodes[idx].x);
                        }
                    }
                }
                adjustments[idx] = coupling_sum;
            }

            // Aplicar os ajustes + Threshold de Estabilidade
            for (node, adj) in self.nodes.iter_mut().zip(adjustments.iter()) {
                node.x += adj;
                // Inibição Homeostática: Se a energia for baixa, silencia rápido (GABA)
                if node.x.abs() < 0.01 {
                    node.x *= 0.7; 
                }
            }

            // 3. Propagação Sináptica Local (Eventos de Spike)
            let mut active_indices = Vec::new();
            for (i, node) in self.nodes.iter().enumerate() {
                if node.has_spiked {
                    active_indices.push(i);
                }
            }

            let influence = 0.15; 
            for &idx in &active_indices {
                let x = idx % grid_w;
                let y = idx / grid_w;
                let neighbors = [
                    (x as i32 + 1, y as i32), (x as i32 - 1, y as i32),
                    (x as i32, y as i32 + 1), (x as i32, y as i32 - 1)
                ];
                for (nx, ny) in neighbors {
                    if nx >= 0 && nx < grid_w as i32 && ny >= 0 && ny < grid_w as i32 {
                        let n_idx = (ny as usize * grid_w) + nx as usize;
                        if n_idx < num_nodes {
                            self.nodes[n_idx].x += influence;
                        }
                    }
                }
            }
        }

        self.tick_count += 1;
    }

    /// snapshot_grid: mapeia a ativação para o grid NxN
    pub fn snapshot_grid(&self) -> Vec<f32> {
        let total_cells = self.grid_size * self.grid_size;
        let num_nodes = self.nodes.len();

        if num_nodes == 0 {
            return vec![0.0; total_cells];
        }

        let mut grid = vec![0.0f32; total_cells];

        if num_nodes >= total_cells {
            let ratio = num_nodes as f64 / total_cells as f64;

            for cell_idx in 0..total_cells {
                let start = (cell_idx as f64 * ratio) as usize;
                let end = (((cell_idx + 1) as f64 * ratio) as usize).min(num_nodes);

                if start < num_nodes && end > start {
                    let sum: f32 = self.nodes[start..end]
                        .iter()
                        .map(|n| n.posicao_x())
                        .sum();
                    grid[cell_idx] = sum / (end - start) as f32;
                }
            }
        } else {
            let ratio = num_nodes as f64 / total_cells as f64;
            for cell_idx in 0..total_cells {
                let node_idx = (cell_idx as f64 * ratio) as usize;
                let node_idx = node_idx.min(num_nodes - 1);
                grid[cell_idx] = self.nodes[node_idx].posicao_x();
            }
        }

        grid
    }

    /// Calcula a energia total (soma absoluta dos estados)
    pub fn total_energy(&self) -> f32 {
        self.nodes.par_iter().map(|n| n.x.abs()).sum()
    }

    pub fn reset_all(&mut self) {
        self.nodes.par_iter_mut().for_each(|node| {
            node.reset();
        });
        self.tick_count = 0;
        log::info!("🔄 Fauna CfC resetada.");
    }

    pub fn active_count(&self, _threshold_unused: f32) -> usize {
        self.nodes.par_iter().filter(|n| n.has_spiked).count()
    }
}
