//! SimHash — Locality Sensitive Hashing para lookup angular sub-linear.
//!
//! Preserva similaridade de cosseno: P(hash(a) == hash(b)) = 1 - angle(a,b)/π
//!
//! Uso: indexar pesos ternários → dado um input embedding, encontrar os
//! neurônios mais prováveis de ativar sem computar o matmul completo.
//!
//! Referência: SLIDE (Rice 2020) — https://arxiv.org/abs/1903.03129

use std::collections::HashMap;

/// Tabela SimHash para lookup de neurônios.
pub struct SimHashTable {
    /// Vetores de projeção gaussianos normalizados: [num_tables * num_bits, dim_in]
    projections: Vec<Vec<f32>>,
    /// Mapas hash → lista de neuron indices (um por tabela)
    tables: Vec<HashMap<u32, Vec<usize>>>,
    pub num_bits: usize,
    pub num_tables: usize,
    pub dim_in: usize,
    pub num_neurons: usize,
}

impl SimHashTable {
    /// Constrói tabelas LSH para uma matriz de pesos W [dim_out, dim_in].
    /// Cada linha de W é indexada pelo seu hash angular.
    ///
    /// # Parâmetros
    /// - `weights`: pesos F32 ou ternários [dim_out, dim_in] row-major
    /// - `num_bits`: bits por hash (mais bits → buckets menores → mais precisão)
    /// - `num_tables`: tabelas (mais tabelas → mais recall, mais memória)
    pub fn build(weights: &[f32], dim_out: usize, dim_in: usize, num_bits: usize, num_tables: usize) -> Self {
        use std::f32::consts::PI;

        let total_proj = num_bits * num_tables;
        let mut projections = Vec::with_capacity(total_proj);

        // LCG + Box-Muller para projections gaussianas determinísticas
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut next_gauss = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 33) as f32 / u32::MAX as f32 + 1e-9;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 33) as f32 / u32::MAX as f32 * 2.0 * PI;
            (-2.0 * u1.ln()).sqrt() * u2.cos()
        };

        for _ in 0..total_proj {
            let mut p: Vec<f32> = (0..dim_in).map(|_| next_gauss()).collect();
            let n = p.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
            p.iter_mut().for_each(|v| *v /= n);
            projections.push(p);
        }

        // Indexar cada neurônio (linha de W) em todas as tabelas
        let mut tables: Vec<HashMap<u32, Vec<usize>>> = vec![HashMap::new(); num_tables];
        for ni in 0..dim_out {
            let row = &weights[ni * dim_in..(ni + 1) * dim_in];
            for t in 0..num_tables {
                let h = Self::hash_row(row, &projections, t, num_bits);
                tables[t].entry(h).or_default().push(ni);
            }
        }

        SimHashTable { projections, tables, num_bits, num_tables, dim_in, num_neurons: dim_out }
    }

    /// Computa hash SimHash de um vetor usando as projeções da tabela t.
    fn hash_row(row: &[f32], projs: &[Vec<f32>], t: usize, num_bits: usize) -> u32 {
        let mut h: u32 = 0;
        for b in 0..num_bits {
            let proj = &projs[t * num_bits + b];
            let dot: f32 = row.iter().zip(proj.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 { h |= 1 << b; }
        }
        h
    }

    /// Query: retorna neurônios candidatos para o input dado.
    /// Multi-probe: união dos buckets de todas as tabelas (sem duplicatas).
    pub fn query(&self, input: &[f32]) -> Vec<usize> {
        assert_eq!(input.len(), self.dim_in, "input dim mismatch");
        let mut seen = vec![false; self.num_neurons];
        for t in 0..self.num_tables {
            let h = Self::hash_row(input, &self.projections, t, self.num_bits);
            if let Some(bucket) = self.tables[t].get(&h) {
                for &idx in bucket { seen[idx] = true; }
            }
        }
        seen.into_iter().enumerate().filter(|(_, c)| *c).map(|(i, _)| i).collect()
    }

    /// Query com probe vizinho: também busca buckets com 1 bit diferente (mais recall).
    pub fn query_multiprobe(&self, input: &[f32], flip_bits: usize) -> Vec<usize> {
        let mut seen = vec![false; self.num_neurons];
        for t in 0..self.num_tables {
            let h = Self::hash_row(input, &self.projections, t, self.num_bits);
            // Bucket exato
            if let Some(b) = self.tables[t].get(&h) {
                for &idx in b { seen[idx] = true; }
            }
            // Buckets com flip_bits bits invertidos
            for bit in 0..self.num_bits.min(flip_bits * 8) {
                let h_flip = h ^ (1 << bit);
                if let Some(b) = self.tables[t].get(&h_flip) {
                    for &idx in b { seen[idx] = true; }
                }
            }
        }
        seen.into_iter().enumerate().filter(|(_, c)| *c).map(|(i, _)| i).collect()
    }

    /// Recall@k: fração dos top-k neurônios ativos (|matmul|) nos candidatos.
    pub fn recall_at_k(&self, weights: &[f32], input: &[f32], k: usize) -> RecallResult {
        let dim_out = self.num_neurons;
        let dim_in = self.dim_in;

        let t0 = std::time::Instant::now();
        let mut outputs: Vec<(f32, usize)> = (0..dim_out)
            .map(|j| {
                let row = &weights[j * dim_in..(j + 1) * dim_in];
                let dot: f32 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                (dot.abs(), j)
            })
            .collect();
        let full_us = t0.elapsed().as_micros();

        outputs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let top_k: std::collections::HashSet<usize> = outputs[..k.min(dim_out)]
            .iter().map(|(_, i)| *i).collect();

        let t1 = std::time::Instant::now();
        let candidates = self.query(input);
        let query_us = t1.elapsed().as_micros();

        let cands_set: std::collections::HashSet<usize> = candidates.iter().cloned().collect();
        let recall = top_k.intersection(&cands_set).count() as f32 / k as f32;

        RecallResult {
            recall,
            k,
            n_candidates: candidates.len(),
            candidate_pct: candidates.len() as f32 / dim_out as f32 * 100.0,
            speedup: full_us as f32 / query_us.max(1) as f32,
            full_us,
            query_us,
        }
    }

    /// Estatísticas de balanceamento dos buckets.
    pub fn bucket_stats(&self) -> BucketStats {
        let all_sizes: Vec<usize> = self.tables.iter()
            .flat_map(|t| t.values().map(|v| v.len()))
            .collect();
        let total = all_sizes.len();
        let max_size = *all_sizes.iter().max().unwrap_or(&0);
        let avg_size = all_sizes.iter().sum::<usize>() as f32 / total.max(1) as f32;
        let total_entries: usize = all_sizes.iter().sum();
        BucketStats { n_buckets: total, max_size, avg_size, total_entries }
    }
}

#[derive(Debug)]
pub struct RecallResult {
    pub recall: f32,
    pub k: usize,
    pub n_candidates: usize,
    pub candidate_pct: f32,
    pub speedup: f32,
    pub full_us: u128,
    pub query_us: u128,
}

#[derive(Debug)]
pub struct BucketStats {
    pub n_buckets: usize,
    pub max_size: usize,
    pub avg_size: f32,
    pub total_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_identical_vectors() {
        // Dois vetores idênticos devem cair no mesmo bucket (sempre)
        let dim = 64usize;
        let w: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let weights: Vec<f32> = w.clone(); // 1 neurônio
        let lsh = SimHashTable::build(&weights, 1, dim, 8, 4);
        let candidates = lsh.query(&w);
        assert!(!candidates.is_empty(), "Identical vector should always find itself");
    }

    #[test]
    fn test_simhash_orthogonal_vectors() {
        // Vetores ortogonais raramente devem colidir
        let dim = 64usize;
        let w1: Vec<f32> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let w2: Vec<f32> = (0..dim).map(|i| if i % 2 != 0 { 1.0 } else { 0.0 }).collect();

        // Criar LSH com 2 neurônios (w1 e w2)
        let weights: Vec<f32> = w1.iter().chain(w2.iter()).cloned().collect();
        let lsh = SimHashTable::build(&weights, 2, dim, 12, 8);

        // Query com w1: deve preferir neurônio 0 (w1)
        let cands_w1 = lsh.query(&w1);
        // w1 deve estar nos candidatos (mesmo hash)
        assert!(cands_w1.contains(&0), "w1 should find itself");
    }
}
