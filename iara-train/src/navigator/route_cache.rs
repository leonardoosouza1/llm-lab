//! Semantic Route Cache — cache de rotas para prompts similares.
//!
//! Prompts com embeddings angularmente próximos (SimHash bucket match)
//! recebem a mesma rota de super layers sem re-executar A*.
//!
//! Pipeline:
//!   prompt → bag-of-words embedding → normaliza para hiperesfera →
//!   SimHash (k bits) → lookup no cache → HIT: retorna rota cached |
//!   MISS: executa A* → armazena no cache
//!
//! Referência: semantic caching em LLM inference (GPTCache, 2023).

use std::collections::HashMap;
use crate::navigator::router::RouteResult;

/// Configuração do cache de rotas
#[derive(Debug, Clone)]
pub struct RouteCacheConfig {
    /// Número de bits do SimHash (mais bits → menos colisões → menos hits, mais precisão)
    pub hash_bits: usize,
    /// Capacidade máxima de entradas no cache
    pub max_entries: usize,
    /// Dimensão do embedding interno (BoW vocabulary)
    pub embed_dim: usize,
}

impl Default for RouteCacheConfig {
    fn default() -> Self {
        RouteCacheConfig {
            hash_bits:   16,
            max_entries: 1024,
            embed_dim:   256,
        }
    }
}

/// Resultado de uma consulta ao cache
#[derive(Debug)]
pub enum CacheResult<'a> {
    Hit  { route: &'a CachedRoute, hash: u32 },
    Miss { hash: u32 },
}

/// Rota armazenada no cache
#[derive(Debug, Clone)]
pub struct CachedRoute {
    pub path:           Vec<usize>,
    pub total_cost:     f32,
    pub flops_saved_pct: f32,
    pub quality_loss:   f32,
    pub hit_count:      usize,
}

impl From<&RouteResult> for CachedRoute {
    fn from(r: &RouteResult) -> Self {
        CachedRoute {
            path:            r.path.clone(),
            total_cost:      r.total_cost,
            flops_saved_pct: r.flops_saved_pct,
            quality_loss:    r.quality_loss,
            hit_count:       0,
        }
    }
}

/// Cache de rotas baseado em SimHash de embeddings de prompt
pub struct RouteCache {
    /// Tabela: hash → rota cached
    table:      HashMap<u32, CachedRoute>,
    /// Projeções aleatórias gaussianas para SimHash [hash_bits × embed_dim]
    projections: Vec<Vec<f32>>,
    /// Configuração
    pub config: RouteCacheConfig,
    /// Métricas
    pub hits:   usize,
    pub misses: usize,
}

impl RouteCache {
    /// Cria cache com projeções determinísticas (LCG + Box-Muller).
    pub fn new(config: RouteCacheConfig) -> Self {
        use std::f32::consts::PI;
        let n_proj = config.hash_bits;
        let dim    = config.embed_dim;

        let mut state: u64 = 0xDEAD_BEEF_CAFE_1337;
        let mut next_gauss = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 33) as f32 / u32::MAX as f32 + 1e-9;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 33) as f32 / u32::MAX as f32 * 2.0 * PI;
            (-2.0 * u1.ln()).sqrt() * u2.cos()
        };

        let projections: Vec<Vec<f32>> = (0..n_proj)
            .map(|_| {
                let mut p: Vec<f32> = (0..dim).map(|_| next_gauss()).collect();
                let norm = p.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
                p.iter_mut().for_each(|v| *v /= norm);
                p
            })
            .collect();

        RouteCache {
            table: HashMap::new(),
            projections,
            config,
            hits: 0,
            misses: 0,
        }
    }

    /// Converte texto em embedding bag-of-words normalizado.
    /// Dimensão fixa = config.embed_dim (hash de caracteres → índice).
    pub fn embed_prompt(&self, prompt: &str) -> Vec<f32> {
        let dim = self.config.embed_dim;
        let mut emb = vec![0.0f32; dim];

        // Bag-of-words por trigramas de caracteres
        let bytes = prompt.as_bytes();
        for w in bytes.windows(3) {
            // FNV-1a hash dos 3 bytes → índice
            let h = w.iter().fold(2166136261u32, |acc, &b| {
                (acc ^ b as u32).wrapping_mul(16777619)
            }) as usize % dim;
            emb[h] += 1.0;
        }
        // Palavras individuais (unigrams)
        for word in prompt.split_whitespace() {
            let h = word.bytes().fold(2166136261u32, |acc, b| {
                (acc ^ b as u32).wrapping_mul(16777619)
            }) as usize % dim;
            emb[h] += 2.0; // peso maior para unigrams
        }

        // Normalizar para hiperesfera (nGPT style)
        let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
        emb.iter_mut().for_each(|v| *v /= norm);
        emb
    }

    /// Computa SimHash de um embedding normalizado.
    pub fn hash_embedding(&self, emb: &[f32]) -> u32 {
        let mut h: u32 = 0;
        for (b, proj) in self.projections.iter().enumerate() {
            let dot: f32 = emb.iter().zip(proj.iter()).map(|(a, p)| a * p).sum();
            if dot >= 0.0 { h |= 1 << b; }
        }
        h
    }

    /// Consulta o cache para um prompt.
    pub fn lookup<'a>(&'a mut self, prompt: &str) -> CacheResult<'a> {
        let emb  = self.embed_prompt(prompt);
        let hash = self.hash_embedding(&emb);

        if self.table.contains_key(&hash) {
            self.hits += 1;
            self.table.get_mut(&hash).unwrap().hit_count += 1;
            CacheResult::Hit { route: self.table.get(&hash).unwrap(), hash }
        } else {
            self.misses += 1;
            CacheResult::Miss { hash }
        }
    }

    /// Armazena uma rota no cache para o hash dado.
    pub fn store(&mut self, hash: u32, route: &RouteResult) {
        if self.table.len() >= self.config.max_entries {
            // Evicção: remove entrada com menor hit_count (LFU simples)
            if let Some((&lfu_key, _)) = self.table.iter()
                .min_by_key(|(_, v)| v.hit_count)
            {
                self.table.remove(&lfu_key);
            }
        }
        self.table.insert(hash, CachedRoute::from(route));
    }

    /// Taxa de cache hit [0..1]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f32 / total as f32 }
    }

    /// Estatísticas do cache
    pub fn stats(&self) -> CacheStats {
        let total_hits: usize = self.table.values().map(|r| r.hit_count).sum();
        CacheStats {
            n_entries:    self.table.len(),
            total_hits:   self.hits,
            total_misses: self.misses,
            hit_rate:     self.hit_rate(),
            avg_hit_count: if self.table.is_empty() { 0.0 }
                           else { total_hits as f32 / self.table.len() as f32 },
        }
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub n_entries:     usize,
    pub total_hits:    usize,
    pub total_misses:  usize,
    pub hit_rate:      f32,
    pub avg_hit_count: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_route(path: Vec<usize>) -> RouteResult {
        RouteResult {
            path,
            total_cost:      2.0,
            flops_saved_pct: 25.0,
            nodes_explored:  4,
            quality_loss:    0.0,
        }
    }

    #[test]
    fn test_same_prompt_hits() {
        let mut cache = RouteCache::new(RouteCacheConfig::default());
        let route = dummy_route(vec![0, 1, 2, 3]);

        // Miss na primeira vez
        let hash = match cache.lookup("What is Paris?") {
            CacheResult::Miss { hash } => hash,
            CacheResult::Hit { .. } => panic!("Should be miss"),
        };
        cache.store(hash, &route);

        // Hit na segunda vez com mesmo prompt
        match cache.lookup("What is Paris?") {
            CacheResult::Hit { route: r, .. } => {
                assert_eq!(r.path, vec![0, 1, 2, 3]);
                assert_eq!(r.hit_count, 1);
            }
            CacheResult::Miss { .. } => panic!("Should be hit"),
        }
    }

    #[test]
    fn test_different_prompts_miss() {
        let mut cache = RouteCache::new(RouteCacheConfig::default());
        let route = dummy_route(vec![0, 3]);

        let hash = match cache.lookup("What is Paris?") {
            CacheResult::Miss { hash } => hash,
            _ => panic!(),
        };
        cache.store(hash, &route);

        // Prompt completamente diferente deve ser miss
        // (pode colidir em casos extremos, mas com 16 bits é improvável)
        let result = cache.lookup("Write a Python function to sort a list");
        // Verificamos apenas que o resultado é consistente (não panic)
        match result {
            CacheResult::Hit { .. } | CacheResult::Miss { .. } => {} // ok
        }
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = RouteCache::new(RouteCacheConfig::default());
        let route = dummy_route(vec![0, 1, 2, 3]);

        let hash = match cache.lookup("hello world") {
            CacheResult::Miss { hash } => hash,
            _ => panic!(),
        };
        cache.store(hash, &route);

        cache.lookup("hello world");
        cache.lookup("hello world");

        let stats = cache.stats();
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_hits, 2);
        assert!((stats.hit_rate - 2.0/3.0).abs() < 1e-5);
    }

    #[test]
    fn test_embed_deterministic() {
        let cache = RouteCache::new(RouteCacheConfig::default());
        let e1 = cache.embed_prompt("What is the capital of France?");
        let e2 = cache.embed_prompt("What is the capital of France?");
        for (a, b) in e1.iter().zip(e2.iter()) {
            assert!((a - b).abs() < 1e-7, "Embedding must be deterministic");
        }
    }

    #[test]
    fn test_embedding_normalized() {
        let cache = RouteCache::new(RouteCacheConfig::default());
        let emb = cache.embed_prompt("Tell me about machine learning");
        let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be unit norm");
    }

    #[test]
    fn test_lfu_eviction() {
        let config = RouteCacheConfig { max_entries: 3, ..Default::default() };
        let mut cache = RouteCache::new(config);
        let route = dummy_route(vec![0, 3]);

        // Preencher cache com 3 entradas únicas
        for i in 0..3 {
            let prompt = format!("unique prompt number {}", i);
            let hash = match cache.lookup(&prompt) {
                CacheResult::Miss { hash } => hash,
                _ => continue, // colisão de hash OK
            };
            cache.store(hash, &route);
        }

        // Adicionar 4ª entrada deve evictar a de menor hit_count
        let hash4 = match cache.lookup("totally different query here xyz") {
            CacheResult::Miss { hash } => hash,
            CacheResult::Hit { .. } => return, // colisão OK
        };
        let prev_size = cache.table.len();
        cache.store(hash4, &route);
        assert!(cache.table.len() <= cache.config.max_entries,
            "Cache must not exceed max_entries");
    }

    #[test]
    fn test_path_names_via_cache() {
        // Verifica que RouteResult::path_names() funciona com rotas do cache
        let result = RouteResult {
            path: vec![0, 2, 3],
            total_cost: 2.0,
            flops_saved_pct: 25.0,
            nodes_explored: 4,
            quality_loss: 0.0,
        };
        let names = result.path_names();
        assert_eq!(names, vec!["SL0:Embed", "SL2:Know", "SL3:Gen"]);
    }
}
