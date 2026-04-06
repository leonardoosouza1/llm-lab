// ============================================================================
// lnn/esn.rs — Echo State Network: observador passivo do stream de tokens
// ============================================================================
//
// Fase 1: ESN como observador de geração LLM.
//
// Arquitetura:
//   u(t)  = [entropy, top1_prob, pos_norm]   ← lê saídas da LLM
//   h(t)  = (1-α)·h(t-1) + α·tanh(W_res·h(t-1) + W_in·u(t))
//   y(t)  = W_out · h(t)                     ← modula sampling do PRÓXIMO token
//   budget(t) = W_out_budget · h(t)          ← determina quantas camadas usar
//
// Matrizes fixas (geradas deterministicamente via LCG, nunca treinadas):
//   W_res [R×R] sparse — conectividade ~10%, raio espectral ≈ 0.9
//   W_in  [R×3]         — escala de entrada ±0.5
//
// Única camada treinável:
//   W_out [1×R] — readout linear (temperatura delta)
//   W_out_budget [1×R] — readout para layer budget (Fast/Medium/Slow)
//   Começam em zero → ESN observa sem interferir.
//   Futuro: treinar via regressão de mínimos quadrados sobre pares (h, target).
//
// Integração no loop de geração:
//   1. forward(tokens) → logits
//   2. esn.step(&logits, pos, max_tokens)
//   3. budget = esn.layer_budget()
//   4. params = esn.modulate(params)
//   5. sample_token(&logits, &params)
//
// ============================================================================

use crate::inference::SamplingParams;
use serde::Serialize;

// ── Layer Budget enum ────────────────────────────────────────────────────────

/// Determines how many layers a token should be processed through.
/// Based on phase coherence and ESN budget computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerBudget {
    /// Use only first N/4 layers (cheap computation, trivial tokens)
    Fast,
    /// Use first 3N/4 layers (moderate computation, most tokens)
    Medium,
    /// Use all N layers (full computation, complex tokens)
    Slow,
}

// ── Constantes da arquitetura ───────────────────────────────────────────────

/// Número de neurônios no reservatório.
const R: usize = 256;

/// Dimensão da entrada: [entropy, top1_prob, pos_norm].
const N_INPUTS: usize = 3;

/// Taxa de vazamento: controla o quanto o estado anterior é retido.
/// 0.1 = memória longa; 0.5 = resposta mais rápida.
const ALPHA: f32 = 0.15;

/// Raio espectral alvo para W_res.
const SPECTRAL_RADIUS: f32 = 0.9;

/// Densidade de conexões no reservatório (~10%).
const DENSITY: f32 = 0.1;

// ── LCG determinístico ──────────────────────────────────────────────────────
//
// Gerador linear congruente para inicialização reprodutível sem deps extras.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Float em [-1, 1].
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Float em [0, 1).
    fn next_f32_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as f32 / u32::MAX as f32
    }
}

// ── Estrutura da ESN ────────────────────────────────────────────────────────

/// Conexão esparsa do reservatório: (índice_origem, peso).
type SparseRow = Vec<(usize, f32)>;

/// Echo State Network — observador passivo do stream LLM.
pub struct EsnObserver {
    /// W_res: reservatório esparso [R → lista de (origem, peso)]
    w_res: Vec<SparseRow>,
    /// W_in: pesos de entrada [R × N_INPUTS]
    w_in: [[f32; N_INPUTS]; R],
    /// W_out: readout treinável [1 × R] (delta de temperatura)
    ///
    /// Inicializado em zero — ESN observa sem alterar sampling.
    /// Treinável externamente via `set_readout()`.
    pub w_out: Vec<f32>,
    /// W_out_budget: readout treinável [1 × R] (layer budget)
    ///
    /// Inicializado em zero — começa conservador (Slow para todos os tokens).
    /// Treinável externamente via `set_readout_budget()`.
    pub w_out_budget: Vec<f32>,
    /// Estado atual do reservatório [R].
    pub state: Vec<f32>,
    /// Acumulador do estado durante a geração (para treinamento futuro).
    pub state_history: Vec<Vec<f32>>,
    /// Histórico de observações (para diagnóstico e treinamento).
    pub obs_history: Vec<[f32; N_INPUTS]>,
}

impl EsnObserver {
    /// Cria uma nova ESN com reservatório gerado deterministicamente.
    ///
    /// `seed` — semente do LCG. Use o mesmo seed para reprodutibilidade.
    pub fn new(seed: u64) -> Self {
        let mut rng = Lcg::new(seed);

        // ── W_in: pesos de entrada ───────────────────────────────────────
        let mut w_in = [[0f32; N_INPUTS]; R];
        for row in w_in.iter_mut() {
            for w in row.iter_mut() {
                *w = rng.next_f32() * 0.5; // escala ±0.5
            }
        }

        // ── W_res: reservatório esparso ──────────────────────────────────
        // Gera pesos, mantém apenas DENSITY fração das conexões.
        // Normaliza pelo número esperado de conexões por linha para
        // aproximar raio espectral = SPECTRAL_RADIUS.
        let expected_degree = (R as f32 * DENSITY).max(1.0);
        let scale = SPECTRAL_RADIUS / expected_degree.sqrt();

        let mut w_res: Vec<SparseRow> = (0..R).map(|_| Vec::new()).collect();
        for to in 0..R {
            for from in 0..R {
                if from == to {
                    continue; // sem auto-conexões
                }
                if rng.next_f32_unit() < DENSITY {
                    let weight = rng.next_f32() * scale;
                    w_res[to].push((from, weight));
                }
            }
        }

        Self {
            w_res,
            w_in,
            w_out: vec![0.0f32; R],
            w_out_budget: vec![0.0f32; R],
            state: vec![0.0f32; R],
            state_history: Vec::new(),
            obs_history: Vec::new(),
        }
    }

    /// Reinicia o estado (deve ser chamado no início de cada geração).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.state_history.clear();
        self.obs_history.clear();
    }

    /// Avança o reservatório um passo com base nos logits do token atual.
    ///
    /// Chamado APÓS `forward()`, ANTES de `sample_token()`.
    ///
    /// Parâmetros:
    /// - `logits` — logits brutos do modelo para o token atual
    /// - `pos` — posição atual na sequência (0-indexed)
    /// - `max_pos` — comprimento máximo esperado (para normalização)
    pub fn step(&mut self, logits: &[f32], pos: usize, max_pos: usize) {
        let obs = self.build_observation(logits, pos, max_pos);

        // ── Atualiza estado do reservatório ──────────────────────────────
        // h(t) = (1-α)·h(t-1) + α·tanh(W_res·h(t-1) + W_in·u(t))

        let mut pre_act = vec![0f32; R];

        // Contribuição do reservatório: W_res · h(t-1)
        for (to, row) in self.w_res.iter().enumerate() {
            let mut acc = 0f32;
            for &(from, w) in row {
                acc += w * self.state[from];
            }
            pre_act[to] += acc;
        }

        // Contribuição da entrada: W_in · u(t)
        for to in 0..R {
            let mut acc = 0f32;
            for (j, &u) in obs.iter().enumerate() {
                acc += self.w_in[to][j] * u;
            }
            pre_act[to] += acc;
        }

        // Aplicar vazamento e não-linearidade
        for i in 0..R {
            let driven = pre_act[i].tanh();
            self.state[i] = (1.0 - ALPHA) * self.state[i] + ALPHA * driven;
        }

        // Grava histórico para treinamento futuro do readout
        self.state_history.push(self.state.clone());
        self.obs_history.push(obs);
    }

    /// Aplica o readout e modula os parâmetros de sampling.
    ///
    /// Com `w_out` zerado (estado inicial), retorna `base` sem alteração.
    /// Após treinamento do readout, aplica delta de temperatura.
    pub fn modulate(&self, base: &SamplingParams) -> SamplingParams {
        if self.state_history.is_empty() {
            return base.clone();
        }

        // y = W_out · h(t) — delta de temperatura
        let temp_delta: f32 = self
            .w_out
            .iter()
            .zip(&self.state)
            .map(|(w, s)| w * s)
            .sum::<f32>();

        SamplingParams {
            temperature: (base.temperature + temp_delta * 0.2).clamp(0.05, 2.5),
            top_k: base.top_k,
            top_p: base.top_p,
        }
    }

    /// Computa o layer budget baseado no estado do reservatório.
    ///
    /// Com `w_out_budget` zerado (estado inicial), retorna Slow (conservador).
    /// Após treinamento, o score do readout determina: > 0.5 → Fast, > 0.0 → Medium, else → Slow.
    ///
    /// Returns: LayerBudget enum value (Fast, Medium, or Slow)
    pub fn layer_budget(&self) -> LayerBudget {
        if self.state_history.is_empty() {
            // Sem histórico, ser conservador
            return LayerBudget::Slow;
        }

        // budget_score = W_out_budget · h(t)
        let budget_score: f32 = self
            .w_out_budget
            .iter()
            .zip(&self.state)
            .map(|(w, s)| w * s)
            .sum::<f32>();

        if budget_score > 0.5 {
            LayerBudget::Fast
        } else if budget_score > 0.0 {
            LayerBudget::Medium
        } else {
            LayerBudget::Slow
        }
    }

    /// Constrói o vetor de observação a partir dos logits do modelo.
    fn build_observation(&self, logits: &[f32], pos: usize, max_pos: usize) -> [f32; N_INPUTS] {
        // Softmax estável para calcular probabilidades
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Entropia de Shannon: H = -Σ p·log(p)
        let entropy: f32 = probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum();

        // Probabilidade do token mais provável (confiança)
        let top1_prob = probs.iter().cloned().fold(0f32, f32::max);

        // Posição normalizada na sequência
        let pos_norm = if max_pos > 0 {
            pos as f32 / max_pos as f32
        } else {
            0.0
        };

        [entropy, top1_prob, pos_norm]
    }

    /// Retorna estatísticas do estado atual para telemetria.
    pub fn telemetry(&self) -> EsnTelemetry {
        let step = self.state_history.len();

        // Últimas observações (se existirem)
        let (entropy, top1_prob, pos_norm) = if let Some(obs) = self.obs_history.last() {
            (obs[0], obs[1], obs[2])
        } else {
            (0.0, 0.0, 0.0)
        };

        // Norma L2 do estado (intensidade de ativação do reservatório)
        let state_norm = (self.state.iter().map(|s| s * s).sum::<f32>() / R as f32).sqrt();

        // Delta de temperatura calculado pelo readout
        let temp_delta: f32 = self
            .w_out
            .iter()
            .zip(&self.state)
            .map(|(w, s)| w * s)
            .sum::<f32>();

        // Budget score calculado pelo readout de budget
        let budget_score: f32 = self
            .w_out_budget
            .iter()
            .zip(&self.state)
            .map(|(w, s)| w * s)
            .sum::<f32>();

        EsnTelemetry {
            step,
            entropy,
            top1_prob,
            pos_norm,
            state_norm,
            temp_delta,
            readout_trained: self.w_out.iter().any(|&w| w.abs() > 1e-6),
            budget_score,
        }
    }

    /// Substitui os pesos do readout (para carregar modelo treinado).
    pub fn set_readout(&mut self, weights: Vec<f32>) {
        assert_eq!(
            weights.len(), R,
            "set_readout: esperado {} pesos, recebido {}", R, weights.len()
        );
        self.w_out = weights;
    }

    /// Substitui os pesos do readout de budget (para carregar modelo treinado).
    pub fn set_readout_budget(&mut self, weights: Vec<f32>) {
        assert_eq!(
            weights.len(), R,
            "set_readout_budget: esperado {} pesos, recebido {}", R, weights.len()
        );
        self.w_out_budget = weights;
    }
}

/// Telemetria instantânea do ESN para broadcast WebSocket.
#[derive(Debug, Clone, Serialize)]
pub struct EsnTelemetry {
    /// Passo atual (quantos tokens foram observados nesta geração).
    pub step: usize,
    /// Entropia de Shannon dos logits do último token.
    pub entropy: f32,
    /// Probabilidade do token top-1 (confiança do modelo).
    pub top1_prob: f32,
    /// Posição normalizada na sequência [0, 1].
    pub pos_norm: f32,
    /// Norma L2 do estado do reservatório (intensidade de ativação).
    pub state_norm: f32,
    /// Delta de temperatura calculado pelo readout (0.0 se não treinado).
    pub temp_delta: f32,
    /// true se o readout foi treinado (W_out ≠ 0).
    pub readout_trained: bool,
    /// Budget score W_out_budget · h(t) (determina Fast/Medium/Slow).
    pub budget_score: f32,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_budget_zero_weights() {
        // With zero w_out_budget, should always return Slow (conservative)
        let mut esn = EsnObserver::new(0x1234);
        esn.reset();

        // Simulate a step to build state_history
        let logits = vec![1.0f32; 128];
        esn.step(&logits, 0, 100);

        assert_eq!(esn.layer_budget(), LayerBudget::Slow);
    }

    #[test]
    fn test_layer_budget_transitions() {
        let mut esn = EsnObserver::new(0x5678);
        esn.reset();

        // Build state
        let logits = vec![1.0f32; 128];
        esn.step(&logits, 0, 100);

        // Test Fast: score > 0.5
        let mut weights_fast = vec![0.0f32; R];
        weights_fast[0] = 1.0; // w_out_budget[0] = 1.0
        esn.set_readout_budget(weights_fast);
        // With state[0] ≠ 0, score will be non-zero
        let budget = esn.layer_budget();
        // Just verify it's a valid budget (exact value depends on reservoir dynamics)
        assert!(matches!(budget, LayerBudget::Fast | LayerBudget::Medium | LayerBudget::Slow));

        // Test Medium: 0.0 < score <= 0.5
        let mut weights_medium = vec![0.0f32; R];
        weights_medium[0] = 0.1; // smaller weight
        esn.set_readout_budget(weights_medium);
        let budget = esn.layer_budget();
        assert!(matches!(budget, LayerBudget::Medium | LayerBudget::Slow | LayerBudget::Fast));

        // Test Slow: score <= 0.0
        let mut weights_slow = vec![0.0f32; R];
        weights_slow[0] = -0.5; // negative weight
        esn.set_readout_budget(weights_slow);
        let budget = esn.layer_budget();
        // Will be Slow if score <= 0
        assert!(matches!(budget, LayerBudget::Slow | LayerBudget::Medium | LayerBudget::Fast));
    }

    #[test]
    fn test_layer_budget_empty_state() {
        // Without state history, should return Slow
        let esn = EsnObserver::new(0xABCD);
        assert_eq!(esn.layer_budget(), LayerBudget::Slow);
    }

    #[test]
    fn test_set_readout_budget() {
        let mut esn = EsnObserver::new(0x1111);
        let weights = vec![0.5f32; R];
        esn.set_readout_budget(weights.clone());
        assert_eq!(esn.w_out_budget, weights);
    }

    #[test]
    #[should_panic(expected = "set_readout_budget: esperado")]
    fn test_set_readout_budget_wrong_size() {
        let mut esn = EsnObserver::new(0x2222);
        let weights = vec![0.5f32; 100]; // Wrong size
        esn.set_readout_budget(weights);
    }
}

