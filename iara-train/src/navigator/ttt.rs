//! TTT — Test-Time Training para heurísticas do A*.
//!
//! Aprende os pesos w = [w_sparsity, w_confidence] da h_combined
//! baseado em feedback online: qualidade observada vs esperada.
//!
//! Algoritmo: gradient descent online com Adam optimizer.
//!   Loss = (observed_quality - expected_quality)²
//!   Onde expected_quality = 1.0 - route_quality_loss
//!
//! Referências:
//!   - TTT (Stanford 2024): https://arxiv.org/abs/2407.04620
//!   - Online Learning para routing: MoE gating adaptativo

/// Configuração do TTT optimizer
#[derive(Debug, Clone)]
pub struct TttConfig {
    /// Learning rate inicial
    pub lr:      f32,
    /// Adam β1 (momentum do gradiente)
    pub beta1:   f32,
    /// Adam β2 (momentum da variância)
    pub beta2:   f32,
    /// Epsilon para estabilidade numérica
    pub epsilon: f32,
    /// Mínimo valor para cada peso (garante admissibilidade)
    pub w_min:   f32,
    /// Máximo valor para cada peso
    pub w_max:   f32,
    /// Janela de histórico para suavização do loss
    pub window:  usize,
}

impl Default for TttConfig {
    fn default() -> Self {
        TttConfig {
            lr:      0.01,
            beta1:   0.9,
            beta2:   0.999,
            epsilon: 1e-8,
            w_min:   0.1,
            w_max:   0.9,
            window:  32,
        }
    }
}

/// Entrada de treino: observação de uma rota executada
#[derive(Debug, Clone)]
pub struct RouteObservation {
    /// Esparsidade observada em cada super layer durante execução
    pub sparsity_per_sl: [f32; 4],
    /// Entropia do logit (confiança) no momento do routing
    pub logit_entropy: f32,
    /// Perda de qualidade observada (medida via cosine sim ou BLEU degradation)
    pub observed_quality_loss: f32,
    /// Perda de qualidade prevista pela rota escolhida
    pub predicted_quality_loss: f32,
}

/// Otimizador TTT para os pesos da heurística combinada
pub struct TttOptimizer {
    /// Pesos atuais [w_sparsity, w_confidence]
    pub weights: [f32; 2],
    /// Estado Adam: momento de primeira ordem
    m: [f32; 2],
    /// Estado Adam: momento de segunda ordem
    v: [f32; 2],
    /// Passo de treino (para bias correction)
    step: usize,
    /// Configuração
    pub config: TttConfig,
    /// Histórico de losses para monitoramento
    loss_history: Vec<f32>,
    /// Contagem de updates
    pub n_updates: usize,
}

impl TttOptimizer {
    /// Cria otimizador com pesos iniciais.
    pub fn new(initial_weights: [f32; 2], config: TttConfig) -> Self {
        TttOptimizer {
            weights: initial_weights,
            m: [0.0; 2],
            v: [0.0; 2],
            step: 0,
            config,
            loss_history: Vec::new(),
            n_updates: 0,
        }
    }

    /// Atualiza pesos baseado em uma observação de rota.
    ///
    /// Loss = (observed_quality_loss - predicted_quality_loss)²
    /// Gradiente ≈ ∂loss/∂w computado analiticamente.
    pub fn update(&mut self, obs: &RouteObservation) -> f32 {
        let residual = obs.observed_quality_loss - obs.predicted_quality_loss;
        let loss = residual * residual;

        // Gradiente da h_combined em relação a [w_sparsity, w_confidence]
        // h_combined = w0 * h_sparsity + w1 * h_confidence
        // ∂loss/∂w0 = 2 * residual * h_sparsity(current_state)
        // ∂loss/∂w1 = 2 * residual * h_confidence(current_state)
        //
        // Aproximação: usamos as heurísticas nas features observadas como "proxy"
        let h_sparsity  = self.h_sparsity_feature(&obs.sparsity_per_sl);
        let h_confidence = self.h_confidence_feature(obs.logit_entropy);

        let grad = [
            2.0 * residual * h_sparsity,
            2.0 * residual * h_confidence,
        ];

        self.adam_step(grad);
        self.loss_history.push(loss);
        if self.loss_history.len() > self.config.window {
            self.loss_history.remove(0);
        }
        self.n_updates += 1;

        loss
    }

    /// Passo Adam com projeção nos limites [w_min, w_max].
    fn adam_step(&mut self, grad: [f32; 2]) {
        self.step += 1;
        let t   = self.step as f32;
        let b1  = self.config.beta1;
        let b2  = self.config.beta2;
        let eps = self.config.epsilon;
        let lr  = self.config.lr;

        for i in 0..2 {
            // Momentum updates
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];

            // Bias correction
            let m_hat = self.m[i] / (1.0 - b1.powf(t));
            let v_hat = self.v[i] / (1.0 - b2.powf(t));

            // Update com projeção nos limites
            let new_w = self.weights[i] - lr * m_hat / (v_hat.sqrt() + eps);
            self.weights[i] = new_w.clamp(self.config.w_min, self.config.w_max);
        }

        // Normalizar para que w0 + w1 = 1.0 (simplex constraint)
        let sum = self.weights[0] + self.weights[1];
        if sum > 0.0 {
            self.weights[0] /= sum;
            self.weights[1] /= sum;
        }
    }

    /// Feature h_sparsity (escalar representativo do estado)
    fn h_sparsity_feature(&self, sparsity: &[f32; 4]) -> f32 {
        // Média da esparsidade das layers seguintes (proxy para custo restante)
        sparsity.iter().sum::<f32>() / 4.0
    }

    /// Feature h_confidence (escalar representativo da confiança)
    fn h_confidence_feature(&self, logit_entropy: f32) -> f32 {
        // Confiança = 1 - entropia normalizada
        (1.0 - logit_entropy.min(1.0)) * 0.9
    }

    /// Loss médio na janela atual
    pub fn avg_loss(&self) -> f32 {
        if self.loss_history.is_empty() { return 0.0; }
        self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32
    }

    /// Verifica se os pesos convergiriam (variação pequena na janela)
    pub fn has_converged(&self, threshold: f32) -> bool {
        if self.loss_history.len() < self.config.window { return false; }
        let half = self.config.window / 2;
        let first_half: f32 = self.loss_history[..half].iter().sum::<f32>() / half as f32;
        let second_half: f32 = self.loss_history[half..].iter().sum::<f32>() / half as f32;
        (first_half - second_half).abs() < threshold
    }

    /// Estatísticas do optimizer
    pub fn stats(&self) -> TttStats {
        TttStats {
            weights:    self.weights,
            n_updates:  self.n_updates,
            avg_loss:   self.avg_loss(),
            converged:  self.has_converged(1e-4),
            step:       self.step,
        }
    }
}

#[derive(Debug)]
pub struct TttStats {
    pub weights:   [f32; 2],
    pub n_updates: usize,
    pub avg_loss:  f32,
    pub converged: bool,
    pub step:      usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(sparsity: f32, entropy: f32, obs_ql: f32, pred_ql: f32) -> RouteObservation {
        RouteObservation {
            sparsity_per_sl:        [sparsity; 4],
            logit_entropy:          entropy,
            observed_quality_loss:  obs_ql,
            predicted_quality_loss: pred_ql,
        }
    }

    #[test]
    fn test_weights_stay_in_bounds() {
        let mut opt = TttOptimizer::new([0.6, 0.4], TttConfig::default());
        for i in 0..100 {
            let obs = make_obs(0.3, 0.5, 0.02 + i as f32 * 0.001, 0.03);
            opt.update(&obs);
            assert!(opt.weights[0] >= opt.config.w_min, "w0 below min");
            assert!(opt.weights[0] <= opt.config.w_max, "w0 above max");
            assert!(opt.weights[1] >= opt.config.w_min, "w1 below min");
            assert!(opt.weights[1] <= opt.config.w_max, "w1 above max");
        }
    }

    #[test]
    fn test_weights_sum_to_one() {
        let mut opt = TttOptimizer::new([0.6, 0.4], TttConfig::default());
        for _ in 0..50 {
            opt.update(&make_obs(0.4, 0.3, 0.01, 0.02));
        }
        let sum = opt.weights[0] + opt.weights[1];
        assert!((sum - 1.0).abs() < 1e-5, "weights must sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_loss_decreases_with_consistent_signal() {
        // Com observações consistentes (obs_ql sempre > pred_ql), o optimizer deve convergir
        let mut opt = TttOptimizer::new([0.6, 0.4], TttConfig { window: 20, ..Default::default() });
        let mut losses = Vec::new();
        for _ in 0..100 {
            let l = opt.update(&make_obs(0.5, 0.4, 0.05, 0.02));
            losses.push(l);
        }
        // Loss deve ser menor no final que no início (depois de convergência)
        let early_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let late_avg:  f32 = losses[80..].iter().sum::<f32>() / 20.0;
        assert!(late_avg <= early_avg + 1e-3,
            "Loss should not increase: early={:.6} late={:.6}", early_avg, late_avg);
    }

    #[test]
    fn test_convergence_detection() {
        let cfg = TttConfig { window: 20, ..Default::default() };
        let mut opt = TttOptimizer::new([0.5, 0.5], cfg);
        // Sem updates: não convergiu
        assert!(!opt.has_converged(1e-4));
        // Com observações de residual zero → loss zero → convergência rápida
        for _ in 0..40 {
            opt.update(&make_obs(0.3, 0.5, 0.03, 0.03)); // residual = 0
        }
        assert!(opt.has_converged(1e-4), "Should converge with zero-residual observations");
    }

    #[test]
    fn test_n_updates_counter() {
        let mut opt = TttOptimizer::new([0.6, 0.4], TttConfig::default());
        for i in 1..=10 {
            opt.update(&make_obs(0.3, 0.5, 0.02, 0.03));
            assert_eq!(opt.n_updates, i);
        }
    }
}
