// ============================================================================
// physics.rs — Neurônio CfC: Closed-form Continuous (Architecture MIT)
// ============================================================================
// Implementação da aproximação de tempo contínuo estável e rápida.
// x(t + dt) = x(t) * exp(-dt/tau) + (1 - exp(-dt/tau)) * f(I, w)
//
// Esta arquitetura resolve o problema da instabilidade de Euler e permite
// que a rede se comporte como um sistema de equações diferenciais líquidas.
// ============================================================================

/// Constantes de escala para o modelo CfC
const SCALE_TAU_MAX: f32 = 0.25; // Decaimento máximo (memória longa)
const SCALE_TAU_MIN: f32 = 0.005; // Decaimento mínimo (resposta rápida)

/// Neurônio CfC — A evolução do Nó Líquido (Inspirado em Hasani et al., MIT)
#[derive(Debug, Clone)]
pub struct NeuronioCfC {
    pub x: f32,           // Estado atual (Ativação)
    pub tau: f32,         // Constante de tempo líquida (LTC)
    pub weight: f32,      // Peso sináptico original da LLM
    pub has_spiked: bool, // Compatibilidade com interface de eventos
}

impl NeuronioCfC {
    /// Cria um neurônio CfC a partir de um peso de LLM
    pub fn from_weight(w: f32) -> Self {
        let w = if w.is_finite() { w } else { 0.0 };
        let abs_w = w.abs();

        // LTC: Pesos pequenos geram constantes de tempo grandes (memória)
        // Pesos grandes geram constantes pequenas (mudança rápida)
        let tau = (1.0 / (abs_w * 10.0 + 1.0)).clamp(SCALE_TAU_MIN, SCALE_TAU_MAX);

        NeuronioCfC {
            x: 0.0,
            tau,
            weight: w,
            has_spiked: false,
        }
    }

    /// Atualização via Solução de Forma Fechada (Closed-form)
    /// 
    /// x(t+dt) = x(t)*decay + (1-decay)*target
    #[inline]
    pub fn tick(&mut self, i_input: f32, dt: f32) {
        self.has_spiked = false;

        // Alvo do neurônio baseado no estímulo e no seu peso interno
        // Usamos tanh para manter a ativação no range [-1, 1] e garantir estabilidade
        let target = (self.weight * i_input).tanh();

        // Coeficiente de decaimento exato para o intervalo dt
        let decay = (-dt / self.tau).exp();

        // O novo estado é uma interpolação exponencial entre o estado atual e o alvo
        let new_x = self.x * decay + (1.0 - decay) * target;

        // Detecção de "evento" (spike conceitual) para compatibilidade visual
        // Se a mudança for brusca, consideramos um disparo
        if (new_x - self.x).abs() > 0.5 {
            self.has_spiked = true;
        }

        self.x = new_x;
    }

    /// Retorna a ativação para visualização
    pub fn posicao_x(&self) -> f32 {
        self.x
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.has_spiked = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfc_convergence() {
        let mut n = NeuronioCfC::from_weight(1.0);
        let dt = 0.1;

        // Estímulo constante 1.0 -> target deve ser tanh(1*1) = 0.7615
        for _ in 0..100 {
            n.tick(1.0, dt);
        }
        assert!((n.x - 0.7615).abs() < 0.01);
    }

    #[test]
    fn test_cfc_decay() {
        let mut n = NeuronioCfC::from_weight(1.0);
        n.x = 1.0;
        let dt = 0.1;

        // Sem estímulo -> target = tanh(0) = 0. Deve decair para zero
        for _ in 0..100 {
            n.tick(0.0, dt);
        }
        assert!(n.x.abs() < 0.01);
    }
}
