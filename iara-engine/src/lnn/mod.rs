// ============================================================================
// lnn/ — Liquid Neural Network: módulos de aprendizado dinâmico
// ============================================================================
//
// Fase 1: ESN (Echo State Network) como observador passivo do stream LLM.
//   → Não toca nos pesos do modelo; lê saídas, modula parâmetros de sampling.
//
// Fase 2: Gravity System + CfC Layer para correção de quantização.
//   → Identifica tokens críticos (alta gravidade semântica)
//   → Aplica correção adaptativa via neurônio líquido (CfC)
//   → Integra-se ao loop de geração entre as camadas do transformer
//
// Roadmap:
//   Fase 1 — ESN observador (implementado)
//   Fase 2 — Gravity + CfC (este módulo)
//   Fase 3 — Treinamento online do readout W_out por reinforcement signal
//   Fase 4 — LNN como memória episódica (estado persistente entre gerações)
//   Fase 5 — Dual pipeline: LNN + LLM em paralelo via UMA zero-copy
// ============================================================================

pub mod esn;
pub mod gravity;
pub mod cfc;

#[allow(unused_imports)]
pub use esn::EsnObserver;
#[allow(unused_imports)]
pub use gravity::{GravitySystem, GravityConfig, GravityResult, GravityTelemetry};
#[allow(unused_imports)]
pub use cfc::{CfcLayer, CfcConfig, CfcTelemetry};
