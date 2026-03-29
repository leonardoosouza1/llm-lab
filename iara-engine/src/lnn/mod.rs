// ============================================================================
// lnn/ — Liquid Neural Network: módulos de aprendizado dinâmico
// ============================================================================
//
// Fase 1: ESN (Echo State Network) como observador passivo do stream LLM.
//   → Não toca nos pesos do modelo; lê saídas, modula parâmetros de sampling.
//
// Roadmap:
//   Fase 1 — ESN observador (este módulo)
//   Fase 2 — Treinamento online do readout W_out por reinforcement signal
//   Fase 3 — LNN como memória episódica (estado persistente entre gerações)
//   Fase 4 — Dual pipeline: LNN + LLM em paralelo via UMA zero-copy
// ============================================================================

pub mod esn;

pub use esn::{EsnObserver, EsnTelemetry};
