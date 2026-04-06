// ============================================================================
// experimental/ — Módulos exploratórios e pesquisa (não integrados ao pipeline)
// ============================================================================
// Estes módulos são pesquisa/prova-de-conceito e NÃO fazem parte do caminho
// de inferência principal. Nenhum código em api/, qwen/, model.rs, kv_cache.rs
// ou main.rs depende deles.
//
// Para o path de inferência ativo, veja:
//   src/model.rs        — CausalLM, carga de pesos, forward GPU
//   src/inference/      — forward_traced, sampling
//   src/model/kv_cache.rs — GpuKvCache
//   src/api/            — HTTP handlers
//   src/transport.rs    — servidor WS
// ============================================================================

pub mod actuator;
pub mod gpu_orchestrator;
pub mod homeostasis;
pub mod inference_stub;
pub mod liquid_brain;
pub mod liquid_network;
pub mod nlm_orchestrator;
pub mod orchestrator;
pub mod physics;
pub mod physics_gpu;
pub mod pipeline;
pub mod prompt;
pub mod proprioception;
pub mod simulation;
pub mod iara_physics;
