// ============================================================================
// generation/mod.rs — Loop autoregressivo com ESN (Fase 1)
// ============================================================================
//
// Fluxo por token:
//   1. model.forward(input)  → logits [vocab_size]
//   2. esn.step(logits, pos) → estado do reservatório evolui (observação)
//   3. params = esn.modulate(params) → ajusta temperatura (no-op se não treinado)
//   4. sample_token(logits, params)  → next_id
//   5. broadcast: GenerationUpdate + EsnTelemetry para frontend
//
// A ESN é reinicializada no início de cada geração via esn.reset().
// O reservatório W_res e W_in são fixos (gerados uma vez na criação).
// Apenas W_out é treinável — começa em zero (Fase 1a).
// ============================================================================

