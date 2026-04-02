---
name: IARA — Bugs Críticos Identificados (2026-04-02)
description: Bugs que impedem geração básica de funcionar via WebSocket e falha silenciosa no frontend
type: project
---

## Bug Principal: Pipeline WebSocket tem erros silenciosos

**Arquivo:** `iara-engine/src/main.rs` — função `run_engine_loop()`

### Bug #1 — Prefill silencioso (linha ~206-210)
```rust
if let Err(e) = m.forward(&[tok]) {
    log::error!("Prefill error: {}", e);
    break;  // ← sem notificar cliente WS, sem propagar erro
}
```

### Bug #2 — Decode loop silencioso (linha ~230)
```rust
Err(e) => { log::error!("Forward error: {}", e); break; }
// ← gera_step nunca inicializado, potencial panic
```

### Bug #3 — Frontend não envia via HTTP (legacy_updates.js:240-249)
```javascript
sendBtn.addEventListener('click', () => {
    wsManager.resonate(text);  // ← ERRADO, deveria ser api.generate()
});
```

### Bug #4 — store.generating nunca resetado no caminho WS
O `store.generating` só é resetado no fluxo HTTP. Via WS, UI trava em "Gerando..." infinitamente.

## Contraste: handlers.rs (HTTP) está CORRETO
O endpoint POST /api/generate propaga erros corretamente com `?`, retorna HTTP 500 com mensagem. O problema é que o frontend não usa esse caminho.

**Why:** Confusão histórica entre dois protocolos (HTTP vs WS) durante desenvolvimento incremental. Frontend evoluiu para WS mas error handling não acompanhou.

**How to apply:** Ao trabalhar no projeto, priorizar corrigir run_engine_loop para propagar erros via frame_tx, e corrigir frontend para resetar store.generating ao receber status final.
