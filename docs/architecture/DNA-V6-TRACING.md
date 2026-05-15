# DNA-V6 + Engine + Tracing — Integrated Design

Status: design approved 2026-05-15. Implementation in 3 phases.

## Integrating principle

The three pieces are designed as one system:

- **DNA-V6** carries metadata (layer index, importance, readable names, lineage).
- **Engine** emits structured trace events that reference DNA tensor names.
- **Tracing** is a first-class subsystem, not env-var hacks — and it spans both
  `byte-language-lab` (brain, byte-level) and `llm-lab` (engine, token-level),
  joined into one replay timeline.

## 1. DNA-V6 file format

```
[HEADER "DNA6"]
  version: u32
  lineage block:
    origin_model:   str   (e.g. "Qwen3-4B")
    architecture:   str   (e.g. "qwen3")
    orig_dtype:     str   (e.g. "Q4_K_M")
    converted_utc:  i64
    source_models:  [] (reserved — empty now; multi-model fusion later)
  layer_index: [ {layer_id, file_offset, n_tensors} ]   ← enables skip + streaming
[LAYER 0]   attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate, up, down
[LAYER 1]   ...                          ← all tensors of a layer contiguous
...                                         on disk = locality
[GLOBAL]    embed_tokens, final_norm, lm_head
[FOOTER: JSON]
  importance_score per layer (gravity metric)
  spectrum stats per tensor (optional)
  trace_hints: which tensors to observe by default
```

Wins over DNA-V5 (flat chunk list):
- per-layer grouping → contiguous reads, streaming load, layer skip
- layer_index → O(1) seek to any layer without scanning
- lineage → provenance; reserved `source_models[]` for multi-model fusion
- importance metadata → trace + future pruning/routing

DNA-V5 stays loadable (version field dispatches). V6 is the evolution.

## 2. Engine — tracing as first-class

Replaces `IARA_DBG_HIDDEN` / `IARA_SECTION_PROF` env hacks.

```rust
enum TraceLevel { Off, Perf, Layer, Byte, Full }

struct TraceEvent {
    ts_us:    u64,
    source:   Source,         // Brain | Engine
    trace_id: u64,            // shared across brain+engine for one query
    seq:      u32,            // token index
    stage:    Stage,          // Tokenize | Embed | Layer{n, sub} | Logits | Sample
    layer:    Option<u32>,
    stats:    StageStats,     // l2, top_k:[(idx,val)], timing_us, nan_flag
    full:     Option<Vec<f32>>, // only with --trace-full
}

trait TraceSink {
    fn emit(&self, ev: TraceEvent);
}
// impls: RingSink (live WS broadcast), FileSink (.trace JSONL append)
```

`TraceLevel::Off` → emit compiles to nothing measurable. Each forward stage
calls `sink.emit()` guarded by the active level.

## 3. Tracing — 3 toggleable levels

| Level | Captures |
|---|---|
| `Perf`  | timing per kernel / per section |
| `Layer` | per layer: norm energy, attention pattern, MLP activation rate, hot neurons |
| `Byte`  | input bytes → BPE merges → token IDs; embed lookup; logits → sampled token; "how the decision was reached" |

Levels stack (Full = all). Each independently flaggable.

### Cross-project: brain ⇄ engine

`byte-language-lab` (brain, byte-level HDC) and `llm-lab` (engine, token-level)
share the `.trace` schema and a `trace_id`. One query produces brain events
(byte processing) and engine events (token processing); the replay tool stitches
them into a single timeline showing the byte ⇄ token transition.

## 4. `.trace` file — hybrid detail

- **Default**: compact stats per stage (L2, top-k, timing, nan) — ~KB/token,
  reconstructs the "path".
- **`--trace-full`**: also dumps full activation tensors per layer — ~MB/token,
  for deep offline bisection.

Format: JSONL (one `TraceEvent` per line) — greppable, streamable, replayable.

## Phases

| Phase | Deliverable |
|---|---|
| **F1 — Tracing** | `TraceLevel` + `TraceEvent` + `TraceSink`; wire all forward stages; `.trace` JSONL + replay tool; WS live to existing `trace-ui` |
| **F2 — DNA-V6** | rich header + per-layer index + lineage; `gguf_to_dna6` converter; engine `load_dna_v6` |
| **F3 — Perf**   | fused gate+up mmvq kernel (`has_fusion`) — close the 13% gap to llama.cpp |

Execution order chosen by user: introspection (F1) first.
