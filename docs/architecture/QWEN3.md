# Qwen3 Architecture Notes

Cross-checked against `models/qwen3-4b/config.json` and `tokenizer_config.json`.

## Diff vs Qwen2.5

| Param | Qwen2.5-1.5B | Qwen3-4B | Effect |
|---|---|---|---|
| `num_hidden_layers` | 28 | 36 | +29% depth |
| `hidden_size` | 1536 | 2560 | +67% |
| `intermediate_size` (FFN) | 8960 | 9728 | +9% |
| `num_attention_heads` | 12 | 32 | 2.67× |
| `num_key_value_heads` | 2 | 8 | 4× KV |
| GQA ratio (Q:KV) | 6:1 | 4:1 | Less aggressive |
| `head_dim` | 128 (1536/12) | 128 (explicit) | Same |
| `tie_word_embeddings` | false | **true** | Saves ~780 MB (vocab×hidden×2 bytes) |
| `attention_bias` | true (Q/K/V bias) | **false** | Fewer params |
| `rope_theta` | 1000000 | 1000000 | Long-ctx aware |
| `max_position_embeddings` | 32768 | 40960 | +25% ctx |
| **QK norm** | no | **yes** | RMSNorm Q and K per-head before RoPE |
| **think mode** | no | **yes** | `<think>` (151667) / `</think>` (151668) tokens |
| `eos_token_id` | 151643 | [151645, 151643] | Multi-EOS |

## Architectural Insights

### 1. QK Normalization
Qwen3 applies RMSNorm to Q and K **per-head** before RoPE.

Q: `[n_tokens, num_heads, head_dim]` → norm each head's `head_dim` slice.
K: `[n_tokens, num_kv_heads, head_dim]` → same.

`iara-engine/src/model/mod.rs:3281` already implements this — kernel treats each head as a "token" so block size = `n_tokens * num_heads`.

Why it matters: stabilizes long-context attention. Tied to longer `max_position_embeddings`.

### 2. Tied Word Embeddings
`lm_head.weight == embed_tokens.weight`. Saves disk + VRAM ~780 MB at vocab=151936, hidden=2560.

`iara-engine/src/model/architecture.rs:195` reads from config.json.
`iara-train/src/bin/dna_compress.rs` writes `lm_head.weight` as a duplicate TRAW when tied (line ~177 logic), so loader doesn't need a tie resolver.

### 3. No QKV Bias
`attention_bias=false` → fewer params + simpler kernels. `iara-engine/src/model/mod.rs:603` checks `Option<bias>` and skips.

### 4. Reasoning / Thinking Tokens
- `<think>...</think>` block emitted BEFORE final answer when `enable_thinking=true` (default).
- `chat_template` injects empty `<think>\n\n</think>\n\n` if `enable_thinking=false` — disables reasoning but stays format-compatible.

For classification/labeling tasks: **always disable thinking** to save tokens.

### 5. GQA 4:1 (vs 6:1 in Qwen2.5)
Less aggressive grouping → more KV cache per layer but better quality.
KV cache size = `n_layers × 2 × num_kv_heads × head_dim × seq_len × 2 bytes`
For Qwen3-4B at 4096 ctx: `36 × 2 × 8 × 128 × 4096 × 2 = 604 MB`

## Insights for byte-language-lab

1. **QK-norm pattern** could stabilize the curator's HV→logits projection. Currently raw softmax of dot products; adding per-domain RMSNorm before softmax may reduce class collapse.

2. **Tied embeddings** is essentially `lm_head = embed_tokens.T`. The brain's expert centroids could be tied to a single embedding matrix → less drift between input encoding and output decoding.

3. **`<think>` tokens** are explicit reasoning brackets. Brain's pipeline already has implicit stages; adopting `<think>` markers in WAL would make replay/audit cleaner.

4. **GQA**: brain's static MoE has 10 experts but routes 1-of-10. Could explore 8-of-32 dense queries (Q heads) sharing 8 expert centroids (KV heads) for richer routing.

## Loading Notes

- Sharded safetensors (`model-0000{1,2,3}-of-00003.safetensors` + `index.json`) — `iara-engine` and now `dna_compress` (extended 2026-05-14) both support this via `weight_map` parsing.
- VRAM at bf16: ~8 GB. On RX 6750 XT (12 GB) leaves ~4 GB for KV cache + scratch.
- Initial token throughput: **14 tok/s** at 36L-2560h on ROCm/HIP (vs ~40 tok/s for Qwen2.5-1.5B).
