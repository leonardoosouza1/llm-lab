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

## DNA-V5 + Kernel Optimization (2026-05-15)

Qwen3-4B runs in 3 formats on the RX 6750 XT:

| Format | Disk | VRAM | Speed | Executor |
|---|---|---|---|---|
| safetensors | 7.6 GB | 9.1 GB | 14 t/s | iara-engine native |
| GGUF Q4_K_M | 2.4 GB | 2.5 GB | 88 t/s | llama.cpp via FFI |
| DNA-V5 | 3.1 GB | 3.6 GB | 77 t/s | iara-engine native |

### Two inference backends

- `.gguf` → `LlamaEngine` (llama.cpp via `llama-sys` FFI)
- `.dna` / `.iara` / safetensors → `CausalLM` (iara-engine native HIP kernels)
- `.dna` takes precedence over `.gguf`; `IARA_NO_LLAMA=1` forces native for GGUF too.

### DNA-V5 format

`.dna` magic `DNA4` + `version=5`. New `TBLK` chunk stores ggml block-quantized
tensors (Q4_K/Q6_K/Q8_0) directly — no SVD. `DnaTensor::Block` → `LayerWeight::Q4K/Q6K`
reuses the GGUF-path kernels. Converter: `scripts/dna/gguf_to_dna5.py`.

Why not SVD (DNA-V4): Qwen2.5-1.5B V_proj is [256,1536] — only 256 SVs, so rank-256
SVD was lossless by luck of the narrow shape. Qwen3-4B V_proj is [1024,2560] — 1024
SVs, rank-256 keeps only ~50% energy → 42% drift → garbage. Block quantization keeps
full rank at 4-5 bits/weight.

### Kernel work — closing the gap to llama.cpp

DNA path was 37 t/s. Profiling (`IARA_SECTION_PROF=1`): MLP 44%, and the synthetic
Q6_K `lm_head` (vocab 151936) was a hidden GEMV bottleneck.

Fix: ported llama.cpp's `vec_dot_q6_K_q8_1` (mmvq) to `hip-ffi/kernels/mmvq_llama.hip`
as `mmvq_q6k_llama` (dp4a `sdot4`, warp-reduce, 1 warp/row). Joins the existing
`mmvq_q4k_llama`. Backend dispatches via `use_mmvq_llama` (default ON).

Result: 37 → 77 t/s (2.1×). Gap to GGUF: 13% slower.

RDNA2 note: llama.cpp `calc_nwarps` uses nwarps=1 for Q4_K/Q6_K at ncols_dst=1 on
RDNA2 (nwarps=8 only on RDNA3/4). Our 1-warp-per-row kernels already match. The
residual 13% is structural — separate kernel launches + per-section syncs vs
llama.cpp's fused graph. Closing further needs a fused gate+up mmvq kernel
(`has_fusion` path) or attention-decode tuning.

`IARA_FUSE_MLP=1` exists but is slower (62 t/s) — the old `fused_gate_up_silu`
kernel predates the mmvq port; keep it OFF.
