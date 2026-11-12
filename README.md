# llm-lab: IARA Engine

**An LLM inference engine written from scratch in Rust, with hand-written HIP kernels. It runs Qwen on a consumer AMD
card (RX 6750 XT, gfx1031) and streams per-layer telemetry while it generates.**

It does not use PyTorch, candle or `llama-cpp-python`. Weights (`safetensors`, `GGUF`, or the in-house `.dna` / `.iara`
containers) are loaded straight into VRAM through a `ComputeBackend` trait with two implementations, ROCm/HIP and
OpenCL. The engine runs the transformer forward pass, exposes a native API plus an OpenAI-compatible `/v1` surface, and
broadcasts layer-by-layer telemetry over WebSocket.

## Highlights

All numbers were measured on one AMD RX 6750 XT (12 GB, RDNA2). "Reported" means the number comes from a log line that
has not been re-measured since.

- **Qwen3-4B, native path: 77 tok/s** in the DNA-V5 block-quant format (3.6 GB VRAM). That is 5.5x the engine's own
  bf16 safetensors path (14 tok/s) and 13% slower than llama.cpp on GGUF Q4_K_M (88 tok/s) on the same card.
  [docs/architecture/QWEN3.md](docs/architecture/QWEN3.md)
- **One kernel port gave 2.1x (37 to 77 tok/s).** Profiling showed MLP at 44% of the time, plus a hidden GEMV
  bottleneck in the Q6_K `lm_head` (vocab 151,936). Porting llama.cpp's `vec_dot_q6_K_q8_1` (dp4a, one warp per row)
  to HIP fixed it. The kernel is in `hip-ffi/kernels/mmvq_llama.hip`.
- **Qwen2.5-1.5B on the native ROCm path: 69.3 tok/s decode** (14 tokens in 202 ms, 751 ms end to end for a short
  Portuguese chat reply).
- **`iara-semente`: a byte-level model trained from scratch in Rust/HIP**, with 19 kernels and the backward pass
  written by hand (no autograd). It runs at 116k tokens/s. Analytic gradients agree with finite differences of an f64
  host loss to between 1e-6 and 7.9e-3 (worst case).
- **Multi-digit arithmetic, 300 problems: Qwen2.5-1.5B Q4_K_M gets 62.0% and a digit-by-digit byte solver gets 94.0%.**
  The byte solver is exact by construction for + and −, so this measures where the LLM fails (0% on 3+ digit
  multiplication). It is not a model-vs-model contest. [BENCH_RESULTS.md](BENCH_RESULTS.md)

## How it works

```
   safetensors / GGUF / .dna / .iara
                  │
                  ▼
        ┌────────────────────┐    HTTP :3000  /api/*, /v1/* (OpenAI), /acp/v1/*
        │    iara-engine     │    WS   :8080  per-token, per-layer telemetry
        │  (Rust, tokio/axum)│    Web  :3000  dashboard (frontend/)
        └────────────────────┘
             │            │
        ROCm/HIP        OpenCL           backend chosen at runtime behind ComputeBackend
     hipBLAS + .hip    JIT C kernels
     kernels; GGUF via llama.cpp FFI (llama-sys)
```

- **Generation flow:** tokenize (ChatML) → reset KV cache → batched prefill → per-token `forward_traced` (RMSNorm →
  QKV → RoPE → KV cache → attention → MLP, with telemetry per layer) → sample → JSON response + WebSocket broadcast.
- **Telemetry:** each forward pass emits a `LayerTrace` per layer with attention energy, MLP energy, residual norm,
  the top-10 neurons by magnitude, and timing.
- **Formats:**
  - `.gguf` goes to llama.cpp through FFI.
  - `.dna`, `.iara` and safetensors go to the native HIP path.
  - DNA-V5 stores ggml block-quantized tensors (Q4_K/Q6_K/Q8_0) in a TLV container and reuses the GGUF-path kernels.
  - DNA-V4 (SVD-factorized) is kept for reference only; see "What didn't work".
- **Device-side quantization kernels:** Q4_K, Q6_K, FP8 (e4m3), and 2-bit ternary (BitNet-style).

| Crate | Role |
|---|---|
| `iara-engine` | server, inference pipeline, telemetry, OpenAI surface |
| `compute-core` | `ComputeBackend` trait and parameter structs (no hardware code) |
| `rocm-backend` / `hip-ffi` | ROCm/HIP backend, HIP + hipBLAS bindings, `.hip` kernels |
| `opencl-backend` | OpenCL backend (JIT C kernels + naive GEMM) |
| `ck-backend` | Composable Kernel backend stub |
| `llama-sys` | thin llama.cpp FFI for the GGUF path |
| `iara-train` | LoRA fine-tuning, DNA compression, ternary MoE bench example |
| `iara-observe`, `iara-explore` | observability, exploration binaries (embedding-only "0-layer" probes, measurements) |
| `iara-byte-bench` | accuracy harness (arithmetic bench above) |
| `iara-semente` | from-scratch byte model, Rust/HIP, hand-written backward (experimental) |

## Results

| What | Number | Baseline / note | Status |
|---|---|---|---|
| Qwen3-4B decode, DNA-V5 native | 77 tok/s, 3.6 GB VRAM | GGUF Q4_K_M via llama.cpp 88 tok/s; safetensors native 14 tok/s | measured |
| Q6_K mmvq kernel port | 37 → 77 tok/s | same model and format | measured |
| Qwen2.5-1.5B decode, native ROCm | 69.3 tok/s | 14 tokens / 202 ms | measured |
| `iara-semente` byte model | 116k tokens/s; 1.44M equivalent weights in 268 KB | gradient check vs f64 finite differences, worst 7.9e-3 | measured |
| 30B MoE (Qwen3-30B-A3B) experts in ternary, FFN only, Rust + `gemv_ternary` | 15.23 ms/token for 48 FFN layers, 7.2 GB (4.0x smaller than FP8) | 238x the Python version; per-layer cosine 0.873, end-to-end quality not measured | partial |
| wave64 vs wave32 GEMV on RDNA2 | 88.4 vs 68 tok/s | memory-bound GEMV; wave64 kernels compiled as a separate `.hsaco` | reported |
| Arithmetic, 300 problems | Qwen 62.0% vs byte solver 94.0% | byte solver exact by construction | reported (repo file) |

## What didn't work (and why)

- **SVD compression (DNA-V4) does not shrink Qwen.** Truncated SVD at 95% energy took Qwen2.5-1.5B from 2620 MB to
  2550 MB (1.03x), because the MLP needs more than 80% of full rank. On Qwen3-4B, rank-256 `v_proj` keeps only about
  50% of the energy and the output drifts 42% (garbage). Rank 256 had been lossless on Qwen2.5-1.5B only because that
  `v_proj` is [256, 1536]. DNA-V5 (block quantization, full rank) replaced it.
- **The remaining 13% gap to llama.cpp comes from launch overhead.** An older profile measured GEMV effective bandwidth
  at 73 GB/s, about 17% of peak, with roughly 395 kernel launches per token (reported). llama.cpp runs a fused graph.
  The existing fused gate+up kernel (`IARA_FUSE_MLP=1`) is slower (62 tok/s) because it predates the mmvq port.
- **The native backend is not thread-safe.** Two to four concurrent workers hit HIP error 700 (illegal memory access)
  through races on shared buffers, and two processes on the same GPU crash. Requests are serialized behind one mutex.
- **Flash-decode kernels assume `head_dim = 128`** (4 elements × 32 lanes). Models with `head_dim = 64` (Qwen2.5-0.5B)
  give wrong output without any error. The context limit on that path is 1024.
- **"Pointer weights" (`iara-semente`) lost to a dense control.** Storing each weight as a learned address into a
  shared bank lost 17% perplexity at equal disk size and 25% at equal parameter count. The kernels and gradient check
  are correct. The idea is what failed.

## Hardware and stack

- AMD RX 6750 XT, 12 GB, Navi22 **gfx1031**, run as gfx1030 with `HSA_OVERRIDE_GFX_VERSION=10.3.0`. The card is not
  officially supported by ROCm. Kernels are wave32 except the dedicated wave64 GEMV set.
- ROCm 7.2.1, hipBLAS, Rust 2021, tokio/axum. An Intel Iris Plus (Gen 11) iGPU is supported through OpenCL.
- RDNA2 has no matrix cores and no fast bf16 path, so the engine keeps fp16/fp32 on the device and uses dp4a for
  quantized dot products.

## Build and run

The repo ships no weights (`models/` is git-ignored).

```bash
git clone https://github.com/leonardoosouza1/llm-lab.git
cd llm-lab
mkdir -p models
# single-file GGUF (llama.cpp path):
wget -c "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
     -O models/qwen2.5-1.5b.gguf
# or safetensors for the native path: config.json, tokenizer.json, model.safetensors in models/<name>/

export ROCM_PATH=/opt/rocm             # your ROCm install
./scripts/run_iara.sh release          # builds with --features rocm, serves :3000 (HTTP) and :8080 (WS)
./scripts/run_igpu.sh release          # OpenCL instead
```

A GPU feature is required. A plain `cargo build --release` produces a binary that exits with "no compute backend
available", so build with `--features rocm` or `--features opencl`.

```bash
curl -s localhost:3000/api/load-model -H 'content-type: application/json' -d '{"name":"qwen2.5-1.5b"}'
curl -s localhost:3000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen2.5-1.5b", "messages": [{"role":"user","content":"Hello!"}], "max_tokens": 60}'
```

| Method | Route | Purpose |
|---|---|---|
| GET | `/health`, `/ping` | status, VRAM |
| GET | `/api/models` | models found in `models/` |
| POST | `/api/load-model`, `/api/unload-model` | load (async, 202) / free VRAM |
| POST | `/api/generate` | native generation (temperature, top-k/p, optional layer `budget`) |
| POST | `/api/trace` | single-token trace with full per-layer telemetry |
| POST / GET | `/v1/chat/completions`, `/v1/models` | OpenAI-compatible (non-streaming) |
| * | `/acp/v1/*` | Agent Communication Protocol (runs, agents, events) |
| WS | `:8080` | live telemetry stream |

## Status

- Working: native and GGUF inference, the telemetry dashboard, both backends, the OpenAI `/v1` surface (no SSE
  streaming, no tool calls or vision).
- Limits: one model loaded at a time, requests serialized, the OpenCL GEMM is a naive K-loop, and VLMs load only the
  text decoder.
- Next: a fused gate+up mmvq kernel to close the remaining 13%, flash decode for `head_dim = 64`, and per-request
  buffers for concurrency.
- New experiments (draft-verify, layer-parallel decode) live in the sibling Rust workspace `iara-lab`, which reuses
  `hip-ffi` from this repo.

## License

[MIT](LICENSE), by leonardoosouza1.
