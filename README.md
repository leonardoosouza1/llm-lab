# IARA Engine

**A from-scratch LLM inference engine in pure Rust — multi-GPU backend, runs on a consumer AMD card, and lets you _watch the model think_ in real time.**

No PyTorch. No candle. No `llama-cpp-python`. IARA loads `safetensors` / `GGUF` / its own quantized formats straight into VRAM through a hand-written compute backend (ROCm/HIP **or** OpenCL, auto-detected at runtime), runs the full transformer forward pass, and streams **per-layer telemetry** — attention energy, MLP activation, hot neurons — to a live dashboard, token by token.

It speaks the **OpenAI API**, so anything that talks to OpenAI talks to IARA.

```
   safetensors / GGUF / .iara / .dna
                  │
                  ▼
        ┌───────────────────┐     HTTP  :3000  ── /api/* + /v1/* (OpenAI)
        │    IARA Engine     │     WS    :8080  ── per-layer telemetry stream
        │   (Rust · tokio)   │     Web   :3000  ── live "watch it think" dashboard
        └───────────────────┘
            │            │
     ROCm/HIP        OpenCL          (runtime auto-detect — no rebuild)
   AMD RX 6750 XT   Intel iGPU / any OpenCL 3.0 GPU
```

> Part of the **IARA** ecosystem. Built and tuned on hardware AMD doesn't officially
> support for ML (an RX 6750 XT gaming card), forcing `HSA_OVERRIDE_GFX_VERSION=10.3.0`.

---

## Three things this PoC shows

### 1. 👁️ You can watch the model think
Every forward pass emits a `LayerTrace` per transformer layer — attention energy, MLP energy,
residual norm, the hottest neurons, and per-layer timing. The Vue dashboard at
`http://localhost:3000` renders this as a **live heatmap that lights up layer-by-layer as each
token is generated**, plus the top-k candidate tokens and an attention view. It turns an opaque
matmul pipeline into something you can actually see.

<!-- TODO(Leonardo): drop a GIF of the dashboard here — docs/dashboard.gif — it's the strongest visual asset -->

### 2. ⚙️ It's a real systems project (no ML framework)
- **Dynamic GPU backend** behind a single `ComputeBackend` trait. **ROCm/HIP** (hand-written
  hipBLAS + HIP kernels) and **OpenCL** (JIT-compiled C kernels) are selected at runtime —
  same binary, different hardware.
- **Multi-format loader:** `safetensors` (zero-copy mmap, BF16 on device), `GGUF` (via a thin
  llama.cpp FFI), and two in-house formats — **`.iara`** (quantization-recipe container) and
  **`.dna`** (SVD-factorized weights).
- **Quantization, device-side:** `Q4_K`, `Q6_K`, `FP8 (e4m3)`, and **BitNet 2-bit ternary**
  (16× smaller than BF16) — ported/written from scratch.
- **Universal architecture auto-detect:** Qwen, Llama, Mistral, Phi, Gemma… read straight from
  `config.json` / GGUF metadata.

### 3. 🔌 It's OpenAI-compatible
A drop-in `/v1/chat/completions` + `/v1/models` surface means existing tooling
(`openai-python`, **`llm_bridge`**, LangChain, …) points at IARA with one base-URL change.
Chat messages are rendered to the correct ChatML template internally and routed through the same
generation path as the native API.

---

## Quickstart

### 1 · Clone & get a model

The engine ships **no weights** (they're git-ignored). Grab any supported model into `models/`.
A single-file GGUF is the easiest start (uses the fast llama.cpp path):

```bash
git clone https://github.com/leonardoosouza1/llm-lab.git
cd llm-lab
mkdir -p models

# Option A — GGUF (one file, recommended for a first run)
wget -c "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
     -O models/qwen2.5-1.5b.gguf

# Option B — safetensors (full native backend + telemetry)
#   download config.json, tokenizer.json and model.safetensors into models/qwen2.5-1.5b/
```

### 2 · Run

```bash
# AMD (ROCm) — override ROCM_PATH if yours isn't /opt/rocm
./scripts/run_iara.sh release        # builds + serves on :3000 / :8080

# Intel iGPU / other OpenCL GPU
./scripts/run_igpu.sh release
```

First build of the workspace takes a while (13 crates, LTO). Then open **http://localhost:3000**.

### 3 · Generate

Load a model, then call it — native API or OpenAI-compatible, your choice:

```bash
# Load (returns 202; first load JIT-compiles + pre-warms GPU kernels)
curl -s localhost:3000/api/load-model -H 'content-type: application/json' \
     -d '{"name":"qwen2.5-1.5b"}'

# OpenAI-compatible
curl -s localhost:3000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen2.5-1.5b",
  "messages": [{"role":"user","content":"Explique o que é um byte em uma frase."}],
  "max_tokens": 80
}'
```

…or point the official OpenAI SDK at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:3000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

---

## API

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/health` | Server + model status, VRAM info |
| `GET`  | `/api/models` | Discover models in `models/` |
| `POST` | `/api/load-model` | Load a model into VRAM (background, 202) |
| `POST` | `/api/unload-model` | Free VRAM |
| `POST` | `/api/generate` | Native generation (temperature, top-k/p, optional layer `budget`) |
| `POST` | `/api/trace` | Deep single-token trace (rich per-layer telemetry) |
| `POST` | **`/v1/chat/completions`** | **OpenAI-compatible chat** (non-streaming) |
| `GET`  | **`/v1/models`** | **OpenAI-compatible model list** |
| `WS`   | `:8080` | Live per-token / per-layer telemetry stream |
| `*`    | `/acp/v1/*` | Agent Communication Protocol (runs, agents, events) |

---

## Architecture

A Cargo workspace; the engine never imports a hardware backend directly — only the
`ComputeBackend` trait.

| Crate | Role |
|-------|------|
| `iara-engine` | tokio + axum server, inference pipeline, telemetry, OpenAI surface |
| `compute-core` | `ComputeBackend` trait + param structs (no hardware code) |
| `rocm-backend` | AMD ROCm/HIP impl — hipBLAS + hand-written HIP kernels |
| `opencl-backend` | OpenCL impl — JIT C kernels + GEMM |
| `hip-ffi` | low-level HIP / hipBLAS bindings |
| `llama-sys` | thin llama.cpp FFI (GGUF fast path) |
| `iara-train` | LoRA fine-tuning + DNA compression |
| `iara-observe` | telemetry / observability subsystem |
| `iara-byte-bench` | accuracy benchmark harness |

**Generation flow:** `POST /api/generate` → tokenize (ChatML) → reset KV cache → batched prefill →
per-token `forward_traced` (RMSNorm → QKV → RoPE → KV-cache → attention → MLP, telemetry per layer)
→ sample → JSON response + WebSocket broadcast.

See [`CLAUDE.md`](CLAUDE.md) and [`iara-engine/CLAUDE.md`](iara-engine/CLAUDE.md) for the deep dive.

---

## Performance & benchmarks

Measured on a single **AMD RX 6750 XT (12 GB, RDNA2, gfx1032)**:

- **~160 tok/s** for Qwen2.5-1.5B (Q4_K_M) via the llama.cpp backend.
- **77 tok/s** for the native `.dna` SVD path after porting the llama.cpp `Q6_K` mmvq kernel
  (up from 37 tok/s — a 2× kernel win).
- Kernel **pre-warm on load** eliminates the 30–60 s JIT stall on the first request.

The engine also drives a head-to-head arithmetic benchmark (Qwen-1.5B vs a byte-level model):
**62% vs 94%** over 300 problems — see [`BENCH_RESULTS.md`](BENCH_RESULTS.md).

---

## Hardware & reproducibility

| Backend | Hardware | Cargo feature |
|---------|----------|---------------|
| ROCm/HIP | AMD RX 6750 XT (gfx1032, RDNA2) | `--features rocm` |
| OpenCL | Intel Iris Plus Gen 11 / any OpenCL 3.0 GPU | `--features opencl` |

```bash
# RDNA2 needs the gfx override (card not officially ROCm-supported) + wave32
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm        # point at your install
```

---

## Status & honest limitations

- ✅ **Native + GGUF inference, telemetry dashboard, multi-backend** are the battle-tested core.
- 🆕 **OpenAI `/v1` surface** is new in this PoC — non-streaming only (no SSE yet); chat content
  is treated as text (no tool-calls / vision).
- ⚠️ **One model at a time** (`Arc<Mutex<Option<CausalLM>>>`); the native backend serializes
  requests behind that mutex (the GGUF batch server handles concurrency).
- ⚠️ OpenCL GEMM is a naive K-loop — correct but slow for large models.
- 🚧 VLMs load the text decoder only (vision encoder is future work).

---

## License

[MIT](LICENSE) © 2026 Leonardo Souza
