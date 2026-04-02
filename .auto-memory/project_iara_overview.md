---
name: IARA LLM Engine — Visão Geral do Projeto
description: Motor de inferência nativo Qwen3 em Rust, sem PyTorch/candle, com backends ROCm e OpenCL, servidor HTTP+WebSocket, dashboard Vue3
type: project
---

Motor de inferência nativo para modelos Qwen3 (8B, 4B, 1.5B) escrito do zero em Rust. Zero frameworks — safetensors → VRAM via trait ComputeBackend.

**Estrutura:** workspace Cargo com crates compute-core, rocm-backend, opencl-backend, hip-ffi, iara-engine

**Hardware:** AMD RX 6750 XT (ROCm/HIP, GFX1032, 12GB) + Intel iGPU (OpenCL, fallback)

**Portas:** HTTP :3000 (REST API axum), WebSocket :8080 (telemetria streaming)

**Caminho local:** `/home/leonardo/projects/LLM/llm-lab/`

**Why:** Projeto pessoal de pesquisa — interpretabilidade mecanística de LLMs, Liquid Neural Networks, arquitetura de agents autônomos.

**How to apply:** Ao sugerir mudanças, respeitar a constraint "zero frameworks" e a separação via trait ComputeBackend. Nunca importar hip-ffi/opencl diretamente no iara-engine.
