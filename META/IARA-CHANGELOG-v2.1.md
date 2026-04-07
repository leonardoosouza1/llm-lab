# IARA Engine v2.1 — Changelog de Otimizacao

## Data: 2026-04-07

## Resumo

Sprint de otimizacao focada em eliminar gargalos GPU-CPU no hot path de inferencia.
Baseline: 7.1 tok/s (IARA BF16) vs 6.7 tok/s (Ollama Q4) no qwen2.5-1.5b, AMD RX 6750 XT (GFX1032, 12 GB GDDR6).

Objetivo: mover toda computacao para GPU, reduzir syncs, agrupar kernel launches,
e zerar alocacoes no loop de decode. Documento complementar ao evolution paper.

---

## Arquivos Criados

| Arquivo | Descricao |
|---------|-----------|
| `iara-engine/src/inference/bottleneck.rs` | BottleneckProfiler: profiling lock-free com ring buffer, zero-cost quando desabilitado |
| `iara-engine/src/inference/sampling.rs` (SamplingBuffer) | Buffers reutilizaveis para sampling zero-alloc no hot path |
| `compare_ollama.sh` | Benchmark head-to-head IARA vs Ollama (sequencial, GPU-safe) |
| `run_stress.sh` | Script de stress test para benchmarking |

## Arquivos Modificados

### `compute-core/src/params.rs`
- Adicionado `SgemmStridedBatchedParams` com campos `a_offset`, `b_offset`, `c_offset`
- Adicionado `stride_b: usize` com suporte a broadcast (`stride_b = 0`)
- Documentacao de offsets e strides para GQA batched

### `compute-core/src/backend.rs`
- Trait `ComputeBackend` estendido com `sgemm_strided_batched()`
- Adicionado `mem_info() -> Option<(usize, usize)>` para consulta VRAM

### `rocm-backend/src/backend.rs`
- Implementacao de `sgemm_strided_batched()` via `hipblasSgemmStridedBatched`
- Implementacao de `mem_info()` via `hipMemGetInfo`

### `hip-ffi/src/blas.rs`
- Bindings FFI para `hipblasSgemmStridedBatched`

### `iara-engine/src/inference/forward.rs`
- Nova funcao `forward_traced_buffered()` que aceita `&mut SamplingBuffer`
- `forward_traced()` refatorado como wrapper de `forward_traced_buffered()`
- Forward pass inteiro em GPU; unico `to_host()` e dos logits finais

### `iara-engine/src/inference/mod.rs`
- Re-exporta `SamplingBuffer`, `sample_token_buffered`, `BottleneckProfiler`
- Adicionado modulo `bottleneck`

### `iara-engine/src/inference/sampling.rs`
- Novo `SamplingBuffer` struct com `probs`, `indexed`, `order` pre-alocados
- `sample_token_buffered()` reutiliza buffers entre tokens (zero alloc)
- `with_vocab_size()` para pre-alocacao exata

### `iara-engine/src/api/handlers.rs`
- Novo handler `unload_model_handler()` em `POST /api/unload-model`
- `HealthResponse` expandido: `status`, `model`, `version`, `timestamp`, `model_loading`, `model_vram_gb`, `vram_free_gb`, `vram_total_gb`
- `generate_internal()` agora usa `forward_traced_buffered()` com `SamplingBuffer` compartilhado

### `iara-engine/src/model/mod.rs`
- Forward pass GPU-only via `sgemm_strided_batched` para atencao GQA
- Offsets para sub-views de cabecote dentro de tensores contiguos

### `iara-engine/src/main.rs`
- Import e uso de `forward_traced_buffered` + `SamplingBuffer` no engine loop
- Rota `/api/unload-model` registrada no router

---

## Otimizacoes por Categoria

### GPU Pipeline
- **P0: Sync elimination (32 -> 3 per token)**
  - Antes: 7x `to_host()` por camada (wq, wk, wv, wo, wgate, wup, wdown) = 7 * 36 = 252 syncs + extras
  - Depois: GEMM inteiro em GPU via hipBLAS; unico download e dos logits finais para sampling
  - Syncs restantes: (1) logits download, (2) KV cache management, (3) token ID upload

- **P2: GQA grouped batched (1008 -> 448 launches)**
  - Antes: 1 kernel launch por cabecote Q (32 heads * 36 layers = 1152 na atencao sozinha)
  - Depois: `hipblasSgemmStridedBatched` agrupa todas as cabecas Q por grupo KV em 1 chamada
  - GQA 32Q/8KV: 8 launches por camada (1 por grupo) vs 32 individuais
  - Total estimado: ~448 launches/token (atencao + MLP + norms)

### Memory
- **P4: SamplingBuffer zero-alloc**
  - Antes: 3x `Vec::new()` por token (`probs`, `indexed`, `order`) de ~152K elements cada
  - Depois: `SamplingBuffer::with_vocab_size()` pre-aloca uma vez; reutilizado via `&mut`
  - Elimina pressao no allocator durante decode

- **Unload endpoint para VRAM management**
  - `POST /api/unload-model` descarrega modelo e libera VRAM
  - Permite benchmark sequencial IARA -> Ollama sem reiniciar servidor

### Compute
- **P1: Batched prefill**
  - Forward pass aceita `token_ids: &[u32]` (slice, nao token unico)
  - Prefill de N tokens do prompt em 1 forward (KV cache preenchido em batch)
  - Antes: 50 tokens de prompt = 50 forwards sequenciais

- **P2: stride_b=0 broadcast GEMM**
  - `SgemmStridedBatchedParams.stride_b = 0` permite broadcast de uma mesma matriz B para todos os batches
  - Usado em GQA: K/V de 1 KV-head compartilhado com 4 Q-heads sem duplicar dados
  - Elimina copias redundantes de pesos KV

### Observability
- **P3: BottleneckProfiler**
  - Lock-free via `AtomicU64` (Relaxed ordering)
  - Ring buffer de 10K eventos (zero alloc no hot path)
  - Macro `profile_section!` para instrumentacao ergonomica
  - RAII `ProfileSection` guard para timing automatico
  - Zero-cost quando desabilitado: early atomic check + `#[inline(always)]`

- **Unified health endpoint**
  - `GET /health` agora retorna: status, model info, version, VRAM free/total
  - Permite monitoramento externo e scripts de benchmark

---

## API Changes

### Novos Endpoints

| Metodo | Rota | Descricao |
|--------|------|-----------|
| POST | `/api/unload-model` | Descarrega modelo da VRAM, reseta flags |

### Endpoints Modificados

| Metodo | Rota | Mudanca |
|--------|------|---------|
| GET | `/health` | JSON schema expandido (ver Breaking Changes) |

---

## Breaking Changes

### `SgemmStridedBatchedParams` (compute-core)
Campos obrigatorios adicionados:
```rust
pub a_offset: usize,  // offset em elementos do batch 0 em A
pub b_offset: usize,  // offset em elementos do batch 0 em B
pub c_offset: usize,  // offset em elementos do batch 0 em C
```
Todos os call sites que constroem `SgemmStridedBatchedParams` precisam incluir esses campos.

### `HealthResponse` JSON schema
Antes:
```json
{ "status": "ok" }
```
Depois:
```json
{
  "status": "ok",
  "model": "36L-4096h",
  "version": "2.0.0",
  "timestamp": "2026-04-07T...",
  "model_loading": false,
  "model_vram_gb": 3.2,
  "vram_free_gb": 8.1,
  "vram_total_gb": 12.0
}
```

---

## How to Benchmark

```bash
# Terminal 1 — servidor IARA em release
./run_iara.sh release

# Terminal 2 — stress test + comparacao
./run_stress.sh           # carrega modelo + mede tok/s
./compare_ollama.sh       # head-to-head IARA vs Ollama (sequencial, GPU-safe)
```

O `compare_ollama.sh` automatiza: benchmark IARA -> unload -> start Ollama -> benchmark Ollama -> comparacao.

---

## Metrics (Before -> After)

| Metric | Before | After (expected) |
|--------|--------|-----------------|
| GPU syncs/token | 32 | 3 |
| Kernel launches/token | 1008 | 448 |
| Prefill (50 tokens) | 50 forwards | 1 forward |
| Sampling alloc/token | 3x Vec<152K> | 0 (reused) |
| tok/s (qwen2.5-1.5b) | 7.1 | 15-25 (pending validation) |

**Hardware:** AMD RX 6750 XT (RDNA2, 12 GB GDDR6), ROCm 6.4.4, HSA_OVERRIDE_GFX_VERSION=10.3.0

---

## Referencia

- Baseline benchmark: `compare_ollama.sh` header
- BottleneckProfiler: `iara-engine/src/inference/bottleneck.rs`
- SamplingBuffer: `iara-engine/src/inference/sampling.rs`
- GQA batched: `compute-core/src/params.rs` (SgemmStridedBatchedParams)
- Forward GPU-only: `iara-engine/src/inference/forward.rs` (forward_traced_buffered)
- Health endpoint: `iara-engine/src/api/handlers.rs` (HealthResponse)
