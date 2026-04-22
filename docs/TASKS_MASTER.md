# LLM-LAB — Task Graph para Agents
> Documento master de orquestração. Cada task é auto-contida e "mastigada" para agents executarem sem pesquisar.
> Gerado: 2026-04-06

---

## Grafo de Dependências

```
TASK-00 (Benchmark Baseline)
    │
    ├──► TASK-01 (Fused Kernel: Norm+Bias+RoPE)     [sem deps]
    ├──► TASK-02 (FP8 Quantization)                  [sem deps]
    ├──► TASK-03 (Composable Kernel GEMM)             [sem deps]
    │
    ├──► TASK-04 (BitNet Ternary Inference)           [depende de TASK-03]
    ├──► TASK-05 (NTC KV-Cache Compression)           [depende de TASK-02]
    │
    ├──► TASK-06 (SAE Interpretability Module)        [depende de TASK-00]
    │
    └──► TASK-07 (Dashboard: novas visualizações)     [depende de TASK-01..06]

TASK-08 (Benchmark Final + Comparação)               [depende de TASK-01..07]
```

### Paralelismo Possível

```
Wave 1 (paralelo):  TASK-00
Wave 2 (paralelo):  TASK-01, TASK-02, TASK-03
Wave 3 (paralelo):  TASK-04, TASK-05, TASK-06
Wave 4 (sequencial): TASK-07
Wave 5 (sequencial): TASK-08
```

---

## TASK-00: Benchmark Baseline

### Objetivo
Criar suite de benchmarks ANTES de qualquer melhoria. Todos os resultados de TASK-08 serão comparados contra esta baseline.

### Arquivos Envolvidos
- **Criar:** `iara-engine/benches/baseline.rs`
- **Criar:** `iara-engine/benches/utils.rs`
- **Criar:** `benchmarks/baseline_results.json`
- **Modificar:** `iara-engine/Cargo.toml` (adicionar `[[bench]]` section)

### Especificação Detalhada

#### 1. Adicionar dependência de benchmark ao Cargo.toml do iara-engine

```toml
# Adicionar ao final do Cargo.toml do iara-engine
[[bench]]
name = "baseline"
harness = false

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
serde_json = "1"
```

#### 2. Criar `iara-engine/benches/utils.rs`

```rust
use std::time::Instant;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub metric: String,
    pub value: f64,
    pub unit: String,
    pub iterations: u64,
    pub timestamp: String,
    pub hardware: String,
    pub config: BenchConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BenchConfig {
    pub model: String,
    pub quantization: String,       // "bf16", "q4", "fp8", "bitnet"
    pub kv_cache_mode: String,      // "f32", "int8", "ntc"
    pub max_seq_len: usize,
    pub batch_size: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BaselineReport {
    pub version: String,
    pub date: String,
    pub gpu: String,
    pub rocm_version: String,
    pub results: Vec<BenchmarkResult>,
}

pub fn measure<F: FnMut()>(name: &str, iterations: u64, mut f: F) -> BenchmarkResult {
    // Warmup: 3 iterações
    for _ in 0..3 { f(); }

    let mut times = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64() * 1000.0); // ms
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p99 = times[(times.len() as f64 * 0.99) as usize];

    BenchmarkResult {
        name: name.to_string(),
        metric: "latency_ms".to_string(),
        value: median,
        unit: "ms".to_string(),
        iterations,
        timestamp: chrono::Utc::now().to_rfc3339(),
        hardware: "AMD RX 6750 XT (gfx1031, 12GB, ROCm override 10.3.0)".to_string(),
        config: BenchConfig {
            model: String::new(),
            quantization: String::new(),
            kv_cache_mode: String::new(),
            max_seq_len: 0,
            batch_size: 1,
        },
    }
}
```

#### 3. Criar `iara-engine/benches/baseline.rs`

Benchmarks a medir (todos com Qwen3-8B ou menor modelo disponível):

| Benchmark | O que mede | Métrica |
|-----------|-----------|---------|
| `single_token_latency` | Latência de 1 token (forward pass completo) | ms/token |
| `throughput_128_tokens` | Gerar 128 tokens seguidos | tokens/segundo |
| `model_load_time` | Tempo de carregar modelo do disco | segundos |
| `vram_usage_bf16` | VRAM total com pesos BF16 | MB |
| `vram_usage_q4` | VRAM total com pesos Q4 | MB |
| `kv_cache_f32_vram` | VRAM do KV-cache F32 em 512 tokens | MB |
| `kv_cache_int8_vram` | VRAM do KV-cache INT8 em 512 tokens | MB |
| `gemm_bf16_latency` | GEMM BF16 isolado (4096×4096) | ms |
| `rms_norm_latency` | RMSNorm isolado (4096 dims) | µs |
| `rope_latency` | RoPE isolado (4096 dims, 32 heads) | µs |
| `softmax_latency` | Softmax isolado (32 heads × 512 seq) | µs |
| `attention_latency` | Attention completo (1 layer) | ms |
| `mlp_latency` | MLP completo (1 layer) | ms |
| `esn_step_latency` | ESN observer step | µs |
| `cfc_step_latency` | CfC correction step | µs |

**Implementação:** usar `criterion` groups. Cada benchmark deve:
1. Carregar modelo uma vez (setup)
2. Rodar N iterações com warmup
3. Reportar median, mean, p99
4. Salvar resultado em JSON

#### 4. Salvar resultados em `benchmarks/baseline_results.json`

Formato:
```json
{
  "version": "baseline-v1",
  "date": "2026-04-XX",
  "gpu": "AMD RX 6750 XT 12GB",
  "rocm_version": "6.x (HSA_OVERRIDE=10.3.0)",
  "results": [
    {"name": "single_token_latency", "value": 85.0, "unit": "ms", ...},
    ...
  ]
}
```

### Critério de Aceite
- [ ] `cargo bench --bench baseline` roda sem erros
- [ ] Arquivo `benchmarks/baseline_results.json` gerado com todos os 16 benchmarks
- [ ] Cada benchmark tem median, mean e p99
- [ ] VRAM medido via `backend.mem_info()`
- [ ] Resultados são reproduzíveis (variance < 10% entre runs)

### Testes
```bash
# Deve compilar
cargo build --release --bench baseline

# Deve rodar (ignora erros de GPU se não disponível)
cargo bench --bench baseline -- --output-format json > benchmarks/baseline_results.json
```

---

## TASK-01: Fused Kernel — Norm + Bias + RoPE

### Objetivo
Fundir 3 operações que acontecem TODA iteração (RMSNorm → add_bias → RoPE) em um único kernel HIP. Elimina 2 round-trips GPU memory ↔ compute. O trait `ComputeBackend` já tem `fused_norm_bias_rope` declarado mas a implementação no rocm-backend provavelmente faz 3 chamadas separadas.

### Contexto Técnico
No forward pass (`iara-engine/src/model/mod.rs`), para Q e K de cada layer:
```
x_norm = rms_norm(x, norm_weight)     // Lê x, escreve x_norm
q = x_norm @ Wq                       // GEMM (não funde)
q = q + bias_q                        // Lê q, escreve q (se Qwen3)
rope_inplace(q, pos)                  // Lê q, escreve q
```

O kernel fusionado faz: `q_ready = rope(norm(x) @ Wq + bias, pos)` — mas como GEMM é separado (hipBLAS), fundimos apenas o **pós-GEMM**: `q = rope(q + bias, pos)` em 1 kernel.

Se o modelo não tem bias (Llama), o kernel pula o add_bias (branch no shader).

### Arquivos Envolvidos
- **Modificar:** `hip-ffi/src/kernels.rs` — adicionar kernel HIP `fused_bias_rope_kernel`
- **Modificar:** `rocm-backend/src/backend.rs` — implementar `fused_norm_bias_rope()` usando o novo kernel
- **Modificar:** `iara-engine/src/model/mod.rs` — usar path fusionado no forward pass
- **Criar:** `iara-engine/tests/test_fused_kernel.rs` — testes de corretude

### Especificação do Kernel HIP

```c
// hip-ffi/src/kernels.rs — adicionar este kernel como string constante

extern "C" __global__ void fused_bias_rope_kernel(
    float* __restrict__ qk,          // [n_heads, head_dim] — Q ou K, in-place
    const float* __restrict__ bias,   // [n_heads * head_dim] ou nullptr
    const int n_heads,
    const int head_dim,
    const int position,               // posição atual no sequence
    const float rope_theta,           // base theta (ex: 1000000.0 para Qwen3)
    const int has_bias                // 0 ou 1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * head_dim;
    if (idx >= total) return;

    // 1. Add bias (se existir)
    if (has_bias) {
        qk[idx] += bias[idx];
    }

    // 2. RoPE in-place
    int head = idx / head_dim;
    int d = idx % head_dim;
    int half = head_dim / 2;

    if (d < half) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * d) / (float)head_dim);
        float angle = (float)position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float x0 = qk[head * head_dim + d];
        float x1 = qk[head * head_dim + d + half];

        qk[head * head_dim + d]        = x0 * cos_a - x1 * sin_a;
        qk[head * head_dim + d + half] = x0 * sin_a + x1 * cos_a;
    }
}
```

**Notas de implementação:**
- O kernel atual `rope_inplace` em `hip-ffi/src/kernels.rs` já tem a lógica de RoPE. Consultar como referência.
- `add_bias_inplace` já existe. A lógica é trivial (soma elementwise).
- O kernel fusionado apenas combina ambos em 1 launch.
- Thread block: 256 threads. Grid: `(n_heads * head_dim + 255) / 256`.

### Modificação no rocm-backend

Em `rocm-backend/src/backend.rs`, o método `fused_norm_bias_rope()` deve:
1. Verificar se `bias` é `Some` → setar `has_bias = 1`
2. Lançar `fused_bias_rope_kernel` com os parâmetros corretos
3. Se `bias` é `None` → setar `has_bias = 0` (kernel pula o add)

### Modificação no forward pass

Em `iara-engine/src/model/mod.rs`, no loop de layers do forward:
```rust
// ANTES (3 chamadas):
// backend.add_bias_inplace(&bias_params)?;
// backend.rope_inplace(&rope_params)?;

// DEPOIS (1 chamada):
backend.fused_norm_bias_rope(&FusedNormBiasRopeParams {
    qk: q_buf.clone(),
    bias: layer.q_bias.clone(), // Option<AnyTensor>
    n_heads: cfg.n_heads,
    head_dim: cfg.head_dim(),
    position: pos,
    rope_theta: cfg.rope_theta,
})?;
```

### Testes de Corretude

Criar `iara-engine/tests/test_fused_kernel.rs`:
1. Gerar tensor Q random [32, 128] (32 heads, head_dim=128)
2. Gerar bias random [32 * 128]
3. Executar path separado: `add_bias` → `rope`
4. Executar path fusionado: `fused_bias_rope`
5. Comparar resultados: `max_abs_diff < 1e-5`
6. Testar com `bias = None` (Llama path)
7. Testar com diferentes posições (0, 1, 512, 1023)

### Critério de Aceite
- [ ] Kernel compila e roda sem erros
- [ ] Teste de corretude passa (max_abs_diff < 1e-5 vs path separado)
- [ ] Forward pass produz output idêntico ao baseline (diff < 1e-4 no logit final)
- [ ] Funciona com bias (Qwen3) e sem bias (Llama)
- [ ] 1 kernel launch em vez de 2-3

### Benchmark Esperado
- **Antes:** 2-3 kernel launches por Q e K por layer = 4-6 launches/layer
- **Depois:** 1 kernel launch por Q e K por layer = 2 launches/layer
- **Estimativa:** ~5-15% speedup no attention portion (memory-bound reduction)

---

## TASK-02: FP8 Quantization (E4M3/E5M2)

### Objetivo
Adicionar quantização FP8 como opção intermediária entre BF16 (2 bytes) e Q4 (0.5 bytes). FP8 E4M3 (1 byte) oferece 2× compressão vs BF16 com melhor qualidade que Q4.

### Contexto Técnico
A RX 6750 XT (RDNA2) NÃO tem hardware FP8 nativo. Porém, podemos:
1. **Armazenar** pesos em FP8 (1 byte/param = 50% menos VRAM)
2. **Dequantizar** para F32 on-the-fly antes do GEMM (kernel de conversão)
3. O custo de dequant é amortizado pelo ganho de bandwidth (pesos cabem melhor no cache)

FP8 E4M3: 1 bit sign, 4 bits exponent, 3 bits mantissa. Range: ±448, precision: ~0.1%
FP8 E5M2: 1 bit sign, 5 bits exponent, 2 bits mantissa. Range: ±57344, precision: ~0.4%

Usar E4M3 para pesos (melhor precision). E5M2 para gradientes (se treinar, futuro).

### Arquivos Envolvidos
- **Criar:** `iara-engine/src/storage/fp8.rs` — FP8 encode/decode, quantize/dequantize
- **Modificar:** `compute-core/src/backend.rs` — adicionar `gemm_fp8_f32()` e `upload_fp8()` ao trait
- **Modificar:** `compute-core/src/params.rs` — adicionar `GemmFp8Params`
- **Criar:** `hip-ffi/src/fp8_kernels.rs` — kernel de dequantização FP8 → F32
- **Modificar:** `rocm-backend/src/backend.rs` — implementar `gemm_fp8_f32()`
- **Modificar:** `iara-engine/src/model/mod.rs` — adicionar path FP8 no loading e forward
- **Criar:** `iara-engine/tests/test_fp8.rs` — testes de corretude

### Especificação: FP8 E4M3

```rust
// iara-engine/src/storage/fp8.rs

/// FP8 E4M3 format: 1 sign + 4 exponent + 3 mantissa
/// Range: [-448, 448], Precision: ~3 decimal digits for small values
/// Special: NaN = 0x7F (max), no Inf representation

/// Converte f32 → fp8_e4m3 (u8)
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    // Clamp to FP8 range
    let clamped = val.clamp(-448.0, 448.0);

    let bits = clamped.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;  // F32 bias
    let mant = (bits >> 20) & 0x7;                  // Top 3 mantissa bits

    // FP8 E4M3 bias = 7
    let fp8_exp = (exp + 7).clamp(0, 15) as u8;

    if fp8_exp == 0 {
        return (sign as u8) << 7;  // Subnormal → 0
    }
    if fp8_exp >= 15 {
        return ((sign as u8) << 7) | 0x7E;  // Clamp to max (not NaN)
    }

    ((sign as u8) << 7) | (fp8_exp << 3) | (mant as u8)
}

/// Converte fp8_e4m3 (u8) → f32
pub fn fp8_e4m3_to_f32(val: u8) -> f32 {
    let sign = (val >> 7) & 1;
    let exp = ((val >> 3) & 0xF) as i32;
    let mant = (val & 0x7) as u32;

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    // Reconstruct F32
    let f32_exp = (exp - 7 + 127) as u32;  // Unbias FP8, rebias F32
    let f32_mant = mant << 20;              // Place in F32 mantissa position
    let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mant;

    f32::from_bits(f32_bits)
}

/// Quantiza tensor BF16 → FP8 E4M3 com per-channel scaling
pub struct Fp8Quantized {
    pub data: Vec<u8>,              // FP8 values
    pub scales: Vec<f32>,           // Per-channel (per output row) scales
    pub shape: Vec<usize>,          // Original shape
}

pub fn quantize_bf16_to_fp8(
    bf16_data: &[u16],
    shape: &[usize],  // [out_features, in_features]
) -> Fp8Quantized {
    let rows = shape[0];
    let cols = shape[1];
    let mut fp8_data = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows);

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;

        // Find absmax of row (convert bf16 → f32 first)
        let mut absmax: f32 = 0.0;
        for i in start..end {
            let val = bf16_to_f32(bf16_data[i]);
            absmax = absmax.max(val.abs());
        }

        // Scale: map absmax → 448.0 (FP8 E4M3 max)
        let scale = if absmax > 0.0 { 448.0 / absmax } else { 1.0 };
        scales.push(1.0 / scale);  // Store inverse for dequant

        // Quantize
        for i in start..end {
            let val = bf16_to_f32(bf16_data[i]) * scale;
            fp8_data.push(f32_to_fp8_e4m3(val));
        }
    }

    Fp8Quantized { data: fp8_data, scales, shape: shape.to_vec() }
}
```

### Kernel HIP: FP8 Dequant + GEMM

```c
// hip-ffi/src/fp8_kernels.rs — como string constante

// Opção A: Dequant separado (mais simples, implementar primeiro)
extern "C" __global__ void fp8_dequant_kernel(
    const unsigned char* __restrict__ fp8_data,  // [M, K] FP8
    float* __restrict__ f32_out,                  // [M, K] F32
    const float* __restrict__ scales,             // [M] per-row scales
    const int M,
    const int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    int row = idx / K;
    unsigned char fp8 = fp8_data[idx];

    // Decode FP8 E4M3
    int sign = (fp8 >> 7) & 1;
    int exp = (fp8 >> 3) & 0xF;
    int mant = fp8 & 0x7;

    float val;
    if (exp == 0 && mant == 0) {
        val = 0.0f;
    } else {
        unsigned int f32_exp = (unsigned int)(exp - 7 + 127);
        unsigned int f32_bits = ((unsigned int)sign << 31) | (f32_exp << 23) | ((unsigned int)mant << 20);
        val = __int_as_float(f32_bits);
    }

    f32_out[idx] = val * scales[row];
}
```

**Fluxo no forward pass:**
1. Peso armazenado como FP8 (1 byte/param) no GPU
2. Antes do GEMM: lançar `fp8_dequant_kernel` → F32 temporário
3. GEMM F32 normal via hipBLAS
4. F32 temporário é reutilizado (scratchpad pré-alocado)

**VRAM estimado para Qwen3-8B:**
- BF16: ~16 GB (não cabe na 12GB)
- FP8: ~8 GB (cabe!)
- Q4: ~4 GB (cabe com folga)

### Modificação no trait ComputeBackend

```rust
// compute-core/src/backend.rs — adicionar ao trait:

fn upload_fp8(&self, data: &[u8], scales: &[f32], shape: &[usize]) -> AnyTensor;
fn dequant_fp8_to_f32(&self, fp8: &AnyTensor, scales: &AnyTensor, output: &AnyTensor, m: usize, k: usize) -> Result<(), BackendError>;
```

### Modificação no model loading

```rust
// iara-engine/src/model/mod.rs — no loading loop

// Adicionar env var: IARA_FP8=1
// Se ativo:
//   1. Ler pesos BF16 do safetensors
//   2. Converter BF16 → FP8 via quantize_bf16_to_fp8()
//   3. Upload FP8 + scales para GPU
//   4. No forward: dequant → GEMM → resultado
```

### Critério de Aceite
- [ ] `f32_to_fp8_e4m3` e `fp8_e4m3_to_f32` roundtrip com error < 2%
- [ ] Quantização BF16 → FP8 → F32 tem NRMSE < 1% vs BF16 original
- [ ] Kernel `fp8_dequant_kernel` compila e roda no HIP
- [ ] Forward pass com `IARA_FP8=1` produz texto coerente
- [ ] VRAM usage ~50% menor que BF16 (medir via `backend.mem_info()`)
- [ ] Throughput: no máximo 20% mais lento que BF16 puro (dequant overhead)

### Testes
```rust
// test_fp8.rs
#[test] fn test_fp8_roundtrip_accuracy() { ... }      // f32→fp8→f32, error < 2%
#[test] fn test_fp8_quantize_tensor() { ... }          // BF16→FP8, NRMSE < 1%
#[test] fn test_fp8_dequant_kernel() { ... }           // GPU kernel output matches CPU
#[test] fn test_fp8_forward_coherent() { ... }         // Geração de texto é coerente
#[test] fn test_fp8_vram_reduction() { ... }           // VRAM < 60% do BF16
```

---

## TASK-03: Composable Kernel GEMM

### Objetivo
Substituir chamadas `hipblasSgemm` / `hipblasGemmEx` por kernels AMD Composable Kernel (CK) com fusão de operações. CK permite GEMM + activation (SiLU, GELU) em 1 kernel, eliminando write-back intermediário.

### Contexto Técnico
O llm-lab atualmente usa hipBLAS para todas as multiplicações de matrizes. hipBLAS é genérico e não funde operações. O Composable Kernel é a resposta da AMD ao cuBLAS + cuDNN fusionado:

- GEMM com epilogue customizável (bias + activation inline)
- Tile-based programming model (auto-tuning por shape)
- Suporta BF16, FP16, FP8, INT8 inputs
- Compilado como C++ template — gera kernel especializado por shape

**Limitação:** CK precisa ser compilado com hipcc. Isso adiciona tempo de build. Solução: feature flag.

### Arquivos Envolvidos
- **Criar:** `ck-backend/` — novo crate workspace member
- **Criar:** `ck-backend/Cargo.toml`
- **Criar:** `ck-backend/build.rs` — compilação dos kernels CK via hipcc
- **Criar:** `ck-backend/src/lib.rs`
- **Criar:** `ck-backend/src/gemm.rs` — wrapper Rust para CK GEMM
- **Criar:** `ck-backend/src/fused_gemm_activation.rs` — GEMM + SiLU/GELU fusionado
- **Modificar:** `Cargo.toml` (workspace) — adicionar `ck-backend`
- **Modificar:** `rocm-backend/Cargo.toml` — dep opcional em `ck-backend`
- **Modificar:** `rocm-backend/src/backend.rs` — usar CK GEMM quando disponível

### Pré-requisitos de Build
```bash
# CK headers devem estar instalados via ROCm
# Localização típica: /opt/rocm/include/ck/
# Se não instalado:
sudo apt install composable-kernel-dev
# Ou via git:
# git clone https://github.com/ROCm/composable_kernel /opt/ck
```

### Especificação: CK GEMM Wrapper

O Composable Kernel usa templates C++. Precisamos:
1. Arquivo `.cpp` com instanciações dos templates que queremos
2. Compilar com `hipcc` no build.rs
3. Expor via `extern "C"` para Rust FFI

```cpp
// ck-backend/kernels/gemm_silu.cpp

#include "ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using F32  = float;
using BF16 = ck::bhalf_t;

// GEMM: C = A @ B.T  (row-major A, col-major B, row-major C)
// Com epilogue: C = SiLU(A @ B.T)

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
    ck::tensor_layout::gemm::RowMajor,     // A layout
    ck::tensor_layout::gemm::ColumnMajor,  // B layout
    ck::tensor_layout::gemm::RowMajor,     // C layout
    BF16,                                    // A type
    BF16,                                    // B type
    F32,                                     // C type
    F32,                                     // Compute type
    ck::tensor_operation::element_wise::PassThrough,   // A elementwise
    ck::tensor_operation::element_wise::PassThrough,   // B elementwise
    ck::tensor_operation::element_wise::SiLU,          // C elementwise (SiLU fusion!)
    256,   // BlockSize
    128,   // MPerBlock
    128,   // NPerBlock
    64,    // KPerBlock
    8,     // AK1
    8,     // BK1
    32,    // MPerXDL
    32,    // NPerXDL
    2,     // MXdlPerWave
    2,     // NXdlPerWave
    // Shuffle configs...
>;

extern "C" int ck_gemm_bf16_silu(
    const void* a, const void* b, void* c,
    int M, int N, int K,
    void* hip_stream
) {
    auto gemm = DeviceGemmInstance{};
    auto argument = gemm.MakeArgument(
        static_cast<const BF16*>(a), static_cast<const BF16*>(b),
        static_cast<F32*>(c),
        M, N, K,
        K,  // LDA (A is [M, K])
        K,  // LDB (B is [N, K], col-major → stride K)
        N,  // LDC (C is [M, N])
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::SiLU{}
    );

    if (!gemm.IsSupportedArgument(argument)) return -1;

    auto invoker = gemm.MakeInvoker();
    invoker.Run(argument, StreamConfig{static_cast<hipStream_t>(hip_stream)});
    return 0;
}

// Versão sem activation (PassThrough) para attention Q@K.T
extern "C" int ck_gemm_bf16_passthrough(
    const void* a, const void* b, void* c,
    int M, int N, int K,
    void* hip_stream
) {
    // Mesma struct mas com PassThrough no epilogue
    // ...
}
```

### build.rs

```rust
// ck-backend/build.rs
fn main() {
    let ck_include = std::env::var("CK_INCLUDE")
        .unwrap_or_else(|_| "/opt/rocm/include".to_string());

    // Compilar kernels CK com hipcc
    let status = std::process::Command::new("hipcc")
        .args(&[
            "-std=c++17",
            "-O3",
            &format!("-I{}", ck_include),
            "-DNDEBUG",
            "--offload-arch=gfx1030",  // Target para override
            "-shared", "-fPIC",
            "-o", "libck_kernels.so",
            "kernels/gemm_silu.cpp",
            "kernels/gemm_gelu.cpp",
            "kernels/gemm_passthrough.cpp",
        ])
        .status()
        .expect("Failed to compile CK kernels");

    assert!(status.success(), "hipcc compilation failed");

    println!("cargo:rustc-link-search=native={}", std::env::current_dir().unwrap().display());
    println!("cargo:rustc-link-lib=dylib=ck_kernels");
}
```

### Wrapper Rust

```rust
// ck-backend/src/gemm.rs

extern "C" {
    fn ck_gemm_bf16_silu(a: *const u8, b: *const u8, c: *mut f32, m: i32, n: i32, k: i32, stream: *mut std::ffi::c_void) -> i32;
    fn ck_gemm_bf16_passthrough(a: *const u8, b: *const u8, c: *mut f32, m: i32, n: i32, k: i32, stream: *mut std::ffi::c_void) -> i32;
    fn ck_gemm_bf16_gelu(a: *const u8, b: *const u8, c: *mut f32, m: i32, n: i32, k: i32, stream: *mut std::ffi::c_void) -> i32;
}

pub enum CkActivation {
    None,       // PassThrough
    SiLU,       // gate_proj + up_proj path
    GELU,       // Phi/Gemma path
}

pub fn ck_gemm_bf16(
    a: *const u8, b: *const u8, c: *mut f32,
    m: usize, n: usize, k: usize,
    activation: CkActivation,
    stream: *mut std::ffi::c_void,
) -> Result<(), String> {
    let ret = unsafe {
        match activation {
            CkActivation::None => ck_gemm_bf16_passthrough(a, b, c, m as i32, n as i32, k as i32, stream),
            CkActivation::SiLU => ck_gemm_bf16_silu(a, b, c, m as i32, n as i32, k as i32, stream),
            CkActivation::GELU => ck_gemm_bf16_gelu(a, b, c, m as i32, n as i32, k as i32, stream),
        }
    };
    if ret == 0 { Ok(()) } else { Err(format!("CK GEMM failed: {}", ret)) }
}
```

### Integração no rocm-backend

```rust
// rocm-backend/src/backend.rs — modificar gemm_bf16_f32()

fn gemm_bf16_f32(&self, p: &GemmBf16Params) -> Result<(), BackendError> {
    #[cfg(feature = "composable-kernel")]
    {
        // Tentar CK primeiro (mais rápido se shape suportado)
        if let Ok(()) = ck_backend::ck_gemm_bf16(
            p.a_ptr(), p.b_ptr(), p.c_ptr(),
            p.m, p.n, p.k,
            ck_backend::CkActivation::None,
            self.stream,
        ) {
            return Ok(());
        }
    }

    // Fallback: hipBLAS (sempre funciona)
    self.hipblas_gemm(p)
}
```

### MLP Fusionado (Maior Ganho)

No MLP do transformer, o padrão atual é:
```
gate = x @ Wgate        // GEMM 1
up   = x @ Wup          // GEMM 2
out  = silu(gate) * up   // Kernel separado
down = out @ Wdown       // GEMM 3
```

Com CK, `gate = silu(x @ Wgate)` vira 1 operação (GEMM + SiLU epilogue). Eliminamos 1 kernel launch e 1 write-back de `gate` intermediário.

### Critério de Aceite
- [ ] `ck-backend` compila com `hipcc` (com CK headers instalados)
- [ ] Feature flag: `cargo build --features composable-kernel`
- [ ] GEMM CK produz output idêntico ao hipBLAS (max diff < 1e-4)
- [ ] MLP fusionado (GEMM+SiLU) funciona corretamente
- [ ] Fallback para hipBLAS quando CK não está disponível
- [ ] Texto gerado é idêntico ao baseline

### Benchmark Esperado
- **GEMM isolado:** CK 10-30% mais rápido que hipBLAS para shapes típicos (4096×4096, 4096×11008)
- **MLP fusionado:** 20-40% speedup (elimina 1 kernel launch + 1 intermediate buffer write)
- **Forward completo:** 10-20% speedup total

---

## TASK-04: BitNet Ternary Inference

### Objetivo
Adicionar suporte a pesos ternários {-1, 0, +1} para inferência. Elimina multiplicação matricial — apenas soma e subtração. Baseado no BitNet b1.58 da Microsoft.

### Dependência
**TASK-03** (Composable Kernel) — usa CK para comparison/fallback benchmarking. Porém, BitNet NÃO usa CK para compute (é ADD/SUB only).

### Contexto Técnico
O Universe já implementa BitNet em Rust (`layers.rs`). O llm-lab tem BitNet P+T+T mencionado no roadmap (IARA_ROADMAP.md Phase 3). A Microsoft lançou BitNet.cpp com kernels otimizados para x86/ARM.

**Ternary GEMV (Kernel Core):**
```
y[i] = sum(x[j] * w[i][j])  onde w ∈ {-1, 0, +1}
     = sum(x[j] where w==+1) - sum(x[j] where w==-1)
```

Isso é apenas gather + sum, sem multiplicação!

**Packing:** 2 bits por peso → 4 pesos por byte
```
Encoding: -1 = 0b00, 0 = 0b01, +1 = 0b10
Pack 4 weights per byte: [w3:w2:w1:w0]
```

### Arquivos Envolvidos
- **Criar:** `iara-engine/src/storage/bitnet.rs` — packing, unpacking, quantização ternária
- **Criar:** `hip-ffi/src/bitnet_kernels.rs` — kernel HIP para GEMV ternário
- **Modificar:** `compute-core/src/backend.rs` — adicionar `gemv_ternary()` ao trait
- **Modificar:** `compute-core/src/params.rs` — adicionar `GemvTernaryParams`
- **Modificar:** `rocm-backend/src/backend.rs` — implementar `gemv_ternary()`
- **Modificar:** `iara-engine/src/model/mod.rs` — path ternário no forward
- **Criar:** `iara-engine/tests/test_bitnet.rs`

### Especificação: Ternary Quantization

```rust
// iara-engine/src/storage/bitnet.rs

/// Quantiza BF16 weights para ternário {-1, 0, +1}
/// Algoritmo: absmean thresholding (BitNet b1.58)
pub struct TernaryWeight {
    pub packed: Vec<u8>,            // 2 bits per weight, 4 per byte
    pub scale: f32,                 // Per-tensor absmax scale
    pub shape: [usize; 2],         // [out_features, in_features]
}

pub fn quantize_to_ternary(bf16_data: &[u16], shape: [usize; 2]) -> TernaryWeight {
    let n = bf16_data.len();

    // 1. Convert to f32 and compute absmax
    let f32_data: Vec<f32> = bf16_data.iter().map(|&b| bf16_to_f32(b)).collect();
    let absmax = f32_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = absmax;

    // 2. Normalize to [-1, 1]
    let normalized: Vec<f32> = f32_data.iter().map(|&v| v / scale).collect();

    // 3. Quantize: round-to-nearest ternary
    //    threshold = mean(|w_normalized|)
    let abs_mean: f32 = normalized.iter().map(|v| v.abs()).sum::<f32>() / n as f32;

    let ternary: Vec<i8> = normalized.iter().map(|&v| {
        if v > abs_mean { 1 }
        else if v < -abs_mean { -1 }
        else { 0 }
    }).collect();

    // 4. Pack: 4 weights per byte
    //    Encoding: -1 → 0b00, 0 → 0b01, +1 → 0b10
    let packed_len = (n + 3) / 4;
    let mut packed = vec![0u8; packed_len];

    for i in 0..n {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let code = match ternary[i] {
            -1 => 0b00u8,
             0 => 0b01u8,
             1 => 0b10u8,
             _ => unreachable!(),
        };
        packed[byte_idx] |= code << bit_offset;
    }

    TernaryWeight { packed, scale, shape }
}
```

### Kernel HIP: Ternary GEMV

```c
// hip-ffi/src/bitnet_kernels.rs

extern "C" __global__ void ternary_gemv_kernel(
    const float* __restrict__ x,          // [K] input vector
    const unsigned char* __restrict__ w,  // [M, K/4] packed ternary weights
    float* __restrict__ y,                // [M] output vector
    const float scale,                    // Weight scale
    const int M,                          // Output dim
    const int K                           // Input dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    int packed_k = (K + 3) / 4;

    for (int pk = 0; pk < packed_k; pk++) {
        unsigned char byte = w[row * packed_k + pk];

        for (int sub = 0; sub < 4 && (pk * 4 + sub) < K; sub++) {
            int code = (byte >> (sub * 2)) & 0x3;
            int col = pk * 4 + sub;

            // code: 0b00 = -1, 0b01 = 0, 0b10 = +1
            if (code == 0) {        // -1
                sum -= x[col];
            } else if (code == 2) { // +1
                sum += x[col];
            }
            // code == 1 (zero): skip (no operation)
        }
    }

    y[row] = sum * scale;
}
```

**Nota de otimização:** Para produção, usar shared memory com reductin warp-level. O kernel acima é a versão funcional baseline. Uma v2 pode usar:
- Shared memory para tile de x
- Warp-level reduction com `__shfl_down()`
- Vectorized loads (4 bytes = 16 weights por load)

### Integração no Forward Pass

```rust
// Em model/mod.rs, no forward loop:

match cfg.weight_mode {
    WeightMode::BF16 => backend.gemm_bf16_f32(&gemm_params)?,
    WeightMode::Q4   => backend.gemv_q4(&q4_params)?,
    WeightMode::FP8  => { backend.dequant_fp8(&fp8, &f32_tmp)?; backend.sgemm(&gemm_params)?; },
    WeightMode::Ternary => backend.gemv_ternary(&ternary_params)?,
}
```

**Env var:** `IARA_BITNET=1` para ativar quantização ternária on load.

### Critério de Aceite
- [ ] `quantize_to_ternary()` produz packing correto (decode matches encode)
- [ ] Kernel `ternary_gemv_kernel` compila e roda no HIP
- [ ] GEMV ternário produz output com NRMSE < 5% vs BF16 GEMM (ternary é lossy)
- [ ] Forward pass com `IARA_BITNET=1` gera texto (qualidade reduzida é esperada)
- [ ] VRAM: ~87.5% menor que BF16 (2 bits vs 16 bits)
- [ ] Zero multiplicações no kernel (apenas ADD/SUB)

### Benchmark Esperado
- **VRAM:** Qwen3-8B em ~1 GB (vs 16GB BF16)
- **Speed:** Potencialmente mais rápido que BF16 GEMM (memory-bound → compute-free)
- **Quality:** Degradação esperada sem fine-tuning ternário. Isso é PoC para eventual IARA-1B treinada nativa ternária.

---

## TASK-05: NTC KV-Cache Compression

### Objetivo
Substituir o KV-cache linear (F32 ou INT8) por uma representação neural comprimida inspirada no NVIDIA NTC. Em vez de armazenar K/V para cada token, treinar uma latent grid + MLP decoder que reconstrói K/V sob demanda.

### Dependência
**TASK-02** (FP8) — os pesos da MLP decoder usam FP8 para economizar VRAM.

### Contexto Técnico
KV-cache atual: `[max_seq_len, kv_dim]` por layer, por K e V. Para Qwen3-8B:
- 36 layers × 2 (K+V) × 1024 seq × 1024 kv_dim × 4 bytes (F32) = **288 MB**
- Com INT8: **72 MB**
- Com NTC: target **18 MB** (4× menos que INT8)

**Insight:** tokens adjacentes têm K/V similares (temporal locality). A MLP aprende esse padrão.

**Arquitetura NTC para KV:**
```
Input: position index p ∈ [0, max_seq_len]
  ↓
Positional Encoding: γ(p) = [sin(2^0·π·p/L), cos(2^0·π·p/L), ..., sin(2^F·π·p/L), cos(2^F·π·p/L)]
  → [2F] dimensional
  ↓
Latent Grid Lookup: bilinear interpolation em grid [G, latent_dim]
  → [latent_dim] dimensional
  ↓
Concat: [γ(p), latent] → [2F + latent_dim]
  ↓
MLP: Linear(2F + latent_dim, hidden) → ReLU → Linear(hidden, kv_dim)
  → [kv_dim] = K ou V reconstruído para posição p
```

### Arquivos Envolvidos
- **Criar:** `iara-engine/src/model/ntc_cache.rs` — NTC KV-cache struct + training + inference
- **Criar:** `hip-ffi/src/ntc_kernels.rs` — kernels GPU para positional encoding + MLP forward
- **Modificar:** `compute-core/src/backend.rs` — adicionar `ntc_cache_encode()`, `ntc_cache_decode()`
- **Modificar:** `iara-engine/src/model/kv_cache.rs` — adicionar `KvCacheMode::NTC`
- **Modificar:** `iara-engine/src/model/mod.rs` — integrar NTC cache no forward
- **Criar:** `iara-engine/tests/test_ntc_cache.rs`

### Especificação

```rust
// iara-engine/src/model/ntc_cache.rs

/// NTC-compressed KV Cache
/// Em vez de [max_seq, kv_dim], armazena:
///   - latent_grid: [grid_size, latent_dim] — aprox. 64×32 = 2KB por layer/kv
///   - mlp_weights: ~8KB por layer/kv (2 linear layers)
///   - Total: ~10KB por layer/kv vs ~4MB (F32) ou ~1MB (INT8)

pub struct NtcKvCache {
    pub grid_size: usize,          // Número de pontos na grid (ex: 64)
    pub latent_dim: usize,         // Dimensão do latente (ex: 32)
    pub pe_freqs: usize,           // Número de frequências positional encoding (ex: 16)
    pub hidden_dim: usize,         // Hidden dim da MLP (ex: 128)
    pub kv_dim: usize,             // Output dim (= n_kv_heads * head_dim)

    // Parâmetros treináveis (por layer, per K/V)
    pub latent_grid: Vec<f32>,     // [grid_size, latent_dim]
    pub w1: Vec<f32>,              // [2*pe_freqs + latent_dim, hidden_dim]
    pub b1: Vec<f32>,              // [hidden_dim]
    pub w2: Vec<f32>,              // [hidden_dim, kv_dim]
    pub b2: Vec<f32>,              // [kv_dim]

    // Buffer de tokens recentes (não comprimidos, janela de 32)
    pub recent_buffer: Vec<f32>,   // [recent_window, kv_dim] — últimos N tokens exatos
    pub recent_window: usize,      // ex: 32
    pub current_len: usize,

    // Training state
    pub optimizer_state: AdamState,
}

/// Treinamento online: a cada N tokens, re-treinar MLP+grid nos K/V acumulados
impl NtcKvCache {
    pub fn new(grid_size: usize, latent_dim: usize, pe_freqs: usize, hidden_dim: usize, kv_dim: usize) -> Self {
        let input_dim = 2 * pe_freqs + latent_dim;
        Self {
            grid_size, latent_dim, pe_freqs, hidden_dim, kv_dim,
            latent_grid: random_init(grid_size * latent_dim, 0.01),
            w1: xavier_init(input_dim, hidden_dim),
            b1: vec![0.0; hidden_dim],
            w2: xavier_init(hidden_dim, kv_dim),
            b2: vec![0.0; kv_dim],
            recent_buffer: vec![0.0; 32 * kv_dim],
            recent_window: 32,
            current_len: 0,
            optimizer_state: AdamState::new(),
        }
    }

    /// Append novo K ou V (chamado no forward pass)
    pub fn append(&mut self, kv_vector: &[f32]) {
        // 1. Adicionar ao recent_buffer (circular)
        let pos_in_recent = self.current_len % self.recent_window;
        self.recent_buffer[pos_in_recent * self.kv_dim .. (pos_in_recent+1) * self.kv_dim]
            .copy_from_slice(kv_vector);

        self.current_len += 1;

        // 2. A cada recent_window tokens, comprimir recent_buffer na NTC
        if self.current_len > 0 && self.current_len % self.recent_window == 0 {
            self.compress_recent();
        }
    }

    /// Comprime os últimos recent_window tokens na representação NTC
    fn compress_recent(&mut self) {
        // Mini-training: 50 steps de SGD/Adam
        // Loss: MSE entre MLP(pos_encoding(p)) e K/V real
        // Target: recent_buffer[0..recent_window]
        // Input: posições [current_len - recent_window .. current_len]

        for _step in 0..50 {
            let mut total_loss = 0.0;

            for i in 0..self.recent_window {
                let abs_pos = self.current_len - self.recent_window + i;
                let target = &self.recent_buffer[i * self.kv_dim .. (i+1) * self.kv_dim];

                // Forward
                let predicted = self.forward_single(abs_pos);

                // Loss: MSE
                let loss: f32 = predicted.iter().zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f32>() / self.kv_dim as f32;
                total_loss += loss;

                // Backward + Adam update (simplified)
                self.backward_and_update(abs_pos, &predicted, target);
            }
        }
    }

    /// Reconstrói K ou V para posição `pos`
    pub fn decode(&self, pos: usize) -> Vec<f32> {
        // Se pos está no recent_buffer, retornar exato (sem loss)
        if self.current_len > 0 && pos >= self.current_len.saturating_sub(self.recent_window) {
            let idx = pos % self.recent_window;
            return self.recent_buffer[idx * self.kv_dim .. (idx+1) * self.kv_dim].to_vec();
        }

        // Senão, decodificar via NTC
        self.forward_single(pos)
    }

    /// Forward pass da MLP decodera
    fn forward_single(&self, pos: usize) -> Vec<f32> {
        // 1. Positional Encoding
        let pe = positional_encoding(pos, self.pe_freqs, self.current_len.max(1));

        // 2. Latent Grid Lookup (interpolação linear)
        let grid_pos = (pos as f32 / self.current_len.max(1) as f32) * (self.grid_size - 1) as f32;
        let g0 = grid_pos.floor() as usize;
        let g1 = (g0 + 1).min(self.grid_size - 1);
        let alpha = grid_pos.fract();

        let latent: Vec<f32> = (0..self.latent_dim).map(|d| {
            self.latent_grid[g0 * self.latent_dim + d] * (1.0 - alpha)
            + self.latent_grid[g1 * self.latent_dim + d] * alpha
        }).collect();

        // 3. Concat: [pe, latent]
        let input: Vec<f32> = pe.iter().chain(latent.iter()).cloned().collect();

        // 4. MLP: Linear → ReLU → Linear
        let hidden = linear_relu(&input, &self.w1, &self.b1, self.hidden_dim);
        let output = linear(&hidden, &self.w2, &self.b2, self.kv_dim);

        output
    }
}

fn positional_encoding(pos: usize, n_freqs: usize, max_len: usize) -> Vec<f32> {
    let mut pe = Vec::with_capacity(2 * n_freqs);
    let norm_pos = pos as f32 / max_len as f32;  // [0, 1]

    for f in 0..n_freqs {
        let freq = (2.0f32).powi(f as i32) * std::f32::consts::PI;
        pe.push((freq * norm_pos).sin());
        pe.push((freq * norm_pos).cos());
    }
    pe
}
```

### Integração no Forward Pass

```rust
// model/kv_cache.rs — adicionar variante
pub enum KvCacheMode {
    F32,
    Int8,
    NTC { grid_size: usize, latent_dim: usize },
}

// model/mod.rs — no attention:
match kv_cache_mode {
    KvCacheMode::F32 | KvCacheMode::Int8 => {
        // Path atual: copy_to_kv_cache, read direto
    },
    KvCacheMode::NTC { .. } => {
        // 1. Append K, V ao NTC cache
        ntc_k_cache.append(&k_vector);
        ntc_v_cache.append(&v_vector);

        // 2. Para attention: decodificar K, V de todas posições [0..current_len]
        for p in 0..current_len {
            k_decoded[p] = ntc_k_cache.decode(p);
            v_decoded[p] = ntc_v_cache.decode(p);
        }

        // 3. Attention normal com k_decoded, v_decoded
    }
}
```

**Env var:** `IARA_NTC_CACHE=1` para ativar.

### Critério de Aceite
- [ ] NTC cache comprime/decomprime com MSE < 0.01 para tokens recentes
- [ ] Recent buffer retorna valores exatos (zero loss para últimos 32 tokens)
- [ ] Training online (50 steps) converge em < 10ms (amortizado)
- [ ] Forward pass completo com NTC cache gera texto coerente
- [ ] VRAM do KV-cache: ~4× menor que INT8 (~18 MB vs ~72 MB)
- [ ] Latência por token: no máximo 2× mais lento que F32 cache (decode overhead)

### Benchmark Esperado
- **VRAM:** 72 MB (INT8) → ~18 MB (NTC) para 1024 seq, 36 layers
- **Quality:** Leve degradação para tokens antigos (>32 posições atrás). Tokens recentes são exatos.
- **Latência:** Overhead de decode. Amortizado para contextos longos onde cache seria evicted anyway.
- **Contexto efetivo:** "Infinito" — NTC não evicta, apenas comprime. A MLP memoriza tudo.

---

## TASK-06: SAE Interpretability Module

### Objetivo
Adicionar Sparse Autoencoder (SAE) como módulo de observabilidade que decompõe ativações de qualquer layer em features interpretáveis e sparse.

### Dependência
**TASK-00** (Baseline) — para ter telemetria de referência.

### Contexto Técnico
SAE treina ONLINE durante inferência (como o ESN). Observa ativações de uma layer e aprende a decompor em features sparse:

```
Encoder: z = ReLU(W_enc @ (x - b_dec) + b_enc)    // z é sparse (maioria zeros)
Decoder: x_hat = W_dec @ z + b_dec                  // Reconstrução
Loss: MSE(x, x_hat) + λ * ||z||_1                   // Sparsity penalty
```

Cada dimensão de `z` corresponde a uma "feature" interpretável (ex: "token é verbo", "contexto é pergunta", "incerteza alta").

### Arquivos Envolvidos
- **Criar:** `iara-engine/src/lnn/sae.rs` — Sparse Autoencoder struct + training
- **Modificar:** `iara-engine/src/inference/forward.rs` — hook SAE após cada layer
- **Modificar:** `iara-engine/src/main.rs` — WebSocket messages para SAE state
- **Criar:** `iara-engine/tests/test_sae.rs`

### Especificação

```rust
// iara-engine/src/lnn/sae.rs

/// Sparse Autoencoder para interpretabilidade de layers LLM
pub struct SparseAutoencoder {
    pub input_dim: usize,     // = hidden_size (ex: 4096)
    pub latent_dim: usize,    // = expansion × hidden_size (ex: 4× = 16384)
    pub sparsity_coeff: f32,  // λ para L1 penalty (ex: 0.001)

    // Parâmetros
    pub w_enc: Vec<f32>,      // [latent_dim, input_dim]
    pub b_enc: Vec<f32>,      // [latent_dim]
    pub w_dec: Vec<f32>,      // [input_dim, latent_dim]
    pub b_dec: Vec<f32>,      // [input_dim]

    // Training state
    pub step: u64,
    pub running_loss: f32,
    pub running_sparsity: f32,  // Fração média de zeros em z
    pub adam: AdamOptimizer,

    // Feature tracking
    pub feature_activations: Vec<f32>,  // [latent_dim] — contagem de ativações por feature
    pub feature_labels: Vec<String>,    // [latent_dim] — labels humanas (populated externally)
}

impl SparseAutoencoder {
    pub fn new(input_dim: usize, expansion: usize, sparsity_coeff: f32) -> Self {
        let latent_dim = input_dim * expansion;
        Self {
            input_dim,
            latent_dim,
            sparsity_coeff,
            w_enc: kaiming_init(latent_dim, input_dim),
            b_enc: vec![0.0; latent_dim],
            w_dec: kaiming_init(input_dim, latent_dim),  // Initialized as unit norm columns
            b_dec: vec![0.0; input_dim],
            step: 0,
            running_loss: 0.0,
            running_sparsity: 0.0,
            adam: AdamOptimizer::new(/* lr */ 3e-4, /* beta1 */ 0.9, /* beta2 */ 0.999),
            feature_activations: vec![0.0; latent_dim],
            feature_labels: vec![String::new(); latent_dim],
        }
    }

    /// Forward + backward em 1 step (online training)
    /// Retorna: (features_ativas: Vec<(usize, f32)>, reconstruction_loss: f32)
    pub fn observe(&mut self, x: &[f32]) -> (Vec<(usize, f32)>, f32) {
        assert_eq!(x.len(), self.input_dim);

        // 1. Encode: z = ReLU(W_enc @ (x - b_dec) + b_enc)
        let x_centered: Vec<f32> = x.iter().zip(&self.b_dec).map(|(xi, bi)| xi - bi).collect();
        let z_pre: Vec<f32> = matmul_1d(&self.w_enc, &x_centered, self.latent_dim, self.input_dim);
        let z: Vec<f32> = z_pre.iter().zip(&self.b_enc).map(|(zi, bi)| (zi + bi).max(0.0)).collect();

        // 2. Decode: x_hat = W_dec @ z + b_dec
        let x_hat_pre: Vec<f32> = matmul_1d(&self.w_dec, &z, self.input_dim, self.latent_dim);
        let x_hat: Vec<f32> = x_hat_pre.iter().zip(&self.b_dec).map(|(xi, bi)| xi + bi).collect();

        // 3. Loss: MSE + L1
        let mse: f32 = x.iter().zip(&x_hat).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / self.input_dim as f32;
        let l1: f32 = z.iter().map(|zi| zi.abs()).sum::<f32>() / self.latent_dim as f32;
        let loss = mse + self.sparsity_coeff * l1;

        // 4. Backward (gradient computation)
        // d_loss/d_x_hat = 2(x_hat - x) / input_dim
        let d_x_hat: Vec<f32> = x_hat.iter().zip(x).map(|(xh, xt)| 2.0 * (xh - xt) / self.input_dim as f32).collect();

        // d_loss/d_z = W_dec.T @ d_x_hat + lambda * sign(z) / latent_dim
        let d_z_recon: Vec<f32> = matmul_transposed_1d(&self.w_dec, &d_x_hat, self.latent_dim, self.input_dim);
        let d_z: Vec<f32> = d_z_recon.iter().zip(&z).enumerate().map(|(_i, (dr, zi))| {
            if *zi > 0.0 {
                dr + self.sparsity_coeff * zi.signum() / self.latent_dim as f32
            } else {
                0.0  // ReLU gradient = 0 when z <= 0
            }
        }).collect();

        // 5. Update weights via Adam
        // (simplified: update w_enc, b_enc, w_dec, b_dec)
        self.adam_update_weights(&x_centered, &z, &d_x_hat, &d_z);

        // 6. Track active features
        let active_features: Vec<(usize, f32)> = z.iter().enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, &v)| { self.feature_activations[i] += 1.0; (i, v) })
            .collect();

        let sparsity = 1.0 - (active_features.len() as f32 / self.latent_dim as f32);

        // 7. Running stats
        let alpha = 0.99;
        self.running_loss = alpha * self.running_loss + (1.0 - alpha) * loss;
        self.running_sparsity = alpha * self.running_sparsity + (1.0 - alpha) * sparsity;
        self.step += 1;

        (active_features, loss)
    }

    /// Top-K features mais ativas (para dashboard)
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32, &str)> {
        let mut indexed: Vec<(usize, f32)> = self.feature_activations.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);
        indexed.iter().map(|(i, v)| (*i, *v, self.feature_labels[*i].as_str())).collect()
    }
}
```

### Integração no Forward Pass

```rust
// inference/forward.rs — após cada layer (ou layer específica):

// SAE observa hidden state APÓS residual + MLP de cada layer
if let Some(sae) = &mut model.sae {
    let hidden_f32 = backend.download_f32(&hidden_buf);
    let (active_features, loss) = sae.observe(&hidden_f32);

    // Enviar via WebSocket
    if let Some(tx) = &broadcast_tx {
        let msg = serde_json::json!({
            "type": "sae_state",
            "layer": layer_idx,
            "active_features": active_features.len(),
            "top_features": sae.top_features(10),
            "sparsity": sae.running_sparsity,
            "loss": sae.running_loss,
            "step": sae.step,
        });
        let _ = tx.send(msg.to_string());
    }
}
```

**Env var:** `IARA_SAE=1` para ativar. `IARA_SAE_LAYER=15` para observar layer específica (default: penúltima).

### WebSocket Message

```json
{
    "type": "sae_state",
    "layer": 15,
    "active_features": 127,
    "top_features": [
        [42, 1523.0, ""],
        [1087, 892.0, ""],
        [3205, 654.0, ""]
    ],
    "sparsity": 0.992,
    "loss": 0.0042,
    "step": 15000
}
```

### Critério de Aceite
- [ ] SAE treina online com loss convergindo (decreasing over steps)
- [ ] Sparsity > 95% (maioria das features são zero)
- [ ] Overhead de latência < 5ms por token (CPU-side, non-blocking)
- [ ] WebSocket message `sae_state` enviada a cada token
- [ ] `top_features()` retorna as features mais frequentes
- [ ] Sem impacto no output da inferência (SAE é read-only observer)

---

## TASK-07: Dashboard — Novas Visualizações

### Objetivo
Adicionar ao frontend Vue3 as visualizações para os novos módulos: FP8 status, BitNet stats, NTC cache metrics, SAE features.

### Dependência
**TASK-01 a TASK-06** — precisa dos WebSocket messages definidos em cada task.

### Arquivos Envolvidos
- **Modificar:** `frontend/index.html` — novos tabs/panels
- **Modificar:** `frontend/src/api/websocket.js` — handlers para novas messages
- **Modificar:** `frontend/src/store/index.js` — state para novos dados
- **Criar:** `frontend/src/views/QuantView.js` — painel de quantização (FP8/BitNet/Q4)
- **Criar:** `frontend/src/views/NtcView.js` — painel NTC cache
- **Criar:** `frontend/src/views/SaeView.js` — painel SAE interpretabilidade
- **Criar:** `frontend/src/charts/sae_features_chart.js` — visualização de features
- **Criar:** `frontend/src/charts/ntc_grid_chart.js` — visualização da latent grid

### Especificação por Painel

#### QuantView (Tab: "Quant ⚡")
Mostra estado da quantização ativa:
- Dropdown mostrando modo atual (BF16 / Q4 / FP8 / BitNet)
- VRAM usage bar chart (antes vs depois)
- Per-layer quantization error (heatmap: layer × metric)
- Widget: comparação de NRMSE entre modos

#### NtcView (Tab: "NTC 🧠")
Mostra estado do NTC KV-cache:
- Latent grid como heatmap 2D (grid_size × latent_dim) — atualiza a cada compression step
- Reconstruction loss over time (line chart)
- VRAM: NTC vs F32 vs INT8 (bar chart comparativo)
- Tokens "recentes" (exact) vs "comprimidos" (NTC) highlight

#### SaeView (Tab: "SAE 🔬")
Mostra features ativas do SAE:
- Top-20 features como bar chart horizontal (feature index × activation count)
- Sparsity gauge (ex: 99.2% sparse)
- Reconstruction loss over time
- Feature activation heatmap: tokens × features (últimos 50 tokens × top-20 features)
- Click numa feature para ver em quais tokens ela ativa

### WebSocket Dispatch

```javascript
// frontend/src/api/websocket.js — adicionar handlers:

case 'sae_state':
    store.commit('updateSae', msg);
    break;
case 'ntc_state':
    store.commit('updateNtc', msg);
    break;
case 'quant_status':
    store.commit('updateQuant', msg);
    break;
```

### Vuex Store

```javascript
// frontend/src/store/index.js — adicionar state:

state: {
    // ... existing state ...
    sae: {
        active: false,
        layer: null,
        activeFeatures: 0,
        topFeatures: [],
        sparsity: 0,
        loss: 0,
        step: 0,
        history: [],  // últimos 100 steps
    },
    ntc: {
        active: false,
        gridSize: 0,
        latentDim: 0,
        vramMb: 0,
        reconstructionLoss: 0,
        compressedTokens: 0,
        recentTokens: 0,
        history: [],
    },
    quant: {
        mode: 'bf16',
        vramMb: 0,
        nrmse: 0,
        perLayerError: [],
    }
}
```

### Critério de Aceite
- [ ] 3 novos tabs visíveis no dashboard (Quant, NTC, SAE)
- [ ] Dados renderizam corretamente quando módulos estão ativos
- [ ] Tabs mostram "Módulo não ativo" quando env var não está setada
- [ ] Heatmaps e charts atualizam em real-time via WebSocket
- [ ] Sem erros no console do browser
- [ ] Estilo consistente com tema Amazonian bio-tech existente

---

## TASK-08: Benchmark Final + Comparação

### Objetivo
Rodar a MESMA suite de benchmarks do TASK-00 com TODAS as melhorias ativas. Gerar relatório comparativo.

### Dependência
**TASK-01 a TASK-07** — todas as tasks anteriores.

### Arquivos Envolvidos
- **Criar:** `iara-engine/benches/improved.rs` — benchmarks pós-melhoria
- **Criar:** `benchmarks/improved_results.json`
- **Criar:** `benchmarks/COMPARISON.md` — relatório comparativo

### Especificação

#### Cenários de Benchmark

| Cenário | Config | O que testa |
|---------|--------|-------------|
| A | BF16 + F32 cache (baseline rerun) | Regressão check |
| B | BF16 + fused kernel | TASK-01 impact |
| C | FP8 + F32 cache | TASK-02 impact |
| D | BF16 + CK GEMM | TASK-03 impact |
| E | BitNet ternário | TASK-04 impact |
| F | BF16 + NTC cache | TASK-05 impact |
| G | FP8 + CK GEMM + fused + NTC cache | All combined |

Para cada cenário, medir:
1. **Single token latency** (ms)
2. **Throughput 128 tokens** (tok/s)
3. **VRAM total** (MB)
4. **VRAM KV-cache** (MB)
5. **Perplexity** (se possível, em prompt de referência)
6. **Output text quality** (manual: coerência 1-5)

#### Formato do Relatório

```markdown
# LLM-Lab Benchmark Comparison

## Hardware
- GPU: AMD RX 6750 XT (12GB, gfx1031, HSA_OVERRIDE=10.3.0)
- CPU: [detectar]
- RAM: [detectar]
- ROCm: [version]

## Results

| Metric | Baseline | +Fused | +FP8 | +CK | +BitNet | +NTC | All |
|--------|----------|--------|------|-----|---------|------|-----|
| tok/s  |          |        |      |     |         |      |     |
| ms/tok |          |        |      |     |         |      |     |
| VRAM   |          |        |      |     |         |      |     |
| ...    |          |        |      |     |         |      |     |

## Analysis
[Análise de cada melhoria e impacto combinado]
```

### Critério de Aceite
- [ ] Todos os 7 cenários executados com sucesso
- [ ] JSON com resultados completos em `benchmarks/improved_results.json`
- [ ] Relatório markdown em `benchmarks/COMPARISON.md`
- [ ] Nenhum cenário teve regressão vs baseline (exceto latência esperada do NTC decode)
- [ ] Cenário G (all combined) tem throughput ≥ baseline OU VRAM ≤ 50% do baseline

---

## Instruções para Agents

### Antes de começar qualquer TASK:
1. Ler TODO este documento
2. Ler os arquivos listados em "Arquivos Envolvidos"
3. NÃO pesquisar na internet — toda informação necessária está neste spec
4. Seguir a especificação EXATAMENTE como descrita
5. Ao criar código novo, seguir o estilo existente do projeto (Rust edition 2021, snake_case, doc comments)

### Ao finalizar uma TASK:
1. Rodar `cargo build --release` — deve compilar sem warnings
2. Rodar todos os testes listados no critério de aceite
3. Gerar diff do que foi alterado
4. Documentar qualquer desvio do spec e o motivo

### Convenções do Projeto
- **Rust:** edition 2021, `#[allow(dead_code)]` nos módulos experimentais
- **Erros:** usar `BackendError` do compute-core, nunca `unwrap()` em produção
- **GPU buffers:** sempre via `AnyTensor` (type-erased `Arc<dyn Any>`)
- **Feature flags:** novos módulos atrás de features (ex: `--features composable-kernel`)
- **Env vars:** prefixo `IARA_` (ex: `IARA_FP8=1`)
- **WebSocket:** JSON com campo `"type"` como discriminante
- **Frontend:** Vue3 Options API (sem Composition API), inline JavaScript (sem bundler)

---

*Fim do Task Graph — LLM-Lab v2.0*
*9 tasks, 4 waves de execução, ~20 arquivos novos, ~15 arquivos modificados*
