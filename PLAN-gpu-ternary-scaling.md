# 📜 PLAN: GPU Ternary Integration & Massive Scaling (Millions)

**Status:** 🏗️ Planning Phase
**Target Hardware:** AMD RX 6750 XT (Navi 22)
**Core Strategy:** BitNet-style Ternary Quantization on GPU + Spatial Hashing for N-Body Physics.

---

## 🎯 Success Criteria
- [ ] Implementação de **Bit-Packing (2 bits/peso)** na GPU, reduzindo uso de VRAM em 16x vs FP32.
- [ ] Kernel de **Física Local (Pauli + KAN)** escalável para >1.000.000 de nós.
- [ ] Latência de integração <16ms (60 FPS) para a malha líquida completa.
- [ ] Integração de Logit Interception usando o estado global da malha quantizada.

---

## 🛠️ Tech Stack & Architecture
- **Rust (rust-pro skill):** Orquestração e gerenciamento de buffers via `rocm-rs`.
- **HIP Kernels (Navi 22 Opt):** Escrita de kernels manuais para manipulação de bits e computação paralela.
- **Ternary Logic (1.58b):** Pesos armazenados como `u32` (16 pesos por DWORD), processados via operações bitwise na GPU.
- **Spatial Hashing:** Grid 3D dinâmico para busca de vizinhos (Essential for Pauli Repulsion at scale).

---

## 📋 Task Breakdown

### Phase 1: Architecture & Memory Layout (P0)
- **Task 1.1:** Definir especificação do buffer `TernaryWeightStream`. 
  - *Agent:* `backend-specialist`
  - *Input:* Hidden Dim (4096)
  - *Output:* Memory layout para 2-bit weights.
- **Task 1.2:** Implementar lógica de Bit-Packing em Rust para converter pesos FP32 do Qwen/KAN para o formato ternário.
  - *Agent:* `rust-pro`
  - *Verify:* Teste unitário comparando precisão original vs ternária.

### Phase 2: High-Performance GPU Kernels (P1)
- **Task 2.1:** Desenvolver `Ternary_GEMV_Kernel` (Matrix-Vector Multiplication).
  - *Agent:* `performance-optimizer`
  - *Skill:* rust-pro (Unsafe/FFI)
  - *Logic:* Unpack bits no registro da GPU e realizar acumulação (Soma/Sub).
- **Task 2.2:** Desenvolver `Spatial_Hash_Grid_Kernel`.
  - *Agent:* `performance-optimizer`
  - *Logic:* Atribuir cada neurônio a uma célula do grid para otimizar a Pressão de Pauli.
- **Task 2.3:** Implementar `Pauli_Repulsion_Spatial_Kernel`.
  - *Agent:* `rust-pro`
  - *Verify:* Comparar força resultante com versão $O(N^2)$ em pequena escala.

### Phase 3: Integration & Intuition Filter (P2)
- **Task 3.1:** Vincular ativações do `QwenModel` (Candle) aos buffers do `LiquidNetwork` (Vulkan/HIP) sem cópia via host.
  - *Agent:* `backend-specialist`
- **Task 3.2:** Implementar o "Filtro de Intuição" no `generate.rs` usando os estados ternários agregados.
  - *Agent:* `rust-pro`

---

## 🚀 Phase X: Verification Checklist
- [ ] **Build:** `cargo build --release` sem warnings de linkagem.
- [ ] **VRAM Audit:** Verificar se 1.000.000 de nós ocupam menos de 2GB de VRAM.
- [ ] **Performance:** Rodar `verify_all.py` para validar estabilidade do container ROCm.
- [ ] **Benchmark:** Medir tokens/sec com e sem o filtro de intuição ternário.

---

## ⚠️ Risks & Mitigations
- **Risk:** Divergência de precisão devido à quantização agressiva.
- **Mitigation:** Implementar "Scaling Factor" dinâmico por camada para compensar a perda de magnitude.
- **Risk:** Overhead de Unpacking na GPU.
- **Mitigation:** Usar instruções `LDS` (Local Data Share) para carregar pesos packeados uma única vez por wavefront.
