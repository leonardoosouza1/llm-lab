# qwen/ — Forward Pass + Sampling

## Módulos

| Arquivo | Papel |
|---------|-------|
| `forward.rs` | `forward_traced()`: forward pass com telemetria completa por camada |
| `sampling.rs` | `sample_token()`: temperatura, top-k, top-p, multinomial |
| `mod.rs` | Re-exporta ambos |

---

## forward.rs — forward_traced()

**Assinatura:**
```rust
pub fn forward_traced(
    model: &mut CausalLM,
    token_ids: &[u32],
    sampling: &SamplingParams,
) -> ModelResult<GenerationStep>
```

**Retorna:** `GenerationStep` com token gerado + 36 `LayerTrace` + top-40 logits.

### Fluxo por camada (36×):

```
1. RMSNorm(x, input_norm_w)
2. Q = normed @ wq    [n, q_dim]   (wq em GPU → to_host() → GEMM host)
3. K = normed @ wk    [n, kv_dim]
4. V = normed @ wv    [n, kv_dim]
5. pos_start = layer.kv_k.seq_len()   ← posição absoluta no contexto
6. apply_rope(&mut Q, ..., pos_start, rope_theta)   ← rotação posicional em Q
7. apply_rope(&mut K, ..., pos_start, rope_theta)   ← rotação posicional em K
8. layer.kv_k.append(&K)   ← K rotacionado no cache
9. layer.kv_v.append(&V)   ← V sem rotação no cache
10. Attention: scores[qi,ki] = (Q[qi] · K[ki]) / sqrt(head_dim)
11. Softmax(scores) → weighted sum sobre V → attn [n, q_dim]
12. attn_proj = attn @ wo
13. x = x + attn_proj   ← residual
14. RMSNorm(x, post_attn_norm_w)
15. gate = normed2 @ wgate
16. up   = normed2 @ wup
17. swiglu = SiLU(gate) × up
18. mlp = swiglu @ wdown
19. x = x + mlp   ← residual
20. Captura LayerTrace { attention_energy, mlp_energy, residual_norm, hot_neurons, time_us }
```

### GQA (Grouped Query Attention)
Qwen3-8B: 32 Q heads, 8 KV heads.
Mapeamento: `kv_h = h_i * n_kv / n_heads`
Ex: heads Q 0-3 → KV head 0, heads Q 4-7 → KV head 1, etc.

### Telemetria capturada por camada (LayerTrace)

```rust
pub struct LayerTrace {
    pub layer_idx: usize,
    pub attention_energy: f32,   // sum(Q²) — intensidade do padrão
    pub mlp_energy: f32,         // sum(gate²) — ativação MLP
    pub residual_norm: f32,      // L2(x) / sqrt(n*h) após residual
    pub max_activation: f32,     // max(|x|) no hidden state
    pub hot_neurons: Vec<usize>, // top-10 índices mais ativos
    pub time_us: u128,           // microssegundos nesta camada
}
```

### Coleta de logits

```rust
// Top-40 dos logits BRUTOS (pré-sampling) — para telemetria no frontend
let top_logits: Vec<(u32, f32)> = sorted[..40].to_vec();

// Token AMOSTRADO — temperatura/top-k/top-p aplicados aqui
let token_id = sample_token(&logits, sampling);
```

Os `top_logits` refletem o que o modelo "pensa" (distribuição real).
O token amostrado pode ser qualquer um deles (não necessariamente o #1).

---

## sampling.rs — sample_token()

**Assinatura:**
```rust
pub fn sample_token(logits: &[f32], params: &SamplingParams) -> u32
```

### SamplingParams

```rust
pub struct SamplingParams {
    pub temperature: f32,  // < 0.001 → greedy argmax
    pub top_k: usize,      // 0 = desabilitado
    pub top_p: f32,        // 1.0 = desabilitado
}
impl Default { temperature: 0.7, top_k: 40, top_p: 0.9 }
```

### Algoritmo

```
1. temperature < 0.001? → retorna argmax (greedy, sem rand)
2. Escala: logits * (1/temperature)
3. Softmax → probabilidades
4. Top-K: zera tudo abaixo do K-ésimo (select_nth_unstable: O(n))
5. Top-P: ordena desc, acumula cumsum, zera após >= top_p
6. Renormaliza (sum > 0, senão fallback greedy)
7. WeightedIndex::new(&probs).sample(&mut thread_rng()) → token_id
```

**Dependência:** `rand = "0.8"` (WeightedIndex + thread_rng)

### Casos especiais
- Todos os logits muito negativos após filtros → sum=0 → fallback greedy
- `top_k > vocab_size` → sem efeito (sem filtro)
- `top_p = 1.0` → sem filtro nucleus

---

## Gotchas
- **`to_host()` por camada**: cada camada faz 7× GPU→CPU (wq,wk,wv,wo,wgate,wup,wdown).
  Isso é o gargalo atual. rocBLAS eliminaria esses transfers.
- **RoPE layout "halves"**: convenção HuggingFace — par (x[i], x[i+head_dim/2]), NÃO (x[2i], x[2i+1]).
- **KV cache stores rotated K**: K é rotacionado ANTES do append. Não re-rotacione no decode.
- **Attention energy ≠ attention pattern**: é a norma de Q pós-RoPE, não a entropia da atenção.
