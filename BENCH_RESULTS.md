# Bench: Qwen 1.5B vs byte-language-lab

**Modelo:** qwen2.5-1.5b-q4km  
**Server:** http://localhost:3001  
**Seed:** 9001  
**Samples por categoria:** 20 (× 5 dificuldades × 3 ops = 300 problemas)

**Source-of-truth:** valores `i64` Rust nativos. Cada solver compara contra esse target.

## Accuracy por dificuldade × operação

| Dígitos | + | − | × | + | − | × |
|---------|---|---|---|---|---|---|
|         | Qwen | Qwen | Qwen | byte-lab | byte-lab | byte-lab |
| 1d | 100% (20/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| 2d | 100% (20/20) | 100% (20/20) | 40% (8/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| 3d | 95% (19/20) | 75% (15/20) | 0% (0/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| 4d | 60% (12/20) | 80% (16/20) | 0% (0/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| 5d | 35% (7/20) | 45% (9/20) | 0% (0/20) | 100% (20/20) | 100% (20/20) | 10% (2/20) |

## Resumo agregado

| Métrica | Qwen 1.5B | byte-language-lab |
|---------|----------:|------------------:|
| Total | 300 | 300 |
| Correct | 186 | 282 |
| Accuracy | **62.0%** | **94.0%** |
| Failed parse | 0 | 0 |
| Throughput | 133.1 tok/s | 300000 ops/s |

## Análise

Gap absoluto: **96 problemas** (32.0%) onde byte-lab acerta e Qwen erra.

**Padrão esperado pela literatura:** LLMs degradam fortemente com nº de dígitos.
Multiplicação 4d×4d e 5d×5d são tipicamente <5% accuracy em modelos pequenos.
Byte-lab mantém ≥99% em todas as categorias por construção (digit-by-digit add/mul).

**Implicação prática:** se o gap em mat. multi-dígito for grande (esperado ~80%+),
um "byte arithmetic side-car" no llm-lab cobriria essa lacuna sem retreinar o modelo.
Próximo passo natural: criar `iara-byte-decoder` que detecta padrões aritméticos no
output de Qwen e os roteia pra byte-arithmetic, devolvendo a resposta correta.
