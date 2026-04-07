# Kuramoto Router - Quick Start Guide

## One-Minute Overview

The Kuramoto Router predicts which MLP neurons will be active in a transformer model. This enables future sparse computation optimizations.

**Current state**: Measurement and prediction accuracy validation (no speedup yet).

## Enable It

```bash
# Build
cargo build --release --features rocm

# Run with routing enabled
export IARA_ROUTING=1
./target/release/iara-engine
```

## Disable It (Default)

```bash
# Default: routing disabled, zero overhead
./target/release/iara-engine
```

## Configure Active Neurons

```bash
# Default 25% of neurons
export IARA_ROUTING_RATIO=0.25

# Try 10% (aggressive)
export IARA_ROUTING_RATIO=0.10

# Try 50% (conservative)
export IARA_ROUTING_RATIO=0.50
```

## Watch the Logs

```bash
# See detailed per-token routing
export RUST_LOG=iara_engine::model=trace
./target/release/iara-engine 2>&1 | head -100

# See initialization only
export RUST_LOG=iara_engine::model=info
./target/release/iara-engine 2>&1 | head -20
```

## Expected Output

```
🧭 Initializing Kuramoto router for sparse MLP prediction...
→ router layer 0 initialized
→ router layer 1 initialized
→ router layer 2 initialized
...
→ router layer 31 initialized

[During inference]
layer 0: token 0 routing: predicted 224/896 (25.0%), actual 412/896 (46.0%)
layer 1: token 0 routing: predicted 239/896 (26.7%), actual 387/896 (43.2%)
...
Kuramoto router stats — predictions: 32, avg_accuracy: 0.541, avg_active_ratio: 0.251
```

## What to Look For

### Prediction Accuracy
- `avg_accuracy` should improve over time (starts ~0.5, can reach ~0.7+)
- Shows how well router predicts real active neurons
- Improves with token count as EMA calibrates

### Active Ratio
- `avg_active_ratio` should match your `IARA_ROUTING_RATIO`
- If you set 0.25 (25%), should see ~0.25 in stats
- Confirms routing is balanced across layers

### Per-Token Statistics
- `predicted` = neurons router thinks will be active
- `actual` = neurons that survived SiLU gating
- Mismatch is normal and measured via accuracy

## Performance Impact (Phase 1)

- **Latency**: +1-2ms per 256 tokens (download overhead)
- **Memory**: +100KB (router state)
- **Real speedup**: 0% (no computation skipped yet)

Phase 2 will add sparse kernels for real speedup (2-4x MLP, ~1.3x model).

## Try Different Configurations

```bash
# Test 1: Default routing
IARA_ROUTING=1 IARA_ROUTING_RATIO=0.25 ./target/release/iara-engine

# Test 2: Aggressive routing
IARA_ROUTING=1 IARA_ROUTING_RATIO=0.10 ./target/release/iara-engine

# Test 3: Conservative routing
IARA_ROUTING=1 IARA_ROUTING_RATIO=0.50 ./target/release/iara-engine

# Test 4: Baseline (no routing)
IARA_ROUTING=0 ./target/release/iara-engine
```

Measure latency and accuracy for each. Higher ratio = better accuracy, lower potential speedup.

## Understanding the Math

### Kuramoto Phase Coherence

1. Extract phase angles from hidden state pairs
2. Compare with "tuning fork" phases (from gate_proj weights)
3. Compute coherence = |mean(cos(phase_diff), sin(phase_diff))|
4. Keep top-K neurons by coherence

### SiLU Gating

- Gate activation: silu(W_gate × hidden) * (W_up × hidden)
- Many gates → 0 (naturally sparse)
- Router predicts which will survive
- Accuracy measures prediction skill

## Debugging

### Router didn't initialize?

```bash
RUST_LOG=iara_engine::model=debug ./target/release/iara-engine 2>&1 | grep -i "kuramoto\|router"
```

Should see:
- "Initializing Kuramoto router"
- "router layer X initialized" for each layer

### Accuracy very low?

- Might need more tokens for calibration (EMA is slow)
- Try different `IARA_ROUTING_RATIO`
- Check if model is quantized (Q4 gate_proj not supported yet)

### No log output?

```bash
export RUST_LOG=trace
./target/release/iara-engine 2>&1 | head -500
```

Look for errors or initialization messages.

## Code Location

| Component | File | Lines |
|-----------|------|-------|
| Router API | `src/inference/kuramoto_router.rs` | 552 |
| Integration | `src/model/mod.rs` | ~300, 600-650, 982-1015, 1365-1400, 1841-1898 |
| Documentation | `KURAMOTO_ROUTER_INTEGRATION.md` | - |

## Vocabulary

- **Router**: Predicts active neurons using phase coherence
- **Tuning fork**: Phase angles extracted from gate_proj rows
- **Coherence**: How well hidden state matches tuning fork pattern
- **Active ratio**: Fraction of neurons to predict as active (0.0-1.0)
- **Calibration**: EMA update of router accuracy statistics
- **Sparsity**: Fraction of neurons with zero activation

## Next Phase

Once Phase 1 measurement is validated, Phase 2 will:

1. Implement sparse GPU kernels
2. Actually skip zero neurons in down-projection
3. Achieve 2-4x MLP speedup
4. Result in ~1.3x overall model speedup

This guide tests the foundation for that work.

## Questions?

- Check `KURAMOTO_ROUTER_INTEGRATION.md` for detailed API
- Check `IMPLEMENTATION_SUMMARY.md` for design decisions
- Review logs with `RUST_LOG=iara_engine::model=trace`
- Inspect `src/model/mod.rs` lines 1841-1898 for integration code
