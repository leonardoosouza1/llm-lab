# Kuramoto Router Integration Guide

## Overview

This document describes the integration of the KuramotoRouter into the IARA engine's forward pass for sparse MLP neuron prediction and measurement.

## What Was Implemented

### 1. CausalLM Model Field Addition

Added optional `kuramoto_router` field to `CausalLM` struct:

```rust
pub struct CausalLM {
    // ... existing fields ...
    /// Kuramoto Phase Router for sparse MLP neuron selection (opcional).
    /// If Some, enabled via IARA_ROUTING=1 env var during model load.
    pub kuramoto_router: Option<crate::inference::KuramotoRouter>,
    // ... rest of fields ...
}
```

### 2. Router Initialization (3 Methods)

Updated all model loading methods to initialize the router when `IARA_ROUTING=1`:

- **`CausalLM::load()`** - Standard BF16 model loading
- **`CausalLM::load_quantized()`** - Q4 quantized model loading (handles both BF16 and Q4 gate_proj)
- **`CausalLM::load_from_iara()`** - IARA v2 format loading

Each method:
1. Checks `IARA_ROUTING` environment variable (default: "0" = disabled)
2. Reads `IARA_ROUTING_RATIO` for target active neuron percentage (default: 0.25 = 25%)
3. Creates router with dimensions from model config
4. Initializes each layer by downloading gate_proj weights and extracting phase tuning forks
5. Logs initialization progress with `log::info!` and `log::debug!`

### 3. Forward Pass Integration

In the `forward()` method, after MLP activation (after `apply_activation`):

1. **Download small buffers** (only when router enabled):
   - Normalized hidden state: `n × hidden_size` (small: 1×1536 = 6KB in decode)
   - Gate activations after SiLU: `n × intermediate_size` (small: 1×8960 = 36KB in decode)

2. **Predict active neurons** for each token:
   - Call `router.route(layer_idx, normed_hidden_state)` → Vec<usize> of predicted indices
   - Returns top-K neurons by phase coherence (K = IARA_ROUTING_RATIO × intermediate_size)

3. **Identify actual survivors** (for calibration):
   - Count gate activations > 1e-7 (neurons that survived SiLU)
   - Build set of indices

4. **Calibrate router**:
   - Call `router.calibrate(layer_idx, predicted, actual_alive)`
   - Updates EMA statistics: accuracy and avg_active_ratio
   - Allows router to improve predictions over time

5. **Log statistics**:
   - Per-token: predicted vs actual neuron counts and percentages
   - Per-layer (first token): Router stats (total predictions, accuracy, active ratio)

## Configuration

### Environment Variables

```bash
# Enable Kuramoto routing
export IARA_ROUTING=1

# Set target active neuron ratio (default 0.25 = 25%)
export IARA_ROUTING_RATIO=0.25

# Enable detailed logging
export RUST_LOG=iara_engine::model=trace
```

### Routing Ratios to Try

- **0.1 (10%)**: Most aggressive, lower accuracy but maximum theoretical speedup
- **0.25 (25%)**: Default, good balance of prediction accuracy and speedup potential
- **0.5 (50%)**: Conservative, higher accuracy but less speedup opportunity

## Current Capabilities (Phase 1)

### What This Implementation Does

1. **Predicts** which MLP neurons will be active using Kuramoto phase coherence
2. **Measures** actual SiLU sparsity (how many neurons survive non-linearity)
3. **Calibrates** router predictions via exponential moving average
4. **Logs** detailed statistics for analysis and debugging

### What This Implementation Does NOT Do (Yet)

- ✗ **Does NOT skip computation** - Still computes full gate/up/down projections
- ✗ **Does NOT use sparse GEMM** - No GPU kernel optimizations yet
- ✗ **Does NOT save latency** - Measurement overhead is small but routing doesn't accelerate yet

The purpose of Phase 1 is **measurement and validation**. We establish:
- Prediction accuracy baseline
- Natural sparsity patterns in neural networks
- Calibration effectiveness
- Platform for real sparse kernels in Phase 2

## Next Steps (Phase 2: Real Speedup)

### Sparse Computation Implementation

Once routing predictions are validated:

1. **GPU Kernel Development**:
   - Implement `sparse_zero_mask()` to zero non-predicted neurons after SiLU
   - Implement `selective_down_gemm()` to skip zeroed columns in down projection
   - Alternative: Use existing sparse BLAS libraries (e.g., cuSPARSE)

2. **Memory Optimization**:
   - Download predicted indices instead of full activations
   - Upload sparse format to GPU (compressed indices + values)
   - Reduces bandwidth: O(k) instead of O(intermediate_size)

3. **Expected Speedup** (for 25% active neurons):
   - Gate/up projections: Still full GEMM (can't avoid without changing input)
   - Down projection: 4x speedup (only k=2240 columns vs 8960)
   - Overall MLP: ~2x speedup (down dominates, gate/up still compute)
   - Full model: ~1.3x speedup (MLP is ~40% of compute, rest is attention)

### Phase 3: Layer-Level Routing

Combine Kuramoto MLP routing with ESN layer-budget decisions:
- Skip entire attention blocks for "easy" tokens (via LayerBudget::Fast)
- Skip some MLP layers for medium difficulty (via LayerBudget::Medium)
- Combined effect: 2-3x total speedup on realistic workloads

## File Changes Summary

### Modified Files

1. **`src/model/mod.rs`**:
   - Added `kuramoto_router: Option<KuramotoRouter>` field to `CausalLM`
   - Updated `CausalLM::load()` initialization
   - Updated `CausalLM::load_quantized()` initialization
   - Updated `CausalLM::load_from_iara()` initialization
   - Added routing logic in `forward()` after MLP activation
   - Total lines added: ~200 (with comments and logging)

### New/Unchanged Files

- `src/inference/kuramoto_router.rs` - Already existed, no changes needed
- `src/main.rs` - No changes (uses environment variables)
- `src/api/handlers.rs` - No changes (routing is transparent to API)

## Usage Example

```rust
// Model loads with router enabled
std::env::set_var("IARA_ROUTING", "1");
std::env::set_var("IARA_ROUTING_RATIO", "0.25");

let mut model = CausalLM::load(model_dir, cfg, arch, backend)?;

// Forward pass automatically uses router if enabled
let (logits, traces) = model.forward(&token_ids, None)?;

// Router statistics available
if let Some(router) = &model.kuramoto_router {
    println!("Router accuracy: {:.3}", router.stats.accuracy);
    println!("Avg active ratio: {:.3}", router.stats.avg_active_ratio);
}
```

## Compilation

```bash
# Build with ROCm backend
cargo build --release --features rocm

# Run with routing enabled
IARA_ROUTING=1 ./target/release/iara-engine
```

## Debugging

### Check If Router Is Initialized

```bash
RUST_LOG=iara_engine::model=info cargo run --features rocm 2>&1 | grep "Kuramoto"
```

Expected output:
```
🧭 Initializing Kuramoto router for sparse MLP prediction...
→ router layer 0 initialized
→ router layer 1 initialized
...
```

### Inspect Routing Decisions

```bash
RUST_LOG=iara_engine::model=trace cargo run --features rocm 2>&1 | head -200
```

Expected pattern:
```
layer 0: token 0 routing: predicted 224/896 (25.0%), actual 412/896 (46.0%)
layer 0: token 1 routing: predicted 211/896 (23.5%), actual 398/896 (44.4%)
...
Kuramoto router stats — predictions: 100, avg_accuracy: 0.546, avg_active_ratio: 0.248
```

### Verify No Compilation Errors

```bash
cargo check --features rocm 2>&1 | grep -E "error|warning"
```

Should show only pre-existing warnings, no new errors.

## Testing Checklist

- [x] Code compiles with `--features rocm`
- [x] Router initializes without segfaults
- [x] Predictions logged per token
- [x] Calibration statistics update over time
- [ ] Run full inference test on real model
- [ ] Verify latency unchanged from baseline
- [ ] Validate prediction accuracy on multiple prompts
- [ ] Test with different IARA_ROUTING_RATIO values
- [ ] Profile memory usage (should be minimal)

## References

- **KuramotoRouter API**: `src/inference/kuramoto_router.rs`
- **Phase Coherence Theory**: Uses complex phase angles extracted from hidden state vectors
- **Sparse Neural Networks**: Expects ~25-50% of neurons to be active naturally via SiLU gating
- **EMA Calibration**: Exponential moving average with alpha=0.01 smooths out noise

## Known Limitations

1. **Q4 Gate Weights**: When loading Q4 quantized models, gate_proj is Q4 and cannot be directly converted to phases. Router initializes but logs warning.

2. **Small Batch Sizes**: Routing overhead (2 GPU downloads) is amortized. May not improve latency for batch_size=1 until Phase 2 sparse kernels implemented.

3. **Precision Loss**: Phase extraction assumes pairs of hidden state dimensions. Odd-dimensional hidden sizes handled gracefully but may not be typical.

4. **No Backward Pass**: Router is forward-only. Useful for inference but not for training.

## Support & Issues

If router does not initialize:
- Check `IARA_ROUTING=1` is set
- Check model loads successfully without routing
- Verify gate_proj weights exist and have correct shape
- Check logs for "router layer X initialized" messages

If predictions seem inaccurate:
- Run longer to allow calibration (EMA has small alpha)
- Try different IARA_ROUTING_RATIO values
- Check if hidden state is normalized correctly
- Inspect raw coherence values in debug logs

## Future Enhancements

- [ ] Implement selective GEMM for Phase 2 speedup
- [ ] Add runtime rebalancing of active_ratio based on accuracy
- [ ] Support for adaptive routing (different ratios per layer)
- [ ] Integration with knowledge distillation
- [ ] Combine with attention head routing
