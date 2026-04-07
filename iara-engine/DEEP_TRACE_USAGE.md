# Deep Trace API Usage Guide

## Quick Start

The deep trace API captures detailed per-token telemetry showing exactly how each layer transforms the hidden state as a token flows through the model.

### Basic Request

```bash
curl -X POST http://localhost:3000/api/trace \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "token_index": 4
  }'
```

This traces the 5th token (index 4) of the prompt, after prefilling with tokens 0-3.

### Response Structure

```json
{
  "token_id": 5387,
  "token_str": "Paris",
  "time_ms": 1250,
  "layers": [
    {
      "layer_idx": 0,
      "hidden_norm": 42.356,
      "attn_entropy": 2.143,
      "attn_top_positions": [
        [4, 0.542],
        [3, 0.287],
        [2, 0.089],
        [1, 0.056],
        [0, 0.026]
      ],
      "mlp_activation_rate": 78.345,
      "mlp_top_neurons": [
        [2341, 1.234],
        [2412, 1.189],
        [1876, 0.987],
        [3421, 0.876],
        [2567, 0.845],
        [4123, 0.754],
        [1234, 0.723],
        [3456, 0.687],
        [2890, 0.654],
        [1999, 0.623]
      ],
      "residual_delta": 15.678,
      "phase_coherence": 0.456,
      "time_us": 0
    },
    // ... more layers ...
  ],
  "output": {
    "logit_entropy": 5.234,
    "top_tokens": [
      [5387, 0.325],
      [1234, 0.198],
      [5678, 0.154],
      [9012, 0.123],
      [3456, 0.089],
      [7890, 0.067],
      [2341, 0.023],
      [4567, 0.012],
      [8901, 0.005],
      [1111, 0.003]
    ],
    "confidence": 0.325,
    "chosen_token_rank": 0
  }
}
```

## Understanding the Metrics

### Layer Metrics

#### `hidden_norm` (float)
L2 norm of the hidden state entering the layer.

**Interpretation:**
- **Low (0-10)**: Quiet input, information is sparse
- **Medium (10-30)**: Normal activation level
- **High (>50)**: Very active input, potential saturation

**Use Case:** Detect if layers are receiving sufficient signal.

#### `attn_entropy` (float, range: [0, ln(seq_len)])
Shannon entropy of the attention score distribution. Measures focus.

**Interpretation:**
- **0.0-1.0**: Highly focused (attending to few positions)
- **1.0-3.0**: Moderate focus
- **3.0+**: Diffuse attention (attending broadly)

**Example:** If sequence length is 100:
- entropy = 0.5 → very focused (maybe 1-2 positions get >80% attention)
- entropy = 2.3 → moderate (multiple positions contribute)
- entropy = 4.6 → completely uniform (all positions equal)

**Use Case:** Understand if attention is becoming too narrow or too broad.

#### `attn_top_positions` (array of [position, weight])
Top-5 positions that received the highest attention weights.

**Format:** `[[position_idx, attention_weight], ...]`

**Interpretation:**
- Position indices show WHAT the model is attending to
- Weights show HOW MUCH attention each position receives
- Should sum to ~1.0 (softmax normalization)

**Use Case:** Debug attention patterns; verify causal masking is working.

#### `mlp_activation_rate` (float, percent: 0-100)
Percentage of MLP neurons that have non-zero output after SiLU gating.

**Interpretation:**
- **0-20%**: Very sparse (only critical neurons fire)
- **40-60%**: Moderate (typical activation level)
- **80-100%**: Dense (most neurons active)

**Use Case:** Monitor MLP sparsity; detect potential saturation or under-utilization.

#### `mlp_top_neurons` (array of [neuron_idx, activation])
Top-10 most activated MLP neurons by absolute value.

**Format:** `[[neuron_index, activation_magnitude], ...]`

**Interpretation:**
- Shows WHICH neurons are most active
- Activations can be positive or negative
- High activation = significant contribution to residual

**Use Case:** Identify key neurons; study feature specialization across layers.

#### `residual_delta` (float)
L2 norm of the change this layer made to the hidden state.

**Interpretation:**
- **0.0-2.0**: Minimal change (skip connection dominates)
- **5.0-15.0**: Significant transformation
- **>30.0**: Major modification (unusual, might indicate issues)

**Use Case:** Detect if layers are contributing meaningfully or if gradients might vanish.

#### `phase_coherence` (float, range: [0.0, 1.0])
Kuramoto order parameter measuring phase alignment in the hidden state.

Interprets hidden state as complex pairs: `(h[0], h[1]), (h[2], h[3]), ...`

**Interpretation:**
- **0.0-0.3**: Random/chaotic phases
- **0.3-0.7**: Partial alignment
- **0.7-1.0**: Strong phase synchronization

**Use Case:** Monitor information structure; detect when phases become organized vs chaotic.

### Output Metrics

#### `logit_entropy` (float, range: [0, ln(vocab_size)])
Shannon entropy of the final logit distribution (after softmax).

**Interpretation:**
- **0.0-2.0**: Model is very confident (few likely tokens)
- **3.0-5.0**: Moderate confidence (several plausible tokens)
- **>7.0**: Model is uncertain (broad distribution)

**Use Case:** Assess model confidence; detect over/under-confidence.

#### `top_tokens` (array of [token_id, probability])
Top-10 most likely tokens with their probabilities.

**Format:** `[[token_id, probability], ...]`

**Sum:** Should equal ~1.0 (but not exactly since we only show top-10)

**Use Case:** See alternative predictions; understand model's top choices.

#### `confidence` (float, range: [0.0, 1.0])
Probability assigned to the chosen token.

**Interpretation:**
- **0.9-1.0**: Very confident (clear choice)
- **0.5-0.9**: Moderately confident
- **0.0-0.2**: Very uncertain (close competition)

**Use Case:** Measure how certain the model is about its prediction.

#### `chosen_token_rank` (int, range: [-1, 9])
Rank of the chosen token in the top-10 list.

**Values:**
- **0**: Top choice
- **1-9**: 2nd through 10th choice
- **-1**: Not in top-10

**Use Case:** Verify top choice is being selected; detect when next token beats primary.

## Practical Examples

### Example 1: Analyze Attention Patterns

```python
import requests
import json

response = requests.post(
    "http://localhost:3000/api/trace",
    json={
        "prompt": "The quick brown fox jumps over",
        "token_index": 6  # Trace "the" at end
    }
)

trace = response.json()

# Check middle layer's attention
middle_layer = trace['layers'][len(trace['layers']) // 2]
print(f"Entropy: {middle_layer['attn_entropy']:.3f}")
print("Top positions:", middle_layer['attn_top_positions'])

# Interpret: is attention sharp or diffuse?
if middle_layer['attn_entropy'] < 1.5:
    print("→ Very focused attention")
elif middle_layer['attn_entropy'] > 3.0:
    print("→ Diffuse attention")
else:
    print("→ Moderate focus")
```

### Example 2: Check MLP Sparsity

```python
response = requests.post(
    "http://localhost:3000/api/trace",
    json={"prompt": "Hello world", "token_index": 0}
)

trace = response.json()

for layer in trace['layers']:
    sparsity = 100 - layer['mlp_activation_rate']
    print(f"Layer {layer['layer_idx']:2d}: {sparsity:5.1f}% sparse")
```

### Example 3: Monitor Confidence Degradation

```python
# Trace multiple tokens in sequence to see confidence change
prompts = [
    {"prompt": "The", "token_index": 0},
    {"prompt": "The answer", "token_index": 1},
    {"prompt": "The answer is", "token_index": 2},
    {"prompt": "The answer is forty-two", "token_index": 3},
]

for req in prompts:
    response = requests.post("http://localhost:3000/api/trace", json=req)
    trace = response.json()
    print(f"Token {trace['token_id']}: confidence={trace['output']['confidence']:.3f}")
```

### Example 4: Identify High-Impact Neurons

```python
response = requests.post(
    "http://localhost:3000/api/trace",
    json={"prompt": "What is AI?", "token_index": 2}
)

trace = response.json()

# Find which neurons are most consistently active
neuron_frequency = {}
for layer in trace['layers']:
    for neuron_idx, activation in layer['mlp_top_neurons'][:3]:  # Top 3 per layer
        neuron_frequency[neuron_idx] = neuron_frequency.get(neuron_idx, 0) + 1

print("Most frequently activated neurons:")
for neuron_idx, count in sorted(neuron_frequency.items(), key=lambda x: -x[1]):
    print(f"  Neuron {neuron_idx}: active in {count} layers")
```

## Performance Notes

- **Typical latency**: 800-1200ms per trace (vs ~350ms standard forward)
- **GPU memory**: Uses same scratchpad as forward()
- **CPU memory**: ~48 MB for result JSON (12 layers, 100 token sequence)
- **Bottleneck**: GPU→CPU downloads (unavoidable for rich telemetry)

## Tips for Debugging

1. **Entropy anomalies**: If entropy suddenly drops in a layer, attention became focused
2. **Residual delta spikes**: Large deltas indicate heavy layer modification
3. **Confidence oscillation**: Confidence bouncing up/down suggests model uncertainty
4. **Sparsity patterns**: Consistent low activation might indicate dead neurons
5. **Phase coherence trend**: Coherence increasing through layers suggests structure formation

## Error Handling

```bash
# Empty prompt
{"error": "Empty prompt"}

# Model not loaded
{"error": "No model loaded"}

# Token index out of range
# (Automatically uses last token of prompt)

# Deep trace fails
{"error": "Deep trace failed: ..."}
```

## Integration Examples

### JavaScript/TypeScript
```javascript
const response = await fetch('http://localhost:3000/api/trace', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "The answer is",
    token_index: 2
  })
});
const trace = await response.json();
console.log(trace.output.confidence);
```

### Python with Pandas
```python
import requests
import pandas as pd

response = requests.post('http://localhost:3000/api/trace', json={...})
trace = response.json()

# Convert to DataFrame for analysis
layers_df = pd.DataFrame(trace['layers'])
print(layers_df[['layer_idx', 'hidden_norm', 'attn_entropy', 'mlp_activation_rate']])
```

### cURL with jq
```bash
curl -X POST http://localhost:3000/api/trace \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","token_index":0}' | \
  jq '.layers[] | {idx: .layer_idx, entropy: .attn_entropy, delta: .residual_delta}'
```
