# IARA Trace API Reference

## Endpoint

```
POST /api/trace
```

## Request Format

```json
{
  "prompt": "your text here"
}
```

### Example Request

```bash
curl -X POST http://localhost:8000/api/trace \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'
```

### JavaScript Example

```javascript
const response = await fetch('/api/trace', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'Your prompt here' })
});
const trace = await response.json();
```

## Response Format

Complete BrainTrace object with all generation steps and layer traces.

### Full Response Structure

```json
{
  "prompt": "Hello world",
  "response": "Hello world is a classic",
  "steps": [
    {
      "token_id": 12345,
      "token_str": "Hello",
      "top_logits": [
        [12345, 4.5],
        [67890, 3.2],
        [11111, 2.8],
        [22222, 1.5],
        [33333, 0.9]
      ],
      "layer_traces": [
        {
          "layer_idx": 0,
          "attention_energy": 0.234,
          "mlp_energy": 0.456,
          "residual_norm": 0.678,
          "max_activation": 1.356,
          "hot_neurons": [5, 12, 23, 45, 67],
          "time_us": 87,
          "mlp_activation_rate": 0.62,
          "phase_coherence": 0.78,
          "hidden_norm": 1.23
        },
        {
          "layer_idx": 1,
          "attention_energy": 0.145,
          "mlp_energy": 0.312,
          "residual_norm": 0.521,
          "max_activation": 1.042,
          "hot_neurons": [3, 8, 15, 31],
          "time_us": 87,
          "mlp_activation_rate": 0.48,
          "phase_coherence": 0.65,
          "hidden_norm": 1.15
        },
        // ... 26 more layers (total 28) ...
        {
          "layer_idx": 27,
          "attention_energy": 0.876,
          "mlp_energy": 0.432,
          "residual_norm": 0.923,
          "max_activation": 1.846,
          "hot_neurons": [101, 145, 203],
          "time_us": 87,
          "mlp_activation_rate": 0.71,
          "phase_coherence": 0.82,
          "hidden_norm": 1.31
        }
      ],
      "active_layers": [0, 2, 5, 7, 12, 15, 18, 20, 23, 26, 27],
      "time_ms": 25
    },
    // ... one step per generated token ...
  ],
  "iara_snapshots": [],
  "layer_activation_path": [],
  "energy_trajectory": [],
  "attention_heatmap": [],
  "finalized_at": "2026-04-07T09:36:00Z"
}
```

## Response Fields Explained

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Original input prompt |
| `response` | string | Complete generated response |
| `steps` | GenerationStep[] | One per generated token (or prompt token) |
| `finalized_at` | ISO8601 string | When trace was completed |
| `iara_snapshots` | array | (optional) IARA simulation state |
| `layer_activation_path` | array | (optional) Layer activation sequence |
| `energy_trajectory` | array | (optional) Energy over time |
| `attention_heatmap` | array | (optional) Attention visualization data |

### GenerationStep Fields

| Field | Type | Description |
|-------|------|-------------|
| `token_id` | u32 | Token ID from vocabulary |
| `token_str` | string | Human-readable token text |
| `top_logits` | [u32, f32][] | Top 5 predictions: [token_id, logit_value] |
| `layer_traces` | LayerTrace[] | Exactly 28 elements (one per layer) |
| `active_layers` | usize[] | Layer indices above activation threshold |
| `time_ms` | u128 | Inference latency for this token (milliseconds) |

### LayerTrace Fields

| Field | Type | Description | Visualization Use |
|-------|------|-------------|-------------------|
| `layer_idx` | usize | Layer number (0-27) | Layer position |
| `attention_energy` | f32 | Energy in attention head | Tooltip display |
| `mlp_energy` | f32 | Energy in MLP feed-forward | Tooltip display |
| `residual_norm` | f32 | Magnitude of residual changes (0-2 typical) | **Bar color** (blue→red) |
| `max_activation` | f32 | Peak neuron activation | Used in stats |
| `hot_neurons` | usize[] | Indices of active neurons (0-255) | **Neuron grid coloring** |
| `time_us` | u128 | Microseconds spent in layer | Latency breakdown |
| `mlp_activation_rate` | f32 | Sparsity: % of neurons alive (0-1) | **Bar width** |
| `phase_coherence` | f32 | Synchronization metric (0-1) | **Dot color** (green/yellow/red) |
| `hidden_norm` | f32 | Hidden state magnitude | Temporal tracking |

## Color Mapping

### residual_norm → Bar Color

```
residual_norm value:     0.0          max/2          max
Color gradient:        BLUE        ORANGE         RED
CSS class:            cold       moderate        hot
RGB approx:         #003366      #ff6633        #ff0000
```

### phase_coherence → Dot Color

```
phase_coherence:     0.0-0.4      0.4-0.6       0.6-1.0
Color:               RED          YELLOW        GREEN
Meaning:            Chaotic      Medium       Coherent
```

### mlp_activation_rate → Bar Width

```
Bar width = mlp_activation_rate × 100%
Example: 0.62 → 62% width
```

## Mock Data Fallback

If server returns error or is unavailable, the visualization falls back to mock data (client-side generation). This allows testing without a running server.

Mock data characteristics:
- 28 layers per token
- Realistic activation patterns (higher in middle layers)
- Synthetic attention with temporal coherence
- Random top-5 logits
- 1-256 hot neurons per layer randomly distributed

## Expected HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response normally |
| 400 | Bad request | Check prompt format |
| 500 | Server error | Check server logs, fall back to mock |
| 503 | Service unavailable | Server loading model |

## Error Handling

When a trace fails:

```javascript
try {
  const response = await fetch('/api/trace', { /* ... */ });
  if (!response.ok) throw new Error(`Server error: ${response.status}`);
  const trace = await response.json();
  // Process trace
} catch (error) {
  console.error('Trace failed:', error.message);
  // Fall back to mock data or show error message
}
```

## Performance Expectations

| Metric | Typical Value |
|--------|---------------|
| Response time (3B model) | 100-500ms per token |
| Total trace size (10 tokens) | 200-500KB JSON |
| Number of layers | Always 28 (hardcoded) |
| Neurons per layer | 256 (from 8×8×4 feed-forward) |
| Max tokens | Limited by memory (1000+ typical) |

## Data Validation

Before processing response, verify:

```javascript
function validateTrace(trace) {
  if (!trace.prompt || !trace.response) return false;
  if (!Array.isArray(trace.steps)) return false;

  for (const step of trace.steps) {
    if (!step.layer_traces || step.layer_traces.length !== 28) return false;
    for (const layer of step.layer_traces) {
      if (layer.layer_idx < 0 || layer.layer_idx >= 28) return false;
      if (!Array.isArray(layer.hot_neurons)) return false;
    }
  }
  return true;
}
```

## Extended Fields

The visualization expects these additional fields in LayerTrace for full functionality:

- `mlp_activation_rate` (f32): Neuron sparsity
- `phase_coherence` (f32): Synchronization metric
- `hidden_norm` (f32): Hidden state magnitude

**Optional**: If not provided by server:
- `mlp_activation_rate` defaults to `max_activation / 2`
- `phase_coherence` defaults to `0.5 + random(0.3)`
- `hidden_norm` defaults to `1.0`

The visualization will still work with just the core fields, but uses estimates for missing extended fields.

## Integration Steps

1. **Implement the endpoint**:
   ```rust
   POST /api/trace {prompt: String} → GenerationStep[]
   ```

2. **Generate LayerTrace for each layer**:
   - Capture during forward pass
   - Include residual_norm, mlp_energy, attention_energy
   - Track hot_neurons in MLP layer
   - Extend with mlp_activation_rate, phase_coherence

3. **Return BrainTrace object**:
   - One entry per token in response
   - Each step with all 28 layer traces
   - Set finalized_at to current timestamp

4. **Handle errors gracefully**:
   - Return 500 on inference errors
   - Return 400 on invalid prompt
   - Client will show error message

## Example: Minimal Working Response

```json
{
  "prompt": "Hello",
  "response": "Hello world",
  "steps": [
    {
      "token_id": 123,
      "token_str": "world",
      "top_logits": [[123, 4.5], [456, 3.2], [789, 2.1], [101, 1.5], [112, 0.8]],
      "layer_traces": [
        {
          "layer_idx": 0,
          "attention_energy": 0.1,
          "mlp_energy": 0.2,
          "residual_norm": 0.3,
          "max_activation": 0.6,
          "hot_neurons": [1, 2, 3],
          "time_us": 100,
          "mlp_activation_rate": 0.5,
          "phase_coherence": 0.7,
          "hidden_norm": 1.0
        },
        // ... 27 more layers ...
      ],
      "active_layers": [0, 5, 10, 15, 20, 25],
      "time_ms": 50
    }
  ],
  "finalized_at": "2026-04-07T09:36:00Z"
}
```

This minimal response will render all visualization components.

## Debugging Tips

**Check response in browser console**:
```javascript
// After a trace is loaded
console.log(JSON.stringify(currentTrace, null, 2));
```

**Validate structure**:
```javascript
console.log('Steps:', currentTrace.steps.length);
console.log('Layers per step:', currentTrace.steps[0].layer_traces.length);
console.log('Sample layer:', currentTrace.steps[0].layer_traces[0]);
```

**Check server response directly**:
```bash
curl -X POST http://localhost:8000/api/trace \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}' | jq .
```

## CORS Considerations

If serving visualization from different origin than API server:

```
Server Response Headers:
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: POST, GET, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

Or more restrictively:
```
Access-Control-Allow-Origin: https://your-domain.com
```
