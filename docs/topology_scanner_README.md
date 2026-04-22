# IARA Neuron Topology Scanner

Probes the IARA HTTP inference server with 200 diverse text samples and builds
a statistical map of neuron activation across the model's MLP layers.

## Model / Architecture

| Field              | Value                        |
|--------------------|------------------------------|
| Model              | qwen2.5-1.5b                 |
| Server             | `http://localhost:3000`      |
| Layers             | 28                           |
| MLP neurons / layer | 8,960                       |
| Total neurons      | 250,880                      |

## Files

```
topology_scanner/
├── scanner.py           — main async scanner (this README explains it)
└── neuron_topology.json — output written after each full run
```

## Requirements

```bash
pip install httpx tqdm
```

## Usage

```bash
# Make sure IARA is running first:
#   cd ~/projects/LLM/llm-lab && ./run_iara.sh

python scanner.py
```

Expected output:

```
IARA Neuron Topology Scanner
Target: http://localhost:3000  |  Model: qwen2.5-1.5b
Architecture: 28 layers x 8960 neurons = 250,880 total
Dataset: 200 samples  |  max_tokens=30  temp=0.0

Checking server health... OK

Scanning: 100%|████████████████████| 200/200 [05:12<00:00, 1.56s/sample]

Scan complete: 200 samples processed, 0 errors
Results summary:
  Recorded entries : 28
  Hot  (>80% freq) : 3
  Dead (<1%  freq) : 0
  Normal           : 25

Saved -> /home/leonardo/projects/LLM/llm-lab/topology_scanner/neuron_topology.json
```

## Output JSON Schema

```json
{
  "meta": {
    "model": "qwen2.5-1.5b",
    "total_samples": 200,
    "num_layers": 28,
    "neurons_per_layer": 8960,
    "total_neurons": 250880,
    "recorded_neurons": 28,
    "used_fallback": true,
    "fallback_note": "...",
    "scanned_at": "2026-04-12T..."
  },
  "summary": {
    "dead_neurons": 0,
    "hot_neurons": 3,
    "normal_neurons": 25
  },
  "neurons": [
    {
      "neuron_id": 8960,
      "layer_idx": 1,
      "neuron_idx": 0,
      "activation_count": 198,
      "freq_pct": 99.0,
      "mean_energy": 3.1415,
      "classification": "hot"
    }
  ],
  "top_hot":  [ ... top 20 hottest ... ],
  "top_dead": [ ... top 20 deadest ... ]
}
```

### Neuron ID encoding

```
neuron_id = layer_idx * 8960 + neuron_idx
```

This gives each of the 250,880 neurons a unique flat integer ID, making it
easy to cross-reference results with weight tensors or visualizations.

### Classifications

| Class    | Criterion               | Interpretation                          |
|----------|-------------------------|-----------------------------------------|
| `hot`    | activation freq > 80%   | Almost always active — polysemantic hub |
| `dead`   | activation freq < 1%    | Rarely fires — candidate for pruning    |
| `normal` | 1% ≤ freq ≤ 80%         | Domain-specific activation              |

## Two-Mode Operation

### Mode A — Full neuron data (preferred)

When the IARA server populates `LayerTrace.hot_neurons`, each entry in the
output represents a specific neuron within a layer.

To enable this, `forward.rs` must fill `hot_neurons` with the actual top-k
neuron indices from the MLP gate activations, e.g.:

```rust
// In forward.rs, after running the MLP block:
let top_k = 10;
let mut indexed: Vec<(usize, f32)> = gate_activations
    .iter().copied().enumerate().collect();
indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
trace.hot_neurons = indexed.iter().take(top_k).map(|(i, _)| *i).collect();
```

### Mode B — Layer-level fallback (current default)

When `hot_neurons` is empty (placeholder), the scanner falls back to
`most_active_layers` and `layer_timeline` from the `BrainAnalysis` response.
Each layer is represented by a single aggregate entry (`neuron_idx = 0`).

`meta.used_fallback = true` signals this mode in the output JSON.

## IARA API Reference

| Endpoint                    | Method | Description                              |
|-----------------------------|--------|------------------------------------------|
| `/health`                   | GET    | Server readiness                         |
| `/api/generate`             | POST   | Run inference, stores trace internally   |
| `/api/debug/last-trace`     | GET    | Return `BrainAnalysis` for last generate |

### POST /api/generate payload

```json
{
  "model":       "qwen2.5-1.5b",
  "prompt":      "...",
  "max_tokens":  30,
  "temperature": 0.0,
  "stream":      false
}
```
