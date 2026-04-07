# IARA Deep Trace Visualization

Interactive HTML visualization for analyzing the deep trace data from the IARA inference engine. Shows how tokens flow through transformer layers and what activations occur at each step.

## File

- **trace.html** - Single-file self-contained visualization (32KB)

## Features

### 1. Layer Flow Graph
Vertical visualization of all 28 transformer layers showing per-token activation:

- **Bar width**: MLP neuron activation rate (% of neurons alive)
- **Bar color**: Residual delta intensity (blue=cold/passive, red=hot/active)
- **Phase dot**: Phase coherence indicator (green=coherent, yellow=medium, red=chaotic)
- **Click**: Select a layer to inspect its neuron activations and attention patterns

The bars represent how much each layer transformed the token representation.

### 2. Attention Heatmap (Right Panel)
When a layer is selected, displays the attention pattern for that layer:
- X-axis: Context positions attended to (0 to current token position)
- Y-axis: Attention heads (8 heads)
- Color intensity: Attention weight (brighter = stronger attention)

### 3. MLP Neuron Activation Map (Bottom Left)
Grid of 256 neurons (16x16) for the selected layer showing:
- **Bright/Green**: Active neurons (firing during forward pass)
- **Gray**: Dormant neurons (inactive but not permanently dead)
- **Dark**: Dead neurons (no activation)

### 4. Summary Statistics (Right Panel)
Real-time metrics for the selected token:
- **Active Layers**: Count of layers above activation threshold
- **Avg MLP Activation**: Average neuron activation rate across all layers
- **Logit Entropy**: Confidence/uncertainty in predictions
- **Confidence**: Probability of top predicted token
- **Top Predictions**: Top-5 tokens with their logit probabilities
- **Forward Time**: Inference latency in milliseconds

## Usage

### With Mock Data (Default)
1. Open `trace.html` in a browser
2. Enter a prompt in the text field (default: "Hello world")
3. Click **Trace** button to generate mock trace data
4. Select tokens from the dropdown to analyze different steps
5. Click layer bars to inspect individual layer activations

### With Real Server
1. Start the IARA inference server (it will serve static files from this directory)
2. Uncheck the **Mock Data** checkbox
3. Enter a prompt and click **Trace**
4. The page will POST to `/api/trace` with the prompt
5. Results will be visualized in real-time

## API Endpoint

The page expects a `POST /api/trace` endpoint that accepts:

```json
{
  "prompt": "your text here"
}
```

And returns a trace object with this structure:

```json
{
  "prompt": "string",
  "response": "string",
  "steps": [
    {
      "token_id": 12345,
      "token_str": "token",
      "top_logits": [[token_id, logit_value], ...],
      "layer_traces": [
        {
          "layer_idx": 0,
          "attention_energy": 0.5,
          "mlp_energy": 0.3,
          "residual_norm": 0.7,
          "max_activation": 1.4,
          "hot_neurons": [1, 5, 12, ...],
          "time_us": 87,
          "mlp_activation_rate": 0.6,
          "phase_coherence": 0.8,
          "hidden_norm": 1.2
        },
        ...28 layers total...
      ],
      "active_layers": [0, 2, 5, ...],
      "time_ms": 25
    },
    ...one per token in response...
  ],
  "finalized_at": "2026-04-07T09:36:00Z"
}
```

## Mock Data

The visualization includes a realistic mock data generator that creates plausible traces without needing a server:

- Simulates 28 layers with realistic layer-specific activation patterns
- Generates synthetic attention patterns with temporal coherence
- Creates MLP activation patterns that vary per layer
- Includes top-5 logit predictions

Perfect for testing the visualization UI before integration with the actual inference engine.

## Styling

Dark theme matching the IARA dashboard:
- Dark background (#0a0e27)
- Cyan/green accents (#00d9ff, #00ff99)
- Red/orange highlights for high activations
- Monospace font (Courier New) for data display

## Performance Notes

- Renders 28 layers with smooth interactions
- Tooltip support for detailed per-layer metrics
- Responsive scrolling for long traces
- All visualization uses D3.js for smooth transitions and interactivity

## Integration with IARA Engine

To serve this from the IARA engine's static file server:

1. Place `trace.html` in a `frontend/` directory
2. Configure the server to serve static files from `frontend/`
3. Ensure the server exposes a `POST /api/trace` endpoint
4. Access via `http://localhost:PORT/trace.html` (or equivalent)

The visualization will automatically detect whether to use mock data or real server based on the checkbox state.
