# Trace Visualization - Technical Implementation Notes

## Architecture

### Single-File Design
- **trace.html** is a completely self-contained file (32KB)
- No external dependencies except D3.js (loaded from CDN)
- All CSS, HTML, and JavaScript inline for easy deployment
- Works as a static file served directly by the IARA engine

### Data Flow

```
User Input
    ↓
tracePrompt()
    ↓
[Check Mock Flag]
    ├─→ Generate Mock Data (client-side)
    └─→ POST /api/trace (server request)
    ↓
Store in currentTrace
    ↓
updateVisualization()
    ├─→ updateTokenSelector()
    ├─→ updateLayerFlow()
    ├─→ updateMlpNeurons()
    ├─→ updateAttentionHeatmap()
    └─→ updateStats()
```

## Component Breakdown

### 1. Layer Flow Graph
**File Section**: Lines 570-615

Renders 28 layers as interactive bars with:
- **Color mapping** (`residual_norm` → color class):
  - 0-40% of max: `cold` (blue gradient)
  - 40-70% of max: `moderate` (orange gradient)
  - 70-100% of max: `hot` (red gradient)

- **Width** (`mlp_activation_rate`):
  - Direct percentage of neurons alive
  - 0-100% bar width in container

- **Phase coherence dot**:
  - `< 0.4`: Chaotic (red)
  - `0.4-0.6`: Medium (yellow)
  - `> 0.6`: Coherent (green)

**Interactivity**:
- Hover shows tooltip with all layer metrics
- Click selects layer for neuron map and attention heatmap
- Selection highlighted with cyan border

### 2. Token Selector
**File Section**: Lines 553-568

Dropdown populated with all tokens from current trace:
- Labeled as "Token N: token_str"
- Updates visualization on selection
- Synced with `selectedTokenIdx` variable

### 3. MLP Neuron Activation Map
**File Section**: Lines 616-633

16x16 grid of 256 neurons showing activation state:
- Uses `hot_neurons` array from LayerTrace
- Color coded:
  - Alive (in hot_neurons set): Bright green with glow
  - Dormant (random 30%): Gray
  - Dead (rest): Dark
- Hover shows neuron ID

### 4. Attention Heatmap
**File Section**: Lines 634-658

Synthetic attention matrix for selected layer:
- Dimensions: `contextLen × numHeads` (max 32 × 8)
- Values: Exponential decay based on position distance
- Color: HSL color space with intensity mapped to attention weight
- Uses realistic temporal coherence pattern

**Attention Calculation**:
```javascript
attention = exp(-(contextLen - 1 - pos) / 4) × (0.5 + sin(head / 4) × 0.2)
```
This creates recent-token bias typical in transformer attention.

### 5. Summary Statistics
**File Section**: Lines 659-722

Six stat cards with visualizations:

**Active Layers**:
- Count of layers where `residual_norm > LAYER_THRESHOLD` (0.3)
- Bar width: (count / 28) × 100%
- Identifies which layers are meaningfully processing the token

**Avg MLP Activation**:
- Mean of `mlp_activation_rate` across all layers
- Bar width: percentage directly
- Shows overall network utilization

**Logit Entropy**:
- Calculated as: sum(logit_i²) / num_logits
- Higher = more confident (sharper peak in distribution)
- Indicates decision certainty

**Confidence Score**:
- Softmax approximation: exp(top_logit) / (exp(top) + exp(second))
- Percentage confidence in top prediction
- Bar fills to show confidence level

**Top Predictions**:
- Top-5 logit predictions with softmax probabilities
- Each with individual probability bar
- Shows full top-5 distribution

**Forward Time**:
- Milliseconds for forward pass (from `step.time_ms`)
- Simple text display of latency

### 6. Mock Data Generator
**File Section**: Lines 320-384

Generates realistic synthetic traces:

**Per-Layer Metrics** (loop over 28 layers):
```javascript
residualDelta = random(0.1, 0.8)
mlpActivationRate = sin(layerIdx/28 * π) * 0.8 + 0.2
phaseCoherence = 0.5 + sin((tokenIdx + layerIdx) / 5) * 0.3
```

**Characteristics**:
- Layer depth pattern: middle layers most active
- Temporal correlation: phase coherence varies smoothly
- Realistic MLP sparsity: 20-80% activation range
- Variable hot neurons: 0-8 per layer

**Per-Token**:
- Generates one step per word in prompt
- Creates active_layers set from layers exceeding threshold
- Random top-5 logits with realistic distribution

### 7. Interactive Features

**Tooltip** (Lines 729-737):
- Shows on layer bar hover
- Displays: Layer #, Residual, Attention Energy, MLP Energy, MLP Rate, Phase, Hot Neurons
- Positioned relative to cursor

**Selection State**:
- `selectedTokenIdx`: Currently displayed token
- `selectedLayerIdx`: Currently selected layer (null = none)
- Selection highlights in layer flow
- Updates dependent panels (neurons, attention)

**Keyboard**:
- Enter key on prompt input triggers trace
- Dropdown selection via token selector

## Data Structures

### BrainTrace (from server)
```javascript
{
  prompt: string,
  response: string,
  steps: GenerationStep[],
  iara_snapshots: SimulationSnapshot[],
  layer_activation_path: usize[][],
  energy_trajectory: f32[],
  attention_heatmap: f32[][],
  finalized_at: string?
}
```

### GenerationStep (per token)
```javascript
{
  token_id: u32,
  token_str: string,
  top_logits: [u32, f32][],
  layer_traces: LayerTrace[],
  active_layers: usize[],
  time_ms: u128
}
```

### LayerTrace (per layer per step)
```javascript
{
  layer_idx: usize,
  attention_energy: f32,
  mlp_energy: f32,
  residual_norm: f32,
  max_activation: f32,
  hot_neurons: usize[],
  time_us: u128,
  // Extended (from mock or server)
  mlp_activation_rate: f32,
  phase_coherence: f32,
  hidden_norm: f32
}
```

## Styling Strategy

### Color Palette
- **Dark background**: #0a0e27 (deep blue-black)
- **Panel background**: #0f1535 (slightly lighter)
- **Borders**: #1e3a5f (dark blue)
- **Accent (cyan)**: #00d9ff (bright cyan)
- **Success (green)**: #00ff99 (bright green)
- **Warning (orange)**: #ff6633, #ffaa33
- **Error (red)**: #ff3333, #ff3366

### Typography
- Monospace: 'Courier New' for all text
- Header: 18px, bold
- Titles: 13px, uppercase, letter-spacing
- Data: 14px (values), 12px (labels), 11px (small)

### Layout
- Flexbox for main container and panels
- CSS Grid for stats (1-column) and neuron grid (16x16)
- Nested flex for header controls
- Scrollable regions with custom scrollbar styling

## Performance Considerations

### Rendering
- 28 layer bars: DOM creation + CSS styling
- 256 neurons: Grid creation (one div per neuron)
- 256 heatmap cells: Grid creation (8 × up to 32)
- Total DOM nodes per render: ~600 nodes

**Optimization**:
- Only re-render changed panels
- Use `innerHTML` for fast batch updates
- CSS transitions for smooth interactions

### Memory
- One BrainTrace in memory (steps × layers = typically < 1MB)
- Mock data generation: O(num_tokens × 28) = negligible

### Interactivity
- Layer selection: O(1) update
- Token selection: Full visualization redraw (~200ms)
- No external API calls with mock enabled

## Browser Compatibility

- Requires ES6+ (arrow functions, template literals, const/let)
- D3.js v7 (modern JavaScript)
- CSS Grid and Flexbox support
- CSS custom properties not used (wide support)

**Tested on**:
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Integration Checklist

When integrating with IARA engine:

- [ ] Place trace.html in frontend/ directory
- [ ] Configure server to serve static files from frontend/
- [ ] Implement POST /api/trace endpoint
- [ ] Endpoint should accept `{ prompt: string }`
- [ ] Endpoint should return GenerationStep[] array with all required fields
- [ ] Ensure LayerTrace includes extended fields if possible:
  - `mlp_activation_rate`: neuron sparsity (0-1)
  - `phase_coherence`: synchronization metric (0-1)
  - `hidden_norm`: residual magnitude tracking
- [ ] Add Access-Control-Allow-Origin headers if served on different origin
- [ ] Test with real inference engine data
- [ ] Verify layer count (hardcoded to 28 - change if needed)

## Future Enhancement Ideas

1. **Multi-token comparison**: Side-by-side visualization of adjacent tokens
2. **Energy trajectory graph**: Show energy progression over generation
3. **Gradient highlights**: Visualize which tokens are "important" by learning signal
4. **Attention flow animation**: Animate attention patterns as token progresses
5. **Export functionality**: Download trace data as JSON
6. **Real-time streaming**: WebSocket support for live trace streaming
7. **Performance profiling**: Per-layer latency breakdown
8. **Neuron clustering**: Identify correlated neuron groups
9. **Saliency maps**: Token importance based on gradient
10. **Model comparison**: Visualize two different models side-by-side
