# IARA Deep Trace Visualization - Quick Start

## What You Get

A single-file interactive HTML visualization that shows how tokens flow through all 28 layers of the IARA transformer model during inference.

## File Location

```
/home/leonardo/projects/LLM/llm-lab/iara-engine/frontend/trace.html
```

## Quickest Start (Right Now)

### Option 1: Test Locally in Browser

```bash
# Navigate to the frontend directory
cd /home/leonardo/projects/LLM/llm-lab/iara-engine/frontend

# Open in browser (Linux)
xdg-open trace.html

# Or macOS
open trace.html

# Or Windows
start trace.html
```

The visualization comes with mock data enabled by default. You'll see:
- 28 transformer layers with synthetic activation patterns
- Token flow visualization
- Interactive layer selection
- Attention patterns and neuron maps

### Option 2: Serve from IARA Engine

When the IARA engine's API server is running:

1. Ensure the engine is configured to serve static files from `frontend/`
2. Open: `http://localhost:YOUR_PORT/trace.html`
3. Uncheck "Mock Data" to use real server data
4. Enter a prompt and click "Trace"

## UI Walkthrough

### 1. Top Bar: Input Section
```
[Prompt input] [Trace button] [Token selector] [Mock Data checkbox]
```

- **Prompt input**: Enter text to analyze (e.g., "Hello world")
- **Trace button**: Send prompt or generate mock data
- **Token selector**: Choose which generated token to inspect
- **Mock Data**: Toggle between synthetic and real server data

### 2. Main Visualization: Layer Flow Graph (Left, Top)
Shows all 28 layers stacked vertically. Each layer displays:

```
Layer# ░░░░░░░░░░░░░░░░░░░░░░ ●
└─┬──┘ └────┬──────────────────┘ └─┬──┘
  │         │                       └─ Phase coherence dot
  │         │
  │         └─ MLP Activation Bar (width = % of neurons alive)
  │            Color: Blue (cold/passive) → Red (hot/active)
  │
  └─ Layer index (0-27)
```

**Interaction**: Click any layer bar to select it and see:
- Its attention heatmap (right panel)
- Its neuron activation grid (left panel, bottom)

### 3. Neuron Activation Map (Left, Bottom)
16×16 grid of 256 neurons in the selected layer:

```
██████████████████
██████░░████░░████  (bright = active, gray = dormant, dark = dead)
██████████████████
░░░░░░░░░░░░░░░░░░
```

The grid updates when you select different layers or tokens.

### 4. Summary Statistics (Right Panel)
Live stats for the current token:

| Metric | Shows |
|--------|-------|
| **Active Layers** | How many layers are meaningfully transforming this token |
| **Avg MLP Activation** | Network utilization (% of neurons firing) |
| **Logit Entropy** | Prediction confidence (higher = more certain) |
| **Confidence** | Probability of top predicted token |
| **Top Predictions** | Best 5 next-token predictions with probabilities |
| **Forward Time** | Inference latency in milliseconds |

### 5. Attention Heatmap (Right, Bottom)
When you select a layer, see its attention pattern:

```
Context Positions (0 to current)
           ↓
Heads →  [████]  (brighter = more attention to that position)
         [████]
         [████]
         [████]
         [████]
         [████]
         [████]
         [████]
```

Recent positions typically have more attention (recency bias).

## Common Tasks

### Inspect a Specific Layer
1. Enter a prompt (e.g., "Hello world")
2. Click "Trace"
3. In the layer flow graph, click on a layer bar
4. See:
   - Its 256 neurons (which ones fired?)
   - Its attention pattern (where does it look?)
   - Its metrics in the tooltip

### Compare Different Tokens
1. Generate a trace with multiple tokens
2. Use the "Token selector" dropdown
3. Select different tokens to see how their layers activate differently

### Understand Model Behavior
1. Look for patterns:
   - Which layers are typically active?
   - Do middle layers activate more than early/late?
   - Which tokens activate different layers?

2. Check statistics:
   - High confidence (>80%) = model is sure
   - Low activation rate = sparse networks (efficient)
   - High entropy = uncertain prediction

### Export Data (For Analysis)
Open browser DevTools (F12) and run:
```javascript
console.log(JSON.stringify(currentTrace, null, 2));
```

Then copy/paste into a JSON file for external analysis.

## Color Legend

| Color | Meaning |
|-------|---------|
| **Cyan (#00d9ff)** | Primary accent, highlights, selection |
| **Green (#00ff99)** | Coherent/active/positive signals |
| **Red (#ff3333)** | High activation, chaotic, errors |
| **Orange (#ff9933)** | Moderate activation, warnings |
| **Blue (#0066cc)** | Low activation, passive, cold |
| **Yellow (#ffff00)** | Medium coherence/state |
| **Gray (#888888)** | Inactive, dormant, disabled |

## Behind the Scenes: What the Visualization Shows

### Layer Bars (Width)
The bar width shows **MLP Activation Rate** - the percentage of neurons in the MLP layer that are actually firing (active) during the forward pass.

- Narrow bar = most neurons sleeping (sparse)
- Wide bar = most neurons awake (dense)

**Why it matters**: Sparse activations = smaller effective model = faster inference

### Layer Bars (Color)
The bar color shows **Residual Delta** - how much that layer changed the token representation.

- Blue = minimal change = layer is passively passing through
- Red = major change = layer is transforming the token significantly

**Why it matters**: Shows which layers do the real work vs. which are redundant

### Phase Coherence Dot
The dot shows **Phase Coherence** - how synchronized are the neurons in that layer.

- Green = synchronized = coherent pattern
- Yellow = mixed = moderate coherence
- Red = chaotic = neurons firing randomly

**Why it matters**: Coherent layers might be learning meaningful patterns; chaotic layers might be struggling

### Attention Pattern
Shows which positions in the context are being paid attention to by each head.

- Brightness = how much attention
- Typical pattern: recent tokens get more attention (recency bias)

**Why it matters**: Shows if attention is working correctly (looking at relevant context)

### Neuron Activation
Bright neurons are the ones that fired for this token in this layer.

- Bright green = active neuron
- Gray = dormant (could activate but didn't)
- Dark = dead (almost never active)

**Why it matters**: Dead neurons waste parameters; many dormant neurons suggest sparsity opportunity

## Tips & Tricks

1. **Layer 0-8** (early): Often very active - these do semantic tokenization
2. **Layer 8-20** (middle): Most variable - layer-specific processing
3. **Layer 20-28** (late): Often calmer - refinement and output preparation

4. **High confidence tokens**: Usually have fewer active layers (less need to compute)
5. **Uncertain tokens**: Typically activate more layers (the model is "thinking harder")

6. **Watching phase coherence**:
   - Early layers: usually high (clean signal)
   - Middle layers: variable (complex processing)
   - Late layers: moderate to high (converging to output)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter | In prompt box = submit trace |
| Click | On layer bar = select layer |
| Hover | On layer bar = show tooltip |

## What's Generated

When you click "Trace", the system either:

**With Mock Data (Default)**:
- Generates realistic synthetic trace data client-side
- No server needed
- Perfect for UI testing

**With Real Server**:
- Sends POST to `/api/trace` with your prompt
- Server performs actual inference
- Returns real tensor activation data
- Visualizes actual model behavior

## Still Getting Mock Data?

If you want to see real model behavior, you need:

1. **Running IARA Engine Server**
   ```bash
   cd /home/leonardo/projects/LLM/llm-lab/iara-engine
   cargo run --bin iara-engine
   ```

2. **Server must expose `/api/trace` endpoint**
   - The endpoint should perform inference
   - Return GenerationStep array with LayerTrace data

3. **Uncheck "Mock Data"** in the visualization

4. **Make sure CORS allows browser access** (if on different origin)

## Troubleshooting

### "Failed to trace: Server error 404"
- Server isn't running
- Or `/api/trace` endpoint doesn't exist
- Solution: Start IARA engine or enable Mock Data

### Visualization shows blank panels
- JavaScript error (check browser console with F12)
- D3.js didn't load from CDN (no internet)
- Solution: Open console, check for errors

### Layer bars don't show color
- CSS didn't load properly
- Solution: Refresh page, clear cache

### Numbers look weird
- Data is invalid JSON
- Solution: Check server response format with F12 Network tab

## Performance

- Opens instantly (single HTML file, 32KB)
- Renders smooth even with mock data
- Handles 28 layers × 8 tokens without lag
- All visualization happens in browser (no server calls after trace)

## Next Steps

1. Try with mock data first (get familiar with UI)
2. Review the layer activation patterns
3. Read TECHNICAL_NOTES.md for architecture details
4. Integrate with real IARA engine server
5. Experiment with different prompts and model sizes

---

**Questions?** Check:
- README.md - Feature overview
- TECHNICAL_NOTES.md - Implementation details
- Browser console (F12) - Error messages and data inspection
