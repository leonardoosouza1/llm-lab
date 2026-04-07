# IARA Deep Trace Visualization - Complete Index

## Welcome!

You have received a complete, production-ready visualization system for analyzing deep transformer model inference traces. Everything you need is in this folder.

## Start Here (Pick Your Path)

### Path 1: I Want to Use It Right Now
1. Read: **QUICKSTART.md** (5 min read)
2. Action: Open `trace.html` in browser
3. Experiment: Use mock data to explore

### Path 2: I Need to Integrate with My Server
1. Read: **API_REFERENCE.md** (10 min read)
2. Implement: The POST /api/trace endpoint per spec
3. Configure: Server to serve static files
4. Test: Integration with real inference data

### Path 3: I Want to Understand the Code
1. Read: **TECHNICAL_NOTES.md** (15 min read)
2. Review: Component breakdown and architecture
3. Explore: The 13 JavaScript functions in trace.html
4. Customize: Extend for your use case

### Path 4: I Want the Big Picture
1. Read: **SUMMARY.md** (10 min read)
2. Review: Success criteria and features list
3. Check: Integration requirements
4. Plan: Your deployment strategy

### Path 5: I Need Quick Reference
1. Open: **QUICK_REFERENCE.txt** (reference anytime)
2. Use: For keyboard shortcuts and color meanings
3. Consult: For troubleshooting table
4. Share: With team members

## File Guide

### Core Visualization

**trace.html** (32 KB) ⭐ THE MAIN FILE
- Single self-contained HTML file
- All CSS and JavaScript inline
- Works immediately with mock data
- Zero external dependencies (except D3.js CDN)
- Production-ready for deployment

Key Features:
- 28-layer transformer visualization
- Interactive layer selection
- Attention heatmaps
- Neuron activation maps
- Real-time statistics
- Mock data generator (no server needed)
- API integration ready (POST /api/trace)

### Documentation Suite

**README.md** (4.6 KB)
- What it is and what it does
- Feature overview
- Component descriptions
- Quick navigation guide

**QUICKSTART.md** (9.4 KB) ← START HERE
- Step-by-step UI walkthrough
- How to interact with visualization
- Common tasks (inspect layers, compare tokens)
- Troubleshooting guide
- Tips and tricks

**TECHNICAL_NOTES.md** (8.4 KB)
- Architecture breakdown
- 13 JavaScript functions explained
- Data structures and flow
- Color mapping logic
- Performance characteristics
- Integration checklist

**API_REFERENCE.md** (9.7 KB)
- Complete API specification
- Request/response format examples
- Required LayerTrace fields
- Data validation guidelines
- Error handling
- CORS considerations

**SUMMARY.md** (13 KB)
- Project overview
- Feature matrix
- Architecture summary
- Success criteria (all met)
- What's included (what's not)
- Future enhancement ideas

**QUICK_REFERENCE.txt** (11 KB)
- One-page cheat sheet
- Color legend
- Keyboard shortcuts
- Troubleshooting table
- API spec summary
- Copy/paste ready

**INDEX.md** (this file)
- Complete file guide
- Navigation paths
- Quick lookup
- Getting help

## Quick Start (2 minutes)

```bash
# Open visualization with mock data
firefox /home/leonardo/projects/LLM/llm-lab/iara-engine/frontend/trace.html

# Or via file manager
Double-click trace.html
```

That's it! Mock data is enabled by default. You'll see:
- 28 transformer layers
- Synthetic but realistic activation patterns
- Interactive visualization
- All features working

## Integration (For Developers)

To connect with your IARA inference engine:

1. **Expose API endpoint** `POST /api/trace`:
   ```json
   Request:  {"prompt": "text"}
   Response: BrainTrace with GenerationStep[] (28 layers each)
   ```

2. **Configure static file serving**:
   ```
   Serve /frontend/ as static directory
   ```

3. **Test**:
   - Start engine: `cargo run --bin iara-engine`
   - Open: `http://localhost:PORT/trace.html`
   - Uncheck: "Mock Data"
   - Enter prompt, click "Trace"

See **API_REFERENCE.md** for complete specification.

## What You're Visualizing

### The Data Story
```
Inference Process:
  Input Prompt → Token-by-token Generation
       ↓
  Per Token:
    - All 28 layers process the token
    - Each layer transforms the hidden state
    - Some neurons activate (MLP layer)
    - Attention heads focus on context
       ↓
  Capture Data:
    - How much each layer changed (residual_norm)
    - Which neurons fired (hot_neurons)
    - Which positions attended to (attention)
    - Network's confidence (logits)
       ↓
  Visualize:
    - Layer flow graph (main)
    - Neuron activation grid
    - Attention heatmap
    - Prediction statistics
```

### The Visualization Maps

**Layer Bars** show:
- **Width**: % of neurons alive (sparsity)
- **Color**: How much layer changed token (intensity)
- **Dot**: Are neurons synchronized (coherence)

**Neuron Grid** shows:
- **Bright**: Neurons that fired
- **Gray**: Dormant neurons
- **Dark**: Dead neurons

**Attention Pattern** shows:
- **Position**: What context positions attended to
- **Head**: Each of 8 attention heads
- **Brightness**: Attention weight

**Statistics** show:
- **Active Layers**: How many layers doing meaningful work
- **Activation Rate**: Overall network utilization
- **Confidence**: How sure model is about prediction
- **Top Predictions**: Best 5 next-token candidates

## Color Legend

| Color | Context | Meaning |
|-------|---------|---------|
| **Cyan** (#00d9ff) | UI accent | Primary highlight, selection |
| **Green** (#00ff99) | Visualization | Active/healthy/coherent signals |
| **Red** (#ff3366) | Visualization | High activation, chaotic, errors |
| **Orange** (#ff9933) | Visualization | Moderate activation, warnings |
| **Blue** (#0066cc) | Visualization | Low activation, passive, cold |
| **Yellow** (#ffff00) | Visualization | Medium state, uncertain |
| **Gray** | Visualization | Inactive, dormant, dead |
| **Black** (#0a0e27) | Background | Dark theme background |

## Keyboard & Mouse

| Action | Effect |
|--------|--------|
| Click layer bar | Select that layer (shows neurons + attention) |
| Click layer bar again | Deselect layer |
| Hover layer bar | Show tooltip with detailed metrics |
| Enter key (in prompt) | Submit trace |
| Change token dropdown | View different token's activations |
| Change prompt | New text to analyze |
| Click "Trace" button | Generate trace (mock or server) |
| Check "Mock Data" | Use synthetic data (no server) |
| Uncheck "Mock Data" | Use real /api/trace server |

## Understanding the Metrics

### Active Layers
- Count of layers making meaningful changes to token
- **High**: Model uncertain, working harder
- **Low**: Model confident, efficient computation
- **Threshold**: residual_norm > 0.3

### MLP Activation Rate
- Percentage of neurons firing on average
- **High %**: Dense computation, slower, expressive
- **Low %**: Sparse computation, faster, efficient
- **Ideal**: Usually 40-70% for good models

### Logit Entropy
- Measure of prediction confidence
- **High**: Uncertain (multiple good options)
- **Low**: Confident (clear winner)
- **Typical**: 0.5-5.0 (log scale)

### Confidence Score
- Probability of top predicted token
- **>80%**: Very confident
- **50-80%**: Moderate confidence
- **<50%**: Very uncertain

### Phase Coherence
- Neuron synchronization metric
- **Green (>0.6)**: Coherent, organized learning
- **Yellow (0.4-0.6)**: Mixed patterns
- **Red (<0.4)**: Chaotic, random firing

## Common Questions

**Q: Do I need a server?**
A: No! Mock data is enabled by default. Works offline.

**Q: How do I see real inference data?**
A: Start IARA engine, uncheck "Mock Data", enter prompt.

**Q: What browser do I need?**
A: Chrome, Firefox, Safari, or Edge (2020+).

**Q: Can I customize colors?**
A: Edit CSS in trace.html (lines 9-400).

**Q: Can I zoom or pan?**
A: Scroll to navigate, mouse wheel to zoom panels.

**Q: How many tokens can I trace?**
A: Limited by memory (typically 100-1000).

**Q: Can I export the data?**
A: Yes! F12 → Console → copy(JSON.stringify(currentTrace)).

**Q: Does it work offline?**
A: Yes, with mock data. Real server data requires network.

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| trace.html | 32 KB | Main visualization (core) |
| README.md | 4.6 KB | Feature overview |
| QUICKSTART.md | 9.4 KB | User guide |
| TECHNICAL_NOTES.md | 8.4 KB | Architecture details |
| API_REFERENCE.md | 9.7 KB | Server integration |
| SUMMARY.md | 13 KB | Project overview |
| QUICK_REFERENCE.txt | 11 KB | Cheat sheet |
| INDEX.md | This file | Navigation |
| **Total** | **108 KB** | Everything |

## Where to Go for Help

### I want to...

**...get started quickly**
→ QUICKSTART.md (5 min)

**...understand what I'm looking at**
→ TECHNICAL_NOTES.md (understand the visualization)
→ README.md (understand the features)

**...integrate with my server**
→ API_REFERENCE.md (complete specification)

**...understand the project**
→ SUMMARY.md (big picture overview)

**...find a specific feature**
→ QUICK_REFERENCE.txt (cheat sheet)

**...debug something**
→ QUICK_REFERENCE.txt (troubleshooting)
→ Browser F12 console (view data)

**...understand the code**
→ TECHNICAL_NOTES.md (architecture)
→ trace.html (read the 936 lines)

## Common Tasks

### View a Token's Activations
1. Open trace.html
2. Enter prompt
3. Click "Trace"
4. Select token from dropdown
5. See layer activations update

### Inspect a Layer
1. In layer flow graph
2. Click any layer bar
3. See neuron grid (left, bottom)
4. See attention heatmap (right, bottom)
5. See metrics in tooltip

### Compare Two Tokens
1. Generate trace
2. Select first token from dropdown
3. Note activation patterns
4. Select second token
5. Compare with first

### Debug Model Behavior
1. Enter prompt that seems wrong
2. Click "Trace"
3. Check which layers are active
4. Look for red dots (chaotic)
5. Check confidence score
6. View top predictions

### Test Server Integration
1. Start IARA engine
2. Open trace.html
3. Uncheck "Mock Data"
4. Enter test prompt
5. Click "Trace"
6. Verify visualization updates

## Deployment

### Standalone (With Mock Data)
```
Copy trace.html anywhere
Open in browser
Works immediately
No setup needed
```

### With IARA Engine Server
```
1. Place frontend/ folder in iara-engine/
2. Configure server to serve /frontend/
3. Implement POST /api/trace endpoint
4. Access via http://localhost:PORT/trace.html
5. Uncheck "Mock Data" to use server
```

## Support & Documentation

All questions should be answerable by the documentation:

1. **How do I use it?** → QUICKSTART.md
2. **What does it show?** → README.md
3. **How does it work?** → TECHNICAL_NOTES.md
4. **How do I integrate?** → API_REFERENCE.md
5. **What's the big picture?** → SUMMARY.md
6. **Quick lookup?** → QUICK_REFERENCE.txt
7. **Where do I go?** → INDEX.md (this file)

## Project Statistics

- **Files**: 8 (1 HTML + 7 documentation)
- **Size**: 108 KB (32 KB visualization + 76 KB docs)
- **Functions**: 13 in JavaScript
- **CSS Classes**: 51
- **HTML Lines**: 936
- **Layers Visualized**: 28
- **Neurons per Layer**: 256
- **Supported Tokens**: Unlimited (memory)
- **Dependencies**: D3.js v7 (CDN only)
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

## What's Included

✓ Production-ready visualization
✓ Mock data generator (no server needed)
✓ Dark theme UI (IARA dashboard matching)
✓ Comprehensive documentation
✓ API integration ready
✓ Detailed technical notes
✓ Quick reference guide
✓ Multiple navigation paths

## What's NOT Included

✗ Web server (you provide or use static hosting)
✗ Backend inference engine (you provide IARA engine)
✗ Real model weights (you provide or download)
✗ Custom color schemes (edit CSS yourself)
✗ 3D visualization (2D is clearer)
✗ Real-time streaming (works with single traces)

## License & Attribution

This visualization system was created for analyzing IARA inference engine traces. Use freely for analysis, debugging, and demonstration purposes.

## Next Steps

1. **Right now**: Open trace.html in browser
2. **In 5 minutes**: Read QUICKSTART.md
3. **In 30 minutes**: Read API_REFERENCE.md if integrating
4. **In 1 hour**: Have working integration with server

---

**Version**: 1.0 (April 7, 2026)
**Status**: Production Ready
**Location**: /home/leonardo/projects/LLM/llm-lab/iara-engine/frontend/
**Access**: http://localhost:PORT/trace.html (when deployed)
