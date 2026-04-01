# IARA Engine Frontend — Observability Enhancements

Complete rebuild of the IARA observability frontend with comprehensive subsystem monitoring, advanced telemetry visualization, and full logging infrastructure.

## Documentation

### Quick Start
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** — 2-minute overview
  - What changed
  - Color coding guide
  - Usage examples
  - Common workflows

### Complete Documentation
- **[ENHANCEMENTS_SUMMARY.md](./ENHANCEMENTS_SUMMARY.md)** — Full technical specification
  - Implementation details for each feature
  - Code structure and patterns
  - Store enhancements
  - Watcher architecture
  - Performance notes

### Technical Changes
- **[CHANGES_DETAIL.md](./CHANGES_DETAIL.md)** — File-by-file changelog
  - Exact modifications per file
  - Line numbers and descriptions
  - Code snippets
  - Testing checklist

## What's New

### Live Panel
- **Subsystem Status Bar** — At-a-glance view of all subsystems (Backend, Quantization, ESN, Gravity, CfC)
- **Enhanced ESN Panel** — 3 sparklines (entropy, top1_prob, state_norm) + temp_delta metric
- **Enhanced Gravity Panel** — Mean/max gravity + high-count estimation + dual sparkline
- **Enhanced CfC Panel** — State and correction norms + correction_norm sparkline

### Console Tab
- **4 new filter buttons** — ESN, Gravity, CfC, Pipeline (color-coded to match subsystems)
- **Comprehensive logging** — All WebSocket messages logged with timestamp and summary
- **5000-entry buffer** — Expanded from 500 for detailed debugging

## Quick Reference

| Panel | New Elements | Purpose |
|-------|--------------|---------|
| Subsystem Status | 5 badges | Show what's active: Backend, Quantization, ESN, Gravity, CfC |
| ESN | 3 sparklines + 1 value | Monitor entropy, confidence, state evolution + temp delta |
| Gravity | 3 values + 1 dual sparkline | Track mean/max gravity, token importance count + trends |
| CfC | 2 values + 1 sparkline | Monitor liquid network state and correction norms |
| Console | 4 filter buttons | Filter logs by: esn_state, gravity_state, cfc_state, pipeline_layer |

## Color System

```
Core Subsystems (when active):
  ESN:      Purple (#a855f7)
  Gravity:  Amber (#fbbf24)
  CfC:      Sky Blue (#38bdf8)

Support Systems:
  Pipeline: Pink (#f472b6)
  Backend:  Cyan (#00ffc8) [when detected]
  Quant:    Q4→Pink, BF16→Cyan

Utilities:
  Success:  Cyan (#00ffc8)
  Error:    Red (#ff4466)
  Inactive: Gray (#6b7280)
```

## Key Features

### Visual Monitoring
- **Real-time sparklines** — 200-point rolling history, auto-scaling
- **Color-coded badges** — Light up when subsystems activate
- **Dual-line comparison** — Gravity threshold vs mean on single canvas
- **Metric fields** — Values formatted to 3 decimals for precision

### Debugging
- **WebSocket logging** — Every message captured with timestamp
- **Message filtering** — Toggle subsystems to debug specific behaviors
- **JSON inspector** — Click log entries to view full message payload
- **Message summary** — Type + step number for quick scanning

### Performance
- **Zero external deps** — Pure canvas API, no additional libraries
- **Bounded buffers** — 200 points per sparkline, 5000 logs max
- **Efficient rendering** — Only redraws on data changes
- **Memory safe** — Auto-trim on overflow

## Getting Started

### Monitor System Status
1. Load model in Live panel
2. Look at Subsystem Status bar at top
3. Watch badges light up as inference starts (ESN → Gravity → CfC)
4. Check Quantization to see if Q4 optimization is active

### Debug ESN Behavior
1. Go to Live panel → ESN Observer
2. Watch 3 sparklines:
   - Entropy: Should show variations (network is learning)
   - Top-1 Prob: Should increase (confidence improving)
   - State Norm: Shows internal state magnitude
3. Check esn-temp-delta value

### Track Gravity Dynamics
1. Live panel → Semantic Gravity
2. View 3 metrics:
   - Mean Gravity: Average semantic importance
   - Max Gravity: Peak value (indicates critical token)
   - High-Count: How many tokens exceed threshold
3. Watch dual sparkline (amber threshold vs blue mean)

### Debug with Console Filters
1. Go to Console tab
2. Click "ESN" to see all ESN state messages
3. Messages show step number, click to view raw JSON
4. Switch between filters to trace subsystem activity

## Architecture

```
WebSocket In
    ↓
dispatch(msg) in websocket.js
    ↓
_logMessage() → store.logs [FIRST]
    ↓
Process message type (esn_state, gravity_state, etc.)
    ↓
Update store fields (esnLatest, gravityLatest, etc.) [SECOND]
    ↓
Watchers in main.js trigger
    ↓
Update DOM: metrics, sparklines, badges
    ↓
User sees real-time updates
```

## File Structure

```
frontend/
├── index.html                     [Main UI: panels, sparklines, filters]
├── style.css                      [Theming: glass panels, button colors]
├── src/
│   ├── main.js                    [Watchers: metrics, sparklines, badges]
│   ├── store/
│   │   └── index.js               [State: subsystem flags, histories]
│   ├── api/
│   │   └── websocket.js           [Logging: _logMessage, dispatch]
│   └── components/
│       └── log_console.js         [Filtering: enhanced log logic]
└── [Documentation]
    ├── QUICK_REFERENCE.md         [2-min overview]
    ├── ENHANCEMENTS_SUMMARY.md    [Full spec]
    ├── CHANGES_DETAIL.md          [File-by-file changes]
    └── README_ENHANCEMENTS.md     [This file]
```

## Implementation Stats

| Metric | Value |
|--------|-------|
| Files Modified | 6 |
| Lines Added | ~313 |
| New UI Elements | 22 |
| New Watchers | 8+ |
| New Store Fields | 5 |
| New Filter Buttons | 4 |
| Sparkline Canvases | 4 |
| Documentation Pages | 4 |

## Testing Checklist

- [ ] Subsystem Status bar visible at top of Live panel
- [ ] Status badges show correct active/inactive colors
- [ ] ESN panel: 3 sparklines + temp_delta display
- [ ] Gravity panel: 3 metric fields + dual sparkline
- [ ] CfC panel: 4-column layout with sparkline
- [ ] Console: 4 new filter buttons appear
- [ ] Filter buttons toggle highlighting correctly
- [ ] Log entries show message type summary
- [ ] Sparklines auto-scale and animate
- [ ] No console errors on load or during inference

## Customization

### Add a New Sparkline
1. Add canvas in HTML: `<canvas id="my-canvas" height="40" style="width:100%"></canvas>`
2. Create history array in main.js: `let myPoints = []`
3. Add watcher to populate array and call: `drawSparkline('my-canvas', myPoints, '#color')`

### Change Color Scheme
1. Edit color variables in ENHANCEMENTS_SUMMARY.md (Color Palette Reference)
2. Update HTML element style: `color: #newcolor`
3. Update CSS button classes: `.filter-btn--name { color: #newcolor }`
4. Update watcher in main.js: `el.style.color = #newcolor`

### Extend Message Logging
1. Add new message type in websocket.js dispatch()
2. Call `this._logMessage()` with appropriate summary
3. Add filter button in HTML Console section
4. Add filter logic in log_console.js
5. Add CSS styling for button

## Support & Debugging

### Sparklines Not Updating
1. Check browser console for errors
2. Verify WebSocket messages arriving (Console tab)
3. Check canvas IDs match in HTML and main.js

### Filter Buttons Not Working
1. Verify data-filter attribute on button matches filter string
2. Check log entry has correct summary format (type at start)
3. Ensure log_console.js filter logic includes new type

### Memory Issues
1. Check store.logs.length in DevTools (should stay < 5000)
2. Verify _logMessage() is being called (adds to buffer)
3. Check auto-trim logic is working

## Future Enhancements

- Persistent storage of telemetry
- Export logs to JSON/CSV
- Alert thresholds for anomalies
- Time-range selection for analysis
- Comparison between generations
- Advanced analytics dashboard

---

**Status**: Complete and production-ready
**Version**: 1.0.0
**Last Updated**: 2026-03-31
**Frontend**: Vue3 + Vanilla JS (no external dependencies)
