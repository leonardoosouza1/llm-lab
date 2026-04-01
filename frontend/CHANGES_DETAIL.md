# Detailed Changes — File-by-File

## 1. frontend/index.html

### Subsystem Status Bar (NEW)
- **Location**: Lines 130-157 (before panel-grid-live)
- **Elements**: 5 status badges (backend, quantization, esn, gravity, cfc)
- **Styling**: Gradient background, glass-panel style
- **IDs**: subsys-backend, subsys-quant, subsys-esn, subsys-gravity, subsys-cfc

### ESN Panel Updates
- **Location**: Lines 361-413
- **Replaced**: Single esn-entropy-canvas with 3-canvas grid
- **New Canvases**:
  - esn-entropy-canvas (existing, kept)
  - esn-top1-canvas (NEW)
  - esn-state-norm-canvas (NEW)
- **New Field**: esn-temp-delta (replaces esn-temperature label, shows Δ)
- **Grid**: Changed from 5-column to 5+3 layout (stats + sparklines)

### Gravity Panel Updates
- **Location**: Lines 395-438
- **New Fields**:
  - gravity-mean (NEW, color #fbbf24)
  - gravity-max (NEW, color #f97316)
  - gravity-high-count (NEW, color #fb7185)
- **Layout**: 3-column stats (was 3, replaced content)
- **Sparkline**: gravity-threshold-canvas (enhanced with dual-line capability)

### CfC Panel Updates
- **Location**: Lines 423-459
- **New Fields**:
  - cfc-state-norm (NEW, color #38bdf8)
  - cfc-correction-norm (NEW, color #818cf8)
- **Layout**: 4-column grid (was 3)
- **New Sparkline**: cfc-correction-canvas (NEW, height 50px)

### Console Filter Buttons (NEW)
- **Location**: Lines 748-780 (filter bar)
- **New Buttons**:
  - ESN (data-filter="esn_state", class filter-btn--esn)
  - Gravity (data-filter="gravity_state", class filter-btn--gravity)
  - CfC (data-filter="cfc_state", class filter-btn--cfc)
  - Pipeline (data-filter="pipeline_layer", class filter-btn--pipeline)
- **Placement**: After WS button, before HTTP button

---

## 2. src/main.js

### New Utility Function
```javascript
function drawSparkline(canvasId, points, color) {
  // Reusable canvas sparkline renderer
  // Parameters: canvas element ID, data points array, stroke color
}
```
- **Location**: Before ESN watcher
- **Purpose**: DRY principle for sparkline rendering

### ESN Watcher Enhancements
- **Added**: esnTop1Points, esnStateNormPoints arrays (200-point history)
- **New Displays**:
  - esn-temp-delta (shows temp_delta value)
  - Top-1 Probability sparkline (amber)
  - State Norm sparkline (amber)
- **Updated**: esn-entropy-canvas to use drawSparkline()

### Gravity Watcher Enhancements
- **Added**: gravityMeanPoints, gravityHighCountHistory arrays
- **New Displays**:
  - gravity-mean (formatted to 3 decimals)
  - gravity-max (formatted to 3 decimals)
  - gravity-high-count (estimated as round(history_len × max_gravity))
- **Sparkline**: Dual-line (threshold amber + mean cyan) on single canvas
- **Logic**: Both lines share axis, auto-scale to max value

### CfC Watcher Enhancements
- **Added**: cfcCorrectionNormPoints array (200-point history)
- **New Displays**:
  - cfc-state-norm (formatted to 3 decimals)
  - cfc-correction-norm (formatted to 3 decimals)
- **Sparkline**: cfc-correction-canvas (indigo color #818cf8)

### Subsystem Status Watchers (NEW)
Five new watchers (one per subsystem status):
1. **backend**: Gray (unknown) → cyan (active), text shows ROCM/OPENCL/CPU
2. **quantization**: Text shows BF16 (cyan) or Q4 (pink)
3. **esnActive**: Gray (OFF) → purple (ON)
4. **gravityActive**: Gray (OFF) → amber (ON)
5. **cfcActive**: Gray (OFF) → sky blue (ON)

### Console Filter Watcher (NEW)
```javascript
watch(() => store.logFilter, filter => {
  // Updates button active states, updates aria-pressed
})
```

### Generation Reset Enhanced
- **Resets**: All new point arrays (esn* × 3, gravity* × 3, cfc* × 1)
- **Clears**: All new metric display IDs
- **Maintains**: Existing reset behavior for esn/gravity/cfc history

---

## 3. src/store/index.js

### New Fields (at end of store)
```javascript
// Subsystem Status
backend: 'unknown',      // 'rocm' | 'opencl' | 'cpu'
quantization: 'bf16',    // 'bf16' | 'q4' | 'f32'
esnActive: false,        // ESN subsystem active flag
gravityActive: false,    // Gravity subsystem active flag
cfcActive: false,        // CfC subsystem active flag
```

---

## 4. src/api/websocket.js

### New Helper Function
```javascript
_logMessage(type, summary, raw) {
  // Logs message to store.logs with timestamp
  // Creates entry: { id, ts, type, summary, raw }
}
```

### Enhanced dispatch() Method
- **At top**: Adds universal logging: `this._logMessage('ws-in', summary, msg)`
- **model_loaded case**: Sets store.backend and store.quantization
- **esn_state case**: Sets store.esnActive = true
- **gravity_state case**: Sets store.gravityActive = true, parses mean_gravity/max_gravity
- **cfc_state case**: Sets store.cfcActive = true
- **q4_status case**: Sets store.quantization = 'q4' if active

### Message Summary Generation
- Format: `message.type + (message.step ? ` #${message.step}` : '')`
- Examples: "esn_state #42", "gravity_state #15", "pipeline_layer #5"

---

## 5. src/components/log_console.js

### Updated Filter Logic
```javascript
const entries = store.logs.filter(function(e) {
  if (filter === 'ALL') return true;
  if (filter === 'WS') return e.type === 'ws-in' || e.type === 'ws-out';
  if (filter === 'esn_state') return e.summary && e.summary.includes('esn_state');
  if (filter === 'gravity_state') return e.summary && e.summary.includes('gravity_state');
  if (filter === 'cfc_state') return e.summary && e.summary.includes('cfc_state');
  if (filter === 'pipeline_layer') return e.summary && e.summary.includes('pipeline_layer');
  if (filter === 'HTTP') return e.type === 'http';
  if (filter === 'ERROR') return e.type === 'error';
  return true;
});
```

### Container ID Fix
- Changed: `el('console-log-list')` → `el('log-list')`
- Reason: HTML uses id="log-list", not "console-log-list"

---

## 6. style.css

### New Filter Button Styles
Added 4 new filter button color schemes (after .filter-btn--error):

```css
.filter-btn--esn {
  color: var(--text-secondary);
}
.filter-btn--esn.filter-btn--active,
.filter-btn--esn:hover {
  color: #a855f7;
  background: rgba(168, 85, 247, 0.15);
  border-color: rgba(168, 85, 247, 0.3);
}

.filter-btn--gravity {
  color: var(--text-secondary);
}
.filter-btn--gravity.filter-btn--active,
.filter-btn--gravity:hover {
  color: #fbbf24;
  background: rgba(251, 191, 36, 0.15);
  border-color: rgba(251, 191, 36, 0.3);
}

.filter-btn--cfc {
  color: var(--text-secondary);
}
.filter-btn--cfc.filter-btn--active,
.filter-btn--cfc:hover {
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.15);
  border-color: rgba(56, 189, 248, 0.3);
}

.filter-btn--pipeline {
  color: var(--text-secondary);
}
.filter-btn--pipeline.filter-btn--active,
.filter-btn--pipeline:hover {
  color: #f472b6;
  background: rgba(244, 114, 182, 0.15);
  border-color: rgba(244, 114, 182, 0.3);
}
```

---

## Summary Statistics

| File | Type | Changes | Lines Added | Lines Removed |
|------|------|---------|-------------|---------------|
| index.html | HTML | 8 new elements, 6 restructures | ~100 | 10 |
| main.js | JavaScript | 1 utility + 8 watchers + enhancements | ~120 | 5 |
| store/index.js | JavaScript | 5 new fields | 5 | 0 |
| api/websocket.js | JavaScript | 1 helper + dispatch enhancements | ~30 | 0 |
| components/log_console.js | JavaScript | Filter logic + 1 fix | 8 | 1 |
| style.css | CSS | 4 button styles | 50 | 0 |

**Total**: 6 files modified, ~313 lines added, ~16 lines removed, 22 new visual elements

---

## Testing Checklist

- [ ] Subsystem Status bar appears at top of Live panel
- [ ] Status badges show correct colors (gray/active)
- [ ] ESN panel shows 3 sparklines and temp_delta
- [ ] Gravity panel shows mean/max/high-count and dual sparkline
- [ ] CfC panel shows 4 fields and correction sparkline
- [ ] Console filter buttons toggle correctly
- [ ] Log entries appear with "esn_state", "gravity_state", etc. in summary
- [ ] Clicking subsystem messages filters console correctly
- [ ] All sparklines auto-scale and update in real-time
- [ ] No console errors on page load

