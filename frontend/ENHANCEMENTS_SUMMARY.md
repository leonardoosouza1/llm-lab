# IARA LLM Engine — Frontend Observability Enhancements

## Overview
Complete observability frontend enhancement for the IARA engine with comprehensive subsystem monitoring, advanced telemetry visualization, and enhanced logging capabilities.

## Implementation Summary

### 1. Enhanced ESN Panel

**Location**: `index.html` (lines 361-420) + `src/main.js` (ESN watcher)

**New Features**:
- **Sparklines for multiple metrics**: 
  - Entropy sparkline (purple, #a855f7) — existing, updated
  - Top-1 Probability sparkline (amber, #fbbf24) — NEW
  - State Norm sparkline (amber, #fbbf24) — NEW
- **New field**: Temp Delta (Δ) displayed in control area
- **Three-canvas grid layout**: All sparklines update independently, maintaining 200-point history each

**Technical Details**:
- `esnTop1Points` and `esnStateNormPoints` arrays track metrics over time
- `drawSparkline()` utility function creates lightweight canvas-based visualizations
- Auto-scales to max value within window for visibility
- Real-time updates via `store.esnLatest` watcher

### 2. Enhanced Gravity Panel

**Location**: `index.html` (lines 395-438) + `src/main.js` (Gravity watcher)

**New Features**:
- **Mean Gravity field** — shows average semantic gravity across history
- **Max Gravity field** — shows peak semantic gravity value
- **High-Gravity Token Count** — estimates significant tokens based on threshold and max_gravity
- **Dual sparkline overlay**:
  - Threshold line (amber, #fbbf24)
  - Mean gravity line (sky blue, #38bdf8) — overlaid for comparison
  - Auto-scaling with shared max value for clear comparison

**Technical Details**:
- `gravityThresholdPoints` and `gravityMeanPoints` maintain 200-entry histories
- High-gravity count estimated as: `round(history_len × max_gravity)`
- Dual-line rendering in single canvas for efficient display
- Both lines independently tracked and redrawn each dispatch

### 3. Enhanced CfC Panel

**Location**: `index.html` (lines 423-459) + `src/main.js` (CfC watcher)

**New Features**:
- **State Norm field** — h vector norm from CfC state
- **Correction Norm field** — magnitude of liquid correction vector
- **Correction Norm Sparkline** — fourth CfC visualization showing correction magnitude over time (indigo, #818cf8)
- **4-column layout**: Consolidates all key metrics into compact display

**Technical Details**:
- `cfcCorrectionNormPoints` maintains 200-entry rolling history
- Sparkline uses `drawSparkline()` utility with dedicated indigo color
- Auto-resets on generation start
- Tracks both structural (state_norm) and dynamic (correction_norm) properties

### 4. Subsystem Status Bar

**Location**: `index.html` (lines 130-157) — above Live panel grid

**Features**:
- **Position**: Top of Live panel, gradient background (purple + cyan blend)
- **Badges**: Backend, Quantization, ESN, Gravity, CfC
- **Color System**:
  - **Backend**: gray (unknown) → cyan (#00ffc8) when active
  - **Quantization**: cyan (#00ffc8) for BF16, pink (#f472b6) for Q4
  - **ESN**: gray (off) → purple (#a855f7) when active
  - **Gravity**: gray (off) → amber (#fbbf24) when active
  - **CfC**: gray (off) → sky blue (#38bdf8) when active

**Technical Details**:
- New store fields: `backend`, `quantization`, `esnActive`, `gravityActive`, `cfcActive`
- Reactive watchers update badge colors and status text in real-time
- Displays system capabilities at a glance during model inference
- Backend detected from `model_loaded` message
- Quantization updated from `q4_status` message

### 5. Enhanced Console Tab Filters

**Location**: `index.html` (filter bar, lines 748-780) + `src/components/log_console.js` + `style.css`

**New Filter Buttons**:
- **ESN** — filters to `esn_state` messages only (purple theme)
- **Gravity** — filters to `gravity_state` messages only (amber theme)
- **CfC** — filters to `cfc_state` messages only (sky blue theme)
- **Pipeline** — filters to `pipeline_layer` messages only (pink theme)
- **WS**, **HTTP**, **ERROR** — existing filters (cyan, green, red)

**Filter Logic**:
```javascript
if (filter === 'esn_state') return e.summary && e.summary.includes('esn_state');
if (filter === 'gravity_state') return e.summary && e.summary.includes('gravity_state');
if (filter === 'cfc_state') return e.summary && e.summary.includes('cfc_state');
if (filter === 'pipeline_layer') return e.summary && e.summary.includes('pipeline_layer');
```

**CSS Styling**:
- Each filter button has dedicated color and hover/active states
- Matches IARA color palette (purple, amber, sky blue, pink)
- Consistent with existing WS/HTTP/ERROR styling pattern

### 6. WebSocket Message Logging

**Location**: `src/api/websocket.js` (dispatch function)

**Features**:
- **Comprehensive logging**: Every incoming message is logged to `store.logs`
- **Log entry structure**:
  ```javascript
  {
    id: unique timestamp-based ID,
    ts: HH:MM:SS timestamp,
    type: 'ws-in',
    summary: message type + step number (e.g., "esn_state #42"),
    raw: full message object
  }
  ```
- **Log buffer management**: Keeps last 5000 entries (up from 500)
- **Message tracking**:
  - ESN state → logs as "esn_state #step"
  - Gravity state → logs as "gravity_state #step"
  - CfC state → logs as "cfc_state #step"
  - Pipeline layer → logs as "pipeline_layer #layer_idx"
  - All other types logged with message.type

**Technical Details**:
- `_logMessage(type, summary, raw)` helper function
- Automatic subsystem activation:
  - `store.esnActive = true` when first `esn_state` received
  - `store.gravityActive = true` when first `gravity_state` received
  - `store.cfcActive = true` when first `cfc_state` received
- Message fields parsed for future enhancements (mean_gravity, max_gravity, etc.)

## Store Enhancements

**File**: `src/store/index.js`

**New Fields**:
```javascript
// Subsystem Status
backend: 'unknown',         // 'rocm' | 'opencl' | 'cpu'
quantization: 'bf16',       // 'bf16' | 'q4' | 'f32'
esnActive: false,           // ESN subsystem active flag
gravityActive: false,       // Gravity subsystem active flag
cfcActive: false,           // CfC subsystem active flag
```

## Main.js Watchers

**Key watchers added**:
1. **ESN**: Entropy, top1_prob, state_norm sparklines + temp_delta
2. **Gravity**: Mean + max + high-count + dual sparkline
3. **CfC**: state_norm + correction_norm + sparkline
4. **Subsystem Status**: Backend, quantization, ESN, Gravity, CfC badges
5. **Console Filter**: Updates button states on filter change
6. **Generation Reset**: Clears all histories, resets values on new generation

**Utilities**:
- `drawSparkline(canvasId, points, color)` — reusable sparkline renderer

## Color Palette Reference

| Subsystem | Primary Color | Hex | Use Case |
|-----------|---------------|-----|----------|
| ESN | Purple | #a855f7 | Entropy, training status |
| Gravity | Amber | #fbbf24 | Threshold, mean gravity |
| CfC | Sky Blue | #38bdf8 | Liquid network state |
| Q4 | Pink | #f472b6 | Quantization status |
| Success | Cyan | #00ffc8 | Active systems, healthy state |
| Error | Red | #ff4466 | Failures, inactive systems |
| Indigo | Indigo | #818cf8 | CfC correction norm |
| Orange | Orange | #f97316 | Secondary metrics (history length) |

## Files Modified

1. **frontend/index.html**
   - ESN panel: Added 2 new sparkline canvases, temp_delta field
   - Gravity panel: Added 3 new metric fields, updated sparkline
   - CfC panel: Added 2 new fields, sparkline, 4-column layout
   - Live panel: Added Subsystem Status bar at top
   - Console: Added 4 new filter buttons

2. **src/main.js**
   - Added `drawSparkline()` utility
   - Enhanced ESN watcher with 3 parallel sparklines
   - Enhanced Gravity watcher with dual sparkline, high-count estimation
   - Enhanced CfC watcher with correction_norm sparkline
   - Added 5 subsystem status watchers
   - Updated generation reset to clear all new histories

3. **src/store/index.js**
   - Added 5 new subsystem status fields

4. **src/api/websocket.js**
   - Added `_logMessage()` helper function
   - Comprehensive message logging on every dispatch
   - Subsystem activation tracking

5. **src/components/log_console.js**
   - Enhanced filter logic for 4 new message types
   - Fixed container element ID

6. **style.css**
   - Added CSS for 4 new filter button states
   - Color theming: purple, amber, sky blue, pink

## Usage & Interaction

### Live Panel
1. **Subsystem Status Bar** shows what systems are active
2. **ESN Observer** displays triple sparklines for entropy, confidence, state
3. **Semantic Gravity** shows dual sparklines comparing threshold vs mean
4. **CfC Correction** displays state and correction norms with trend

### Console Tab
1. Click filter buttons to view specific message types
2. Each button highlights when active
3. Logs show timestamp, message type, and summary
4. Click any log entry to view full JSON in inspector

### Monitoring
- All sparklines auto-scale to visible range
- Rolling 200-point history prevents memory bloat
- Colors follow IARA design system for quick visual scanning
- Subsystem badges light up as systems activate during inference

## Future Enhancements

- Alert thresholds for high-gravity tokens
- CfC feedback visualization (state corrections over time)
- Gravity semantic distance matrix heatmap
- ESN reservoir activity distribution
- Multi-generation comparison charts
- Export telemetry to CSV/JSON

---

**Implementation Date**: 2026-03-31
**Frontend Version**: Vue3 + Vanilla JS modules
**Dependencies**: None (canvas API, no additional libraries)
