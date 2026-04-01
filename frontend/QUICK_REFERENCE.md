# Frontend Enhancements — Quick Reference

## What Changed

### Visual Enhancements
- **ESN Panel**: 3 sparklines (entropy, top1_prob, state_norm) + temp_delta value
- **Gravity Panel**: 3 metric fields (mean, max, high-count) + dual sparkline
- **CfC Panel**: 2 metric fields (state_norm, correction_norm) + sparkline
- **Status Bar**: 5 badges showing Backend, Quantization, ESN, Gravity, CfC active/inactive

### Console Improvements
- 4 new filter buttons: ESN, Gravity, CfC, Pipeline (color-coded)
- All WebSocket messages now logged with timestamp and summary
- 5000 message buffer (up from 500)

## Key Files Modified

| File | Purpose | Changes |
|------|---------|---------|
| `index.html` | UI Layout | Added panels, status bar, filter buttons |
| `src/main.js` | Reactive Logic | Added watchers for sparklines, badges |
| `src/store/index.js` | State | Added backend, quantization, subsystem flags |
| `src/api/websocket.js` | Logging | Added `_logMessage()`, message tracking |
| `src/components/log_console.js` | Filtering | Enhanced filter logic |
| `style.css` | Theming | Added filter button colors |

## Color Coding

```
ESN:      Purple (#a855f7) ▢
Gravity:  Amber (#fbbf24)  ▢
CfC:      Sky Blue (#38bdf8) ▢
Pipeline: Pink (#f472b6)   ▢
Success:  Cyan (#00ffc8)   ▢
```

## Usage Examples

### Check System Status
Look at the status bar at top of Live panel:
- **Backend**: Shows "ROCM", "OPENCL", or "unknown"
- **Quantization**: Shows "BF16" (cyan) or "Q4" (pink)
- **Subsystems**: ESN/Gravity/CfC show "ON" (colored) or "OFF" (gray)

### Monitor ESN Training
Watch the ESN Observer panel for:
1. Entropy sparkline (purple) — should show variations
2. Top-1 Prob sparkline (amber) — confidence increasing over time
3. State Norm sparkline (amber) — network state evolution

### Track Gravity Dynamics
In Gravity panel:
1. Mean Gravity (amber) vs Threshold (sky blue) overlap shows active region
2. Max Gravity value indicates peak importance
3. High-count shows how many tokens exceed threshold

### Debug with Console Filters
In Console tab:
1. Click "ESN" to see all ESN state updates
2. Click "Gravity" to trace semantic importance evolution
3. Click "CfC" to monitor liquid network corrections
4. Click "Pipeline" to follow forward pass progress

## Data Flow

```
WebSocket Message
       ↓
dispatch() in websocket.js
       ↓
_logMessage() → store.logs
       ↓
store.esnLatest / gravityLatest / cfcLatest
       ↓
Watchers in main.js
       ↓
DOM Updates + Canvas Sparklines
       ↓
Rendered in Live Panel / Console Tab
```

## Performance Notes

- Sparklines: 200-point rolling history per line
- Canvas: Lightweight 2D rendering (no external libraries)
- Logging: 5000 message buffer, auto-trim on overflow
- Store: Reactive updates only when data changes
- No FPS impact: Separate canvas per metric

## Debugging Tips

1. **Check console filters**: If messages not showing, verify filter button is active
2. **Verify timestamps**: Log entries show HH:MM:SS, helps trace timing issues
3. **Inspect raw JSON**: Click log entry → view full message in JSON inspector
4. **Watch subsystem badges**: They light up when first subsystem message arrives
5. **Monitor memory**: Open DevTools, watch store.logs.length (should stay under 5000)

## Next Steps

To add more observability:
1. Add fields to `store` (new metrics)
2. Update WebSocket `dispatch()` to track new fields
3. Add canvas in HTML (if sparkline needed)
4. Add watcher in `main.js` with drawing logic
5. Add filter button in Console (if needed)
6. Add CSS for theming (if needed)

---

Last updated: 2026-03-31
