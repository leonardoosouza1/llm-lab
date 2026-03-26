// frontend/src/views/app.js

import { store } from '../store/index.js';
import { api } from '../api/http.js';
import { wsManager } from '../api/websocket.js';
import { initFrameHeatmap, triggerFrameHeatmap } from '../charts/frame_heatmap.js';
import { initLogitsChart, updateLogitsChart } from '../charts/logits_chart.js';
import { initForceGraph, updateForceGraph, isForceGraphInitialized } from '../charts/force_graph.js';
import { renderTracePanel } from '../charts/energy_chart.js';
import { renderConsoleLogs, setConsoleAutoScroll } from '../components/log_console.js';
import { renderLayerGrid, renderLayerHeatmapCells, setLayerSort } from '../components/layer_grid.js';
import { el } from '../utils.js';

export function mountVueApp() {
  if (typeof Vue === 'undefined') {
    // Vue not loaded — fallback to pure DOM mode (this case should not happen if CDN is loaded correctly)
    console.error('Vue not loaded — running in DOM-only mode');
    return;
  }

  const { createApp, reactive, ref, computed, onMounted, watch, nextTick } = Vue;

  const app = createApp({
    setup() {
      const s = store; // Use the imported store

      // Tab switching
      function switchTab(tab) {
        s.activeTab = tab;
        nextTick(function() {
          if (tab === 'graph' && !isForceGraphInitialized()) {
            initForceGraph();
          }
          if (tab === 'layers') {
            renderLayerGrid();
            renderLayerHeatmapCells();
          }
          if (tab === 'trace' && s.trace) {
            renderTracePanel();
          }
          if (tab === 'console') {
            renderConsoleLogs();
          }
        });
      }

      // Generate via HTTP
      async function generate() {
        if (s.generating || !s.prompt.trim()) return;
        await api.generate(s.prompt, s.maxTokens, s.temperature);
      }

      // Resonate via WS
      function resonate() {
        if (!s.prompt.trim()) return;
        wsManager.resonate(s.prompt);
        s.pipelineComplete = false;
        s.layerData = {};
      }

      // Send stimulus via WS
      function sendStimulus() {
        if (!s.prompt.trim()) return;
        wsManager.stimulus(s.prompt);
      }

      // Reset
      function reset() {
        wsManager.reset();
        s.layerData = {};
        s.topTokens = [];
        s.pipelineComplete = false;
        s.responseText = '';
        s.trace = null;
      }

      // List layers
      function listLayers() {
        wsManager.listLayers();
      }

      // Select layer
      function selectLayer(name) {
        wsManager.selectLayer(name);
        s.selectedLayer = name;
      }

      // Fetch trace
      function fetchTrace() {
        api.fetchTrace();
      }

      // Console filter
      function setLogFilter(f) {
        s.logFilter = f;
        renderConsoleLogs();
      }

      // Layer sort
      // setLayerSort is directly imported and used

      // Watch layerData changes
      watch(
        () => Object.keys(s.layerData).length,
        function() {
          if (s.activeTab === 'layers') {
            renderLayerGrid();
            renderLayerHeatmapCells();
          }
          if (s.activeTab === 'graph' && isForceGraphInitialized()) {
            updateForceGraph(s.layerData);
          }
        }
      );

      // Watch topTokens
      watch(
        () => s.topTokens.length,
        function() {
          updateLogitsChart();
        }
      );

      // Watch logs
      watch(
        () => s.logs.length,
        function() {
          if (s.activeTab === 'console') {
            renderConsoleLogs();
          }
        }
      );

      // Watch trace
      watch(
        () => s.trace,
        function(t) {
          if (t && s.activeTab === 'trace') {
            renderTracePanel();
          }
        }
      );

      // --- Event listeners for non-Vue managed elements inside #app ---
      // This listener handles the tab navigation using document.querySelectorAll
      // The original app.js had a delegated click listener on document.
      // This can be kept if it's the only way to catch clicks on data-tab elements
      // that are not part of a Vue component template or are dynamically added.
      // For now, let's keep the Vue way and assume buttons are within Vue template.
      // If we decide to keep the document-level delegation, it will go to main.js

      onMounted(function() {
        // Resize canvases for proper display
        nextTick(function() {
          initFrameHeatmap();
          initLogitsChart();
        });

        // Initialize button actions for layer sorting
        document.querySelectorAll('.layers-controls button[data-sort]').forEach(button => {
          button.addEventListener('click', (event) => {
            document.querySelectorAll('.layers-controls button[data-sort]').forEach(btn => btn.classList.remove('ctrl-btn--active'));
            event.currentTarget.classList.add('ctrl-btn--active');
            setLayerSort(event.currentTarget.dataset.sort);
          });
        });

        // Initialize button actions for console log filter
        document.querySelectorAll('.console-filter-bar button[data-filter]').forEach(button => {
            button.addEventListener('click', (event) => {
                document.querySelectorAll('.console-filter-bar button[data-filter]').forEach(btn => btn.classList.remove('filter-btn--active'));
                event.currentTarget.classList.add('filter-btn--active');
                setLogFilter(event.currentTarget.dataset.filter);
            });
        });

        // Console clear button
        el('console-clear-btn')?.addEventListener('click', () => {
          s.logs = [];
        });

        // Console pause button
        el('console-pause-btn')?.addEventListener('click', (event) => {
          const btn = event.currentTarget;
          const isPaused = btn.getAttribute('aria-pressed') === 'true';
          btn.setAttribute('aria-pressed', String(!isPaused));
          setConsoleAutoScroll(isPaused); // if paused, set auto-scroll to false
        });

      });


      return {
        store: s,
        switchTab,
        generate,
        resonate,
        sendStimulus,
        reset,
        listLayers,
        selectLayer,
        fetchTrace,
        setLogFilter,
        setLayerSort,
      };
    },
  });

  // Only mount if #app exists (new layout)
  const appEl = el('app');
  if (appEl) {
    app.mount('#app');
  }
}