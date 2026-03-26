// frontend/src/api/http.js

import { store } from '../store/index.js';
import { addLog } from '../dom/legacy_updates.js'; // The addLog that pushes to store.logs
import { populateModelSelect, updateModelUI, updateModelLoadingBanner, updateVramBar } from '../components/model_loader.js';
import { setApiInstance } from './websocket.js'; // To resolve circular dependency
import { renderTracePanel } from '../charts/energy_chart.js'; // Will be created later

export const api = {
  async health() {
    try {
      const res = await fetch('http://localhost:3000/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      store.httpHealthy = true;
      store.modelReady = data.status === 'ok';
      store.modelLoading = data.model_loading || false;
      if (data.status === 'loading') {
        store.loadingMsg = 'Carregando modelo...';
      }
      updateModelLoadingBanner();
      updateModelUI();
      updateVramBar(data);
      addLog('http', data, `Health: ${data.status} — model: ${data.model}`);
      return data;
    } catch (e) {
      store.httpHealthy = false;
      addLog('error', { error: e.message }, `Health check failed: ${e.message}`);
      return null;
    }
  },

  async generate(prompt, maxTokens, temperature) {
    store.generating = true;
    store.responseText = '';
    store.evalCount = 0;
    store.totalDurationMs = 0;
    store.pipelineComplete = false;

    const body = {
      model: store.modelName,
      prompt,
      max_tokens: maxTokens,
      temperature,
    };
    if (store.selectedImageBase64) {
      body.image_base64 = store.selectedImageBase64;
    }

    addLog('http', body, `POST /api/generate — prompt: "${prompt}"`);

    try {
      const res = await fetch('http://localhost:3000/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      store.responseText = data.response || '';
      store.evalCount = data.eval_count || 0;
      store.totalDurationMs = data.total_duration ? data.total_duration / 1e6 : 0;
      store.generating = false;

      // Performance metrics
      if (store.evalCount > 0 && store.totalDurationMs > 0) {
        store.tokensPerSec = store.evalCount / (store.totalDurationMs / 1000);
      }
      store.msPerLayer = Object.values(store.layerData)
        .reduce((s, l) => s + (l.time_us || 0), 0) / 1000 / 36;

      addLog('http', data, `Generate done — ${store.evalCount} tokens in ${store.totalDurationMs.toFixed(0)}ms`);

      // updateGenerateResultUI(data); // This is handled by Vue app
      return data;
    } catch (e) {
      store.generating = false;
      addLog('error', { error: e.message }, `Generate failed: ${e.message}`);
      return null;
    }
  },

  async fetchTrace() {
    try {
      const res = await fetch('http://localhost:3000/api/debug/last-trace');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      store.trace = data;
      store.anomalyCount = (data.anomalies || []).length;
      addLog('http', data, `Trace fetched — resonance: ${data.iara_resonance_score}`);
      renderTracePanel(); // This will need to be made reactive/component later
      return data;
    } catch (e) {
      addLog('error', { error: e.message }, `Fetch trace failed: ${e.message}`);
      return null;
    }
  },

  async fetchModels() {
    try {
      const res = await fetch('http://localhost:3000/api/models');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      store.availableModels = data.models || [];
      populateModelSelect();
      addLog('http', data, `Models: ${store.availableModels.map(m => m.name).join(', ')}`);
      return data.models;
    } catch (e) {
      addLog('error', { error: e.message }, `fetchModels failed: ${e.message}`);
      return null;
    }
  },

  async loadModel(name) {
    if (!name) return;
    try {
      store.modelLoading = true;
      store.modelReady = false;
      store.loadingMsg = `Carregando '${name}'...`;
      updateModelUI();
      const res = await fetch('http://localhost:3000/api/load-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      addLog('http', data, `load-model accepted: ${name}`);
      return data;
    } catch (e) {
      store.modelLoading = false;
      store.loadingMsg = `Erro: ${e.message}`;
      updateModelUI();
      addLog('error', { error: e.message }, `loadModel failed: ${e.message}`);
      return null;
    }
  },
};

// Set API instance for websocket module to resolve circular dependency
setApiInstance(api);