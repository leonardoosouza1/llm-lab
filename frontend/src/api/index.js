// frontend/src/api/index.js
import { store } from '../store/index.js';
import { wsManager } from './websocket.js';

/**
 * Unified API Service Entry Point
 */
// Same origin the dashboard was served from — works on localhost, LAN IP, or a remote host.
const HTTP_URL = window.location.origin;

export const api = {
  // Re-export WebSocket for legacy or unified access
  ws: wsManager,

  init() {
    wsManager.connect();
  },

  async health() {
    try {
      const res = await fetch(`${HTTP_URL}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      store.httpHealthy = true;
      store.modelReady = data.status === 'ok';
      store.modelLoading = data.model_loading || false;
      if (data.model_loading) {
        store.loadingMsg = 'Carregando modelo...';
      } else if (data.status !== 'ok' && !store.modelReady) {
        store.loadingMsg = 'Aguardando servidor...';
      }
      if (data.model && data.status === 'ok') {
        store.modelName = data.model;
      }
      return data;
    } catch (err) {
      store.httpHealthy = false;
      return null;
    }
  },

  async fetchModels() {
    try {
      const res = await fetch(`${HTTP_URL}/api/models`);
      const data = await res.json();
      store.availableModels = data.models || data;
      store.httpHealthy = true;
    } catch (err) {
      store.httpHealthy = false;
      console.error('Failed to fetch models:', err);
    }
  },

  async loadModel(name) {
    if (!name) return;
    store.modelLoading = true;
    store.modelReady = false;
    store.loadingMsg = `Carregando '${name}'...`;
    
    try {
      const res = await fetch(`${HTTP_URL}/api/load-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      store.modelLoading = false;
      store.loadingMsg = `Erro ao carregar: ${err.message}`;
      console.error('loadModel failed:', err);
    }
  },

  async generate(prompt, maxTokens, temperature) {
    store.generating = true;
    const body = {
      model: store.modelName,
      prompt,
      max_tokens: maxTokens,
      temperature
    };
    if (store.selectedImageBase64) {
      body.image_base64 = store.selectedImageBase64;
    }

    try {
      const res = await fetch(`${HTTP_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      store.responseText = data.response || '';
      store.generating = false;
    } catch (err) {
      store.generating = false;
      console.error('Generate failed:', err);
    }
  }
};
