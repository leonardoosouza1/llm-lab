// frontend/src/views/LiveView.js
import { store } from '../store/index.js';
import { api } from '../api/index.js';

export const LiveView = {
  // Logic for the Live Dashboard panel
  init() {
    console.log('⚡ LiveView Module Initialized');
  },

  async handleGenerate() {
    if (!store.prompt || store.generating) return;
    await api.generate(store.prompt, store.maxTokens, store.temperature);
  },

  handleResonate() {
    if (!store.prompt || store.generating) return;
    api.ws.resonate(store.prompt);
  },

  async handleLoadModel(modelName) {
    if (!modelName || store.modelLoading) return;
    await api.loadModel(modelName);
  }
};
