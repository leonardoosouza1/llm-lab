// frontend/src/views/SaeView.js
import { store } from '../store/index.js';

export const SaeView = {
  init() {
    console.log('🧠 SaeView Module Initialized');
  },

  toggleSaeActive(enabled) {
    store.sae.active = enabled;
    console.log(`SAE interpretability: ${enabled ? 'enabled' : 'disabled'}`);
  },

  getSparsityPercent() {
    return (store.sae.sparsity * 100).toFixed(2);
  },

  getTopFeatures(limit = 20) {
    return store.sae.topFeatures.slice(0, limit);
  }
};
