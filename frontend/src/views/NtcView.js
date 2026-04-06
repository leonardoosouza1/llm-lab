// frontend/src/views/NtcView.js
import { store } from '../store/index.js';

export const NtcView = {
  init() {
    console.log('🗜️ NtcView Module Initialized');
  },

  toggleNtcActive(enabled) {
    store.ntc.active = enabled;
    console.log(`NTC KV-Cache compression: ${enabled ? 'enabled' : 'disabled'}`);
  },

  getCompressionRatio() {
    if (store.ntc.recentTokens === 0) return 0;
    return (store.ntc.recentTokens / Math.max(1, store.ntc.compressedTokens)).toFixed(2);
  }
};
