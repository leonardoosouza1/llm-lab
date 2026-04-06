// frontend/src/views/QuantView.js
import { store } from '../store/index.js';

export const QuantView = {
  init() {
    console.log('💾 QuantView Module Initialized');
  },

  setQuantMode(mode) {
    // Send mode change to backend via WebSocket (if needed)
    if (store.quant.mode !== mode) {
      store.quant.mode = mode;
      console.log(`Quantization mode changed to: ${mode}`);
    }
  },

  getQuantModes() {
    return ['BF16', 'Q4', 'FP8', 'BitNet'];
  }
};
