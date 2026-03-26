// frontend/src/main.js
import { store } from './store/index.js';
import { api } from './api/index.js';

// Import Views
// Note: In a CDN environment without a bundler, we assume these are exported via ESM
import { LiveView } from './views/LiveView.js';
import { LayersView } from './views/LayersView.js';
import { Charts } from './utils/charts.js';

const app = Vue.createApp({
  data() {
    return {
      store
    };
  },
  mounted() {
    console.log('🚀 IARA Neural Dashboard Modularized Started');
    api.init();
    api.fetchModels();
    
    // Initialize Charts after mount
    Vue.nextTick(() => {
      Charts.initLogitsChart('logits-canvas');
    });

    // Tab switching bridge
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('#tab-nav [data-tab]');
      if (btn) this.setTab(btn.dataset.tab);
    });
  },
  methods: {
    setTab(tab) {
      this.store.activeTab = tab;
    }
  }
});

// Register Components
// app.component('layers-view', LayersView);
// app.component('live-view', LiveView);

app.mount('#app');