// frontend/src/components/model_loader.js

import { el, addLog } from '../utils.js'; // addLog will be replaced by the store-bound version
import { store } from '../store/index.js';

/**
 * Updates the VRAM bar in the UI.
 * @param {object} health - Health data from the API.
 */
export function updateVramBar(health) {
  const section   = el('vram-section');
  const barModel  = el('vram-bar-model');
  const label     = el('vram-label');
  const chipModel = el('vram-model-chip');
  const chipFree  = el('vram-free-chip');
  if (!section) return;

  const modelGb = health.model_vram_gb || 0;
  const totalGb = health.vram_total_gb || 0;
  const freeGb  = health.vram_free_gb  || 0;
  const usedGb  = totalGb - freeGb;

  if (modelGb === 0 && totalGb === 0) { section.style.display = 'none'; return; }
  section.style.display = 'block';

  // Barra mostra pesos do modelo como % do total
  const pct = totalGb > 0 ? Math.min(100, (modelGb / totalGb) * 100) : 0;
  barModel.style.width = pct.toFixed(1) + '%';

  // Cor da barra: verde < 70%, amarelo < 90%, vermelho >= 90%
  const usedPct = totalGb > 0 ? (usedGb / totalGb) * 100 : 0;
  if (usedPct >= 90)      barModel.style.background = 'linear-gradient(90deg,#dc2626,#ef4444)';
  else if (usedPct >= 70) barModel.style.background = 'linear-gradient(90deg,#d97706,#f59e0b)';
  else                    barModel.style.background = 'linear-gradient(90deg,#7c3aed,#a855f7)';

  label.textContent = totalGb > 0
    ? `${usedGb.toFixed(1)} / ${totalGb.toFixed(1)} GB`
    : `${modelGb.toFixed(2)} GB`;
  chipModel.textContent = `pesos: ${modelGb.toFixed(2)} GB`;
  chipFree.textContent  = freeGb > 0 ? `livre: ${freeGb.toFixed(1)} GB` : 'livre: —';
}

/**
 * Populates the <select id="model-select"> from store.availableModels.
 */
export function populateModelSelect() {
  const sel = el('model-select');
  if (!sel) return;

  // Keep current selection if still valid
  const prev = sel.value;
  sel.innerHTML = '<option value="" disabled>— selecione um modelo —</option>';

  store.availableModels.forEach(function(m) {
    const opt = document.createElement('option');
    opt.value = m.name;
    const layers = m.num_layers ? `${m.num_layers}L` : '';
    const hidden = m.hidden_size ? `${m.hidden_size}h` : '';
    const dtype  = m.torch_dtype ? ` (${m.torch_dtype})` : '';
    const badge  = m.model_kind === 'vlm' ? ' 🖼 VLM' :
                   m.model_kind === 'causal_lm' ? ' 📝 LLM' : '';
    opt.textContent = `${m.name}${badge}  ${layers} ${hidden}${dtype}`.trim();
    if (m.model_kind === 'vlm') opt.style.color = '#00d4ff';
    sel.appendChild(opt);
  });

  if (prev && store.availableModels.some(m => m.name === prev)) {
    sel.value = prev;
  } else {
    sel.value = '';
  }
}

/**
 * Updates model-related UI elements.
 */
export function updateModelUI() {
  const badge    = el('model-status-badge');
  const progress = el('model-load-progress');
  const loadMsg  = el('model-load-msg');
  const loadBtn  = el('load-model-btn');
  const infoRow  = el('model-info-row');

  if (store.modelLoading) {
    if (badge)    { badge.className = 'model-status-badge model-status--loading'; badge.textContent = 'loading'; }
    if (progress) progress.style.display = 'flex';
    if (loadMsg)  loadMsg.textContent = store.loadingMsg;
    if (loadBtn)  loadBtn.disabled = true;
  } else if (store.modelReady) {
    if (badge)    { badge.className = 'model-status-badge model-status--ready'; badge.textContent = 'ready'; }
    if (progress) progress.style.display = 'none';
    if (loadBtn)  loadBtn.disabled = false;
    // Enable prompt input
    const promptEl = el('prompt-input');
    const sendEl   = el('send-btn');
    if (promptEl) { promptEl.disabled = false; promptEl.placeholder = 'Digite seu prompt...'; }
    if (sendEl)   sendEl.disabled = false;
    // Show model info chips
    if (infoRow) {
      const m = store.availableModels.find(function(x) { return x.name === store.modelName; });
      if (m) {
        infoRow.style.display = 'flex';
        infoRow.innerHTML = [
          m.model_kind === 'vlm' ? `<span class="model-info-chip model-chip-vlm">VLM 🖼</span>` : '',
          m.has_vision           ? `<span class="model-info-chip model-chip-vision">vision</span>` : '',
          m.num_layers        ? `<span class="model-info-chip"><strong>${m.num_layers}</strong> layers</span>` : '',
          m.hidden_size       ? `<span class="model-info-chip">hidden <strong>${m.hidden_size}</strong></span>` : '',
          m.intermediate_size ? `<span class="model-info-chip">ffn <strong>${m.intermediate_size}</strong></span>` : '',
          m.num_heads         ? `<span class="model-info-chip"><strong>${m.num_heads}</strong> heads</span>` : '',
          m.num_kv_heads      ? `<span class="model-info-chip"><strong>${m.num_kv_heads}</strong> kv</span>` : '',
          m.vocab_size        ? `<span class="model-info-chip">vocab <strong>${m.vocab_size}</strong></span>` : '',
          m.torch_dtype       ? `<span class="model-info-chip">${m.torch_dtype}</span>` : '',
        ].join('');
      }
    }
  } else {
    if (badge)    { badge.className = 'model-status-badge model-status--idle'; badge.textContent = 'idle'; }
    if (progress) progress.style.display = 'none';
    if (loadBtn)  loadBtn.disabled = false;
  }
}

/**
 * Updates the model loading banner visibility and message.
 */
export function updateModelLoadingBanner() {
  const banner = el('model-loading-banner');
  if (!banner) return;
  if (store.modelReady) {
    banner.style.display = 'none';
  } else {
    banner.style.display = 'flex';
    const msgEl = banner.querySelector('#loading-msg');
    if (msgEl) msgEl.textContent = store.loadingMsg;
  }
}

/**
 * Shows/hides the image upload area based on model capabilities.
 */
export function updateImageUploadArea() {
  const area = el('image-upload-area');
  if (!area) return;
  const show = store.modelKind === 'vlm' || store.hasVision;
  area.style.display = show ? 'flex' : 'none';
}

/**
 * Binds model selection and loading events.
 * @param {object} api - The HTTP API manager instance.
 */
export function bindModelEvents(api) {
  const loadModelBtn = el('load-model-btn');
  const modelSelectEl = el('model-select');

  if (modelSelectEl) {
    modelSelectEl.addEventListener('change', function() {
      store.selectedModelName = modelSelectEl.value;
      const m = store.availableModels.find(function(x) { return x.name === modelSelectEl.value; });
      store.modelKind = m ? (m.model_kind || 'unknown') : 'unknown';
      store.hasVision = m ? (m.has_vision || false) : false;
      updateImageUploadArea();

      // Show info chips for selected (not yet loaded) model
      const infoRow = el('model-info-row');
      if (infoRow && !store.modelLoading) {
        if (m) {
          infoRow.style.display = 'flex';
          infoRow.innerHTML = [
            m.num_layers        ? `<span class="model-info-chip"><strong>${m.num_layers}</strong> layers</span>` : '',
            m.hidden_size       ? `<span class="model-info-chip">hidden <strong>${m.hidden_size}</strong></span>` : '',
            m.intermediate_size ? `<span class="model-info-chip">ffn <strong>${m.intermediate_size}</strong></span>` : '',
            m.num_heads         ? `<span class="model-info-chip"><strong>${m.num_heads}</strong> heads</span>` : '',
            m.num_kv_heads      ? `<span class="model-info-chip"><strong>${m.num_kv_heads}</strong> kv</span>` : '',
            m.vocab_size        ? `<span class="model-info-chip">vocab <strong>${m.vocab_size}</strong></span>` : '',
            m.torch_dtype       ? `<span class="model-info-chip">${m.torch_dtype}</span>` : '',
          ].join('');
        }
      }
    });
  }

  if (loadModelBtn) {
    loadModelBtn.addEventListener('click', function() {
      const name = modelSelectEl ? modelSelectEl.value : store.selectedModelName;
      if (!name) { /* addLog('error', null, 'Selecione um modelo primeiro'); */ return; } // TODO: Re-introduce addLog
      store.selectedModelName = name;
      api.loadModel(name);
    });
  }
}
