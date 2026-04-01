// frontend/src/components/log_console.js

import { el, syntaxHighlight, escapeHtml } from '../utils.js';
import { store } from '../store/index.js';

let _consoleAutoScroll = true;
let _selectedLogId = null;

export function renderConsoleLogs() {
  const container = el('log-list');
  if (!container) return;

  const filter = store.logFilter;
  const entries = store.logs.filter(function(e) {
    if (filter === 'ALL') return true;
    if (filter === 'WS')   return e.type === 'ws-in' || e.type === 'ws-out';
    if (filter === 'esn_state') return e.summary && e.summary.includes('esn_state');
    if (filter === 'gravity_state') return e.summary && e.summary.includes('gravity_state');
    if (filter === 'cfc_state') return e.summary && e.summary.includes('cfc_state');
    if (filter === 'pipeline_layer') return e.summary && e.summary.includes('pipeline_layer');
    if (filter === 'HTTP') return e.type === 'http';
    if (filter === 'ERROR') return e.type === 'error';
    return true;
  });

  container.innerHTML = '';
  entries.forEach(function(entry) {
    const div = document.createElement('div');
    div.style.cssText = `
      display:flex;align-items:flex-start;gap:8px;
      padding:4px 8px;border-bottom:1px solid rgba(255,255,255,0.03);
      cursor:pointer;font-family:monospace;font-size:11px;
    `;
    div.addEventListener('click', function() {
      _selectedLogId = entry.id;
      renderJsonInspector(entry.raw);
    });
    div.addEventListener('mouseover', function() {
      this.style.background = 'rgba(255,255,255,0.03)';
    });
    div.addEventListener('mouseout', function() {
      this.style.background = '';
    });

    const typeColors = {
      'ws-in':  '#00d4ff',
      'ws-out': '#00ff88',
      'http':   '#a855f7',
      'error':  '#ff4466',
    };
    const color = typeColors[entry.type] || '#6b7fa3';

    const rawStr = typeof entry.raw === 'string'
      ? entry.raw
      : JSON.stringify(entry.raw);
    const truncated = rawStr.length > 80 ? rawStr.slice(0, 80) + '…' : rawStr;

    div.innerHTML = `
      <span style="color:#555568;flex-shrink:0">${entry.ts}</span>
      <span style="color:${color};flex-shrink:0;width:52px">${entry.type}</span>
      <span style="color:#8888a0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(truncated)}</span>
    `;

    container.appendChild(div);
  });

  if (_consoleAutoScroll) {
    container.scrollTop = container.scrollHeight;
  }
}

export function renderJsonInspector(raw) {
  const inspector = el('json-inspector');
  if (!inspector) return;

  const str = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
  inspector.innerHTML = `<pre style="margin:0;white-space:pre-wrap;word-break:break-all">${syntaxHighlight(str)}</pre>`;
}

export function setConsoleAutoScroll(value) {
  _consoleAutoScroll = value;
}