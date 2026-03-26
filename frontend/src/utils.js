// frontend/src/utils.js

/**
 * Utility: safe element getter
 * @param {string} id
 * @returns {HTMLElement | null}
 */
export function el(id) { return document.getElementById(id); }

/**
 * Utility: thermal color mapping
 * Maps 0→#1a1a2e, 0.5→#a855f7, 1→#00ffc8
 * @param {number} value
 * @returns {string} CSS color string
 */
export function thermalColor(value) {
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.5) {
    // #1a1a2e → #a855f7
    const t = v * 2;
    const r = Math.round(0x1a + (0xa8 - 0x1a) * t);
    const g = Math.round(0x1a + (0x55 - 0x1a) * t);
    const b = Math.round(0x2e + (0xf7 - 0x2e) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    // #a855f7 → #00ffc8
    const t = (v - 0.5) * 2;
    const r = Math.round(0xa8 + (0x00 - 0xa8) * t);
    const g = Math.round(0x55 + (0xff - 0x55) * t);
    const b = Math.round(0xf7 + (0xc8 - 0xf7) * t);
    return `rgb(${r},${g},${b})`;
  }
}

/**
 * Utility: thermal color as [r,g,b] array
 * @param {number} value
 * @returns {number[]} [r,g,b]
 */
export function thermalColorRGB(value) {
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.5) {
    const t = v * 2;
    return [
      Math.round(0x1a + (0xa8 - 0x1a) * t),
      Math.round(0x1a + (0x55 - 0x1a) * t),
      Math.round(0x2e + (0xf7 - 0x2e) * t),
    ];
  } else {
    const t = (v - 0.5) * 2;
    return [
      Math.round(0xa8 + (0x00 - 0xa8) * t),
      Math.round(0x55 + (0xff - 0x55) * t),
      Math.round(0xf7 + (0xc8 - 0xf7) * t),
    ];
  }
}

/**
 * Utility: timestamp HH:MM:SS.mmm
 * @returns {string}
 */
export function timestamp() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const ms = String(d.getMilliseconds()).padStart(3, '0');
  return `${hh}:${mm}:${ss}.${ms}`;
}

/**
 * Utility: HTML escape
 * @param {string} text
 * @returns {string}
 */
export function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = String(text);
  return d.innerHTML;
}

/**
 * Utility: JSON syntax highlight
 * @param {any} json
 * @returns {string}
 */
export function syntaxHighlight(json) {
  const str = typeof json === 'string' ? json : JSON.stringify(json, null, 2);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/("(\u[a-zA-Z0-9]{4}|\[^u]|[^"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function(match) {
      if (/^"/.test(match)) {
        if (/:$/.test(match)) {
          return `<span style="color:#00d4ff">${match}</span>`;
        }
        return `<span style="color:#00ff88">${match}</span>`;
      }
      if (/true|false/.test(match)) {
        return `<span style="color:#a855f7">${match}</span>`;
      }
      if (/null/.test(match)) {
        return `<span style="color:#ff4466">${match}</span>`;
      }
      return `<span style="color:#ffb347">${match}</span>`;
    });
}

// Log helper - now accepts a log handler function
let _logId = 0;
export function createLogEntry(typeOrLevel, raw, text) {
  // Support old-style addLog(message, level) and new-style addLog(type, raw, text)
  let type, rawObj, textStr;
  if (text === undefined) {
    // Legacy call: addLog(message, level)
    type = (raw === 'error') ? 'error' : (raw === 'warning' ? 'error' : 'http');
    rawObj = typeOrLevel;
    textStr = String(typeOrLevel);
  } else {
    type = typeOrLevel;
    rawObj = raw;
    textStr = text || (typeof raw === 'string' ? raw : JSON.stringify(raw));
  }

  return {
    id: ++_logId,
    ts: timestamp(),
    type,
    raw: rawObj,
    text: textStr,
  };
}