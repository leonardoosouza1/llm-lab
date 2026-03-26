// frontend/src/utils/dom.js

export function el(id) { return document.getElementById(id); }

export function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = String(text);
  return d.innerHTML;
}

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
