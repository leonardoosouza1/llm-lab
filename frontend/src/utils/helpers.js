// frontend/src/utils/helpers.js

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

export function timestamp() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const ms = String(d.getMilliseconds()).padStart(3, '0');
  return `${hh}:${mm}:${ss}.${ms}`;
}
