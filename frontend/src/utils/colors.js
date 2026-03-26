// frontend/src/utils/colors.js

/**
 * Generates a thermal color string based on a normalized value [0, 1].
 */
export function thermalColor(v) {
  const colors = [
    [0, 20, 40],    // Deep blue
    [0, 100, 255],  // Blue
    [0, 255, 200],  // Cyan
    [255, 255, 0],  // Yellow
    [255, 128, 0],  // Orange
    [255, 0, 0]     // Red
  ];

  if (v <= 0) return `rgb(${colors[0].join(',')})`;
  if (v >= 1) return `rgb(${colors[colors.length - 1].join(',')})`;

  const scaled = v * (colors.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;

  const c1 = colors[idx];
  const c2 = colors[idx + 1];

  const r = Math.round(c1[0] + (c2[0] - c1[0]) * frac);
  const g = Math.round(c1[1] + (c2[1] - c1[1]) * frac);
  const b = Math.round(c1[2] + (c2[2] - c1[2]) * frac);

  return `rgb(${r},${g},${b})`;
}
