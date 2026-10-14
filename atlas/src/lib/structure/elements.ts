import { ELEMENTS, type ElementRow } from './elements.generated';

const BY_SYMBOL = new Map<string, ElementRow>(ELEMENTS.map((e) => [e.symbol, e]));

/** Radius used when an element has no tabulated covalent radius (flagged in the UI). */
export const FALLBACK_RADIUS = 1.5;

export function element(symbol: string): ElementRow {
  const e = BY_SYMBOL.get(symbol);
  if (!e) throw new Error(`Unknown element symbol "${symbol}"`);
  return e;
}

export function covalentRadius(symbol: string): { r: number; tabulated: boolean } {
  const r = element(symbol).covalentRadius;
  return r === null ? { r: FALLBACK_RADIUS, tabulated: false } : { r, tabulated: true };
}

export function hexToLinearRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  const srgb = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => v / 255);
  return srgb.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)) as [number, number, number];
}

/**
 * Muted Jmol colour for the canvas: the Jmol hue, desaturated 30% towards its own
 * luminance so element colours stay distinguishable without shouting. On the light
 * theme near-white elements (H, He, …) are pulled down so they stay visible against
 * the background; on the dark theme near-black ones are lifted. sRGB in [0, 1].
 */
export function elementColor(symbol: string, theme: 'light' | 'dark', desaturate = 0.3): [number, number, number] {
  const n = parseInt(element(symbol).color.slice(1), 16);
  let rgb: [number, number, number] = [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  let lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  rgb = rgb.map((c) => c + (lum - c) * desaturate) as typeof rgb;
  lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  if (theme === 'light' && lum > 0.84) rgb = rgb.map((c) => c * (0.84 / lum)) as typeof rgb;
  if (theme === 'dark' && lum < 0.24) rgb = rgb.map((c) => Math.min(1, c + (0.24 - lum))) as typeof rgb;
  return rgb;
}
