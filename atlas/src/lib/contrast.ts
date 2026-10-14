// WCAG 2.x relative luminance and contrast ratio, plus a parser that reads the
// design tokens straight from global.css so swatches and tests use real values.

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  if (!/^[0-9a-f]{6}$/i.test(full)) throw new Error(`Not a hex colour: ${hex}`);
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as [number, number, number];
}

export function relativeLuminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex).map((v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  const [hi, lo] = la > lb ? [la, lb] : [lb, la];
  return (hi + 0.05) / (lo + 0.05);
}

export type ThemeTokens = Record<string, string>;

/** Extract `--atlas-*: #hex` declarations for the light (:root) and dark themes. */
export function parseThemeTokens(css: string): { light: ThemeTokens; dark: ThemeTokens } {
  const block = (selector: RegExp): ThemeTokens => {
    const m = css.match(selector);
    if (!m?.[1]) throw new Error(`Theme block not found: ${selector}`);
    const out: ThemeTokens = {};
    for (const d of m[1].matchAll(/--atlas-([a-z0-9-]+):\s*(#[0-9a-fA-F]{3,6})\s*;/g)) {
      out[d[1]!] = d[2]!.toLowerCase();
    }
    return out;
  };
  return {
    light: block(/:root\s*\{([^}]*)\}/),
    dark: block(/\[data-theme="dark"\]\s*\{([^}]*)\}/),
  };
}

/** Tokens used as text (or as text-like marks) that must meet WCAG AA on canvas and surface. */
export const TEXT_TOKENS = [
  'ink',
  'ink-muted',
  'ink-faint',
  'arch',
  'phys',
  'lit',
  'caveat',
  'fail',
  'verified',
  'neutral',
] as const;
