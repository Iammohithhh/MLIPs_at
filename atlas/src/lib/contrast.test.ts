import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { contrastRatio, parseThemeTokens, relativeLuminance, TEXT_TOKENS } from './contrast';

const css = readFileSync(new URL('../styles/global.css', import.meta.url), 'utf8');
const themes = parseThemeTokens(css);

describe('WCAG maths', () => {
  it('matches reference values', () => {
    expect(relativeLuminance('#ffffff')).toBeCloseTo(1, 10);
    expect(relativeLuminance('#000000')).toBe(0);
    expect(contrastRatio('#000', '#fff')).toBeCloseTo(21, 10);
    // #767676 on white is the classic 4.54:1 AA threshold example.
    expect(contrastRatio('#767676', '#ffffff')).toBeCloseTo(4.54, 2);
  });
});

describe('design tokens', () => {
  it('defines the same token names in both themes', () => {
    expect(Object.keys(themes.dark).sort()).toEqual(Object.keys(themes.light).sort());
  });

  for (const theme of ['light', 'dark'] as const) {
    for (const bg of ['canvas', 'surface', 'surface-2'] as const) {
      for (const fg of TEXT_TOKENS) {
        it(`${theme}: ${fg} on ${bg} meets WCAG AA (≥ 4.5:1)`, () => {
          const t = themes[theme];
          expect(t[fg], `missing --atlas-${fg}`).toBeDefined();
          expect(contrastRatio(t[fg]!, t[bg]!)).toBeGreaterThanOrEqual(4.5);
        });
      }
    }
  }
});

describe('model series colours (Observatory)', () => {
  for (const theme of ['light', 'dark'] as const) {
    for (const m of ['m1', 'm2', 'm3', 'm4']) {
      it(`${theme}: ${m} is a graphical mark ≥ 3:1 on canvas and surface`, () => {
        const t = themes[theme];
        expect(t[m], `missing --atlas-${m}`).toBeDefined();
        expect(contrastRatio(t[m]!, t.canvas!)).toBeGreaterThanOrEqual(3);
        expect(contrastRatio(t[m]!, t.surface!)).toBeGreaterThanOrEqual(3);
      });
    }
  }
});
