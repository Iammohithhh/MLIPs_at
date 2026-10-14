import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { assertTexIntact, katexOptions } from './katex-options.mjs';

describe('linked-symbol macro', () => {
  it('wraps symbols in data-sym spans', () => {
    const html = katex.renderToString(String.raw`\sym{rc}{r_c} + \sym{p}{p}`, katexOptions);
    expect(html).toContain('data-sym="rc"');
    expect(html).toContain('data-sym="p"');
  });

  it('keeps MathML output for screen readers', () => {
    const html = katex.renderToString(String.raw`\sym{rc}{r_c}`, katexOptions);
    expect(html).toContain('<math');
  });

  it('rejects untrusted HTML extensions', () => {
    const html = katex.renderToString(String.raw`\htmlClass{x}{y}`, katexOptions);
    expect(html).not.toContain('class="enclosing x"');
  });

  it('fails on LaTeX-incompatible input', () => {
    expect(() => katex.renderToString(String.raw`\frac{1}`, katexOptions)).toThrow();
  });
});

describe('assertTexIntact', () => {
  it('accepts intact TeX', () => {
    expect(() => assertTexIntact(String.raw`\vect{F}_i = -\frac{\partial E}{\partial \vect{R}_i}`)).not.toThrow();
    expect(() => assertTexIntact(String.raw`\mathcal{R}_i^{(T)} \subseteq \{ j : r_{ij} < T r_c \}`)).not.toThrow();
    expect(() => assertTexIntact('r_c')).not.toThrow();
  });

  it('catches escapes consumed by a normal string literal', () => {
    expect(() => assertTexIntact('\vect{F}')).toThrow(/control character/);
    expect(() => assertTexIntact('\beta')).toThrow(/control character/);
    expect(() => assertTexIntact('sym{T}{T}')).toThrow(/lost its backslash/);
    expect(() => assertTexIntact('x = frac{1}{2}')).toThrow(/lost its backslash/);
  });
});
