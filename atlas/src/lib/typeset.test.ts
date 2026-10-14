import { describe, expect, it } from 'vitest';
import { typesetInline } from './typeset';

describe('typesetInline', () => {
  it('typesets single-letter subscripted symbols', () => {
    expect(typesetInline('within r_c of i')).toBe('within <i>r</i><sub>c</sub> of i');
    expect(typesetInline('2·r_c')).toBe('2·<i>r</i><sub>c</sub>');
  });
  it('leaves identifiers and longer words alone', () => {
    expect(typesetInline('snake_case and node_feats')).toBe('snake_case and node_feats');
  });
  it('escapes HTML', () => {
    expect(typesetInline('a < b & <script>')).toBe('a &lt; b &amp; &lt;script&gt;');
  });
});
