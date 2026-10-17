import { describe, expect, it } from 'vitest';
import { fmtSigned } from './format';

describe('fmtSigned', () => {
  it('signs values and uses a true minus', () => {
    expect(fmtSigned(0.70756, 4, 'eV')).toBe('+0.7076 eV');
    expect(fmtSigned(-6.2541, 3, 'eV/Å')).toBe('−6.254 eV/Å');
    expect(fmtSigned(0, 3)).toBe('0.000');
  });
  it('keeps the exponent sign of small values (regression: +2.43e15 for 2.43e-15)', () => {
    expect(fmtSigned(2.4296e-15, 3, 'eV/Å')).toBe('+2.43e−15 eV/Å');
    expect(fmtSigned(-2.5163e-15, 3)).toBe('−2.52e−15');
    expect(fmtSigned(-4e-4, 3)).toBe('−4.00e−4');
  });
  it('shows missing values as missing, never as zero', () => {
    expect(fmtSigned(null)).toBe('—');
    expect(fmtSigned(undefined)).toBe('—');
    expect(fmtSigned(NaN)).toBe('—');
  });
});

describe('fmtD', () => {
  it('signs displacements with a true minus and leaves the baseline unsigned', async () => {
    const { fmtD } = await import('./format');
    expect(fmtD(0.2)).toBe('+0.200');
    expect(fmtD(-0.125)).toBe('−0.125');
    expect(fmtD(0)).toBe('0.000');
  });
});
