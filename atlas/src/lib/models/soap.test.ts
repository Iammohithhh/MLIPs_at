import { describe, expect, it } from 'vitest';
import { mulberry32, randomRotation } from '../chem/rng';
import {
  DEFAULT_SOAP,
  GAP18_SOAP,
  bispectrum,
  degeneratePair,
  dot,
  exactKernel,
  fcut,
  kernelDistance,
  latticeEnv,
  normalise,
  overlap,
  pairAngles,
  powerByL,
  powerSpectrum,
  scaledBesselI,
  soapLength,
  soapVector,
  type SoapOptions,
  type Vec3,
} from './soap';

const rot = (vs: Vec3[], R: number[]): Vec3[] =>
  vs.map((v) => [R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2], R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2], R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2]]);

const randomEnv = (n: number, seed: number, R = 4): Vec3[] => {
  const rand = mulberry32(seed);
  return Array.from({ length: n }, () => {
    const r = 1.5 + rand() * (R - 1.5);
    const z = 2 * rand() - 1;
    const ph = 2 * Math.PI * rand();
    const s = Math.sqrt(1 - z * z);
    return [r * s * Math.cos(ph), r * s * Math.sin(ph), r * z] as Vec3;
  });
};

describe('scaled modified spherical Bessel functions', () => {
  // Closed forms: i0 = sinh x / x, i1 = (x cosh x − sinh x)/x², i2 = ((x² + 3) sinh x − 3x cosh x)/x³.
  const closed = (l: number, x: number) => {
    const sh = Math.sinh(x);
    const ch = Math.cosh(x);
    return [sh / x, (x * ch - sh) / (x * x), ((x * x + 3) * sh - 3 * x * ch) / x ** 3][l]! * Math.exp(-x);
  };
  it.each([0.01, 0.3, 1.0, 1.99, 2.01, 5, 17, 60])('matches the closed forms at x = %f', (x) => {
    const I = scaledBesselI(4, x);
    for (let l = 0; l <= 2; l++) {
      // The closed forms cancel catastrophically at small x; compare there only to the series' leading term.
      if (x < 0.5 && l > 0) continue;
      expect(I[l]).toBeCloseTo(closed(l, x), 12);
    }
  });
  it('agrees between the series and the recurrence across x = 2', () => {
    const a = scaledBesselI(12, 1.99999999);
    const b = scaledBesselI(12, 2);
    for (let l = 0; l <= 12; l++) expect(b[l]! / a[l]!).toBeCloseTo(1, 6);
  });
  it('satisfies the recurrence i_{l−1} − i_{l+1} = (2l+1)/x i_l at large x', () => {
    const x = 120;
    const I = scaledBesselI(14, x);
    for (let l = 1; l < 14; l++) expect((I[l - 1]! - I[l + 1]!) / (((2 * l + 1) / x) * I[l]!)).toBeCloseTo(1, 10);
  });
  it('is δ_l0 at x = 0 and behaves as x^l/(2l+1)!! for small x', () => {
    expect(Array.from(scaledBesselI(3, 0))).toEqual([1, 0, 0, 0]);
    expect(scaledBesselI(3, 1e-3)[3]! / (1e-9 / 105)).toBeCloseTo(1, 2);
  });
});

describe('cutoff', () => {
  it('is 1 inside r_c − d, ½ halfway down the ramp, 0 beyond r_c', () => {
    expect(fcut(3.9, 5, 1)).toBe(1);
    expect(fcut(4.5, 5, 1)).toBeCloseTo(0.5, 14);
    expect(fcut(5, 5, 1)).toBeCloseTo(0, 14);
    expect(fcut(5.01, 5, 1)).toBe(0);
  });
});

describe('SOAP power spectrum', () => {
  const o: SoapOptions = { ...DEFAULT_SOAP, rc: 4.5, lmax: 5, nr: 14 };
  const env = randomEnv(9, 4);

  it('has (lmax + 1) · nr(nr + 1)/2 components, and the Si GAP vector 716 = 13 · 55 + 1', () => {
    expect(powerSpectrum(env, o).length).toBe(soapLength(o));
    expect((GAP18_SOAP.lmax + 1) * (GAP18_SOAP.nmax * (GAP18_SOAP.nmax + 1)) / 2 + 1).toBe(GAP18_SOAP.dims);
  });

  it('is invariant under rotations, reflections and permutations of the neighbours', () => {
    const p = powerSpectrum(env, o);
    const R = randomRotation(mulberry32(9));
    const q = powerSpectrum(rot(env, R), o);
    const mirror = powerSpectrum(env.map((v) => [-v[0], v[1], v[2]] as Vec3), o);
    const perm = powerSpectrum([...env].reverse(), o);
    const scale = Math.sqrt(dot(p, p));
    for (let k = 0; k < p.length; k++) {
      expect(Math.abs(q[k]! - p[k]!) / scale).toBeLessThan(1e-12);
      expect(Math.abs(mirror[k]! - p[k]!) / scale).toBeLessThan(1e-12);
      expect(Math.abs(perm[k]! - p[k]!) / scale).toBeLessThan(1e-12);
    }
  });

  it('dotted with another spectrum, reproduces the exact rotation-averaged overlap (closed-form radial integrals)', () => {
    const other = randomEnv(7, 11);
    const fine: SoapOptions = { ...o, nr: 40 };
    const grid = dot(powerSpectrum(env, fine), powerSpectrum(other, fine));
    const exact = exactKernel(env, other, fine);
    expect(grid / exact).toBeCloseTo(1, 8);
    const self = dot(powerSpectrum(env, fine), powerSpectrum(env, fine));
    expect(self / exactKernel(env, env, fine)).toBeCloseTo(1, 8);
  });

  it('is the Haar average of the squared overlap (Monte Carlo over random rotations)', () => {
    const o2: SoapOptions = { ...o, lmax: 24, nr: 30 };
    const a: Vec3[] = [[1.8, 0, 0], [0, 2.2, 0.4]];
    const b: Vec3[] = [[0, 0, 2.0], [1.5, 1.2, 0], [-1.9, 0.3, 0.2]];
    const rand = mulberry32(5);
    const n = 40000;
    let s = 0;
    for (let k = 0; k < n; k++) s += overlap(a, b, randomRotation(rand), o2) ** 2;
    expect((s / n) / exactKernel(a, b, o2)).toBeCloseTo(1, 1.5);
  });

  it('converges as the radial grid is refined', () => {
    const other = randomEnv(7, 11);
    const k = (nr: number) => dot(soapVector(env, { ...o, nr }), soapVector(other, { ...o, nr }));
    expect(Math.abs(k(20) - k(40))).toBeLessThan(2e-5);
    expect(Math.abs(k(28) - k(40))).toBeLessThan(1e-6);
  });

  it('splits |p|² over angular channels that sum to one', () => {
    const share = powerByL(powerSpectrum(env, o), o);
    expect(share.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12);
  });

  it('gives a normalised kernel of 1 for an environment with itself and < 1 for fcc vs hcp', () => {
    const d = 2.556;
    const fcc = soapVector(latticeEnv('fcc', d, 5), DEFAULT_SOAP);
    const hcp = soapVector(latticeEnv('hcp', d, 5), DEFAULT_SOAP);
    expect(kernelDistance(fcc, fcc, 4)).toBeCloseTo(0, 6);
    expect(kernelDistance(fcc, hcp, 4)).toBeGreaterThan(1e-3);
    expect(dot(fcc, fcc)).toBeCloseTo(1, 12);
  });
});

describe('lattices', () => {
  it('have the right first shells at the nearest-neighbour distance', () => {
    const d = 2.5;
    const shell = (k: Parameters<typeof latticeEnv>[0]) => latticeEnv(k, d, d * 1.01).length;
    expect(shell('fcc')).toBe(12);
    expect(shell('hcp')).toBe(12);
    expect(shell('bcc')).toBe(8);
    expect(shell('sc')).toBe(6);
    expect(shell('diamond')).toBe(4);
    // hcp's first neighbour beyond the shared 12 is at √2 d (6 atoms), like fcc's; they differ by angles.
    expect(latticeEnv('hcp', d, d * 1.42).length).toBe(18);
    expect(latticeEnv('fcc', d, d * 1.42).length).toBe(18);
  });
});

describe('the degenerate pair (Pozdnyakov et al. 2020, Fig. 1a)', () => {
  const o: SoapOptions = { ...DEFAULT_SOAP, rc: 4, lmax: 6, nr: 18 };
  const { plus, minus } = degeneratePair(2);

  it('has the same list of angles, 45°, 45°, 90°, 135°, 135°, 180°', () => {
    const a = pairAngles(plus);
    const b = pairAngles(minus);
    const want = [45, 45, 90, 135, 135, 180];
    // acos loses precision next to 180°.
    a.forEach((x, k) => expect(x).toBeCloseTo(want[k]!, 4));
    b.forEach((x, k) => expect(x).toBeCloseTo(want[k]!, 4));
  });

  it('cannot be superimposed: the gaps around the circle differ (1,2,1,4 vs 1,1,3,3 × 45°)', () => {
    const gaps = (ks: number[]) => ks.map((k, i) => (ks[(i + 1) % ks.length]! - k + 8) % 8).sort();
    expect(gaps([0, 1, 3, 4])).not.toEqual(gaps([0, 1, 2, 5]));
  });

  it('has identical SOAP power spectra but different bispectra', () => {
    const p = powerSpectrum(plus, o);
    const q = powerSpectrum(minus, o);
    const scale = Math.sqrt(dot(p, p));
    for (let k = 0; k < p.length; k++) expect(Math.abs(p[k]! - q[k]!) / scale).toBeLessThan(1e-12);
    const centres = [1.5, 2.0, 2.5];
    const b1 = normalise(bispectrum(plus, o, centres));
    const b2 = normalise(bispectrum(minus, o, centres));
    expect(Math.sqrt(Math.max(0, 2 - 2 * dot(b1, b2)))).toBeGreaterThan(1e-3);
  });

  it('bispectrum is rotation invariant', () => {
    const centres = [1.5, 2.0, 2.5];
    const R = randomRotation(mulberry32(3));
    const b1 = bispectrum(plus, o, centres);
    const b2 = bispectrum(rot(plus, R), o, centres);
    const s = Math.sqrt(dot(b1, b1));
    for (let k = 0; k < b1.length; k++) expect(Math.abs(b1[k]! - b2[k]!) / s).toBeLessThan(1e-10);
  });

  it('lifting one neighbour breaks the power-spectrum degeneracy', () => {
    const lifted = degeneratePair(2, 0.3);
    const d = kernelDistance(soapVector(lifted.plus, o), soapVector(lifted.minus, o), 1);
    expect(d).toBeGreaterThan(1e-4);
  });
});
