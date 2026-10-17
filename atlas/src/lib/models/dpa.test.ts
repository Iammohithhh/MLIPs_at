import { describe, expect, it } from 'vitest';
import { mulberry32, randomRotation } from '~/lib/chem/rng';
import {
  BIAS_DATASETS,
  CU_D,
  closePacked,
  DPA3_G1,
  FRAME_SPECS,
  SWITCHES,
  attentionRow,
  cutoffScan,
  derivs,
  dpa1Descriptor,
  dpa3Descriptor,
  fitBias,
  framesFor,
  jumpsAt,
  makeToy,
  pyramid,
  relDiff,
  rotate,
  sDeepPotSE,
  sDpa3,
  sPoly,
  seDescriptor,
  shuffle,
  square,
  tShape,
  tetrahedron,
  wPoly,
  type Vec3,
} from './dpa';

const toy = makeToy();
const fcc = 'fcc' as const;
const hcp = 'hcp' as const;
const env = (k: 'fcc' | 'hcp', R: number): Vec3[] => closePacked(k, CU_D, R);

describe('switching functions', () => {
  it('match the papers at the ends of the switching interval', () => {
    expect(sDeepPotSE(2, 3, 4)).toBeCloseTo(0.5, 12);
    expect(sDeepPotSE(3.5, 3, 4)).toBeCloseTo(0.5 / 3.5, 12);
    expect(wPoly(3.5, 3, 4)).toBeCloseTo(0.5, 12);
    expect(sPoly(4, 3, 4)).toBe(0);
    expect(sDpa3(DPA3_G1.rcs, DPA3_G1.rcs, DPA3_G1.rc)).toBeCloseTo(Math.exp(-1), 12);
  });

  it('have the continuity each paper claims at r_c', () => {
    const rcs = 5.3;
    const rc = 6;
    const j = Object.fromEntries(SWITCHES.map((s) => [s.id, jumpsAt((r) => s.f(r, rcs, rc), rc)]));
    // Hard cutoff: the value jumps by 1/r_c.
    expect(j.hard!.value).toBeCloseTo(1 / rc, 6);
    // DeepPot-SE cosine: C¹; the second derivative jumps by π²/(2Δ² r_c).
    expect(Math.abs(j['deeppot-se']!.value)).toBeLessThan(1e-9);
    expect(Math.abs(j['deeppot-se']!.first)).toBeLessThan(1e-5);
    expect(j['deeppot-se']!.second).toBeCloseTo(Math.PI ** 2 / (2 * 0.7 ** 2 * rc), 2);
    // DPA-1/DPA-2 polynomial: C².
    expect(Math.abs(j.poly!.value)).toBeLessThan(1e-9);
    expect(Math.abs(j.poly!.first)).toBeLessThan(1e-5);
    expect(Math.abs(j.poly!.second)).toBeLessThan(1e-3);
    // DPA-3 double exponential: not exactly zero at r_c — a jump of exp(−exp(20 · 0.7/5.3)) ≈ 8 × 10⁻⁷.
    expect(j.dpa3!.value).toBeCloseTo(Math.exp(-Math.exp((20 * 0.7) / 5.3)), 9);
    expect(j.dpa3!.value).toBeGreaterThan(7e-7);
    expect(j.dpa3!.value).toBeLessThan(9e-7);
  });

  it('central differences agree with the analytic derivative of the polynomial switch', () => {
    const [, d1] = derivs((r) => wPoly(r, 3, 4), 3.5);
    // dw/dr at u = 1/2: 30 u²(−u² + 2u − 1)/Δ = −30/16.
    expect(d1).toBeCloseTo(-30 / 16, 6);
  });
});

describe('descriptor symmetries', () => {
  const o = { rc: 6, rcs: 5.3 };
  const base = env(hcp, 6.2);
  const R = randomRotation(mulberry32(3));
  const moved = shuffle(rotate(base, R), 5);
  it('all three descriptors are invariant under rotation and permutation', () => {
    expect(relDiff(seDescriptor(toy, base, o).D, seDescriptor(toy, moved, o).D)).toBeLessThan(1e-12);
    expect(relDiff(dpa1Descriptor(toy, base, o).D, dpa1Descriptor(toy, moved, o).D)).toBeLessThan(1e-12);
    expect(relDiff(dpa3Descriptor(toy, base, o).v, dpa3Descriptor(toy, moved, o).v)).toBeLessThan(1e-12);
  });

  it('the DeepPot-SE descriptor is exactly its radial part plus its dipole part (Gram identity)', () => {
    const r = seDescriptor(toy, env(fcc, 5), { rc: 5, rcs: 4.5 });
    r.D.forEach((d, k) => expect(d).toBeCloseTo(r.radial[k]! + r.dipole[k]!, 12));
  });
});

describe('what each descriptor can tell apart', () => {
  it('fcc and hcp at a first-shell cutoff: DeepPot-SE and DPA-1 see nothing, the angle embedding does', () => {
    const o = { rc: 3.2, rcs: 2.9 };
    const a = env(fcc, 3.3);
    const b = env(hcp, 3.3);
    expect(a.length).toBe(12);
    expect(b.length).toBe(12);
    // Every filtered dipole vanishes in both sites, so only the radial sums are left.
    expect(Math.max(...seDescriptor(toy, b, o).dipoleNorms)).toBeLessThan(1e-12);
    expect(relDiff(seDescriptor(toy, a, o).D, seDescriptor(toy, b, o).D)).toBeLessThan(1e-12);
    const d1a = dpa1Descriptor(toy, a, o);
    expect(Math.max(...d1a.updateNorm)).toBeLessThan(1e-12);
    expect(relDiff(d1a.D, dpa1Descriptor(toy, b, o).D)).toBeLessThan(1e-12);
    expect(relDiff(dpa3Descriptor(toy, a, o).v, dpa3Descriptor(toy, b, o).v)).toBeGreaterThan(1e-5);
  });

  it('fcc and hcp stay indistinguishable for DeepPot-SE and DPA-1 until hcp’s 1.633 d shell enters', () => {
    const d = 3.615 / Math.SQRT2;
    for (const rc of [3.8, 4.1]) {
      const o = { rc, rcs: rc - 0.5 };
      const a = env(fcc, rc + 0.1);
      const b = env(hcp, rc + 0.1);
      expect(relDiff(seDescriptor(toy, a, o).D, seDescriptor(toy, b, o).D)).toBeLessThan(1e-12);
      expect(relDiff(dpa1Descriptor(toy, a, o).D, dpa1Descriptor(toy, b, o).D)).toBeLessThan(1e-12);
    }
    const o = { rc: 4.3, rcs: 3.8 };
    expect(Math.sqrt(8 / 3) * d).toBeLessThan(4.3);
    expect(relDiff(seDescriptor(toy, env(fcc, 4.4), o).D, seDescriptor(toy, env(hcp, 4.4), o).D)).toBeGreaterThan(1e-6);
  });

  it('pyramid vs T: equal |Σ r̂|², so DeepPot-SE is blind; DPA-1’s gate separates them', () => {
    const o = { rc: 4, rcs: 3.5 };
    const a = pyramid(2.2);
    const b = tShape(2.2);
    expect(relDiff(seDescriptor(toy, a, o).D, seDescriptor(toy, b, o).D)).toBeLessThan(1e-12);
    expect(relDiff(dpa1Descriptor(toy, a, o).D, dpa1Descriptor(toy, b, o).D)).toBeGreaterThan(1e-3);
    // Without the gate, attention only sees distances and cannot help.
    expect(relDiff(dpa1Descriptor(toy, a, o, 2, false).D, dpa1Descriptor(toy, b, o, 2, false).D)).toBeLessThan(1e-12);
    expect(relDiff(dpa3Descriptor(toy, a, o).v, dpa3Descriptor(toy, b, o).v)).toBeGreaterThan(1e-3);
  });

  it('tetrahedron vs square: both dipoles vanish, so only the angle embedding separates them', () => {
    const o = { rc: 4, rcs: 3.5 };
    const a = tetrahedron(2.2);
    const b = square(2.2);
    expect(relDiff(seDescriptor(toy, a, o).D, seDescriptor(toy, b, o).D)).toBeLessThan(1e-12);
    expect(relDiff(dpa1Descriptor(toy, a, o).D, dpa1Descriptor(toy, b, o).D)).toBeLessThan(1e-12);
    expect(relDiff(dpa3Descriptor(toy, a, o).v, dpa3Descriptor(toy, b, o).v)).toBeGreaterThan(1e-3);
  });
});

describe('smooth softmax', () => {
  it('weights of neighbours with w → 0 vanish only for the switched forms', () => {
    const x = [0.3, -0.2, 0.9];
    const w = [1, 1, 1e-9];
    expect(attentionRow(x, w, 'plain')[2]).toBeGreaterThan(0.3);
    expect(attentionRow(x, w, 'switched')[2]).toBeLessThan(1e-8);
    expect(attentionRow(x, w, 'smooth')[2]).toBeLessThan(1e-8);
    const sum = attentionRow(x, [1, 1, 1], 'smooth').reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 12);
  });

  it('an atom crossing r_c makes plain and switched softmax jump; softmax* leaves ≈ e^−20', () => {
    const fixed: Vec3[] = [
      [2.4, 0.3, 0.1],
      [-0.5, 2.6, 0.4],
      [0.2, -0.6, 3.1],
      [-2.9, -1.2, 0.5],
    ];
    const dir: Vec3 = [0.6, 0.64, 0.48];
    const s = cutoffScan(toy, fixed, dir, [4, 5, 5.9], 4.5, 6);
    expect(Math.abs(s.jump.plain)).toBeGreaterThan(1e-3);
    expect(Math.abs(s.jump.switched)).toBeGreaterThan(1e-3);
    expect(Math.abs(s.jump.smooth)).toBeLessThan(1e-6);
    expect(Math.abs(s.jump.smooth)).toBeLessThan(1e-4 * Math.abs(s.jump.switched));
  });
});

describe('energy bias', () => {
  const water = FRAME_SPECS.filter((f) => f.counts[0] === 2 * f.counts[1]);
  it('water frames alone fix only 2 e(H) + e(O): rank 1, with a null direction ∝ (1, −2)', () => {
    const fit = fitBias(framesFor(BIAS_DATASETS[0]!, water));
    expect(fit.rank).toBe(1);
    const n = fit.nullDir!;
    expect(Math.abs(n[1]! / n[0]! + 2)).toBeLessThan(1e-9);
    // The fitted combination is right even though the parts are not.
    const [mh, mo] = BIAS_DATASETS[0]!.mu;
    const comb = 2 * fit.bias[0]! + fit.bias[1]!;
    expect(Math.abs(comb - (2 * mh + mo))).toBeLessThan(1.5);
  });

  it('adding H₂ and O₂ makes the bias identifiable, close to the setup’s reference energies', () => {
    for (const ds of BIAS_DATASETS) {
      const fit = fitBias(framesFor(ds, FRAME_SPECS));
      expect(fit.rank).toBe(2);
      expect(Math.abs(fit.bias[0]! - ds.mu[0])).toBeLessThan(1.5);
      expect(Math.abs(fit.bias[1]! - ds.mu[1])).toBeLessThan(1.5);
    }
  });

  it('one bias for both setups leaves errors of hundreds of eV per atom; one per setup does not', () => {
    const merged = fitBias([...framesFor(BIAS_DATASETS[0]!, FRAME_SPECS), ...framesFor(BIAS_DATASETS[1]!, FRAME_SPECS)]);
    expect(merged.rmsPerAtom).toBeGreaterThan(50);
    for (const ds of BIAS_DATASETS) expect(fitBias(framesFor(ds, FRAME_SPECS)).rmsPerAtom).toBeLessThan(1.5);
  });
});
