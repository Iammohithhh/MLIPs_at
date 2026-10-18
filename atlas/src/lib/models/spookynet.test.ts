import { describe, expect, it } from 'vitest';
import { mulberry32, randomRotation } from '../chem/rng';
import {
  MAGIC_THETA,
  PAIRS,
  RC_PAPER,
  attentionExact,
  attentionFavor,
  bernstein,
  chargeCorrection,
  drawOmega,
  electrostatics,
  elementDescriptor,
  fcut,
  featureMap,
  fragmentPair,
  fswitch,
  legendre,
  makeToySpooky,
  maxDiff,
  powerSpectrum,
  radialBasis,
  realY,
  runToy,
  shBlock,
  subset,
  trigonalTwist,
  type Molecule,
  type PairKind,
  type Vec3,
} from './spookynet';

const rot = (m: number[], v: Vec3): Vec3 => [m[0]! * v[0] + m[1]! * v[1] + m[2]! * v[2], m[3]! * v[0] + m[4]! * v[1] + m[5]! * v[2], m[6]! * v[0] + m[7]! * v[1] + m[8]! * v[2]];
const rotate = (mol: Molecule, m: number[]): Molecule => ({ numbers: mol.numbers, positions: mol.positions.map((p) => rot(m, p)) });
const split = (mol: Molecule, nA: number) => [subset(mol, [...Array(nA).keys()]), subset(mol, Array.from({ length: mol.numbers.length - nA }, (_, i) => i + nA))] as const;
const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);

describe('basis functions (Eqs. 14–17)', () => {
  it('f_cut is 1 at r = 0, 0 at and beyond r_cut, and flat at both ends', () => {
    expect(fcut(0)).toBe(1);
    expect(fcut(RC_PAPER)).toBe(0);
    expect(fcut(RC_PAPER + 1)).toBe(0);
    expect(fcut(RC_PAPER - 1e-3)).toBeLessThan(1e-300 + 1e-12);
    const h = 1e-4;
    expect(Math.abs((fcut(h) - fcut(0)) / h)).toBeLessThan(1e-3);
  });

  it('Bernstein polynomials are a partition of unity, so Σ_k ρ_k(r) = f_cut(r)', () => {
    for (const K of [2, 4, 8, 16]) {
      for (const x of [0, 0.13, 0.5, 0.97, 1]) expect(sum(Array.from({ length: K }, (_, k) => bernstein(k, K - 1, x)))).toBeCloseTo(1, 12);
      for (const r of [0.1, 1, 2.5, 4, 5.2]) expect(sum(radialBasis(r, K))).toBeCloseTo(fcut(r), 12);
    }
  });

  it('Eq. (17) gives the real harmonics without √(4π/(2l+1)): Σ_m Y(u)Y(v) = P_l(u·v)', () => {
    const rand = mulberry32(3);
    for (let n = 0; n < 20; n++) {
      const u: Vec3 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const v: Vec3 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v));
      for (let l = 0; l <= 4; l++) expect(sum(shBlock(l, u).map((y, i) => y * shBlock(l, v)[i]!))).toBeCloseTo(legendre(l, c), 12);
    }
  });

  it('l = 1 is (y, z, x) and l = 2 matches the reference code’s irreps', () => {
    const v: Vec3 = [0.3, -0.5, 0.81];
    const [x, y, z] = v.map((c) => c / Math.hypot(...v)) as Vec3;
    expect(shBlock(1, v).map((c) => +c.toFixed(12))).toEqual([y, z, x].map((c) => +c.toFixed(12)));
    const s3 = Math.sqrt(3);
    const d = shBlock(2, v);
    const code = [s3 * x * y, s3 * y * z, 0.5 * (3 * z * z - 1), s3 * x * z, (s3 / 2) * (x * x - y * y)];
    d.forEach((c, i) => expect(c).toBeCloseTo(code[i]!, 12));
    expect(realY(2, 0, [0, 0, 2])).toBeCloseTo(1, 12);
  });

  it('the power spectrum equals the double sum of Legendre polynomials', () => {
    const vs = trigonalTwist(23, 61).positions.slice(1);
    for (let l = 0; l <= 4; l++) {
      let ref = 0;
      for (const a of vs) for (const b of vs) ref += legendre(l, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b)));
      expect(powerSpectrum(vs, l)).toBeCloseTo(ref, 10);
    }
  });
});

describe('the trigonal twist', () => {
  it('keeps the s, p and d invariants of the centre fixed; f changes as (1 + cos 3τ)', () => {
    const at = (tau: number, theta = MAGIC_THETA) => trigonalTwist(tau, theta).positions.slice(1);
    for (const tau of [0, 10, 25, 45, 60]) {
      expect(powerSpectrum(at(tau), 0)).toBeCloseTo(36, 10);
      expect(powerSpectrum(at(tau), 1)).toBeLessThan(1e-20);
      expect(powerSpectrum(at(tau), 2)).toBeLessThan(1e-20);
    }
    // a_3 ∝ |1 + e^{3iτ}|² = 2(1 + cos 3τ): zero for the octahedron, largest for the prism
    expect(powerSpectrum(at(60), 3)).toBeLessThan(1e-20);
    expect(powerSpectrum(at(0), 3)).toBeGreaterThan(1);
    const ratio = powerSpectrum(at(20), 3) / powerSpectrum(at(0), 3);
    expect(ratio).toBeCloseTo((1 + Math.cos(Math.PI / 3)) / 2, 10);
    // away from the magic angle the d invariant is no longer zero, but still independent of τ
    expect(powerSpectrum(at(30, 50), 2)).toBeGreaterThan(1e-3);
    expect(powerSpectrum(at(0, 50), 2)).toBeCloseTo(powerSpectrum(at(60, 50), 2), 12);
    expect(powerSpectrum(at(0, 50), 1)).toBeCloseTo(powerSpectrum(at(60, 50), 1), 12);
  });

  it('the antiprism at the magic angle is a regular octahedron', () => {
    const p = trigonalTwist(60).positions.slice(1);
    const ds = p.flatMap((a, i) => p.slice(i + 1).map((b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])));
    const short = ds.filter((d) => d < 2.9);
    expect(short.length).toBe(12);
    short.forEach((d) => expect(d).toBeCloseTo(2 * Math.SQRT2, 10));
  });

  it('one toy update leaves the metal’s features unchanged along the twist, two do not', () => {
    const model = makeToySpooky();
    const ref1 = runToy(model, trigonalTwist(60), { modules: 1 });
    const ref2 = runToy(model, trigonalTwist(60), { modules: 2 });
    let moved2 = 0;
    for (const tau of [0, 15, 30, 45]) {
      const r1 = runToy(model, trigonalTwist(tau), { modules: 1 });
      expect(maxDiff([r1.xs[0]![0]!], [ref1.xs[0]![0]!])).toBeLessThan(1e-12);
      // the ligands see the twist in the first update already
      expect(maxDiff([r1.xs[0]![1]!], [ref1.xs[0]![1]!])).toBeGreaterThan(1e-6);
      const r2 = runToy(model, trigonalTwist(tau), { modules: 2 });
      moved2 = Math.max(moved2, maxDiff([r2.xs[1]![0]!], [ref2.xs[1]![0]!]));
    }
    expect(moved2).toBeGreaterThan(1e-4);
    // at any polar angle the twist stays invisible to one update; the polar angle itself is not
    const a = runToy(model, trigonalTwist(0, 50), { modules: 1 });
    const b = runToy(model, trigonalTwist(60, 50), { modules: 1 });
    expect(maxDiff([a.xs[0]![0]!], [b.xs[0]![0]!])).toBeLessThan(1e-12);
    const c = runToy(model, trigonalTwist(0, MAGIC_THETA), { modules: 1 });
    expect(maxDiff([a.xs[0]![0]!], [c.xs[0]![0]!])).toBeGreaterThan(1e-6);
  });
});

describe('element descriptors (Supplementary Information, element descriptor table)', () => {
  it('match the published unscaled rows divided by the Rn row', () => {
    const rn = [86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14];
    const table: Record<number, number[]> = {
      1: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      6: [6, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
      8: [8, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],
      16: [16, 2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],
      26: [26, 2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 6, 0],
    };
    for (const [Z, row] of Object.entries(table)) elementDescriptor(Number(Z)).forEach((v, i) => expect(v).toBeCloseTo(row[i]! / rn[i]!, 14));
  });
});

describe('toy SpookyNet', () => {
  const model = makeToySpooky();

  it('is invariant to rotations and permutations', () => {
    const mol = fragmentPair('benzene-water', 3.4);
    const ref = runToy(model, mol, { Q: 1, S: 1 });
    const r = runToy(model, rotate(mol, randomRotation(mulberry32(5))), { Q: 1, S: 1 });
    expect(r.energy).toBeCloseTo(ref.energy, 10);
    const perm = [...mol.numbers.keys()].reverse();
    const p = runToy(model, subset(mol, perm), { Q: 1, S: 1 });
    expect(p.energy).toBeCloseTo(ref.energy, 10);
    p.q.forEach((q, i) => expect(q).toBeCloseTo(ref.q[perm[i]!]!, 10));
  });

  it('electronic embedding: Σ a_i = Ψ, set by the elements alone, zero for Ψ = 0', () => {
    for (const Q of [-2, -1, 1, 3]) {
      const a = runToy(model, fragmentPair('benzene-water', 3.4), { Q }).aQ;
      expect(sum(a)).toBeCloseTo(Q, 12);
      // geometry plays no role: same weights for a different arrangement of the same atoms
      const b = runToy(model, fragmentPair('benzene-water', 9), { Q }).aQ;
      a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, 14));
      // atoms of one element share one weight
      expect(a[0]).toBeCloseTo(a[5]!, 14);
    }
    expect(runToy(model, fragmentPair('LiF', 2), { Q: 0 }).aQ).toEqual([0, 0]);
    expect(sum(runToy(model, fragmentPair('LiF', 2), { S: 2 }).aS)).toBeCloseTo(2, 12);
  });

  it('charges sum to Q (Eq. 24) and the correction is shared per atom', () => {
    const qt = [0.3, -0.1, 0.05, 0.2];
    const q = chargeCorrection(qt, 1);
    expect(sum(q)).toBeCloseTo(1, 14);
    q.forEach((v, i) => expect(v - qt[i]!).toBeCloseTo((1 - 0.45) / 4, 14));
  });

  it('beyond the cutoff nothing depends on the separation', () => {
    for (const kind of Object.keys(PAIRS) as PairKind[]) {
      for (const nonlocal of ['off', 'exact'] as const) {
        const a = runToy(model, fragmentPair(kind, 7), { Q: 1, nonlocal });
        const b = runToy(model, fragmentPair(kind, 15), { Q: 1, nonlocal });
        expect(Math.abs(a.energy - b.energy)).toBeLessThan(1e-12);
        expect(maxDiff([a.q], [b.q])).toBeLessThan(1e-12);
      }
    }
  });

  it('local-only and neutral: separated fragments add up exactly', () => {
    for (const kind of ['LiF', 'NaCl', 'benzene-water'] as PairKind[]) {
      const mol = fragmentPair(kind, 9);
      const [A, B] = split(mol, PAIRS[kind].nA);
      const e = runToy(model, mol, { nonlocal: 'off' }).energy;
      expect(e).toBeCloseTo(runToy(model, A, { nonlocal: 'off' }).energy + runToy(model, B, { nonlocal: 'off' }).energy, 10);
    }
  });

  it('with nonlocal attention unlike fragments do not add up, identical ones do', () => {
    for (const kind of ['LiF', 'benzene-water'] as PairKind[]) {
      const mol = fragmentPair(kind, 9);
      const [A, B] = split(mol, PAIRS[kind].nA);
      const gap = runToy(model, mol).energy - runToy(model, A).energy - runToy(model, B).energy;
      expect(Math.abs(gap)).toBeGreaterThan(1e-3);
    }
    const mol = fragmentPair('benzene-benzene', 9);
    const [A] = split(mol, 12);
    expect(runToy(model, mol).energy).toBeCloseTo(2 * runToy(model, A).energy, 10);
  });

  it('a charge on identical separated fragments splits in half, as if each were given Q/2', () => {
    const mol = fragmentPair('benzene-benzene', 9);
    const [A] = split(mol, 12);
    for (const nonlocal of ['off', 'exact'] as const) {
      const r = runToy(model, mol, { Q: 1, nonlocal });
      expect(sum(r.q.slice(0, 12))).toBeCloseTo(0.5, 12);
      expect(sum(r.aQ.slice(0, 12))).toBeCloseTo(0.5, 12);
      expect(r.energy).toBeCloseTo(2 * runToy(model, A, { Q: 0.5, nonlocal }).energy, 10);
    }
  });

  it('electrostatics: switched kernel is Coulomb beyond 3r_c/4 and finite at contact', () => {
    expect(fswitch(0.1, 1, 3)).toBe(1);
    expect(fswitch(3.1, 1, 3)).toBe(0);
    expect(fswitch(2, 1, 3)).toBeCloseTo(0.5, 12);
    const far = electrostatics([1, -1], [[0, 0, 0], [5, 0, 0]]);
    expect(far).toBeCloseTo(-14.399645 / 5, 10);
    expect(Number.isFinite(electrostatics([1, 1], [[0, 0, 0], [0, 0, 0]]))).toBe(true);
  });
});

describe('attention (Eqs. 19–20)', () => {
  const model = makeToySpooky();
  const { Q, K, V } = runToy(model, fragmentPair('benzene-water', 3.4)).attention[1]!;

  it('exact attention rows sum to one', () => {
    attentionExact(Q, K, V).weights.forEach((w) => expect(sum(w)).toBeCloseTo(1, 12));
  });

  it('positive random features estimate exp(q·k/√d) without bias', () => {
    const d = 12;
    const q = Array.from({ length: d }, (_, i) => 0.8 * Math.sin(i + 1));
    const k = Array.from({ length: d }, (_, i) => 0.8 * Math.cos(2 * i + 1));
    const exact = Math.exp(sum(q.map((v, i) => v * k[i]!)) / Math.sqrt(d));
    let mean = 0;
    const n = 400;
    for (let s = 0; s < n; s++) {
      const om = drawOmega(1000 + s, 96, d);
      const a = featureMap(q, om, 'positive');
      const b = featureMap(k, om, 'positive');
      mean += sum(a.map((x, i) => x * b[i]!)) / n;
    }
    expect(Math.abs(mean - exact) / exact).toBeLessThan(0.02);
  });

  it('FAVOR+ rows sum to one, its error falls with m, and its normaliser stays positive', () => {
    const ex = attentionExact(Q, K, V);
    const err = (m: number) => {
      let e = 0;
      for (let s = 0; s < 16; s++) e += maxDiff(attentionFavor(Q, K, V, drawOmega(200 + s, m, 12)).weights, ex.weights) / 16;
      return e;
    };
    attentionFavor(Q, K, V, drawOmega(1, 12, 12)).weights.forEach((w) => expect(sum(w)).toBeCloseTo(1, 12));
    expect(err(768)).toBeLessThan(err(12) / 2);
    // sharpen the attention: trigonometric features produce non-positive normalisers, positive ones never do
    const Qs = Q.map((q) => q.map((x) => 3 * x));
    const Ks = K.map((k) => k.map((x) => 3 * x));
    let negTrig = 0;
    for (let s = 0; s < 16; s++) {
      const om = drawOmega(300 + s, 48, 12);
      expect(attentionFavor(Qs, Ks, V, om, 'positive').minDenominator).toBeGreaterThan(0);
      if (attentionFavor(Qs, Ks, V, om, 'trig').minDenominator <= 0) negTrig++;
    }
    expect(negTrig).toBeGreaterThan(4);
  });

  it('orthogonal blocks are orthogonal before their rows are rescaled', () => {
    const om = drawOmega(9, 12, 12);
    const unit = om.map((r) => r.map((v) => v / Math.hypot(...r)));
    for (let i = 0; i < 12; i++) for (let j = 0; j < i; j++) expect(Math.abs(sum(unit[i]!.map((v, c) => v * unit[j]![c]!)))).toBeLessThan(1e-12);
  });
});
