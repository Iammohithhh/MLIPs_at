import { describe, expect, it } from 'vitest';
import raw from '~/data/models/mtp-si.json';
import { mulberry32 } from '../chem/rng';
import { barrier1d, boltzmannFit, coefficients, grade, grade1d, interpolate1d, invert, logAbsDet, maxvol, rms1d, type Matrix } from './active-learning';
import { fitSi, spearman, type SiData } from './mtp-fit';

const data = raw as unknown as SiData;

describe('the one-dimensional example of Podryabinkin & Shapeev (2017), Sec. 4.1', () => {
  it('reproduces the passive fit 0.93x² + 0.21x³ with RMS error 0.25 and its barrier near x = −3', () => {
    const t = boltzmannFit(1);
    expect(t[0]).toBeCloseTo(0.93, 2);
    expect(t[1]).toBeCloseTo(0.21, 2);
    expect(rms1d(t)).toBeCloseTo(0.25, 2);
    const b = barrier1d(t);
    expect(b.x).toBeGreaterThan(-3.1);
    expect(b.x).toBeLessThan(-2.9);
    expect(b.e).toBeGreaterThan(t[0]! * 16 - t[1]! * 64); // the model turns down again towards x = −4
  });

  it('the D-optimal pair is x = ±4, giving x² + 3 × 10⁻⁴ x³ with RMS error 0.46', () => {
    let best = { x1: 0, x2: 0, v: -Infinity };
    for (let i = 0; i <= 80; i++)
      for (let j = i + 1; j <= 80; j++) {
        const x1 = -4 + i / 10;
        const x2 = -4 + j / 10;
        const v = logAbsDet([
          [x1 * x1, x1 ** 3],
          [x2 * x2, x2 ** 3],
        ]);
        if (v > best.v) best = { x1, x2, v };
      }
    expect([best.x1, best.x2]).toEqual([-4, 4]);
    const t = interpolate1d(-4, 4)!;
    expect(t[0]).toBeCloseTo(1, 12);
    expect(t[1]).toBeCloseTo(Math.exp(-8), 12);
    expect(t[1]).toBeCloseTo(3e-4, 4);
    expect(rms1d(t)).toBeCloseTo(0.46, 2);
  });

  it('γ ≤ 1 everywhere between the active points and > 1 outside them', () => {
    for (let x = -4; x <= 4; x += 0.05) expect(grade1d(x, -4, 4)).toBeLessThanOrEqual(1 + 1e-12);
    expect(grade1d(4.5, -4, 4)).toBeGreaterThan(1);
    expect(grade1d(1, -1, 2)).toBeLessThanOrEqual(1);
    expect(grade1d(-3, -1, 2)).toBeGreaterThan(1);
  });
});

describe('maxvol and the extrapolation grade', () => {
  const rand = mulberry32(11);
  const B: Matrix = Array.from({ length: 60 }, () => Array.from({ length: 6 }, () => 2 * rand() - 1));

  it('every candidate is an interpolation of the active set (|c_j| ≤ 1) and the active rows have γ = 1', () => {
    const a = maxvol(B)!;
    expect(a.rows.length).toBe(6);
    for (const b of B) expect(grade(b, a.Ainv)).toBeLessThanOrEqual(1 + 1e-9);
    for (const i of a.rows) expect(grade(B[i]!, a.Ainv)).toBeCloseTo(1, 12);
  });

  it('replacing row k by b multiplies |det A| by |c_k| (Cramer’s rule)', () => {
    const a = maxvol(B)!;
    const b = B[0]!.map((x) => 3 * x);
    const c = coefficients(b, a.Ainv);
    const A = a.rows.map((i) => B[i]!);
    for (let k = 0; k < 6; k++) {
      const A2 = A.map((r, i) => (i === k ? b : r));
      expect(Math.exp(logAbsDet(A2) - logAbsDet(A))).toBeCloseTo(Math.abs(c[k]!), 9);
    }
  });

  it('γ does not depend on how the basis functions are scaled', () => {
    const a = maxvol(B)!;
    const D = [1, 1e3, 1e-2, 7, 0.5, 40];
    const Bs = B.map((r) => r.map((x, j) => x * D[j]!));
    const As = invert(a.rows.map((i) => Bs[i]!))!;
    const x = [0.3, 2, -1.5, 0.9, 1.1, -0.4];
    expect(grade(x.map((v, j) => v * D[j]!), As)).toBeCloseTo(grade(x, a.Ainv), 9);
  });

  it('returns null when the candidates do not span every basis function', () => {
    const flat = B.map((r) => [...r.slice(0, 5), r[0]! + r[1]!]);
    expect(maxvol(flat)).toBeNull();
  });
});

describe('linear MTP on the Zuo et al. Si cells (precomputed basis values)', () => {
  it('has the 214 training and 25 test cells of the mlearn Si set', () => {
    expect(data.configs.filter((c) => c.split === 'train').length).toBe(214);
    expect(data.configs.filter((c) => c.split === 'test').length).toBe(25);
    expect(new Set(data.configs.map((c) => c.group))).toEqual(new Set(['AIMD-NVT', 'Vacancy', 'Elastic', 'Surface']));
    expect(data.configs.every((c) => c.b.length === data.basis.length)).toBe(true);
  });

  it('more basis functions fit the training cells better', () => {
    const e = [8, 10, 12, 14].map((L) => fitSi(data, { levelMax: L, leaveOut: null, fitOn: 'all', lambda: 1e-10 }).rmse.train);
    for (let k = 1; k < e.length; k++) expect(e[k]!).toBeLessThan(e[k - 1]!);
  });

  it('test cells from the same groups interpolate (γ near or below 1); left-out surfaces extrapolate with large errors', () => {
    const all = fitSi(data, { levelMax: 12, leaveOut: null, fitOn: 'all', lambda: 1e-10 });
    expect(all.active?.rows.length).toBe(all.m);
    for (const c of all.cells.filter((x) => x.role === 'train')) expect(c.gamma).toBeLessThanOrEqual(1 + 1e-6);
    const noSurf = fitSi(data, { levelMax: 12, leaveOut: 'Surface', fitOn: 'all', lambda: 1e-10 });
    const surf = noSurf.cells.filter((c) => c.role === 'left-out');
    const med = (v: number[]) => [...v].sort((p, q) => p - q)[Math.floor(v.length / 2)]!;
    expect(med(surf.map((c) => c.gamma))).toBeGreaterThan(2);
    expect(noSurf.rmse.leftOut!).toBeGreaterThan(3 * noSurf.rmse.test);
  });

  it('error and γ are rank-correlated over all cells when surfaces are left out', () => {
    const r = fitSi(data, { levelMax: 12, leaveOut: 'Surface', fitOn: 'all', lambda: 1e-10 });
    const ok = r.cells.filter((c) => c.role !== 'train');
    expect(spearman(ok.map((c) => c.gamma), ok.map((c) => c.error))).toBeGreaterThan(0.3);
  });
});
