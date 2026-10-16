import { describe, expect, it } from 'vitest';
import {
  angleScan,
  blindAngle,
  cosineCutoff,
  filterAt,
  forward,
  makeToySchNet,
  outerDistance,
  rbfCentres,
  rbfExpand,
  ssp,
  triatomic,
  type Vec3,
} from './schnet';

const R_SI = 2.3517; // diamond Si nearest-neighbour distance, a√3/4 with a = 5.431 Å

describe('SchNet building blocks', () => {
  it('shifted softplus: ssp(0) = 0, ln(0.5eˣ + 0.5) everywhere, linear for large x', () => {
    expect(ssp(0)).toBe(0);
    for (const x of [-3, -0.5, 0.7, 4]) expect(ssp(x)).toBeCloseTo(Math.log(0.5 * Math.exp(x) + 0.5), 12);
    expect(ssp(800)).toBeCloseTo(800 - Math.LN2, 9);
  });

  it('cosine cutoff is 1 at r = 0, ½ at r_c/2, and reaches 0 with zero slope at r_c', () => {
    expect(cosineCutoff(0, 5)).toBe(1);
    expect(cosineCutoff(2.5, 5)).toBeCloseTo(0.5, 12);
    expect(cosineCutoff(5, 5)).toBe(0);
    const h = 1e-5;
    expect((cosineCutoff(5, 5) - cosineCutoff(5 - h, 5)) / h).toBeLessThan(1e-4);
  });

  it('paper RBF grid: 0.1 Å spacing, 0–30 Å gives 301 Gaussians, peak at the nearest centre', () => {
    const spec = { spacing: 0.1, gamma: 10, max: 30 };
    expect(rbfCentres(spec)).toHaveLength(301);
    const e = rbfExpand(2.35, spec);
    const k = e.indexOf(Math.max(...e));
    expect([23, 24]).toContain(k);
    expect(e[k]!).toBeGreaterThan(0.97);
  });

  it('filters vanish beyond the cutoff', () => {
    const m = makeToySchNet({ rc: 4 });
    expect(Array.from(filterAt(m, 0, 4.2)).every((v) => v === 0)).toBe(true);
    expect(Array.from(filterAt(m, 0, 2)).some((v) => v !== 0)).toBe(true);
  });
});

describe('invariances of the toy SchNet', () => {
  const m = makeToySchNet({ rc: 5, blocks: 3 });
  const Z = [14, 8, 1, 6];
  const P: Vec3[] = [
    [0, 0, 0],
    [1.6, 0.2, -0.1],
    [-0.4, 1.3, 0.9],
    [0.3, -1.1, 1.7],
  ];

  it('rotation and translation leave every feature unchanged', () => {
    const c = Math.cos(0.7);
    const s = Math.sin(0.7);
    const Q: Vec3[] = P.map(([x, y, z]) => [c * x - s * z + 3, y - 2, s * x + c * z + 0.5]);
    const a = forward(m, Z, P);
    const b = forward(m, Z, Q);
    a.forEach((L, t) => L.forEach((x, i) => x.forEach((v, f) => expect(b[t]![i]![f]!).toBeCloseTo(v, 10))));
  });

  it('permuting atoms permutes features', () => {
    const perm = [2, 0, 3, 1];
    const a = forward(m, Z, P);
    const b = forward(m, perm.map((p) => Z[p]!), perm.map((p) => P[p]!));
    perm.forEach((p, i) => a[3]![p]!.forEach((v, f) => expect(b[3]![i]![f]!).toBeCloseTo(v, 10)));
  });
});

describe('what a distance-only message can see: the bond angle', () => {
  const thetas = [60, 80, 100, 109.47, 120, 150, 180];

  it('r_jk = 2r sin(θ/2): the tetrahedral angle in Si puts j and k 3.84 Å apart', () => {
    expect(outerDistance(R_SI, 109.4712)).toBeCloseTo(3.840, 3);
    expect(outerDistance(1, 180)).toBeCloseTo(2, 12);
  });

  it('blind angle θ* = 2 arcsin(r_c / 2r); 180° once r_c ≥ 2r', () => {
    expect(blindAngle(R_SI, 3.84)).toBeCloseTo(109.47, 1);
    expect(blindAngle(R_SI, 5)).toBe(180);
    expect(outerDistance(R_SI, blindAngle(R_SI, 3.5))).toBeCloseTo(3.5, 10);
  });

  it('after one block the centre atom cannot tell any two angles apart (exactly)', () => {
    const m = makeToySchNet({ rc: 5, blocks: 3 });
    const scan = angleScan(m, [14, 14, 14], R_SI, thetas);
    for (const d of scan.diff[0]!) expect(d).toBeLessThan(1e-12);
  });

  it('with r_jk < r_c the second block carries angle information', () => {
    const m = makeToySchNet({ rc: 5, blocks: 3 });
    const scan = angleScan(m, [14, 14, 14], R_SI, thetas);
    expect(Math.max(...scan.diff[1]!)).toBeGreaterThan(1e-5);
    expect(Math.max(...scan.diff[2]!)).toBeGreaterThan(1e-5);
  });

  it('when j and k are outside each other’s cutoff, no depth helps: all angles look alike', () => {
    const rc = 3.5;
    const m = makeToySchNet({ rc, blocks: 6 });
    const wide = [blindAngle(R_SI, rc) + 0.5, 120, 140, 160, 179];
    const scan = angleScan(m, [14, 14, 14], R_SI, wide);
    for (const row of scan.diff) for (const d of row) expect(d).toBeLessThan(1e-12);
  });

  it('the triatomic geometry has the requested bond lengths and angle', () => {
    const [i, j, k] = triatomic(2, 72);
    const d = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    expect(d(i!, j!)).toBeCloseTo(2, 12);
    expect(d(i!, k!)).toBeCloseTo(2, 12);
    expect(d(j!, k!)).toBeCloseTo(outerDistance(2, 72), 12);
  });
});
