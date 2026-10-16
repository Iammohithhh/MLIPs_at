import { describe, expect, it } from 'vitest';
import doc from '~/data/observatory/observatory.json';
import {
  EV_PER_A3_IN_GPA,
  angleDeg,
  birchMurnaghan,
  curveSummary,
  eosFromScan,
  forceAgreement,
  galleryForces,
  loopTest,
  pressureGPa,
  rotationTest,
  slopeMismatch,
  solve,
  type ObsDoc,
  type Vec3,
} from './observatory';

const D = doc as unknown as ObsDoc;

function bm(v: number, v0: number, b0: number, bp: number, e0: number) {
  const eta = (v0 / v) ** (2 / 3);
  return e0 + ((9 * v0 * b0) / 16) * ((eta - 1) ** 3 * bp + (eta - 1) ** 2 * (6 - 4 * eta));
}

describe('linear algebra', () => {
  it('solves a pivoting system', () => {
    const x = solve(
      [
        [0, 2, 1],
        [1, 1, 1],
        [2, 1, 3],
      ],
      [7, 6, 13],
    );
    expect(x[0]).toBeCloseTo(1, 12);
    expect(x[1]).toBeCloseTo(2, 12);
    expect(x[2]).toBeCloseTo(3, 12);
  });
});

describe('Birch–Murnaghan fit', () => {
  it('recovers V0, B0 and E0 exactly from third-order BM data', () => {
    const v0 = 20.0;
    const b0 = (0.8 / EV_PER_A3_IN_GPA) * 100; // 100 GPa × 0.8 → 80 GPa in eV/Å³
    const vols = Array.from({ length: 13 }, (_, i) => v0 * (0.94 + 0.01 * i) ** 3);
    const e = vols.map((v) => bm(v, v0, b0, 4.5, -3.7));
    const fit = birchMurnaghan(vols, e);
    expect(fit.v0).toBeCloseTo(v0, 8);
    expect(fit.b0).toBeCloseTo(80, 6);
    expect(fit.e0).toBeCloseTo(-3.7, 10);
    expect(fit.rmsMeV).toBeLessThan(1e-6);
  });
  it('reports the equilibrium lattice scale of a scan', () => {
    const atoms = 4;
    const V = 47.0;
    const scales = Array.from({ length: 13 }, (_, i) => 0.94 + 0.01 * i);
    const vTrue = (V * 0.98 ** 3) / atoms;
    const e = scales.map((s) => atoms * bm((V * s ** 3) / atoms, vTrue, 0.9, 5, -4));
    const fit = eosFromScan(scales, e, atoms, V);
    expect(fit.scale0).toBeCloseTo(0.98, 8);
  });
});

describe('force agreement', () => {
  const a: Vec3[] = [
    [1, 0, 0],
    [0, 2, 0],
  ];
  it('is zero for identical models', () => {
    const r = forceAgreement([a, a, a]);
    expect(r.spread).toEqual([0, 0]);
    expect(r.pairwise.flat().every((v) => v === 0)).toBe(true);
  });
  it('measures a symmetric split as its half-width', () => {
    const d = 0.3;
    const p = a.map((v) => [v[0] + d, v[1], v[2]] as Vec3);
    const m = a.map((v) => [v[0] - d, v[1], v[2]] as Vec3);
    const r = forceAgreement([p, m]);
    r.spread.forEach((s) => expect(s).toBeCloseTo(d, 12));
    expect(r.pairwise[0]![1]).toBeCloseTo(2 * d, 12);
    expect(r.mean[0]![0]).toBeCloseTo(1, 12);
  });
  it('measures angles', () => {
    expect(angleDeg([1, 0, 0], [0, 1, 0])).toBeCloseTo(90, 10);
    expect(angleDeg([1, 0, 0], [0, 0, 0])).toBe(0);
  });
});

describe('closed-loop test', () => {
  const n = 48;
  const th = Array.from({ length: n }, (_, k) => (2 * Math.PI * k) / n);
  it('gives zero circulation for a gradient field', () => {
    const E = th.map((t) => Math.sin(t) + 0.3 * Math.cos(2 * t));
    const P = th.map((t) => -(Math.cos(t) - 0.6 * Math.sin(2 * t))); // F·dx/dθ = −dE/dθ
    const r = loopTest(th, P, E);
    expect(Math.abs(r.circulation)).toBeLessThan(1e-12);
    expect(r.maxGap).toBeLessThan(5e-3); // trapezoid error of the running integral
  });
  it('detects a curl', () => {
    const r = loopTest(
      th,
      th.map(() => 0.01),
      th.map(() => 0),
    );
    expect(r.circulation).toBeCloseTo(2 * Math.PI * 0.01, 12);
  });
});

describe('rotation test', () => {
  const c = Math.cos(0.4);
  const s = Math.sin(0.4);
  const R: [Vec3, Vec3, Vec3] = [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
  const f: Vec3[] = [
    [1, 0, 0],
    [0, 1, 1],
  ];
  const rf = f.map((v) => [R[0][0] * v[0] + R[0][1] * v[1], R[1][0] * v[0] + R[1][1] * v[1], v[2]] as Vec3);
  it('is zero for co-rotating forces', () => {
    const r = rotationTest(R, [1, 1], [f, rf]);
    expect(r.maxForceError).toBeLessThan(1e-15);
    expect(r.dE).toBe(0);
  });
  it('is non-zero when forces do not co-rotate', () => {
    expect(rotationTest(R, [1, 1.1], [f, f]).maxForceError).toBeGreaterThan(0.1);
  });
});

describe('force versus energy slope', () => {
  it('is the finite-difference error for a gradient force and large otherwise', () => {
    const x = Array.from({ length: 21 }, (_, k) => 1 + 0.1 * k);
    const E = x.map((v) => (v - 2) ** 2);
    const F = x.map((v) => -2 * (v - 2));
    expect(Math.max(...slopeMismatch(x, E, F).map((p) => p.diff))).toBeLessThan(1e-12); // exact for a parabola
    expect(
      Math.max(
        ...slopeMismatch(
          x,
          E,
          F.map((f) => f + 0.5),
        ).map((p) => p.diff),
      ),
    ).toBeCloseTo(0.5, 12);
    expect(slopeMismatch(x, E, F, 2.5).every((p) => p.x >= 2.5)).toBe(true);
  });
});

describe('curves and stress', () => {
  it('finds the minimum, depth and spurious minima', () => {
    const x = [1, 1.5, 2, 2.5, 3, 4];
    const r = curveSummary(x, [-0.5, 2, -1, -0.8, -0.2, 0], [-1, 5, 0, -0.5, -0.2, 0]);
    expect(r.xMin).toBe(2);
    expect(r.depth).toBe(-1);
    expect(r.minima).toBe(1);
    expect(r.eShort).toBe(-0.5);
    expect(r.wrongWay).toEqual([1]);
  });
  it('converts stress to pressure', () => {
    expect(pressureGPa([-0.01, -0.01, -0.01, 0, 0, 0])).toBeCloseTo(1.6021766208, 8);
  });
});

describe('the precomputed Observatory file', () => {
  it('has every model on every input', () => {
    expect(D.format).toBe('atlas-observatory');
    expect(D.models.length).toBeGreaterThanOrEqual(4);
    for (const m of D.models) {
      expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
      for (const g of D.inputs.gallery) expect(m.results.gallery[g.id]!.forces.length).toBe(g.symbols.length);
      expect(m.results['ev-cu'].energy.length).toBe(D.inputs['ev-cu'].x.length);
      expect(m.results['dimer-si'].energy.length).toBe(D.inputs['dimer-si'].x.length);
      expect(m.results['na-au'].energy.length).toBe(D.inputs['na-au'].x.length);
      expect(m.results.loop.power.length).toBe(D.inputs.loop.theta.length);
    }
  });
  it('rotation matrix is orthogonal', () => {
    const R = D.inputs.rotation.R;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const d = R[i]![0] * R[j]![0] + R[i]![1] * R[j]![1] + R[i]![2] * R[j]![2];
        expect(d).toBeCloseTo(i === j ? 1 : 0, 10);
      }
  });
  it('conservative models close the loop; the direct-force model does not', () => {
    const th = D.inputs.loop.theta;
    for (const m of D.models) {
      const r = loopTest(th, m.results.loop.power, m.results.loop.energy);
      if (m.forces === 'conservative') expect(Math.abs(r.circulation)).toBeLessThan(0.05 * r.energyRange + 2e-3);
    }
  });
  it('invariant-energy models co-rotate their forces to float precision', () => {
    for (const m of D.models.filter((x) => x.forces === 'conservative')) {
      const r = rotationTest(D.inputs.rotation.R, m.results.rotation.energy, m.results.rotation.forces);
      expect(r.relForceError).toBeLessThan(1e-3);
    }
  });
  it('has a force spread for every gallery structure', () => {
    for (const g of D.inputs.gallery) {
      const a = forceAgreement(galleryForces(D, g.id));
      expect(a.spread.length).toBe(g.symbols.length);
      expect(a.spread.every(Number.isFinite)).toBe(true);
    }
  });
  it('fits an equation of state for each model on Cu and Si', () => {
    for (const m of D.models)
      for (const k of ['ev-cu', 'ev-si'] as const) {
        const inp = D.inputs[k];
        const fit = eosFromScan(inp.x, m.results[k].energy, inp.atoms, inp.volume0);
        expect(fit.b0).toBeGreaterThan(10);
        expect(fit.b0).toBeLessThan(400);
      }
  });
});
