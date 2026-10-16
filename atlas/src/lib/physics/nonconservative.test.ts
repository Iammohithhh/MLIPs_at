import { describe, expect, it } from 'vitest';
import {
  RMS_CURL_UNIT,
  R_DISC,
  amplitudeFor,
  circulation,
  circulationExact,
  deltaF1Curl,
  deltaF1Gradient,
  deltaV1,
  drift,
  force,
  gradV,
  launch,
  potential,
  rmsGradientUnit,
  rmsOverDisc,
  run,
  wellMesh,
  type Vec2,
} from './nonconservative';

const central = (f: (p: Vec2) => number, p: Vec2, h = 1e-5): Vec2 => [
  (f([p[0] + h, p[1]]) - f([p[0] - h, p[1]])) / (2 * h),
  (f([p[0], p[1] + h]) - f([p[0], p[1] - h])) / (2 * h),
];
const pts: Vec2[] = [
  [0.3, -0.7],
  [1.1, 0.4],
  [-1.6, 1.2],
  [0, 0],
];

describe('fields', () => {
  it('gradV and the gradient error are the gradients of their potentials', () => {
    for (const p of pts) {
      const g = central(potential, p);
      expect(gradV(p)[0]).toBeCloseTo(g[0], 7);
      expect(gradV(p)[1]).toBeCloseTo(g[1], 7);
      const d = central(deltaV1, p);
      expect(deltaF1Gradient(p)[0]).toBeCloseTo(-d[0], 7);
      expect(deltaF1Gradient(p)[1]).toBeCloseTo(-d[1], 7);
    }
  });

  it('the rotational error has curl 2 and divergence 0', () => {
    const h = 1e-5;
    for (const p of pts) {
      const fx = (q: Vec2) => deltaF1Curl(q)[0];
      const fy = (q: Vec2) => deltaF1Curl(q)[1];
      const curl = central(fy, p, h)[0] - central(fx, p, h)[1];
      const div = central(fx, p, h)[0] + central(fy, p, h)[1];
      expect(curl).toBeCloseTo(2, 8);
      expect(div).toBeCloseTo(0, 8);
    }
  });

  it('both errors are scaled to the same RMS force error on the disc', () => {
    expect(rmsOverDisc(deltaF1Curl)).toBeCloseTo(RMS_CURL_UNIT, 3);
    expect(RMS_CURL_UNIT).toBeCloseTo(R_DISC / Math.SQRT2, 12);
    for (const sigma of [0.01, 0.04]) {
      const g = amplitudeFor('gradient', sigma);
      const c = amplitudeFor('direct', sigma);
      expect(rmsOverDisc((p) => [g * deltaF1Gradient(p)[0], g * deltaF1Gradient(p)[1]])).toBeCloseTo(sigma, 6);
      expect(rmsOverDisc((p) => [c * deltaF1Curl(p)[0], c * deltaF1Curl(p)[1]])).toBeCloseTo(sigma, 4);
    }
    expect(rmsGradientUnit()).toBeGreaterThan(0.5);
  });
});

describe('closed-loop work', () => {
  it('vanishes for gradient forces and equals amp·2·πR² for the curl error (Stokes)', () => {
    for (const R of [0.5, 1, 1.7]) {
      expect(Math.abs(circulation('exact', 0, R))).toBeLessThan(1e-12);
      expect(Math.abs(circulation('gradient', 0.3, R))).toBeLessThan(1e-12);
      const a = amplitudeFor('direct', 0.03);
      expect(circulation('direct', a, R)).toBeCloseTo(circulationExact(a, R), 10);
    }
  });
});

describe('velocity Verlet trajectories', () => {
  const base = { sigma: 0.03, tEnd: 120, ...launch(1) };

  it('the exact model conserves its energy to O(h²) with no drift', () => {
    const errs = [0.1, 0.05].map((dt) => drift(run({ ...base, model: 'exact', dt }).eModel).maxAbs);
    expect(errs[0]! / errs[1]!).toBeGreaterThan(3.5);
    expect(errs[0]! / errs[1]!).toBeLessThan(4.5);
    expect(errs[1]!).toBeLessThan(1e-3);
  });

  it('a gradient error conserves the model’s own energy; the reference energy only fluctuates', () => {
    const tr = run({ ...base, model: 'gradient', dt: 0.05 });
    expect(drift(tr.eModel).maxAbs).toBeLessThan(1e-3);
    // eRef = eModel − amp·δV₁ and δV₁ ranges over [−½, ½], so the excursion is at most amp.
    expect(drift(tr.eRef).maxAbs).toBeLessThan(amplitudeFor('gradient', base.sigma) + 1e-3);
  });

  it('a curl error pumps energy: dE/dt = amp·L and dL/dt = amp·r² (isotropic well)', () => {
    const dt = 0.01;
    const tr = run({ ...base, model: 'direct', dt });
    const a = amplitudeFor('direct', base.sigma);
    for (const k of [500, 4000, 9000]) {
      const dE = (tr.eRef[k + 1]! - tr.eRef[k - 1]!) / (2 * dt);
      const dL = (tr.L[k + 1]! - tr.L[k - 1]!) / (2 * dt);
      const r2 = tr.x[k]! ** 2 + tr.y[k]! ** 2;
      expect(dE).toBeCloseTo(a * tr.L[k]!, 3);
      expect(dL).toBeCloseTo(a * r2, 3);
    }
    expect(drift(tr.eRef).net).toBeGreaterThan(1);
  });

  it('the curl-driven drift is not an integrator error: it barely changes with the time step', () => {
    const e = [0.1, 0.025].map((dt) => drift(run({ ...base, model: 'direct', dt }).eRef).net);
    expect(Math.abs(e[0]! - e[1]!) / e[1]!).toBeLessThan(0.05);
  });

  it('launched clockwise, the energy first falls, then rises once L changes sign', () => {
    const tr = run({ ...base, ...launch(-1), model: 'direct', dt: 0.05, tEnd: 200 });
    let kMin = 0;
    tr.eRef.forEach((v, k) => {
      if (v < tr.eRef[kMin]!) kMin = k;
    });
    expect(tr.eRef[kMin]!).toBeLessThan(tr.eRef[0]! - 0.3);
    expect(tr.eRef[tr.eRef.length - 1]!).toBeGreaterThan(tr.eRef[kMin]! + 0.3);
    expect(Math.abs(tr.L[kMin]!)).toBeLessThan(0.15);
  });

  it('force() adds the scaled error to −∇V', () => {
    const p: Vec2 = [0.4, 0.9];
    const a = 0.2;
    const f = force('direct', p, a);
    expect(f[0]).toBeCloseTo(-gradV(p)[0] - a * p[1], 12);
    expect(f[1]).toBeCloseTo(-gradV(p)[1] + a * p[0], 12);
  });
});

describe('wellMesh', () => {
  it('samples z = V on a polar grid with a circular rim', () => {
    const m = wellMesh(3, 10, 24);
    expect(m.heights.length).toBe(1 + 10 * 24);
    expect(m.indices.length).toBe(3 * 24 + 6 * 24 * 9);
    expect(Math.max(...m.indices)).toBe(m.heights.length - 1);
    expect(m.heights[0]).toBe(0);
    const rim = m.heights[m.heights.length - 1]!;
    expect(rim).toBeCloseTo(potential([3, 0]), 5);
    for (let k = 0; k < m.heights.length; k++) {
      const p: Vec2 = [m.positions[3 * k]!, m.positions[3 * k + 1]!];
      expect(m.positions[3 * k + 2]).toBeCloseTo(potential(p), 4);
    }
  });
});
