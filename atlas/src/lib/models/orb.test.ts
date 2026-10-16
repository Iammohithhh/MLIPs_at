import { describe, expect, it } from 'vitest';
import cu from '~/data/structures/cu-fcc.json';
import si from '~/data/structures/si-diamond.json';
import nacl from '~/data/structures/nacl-rocksalt.json';
import zno from '~/data/structures/zno-wurtzite.json';
import type { Structure } from '~/lib/structure/types';
import { shBlock } from '../physics/spherical-harmonics';
import { mulberry32 } from '../chem/rng';
import { shellAround } from './three-body';
import {
  ORB_K20,
  ORB_KINF,
  ORB_RMAX,
  axisRot,
  besselBasis,
  deltaRot,
  droppedTiedNeighbour,
  equigradExperiment,
  expAntisym,
  features,
  fitReadout,
  fittedDeltaRot,
  frob,
  dot,
  limitCut,
  makeSample,
  makeToyOrb,
  netTorque,
  orbEnvelope,
  orbForces,
  orbForward,
  randomRowRotation,
  rotateAll,
  rowTimes,
  scoreFit,
  selectNeighbours,
  shAll,
  shells,
  swSite,
  withRadial,
  type Vec3,
} from './orb';

const env = (s: unknown, R = ORB_RMAX) => shellAround(s as Structure, R).map((n) => n.p as Vec3);
const cuEnv = env(cu);
const si5 = env(si, 2.5);
const norm = (v: Vec3) => Math.hypot(...v);

describe('Orb-v3 edge basis', () => {
  it('the envelope is 1 at 0, 0 at r_max, with zero slope and curvature there', () => {
    expect(orbEnvelope(0)).toBe(1);
    expect(orbEnvelope(6)).toBe(0);
    expect(orbEnvelope(7)).toBe(0);
    const h = 1e-4;
    expect(Math.abs(orbEnvelope(6 - h)) / h).toBeLessThan(1e-6);
    const curv = (orbEnvelope(6 - 2 * h) - 2 * orbEnvelope(6 - h) + orbEnvelope(6)) / (h * h);
    expect(Math.abs(curv)).toBeLessThan(1e-2);
    let prev = 1;
    for (let r = 0; r <= 6; r += 0.01) {
      expect(orbEnvelope(r)).toBeLessThanOrEqual(prev + 1e-12);
      prev = orbEnvelope(r);
    }
  });

  it('uses 8 Bessel functions √(2/r_max) sin(nπr/r_max)/r', () => {
    const b = besselBasis(2.0);
    expect(b).toHaveLength(8);
    expect(b[2]).toBeCloseTo((Math.sqrt(2 / 6) * Math.sin((3 * Math.PI * 2) / 6)) / 2, 14);
  });

  it('the explicit l ≤ 3 harmonics equal the Physics Lab convention', () => {
    const rand = mulberry32(2);
    for (let t = 0; t < 30; t++) {
      const v: Vec3 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const ref = [0, 1, 2, 3].flatMap((l) => shBlock(l, v));
      shAll(v).forEach((y, i) => expect(y).toBeCloseTo(ref[i]!, 12));
    }
  });
});

describe('the neighbour limit in real crystals (6 Å, k = 20)', () => {
  const cuts = Object.fromEntries(
    ([['Cu', cu], ['Si', si], ['NaCl', nacl], ['ZnO', zno]] as const).map(([n, s]) => [n, limitCut(env(s).map(norm), ORB_K20)]),
  );

  it('fcc Cu: 78 neighbours inside 6 Å; the 20th place splits the 24-atom third shell', () => {
    expect(shells(cuEnv.map(norm)).map(([, m]) => m)).toEqual([12, 6, 24, 12, 24]);
    expect(cuts.Cu).toMatchObject({ inside: 78, tied: 24, tiedKept: 2, dropped: 58 });
    expect(cuts.Cu!.rk).toBeCloseTo(3.615 * Math.sqrt(1.5), 9);
  });

  it('diamond Si and rock-salt NaCl: the cut also falls inside a shell', () => {
    expect(cuts.Si).toMatchObject({ inside: 46, tied: 12, tiedKept: 4 });
    expect(cuts.NaCl).toMatchObject({ inside: 32, tied: 8, tiedKept: 2 });
  });

  it('the "-inf" checkpoints keep up to 120, more than any of these crystals has inside 6 Å', () => {
    for (const c of Object.values(cuts)) expect(c.inside).toBeLessThanOrEqual(ORB_KINF);
    expect(selectNeighbours(cuEnv.map(norm), ORB_KINF)).toHaveLength(78);
    expect(selectNeighbours(cuEnv.map(norm), null)).toHaveLength(78);
  });
});

describe('the toy GNS', () => {
  const m = makeToyOrb();
  const md = makeToyOrb({ mode: 'distance' });

  it('is invariant to the order of neighbours when nothing is truncated', () => {
    const rev = [...cuEnv].reverse();
    expect(orbForward(m, rev, null).energy).toBeCloseTo(orbForward(m, cuEnv, null).energy, 12);
  });

  it('is not rotationally invariant with Orb-v3 edge features, and is with distances only', () => {
    const R = randomRowRotation(mulberry32(4));
    const rot = rotateAll(cuEnv, R);
    expect(Math.abs(orbForward(m, rot, null).energy - orbForward(m, cuEnv, null).energy)).toBeGreaterThan(1e-5);
    expect(orbForward(md, rot, null).energy).toBeCloseTo(orbForward(md, cuEnv, null).energy, 12);
  });

  it('is smooth as an atom crosses r_max when every neighbour is kept', () => {
    const j = cuEnv.length - 1;
    const E = (r: number) => orbForward(m, withRadial(cuEnv, j, r), null).energy;
    expect(Math.abs(E(6 - 1e-7) - E(6 + 1e-7))).toBeLessThan(1e-10);
    const F = (r: number) => -(E(r + 1e-5) - E(r - 1e-5)) / 2e-5;
    expect(Math.abs(F(6 - 1e-3))).toBeLessThan(1e-6);
  });

  describe('crossing the 20th-neighbour distance in Cu', () => {
    const j = droppedTiedNeighbour(cuEnv, ORB_K20);
    const r0 = norm(cuEnv[j]!);
    const step = (mm: typeof m, k: number | null) => {
      const E = (r: number) => orbForward(mm, withRadial(cuEnv, j, r), k).energy;
      const F = (r: number) => -(E(r + 1e-6) - E(r - 1e-6)) / 2e-6;
      return { dE: E(r0 + 1e-7) - E(r0 - 1e-7), Fin: F(r0 - 1e-4), Fout: F(r0 + 1e-4) };
    };

    it('the moved atom is a third-shell atom the limit drops', () => {
      expect(r0).toBeCloseTo(4.4275, 3);
      expect(selectNeighbours(cuEnv.map(norm), ORB_K20)).not.toContain(j);
      expect(selectNeighbours(withRadial(cuEnv, j, r0 - 0.01).map(norm), ORB_K20)).toContain(j);
    });

    it('with direction-dependent features the energy steps; without the limit it does not', () => {
      expect(Math.abs(step(m, ORB_K20).dE)).toBeGreaterThan(1e-6);
      expect(Math.abs(step(m, null).dE)).toBeLessThan(1e-9);
    });

    it('with distance-only features the energy is continuous but the force on the atom drops to zero', () => {
      const s = step(md, ORB_K20);
      expect(Math.abs(s.dE)).toBeLessThan(1e-9);
      expect(Math.abs(s.Fin)).toBeGreaterThan(1e-4);
      expect(Math.abs(s.Fout)).toBe(0);
      const t = step(md, null);
      expect(t.Fout).toBeCloseTo(t.Fin, 4);
    });

    it('the path stays clear of the other atoms', () => {
      for (let r = 4.15; r <= 4.7; r += 0.05) {
        const p = withRadial(cuEnv, j, r)[j]!;
        for (const [i, q] of cuEnv.entries()) if (i !== j) expect(norm([p[0] - q[0], p[1] - q[1], p[2] - q[2]])).toBeGreaterThan(2.0);
      }
    });
  });
});

describe('Equigrad', () => {
  const m = makeToyOrb({ d: 64 });
  const E = (v: Vec3[]) => orbForward(m, v, null).energy;

  it('exp of an antisymmetric matrix is a proper rotation; axisRot turns by +θ', () => {
    const A = [0, -0.3, 0.2, 0.3, 0, -0.5, -0.2, 0.5, 0];
    const R = expAntisym(A);
    const RtR = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => R[i]! * R[j]! + R[3 + i]! * R[3 + j]! + R[6 + i]! * R[6 + j]!));
    RtR.forEach((x, k) => expect(x).toBeCloseTo(k % 4 === 0 ? 1 : 0, 12));
    const v = rowTimes([1, 0, 0], axisRot([0, 0, 1], Math.PI / 2));
    expect(v[1]).toBeCloseTo(1, 12);
  });

  it('Δ_rot vanishes for an invariant energy and not for the Orb-like toy', () => {
    expect(frob(deltaRot(swSite, si5))).toBeLessThan(1e-8);
    expect(frob(deltaRot((v) => orbForward(makeToyOrb({ mode: 'distance' }), v, null).energy, si5))).toBeLessThan(1e-8);
    expect(frob(deltaRot(E, si5))).toBeGreaterThan(1e-3);
  });

  it('is antisymmetric and equals minus the net torque of the conservative forces', () => {
    const D = deltaRot(E, si5);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) expect(D[3 * r + c]!).toBeCloseTo(-D[3 * c + r]!, 8);
    const T = netTorque(si5, orbForces(E, si5));
    expect(T[0]).toBeCloseTo(-D[5]!, 6);
    expect(T[1]).toBeCloseTo(D[2]!, 6);
    expect(T[2]).toBeCloseTo(-D[1]!, 6);
  });

  it('for a linear readout, ‖Δ_rot‖ is the norm of w against the feature Jacobian', () => {
    const s = makeSample(m, si5);
    const w = new Float64Array(s.phi.length).map((_, i) => Math.sin(i + 1));
    const D = deltaRot((v) => dot(w, features(m, v)), si5);
    expect(fittedDeltaRot(w, s)).toBeCloseTo(frob(D), 6);
  });

  describe('fitting the readout on Si₅', () => {
    const ex = equigradExperiment(m, si5);
    const score = (set: 'fixed' | 'rotated', lam: number) => scoreFit(fitReadout(ex.train[set], lam), ex.train[set], ex.test);

    it('without penalty or augmentation the fit is good in one orientation and poor in others', () => {
      const s = score('fixed', 0);
      expect(s.trainRmse).toBeLessThan(0.02);
      expect(s.rmse).toBeGreaterThan(10 * s.trainRmse);
    });

    it('the Equigrad penalty lowers ‖Δ_rot‖ and the orientation spread, at some cost in training error', () => {
      const a = score('fixed', 0);
      const b = score('fixed', 100);
      expect(b.deltaRot).toBeLessThan(a.deltaRot / 10);
      expect(b.spread).toBeLessThan(a.spread / 10);
      expect(b.trainRmse).toBeGreaterThan(a.trainRmse);
      expect(b.rmse).toBeLessThan(a.rmse);
    });

    it('augmentation alone already narrows the spread; adding the penalty narrows it further', () => {
      const f = score('fixed', 0);
      const r = score('rotated', 0);
      const rq = score('rotated', 1);
      expect(r.spread).toBeLessThan(f.spread / 5);
      expect(rq.spread).toBeLessThan(r.spread / 3);
    });

    it('with λ = 0 the fit is ordinary (ridge) least squares on standardised features', () => {
      const w = fitReadout(ex.train.fixed, 0, 1e-10);
      // Normal equations: residuals are orthogonal to every feature.
      const res = ex.train.fixed.map((s) => dot(w, s.phi) - s.y);
      const n = ex.train.fixed[0]!.phi.length;
      for (let i = 0; i < n; i += 7) {
        const g = ex.train.fixed.reduce((a, s, k) => a + s.phi[i]! * res[k]!, 0);
        expect(Math.abs(g) / ex.train.fixed.length).toBeLessThan(1e-6);
      }
    });
  });
});
