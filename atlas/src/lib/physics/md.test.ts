import { describe, expect, it } from 'vitest';
import {
  LJ_STATE,
  SAMPLE_DT,
  FRAME_DT,
  TIMESTEPS,
  Unwrapper,
  boxLength,
  cutoffJump,
  diffusionCoefficient,
  energyDiagnostics,
  fccSites,
  lammpsInput,
  linearFit,
  lj,
  ljTimePs,
  logLogSlope,
  msd,
  pairHistogram,
  pairPotential,
  rdfFromHistogram,
  smoothTableFile,
  stepsPer,
  verletOscillatorError,
  type CutoffTreatment,
} from './md';

// Deterministic pseudo-random numbers (mulberry32).
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('pair potentials', () => {
  it('Lennard-Jones has its minimum −ε at 2^(1/6) σ', () => {
    const v = lj(2 ** (1 / 6));
    expect(v.e).toBeCloseTo(-1, 12);
    expect(v.f).toBeCloseTo(0, 12);
  });

  it.each(['truncated', 'shifted', 'smooth'] as CutoffTreatment[])('%s: force is −dE/dr inside the cutoff', (t) => {
    const h = 1e-6;
    for (const r of [0.95, 1.12, 1.5, 2.0, 2.4]) {
      const fd = -(pairPotential(r + h, t).e - pairPotential(r - h, t).e) / (2 * h);
      expect(pairPotential(r, t).f).toBeCloseTo(fd, 6);
    }
  });

  it('truncated LJ jumps in energy and force at r_c; shifting removes only the energy jump', () => {
    const rc = LJ_STATE.rc;
    expect(cutoffJump('truncated').e).toBeCloseTo(4 * (rc ** -12 - rc ** -6), 12);
    expect(cutoffJump('truncated').e).toBeCloseTo(-0.016316891, 8);
    expect(cutoffJump('shifted').e).toBeCloseTo(0, 12);
    expect(cutoffJump('shifted').f).toBeCloseTo(cutoffJump('truncated').f, 12);
    expect(Math.abs(cutoffJump('shifted').f)).toBeGreaterThan(0.03);
    expect(cutoffJump('smooth')).toEqual({ e: 0, f: 0 });
  });

  it('the smooth envelope sends E, F and dF/dr continuously to zero at r_c', () => {
    const rc = LJ_STATE.rc;
    const near = pairPotential(rc - 1e-4, 'smooth');
    expect(Math.abs(near.e)).toBeLessThan(1e-9);
    expect(Math.abs(near.f)).toBeLessThan(1e-6);
    const h = 1e-5;
    const df = (pairPotential(rc - 2 * h, 'smooth').f - pairPotential(rc - h, 'smooth').f) / h;
    expect(Math.abs(df)).toBeLessThan(1e-3);
  });

  it('writes a LAMMPS table whose forces are consistent with its energies', () => {
    const txt = smoothTableFile('LJ_SMOOTH', 400, 0.5);
    const lines = txt.split('\n');
    expect(lines).toContain('LJ_SMOOTH');
    expect(lines).toContain('N 400 R 0.5 2.5');
    const rows = lines.filter((l) => /^\d+ /.test(l)).map((l) => l.split(' ').map(Number));
    expect(rows).toHaveLength(400);
    expect(rows[0]![1]).toBeCloseTo(0.5, 12);
    expect(rows.at(-1)![1]).toBeCloseTo(2.5, 12);
    expect(rows.at(-1)![2]).toBeCloseTo(0, 12);
    for (let i = 100; i < 398; i += 37) {
      const [, r0, e0] = rows[i - 1]!;
      const [, , , f] = rows[i]!;
      const [, r2, e2] = rows[i + 1]!;
      const fd = -(e2! - e0!) / (r2! - r0!);
      // Central difference on a 5 mσ grid: O(h²) relative error.
      expect(Math.abs(f! - fd)).toBeLessThan(2e-3 * Math.abs(f!) + 1e-4);
    }
  });
});

describe('LAMMPS input', () => {
  it('uses the bench/in.lj state point with an NVE integrator', () => {
    const s = lammpsInput({ treatment: 'truncated', dt: 0.005 });
    expect(s).toContain('lattice fcc 0.8442');
    expect(s).toContain('velocity all create 1.44 87287 loop geom');
    expect(s).toContain('pair_style lj/cut 2.5');
    expect(s).toContain('pair_modify shift no');
    expect(s).toContain('fix 1 all nve');
    expect(s).toContain('timestep 0.005');
  });

  it('switches pair style per treatment', () => {
    expect(lammpsInput({ treatment: 'shifted', dt: 0.005 })).toContain('pair_modify shift yes');
    const smooth = lammpsInput({ treatment: 'smooth', dt: 0.005 });
    expect(smooth).toContain('pair_style table spline 4000');
    expect(smooth).not.toContain('lj/cut');
  });

  it('every offered time step divides the sample and frame intervals', () => {
    for (const dt of TIMESTEPS) {
      expect(stepsPer(SAMPLE_DT, dt) * dt).toBeCloseTo(SAMPLE_DT, 12);
      expect(stepsPer(FRAME_DT, dt) * dt).toBeCloseTo(FRAME_DT, 12);
    }
    expect(() => stepsPer(0.02, 0.003)).toThrow();
  });

  it('the box holds 4N sites and is large enough for minimum image at r_c + skin', () => {
    const L = boxLength();
    expect(fccSites().length / 3).toBe(4 * LJ_STATE.cells ** 3);
    expect(L / 2).toBeGreaterThan(LJ_STATE.rc + 0.3);
    expect(fccSites().length / 3 / L ** 3).toBeCloseTo(LJ_STATE.density, 12);
  });
});

describe('trajectory diagnostics', () => {
  it('unwraps atoms that cross the periodic boundary', () => {
    const L = 5;
    const u = new Unwrapper(L, Float64Array.from([4.9, 0, 0]));
    u.push(Float64Array.from([0.1, 0, 0])); // crossed +x
    const out = u.push(Float64Array.from([0.4, 0, 0]));
    expect(out[0]).toBeCloseTo(5.4, 12);
    expect(msd(Float64Array.from([4.9, 0, 0]), out)).toBeCloseTo(0.25, 12);
  });

  it('g(r) of an ideal gas is 1 within noise', () => {
    const L = 10;
    const n = 400;
    const rand = rng(7);
    const pos = Float64Array.from({ length: 3 * n }, () => rand() * L);
    const { g } = rdfFromHistogram(pairHistogram(pos, L, 5, 10), 1, n, L, 5);
    // Skip the smallest shells (few pairs, large noise).
    g.slice(3).forEach((v) => expect(v).toBeGreaterThan(0.85));
    g.slice(3).forEach((v) => expect(v).toBeLessThan(1.15));
  });

  it('g(r) of the fcc lattice peaks at the nearest-neighbour distance with 12 neighbours', () => {
    const L = boxLength();
    const pos = fccSites();
    const n = pos.length / 3;
    const rmax = L / 2;
    const h = pairHistogram(pos, L, rmax, 200);
    const a = Math.cbrt(4 / LJ_STATE.density);
    const dr = rmax / 200;
    const k = Math.floor(a / Math.SQRT2 / dr);
    // Each pair counted once: N·12/2 nearest-neighbour pairs.
    expect(h[k]).toBe((n * 12) / 2);
    const { g } = rdfFromHistogram(h, 1, n, L, rmax);
    expect(g.indexOf(Math.max(...g))).toBe(k);
  });

  it('fits drift and fluctuation of a noisy linear energy', () => {
    const t = Array.from({ length: 200 }, (_, i) => i * 0.1);
    const e = t.map((x, i) => -5 + 2e-4 * x + (i % 2 ? 1e-3 : -1e-3));
    const d = energyDiagnostics(t, e);
    expect(d.drift).toBeCloseTo(2e-4, 5);
    expect(d.rms).toBeCloseTo(1e-3, 5);
    expect(linearFit([0, 1, 2], [1, 3, 5])).toEqual({ slope: 2, intercept: 1 });
  });

  it('recovers D from MSD = 6Dt', () => {
    const t = Array.from({ length: 50 }, (_, i) => i * 0.1);
    expect(diffusionCoefficient(t, t.map((x) => 6 * 0.03 * x + 0.2))).toBeCloseTo(0.03, 12);
  });
});

describe('integrator error', () => {
  it('velocity Verlet energy error scales as dt² (slope 2 on log-log axes)', () => {
    const dts = [0.2, 0.1, 0.05, 0.025];
    const err = dts.map((dt) => verletOscillatorError(dt, 200));
    expect(logLogSlope(dts, err)).toBeCloseTo(2, 1);
    // Bounded, not drifting: a 10× longer run has the same worst error.
    expect(verletOscillatorError(0.1, 2000)).toBeCloseTo(err[1]!, 6);
  });

  it('converts the LJ time unit to picoseconds', () => {
    // σ = 3.4 Å, ε/k_B = 120 K, m = 39.948 u → τ ≈ 2.15 ps.
    expect(ljTimePs(3.4, 120, 39.948)).toBeCloseTo(2.151, 2);
  });
});
