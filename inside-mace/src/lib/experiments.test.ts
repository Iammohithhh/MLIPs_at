// Checks that hold for every committed experiment (Si bond stretch, ZnO axial bond, …). The Si-specific
// geometry and URL tests stay in experiment.test.ts; these run over whatever manifests are on disk, so a
// newly generated experiment is checked before any page uses it.

import { describe, expect, it } from 'vitest';
import { axialForce, energyChange, forceChange, framePositions, mic, norm, okFrameAt, parseManifest } from './experiment';

const files = import.meta.glob<{ default: unknown }>('../data/experiments/*/manifest.json', { eager: true });
const manifests = Object.entries(files).map(([path, mod]) => [path.split('/').at(-2)!, parseManifest(mod.default)] as const);

describe('committed experiments', () => {
  it('include the Si and ZnO experiments', () => {
    expect(manifests.map(([id]) => id).sort()).toEqual(['si-bond-stretch', 'zno-axial-bond']);
  });
});

describe.each(manifests)('%s', (id, m) => {
  const B = m.displacement.baselineIndex;
  const moved = m.displacement.atom;

  it('is a real precomputed MACE-MP-0a medium run with matching id', () => {
    expect(m.id).toBe(id);
    expect(m.status).toBe('PRECOMPUTED');
    expect(m.provenance.sha256).toBe('01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845');
    expect(m.provenance.dtype).toBe('float64');
    expect(m.structure.symbols.length).toBe(m.structure.n);
  });

  it('baseline restoration reproduces the original outputs', () => {
    expect(m.frames[B]!.d).toBe(0);
    expect(m.checks.restoration.energyAbsDiff).toBeLessThan(1e-9);
    expect(m.checks.restoration.forceMaxAbsDiff).toBeLessThan(1e-9);
    expect(energyChange(m, B)).toBe(0);
    expect(norm(forceChange(m, B, moved)!)).toBe(0);
  });

  it('energy changes are E(frame) − E(baseline) and match the stored dE', () => {
    for (const f of m.frames) if (f.status === 'ok') expect(energyChange(m, f.k)!).toBeCloseTo(f.dE, 9);
  });

  it('force along the path agrees with the slope of the stored energies (central differences)', () => {
    for (let k = 1; k < m.frames.length - 1; k++) {
      const a = okFrameAt(m, k - 1);
      const c = okFrameAt(m, k + 1);
      if (!a || !c || !okFrameAt(m, k)) continue;
      const slope = (c.energy - a.energy) / (c.d - a.d);
      expect(Math.abs(axialForce(m, k)! + slope)).toBeLessThan(0.02 * Math.max(1, Math.abs(slope)));
    }
  });

  it('the path is locally stable: F·û decreases as d increases (positive curvature of E(d))', () => {
    const ok = m.frames.filter((f) => f.status === 'ok');
    for (let i = 1; i < ok.length; i++) expect(axialForce(m, ok[i]!.k)!).toBeLessThan(axialForce(m, ok[i - 1]!.k)!);
  });

  it('site energies of all atoms sum to the total energy', () => {
    for (const f of m.frames) {
      if (f.status !== 'ok') continue;
      const s = f.nodeEnergy.reduce((a, b) => a + b, 0);
      expect(Math.abs(s - f.energy)).toBeLessThan(m.structure.n * 1e-6);
    }
  });

  it('forces sum to zero in the periodic cell', () => {
    for (const f of m.frames) {
      if (f.status !== 'ok') continue;
      for (let c = 0; c < 3; c++) expect(Math.abs(f.forces.reduce((a, F) => a + F[c]!, 0))).toBeLessThan(m.structure.n * 1e-6);
    }
  });

  it('finite-difference checks are within documented tolerance and converge', () => {
    for (const row of m.checks.finiteDifference) {
      expect(row.steps.at(-1)!.absDiff).toBeLessThan(2e-6);
      expect(row.steps.at(-1)!.absDiff).toBeLessThan(row.steps[0]!.absDiff + 1e-9);
    }
  });

  it('rotating the model input leaves energy invariant and rotates forces as F → R F', () => {
    const R = m.checks.rotation.R;
    for (const run of m.checks.rotation.runs) {
      expect(run.energyAbsDiff).toBeLessThan(1e-8);
      const f = okFrameAt(m, run.frame)!;
      f.forces.forEach((F, i) => {
        const RF = R.map((row) => row[0] * F[0] + row[1] * F[1] + row[2] * F[2]);
        const G = run.forces[i]!;
        for (let c = 0; c < 3; c++) expect(Math.abs(RF[c]! - G[c]!)).toBeLessThan(5e-7);
      });
    }
  });

  it('only the moved atom moves, along the axis towards its partner, keeping its index', () => {
    const k = Math.min(B + 4, m.frames.length - 1);
    const p = framePositions(m, k)!;
    const p0 = framePositions(m, B)!;
    p.forEach((x, i) => {
      const dd = norm([x[0] - p0[i]![0], x[1] - p0[i]![1], x[2] - p0[i]![2]]);
      expect(dd).toBeCloseTo(i === moved ? Math.abs(m.frames[k]!.d) : 0, 6);
    });
    const v = mic(m, p0[moved]!, p0[m.displacement.partner]!);
    const u = m.displacement.axis;
    expect(v[0] * u[0] + v[1] * u[1] + v[2] * u[2]).toBeCloseTo(norm(v), 5);
  });

  it('the local set is the moved atom followed by distinct neighbours', () => {
    expect(m.local[0]).toBe(moved);
    expect(new Set(m.local).size).toBe(m.local.length);
  });
});
