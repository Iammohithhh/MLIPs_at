import { describe, expect, it } from 'vitest';
import raw from '../data/experiments/si-bond-stretch/manifest.json';
import {
  axialForce,
  baseline,
  blockScaleMax,
  energyChange,
  featureChange,
  forceChange,
  framePositions,
  invert3,
  mic,
  norm,
  okFrameAt,
  parseManifest,
  relativeFeatureChange,
  siteEnergyChange,
  snapToFrame,
  type Manifest,
} from './experiment';
import { decodeState, defaultState, encodeState } from './urlState';

const m = parseManifest(raw);
const B = m.displacement.baselineIndex;
const moved = m.displacement.atom;

function withFailedFrame(k: number): Manifest {
  const copy = structuredClone(raw) as { frames: Record<string, unknown>[] };
  const f = copy.frames[k]!;
  copy.frames[k] = { k: f.k, d: f.d, status: 'failed', error: 'RuntimeError: test', seconds: 0 };
  return parseManifest(copy);
}

describe('manifest', () => {
  it('is the real precomputed MACE-MP-0a medium run', () => {
    expect(m.status).toBe('PRECOMPUTED');
    expect(m.provenance.sha256).toBe('01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845');
    expect(m.provenance.dtype).toBe('float64');
  });

  it('baseline restoration reproduces the original outputs', () => {
    expect(m.checks.restoration.energyAbsDiff).toBeLessThan(1e-9);
    expect(m.checks.restoration.forceMaxAbsDiff).toBeLessThan(1e-9);
    expect(energyChange(m, B)).toBe(0);
    expect(norm(forceChange(m, B, moved)!)).toBe(0);
  });

  it('energy changes are E(frame) − E(baseline) and match the stored dE', () => {
    for (const f of m.frames) if (f.status === 'ok') expect(energyChange(m, f.k)!).toBeCloseTo(f.dE, 9);
  });

  it('forces are −∂E/∂x: the moved atom pushes back towards the baseline (restoring)', () => {
    for (const f of m.frames) {
      if (f.status !== 'ok' || f.d === 0) continue;
      expect(Math.sign(axialForce(m, f.k)!)).toBe(-Math.sign(f.d));
    }
  });

  it('force along the path agrees with the slope of the stored energies (central differences)', () => {
    for (let k = 1; k < m.frames.length - 1; k++) {
      const a = okFrameAt(m, k - 1)!;
      const c = okFrameAt(m, k + 1)!;
      const slope = (c.energy - a.energy) / (c.d - a.d);
      expect(Math.abs(axialForce(m, k)! + slope)).toBeLessThan(0.02 * Math.max(1, Math.abs(slope)));
    }
  });

  it('site energies of all atoms sum to the total energy', () => {
    for (const f of m.frames) {
      if (f.status !== 'ok') continue;
      const s = f.nodeEnergy.reduce((a, b) => a + b, 0);
      expect(Math.abs(s - f.energy)).toBeLessThan(1e-4); // 216 values each rounded to 1e-7
    }
  });

  it('finite-difference checks are within documented tolerance', () => {
    for (const row of m.checks.finiteDifference) expect(row.steps.at(-1)!.absDiff).toBeLessThan(2e-6);
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
});

describe('frames', () => {
  it('snap to evaluated frames and never interpolate', () => {
    expect(snapToFrame(m, 0)).toBe(B);
    expect(snapToFrame(m, 0.011)).toBe(B);
    expect(snapToFrame(m, 0.013)).toBe(B + 1);
    expect(snapToFrame(m, 99)).toBe(m.frames.length - 1);
  });

  it('only the moved atom changes position, along the axis, keeping its index', () => {
    const k = B + 4;
    const p = framePositions(m, k)!;
    const p0 = framePositions(m, B)!;
    p.forEach((x, i) => {
      const dd = norm([x[0] - p0[i]![0], x[1] - p0[i]![1], x[2] - p0[i]![2]]);
      expect(dd).toBeCloseTo(i === moved ? Math.abs(m.frames[k]!.d) : 0, 6);
    });
  });

  it('missing data is null, never zero', () => {
    const f = withFailedFrame(B + 2);
    expect(energyChange(f, B + 2)).toBeNull();
    expect(forceChange(f, B + 2, moved)).toBeNull();
    expect(siteEnergyChange(f, B + 2, moved)).toBeNull();
    expect(featureChange(f, B + 2, 0, 'h0', moved)).toBeNull();
    expect(framePositions(f, B + 2)).toBeNull();
  });

  it('the last layer has no l = 1 block (MACE-MP-0a medium outputs scalars only), reported as null', () => {
    expect(featureChange(m, B + 1, 1, 'h1', moved)).toBeNull();
    expect(featureChange(m, B + 1, 0, 'h1', moved)).not.toBeNull();
  });
});

describe('metrics', () => {
  it('relative change uses a floor so symmetry-vanishing baselines cannot explode', () => {
    for (const block of ['h0', 'h1'] as const) {
      for (let i = 0; i < m.structure.n; i++) {
        const r = relativeFeatureChange(m, m.frames.length - 1, 0, block, i);
        if (r !== null) expect(Number.isFinite(r)).toBe(true);
      }
    }
  });

  it('colour-scale maximum is fixed per block across frames', () => {
    const max = blockScaleMax(m, 0, 'h0')!;
    for (const f of m.frames) if (f.status === 'ok') for (const v of f.change[0]!.h0) expect(v).toBeLessThanOrEqual(max);
    expect(blockScaleMax(m, 1, 'h1')).toBeNull();
  });
});

describe('geometry', () => {
  it('inverts the cell and finds minimum images', () => {
    const inv = invert3(m.structure.cell);
    const c = m.structure.cell;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const v = c[i]![0] * inv[0]![j]! + c[i]![1] * inv[1]![j]! + c[i]![2] * inv[2]![j]!;
        expect(v).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    const p = m.structure.positions;
    const partner = m.displacement.partner;
    const v = mic(m, p[moved]!, p[partner]!);
    expect(norm(v)).toBeCloseTo(m.shells[0]!, 2);
    const u = m.displacement.axis;
    expect(v[0] * u[0] + v[1] * u[1] + v[2] * u[2]).toBeCloseTo(norm(v), 5); // +d moves towards the partner
  });
});

describe('url state', () => {
  const ck = 'mace-mp-0a-medium';
  it('round-trips', () => {
    const s = { ...defaultState(m, ck), frame: B + 3, stage: 'messages' as const, channel: 17, view: 'difference' as const, atom: m.local[3]! };
    const back = decodeState(encodeState(s), m, ck);
    expect(back.issues).toEqual([]);
    expect(back.state).toEqual(s);
  });

  it('rejects unsupported values and reports why', () => {
    const r = decodeState('exp=si-bond-stretch&frame=999&ch=500&stage=magic&atom=-1', m, ck);
    expect(r.state).toEqual(defaultState(m, ck));
    expect(r.issues.length).toBe(4);
    expect(decodeState('exp=other', m, ck).issues[0]).toMatch(/not available/);
    expect(decodeState('ckpt=mace-mp-0b3', m, ck).issues[0]).toMatch(/not the one evaluated/);
  });

  it('never restores a failed frame', () => {
    const f = withFailedFrame(B + 2);
    const r = decodeState(`frame=${B + 2}`, f, ck);
    expect(r.state.frame).toBe(B);
    expect(r.issues[0]).toMatch(/failed/);
  });

  it('baseline accessor returns the d = 0 frame', () => {
    expect(baseline(m).d).toBe(0);
  });
});
