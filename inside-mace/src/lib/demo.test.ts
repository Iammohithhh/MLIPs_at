import { describe, expect, it } from 'vitest';
import raw from '../data/experiments/si-bond-stretch/manifest.json';
import { axialForce, energyChange, okFrameAt, parseManifest } from './experiment';
import { demoSteps } from './demo';
import { DEMO_STEPS, decodeState, defaultState, encodeState } from './urlState';

const m = parseManifest(raw);
const ck = String(m.provenance.checkpoint);

describe('demonstration sequence', () => {
  const steps = demoSteps(m);
  it('has the documented number of steps, each on an evaluated frame', () => {
    expect(steps).toHaveLength(DEMO_STEPS);
    for (const s of steps) expect(okFrameAt(m, s.state.frame!)).not.toBeNull();
  });
  it('step 1 claims zero forces only because the stored baseline forces are exactly zero', () => {
    const f = okFrameAt(m, m.displacement.baselineIndex)!;
    expect(Math.max(...f.forces.map((v) => Math.hypot(...v)))).toBe(0);
  });
  it('captions quote the stored numbers', () => {
    const last = m.frames.length - 1;
    expect(steps[3]!.caption).toContain(`+${energyChange(m, last)!.toFixed(3)} eV`);
    expect(steps[1]!.caption).toContain(`−${Math.abs(axialForce(m, last)!).toFixed(2)} eV/Å`);
  });
  it('presentation state round-trips through the URL and bad values are reported', () => {
    const s = { ...defaultState(m, ck), present: true, demo: 3, frame: 24 };
    expect(decodeState(encodeState(s), m, ck).state).toEqual(s);
    expect(encodeState(defaultState(m, ck))).not.toContain('present');
    const r = decodeState('present=yes&demo=9', m, ck);
    expect(r.state.present).toBe(false);
    expect(r.state.demo).toBe(0);
    expect(r.issues).toHaveLength(2);
  });
});
