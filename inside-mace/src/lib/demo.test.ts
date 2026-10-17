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

import znoRaw from '../data/experiments/zno-axial-bond/manifest.json';

describe('demonstration sequence for the unrelaxed ZnO run', () => {
  const z = parseManifest(znoRaw);
  const steps = demoSteps(z);
  const last = z.frames.length - 1;
  it('step 1 does not claim zero forces and quotes the stored largest baseline force', () => {
    expect(steps[0]!.caption).not.toMatch(/zero/);
    expect(steps[0]!.caption).toContain('not relaxed');
    const f = okFrameAt(z, z.displacement.baselineIndex)!;
    const max = Math.max(...f.forces.map((v) => Math.hypot(...v)));
    expect(steps[0]!.caption).toContain(`${max.toFixed(3)} eV/Å`);
  });
  it('captions quote the stored ZnO numbers and URLs carry the experiment id', () => {
    expect(steps[3]!.caption).toContain(`+${energyChange(z, last)!.toFixed(3)} eV`);
    expect(steps[1]!.caption).toContain(`−${Math.abs(axialForce(z, last)!).toFixed(2)} eV/Å`);
    const zck = String(z.provenance.checkpoint);
    expect(encodeState({ ...defaultState(z, zck), ...steps[1]!.state, present: true, demo: 2 })).toContain('exp=zno-axial-bond');
  });
});
