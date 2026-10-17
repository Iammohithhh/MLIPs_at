import { describe, expect, it } from 'vitest';
import raw from '../data/experiments/si-bond-stretch/manifest.json';
import { energyChange, parseManifest } from './experiment';
import { figureText } from './figureText';
import { defaultState } from './urlState';

const m = parseManifest(raw);
const ck = String(m.provenance.checkpoint);

describe('still-image export text', () => {
  it('describes the selected evaluated frame with units, status and source', () => {
    const s = { ...defaultState(m, ck), frame: 24, view: 'split' as const };
    const t = figureText(m, s, 'element', 0.5, 'http://x/?frame=24');
    expect(t.title).toContain('+0.300 Å');
    expect(t.status).toMatch(/PRECOMPUTED/);
    expect(t.settings.join(' ')).toContain(`+${energyChange(m, 24)!.toFixed(4)} eV`);
    expect(t.settings.join(' ')).toContain('Frame 25/25');
    expect(t.settings[0]).toContain('01bfe22100139f42');
    expect(t.source).toBe('http://x/?frame=24');
    expect(t.legend[0]).toContain('0.5 Å of length per eV/Å');
    expect(t.panes).toEqual(['Baseline · d = 0', 'Perturbed · d = +0.300 Å']);
  });

  it('names force-change arrows in the difference view', () => {
    const t = figureText(m, { ...defaultState(m, ck), frame: 24, view: 'difference' as const }, 'element', 0.5, 'u');
    expect(t.legend[0]).toMatch(/^Force-change arrows ΔF = F\(d\) − F\(0\)/);
    expect(figureText(m, { ...defaultState(m, ck), frame: 24 }, 'element', 0.5, 'u').legend[0]).toMatch(/^Force arrows/);
  });
});

describe('still-image export text for another experiment', () => {
  it('names the moved element, structure and bond criterion of that experiment', async () => {
    const zno = parseManifest((await import('../data/experiments/zno-axial-bond/manifest.json')).default);
    const t = figureText(zno, { ...defaultState(zno, ck), frame: 24 }, 'element', 0.5, 'u');
    expect(t.title).toContain('Moving one Zn atom +0.300 Å');
    expect(t.settings[1]).toContain('ZnO wurtzite 5×5×3 supercell, 300 atoms');
    expect(t.legend[1]).toContain('Zn–O bonds < 2.3 Å');
    expect(t.legend[1]).toContain('Zn slate blue, O brick red');
    expect(t.legend[1]).not.toContain('Si');
    const si = figureText(m, defaultState(m, ck), 'element', 0.5, 'u');
    expect(si.legend[1]).toContain('Si grey');
    expect(si.legend[1]).toContain('Si–Si bonds < 2.6 Å');
  });
});
