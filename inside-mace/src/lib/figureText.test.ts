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
});
