import { describe, expect, it } from 'vitest';
import si from '../data/experiments/si-bond-stretch/manifest.json';
import zno from '../data/experiments/zno-axial-bond/manifest.json';
import { parseManifest } from './experiment';
import { ELEMENT_DISPLAY, baselineFacts, composition, experimentInfo } from './experimentInfo';

const mSi = parseManifest(si);
const mZnO = parseManifest(zno);

describe('experiment info', () => {
  it('describes every committed experiment and every element in it', () => {
    for (const m of [mSi, mZnO]) {
      expect(experimentInfo(m.id)).not.toBeNull();
      for (const s of new Set(m.structure.symbols)) expect(ELEMENT_DISPLAY[s]).toBeDefined();
    }
    expect(experimentInfo('unknown')).toBeNull();
  });

  it('element colours stay clear of the mint (moved atom) and amber (change) accents', () => {
    for (const { color } of Object.values(ELEMENT_DISPLAY)) expect(['#6fd3bd', '#f0a945']).not.toContain(color);
  });

  it('Si diamond baseline: no forces and no layer-1 vector features (tetrahedral sites)', () => {
    const f = baselineFacts(mSi);
    expect(f.forceFree).toBe(true);
    expect(f.vectorsVanish).toBe(true);
  });

  it('ZnO wurtzite baseline (not relaxed with the model): non-zero forces along c and non-zero vectors (C3v sites)', () => {
    const f = baselineFacts(mZnO);
    expect(f.forceFree).toBe(false);
    expect(f.maxForce).toBeCloseTo(0.0371, 3);
    expect(f.vectorsVanish).toBe(false);
    const F = mZnO.frames[mZnO.displacement.baselineIndex]!;
    if (F.status !== 'ok') throw new Error('baseline must be ok');
    for (const v of F.forces) expect(Math.abs(v[0]) + Math.abs(v[1])).toBeLessThan(1e-5); // along c only
  });

  it('bond criterion separates first neighbours from the second shell', () => {
    expect(mSi.shells[0]!).toBeLessThan(experimentInfo(mSi.id)!.bondCutoff);
    expect(mSi.shells[1]!).toBeGreaterThan(experimentInfo(mSi.id)!.bondCutoff);
    const z = experimentInfo(mZnO.id)!.bondCutoff;
    expect(mZnO.shells.filter((r) => r < z)).toEqual([1.975, 1.99]);
  });

  it('composition counts species', () => {
    expect(composition(mSi)).toEqual([{ symbol: 'Si', count: 216 }]);
    expect(composition(mZnO)).toEqual([{ symbol: 'Zn', count: 150 }, { symbol: 'O', count: 150 }]);
  });
});

describe('demonstration captions per experiment', () => {
  it('claim zero baseline forces for Si only, and quote the stored force for ZnO', async () => {
    const { demoSteps } = await import('./demo');
    expect(demoSteps(mSi)[0]!.caption).toContain('A periodic silicon crystal (216 atoms)');
    expect(demoSteps(mSi)[0]!.caption).toContain('zero by symmetry');
    const z = demoSteps(mZnO)[0]!.caption;
    expect(z).toContain('zinc oxide (wurtzite) crystal (300 atoms)');
    expect(z).not.toContain('zero');
    expect(z).toContain('largest 0.037 eV/Å');
    expect(demoSteps(mZnO)[1]!.caption).toContain('it pushes back');
  });
});
