import { describe, expect, it } from 'vitest';
import raw from '~/data/xray/nequip-oam-s.json';
import { element } from '~/lib/structure/elements';
import {
  besselFeature,
  channelMatrix,
  channelValues,
  channelVectors,
  dataflowShapes,
  energyParts,
  hasCrossProductPath,
  mostVariedChannel,
  parameterShares,
  polynomialCutoff,
  symmetryCheck,
  tensorBlocks,
  tpPaths,
  transformedView,
  zblPair,
  zblWithCutoff,
  type NequipDoc,
} from './xray-nequip';

const doc = raw as unknown as NequipDoc;
const hp = doc.model.hyperparameters;
const sys = (id: string) => doc.systems.find((s) => s.id === id)!;
const Z = (sym: string) => element(sym).z;

describe('provenance', () => {
  it('names the package, its hash and licence, and is labelled PRECOMPUTED', () => {
    expect(doc.status).toBe('PRECOMPUTED');
    expect(doc.model.checkpoint).toBe('nequip-oam-s-0-1');
    expect(doc.model.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(doc.model.licence).toBe('CC-BY-4.0');
    expect(hp.r_max).toBe(4.5);
    expect(hp.num_layers).toBe(2);
    expect(hp.feature_irreps_hidden).toEqual(['128x0e+64x1e', '128x0e']);
  });

  it('counts 0.6 M trainable parameters, most of them in the layer-2 self-connection', () => {
    const { total, parts } = parameterShares(doc);
    expect(total).toBe(616360);
    const sc = parts.find((p) => p.name.endsWith('conv.sc'))!;
    expect(sc.count).toBe(128 * 32 * 128); // 128x0e ⊗ 32 type-embedding scalars → 128x0e, fully connected
    expect(sc.share).toBeGreaterThan(0.8);
  });
});

describe('closed forms reproduce the stored network tensors', () => {
  it('polynomial cutoff: 1 at 0, 0 at r_c with zero slope', () => {
    const p = hp.polynomial_cutoff_p;
    expect(polynomialCutoff(0, p)).toBe(1);
    expect(polynomialCutoff(1, p)).toBe(0);
    const h = 1e-6;
    expect(Math.abs((polynomialCutoff(1 - h, p) - polynomialCutoff(1 - 2 * h, p)) / h)).toBeLessThan(1e-6);
  });

  it('the Bessel features with the trained frequencies match every stored edge', () => {
    for (const s of doc.systems) {
      s.edges.vec.forEach((v, e) => {
        // the molecule has no periodic images, so its bond lengths come from unrounded positions;
        // crystal edges use the stored vectors (5 significant figures)
        const exact = !s.cell;
        const pi = s.positions[s.edges.dst[e]!]!;
        const pj = s.positions[s.edges.src[e]!]!;
        const r = exact ? Math.hypot(pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]) : Math.hypot(...v);
        for (let n = 0; n < hp.num_bessels; n++) {
          const want = s.edges.radial[e]![n]!;
          const got = besselFeature(r, n, hp.bessel_weights, hp.r_max, hp.polynomial_cutoff_p);
          expect(Math.abs(got - want)).toBeLessThan(exact ? 1e-4 * Math.max(1e-2, Math.abs(want)) : 2e-4);
        }
        expect(polynomialCutoff(r / hp.r_max, hp.polynomial_cutoff_p)).toBeCloseTo(s.edges.cutoff[e]!, 4);
      });
    }
  });

  it('edge harmonics are of the bond from the receiving atom to its neighbour (−vec)', () => {
    const s = sys('ethanol');
    s.edges.vec.forEach((v, e) => {
      const r = Math.hypot(...v);
      const y = s.edges.sh[e]!;
      expect(y[0]).toBeCloseTo(1, 4);
      // component normalisation: Y₁ = √3 r̂, in (x, y, z) order
      for (let k = 0; k < 3; k++) expect(y[1 + k]!).toBeCloseTo((-Math.sqrt(3) * v[k]!) / r, 3);
    });
  });

  it('the ZBL term with cutoff reproduces the network’s ZBL energies of the dimers', () => {
    for (const d of doc.dimers) {
      const [a, b] = d.pair.split('–') as [string, string];
      d.r.forEach((r, k) => {
        const want = d.zbl[k]!;
        const got = zblWithCutoff(Z(a), Z(b), r, hp.r_max, hp.polynomial_cutoff_p);
        expect(Math.abs(got - want)).toBeLessThan(1e-5 * Math.max(1, Math.abs(want)));
      });
    }
  });

  it('…and the per-atom ZBL energies of the crystals (½ of every pair within r_c)', () => {
    for (const s of doc.systems) {
      const per = new Array<number>(s.n).fill(0);
      s.edges.vec.forEach((v, e) => {
        const r = Math.hypot(...v);
        per[s.edges.dst[e]!]! += 0.5 * zblWithCutoff(Z(s.symbols[s.edges.dst[e]!]!), Z(s.symbols[s.edges.src[e]!]!), r, hp.r_max, hp.polynomial_cutoff_p);
      });
      per.forEach((x, i) => expect(x).toBeCloseTo(s.zbl[i]!, 3));
    }
  });

  it('bare ZBL tends to the Coulomb repulsion of two nuclei at short range', () => {
    const r = 1e-9; // φ(0) = 0.02817 + 0.28022 + 0.50986 + 0.18175 = 1
    expect(zblPair(14, 14, r) / ((14.399645 * 14 * 14) / r)).toBeCloseTo(1, 6);
  });
});

describe('energy bookkeeping', () => {
  it.each(['ethanol', 'zno-wurtzite', 'si-diamond'])('%s: learned + shift + ZBL sum to each site and the total', (id) => {
    const s = sys(id);
    const parts = energyParts(s);
    parts.forEach((p, i) => expect(p.total).toBeCloseTo(s.nodeEnergy[i]!, 2));
    expect(s.nodeEnergy.reduce((a, x) => a + x, 0)).toBeCloseTo(s.energy, 3);
  });

  it('isolated atoms sit exactly at their shift; the dimer curves join it smoothly at r_c', () => {
    for (const d of doc.dimers) {
      const last = d.total[d.total.length - 1]!;
      expect(d.r[d.r.length - 1]!).toBeGreaterThan(hp.r_max);
      expect(last).toBeCloseTo(d.e0[0]! + d.e0[1]!, 5);
      d.total.forEach((t, k) => expect(Math.abs(t - d.learned[k]! - d.zbl[k]!)).toBeLessThan(1e-5 * Math.max(1, Math.abs(d.learned[k]!), Math.abs(d.zbl[k]!))));
    }
  });

  it('the ZBL prior is a large positive term even at equilibrium bond lengths', () => {
    const si = energyParts(sys('si-diamond'));
    expect(si[0]!.zbl).toBeGreaterThan(1.4);
    expect(si[0]!.learned).toBeLessThan(-si[0]!.zbl);
  });
});

describe('symmetry in the real network', () => {
  it('rotation: scalars and energy unchanged, vectors and forces co-rotate', () => {
    const r = symmetryCheck(sys('ethanol'), 'rotated')!;
    expect(r.det).toBeCloseTo(1, 6);
    expect(r.dE).toBeLessThan(1e-9);
    expect(r.forces).toBeLessThan(1e-3);
    r.layers.forEach((L) => expect(L.l0).toBeLessThan(1e-3));
    expect(r.layers[0]!.hl1!).toBeLessThan(2e-3);
    expect(r.layers[0]!.ml1!).toBeLessThan(2e-3);
  });

  it('mirror: no 1 ⊗ 1 → 1 path, so the energy is exactly invariant and vectors transform as polar vectors', () => {
    expect(hasCrossProductPath(doc)).toBe(false);
    const m = symmetryCheck(sys('ethanol'), 'mirrored')!;
    expect(m.det).toBeCloseTo(-1, 6);
    expect(m.dE).toBe(0);
    expect(m.forces).toBe(0);
    expect(m.layers[0]!.hl1).toBe(0);
  });

  it('transformed views keep the energy split', () => {
    const e = sys('ethanol');
    for (const w of ['rotated', 'mirrored'] as const) {
      const v = transformedView(e, w)!;
      v.nodeEnergy.forEach((x, i) => expect(x).toBeCloseTo(e.nodeEnergy[i]!, 4));
    }
    expect(transformedView(sys('si-diamond'), 'rotated')).toBeNull();
  });

  it('diamond: by inversion symmetry every vector feature vanishes', () => {
    const s = sys('si-diamond');
    expect(s.layers[0]!.h.l1!.flat().flat().every((x) => x === 0)).toBe(true);
  });
});

describe('tensor access', () => {
  it('lists the real tensor-product paths of each layer', () => {
    const e = sys('ethanol');
    expect(tpPaths(e.layers[0]!).map((p) => p.label)).toEqual(['0 ⊗ 0 → 0', '0 ⊗ 1 → 1']);
    expect(tpPaths(e.layers[1]!).map((p) => p.label)).toEqual(['0 ⊗ 0 → 0', '1 ⊗ 1 → 0']);
  });

  it('blocks, values, vectors and matrices agree in size', () => {
    const e = sys('ethanol');
    const shapes = dataflowShapes(doc, e);
    expect(shapes.h1).toBe(`[9, ${128 + 64 * 3}]`);
    expect(shapes.m2).toBe('[9, 192]');
    for (const id of ['embed', 'm1', 'h1', 'm2', 'h2'] as const) {
      for (const b of tensorBlocks(e, id)) {
        const m = channelMatrix(e, id, b.key)!;
        expect(m.length).toBe(9);
        expect(m[0]!.length).toBe(b.channels);
        const c = mostVariedChannel(e, id, b.key);
        expect(channelValues(e, id, b.key, c)!.length).toBe(9);
        if (b.l === 1) expect(channelVectors(e, id, b.key, c)!.length).toBe(9);
      }
    }
  });
});
