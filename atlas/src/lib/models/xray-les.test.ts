import { describe, expect, it } from 'vitest';
import raw from '~/data/xray/maceles-off.json';
import {
  becSumRule,
  coulombConstant,
  dimerMultipoles,
  erf,
  lesEnergy,
  lesKernel,
  lesPotential,
  netCharge,
  polarization,
  receptiveField,
  rotationCheck,
  separatedRows,
  slopeThroughOrigin,
  stepValues,
  type LesDoc,
  type Vec3,
} from './xray-les';

const doc = raw as unknown as LesDoc;
const hp = doc.model.hyperparameters;
const les = hp.les;
const sys = (id: string) => doc.systems.find((s) => s.id === id)!;

describe('provenance', () => {
  it('names the checkpoint, its hashes and licence, and is labelled PRECOMPUTED', () => {
    expect(doc.status).toBe('PRECOMPUTED');
    expect(doc.model.checkpoint).toBe('maceles-off');
    expect(doc.model.md5).toBe('cc10e937b55e09f05b16dba756e2311b');
    expect(doc.model.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(doc.model.licence).toBe('CC BY-NC 4.0');
  });

  it('reads the architecture of Kim et al. Table II off the file', () => {
    expect(hp.r_max).toBe(4.5);
    expect(hp.num_interactions).toBe(2);
    expect(hp.hidden_irreps).toBe('192x0e+192x1o');
    expect(hp.atomic_numbers).toEqual([1, 6, 7, 8, 9, 15, 16, 17, 35, 53]);
    expect(receptiveField(hp)).toBe(9);
    expect(les.sigma).toBe(1);
    expect(les.use_atomwise).toBe(false);
  });
});

describe('closed forms', () => {
  it('erf agrees with tabulated values', () => {
    expect(erf(0)).toBe(0);
    expect(erf(0.5)).toBeCloseTo(0.5204998778130465, 14);
    expect(erf(1)).toBeCloseTo(0.8427007929497149, 14);
    expect(erf(2.5)).toBeCloseTo(0.9995930479825550, 14);
    expect(erf(4)).toBeCloseTo(0.9999999845827421, 14);
    expect(erf(-1.3)).toBeCloseTo(-0.9340079449406524, 14);
  });

  it('the kernel is finite at contact and Coulombic far away', () => {
    expect(lesKernel(0, 1)).toBeCloseTo(Math.sqrt(2 / Math.PI), 12);
    expect(lesKernel(1e-6, 1)).toBeCloseTo(Math.sqrt(2 / Math.PI), 8);
    expect(lesKernel(12, 1) * 12).toBeCloseTo(1, 12);
  });

  it('the stored prefactor is 0.47 % below 1/(2ε₀): k = 14.331 instead of 14.400 eV Å', () => {
    expect(les.norm_factor).toBeCloseTo(90.0474, 4);
    expect(coulombConstant(les.norm_factor)).toBeCloseTo(14.3315, 4);
    expect(doc.model.halfInvEps0).toBeCloseTo(90.4756, 3);
    expect(1 - les.norm_factor / doc.model.halfInvEps0).toBeCloseTo(0.00473, 4);
  });

  it('E_lr is even in q; mean-removed P does not depend on the origin', () => {
    const pos: Vec3[] = [
      [0, 0, 0],
      [1.1, 0.2, 0],
      [-0.4, 0.9, 0.3],
    ];
    const q = [0.3, -0.5, 0.1];
    expect(lesEnergy(q, pos, 1, 90)).toBeCloseTo(lesEnergy(q.map((x) => -x), pos, 1, 90), 14);
    const moved = pos.map((p) => [p[0] + 5, p[1] - 2, p[2] + 1] as Vec3);
    polarization(q, pos).forEach((x, k) => expect(polarization(q, moved)[k]).toBeCloseTo(x, 12));
  });

  it('slope through the origin and correlation', () => {
    const { slope, r } = slopeThroughOrigin([1, 2, 3], [-2, -4, -6]);
    expect(slope).toBeCloseTo(-2, 12);
    expect(r).toBeCloseTo(-1, 12);
  });
});

describe('the stored forward passes', () => {
  it('E_lr and φ recomputed from the stored charges match the pipeline', () => {
    for (const s of doc.systems) {
      expect(lesEnergy(s.q, s.positions, les.sigma, les.norm_factor)).toBeCloseTo(s.eLr, 3);
      const phi = lesPotential(s.q, s.positions, les.sigma, les.norm_factor);
      phi.forEach((x, i) => expect(Math.abs(x - s.phi[i]!)).toBeLessThan(1e-3 * Math.max(1, Math.abs(x))));
      // q = q⁽¹⁾ + q⁽²⁾
      s.q.forEach((x, i) => expect(s.qLayers[0][i]! + s.qLayers[1][i]!).toBeCloseTo(x, 4));
      // P from the stored charges
      polarization(s.q, s.positions).forEach((x, k) => expect(x).toBeCloseTo(s.P[k]!, 3));
    }
  });

  it('latent charges do not sum to zero, yet the BEC acoustic sum rule holds exactly', () => {
    for (const s of doc.systems) {
      expect(Math.abs(netCharge(s.q))).toBeGreaterThan(1e-3);
      expect(becSumRule(s.Z)).toBeLessThan(1e-3); // stored to 4 significant figures
    }
  });

  it('rotating ethanol leaves charges and energies unchanged and co-rotates the forces', () => {
    const r = rotationCheck(sys('ethanol'))!;
    expect(r.dE).toBeLessThan(1e-7);
    expect(r.dq).toBeLessThan(1e-4);
    expect(r.dFsr).toBeLessThan(1e-4);
    expect(r.dFlr).toBeLessThan(1e-4);
  });

  it('the dipeptide’s dipole and BECs point opposite to DFT', () => {
    const d = sys('dipeptide');
    const P = d.P;
    const mu = d.dft!.dipole;
    expect(P[0] * mu[0] + P[1] * mu[1] + P[2] * mu[2]).toBeLessThan(0);
    const diag = (Z: Vec3[][]) => Z.flatMap((Zi) => [0, 1, 2].map((a) => Zi[a]![a]!));
    expect(slopeThroughOrigin(diag(d.dft!.Z), diag(d.Z)).slope).toBeLessThan(-0.5);
  });

  it('per-atom views have one value per atom', () => {
    const s = sys('ethanol');
    for (const id of ['h1', 'h2', 'q', 'phi', 'energy'] as const) expect(stepValues(s, id, 0).values).toHaveLength(s.n);
    const e = stepValues(s, 'energy', 0).values.reduce((a, b) => a + b, 0);
    expect(e).toBeCloseTo(s.eLr, 3);
  });
});

describe('water dimer: what couples the two molecules', () => {
  it('once the graph splits, the local energy and charges are exactly the monomers’', () => {
    const rows = separatedRows(doc);
    expect(rows.length).toBeGreaterThan(15);
    for (const r of rows) {
      expect(r.dEsr).toBeLessThan(1e-8);
      expect(r.dq).toBeLessThan(1e-4);
      expect(Math.abs(r.eLrInter)).toBeGreaterThan(1e-5);
    }
    // just inside r_c the local part still contributes
    const close = doc.dimer.rows.find((r) => r.minGap < hp.r_max && r.minGap > 4)!;
    const [A, B] = doc.dimer.monomers;
    expect(Math.abs(close.eSr - A!.eSr - B!.eSr)).toBeGreaterThan(1e-3);
  });

  it('the latent tail approaches the multipole expansion of the monomers’ charges', () => {
    const [A, B] = doc.dimer.monomers;
    for (const d of [20, 25, 30]) {
      const row = doc.dimer.rows.find((r) => r.dOO === d)!;
      const exact = row.eLr - A!.eLr - B!.eLr;
      const m = dimerMultipoles(doc, d);
      expect(Math.abs(m.total - exact) / Math.abs(exact)).toBeLessThan(0.08);
    }
  });
});

describe('validation on the les_fit test set', () => {
  it('reproduces DFT forces at the 10–35 meV/Å level, and the LES term cannot be dropped', () => {
    for (const s of doc.validation.subsets) {
      expect(s.forceRmse).toBeLessThan(0.035);
      expect(s.forceRmseWithoutLr).toBeGreaterThan(2 * s.forceRmse);
      expect(s.energyRmsePerAtom).toBeLessThan(0.002);
    }
  });

  it('BECs and dipoles are strongly anti-correlated with DFT in every subset', () => {
    for (const s of doc.validation.subsets) {
      expect(s.becCorr).toBeLessThan(-0.85);
      expect(s.dipoleCorr).toBeLessThan(-0.95);
      expect(s.becSlope).toBeLessThan(0);
    }
  });
});
