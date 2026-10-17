import { describe, expect, it } from 'vitest';
import { mulberry32, randomRotation } from '../chem/rng';
import {
  DEFAULT_GAP,
  SI_A0,
  cellDescriptors,
  cholSolve,
  cholesky,
  diamondCell,
  fitGap,
  leastSquares,
  neighbourVectors,
  predict,
  rattled,
  selectSparse,
  strained,
  swEnergy,
  trainingCells,
  type Cell,
  type TrainingConfig,
} from './gap';
import { DEFAULT_SOAP, soapVector, type SoapOptions } from './soap';

const o: SoapOptions = { ...DEFAULT_SOAP, lmax: 4, nr: 14 };

const configsOf = (cells: Cell[]): TrainingConfig[] => cells.map((c) => ({ envs: cellDescriptors(c, o), energy: swEnergy(c) }));

describe('reference and cells', () => {
  it('Stillinger–Weber diamond Si: −4.3366 eV per atom at a = 5.431 Å, a minimum under strain', () => {
    const e0 = swEnergy(diamondCell()) / 8;
    expect(e0).toBeCloseTo(-2 * 2.1683, 3);
    expect(swEnergy(strained(diamondCell(), 0.02)) / 8).toBeGreaterThan(e0);
    expect(swEnergy(strained(diamondCell(), -0.02)) / 8).toBeGreaterThan(e0);
  });

  it('finds 4 + 12 + 12 neighbours within 5 Å in diamond Si', () => {
    const nb = neighbourVectors(diamondCell(), 5.0);
    nb.forEach((v) => expect(v.length).toBe(28));
  });

  it('predicted energies do not depend on rotating the whole cell (descriptors are invariant)', () => {
    const c = rattled(diamondCell(), 0.1, 3);
    const R = randomRotation(mulberry32(2));
    const envs = neighbourVectors(c, o.rc);
    const a = cellDescriptors(c, o);
    const b = envs.map((env) => env.map((v) => [R[0]! * v[0] + R[1]! * v[1] + R[2]! * v[2], R[3]! * v[0] + R[4]! * v[1] + R[5]! * v[2], R[6]! * v[0] + R[7]! * v[1] + R[8]! * v[2]] as [number, number, number]));
    b.forEach((env, i) => {
      const q = soapVector(env, o);
      for (let k = 0; k < q.length; k++) expect(Math.abs(q[k]! - a[i]![k]!)).toBeLessThan(1e-12);
    });
  });
});

describe('linear algebra', () => {
  it('solves a symmetric positive-definite system by Cholesky', () => {
    const A = new Float64Array([4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2]);
    const x = cholSolve(cholesky(A, 3), 3, [1, 2, 3]);
    for (let i = 0; i < 3; i++) expect(A[3 * i]! * x[0]! + A[3 * i + 1]! * x[1]! + A[3 * i + 2]! * x[2]!).toBeCloseTo([1, 2, 3][i]!, 12);
  });
});

describe('least squares', () => {
  it('matches the normal equations on a well-conditioned problem', () => {
    const rand = mulberry32(8);
    const rows = 9;
    const cols = 4;
    const A = Float64Array.from({ length: rows * cols }, () => rand() - 0.5);
    const b = Array.from({ length: rows }, () => rand());
    const x = leastSquares(A, rows, cols, b);
    const AtA = new Float64Array(cols * cols);
    const Atb = new Float64Array(cols);
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < cols; j++) for (let r = 0; r < rows; r++) AtA[i * cols + j]! += A[r * cols + i]! * A[r * cols + j]!;
      for (let r = 0; r < rows; r++) Atb[i]! += A[r * cols + i]! * b[r]!;
    }
    const y = cholSolve(cholesky(AtA, cols), cols, Atb);
    for (let i = 0; i < cols; i++) expect(x[i]).toBeCloseTo(y[i]!, 12);
  });
});

describe('sparse GAP on total energies', () => {
  const cells = trainingCells({ count: 24, maxStrain: 0.03, maxRattle: 0.12, seed: 7 });
  const train = configsOf(cells);
  const all = train.flatMap((c) => c.envs);
  const sparseIdx = selectSparse(all, 40, DEFAULT_GAP.zeta);
  const model = fitGap(train, sparseIdx.map((i) => all[i]!));

  it('picks distinct representative environments', () => {
    expect(new Set(sparseIdx).size).toBe(sparseIdx.length);
  });

  it('fits the training energies to well under a meV per atom', () => {
    const rmse = Math.sqrt(train.reduce((s, c) => s + ((predict(model, c.envs).energy - c.energy) / 8) ** 2, 0) / train.length);
    expect(rmse).toBeLessThan(1e-3);
  });

  it('predicts held-out cells from the same distribution', () => {
    const test = configsOf(trainingCells({ count: 6, maxStrain: 0.03, maxRattle: 0.12, seed: 99 }));
    for (const c of test) expect(Math.abs(predict(model, c.envs).energy - c.energy) / 8).toBeLessThan(0.01);
  });

  it('is exact in the noiseless limit when every environment is a sparse point', () => {
    const small = configsOf(trainingCells({ count: 3, maxStrain: 0.02, maxRattle: 0.1, seed: 5 }));
    const envs = small.flatMap((c) => c.envs);
    const m = fitGap(small, envs, { ...DEFAULT_GAP, sigmaE: 1e-5 });
    for (const c of small) expect(predict(m, c.envs).energy).toBeCloseTo(c.energy, 3);
  });

  it('per-atom σ vanishes at a representative environment and grows far from the data', () => {
    const s0 = predict(model, [model.sparse[0]!]).sigma[0]!;
    expect(s0).toBeLessThan(0.01);
    const near = predict(model, cellDescriptors(diamondCell(), o)).sigma[0]!;
    const far = predict(model, cellDescriptors(strained(diamondCell(), 0.15), o)).sigma[0]!;
    expect(far).toBeGreaterThan(5 * near);
  });

  it('errs much more outside the trained strain range', () => {
    const err = (eps: number) => {
      const c = strained(diamondCell(SI_A0), eps);
      return Math.abs(predict(model, cellDescriptors(c, o)).energy - swEnergy(c)) / 8;
    };
    expect(err(0.12)).toBeGreaterThan(10 * err(0.0));
  });
});
