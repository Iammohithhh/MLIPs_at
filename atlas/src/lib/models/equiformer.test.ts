import { describe, expect, it } from 'vitest';
import { axisRotation, randomRotationMatrix } from './equiformer.testutil';
import {
  activationError,
  attend,
  attentionScene,
  cutoffJump,
  eqv2DefaultGrid,
  evalAt,
  fromGrid,
  krWeights,
  layerNorm,
  logit,
  makeGrid,
  makeToyAttention,
  randomCoefs,
  smoothLeakyRelu,
  swigluExactGrid,
  toGrid,
} from './equiformer';

const R = randomRotationMatrix(5);
const dist = (p: number[]) => Math.hypot(p[0]!, p[1]!, p[2]!);

describe('Kostelec–Rockmore quadrature (e3nn grid)', () => {
  it('integrates polynomials in cos β of degree < N_β exactly', () => {
    for (const N of [4, 8, 14, 20]) {
      const w = krWeights(N);
      for (let d = 0; d < N; d++) {
        let q = 0;
        for (let j = 0; j < N; j++) q += w[j]! * Math.cos((Math.PI * (j + 0.5)) / N) ** d;
        const exact = d % 2 === 0 ? 2 / (d + 1) : 0;
        expect(q).toBeCloseTo(exact, 12);
      }
    }
  });
  it('rejects odd N_β, as e3nn does', () => {
    expect(() => krWeights(7)).toThrow();
  });
  it('weights sum to 4π over the whole grid', () => {
    const g = makeGrid(3, 8, 9);
    expect(g.W.reduce((s, v) => s + v, 0)).toBeCloseTo(4 * Math.PI, 12);
  });
});

describe('G and G⁻¹', () => {
  it('round-trip a degree-L feature exactly on the EquiformerV2 default grid', () => {
    for (const L of [2, 4, 6]) {
      const d = eqv2DefaultGrid(L);
      const g = makeGrid(L, d.nBeta, d.nAlpha);
      const x = randomCoefs(L, 3);
      const y = fromGrid(g, toGrid(g, x));
      x.forEach((v, c) => expect(y[c]).toBeCloseTo(v, 12));
    }
  });
  it('G samples the function f(û) = Σ x_lm Y_lm(û)', () => {
    const g = makeGrid(3, 8, 9);
    const x = randomCoefs(3, 4);
    const v = toGrid(g, x);
    g.dirs.forEach((u, p) => expect(v[p]).toBeCloseTo(evalAt(x, 3, u), 12));
  });
  it('reproduces EquiformerV2’s default resolutions (so3.py SO3_Grid)', () => {
    expect(eqv2DefaultGrid(6)).toEqual({ nBeta: 14, nAlpha: 15 });
    expect(eqv2DefaultGrid(6, 2)).toEqual({ nBeta: 14, nAlpha: 5 });
  });
});

describe('equivariance of activations on the sphere', () => {
  it('the gate is exactly equivariant', () => {
    const g = makeGrid(4, 10, 11);
    expect(activationError('gate', g, randomCoefs(4, 1), randomCoefs(4, 2), 0.3, R)).toBeLessThan(1e-12);
  });
  it('SiLU on the grid is not exact, and its error falls as the grid is refined', () => {
    const L = 4;
    const x = randomCoefs(L, 1, 2);
    const errs = [10, 16, 24, 40].map((n) => activationError('s2', makeGrid(L, n, n + 1), x, x, 0, R));
    expect(errs[0]!).toBeGreaterThan(1e-4);
    for (let k = 1; k < errs.length; k++) expect(errs[k]!).toBeLessThan(errs[k - 1]!);
    expect(errs[3]!).toBeLessThan(errs[0]! / 100);
  });
  it('the separable form only changes the degree-0 output', () => {
    const L = 3;
    const g = makeGrid(L, 12, 13);
    const x = randomCoefs(L, 8);
    const a = activationError('s2', g, x, x, 0, R);
    const b = activationError('sep-s2', g, x, x, 0.4, R);
    expect(a).toBeGreaterThan(1e-6);
    expect(b).toBeGreaterThan(1e-6);
  });
  it('SwiGLU-S² is exact once N_β ≥ 3L + 1 and N_α ≥ 3L + 1, and not one point before', () => {
    for (const L of [2, 3, 4, 6]) {
      const x1 = randomCoefs(L, 21, 1.5);
      const x2 = randomCoefs(L, 22, 1.5);
      const e = swigluExactGrid(L);
      expect(activationError('swiglu-s2', makeGrid(L, e.nBeta, e.nAlpha), x1, x2, 0.2, R)).toBeLessThan(1e-11);
      expect(activationError('swiglu-s2', makeGrid(L, e.nBeta - 2, e.nAlpha), x1, x2, 0.2, R)).toBeGreaterThan(1e-8);
      expect(activationError('swiglu-s2', makeGrid(L, e.nBeta, e.nAlpha - 1), x1, x2, 0.2, R)).toBeGreaterThan(1e-8);
    }
  });
  it('matches EquiformerV3’s grids: (8, 8) for L = 2 and (20, 20) for L = 6', () => {
    expect(swigluExactGrid(2)).toEqual({ nBeta: 8, nAlpha: 7 });
    expect(swigluExactGrid(6)).toEqual({ nBeta: 20, nAlpha: 19 });
  });
  it('with m_max = 2 in the edge frame, N_α = 7 suffices for rotations about the bond: V3’s (8, 20) attention grid', () => {
    const L = 6;
    const e = swigluExactGrid(L, 2);
    expect(e).toEqual({ nBeta: 20, nAlpha: 7 });
    const roll = axisRotation([0, 0, 1], 1.234);
    const x1 = randomCoefs(L, 31, 1, 2);
    const x2 = randomCoefs(L, 32, 1, 2);
    expect(activationError('swiglu-s2', makeGrid(L, 20, 8, 2), x1, x2, 0.1, roll)).toBeLessThan(1e-11);
    expect(activationError('swiglu-s2', makeGrid(L, 20, 6, 2), x1, x2, 0.1, roll)).toBeGreaterThan(1e-8);
  });
  it('SiLU on EquiformerV2’s 18 × 18 grid at L = 6 gives a small but nonzero error', () => {
    const L = 6;
    const x = randomCoefs(L, 41);
    const err = activationError('s2', makeGrid(L, 18, 18), x, x, 0, R);
    expect(err).toBeGreaterThan(1e-6);
    expect(err).toBeLessThan(0.05);
  });
});

describe('attention near the cutoff', () => {
  const t = makeToyAttention();
  const scene = attentionScene(5);
  const dists = scene.positions.map(dist);

  it('weights sum to one over neighbours inside the cutoff', () => {
    for (const mode of ['hard', 'message', 'smooth'] as const) {
      const w = attend(t, dists, mode).weights;
      expect(w.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    }
  });
  it('a hard cutoff and a message-only envelope both jump; the envelope inside the softmax does not', () => {
    const k = scene.mover;
    expect(Math.abs(cutoffJump(t, dists, k, 'hard'))).toBeGreaterThan(1e-3);
    expect(Math.abs(cutoffJump(t, dists, k, 'message'))).toBeGreaterThan(1e-3);
    expect(Math.abs(cutoffJump(t, dists, k, 'smooth'))).toBeLessThan(1e-9);
  });
  it('with the message-only envelope the jump is entirely the renormalisation of the other weights', () => {
    const k = scene.mover;
    const inside = dists.slice();
    inside[k] = t.rcut - 1e-9;
    const outside = dists.slice();
    outside[k] = t.rcut + 1e-9;
    const wIn = attend(t, inside, 'message').weights;
    const wOut = attend(t, outside, 'message').weights;
    // The mover takes a finite share at r_c, so every other weight drops by the factor (1 − a_mover).
    for (let j = 0; j < k; j++) expect(wIn[j]!).toBeCloseTo(wOut[j]! * (1 - wIn[k]!), 9);
    expect(wIn[k]!).toBeGreaterThan(1e-3);
  });
  it('LayerNorm makes the logits independent of the scale of the invariant features', () => {
    for (const s of [0.5, 3, 30]) {
      expect(logit(t, 3.1, { ln: true, scale: s })).toBeCloseTo(logit(t, 3.1, { ln: true, scale: 1 }), 3);
    }
    expect(Math.abs(logit(t, 3.1, { ln: false, scale: 30 }) - logit(t, 3.1, { ln: false }))).toBeGreaterThan(1);
  });
  it('without LayerNorm, large features collapse the attention onto one neighbour', () => {
    const big = attend(t, dists, 'hard', { ln: false, scale: 40 }).entropy;
    const normed = attend(t, dists, 'hard', { ln: true, scale: 40 }).entropy;
    expect(big).toBeLessThan(0.1);
    expect(normed).toBeGreaterThan(0.3);
  });
  it('LayerNorm output has zero mean and unit variance; SmoothLeakyReLU has slopes 0.2 and 1 at the extremes', () => {
    const f = layerNorm(Float64Array.from([1, 4, -2, 0.5]));
    expect(f.reduce((s, v) => s + v, 0)).toBeCloseTo(0, 12);
    expect(f.reduce((s, v) => s + v * v, 0) / 4).toBeCloseTo(1, 4);
    expect(smoothLeakyRelu(-50) / -50).toBeCloseTo(0.2, 10);
    expect(smoothLeakyRelu(50) / 50).toBeCloseTo(1, 10);
  });
});
