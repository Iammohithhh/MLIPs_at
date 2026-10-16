import { describe, expect, it } from 'vitest';
import zno from '~/data/structures/zno-wurtzite.json';
import type { Structure } from '~/lib/structure/types';
import { shellAround } from './three-body';
import { mulberry32, randomRotation } from '../chem/rng';
import {
  applyMat,
  axisRotation,
  collinearSwitch,
  det3,
  ecse,
  emptiestDirection,
  makeToyPet,
  pairFrame,
  petCutoff,
  petForward,
  rotateEnv,
  rotationSpread,
  withProbe,
  type EcseOptions,
  type Environment,
  type PetOptions,
} from './pet';

const PET: PetOptions = { rc: 4.5, width: 0.5 };
const ECSE: EcseOptions = { rin: 2.6, rinWidth: 0.5, omega: 0.05, dOmega: 0.1 };
const model = makeToyPet({ species: 3 });

const SPECIES: Record<string, number> = { Zn: 0, O: 1 };
const env: Environment = {
  centre: 0,
  neighbours: shellAround(zno as unknown as Structure, 4.5).map((n) => ({ v: [...n.p], s: SPECIES[n.symbol]! })),
};

describe('PET switching functions (App. F.1)', () => {
  it('f_c is 1 inside, 0 outside, monotone and flat at both ends', () => {
    expect(petCutoff(3.9, 4.5, 0.5)).toBe(1);
    expect(petCutoff(4.5, 4.5, 0.5)).toBe(0);
    expect(petCutoff(4.25, 4.5, 0.5)).toBeCloseTo(0.5, 12);
    let prev = 1;
    for (let r = 4.0; r <= 4.5; r += 0.005) {
      const f = petCutoff(r, 4.5, 0.5);
      expect(f).toBeLessThanOrEqual(prev + 1e-15);
      prev = f;
    }
    // C^∞ joins: the slope vanishes at both ends of the switching region.
    const h = 1e-4;
    expect(Math.abs(petCutoff(4.0 + h, 4.5, 0.5) - 1) / h).toBeLessThan(1e-6);
    expect(petCutoff(4.5 - h, 4.5, 0.5) / h).toBeLessThan(1e-6);
  });

  it('q_c1 switches from 0 to 1 between ω and ω + Δω', () => {
    expect(collinearSwitch(0.04, 0.05, 0.1)).toBe(0);
    expect(collinearSwitch(0.1, 0.05, 0.1)).toBeCloseTo(0.5, 12);
    expect(collinearSwitch(0.2, 0.05, 0.1)).toBe(1);
  });
});

describe('local frames', () => {
  it('are proper rotations whose first axis points at neighbour j', () => {
    const rand = mulberry32(3);
    for (let t = 0; t < 20; t++) {
      const a: [number, number, number] = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const b: [number, number, number] = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      const F = pairFrame(a, b);
      expect(det3(F)).toBeCloseTo(1, 12);
      const fa = applyMat(F, a);
      expect(Math.abs(fa[1])).toBeLessThan(1e-12);
      expect(Math.abs(fa[2])).toBeLessThan(1e-12);
      expect(fa[0]).toBeGreaterThan(0);
      // j′ lies in the frame's x–z plane.
      expect(Math.abs(applyMat(F, b)[1])).toBeLessThan(1e-12);
    }
  });
});

describe('the toy PET block', () => {
  it('uses the ZnO wurtzite environment within 4.5 Å', () => {
    expect(env.neighbours.length).toBeGreaterThan(10);
    expect(Math.hypot(...env.neighbours[0]!.v)).toBeCloseTo(1.98, 1);
  });

  it('is invariant to permuting neighbours', () => {
    const shuffled = { ...env, neighbours: [...env.neighbours].reverse() };
    expect(petForward(model, shuffled, PET).energy).toBeCloseTo(petForward(model, env, PET).energy, 12);
  });

  it('is not rotationally invariant: L acts on Cartesian components', () => {
    const { mean, std } = rotationSpread((e) => petForward(model, e, PET).energy, env, 50);
    expect(std / Math.abs(mean)).toBeGreaterThan(1e-3);
  });

  it('attention to an atom at the cutoff vanishes with PET weights but not with plain softmax', () => {
    const u = emptiestDirection(env, 3.5, 5.5);
    const at = withProbe(env, u, 4.5 - 1e-9, 2);
    const pet = petForward(model, at, PET);
    const plain = petForward(model, at, { ...PET, mode: 'plain' });
    expect(pet.attention.at(-1)!).toBeLessThan(1e-12);
    expect(plain.attention.at(-1)!).toBeGreaterThan(1e-3);
    const sum = pet.attention.reduce((a, b) => a + b, pet.selfAttention);
    expect(sum).toBeCloseTo(1, 12);
  });

  it('is continuous as an atom crosses the cutoff with PET attention, and jumps without it', () => {
    const u = emptiestDirection(env, 3.5, 5.5);
    const E = (d: number, mode: 'pet' | 'plain') => petForward(model, withProbe(env, u, d, 2), { ...PET, mode }).energy;
    const eps = 1e-7;
    expect(Math.abs(E(4.5 - eps, 'pet') - E(4.5 + eps, 'pet'))).toBeLessThan(1e-9);
    expect(Math.abs(E(4.5 - eps, 'plain') - E(4.5 + eps, 'plain'))).toBeGreaterThan(1e-3);
    // Outside the cutoff the probe is simply absent.
    expect(E(4.6, 'pet')).toBeCloseTo(petForward(model, env, PET).energy, 12);
  });

  it('keeps the probe path clear of the crystal', () => {
    const u = emptiestDirection(env, 3.5, 5.5);
    for (let d = 3.5; d <= 5.5; d += 0.1)
      for (const n of env.neighbours) {
        const p = withProbe(env, u, d, 2).neighbours.at(-1)!.v;
        expect(Math.hypot(p[0] - n.v[0], p[1] - n.v[1], p[2] - n.v[2])).toBeGreaterThan(1.2);
      }
  });
});

describe('ECSE symmetrisation', () => {
  it('makes the backbone exactly invariant to rotations', () => {
    const ref = ecse(model, env, PET, ECSE).energy;
    const rand = mulberry32(11);
    for (let t = 0; t < 8; t++) {
      const y = ecse(model, rotateEnv(env, randomRotation(rand)), PET, ECSE).energy;
      expect(Math.abs(y - ref)).toBeLessThan(1e-10 * Math.max(1, Math.abs(ref)));
    }
    for (const th of [0.3, 1.7, 4.0]) expect(ecse(model, rotateEnv(env, axisRotation([1, 2, 3], th)), PET, ECSE).energy).toBeCloseTo(ref, 10);
  });

  it('uses one frame per ordered pair of the four Zn–O bonds at r_in = 2.6 Å', () => {
    expect(ecse(model, env, PET, ECSE).frames).toBe(12);
  });

  it('is permutation invariant', () => {
    const shuffled = { ...env, neighbours: [...env.neighbours].reverse() };
    expect(ecse(model, shuffled, PET, ECSE).energy).toBeCloseTo(ecse(model, env, PET, ECSE).energy, 10);
  });

  it('is continuous with smooth weights and jumps with a plain average when a frame atom crosses r_in', () => {
    // Stretch the first O neighbour radially through r_in.
    const stretch = (d: number): Environment => ({
      ...env,
      neighbours: env.neighbours.map((n, k) => {
        if (k !== 0) return n;
        const r = Math.hypot(...n.v);
        return { ...n, v: [(n.v[0] * d) / r, (n.v[1] * d) / r, (n.v[2] * d) / r] };
      }),
    });
    const eps = 1e-9;
    const y = (d: number, weights: 'smooth' | 'plain') => ecse(model, stretch(d), PET, { ...ECSE, weights }).energy;
    expect(Math.abs(y(2.6 - eps, 'smooth') - y(2.6 + eps, 'smooth'))).toBeLessThan(1e-7);
    expect(Math.abs(y(2.6 - eps, 'plain') - y(2.6 + eps, 'plain'))).toBeGreaterThan(1e-4);
  });
});
