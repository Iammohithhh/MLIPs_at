import { describe, expect, it } from 'vitest';
import { A_CU, MORSE_CU, cleavageEnergy, coordination, crossPairs, fcc111, morse } from './cleavage';

const S = fcc111(A_CU, 4, 2, 8);

describe('rigid fcc(111) cleavage with a Morse potential (toy)', () => {
  it('builds an fcc(111) block whose interior atoms have 12 nearest neighbours', () => {
    expect(S.atoms).toHaveLength(4 * 2 * 2 * 8);
    const c = coordination(S, 0, 3.0);
    S.atoms.forEach((a, i) => {
      if (a.layer > 0 && a.layer < 7) expect(c[i]).toBe(12);
    });
  });

  it('opening the gap leaves 9-coordinated atoms on both new surfaces', () => {
    const c = coordination(S, 4, 3.0);
    S.atoms.forEach((a, i) => {
      if (a.layer === 3 || a.layer === 4) expect(c[i]).toBe(9);
    });
  });

  it('with nearest neighbours only, the cleavage energy is 3 broken bonds per surface atom', () => {
    const rc = 3.0; // between the first (2.556 Å) and second (3.615 Å) neighbour shells
    const pairs = crossPairs(S, rc);
    const perLayer = S.atoms.filter((a) => a.layer === 0).length;
    const expected = (1000 * 3 * perLayer * -morse(S.nn, MORSE_CU, rc)) / (S.cell[0] * S.cell[1]);
    expect(cleavageEnergy(S, pairs, 10, MORSE_CU, rc)).toBeCloseTo(expected, 8);
  });

  it('is zero at zero gap, rises monotonically and is flat once the gap exceeds the cutoff', () => {
    const rc = 7;
    const pairs = crossPairs(S, rc);
    expect(cleavageEnergy(S, pairs, 0, MORSE_CU, rc)).toBeCloseTo(0, 10);
    let prev = 0;
    for (let d = 0.25; d <= 6; d += 0.25) {
      const e = cleavageEnergy(S, pairs, d, MORSE_CU, rc);
      expect(e).toBeGreaterThan(prev - 1e-9);
      prev = e;
    }
    expect(cleavageEnergy(S, pairs, rc + 0.1, MORSE_CU, rc)).toBeCloseTo(cleavageEnergy(S, pairs, 20, MORSE_CU, rc), 10);
  });
});
