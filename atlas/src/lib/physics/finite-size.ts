// What a periodic interface cell of lateral size L can represent (Failure Atlas: small → large cells).
// THEORY: exact consequences of putting whole ions and whole electrons into a cell of area A = L².
//  - one ion pair in an electrolyte slab of thickness h is a concentration 1/(N_A·A·h);
//  - one electron on the electrode is a surface charge e/A;
//  - moving one electron across a double layer of thickness d and permittivity ε_r shifts the potential by
//    e·d/(ε₀ε_r A) (the capacitor relation used in the discharge lab).
// Cost estimates assume conventional DFT scaling ∝ N³ in the number of atoms; that is an assumption.

import { shiftPerElectron } from './newns-anderson';

/** Avogadro constant (exact, SI 2019). */
export const NA = 6.02214076e23;
/** Elementary charge per Å² expressed in μC/cm² (1.602176634e-19 C / 1e-16 cm²). */
export const E_PER_A2_UC_CM2 = 1602.176634;

/** Concentration (mol/L) of one ion pair in an electrolyte slab of area L² (Å²) and thickness h (Å). */
export const concentrationPerPair = (L: number, h: number) => 1 / (NA * L * L * h * 1e-27);

/** Surface charge density (μC/cm²) of one elementary charge on area L². */
export const sigmaPerElectron = (L: number) => E_PER_A2_UC_CM2 / (L * L);

/** Potential shift (V) per electron moved across the double layer of a cell of side L. */
export const potentialPerElectron = (L: number, d = 3, er = 6) => shiftPerElectron(L * L, d, er);

/** Smallest cell side (Å) for which one ion pair is at most concentration c (mol/L). */
export const sideForConcentration = (c: number, h: number) => Math.sqrt(1 / (NA * c * h * 1e-27));

/** Smallest cell side (Å) for which one transferred electron shifts the potential by at most dU (V). */
export const sideForPotential = (dU: number, d = 3, er = 6) => Math.sqrt(potentialPerElectron(1, d, er) / dU);

/** Atoms in an Au(111)/water cell: `layers` metal layers plus water of thickness h at 0.0334 molecules/Å³. */
export function atomCount(L: number, h: number, layers = 4) {
  const auPerA2 = 1 / ((Math.sqrt(3) / 2) * (4.078 / Math.SQRT2) ** 2);
  const metal = Math.round(layers * auPerA2 * L * L);
  const water = Math.round(0.0334 * L * L * h);
  return { metal, water, total: metal + 3 * water };
}

/** Cost relative to a reference cell, assuming cubic scaling with atom count. */
export const relativeCost = (n: number, nRef: number) => (n / nRef) ** 3;
