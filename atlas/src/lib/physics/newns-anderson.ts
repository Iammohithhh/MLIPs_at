// Ion discharge at an electrode in the Newns–Anderson (Gurney) picture (Electrochemistry Lab, panel 4).
// TOY MODEL — a model Hamiltonian, not an electronic-structure calculation. A cation's empty
// neutralisation level sits at −I below the vacuum level far from the metal; near a conductor its image
// shifts it up by k/(4z) and coupling to the metal broadens it into a Lorentzian of half-width Δ(z).
// In the wide-band limit its occupation is n = ½ − (1/π)·arctan((ε − E_F)/Δ). The electrode's Fermi
// level is either held fixed (constant potential) or moves as the metal gives up the transferred
// charge through a Helmholtz capacitance C = ε₀ε_r A/d (constant charge: a finite cell, no potentiostat).

import { COULOMB_K } from '../chem/elements';
import { EPS0 } from './slab-dipole';

export interface DischargeParams {
  /** Ionisation energy of the neutral atom (eV): the ion's level is −I far away. */
  I: number;
  /** Work function of the uncharged electrode (eV): E_F = −W at zero charge. */
  W: number;
  /** Level half-width at z0 (eV) and its decay length (Å). */
  delta0: number;
  z0: number;
  lambda: number;
  /** Electrode potential relative to the potential of zero charge (V); positive lowers E_F. */
  phi: number;
}

export const DEFAULT_DISCHARGE: DischargeParams = { I: 5.0, W: 5.3, delta0: 1.2, z0: 2.0, lambda: 0.9, phi: 0 };

/** Level position (eV, vacuum = 0) at height z (Å) above the image plane. */
export const levelAt = (z: number, p: DischargeParams) => -p.I + COULOMB_K / (4 * z);

/** Lorentzian half-width (eV) at height z. */
export const widthAt = (z: number, p: DischargeParams) => p.delta0 * Math.exp(-(z - p.z0) / p.lambda);

/** Wide-band-limit occupation of a level ε with half-width Δ against Fermi level E_F. */
export const occupation = (eps: number, delta: number, ef: number) => 0.5 - Math.atan((eps - ef) / delta) / Math.PI;

/** Fermi-level shift per transferred electron (eV) for a Helmholtz layer of area A (Å²), thickness d (Å), permittivity εr. */
export const shiftPerElectron = (area: number, d = 3, er = 6) => d / (EPS0 * er * area);

export interface DischargeState {
  z: number;
  eps: number;
  delta: number;
  /** Fermi level after any shift (eV). */
  ef: number;
  /** Electrons on the ion's level, and the ion's charge 1 − n (e). */
  n: number;
  charge: number;
}

/**
 * Self-consistent occupation. With constant potential, E_F = −W − eφ. With constant charge the metal loses
 * n electrons to the ion, which lowers its Fermi level by n·s (s = e²/C per electron); n is then the fixed
 * point of n = f(ε, Δ, E_F(n)), found by bisection (the right side decreases with n, so it is unique).
 */
export function discharge(z: number, p: DischargeParams, mode: 'potential' | 'charge', area = 100): DischargeState {
  const eps = levelAt(z, p);
  const delta = widthAt(z, p);
  const ef0 = -p.W - p.phi;
  if (mode === 'potential') {
    const n = occupation(eps, delta, ef0);
    return { z, eps, delta, ef: ef0, n, charge: 1 - n };
  }
  const s = shiftPerElectron(area);
  let lo = 0;
  let hi = 1;
  for (let k = 0; k < 60; k++) {
    const mid = (lo + hi) / 2;
    if (mid - occupation(eps, delta, ef0 - s * mid) < 0) lo = mid;
    else hi = mid;
  }
  const n = (lo + hi) / 2;
  return { z, eps, delta, ef: ef0 - s * n, n, charge: 1 - n };
}

/** Lorentzian density of states of the level (1/eV). */
export const levelDos = (e: number, eps: number, delta: number) => delta / Math.PI / ((e - eps) ** 2 + delta * delta);
