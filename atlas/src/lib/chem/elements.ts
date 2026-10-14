// Element data for the 3D viewer. Colours are a CPK/Jmol-derived palette tuned
// for contrast on both themes; radii are *display* radii in ångström, chosen for
// legibility (ball-and-stick for molecules, near-ionic for ionic crystals). They
// are not physical radii and are never used in any calculation.

export type El = 'H' | 'O' | 'Na' | 'Cl' | 'Au';

export interface ElementStyle {
  name: string;
  /** sRGB hex colour. */
  color: string;
  /** Display radius in ball-and-stick view (Å). */
  radius: number;
  /** Display radius in space-filling-like view of ionic crystals (Å). */
  ionicRadius: number;
  /** 0 = dielectric shading, 1 = metallic shading. */
  metal: number;
}

export const ELEMENTS: Record<El, ElementStyle> = {
  H: { name: 'Hydrogen', color: '#eef1f5', radius: 0.3, ionicRadius: 0.3, metal: 0 },
  O: { name: 'Oxygen', color: '#e2483d', radius: 0.58, ionicRadius: 0.58, metal: 0 },
  Na: { name: 'Sodium', color: '#8f5fe8', radius: 0.92, ionicRadius: 0.82, metal: 0 },
  Cl: { name: 'Chlorine', color: '#2fb866', radius: 1.0, ionicRadius: 1.25, metal: 0 },
  Au: { name: 'Gold', color: '#e3ad35', radius: 1.2, ionicRadius: 1.2, metal: 1 },
};

/** e²/(4πε₀) in eV·Å (CODATA 2018 constants). */
export const COULOMB_K = 14.399645;
