// Text block for a still-image export of the current instrument state (brief §14): descriptive title,
// experiment settings with units, computation status, legends, attribution and the source URL. Pure and
// tested; every number comes from the selected evaluated frame.
import { axialForce, energyChange, okFrameAt, type Manifest } from './experiment';
import type { ExperimentState } from './urlState';
import { ELEMENT_DISPLAY, composition, experimentInfo } from './experimentInfo';

export interface FigureText {
  title: string;
  status: string;
  settings: string[];
  legend: string[];
  attribution: string;
  source: string;
  /** Labels for side-by-side panes, left to right (empty for a single scene). */
  panes: string[];
}

export function figureText(m: Manifest, s: ExperimentState, colour: 'element' | 'change', forceScale: number, url: string): FigureText {
  const d = m.displacement.values[s.frame] ?? NaN;
  const ok = okFrameAt(m, s.frame) !== null;
  const sign = (x: number) => (x >= 0 ? '+' : '−');
  const dE = energyChange(m, s.frame);
  const fa = axialForce(m, s.frame);
  const viewLabel = { baseline: 'baseline (d = 0)', perturbed: 'perturbed', difference: 'difference: ΔF arrows', split: 'baseline | perturbed, same camera' }[s.view];
  const p = m.provenance;
  const info = experimentInfo(m.id);
  const movedSymbol = m.structure.symbols[m.displacement.atom] ?? '';
  const elementLegend = composition(m)
    .map(({ symbol }) => `${symbol} ${ELEMENT_DISPLAY[symbol]?.colorName ?? 'grey'}`)
    .join(', ');
  return {
    title: `Moving one ${movedSymbol} atom ${sign(d)}${Math.abs(d).toFixed(3)} Å: what a real MACE checkpoint predicts`,
    status: 'PRECOMPUTED model evaluation (not live), interactive inspection',
    settings: [
      `${String(p.checkpoint)} · sha256 ${String(p.sha256).slice(0, 16)}… · mace-torch ${String(p['mace-torch'])} · ${String(p.dtype)} on ${String(p.device)}`,
      `${info?.structure ?? m.title} ${m.structure.supercell.join('×')} supercell, ${m.structure.n} atoms, periodic; atom ${m.displacement.atom} moved along ${m.displacement.axisLabel}`,
      `Frame ${s.frame + 1}/${m.frames.length}, d = ${sign(d)}${Math.abs(d).toFixed(3)} Å · view: ${viewLabel}`,
      ok && dE !== null && fa !== null
        ? `ΔE total = ${sign(dE)}${Math.abs(dE).toFixed(4)} eV · F·û on moved atom = ${sign(fa)}${Math.abs(fa).toFixed(3)} eV/Å`
        : 'This frame has no evaluated outputs.',
    ],
    legend: [
      `${s.view === 'difference' ? 'Force-change arrows ΔF = F(d) − F(0) (amber)' : 'Force arrows'}: ${forceScale} Å of length per eV/Å (scene units); arrows under 0.12 eV/Å hidden`,
      colour === 'change'
        ? `Atom colour: ‖Δh‖ of layer ${s.layer + 1} scalar features (128 channels), grey = 0 to amber = largest over all frames; moved atom in mint`
        : `Atom colour: ${elementLegend}, moved atom mint · thin lines: model graph edges (r < ${m.rMax} Å), not bonds · sticks: ${info?.bondLabel ?? 'bonds < 2.6 Å'}`,
    ],
    attribution: `MACE-MP-0a medium (${String(p.licence)}), Batatia et al.; trained on ${String(p.trainingData)}. Observed changes in a fixed model, not error estimates.`,
    source: url,
    panes: s.view === 'split' ? ['Baseline · d = 0', `Perturbed · d = ${sign(d)}${Math.abs(d).toFixed(3)} Å`] : [],
  };
}
