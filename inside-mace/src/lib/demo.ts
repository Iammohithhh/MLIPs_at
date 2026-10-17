// The scripted demonstration sequence for a short screen recording (brief §14). Each step is a fully
// specified, reproducible instrument state on evaluated frames; captions quote numbers computed from the
// manifest, never typed in by hand. Pure module, covered by demo.test.ts.
import { axialForce, energyChange, type Manifest } from './experiment';
import type { ExperimentState } from './urlState';

export interface DemoStep {
  n: number;
  title: string;
  caption: string;
  /** Caption when the screen is too narrow for the side-by-side view (one scene on mobile). */
  captionNarrow?: string;
  state: Partial<ExperimentState>;
  colour: 'element' | 'change';
}

function signed(x: number | null, digits: number, unit: string): string {
  if (x === null) return 'not available';
  return `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(digits)} ${unit}`;
}

export function demoSteps(m: Manifest): DemoStep[] {
  const B = m.displacement.baselineIndex;
  const last = m.frames.length - 1;
  const d = m.displacement.values[last]!;
  const atom = m.displacement.atom;
  const layers = m.baseline.norms.length;
  return [
    {
      n: 1,
      title: 'The structure and its predicted forces',
      caption: `A periodic silicon crystal (${m.structure.n} atoms) evaluated by a real MACE checkpoint. In the perfect crystal every predicted force is zero by symmetry, so no arrows are drawn.`,
      state: { frame: B, view: 'perturbed', atom, mode: 'lab' },
      colour: 'element',
    },
    {
      n: 2,
      title: 'Move one atom along the evaluated path',
      caption: `Atom ${atom} moved ${Math.abs(d).toFixed(3)} Å along ${m.displacement.axisLabel} (evaluated frame ${last + 1} of ${m.frames.length}). The force on it along the path is now ${signed(axialForce(m, last), 2, 'eV/Å')}: it pushes back.`,
      state: { frame: last, view: 'perturbed', atom, mode: 'lab' },
      colour: 'element',
    },
    {
      n: 3,
      title: 'What changes inside the model',
      caption: `Amber shows how much each atom’s layer-${layers} scalar features changed (‖Δh‖, fixed scale over all frames). Observed changes in a fixed model, not an explanation of its reasoning.`,
      state: { frame: last, view: 'perturbed', atom, layer: layers - 1, mode: 'lab' },
      colour: 'change',
    },
    {
      n: 4,
      title: 'Energy and force comparison',
      caption: `Baseline beside the moved state, cameras linked. Total energy change ΔE = ${signed(energyChange(m, last), 3, 'eV')}; force on the moved atom along the path ${signed(axialForce(m, last), 2, 'eV/Å')} (F = −∂E/∂r).`,
      captionNarrow: `The moved state (on a narrow screen one scene is shown; tap Baseline to compare). Total energy change ΔE = ${signed(energyChange(m, last), 3, 'eV')}; force on the moved atom along the path ${signed(axialForce(m, last), 2, 'eV/Å')} (F = −∂E/∂r).`,
      state: { frame: last, view: 'split', atom, mode: 'lab' },
      colour: 'element',
    },
    {
      n: 5,
      title: 'Try the same experiment',
      caption: 'Move one atom. Inspect what changes inside a real MACE checkpoint. Every number here is a precomputed model evaluation you can open and check.',
      state: { frame: last, view: 'difference', atom, mode: 'lab' },
      colour: 'element',
    },
  ];
}
