// Every interactive, figure and result on the Atlas carries one of these labels.
// CLAUDE.md lists the five result statuses; the vision (§18) adds evidence grades
// for the Failure Atlas, and two provenance labels keep illustration and
// electronic-structure output from being mistaken for MLIP predictions.

export type Tone = 'arch' | 'phys' | 'lit' | 'caveat' | 'fail' | 'verified' | 'neutral';

export type LabelGroup = 'status' | 'evidence' | 'provenance';

export interface LabelDef {
  /** Uppercase text shown in the badge. */
  text: string;
  group: LabelGroup;
  tone: Tone;
  /** One-sentence definition, shown in tooltips and on the Method page. */
  definition: string;
}

export const LABELS = {
  'toy-model': {
    text: 'TOY MODEL',
    group: 'status',
    tone: 'caveat',
    definition:
      'An explicitly defined, simplified model evaluated in your browser. It illustrates a mechanism; it is not the prediction of any real MLIP or electronic-structure calculation.',
  },
  'live-calculation': {
    text: 'LIVE CALCULATION',
    group: 'status',
    tone: 'phys',
    definition:
      'Computed now from the inputs on screen by a real model or solver — in your browser or on the Atlas CPU inference server — with the exact settings and reproduction code shown.',
  },
  precomputed: {
    text: 'PRECOMPUTED',
    group: 'status',
    tone: 'arch',
    definition:
      'Computed offline by the Atlas pipeline with a pinned code version, checkpoint and settings; a provenance manifest is attached to the data.',
  },
  'published-result': {
    text: 'PUBLISHED RESULT',
    group: 'status',
    tone: 'lit',
    definition:
      'Taken from a published paper or benchmark with a figure, table or section locator. Not re-run by the Atlas.',
  },
  reproduced: {
    text: 'REPRODUCED',
    group: 'status',
    tone: 'verified',
    definition:
      'A published result that the Atlas re-ran itself and matched; both the publication and the Atlas run are linked.',
  },
  theory: {
    text: 'THEORY',
    group: 'evidence',
    tone: 'arch',
    definition: 'Follows from a derivation or an exact property of the model, shown or cited on the page.',
  },
  'toy-demo': {
    text: 'TOY DEMO',
    group: 'evidence',
    tone: 'caveat',
    definition: 'Demonstrated only in a simplified toy model; suggests a mechanism, does not establish it for real systems.',
  },
  hypothesis: {
    text: 'HYPOTHESIS',
    group: 'evidence',
    tone: 'caveat',
    definition: 'A plausible mechanism or investigation not yet supported by published or Atlas calculations.',
  },
  schematic: {
    text: 'SCHEMATIC',
    group: 'provenance',
    tone: 'neutral',
    definition: 'An illustration of a concept. Positions, magnitudes or interactions are not computed and not to scale.',
  },
  'dft-reference': {
    text: 'DFT REFERENCE',
    group: 'provenance',
    tone: 'lit',
    definition:
      'An electronic-structure (e.g. DFT) quantity on a named structure. Energy/force MLIPs do not predict it unless the model explicitly outputs it.',
  },
} as const satisfies Record<string, LabelDef>;

export type LabelKey = keyof typeof LABELS;

export const LABEL_KEYS = Object.keys(LABELS) as LabelKey[];

export function labelsInGroup(group: LabelGroup): LabelKey[] {
  return LABEL_KEYS.filter((k) => LABELS[k].group === group);
}
