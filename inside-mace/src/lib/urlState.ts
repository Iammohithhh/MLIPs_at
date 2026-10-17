// Shareable experiment state ↔ URL query string. Every value is validated against the loaded
// manifest; unknown, obsolete or out-of-range values fall back to defaults and are reported,
// so a stale link never shows a state that was not evaluated.

import type { Manifest } from './experiment';

export const STAGES = ['species', 'neighbours', 'radial', 'messages', 'features', 'energy', 'forces'] as const;
export type Stage = (typeof STAGES)[number];

/** 'split' = baseline and perturbed side by side with synchronised cameras (desktop; one scene on narrow screens). */
export const VIEWS = ['baseline', 'perturbed', 'difference', 'split'] as const;
export type View = (typeof VIEWS)[number];

export const MODES = ['lab', 'walk'] as const;
export type Mode = (typeof MODES)[number];

export interface ExperimentState {
  /** Data schema version the state was made with. */
  v: number;
  experiment: string;
  checkpoint: string;
  mode: Mode;
  atom: number;
  frame: number;
  stage: Stage;
  layer: number;
  channel: number;
  view: View;
}

export const CHANNELS = 128;

export function defaultState(m: Manifest, checkpoint: string): ExperimentState {
  return {
    v: m.schema,
    experiment: m.id,
    checkpoint,
    mode: 'lab',
    atom: m.displacement.atom,
    frame: m.displacement.baselineIndex,
    stage: 'species',
    layer: 0,
    channel: 0,
    view: 'perturbed',
  };
}

export function encodeState(s: ExperimentState): string {
  const p = new URLSearchParams();
  p.set('v', String(s.v));
  p.set('exp', s.experiment);
  p.set('ckpt', s.checkpoint);
  p.set('mode', s.mode);
  p.set('atom', String(s.atom));
  p.set('frame', String(s.frame));
  p.set('stage', s.stage);
  p.set('layer', String(s.layer));
  p.set('ch', String(s.channel));
  p.set('view', s.view);
  return p.toString();
}

function int(raw: string | null): number | null {
  if (raw === null || !/^-?\d+$/.test(raw)) return null;
  return Number(raw);
}

export interface DecodeResult {
  state: ExperimentState;
  /** Human-readable notes about values that were ignored. Empty when the link restored exactly. */
  issues: string[];
}

export function decodeState(query: string, m: Manifest, checkpoint: string): DecodeResult {
  const p = new URLSearchParams(query);
  const s = defaultState(m, checkpoint);
  const issues: string[] = [];
  if (![...p.keys()].length) return { state: s, issues };

  const v = int(p.get('v'));
  if (v !== null && v !== m.schema) issues.push(`link made for data version ${v}; showing version ${m.schema}`);
  const exp = p.get('exp');
  if (exp !== null && exp !== m.id) {
    issues.push(`experiment “${exp}” is not available`);
    return { state: s, issues };
  }
  const ck = p.get('ckpt');
  if (ck !== null && ck !== checkpoint) {
    issues.push(`checkpoint “${ck}” is not the one evaluated here (${checkpoint})`);
    return { state: s, issues };
  }

  const mode = p.get('mode');
  if (mode !== null) (MODES as readonly string[]).includes(mode) ? (s.mode = mode as Mode) : issues.push(`unknown mode “${mode}”`);

  const atom = int(p.get('atom'));
  if (p.has('atom')) atom !== null && atom >= 0 && atom < m.structure.n ? (s.atom = atom) : issues.push('atom index out of range');

  const frame = int(p.get('frame'));
  if (p.has('frame')) {
    if (frame === null || frame < 0 || frame >= m.frames.length) issues.push('displacement frame out of range');
    else if (m.frames[frame]?.status !== 'ok') issues.push(`frame ${frame} failed to evaluate; showing the baseline`);
    else s.frame = frame;
  }

  const stage = p.get('stage');
  if (stage !== null) (STAGES as readonly string[]).includes(stage) ? (s.stage = stage as Stage) : issues.push(`unknown stage “${stage}”`);

  const layers = m.baseline.norms.length;
  const layer = int(p.get('layer'));
  if (p.has('layer')) layer !== null && layer >= 0 && layer < layers ? (s.layer = layer) : issues.push('layer out of range');

  const ch = int(p.get('ch'));
  if (p.has('ch')) ch !== null && ch >= 0 && ch < CHANNELS ? (s.channel = ch) : issues.push('channel out of range');

  const view = p.get('view');
  if (view !== null) (VIEWS as readonly string[]).includes(view) ? (s.view = view as View) : issues.push(`unknown view “${view}”`);

  return { state: s, issues };
}
