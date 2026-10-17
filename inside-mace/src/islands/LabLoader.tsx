// Lazy entry for the instrument: the page's first paint needs no Three.js and no experiment data; the
// island code, renderer and the selected experiment's manifest arrive as separate chunks (brief §15:
// lazy-load data). The experiment comes from the URL (`exp`, validated against index.json); an unknown
// id falls back to the first experiment and the lab reports the ignored value.
import { lazy, Suspense, useEffect, useState } from 'react';
import index from '../data/experiments/index.json';
import { parseManifest, type Manifest } from '../lib/experiment';
import './InterventionLab.css';

const InterventionLab = lazy(() => import('./InterventionLab'));
const MANIFESTS = import.meta.glob<{ default: unknown }>('../data/experiments/*/manifest.json');

export const EXPERIMENTS = index.experiments;
const DEFAULT_ID = EXPERIMENTS[0]!.id;

/** Short labels for the experiment picker; the full title comes from index.json. */
const SHORT: Record<string, string> = { 'si-bond-stretch': 'Si · bond stretch', 'zno-axial-bond': 'ZnO · axial Zn–O bond' };

export function experimentFromQuery(query: string): string {
  const id = new URLSearchParams(query).get('exp');
  return id !== null && EXPERIMENTS.some((e) => e.id === id) ? id : DEFAULT_ID;
}

/** URL for switching experiments: keeps mode, stage, view and presentation; the atom, frame and channel belong to the old experiment. */
export function switchQuery(query: string, id: string): string {
  const p = new URLSearchParams(query);
  for (const k of ['atom', 'frame', 'ch', 'layer']) p.delete(k);
  p.set('exp', id);
  return p.toString();
}

function Loading({ text }: { text: string }) {
  return (
    <div className="lab lab--loading" role="status">
      {text}
    </div>
  );
}

export default function LabLoader() {
  const [id, setId] = useState(() => (typeof window === 'undefined' ? DEFAULT_ID : experimentFromQuery(window.location.search.slice(1))));
  const [manifest, setManifest] = useState<{ id: string; m: Manifest } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setError(null);
    const load = MANIFESTS[`../data/experiments/${id}/manifest.json`];
    if (!load) {
      setError(`No data is committed for experiment “${id}”.`);
      return;
    }
    load()
      .then((mod) => live && setManifest({ id, m: parseManifest(mod.default) }))
      .catch((e: unknown) => live && setError(`The experiment data could not be loaded: ${e instanceof Error ? e.message : String(e)}`));
    return () => {
      live = false;
    };
  }, [id]);

  const choose = (next: string) => {
    if (next === id) return;
    const q = switchQuery(window.location.search.slice(1), next);
    window.history.replaceState(null, '', `${window.location.pathname}?${q}${window.location.hash}`);
    setId(next);
  };

  return (
    <>
      <div className="exp-pick" role="group" aria-label="Experiment">
        <span className="exp-pick__label">Experiment</span>
        {EXPERIMENTS.map((e) => (
          <button key={e.id} aria-pressed={e.id === id} className={e.id === id ? 'is-on' : ''} onClick={() => choose(e.id)} title={e.title}>
            <span>{SHORT[e.id] ?? e.title}</span>
            <span className="exp-pick__n mono">{e.n} atoms</span>
          </button>
        ))}
      </div>
      {error ? (
        <div className="lab lab--loading" role="alert">
          {error}
        </div>
      ) : manifest && manifest.id === id ? (
        <Suspense fallback={<Loading text="Loading the instrument…" />}>
          <InterventionLab key={id} manifest={manifest.m} />
        </Suspense>
      ) : (
        <Loading text="Loading the precomputed evaluations…" />
      )}
    </>
  );
}
