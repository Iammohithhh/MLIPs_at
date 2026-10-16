import { useEffect, useMemo, useState } from 'react';
import './PhysicsLab.css';
import { Stage } from '~/three/Stage';
import { AtomPicker, ImpostorAtoms, Lines, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';

// The literature around the Atlas papers, from OpenAlex (CC0): every Atlas paper, what it cites, and the
// most-cited works that cite it. Layout precomputed by pipeline/atlas_pipeline/citations.py.

interface Work {
  id: string;
  title: string;
  year: number | null;
  cites: number;
  field: string | null;
  atlas: string | null;
  doi: string | null;
  p: [number, number, number];
}
interface Universe {
  source: { retrieved: string };
  nodes: Work[];
  edges: Array<[number, number]>;
}

const SCALE = 70;
let cache: Promise<Universe> | null = null;
const load = () => (cache ??= fetch('/data/citation-universe.json').then((r) => r.json() as Promise<Universe>));

export default function CitationUniverse() {
  const theme = useSceneTheme();
  const [u, setU] = useState<Universe | null>(null);
  const [failed, setFailed] = useState(false);
  const [sel, setSel] = useState<number | null>(null);
  const [year, setYear] = useState(2026);
  const [colour, setColour] = useState<'year' | 'field'>('year');
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    load().then(setU, () => setFailed(true));
  }, []);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setYear((y) => (y >= 2026 ? (setPlaying(false), 2026) : y + 1)), 120);
    return () => clearInterval(t);
  }, [playing]);

  if (!u) return <p className="lab-note">{failed ? 'The citation data could not be loaded.' : 'Loading the citation universe…'}</p>;
  const shown = u.nodes.filter((n) => (n.year ?? 0) <= year).length;
  const s = sel != null ? u.nodes[sel]! : null;
  const deg = sel != null ? u.edges.filter(([a, b]) => a === sel || b === sel) : [];
  const fields = fieldOrder(u);

  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage" style={{ minHeight: 540 }}>
          <Stage view={{ position: [125, -215, 105], target: [0, 0, 0] }} background={theme.canvas} label={`Citation universe: ${shown} works published by ${year}.`} minDistance={20} maxDistance={400}>
            <Scene theme={theme} u={u} sel={sel} setSel={setSel} year={year} colour={colour} fields={fields} />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Colour by">
              <button type="button" aria-pressed={colour === 'year'} onClick={() => setColour('year')}>Colour by year</button>
              <button type="button" aria-pressed={colour === 'field'} onClick={() => setColour('field')}>by field</button>
            </div>
            <button type="button" className="lab-btn" aria-pressed={playing} onClick={() => { if (!playing) setYear(1950); setPlaying(!playing); }}>
              {playing ? 'Pause' : 'Grow from 1950'}
            </button>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li><span className="dot" style={{ background: theme.ink }} /> Atlas paper · size: times cited</li>
            {colour === 'field'
              ? fieldOrder(u)
                  .slice(0, 6)
                  .map((f) => (
                    <li key={f}>
                      <span className="dot" style={{ background: FIELD_COLOURS(theme)[fields.indexOf(f) % 7] }} /> {f}
                    </li>
                  ))
              : [<li key="y"><span className="bar" style={{ background: `linear-gradient(90deg, ${theme.arch}, ${theme.phys}, ${theme.caveat})` }} /> 1960 → 2026</li>]}
          </ul>
        </div>
        <aside className="lab-panel" style={{ margin: 0, alignContent: 'start' }} aria-live="polite">
          <div className="lab-controls">
            <label>
              <span>published by</span>
              <input type="range" min={1950} max={2026} step={1} value={year} onChange={(e) => { setPlaying(false); setYear(Number(e.target.value)); }} />
              <output>{year}</output>
            </label>
          </div>
          <dl className="lab-readout">
            <div><dt>Works shown</dt><dd>{shown} of {u.nodes.length}</dd></div>
            <div><dt>Citations among them</dt><dd>{u.edges.length.toLocaleString('en-US')}</dd></div>
          </dl>
          {s ? (
            <div>
              <p className="eyebrow">{s.year ?? '—'} · cited {s.cites.toLocaleString('en-US')} times · {s.field ?? 'field unknown'}</p>
              <h3 style={{ margin: '0.3rem 0 0.5rem', font: '600 1.08rem/1.35 var(--font-serif)' }}>
                {s.atlas ? <a href={`/papers/#paper-${s.atlas}`}>{s.title}</a> : s.doi ? <a href={s.doi}>{s.title}</a> : s.title}
              </h3>
              <p className="lab-note">
                {s.atlas ? 'An Atlas paper. ' : ''}
                {deg.length} citation link{deg.length === 1 ? '' : 's'} inside this universe (highlighted).{' '}
                <a href={`https://openalex.org/${s.id}`}>OpenAlex record</a>
              </p>
            </div>
          ) : (
            <p className="lab-note">
              The {u.nodes.filter((n) => n.atlas).length} papers the Atlas cites, every work they cite, and the most-cited works that cite
              them; position follows the citation network, so neighbours cite each other. Click a star for its links, or grow the
              field year by year.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}

const FIELD_COLOURS = (t: SceneTheme) => [t.phys, t.arch, t.lit, t.caveat, t.fail, t.verified, t.inkMuted];

/** Fields by number of works, most common first. */
function fieldOrder(u: Universe): string[] {
  const c = new Map<string, number>();
  for (const n of u.nodes) c.set(n.field ?? 'Unknown', (c.get(n.field ?? 'Unknown') ?? 0) + 1);
  return [...c.entries()].sort((a, b) => b[1] - a[1]).map(([f]) => f);
}

function yearColour(t: SceneTheme, y: number | null): [number, number, number] {
  const a = linear(t.arch);
  const b = linear(t.phys);
  const c = linear(t.caveat);
  const f = Math.min(1, Math.max(0, ((y ?? 2000) - 1960) / (2026 - 1960)));
  return f < 0.6 ? mixRGB(a, b, f / 0.6) : mixRGB(b, c, (f - 0.6) / 0.4);
}

function Scene({ theme, u, sel, setSel, year, colour, fields }: { theme: SceneTheme; u: Universe; sel: number | null; setSel: (i: number | null) => void; year: number; colour: 'year' | 'field'; fields: string[] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 200, 700);
  const palette = useMemo(() => FIELD_COLOURS(theme).map(linear), [theme]);
  const linked = useMemo(() => new Set(sel == null ? [] : u.edges.flatMap(([a, b]) => (a === sel ? [b] : b === sel ? [a] : []))), [u, sel]);
  const buffers = useMemo<AtomBuffers>(() => {
    const n = u.nodes.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const bg = linear(theme.canvas);
    u.nodes.forEach((w, i) => {
      pos.set([w.p[0] * SCALE, w.p[1] * SCALE, w.p[2] * SCALE], 3 * i);
      const on = (w.year ?? 0) <= year;
      radii[i] = on ? (w.atlas ? 1.6 : 0.35 + 0.22 * Math.log10(1 + w.cites)) * (sel === i ? 1.5 : 1) : 0.001;
      const c = colour === 'year' ? yearColour(theme, w.year) : palette[fields.indexOf(w.field ?? 'Unknown') % palette.length]!;
      const dim = sel != null && sel !== i && !linked.has(i);
      colors.set(w.atlas ? linear(theme.ink) : dim ? mixRGB(c, bg, 0.75) : c, 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
  }, [u, year, colour, sel, linked, theme, palette, fields]);
  const web = useMemo<ColoredLine[]>(() => {
    // Every citation, very faint: the web is what gives the cloud its shape.
    if (sel != null) return [];
    const c = linear(theme.inkMuted);
    return u.edges
      .filter(([a, b]) => (u.nodes[a]!.year ?? 0) <= year && (u.nodes[b]!.year ?? 0) <= year)
      .map(([a, b]) => ({
        a: u.nodes[a]!.p.map((v) => v * SCALE) as [number, number, number],
        b: u.nodes[b]!.p.map((v) => v * SCALE) as [number, number, number],
        rgba: [c[0], c[1], c[2], 0.045],
      }));
  }, [u, sel, year, theme.inkMuted]);
  const lines = useMemo<ColoredLine[]>(() => {
    if (sel == null) return [];
    const c = linear(theme.ink);
    return u.edges
      .filter(([a, b]) => (a === sel || b === sel) && (u.nodes[a]!.year ?? 0) <= year && (u.nodes[b]!.year ?? 0) <= year)
      .map(([a, b]) => ({
        a: u.nodes[a]!.p.map((v) => v * SCALE) as [number, number, number],
        b: u.nodes[b]!.p.map((v) => v * SCALE) as [number, number, number],
        rgba: [c[0], c[1], c[2], 0.55],
      }));
  }, [u, sel, year, theme.ink]);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={web} />
      <Lines lines={lines} />
      <AtomPicker positions={buffers.positions} radii={buffers.radii} count={buffers.count} onPick={(i) => setSel(i)} />
    </>
  );
}
