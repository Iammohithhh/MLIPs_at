import { useEffect, useId, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import '../PhysicsLab.css';
import './AtomViewer.css';
import { Stage, type StageView } from '~/three/Stage';
import {
  Arrows,
  AtomPicker,
  ImpostorAtoms,
  Lines,
  Shell,
  Sticks,
  linear,
  mixRGB,
  useShadingUniforms,
  type AtomBuffers,
  type ColoredLine,
  type Segment,
} from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { elementColor } from '~/lib/structure/elements';
import { cartToFrac, cellParameters, displayBonds, vecMat } from '~/lib/structure/geometry';
import type { Structure, Vec3 } from '~/lib/structure/types';
import { useLinkedSymbol } from '~/lib/useLinkedSymbol';
import { parseStructureFile } from '~/lib/structure/parsers';
import { DEFAULT_OPTIONS, buildModel, measure, type DisplayModel, type DrawnAtom, type ViewerOptions } from './model';

// The general-purpose atomic viewer: any Atlas structure, periodic or not, with
// replicas, boundary images, display bonds, cutoff shells (including periodic
// images outside the drawn cells), vector layers and distance/angle/dihedral
// measurement. Rendering uses the shared impostor engine in src/three/.

const METALS = new Set(['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Ti', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ag', 'Au', 'Pt', 'Pd', 'Ir', 'Rh', 'Ru', 'Os', 'W', 'Mo']);

type Preset = 'iso' | 'a' | 'b' | 'c';

function viewFor(model: DisplayModel, s: Structure, preset: Preset, fov = 30): StageView {
  let dir: Vec3 = [0.55, -0.72, 0.42];
  if (s.cell && preset !== 'iso') {
    const k = preset === 'a' ? 0 : preset === 'b' ? 1 : 2;
    const v = s.cell[k]!;
    const n = Math.hypot(...v);
    dir = [v[0] / n, v[1] / n, v[2] / n];
    // Looking straight down z with a z-up camera is degenerate; tilt slightly.
    if (Math.abs(dir[2]) > 0.98) dir = [0.02, -0.06, dir[2] > 0 ? 1 : -1];
  }
  const n = Math.hypot(...dir);
  const dist = (model.extent / Math.tan(((fov / 2) * Math.PI) / 180)) * 1.08 + 2;
  const c = model.center;
  return {
    position: [c[0] + (dir[0] / n) * dist, c[1] + (dir[1] / n) * dist, c[2] + (dir[2] / n) * dist],
    target: [...c],
  };
}

const fmt = (v: number, d = 3) => (Math.abs(v) < 5e-10 ? '0' : v.toFixed(d));

export default function AtomViewer({
  structures: given,
  initial = 0,
  allowUpload = true,
}: {
  structures: Structure[];
  initial?: number;
  allowUpload?: boolean;
}) {
  const theme = useSceneTheme();
  const ids = useId();
  const [uploaded, setUploaded] = useState<Structure[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const structures = useMemo(() => [...given, ...uploaded], [given, uploaded]);
  const [which, setWhich] = useState(Math.min(initial, given.length - 1));
  const s = structures[which]!;
  const periodic = !!s.cell && s.pbc.some(Boolean);
  const [opts, setOpts] = useState<ViewerOptions>({ ...DEFAULT_OPTIONS, boundary: periodic });
  const [selected, setSelected] = useState<number[]>([]);
  const [shellOn, setShellOn] = useState(false);
  const [rc, setRc] = useState(3.0);
  const [showCell, setShowCell] = useState(true);
  const vectorKeys = Object.keys(s.vectors ?? {});
  const [showVectors, setShowVectors] = useState(true);
  // Arrow scale is a multiplier on an automatic scale that makes the longest arrow ~2.4 Å.
  const [vecScale, setVecScale] = useState(1);
  const [preset, setPreset] = useState<Preset>('iso');
  const [resetKey, setResetKey] = useState(0);
  const [active] = useLinkedSymbol('atom-viewer');

  const bondList = useMemo(() => displayBonds(s), [s]);
  const effective: ViewerOptions = { ...opts, shell: shellOn ? rc : null };
  const model = useMemo(
    () => buildModel(s, effective, selected.slice(0, 1), bondList),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [s, opts, shellOn, rc, selected[0], bondList],
  );
  const m = measure(model.atoms, selected);
  // The camera frames the structure when it, the repeat or the preset changes — not on every selection.
  const view = useMemo(() => viewFor(buildModel(s, { ...opts, shell: null }), s, preset), [s, opts.repeat, opts.boundary, preset]);

  const choose = (k: number) => {
    setWhich(k);
    setSelected([]);
    const next = structures[k]!;
    setOpts((o) => ({ ...o, boundary: !!next.cell && next.pbc.some(Boolean), repeat: [1, 1, 1] }));
    setPreset('iso');
    setResetKey((x) => x + 1);
  };
  const onUpload = async (file: File | undefined) => {
    if (!file) return;
    setUploadError(null);
    if (file.size > 2_000_000) return setUploadError('File too large (limit 2 MB).');
    try {
      const st = parseStructureFile(file.name, await file.text());
      const next = [...uploaded, st];
      setUploaded(next);
      const k = given.length + next.length - 1;
      setWhich(k);
      setSelected([]);
      setOpts((o) => ({ ...o, boundary: !!st.cell && st.pbc.some(Boolean), repeat: [1, 1, 1] }));
      setPreset('iso');
      setResetKey((x) => x + 1);
    } catch (e) {
      setUploadError((e as Error).message);
    }
  };
  const onPick = (k: number | null, additive: boolean) => {
    if (k === null) return setSelected(additive ? selected : []);
    if (additive) {
      setSelected(selected.includes(k) ? selected.filter((x) => x !== k) : [...selected, k].slice(-4));
    } else {
      setSelected(selected.length === 1 && selected[0] === k ? [] : [k]);
    }
  };

  const sel = selected[0] !== undefined ? model.atoms[selected[0]] : undefined;
  const shellComposition = useMemo(() => {
    const counts = new Map<string, number>();
    model.atoms.filter((a) => a.inShell).forEach((a) => counts.set(a.symbol, (counts.get(a.symbol) ?? 0) + 1));
    return [...counts].map(([el, n]) => `${n} ${el}`).join(', ');
  }, [model]);

  const params = s.cell ? cellParameters(s.cell) : null;
  const vecArr = vectorKeys[0] ? s.vectors![vectorKeys[0]]! : null;
  const vecMax = vecArr ? Math.max(1e-9, ...vecArr.values.map((v) => Math.hypot(...v))) : 1;
  const angPerUnit = (2.4 / vecMax) * vecScale;

  return (
    <div className="lab viewer">
      <div className="viewer-top">
        <label className="viewer-pick" htmlFor={`${ids}-structure`}>
          <span>Structure</span>
          <select id={`${ids}-structure`} value={which} onChange={(e) => choose(Number(e.target.value))}>
            {structures.map((st, k) => (
              <option key={st.id} value={k}>
                {st.title}
              </option>
            ))}
          </select>
        </label>
        {allowUpload && (
          <label className="viewer-upload">
            <input type="file" accept=".xyz,.extxyz,.vasp,POSCAR,CONTCAR" onChange={(e) => onUpload(e.target.files?.[0])} />
            <span>Open your own (XYZ, extXYZ, POSCAR)</span>
          </label>
        )}
        {uploadError && (
          <p className="viewer-error" role="alert">
            {uploadError}
          </p>
        )}
        <div className="seg" role="group" aria-label="Display mode">
          {(['ball-and-stick', 'space-filling'] as const).map((md) => (
            <button key={md} type="button" aria-pressed={opts.mode === md} onClick={() => setOpts({ ...opts, mode: md })}>
              {md === 'ball-and-stick' ? 'Ball & stick' : 'Space-filling'}
            </button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Camera">
          {(['iso', 'a', 'b', 'c'] as const)
            .filter((p) => p === 'iso' || s.cell)
            .map((p) => (
              <button
                key={p}
                type="button"
                aria-pressed={preset === p}
                onClick={() => {
                  setPreset(p);
                  setResetKey((x) => x + 1);
                }}
              >
                {p === 'iso' ? 'Reset view' : `along ${p}`}
              </button>
            ))}
        </div>
      </div>

      <div className="lab-split">
        <div className="lab-stage viewer-stage">
          <Stage
            view={view}
            resetKey={resetKey}
            background={theme.canvas}
            label={`${s.title}: ${model.atoms.length} atoms drawn. ${sel ? `Selected ${sel.symbol} ${sel.index + 1}.` : 'No atom selected.'}`}
            minDistance={2}
            maxDistance={Math.max(60, model.extent * 8)}
          >
            <ViewerScene
              s={s}
              model={model}
              theme={theme}
              selected={selected}
              showCell={showCell && !!s.cell}
              repeat={opts.repeat}
              shell={shellOn && sel ? { center: sel.position, radius: rc } : null}
              shellActive={active === 'rc'}
              vectors={showVectors && vecArr ? { values: vecArr.values, scale: angPerUnit } : null}
              onPick={onPick}
            />
          </Stage>
          <p className="viewer-hint" aria-hidden="true">
            Drag to rotate · scroll to zoom · click an atom · shift-click up to 4 to measure
          </p>
        </div>

        <div className="lab-panel viewer-panel">
          <section aria-labelledby={`${ids}-sel`}>
            <h3 id={`${ids}-sel`} className="viewer-h">
              Selection
            </h3>
            <label className="viewer-pick" htmlFor={`${ids}-atom`}>
              <span>Atom</span>
              <select
                id={`${ids}-atom`}
                value={selected[0] ?? ''}
                onChange={(e) => setSelected(e.target.value === '' ? [] : [Number(e.target.value)])}
              >
                <option value="">none</option>
                {model.atoms.map((a, k) =>
                  a.ghost ? null : (
                    <option key={k} value={k}>
                      {a.symbol} {a.index + 1}
                      {a.image.some((v) => v !== 0) ? ` [${a.image.join(' ')}]` : ''}
                    </option>
                  ),
                )}
              </select>
            </label>
            {sel ? (
              <dl className="lab-readout viewer-readout" aria-live="polite">
                <div>
                  <dt>Atom</dt>
                  <dd>
                    {sel.symbol} {sel.index + 1}
                    {sel.ghost ? ' (periodic image)' : ''}
                  </dd>
                </div>
                <div>
                  <dt>Position</dt>
                  <dd className="mono">{sel.position.map((v) => fmt(v, 3)).join('  ')} Å</dd>
                </div>
                {s.cell && (
                  <div>
                    <dt>Fractional</dt>
                    <dd className="mono">{cartToFrac(sel.position, s.cell).map((v) => fmt(v, 4)).join('  ')}</dd>
                  </div>
                )}
                {sel.untabulated && (
                  <div className="is-alert">
                    <dt>Radius</dt>
                    <dd>no tabulated covalent radius — fallback used</dd>
                  </div>
                )}
              </dl>
            ) : (
              <p className="lab-note">Click an atom, or choose one from the list.</p>
            )}
            {m && (
              <p className="viewer-measure" aria-live="polite">
                <span className="kind">{m.kind}</span>{' '}
                {m.atoms.map((k) => `${model.atoms[k]!.symbol}${model.atoms[k]!.index + 1}`).join('–')} ={' '}
                <strong>
                  {m.value.toFixed(m.kind === 'distance' ? 4 : 2)} {m.unit}
                </strong>
              </p>
            )}
            {selected.length > 0 && (
              <button type="button" className="btn viewer-clear" onClick={() => setSelected([])}>
                Clear selection
              </button>
            )}
          </section>

          <section aria-labelledby={`${ids}-shell`}>
            <h3 id={`${ids}-shell`} className="viewer-h">
              Cutoff shell
            </h3>
            <label className="viewer-check">
              <input type="checkbox" checked={shellOn} onChange={(e) => setShellOn(e.target.checked)} /> Show atoms within{' '}
              <i>r</i>
              <sub>c</sub> of the selected atom
            </label>
            <div className="lab-controls">
              <label htmlFor={`${ids}-rc`} data-sym="rc">
                <span>
                  <i>r</i>
                  <sub>c</sub>
                </span>
                <input id={`${ids}-rc`} type="range" min={1} max={8} step={0.05} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
                <output>{rc.toFixed(2)} Å</output>
              </label>
            </div>
            {shellOn && (
              <p className="lab-note" aria-live="polite">
                {sel ? (
                  <>
                    <strong>{model.shellCount}</strong> atoms within {rc.toFixed(2)} Å{shellComposition ? ` (${shellComposition})` : ''}
                    {model.atoms.some((a) => a.ghost) ? ', including periodic images outside the drawn cells (drawn faded).' : '.'}
                  </>
                ) : (
                  'Select an atom to centre the shell.'
                )}
              </p>
            )}
          </section>

          <section aria-labelledby={`${ids}-disp`}>
            <h3 id={`${ids}-disp`} className="viewer-h">
              Display
            </h3>
            <label className="viewer-check">
              <input type="checkbox" checked={opts.bonds} onChange={(e) => setOpts({ ...opts, bonds: e.target.checked })} /> Bonds{' '}
              <small>(d &lt; 1.15 × sum of covalent radii — a drawing convention)</small>
            </label>
            {periodic && (
              <>
                <label className="viewer-check">
                  <input type="checkbox" checked={showCell} onChange={(e) => setShowCell(e.target.checked)} /> Unit cell
                </label>
                <label className="viewer-check">
                  <input type="checkbox" checked={opts.boundary} onChange={(e) => setOpts({ ...opts, boundary: e.target.checked })} />{' '}
                  Atoms on cell faces, edges and corners
                </label>
                <div className="viewer-repeat" role="group" aria-label="Replicas along a, b, c">
                  <span>Replicas</span>
                  {(['a', 'b', 'c'] as const).map((ax, k) =>
                    s.pbc[k] ? (
                      <label key={ax}>
                        <i>{ax}</i>
                        <select
                          value={opts.repeat[k]}
                          onChange={(e) => {
                            const r = [...opts.repeat] as Vec3;
                            r[k] = Number(e.target.value);
                            setOpts({ ...opts, repeat: r });
                            setSelected([]);
                          }}
                        >
                          {[1, 2, 3, 4].map((n) => (
                            <option key={n} value={n}>
                              {n}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : null,
                  )}
                </div>
              </>
            )}
            {vecArr && (
              <>
                <label className="viewer-check">
                  <input type="checkbox" checked={showVectors} onChange={(e) => setShowVectors(e.target.checked)} />{' '}
                  {vecArr.label} vectors <span className="viewer-tag">{vecArr.status.replace('_', ' ')}</span>
                </label>
                <div className="lab-controls">
                  <label htmlFor={`${ids}-vs`}>
                    <span>arrow scale</span>
                    <input id={`${ids}-vs`} type="range" min={0.25} max={3} step={0.05} value={vecScale} onChange={(e) => setVecScale(Number(e.target.value))} />
                    <output>×{vecScale.toFixed(2)}</output>
                  </label>
                </div>
                <p className="lab-note">
                  Scale: 1 Å of arrow = {(1 / angPerUnit).toFixed(2)} {vecArr.unit}; largest |F| = {vecMax.toFixed(2)} {vecArr.unit}.
                </p>
              </>
            )}
          </section>

          <section aria-labelledby={`${ids}-info`} className="viewer-info">
            <h3 id={`${ids}-info`} className="viewer-h">
              Structure
            </h3>
            <dl>
              <div>
                <dt>Atoms</dt>
                <dd>
                  {s.positions.length} stored · {model.atoms.length} drawn
                </dd>
              </div>
              {params ? (
                <div>
                  <dt>Cell</dt>
                  <dd className="mono">
                    a b c = {params.lengths.map((v) => v.toFixed(3)).join(' ')} Å
                    <br />α β γ = {params.angles.map((v) => v.toFixed(2)).join(' ')}°
                  </dd>
                </div>
              ) : (
                <div>
                  <dt>Boundary</dt>
                  <dd>open (no periodic images)</dd>
                </div>
              )}
              {periodic && (
                <div>
                  <dt>Periodic</dt>
                  <dd>{(['a', 'b', 'c'] as const).filter((_, k) => s.pbc[k]).join(', ') || 'none'}</dd>
                </div>
              )}
              {s.provenance && (
                <div>
                  <dt>Source</dt>
                  <dd>
                    <code>{s.provenance.source}</code>
                    {s.provenance.notes && <span className="viewer-notes">{s.provenance.notes}</span>}
                  </dd>
                </div>
              )}
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
}

function ViewerScene({
  s,
  model,
  theme,
  selected,
  showCell,
  repeat,
  shell,
  shellActive,
  vectors,
  onPick,
}: {
  s: Structure;
  model: DisplayModel;
  theme: SceneTheme;
  selected: number[];
  showCell: boolean;
  repeat: Vec3;
  shell: { center: Vec3; radius: number } | null;
  shellActive: boolean;
  vectors: { values: Vec3[]; scale: number } | null;
  onPick: (k: number | null, additive: boolean) => void;
}) {
  const dist = model.extent * 3;
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, dist * 0.6, dist * 1.8);
  const bg = linear(theme.canvas);
  const accent = linear(theme.phys);
  const measureCol = linear(theme.arch);

  const colorOf = useMemo(() => {
    const cache = new Map<string, [number, number, number]>();
    return (sym: string) => {
      if (!cache.has(sym)) {
        const [r, g, b] = elementColor(sym, theme.dark ? 'dark' : 'light');
        const c = new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
        cache.set(sym, [c.r, c.g, c.b]);
      }
      return cache.get(sym)!;
    };
  }, [theme.dark]);

  const buffers = useMemo<AtomBuffers>(() => {
    const n = model.atoms.length;
    const positions = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const metal = new Float32Array(n);
    const ao = new Float32Array(n);
    const focus = model.shellCenter !== null;
    model.atoms.forEach((a: DrawnAtom, k) => {
      positions.set(a.position, 3 * k);
      radii[k] = a.radius;
      let c = colorOf(a.symbol);
      if (focus && !a.inShell && k !== model.shellCenter) c = mixRGB(c, bg, 0.72);
      if (a.ghost) c = mixRGB(c, bg, 0.3);
      if (selected.includes(k)) c = mixRGB(c, k === selected[0] ? accent : measureCol, 0.55);
      colors.set(c, 3 * k);
      metal[k] = METALS.has(a.symbol) ? 1 : 0;
      ao[k] = a.boundary ? 0.92 : 1;
    });
    return { positions, radii, colors, metal, ao, count: n };
  }, [model, selected, colorOf, bg, accent, measureCol]);

  const sticks = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    for (const [i, j] of model.bonds) {
      const a = model.atoms[i]!;
      const b = model.atoms[j]!;
      const mid: Vec3 = [(a.position[0] + b.position[0]) / 2, (a.position[1] + b.position[1]) / 2, (a.position[2] + b.position[2]) / 2];
      const ca = buffers.colors.subarray(3 * i, 3 * i + 3);
      const cb = buffers.colors.subarray(3 * j, 3 * j + 3);
      out.push({ a: a.position, b: mid, color: [ca[0]!, ca[1]!, ca[2]!], radius: 0.13 });
      out.push({ a: mid, b: b.position, color: [cb[0]!, cb[1]!, cb[2]!], radius: 0.13 });
    }
    return out;
  }, [model, buffers]);

  const lines = useMemo<ColoredLine[]>(() => {
    const out: ColoredLine[] = [];
    if (showCell && s.cell) {
      const cell = s.cell;
      const ink = linear(theme.inkMuted);
      const corner = (f: Vec3): [number, number, number] => vecMat(f, cell) as [number, number, number];
      for (let a = 0; a < repeat[0]; a++)
        for (let b = 0; b < repeat[1]; b++)
          for (let c = 0; c < repeat[2]; c++) {
            const o: Vec3 = [a, b, c];
            const home = a === 0 && b === 0 && c === 0;
            const rgba: [number, number, number, number] = [...ink, home ? 0.75 : 0.3];
            const e = [
              [[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]],
              [[1, 0, 0], [1, 1, 0]], [[1, 0, 0], [1, 0, 1]], [[0, 1, 0], [1, 1, 0]],
              [[0, 1, 0], [0, 1, 1]], [[0, 0, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 1]],
              [[1, 1, 0], [1, 1, 1]], [[1, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]],
            ] as const;
            for (const [p, q] of e) {
              out.push({
                a: corner([o[0] + p[0], o[1] + p[1], o[2] + p[2]]),
                b: corner([o[0] + q[0], o[1] + q[1], o[2] + q[2]]),
                rgba,
              });
            }
          }
    }
    // Measurement path between picked atoms.
    for (let k = 1; k < selected.length; k++) {
      const p = model.atoms[selected[k - 1]!]?.position;
      const q = model.atoms[selected[k]!]?.position;
      if (p && q) out.push({ a: p, b: q, rgba: [...measureCol, 1] });
    }
    return out;
  }, [showCell, s.cell, repeat, selected, model, theme.inkMuted, measureCol]);

  const arrows = useMemo<Segment[]>(() => {
    if (!vectors) return [];
    const col = linear(theme.caveat);
    return model.atoms
      .filter((a) => !a.ghost)
      .map((a) => {
        const v = vectors.values[a.index] ?? [0, 0, 0];
        const len = Math.hypot(...v) || 1;
        // Start on the sphere surface so the whole arrow (its length ∝ |v|) stays visible.
        const s0: Vec3 = [a.position[0] + (v[0] / len) * a.radius, a.position[1] + (v[1] / len) * a.radius, a.position[2] + (v[2] / len) * a.radius];
        return {
          a: s0,
          b: [s0[0] + v[0] * vectors.scale, s0[1] + v[1] * vectors.scale, s0[2] + v[2] * vectors.scale],
          color: col,
          radius: 0.09,
        } as Segment;
      })
      .filter((seg) => Math.hypot(seg.b[0] - seg.a[0], seg.b[1] - seg.a[1], seg.b[2] - seg.a[2]) > 0.05);
  }, [vectors, model, theme.caveat]);

  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      {sticks.length > 0 && <Sticks segments={sticks} uniforms={uniforms} />}
      {arrows.length > 0 && <Arrows arrows={arrows} uniforms={uniforms} head={0.35} />}
      {lines.length > 0 && <Lines lines={lines} />}
      {shell && <Shell center={shell.center} radius={shell.radius} color={theme.phys} opacity={shellActive ? 0.85 : 0.45} grid />}
      <ShiftPicker positions={buffers.positions} radii={buffers.radii} count={buffers.count} onPick={onPick} />
    </>
  );
}

/** AtomPicker plus the shift key: shift-click adds to a measurement. */
function ShiftPicker(props: {
  positions: Float32Array;
  radii: Float32Array;
  count: number;
  onPick: (k: number | null, additive: boolean) => void;
}) {
  const shift = useRef(false);
  useEffect(() => {
    const track = (e: KeyboardEvent | PointerEvent) => {
      shift.current = e.shiftKey;
    };
    window.addEventListener('pointerdown', track, true);
    window.addEventListener('keydown', track);
    window.addEventListener('keyup', track);
    return () => {
      window.removeEventListener('pointerdown', track, true);
      window.removeEventListener('keydown', track);
      window.removeEventListener('keyup', track);
    };
  }, []);
  return <AtomPicker positions={props.positions} radii={props.radii} count={props.count} onPick={(k) => props.onPick(k, shift.current)} />;
}
