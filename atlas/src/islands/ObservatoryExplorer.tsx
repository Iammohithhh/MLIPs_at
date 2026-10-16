import { useEffect, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Observatory.css';
import { Stage } from '~/three/Stage';
import {
  Arrows,
  AtomPicker,
  ImpostorAtoms,
  Lines,
  Shell,
  linear,
  mixRGB,
  useShadingUniforms,
  type AtomBuffers,
  type ColoredLine,
  type Segment,
} from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { covalentRadius, elementColor, hexToLinearRgb } from '~/lib/structure/elements';
import { angleDeg, forceAgreement, galleryForces, pressureGPa, type ObsDoc, type ObsFrame, type Vec3 } from '~/lib/models/observatory';

// Model Observatory, Fig. 1: several released checkpoints on the same rattled structure.
// Arrows are each model's forces in its own colour; the translucent halo around an atom
// has a radius set by the spread of the models' force vectors on that atom. PRECOMPUTED:
// every vector is read from src/data/observatory/observatory.json.

export const MODEL_VARS = ['--atlas-m1', '--atlas-m2', '--atlas-m3', '--atlas-m4'];

export function useModelColors(dark: boolean): string[] {
  const [c, setC] = useState<string[]>(['#3d6bb3', '#c4622d', '#2a8a6a', '#a04e9e']);
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    setC(MODEL_VARS.map((v, i) => cs.getPropertyValue(v).trim() || c[i]!));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark]);
  return c;
}

export function useObservatory(): { doc: ObsDoc | null; failed: boolean } {
  const [doc, setDoc] = useState<ObsDoc | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    import('~/data/observatory/observatory.json').then((m) => live && setDoc(m.default as unknown as ObsDoc)).catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);
  return { doc, failed };
}

const LABEL: Record<string, string> = {
  'si-diamond': 'Si',
  'nacl-rocksalt': 'NaCl',
  'cu-fcc': 'Cu',
  'zno-wurtzite': 'ZnO',
  'au111-slab': 'Au(111)',
  h2o: 'H₂O',
  benzene: 'Benzene',
  ethanol: 'Ethanol',
};

function colorOf(sym: string, dark: boolean): [number, number, number] {
  const c = elementColor(sym, dark ? 'dark' : 'light');
  return hexToLinearRgb(
    '#' +
      c
        .map((v) =>
          Math.round(Math.min(1, Math.max(0, v)) * 255)
            .toString(16)
            .padStart(2, '0'),
        )
        .join(''),
  );
}

/** Halo radius beyond the atom: HALO.perEvA Å per eV/Å of spread, the same on every structure, capped; none below HALO.min. */
const HALO = { perEvA: 3, cap: 1.6, min: 0.01 };

/** Bulk crystals: the only inputs for which a pressure is meaningful. */
const BULK = new Set(['si-diamond', 'nacl-rocksalt', 'cu-fcc', 'zno-wurtzite']);

const len = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);
const f3 = (x: number, d = 2) => x.toFixed(d).replace('-', '−');

export default function ObservatoryExplorer() {
  const theme = useSceneTheme();
  const colors = useModelColors(theme.dark);
  const { doc, failed } = useObservatory();
  const [sid, setSid] = useState('au111-slab');
  const [on, setOn] = useState<boolean[]>([true, true, true, true]);
  const [atom, setAtom] = useState(0);
  const [halo, setHalo] = useState(true);
  const ids = useId();

  if (failed) return <p className="stage-fallback">The precomputed model outputs could not be loaded.</p>;
  if (!doc)
    return (
      <div className="lab obs obs-loading" aria-busy="true">
        Loading the precomputed model outputs…
      </div>
    );

  const frame = doc.inputs.gallery.find((g) => g.id === sid)!;
  const sets = galleryForces(doc, sid);
  const agree = forceAgreement(sets);
  const a = Math.min(atom, frame.symbols.length - 1);
  const maxSpread = Math.max(...agree.spread);
  const worst = agree.spread.indexOf(maxSpread);

  return (
    <div className="lab obs">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            key={sid}
            view={viewFor(frame)}
            background={theme.canvas}
            label={`${LABEL[sid]} rattled by ${doc.settings.rattle.sigma} Å: forces from ${doc.models
              .filter((_, i) => on[i])
              .map((m) => m.name)
              .join(', ')}; halo radius shows the spread between models.`}
            minDistance={4}
            maxDistance={90}
          >
            <ObsScene
              theme={theme}
              frame={frame}
              forces={sets}
              on={on}
              colors={colors}
              spread={halo ? agree.spread : null}
              selected={a}
              onPick={(i) => i !== null && setAtom(i)}
            />
          </Stage>
          <div className="lab-toolbar">
            <div className="seg" role="group" aria-label="Structure">
              {doc.inputs.gallery.map((g) => (
                <button
                  key={g.id}
                  type="button"
                  aria-pressed={g.id === sid}
                  onClick={() => {
                    setSid(g.id);
                    setAtom(0);
                  }}
                >
                  {LABEL[g.id] ?? g.id}
                </button>
              ))}
            </div>
          </div>
          <ul className="stage-legend" aria-hidden="true">
            <li>arrows: forces, common scale per structure</li>
            {halo && <li>halo: +{HALO.perEvA} Å per eV/Å of spread, same on every structure</li>}
            <li>click an atom</li>
          </ul>
        </div>

        <div className="obs-side">
          <fieldset className="obs-models">
            <legend>Models (all trained on MPtrj)</legend>
            {doc.models.map((m, i) => (
              <label key={m.key} className="obs-chip">
                <input type="checkbox" checked={on[i]} onChange={() => setOn(on.map((v, k) => (k === i ? !v : v)))} />
                <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" />
                <span>
                  {m.name}
                  <small>{m.forces === 'direct' ? 'direct forces' : 'F = −∇E'}</small>
                </span>
              </label>
            ))}
            <label className="obs-chip obs-chip--plain">
              <input type="checkbox" checked={halo} onChange={() => setHalo(!halo)} />
              <span>disagreement halo</span>
            </label>
          </fieldset>

          <div className="obs-atom" aria-live="polite">
            <p className="obs-atom-h">
              <strong>
                {frame.symbols[a]}
                {a + 1}
              </strong>{' '}
              · spread {f3(agree.spread[a]!, 3)} eV/Å · consensus |F̄| {f3(len(agree.mean[a]!), 3)} eV/Å
            </p>
            <table className="obs-t">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">|F| (eV/Å)</th>
                  <th scope="col">angle to F̄</th>
                </tr>
              </thead>
              <tbody>
                {doc.models.map((m, i) => (
                  <tr key={m.key}>
                    <th scope="row">
                      <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" /> {m.name}
                    </th>
                    <td>{f3(len(sets[i]![a]!), 3)}</td>
                    <td>{angleDeg(sets[i]![a]!, agree.mean[a]!).toFixed(0)}°</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {worst !== a && (
              <button type="button" className="obs-link" onClick={() => setAtom(worst)}>
                Jump to the most disputed atom ({frame.symbols[worst]}
                {worst + 1}, spread {f3(maxSpread, 3)} eV/Å)
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="lab-panel obs-panel">
        <PairMatrix
          doc={doc}
          pairwise={agree.pairwise}
          colors={colors}
          caption={`Mean |F(a) − F(b)| over the ${frame.symbols.length} atoms of rattled ${LABEL[sid]} (eV/Å)`}
          id={ids}
        />
        <div className="obs-summary">
          <table className="obs-t">
            <caption>Whole structure</caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">RMS |F|</th>
                <th scope="col">RMS |F − F̄|</th>
                <th scope="col">pressure</th>
              </tr>
            </thead>
            <tbody>
              {doc.models.map((m, i) => {
                const r = m.results.gallery[sid]!;
                const rms = Math.sqrt(r.forces.reduce((s, v) => s + len(v) ** 2, 0) / r.forces.length);
                return (
                  <tr key={m.key}>
                    <th scope="row">
                      <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" /> {m.name}
                    </th>
                    <td>{f3(rms, 3)}</td>
                    <td>{f3(agree.deviation[i]!, 3)}</td>
                    <td>{r.stress && BULK.has(sid) ? `${f3(pressureGPa(r.stress), 1)} GPa` : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="obs-note">
            Forces in eV/Å. Pressure only for bulk crystals (a slab or a molecule in a box has no meaningful bulk pressure). Total energies are not compared:
            the checkpoints use different energy references.
          </p>
        </div>
      </div>
    </div>
  );
}

function PairMatrix({ doc, pairwise, colors, caption, id }: { doc: ObsDoc; pairwise: number[][]; colors: string[]; caption: string; id: string }) {
  const M = doc.models.length;
  const max = Math.max(1e-9, ...pairwise.flat());
  const C = 46;
  const L = 22;
  return (
    <figure className="obs-matrix">
      <svg viewBox={`0 0 ${L + M * C + 2} ${L + M * C + 2}`} role="img" aria-labelledby={`${id}-pm`}>
        <title id={`${id}-pm`}>{caption}</title>
        {doc.models.map((m, i) => (
          <g key={m.key}>
            <rect x={L + i * C + C / 2 - 6} y={4} width={12} height={12} rx={2} fill={colors[i]} />
            <rect x={4} y={L + i * C + C / 2 - 6} width={12} height={12} rx={2} fill={colors[i]} />
          </g>
        ))}
        {pairwise.map((row, i) =>
          row.map((v, j) => (
            <g key={`${i}-${j}`}>
              <rect
                className="obs-cell"
                x={L + j * C + 1}
                y={L + i * C + 1}
                width={C - 2}
                height={C - 2}
                rx={3}
                style={{
                  fillOpacity: i === j ? 0.04 : 0.12 + 0.88 * (v / max),
                }}
              >
                <title>
                  {doc.models[i]!.name} vs {doc.models[j]!.name}: {v.toFixed(3)} eV/Å
                </title>
              </rect>
              {i !== j && (
                <text className={`obs-cell-t${v / max > 0.55 ? ' inv' : ''}`} x={L + j * C + C / 2} y={L + i * C + C / 2 + 4} textAnchor="middle">
                  {v < 0.1 ? v.toFixed(3).slice(1) : v.toFixed(2)}
                </text>
              )}
            </g>
          )),
        )}
      </svg>
      <figcaption>{caption}. Rows and columns follow the model colours.</figcaption>
    </figure>
  );
}

function viewFor(frame: ObsFrame): { position: Vec3; target: Vec3 } {
  const n = frame.positions.length;
  const c = frame.positions.reduce<Vec3>((s, p) => [s[0] + p[0] / n, s[1] + p[1] / n, s[2] + p[2] / n], [0, 0, 0]);
  const r = Math.max(3, ...frame.positions.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])));
  const d = r * 3.4 + 4;
  return { position: [d * 0.55, -d * 0.8, d * 0.45], target: [0, 0, 0] };
}

function ObsScene({
  theme,
  frame,
  forces,
  on,
  colors,
  spread,
  selected,
  onPick,
}: {
  theme: SceneTheme;
  frame: ObsFrame;
  forces: Vec3[][];
  on: boolean[];
  colors: string[];
  spread: number[] | null;
  selected: number;
  onPick: (i: number | null) => void;
}) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 18, 90);
  const n = frame.positions.length;
  const c = useMemo<Vec3>(() => frame.positions.reduce<Vec3>((s, p) => [s[0] + p[0] / n, s[1] + p[1] / n, s[2] + p[2] / n], [0, 0, 0]), [frame, n]);
  const P = useMemo(() => frame.positions.map((p) => [p[0] - c[0], p[1] - c[1], p[2] - c[2]] as Vec3), [frame, c]);
  const radii = useMemo(() => frame.symbols.map((s) => Math.max(0.28, 0.5 * covalentRadius(s).r)), [frame]);

  const buffers = useMemo<AtomBuffers>(() => {
    const positions = new Float32Array(n * 3);
    const r = new Float32Array(n);
    const col = new Float32Array(n * 3);
    const stone = linear(theme.dark ? '#5a616b' : '#c9c4b9');
    P.forEach((p, i) => {
      positions.set(p, 3 * i);
      r[i] = radii[i]! * (i === selected ? 1.18 : 1);
      col.set(i === selected ? colorOf(frame.symbols[i]!, theme.dark) : mixRGB(stone, colorOf(frame.symbols[i]!, theme.dark), 0.75), 3 * i);
    });
    return {
      positions,
      radii: r,
      colors: col,
      metal: new Float32Array(n),
      ao: new Float32Array(n).fill(1),
      count: n,
    };
  }, [P, radii, selected, frame, theme.dark, n]);

  const scale = useMemo(() => {
    const m = Math.max(1e-9, ...forces.flatMap((f) => f.map(len)));
    return 1.8 / m;
  }, [forces]);

  const arrows = useMemo<Segment[]>(() => {
    const out: Segment[] = [];
    forces.forEach((f, mi) => {
      if (!on[mi]) return;
      const col = linear(colors[mi]!);
      f.forEach((v, i) => {
        const d = len(v);
        if (d * scale < 0.25) return;
        const u: Vec3 = [v[0] / d, v[1] / d, v[2] / d];
        const r0 = radii[i]! * 0.9;
        const a: Vec3 = [P[i]![0] + u[0] * r0, P[i]![1] + u[1] * r0, P[i]![2] + u[2] * r0];
        out.push({
          a,
          b: [a[0] + v[0] * scale, a[1] + v[1] * scale, a[2] + v[2] * scale],
          color: col,
          radius: i === selected ? 0.055 : 0.04,
        });
      });
    });
    return out;
  }, [forces, on, colors, scale, P, radii, selected]);

  const cell = useMemo<ColoredLine[]>(() => {
    const [A, B, C] = frame.cell;
    if (Math.abs(A[0] - 20) < 1e-6 && Math.abs(B[1] - 20) < 1e-6) return []; // molecule box: not drawn
    const o: Vec3 = [-c[0], -c[1], -c[2]];
    const add = (...v: Vec3[]): Vec3 => v.reduce<Vec3>((s, w) => [s[0] + w[0], s[1] + w[1], s[2] + w[2]], [0, 0, 0]);
    const corners = [0, 1].flatMap((i) => [0, 1].flatMap((j) => [0, 1].map((k) => add(o, i ? A : [0, 0, 0], j ? B : [0, 0, 0], k ? C : [0, 0, 0]))));
    const rgb = linear(theme.ruleStrong);
    const edges: [number, number][] = [
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7],
      [0, 2],
      [1, 3],
      [4, 6],
      [5, 7],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ];
    return edges.map(([p, q]) => ({
      a: corners[p]!,
      b: corners[q]!,
      rgba: [rgb[0], rgb[1], rgb[2], 0.9] as [number, number, number, number],
    }));
  }, [frame, c, theme.ruleStrong]);

  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <AtomPicker positions={buffers.positions} radii={buffers.radii} count={buffers.count} onPick={onPick} />
      {cell.length > 0 && <Lines lines={cell} />}
      {arrows.length > 0 && <Arrows arrows={arrows} uniforms={uniforms} head={0.22} />}
      {spread &&
        spread.map((v, i) =>
          v > HALO.min ? (
            <Shell
              key={i}
              center={P[i]!}
              radius={radii[i]! + 0.1 + Math.min(HALO.cap, HALO.perEvA * v)}
              color={theme.fail}
              opacity={i === selected ? 0.32 : 0.18}
            />
          ) : null,
        )}
    </>
  );
}
