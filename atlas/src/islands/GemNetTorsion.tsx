import { useDeferredValue, useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './Dimenet.css';
import { Stage } from '~/three/Stage';
import { ImpostorAtoms, Lines, Sticks, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine, type Segment } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { PEROXIDE, makeToyDimeNet, makeToyGemNetQ, peroxide, peroxideDistances, peroxideFeatures, relDiff } from '~/lib/models/dimenet';
import { cssColor } from './spooky/colors';
import { elementColor } from '~/lib/structure/elements';
import { ZeroLaneChart } from './dimenet/ZeroLaneChart';

// A torsion in an H–O–O–H chain. With only the three bonds in the graph, every DimeNet
// message depends on bond lengths and the two H–O–O angles, none of which changes with
// the dihedral; a GemNet-style quadruplet term reads the dihedral θ_cabd directly.

const DIHEDRALS = Array.from({ length: 61 }, (_, k) => 3 * k);
const SYMBOLS = ['H', 'O', 'O', 'H'];
const atomRGB = (symbol: string, dark: boolean) => linear(`rgb(${elementColor(symbol, dark ? 'dark' : 'light').map((v) => Math.round(255 * v)).join(',')})`);

export default function GemNetTorsion() {
  const theme = useSceneTheme();
  const ids = useId();
  const [tau, setTau] = useState(111);
  const [rc, setRc] = useState(1.7);
  const rcD = useDeferredValue(rc);
  const models = useMemo(() => {
    const o = { rc: rcD, elements: [1, 8], blocks: 3 };
    return { dime: makeToyDimeNet(o), gem: makeToyGemNetQ(o) };
  }, [rcD]);
  const table = useMemo(
    () => ({ dime: DIHEDRALS.map((t) => peroxideFeatures(models.dime, t)), gem: DIHEDRALS.map((t) => peroxideFeatures(models.gem, t)) }),
    [models],
  );
  const ref = useMemo(() => ({ dime: peroxideFeatures(models.dime, tau), gem: peroxideFeatures(models.gem, tau) }), [models, tau]);
  const dime = table.dime.map((v) => relDiff(v, ref.dime));
  const gem = table.gem.map((v) => relDiff(v, ref.gem));
  const g = peroxideDistances(0);
  const hhMax = peroxideDistances(180).hh;
  const hh = peroxideDistances(tau).hh;
  const regime =
    rc < PEROXIDE.rOH
      ? 'no edges'
      : rc < PEROXIDE.rOO
        ? 'two separate O–H fragments'
        : rc < g.hh
          ? 'the three bonds only'
          : rc < g.hOfar
            ? 'bonds + H···H near cis'
            : rc < hhMax
              ? 'bonds + both H···O (1–3) edges'
              : 'every pair';
  return (
    <div className="lab">
      <div className="lab-split lab-split--wide">
        <div className="lab-stage">
          <Stage
            view={{ position: [0.7, -9, 4], target: [0.72, 0, 0.3] }}
            background={theme.canvas}
            label={`An H–O–O–H chain with a ${tau}° dihedral. With a ${rc.toFixed(2)} Å cutoff the graph holds ${regime}.`}
            minDistance={4}
            maxDistance={30}
          >
            <ChainScene theme={theme} tau={tau} rc={rc} />
          </Stage>
          <ul className="stage-legend" aria-hidden="true">
            <li>
              <span className="dot" style={{ background: cssColor('O', theme.dark) }} /> O
            </li>
            <li>
              <span className="dot" style={{ background: cssColor('H', theme.dark) }} /> H
            </li>
            <li>
              <span className="bar thin" style={{ background: theme.phys }} /> non-bonded edge in the graph
            </li>
          </ul>
        </div>
        <ZeroLaneChart
          ariaLabel={`Relative change of the chain's features between ${tau} degrees and every other dihedral, for a toy DimeNet and the same network with a GemNet quadruplet term, at a ${rc.toFixed(2)} Å cutoff.`}
          series={[
            { key: 'd', label: 'DimeNet (triplets), 3 blocks', color: theme.fail, xs: DIHEDRALS, ys: dime, width: 3.4 },
            { key: 'g', label: '+ GemNet quadruplet term (Eq. 11)', color: theme.arch, xs: DIHEDRALS, ys: gem },
          ]}
          xMin={0}
          xMax={180}
          xTicks={[0, 30, 60, 90, 120, 150, 180]}
          xFmt={(v) => `${v}°`}
          xLabel="other dihedral τ′"
          yLabel="‖h(τ′) − h(τ)‖ / ‖h(τ)‖ (log)"
          logLo={-8}
          logHi={0}
          cursor={tau}
        />
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <label htmlFor={`${ids}-t`}>
            <span>dihedral τ (H–O–O–H)</span>
            <input id={`${ids}-t`} type="range" min={0} max={180} step={1} value={tau} onChange={(e) => setTau(Number(e.target.value))} />
            <output>{tau}°</output>
          </label>
          <label htmlFor={`${ids}-rc`}>
            <span>
              cutoff r<sub>c</sub>
            </span>
            <input id={`${ids}-rc`} type="range" min={1.2} max={3} step={0.02} value={rc} onChange={(e) => setRc(Number(e.target.value))} />
            <output>{rc.toFixed(2)} Å</output>
          </label>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div>
            <dt>graph</dt>
            <dd>{regime}</dd>
          </div>
          <div>
            <dt>H···O (1–3), fixed</dt>
            <dd>{g.hOfar.toFixed(2)} Å</dd>
          </div>
          <div>
            <dt>H···H (1–4) at τ</dt>
            <dd>{hh.toFixed(2)} Å</dd>
          </div>
        </dl>
        <p className="lab-note">
          Between the O–O bond ({PEROXIDE.rOO} Å) and the shortest H···H distance ({g.hh.toFixed(2)} Å, at τ = 0) the graph
          is the bare chain. Every DimeNet triplet then involves two bonds sharing an oxygen and the H–O–O angle between
          them, and none of these changes when one OH rotates about the O–O axis: the red curve sits on the zero lane for
          every dihedral and every depth. The quadruplet term of GemNet’s Eq. (11) passes the message from the far H to the
          near one through the O–O bond with the dihedral θ<sub>cabd</sub> in its basis, so it sees the torsion at once.
          Raise the cutoff past {g.hOfar.toFixed(2)} Å and the H···O edges make the angle at O depend on τ: DimeNet sees it
          too. With all distances inside the cutoff, distances alone fix the geometry up to a mirror image.
        </p>
      </div>
    </div>
  );
}

function ChainScene({ theme, tau, rc }: { theme: SceneTheme; tau: number; rc: number }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 6, 30);
  const pos = useMemo(() => peroxide(tau), [tau]);
  const atoms = useMemo<AtomBuffers>(() => {
    const p = new Float32Array(12);
    const colors = new Float32Array(12);
    pos.forEach((v, k) => {
      p.set(v, 3 * k);
      colors.set(atomRGB(SYMBOLS[k]!, theme.dark), 3 * k);
    });
    return { positions: p, radii: new Float32Array([0.22, 0.34, 0.34, 0.22]), colors, metal: new Float32Array(4), ao: new Float32Array(4).fill(1), count: 4 };
  }, [pos, theme]);
  const bonds = useMemo<Segment[]>(() => {
    const c = mixRGB(linear(theme.ink), linear(theme.canvas), 0.3);
    return [
      [0, 1],
      [1, 2],
      [2, 3],
    ].map(([a, b]) => ({ a: pos[a!] as [number, number, number], b: pos[b!] as [number, number, number], color: c, radius: 0.06 }));
  }, [pos, theme]);
  const extra = useMemo<ColoredLine[]>(() => {
    const c = linear(theme.phys);
    const out: ColoredLine[] = [];
    for (const [a, b] of [
      [0, 2],
      [1, 3],
      [0, 3],
    ] as const) {
      const r = Math.hypot(pos[a]![0] - pos[b]![0], pos[a]![1] - pos[b]![1], pos[a]![2] - pos[b]![2]);
      if (r < rc) out.push({ a: pos[a] as [number, number, number], b: pos[b] as [number, number, number], rgba: [c[0], c[1], c[2], 1] });
    }
    return out;
  }, [pos, rc, theme]);
  return (
    <>
      <ImpostorAtoms buffers={atoms} version={0} uniforms={uniforms} />
      <Sticks segments={bonds} uniforms={uniforms} />
      <Lines lines={extra} />
    </>
  );
}
