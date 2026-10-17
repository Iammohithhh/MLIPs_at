import { useId, useMemo, useState } from 'react';
import './PhysicsLab.css';
import './SevenNet.css';
import { useSceneTheme } from '~/three/theme';
import {
  TOY_RC,
  evaluateNaive,
  evaluateSerial,
  evaluateSevenNet,
  makeSystem,
  makeToyModel,
  neighbourList,
  type CommSwitches,
  type ParallelEvaluation,
  type System2D,
} from '~/lib/models/sevennet';

// A toy invariant message-passing network (src/lib/models/sevennet.ts) evaluated three ways on the
// same 300 atoms: on one process, with SevenNet's scheme (r_c ghosts, features exchanged between
// layers, gradients and partial forces sent back), and with the naive T·r_c ghost shell. Any of
// SevenNet's three exchanges can be switched off to see what it carries.

const sys: System2D = makeSystem();
const nbrs = neighbourList(sys, TOY_RC);
const S = 30; // px per toy unit
const TOL = 1e-9;

export default function SevenNetParallel() {
  const theme = useSceneTheme();
  const ids = useId();
  const [P, setP] = useState(3);
  const [T, setT] = useState(3);
  const [scheme, setScheme] = useState<'sevennet' | 'naive'>('sevennet');
  const [sw, setSw] = useState<CommSwitches>({ features: true, gradients: true, forces: true });
  const [rank, setRank] = useState(1);
  const r = Math.min(rank, P - 1);
  const model = useMemo(() => makeToyModel(T), [T]);
  const ref = useMemo(() => evaluateSerial(model, sys, 0, 'none', nbrs), [model]);
  const par = useMemo<ParallelEvaluation>(
    () => (scheme === 'naive' ? evaluateNaive(model, sys, P, 0, 'none', nbrs) : evaluateSevenNet(model, sys, P, sw, 0, 'none', nbrs)),
    [model, P, scheme, sw],
  );
  const dom = par.domains[r]!;
  const ghosts = new Set(dom.ghosts);
  const owned = new Set(dom.owned);
  const shell = scheme === 'naive' ? T * TOY_RC : TOY_RC;
  let fmax = 0;
  for (const f of ref.forces) fmax = Math.max(fmax, Math.abs(f));
  const err = new Float64Array(sys.n);
  let wrong = 0;
  let emax = 0;
  for (let i = 0; i < sys.n; i++) {
    err[i] = Math.hypot(par.forces[2 * i]! - ref.forces[2 * i]!, par.forces[2 * i + 1]! - ref.forces[2 * i + 1]!);
    if (err[i]! > TOL * fmax) wrong++;
    emax = Math.max(emax, err[i]!);
  }
  const dE = par.energy - ref.energy;
  const arrow = 0.9 / fmax; // toy units of length per force unit
  const W = sys.width * S;
  const H = (sys.height + 0.9) * S;
  const y0 = 0.75 * S;
  const toggle = (k: keyof CommSwitches) => setSw((s) => ({ ...s, [k]: !s[k] }));
  const exact = Math.abs(dE) < 1e-9 * Math.abs(ref.energy) && wrong === 0;

  return (
    <div className="lab">
      <div className="sn-canvas">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`300 atoms split into ${P} strips. Rank ${r}'s owned atoms are filled blue and its ghost atoms are green rings. ${wrong} atoms have forces that differ from the serial result.`}>
          <defs>
            <marker id={`${ids}-head`} viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L6,3 L0,6 z" style={{ fill: theme.inkMuted }} />
            </marker>
            <marker id={`${ids}-bad`} viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L6,3 L0,6 z" style={{ fill: theme.fail }} />
            </marker>
          </defs>
          <rect x={Math.max(0, dom.lo - shell) * S} y={y0 - 0.5 * S} width={(Math.min(sys.width, dom.hi + shell) - Math.max(0, dom.lo - shell)) * S} height={sys.height * S + 0.5 * S} style={{ fill: theme.phys, opacity: 0.1 }} />
          <rect x={dom.lo * S} y={y0 - 0.5 * S} width={(dom.hi - dom.lo) * S} height={sys.height * S + 0.5 * S} style={{ fill: theme.arch, opacity: 0.12 }} />
          {par.domains.map((d, k) => (
            <g key={k}>
              {k > 0 ? <line x1={d.lo * S} x2={d.lo * S} y1={y0 - 0.5 * S} y2={H} style={{ stroke: theme.ruleStrong, strokeDasharray: '5 4', strokeWidth: 1.5 }} /> : null}
              <text x={((d.lo + d.hi) / 2) * S} y={0.5 * S} textAnchor="middle" className={`sn-rank${k === r ? ' is-on' : ''}`} onClick={() => setRank(k)}>
                rank {k}
              </text>
            </g>
          ))}
          {Array.from({ length: sys.n }, (_, i) => {
            const x = sys.pos[2 * i]! * S;
            const y = y0 + sys.pos[2 * i + 1]! * S;
            const rad = (sys.species[i] ? 0.3 : 0.24) * S;
            const bad = err[i]! > TOL * fmax;
            const style = owned.has(i)
              ? { fill: theme.arch }
              : ghosts.has(i)
                ? { fill: theme.canvas, stroke: theme.phys, strokeWidth: 2.2 }
                : { fill: theme.dark ? '#3c4552' : '#c9ced6' };
            const fx = par.forces[2 * i]! * arrow * S;
            const fy = par.forces[2 * i + 1]! * arrow * S;
            const gx = ref.forces[2 * i]! * arrow * S;
            const gy = ref.forces[2 * i + 1]! * arrow * S;
            return (
              <g key={i}>
                <circle cx={x} cy={y} r={rad} style={style} />
                {bad ? <circle cx={x} cy={y} r={rad + 4} style={{ fill: 'none', stroke: theme.fail, strokeWidth: 2 }} /> : null}
                {bad ? <line x1={x} y1={y} x2={x + gx} y2={y + gy} style={{ stroke: theme.inkMuted, strokeWidth: 1.2, strokeDasharray: '3 2' }} /> : null}
                <line x1={x} y1={y} x2={x + fx} y2={y + fy} markerEnd={`url(#${ids}-${bad ? 'bad' : 'head'})`} style={{ stroke: bad ? theme.fail : theme.inkMuted, strokeWidth: bad ? 1.8 : 1, opacity: bad ? 1 : 0.7 }} />
              </g>
            );
          })}
        </svg>
        <ul className="stage-legend sn-legend" aria-hidden="true">
          <li>
            <span className="dot" style={{ background: theme.arch }} /> owned by rank {r}
          </li>
          <li>
            <span className="ring" style={{ borderColor: theme.phys }} /> its ghosts ({scheme === 'naive' ? `${T}·r_c` : 'r_c'})
          </li>
          <li>
            <span className="ring" style={{ borderColor: theme.fail }} /> force differs from serial <small>(dashed: serial)</small>
          </li>
          <li>
            <small>tap a rank label to select it</small>
          </li>
        </ul>
      </div>
      <div className="lab-panel">
        <div className="lab-controls">
          <div className="seg" role="group" aria-label="Scheme">
            <button type="button" aria-pressed={scheme === 'sevennet'} onClick={() => setScheme('sevennet')}>
              SevenNet
            </button>
            <button type="button" aria-pressed={scheme === 'naive'} onClick={() => setScheme('naive')}>
              naive T·r<sub>c</sub> shell
            </button>
          </div>
          <label htmlFor={`${ids}-p`}>
            <span>ranks P</span>
            <input id={`${ids}-p`} type="range" min={1} max={4} step={1} value={P} onChange={(e) => setP(Number(e.target.value))} />
            <output>{P}</output>
          </label>
          <label htmlFor={`${ids}-t`}>
            <span>layers T</span>
            <input id={`${ids}-t`} type="range" min={1} max={5} step={1} value={T} onChange={(e) => setT(Number(e.target.value))} />
            <output>{T}</output>
          </label>
          <fieldset className="sn-switches" disabled={scheme === 'naive'}>
            <legend>SevenNet’s exchanges</legend>
            <label>
              <input type="checkbox" checked={sw.features} onChange={() => toggle('features')} /> ghost features h after each layer (forward)
            </label>
            <label>
              <input type="checkbox" checked={sw.gradients} onChange={() => toggle('gradients')} /> ghost gradients ∂E/∂h (reverse)
            </label>
            <label>
              <input type="checkbox" checked={sw.forces} onChange={() => toggle('forces')} /> ghost partial forces (reverse)
            </label>
          </fieldset>
        </div>
        <dl className="lab-readout" aria-live="polite">
          <div className={exact ? 'is-good' : 'is-alert'}>
            <dt>vs one process</dt>
            <dd>{exact ? 'identical to rounding' : 'wrong'}</dd>
          </div>
          <div className={Math.abs(dE) > 1e-9 * Math.abs(ref.energy) ? 'is-alert' : undefined}>
            <dt>energy difference</dt>
            <dd>{Math.abs(dE) > 1e-9 * Math.abs(ref.energy) ? dE.toFixed(3) : '0 to rounding'}</dd>
          </div>
          <div className={wrong ? 'is-alert' : undefined}>
            <dt>atoms with wrong forces</dt>
            <dd>
              {wrong} {wrong ? <small>max {((100 * emax) / fmax).toPrecision(2)} % of the largest force</small> : <small>all within rounding</small>}
            </dd>
          </div>
          <div>
            <dt>node updates</dt>
            <dd>
              {par.updates} <small>serial {T * sys.n}</small>
            </dd>
          </div>
          <div>
            <dt>exchanges per force call</dt>
            <dd>{par.messages}</dd>
          </div>
          <div>
            <dt>floats received</dt>
            <dd>
              {par.comm.positions + par.comm.features + par.comm.gradients + par.comm.forces}{' '}
              <small>
                pos {par.comm.positions} · h {par.comm.features} · ∂E/∂h {par.comm.gradients} · F {par.comm.forces}
              </small>
            </dd>
          </div>
        </dl>
        <p className="lab-note">
          With all three exchanges SevenNet’s answer equals the single-process one: energy and every force agree to rounding, as the
          paper requires of a parallel model that “does not alter the outputs”. Drop the feature exchange and ghosts carry stale
          features, so the energy itself is wrong. Drop the gradient exchange and the energy is still right but forces near the strip
          boundaries miss the part of ∂E/∂h that arrived through neighbours on other ranks (the second term of Eq. 8). Drop the
          force exchange and the bond forces computed on the far side of a boundary never reach their owner (Eq. 6). With T = 1 there
          is nothing to exchange between layers. The naive shell needs no exchange at all, at the price of the extra node updates.
        </p>
      </div>
    </div>
  );
}
