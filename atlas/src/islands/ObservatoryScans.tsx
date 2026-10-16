import { useState } from 'react';
import './PhysicsLab.css';
import './Observatory.css';
import LineChart, { type Series } from './obs/LineChart';
import { useModelColors, useObservatory } from './ObservatoryExplorer';
import { useSceneTheme } from '~/three/theme';
import { curveSummary, eosFromScan, loopTest, relative, rotationTest, type ObsDoc } from '~/lib/models/observatory';

// Model Observatory, Figs. 2–3: scans and symmetry/conservation tests from the same
// precomputed file. Energies are always shown relative to a point of the same model's
// own curve, because the checkpoints do not share an energy reference.

type ScanId = 'ev-cu' | 'ev-si' | 'dimer-si' | 'dimer-cu' | 'na-au';

const SCANS: { id: ScanId; label: string }[] = [
  { id: 'ev-cu', label: 'Cu: E(V)' },
  { id: 'ev-si', label: 'Si: E(V)' },
  { id: 'dimer-si', label: 'Si₂ dimer' },
  { id: 'dimer-cu', label: 'Cu₂ dimer' },
  { id: 'na-au', label: 'Na on Au(111)' },
];

const short = (name: string) => name.split(' ')[0]!.replace(/-MP-0a|-0$/, '');
const fmtX = (v: number) => (v === 0 ? '0' : v.toExponential(1));
const fmtS = (v: number, d = 2) => v.toFixed(d).replace('-', '−');

export function ObservatoryScans() {
  const theme = useSceneTheme();
  const colors = useModelColors(theme.dark);
  const { doc, failed } = useObservatory();
  const [scan, setScan] = useState<ScanId>('ev-cu');
  if (failed) return <p className="stage-fallback">The precomputed model outputs could not be loaded.</p>;
  if (!doc)
    return (
      <div className="lab obs obs-loading" aria-busy="true">
        Loading the precomputed scans…
      </div>
    );

  return (
    <div className="lab obs">
      <div className="lab-panel obs-panel obs-panel--top">
        <div className="seg" role="group" aria-label="Scan">
          {SCANS.map((s) => (
            <button key={s.id} type="button" aria-pressed={s.id === scan} onClick={() => setScan(s.id)}>
              {s.label}
            </button>
          ))}
        </div>
        {scan.startsWith('ev') ? (
          <EvScan doc={doc} id={scan as 'ev-cu' | 'ev-si'} colors={colors} />
        ) : scan === 'na-au' ? (
          <NaAu doc={doc} colors={colors} />
        ) : (
          <Dimer doc={doc} id={scan as 'dimer-si' | 'dimer-cu'} colors={colors} />
        )}
      </div>
    </div>
  );
}

function EvScan({ doc, id, colors }: { doc: ObsDoc; id: 'ev-cu' | 'ev-si'; colors: string[] }) {
  const inp = doc.inputs[id];
  const fits = doc.models.map((m) => eosFromScan(inp.x, m.results[id].energy, inp.atoms, inp.volume0));
  const series: Series[] = doc.models.map((m, i) => ({
    key: m.key,
    name: m.name,
    short: short(m.name),
    color: colors[i]!,
    y: m.results[id].energy.map((e) => 1000 * (e / inp.atoms - fits[i]!.e0)),
  }));
  const el = id === 'ev-cu' ? 'Cu' : 'Si';
  return (
    <>
      <LineChart
        x={inp.x}
        series={series}
        xLabel="lattice scale a / a_gallery"
        yLabel="E − E_min (meV/atom)"
        fmtY={(v) => v.toFixed(0)}
        fmtX={(v) => v.toFixed(2)}
        marks={[{ x: 1, label: 'gallery a' }]}
        caption={`Energy per atom of ${el} under isotropic strain from each model, relative to that model's own fitted minimum.`}
      />
      <div className="obs-scroll">
        <table className="obs-t">
          <caption>Third-order Birch–Murnaghan fit to each model’s 13 points</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">a₀ / a_gallery</th>
              <th scope="col">V₀ (Å³/atom)</th>
              <th scope="col">B₀ (GPa)</th>
              <th scope="col">fit RMS (meV/atom)</th>
            </tr>
          </thead>
          <tbody>
            {doc.models.map((m, i) => (
              <tr key={m.key}>
                <th scope="row">
                  <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" /> {m.name}
                </th>
                <td>{fits[i]!.scale0.toFixed(4)}</td>
                <td>{fits[i]!.v0.toFixed(2)}</td>
                <td>{fits[i]!.b0.toFixed(0)}</td>
                <td>{fits[i]!.rmsMeV.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Dimer({ doc, id, colors }: { doc: ObsDoc; id: 'dimer-si' | 'dimer-cu'; colors: string[] }) {
  const x = doc.inputs[id].x;
  const sums = doc.models.map((m) => curveSummary(x, m.results[id].energy, m.results[id].force));
  const series: Series[] = doc.models.map((m, i) => ({
    key: m.key,
    name: m.name,
    short: short(m.name),
    color: colors[i]!,
    y: relative(m.results[id].energy, x.length - 1),
  }));
  const el = id === 'dimer-si' ? 'Si' : 'Cu';
  return (
    <>
      <LineChart
        x={x}
        series={series}
        xLabel="separation d (Å)"
        yLabel="E(d) − E(5 Å) (eV)"
        zero
        yClip={[-8, 8]}
        fmtY={(v) => v.toFixed(1)}
        fmtX={(v) => v.toFixed(1)}
        caption={`Energy of an isolated ${el}₂ dimer against separation from each model, relative to its own value at 5 Å.`}
      />
      <div className="obs-scroll">
        <table className="obs-t">
          <caption>What each curve does (energies relative to d = 5 Å; axis clipped at ±8 eV)</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">minimum at</th>
              <th scope="col">well depth</th>
              <th scope="col">E at 1.0 Å</th>
              <th scope="col">local minima</th>
              <th scope="col">attractive inside the wall</th>
            </tr>
          </thead>
          <tbody>
            {doc.models.map((m, i) => (
              <tr key={m.key}>
                <th scope="row">
                  <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" /> {m.name}
                </th>
                <td>{sums[i]!.xMin.toFixed(1)} Å</td>
                <td>{fmtS(sums[i]!.depth)} eV</td>
                <td>{fmtS(sums[i]!.eShort, 1)} eV</td>
                <td>{sums[i]!.minima}</td>
                <td>{sums[i]!.wrongWay.length ? sums[i]!.wrongWay.map((v) => v.toFixed(1)).join(', ') + ' Å' : 'no'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function NaAu({ doc, colors }: { doc: ObsDoc; colors: string[] }) {
  const x = doc.inputs['na-au'].x;
  const series: Series[] = doc.models.map((m, i) => ({
    key: m.key,
    name: m.name,
    short: short(m.name),
    color: colors[i]!,
    y: relative(m.results['na-au'].energy, x.length - 1),
  }));
  const fz: Series[] = doc.models.map((m, i) => ({
    key: m.key,
    name: m.name,
    short: short(m.name),
    color: colors[i]!,
    y: m.results['na-au'].fz,
  }));
  const sums = doc.models.map((m) => curveSummary(x, m.results['na-au'].energy));
  return (
    <>
      <div className="obs-two">
        <LineChart
          x={x}
          series={series}
          xLabel="height above the top Au layer (Å)"
          yLabel="E(h) − E(8 Å) (eV)"
          zero
          yClip={[-4, 3]}
          height={280}
          fmtY={(v) => v.toFixed(1)}
          fmtX={(v) => v.toFixed(1)}
          caption="Energy of a neutral Na atom above the fcc hollow of Au(111), relative to 8 Å, from each model."
        />
        <LineChart
          x={x}
          series={fz}
          xLabel="height above the top Au layer (Å)"
          yLabel="F_z on Na (eV/Å)"
          zero
          yClip={[-3, 3]}
          height={280}
          fmtY={(v) => v.toFixed(2)}
          fmtX={(v) => v.toFixed(1)}
          caption="Vertical force on the Na atom; negative pulls it toward the surface."
        />
      </div>
      <div className="obs-scroll">
        <table className="obs-t">
          <caption>Adsorption curve, relative to 8 Å</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">minimum at</th>
              <th scope="col">well depth</th>
              <th scope="col">cutoff (Å)</th>
            </tr>
          </thead>
          <tbody>
            {doc.models.map((m, i) => (
              <tr key={m.key}>
                <th scope="row">
                  <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" /> {m.name}
                </th>
                <td>{sums[i]!.xMin.toFixed(1)} Å</td>
                <td>{fmtS(sums[i]!.depth)} eV</td>
                <td>{m.cutoff ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function ObservatoryTests() {
  const theme = useSceneTheme();
  const colors = useModelColors(theme.dark);
  const { doc, failed } = useObservatory();
  const [view, setView] = useState<'energy' | 'gap'>('gap');
  if (failed) return <p className="stage-fallback">The precomputed model outputs could not be loaded.</p>;
  if (!doc)
    return (
      <div className="lab obs obs-loading" aria-busy="true">
        Loading the precomputed tests…
      </div>
    );
  const th = doc.inputs.loop.theta;
  const deg = th.map((t) => (t * 180) / Math.PI);
  const loops = doc.models.map((m) => loopTest(th, m.results.loop.power, m.results.loop.energy));
  const rots = doc.models.map((m) => rotationTest(doc.inputs.rotation.R, m.results.rotation.energy, m.results.rotation.forces));
  const series: Series[] =
    view === 'gap'
      ? doc.models.map((m, i) => ({
          key: m.key,
          name: m.name,
          short: short(m.name),
          color: colors[i]!,
          y: loops[i]!.workEnergy.map((w, k) => 1000 * (w - loops[i]!.energy[k]!)),
        }))
      : doc.models.flatMap((m, i) => [
          {
            key: m.key,
            name: m.name,
            short: short(m.name),
            color: colors[i]!,
            y: loops[i]!.energy.map((e) => 1000 * e),
          },
          {
            key: `${m.key}-w`,
            name: `${m.name}: −∫F·dx`,
            short: `${short(m.name)} work`,
            color: colors[i]!,
            y: loops[i]!.workEnergy.map((e) => 1000 * e),
            dashed: true,
          },
        ]);
  return (
    <div className="lab obs">
      <div className="lab-panel obs-panel obs-panel--top">
        <div className="seg" role="group" aria-label="Loop view">
          <button type="button" aria-pressed={view === 'gap'} onClick={() => setView('gap')}>
            work − energy
          </button>
          <button type="button" aria-pressed={view === 'energy'} onClick={() => setView('energy')}>
            energy and work
          </button>
        </div>
        <LineChart
          x={deg}
          series={series}
          xLabel="loop angle θ (°)"
          yLabel={view === 'gap' ? '−∫F·dx − ΔE (meV)' : 'E(θ) − E(0) (meV)'}
          zero
          fmtY={(v) => v.toFixed(0)}
          fmtX={(v) => v.toFixed(0)}
          caption={
            view === 'gap'
              ? 'Running work done by each model’s forces minus its own energy change around the closed loop; zero for a conservative model.'
              : 'Energy along the loop (solid) and the energy implied by the work of the forces (dashed).'
          }
        />
        <div className="obs-two obs-two--tables">
          <div className="obs-scroll">
            <table className="obs-t">
              <caption>
                Closed loop in {3 * doc.inputs.loop.atoms}-dimensional configuration space (rattled Si, {doc.inputs.loop.atoms} atoms,{' '}
                {doc.inputs.loop.theta.length} points)
              </caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">∮ F · dx (meV)</th>
                  <th scope="col">largest gap (meV)</th>
                  <th scope="col">ΔE range (meV)</th>
                </tr>
              </thead>
              <tbody>
                {doc.models.map((m, i) => (
                  <tr key={m.key}>
                    <th scope="row">
                      <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" /> {m.name}
                    </th>
                    <td>{fmtS(1000 * loops[i]!.circulation, 2)}</td>
                    <td>{(1000 * loops[i]!.maxGap).toFixed(2)}</td>
                    <td>{(1000 * loops[i]!.energyRange).toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="obs-scroll">
            <table className="obs-t">
              <caption>
                Rigid rotation by {doc.settings.rotation.angleDeg}° (rattled ZnO, {doc.inputs.rotation.frames[0].symbols.length} atoms)
              </caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">|ΔE| (meV)</th>
                  <th scope="col">max |F(Rx) − RF(x)| (meV/Å)</th>
                  <th scope="col">relative</th>
                </tr>
              </thead>
              <tbody>
                {doc.models.map((m, i) => (
                  <tr key={m.key}>
                    <th scope="row">
                      <span className="obs-sw" style={{ background: colors[i] }} aria-hidden="true" /> {m.name}
                    </th>
                    <td>{fmtX(1000 * rots[i]!.dE)}</td>
                    <td>{fmtX(1000 * rots[i]!.maxForceError)}</td>
                    <td>{fmtX(rots[i]!.relForceError)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
