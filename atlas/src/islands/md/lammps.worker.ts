/// <reference lib="webworker" />
// One LAMMPS instance (lammps.js, LAMMPS compiled to WebAssembly) per worker.
// The worker only integrates and samples; all diagnostics are computed on the
// main thread by the tested functions in ~/lib/physics/md.

import createModule from 'lammps.js';
import type { LammpsModule, LAMMPSWeb, BufferView } from 'lammps.js';
import { SAMPLE_DT, TABLE_PATH, lammpsInput, smoothTableFile, stepsPer, energyDiagnostics, type RunConfig } from '~/lib/physics/md';

export type WorkerRequest =
  | { type: 'setup'; config: RunConfig }
  | { type: 'advance'; samples: number }
  | { type: 'sweep'; treatment: RunConfig['treatment']; dts: number[]; duration: number };

export interface Sample {
  t: number;
  etotal: number;
  pe: number;
  ke: number;
  temp: number;
}

export type WorkerResponse =
  | { type: 'frame'; samples: Sample[]; positions: Float64Array; step: number }
  | { type: 'sweep-point'; dt: number; rms: number; drift: number }
  | { type: 'sweep-done' }
  | { type: 'error'; message: string };

let mod: LammpsModule | null = null;
let lmp: LAMMPSWeb | null = null;
let dt = 0.005;

async function getModule(): Promise<LammpsModule> {
  if (!mod) {
    mod = await createModule({ print: () => {}, printErr: () => {} });
    try {
      mod.FS.mkdir('/work');
    } catch {
      /* exists */
    }
    mod.FS.writeFile(TABLE_PATH, smoothTableFile());
  }
  return mod;
}

function heapFor(m: LammpsModule, view: BufferView): ArrayLike<number | bigint> {
  const S = m.ScalarType;
  const bytes = view.type === S.Float64 || view.type === S.Int64 ? 8 : 4;
  const start = view.ptr / bytes;
  const heap = view.type === S.Float64 ? m.HEAPF64 : view.type === S.Int32 ? m.HEAP32 : view.type === S.Int64 ? m.HEAP64 : m.HEAPF32;
  return heap.subarray(start, start + view.length);
}

/** Positions ordered by atom id (LAMMPS reorders atoms in memory). */
function positionsById(m: LammpsModule, l: LAMMPSWeb): Float64Array {
  const snap = l.syncParticles();
  const pos = heapFor(m, snap.positions);
  const ids = heapFor(m, snap.ids);
  const out = new Float64Array(snap.count * 3);
  for (let i = 0; i < snap.count; i++) {
    const id = Number(ids[i]) - 1;
    out[3 * id] = Number(pos[3 * i]);
    out[3 * id + 1] = Number(pos[3 * i + 1]);
    out[3 * id + 2] = Number(pos[3 * i + 2]);
  }
  return out;
}

function sample(l: LAMMPSWeb): Sample {
  return {
    t: l.getCurrentStep() * dt,
    etotal: l.getThermo('etotal'),
    pe: l.getThermo('pe'),
    ke: l.getThermo('ke'),
    temp: l.getThermo('temp'),
  };
}

function fail(l: LAMMPSWeb | null, e: unknown): never {
  const msg = l?.getLastErrorMessage() || (e instanceof Error ? e.message : String(e));
  throw new Error(msg);
}

async function setup(config: RunConfig): Promise<LAMMPSWeb> {
  const m = await getModule();
  if (lmp) {
    try {
      lmp.stop();
    } catch {
      /* already stopped */
    }
  }
  const l = new m.LAMMPSWeb();
  l.start();
  dt = config.dt;
  try {
    l.runScript(lammpsInput(config) + '\n');
  } catch (e) {
    fail(l, e);
  }
  return l;
}

function advance(l: LAMMPSWeb, samples: number): Sample[] {
  const n = stepsPer(SAMPLE_DT, dt);
  const out: Sample[] = [];
  for (let k = 0; k < samples; k++) {
    try {
      l.runCommand(`run ${n} pre no post no`);
    } catch (e) {
      fail(l, e);
    }
    out.push(sample(l));
  }
  return out;
}

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  try {
    if (req.type === 'setup') {
      lmp = await setup(req.config);
      const positions = positionsById(mod!, lmp);
      post({ type: 'frame', samples: [sample(lmp)], positions, step: 0 }, [positions.buffer]);
    } else if (req.type === 'advance') {
      if (!lmp) throw new Error('LAMMPS is not set up');
      const samples = advance(lmp, req.samples);
      const positions = positionsById(mod!, lmp);
      post({ type: 'frame', samples, positions, step: lmp.getCurrentStep() }, [positions.buffer]);
    } else if (req.type === 'sweep') {
      // Fresh runs from the same initial state at each time step; the playback
      // instance (if any) is replaced and must be set up again afterwards.
      for (const step of req.dts) {
        const l = await setup({ treatment: req.treatment, dt: step });
        lmp = l;
        const s = [sample(l), ...advance(l, Math.round(req.duration / SAMPLE_DT))];
        const d = energyDiagnostics(
          s.map((x) => x.t),
          s.map((x) => x.etotal),
        );
        post({ type: 'sweep-point', dt: step, rms: d.rms, drift: d.drift });
      }
      post({ type: 'sweep-done' });
    }
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
