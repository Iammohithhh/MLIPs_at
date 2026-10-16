import { useEffect, useMemo, useState } from 'react';
import { Stage } from '~/three/Stage';
import { Arrows, ImpostorAtoms, Lines, Shell, linear, mixRGB, useShadingUniforms, type AtomBuffers, type ColoredLine } from '~/three/primitives';
import { useSceneTheme, type SceneTheme } from '~/three/theme';
import { ELEMENTS as TABLE } from '~/lib/structure/elements.generated';

// Endpoint geometry of one literature migration path (Zenodo Dataset-2 of Bheemaguli et al.), written by
// pipeline/atlas_pipeline/migration_barriers.py to public/data/migration-paths.json and loaded on demand.

export interface HopPath {
  id: string;
  lattice: number[][];
  elements: string[];
  positions: number[][];
  mover: number;
  end: number[];
  hop: number;
}

const BY_SYMBOL = new Map(TABLE.map((e) => [e.symbol, e]));
let cache: Promise<Map<string, HopPath>> | null = null;
export function loadPaths(): Promise<Map<string, HopPath>> {
  cache ??= fetch('/data/migration-paths.json')
    .then((r) => r.json() as Promise<{ paths: HopPath[] }>)
    .then((d) => new Map(d.paths.map((p) => [p.id, p])));
  return cache;
}

export function useHopPath(id: string | null) {
  const [path, setPath] = useState<HopPath | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!id) return;
    let live = true;
    loadPaths()
      .then((m) => live && setPath(m.get(id) ?? null))
      .catch(() => live && setError(true));
    return () => {
      live = false;
    };
  }, [id]);
  return { path: path && path.id === id ? path : null, error };
}

export default function HopViewer({ path, label }: { path: HopPath; label: string }) {
  const theme = useSceneTheme();
  const L = path.lattice;
  const centre = [0, 1, 2].map((j) => (L[0]![j]! + L[1]![j]! + L[2]![j]!) / 2) as [number, number, number];
  const size = Math.max(...[0, 1, 2].map((i) => Math.hypot(...(L[i] as [number, number, number]))));
  return (
    <Stage key={path.id} view={{ position: [size * 1.6, -size * 2.4, size * 1.2], target: [0, 0, 0] }} background={theme.canvas} label={label} minDistance={size * 0.6} maxDistance={size * 6}>
      <HopScene theme={theme} path={path} centre={centre} />
    </Stage>
  );
}

function HopScene({ theme, path, centre }: { theme: SceneTheme; path: HopPath; centre: [number, number, number] }) {
  const uniforms = useShadingUniforms(theme.canvas, theme.dark, 20, 200);
  const sh = (p: number[]): [number, number, number] => [p[0]! - centre[0], p[1]! - centre[1], p[2]! - centre[2]];
  const buffers = useMemo<AtomBuffers>(() => {
    const n = path.elements.length;
    const pos = new Float32Array(n * 3);
    const radii = new Float32Array(n);
    const colors = new Float32Array(n * 3);
    const moverEl = path.elements[path.mover];
    const fade = linear(theme.canvas);
    path.elements.forEach((el, i) => {
      const row = BY_SYMBOL.get(el);
      pos.set(sh(path.positions[i]!), 3 * i);
      const r = row?.covalentRadius ?? 1.2;
      const same = el === moverEl;
      radii[i] = i === path.mover ? 1.1 : Math.max(0.28, Math.min(0.8, r * 0.5));
      const c = linear(row?.color ?? '#999999');
      colors.set(i === path.mover ? linear(theme.caveat) : same ? c : mixRGB(c, fade, 0.45), 3 * i);
    });
    return { positions: pos, radii, colors, metal: new Float32Array(n), ao: new Float32Array(n).fill(1), count: n };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, theme]);
  const box = useMemo<ColoredLine[]>(() => {
    const [a, b, c] = path.lattice as [number[], number[], number[]];
    const o = sh([0, 0, 0]);
    const add = (p: number[], q: number[]) => [p[0]! + q[0]!, p[1]! + q[1]!, p[2]! + q[2]!];
    const v = [[0, 0, 0], a, b, c, add(a, b), add(a, c), add(b, c), add(add(a, b), c)].map((p) => [p[0]! + o[0], p[1]! + o[1], p[2]! + o[2]] as [number, number, number]);
    const e: Array<[number, number]> = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]];
    const col = linear(theme.inkMuted);
    return e.map(([i, j]) => ({ a: v[i]!, b: v[j]!, rgba: [col[0], col[1], col[2], 0.5] }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, theme.inkMuted]);
  const start = sh(path.positions[path.mover]!);
  const end = sh(path.end);
  return (
    <>
      <ImpostorAtoms buffers={buffers} version={0} uniforms={uniforms} />
      <Lines lines={box} />
      <Arrows arrows={[{ a: start, b: end, color: linear(theme.caveat), radius: 0.12 }]} uniforms={uniforms} head={0.45} />
      <Shell center={end} radius={1.1} color={theme.caveat} opacity={0.35} overlay />
    </>
  );
}
