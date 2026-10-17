// Copied from atlas/src/three/Stage.tsx (MLIP Atlas) for inside-mace; keep in sync deliberately, not automatically.
// inside-mace changes: `preserveDrawingBuffer` (for still-image export); optional `sync` (cameraSync.ts) links the orbit cameras of side-by-side Stages.
// A Canvas configured for Atlas figures: render-on-demand, z-up orbit camera,
// colour-managed output with no tone mapping (so the scene background matches
// the page canvas exactly), and a static message when WebGL is unavailable.

import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useEffect, useRef, type ReactNode } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { CameraSync } from './cameraSync';

export interface StageView {
  position: [number, number, number];
  target: [number, number, number];
}

export function Stage({
  view,
  resetKey,
  background,
  label,
  children,
  fov = 30,
  minDistance = 8,
  maxDistance = 140,
  sync,
  preserveDrawingBuffer = false,
}: {
  view: StageView;
  /** Change to animate back to `view`. */
  resetKey?: number;
  background: string;
  /** Accessible description of the scene. */
  label: string;
  children: ReactNode;
  fov?: number;
  minDistance?: number;
  maxDistance?: number;
  sync?: CameraSync;
  /** Keep the last frame readable so the canvas can be exported as an image. */
  preserveDrawingBuffer?: boolean;
}) {
  return (
    <Canvas
      flat
      frameloop="demand"
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer }}
      camera={{ position: view.position, fov, near: 0.5, far: 600, up: [0, 0, 1] }}
      onCreated={({ scene, camera }) => {
        scene.background = new THREE.Color(background);
        camera.up.set(0, 0, 1);
      }}
      fallback={<p className="stage-fallback">This figure needs WebGL, which is not available in this browser.</p>}
      aria-label={label}
      role="img"
    >
      <Background color={background} />
      <Controls view={view} resetKey={resetKey} minDistance={minDistance} maxDistance={maxDistance} sync={sync} />
      {children}
    </Canvas>
  );
}


function Background({ color }: { color: string }) {
  const scene = useThree((s) => s.scene);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    scene.background = new THREE.Color(color);
    invalidate();
  }, [scene, color, invalidate]);
  return null;
}

function Controls({
  view,
  resetKey,
  minDistance,
  maxDistance,
  sync,
}: {
  view: StageView;
  resetKey?: number;
  minDistance: number;
  maxDistance: number;
  sync?: CameraSync;
}) {
  const ref = useRef<OrbitControlsImpl>(null);
  const camera = useThree((s) => s.camera);
  const invalidate = useThree((s) => s.invalidate);
  const first = useRef(true);

  useEffect(() => {
    const controls = ref.current;
    if (!controls) return;
    if (first.current) {
      first.current = false;
      controls.target.set(...view.target);
      controls.update();
      return;
    }
    camera.position.set(...view.position);
    controls.target.set(...view.target);
    controls.update();
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey]);

  useEffect(() => {
    const controls = ref.current;
    if (!sync || !controls) return;
    let applying = false;
    const sub = sync.subscribe((p) => {
      applying = true;
      camera.position.set(...p.position);
      controls.target.set(...p.target);
      controls.update();
      applying = false;
      invalidate();
    });
    const onChange = () => {
      if (applying) return;
      sync.publish(sub.id, { position: camera.position.toArray() as [number, number, number], target: controls.target.toArray() as [number, number, number] });
    };
    controls.addEventListener('change', onChange);
    return () => {
      controls.removeEventListener('change', onChange);
      sub.off();
    };
  }, [sync, camera, invalidate]);

  return (
    <OrbitControls
      ref={ref}
      makeDefault
      enableDamping
      dampingFactor={0.12}
      rotateSpeed={0.7}
      minDistance={minDistance}
      maxDistance={maxDistance}
      target={view.target}
    />
  );
}
