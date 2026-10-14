// A Canvas configured for Atlas figures: render-on-demand, z-up orbit camera,
// colour-managed output with no tone mapping (so the scene background matches
// the page canvas exactly), and a static message when WebGL is unavailable.

import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useEffect, useRef, type ReactNode } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';

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
}) {
  return (
    <Canvas
      flat
      frameloop="demand"
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
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
      <Controls view={view} resetKey={resetKey} minDistance={minDistance} maxDistance={maxDistance} />
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
}: {
  view: StageView;
  resetKey?: number;
  minDistance: number;
  maxDistance: number;
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
