// Copied from atlas/src/three/Stage.tsx (MLIP Atlas) for inside-mace; keep in sync deliberately, not automatically.
// inside-mace changes: `preserveDrawingBuffer` (for still-image export); optional `sync` (cameraSync.ts) links the orbit cameras of side-by-side Stages;
// rendering pauses (frameloop 'never') while the canvas is off screen or the tab is hidden, and redraws once on return.
// A Canvas configured for Atlas figures: render-on-demand, z-up orbit camera,
// colour-managed output with no tone mapping (so the scene background matches
// the page canvas exactly), and a static message when WebGL is unavailable.

import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import * as THREE from 'three';
import { gsap } from 'gsap';
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
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const visible = useOnScreen(canvasRef);
  return (
    <Canvas
      ref={canvasRef}
      data-rendering={visible ? 'on-demand' : 'paused'}
      flat
      frameloop={visible ? 'demand' : 'never'}
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
      <RedrawOnReturn visible={visible} />
      <Controls view={view} resetKey={resetKey} minDistance={minDistance} maxDistance={maxDistance} sync={sync} />
      {children}
    </Canvas>
  );
}


/** True while the element intersects the viewport and the document is visible (brief §15: pause hidden work). */
function useOnScreen(ref: React.RefObject<HTMLElement | null>): boolean {
  const [inView, setInView] = useState(true);
  const [pageVisible, setPageVisible] = useState(true);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver((entries) => setInView(entries.some((e) => e.isIntersecting)), { rootMargin: '100px' });
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);
  useEffect(() => {
    const on = () => setPageVisible(document.visibilityState !== 'hidden');
    on();
    document.addEventListener('visibilitychange', on);
    return () => document.removeEventListener('visibilitychange', on);
  }, []);
  return inView && pageVisible;
}

/** State may change while paused (e.g. the slider below a scrolled-away scene); draw once when visible again. */
function RedrawOnReturn({ visible }: { visible: boolean }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    if (visible) invalidate();
  }, [visible, invalidate]);
  return null;
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
    // inside-mace: animate the preset change with GSAP (orientation stays z-up), or jump under reduced motion.
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      camera.position.set(...view.position);
      controls.target.set(...view.target);
      controls.update();
      invalidate();
      return;
    }
    const tl = gsap.timeline({
      onUpdate: () => {
        controls.update();
        invalidate();
      },
    });
    tl.to(camera.position, { x: view.position[0], y: view.position[1], z: view.position[2], duration: 0.7, ease: 'power2.inOut' }, 0);
    tl.to(controls.target, { x: view.target[0], y: view.target[1], z: view.target[2], duration: 0.7, ease: 'power2.inOut' }, 0);
    return () => {
      tl.kill();
    };
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
