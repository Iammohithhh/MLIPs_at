// Scene primitives for atomistic figures. Atoms are ray-cast sphere impostors:
// one camera-facing quad per atom whose fragment shader intersects the view ray
// with the exact sphere and writes the true depth — pixel-perfect spheres at any
// zoom, at the cost of two triangles per atom. Bonds and arrows are instanced
// meshes sharing the same lighting (shading.ts), so everything reads as one material system.

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { SHADING_GLSL, SHADING_UNIFORMS } from './shading';

// ---------------------------------------------------------------------------
// Shared uniforms (background, depth cueing, theme)
// ---------------------------------------------------------------------------

export interface ShadingUniforms {
  uBg: { value: THREE.Color };
  uFogNear: { value: number };
  uFogFar: { value: number };
  uDark: { value: number };
  [key: string]: THREE.IUniform;
}

export function useShadingUniforms(bg: string, dark: boolean, fogNear: number, fogFar: number): ShadingUniforms {
  const uniforms = useMemo<ShadingUniforms>(
    () => ({
      uBg: { value: new THREE.Color(bg) },
      uFogNear: { value: fogNear },
      uFogFar: { value: fogFar },
      uDark: { value: dark ? 1 : 0 },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    uniforms.uBg.value.set(bg);
    uniforms.uFogNear.value = fogNear;
    uniforms.uFogFar.value = fogFar;
    uniforms.uDark.value = dark ? 1 : 0;
    invalidate();
  }, [bg, dark, fogNear, fogFar, uniforms, invalidate]);
  return uniforms;
}

// ---------------------------------------------------------------------------
// Impostor atoms
// ---------------------------------------------------------------------------

export interface AtomBuffers {
  /** xyz per atom (world space, Å). */
  positions: Float32Array;
  radii: Float32Array;
  /** Linear RGB per atom. */
  colors: Float32Array;
  metal: Float32Array;
  /** Ambient occlusion factor per atom, 0–1. */
  ao: Float32Array;
  count: number;
}

const IMPOSTOR_VERT = /* glsl */ `
attribute vec3 aCenter;
attribute float aRadius;
attribute vec3 aColor;
attribute float aMetal;
attribute float aAO;
uniform float uRadiusScale;
varying vec3 vCenter;
varying float vRadius;
varying vec3 vPos;
varying vec3 vColor;
varying float vMetal;
varying float vAO;
void main() {
  vec4 cv = modelViewMatrix * vec4(aCenter, 1.0);
  vec3 c = cv.xyz;
  float r = aRadius * uRadiusScale;
  float d = max(length(c), 1e-3);
  vec3 fwd = c / d;
  vec3 up0 = abs(fwd.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  vec3 right = normalize(cross(fwd, up0));
  vec3 up = cross(right, fwd);
  // Radius of the silhouette cone's section through the sphere centre.
  float s = r / sqrt(max(1e-4, 1.0 - (r * r) / (d * d))) * 1.03;
  vec3 p = c + (right * position.x + up * position.y) * s;
  vCenter = c;
  vRadius = r;
  vPos = p;
  vColor = aColor;
  vMetal = aMetal;
  vAO = aAO;
  gl_Position = projectionMatrix * vec4(p, 1.0);
}
`;

const IMPOSTOR_FRAG = /* glsl */ `
uniform mat4 projectionMatrix;
${SHADING_UNIFORMS}
varying vec3 vCenter;
varying float vRadius;
varying vec3 vPos;
varying vec3 vColor;
varying float vMetal;
varying float vAO;
${SHADING_GLSL}
void main() {
  if (vRadius <= 0.0) discard;
  vec3 dir = normalize(vPos);
  float b = dot(dir, vCenter);
  float cc = dot(vCenter, vCenter) - vRadius * vRadius;
  float disc = b * b - cc;
  if (disc < 0.0) discard;
  float t = b - sqrt(disc);
  vec3 hit = dir * t;
  vec3 n = (hit - vCenter) / vRadius;
  vec4 clip = projectionMatrix * vec4(hit, 1.0);
  gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;
  vec3 col = atlasShade(vColor, n, -dir, vMetal, vAO);
  col = atlasFog(col, -hit.z);
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}
`;

export function ImpostorAtoms({
  buffers,
  version,
  uniforms,
  radiusScale = 1,
}: {
  buffers: AtomBuffers;
  /** Bump to re-upload buffers. */
  version: number;
  uniforms: ShadingUniforms;
  radiusScale?: number;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const capacity = buffers.radii.length;

  const geometry = useMemo(() => {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    g.setAttribute('aCenter', new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3));
    g.setAttribute('aRadius', new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1));
    g.setAttribute('aColor', new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3));
    g.setAttribute('aMetal', new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1));
    g.setAttribute('aAO', new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1));
    return g;
  }, [capacity]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: IMPOSTOR_VERT,
        fragmentShader: IMPOSTOR_FRAG,
        uniforms: { ...uniforms, uRadiusScale: { value: radiusScale } },
      }),
    // Uniform objects are shared by reference; radiusScale is updated below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [uniforms],
  );

  useEffect(() => {
    material.uniforms.uRadiusScale!.value = radiusScale;
    invalidate();
  }, [material, radiusScale, invalidate]);

  useLayoutEffect(() => {
    const set = (name: string, data: Float32Array) => {
      const attr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute;
      (attr.array as Float32Array).set(data.subarray(0, (attr.array as Float32Array).length));
      attr.needsUpdate = true;
    };
    set('aCenter', buffers.positions);
    set('aRadius', buffers.radii);
    set('aColor', buffers.colors);
    set('aMetal', buffers.metal);
    set('aAO', buffers.ao);
    geometry.instanceCount = buffers.count;
    invalidate();
  }, [geometry, buffers, version, invalidate]);

  useEffect(() => () => {
    geometry.dispose();
    material.dispose();
  }, [geometry, material]);

  return <mesh geometry={geometry} material={material} frustumCulled={false} />;
}

// ---------------------------------------------------------------------------
// Instanced sticks (bonds, arrow shafts) and cones (arrow heads)
// ---------------------------------------------------------------------------

const MESH_VERT = /* glsl */ `
varying vec3 vN;
varying vec3 vV;
varying vec3 vCol;
void main() {
  vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
  vN = normalize((modelViewMatrix * instanceMatrix * vec4(normal, 0.0)).xyz);
  vV = -mv.xyz;
  #ifdef USE_INSTANCING_COLOR
    vCol = instanceColor;
  #else
    vCol = vec3(0.6);
  #endif
  gl_Position = projectionMatrix * mv;
}
`;

const MESH_FRAG = /* glsl */ `
${SHADING_UNIFORMS}
varying vec3 vN;
varying vec3 vV;
varying vec3 vCol;
${SHADING_GLSL}
void main() {
  vec3 n = normalize(vN);
  vec3 v = normalize(vV);
  if (!gl_FrontFacing) n = -n;
  vec3 col = atlasShade(vCol, n, v, 0.0, 1.0);
  col = atlasFog(col, length(vV));
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}
`;

export interface Segment {
  a: [number, number, number];
  b: [number, number, number];
  /** Linear RGB. */
  color: [number, number, number];
  radius: number;
}

const Y = new THREE.Vector3(0, 1, 0);

function segmentMatrix(s: Segment, m: THREE.Matrix4, lengthScale = 1) {
  const a = new THREE.Vector3(...s.a);
  const b = new THREE.Vector3(...s.b);
  const dir = b.clone().sub(a);
  const len = dir.length();
  const q = new THREE.Quaternion().setFromUnitVectors(Y, dir.clone().normalize());
  const mid = a.clone().add(b).multiplyScalar(0.5);
  m.compose(mid, q, new THREE.Vector3(s.radius, len * lengthScale, s.radius));
}

export function Sticks({ segments, uniforms }: { segments: Segment[]; uniforms: ShadingUniforms }) {
  const invalidate = useThree((s) => s.invalidate);
  const geometry = useMemo(() => new THREE.CylinderGeometry(1, 1, 1, 18, 1, true), []);
  const material = useMemo(
    () => new THREE.ShaderMaterial({ vertexShader: MESH_VERT, fragmentShader: MESH_FRAG, uniforms, side: THREE.DoubleSide }),
    [uniforms],
  );
  const ref = useRef<THREE.InstancedMesh>(null);
  const capacity = Math.max(1, segments.length);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    segments.forEach((s, i) => {
      segmentMatrix(s, m);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, c.setRGB(...s.color));
    });
    mesh.count = segments.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    invalidate();
  }, [segments, invalidate]);

  return (
    <instancedMesh
      key={capacity}
      ref={ref}
      args={[geometry, material, capacity]}
      frustumCulled={false}
    />
  );
}

/** Arrows from a to b: a shaft (Sticks) plus a cone head of length `head`. */
export function Arrows({
  arrows,
  uniforms,
  head = 0.32,
}: {
  arrows: Segment[];
  uniforms: ShadingUniforms;
  head?: number;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const shafts = useMemo<Segment[]>(
    () =>
      arrows.map((s) => {
        const d = [s.b[0] - s.a[0], s.b[1] - s.a[1], s.b[2] - s.a[2]];
        const len = Math.hypot(d[0]!, d[1]!, d[2]!) || 1;
        const k = Math.max(0, (len - head) / len);
        return { ...s, b: [s.a[0] + d[0]! * k, s.a[1] + d[1]! * k, s.a[2] + d[2]! * k] };
      }),
    [arrows, head],
  );
  const coneGeo = useMemo(() => new THREE.ConeGeometry(1, 1, 20, 1, false), []);
  const material = useMemo(
    () => new THREE.ShaderMaterial({ vertexShader: MESH_VERT, fragmentShader: MESH_FRAG, uniforms, side: THREE.DoubleSide }),
    [uniforms],
  );
  const ref = useRef<THREE.InstancedMesh>(null);
  const capacity = Math.max(1, arrows.length);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    arrows.forEach((s, i) => {
      const a = new THREE.Vector3(...s.a);
      const b = new THREE.Vector3(...s.b);
      const dir = b.clone().sub(a).normalize();
      const base = b.clone().sub(dir.clone().multiplyScalar(head));
      const mid = base.clone().add(b).multiplyScalar(0.5);
      m.compose(mid, new THREE.Quaternion().setFromUnitVectors(Y, dir), new THREE.Vector3(s.radius * 2.4, head, s.radius * 2.4));
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, c.setRGB(...s.color));
    });
    mesh.count = arrows.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    invalidate();
  }, [arrows, head, invalidate]);

  return (
    <group>
      <Sticks segments={shafts} uniforms={uniforms} />
      <instancedMesh key={capacity} ref={ref} args={[coneGeo, material, capacity]} frustumCulled={false} />
    </group>
  );
}

// ---------------------------------------------------------------------------
// Translucent shells (cutoff spheres) with a Fresnel rim
// ---------------------------------------------------------------------------

const SHELL_VERT = /* glsl */ `
varying vec3 vN;
varying vec3 vV;
varying vec3 vObj;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  vObj = position;
  gl_Position = projectionMatrix * mv;
}
`;

const SHELL_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uGrid;
varying vec3 vN;
varying vec3 vV;
varying vec3 vObj;
void main() {
  float ndv = abs(dot(normalize(vN), normalize(vV)));
  float rim = pow(1.0 - ndv, 2.2);
  float a = uOpacity * (0.10 + 0.90 * rim);
  if (uGrid > 0.5) {
    vec3 p = normalize(vObj);
    float lat = asin(clamp(p.z, -1.0, 1.0));
    float lon = atan(p.y, p.x);
    float g = max(smoothstep(0.965, 1.0, abs(cos(lat * 9.0))), smoothstep(0.975, 1.0, abs(cos(lon * 9.0))) * (1.0 - abs(p.z)));
    a = a * 0.45 + g * uOpacity * 0.55;
  }
  gl_FragColor = vec4(uColor, a);
  #include <colorspace_fragment>
}
`;

export function Shell({
  center,
  radius,
  color,
  opacity = 0.5,
  grid = false,
  overlay = false,
}: {
  center: [number, number, number];
  radius: number;
  color: string;
  opacity?: number;
  grid?: boolean;
  /** Draw through other objects (no depth test). */
  overlay?: boolean;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: SHELL_VERT,
        fragmentShader: SHELL_FRAG,
        uniforms: { uColor: { value: new THREE.Color(color) }, uOpacity: { value: opacity }, uGrid: { value: grid ? 1 : 0 } },
        transparent: true,
        depthWrite: false,
        depthTest: !overlay,
        side: THREE.DoubleSide,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [overlay],
  );
  useEffect(() => {
    material.uniforms.uColor!.value.set(color);
    material.uniforms.uOpacity!.value = opacity;
    material.uniforms.uGrid!.value = grid ? 1 : 0;
    invalidate();
  }, [material, color, opacity, grid, invalidate]);
  return (
    <mesh position={center} scale={radius} material={material} renderOrder={overlay ? 40 : 10}>
      <sphereGeometry args={[1, 72, 48]} />
    </mesh>
  );
}

/** Translucent axis-aligned box (e.g. a summation cube), same Fresnel-free flat tint plus edges. */
export function TintBox({
  center,
  size,
  color,
  opacity = 0.12,
}: {
  center: [number, number, number];
  size: [number, number, number];
  color: string;
  opacity?: number;
}) {
  return (
    <group position={center}>
      <mesh renderOrder={10}>
        <boxGeometry args={size} />
        <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <lineSegments renderOrder={11}>
        <edgesGeometry args={[new THREE.BoxGeometry(...size)]} />
        <lineBasicMaterial color={color} transparent opacity={0.9} />
      </lineSegments>
    </group>
  );
}

// ---------------------------------------------------------------------------
// Lines: cell edges and coupling webs
// ---------------------------------------------------------------------------

export function CellBox({
  origin,
  size,
  color,
  opacity = 0.8,
}: {
  origin: [number, number, number];
  size: [number, number, number];
  color: string;
  opacity?: number;
}) {
  const geo = useMemo(() => {
    const box = new THREE.BoxGeometry(...size);
    const e = new THREE.EdgesGeometry(box);
    box.dispose();
    return e;
  }, [size[0], size[1], size[2]]);
  return (
    <lineSegments
      geometry={geo}
      position={[origin[0] + size[0] / 2, origin[1] + size[1] / 2, origin[2] + size[2] / 2]}
    >
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </lineSegments>
  );
}

export interface ColoredLine {
  a: [number, number, number];
  b: [number, number, number];
  /** Linear RGB + alpha. */
  rgba: [number, number, number, number];
}

export function Lines({ lines, overlay = false }: { lines: ColoredLine[]; /** Draw on top of everything (no depth test). */ overlay?: boolean }) {
  const invalidate = useThree((s) => s.invalidate);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(lines.length * 6);
    const col = new Float32Array(lines.length * 8);
    lines.forEach((l, i) => {
      pos.set(l.a, i * 6);
      pos.set(l.b, i * 6 + 3);
      col.set(l.rgba, i * 8);
      col.set(l.rgba, i * 8 + 4);
    });
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 4));
    return g;
  }, [lines]);
  useEffect(() => {
    invalidate();
    return () => geometry.dispose();
  }, [geometry, invalidate]);
  return (
    <lineSegments geometry={geometry} renderOrder={overlay ? 30 : 5}>
      <lineBasicMaterial vertexColors transparent depthWrite={false} depthTest={!overlay} />
    </lineSegments>
  );
}

// ---------------------------------------------------------------------------
// Picking: exact ray–sphere intersection against atom centres
// ---------------------------------------------------------------------------

export function pickSphere(
  ray: THREE.Ray,
  positions: Float32Array,
  radii: Float32Array,
  count: number,
): number | null {
  let best = -1;
  let bestT = Infinity;
  const o = ray.origin;
  const d = ray.direction;
  for (let i = 0; i < count; i++) {
    const r = radii[i]!;
    if (r <= 0) continue;
    const cx = positions[3 * i]! - o.x;
    const cy = positions[3 * i + 1]! - o.y;
    const cz = positions[3 * i + 2]! - o.z;
    const b = cx * d.x + cy * d.y + cz * d.z;
    const c = cx * cx + cy * cy + cz * cz - r * r;
    const disc = b * b - c;
    if (disc < 0) continue;
    const t = b - Math.sqrt(disc);
    if (t > 0 && t < bestT) {
      bestT = t;
      best = i;
    }
  }
  return best >= 0 ? best : null;
}

/** Click (not drag) on the canvas → index of the atom under the pointer, or null. */
export function AtomPicker({
  positions,
  radii,
  count,
  onPick,
}: {
  positions: Float32Array;
  radii: Float32Array;
  count: number;
  onPick: (i: number | null) => void;
}) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const latest = useRef({ positions, radii, count, onPick });
  latest.current = { positions, radii, count, onPick };

  useEffect(() => {
    const el = gl.domElement;
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY };
    };
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
      down = null;
      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      const rc = new THREE.Raycaster();
      rc.setFromCamera(ndc, camera);
      const { positions: p, radii: r, count: n, onPick: cb } = latest.current;
      cb(pickSphere(rc.ray, p, r, n));
    };
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointerup', onUp);
    return () => {
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointerup', onUp);
    };
  }, [gl, camera]);
  return null;
}

/** Linear RGB triple for an sRGB CSS colour. */
export function linear(css: string): [number, number, number] {
  const c = new THREE.Color(css);
  return [c.r, c.g, c.b];
}

/** Mix two linear RGB colours. */
export function mixRGB(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

// ---------------------------------------------------------------------------
// Shaded surfaces with per-vertex colour (e.g. spherical-harmonic lobes)
// ---------------------------------------------------------------------------

const SURFACE_VERT = /* glsl */ `
attribute vec3 color;
varying vec3 vN;
varying vec3 vV;
varying vec3 vCol;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  vCol = color;
  gl_Position = projectionMatrix * mv;
}
`;

const SURFACE_FRAG = /* glsl */ `
${SHADING_UNIFORMS}
varying vec3 vN;
varying vec3 vV;
varying vec3 vCol;
${SHADING_GLSL}
void main() {
  vec3 n = normalize(vN);
  vec3 v = normalize(vV);
  if (!gl_FrontFacing) n = -n;
  vec3 col = atlasShade(vCol, n, v, 0.0, 1.0);
  col = atlasFog(col, length(vV));
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}
`;

export interface SurfaceData {
  positions: Float32Array;
  /** Linear RGB per vertex. */
  colors: Float32Array;
  indices: Uint32Array;
}

export function ShadedSurface({ data, uniforms }: { data: SurfaceData; uniforms: ShadingUniforms }) {
  const invalidate = useThree((s) => s.invalidate);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    g.setAttribute('color', new THREE.BufferAttribute(data.colors, 3));
    g.setIndex(new THREE.BufferAttribute(data.indices, 1));
    g.computeVertexNormals();
    return g;
  }, [data]);
  const material = useMemo(
    () => new THREE.ShaderMaterial({ vertexShader: SURFACE_VERT, fragmentShader: SURFACE_FRAG, uniforms, side: THREE.DoubleSide }),
    [uniforms],
  );
  useEffect(() => {
    invalidate();
    return () => geometry.dispose();
  }, [geometry, invalidate]);
  return <mesh geometry={geometry} material={material} />;
}
