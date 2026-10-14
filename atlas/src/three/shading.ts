// Shared GLSL for every Atlas surface, so impostor atoms, bonds and arrows are lit
// identically. Lighting is defined in view space (a studio rig that moves with
// the camera): a key light from the upper left, a weak fill from the right, a
// hemisphere ambient term, Blinn–Phong highlights, a fake sky/floor reflection
// for metals, silhouette darkening for outlines and depth cueing towards the
// page background. Colours are linear; the renderer converts to sRGB.

export const SHADING_UNIFORMS = /* glsl */ `
uniform vec3 uBg;
uniform float uFogNear;
uniform float uFogFar;
uniform float uDark;
`;

export const SHADING_GLSL = /* glsl */ `
vec3 atlasShade(vec3 base, vec3 n, vec3 v, float metal, float ao) {
  vec3 L1 = normalize(vec3(-0.45, 0.70, 0.55));
  vec3 L2 = normalize(vec3(0.70, -0.15, 0.45));
  float d1 = max(dot(n, L1), 0.0);
  float d2 = max(dot(n, L2), 0.0);
  float hemi = 0.5 + 0.5 * n.y;
  vec3 amb = mix(vec3(0.20, 0.21, 0.25), vec3(0.58, 0.60, 0.66), hemi) * mix(1.0, 0.8, uDark);
  float ndv = clamp(dot(n, v), 0.0, 1.0);

  vec3 diffuse = base * (amb * ao + (d1 * 0.92 + d2 * 0.22));

  // Metals: a soft "studio" environment (bright sky, darker floor) tinted by the base colour.
  vec3 r = reflect(-v, n);
  vec3 env = mix(base * 0.30, base * 1.30 + 0.04, smoothstep(-0.35, 0.65, r.y));
  vec3 col = mix(diffuse, env * (0.55 + 0.45 * ao) + diffuse * 0.25, metal * 0.7);

  vec3 h = normalize(L1 + v);
  float spec = pow(max(dot(n, h), 0.0), mix(64.0, 26.0, metal));
  vec3 specCol = mix(vec3(1.0), base * 1.5 + 0.1, metal);
  col += specCol * spec * mix(0.32, 0.85, metal);

  // Thin rim light and silhouette darkening (reads as an outline at any zoom).
  col += pow(1.0 - ndv, 4.0) * 0.05;
  col *= mix(0.40, 1.0, smoothstep(0.0, 0.30, ndv));
  return col;
}

vec3 atlasFog(vec3 col, float viewDepth) {
  float f = smoothstep(uFogNear, uFogFar, viewDepth);
  return mix(col, uBg, f * 0.70);
}
`;
