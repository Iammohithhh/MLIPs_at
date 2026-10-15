// Read the Atlas design tokens (CSS custom properties) for use in WebGL scenes,
// and follow theme changes (the toggle sets data-theme on <html>).

import { useEffect, useState } from 'react';

export interface SceneTheme {
  dark: boolean;
  canvas: string;
  surface: string;
  ink: string;
  inkMuted: string;
  rule: string;
  ruleStrong: string;
  arch: string;
  phys: string;
  lit: string;
  caveat: string;
  fail: string;
  verified: string;
}

const FALLBACK: SceneTheme = {
  dark: false,
  canvas: '#f6f4ef',
  surface: '#fbfaf7',
  ink: '#2b323c',
  inkMuted: '#545c67',
  rule: '#e2ded5',
  ruleStrong: '#c9c4b9',
  arch: '#3f5d96',
  phys: '#3a7069',
  lit: '#6b5a8c',
  caveat: '#8c5f25',
  fail: '#9f4740',
  verified: '#3f7a5c',
};

function read(): SceneTheme {
  if (typeof document === 'undefined') return FALLBACK;
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
  return {
    dark: document.documentElement.dataset.theme === 'dark',
    canvas: v('--atlas-canvas', FALLBACK.canvas),
    surface: v('--atlas-surface', FALLBACK.surface),
    ink: v('--atlas-ink', FALLBACK.ink),
    inkMuted: v('--atlas-ink-muted', FALLBACK.inkMuted),
    rule: v('--atlas-rule', FALLBACK.rule),
    ruleStrong: v('--atlas-rule-strong', FALLBACK.ruleStrong),
    arch: v('--atlas-arch', FALLBACK.arch),
    phys: v('--atlas-phys', FALLBACK.phys),
    lit: v('--atlas-lit', FALLBACK.lit),
    caveat: v('--atlas-caveat', FALLBACK.caveat),
    fail: v('--atlas-fail', FALLBACK.fail),
    verified: v('--atlas-verified', FALLBACK.verified),
  };
}

export function useSceneTheme(): SceneTheme {
  const [theme, setTheme] = useState<SceneTheme>(read);
  useEffect(() => {
    setTheme(read());
    const obs = new MutationObserver(() => setTheme(read()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);
  return theme;
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const on = () => setReduced(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return reduced;
}
