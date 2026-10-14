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
}

const FALLBACK: SceneTheme = {
  dark: false,
  canvas: '#f9f9f7',
  surface: '#ffffff',
  ink: '#0f172a',
  inkMuted: '#475569',
  rule: '#dde1e6',
  ruleStrong: '#b9c0ca',
  arch: '#1d4ed8',
  phys: '#0f766e',
  lit: '#6a4c9c',
  caveat: '#a14a07',
  fail: '#b91c1c',
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
