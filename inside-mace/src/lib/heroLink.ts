// The hero's two actions ("Move an atom", "Walk me through it") switch the instrument's mode. Without
// JavaScript they are plain links (`?mode=lab`, `?mode=walk&stage=species`); with it, they keep the
// visitor's current experiment, atom, frame, view and channel from the address bar, so switching modes
// preserves context (brief §2). Presentation-mode keys are dropped: the actions lead to the full instrument.

export function heroHref(currentQuery: string, mode: 'lab' | 'walk'): string {
  const p = new URLSearchParams(currentQuery);
  p.delete('present');
  p.delete('demo');
  p.set('mode', mode);
  if (mode === 'walk' && !p.has('stage')) p.set('stage', 'species');
  return `?${p.toString()}#lab`;
}
