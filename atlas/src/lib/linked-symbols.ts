// Linked symbols: hovering or focusing any element with data-sym="key" highlights
// every element on the page that carries the same key — symbols inside KaTeX
// equations, symbol-legend chips, SVG marks in figures — and broadcasts the key so
// canvas/3D islands (which have no DOM per object) can highlight their counterpart.

export const SYMBOL_EVENT = 'atlas:symbol';
const ACTIVE_CLASS = 'sym-active';

export interface SymbolEventDetail {
  /** The active symbol key, or null when nothing is highlighted. */
  sym: string | null;
  /** Where the change came from, so an island can ignore its own echoes. */
  source: string;
}

let current: string | null = null;

function escapeAttr(value: string): string {
  return typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(value) : value.replace(/["\\]/g, '\\$&');
}

/** Highlight `sym` everywhere (null clears) and notify listeners. */
export function setActiveSymbol(sym: string | null, source = 'dom'): void {
  if (sym === current) return;
  if (current !== null) {
    document
      .querySelectorAll(`[data-sym="${escapeAttr(current)}"]`)
      .forEach((el) => el.classList.remove(ACTIVE_CLASS));
  }
  current = sym;
  if (sym !== null) {
    document.querySelectorAll(`[data-sym="${escapeAttr(sym)}"]`).forEach((el) => el.classList.add(ACTIVE_CLASS));
  }
  document.dispatchEvent(
    new CustomEvent<SymbolEventDetail>(SYMBOL_EVENT, { detail: { sym, source } }),
  );
}

export function getActiveSymbol(): string | null {
  return current;
}

/** Subscribe to symbol changes; returns an unsubscribe function. */
export function onSymbolChange(handler: (detail: SymbolEventDetail) => void): () => void {
  const listener = (e: Event) => handler((e as CustomEvent<SymbolEventDetail>).detail);
  document.addEventListener(SYMBOL_EVENT, listener);
  return () => document.removeEventListener(SYMBOL_EVENT, listener);
}

function symbolOf(target: EventTarget | null): string | null {
  if (!(target instanceof Element)) return null;
  const el = target.closest('[data-sym]');
  return el?.getAttribute('data-sym') ?? null;
}

let installed = false;

/** Install document-level delegation once; works for islands rendered later. */
export function installLinkedSymbols(): void {
  if (installed || typeof document === 'undefined') return;
  installed = true;
  document.addEventListener('pointerover', (e) => {
    const sym = symbolOf(e.target);
    if (sym !== null) setActiveSymbol(sym);
  });
  document.addEventListener('pointerout', (e) => {
    const from = symbolOf(e.target);
    const to = symbolOf(e.relatedTarget);
    if (from !== null && to !== from) setActiveSymbol(to);
  });
  document.addEventListener('focusin', (e) => {
    const sym = symbolOf(e.target);
    if (sym !== null) setActiveSymbol(sym);
  });
  document.addEventListener('focusout', (e) => {
    if (symbolOf(e.target) !== null && symbolOf(e.relatedTarget) === null) setActiveSymbol(null);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setActiveSymbol(null);
  });
}
