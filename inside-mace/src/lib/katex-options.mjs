// Copied from atlas/src/lib/katex-options.mjs (MLIP Atlas) for inside-mace; keep in sync deliberately, not automatically.
// Shared KaTeX configuration for build-time maths (Markdown/MDX via rehype-katex,
// and the <Math> component). Plain .mjs so astro.config.mjs can import it.
//
// Linked symbols: write \sym{key}{tex} in an equation. It renders `tex` wrapped in
// a span carrying data-sym="key"; hovering it highlights every element on the page
// with the same data-sym (other equations, legends, figure marks, 3D scene objects).

/** Only the \htmlData extension is trusted, and only for data-sym. */
function trust(context) {
  return context.command === '\\htmlData';
}

/** Fail the build on any LaTeX-incompatible input except our own \htmlData use. */
function strict(errorCode) {
  return errorCode === 'htmlExtension' ? 'ignore' : 'error';
}

/** @type {import('katex').KatexOptions} */
export const katexOptions = {
  output: 'htmlAndMathml',
  throwOnError: true,
  strict,
  trust,
  macros: {
    '\\sym': '\\htmlData{sym=#1}{#2}',
    '\\vect': '\\mathbf{#1}',
    '\\dd': '\\mathrm{d}',
    '\\ee': '\\mathrm{e}',
  },
};

const LOST_ESCAPE =
  /(^|[^\\A-Za-z])(sym|frac|dfrac|tfrac|mathrm|mathbf|mathcal|vect|sqrt|left|right|partial|nabla|sum|int|cdot|alpha|beta|epsilon|varepsilon|rho|phi|varphi)\{?(?![A-Za-z])/;

/**
 * Throw if TeX was mangled by string-escape processing, e.g. tex="\vect{F}"
 * in an Astro attribute turns "\v" into a vertical tab and "\s" into "s".
 * Pass TeX as tex={String.raw`…`} instead.
 * @param {string} tex
 */
export function assertTexIntact(tex) {
  // Control characters other than newline mean an escape like \b, \f, \t, \v or \r was consumed.
  if (/[\u0000-\u0009\u000b-\u001f]/.test(tex)) {
    throw new Error(`TeX contains a control character (a consumed escape?): ${JSON.stringify(tex)}. Use String.raw.`);
  }
  const m = tex.match(LOST_ESCAPE);
  if (m) {
    throw new Error(`TeX command "${m[2]}" has lost its backslash in ${JSON.stringify(tex)}. Use String.raw.`);
  }
}
