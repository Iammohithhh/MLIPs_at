// Number formatting for displayed values: explicit sign, true minus sign (U+2212), missing shown as "—".
// Pure module, covered by format.test.ts.

/** Signed value with `digits` decimals; magnitudes below 10^−digits switch to exponent notation. */
export function fmtSigned(x: number | null | undefined, digits = 3, unit = ''): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—';
  const a = Math.abs(x);
  // format the magnitude, so the only minus signs left are inside the exponent
  const s = a < 10 ** -digits && a !== 0 ? a.toExponential(2).replace('e-', 'e−').replace('e+', 'e') : a.toFixed(digits);
  return `${x > 0 ? '+' : x < 0 ? '−' : ''}${s}${unit ? ` ${unit}` : ''}`;
}
