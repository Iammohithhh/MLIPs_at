import { elementColor } from '~/lib/structure/elements';

/** CSS colour of an element's atom in the 3D scenes, for legends. */
export function cssColor(symbol: string, dark: boolean): string {
  const c = elementColor(symbol, dark ? 'dark' : 'light');
  return `rgb(${c.map((v) => Math.round(255 * v)).join(' ')})`;
}
