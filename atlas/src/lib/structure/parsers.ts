// Client-side parsers for uploaded structures. Files never leave the browser.
// Supported: (extended) XYZ and VASP POSCAR/CONTCAR (VASP 5 format with species
// line; Cartesian or Direct coordinates; selective dynamics flags ignored).
import { fracToCart } from './geometry';
import { structureSchema } from './schema';
import type { Mat3, Structure, Vec3 } from './types';

export const MAX_UPLOAD_ATOMS = 5000;

export class ParseError extends Error {}

const num = (s: string | undefined, what: string): number => {
  const v = Number(s);
  if (s === undefined || !Number.isFinite(v)) throw new ParseError(`Expected a number for ${what}, got "${s ?? ''}"`);
  return v;
};

function slug(name: string): string {
  return (
    name
      .toLowerCase()
      .replace(/\.[a-z0-9]+$/, '')
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'upload'
  );
}

/** Parse key=value pairs from an extended-XYZ comment line (quoted values allowed). */
export function parseExtxyzInfo(line: string): Record<string, string> {
  const out: Record<string, string> = {};
  const re = /([A-Za-z_][\w-]*)\s*=\s*("([^"]*)"|'([^']*)'|(\S+))/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line))) out[m[1]!.toLowerCase()] = m[3] ?? m[4] ?? m[5] ?? '';
  return out;
}

export function parseXYZ(text: string, name = 'upload.xyz'): Structure {
  const lines = text.replace(/\r/g, '').split('\n');
  const n = Math.round(num(lines[0]?.trim(), 'the atom count on line 1'));
  if (n < 1) throw new ParseError('The atom count must be at least 1');
  if (n > MAX_UPLOAD_ATOMS) throw new ParseError(`Too many atoms (${n}); the limit is ${MAX_UPLOAD_ATOMS}`);
  const info = parseExtxyzInfo(lines[1] ?? '');
  let cell: Mat3 | null = null;
  let pbc: [boolean, boolean, boolean] = [false, false, false];
  if (info['lattice']) {
    const v = info['lattice'].trim().split(/\s+/).map((x) => num(x, 'Lattice'));
    if (v.length !== 9) throw new ParseError('Lattice must have 9 numbers');
    cell = [
      [v[0]!, v[1]!, v[2]!],
      [v[3]!, v[4]!, v[5]!],
      [v[6]!, v[7]!, v[8]!],
    ];
    pbc = [true, true, true];
  }
  if (info['pbc']) {
    const f = info['pbc'].trim().split(/\s+/).map((x) => /^(t|true|1)$/i.test(x));
    if (f.length === 3) pbc = [f[0]!, f[1]!, f[2]!];
  }
  const symbols: string[] = [];
  const positions: Vec3[] = [];
  for (let k = 0; k < n; k++) {
    const parts = (lines[2 + k] ?? '').trim().split(/\s+/);
    if (parts.length < 4) throw new ParseError(`Line ${3 + k}: expected "symbol x y z"`);
    symbols.push(normaliseSymbol(parts[0]!));
    positions.push([num(parts[1], `x on line ${3 + k}`), num(parts[2], `y on line ${3 + k}`), num(parts[3], `z on line ${3 + k}`)]);
  }
  if (!pbc.some(Boolean)) cell = null;
  return finish({ id: slug(name), title: `Uploaded: ${name}`, symbols, positions, cell, pbc }, name);
}

export function parsePOSCAR(text: string, name = 'POSCAR'): Structure {
  const lines = text.replace(/\r/g, '').split('\n');
  const scale = num(lines[1]?.trim().split(/\s+/)[0], 'the scale factor on line 2');
  if (scale <= 0) throw new ParseError('Negative (volume) scale factors are not supported');
  const vec = (i: number): Vec3 => {
    const p = (lines[i] ?? '').trim().split(/\s+/);
    return [num(p[0], `lattice line ${i + 1}`) * scale, num(p[1], `lattice line ${i + 1}`) * scale, num(p[2], `lattice line ${i + 1}`) * scale];
  };
  const cell: Mat3 = [vec(2), vec(3), vec(4)];
  const species = (lines[5] ?? '').trim().split(/\s+/);
  if (species.some((s) => /^\d+$/.test(s))) throw new ParseError('VASP 4 files without a species line are not supported');
  const counts = (lines[6] ?? '').trim().split(/\s+/).map((c) => Math.round(num(c, 'species counts on line 7')));
  if (counts.length !== species.length) throw new ParseError('Species and count lines differ in length');
  const total = counts.reduce((a, b) => a + b, 0);
  if (total > MAX_UPLOAD_ATOMS) throw new ParseError(`Too many atoms (${total}); the limit is ${MAX_UPLOAD_ATOMS}`);
  let line = 7;
  if (/^s/i.test((lines[line] ?? '').trim())) line++; // Selective dynamics
  const mode = (lines[line] ?? '').trim();
  const direct = /^d/i.test(mode);
  if (!direct && !/^[ck]/i.test(mode)) throw new ParseError(`Line ${line + 1}: expected Direct or Cartesian`);
  line++;
  const symbols: string[] = [];
  const positions: Vec3[] = [];
  species.forEach((sp, s) => {
    for (let k = 0; k < counts[s]!; k++) {
      const p = (lines[line] ?? '').trim().split(/\s+/);
      const v: Vec3 = [num(p[0], `coordinates on line ${line + 1}`), num(p[1], `coordinates on line ${line + 1}`), num(p[2], `coordinates on line ${line + 1}`)];
      positions.push(direct ? fracToCart(v, cell) : [v[0] * scale, v[1] * scale, v[2] * scale]);
      symbols.push(normaliseSymbol(sp));
      line++;
    }
  });
  return finish(
    { id: slug(name), title: `Uploaded: ${name}`, symbols, positions, cell, pbc: [true, true, true] },
    name,
  );
}

function normaliseSymbol(s: string): string {
  const m = /^([A-Za-z]{1,2})/.exec(s);
  if (!m) throw new ParseError(`Not an element symbol: "${s}"`);
  const t = m[1]!;
  return t[0]!.toUpperCase() + (t[1] ?? '').toLowerCase();
}

function finish(s: Structure, name: string): Structure {
  const r = structureSchema.safeParse(s);
  if (!r.success) throw new ParseError(r.error.issues.map((i) => i.message).join('; '));
  return { ...s, provenance: { source: `uploaded file ${name} (parsed in your browser; not stored)` } };
}

/** Pick a parser from the file name (and content, for extension-less POSCAR/CONTCAR). */
export function parseStructureFile(name: string, text: string): Structure {
  if (/\.(ext)?xyz$/i.test(name)) return parseXYZ(text, name);
  if (/poscar|contcar|\.vasp$/i.test(name)) return parsePOSCAR(text, name);
  if (/^\s*\d+\s*$/.test(text.split('\n')[0] ?? '')) return parseXYZ(text, name);
  return parsePOSCAR(text, name);
}
