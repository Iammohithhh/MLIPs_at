import { describe, expect, it } from 'vitest';
import { neighboursOf } from './geometry';
import { ParseError, parseExtxyzInfo, parsePOSCAR, parseStructureFile, parseXYZ } from './parsers';

describe('extended XYZ', () => {
  it('reads plain XYZ as an open molecule', () => {
    const s = parseXYZ('3\nwater\nO 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0\n', 'water.xyz');
    expect(s.symbols).toEqual(['O', 'H', 'H']);
    expect(s.cell).toBeNull();
    expect(s.id).toBe('water');
  });

  it('reads Lattice and pbc from the comment line', () => {
    const s = parseXYZ(
      '2\nLattice="5.64 0 0 0 5.64 0 0 0 5.64" Properties=species:S:1:pos:R:3 pbc="T T F"\nNa 0 0 0\nCl 2.82 0 0\n',
      'nacl.extxyz',
    );
    expect(s.cell![0]![0]).toBe(5.64);
    expect(s.pbc).toEqual([true, true, false]);
  });

  it('parses quoted and unquoted key=value pairs', () => {
    expect(parseExtxyzInfo('energy=-1.5 config_type="bulk md" pbc=\'T T T\'')).toEqual({
      energy: '-1.5',
      config_type: 'bulk md',
      pbc: 'T T T',
    });
  });

  it('rejects malformed files with a useful message', () => {
    expect(() => parseXYZ('two\n\n')).toThrow(ParseError);
    expect(() => parseXYZ('1\n\nXx 0 0 0\n')).toThrow(/unknown element/);
    expect(() => parseXYZ('1\n\nH 0 zero 0\n')).toThrow(/line 3/);
    expect(() => parseXYZ('9999\n\n')).toThrow(/limit/);
  });
});

describe('POSCAR', () => {
  const poscar = `NaCl
1.0
5.64 0 0
0 5.64 0
0 0 5.64
Na Cl
4 4
Direct
0 0 0
0 0.5 0.5
0.5 0 0.5
0.5 0.5 0
0.5 0 0
0.5 0.5 0.5
0 0 0.5
0 0.5 0
`;
  it('reads Direct coordinates into a periodic structure with correct geometry', () => {
    const s = parsePOSCAR(poscar);
    expect(s.symbols.filter((x) => x === 'Cl')).toHaveLength(4);
    expect(neighboursOf(s, 0, 3)).toHaveLength(6);
  });

  it('applies the scale factor and skips selective dynamics', () => {
    const scaled = poscar.replace('1.0', '2.0').replace(/5\.64/g, '2.82').replace('Direct', 'Selective dynamics\nDirect');
    const s = parsePOSCAR(scaled);
    expect(s.cell![0]![0]).toBeCloseTo(5.64, 12);
    expect(neighboursOf(s, 0, 3)).toHaveLength(6);
  });

  it('is chosen for CONTCAR names', () => {
    expect(parseStructureFile('CONTCAR', poscar).cell).not.toBeNull();
  });
});
