import { describe, expect, it } from 'vitest';
import { correctionUrl } from './contribute';

describe('contribution links', () => {
  it('pre-fill the correction form with the record id and the quoted statement', () => {
    const u = new URL(correctionUrl('claims', 'x-y', 'A statement.'));
    expect(u.pathname).toBe('/Iammohithhh/MLIPs_at/issues/new');
    expect(u.searchParams.get('template')).toBe('correction.yml');
    expect(u.searchParams.get('record')).toBe('claims/x-y');
    expect(u.searchParams.get('problem')).toContain('> A statement.');
  });
});
