import { describe, expect, it } from 'vitest';
import { heroHref } from './heroLink';

describe('hero actions keep the visitor’s context', () => {
  it('keeps experiment, frame and view; sets the mode', () => {
    expect(heroHref('exp=zno-axial-bond&frame=20&view=difference&mode=lab', 'walk')).toBe('?exp=zno-axial-bond&frame=20&view=difference&mode=walk&stage=species#lab');
  });
  it('keeps an existing stage and drops presentation keys', () => {
    expect(heroHref('mode=walk&stage=forces&present=1&demo=3', 'lab')).toBe('?mode=lab&stage=forces#lab');
  });
  it('works from an empty address', () => {
    expect(heroHref('', 'lab')).toBe('?mode=lab#lab');
  });
});
