// Model families that have a full monograph page. A family gets a page only when it
// meets the MACE page's bar (architecture walkthrough, equations at three depths,
// evidence, failure modes, checkpoints); the others appear as "in review" stubs.
// Closely related families can share one page (value = page slug).
const PAGES: ReadonlyMap<string, string> = new Map([
  ['mace', 'mace'],
  ['nequip', 'nequip'],
  ['allegro', 'allegro'],
  ['behler-parrinello', 'behler-parrinello'],
  ['schnet', 'schnet'],
  ['m3gnet', 'm3gnet-chgnet'],
  ['chgnet', 'm3gnet-chgnet'],
  ['pet', 'pet'],
  ['orb', 'orb'],
  ['esen', 'esen-uma'],
  ['uma', 'esen-uma'],
  ['grace', 'grace'],
  ['sevennet', 'sevennet'],
  ['dpa', 'dpa'],
]);

export const MODEL_PAGES: ReadonlySet<string> = new Set(PAGES.keys());

export const modelPageUrl = (id: string): string | null => {
  const slug = PAGES.get(id);
  return slug ? `/models/${slug}/` : null;
};
