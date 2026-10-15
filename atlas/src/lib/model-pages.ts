// Model families that have a full monograph page. A family gets a page only when it
// meets the MACE page's bar (architecture walkthrough, equations at three depths,
// evidence, failure modes, checkpoints); the others appear as "in review" stubs.
export const MODEL_PAGES: ReadonlySet<string> = new Set(['mace', 'nequip', 'allegro']);

export const modelPageUrl = (id: string): string | null => (MODEL_PAGES.has(id) ? `/models/${id}/` : null);
