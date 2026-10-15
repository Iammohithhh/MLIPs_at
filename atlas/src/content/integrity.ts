// Loads every YAML record, validates it with the shared schemas, then checks the
// rules that span records: references resolve, nothing published rests on
// unreviewed content, sources are verified against full text before publication,
// benchmark results name a protocol and metric their benchmark defines, a
// checkpoint's training level of theory is one its dataset provides, and on-site
// links point at real pages. Run by the Astro integration (build fails on any
// problem) and by the test suite.

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { parseDocument } from 'yaml';
import {
  COLLECTION_NAMES,
  ID_PATTERN,
  SCHEMAS,
  collectRefs,
  type BenchmarkResult,
  type Benchmark,
  type Checkpoint,
  type Claim,
  type CollectionName,
  type Dataset,
  type Experiment,
  type FailureCase,
  type RefUse,
  type Review,
} from './schemas';

export interface LoadedRecord {
  collection: CollectionName;
  id: string;
  file: string;
  data: Record<string, unknown> & { review: Review };
  refs: RefUse[];
}

export interface Problem {
  file: string;
  message: string;
}

export const key = (collection: CollectionName, id: string) => `${collection}/${id}`;

/** Parse YAML exactly as Astro's content layer does (YAML 1.2 core + timestamps). */
export function parseYaml(source: string): unknown {
  const doc = parseDocument(source, { customTags: ['timestamp'], merge: true, schema: 'core' });
  if (doc.errors.length > 0) throw doc.errors[0];
  return doc.toJS();
}

export interface RawRecord {
  collection: CollectionName;
  id: string;
  file: string;
  raw: unknown;
}

export function readContentDir(contentRoot: string): { raws: RawRecord[]; problems: Problem[] } {
  const raws: RawRecord[] = [];
  const problems: Problem[] = [];
  if (!existsSync(contentRoot)) return { raws, problems: [{ file: contentRoot, message: 'content directory not found' }] };
  for (const entry of readdirSync(contentRoot, { withFileTypes: true })) {
    if (entry.isDirectory() && !(COLLECTION_NAMES as readonly string[]).includes(entry.name)) {
      problems.push({ file: join(contentRoot, entry.name), message: `unknown collection "${entry.name}"` });
    }
  }
  for (const collection of COLLECTION_NAMES) {
    const dir = join(contentRoot, collection);
    if (!existsSync(dir)) continue;
    for (const f of readdirSync(dir)) {
      const file = join(dir, f);
      if (f.startsWith('.')) continue;
      if (extname(f) !== '.yaml') {
        problems.push({ file, message: 'records must be .yaml files' });
        continue;
      }
      const id = basename(f, '.yaml');
      try {
        raws.push({ collection, id, file, raw: parseYaml(readFileSync(file, 'utf8')) });
      } catch (e) {
        problems.push({ file, message: `YAML error: ${(e as Error).message}` });
      }
    }
  }
  return { raws, problems };
}

/** Schema-validate raw records and collect the references each one makes. */
export function validateRecords(raws: RawRecord[]): { records: Map<string, LoadedRecord>; problems: Problem[] } {
  const records = new Map<string, LoadedRecord>();
  const problems: Problem[] = [];
  for (const { collection, id, file, raw } of raws) {
    if (!ID_PATTERN.test(id)) {
      problems.push({ file, message: `file name "${id}" is not a kebab-case id` });
      continue;
    }
    const { result, refs } = collectRefs(() => SCHEMAS[collection].safeParse(raw));
    if (!result.success) {
      for (const issue of result.error.issues) {
        const path = issue.path.length ? issue.path.join('.') : '(record)';
        problems.push({ file, message: `${path}: ${issue.message}` });
      }
      continue;
    }
    records.set(key(collection, id), {
      collection,
      id,
      file,
      data: result.data as LoadedRecord['data'],
      refs,
    });
  }
  return { records, problems };
}

/** Does a site path like "/method/design-system/#fig-cutoff" correspond to a page file? */
export function pageExists(pagesRoot: string, sitePath: string): boolean {
  const path = sitePath.split('#')[0]!.replace(/^\/+|\/+$/g, '');
  const candidates = path === '' ? ['index'] : [path, join(path, 'index')];
  if (candidates.some((c) => ['.astro', '.md', '.mdx'].some((ext) => existsSync(join(pagesRoot, c + ext))))) return true;
  // Dynamic route one level deep, e.g. /failures/<id>/ served by pages/failures/[id].astro.
  const parts = path.split('/');
  if (parts.length < 2) return false;
  const dir = join(pagesRoot, ...parts.slice(0, -1));
  return existsSync(dir) && readdirSync(dir).some((f) => /^\[[^\]]+\]\.(astro|md|mdx)$/.test(f));
}

export function checkIntegrity(records: Map<string, LoadedRecord>, pagesRoot?: string): Problem[] {
  const problems: Problem[] = [];
  const get = <T>(c: CollectionName, id: string) => records.get(key(c, id))?.data as (T & { review: Review }) | undefined;

  for (const rec of records.values()) {
    const p = (message: string) => problems.push({ file: rec.file, message });
    const published = rec.data.review.status === 'published';

    // 1. Every reference resolves; nothing published rests on unpublished records.
    for (const r of rec.refs) {
      const target = records.get(key(r.target, r.id));
      if (!target) {
        p(`unknown ${r.target} id "${r.id}"`);
      } else if (published && target.data.review.status !== 'published') {
        p(`published record references ${r.target}/${r.id}, which is ${target.data.review.status}`);
      }
    }

    // 2. Sources must be checked against full text before publication (vision: † items).
    const verified = (rec.data as { verified_against?: string }).verified_against;
    if (published && verified && verified !== 'full-text') {
      p(`cannot be published while only verified against ${verified}`);
    }

    // 3. Collection-specific cross-record rules.
    switch (rec.collection) {
      case 'claims': {
        const c = rec.data as unknown as Claim;
        if (published) {
          c.sources.forEach((s, i) => {
            if (!s.locator?.trim()) p(`sources.${i}: published claims need a figure/table/section locator`);
          });
        }
        if (c.derivation && pagesRoot && !pageExists(pagesRoot, c.derivation)) {
          p(`derivation page "${c.derivation}" does not exist`);
        }
        break;
      }
      case 'benchmark-results': {
        const r = rec.data as unknown as BenchmarkResult;
        const b = get<Benchmark>('benchmarks', r.benchmark);
        if (b) {
          if (!b.protocols.some((x) => x.id === r.protocol)) p(`protocol "${r.protocol}" is not defined by benchmark "${r.benchmark}"`);
          if (!b.metrics.some((x) => x.id === r.metric)) p(`metric "${r.metric}" is not defined by benchmark "${r.benchmark}"`);
        }
        break;
      }
      case 'checkpoints': {
        const ck = rec.data as unknown as Checkpoint;
        ck.training.forEach((t, i) => {
          const ds = get<Dataset>('datasets', t.dataset);
          if (ds && !ds.levels_of_theory.includes(t.level_of_theory)) {
            p(`training.${i}: dataset "${t.dataset}" does not provide level of theory "${t.level_of_theory}"`);
          }
        });
        if (ck.fine_tuned_from === rec.id) p('a checkpoint cannot be fine-tuned from itself');
        break;
      }
      case 'failure-cases': {
        const f = rec.data as unknown as FailureCase;
        const main = get<Claim>('claims', f.claim);
        if (main && main.grade !== f.grade) p(`grade ${f.grade} differs from its claim's grade ${main.grade}`);
        break;
      }
      case 'experiments': {
        const e = rec.data as unknown as Experiment;
        if (pagesRoot && !pageExists(pagesRoot, e.page)) p(`page "${e.page}" does not exist`);
        break;
      }
      default:
        break;
    }
  }
  return problems;
}

/** Full pipeline: read, validate, cross-check. */
export function auditContent(contentRoot: string, pagesRoot?: string) {
  const read = readContentDir(contentRoot);
  const validated = validateRecords(read.raws);
  const integrity = checkIntegrity(validated.records, pagesRoot);
  return {
    records: validated.records,
    problems: [...read.problems, ...validated.problems, ...integrity],
  };
}

export function formatProblems(problems: Problem[], cwd = ''): string {
  return problems.map((x) => `  ✗ ${x.file.replace(cwd, '').replace(/^\//, '')}: ${x.message}`).join('\n');
}
