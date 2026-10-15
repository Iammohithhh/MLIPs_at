// Content model for every Atlas entity (vision §22–24). One YAML file per record
// in atlas/content/<collection>/<id>.yaml; the file name is the record id.
//
//   Concept ──explains──> PhysicalPrinciple
//      ├─relevant_to─> ModelFamily ─has─> Checkpoint ─trained_on─> Dataset ─computed_at─> LevelOfTheory
//      │                  │                  └─evaluated_in─> BenchmarkResult ─under─> Benchmark(protocol)
//      │                  └─implemented_by─> Implementation
//      ├─demonstrated_by─> Experiment
//      ├─associated_with─> Application <── FailureCase
//      └─described_in──> Paper ── supports ──> Claim
//
// Architecture ≠ checkpoint ≠ dataset ≠ benchmark result: each is its own
// collection, and cross-links are typed references checked by integrity.ts.
// These schemas use plain `zod` (the same instance Astro uses) so the content
// layer, the integrity check and the tests all validate with identical rules.

import { z } from 'zod';

export const COLLECTION_NAMES = [
  'papers',
  'claims',
  'concepts',
  'principles',
  'model-families',
  'checkpoints',
  'implementations',
  'datasets',
  'levels-of-theory',
  'benchmarks',
  'benchmark-results',
  'experiments',
  'applications',
  'failure-cases',
] as const;
export type CollectionName = (typeof COLLECTION_NAMES)[number];

export const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

// ---------------------------------------------------------------------------
// Typed references. While integrity.ts parses a record it installs a sink, and
// every reference field reports (target collection, id) into it. Outside that
// (e.g. in Astro's content layer) the sink is null and refs are plain ids.
// ---------------------------------------------------------------------------
export interface RefUse {
  target: CollectionName;
  id: string;
}
let refSink: RefUse[] | null = null;
const refTargets = new WeakMap<object, CollectionName>();

export function ref(target: CollectionName) {
  const schema = z
    .string()
    .regex(ID_PATTERN, `must be a kebab-case id of a record in ${target}`)
    .superRefine((id) => {
      refSink?.push({ target, id });
    });
  refTargets.set(schema, target);
  return schema;
}

/** Run `fn` (a schema parse) and collect every reference it validated. */
export function collectRefs<T>(fn: () => T): { result: T; refs: RefUse[] } {
  const previous = refSink;
  refSink = [];
  try {
    const result = fn();
    return { result, refs: refSink };
  } finally {
    refSink = previous;
  }
}

// ---------------------------------------------------------------------------
// Shared building blocks
// ---------------------------------------------------------------------------

/** ISO calendar date. YAML timestamps arrive as Date objects; normalise both. */
export const isoDate = z.union([
  z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'use YYYY-MM-DD'),
  z.date().transform((d) => d.toISOString().slice(0, 10)),
]);

const url = z.url();
const nonEmpty = z.string().trim().min(1);

export const REVIEW_STATUSES = ['draft', 'in-review', 'published'] as const;
export const review = z.object({
  status: z.enum(REVIEW_STATUSES),
  reviewers: z.array(nonEmpty).default([]),
  last_checked: isoDate,
  notes: z.string().optional(),
});
export type Review = z.infer<typeof review>;

/**
 * How the Atlas has checked a source. Vision: items "seen only via abstract or
 * secondary summary" (†) must be verified against the full paper before publication.
 */
export const VERIFICATION_LEVELS = ['full-text', 'abstract', 'secondary'] as const;
export const verifiedAgainst = z.enum(VERIFICATION_LEVELS);

/** Pointer into a paper: which figure, table, equation or section supports a claim. */
export const sourceLocator = z.object({
  paper: ref('papers'),
  /** e.g. "Fig. 3b", "Table 2", "Eq. (7)", "Sec. 4.1". Required before publication. */
  locator: z.string().optional(),
  /** Short verbatim excerpt, if helpful for review. */
  quote: z.string().optional(),
});

const assumption = z.object({ term: nonEmpty, value: nonEmpty });

// ---------------------------------------------------------------------------
// Papers and claims — the atoms of provenance
// ---------------------------------------------------------------------------

export const PAPER_TYPES = [
  'article',
  'preprint',
  'conference',
  'review',
  'perspective',
  'software',
  'dataset',
  'thesis',
  'book',
] as const;

export const PAPER_EDGE_TYPES = [
  'extends',
  'reuses-mechanism',
  'compares-against',
  'fine-tunes',
  'benchmarks',
  'responds-to',
] as const;

/** Lanes of the Papers chronology; a navigation aid, not a classification claim. */
export const PAPER_TOPICS = [
  'electrostatics',
  'descriptors-locality',
  'equivariant-message-passing',
  'foundation-models-data',
  'benchmarks-failures',
  'force-fields-water',
  'electrochemistry',
  'other',
] as const;

export const paper = z
  .object({
    title: nonEmpty,
    topic: z.enum(PAPER_TOPICS).default('other'),
    /**
     * "Family, Given-initials" in author order, one quoted string per author (a YAML
     * flow list [Behler, J., Parrinello, M.] would split each name in two). Use
     * `et_al` when the list is truncated.
     */
    authors: z
      .array(z.string().regex(/^[^,]+, \S.*$/, 'write each author as "Family, Initials" in quotes'))
      .min(1),
    et_al: z.boolean().default(false),
    year: z.number().int().min(1900).max(2100),
    venue: z.string().optional(),
    type: z.enum(PAPER_TYPES),
    identifiers: z
      .object({
        doi: z
          .string()
          .regex(/^10\.\d{4,9}\/\S+$/, 'bare DOI, e.g. 10.1063/1.3553717')
          .optional(),
        arxiv: z
          .string()
          .regex(/^\d{4}\.\d{4,5}(v\d+)?$/, 'new-style arXiv id, e.g. 2206.07697')
          .optional(),
        url: url.optional(),
        openalex: z.string().regex(/^W\d+$/).optional(),
      })
      .refine((ids) => ids.doi || ids.arxiv || ids.url, 'give at least one of doi, arxiv or url'),
    /** Typed citation edges to other papers; `documented` = stated by the paper itself. */
    edges: z
      .array(
        z.object({
          type: z.enum(PAPER_EDGE_TYPES),
          target: ref('papers'),
          documented: z.boolean(),
          note: z.string().optional(),
        }),
      )
      .default([]),
    verified_against: verifiedAgainst,
    /** Where the bibliographic metadata came from (filled by the bibliography pipeline). */
    metadata_source: z.enum(['manual', 'crossref', 'arxiv', 'openalex']).default('manual'),
    review,
  })
  .strict();

export const CLAIM_GRADES = ['THEORY', 'PUBLISHED_RESULT', 'REPRODUCED', 'TOY_DEMO', 'HYPOTHESIS'] as const;

export const claim = z
  .object({
    /** One atomic, checkable statement. */
    statement: nonEmpty,
    grade: z.enum(CLAIM_GRADES),
    sources: z.array(sourceLocator).default([]),
    /** For THEORY claims derived on an Atlas page: the page path and anchor. */
    derivation: z.string().regex(/^\//, 'site path, e.g. /method/design-system/#fig-cutoff').optional(),
    scope: z
      .object({
        model_families: z.array(ref('model-families')).default([]),
        checkpoints: z.array(ref('checkpoints')).default([]),
        datasets: z.array(ref('datasets')).default([]),
        benchmarks: z.array(ref('benchmarks')).default([]),
      })
      .default({ model_families: [], checkpoints: [], datasets: [], benchmarks: [] }),
    reproduction: z
      .object({
        experiment: ref('experiments'),
        date: isoDate,
        matches: z.boolean(),
        note: z.string().optional(),
      })
      .optional(),
    /** Required whenever there are sources; on-page derivations have none. */
    verified_against: verifiedAgainst.optional(),
    review,
  })
  .strict()
  .superRefine((c, ctx) => {
    if (c.sources.length > 0 && !c.verified_against) {
      ctx.addIssue({ code: 'custom', message: 'say how the sources were checked', path: ['verified_against'] });
    }
    if (c.grade === 'PUBLISHED_RESULT' && c.sources.length === 0) {
      ctx.addIssue({ code: 'custom', message: 'PUBLISHED_RESULT claims need at least one source', path: ['sources'] });
    }
    if (c.grade === 'THEORY' && c.sources.length === 0 && !c.derivation) {
      ctx.addIssue({ code: 'custom', message: 'THEORY claims need a source or an on-page derivation', path: ['derivation'] });
    }
    if (c.grade === 'REPRODUCED' && (c.sources.length === 0 || !c.reproduction?.matches)) {
      ctx.addIssue({
        code: 'custom',
        message: 'REPRODUCED claims need the published source and a matching Atlas reproduction',
        path: ['reproduction'],
      });
    }
  });

// ---------------------------------------------------------------------------
// Concepts and physical principles
// ---------------------------------------------------------------------------

export const principle = z
  .object({
    title: nonEmpty,
    summary: nonEmpty,
    /** Defining equations in TeX (KaTeX subset); use \sym{key}{…} for linked symbols. */
    equations: z.array(z.object({ tex: nonEmpty, label: z.string().optional() })).default([]),
    papers: z.array(ref('papers')).default([]),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict();

export const concept = z
  .object({
    title: nonEmpty,
    summary: nonEmpty,
    explains: z.array(ref('principles')).default([]),
    relevant_to: z.array(ref('model-families')).default([]),
    demonstrated_by: z.array(ref('experiments')).default([]),
    related: z.array(ref('concepts')).default([]),
    papers: z.array(ref('papers')).default([]),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict();

// ---------------------------------------------------------------------------
// Architecture (model family), implementation, checkpoint
// ---------------------------------------------------------------------------

export const PARADIGMS = [
  'descriptor-regression',
  'invariant-message-passing',
  'equivariant',
  'learned-equivariance',
  'long-range',
  'fast-deployment',
  'foundation',
] as const;

export const LONG_RANGE_METHODS = [
  'none',
  'ewald',
  'latent-ewald',
  'charge-equilibration',
  'explicit-charges',
  'nonlocal-descriptor',
  'other',
] as const;

export const modelFamily = z
  .object({
    name: nonEmpty,
    aliases: z.array(nonEmpty).default([]),
    paradigms: z.array(z.enum(PARADIGMS)).min(1),
    introduced_in: ref('papers'),
    year: z.number().int().min(1990).max(2100),
    summary: nonEmpty,
    locality: z.object({
      /** strictly-local: one cutoff, no messages; message-passing: receptive field grows with layers. */
      kind: z.enum(['strictly-local', 'message-passing', 'local-plus-long-range', 'global']),
      note: z.string().optional(),
    }),
    symmetry: z.enum(['invariant', 'equivariant', 'learned', 'none']),
    body_order: z.string().optional(),
    long_range: z.enum(LONG_RANGE_METHODS).default('none'),
    extends: z.array(ref('model-families')).default([]),
    reuses_mechanism: z
      .array(z.object({ family: ref('model-families'), mechanism: nonEmpty }))
      .default([]),
    implementations: z.array(ref('implementations')).default([]),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict();

export const implementation = z
  .object({
    name: nonEmpty,
    repo: url,
    language: nonEmpty,
    framework: z.enum(['pytorch', 'jax', 'tensorflow', 'julia', 'cpp', 'other']),
    licence: nonEmpty,
    families: z.array(ref('model-families')).min(1),
    versions: z.array(z.object({ version: nonEmpty, date: isoDate, notes: z.string().optional() })).default([]),
    review,
  })
  .strict();

export const OUTPUTS = [
  'energy',
  'forces',
  'stress',
  'charges',
  'dipole',
  'polarisation',
  'magnetic-moments',
  'fermi-level',
  'other',
] as const;

export const checkpoint = z
  .object({
    name: nonEmpty,
    family: ref('model-families'),
    implementation: ref('implementations'),
    version: z.string().optional(),
    release: z.object({ date: isoDate.optional(), source: z.union([ref('papers'), url]) }),
    elements: z.number().int().min(1).max(118).optional(),
    parameters: z
      .object({ total: z.number().positive().optional(), active: z.number().positive().optional() })
      .optional(),
    heads: z.array(z.object({ name: nonEmpty, level_of_theory: ref('levels-of-theory') })).default([]),
    training: z
      .array(
        z.object({
          stage: z.enum(['pretraining', 'finetuning', 'distillation', 'training']),
          dataset: ref('datasets'),
          level_of_theory: ref('levels-of-theory'),
          note: z.string().optional(),
        }),
      )
      .min(1),
    outputs: z.array(z.enum(OUTPUTS)).min(1),
    long_range: z.enum(LONG_RANGE_METHODS).default('none'),
    licence: nonEmpty,
    weights: z.object({ url, sha256: z.string().regex(/^[0-9a-f]{64}$/).optional() }).optional(),
    fine_tuned_from: ref('checkpoints').optional(),
    benchmark_results: z.array(ref('benchmark-results')).default([]),
    known_issues: z.array(ref('claims')).default([]),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict();

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

export const levelOfTheory = z
  .object({
    name: nonEmpty,
    method: z.enum(['DFT', 'hybrid-DFT', 'meta-GGA-DFT', 'wavefunction', 'semi-empirical', 'experiment', 'other']),
    functional: z.string().optional(),
    /** e.g. Hubbard U corrections, spin treatment. */
    corrections: z.array(nonEmpty).default([]),
    basis: z.string().optional(),
    dispersion: z.string().optional(),
    codes: z.array(nonEmpty).default([]),
    notes: z.string().optional(),
    review,
  })
  .strict();

export const DATASET_DOMAINS = [
  'inorganic-materials',
  'molecules',
  'biomolecular',
  'catalysis',
  'porous-materials',
  'liquids-interfaces',
  'mixed',
] as const;

export const dataset = z
  .object({
    name: nonEmpty,
    domain: z.array(z.enum(DATASET_DOMAINS)).min(1),
    summary: nonEmpty,
    introduced_in: ref('papers'),
    levels_of_theory: z.array(ref('levels-of-theory')).min(1),
    configuration_types: z
      .array(z.enum(['relaxation-trajectory', 'rattled', 'aimd', 'single-point', 'md-sampled', 'active-learning', 'other']))
      .default([]),
    /** Size is a claim like any other: give the number with its source. */
    size: z
      .object({
        structures: z.number().int().positive(),
        approximate: z.boolean(),
        source: sourceLocator,
      })
      .optional(),
    derived_from: z.array(ref('datasets')).default([]),
    licence: z.string().optional(),
    url: url.optional(),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict();

// ---------------------------------------------------------------------------
// Benchmarks and results — never merged across protocols
// ---------------------------------------------------------------------------

export const benchmark = z
  .object({
    name: nonEmpty,
    summary: nonEmpty,
    url: url.optional(),
    introduced_in: ref('papers'),
    domain: z.array(z.enum(DATASET_DOMAINS)).min(1),
    metrics: z
      .array(
        z.object({
          id: z.string().regex(ID_PATTERN),
          name: nonEmpty,
          unit: z.string(),
          higher_is_better: z.boolean(),
          description: z.string().optional(),
        }),
      )
      .min(1),
    protocols: z
      .array(
        z.object({
          id: z.string().regex(ID_PATTERN),
          name: nonEmpty,
          description: nonEmpty,
          test_set: z.string().optional(),
        }),
      )
      .min(1),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict()
  .superRefine((b, ctx) => {
    for (const key of ['metrics', 'protocols'] as const) {
      const ids = b[key].map((x) => x.id);
      const dup = ids.find((id, i) => ids.indexOf(id) !== i);
      if (dup) ctx.addIssue({ code: 'custom', message: `duplicate ${key} id "${dup}"`, path: [key] });
    }
  });

export const benchmarkResult = z
  .object({
    benchmark: ref('benchmarks'),
    /** Protocol and metric ids within the benchmark record (checked by integrity.ts). */
    protocol: z.string().regex(ID_PATTERN),
    metric: z.string().regex(ID_PATTERN),
    checkpoint: ref('checkpoints'),
    value: z.number(),
    uncertainty: z.number().nonnegative().optional(),
    retrieved: isoDate,
    status: z.enum(['PUBLISHED_RESULT', 'REPRODUCED']),
    source: z.object({
      kind: z.enum(['leaderboard', 'paper']),
      url: url.optional(),
      paper: ref('papers').optional(),
      locator: z.string().optional(),
      /** Commit or release of the leaderboard data the number was pulled from. */
      revision: z.string().optional(),
    }),
    review,
  })
  .strict()
  .superRefine((r, ctx) => {
    if (r.source.kind === 'leaderboard' && !r.source.url) {
      ctx.addIssue({ code: 'custom', message: 'leaderboard results need the source url', path: ['source', 'url'] });
    }
    if (r.source.kind === 'paper' && !r.source.paper) {
      ctx.addIssue({ code: 'custom', message: 'paper results need the paper id', path: ['source', 'paper'] });
    }
  });

// ---------------------------------------------------------------------------
// Experiments, applications, failure cases
// ---------------------------------------------------------------------------

export const RESULT_STATUSES = ['TOY_MODEL', 'LIVE_CALCULATION', 'PRECOMPUTED', 'PUBLISHED_RESULT', 'REPRODUCED'] as const;
export const COMPUTE_TIERS = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5'] as const;

export const experiment = z
  .object({
    title: nonEmpty,
    question: nonEmpty,
    kind: z.enum(['physics-lab', 'observatory-scan', 'reproduction', 'figure']),
    /** Physics Lab module id from the vision (P1–P12), when applicable. */
    module: z.string().regex(/^P\d{1,2}$/).optional(),
    status: z.enum(RESULT_STATUSES),
    tier: z.enum(COMPUTE_TIERS),
    /** Where it lives on the site. */
    page: z.string().regex(/^\//),
    assumptions: z.array(assumption).min(1),
    concepts: z.array(ref('concepts')).default([]),
    checkpoints: z.array(ref('checkpoints')).default([]),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict();

export const application = z
  .object({
    title: nonEmpty,
    domain: z.enum(['materials', 'energy', 'chemistry', 'biomolecular', 'fundamental-physics']),
    summary: nonEmpty,
    key_physics: z.array(ref('concepts')).default([]),
    requirements: z.array(nonEmpty).default([]),
    /** "Evaluated for related systems" — never "recommended". */
    evaluated_checkpoints: z
      .array(z.object({ checkpoint: ref('checkpoints'), evidence: ref('claims') }))
      .default([]),
    papers: z.array(ref('papers')).default([]),
    claims: z.array(ref('claims')).default([]),
    review,
  })
  .strict();

export const EVIDENCE_GRADES = ['THEORY', 'PUBLISHED_RESULT', 'REPRODUCED', 'TOY_DEMO', 'HYPOTHESIS'] as const;

export const failureCase = z
  .object({
    title: nonEmpty,
    /** The failure, stated as a checkable claim. */
    claim: ref('claims'),
    grade: z.enum(EVIDENCE_GRADES),
    evidence: z.array(ref('claims')).default([]),
    mechanisms: z.array(z.object({ description: nonEmpty, grade: z.enum(EVIDENCE_GRADES), claim: ref('claims').optional() })).default([]),
    diagnostics: z.array(nonEmpty).default([]),
    responses: z.array(z.object({ description: nonEmpty, claim: ref('claims').optional() })).default([]),
    open_questions: z.array(nonEmpty).default([]),
    affected_checkpoints: z.array(ref('checkpoints')).default([]),
    applications: z.array(ref('applications')).default([]),
    review,
  })
  .strict()
  .superRefine((f, ctx) => {
    const needsEvidence = f.grade === 'PUBLISHED_RESULT' || f.grade === 'REPRODUCED';
    if (needsEvidence && f.evidence.length === 0) {
      ctx.addIssue({ code: 'custom', message: `${f.grade} failure cases need evidence claims`, path: ['evidence'] });
    }
  });

// ---------------------------------------------------------------------------

export const SCHEMAS = {
  papers: paper,
  claims: claim,
  concepts: concept,
  principles: principle,
  'model-families': modelFamily,
  checkpoints: checkpoint,
  implementations: implementation,
  datasets: dataset,
  'levels-of-theory': levelOfTheory,
  benchmarks: benchmark,
  'benchmark-results': benchmarkResult,
  experiments: experiment,
  applications: application,
  'failure-cases': failureCase,
} as const satisfies Record<CollectionName, z.ZodType>;

export type Paper = z.infer<typeof paper>;
export type Claim = z.infer<typeof claim>;
export type Concept = z.infer<typeof concept>;
export type Principle = z.infer<typeof principle>;
export type ModelFamily = z.infer<typeof modelFamily>;
export type Checkpoint = z.infer<typeof checkpoint>;
export type Implementation = z.infer<typeof implementation>;
export type Dataset = z.infer<typeof dataset>;
export type LevelOfTheory = z.infer<typeof levelOfTheory>;
export type Benchmark = z.infer<typeof benchmark>;
export type BenchmarkResult = z.infer<typeof benchmarkResult>;
export type Experiment = z.infer<typeof experiment>;
export type Application = z.infer<typeof application>;
export type FailureCase = z.infer<typeof failureCase>;

// ---------------------------------------------------------------------------
// Relation map, derived from the schemas themselves (so diagrams and docs can
// never drift from what is validated). Walks object/array/optional/default/union
// wrappers and reports every ref() field with its dotted path.
// ---------------------------------------------------------------------------

export interface Relation {
  from: CollectionName;
  /** Dotted field path; `[]` marks array elements, e.g. "training[].dataset". */
  field: string;
  to: CollectionName;
}

interface ZodDefLike {
  type: string;
  shape?: Record<string, unknown>;
  element?: unknown;
  innerType?: unknown;
  options?: unknown[];
  in?: unknown;
}

function defOf(schema: unknown): ZodDefLike | undefined {
  return (schema as { _zod?: { def?: ZodDefLike } })?._zod?.def;
}

function walkRefs(schema: unknown, path: string, out: Array<{ field: string; to: CollectionName }>): void {
  const target = refTargets.get(schema as object);
  if (target) {
    out.push({ field: path, to: target });
    return;
  }
  const def = defOf(schema);
  if (!def) return;
  switch (def.type) {
    case 'object':
      for (const [k, v] of Object.entries(def.shape ?? {})) walkRefs(v, path ? `${path}.${k}` : k, out);
      break;
    case 'array':
      walkRefs(def.element, `${path}[]`, out);
      break;
    case 'optional':
    case 'nullable':
    case 'default':
    case 'prefault':
    case 'readonly':
    case 'nonoptional':
    case 'catch':
      walkRefs(def.innerType, path, out);
      break;
    case 'union':
      for (const o of def.options ?? []) walkRefs(o, path, out);
      break;
    case 'pipe':
      walkRefs(def.in, path, out);
      break;
    default:
      break;
  }
}

export function schemaRelations(): Relation[] {
  const out: Relation[] = [];
  for (const from of COLLECTION_NAMES) {
    const found: Array<{ field: string; to: CollectionName }> = [];
    walkRefs(SCHEMAS[from], '', found);
    for (const f of found) out.push({ from, ...f });
  }
  return out;
}
