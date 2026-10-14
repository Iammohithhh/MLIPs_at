// Astro content collections: one per Atlas entity, loaded from content/<name>/*.yaml
// and validated with the shared schemas in src/content/schemas.ts. Cross-record
// rules run in the content-integrity integration (see astro.config.mjs).
import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { SCHEMAS, type CollectionName } from './content/schemas';

const records = <C extends CollectionName>(name: C) =>
  defineCollection({
    loader: glob({ pattern: '*.yaml', base: `./content/${name}` }),
    schema: SCHEMAS[name],
  });

export const collections = {
  papers: records('papers'),
  claims: records('claims'),
  concepts: records('concepts'),
  principles: records('principles'),
  'model-families': records('model-families'),
  checkpoints: records('checkpoints'),
  implementations: records('implementations'),
  datasets: records('datasets'),
  'levels-of-theory': records('levels-of-theory'),
  benchmarks: records('benchmarks'),
  'benchmark-results': records('benchmark-results'),
  experiments: records('experiments'),
  applications: records('applications'),
  'failure-cases': records('failure-cases'),
};
