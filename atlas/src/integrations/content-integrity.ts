// Runs the cross-record content audit when Astro starts. `astro build` fails on
// any problem; `astro dev` reports problems as warnings so editing stays fluid.
import { fileURLToPath } from 'node:url';
import type { AstroIntegration } from 'astro';
import { auditContent, formatProblems } from '../content/integrity';

export default function contentIntegrity(): AstroIntegration {
  return {
    name: 'atlas:content-integrity',
    hooks: {
      'astro:config:setup': ({ config, command, logger }) => {
        const root = fileURLToPath(config.root);
        const { records, problems } = auditContent(`${root}content`, `${root}src/pages`);
        if (problems.length === 0) {
          logger.info(`${records.size} records: schema and integrity checks passed`);
          return;
        }
        const message = `${problems.length} content problem(s):\n${formatProblems(problems, root)}`;
        if (command === 'build') throw new Error(message);
        logger.warn(message);
      },
    },
  };
}
