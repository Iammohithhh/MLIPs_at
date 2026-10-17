// @ts-check
// Adapted from atlas/astro.config.mjs (MLIP Atlas): same React islands, build-time KaTeX and
// HTML-aware whitespace compression; without the Atlas content-integrity integration, MDX and Tailwind.
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { katexOptions } from './src/lib/katex-options.mjs';

export default defineConfig({
  trailingSlash: 'ignore',
  // Astro 7's default ('jsx') whitespace handling drops spaces between prose and inline elements.
  compressHTML: true,
  integrations: [react()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, katexOptions]],
    }),
  },
});
