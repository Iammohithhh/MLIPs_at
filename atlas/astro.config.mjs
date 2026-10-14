// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { katexOptions } from './src/lib/katex-options.mjs';

// https://docs.astro.build/en/reference/configuration-reference/
export default defineConfig({
  trailingSlash: 'ignore',
  // Lossless, HTML-aware whitespace compression. Astro 7's default ('jsx') drops
  // line breaks between prose and inline elements ("orderp"), which breaks text.
  compressHTML: true,
  integrations: [react(), mdx()],
  markdown: {
    // KaTeX runs at build time: pages ship HTML + MathML, no maths JS.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, katexOptions]],
    }),
  },
  vite: {
    plugins: [tailwindcss()],
  },
});
