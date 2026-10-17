// Copied from atlas/vitest.config.ts (MLIP Atlas) for inside-mace; keep in sync deliberately, not automatically.
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: { '~': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
});
