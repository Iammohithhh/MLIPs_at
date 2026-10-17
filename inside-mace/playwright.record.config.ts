import { defineConfig } from '@playwright/test';

// Records the scripted demonstration (brief §14) as a video: `npx astro build && npm run record:demo`.
// A still of every step is always saved; a video is added with RECORD_VIDEO=1 (needs Playwright's ffmpeg,
// `npx playwright install ffmpeg`). Output in recordings/ (git-ignored). Same browser setup as playwright.config.ts.
const executablePath = process.env.PW_CHROMIUM ?? (process.platform === 'linux' ? '/opt/pw-browsers/chromium' : undefined);
const args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
const size = { width: 1280, height: 720 };

export default defineConfig({
  testDir: 'record',
  testMatch: '*.record.ts',
  outputDir: 'recordings',
  timeout: 120_000,
  webServer: { command: 'python3 -m http.server 4330 --bind 127.0.0.1 --directory dist', port: 4330, reuseExistingServer: true },
  use: { baseURL: 'http://localhost:4330', viewport: size, video: process.env.RECORD_VIDEO ? { mode: 'on', size } : 'off', launchOptions: { executablePath, args } },
  projects: process.platform === 'linux' ? [{ name: 'chromium' }] : [{ name: 'edge', use: { channel: 'msedge' } }],
});
