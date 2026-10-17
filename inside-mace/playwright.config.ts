import { defineConfig, devices } from '@playwright/test';

// Chromium on Linux (cloud/CI). On the owner's Windows machine use `--project=edge` (channel msedge).
const executablePath = process.env.PW_CHROMIUM ?? (process.platform === 'linux' ? '/opt/pw-browsers/chromium' : undefined);
const args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'];

export default defineConfig({
  testDir: 'e2e',
  timeout: 60_000,
  // astro preview detaches itself in non-interactive shells, so serve the built site with Python's static server.
  webServer: { command: 'python3 -m http.server 4330 --bind 127.0.0.1 --directory dist', port: 4330, reuseExistingServer: true },
  use: { baseURL: 'http://localhost:4330', launchOptions: { executablePath, args } },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile', use: { ...devices['Pixel 7'], launchOptions: { executablePath, args } } },
    { name: 'edge', use: { channel: 'msedge', viewport: { width: 1440, height: 1000 } } },
  ],
});
