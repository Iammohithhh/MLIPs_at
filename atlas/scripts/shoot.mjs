// Screenshot pages for visual review: node scripts/shoot.mjs <outDir> [path ...]
// Needs `npx astro preview` running on :4321. Shoots light/dark × desktop/mobile,
// reports console errors, page errors and horizontal overflow.
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const [out = 'shots', ...paths] = process.argv.slice(2);
const pages = paths.length ? paths : ['/'];
const base = process.env.BASE ?? 'http://localhost:4321';
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chromium', args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
const problems = [];
for (const theme of ['light', 'dark']) {
  for (const [w, h, tag] of [
    [1440, 900, 'desk'],
    [390, 844, 'mob'],
  ]) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: theme });
    const page = await ctx.newPage();
    page.on('console', (m) => m.type() === 'error' && problems.push(`${tag}/${theme} console: ${m.text()}`));
    page.on('pageerror', (e) => problems.push(`${tag}/${theme} pageerror: ${e.message}`));
    for (const p of pages) {
      await page.goto(base + p, { waitUntil: 'networkidle' });
      await page.waitForTimeout(Number(process.env.WAIT ?? 1200));
      const sw = await page.evaluate(() => document.documentElement.scrollWidth);
      if (sw > w + 1) problems.push(`${tag}/${theme} ${p}: horizontal overflow (${sw}px)`);
      const name = p.replace(/\//g, '_').replace(/^_|_$/g, '') || 'home';
      await page.screenshot({ path: `${out}/${name}-${tag}-${theme}.png`, fullPage: process.env.FULL !== '0' });
    }
    await ctx.close();
  }
}
await browser.close();
console.log(problems.length ? problems.join('\n') : 'no errors');
