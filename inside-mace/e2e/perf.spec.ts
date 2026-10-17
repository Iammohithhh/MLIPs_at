// Interaction performance on an emulated mid-range phone (brief §15): Pixel 7 viewport and touch, CPU
// throttled 4× through the Chrome DevTools Protocol, SwiftShader (software) WebGL — a pessimistic GPU.
// Measures: time to an interactive instrument, the latency from a frame step to the updated readout
// (median and worst of 12 steps, including the lazy frame-chunk fetch), long tasks, and the JS heap.
// Results are attached to the test report and printed; the budgets are generous upper bounds so the
// check catches regressions, not CI noise.

import { expect, test } from '@playwright/test';

test('mobile interaction stays responsive under 4× CPU throttling', async ({ page, context, browserName }, info) => {
  test.skip(browserName !== 'chromium' || info.project.name !== 'mobile', 'CDP throttling: Chromium mobile project only');
  const cdp = await context.newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  await cdp.send('Performance.enable');

  await page.addInitScript(() => {
    (window as unknown as { __long: number[] }).__long = [];
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) (window as unknown as { __long: number[] }).__long.push(e.duration);
    }).observe({ type: 'longtask', buffered: true });
  });

  const t0 = Date.now();
  await page.goto('/');
  const out = page.locator('output.lab__disp-value');
  await expect(out).toContainText('d = +0.000 Å', { timeout: 30_000 });
  await expect(page.locator('canvas').first()).toBeVisible({ timeout: 30_000 });
  const interactiveMs = Date.now() - t0;

  const next = page.getByRole('button', { name: 'Next evaluated frame' });
  const steps: number[] = [];
  for (let i = 1; i <= 12; i++) {
    const d = `d = +${(0.025 * i).toFixed(3)} Å`;
    const s = Date.now();
    await next.click();
    await expect(out).toContainText(d);
    await expect(page.locator('.lab__readouts dd').first()).not.toContainText('—');
    steps.push(Date.now() - s);
  }
  const sorted = [...steps].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const worst = sorted.at(-1)!;
  const long = await page.evaluate(() => (window as unknown as { __long: number[] }).__long);
  const metrics = (await cdp.send('Performance.getMetrics')).metrics;
  const heapMB = (metrics.find((x) => x.name === 'JSHeapUsedSize')?.value ?? NaN) / 2 ** 20;

  const result = {
    setup: 'Pixel 7 emulation, CPU throttling 4×, SwiftShader WebGL, static dist/ over localhost',
    interactiveMs,
    frameStepMs: { median, worst, all: steps },
    longTasks: { count: long.length, maxMs: Math.round(Math.max(0, ...long)) },
    jsHeapUsedMB: Math.round(heapMB * 10) / 10,
  };
  console.log('mobile-perf', JSON.stringify(result));
  await info.attach('mobile-perf.json', { body: JSON.stringify(result, null, 1), contentType: 'application/json' });

  expect(median).toBeLessThan(1500);
  expect(worst).toBeLessThan(4000);
  expect(heapMB).toBeLessThan(400);
});
