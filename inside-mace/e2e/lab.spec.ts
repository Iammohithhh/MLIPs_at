import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(m.text());
  });
  return errors;
}

test('opens on a real interaction with checkpoint and status visible', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Inside an atomic prediction.' })).toBeVisible();
  await expect(page.getByText('Precomputed model evaluations; interactive inspection')).toBeVisible();
  await expect(page.getByText(/mace-mp-0a-medium · sha256 01bfe221/)).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();
  expect(errors).toEqual([]);
});

test('displacement snaps to evaluated frames and updates numbers and URL', async ({ page }) => {
  await page.goto('/');
  const out = page.locator('output.lab__disp-value');
  await expect(out).toContainText('d = +0.000 Å');
  await expect(page.locator('.lab__readouts dd').first()).toContainText('0.0000 eV');
  await page.getByRole('button', { name: 'Next evaluated frame' }).click();
  await expect(out).toContainText('d = +0.025 Å');
  await expect(page).toHaveURL(/frame=13/);
  // ΔE for frame 13 from the manifest: +0.002634 eV
  await expect(page.locator('.lab__readouts dd').first()).toContainText('+0.0026 eV');
});

test('keyboard operates the slider', async ({ page }) => {
  await page.goto('/');
  await page.locator('#disp').focus();
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = −0.050 Å');
});

test('baseline / perturbed / difference views and reset', async ({ page }) => {
  await page.goto('/?frame=20');
  for (const name of ['Baseline', 'Difference', 'Perturbed']) {
    await page.getByRole('radio', { name }).click();
    await expect(page.getByRole('radio', { name })).toHaveAttribute('aria-checked', 'true');
  }
  await page.getByRole('button', { name: 'Reset experiment' }).click();
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.000 Å');
});

test('shared URL restores the same state; bad values are reported, not shown', async ({ page }) => {
  await page.goto('/?v=1&exp=si-bond-stretch&ckpt=mace-mp-0a-medium&frame=4&view=difference&ch=17');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = −0.200 Å');
  await expect(page.getByRole('radio', { name: 'Difference' })).toHaveAttribute('aria-checked', 'true');
  await expect(page.getByText('channel 17, l = 0')).toBeVisible();
  await page.goto('/?frame=999');
  await expect(page.getByRole('alert')).toContainText('out of range');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.000 Å');
});

test('no horizontal overflow', async ({ page }) => {
  await page.goto('/?frame=20');
  await page.waitForTimeout(1000);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test('without WebGL the page says so and the numbers stay usable', async ({ page }) => {
  await page.addInitScript(() => {
    const orig = HTMLCanvasElement.prototype.getContext;
    // @ts-expect-error test override
    HTMLCanvasElement.prototype.getContext = function (type: string, ...rest: unknown[]) {
      if (/webgl/.test(type)) return null;
      return orig.call(this, type, ...rest);
    };
  });
  await page.goto('/?frame=16');
  await expect(page.getByText('3D view unavailable.')).toBeVisible();
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.100 Å');
  await expect(page.locator('.lab__readouts dd').first()).toContainText('+0.0464 eV');
});

test('reduced motion is respected (no transitions)', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  const d = await page.locator('.lab__views button').first().evaluate((el) => getComputedStyle(el).transitionDuration);
  expect(parseFloat(d)).toBeLessThan(0.001);
});

test('guided walkthrough: seven stages, equation on demand, URL keeps the stage', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await page.getByRole('link', { name: 'Walk me through it' }).click();
  await expect(page.getByText('Stage 1 of 7')).toBeVisible();
  for (let n = 2; n <= 7; n++) {
    await page.getByRole('button', { name: 'Next stage →' }).click();
    await expect(page.getByText(`Stage ${n} of 7`)).toBeVisible();
  }
  await expect(page).toHaveURL(/stage=forces/);
  await expect(page.getByText(/not the slope of its own site energy/)).toBeVisible();
  await page.getByRole('button', { name: 'Show equation and code' }).click();
  await expect(page.locator('.sd__eq .katex')).toBeVisible();
  await expect(page.getByText(/mace\/modules\/utils.py:22/)).toBeVisible();
  await page.getByRole('navigation', { name: 'Computation stages' }).getByRole('button', { name: /Neighbour graph/ }).click();
  await expect(page.getByText('Stage 2 of 7')).toBeVisible();
  // switching mode keeps the displacement frame
  await page.getByRole('button', { name: 'Next evaluated frame' }).click();
  await page.getByRole('tab', { name: 'Move an atom' }).click();
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.025 Å');
  expect(errors).toEqual([]);
});

test('atom selection updates readouts and URL', async ({ page }) => {
  await page.goto('/?frame=16');
  await page.getByLabel('Selected atom', { exact: true }).selectOption('106');
  await expect(page).toHaveURL(/atom=106/);
  await expect(page.getByText('|F| on atom 106')).toBeVisible();
  await expect(page.getByText(/Layer 1 features h of atom 106/)).toBeVisible();
});
