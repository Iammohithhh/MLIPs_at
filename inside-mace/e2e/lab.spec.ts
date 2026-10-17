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
  await expect(page.getByText(/force arrows: 0.5 Å of length per eV\/Å/)).toBeVisible();
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
  await expect(page.getByText(/learned numbers selected by the element alone/)).toBeVisible();
  for (let n = 2; n <= 7; n++) {
    await page.getByRole('button', { name: 'Next stage →' }).click();
    await expect(page.getByText(`Stage ${n} of 7`)).toBeVisible();
  }
  await expect(page).toHaveURL(/stage=forces/);
  await expect(page.getByText(/not the slope of its own site energy/)).toBeVisible();
  await page.getByRole('button', { name: 'Show equation and code' }).click();
  await expect(page.locator('.sd__eq .katex')).toBeVisible();
  await expect(page.getByText(/mace\/modules\/utils.py:22/)).toBeVisible();
  await expect(page.getByText(/autograd force vs central finite difference/)).toBeVisible();
  await page.getByRole('navigation', { name: 'Computation stages' }).getByRole('button', { name: /Neighbour graph/ }).click();
  await expect(page.getByText('Stage 2 of 7')).toBeVisible();
  await expect(page.getByText(/46 edges \(r < 6 Å\) end on the moved atom/)).toBeVisible();
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

test('rotation comparison shows invariant energy and rotated force', async ({ page }) => {
  await page.goto('/?ch=3');
  const rot = page.getByRole('region', { name: 'What changes when the coordinate frame changes?' });
  await expect(rot).toContainText('invariant');
  await expect(rot).toContainText('rotates with the input');
  await expect(rot).toContainText(/Layer 1 vector, ch 3/);
  await expect(rot).toContainText(/rotated run \(/);
});

test('trace the change: predict, reveal stage by stage, exact zeros beyond the reach', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?frame=24');
  const trace = page.locator('section.trace');
  await trace.scrollIntoViewIfNeeded();
  await expect(trace.getByText('Precomputed · frame 25, d = +0.300 Å')).toBeVisible();
  await trace.getByRole('button', { name: /Layer 1 features/ }).click();
  await expect(trace.getByText(/answer the question above to reveal/)).toBeVisible();
  await trace.getByRole('button', { name: 'Yes, they change' }).click();
  await expect(trace.getByRole('status')).toContainText('Computed: 0 of 169 atoms beyond 6 Å changed');
  await trace.getByRole('button', { name: /^5 Forces/ }).click();
  await trace.getByRole('button', { name: 'No, unchanged' }).click();
  await expect(trace.getByRole('status')).toContainText('the answer for this frame is yes');
  await trace.getByRole('checkbox', { name: 'Predict before revealing' }).uncheck();
  await trace.getByRole('button', { name: /Layer 2 features/ }).click();
  await expect(trace.locator('figcaption')).toContainText('194 atoms changed, 22 unchanged');
  expect(errors).toEqual([]);
});

test('side by side: two synchronised scenes on desktop, one scene on mobile', async ({ page }, info) => {
  const errors = collectErrors(page);
  await page.goto('/?frame=20&view=split');
  if (info.project.name === 'mobile') {
    await expect(page.locator('.lab__stage canvas')).toHaveCount(1);
    await expect(page.getByRole('radio', { name: 'Side by side' })).toHaveCount(0);
    await expect(page.getByRole('radio', { name: 'Perturbed' })).toHaveAttribute('aria-checked', 'true');
  } else {
    await expect(page.locator('.lab__stage canvas')).toHaveCount(2);
    await expect(page.getByText('Baseline · d = 0')).toBeVisible();
    await expect(page.getByText('Perturbed · d = +0.200 Å')).toBeVisible();
    await page.getByRole('radio', { name: 'Difference' }).click();
    await expect(page.locator('.lab__stage canvas')).toHaveCount(1);
    await expect(page).toHaveURL(/view=difference/);
  }
  expect(errors).toEqual([]);
});

test('architecture diagram: a block opens its walkthrough stage and the stage emphasises its blocks', async ({ page }) => {
  await page.goto('/');
  const arch = page.locator('figure.arch');
  await arch.getByRole('button', { name: /Readout 2/ }).click();
  await expect(page).toHaveURL(/mode=walk/);
  await expect(page).toHaveURL(/stage=energy/);
  await expect(arch.getByRole('button', { name: /Readout 1/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(arch.getByRole('button', { name: /Spherical harmonics/ })).toHaveAttribute('aria-pressed', 'false');
});

test('walkthrough stage 5 shows baseline and frame feature strips for a local atom', async ({ page }) => {
  await page.goto('/?mode=walk&stage=features&frame=24');
  await expect(page.getByText('Atom 105: baseline vs this frame (d = 0.300 Å)')).toBeVisible();
  await expect(page.locator('.ev').getByText(/Layer 1, 128 vectors \(l = 1\): length/)).toBeVisible();
  await expect(page.locator('.ev').getByText(/Layer 2, 128 scalars/)).toBeVisible();
});

test('presentation mode: scripted demonstration steps, status visible, link back to the full instrument', async ({ page }, info) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Presentation mode' }).click();
  await expect(page).toHaveURL(/present=1/);
  await expect(page.getByRole('heading', { name: 'Inside an atomic prediction.' })).toBeHidden();
  await expect(page.getByText('Precomputed model evaluations; interactive inspection')).toBeVisible();
  await expect(page.locator('.present__caption')).toContainText('1/5');
  await page.getByRole('button', { name: 'Next →' }).click();
  await expect(page.locator('.present__caption')).toContainText('2/5');
  await expect(page).toHaveURL(/frame=24/);
  await page.getByRole('button', { name: 'Next →' }).click();
  await page.getByRole('button', { name: 'Next →' }).click();
  await expect(page.locator('.present__caption')).toContainText('ΔE = +0.575 eV');
  await expect(page.locator('.lab__stage canvas')).toHaveCount(info.project.name === 'mobile' ? 1 : 2);
  await page.getByRole('link', { name: 'Open the full instrument' }).click();
  await expect(page.getByRole('heading', { name: 'Inside an atomic prediction.' })).toBeVisible();
  await expect(page).not.toHaveURL(/present=1/);
  expect(errors).toEqual([]);
});
