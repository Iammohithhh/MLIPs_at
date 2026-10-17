import { expect, test } from '@playwright/test';

// The second committed experiment (wurtzite ZnO) and switching between experiments.
// Expected numbers come from src/data/experiments/zno-axial-bond/manifest.json (MACE-MP-0a medium run).

test('a ZnO link restores the ZnO experiment with its own numbers and texts', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('/?v=1&exp=zno-axial-bond&ckpt=mace-mp-0a-medium&frame=24');
  await expect(page.getByRole('button', { name: /ZnO · axial Zn–O bond/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.300 Å');
  // ΔE(frame 24) = +0.707558 eV
  await expect(page.locator('.lab__readouts dd').first()).toContainText('+0.7076 eV');
  await expect(page.getByRole('alert')).toHaveCount(0);
  // unrelaxed input: stored baseline F·û on the moved Zn = +0.0371 eV/Å
  await expect(page.locator('.lab__baseline-note')).toContainText('+0.0371 eV/Å at d = 0');
  await expect(page.locator('.lab').first()).not.toContainText('zero by symmetry');
  await expect(page.locator('.lab').first()).not.toContainText(/216 (Si )?atoms|\[216/);
  expect(errors).toEqual([]);
});

test('ZnO walkthrough states its own atom and neighbour counts', async ({ page }) => {
  await page.goto('/?exp=zno-axial-bond&mode=walk&stage=species');
  await expect(page.getByText(/Starting point: 300 atoms \(150 Zn, 150 O\)/)).toBeVisible();
  await page.getByRole('button', { name: 'Next stage →' }).click();
  await expect(page.getByText(/the moved Zn atom has 76 neighbours within r_max/)).toBeVisible();
  for (let n = 3; n <= 7; n++) await page.getByRole('button', { name: 'Next stage →' }).click();
  await expect(page.getByText('Stage 7 of 7')).toBeVisible();
  await expect(page.getByText(/unrelaxed geometry already carries a predicted force/)).toBeVisible();
});

test('switching experiments keeps the mode, resets the frame, and updates the URL', async ({ page }) => {
  await page.goto('/?exp=si-bond-stretch&mode=lab&frame=20&view=difference');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.200 Å');
  await page.getByRole('button', { name: /ZnO · axial Zn–O bond/ }).click();
  await expect(page).toHaveURL(/exp=zno-axial-bond/);
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.000 Å');
  await expect(page.getByRole('radio', { name: 'Difference' })).toHaveAttribute('aria-checked', 'true');
  await page.getByRole('button', { name: /Si · bond stretch/ }).click();
  await expect(page).toHaveURL(/exp=si-bond-stretch/);
  await expect(page.getByRole('button', { name: /Si · bond stretch/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.lab__baseline-note')).toHaveCount(0); // Si baseline forces are zero
});

test('an unknown experiment id falls back to silicon and says so', async ({ page }) => {
  await page.goto('/?exp=ethanol');
  await expect(page.getByRole('alert')).toContainText('experiment “ethanol” is not available');
  await expect(page.getByRole('button', { name: /Si · bond stretch/ })).toHaveAttribute('aria-pressed', 'true');
});

test('the hero actions switch mode and keep the experiment and frame', async ({ page }) => {
  await page.goto('/?exp=zno-axial-bond&mode=lab&frame=20');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.200 Å');
  await page.getByRole('link', { name: 'Walk me through it' }).click();
  await expect(page).toHaveURL(/exp=zno-axial-bond/);
  await expect(page).toHaveURL(/mode=walk/);
  await expect(page.getByText('Stage 1 of 7')).toBeVisible();
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.200 Å');
  await expect(page.getByRole('button', { name: /ZnO · axial Zn–O bond/ })).toHaveAttribute('aria-pressed', 'true');
});

test('presentation mode on ZnO: captions from the ZnO run, status and checkpoint visible', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('/?exp=zno-axial-bond&present=1&demo=1');
  await expect(page.getByText(/zinc oxide \(wurtzite\) crystal \(300 atoms\)/)).toBeVisible();
  await expect(page.getByText(/not relaxed with the model/)).toBeVisible();
  await expect(page.getByText('Precomputed model evaluations; interactive inspection')).toBeVisible();
  await expect(page.getByText(/mace-mp-0a-medium · sha256 01bfe221/)).toBeVisible();
  await expect(page.locator('.exp-pick')).toBeHidden(); // minimal controls: the link fixes the experiment
  expect(errors).toEqual([]);
});

test('stage 4 shows per-channel |A| strips for each l, with the Si symmetry zeros as empty baseline rows', async ({ page }) => {
  await page.goto('/?exp=si-bond-stretch&mode=walk&stage=messages&frame=20');
  const strips = page.locator('svg.ev__achan');
  await expect(strips).toHaveCount(4);
  // l = 1 baseline row: every channel is zero in the perfect crystal → fully transparent cells
  const opacities = await strips.nth(1).locator('rect').evaluateAll((rs) => rs.slice(0, 128).map((r) => r.getAttribute('fill')));
  expect(opacities.every((f) => /\/ 0\)$/.test(f ?? ''))).toBe(true);
  await expect(page.getByText(/each l on its own scale/)).toBeVisible();
});
