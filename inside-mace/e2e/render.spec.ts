import { expect, test } from '@playwright/test';

// Brief §15: pause work when the scene is hidden. Each Stage's container reports its frame loop.
test('scene rendering pauses off screen and resumes, redrawing state changed meanwhile', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('/?exp=si-bond-stretch');
  const stage = page.locator('[data-rendering]').first();
  await expect(stage).toHaveAttribute('data-rendering', 'on-demand');
  await page.locator('#prov').scrollIntoViewIfNeeded();
  await expect(stage).toHaveAttribute('data-rendering', 'paused');
  // change the frame while the scene is off screen (URL state still updates)
  await page.evaluate(() => (document.querySelector('[aria-label="Next evaluated frame"]') as HTMLButtonElement).click());
  await expect(page).toHaveURL(/frame=13/);
  await page.locator('.lab__canvas').first().scrollIntoViewIfNeeded();
  await expect(stage).toHaveAttribute('data-rendering', 'on-demand');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.025 Å');
  expect(errors).toEqual([]);
});
