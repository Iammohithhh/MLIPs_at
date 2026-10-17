import { expect, test } from '@playwright/test';

// The five-step demonstration for a short screen recording (brief §14), driven through the page's own presentation
// mode, so the video shows only supported behaviour: stored frames, captions computed from the stored run.
// 1 structure and forces · 2 move one atom along the evaluated path · 3 changes inside the model ·
// 4 energy/force comparison · 5 invitation to try the same experiment.
// Pauses are for the viewer only; they are not model timings.

for (const exp of ['si-bond-stretch', 'zno-axial-bond']) {
  test(`demo sequence: ${exp}`, async ({ page }, info) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`/?exp=${exp}&present=1&demo=1`);
    const caption = page.locator('.present__caption');
    await expect(caption).toContainText('1/5');
    await expect(page.getByText('Precomputed model evaluations; interactive inspection')).toBeVisible();
    await page.waitForTimeout(1500);
    await page.screenshot({ path: info.outputPath(`${exp}-step1.png`) });
    await page.waitForTimeout(2000);
    for (let k = 2; k <= 5; k++) {
      await page.getByRole('button', { name: 'Next →' }).click();
      await expect(caption).toContainText(`${k}/5`);
      await expect(page).toHaveURL(new RegExp(`demo=${k}`));
      await page.waitForTimeout(1500);
      await page.screenshot({ path: info.outputPath(`${exp}-step${k}.png`) });
      await page.waitForTimeout(k === 2 ? 3000 : 2000);
    }
    expect(errors).toEqual([]);
  });
}
