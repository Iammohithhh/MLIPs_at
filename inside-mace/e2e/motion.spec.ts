import { expect, test, type Page } from '@playwright/test';

// GSAP transitions on the architecture diagram (stage emphasis) and the trace reveal run with normal motion and are
// skipped entirely under prefers-reduced-motion. A MutationObserver records inline styles GSAP writes during the change.

async function recordStyles(page: Page, selector: string) {
  await page.evaluate((sel) => {
    const w = window as unknown as { __styled: string[] };
    w.__styled = [];
    new MutationObserver((ms) => {
      for (const m of ms) {
        const el = m.target as Element;
        if (el.matches(sel) && (el.getAttribute('style') ?? '') !== '') w.__styled.push(el.getAttribute('style')!);
      }
    }).observe(document.body, { subtree: true, attributes: true, attributeFilter: ['style'] });
  }, selector);
}
const styled = (page: Page) => page.evaluate(() => (window as unknown as { __styled: string[] }).__styled);

for (const reduce of [false, true]) {
  test(`stage emphasis and trace reveal ${reduce ? 'do not animate under reduced motion' : 'animate with GSAP'}`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion: reduce ? 'reduce' : 'no-preference' });
    await page.goto('/?exp=si-bond-stretch&mode=walk&stage=species&frame=24');
    const arch = page.locator('figure.arch');
    await arch.scrollIntoViewIfNeeded();
    await recordStyles(page, '.arch__block, .trace__reveal');
    await arch.getByRole('button', { name: /Readout 2/ }).click();
    await expect(page).toHaveURL(/stage=energy/);
    await page.locator('.trace').scrollIntoViewIfNeeded();
    await page.getByLabel('Predict before revealing').uncheck();
    await page.getByRole('button', { name: 'Next trace step' }).click();
    await page.waitForTimeout(600);
    const s = await styled(page);
    if (reduce) expect(s).toEqual([]);
    else {
      expect(s.some((x) => /transform|scale/.test(x))).toBe(true);
      expect(s.some((x) => /opacity/.test(x))).toBe(true);
    }
    // after the transitions nothing is left hidden or scaled
    await expect(page.locator('.arch__block.is-on').first()).not.toHaveAttribute('style', /scale|transform/);
  });
}
