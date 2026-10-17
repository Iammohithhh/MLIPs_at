import { expect, test, type Page } from '@playwright/test';

// Colour-by-change regression (fixed in 5c86385): every element must start from the legend's neutral grey, so the
// ZnO baseline in "by feature change" mode shows no brick-red O atoms, while the displaced frame shows amber atoms.
// Pixels are classified from a screenshot of the first scene canvas (SwiftShader WebGL).

type Counts = { red: number; amber: number; total: number };

async function classify(page: Page): Promise<Counts> {
  const canvas = page.locator('.lab__canvas canvas').first();
  await canvas.scrollIntoViewIfNeeded();
  await page.waitForTimeout(400);
  const png = await canvas.screenshot();
  return page.evaluate(async (b64) => {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let red = 0;
    let amber = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i]!, g = d[i + 1]!, bl = d[i + 2]!;
      // O display colour #c96a5f under shading: red dominant, green ≈ blue
      if (r > 110 && r - g > 55 && Math.abs(g - bl) < 30) red++;
      // amber #f0a945 under shading: r > g > b with a large green–blue gap
      if (r > 120 && g - bl > 45 && r - g > 20 && r - g < 110) amber++;
    }
    return { red, amber, total: d.length / 4 };
  }, png.toString('base64'));
}

async function setColour(page: Page, mode: 'element' | 'change') {
  await page.getByLabel(/^Colour/).selectOption(mode);
}

test('ZnO colour by change: grey at the baseline, amber after the displacement', async ({ page }) => {
  await page.goto('/?exp=zno-axial-bond&mode=lab&frame=12');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.000 Å');
  const element = await classify(page);
  await setColour(page, 'change');
  const base = await classify(page);
  await page.goto('/?exp=zno-axial-bond&mode=lab&frame=24');
  await expect(page.locator('output.lab__disp-value')).toContainText('d = +0.300 Å');
  await setColour(page, 'change');
  const moved = await classify(page);
  expect(element.red).toBeGreaterThan(500); // O atoms are visibly red by element
  expect(base.red).toBeLessThan(element.red / 20); // …and not by change at zero displacement
  // the amber displacement-path marker is present in both; atoms add amber only after the move
  expect(moved.amber).toBeGreaterThan(Math.max(2 * base.amber, base.amber + 300));
});
