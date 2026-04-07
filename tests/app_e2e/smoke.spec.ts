import { test, expect } from "@playwright/test";
import path from "path";

const screenshotDir = path.join(__dirname, "screenshots");
const BOARD_SIZE = 11;
const PLAY_CELLS = BOARD_SIZE * BOARD_SIZE; // 121

test("initial board renders with correct cell count", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector('[data-testid="hex-board"]', { timeout: 10_000 });

  await page.screenshot({ path: path.join(screenshotDir, "01-initial-board.png"), fullPage: true });

  const cells = page.locator('[data-testid="hex-cell"]');
  await expect(cells).toHaveCount(PLAY_CELLS);

  // All play cells start empty (white fill)
  const firstFill = await cells.first().locator("polygon").getAttribute("fill");
  expect(firstFill).toBe("white");
});

test("clicking a cell places a red stone", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector('[data-testid="hex-board"]');

  const centerCell = page.locator('[data-cell-id="60"]'); // (5,5)
  await centerCell.click();

  await page.screenshot({ path: path.join(screenshotDir, "02-after-player-click.png"), fullPage: true });

  const fill = await centerCell.locator("polygon").getAttribute("fill");
  expect(fill).toBe("rgb(251, 41, 67)");
});

test("AI places a blue stone after player move", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector('[data-testid="hex-board"]');

  await page.locator('[data-cell-id="60"]').click();

  await expect(page.locator('[data-testid="status"]')).toContainText("thinking");
  await page.screenshot({ path: path.join(screenshotDir, "03-ai-thinking.png"), fullPage: true });

  // Wait for AI to finish
  await expect(page.locator('[data-testid="status"]')).toBeEmpty({ timeout: 20_000 });

  const blueCells = page.locator('[data-testid="hex-cell"] polygon[fill="rgb(6, 154, 243)"]');
  await expect(blueCells).toHaveCount(1);

  await page.screenshot({ path: path.join(screenshotDir, "04-after-ai-move.png"), fullPage: true });
});

test("show/hide ratings toggle", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector('[data-testid="hex-board"]');

  // Trigger AI to populate scores
  await page.locator('[data-cell-id="60"]').click();
  await expect(page.locator('[data-testid="status"]')).toBeEmpty({ timeout: 20_000 });

  await page.locator('[data-testid="toggle-ratings"]').click();
  await page.screenshot({ path: path.join(screenshotDir, "05-ratings-visible.png"), fullPage: true });

  const nonEmptyTexts = await page
    .locator('[data-testid="hex-cell"] text')
    .evaluateAll((els) => els.filter((el) => el.textContent?.trim() !== "").length);
  expect(nonEmptyTexts).toBeGreaterThan(0);

  await page.locator('[data-testid="toggle-ratings"]').click();
  await page.screenshot({ path: path.join(screenshotDir, "06-ratings-hidden.png"), fullPage: true });
});
