import { test, expect, Page } from "@playwright/test";
import path from "path";

const screenshotDir = path.join(__dirname, "screenshots");
const BOARD_SIZE = 11;
const PLAY_CELLS = BOARD_SIZE * BOARD_SIZE; // 121

/** Go to the app and start a standard Human (red) vs AI (blue) game. */
async function startHumanVsAI(page: Page) {
  await page.goto("/");
  // Default setup already has red=human, blue=AI; just confirm.
  await page.getByRole("button", { name: "Start Game" }).click();
  await page.waitForSelector('[data-testid="hex-board"]', { timeout: 10_000 });
}

test("initial board renders with correct cell count", async ({ page }) => {
  await startHumanVsAI(page);

  await page.screenshot({ path: path.join(screenshotDir, "01-initial-board.png"), fullPage: true });

  const cells = page.locator('[data-testid="hex-cell"]');
  await expect(cells).toHaveCount(PLAY_CELLS);

  const firstFill = await cells.first().locator("polygon").getAttribute("fill");
  expect(firstFill).toBe("white");
});

test("clicking a cell places a red stone", async ({ page }) => {
  await startHumanVsAI(page);

  const centerCell = page.locator('[data-cell-id="60"]');
  await centerCell.click();

  await page.screenshot({ path: path.join(screenshotDir, "02-after-player-click.png"), fullPage: true });

  const fill = await centerCell.locator("polygon").getAttribute("fill");
  expect(fill).toBe("rgb(251, 41, 67)");
});

test("AI places a blue stone after player move", async ({ page }) => {
  await startHumanVsAI(page);

  await page.locator('[data-cell-id="60"]').click();

  await expect(page.locator('[data-testid="status"]')).toContainText("thinking");
  await page.screenshot({ path: path.join(screenshotDir, "03-ai-thinking.png"), fullPage: true });

  const blueCells = page.locator('[data-testid="hex-cell"] polygon[fill="rgb(6, 154, 243)"]');
  await expect(blueCells).toHaveCount(1, { timeout: 20_000 });

  await page.screenshot({ path: path.join(screenshotDir, "04-after-ai-move.png"), fullPage: true });
});

test("show/hide ratings toggle", async ({ page }) => {
  await startHumanVsAI(page);

  await page.locator('[data-cell-id="60"]').click();
  await expect(page.locator('[data-testid="hex-cell"] polygon[fill="rgb(6, 154, 243)"]')).toHaveCount(
    1,
    { timeout: 20_000 }
  );

  await page.locator('[data-testid="toggle-ratings"]').click();
  await page.screenshot({ path: path.join(screenshotDir, "05-ratings-visible.png"), fullPage: true });

  const nonEmptyTexts = await page
    .locator('[data-testid="hex-cell"] text')
    .evaluateAll((els) => els.filter((el) => el.textContent?.trim() !== "").length);
  expect(nonEmptyTexts).toBeGreaterThan(0);

  await page.locator('[data-testid="toggle-ratings"]').click();
  await page.screenshot({ path: path.join(screenshotDir, "06-ratings-hidden.png"), fullPage: true });
});

test("human can swap via pie rule when AI plays first", async ({ page }) => {
  await page.goto("/");
  // AI plays red (first), human plays blue (second) → human can swap.
  await page.locator('input[name="red"][value="ai"]').check();
  await page.getByRole("button", { name: "Start Game" }).click();

  // Wait until AI places a red stone
  const redCells = page.locator('[data-testid="hex-cell"] polygon[fill="rgb(251, 41, 67)"]');
  await expect(redCells).toHaveCount(1, { timeout: 20_000 });

  // The one red cell should now be swap-clickable
  await expect(page.locator('[data-testid="swap-hint"]')).toBeVisible();

  const redCell = page.locator('[data-testid="hex-cell"]').filter({ has: page.locator('polygon[fill="rgb(251, 41, 67)"]') }).first();
  await redCell.click();

  // After swap: red becomes human, AI starts thinking as blue
  await expect(page.locator('[data-testid="status"]')).toContainText("thinking");
  await page.screenshot({ path: path.join(screenshotDir, "07-after-human-swap.png"), fullPage: true });
});
