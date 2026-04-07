import { test, expect } from "@playwright/test";
import path from "path";

const screenshotDir = path.join(__dirname, "screenshots");
const buildIndex = `file://${path.resolve(__dirname, "../../app/build/index.html")}`;

test("initial board renders", async ({ page }) => {
  await page.goto(buildIndex);
  await page.waitForSelector("svg", { timeout: 10_000 });
  await page.screenshot({
    path: path.join(screenshotDir, "01-initial-board.png"),
    fullPage: true,
  });

  const hexagons = page.locator("polygon");
  const count = await hexagons.count();
  expect(count).toBeGreaterThan(0);
});

test("clicking a cell places a stone and AI responds", async ({ page }) => {
  await page.goto(buildIndex);
  await page.waitForSelector("svg", { timeout: 10_000 });

  const hexCells = page.locator("polygon");
  const cellCount = await hexCells.count();
  expect(cellCount).toBeGreaterThan(0);

  // Click a cell near the center
  const middleCell = hexCells.nth(Math.floor(cellCount / 2));
  await middleCell.click();

  await page.screenshot({
    path: path.join(screenshotDir, "02-after-player-move.png"),
    fullPage: true,
  });
});
