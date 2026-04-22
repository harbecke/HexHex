import { test, expect } from "@playwright/test";
import path from "node:path";

const OUT_DIR = path.resolve(__dirname, "screenshots");

test.use({ viewport: { width: 1280, height: 900 } });

test("capture setup screen", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("start-game")).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT_DIR, "hexhex-setup.png") });
});

test("capture gameplay with ratings overlay", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("start-game").click();
  await expect(page.getByTestId("hex-board")).toBeVisible();

  // Let the worker + ONNX model finish loading on first use.
  await page.waitForTimeout(1500);

  const click = async (cellId: number) => {
    const cell = page.locator(`[data-cell-id="${cellId}"]`);
    if (await cell.count() === 0) return;
    await cell.click();
  };

  // Human-vs-AI: human plays red. Play a handful of moves, giving the AI time to reply.
  const myMoves = [60, 50, 72, 39, 84];
  for (const id of myMoves) {
    await click(id);
    await page.waitForTimeout(1400);
  }

  // Toggle the move-values overlay (hotkey S).
  await page.keyboard.press("s");
  await expect(page.getByTestId("ratings-info")).toBeVisible();
  await page.waitForTimeout(400);

  await page.screenshot({ path: path.join(OUT_DIR, "hexhex-gameplay.png") });
});
