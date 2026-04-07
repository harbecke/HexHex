import { defineConfig } from "@playwright/test";
import path from "path";

export default defineConfig({
  testDir: "./tests/app_e2e",
  timeout: 30_000,
  use: {
    screenshot: "off",
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        // Allow file:// access and local resource loading
        launchOptions: {
          args: ["--allow-file-access-from-files"],
        },
      },
    },
  ],
});
