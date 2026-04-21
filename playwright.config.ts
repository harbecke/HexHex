import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/app_e2e",
  timeout: 60_000,
  use: {
    baseURL: "http://localhost:4173",
    screenshot: "off",
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  webServer: {
    command: "npm run preview --prefix app",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env["CI"],
    timeout: 30_000,
  },
});
