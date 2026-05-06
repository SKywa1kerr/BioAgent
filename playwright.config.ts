import { defineConfig } from "@playwright/test";

// Electron e2e config. Targets the Vite dev server because the smoke test
// only exercises the renderer-side flows (no full electron:build pipeline).
// For full Electron integration tests, switch to `_electron.launch()` from
// `playwright`'s electron support — out of scope for the smoke commit.
//
// NOTE: this project's Vite dev server runs on port 1420 (see vite.config.ts),
// not the Vite default 5173 — keep these in sync if vite.config.ts changes.
export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: /.*\.spec\.ts$/,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:1420",
    trace: "retain-on-failure",
    headless: true,
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    stdout: "ignore",
    stderr: "pipe",
  },
});
