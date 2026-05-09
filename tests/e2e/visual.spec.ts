// Visual baseline capture — runs against `npm run dev` and writes PNGs into
// docs/superpowers/specs/assets/2026-05-08/. Renders are pure-browser (the
// electronAPI stub from main.tsx kicks in), so flows that don't need a real
// Python sidecar / LLM are exercisable.
import { test, expect, type Page } from "@playwright/test";
import * as path from "path";

const BASELINE_DIR = path.resolve(process.cwd(), "docs/superpowers/specs/assets/2026-05-08");

async function setTheme(page: Page, theme: "light" | "dark"): Promise<void> {
  await page.evaluate((t) => {
    document.documentElement.dataset.theme = t;
    try {
      window.localStorage.setItem("bioagent-theme", t);
    } catch {
      /* ignore */
    }
  }, theme);
  // settle: theme tokens cascade and font swap may relayout
  await page.waitForTimeout(150);
}

async function clearStorage(page: Page): Promise<void> {
  await page.evaluate(() => {
    try {
      window.localStorage.clear();
    } catch {
      /* ignore */
    }
  });
}

async function shoot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: path.join(BASELINE_DIR, name), fullPage: false });
}

test.describe.serial("warm-workbench visual baselines", () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
  });

  test("01–02 init dialog (light + dark)", async ({ page }) => {
    await page.goto("/");
    await clearStorage(page);
    await page.reload();
    await expect(page.locator(".init-dialog-card")).toBeVisible();
    await setTheme(page, "light");
    await shoot(page, "01-init-dialog-light.png");
    await setTheme(page, "dark");
    await shoot(page, "02-init-dialog-dark.png");
  });

  test("03 init provider Anthropic shows proxy hint", async ({ page }) => {
    await page.goto("/");
    await clearStorage(page);
    await page.reload();
    await setTheme(page, "light");
    await page.locator(".init-dialog-select").selectOption("anthropic");
    await expect(page.locator(".init-dialog-hint").first()).toBeVisible();
    await shoot(page, "03-init-provider-anthropic.png");
  });

  test("04 init provider Ollama hides API key", async ({ page }) => {
    await page.goto("/");
    await clearStorage(page);
    await page.reload();
    await setTheme(page, "light");
    await page.locator(".init-dialog-select").selectOption("ollama");
    await expect(page.locator('.init-dialog-field input[type="password"]')).toHaveCount(0);
    await shoot(page, "04-init-provider-ollama.png");
  });

  test("05–06 empty state (light + dark)", async ({ page }) => {
    await page.goto("/");
    // Pre-seed settings so InitDialog won't auto-open. agent.initialized stays
    // false (stubbed electronAPI never sends ready event) but we just want to
    // capture the underlying shell layout.
    await page.evaluate(() => {
      window.localStorage.setItem(
        "bioagent-settings",
        JSON.stringify({
          p: "sjtu",
          k: btoa("dev-key-not-real"),
          u: "https://models.sjtu.edu.cn/api/v1",
          m: "deepseek-chat",
          t: 2400,
        }),
      );
    });
    await page.reload();
    await expect(page.locator(".app-shell")).toBeVisible();
    // dismiss init dialog if it appears (it will, since stub never initializes)
    if (await page.locator(".init-dialog-card").isVisible().catch(() => false)) {
      // Press Escape isn't wired for InitDialog — just hide via DOM for shot.
      await page.evaluate(() => {
        const el = document.querySelector(".init-dialog-scrim") as HTMLElement | null;
        if (el) el.style.display = "none";
      });
    }
    await setTheme(page, "light");
    await shoot(page, "05-empty-state-light.png");
    await setTheme(page, "dark");
    await shoot(page, "06-empty-state-dark.png");
  });

  test("07 settings modal", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => {
      window.localStorage.setItem(
        "bioagent-settings",
        JSON.stringify({ p: "sjtu", k: btoa("k"), u: "https://x/v1", m: "deepseek-chat", t: 2400 }),
      );
    });
    await page.reload();
    await setTheme(page, "light");
    if (await page.locator(".init-dialog-scrim").isVisible().catch(() => false)) {
      await page.evaluate(() => {
        const el = document.querySelector(".init-dialog-scrim") as HTMLElement | null;
        if (el) el.style.display = "none";
      });
    }
    await page.keyboard.press("Control+Comma");
    await expect(page.locator(".settings-modal")).toBeVisible();
    await shoot(page, "07-settings-modal-light.png");
  });

  test("08 command palette", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => {
      window.localStorage.setItem(
        "bioagent-settings",
        JSON.stringify({ p: "sjtu", k: btoa("k"), u: "https://x/v1", m: "deepseek-chat", t: 2400 }),
      );
    });
    await page.reload();
    await setTheme(page, "light");
    if (await page.locator(".init-dialog-scrim").isVisible().catch(() => false)) {
      await page.evaluate(() => {
        const el = document.querySelector(".init-dialog-scrim") as HTMLElement | null;
        if (el) el.style.display = "none";
      });
    }
    await page.keyboard.press("Control+K");
    await expect(page.locator(".command-palette")).toBeVisible();
    await shoot(page, "08-command-palette-light.png");
  });

  test("09 chat collapsed rail", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => {
      window.localStorage.setItem(
        "bioagent-settings",
        JSON.stringify({ p: "sjtu", k: btoa("k"), u: "https://x/v1", m: "deepseek-chat", t: 2400 }),
      );
      window.localStorage.setItem(
        "bioagent-chat-width",
        JSON.stringify({ width: 32, collapsed: true, lastExpandedWidth: 320 }),
      );
    });
    await page.reload();
    await setTheme(page, "light");
    if (await page.locator(".init-dialog-scrim").isVisible().catch(() => false)) {
      await page.evaluate(() => {
        const el = document.querySelector(".init-dialog-scrim") as HTMLElement | null;
        if (el) el.style.display = "none";
      });
    }
    await expect(page.locator(".app-shell.chat-collapsed")).toBeVisible();
    await shoot(page, "09-chat-collapsed-light.png");
  });

  test("10 sidebar collapsed", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => {
      window.localStorage.setItem(
        "bioagent-settings",
        JSON.stringify({ p: "sjtu", k: btoa("k"), u: "https://x/v1", m: "deepseek-chat", t: 2400 }),
      );
      window.localStorage.setItem("bioagent-sidebar-collapsed", "1");
    });
    await page.reload();
    await setTheme(page, "light");
    if (await page.locator(".init-dialog-scrim").isVisible().catch(() => false)) {
      await page.evaluate(() => {
        const el = document.querySelector(".init-dialog-scrim") as HTMLElement | null;
        if (el) el.style.display = "none";
      });
    }
    await expect(page.locator(".app-shell.sidebar-collapsed")).toBeVisible();
    await shoot(page, "10-sidebar-collapsed-light.png");
  });

  test("11 shortcuts overlay", async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => {
      window.localStorage.setItem(
        "bioagent-settings",
        JSON.stringify({ p: "sjtu", k: btoa("k"), u: "https://x/v1", m: "deepseek-chat", t: 2400 }),
      );
    });
    await page.reload();
    await setTheme(page, "light");
    if (await page.locator(".init-dialog-scrim").isVisible().catch(() => false)) {
      await page.evaluate(() => {
        const el = document.querySelector(".init-dialog-scrim") as HTMLElement | null;
        if (el) el.style.display = "none";
      });
    }
    await page.keyboard.press("Shift+/");
    if (await page.locator(".shortcuts-overlay").isVisible().catch(() => false)) {
      await shoot(page, "11-shortcuts-overlay-light.png");
    } else {
      // dispatch '?' directly if Shift+/ didn't register
      await page.evaluate(() => {
        const ev = new KeyboardEvent("keydown", { key: "?" });
        document.dispatchEvent(ev);
      });
      await page.waitForTimeout(200);
      await shoot(page, "11-shortcuts-overlay-light.png");
    }
  });
});
