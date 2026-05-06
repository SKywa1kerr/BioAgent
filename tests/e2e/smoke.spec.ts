import { test, expect } from "@playwright/test";

test.describe("BioAgent Desktop smoke", () => {
  test("renders the app shell and chat composer", async ({ page }) => {
    await page.goto("/");
    // App-shell root class is "app-shell" with a rail-* modifier (see src/App.tsx).
    const shell = page.locator(".app-shell");
    await expect(shell).toBeVisible();

    // The chat textarea is the renderer's primary input. It lives inside the
    // .composer wrapper in ChatPanel.tsx. The textarea is initially disabled
    // (no harness yet) but should still be present in the DOM.
    const composer = page.locator(".composer textarea");
    await expect(composer).toHaveCount(1);
  });

  test("Ctrl+K opens the command palette and Escape closes it", async ({ page }) => {
    await page.goto("/");

    // Wait for the app shell to mount before sending shortcuts — otherwise the
    // global keydown listener (registered in App's effect) may not be wired up yet.
    await expect(page.locator(".app-shell")).toBeVisible();

    await page.keyboard.press("Control+K");
    // CommandPalette renders directly into the React tree (no portal). Look
    // for the inner dialog with class "command-palette" (see CommandPalette.tsx).
    const palette = page.locator(".command-palette");
    await expect(palette).toBeVisible();

    // CommandPalette focuses its input via requestAnimationFrame after opening,
    // and binds the Escape handler on the inner dialog (not document). Press
    // Escape on the input directly so we don't race the rAF focus call.
    const paletteInput = page.locator(".command-palette-input");
    await expect(paletteInput).toBeVisible();
    await paletteInput.press("Escape");
    await expect(palette).toHaveCount(0);
  });

  test("Ctrl+, opens the settings modal and Escape closes it", async ({ page }) => {
    await page.goto("/");

    await expect(page.locator(".app-shell")).toBeVisible();

    await page.keyboard.press("Control+Comma");
    // SettingsModal renders an overlay (settings-modal-overlay) wrapping a
    // dialog with class "settings-modal" (see SettingsModal.tsx).
    const modal = page.locator(".settings-modal");
    await expect(modal).toBeVisible();

    // Settings modal's Escape is handled by App's global keydown listener —
    // dispatching Escape from the page level is enough.
    await page.keyboard.press("Escape");
    await expect(modal).toHaveCount(0);
  });
});
