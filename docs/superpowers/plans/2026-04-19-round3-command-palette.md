# Round 3 Implementation Plan — Command Palette + Onboarding + Keyboard Nav

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ctrl+K command palette, 3-step corner onboarding, and "?" shortcut overlay — with a global focus-visible audit — to BioAgent Desktop.

**Architecture:** Module-level `commandRegistry` (pure JS, testable). Each owner component registers its own commands in `useEffect` (App owns nav/appearance/examples/log; ResultsWorkbench owns export/clear-filters). Ctrl+K / "?" global listeners live in App and respect a `isAnyModalOpen` flag. Onboarding state is a 2-state flag in localStorage. Focus ring is unified via a global `:focus-visible` rule.

**Tech Stack:** React 18 + TypeScript, Vite 5, Electron 33. Pure JS (`.js` + `.d.ts`) for pure modules so `node --test` can consume them; TSX for React components.

**Spec:** `docs/superpowers/specs/2026-04-18-round3-command-palette-design.md`

---

## File Structure

**New files:**
- `src/lib/commands/registry.js` + `.d.ts` — register/get/filter/clear
- `src/lib/commands/fuzzy.js` + `.d.ts` — `fuzzyScore(text, query)`
- `src/lib/commands/shortcuts.ts` — UI single source for shortcut list
- `src/lib/exporters/runExport.ts` — format-agnostic `runExport(fmt, args)` used by ExportMenu and palette
- `src/components/CommandPalette.tsx` + `.css`
- `src/hooks/useOnboarding.ts` + `useOnboarding.js` + `.d.ts` (pure helpers)
- `src/components/OnboardingCoach.tsx` + `.css`
- `src/components/ShortcutsOverlay.tsx` + `.css`
- `tests/test_command_registry.mjs`
- `tests/test_fuzzy.mjs`
- `tests/test_onboarding_store.mjs`

**Modified files:**
- `src/App.tsx` — register cross-cutting commands; mount 3 new components; global Ctrl+K / "?" listeners; track `isAnyModalOpen`
- `src/i18n.ts` — new strings (zh + en)
- `src/components/ChatPanel.tsx` — expose `prefillChat(text)` via forwardRef or a callback prop; empty-state "Open command palette" CTA
- `src/components/workbench/ExportMenu.tsx` — delegate to `runExport`
- `src/components/workbench/ResultsWorkbench.tsx` — register workbench commands via `useWorkbenchCommands` hook
- `src/index.css` (or the root stylesheet in use) — global `:focus-visible`

---

# Phase 1: Command registry + fuzzy search + palette UI

### Task 1.1: Pure registry with tests

**Files:**
- Create: `src/lib/commands/registry.js`
- Create: `src/lib/commands/registry.d.ts`
- Create: `tests/test_command_registry.mjs`

- [ ] **Step 1: Write failing tests** at `tests/test_command_registry.mjs`

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  registerCommand,
  getCommands,
  filterCommands,
  clearCommands,
} from "../src/lib/commands/registry.js";

function makeCmd(id, overrides = {}) {
  return {
    id,
    title: overrides.title ?? id,
    group: overrides.group ?? "nav",
    keywords: overrides.keywords,
    when: overrides.when,
    run: overrides.run ?? (() => {}),
  };
}

test("registerCommand stores commands in insertion order", () => {
  clearCommands();
  registerCommand(makeCmd("a"));
  registerCommand(makeCmd("b"));
  assert.deepEqual(getCommands().map((c) => c.id), ["a", "b"]);
});

test("registerCommand returns an unregister function", () => {
  clearCommands();
  const off = registerCommand(makeCmd("a"));
  off();
  assert.equal(getCommands().length, 0);
});

test("registerCommand with duplicate id replaces the earlier entry", () => {
  clearCommands();
  registerCommand(makeCmd("a", { title: "first" }));
  registerCommand(makeCmd("a", { title: "second" }));
  const cmds = getCommands();
  assert.equal(cmds.length, 1);
  assert.equal(cmds[0].title, "second");
});

test("filterCommands hides entries whose when() returns false", () => {
  clearCommands();
  registerCommand(makeCmd("a", { when: () => true }));
  registerCommand(makeCmd("b", { when: () => false }));
  const ids = filterCommands("").map((c) => c.id);
  assert.deepEqual(ids, ["a"]);
});

test("clearCommands empties the registry", () => {
  registerCommand(makeCmd("a"));
  clearCommands();
  assert.equal(getCommands().length, 0);
});
```

- [ ] **Step 2: Run tests → FAIL (module not found)**

`node --test tests/test_command_registry.mjs`

- [ ] **Step 3: Implement `src/lib/commands/registry.js`**

```js
const commands = new Map();

export function registerCommand(cmd) {
  if (!cmd || typeof cmd.id !== "string" || typeof cmd.run !== "function") {
    throw new Error("registerCommand: id and run are required");
  }
  commands.set(cmd.id, cmd);
  return function unregister() {
    const existing = commands.get(cmd.id);
    if (existing === cmd) commands.delete(cmd.id);
  };
}

export function getCommands() {
  return Array.from(commands.values());
}

export function filterCommands(query) {
  const visible = getCommands().filter((c) => (typeof c.when === "function" ? c.when() : true));
  const q = (query ?? "").trim();
  if (!q) return visible;
  // fuzzy scoring plugged in from fuzzy.js in Task 1.2
  return visible;
}

export function clearCommands() {
  commands.clear();
}
```

- [ ] **Step 4: Create `.d.ts`**

```ts
export type CommandGroup = "nav" | "workbench" | "appearance" | "examples" | "log";

export interface Command {
  id: string;
  title: string;
  group: CommandGroup;
  keywords?: string[];
  shortcut?: string;
  when?: () => boolean;
  run: () => void | Promise<void>;
}

export function registerCommand(cmd: Command): () => void;
export function getCommands(): Command[];
export function filterCommands(query: string): Command[];
export function clearCommands(): void;
```

- [ ] **Step 5: Run tests → 5 passing**

- [ ] **Step 6: Commit**

```bash
git add src/lib/commands/registry.js src/lib/commands/registry.d.ts tests/test_command_registry.mjs
git commit -m "feat(commands): pure registry with tests"
```

---

### Task 1.2: Fuzzy matcher with tests

**Files:**
- Create: `src/lib/commands/fuzzy.js` + `.d.ts`
- Create: `tests/test_fuzzy.mjs`

- [ ] **Step 1: Write failing tests** at `tests/test_fuzzy.mjs`

```js
import test from "node:test";
import assert from "node:assert/strict";
import { fuzzyScore } from "../src/lib/commands/fuzzy.js";

test("exact prefix scores higher than sparse match", () => {
  const exact = fuzzyScore("export csv", "exp");
  const sparse = fuzzyScore("export csv", "ecv");
  assert.ok(exact > sparse, `exact ${exact} should exceed sparse ${sparse}`);
});

test("non-subsequence returns -1", () => {
  assert.equal(fuzzyScore("export csv", "zzz"), -1);
});

test("subsequence match is positive", () => {
  assert.ok(fuzzyScore("export csv", "ecv") > 0);
});

test("case-insensitive", () => {
  assert.ok(fuzzyScore("Export CSV", "csv") > 0);
  assert.ok(fuzzyScore("EXPORT", "exp") > 0);
});

test("empty query scores 0", () => {
  assert.equal(fuzzyScore("anything", ""), 0);
});

test("query longer than text returns -1", () => {
  assert.equal(fuzzyScore("abc", "abcd"), -1);
});
```

- [ ] **Step 2: Run tests → FAIL**

- [ ] **Step 3: Implement `src/lib/commands/fuzzy.js`**

```js
// Returns -1 when query characters are not a subsequence of text.
// Score prefers contiguous matches and early matches.
export function fuzzyScore(text, query) {
  if (typeof text !== "string" || typeof query !== "string") return -1;
  if (query.length === 0) return 0;
  if (query.length > text.length) return -1;
  const t = text.toLowerCase();
  const q = query.toLowerCase();
  let ti = 0;
  let qi = 0;
  let score = 0;
  let lastMatch = -2;
  while (ti < t.length && qi < q.length) {
    if (t[ti] === q[qi]) {
      // contiguous bonus
      score += ti - lastMatch === 1 ? 3 : 1;
      // early bonus (higher when match is near start)
      score += Math.max(0, 10 - ti);
      lastMatch = ti;
      qi++;
    }
    ti++;
  }
  if (qi < q.length) return -1;
  return score;
}
```

- [ ] **Step 4: Create `.d.ts`**

```ts
export function fuzzyScore(text: string, query: string): number;
```

- [ ] **Step 5: Run tests → 6 passing**

- [ ] **Step 6: Plug into registry** — edit `src/lib/commands/registry.js`:

```js
import { fuzzyScore } from "./fuzzy.js";

// ... existing registerCommand / getCommands / clearCommands unchanged

export function filterCommands(query) {
  const visible = getCommands().filter((c) => (typeof c.when === "function" ? c.when() : true));
  const q = (query ?? "").trim();
  if (!q) return visible;
  const scored = [];
  for (const cmd of visible) {
    const hay = [cmd.title, ...(cmd.keywords ?? [])].join(" ");
    const s = fuzzyScore(hay, q);
    if (s >= 0) scored.push({ cmd, s });
  }
  scored.sort((a, b) => b.s - a.s);
  return scored.map((x) => x.cmd);
}
```

- [ ] **Step 7: Extend registry tests** — add to `tests/test_command_registry.mjs`:

```js
test("filterCommands ranks better matches first", () => {
  clearCommands();
  registerCommand(makeCmd("a", { title: "export csv" }));
  registerCommand(makeCmd("b", { title: "toggle theme" }));
  const ids = filterCommands("export").map((c) => c.id);
  assert.deepEqual(ids, ["a"]);
});

test("filterCommands searches keywords", () => {
  clearCommands();
  registerCommand(makeCmd("a", { title: "导出 CSV", keywords: ["export", "csv"] }));
  const ids = filterCommands("export").map((c) => c.id);
  assert.deepEqual(ids, ["a"]);
});
```

- [ ] **Step 8: Re-run both test files → 7 + 6 = 13 passing**

- [ ] **Step 9: Commit**

```bash
git add src/lib/commands/fuzzy.* src/lib/commands/registry.js tests/test_fuzzy.mjs tests/test_command_registry.mjs
git commit -m "feat(commands): fuzzy matcher and scoring in filterCommands"
```

---

### Task 1.3: CommandPalette component

**Files:**
- Create: `src/components/CommandPalette.tsx`
- Create: `src/components/CommandPalette.css`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Add i18n strings**

Append to `zh` dict in `src/i18n.ts`:

```ts
"palette.title": "命令面板",
"palette.placeholder": "搜索命令...",
"palette.empty": "未找到命令",
"palette.groupNav": "导航",
"palette.groupWorkbench": "工作台",
"palette.groupAppearance": "外观",
"palette.groupExamples": "示例 prompt",
"palette.groupLog": "日志",
"palette.cmd.focusChat": "聚焦聊天输入",
"palette.cmd.openSettings": "打开设置",
"palette.cmd.toggleTheme": "切换浅色/深色",
"palette.cmd.toggleLang": "切换中文/English",
"palette.cmd.exportDebug": "导出调试日志",
"palette.cmd.example.base": "分析 base 数据集",
"palette.cmd.example.pro": "分析 pro 数据集",
"palette.cmd.example.trends": "显示突变趋势",
"palette.cmd.example.suggestions": "给出实验建议",
"palette.cmd.tabAnalysis": "切换到 序列分析",
"palette.cmd.tabTrends": "切换到 突变趋势",
"palette.cmd.tabSuggestions": "切换到 实验建议",
"palette.cmd.exportCsv": "导出当前结果为 CSV",
"palette.cmd.exportJson": "导出当前结果为 JSON",
"palette.cmd.exportPdf": "导出当前结果为 PDF",
"palette.cmd.clearFilters": "清除工作台筛选",
```

Append same keys to `en` dict with English values (`"Command palette"`, `"Search commands..."`, `"No commands found"`, etc.).

- [ ] **Step 2: Write `src/components/CommandPalette.tsx`**

```tsx
import { useEffect, useMemo, useRef, useState } from "react";
import { filterCommands } from "../lib/commands/registry";
import type { Command, CommandGroup } from "../lib/commands/registry";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import "./CommandPalette.css";

const GROUP_ORDER: CommandGroup[] = ["nav", "workbench", "appearance", "examples", "log"];
const GROUP_LABEL_KEY: Record<CommandGroup, string> = {
  nav: "palette.groupNav",
  workbench: "palette.groupWorkbench",
  appearance: "palette.groupAppearance",
  examples: "palette.groupExamples",
  log: "palette.groupLog",
};

interface Props {
  open: boolean;
  onClose: () => void;
  language: AppLanguage;
}

export function CommandPalette({ open, onClose, language }: Props) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    setQuery("");
    setSelectedIndex(0);
    requestAnimationFrame(() => inputRef.current?.focus());
    return () => {
      previouslyFocusedRef.current?.focus?.();
    };
  }, [open]);

  const items = useMemo(() => (open ? filterCommands(query) : []), [open, query]);

  useEffect(() => {
    if (selectedIndex >= items.length) setSelectedIndex(Math.max(0, items.length - 1));
  }, [items, selectedIndex]);

  if (!open) return null;

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(items.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const cmd = items[selectedIndex];
      if (cmd) runAndClose(cmd);
    } else if (e.key === "Tab") {
      // Keep focus inside palette.
      e.preventDefault();
    }
  }

  async function runAndClose(cmd: Command) {
    onClose();
    try {
      await cmd.run();
    } catch (err) {
      console.error(`Command "${cmd.id}" failed:`, err);
    }
  }

  const grouped = groupItems(items);

  return (
    <div className="command-palette-scrim" onMouseDown={onClose} role="presentation">
      <div
        className="command-palette"
        role="dialog"
        aria-label={t(language, "palette.title")}
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <input
          ref={inputRef}
          type="text"
          className="command-palette-input"
          placeholder={t(language, "palette.placeholder")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label={t(language, "palette.placeholder")}
        />
        <div className="command-palette-list" role="listbox">
          {items.length === 0 ? (
            <div className="command-palette-empty">{t(language, "palette.empty")}</div>
          ) : (
            GROUP_ORDER.map((g) => {
              const list = grouped.get(g);
              if (!list || list.length === 0) return null;
              return (
                <div key={g} className="command-palette-group">
                  <div className="command-palette-group-label">{t(language, GROUP_LABEL_KEY[g])}</div>
                  {list.map((cmd) => {
                    const flatIndex = items.indexOf(cmd);
                    const active = flatIndex === selectedIndex;
                    return (
                      <button
                        key={cmd.id}
                        type="button"
                        role="option"
                        aria-selected={active}
                        className={`command-palette-item${active ? " is-active" : ""}`}
                        onMouseEnter={() => setSelectedIndex(flatIndex)}
                        onClick={() => runAndClose(cmd)}
                      >
                        <span className="command-palette-item-title">{cmd.title}</span>
                        {cmd.shortcut ? <span className="command-palette-item-shortcut">{cmd.shortcut}</span> : null}
                      </button>
                    );
                  })}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

function groupItems(items: Command[]): Map<CommandGroup, Command[]> {
  const m = new Map<CommandGroup, Command[]>();
  for (const cmd of items) {
    const list = m.get(cmd.group) ?? [];
    list.push(cmd);
    m.set(cmd.group, list);
  }
  return m;
}
```

- [ ] **Step 3: Create `CommandPalette.css`**

```css
.command-palette-scrim {
  position: fixed; inset: 0;
  background: rgba(10, 15, 25, 0.4);
  display: flex; align-items: flex-start; justify-content: center;
  padding-top: 10vh;
  z-index: 50;
}

.command-palette {
  width: min(600px, 92vw);
  background: var(--results-panel-bg, #fff);
  color: var(--results-panel-text, #0f172a);
  border: 1px solid var(--results-panel-border, rgba(0,0,0,0.08));
  border-radius: 12px;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.35);
  overflow: hidden;
  display: flex; flex-direction: column;
  max-height: 60vh;
}

.command-palette-input {
  width: 100%;
  padding: 14px 18px;
  border: none;
  border-bottom: 1px solid var(--results-panel-border, rgba(0,0,0,0.08));
  background: transparent;
  color: inherit;
  font-size: 15px;
  outline: none;
}

.command-palette-list { overflow: auto; padding: 4px 0; }

.command-palette-empty {
  padding: 24px; text-align: center; color: var(--results-panel-muted, #64748b);
  font-size: 13px;
}

.command-palette-group { padding: 4px 0; }
.command-palette-group-label {
  padding: 6px 16px 2px;
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  color: var(--results-panel-muted, #64748b);
}

.command-palette-item {
  display: flex; align-items: center; justify-content: space-between;
  width: 100%;
  padding: 8px 16px;
  background: transparent; border: none; color: inherit;
  font-size: 13px; text-align: left; cursor: pointer;
}
.command-palette-item.is-active,
.command-palette-item:focus-visible {
  background: var(--results-primary-soft, rgba(31, 120, 193, 0.14));
  color: var(--results-primary, #1f78c1);
  outline: none;
}
.command-palette-item-shortcut {
  font-family: ui-monospace, Menlo, monospace;
  font-size: 11px;
  color: var(--results-panel-muted, #64748b);
}
```

- [ ] **Step 4: Typecheck**

`npx tsc --noEmit` → expected: no errors (palette is not yet mounted).

- [ ] **Step 5: Commit**

```bash
git add src/components/CommandPalette.tsx src/components/CommandPalette.css src/i18n.ts
git commit -m "feat(commands): command palette UI with fuzzy filter + keyboard nav"
```

---

### Task 1.4: Mount palette in App + register cross-cutting commands + Ctrl+K

**Files:**
- Modify: `src/App.tsx`

- [ ] **Step 1: Identify current shortcut handler**

Open `src/App.tsx` — find the existing `handleGlobalKeyDown` effect (around line 77-90). That's where Ctrl+L and Ctrl+, live today.

- [ ] **Step 2: Add palette state + cross-cutting commands**

Near the top of `App`:

```tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CommandPalette } from "./components/CommandPalette";
import { registerCommand, clearCommands } from "./lib/commands/registry";
```

Inside `App`:

```tsx
const [paletteOpen, setPaletteOpen] = useState(false);
const chatInputRef = useRef<HTMLTextAreaElement | null>(null);

const prefillChat = useCallback((text: string) => {
  // Implementation wired via ChatPanel prop in Task 1.5.
}, []);

const focusChat = useCallback(() => {
  chatInputRef.current?.focus();
}, []);

const toggleTheme = useCallback(() => {
  setTheme((t) => (t === "light" ? "dark" : "light"));
}, []);
const toggleLanguage = useCallback(() => {
  setLanguage((l) => (l === "zh" ? "en" : "zh"));
}, []);
const openSettings = useCallback(() => setSettingsOpen(true), []);

const isAnyModalOpen = settingsOpen || !!agent.confirmMessage || /* paletteOpen intentionally excluded */ false;
```

- [ ] **Step 3: Register nav/appearance/examples/log commands**

```tsx
useEffect(() => {
  const offs: Array<() => void> = [];

  offs.push(
    registerCommand({
      id: "nav.focus-chat",
      title: t(language, "palette.cmd.focusChat"),
      group: "nav",
      shortcut: "Ctrl+L",
      run: focusChat,
    }),
    registerCommand({
      id: "nav.open-settings",
      title: t(language, "palette.cmd.openSettings"),
      group: "nav",
      shortcut: "Ctrl+,",
      run: openSettings,
    }),
    registerCommand({
      id: "nav.tab-analysis",
      title: t(language, "palette.cmd.tabAnalysis"),
      group: "nav",
      when: () => panelCache.analysis != null,
      run: () => setActiveTab("analysis"),
    }),
    registerCommand({
      id: "nav.tab-trends",
      title: t(language, "palette.cmd.tabTrends"),
      group: "nav",
      when: () => panelCache.trends != null,
      run: () => setActiveTab("trends"),
    }),
    registerCommand({
      id: "nav.tab-suggestions",
      title: t(language, "palette.cmd.tabSuggestions"),
      group: "nav",
      when: () => panelCache.suggestions != null,
      run: () => setActiveTab("suggestions"),
    }),
    registerCommand({
      id: "appearance.toggle-theme",
      title: t(language, "palette.cmd.toggleTheme"),
      group: "appearance",
      run: toggleTheme,
    }),
    registerCommand({
      id: "appearance.toggle-lang",
      title: t(language, "palette.cmd.toggleLang"),
      group: "appearance",
      run: toggleLanguage,
    }),
    registerCommand({
      id: "log.export-debug",
      title: t(language, "palette.cmd.exportDebug"),
      group: "log",
      run: () => agent.exportDebugLog(),
    }),
    registerCommand({
      id: "examples.analyze-base",
      title: t(language, "palette.cmd.example.base"),
      group: "examples",
      run: () => prefillChat("分析 base 数据集"),
    }),
    registerCommand({
      id: "examples.analyze-pro",
      title: t(language, "palette.cmd.example.pro"),
      group: "examples",
      run: () => prefillChat("分析 pro 数据集"),
    }),
    registerCommand({
      id: "examples.trends",
      title: t(language, "palette.cmd.example.trends"),
      group: "examples",
      run: () => prefillChat("显示突变趋势"),
    }),
    registerCommand({
      id: "examples.suggestions",
      title: t(language, "palette.cmd.example.suggestions"),
      group: "examples",
      run: () => prefillChat("给出实验建议"),
    }),
  );

  return () => { offs.forEach((off) => off()); };
}, [language, focusChat, openSettings, toggleTheme, toggleLanguage, agent, panelCache.analysis, panelCache.trends, panelCache.suggestions, prefillChat]);
```

- [ ] **Step 4: Add Ctrl+K listener**

Inside the existing global keydown effect, add before the existing Ctrl+L handler:

```tsx
if (mod && e.key.toLowerCase() === "k") {
  e.preventDefault();
  if (isAnyModalOpen) return;      // ignore while another modal is open
  setPaletteOpen((v) => !v);
  return;
}
```

- [ ] **Step 5: Render `<CommandPalette />`**

At the end of App's JSX (alongside SettingsModal etc.):

```tsx
<CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} language={language} />
```

- [ ] **Step 6: Typecheck + dev smoke**

`npx tsc --noEmit` → expected clean.
`npm run dev` → Ctrl+K opens palette; type "设" → see 打开设置; Enter opens settings modal and palette closes; Esc restores focus.

Known limitation this task leaves: `prefillChat` is a no-op until Task 1.5 wires ChatPanel. Palette example commands will fire but do nothing yet — acceptable checkpoint.

- [ ] **Step 7: Commit**

```bash
git add src/App.tsx
git commit -m "feat(app): register cross-cutting commands + Ctrl+K palette toggle"
```

---

### Task 1.5: ChatPanel prefill + empty-state CTA

**Files:**
- Modify: `src/components/ChatPanel.tsx`
- Modify: `src/App.tsx`

- [ ] **Step 1: Locate ChatPanel input**

Open `src/components/ChatPanel.tsx`. The textarea lives here and is backed by a `useState` string.

- [ ] **Step 2: Expose a `prefillText` prop + ref for focus**

Add to `ChatPanelProps` (find existing interface):

```ts
prefillText?: string | null;
onPrefillConsumed?: () => void;
inputRef?: React.RefObject<HTMLTextAreaElement>;
onOpenPalette?: () => void;
```

Inside the component, add effect:

```tsx
useEffect(() => {
  if (props.prefillText != null) {
    setInputValue(props.prefillText);
    props.inputRef?.current?.focus();
    props.onPrefillConsumed?.();
  }
}, [props.prefillText]);
```

Attach `ref={props.inputRef}` to the textarea.

Inside the empty-state block (where the ready hint is rendered when `messages.length === 0`), add under the existing body:

```tsx
{props.onOpenPalette ? (
  <button
    type="button"
    className="chat-empty-cta"
    onClick={props.onOpenPalette}
  >
    {t(language, "chat.openPalette")}
  </button>
) : null}
```

- [ ] **Step 3: Add i18n strings**

```
zh: "chat.openPalette": "打开命令面板 (Ctrl+K)"
en: "chat.openPalette": "Open command palette (Ctrl+K)"
```

- [ ] **Step 4: Add minimal CSS**

Append to `ChatPanel.css` (or the file that styles the empty state):

```css
.chat-empty-cta {
  margin-top: 8px;
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid var(--results-panel-border);
  background: transparent;
  color: var(--results-primary);
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
}
.chat-empty-cta:hover { background: var(--results-primary-soft); }
```

- [ ] **Step 5: Wire from App.tsx**

Inside App:

```tsx
const [prefillText, setPrefillText] = useState<string | null>(null);
const prefillChat = useCallback((text: string) => setPrefillText(text), []);
```

Replace the stub in Task 1.4's `prefillChat` with this real one.

Pass props to `<ChatPanel ... prefillText={prefillText} onPrefillConsumed={() => setPrefillText(null)} inputRef={chatInputRef} onOpenPalette={() => setPaletteOpen(true)} />`.

- [ ] **Step 6: Dev smoke**

Open palette → pick "分析 pro 数据集" → chat input filled and focused; empty state shows "Open command palette" button.

- [ ] **Step 7: Typecheck + commit**

```bash
npx tsc --noEmit
git add src/components/ChatPanel.tsx src/components/ChatPanel.css src/App.tsx src/i18n.ts
git commit -m "feat(chat): prefill from palette + empty-state CTA"
```

---

# Phase 2: Workbench commands

### Task 2.1: Extract `runExport` helper

**Files:**
- Create: `src/lib/exporters/runExport.ts`
- Modify: `src/components/workbench/ExportMenu.tsx`

- [ ] **Step 1: Create `runExport.ts`**

```ts
import type { WorkbenchSample } from "../../components/workbench/types";
import type { AppLanguage } from "../../i18n";
import { samplesToCsv } from "./csv";
import { samplesToJson } from "./json";
import { buildExportFilename } from "./filename";
import { saveFile } from "./saveFile";

export type ExportFormat = "csv" | "json" | "pdf";

export interface RunExportArgs {
  samples: WorkbenchSample[];
  filters: { statusFilter: string; searchQuery: string; sortKey: string };
  dataset?: string;
  language: AppLanguage;
  onWarn?: (message: string) => void;
}

export async function runExport(fmt: ExportFormat, args: RunExportArgs): Promise<void> {
  if (fmt === "csv") {
    await saveFile({
      filename: buildExportFilename({ dataset: args.dataset, ext: "csv" }),
      mime: "text/csv;charset=utf-8",
      data: samplesToCsv(args.samples),
    });
    return;
  }
  if (fmt === "json") {
    await saveFile({
      filename: buildExportFilename({ dataset: args.dataset, ext: "json" }),
      mime: "application/json;charset=utf-8",
      data: samplesToJson(args.samples, { filters: args.filters }),
    });
    return;
  }
  if (fmt === "pdf") {
    const { exportPdf } = await import("./pdf");
    await exportPdf(args);
    return;
  }
  throw new Error(`Unknown export format: ${fmt}`);
}
```

- [ ] **Step 2: Delegate from ExportMenu**

In `src/components/workbench/ExportMenu.tsx`, replace the format-specific branches inside `exportAs` with:

```tsx
await runExport(fmt, {
  samples,
  filters,
  dataset,
  language,
  onWarn: (text) => setMessage({ tone: "info", text }),
});
```

Import `runExport` at top. Remove the now-unused imports (`samplesToCsv`, `samplesToJson`, `buildExportFilename`, `saveFile` — keep if other code in the file uses them; grep first).

- [ ] **Step 3: Run existing tests**

`node --test tests/*.mjs` → all previous tests still pass (runExport isn't tested directly — its parts already are).

- [ ] **Step 4: Dev smoke**

Export CSV / JSON / PDF from the button → behavior unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/lib/exporters/runExport.ts src/components/workbench/ExportMenu.tsx
git commit -m "refactor(exporters): extract runExport helper shared by UI + palette"
```

---

### Task 2.2: Register workbench commands from `ResultsWorkbench`

**Files:**
- Modify: `src/components/workbench/ResultsWorkbench.tsx`

- [ ] **Step 1: Add registration effect**

Inside `ResultsWorkbench`, after `visibleSamples` / `summarySource` are computed:

```tsx
useEffect(() => {
  const offs: Array<() => void> = [];
  const args = {
    samples: visibleSamples,
    filters: { statusFilter, searchQuery, sortKey },
    dataset,
    language,
  };
  offs.push(
    registerCommand({
      id: "workbench.export-csv",
      title: t(language, "palette.cmd.exportCsv"),
      group: "workbench",
      keywords: ["export", "csv", "导出"],
      when: () => visibleSamples.length > 0,
      run: () => runExport("csv", args),
    }),
    registerCommand({
      id: "workbench.export-json",
      title: t(language, "palette.cmd.exportJson"),
      group: "workbench",
      keywords: ["export", "json", "导出"],
      when: () => visibleSamples.length > 0,
      run: () => runExport("json", args),
    }),
    registerCommand({
      id: "workbench.export-pdf",
      title: t(language, "palette.cmd.exportPdf"),
      group: "workbench",
      keywords: ["export", "pdf", "报告", "导出"],
      when: () => visibleSamples.length > 0,
      run: () => runExport("pdf", args),
    }),
    registerCommand({
      id: "workbench.clear-filters",
      title: t(language, "palette.cmd.clearFilters"),
      group: "workbench",
      keywords: ["clear", "reset", "清除"],
      when: () => hasActiveControls,
      run: reset,
    }),
  );
  return () => { offs.forEach((off) => off()); };
}, [visibleSamples, statusFilter, searchQuery, sortKey, language, dataset, hasActiveControls, reset]);
```

Imports:

```tsx
import { useEffect } from "react";
import { registerCommand } from "../../lib/commands/registry";
import { runExport } from "../../lib/exporters/runExport";
```

- [ ] **Step 2: Verify `when` gates**

If no samples load yet (initial state), Ctrl+K should not show export commands. Manual test: open app → Ctrl+K → workbench group is empty/absent.

- [ ] **Step 3: Dev smoke**

Load a dataset → Ctrl+K → "导出 CSV" / "清除筛选" appear under 工作台; Enter triggers export.

- [ ] **Step 4: Commit**

```bash
git add src/components/workbench/ResultsWorkbench.tsx
git commit -m "feat(workbench): register export + clear-filters commands in palette"
```

---

# Phase 3: Onboarding

### Task 3.1: Pure onboarding store + tests

**Files:**
- Create: `src/hooks/useOnboarding.js` + `.d.ts`
- Create: `tests/test_onboarding_store.mjs`

- [ ] **Step 1: Failing test** at `tests/test_onboarding_store.mjs`

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  readOnboarding,
  writeOnboarding,
  ONBOARDING_STORAGE_KEY,
} from "../src/hooks/useOnboarding.js";

function makeStore() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
  };
}

test("readOnboarding returns false when storage empty", () => {
  assert.equal(readOnboarding(makeStore()), false);
});

test("writeOnboarding(true) then read returns true", () => {
  const s = makeStore();
  writeOnboarding(s, true);
  assert.equal(readOnboarding(s), true);
});

test("readOnboarding tolerates invalid values", () => {
  const s = makeStore();
  s.setItem(ONBOARDING_STORAGE_KEY, "garbage");
  assert.equal(readOnboarding(s), false);
});

test("writeOnboarding(false) removes the key", () => {
  const s = makeStore();
  writeOnboarding(s, true);
  writeOnboarding(s, false);
  assert.equal(s.getItem(ONBOARDING_STORAGE_KEY), null);
});
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement `useOnboarding.js`**

```js
export const ONBOARDING_STORAGE_KEY = "bioagent-onboarding-v1";

export function readOnboarding(storage) {
  try {
    return storage.getItem(ONBOARDING_STORAGE_KEY) === "complete";
  } catch {
    return false;
  }
}

export function writeOnboarding(storage, complete) {
  try {
    if (complete) storage.setItem(ONBOARDING_STORAGE_KEY, "complete");
    else storage.removeItem(ONBOARDING_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
```

- [ ] **Step 4: Create `.d.ts`**

```ts
export const ONBOARDING_STORAGE_KEY: string;
export interface OnboardingStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}
export function readOnboarding(storage: OnboardingStorage): boolean;
export function writeOnboarding(storage: OnboardingStorage, complete: boolean): void;
```

- [ ] **Step 5: Run → 4 passing → Commit**

```bash
git add src/hooks/useOnboarding.js src/hooks/useOnboarding.d.ts tests/test_onboarding_store.mjs
git commit -m "feat(onboarding): pure persistence helpers with tests"
```

---

### Task 3.2: React hook wrapper + OnboardingCoach component

**Files:**
- Create: `src/hooks/useOnboarding.ts`
- Create: `src/components/OnboardingCoach.tsx`
- Create: `src/components/OnboardingCoach.css`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Hook wrapper `useOnboarding.ts`**

```ts
import { useCallback, useEffect, useRef, useState } from "react";
import { readOnboarding, writeOnboarding } from "./useOnboarding.js";

export function useOnboarding() {
  const storageRef = useRef<Storage | null>(null);
  if (storageRef.current === null) {
    try {
      storageRef.current = typeof window !== "undefined" ? window.localStorage : null;
    } catch {
      storageRef.current = null;
    }
  }

  const [complete, setComplete] = useState<boolean>(() => {
    const s = storageRef.current;
    return s ? readOnboarding(s) : false;
  });

  useEffect(() => {
    const s = storageRef.current;
    if (s) writeOnboarding(s, complete);
  }, [complete]);

  const finish = useCallback(() => setComplete(true), []);
  const reset = useCallback(() => setComplete(false), []);

  return { complete, finish, reset };
}
```

- [ ] **Step 2: Add i18n strings**

```
"onboarding.step": "{current}/{total}"
"onboarding.skip": "跳过" / "Skip"
"onboarding.next": "下一步" / "Next"
"onboarding.done": "完成" / "Done"
"onboarding.step1.title": "配置智能体"
"onboarding.step1.body": "点击左上角齿轮图标填写 API key。支持 OpenAI 兼容接口。"
"onboarding.step2.title": "输入分析请求"
"onboarding.step2.body": "在左侧聊天框输入请求，例如：分析 pro 数据集。"
"onboarding.step3.title": "查看与导出结果"
"onboarding.step3.body": "结果在右侧工作台；按 Ctrl+K 打开命令面板可一键导出 CSV/JSON/PDF。"
```
Plus English equivalents.

- [ ] **Step 3: Write `OnboardingCoach.tsx`**

```tsx
import { useEffect, useState } from "react";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import "./OnboardingCoach.css";

interface Props {
  language: AppLanguage;
  onDismiss: () => void;
}

const TOTAL = 3;

export function OnboardingCoach({ language, onDismiss }: Props) {
  const [step, setStep] = useState(1);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onDismiss();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onDismiss]);

  const isLast = step === TOTAL;

  const titleKey = `onboarding.step${step}.title`;
  const bodyKey = `onboarding.step${step}.body`;

  return (
    <aside
      className="onboarding-coach"
      role="dialog"
      aria-label={t(language, titleKey)}
    >
      <header className="onboarding-coach-head">
        <span className="onboarding-coach-step">{t(language, "onboarding.step", { current: step, total: TOTAL })}</span>
        <button type="button" className="onboarding-coach-close" aria-label="close" onClick={onDismiss}>×</button>
      </header>
      <h3 className="onboarding-coach-title">{t(language, titleKey)}</h3>
      <p className="onboarding-coach-body">{t(language, bodyKey)}</p>
      <footer className="onboarding-coach-foot">
        <button type="button" className="onboarding-coach-skip" onClick={onDismiss}>
          {t(language, "onboarding.skip")}
        </button>
        <button
          type="button"
          className="onboarding-coach-next"
          onClick={() => (isLast ? onDismiss() : setStep((s) => s + 1))}
          autoFocus
        >
          {t(language, isLast ? "onboarding.done" : "onboarding.next")}
        </button>
      </footer>
    </aside>
  );
}
```

- [ ] **Step 4: CSS `OnboardingCoach.css`**

```css
.onboarding-coach {
  position: fixed;
  right: 16px;
  bottom: 16px;
  z-index: 45;
  width: 320px;
  padding: 14px 16px;
  background: var(--results-panel-bg, #fff);
  color: var(--results-panel-text, #0f172a);
  border: 1px solid var(--results-panel-border, rgba(0,0,0,0.08));
  border-radius: 12px;
  box-shadow: 0 16px 40px rgba(0,0,0,0.28);
  animation: onboarding-pop 180ms ease-out;
}
@keyframes onboarding-pop {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.onboarding-coach-head {
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;
}
.onboarding-coach-step {
  font-size: 11px; font-weight: 700;
  padding: 2px 8px;
  background: var(--results-primary-soft, rgba(31,120,193,0.14));
  color: var(--results-primary, #1f78c1);
  border-radius: 999px;
}
.onboarding-coach-close {
  border: none; background: transparent; color: inherit;
  cursor: pointer; font-size: 18px; line-height: 1;
}
.onboarding-coach-title { margin: 4px 0 6px; font-size: 15px; font-weight: 700; }
.onboarding-coach-body { margin: 0 0 10px; font-size: 13px; line-height: 1.5; color: var(--results-panel-muted, #5f7287); }
.onboarding-coach-foot { display: flex; align-items: center; justify-content: space-between; }
.onboarding-coach-skip {
  border: none; background: transparent; color: var(--results-panel-muted, #64748b);
  cursor: pointer; font-size: 12px;
}
.onboarding-coach-next {
  padding: 6px 14px; border-radius: 999px; border: none;
  background: var(--results-primary, #1f78c1); color: #fff;
  cursor: pointer; font-size: 12px; font-weight: 700;
}
```

- [ ] **Step 5: Mount in App**

```tsx
import { useOnboarding } from "./hooks/useOnboarding";
import { OnboardingCoach } from "./components/OnboardingCoach";

const onboarding = useOnboarding();
```

Render (outside the main shell, alongside other modals):

```tsx
{!onboarding.complete && !settingsOpen ? (
  <OnboardingCoach language={language} onDismiss={onboarding.finish} />
) : null}
```

- [ ] **Step 6: Typecheck + dev smoke**

Clear localStorage → reload → coach appears with step 1; Next advances; Done or × or Esc dismisses; reload no longer shows it.

- [ ] **Step 7: Commit**

```bash
git add src/hooks/useOnboarding.ts src/components/OnboardingCoach.tsx src/components/OnboardingCoach.css src/App.tsx src/i18n.ts
git commit -m "feat(onboarding): 3-step corner coach with localStorage persistence"
```

---

# Phase 4: Shortcuts overlay + focus audit

### Task 4.1: `shortcuts.ts` + ShortcutsOverlay

**Files:**
- Create: `src/lib/commands/shortcuts.ts`
- Create: `src/components/ShortcutsOverlay.tsx`
- Create: `src/components/ShortcutsOverlay.css`
- Modify: `src/i18n.ts`
- Modify: `src/App.tsx`

- [ ] **Step 1: Shortcuts source**

```ts
// src/lib/commands/shortcuts.ts
export interface ShortcutEntry {
  key: string;
  actionKey: string; // i18n key
}

export const SHORTCUTS: ShortcutEntry[] = [
  { key: "Ctrl+K",               actionKey: "shortcuts.action.palette" },
  { key: "Ctrl+L",               actionKey: "shortcuts.action.focusChat" },
  { key: "Ctrl+,",               actionKey: "shortcuts.action.settings" },
  { key: "Enter",                actionKey: "shortcuts.action.send" },
  { key: "Shift+Enter",          actionKey: "shortcuts.action.newline" },
  { key: "?",                    actionKey: "shortcuts.action.overlay" },
  { key: "Esc",                  actionKey: "shortcuts.action.close" },
];
```

- [ ] **Step 2: i18n**

```
"shortcuts.title": "键盘快捷键"
"shortcuts.action.palette": "命令面板"
"shortcuts.action.focusChat": "聚焦聊天输入"
"shortcuts.action.settings": "打开设置"
"shortcuts.action.send": "发送消息"
"shortcuts.action.newline": "换行"
"shortcuts.action.overlay": "显示此清单"
"shortcuts.action.close": "关闭模态 / 面板"
```
Plus English.

- [ ] **Step 3: Component**

```tsx
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import { SHORTCUTS } from "../lib/commands/shortcuts";
import "./ShortcutsOverlay.css";

interface Props {
  open: boolean;
  onClose: () => void;
  language: AppLanguage;
}

export function ShortcutsOverlay({ open, onClose, language }: Props) {
  if (!open) return null;
  return (
    <div className="shortcuts-scrim" onMouseDown={onClose} role="presentation">
      <div
        className="shortcuts-overlay"
        role="dialog"
        aria-modal="true"
        aria-label={t(language, "shortcuts.title")}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="shortcuts-head">
          <h3>{t(language, "shortcuts.title")}</h3>
          <button type="button" className="shortcuts-close" onClick={onClose} aria-label="close">×</button>
        </header>
        <ul className="shortcuts-list">
          {SHORTCUTS.map((s) => (
            <li key={s.key}>
              <kbd>{s.key}</kbd>
              <span>{t(language, s.actionKey)}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: CSS**

```css
.shortcuts-scrim {
  position: fixed; inset: 0;
  background: rgba(10, 15, 25, 0.4);
  display: flex; align-items: center; justify-content: center;
  z-index: 48;
}
.shortcuts-overlay {
  width: min(420px, 92vw);
  background: var(--results-panel-bg, #fff);
  color: var(--results-panel-text, #0f172a);
  border: 1px solid var(--results-panel-border, rgba(0,0,0,0.08));
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: 0 24px 48px rgba(0, 0, 0, 0.3);
}
.shortcuts-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.shortcuts-head h3 { margin: 0; font-size: 15px; font-weight: 700; }
.shortcuts-close { border: none; background: transparent; font-size: 18px; cursor: pointer; color: inherit; }
.shortcuts-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }
.shortcuts-list li { display: flex; align-items: baseline; gap: 12px; font-size: 13px; }
.shortcuts-list kbd {
  font-family: ui-monospace, Menlo, monospace;
  background: var(--results-panel-subtle, rgba(148,163,184,0.1));
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  min-width: 110px;
}
```

- [ ] **Step 5: Wire `?` listener in App.tsx**

Add state:

```tsx
const [shortcutsOpen, setShortcutsOpen] = useState(false);
```

Inside the global keydown handler, **before** the existing branches:

```tsx
if (e.key === "?" && !e.ctrlKey && !e.metaKey && !e.altKey) {
  const target = e.target as HTMLElement | null;
  const editable =
    target?.tagName === "INPUT" ||
    target?.tagName === "TEXTAREA" ||
    target?.isContentEditable;
  if (editable) return;          // allow typing ? into chat
  if (isAnyModalOpen || paletteOpen || shortcutsOpen) return;
  e.preventDefault();
  setShortcutsOpen(true);
  return;
}
```

Render at end of JSX:

```tsx
<ShortcutsOverlay open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} language={language} />
```

Update `isAnyModalOpen` computation to include `shortcutsOpen` for nested-key handling consistency.

- [ ] **Step 6: Typecheck + dev smoke**

`?` while focus on body → overlay opens; Esc / click scrim closes; typing `?` inside chat textarea does not trigger.

- [ ] **Step 7: Commit**

```bash
git add src/lib/commands/shortcuts.ts src/components/ShortcutsOverlay.tsx src/components/ShortcutsOverlay.css src/App.tsx src/i18n.ts
git commit -m "feat(shortcuts): ? overlay backed by single-source shortcuts.ts"
```

---

### Task 4.2: Global focus-visible audit

**Files:**
- Modify: root stylesheet (find where `index.css` or equivalent lives)

- [ ] **Step 1: Locate root CSS entry**

`grep -rn "import.*\.css" src/main.tsx src/App.tsx | head` — identify the file that imports global styles.

- [ ] **Step 2: Add global rule**

Append to the root CSS file:

```css
*:focus-visible {
  outline: 2px solid var(--results-primary, #1f78c1);
  outline-offset: 2px;
  border-radius: 4px;
}

/* Let custom is-active / hover states keep their look; focus-visible wins only when tab-navigated */
```

- [ ] **Step 3: Manual regression**

Tab through: header buttons → gear icon → chat textarea → send button → workbench filter chips → sort select → ExportMenu trigger → palette items (opened via Ctrl+K). Each should show a visible 2px accent ring.

Known cases to check specifically:
- `SettingsModal` input fields (should already work; verify)
- `SummaryScopeToggle` buttons
- `ResultsTable` expand/collapse buttons
- CommandPalette items (`is-active` and focus-visible should not conflict — test focusing via Tab)

- [ ] **Step 4: Fix any conflicts inline**

If any component's existing `outline: none` suppresses the new ring, either delete that override or replace with explicit `:focus-visible` equivalent. Common offenders: buttons with `border` set (replace `outline` with same border style when `focus-visible`).

- [ ] **Step 5: Commit**

```bash
git add src/index.css  # or wherever the root file lives
git commit -m "feat(a11y): global focus-visible ring for keyboard navigation"
```

---

### Task 4.3: Final regression

- [ ] **Step 1: Full test suite**

`node --test tests/*.mjs` → all passing.

- [ ] **Step 2: Typecheck**

`npx tsc --noEmit` → no errors.

- [ ] **Step 3: Production build**

`npm run build` → succeeds; `pdfmake` chunk still isolated; no new warnings.

- [ ] **Step 4: Electron smoke**

`npm run electron:dev` → run through:
- Clear localStorage → launch → onboarding coach appears → Complete
- Ctrl+K → see nav + appearance + examples + log (workbench absent until dataset loads)
- Load a dataset → Ctrl+K → workbench commands appear; export CSV from palette works
- `?` → shortcuts overlay; Esc closes
- Tab through UI → focus ring visible everywhere

- [ ] **Step 5: Tag**

```bash
git commit --allow-empty -m "chore: Round 3 command palette + onboarding complete"
git tag round-3-command-palette
```

---

## Rollback

- Each PR (P1–P4) is independently revertible.
- localStorage keys: `bioagent-onboarding-v1` (new). Future breaking changes go to `-v2`.
- If Ctrl+K conflicts with a user's accessibility tool, quick mitigation: short-circuit `setPaletteOpen` in App when a future setting disables it.

## Dependency Additions

None. All functionality uses existing dependencies.
