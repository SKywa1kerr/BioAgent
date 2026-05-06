# Desktop polish session handoff — 2026-05-07

This document hands off the in-progress "全面系统性升级" started on
2026-05-06. A new conversation can pick up from here without re-deriving
project state.

## Branch + working state

- Worktree: `D:\Learning\Biology\projects\BioAgent_Desktop\.claude\worktrees\ultimate-bioagent`
- Branch: `worktree-ultimate-bioagent`
- Ahead of `origin/main` by **111 commits** (16 landed this session, 95 prior)
- Working tree: clean
- Verification at handoff: `npm run typecheck` clean, `npm run test:js` 162/162 pass, pytest 49/49 pass (including 5 new structured-logging tests)

## What landed this session (16 commits)

In order:

| # | SHA       | Batch | Subject |
|---|-----------|-------|---------|
| 1 | e6d464c | A4-1  | feat(ui): pure toast reducer with TDD coverage |
| 2 | d45b555 | A4-2  | feat(ui): toast provider, container, and component with framer-motion |
| 3 | 22d4749 | A4-3  | feat(ui): wire toast notifications for agent errors with retry/view-log |
| 4 | 436b646 | A5    | feat(ui): micro-animations for canvas, drawer, and chromatogram skeleton |
| 5 | b061e53 | B2    | feat(workbench): persist workbench prefs (sort, density, filter, scope) |
| 6 | 2f004a3 | B5    | feat(workbench): recent analyses rail wired to query_history MCP tool |
| 7 | dc1b26f | B6    | feat(workbench): manual sample status override with reason and exports |
| 8 | 0e71aaa | B4    | feat(workbench): side-by-side sample comparison view |
| 9 | e413487 | D2    | ci: GitHub Actions matrix for typecheck, JS tests, build, and release |
| 10 | 677fb9b | C4    | feat(sidecar): structured JSON-line logging on stderr |
| 11 | 6eca5c2 | B4-2  | feat(workbench): mutation diff highlighting in compare view |
| 12 | 151631c | C5    | test(e2e): playwright smoke suite for app shell + key shortcuts |
| 13 | 23fcf5b | C1    | refactor(types): zod-validated IPC boundary for agent events and analysis result |
| 14 | 84e91ee | B3    | feat(workbench): chromatogram zoom, scroll, and mini-map overview band |
| 15 | f60c79a | D1    | feat(desktop): custom title bar with platform-aware window chrome |
| 16 | 3dfde3d | C3    | refactor(i18n): externalise zh/en bundles to JSON, keep t() signature |

## Roadmap status

The original roadmap was 4 batches × ~25–30 commits.

### Batch A — visual foundation (DONE)

- A1 字体 — `fe5b680` (prior session: Inter + JetBrains Mono via fontsource)
- A2 设计 tokens — `fe5b680` (same commit; 244-line tokens.css)
- A3 lucide 图标 — `6088f6c` (prior session)
- A4 Toast — three commits this session (e6d464c, d45b555, 22d4749)
- A5 微动效 — `436b646`

### Batch B — workflow upgrades (5/6 DONE)

- B1 拖拽导入 — **NOT STARTED** (deferred from this session at user's request — needs real-machine testing of drag-drop flow)
- B2 列持久化 — `b061e53`
- B3 chromatogram 缩放/mini-map — `84e91ee`
- B4 差异对比 + diff math — `0e71aaa` + `6eca5c2`
- B5 最近分析历史 rail — `2f004a3`
- B6 手动判读 override — `dc1b26f`

### Batch C — engineering quality (4/5 DONE; C2 not done)

- C1 zod IPC 边界 + 去 any — `23fcf5b`
- C2 CSS Modules 拆分 — **NOT STARTED**
- C3 i18n 外置 JSON — `3dfde3d`
- C4 Python 结构化日志 — `677fb9b`
- C5 Playwright e2e — `151631c` (CI integration of `test:e2e` deferred)

### Batch D — desktop feel (1/3 DONE; D3 stashed)

- D1 自定义 titleBar — `f60c79a`
- D2 GitHub Actions CI — `e413487`
- D3 electron-updater — **STASHED, ~80% complete** (see "Stash" section below)

## Stash — D3 electron-updater (in-flight)

`stash@{0}` contains a partial D3 implementation that was **stopped mid-flight** because the parallel C3 commit reshaped i18n.ts under the agent's feet. Specifically the agent had:

- Added `electron-updater` to `package.json` dependencies
- Added `publish` block to the `build` config (provider: "github", owner: "Frank", repo: "BioAgent_Desktop")
- Wired `autoUpdater` setup + IPC handlers (`updater-quit-and-install`, `updater-state` broadcast) in `electron/main.js`
- Extended `electron/preload.js` with `electronUpdater` context-bridge surface
- Created `src/hooks/useUpdater.ts` (the renderer-side state hook)
- Started modifying `src/App.tsx` (action handler registration; toast effect was about to be added when stopped)
- Modified `src/locales/zh.json` and `src/locales/en.json` (added 4 `updater.*` keys directly to the JSON files — the agent correctly noticed C3 had already externalised i18n)

What's missing (the part that didn't land):

- The `useEffect` in `App.tsx` that subscribes to `useUpdater().state` and pushes toasts at `phase === "available"`, `phase === "ready"`, `phase === "error"`.
- The action handler registration for `"updater-install"` so the "Restart now" button on the ready-toast actually triggers `quitAndInstall()`.

To resume D3:

```bash
git stash pop   # apply the partial work
npm install     # ensure electron-updater is in node_modules
# then add the missing useEffect + action handler in src/App.tsx
# then verify: npm run typecheck && npm run test:js
# then commit
```

The stash subject ("D3 electron-updater partial: ...") makes it easy to identify with `git stash list`.

**Caveat**: the `publish` block uses owner=Frank, repo=BioAgent_Desktop. Confirm against the actual GitHub remote before tagging a release.

## Remaining work

In priority order for the next session:

### 1. D3 electron-updater (resume from stash) — 1 small commit

See the "Stash" section above. ~30 min of work.

### 2. B1 拖拽导入 — 3–4 commits, needs your real-machine testing

The biggest user-facing remaining feature. Architecture researched in this session (see `superpowers:Explore` agent's report from 2026-05-06). Summary:

- **Python**: extend `analyze_sequences` in `src-python/bioagent/tools_register.py` line 63 to accept optional `ab1_dir` and `gb_dir` parameters that bypass the `resolve_dataset_dirs()` enum-only path.
- **Electron**: add `ipcMain.handle("analyze-dropped-files", async (event, {ab1Paths, gbPaths}) => ...)` in `electron/main.js`. Validate, pair files by basename prefix, and call `agentHarness.callMcpTool("analyze_sequences", { ab1_dir, gb_dir })`.
- **Frontend**: new `src/components/DropZone.tsx` overlay that activates on dragenter, collects files via `event.dataTransfer.files`, pairs `.ab1` + `.gb` by basename via a new pure helper `src/lib/files/pairAb1Gb.{js,d.ts}`, calls the IPC handler, and pushes a toast with success/failure feedback (toast already wired in A4).

Test path: drag a few `.ab1` + `.gb` files onto the running app and watch them get analysed without copying into `data/ab1_files/`.

Suggested split:

- B1-1: Python signature + tool registration (one commit, runs pytest)
- B1-2: Electron IPC + file pairing helper (one commit, JS tests for pair-by-basename)
- B1-3: Frontend DropZone + wire-up (one commit)

### 3. C2 CSS Modules — 1 large commit

The 1136-line ResultsWorkbench.css and 898-line styles.css are unscoped. Convert to either CSS Modules (rename to `*.module.css`, update each importer to `import styles from './X.module.css'`) or vanilla-extract. CSS Modules is the lower-risk choice given the existing setup.

This is best done **alone** because it touches almost every component file. Schedule it for a focused session and dispatch a single subagent.

### 4. B4 / D3 polish (small follow-ups)

- B4 diff highlight currently renders a thin info strip + per-row CSS classes. Could push further: highlight differing positions in the chromatogram canvas itself (paint unique-position bases with a status-uncertain background tint). Optional.
- D3 once installed: wire `test:e2e` into the GitHub Actions CI workflow (.github/workflows/ci.yml) by adding a separate job that runs `npx playwright install --with-deps chromium && npm run test:e2e`. Currently deferred per the C5 commit message.

## Critical project conventions

A new session must respect these to avoid re-deriving them:

### File patterns

- **Testable utility logic** lives as `.js + .d.ts` pairs (e.g., `src/lib/ui/chatRailState.{js,d.ts}`, `src/lib/workbench/compareSelection.{js,d.ts}`). Tests in `tests/test_*.mjs` import the `.js` directly via `node --test`.
- **TypeScript-only modules** (.ts / .tsx) live where the component / hook is. They're built by Vite, not directly testable by `node --test`.
- **Type declarations** for `.js` modules live alongside as `.d.ts`.

### Test commands

- `npm run typecheck` — `tsc --noEmit`. Project has `noUncheckedIndexedAccess` enabled.
- `npm run test:js` — `node --test "tests/test_*.mjs"`. Currently 162 tests across the codebase.
- `npm run test:py` — `pytest tests`. 49 tests, includes `test_alignment_stdout_clean.py` and `test_llm_client_stdout_clean.py` which **must** stay green (they assert sidecar stdout is silent).
- `npm run test:e2e` — playwright (browsers must be installed first via `npm run test:e2e:install`).

### Design tokens

All colours, spacing, radius, shadow, motion live in `src/styles/tokens.css`. Three palettes: light (root), dark (`[data-theme="dark"]`), high-contrast (`[data-theme="hc"]`). `prefers-reduced-motion` overrides `--duration-*` to 0ms automatically. Component CSS must reference tokens, never raw hex.

### i18n

After C3 (`3dfde3d`), the t() function still has the same signature `t(language, key, params?)`, but the dicts now live in:

- `src/locales/zh.json` (279 keys)
- `src/locales/en.json` (279 keys)

`src/i18n.ts` is now an 18-line wrapper that imports the JSON and delegates to `src/lib/i18n/translate.{js,d.ts}`. To add new strings: edit BOTH JSON files. Keep the same key in both. Vite's `resolveJsonModule` is already on.

### Toast notifications

After A4 (`e6d464c` + `d45b555` + `22d4749`):

- Toast state is a pure reducer at `src/lib/ui/toastReducer.{js,d.ts}` (DEFAULT_DURATION_MS=5000, MAX_VISIBLE=5 with FIFO drop). `durationMs: 0` means persistent.
- React provider at `src/components/ui/ToastProvider.tsx` exposes `useToasts()` returning `{ pushToast, dismissToast, clearAllToasts, registerActionHandler }`.
- Action handlers registered in `src/App.tsx` for `"export-debug-log"` and `"retry-last"`. Add new actionIds by calling `toasts.registerActionHandler(actionId, handler)`.
- The harness (`useAgentHarness.ts`) emits `lastErrorEvent` state, App translates it into toasts. New error sites should set `lastErrorEvent` with a fresh object literal so dedup-by-reference works.

### Animation discipline

After A5 (`436b646`):

- Use 80–160ms duration for transform-based motion (138-140ms is the sweet spot, mirroring `--duration-base`).
- Easing tuple `[0.2, 0.7, 0.2, 1]` to mirror `--easing-standard`.
- No springs, no bouncing. `useReducedMotion()` from framer-motion drives opacity-only fallback.

### IPC types (after C1)

Schemas in `src/types/agentResult.ts` (zod). All renderer-side IPC consumption should go through these types. `applyAgentEvent` validates with `AgentEventSchema.safeParse` at the seam. Don't add new `as Record<string, unknown>` casts; declare a schema instead.

### Subagent dispatch playbook

- Be surgical: name exact files in the prompt to prevent scope creep.
- Specify "do NOT commit" so the controller can stage and review.
- For parallel agents: ensure file sets don't overlap. `package.json` and `i18n.ts` are common contention points — only one parallel agent should touch them at a time.
- Foreground for short tasks (<5 min), background (`run_in_background: true`) for longer ones.
- Always verify `npm run typecheck && npm run test:js` between commits.

## How to resume

In the new session, the first message can be something like:

> 继续昨天的桌面 polish 工作。读 `docs/superpowers/plans/2026-05-07-desktop-polish-handoff.md` 了解状态，然后从 D3 electron-updater 开始（`git stash pop` 拿到半成品，补完 toast 订阅 effect 即可）。之后做 B1 拖拽导入，最后 C2 CSS Modules。

The new Claude will read this file, understand the state, and resume.

---

Generated 2026-05-07 by Claude Opus 4.7 (1M context).
