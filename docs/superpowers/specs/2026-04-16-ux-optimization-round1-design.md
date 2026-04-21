# UX Optimization Round 1 — Core Experience Fixes

**Date:** 2026-04-16
**Scope:** 5 high-priority items that address the most impactful usability and stability gaps.

---

## 1. Settings Persistence + Post-Init Settings Access

### Problem
- `llmApiKey`, `llmBaseUrl`, `llmModel` are stored only in React state — lost on refresh.
- Once initialized, the settings panel disappears with no way to reconfigure.

### Design
- **Persistence:** Save settings to `localStorage` on change. API key encoded with `btoa()` (not true encryption, but prevents casual shoulder-surfing in devtools).
- **Settings Modal:** Add a gear icon button to `ChatPanel` header. Clicking opens a modal overlay with the same 3 fields + Save/Cancel buttons. On save, call `shutdown → re-init` with new settings.
- **Hydration:** On app mount, read saved values from localStorage as initial state. If all 3 fields are populated, show a "Quick Connect" button instead of requiring manual input.

### Files Changed
- `src/App.tsx` → extract settings logic
- New: `src/components/SettingsModal.tsx`

---

## 2. Split App.tsx Into Focused Modules

### Problem
`App.tsx` is ~710 lines containing all state, event handling, and rendering. Every state change re-renders the entire tree.

### Design

**Extract `useAgentHarness` hook:**
- Owns: `initialized`, `isRunning`, `progress`, `messages`, `panelType`, `panelPayload`, `confirmMessage`
- Exposes: `initialize(settings)`, `sendMessage(text)`, `exportDebugLog()`, event subscription
- Handles: IPC communication, event processing (`applyAgentEvent`), detail hydration, trace fallback

**Extract `ChatPanel` component:**
- Props: `messages`, `isRunning`, `progress`, `language`, `onSend`, `onExportDebug`
- Contains: message list, composer, inline progress bar, expand/collapse logic

**Extract `SettingsModal` component:**
- Props: `open`, `onClose`, `initialSettings`, `onSave`

**Resulting App.tsx (~120-150 lines):**
- Layout shell (`app-shell` grid)
- Language/theme toggles
- Wires `useAgentHarness` to `ChatPanel` and `SmartCanvas`

### Files Changed
- `src/App.tsx` (rewrite to shell)
- New: `src/hooks/useAgentHarness.ts`
- New: `src/components/ChatPanel.tsx`
- New: `src/components/SettingsModal.tsx`

---

## 3. Error Boundary

### Problem
Any render error in `AnalysisPanel`, `ChromatogramCanvas`, or chart components crashes the entire app with a white screen.

### Design
- Create a generic `ErrorBoundary` class component (React still requires class-based error boundaries).
- Wrap `SmartCanvas` children and each panel component.
- Fallback UI: error message card with "Retry" button that resets the error state.
- Log caught errors to console and optionally to the debug log via IPC.

### Files Changed
- New: `src/components/ErrorBoundary.tsx`
- `src/components/SmartCanvas.tsx` (wrap children)
- `src/App.tsx` (wrap canvas content)

---

## 4. LLM Retry Mechanism

### Problem
`agent_harness.mjs` makes a single LLM API call with no retry. Transient network errors or rate limits cause immediate failure.

### Design
- Add a `retryLlmCall(fn, maxRetries=2)` utility in `agent_harness.mjs`.
- Retry on: HTTP 429, 500, 502, 503, ETIMEDOUT, ECONNRESET, ECONNREFUSED.
- Backoff: 1s after first failure, 3s after second.
- During retry, emit `onEvent({ type: "thinking", retrying: true })` so the UI can show "Retrying...".
- Non-retryable errors (400, 401, 403) fail immediately.

### Files Changed
- `electron/agent_harness.mjs`

---

## 5. Electron Production Build Fix

### Problem
`main.js:26` hardcodes `http://127.0.0.1:1420` — the Vite dev server URL. Packaged builds cannot connect.

### Design
- Check `app.isPackaged` in `createWindow()`.
- If packaged: `mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))`.
- If dev: keep current `loadURL("http://127.0.0.1:1420")`.
- Also set `ELECTRON_IS_DEV` env var for renderer-side detection if needed later.

### Files Changed
- `electron/main.js`

---

## Implementation Order

1. **Split App.tsx** (foundation — all other changes land cleaner on the new structure)
2. **Error Boundary** (safety net before more changes)
3. **Settings Persistence + Modal** (depends on split)
4. **Electron Production Build Fix** (independent, quick)
5. **LLM Retry** (independent, quick)

Steps 4 and 5 can be done in parallel.
