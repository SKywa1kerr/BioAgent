# UX Optimization Round 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 5 highest-impact UX and stability issues — split the monolithic App.tsx, add error boundaries, persist settings with post-init access, fix Electron production builds, and add LLM retry logic.

**Architecture:** Extract agent communication into a custom hook (`useAgentHarness`), move chat UI to `ChatPanel`, add `SettingsModal` as a controlled overlay. Add `ErrorBoundary` class component wrapping canvas content. Backend changes are isolated to `electron/main.js` and `electron/agent_harness.mjs`.

**Tech Stack:** React 18, TypeScript, Electron 33, OpenAI SDK, Vite 5

**Spec:** `docs/superpowers/specs/2026-04-16-ux-optimization-round1-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/hooks/useAgentHarness.ts` | All agent IPC: init, run, event handling, progress, panel state |
| Create | `src/components/ChatPanel.tsx` | Message list, composer, inline progress, expand/collapse |
| Create | `src/components/SettingsModal.tsx` | Settings overlay with persist/restore, re-init on save |
| Create | `src/components/ErrorBoundary.tsx` | Generic error boundary with retry |
| Create | `src/lib/settingsStorage.ts` | Read/write settings to localStorage (Base64 key encoding) |
| Modify | `src/App.tsx` | Slim down to layout shell (~130 lines) |
| Modify | `src/i18n.ts` | Add new i18n keys for settings modal, error boundary, retry |
| Modify | `src/styles.css` | Add settings modal and error boundary styles |
| Modify | `electron/main.js:13-27` | Production build: `loadFile` vs `loadURL` |
| Modify | `electron/agent_harness.mjs:191-223` | Retry wrapper around LLM call |

---

### Task 1: Create `settingsStorage` utility

**Files:**
- Create: `src/lib/settingsStorage.ts`

This is the foundation other tasks depend on. Pure functions, no React.

- [ ] **Step 1: Create the file with types and functions**

```typescript
// src/lib/settingsStorage.ts
const STORAGE_KEY = "bioagent-settings";

export interface AgentSettings {
  llmApiKey: string;
  llmBaseUrl: string;
  llmModel: string;
}

const DEFAULTS: AgentSettings = {
  llmApiKey: "",
  llmBaseUrl: "https://models.sjtu.edu.cn/api/v1",
  llmModel: "deepseek-chat",
};

export function loadSettings(): AgentSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    return {
      llmApiKey: typeof parsed.k === "string" ? atob(parsed.k) : DEFAULTS.llmApiKey,
      llmBaseUrl: typeof parsed.u === "string" ? parsed.u : DEFAULTS.llmBaseUrl,
      llmModel: typeof parsed.m === "string" ? parsed.m : DEFAULTS.llmModel,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(settings: AgentSettings): void {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        k: btoa(settings.llmApiKey),
        u: settings.llmBaseUrl,
        m: settings.llmModel,
      }),
    );
  } catch {
    // storage full or blocked — silently ignore
  }
}

export function clearSettings(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /d/Learning/Biology/projects/BioAgent_Desktop/.claude/worktrees/ultimate-bioagent && npx tsc --noEmit src/lib/settingsStorage.ts`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/lib/settingsStorage.ts
git commit -m "feat: add settings persistence utility (localStorage + Base64 key encoding)"
```

---

### Task 2: Create `ErrorBoundary` component

**Files:**
- Create: `src/components/ErrorBoundary.tsx`
- Modify: `src/i18n.ts` (add 3 keys)
- Modify: `src/styles.css` (add error boundary styles)

- [ ] **Step 1: Add i18n keys**

Add to the `zh` dict in `src/i18n.ts` (after the `"confirm.ok"` line):

```typescript
"error.boundary.title": "渲染出错",
"error.boundary.body": "此区域发生了意外错误，请点击重试。",
"error.boundary.retry": "重试",
```

Add matching keys to the `en` dict:

```typescript
"error.boundary.title": "Rendering Error",
"error.boundary.body": "An unexpected error occurred in this area. Click retry to recover.",
"error.boundary.retry": "Retry",
```

- [ ] **Step 2: Create the ErrorBoundary component**

```tsx
// src/components/ErrorBoundary.tsx
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackBody?: string;
  retryLabel?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-card">
          <h3>{this.props.fallbackTitle || "Rendering Error"}</h3>
          <p>{this.props.fallbackBody || "An unexpected error occurred."}</p>
          <p className="error-boundary-detail">{this.state.error?.message}</p>
          <button className="primary-button" onClick={this.handleRetry}>
            {this.props.retryLabel || "Retry"}
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
```

- [ ] **Step 3: Add styles to `src/styles.css`**

Append at end of file:

```css
.error-boundary-card {
  display: grid;
  gap: 12px;
  padding: 24px;
  border-radius: 18px;
  border: 1px solid rgba(220, 38, 38, 0.24);
  background: rgba(220, 38, 38, 0.06);
  color: var(--text-main);
}

.error-boundary-card h3 {
  margin: 0;
  font-size: 16px;
}

.error-boundary-card p {
  margin: 0;
  line-height: 1.6;
}

.error-boundary-detail {
  font-family: "Consolas", "SFMono-Regular", monospace;
  font-size: 12px;
  color: var(--text-muted);
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  word-break: break-word;
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /d/Learning/Biology/projects/BioAgent_Desktop/.claude/worktrees/ultimate-bioagent && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/components/ErrorBoundary.tsx src/i18n.ts src/styles.css
git commit -m "feat: add ErrorBoundary component with retry support"
```

---

### Task 3: Extract `useAgentHarness` hook

**Files:**
- Create: `src/hooks/useAgentHarness.ts`

This is the biggest refactor. Extract all agent-related state and logic from `App.tsx` into a custom hook.

The hook must own:
- `initialized`, `isRunning`, `messages`, `progress`, `panelType`, `panelPayload`, `confirmMessage`, `statusMessage`
- Functions: `initialize(settings)`, `sendMessage(text)`, `exportDebugLog()`
- All IPC event handling (`applyAgentEvent`, `hydrateAnalysisResult`, `applyTraceFallback`)

- [ ] **Step 1: Create the hook file**

Create `src/hooks/useAgentHarness.ts`. Move the following from `App.tsx`:
- Lines 18-28: constants and `ProgressState` type
- Lines 29-52: `resolvePanelFromEvent`, `withTimeout`, `wait`
- Lines 69-77: `normalizeApiKey`
- Lines 78-81: `maskSecrets`
- Lines 173-191: `getFriendlyToolName`, `getLifecycleLabel`
- Lines 193-209: state declarations (messages, initialized, isRunning, panelType, panelPayload, confirmMessage, statusMessage, progress)
- Lines 210-214: refs (latestEventRef, assistantMessageCountRef, panelTypeRef, languageRef, runTokenRef)
- Lines 251-327: `pushAssistant`, `setProgressState`, `updateAnalysisPayload`, `hydrateAnalysisResult`
- Lines 329-412: `applyAgentEvent`, `applyTraceFallback`, IPC event subscription useEffect
- Lines 414-510: `initializeHarness`, `handleSend`
- Lines 512-524: `handleExportDebugLog`

The hook signature:

```typescript
export function useAgentHarness(language: AppLanguage) {
  // ... all state and logic ...
  return {
    // State
    initialized, isRunning, messages, progress, statusMessage,
    panelType, panelPayload, confirmMessage,
    // Actions
    initialize, sendMessage, exportDebugLog,
    // Panel control
    setPanelType,
  };
}
```

Import `AgentSettings` from `../lib/settingsStorage` for the `initialize` function signature:

```typescript
async function initialize(settings: AgentSettings) { ... }
```

Instead of reading `llmApiKey`/`llmBaseUrl`/`llmModel` from local state, accept them as a parameter.

- [ ] **Step 2: Verify it compiles (may have temporary errors until App.tsx is updated)**

Run: `npx tsc --noEmit`
Expected: Errors only in `App.tsx` (references to moved code) — those will be fixed in Task 5.

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useAgentHarness.ts
git commit -m "feat: extract useAgentHarness hook from App.tsx"
```

---

### Task 4: Create `ChatPanel` and `SettingsModal` components

**Files:**
- Create: `src/components/ChatPanel.tsx`
- Create: `src/components/SettingsModal.tsx`
- Modify: `src/i18n.ts` (add settings modal i18n keys)
- Modify: `src/styles.css` (add settings modal styles)

- [ ] **Step 1: Add i18n keys for settings modal**

Add to `zh` dict in `src/i18n.ts`:

```typescript
"settings.title": "智能体设置",
"settings.save": "保存并重新连接",
"settings.cancel": "取消",
"settings.quickConnect": "快速连接",
"settings.saved": "设置已保存",
```

Add matching `en` keys:

```typescript
"settings.title": "Agent Settings",
"settings.save": "Save & Reconnect",
"settings.cancel": "Cancel",
"settings.quickConnect": "Quick Connect",
"settings.saved": "Settings saved",
```

- [ ] **Step 2: Create `ChatPanel.tsx`**

Extract from `App.tsx` lines 620-694 (the `<aside className="chat-panel">` block). The component receives props:

```tsx
// src/components/ChatPanel.tsx
import { useRef, useEffect, useState, type KeyboardEvent, type ReactNode } from "react";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";

interface ChatMessage {
  role: string;
  content: string;
}

interface ChatPanelProps {
  messages: ChatMessage[];
  isRunning: boolean;
  progress: { phase: string; progress: number; label: string };
  language: AppLanguage;
  initialized: boolean;
  onSend: (text: string) => void;
  onExportDebug: () => void;
  onToggleLanguage: () => void;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  theme: "light" | "dark";
}
```

Move into this component:
- `renderInlineRichText` (App.tsx:93-102)
- `renderStructuredMessage` (App.tsx:104-166)
- `isLongAssistantMessage` (App.tsx:167-171)
- `renderChatProgress` logic (App.tsx:561-572)
- The message list JSX, composer, expand/collapse state
- The title bar with language/theme/export/settings buttons

Add a gear icon button that calls `onOpenSettings`:
```tsx
<button className="theme-toggle" onClick={onOpenSettings} title={t(language, "settings.title")}>
  {"⚙"}
</button>
```

- [ ] **Step 3: Create `SettingsModal.tsx`**

```tsx
// src/components/SettingsModal.tsx
import { useState } from "react";
import type { AgentSettings } from "../lib/settingsStorage";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (settings: AgentSettings) => void;
  currentSettings: AgentSettings;
  language: AppLanguage;
}

export function SettingsModal({ open, onClose, onSave, currentSettings, language }: SettingsModalProps) {
  const [apiKey, setApiKey] = useState(currentSettings.llmApiKey);
  const [baseUrl, setBaseUrl] = useState(currentSettings.llmBaseUrl);
  const [model, setModel] = useState(currentSettings.llmModel);

  if (!open) return null;

  function handleSave() {
    onSave({ llmApiKey: apiKey, llmBaseUrl: baseUrl, llmModel: model });
  }

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{t(language, "settings.title")}</h3>
        <div className="settings-form">
          <label>
            <span>{t(language, "app.field.apiKey")}</span>
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-..." />
          </label>
          <label>
            <span>{t(language, "app.field.baseUrl")}</span>
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
          </label>
          <label>
            <span>{t(language, "app.field.model")}</span>
            <input value={model} onChange={(e) => setModel(e.target.value)} />
          </label>
          <div className="settings-actions">
            <button className="ghost-button" onClick={onClose}>{t(language, "settings.cancel")}</button>
            <button className="primary-button" onClick={handleSave}>{t(language, "settings.save")}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Add settings modal styles to `src/styles.css`**

```css
.settings-modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 900;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
}

.settings-modal {
  width: min(460px, 90vw);
  padding: 24px;
  border-radius: 20px;
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  box-shadow: var(--panel-shadow);
}

.settings-modal h3 {
  margin: 0 0 16px 0;
  font-size: 18px;
}
```

- [ ] **Step 5: Verify it compiles**

Run: `npx tsc --noEmit`
Expected: Errors only in `App.tsx` (will be resolved in next task)

- [ ] **Step 6: Commit**

```bash
git add src/components/ChatPanel.tsx src/components/SettingsModal.tsx src/i18n.ts src/styles.css
git commit -m "feat: add ChatPanel and SettingsModal components"
```

---

### Task 5: Rewrite `App.tsx` as thin shell

**Files:**
- Modify: `src/App.tsx` (rewrite to ~130 lines)

This ties everything together. `App.tsx` becomes a layout shell that wires the hook to the components.

- [ ] **Step 1: Rewrite App.tsx**

Replace the entire file with:

```tsx
import { useState } from "react";
import { SmartCanvas } from "./components/SmartCanvas";
import { ChatPanel } from "./components/ChatPanel";
import { SettingsModal } from "./components/SettingsModal";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { AnalysisPanel } from "./components/panels/AnalysisPanel";
import { MutationTrendPanel } from "./components/panels/MutationTrendPanel";
import { LabSuggestionPanel } from "./components/panels/LabSuggestionPanel";
import { ConfirmationDialog } from "./components/panels/ConfirmationDialog";
import { useAgentHarness } from "./hooks/useAgentHarness";
import { loadSettings, saveSettings, type AgentSettings } from "./lib/settingsStorage";
import { t, type AppLanguage } from "./i18n";

function getLocalStorageValue<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const saved = window.localStorage.getItem(key);
    if (saved && (allowed as readonly string[]).includes(saved)) return saved as T;
  } catch { /* ignore */ }
  return fallback;
}

export function App() {
  const [language, setLanguage] = useState<AppLanguage>(() => getLocalStorageValue("bioagent-language", ["zh", "en"] as const, "zh"));
  const [theme, setTheme] = useState<"light" | "dark">(() => getLocalStorageValue("bioagent-theme", ["light", "dark"] as const, "dark"));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<AgentSettings>(loadSettings);

  const agent = useAgentHarness(language);

  // Persist language & theme
  // (keep the existing useEffect patterns from the original for theme/language localStorage sync)

  function handleToggleLanguage() {
    setLanguage((l) => {
      const next = l === "zh" ? "en" : "zh";
      try { window.localStorage.setItem("bioagent-language", next); } catch { /* ignore */ }
      return next;
    });
  }

  function handleToggleTheme() {
    setTheme((v) => {
      const next = v === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { window.localStorage.setItem("bioagent-theme", next); } catch { /* ignore */ }
      return next;
    });
  }

  function handleSettingsSave(newSettings: AgentSettings) {
    setSettings(newSettings);
    saveSettings(newSettings);
    setSettingsOpen(false);
    void agent.initialize(newSettings);
  }

  function handleSend(text: string) {
    if (!agent.initialized) {
      // Auto-init if settings are available
      if (settings.llmApiKey) {
        void agent.initialize(settings).then(() => agent.sendMessage(text, settings));
      }
      return;
    }
    void agent.sendMessage(text, settings);
  }

  // Initial settings panel (when not initialized)
  // Render the initial settings form inside the canvas panel if not initialized
  function renderPanel() {
    if (!agent.initialized) {
      return (
        <div className="result-panel">
          <div className="detail-card">
            <h3>{t(language, "app.panel.settings")}</h3>
            <div className="settings-form">
              <label>
                <span>{t(language, "app.field.apiKey")}</span>
                <input type="password" value={settings.llmApiKey} onChange={(e) => setSettings((s) => ({ ...s, llmApiKey: e.target.value }))} placeholder="sk-..." />
              </label>
              <label>
                <span>{t(language, "app.field.baseUrl")}</span>
                <input value={settings.llmBaseUrl} onChange={(e) => setSettings((s) => ({ ...s, llmBaseUrl: e.target.value }))} />
              </label>
              <label>
                <span>{t(language, "app.field.model")}</span>
                <input value={settings.llmModel} onChange={(e) => setSettings((s) => ({ ...s, llmModel: e.target.value }))} />
              </label>
              <div className="settings-actions">
                <button className="primary-button" onClick={() => { saveSettings(settings); void agent.initialize(settings); }}>
                  {t(language, "app.action.init")}
                </button>
              </div>
              <div className="status-line">{agent.statusMessage}</div>
            </div>
          </div>
          <div className="detail-card progress-card clean-progress-card">
            <h3>{t(language, "app.progress.cardTitle")}</h3>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${agent.progress.progress}%` }} />
            </div>
            <div className="progress-meta clean-progress-meta">
              <span>{agent.progress.label}</span>
            </div>
          </div>
        </div>
      );
    }

    if (agent.panelType === "analysis") return <AnalysisPanel result={agent.panelPayload} language={language} />;
    if (agent.panelType === "trends") return <MutationTrendPanel result={agent.panelPayload} language={language} />;
    if (agent.panelType === "suggestions") return <LabSuggestionPanel result={agent.panelPayload} language={language} />;
    if (agent.panelType === "confirmation") {
      return <ConfirmationDialog message={agent.confirmMessage} onConfirm={() => agent.setPanelType("text")} onCancel={() => agent.setPanelType("text")} language={language} />;
    }

    return (
      <div className="detail-card audience-card">
        <h3>{t(language, "app.ready.title")}</h3>
        <p>{t(language, "app.ready.body")}</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <ChatPanel
        messages={agent.messages}
        isRunning={agent.isRunning}
        progress={agent.progress}
        language={language}
        initialized={agent.initialized}
        onSend={handleSend}
        onExportDebug={() => void agent.exportDebugLog()}
        onToggleLanguage={handleToggleLanguage}
        onToggleTheme={handleToggleTheme}
        onOpenSettings={() => setSettingsOpen(true)}
        theme={theme}
      />

      <main className="canvas-panel">
        <SmartCanvas title={t(language, "app.canvasTitle")} panelType={agent.panelType}>
          <ErrorBoundary
            fallbackTitle={t(language, "error.boundary.title")}
            fallbackBody={t(language, "error.boundary.body")}
            retryLabel={t(language, "error.boundary.retry")}
          >
            {renderPanel()}
          </ErrorBoundary>
        </SmartCanvas>
      </main>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSave={handleSettingsSave}
        currentSettings={settings}
        language={language}
      />
    </div>
  );
}
```

Note: The `renderCompactProgress` that was shown inside `SmartCanvas` should be moved into the panel rendering area or into `ChatPanel` as needed. Preserve the existing compact progress rendering inside the canvas area.

- [ ] **Step 2: Verify the full project compiles**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Verify the app runs in dev mode**

Run: `npm run dev` (in a separate terminal)
Check: App loads, settings form appears, can type in fields.

- [ ] **Step 4: Commit**

```bash
git add src/App.tsx
git commit -m "refactor: slim App.tsx to layout shell, wire useAgentHarness + ChatPanel + SettingsModal + ErrorBoundary"
```

---

### Task 6: Fix Electron production build

**Files:**
- Modify: `electron/main.js:13-27`

- [ ] **Step 1: Update `createWindow` in `electron/main.js`**

Replace lines 13-27:

```javascript
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 760,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  } else {
    mainWindow.loadURL("http://127.0.0.1:1420");
  }
}
```

- [ ] **Step 2: Verify electron starts in dev mode**

Run: `npm run electron:dev`
Expected: Window opens and loads the Vite dev server.

- [ ] **Step 3: Commit**

```bash
git add electron/main.js
git commit -m "fix: load dist/index.html in packaged Electron builds"
```

---

### Task 7: Add LLM retry mechanism

**Files:**
- Modify: `electron/agent_harness.mjs`

- [ ] **Step 1: Add retry utility function after the constants (line 10)**

Insert after line 10 (`const MAX_PROMPT_MESSAGES = 8;`):

```javascript
const RETRY_MAX = 2;
const RETRY_DELAYS_MS = [1000, 3000];

function isRetryableError(error) {
  const msg = String(error?.message || error || "").toLowerCase();
  if (/429|rate.?limit|too many requests/i.test(msg)) return true;
  if (/50[023]|bad gateway|service unavailable|internal server/i.test(msg)) return true;
  if (/etimedout|econnreset|econnrefused|socket hang up|network/i.test(msg)) return true;
  return false;
}

async function withRetry(fn, onRetry) {
  let lastError;
  for (let attempt = 0; attempt <= RETRY_MAX; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < RETRY_MAX && isRetryableError(error)) {
        const delay = RETRY_DELAYS_MS[attempt] || 3000;
        if (onRetry) onRetry(attempt + 1, delay, error);
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }
      throw error;
    }
  }
  throw lastError;
}
```

- [ ] **Step 2: Wrap the LLM call in `runTurn` with `withRetry`**

In `runTurn`, replace the direct `client.chat.completions.create(...)` call (around line 209-223) with:

```javascript
const response = await withRetry(
  () => client.chat.completions.create({
    model: this.settings.llmModel || DEFAULT_MODEL,
    temperature: 0,
    max_tokens: 1200,
    timeout: this.settings.llmTimeoutMs || LLM_TIMEOUT_MS,
    messages: [{ role: "system", content: this.buildSystemPrompt() }, ...this.messages.slice(-MAX_PROMPT_MESSAGES)],
    tools: this.mcpTools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    })),
  }),
  (attempt, delay, error) => {
    onEvent({ type: "thinking", retrying: true, attempt, message: `Retrying (${attempt}/${RETRY_MAX})...` });
  },
);
```

- [ ] **Step 3: Verify electron dev mode still works**

Run: `npm run electron:dev`
Test: Send a message, verify normal flow still works.

- [ ] **Step 4: Commit**

```bash
git add electron/agent_harness.mjs
git commit -m "feat: add exponential backoff retry for transient LLM API errors"
```

---

## Execution Checklist

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | settingsStorage utility | — |
| 2 | ErrorBoundary component | — |
| 3 | useAgentHarness hook | Task 1 |
| 4 | ChatPanel + SettingsModal | Task 1 |
| 5 | Rewrite App.tsx | Tasks 1-4 |
| 6 | Electron production build fix | — |
| 7 | LLM retry mechanism | — |

Tasks 1, 2, 6, 7 are independent and can be parallelized.
Tasks 3, 4 depend on Task 1.
Task 5 depends on all frontend tasks (1-4).
