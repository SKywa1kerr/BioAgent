# Command Workbench And Assistant Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the analysis experience into a command-first scientific workbench with a dedicated Assistant page, while preserving the existing Python analysis engine and making the pre-redesign codebase easy to restore.

**Architecture:** Keep the existing React + Electron + Python split, but move the current `AgentPanel` out of the analysis page and add a new top-of-page command surface that renders a visible action plan and executes approved actions through Electron. Use a lightweight frontend action registry for execution metadata and a Python-side command interpreter for deterministic Chinese intent parsing and testability.

**Tech Stack:** React 18, TypeScript, Vite, Electron, Python 3, pytest

---

## File Structure

- `src/App.tsx`
  Own top-level page layout, tab routing, analysis-page command orchestration, command-plan state, result filter state, and confirmation flow.
- `src/App.css`
  Own the new scientific workbench layout, command surface styling, and analysis-page visual hierarchy.
- `src/components/AssistantPage.tsx`
  Full-page assistant shell that hosts the existing `AgentPanel`.
- `src/components/AssistantPage.css`
  Dedicated full-page assistant styling.
- `src/components/CommandWorkbench.tsx`
  Chinese command input, quick actions, current batch summary, and submit controls.
- `src/components/CommandWorkbench.css`
  Styling for the command surface.
- `src/components/ActionPlanCard.tsx`
  Visible parsed-plan preview and confirmation card.
- `src/components/ActionPlanCard.css`
  Styling for the action plan card.
- `src/components/ExecutionTimeline.tsx`
  Compact timeline showing planned, running, confirmed, completed, and failed steps.
- `src/components/ExecutionTimeline.css`
  Styling for the execution timeline.
- `src/components/ResultsWorkbench.tsx`
  Continue to render the result shell, but accept already-filtered samples from `App.tsx`.
- `src/components/TabLayout.tsx`
  Continue to render tab navigation, with the new `Assistant` tab.
- `src/i18n.ts`
  Add new copy for the Assistant tab, command workbench, action summaries, confirmation text, and quick prompts.
- `src/types/index.ts`
  Add command-plan, execution-event, confirmation, and command-action types.
- `src/utils/actionRegistry.ts`
  Frontend source of truth for operational action metadata and execution-risk flags.
- `src/utils/commandExecution.ts`
  Sequentially execute parsed actions against the current app state and Electron IPC surface.
- `electron/main.js`
  Add `interpret-command` and `open-export-folder` IPC handlers.
- `electron/preload.js`
  Expose the new IPC methods through `window.electronAPI`.
- `src/electron.d.ts`
  Type the new preload methods.
- `src-python/bioagent/command_intent.py`
  Deterministic Chinese command interpreter that maps common phrases to canonical actions.
- `src-python/bioagent/main.py`
  Add a `--interpret-command` CLI entry and return JSON action plans.
- `tests/test_command_intent.py`
  Behavioral pytest coverage for the Python command interpreter.
- `tests/test_command_ui_contract.py`
  Contract tests for the new analysis-page and assistant-page composition.
- `tests/test_command_registry_contract.py`
  Contract tests for frontend action IDs, confirmation flags, and new command components.

## Task 1: Freeze The Current Baseline Before Refactoring

**Files:**
- Create local-only archive artifacts outside the repo root:
  - `..\BioAgent_Desktop-pre-command-workbench-2026-04-08.bundle`
  - `..\BioAgent_Desktop-pre-command-workbench-2026-04-08.zip`
  - `..\BioAgent_Desktop-pre-command-workbench-2026-04-08-status.txt`
- Create a recovery tag and branch:
  - `pre-command-workbench-2026-04-08`
  - `archive/pre-command-workbench-2026-04-08`

- [ ] **Step 1: Capture the current git state into an external status file**

```powershell
git status --short | Out-File -Encoding utf8 ..\BioAgent_Desktop-pre-command-workbench-2026-04-08-status.txt
Get-Content ..\BioAgent_Desktop-pre-command-workbench-2026-04-08-status.txt
```

- [ ] **Step 2: Create a full repository bundle for git-level recovery**

```powershell
git bundle create ..\BioAgent_Desktop-pre-command-workbench-2026-04-08.bundle --all
Get-Item ..\BioAgent_Desktop-pre-command-workbench-2026-04-08.bundle | Select-Object FullName,Length
```

- [ ] **Step 3: Create a filesystem snapshot zip for non-git recovery**

```powershell
Compress-Archive -Path .\src,.\src-python,.\electron,.\docs,.\tests,.\package.json,.\package-lock.json,.\vite.config.ts,.\tsconfig.json,.\tsconfig.node.json,.\index.html,.\README.md -DestinationPath ..\BioAgent_Desktop-pre-command-workbench-2026-04-08.zip -Force
Get-Item ..\BioAgent_Desktop-pre-command-workbench-2026-04-08.zip | Select-Object FullName,Length
```

- [ ] **Step 4: Create a named recovery tag and archive branch**

```powershell
git tag -a pre-command-workbench-2026-04-08 -m "Pre-command-workbench redesign recovery point"
git branch archive/pre-command-workbench-2026-04-08
git tag --list pre-command-workbench-2026-04-08
git branch --list archive/pre-command-workbench-2026-04-08
```

- [ ] **Step 5: Create a dedicated implementation worktree and switch future code changes there**

```powershell
git worktree add ..\BioAgent_Desktop-command-workbench feature/desktop-v2
Get-ChildItem ..\BioAgent_Desktop-command-workbench | Select-Object Name
```

Expected: the sibling worktree exists and can be used for all refactor work, while the original workspace remains as a readable snapshot.

## Task 2: Extract The Existing Agent Panel Into A Dedicated Assistant Page

**Files:**
- Create: `src/components/AssistantPage.tsx`
- Create: `src/components/AssistantPage.css`
- Create: `tests/test_command_ui_contract.py`
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Modify: `src/components/TabLayout.tsx`
- Modify: `src/components/TabLayout.css`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Write the failing UI contract test for the new Assistant tab**

```python
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_analysis_tabs_include_assistant_and_analysis_page_drops_agent_panel():
    app = _read("src/App.tsx")
    assert '{ id: "assistant", label: t(language, "tabs.assistant") }' in app
    assert 'activeTab === "assistant"' in app
    assert '<AssistantPage' in app
    assert '<AgentPanel' not in app.split('activeTab === "analysis"')[1]


def test_assistant_page_wraps_agent_panel():
    content = _read("src/components/AssistantPage.tsx")
    assert "export function AssistantPage(" in content
    assert "<AgentPanel" in content
    assert 'className="assistant-page"' in content
```

- [ ] **Step 2: Run the contract test to verify it fails before implementation**

```powershell
python -m pytest tests\test_command_ui_contract.py -q
```

Expected: FAIL because `AssistantPage.tsx` and the new tab do not exist yet.

- [ ] **Step 3: Implement the Assistant page wrapper and move AgentPanel off the analysis page**

```tsx
// src/components/AssistantPage.tsx
import type { AppLanguage, AnalysisContextUpdate, Sample } from "../types";
import { t } from "../i18n";
import { AgentPanel } from "./AgentPanel";
import "./AssistantPage.css";

interface AssistantPageProps {
  language: AppLanguage;
  samples: Sample[];
  selectedSampleId: string | null;
  sourcePath?: string | null;
  genesDir?: string | null;
  plasmid?: string;
  onAnalysisComplete?: (nextAnalysis: AnalysisContextUpdate) => void;
}

export function AssistantPage(props: AssistantPageProps) {
  return (
    <section className="assistant-page" aria-label={t(props.language, "assistant.shellLabel")}>
      <div className="assistant-page-header">
        <div>
          <span className="assistant-page-kicker">{t(props.language, "assistant.kicker")}</span>
          <h2>{t(props.language, "assistant.title")}</h2>
        </div>
        <p>{t(props.language, "assistant.body")}</p>
      </div>
      <AgentPanel {...props} />
    </section>
  );
}
```

```tsx
// src/App.tsx
import { AssistantPage } from "./components/AssistantPage";

function buildTabs(language: AppLanguage) {
  return [
    { id: "analysis", label: t(language, "tabs.analysis") },
    { id: "assistant", label: t(language, "tabs.assistant") },
    { id: "history", label: t(language, "tabs.history") },
    { id: "settings", label: t(language, "tabs.settings") },
  ];
}

{activeTab === "analysis" && (
  <>
    <div className="analysis-command-stage" />
    <div className="analysis-main-surface" />
  </>
)}
{activeTab === "assistant" && (
  <AssistantPage
    language={language}
    samples={samples}
    selectedSampleId={selectedId}
    sourcePath={ab1Dir}
    genesDir={genesDir}
    plasmid={plasmid}
    onAnalysisComplete={handleAnalysisComplete}
  />
)}
```

```ts
// src/i18n.ts
tabs: {
  analysis: "Analysis",
  assistant: "Assistant",
  history: "History",
  settings: "Settings",
},
assistant: {
  shellLabel: "Assistant workspace",
  kicker: "BioAgent Assistant",
  title: "解释与追问",
  body: "这里保留深度解释、追问和诊断，不再占用分析主页面。",
},
```

- [ ] **Step 4: Run the contract test and the TypeScript build**

```powershell
python -m pytest tests\test_command_ui_contract.py -q
npm.cmd run build
```

Expected: the new contract test passes and the Vite build completes successfully.

- [ ] **Step 5: Commit the Assistant-page extraction**

```powershell
git add src\App.tsx src\App.css src\components\AssistantPage.tsx src\components\AssistantPage.css src\components\TabLayout.tsx src\components\TabLayout.css src\i18n.ts tests\test_command_ui_contract.py
git commit -m "feat: move agent panel into assistant page"
```

## Task 3: Add A Python Command Interpreter And Expose It Through Electron

**Files:**
- Create: `src-python/bioagent/command_intent.py`
- Create: `tests/test_command_intent.py`
- Modify: `src-python/bioagent/main.py`
- Modify: `electron/main.js`
- Modify: `electron/preload.js`
- Modify: `src/electron.d.ts`

- [ ] **Step 1: Write failing pytest coverage for common Chinese command patterns**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.command_intent import interpret_command


def test_interpret_command_parses_run_analysis_with_plasmid_and_wrong_filter():
    plan = interpret_command("分析这个数据集，用 pet15b，只看 wrong 样本")
    assert [action["id"] for action in plan["actions"]] == [
        "set_plasmid",
        "run_analysis",
        "filter_results",
    ]
    assert plan["actions"][0]["args"]["plasmid"] == "pet15b"
    assert plan["actions"][2]["args"]["status"] == "wrong"
    assert plan["needsConfirmation"] is True


def test_interpret_command_parses_export_and_open_folder():
    plan = interpret_command("导出当前报告并打开导出目录")
    assert [action["id"] for action in plan["actions"]] == [
        "export_report",
        "open_export_folder",
    ]
    assert plan["needsConfirmation"] is True


def test_interpret_command_returns_import_dataset_for_dataset_request():
    plan = interpret_command("导入新的数据集并开始分析")
    assert plan["actions"][0]["id"] == "import_dataset"
    assert plan["actions"][1]["id"] == "run_analysis"
```

- [ ] **Step 2: Run the interpreter tests to verify they fail**

```powershell
python -m pytest tests\test_command_intent.py -q
```

Expected: FAIL because `bioagent.command_intent` and `interpret_command()` do not exist yet.

- [ ] **Step 3: Implement the interpreter module, CLI entry, and Electron IPC**

```python
# src-python/bioagent/command_intent.py
from __future__ import annotations

from typing import Any


CONFIRMATION_ACTIONS = {"run_analysis", "export_report", "open_export_folder", "import_dataset"}


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def interpret_command(text: str) -> dict[str, Any]:
    normalized = _normalize(text)
    actions: list[dict[str, Any]] = []

    if "pet15b" in normalized:
        actions.append({"id": "set_plasmid", "args": {"plasmid": "pet15b"}})
    elif "pet22b" in normalized:
        actions.append({"id": "set_plasmid", "args": {"plasmid": "pet22b"}})

    if "导入" in text and "数据集" in text:
        actions.append({"id": "import_dataset", "args": {}})

    if "分析" in text or "重跑" in text:
        actions.append({"id": "run_analysis", "args": {}})

    if "wrong" in normalized or "异常" in text:
        actions.append({"id": "filter_results", "args": {"status": "wrong"}})
    elif "ok" in normalized or "通过" in text:
        actions.append({"id": "filter_results", "args": {"status": "ok"}})

    if "导出" in text and "报告" in text:
        actions.append({"id": "export_report", "args": {}})

    if "打开" in text and "导出" in text and "目录" in text:
        actions.append({"id": "open_export_folder", "args": {}})

    summary = " -> ".join(action["id"] for action in actions) or "reply_only"
    return {
        "summary": summary,
        "actions": actions,
        "needsConfirmation": any(action["id"] in CONFIRMATION_ACTIONS for action in actions),
    }
```

```python
# src-python/bioagent/main.py
from .command_intent import interpret_command

parser.add_argument("--interpret-command", help="Interpret a natural-language command and return an action plan")

if args.interpret_command:
    print(json.dumps(interpret_command(args.interpret_command), ensure_ascii=False))
    return
```

```js
// electron/main.js
const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");

ipcMain.handle("interpret-command", async (_event, text) => {
  return new Promise((resolve, reject) => {
    const { cmd, baseArgs, cwd } = getPythonCommand();
    const args = [...baseArgs, "--interpret-command", String(text)];
    const env = getAnalysisEnv();

    execFile(cmd, args, { cwd, env }, (err, stdout, stderr) => {
      if (err) {
        reject(stderr || err.message);
        return;
      }
      resolve(stdout);
    });
  });
});

ipcMain.handle("open-export-folder", async (_event, exportedPath) => {
  if (!exportedPath) return false;
  shell.showItemInFolder(exportedPath);
  return true;
});
```

```js
// electron/preload.js
contextBridge.exposeInMainWorld("electronAPI", {
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  interpretCommand: (text) => ipcRenderer.invoke("interpret-command", text),
  openExportFolder: (exportedPath) => ipcRenderer.invoke("open-export-folder", exportedPath),
  onAnalysisProgress: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("analysis-progress", listener);
    return () => ipcRenderer.removeListener("analysis-progress", listener);
  },
});
```

```ts
// src/electron.d.ts
interface ElectronAPI {
  invoke: (channel: string, ...args: unknown[]) => Promise<unknown>;
  interpretCommand: (text: string) => Promise<unknown>;
  openExportFolder: (exportedPath: string) => Promise<boolean>;
  onAnalysisProgress: (
    callback: (payload: {
      stage: string;
      percent: number;
      processedSamples: number;
      totalSamples: number;
      sampleId?: string | null;
      message?: string;
    }) => void
  ) => () => void;
}
```

- [ ] **Step 4: Run the interpreter tests and the existing agent-side tests**

```powershell
python -m pytest tests\test_command_intent.py tests\test_agent_tools.py tests\test_agent_chat.py -q
```

Expected: PASS for the new command-intent tests and no regression in the current agent-tool tests.

- [ ] **Step 5: Commit the command interpreter and Electron bridge**

```powershell
git add src-python\bioagent\command_intent.py src-python\bioagent\main.py electron\main.js electron\preload.js src\electron.d.ts tests\test_command_intent.py
git commit -m "feat: add command interpretation bridge"
```

## Task 4: Introduce Command Types, Action Registry, And Workbench UI Components

**Files:**
- Create: `src/components/CommandWorkbench.tsx`
- Create: `src/components/CommandWorkbench.css`
- Create: `src/components/ActionPlanCard.tsx`
- Create: `src/components/ActionPlanCard.css`
- Create: `src/components/ExecutionTimeline.tsx`
- Create: `src/components/ExecutionTimeline.css`
- Create: `src/utils/actionRegistry.ts`
- Create: `tests/test_command_registry_contract.py`
- Modify: `src/types/index.ts`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Write failing contract tests for the new command components and action registry**

```python
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_action_registry_exposes_expected_operational_actions():
    content = _read("src/utils/actionRegistry.ts")
    assert 'id: "import_dataset"' in content
    assert 'id: "set_plasmid"' in content
    assert 'id: "run_analysis"' in content
    assert 'id: "filter_results"' in content
    assert 'id: "export_report"' in content
    assert 'needsConfirmation: true' in content


def test_command_workbench_contract_exists():
    content = _read("src/components/CommandWorkbench.tsx")
    assert "export function CommandWorkbench(" in content
    assert 'className="command-workbench"' in content
    assert "quickPrompts" in content


def test_action_plan_card_and_timeline_exist():
    plan = _read("src/components/ActionPlanCard.tsx")
    timeline = _read("src/components/ExecutionTimeline.tsx")
    assert "export function ActionPlanCard(" in plan
    assert "export function ExecutionTimeline(" in timeline
```

- [ ] **Step 2: Run the contract tests to verify they fail**

```powershell
python -m pytest tests\test_command_registry_contract.py -q
```

Expected: FAIL because none of the new files exist yet.

- [ ] **Step 3: Implement the new command types, registry, and presentational components**

```ts
// src/types/index.ts
export type CommandActionId =
  | "import_dataset"
  | "set_ab1_dir"
  | "set_genes_dir"
  | "set_plasmid"
  | "run_analysis"
  | "filter_results"
  | "open_sample"
  | "export_report"
  | "open_export_folder";

export interface PlannedAction {
  id: CommandActionId;
  args: Record<string, unknown>;
}

export interface CommandPlan {
  summary: string;
  actions: PlannedAction[];
  needsConfirmation: boolean;
}

export interface ExecutionEvent {
  id: string;
  actionId: CommandActionId;
  status: "planned" | "running" | "done" | "failed" | "waiting_confirmation";
  message: string;
}
```

```ts
// src/utils/actionRegistry.ts
import type { CommandActionId } from "../types";

export interface ActionDefinition {
  id: CommandActionId;
  label: string;
  needsConfirmation: boolean;
}

export const ACTION_REGISTRY: ActionDefinition[] = [
  { id: "import_dataset", label: "导入数据集", needsConfirmation: true },
  { id: "set_ab1_dir", label: "选择 AB1 目录", needsConfirmation: false },
  { id: "set_genes_dir", label: "选择参考目录", needsConfirmation: false },
  { id: "set_plasmid", label: "切换质粒", needsConfirmation: false },
  { id: "run_analysis", label: "开始分析", needsConfirmation: true },
  { id: "filter_results", label: "筛选结果", needsConfirmation: false },
  { id: "open_sample", label: "打开样本", needsConfirmation: false },
  { id: "export_report", label: "导出报告", needsConfirmation: true },
  { id: "open_export_folder", label: "打开导出目录", needsConfirmation: true },
];
```

```tsx
// src/components/CommandWorkbench.tsx
import type { AppLanguage } from "../types";
import { t } from "../i18n";
import "./CommandWorkbench.css";

interface CommandWorkbenchProps {
  language: AppLanguage;
  draft: string;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onQuickPrompt: (value: string) => void;
  batchLabel: string;
  plasmid: string;
  sampleCount: number;
  isBusy: boolean;
}

const quickPrompts = [
  "分析这个数据集",
  "用 pet15b 重跑",
  "只看 wrong 样本",
  "导出当前报告",
];

export function CommandWorkbench({
  language,
  draft,
  onDraftChange,
  onSubmit,
  onQuickPrompt,
  batchLabel,
  plasmid,
  sampleCount,
  isBusy,
}: CommandWorkbenchProps) {
  return (
    <section className="command-workbench" aria-label={t(language, "command.shellLabel")}>
      <div className="command-workbench-header">
        <div>
          <span className="command-workbench-kicker">{t(language, "command.kicker")}</span>
          <h2>{t(language, "command.title")}</h2>
        </div>
        <div className="command-workbench-meta">
          <span>{batchLabel}</span>
          <span>{plasmid}</span>
          <span>{sampleCount} samples</span>
        </div>
      </div>
      <div className="command-workbench-input">
        <textarea
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          placeholder={t(language, "command.placeholder")}
          rows={3}
          disabled={isBusy}
        />
        <button type="button" onClick={onSubmit} disabled={isBusy || draft.trim().length === 0}>
          {t(language, "command.submit")}
        </button>
      </div>
      <div className="command-workbench-quick-actions">
        {quickPrompts.map((prompt) => (
          <button key={prompt} type="button" onClick={() => onQuickPrompt(prompt)}>
            {prompt}
          </button>
        ))}
      </div>
    </section>
  );
}
```

```tsx
// src/components/ActionPlanCard.tsx
import type { AppLanguage, CommandPlan } from "../types";
import { t } from "../i18n";
import "./ActionPlanCard.css";

interface ActionPlanCardProps {
  language: AppLanguage;
  plan: CommandPlan | null;
  onConfirm?: () => void;
  onCancel?: () => void;
}

export function ActionPlanCard({ language, plan, onConfirm, onCancel }: ActionPlanCardProps) {
  if (!plan) return null;

  return (
    <section className="action-plan-card" aria-label={t(language, "command.planLabel")}>
      <div className="action-plan-card-header">
        <span>{t(language, "command.planKicker")}</span>
        <strong>{plan.summary}</strong>
      </div>
      <ul>
        {plan.actions.map((action) => (
          <li key={`${action.id}-${JSON.stringify(action.args)}`}>{action.id}</li>
        ))}
      </ul>
      {plan.needsConfirmation ? (
        <div className="action-plan-card-actions">
          <button type="button" onClick={onConfirm}>{t(language, "command.confirm")}</button>
          <button type="button" onClick={onCancel}>{t(language, "command.cancel")}</button>
        </div>
      ) : null}
    </section>
  );
}
```

```tsx
// src/components/ExecutionTimeline.tsx
import type { AppLanguage, ExecutionEvent } from "../types";
import "./ExecutionTimeline.css";

interface ExecutionTimelineProps {
  language: AppLanguage;
  events: ExecutionEvent[];
}

export function ExecutionTimeline({ events }: ExecutionTimelineProps) {
  return (
    <section className="execution-timeline">
      {events.map((event) => (
        <article key={event.id} className={`execution-event ${event.status}`}>
          <strong>{event.actionId}</strong>
          <span>{event.message}</span>
        </article>
      ))}
    </section>
  );
}
```

- [ ] **Step 4: Run the new contract tests and a build**

```powershell
python -m pytest tests\test_command_registry_contract.py -q
npm.cmd run build
```

Expected: the registry and component contract tests pass, and the new component files type-check.

- [ ] **Step 5: Commit the command UI foundation**

```powershell
git add src\components\CommandWorkbench.tsx src\components\CommandWorkbench.css src\components\ActionPlanCard.tsx src\components\ActionPlanCard.css src\components\ExecutionTimeline.tsx src\components\ExecutionTimeline.css src\utils\actionRegistry.ts src\types\index.ts src\i18n.ts tests\test_command_registry_contract.py
git commit -m "feat: add command workbench foundation"
```

## Task 5: Wire The Analysis Page To Parse Commands, Ask For Confirmation, And Execute Actions

**Files:**
- Create: `src/utils/commandExecution.ts`
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Modify: `src/components/ResultsWorkbench.tsx`
- Modify: `src/utils/resultSelection.ts`
- Modify: `tests/test_command_ui_contract.py`

- [ ] **Step 1: Extend the UI contract test with the new analysis-page command surface**

```python
def test_analysis_page_uses_command_workbench_and_timeline():
    app = _read("src/App.tsx")
    assert "<CommandWorkbench" in app
    assert "<ActionPlanCard" in app
    assert "<ExecutionTimeline" in app
    assert "window.electronAPI.interpretCommand" in app or ".interpretCommand(" in app
```

- [ ] **Step 2: Run the updated contract test to verify it fails**

```powershell
python -m pytest tests\test_command_ui_contract.py -q
```

Expected: FAIL because the analysis page does not yet call the interpreter or render the new components.

- [ ] **Step 3: Implement command-plan state, confirmation, filtering, and sequential execution**

```ts
// src/utils/commandExecution.ts
import type { CommandPlan, ExecutionEvent, Sample } from "../types";

interface ExecutePlanOptions {
  plan: CommandPlan;
  samples: Sample[];
  runAnalysis: () => Promise<void>;
  exportReport: () => Promise<string | null>;
  importDataset: () => Promise<void>;
  setPlasmid: (value: string) => void;
  setFilter: (value: string | null) => void;
  selectSample: (sampleId: string | null) => void;
  openExportFolder: (path: string) => Promise<void>;
  appendEvent: (event: ExecutionEvent) => void;
}

export async function executePlan(options: ExecutePlanOptions) {
  let lastExportedPath: string | null = null;

  for (const action of options.plan.actions) {
    options.appendEvent({
      id: `${action.id}-${Date.now()}`,
      actionId: action.id,
      status: "running",
      message: `Running ${action.id}`,
    });

    if (action.id === "set_plasmid") {
      options.setPlasmid(String(action.args.plasmid ?? "pet22b"));
    } else if (action.id === "filter_results") {
      options.setFilter(String(action.args.status ?? ""));
    } else if (action.id === "open_sample") {
      options.selectSample(String(action.args.sampleId ?? null));
    } else if (action.id === "import_dataset") {
      await options.importDataset();
    } else if (action.id === "run_analysis") {
      await options.runAnalysis();
    } else if (action.id === "export_report") {
      lastExportedPath = await options.exportReport();
    } else if (action.id === "open_export_folder" && lastExportedPath) {
      await options.openExportFolder(lastExportedPath);
    }

    options.appendEvent({
      id: `${action.id}-${Date.now()}-done`,
      actionId: action.id,
      status: "done",
      message: `Completed ${action.id}`,
    });
  }
}
```

```tsx
// src/App.tsx
const [commandDraft, setCommandDraft] = useState("");
const [commandPlan, setCommandPlan] = useState<CommandPlan | null>(null);
const [executionEvents, setExecutionEvents] = useState<ExecutionEvent[]>([]);
const [resultFilter, setResultFilter] = useState<string | null>(null);
const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);

const filteredSamples = resultFilter
  ? samples.filter((sample) => sample.status === resultFilter)
  : samples;

const appendExecutionEvent = (event: ExecutionEvent) => {
  setExecutionEvents((current) => [...current, event]);
};

const handleInterpretCommand = async () => {
  const raw = await window.electronAPI.interpretCommand(commandDraft.trim());
  const nextPlan = JSON.parse(String(raw)) as CommandPlan;
  setCommandPlan(nextPlan);
  setAwaitingConfirmation(nextPlan.needsConfirmation);
  setExecutionEvents(
    nextPlan.actions.map((action, index) => ({
      id: `${action.id}-${index}`,
      actionId: action.id,
      status: nextPlan.needsConfirmation ? "waiting_confirmation" : "planned",
      message: nextPlan.summary,
    }))
  );
};

const confirmPlan = async () => {
  if (!commandPlan) return;
  setAwaitingConfirmation(false);
  await executePlan({
    plan: commandPlan,
    samples,
    runAnalysis: async () => { await runAnalysis(); },
    exportReport: async () => {
      const result = await invoke("export-excel", samples, ab1Dir);
      return result ? String((result as { exported?: string }).exported ?? "") : null;
    },
    importDataset: handleSelectDatasetDir,
    setPlasmid,
    setFilter: setResultFilter,
    selectSample: setSelectedId,
    openExportFolder: async (exportedPath) => { await window.electronAPI.openExportFolder(exportedPath); },
    appendEvent: appendExecutionEvent,
  });
};
```

```tsx
// src/App.tsx analysis page body
<CommandWorkbench
  language={language}
  draft={commandDraft}
  onDraftChange={setCommandDraft}
  onSubmit={() => void handleInterpretCommand()}
  onQuickPrompt={setCommandDraft}
  batchLabel={datasetImport?.datasetName ?? (ab1Dir ?? plasmid)}
  plasmid={plasmid}
  sampleCount={samples.length}
  isBusy={isAnalyzing}
/>
<ActionPlanCard
  language={language}
  plan={commandPlan}
  onConfirm={awaitingConfirmation ? () => void confirmPlan() : undefined}
  onCancel={awaitingConfirmation ? () => { setCommandPlan(null); setExecutionEvents([]); setAwaitingConfirmation(false); } : undefined}
/>
<ExecutionTimeline language={language} events={executionEvents} />
<ResultsWorkbench
  language={language}
  samples={filteredSamples}
  selectedId={selectedId}
  onSelect={setSelectedId}
>
```

- [ ] **Step 4: Run the contract tests and a production build**

```powershell
python -m pytest tests\test_command_ui_contract.py tests\test_command_registry_contract.py -q
npm.cmd run build
```

Expected: PASS for the contract tests and a successful build with the new command orchestration state.

- [ ] **Step 5: Commit the command execution wiring**

```powershell
git add src\App.tsx src\App.css src\components\ResultsWorkbench.tsx src\utils\commandExecution.ts src\utils\resultSelection.ts tests\test_command_ui_contract.py
git commit -m "feat: wire command plans into analysis page"
```

## Task 6: Add Open-Export-Folder Support And Finalize Operational Action Coverage

**Files:**
- Modify: `electron/main.js`
- Modify: `src/App.tsx`
- Modify: `src/i18n.ts`
- Modify: `tests/test_command_registry_contract.py`

- [ ] **Step 1: Extend the registry contract test so exported-folder behavior is required**

```python
def test_action_registry_keeps_open_export_folder_confirmed():
    content = _read("src/utils/actionRegistry.ts")
    assert '{ id: "open_export_folder", label: "打开导出目录", needsConfirmation: true }' in content
```

- [ ] **Step 2: Run the contract test to verify the export-folder path is still covered**

```powershell
python -m pytest tests\test_command_registry_contract.py -q
```

Expected: PASS after Task 4; if it fails, fix the registry before adding more behavior.

- [ ] **Step 3: Make the export workflow remember the last exported file and expose a clear success message**

```tsx
// src/App.tsx
const [lastExportedPath, setLastExportedPath] = useState<string | null>(null);

const exportCurrentReport = async () => {
  if (!samples.length) return null;
  const result = await invoke("export-excel", samples, ab1Dir);
  const exported = result ? String((result as { exported?: string }).exported ?? "") : "";
  setLastExportedPath(exported || null);
  return exported || null;
};
```

```ts
// src/i18n.ts
command: {
  exportedReady: "报告已导出，可继续打开导出目录。",
  openFolder: "打开导出目录",
}
```

```tsx
// src/App.tsx executePlan binding
exportReport: exportCurrentReport,
openExportFolder: async (exportedPath) => {
  await window.electronAPI.openExportFolder(exportedPath || lastExportedPath || "");
},
```

- [ ] **Step 4: Run the build and the interpreter tests together**

```powershell
python -m pytest tests\test_command_intent.py tests\test_command_registry_contract.py -q
npm.cmd run build
```

Expected: PASS, with export-related command coverage still intact.

- [ ] **Step 5: Commit the export-folder finishing pass**

```powershell
git add electron\main.js src\App.tsx src\i18n.ts tests\test_command_registry_contract.py
git commit -m "feat: support opening exported report folders"
```

## Task 7: Apply The Full Scientific Workbench Visual Redesign

**Files:**
- Modify: `src/App.css`
- Modify: `src/styles.css`
- Modify: `src/components/CommandWorkbench.css`
- Modify: `src/components/ActionPlanCard.css`
- Modify: `src/components/ExecutionTimeline.css`
- Modify: `src/components/AssistantPage.css`
- Modify: `src/components/TabLayout.css`
- Modify: `src/components/ResultsWorkbench.css`

- [ ] **Step 1: Add a contract test ensuring the command workbench is the new primary analysis surface**

```python
def test_app_styles_define_command_workbench_shell():
    css = _read("src/App.css")
    assert ".command-workbench-shell" in css or ".command-workbench" in css
    assert ".analysis-command-stage" in css
    assert ".analysis-main-surface" in css
```

- [ ] **Step 2: Run the CSS contract test to verify it fails before the redesign lands**

```powershell
python -m pytest tests\test_command_ui_contract.py -q
```

Expected: FAIL if the new top-level workbench layout classes have not been added yet.

- [ ] **Step 3: Restyle the analysis page into a command-first scientific workbench**

```css
/* src/App.css */
.analysis-command-stage {
  display: grid;
  gap: 18px;
  margin-bottom: 24px;
}

.analysis-main-surface {
  display: grid;
  gap: 24px;
}

.command-workbench {
  border: 1px solid rgba(31, 41, 55, 0.08);
  background: linear-gradient(180deg, rgba(250, 247, 240, 0.96), rgba(244, 239, 230, 0.92));
  border-radius: 28px;
  padding: 24px 28px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

.execution-timeline {
  display: grid;
  gap: 10px;
}

.execution-event.running {
  border-left: 4px solid #8b5cf6;
}

.execution-event.done {
  border-left: 4px solid #15803d;
}

.execution-event.failed {
  border-left: 4px solid #b91c1c;
}
```

```css
/* src/components/AssistantPage.css */
.assistant-page {
  display: grid;
  gap: 18px;
  padding: 24px;
}

.assistant-page-header {
  display: grid;
  gap: 8px;
  padding: 20px 24px;
  border-radius: 24px;
  background: rgba(248, 245, 238, 0.9);
  border: 1px solid rgba(31, 41, 55, 0.08);
}
```

- [ ] **Step 4: Run the build and manually inspect the redesigned layout in Electron**

```powershell
npm.cmd run build
npm.cmd run electron:dev
```

Expected: the analysis page opens with a large command surface above the result workbench, and the Assistant page looks like a full-page tool rather than a side rail.

- [ ] **Step 5: Commit the visual redesign**

```powershell
git add src\App.css src\styles.css src\components\CommandWorkbench.css src\components\ActionPlanCard.css src\components\ExecutionTimeline.css src\components\AssistantPage.css src\components\TabLayout.css src\components\ResultsWorkbench.css
git commit -m "feat: redesign analysis page as command workbench"
```

## Task 8: Run End-To-End Verification And Update User-Facing Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-08-command-workbench-and-assistant-redesign-design.md`

- [ ] **Step 1: Update README usage text to mention the new Chinese command workflow**

```markdown
## Command Workbench

The analysis page now supports direct Chinese operational commands such as:

- `分析这个数据集`
- `用 pet15b 重跑`
- `只看 wrong 样本`
- `导出当前报告`

Higher-risk actions such as running analysis or exporting reports require confirmation before execution.
```

- [ ] **Step 2: Run the targeted automated checks in a temp directory under the workspace**

```powershell
if (!(Test-Path .\tmp)) { New-Item -ItemType Directory .\tmp | Out-Null }
$env:TEMP = "$PWD\tmp"
$env:TMP = "$PWD\tmp"
python -m pytest tests\test_command_intent.py tests\test_command_ui_contract.py tests\test_command_registry_contract.py tests\test_agent_tools.py tests\test_agent_chat.py -q --basetemp=$PWD\tmp\pytest-command-workbench
npm.cmd run build
```

Expected: PASS for the targeted pytest suite and a successful production build.

- [ ] **Step 3: Run the manual smoke script in the desktop shell**

```text
1. Launch `npm run electron:dev`.
2. Open the Analysis page.
3. Trigger `分析这个数据集`.
4. Confirm the plan when prompted.
5. Trigger `只看 wrong 样本`.
6. Trigger `导出当前报告并打开导出目录`.
7. Switch to the Assistant tab and verify the old agent experience still works.
```

- [ ] **Step 4: Record the final implementation state in the spec file**

```markdown
## Implementation Note

Phase-1 delivery keeps command interpretation deterministic on the Python side and keeps explanation-oriented agent interactions inside the dedicated Assistant page.
```

- [ ] **Step 5: Commit verification and docs**

```powershell
git add README.md docs\superpowers\specs\2026-04-08-command-workbench-and-assistant-redesign-design.md
git commit -m "docs: document command workbench workflow"
```
