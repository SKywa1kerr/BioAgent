# Agent Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled Agent Panel to the analysis page that can answer questions from current analysis context and run a visible multi-step tool loop through the Python sidecar.

**Architecture:** The renderer owns loop execution and message rendering. Electron adds a single `agent-chat` bridge plus reuses existing IPC handlers for concrete actions. Python adds `agent_tools.py` and `agent_chat.py` to hold tool metadata, prompt construction, and structured action parsing, while `main.py` remains the CLI boundary.

**Tech Stack:** Electron 33, React 18, TypeScript 5, Vite 5, Python 3.10+, pytest

---

## File Structure

### Files to create

- `src-python/bioagent/agent_tools.py` - tool specs, tool filtering, local tool execution helpers for agent flow
- `src-python/bioagent/agent_chat.py` - runtime config, prompt construction, structured response parsing
- `tests/test_agent_tools.py` - Python tests for tool registry and tool filtering
- `tests/test_agent_chat.py` - Python tests for response parsing and stop reasons
- `src/components/AgentPanel.tsx` - right-side panel UI and controlled loop runner
- `src/components/AgentPanel.css` - panel styling
- `src/components/ChatMessage.tsx` - message renderer for user, agent, plan, and tool status cards
- `src/components/ChatMessage.css` - message styling

### Files to modify

- `src-python/bioagent/main.py` - add `--agent-chat` entry
- `electron/main.js` - add `agent-chat` IPC and shared Python invocation helper if missing
- `src/types/index.ts` - add agent runtime/message/tool types
- `src/App.tsx` - mount Agent Panel into analysis page and pass context
- `src/App.css` - make analysis page support persistent right panel

---

### Task 1: Add shared agent types in TypeScript

**Files:**
- Modify: `src/types/index.ts`
- Test: `npm run build`

- [ ] **Step 1: Add the failing type usage plan**

Add these types to `src/types/index.ts` near the existing app models:

```ts
export type ToolCategory = "query" | "action";

export type StopReason =
  | "final_reply"
  | "max_rounds_reached"
  | "tool_failed"
  | "invalid_model_output"
  | "permission_denied"
  | "aborted";

export interface TokenUsage {
  input: number;
  output: number;
  total?: number;
}

export interface ToolCall {
  tool: "query_samples" | "query_history" | "get_sample_detail" | "run_analysis" | "export_report";
  args: Record<string, unknown>;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  category: ToolCategory;
}

export interface AgentRuntimeConfig {
  maxRounds: number;
  maxToolCallsPerTurn: number;
  maxRecentMessages: number;
  allowActionTools: boolean;
  includeUsage: boolean;
}

export interface ToolResult {
  tool: ToolCall["tool"];
  ok: boolean;
  summary: string;
  data?: unknown;
}

export interface AgentFailure {
  kind: "tool_failed" | "invalid_model_output" | "permission_denied";
  message: string;
  toolName?: string;
}

export type ChatMessage =
  | { id: string; type: "user"; content: string; timestamp: number }
  | { id: string; type: "agent"; content: string; timestamp: number; usage?: TokenUsage; stopReason?: StopReason }
  | { id: string; type: "plan"; content: string; timestamp: number }
  | { id: string; type: "tool_status"; content: string; timestamp: number; toolName: ToolCall["tool"]; status: "running" | "done" | "failed" };

export type AgentTurnResponse =
  | { action: "reply"; content: string; usage?: TokenUsage; stopReason?: StopReason }
  | { action: "tool_calls"; message: string; calls: ToolCall[]; usage?: TokenUsage };

export interface AgentContext {
  currentAnalysis?: {
    sourcePath?: string;
    samples: Sample[];
    selectedSampleId?: string | null;
  };
  recentToolResults?: ToolResult[];
  history?: ChatMessage[];
  runtime?: AgentRuntimeConfig;
}
```

- [ ] **Step 2: Run TypeScript build to verify the new exports do not break consumers**

Run: `npm run build`
Expected: build may fail only for missing agent UI imports that will be added later, but `src/types/index.ts` itself should not introduce syntax errors.

- [ ] **Step 3: Commit the type layer**

```bash
git add src/types/index.ts
git commit -m "feat: add agent runtime and message types"
```

---

### Task 2: Add Python tool registry and tool filtering

**Files:**
- Create: `src-python/bioagent/agent_tools.py`
- Create: `tests/test_agent_tools.py`
- Test: `tests/test_agent_tools.py`

- [ ] **Step 1: Write failing tests for response parsing and stop reasons**

Create `tests/test_agent_chat.py`:

```python
import pytest

from bioagent.agent_chat import DEFAULT_RUNTIME_CONFIG, parse_agent_response


def test_parse_reply_response():
    result = parse_agent_response('{"action":"reply","content":"done"}')
    assert result["action"] == "reply"
    assert result["content"] == "done"
    assert result["stopReason"] == "final_reply"


def test_parse_tool_calls_response():
    raw = '{"action":"tool_calls","message":"Checking sample detail","calls":[{"tool":"get_sample_detail","args":{"sampleId":"S1"}}]}'
    result = parse_agent_response(raw)
    assert result["action"] == "tool_calls"
    assert result["calls"][0]["tool"] == "get_sample_detail"


def test_invalid_response_raises_value_error():
    with pytest.raises(ValueError):
        parse_agent_response('{"action":"unknown"}')


def test_runtime_defaults_match_spec():
    assert DEFAULT_RUNTIME_CONFIG["maxRounds"] == 3
    assert DEFAULT_RUNTIME_CONFIG["maxToolCallsPerTurn"] == 3
    assert DEFAULT_RUNTIME_CONFIG["allowActionTools"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd src-python && python -m pytest ../tests/test_agent_chat.py -v`
Expected: FAIL because `bioagent.agent_chat` does not exist yet.

- [ ] **Step 3: Implement the minimal runtime config and response parser**

Create `src-python/bioagent/agent_chat.py`:

```python
from __future__ import annotations

import json
from typing import Any

from .agent_tools import build_tools_prompt, filter_tool_specs, get_tool_specs

DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "maxRounds": 3,
    "maxToolCallsPerTurn": 3,
    "maxRecentMessages": 12,
    "allowActionTools": True,
    "includeUsage": True,
}


def build_agent_prompt(context: dict[str, Any]) -> str:
    runtime = {**DEFAULT_RUNTIME_CONFIG, **context.get("runtime", {})}
    tools = filter_tool_specs(get_tool_specs(), allow_action_tools=runtime["allowActionTools"])
    return "\n\n".join(
        [
            "You are BioAgent, a controlled desktop analysis assistant.",
            "Use tools only when needed. Prefer a final answer when current context already resolves the question.",
            build_tools_prompt(tools),
        ]
    )


def parse_agent_response(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_model_output") from exc

    action = data.get("action")
    if action == "reply":
        return {
            "action": "reply",
            "content": data.get("content", ""),
            "usage": data.get("usage"),
            "stopReason": data.get("stopReason", "final_reply"),
        }
    if action == "tool_calls":
        calls = data.get("calls") or []
        if not isinstance(calls, list):
            raise ValueError("invalid_model_output")
        return {
            "action": "tool_calls",
            "message": data.get("message", ""),
            "calls": calls[: DEFAULT_RUNTIME_CONFIG["maxToolCallsPerTurn"]],
            "usage": data.get("usage"),
        }
    raise ValueError("invalid_model_output")
```

- [ ] **Step 4: Run tests to verify parser and defaults pass**

Run: `cd src-python && python -m pytest ../tests/test_agent_chat.py -v`
Expected: PASS for all tests.

- [ ] **Step 5: Commit the parser layer**

```bash
git add src-python/bioagent/agent_chat.py tests/test_agent_chat.py
git commit -m "feat: add agent runtime config and response parser"
```

---

### Task 4: Add `--agent-chat` to the Python sidecar entry point

**Files:**
- Modify: `src-python/bioagent/main.py`
- Test: `tests/test_agent_chat.py`

- [ ] **Step 1: Write the failing tests for tool registry and filtering**

Create `tests/test_agent_tools.py`:

```python
from bioagent.agent_tools import get_tool_specs, filter_tool_specs


def test_get_tool_specs_exposes_expected_tools():
    names = [tool["name"] for tool in get_tool_specs()]
    assert names == [
        "query_samples",
        "query_history",
        "get_sample_detail",
        "run_analysis",
        "export_report",
    ]


def test_filter_tool_specs_blocks_action_tools():
    filtered = filter_tool_specs(get_tool_specs(), allow_action_tools=False)
    names = [tool["name"] for tool in filtered]
    assert "run_analysis" not in names
    assert "export_report" not in names
    assert names == ["query_samples", "query_history", "get_sample_detail"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd src-python && python -m pytest ../tests/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors because `bioagent.agent_tools` does not exist yet.

- [ ] **Step 3: Implement the minimal tool registry**

Create `src-python/bioagent/agent_tools.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    category: str


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="query_samples",
        description="Read current samples or filter them by status, sample id, or analysis id.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "sampleId": {"type": "string"},
                "analysisId": {"type": "string"},
            },
        },
        category="query",
    ),
    ToolSpec(
        name="query_history",
        description="Read recent analysis history summaries.",
        parameters={"type": "object", "properties": {"limit": {"type": "number"}}},
        category="query",
    ),
    ToolSpec(
        name="get_sample_detail",
        description="Read detailed data for a single sample.",
        parameters={"type": "object", "properties": {"sampleId": {"type": "string"}}},
        category="query",
    ),
    ToolSpec(
        name="run_analysis",
        description="Run analysis on AB1 and reference folders using the current desktop workflow.",
        parameters={
            "type": "object",
            "properties": {
                "ab1Dir": {"type": "string"},
                "genesDir": {"type": "string"},
                "plasmid": {"type": "string"},
                "useLLM": {"type": "boolean"},
            },
            "required": ["ab1Dir"],
        },
        category="action",
    ),
    ToolSpec(
        name="export_report",
        description="Export an Excel report for the current samples.",
        parameters={"type": "object", "properties": {}},
        category="action",
    ),
)


def get_tool_specs() -> list[dict[str, Any]]:
    return [asdict(tool) for tool in TOOLS]


def filter_tool_specs(tool_specs: list[dict[str, Any]], allow_action_tools: bool) -> list[dict[str, Any]]:
    if allow_action_tools:
        return tool_specs
    return [tool for tool in tool_specs if tool["category"] != "action"]


def build_tools_prompt(tool_specs: list[dict[str, Any]]) -> str:
    lines = ["Available tools:"]
    for tool in tool_specs:
        lines.append(f"- {tool['name']} ({tool['category']}): {tool['description']}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify the registry passes**

Run: `cd src-python && python -m pytest ../tests/test_agent_tools.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit the registry layer**

```bash
git add src-python/bioagent/agent_tools.py tests/test_agent_tools.py
git commit -m "feat: add agent tool registry and filtering"
```

---

### Task 3: Add Python agent response parsing and runtime config

**Files:**
- Create: `src-python/bioagent/agent_chat.py`
- Create: `tests/test_agent_chat.py`
- Test: `tests/test_agent_chat.py`

- [ ] **Step 1: Write a failing CLI test for `--agent-chat`**

Append this test to `tests/test_agent_chat.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


def test_agent_chat_cli_returns_reply_json():
    root = Path(__file__).resolve().parents[1] / "src-python"
    payload = json.dumps({"mockResponse": '{"action":"reply","content":"ok"}'})
    result = subprocess.run(
        [sys.executable, "-m", "bioagent.main", "--agent-chat", payload],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["action"] == "reply"
    assert data["content"] == "ok"
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `cd src-python && python -m pytest ../tests/test_agent_chat.py::test_agent_chat_cli_returns_reply_json -v`
Expected: FAIL because `main.py` does not yet accept `--agent-chat`.

- [ ] **Step 3: Add the CLI entry in `main.py`**

Insert this import near the other local imports:

```python
from .agent_chat import build_agent_prompt, parse_agent_response
```

Add this argument to the parser:

```python
parser.add_argument("--agent-chat", help="Run a structured agent chat turn from a JSON payload")
```

Add this branch before the existing history/export/analysis branches:

```python
    if args.agent_chat:
        payload = json.loads(args.agent_chat)
        if "mockResponse" in payload:
            print(json.dumps(parse_agent_response(payload["mockResponse"]), ensure_ascii=False))
            return

        build_agent_prompt(payload)
        fallback = {
            "action": "reply",
            "content": "Agent chat is wired, but no live model response was provided.",
            "stopReason": "aborted",
        }
        print(json.dumps(fallback, ensure_ascii=False))
        return
```

- [ ] **Step 4: Run the CLI test to verify it passes**

Run: `cd src-python && python -m pytest ../tests/test_agent_chat.py::test_agent_chat_cli_returns_reply_json -v`
Expected: PASS.

- [ ] **Step 5: Commit the CLI bridge**

```bash
git add src-python/bioagent/main.py tests/test_agent_chat.py
git commit -m "feat: add agent chat CLI entry point"
```

---

### Task 5: Add Electron `agent-chat` IPC bridge

**Files:**
- Modify: `electron/main.js`
- Test: `npm run build`

- [ ] **Step 1: Inspect the existing Python invocation path and add a shared helper if needed**

In `electron/main.js`, prefer a helper shaped like this near the other Python helpers:

```javascript
function getPythonCommand() {
  if (isDev) {
    return {
      command: process.platform === "win32" ? "python" : "python3",
      args: ["-m", "bioagent.main"],
      cwd: path.join(__dirname, "../src-python"),
      env: { ...process.env, PYTHONPATH: path.join(__dirname, "../src-python") },
    };
  }

  const exeName = process.platform === "win32" ? "bioagent.exe" : "bioagent";
  const exePath = path.join(process.resourcesPath, "sidecar", "bioagent-sidecar", exeName);
  return { command: exePath, args: [], cwd: path.dirname(exePath), env: process.env };
}
```

- [ ] **Step 2: Add the `agent-chat` handler**

Add this handler next to the other `ipcMain.handle(...)` registrations:

```javascript
ipcMain.handle("agent-chat", async (_event, payload) => {
  return new Promise((resolve, reject) => {
    const py = getPythonCommand();
    const args = [...py.args, "--agent-chat", JSON.stringify(payload)];

    execFile(py.command, args, { cwd: py.cwd, env: py.env, maxBuffer: 20 * 1024 * 1024 }, (err, stdout, stderr) => {
      if (err) {
        console.error("Agent chat error:", stderr || err.message);
        reject(stderr || err.message);
        return;
      }
      resolve(stdout);
    });
  });
});
```

- [ ] **Step 3: Run the frontend build to verify the new main-process code is syntactically valid**

Run: `npm run build`
Expected: the build may still fail due to missing Agent Panel UI files, but `electron/main.js` should not introduce a new syntax error.

- [ ] **Step 4: Commit the Electron bridge**

```bash
git add electron/main.js
git commit -m "feat: add electron agent chat bridge"
```

---

### Task 6: Add React message renderer and Agent Panel shell

**Files:**
- Create: `src/components/ChatMessage.tsx`
- Create: `src/components/ChatMessage.css`
- Create: `src/components/AgentPanel.tsx`
- Create: `src/components/AgentPanel.css`
- Test: `npm run build`

- [ ] **Step 1: Refine loop state so the request history passed to Python is current**

In `src/components/AgentPanel.tsx`, replace the direct use of stale `messages` inside `handleSubmit` with a local `conversationHistory` array and send that rolling history back to Python each round.

- [ ] **Step 2: Add duplicate-tool guard before execution**

Inside the `tool_calls` branch, insert:

```tsx
const seenCalls = new Set<string>();
for (const call of response.calls.slice(0, DEFAULT_RUNTIME.maxToolCallsPerTurn)) {
  const signature = JSON.stringify(call);
  if (seenCalls.has(signature)) {
    continue;
  }
  seenCalls.add(signature);
  // existing execution body
}
```

- [ ] **Step 3: Surface stop reason and usage more cleanly in the message card**

In `src/components/ChatMessage.tsx`, replace the meta rendering with:

```tsx
{message.type === "agent" && (message.stopReason || message.usage) ? (
  <div className="message-meta">
    {message.stopReason ? <span>stop: {message.stopReason}</span> : null}
    {message.usage ? <span>tokens: {message.usage.total ?? message.usage.input + message.usage.output}</span> : null}
  </div>
) : null}
```

- [ ] **Step 4: Run the build again**

Run: `npm run build`
Expected: PASS.

- [ ] **Step 5: Commit the loop refinements**

```bash
git add src/components/AgentPanel.tsx src/components/ChatMessage.tsx
git commit -m "feat: refine agent loop state and runtime metadata"
```

---

### Task 9: Verify Python and frontend behavior together

**Files:**
- No new files
- Test: `tests/test_agent_tools.py`, `tests/test_agent_chat.py`, `npm run build`

- [ ] **Step 1: Run all Python agent tests**

Run: `cd src-python && python -m pytest ../tests/test_agent_tools.py ../tests/test_agent_chat.py -v`
Expected: PASS.

- [ ] **Step 2: Run the frontend build**

Run: `npm run build`
Expected: PASS.

- [ ] **Step 3: Manual smoke test in desktop dev mode**

Run: `npm run electron:dev`
Expected manual checks:

1. Analysis tab shows a persistent `Agent` panel on the right.
2. Sending a question creates a `user` message and either a final answer or a plan card.
3. Asking for history triggers visible tool status cards and returns a final answer.
4. Asking for sample detail can resolve against the currently loaded sample list.
5. A failing tool call yields a visible failed tool-status card and the loop stops cleanly.

- [ ] **Step 4: Commit integration fixes if needed**

```bash
git add -u
git commit -m "fix: polish agent panel integration"
```

---

## Self-Review

### Spec coverage

- Runtime config: covered in Task 3 and Task 8.
- Stop reasons and usage: covered in Task 3, Task 6, and Task 8.
- Tool categories and filtering: covered in Task 2 and Task 3.
- `agent-chat` sidecar bridge: covered in Task 4 and Task 5.
- Right-side panel UI: covered in Task 6 and Task 7.
- Visible tool-status loop: covered in Task 6 and Task 8.
- Validation: covered in Task 9.

### Placeholder scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Every task contains explicit file paths, code snippets, and commands.

### Type consistency

- `AgentRuntimeConfig`, `AgentTurnResponse`, `ToolCall`, `ToolSpec`, `ToolResult`, `StopReason`, and `ChatMessage` are introduced in Task 1 and referenced consistently later.
- Python runtime config and TypeScript runtime config share the same field names.

- [ ] **Step 1: Create the message renderer**

Create `src/components/ChatMessage.tsx`:

```tsx
import { ChatMessage } from "../types";
import "./ChatMessage.css";

interface Props {
  message: ChatMessage;
}

export function ChatMessageCard({ message }: Props) {
  if (message.type === "tool_status") {
    return (
      <div className={`chat-message tool-status ${message.status}`}>
        <div className="message-label">{message.toolName}</div>
        <div>{message.content}</div>
      </div>
    );
  }

  return (
    <div className={`chat-message ${message.type}`}>
      <div className="message-body">{message.content}</div>
      {message.type === "agent" && (message.stopReason || message.usage) ? (
        <div className="message-meta">
          {message.stopReason ? <span>{message.stopReason}</span> : null}
          {message.usage ? <span>{message.usage.input}/{message.usage.output}</span> : null}
        </div>
      ) : null}
    </div>
  );
}
```

Create `src/components/ChatMessage.css`:

```css
.chat-message {
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
  font-size: 13px;
  line-height: 1.5;
}

.chat-message.user { background: #12344a; color: #fff; margin-left: 28px; }
.chat-message.agent { background: #f6f8fb; color: #1f2937; margin-right: 12px; }
.chat-message.plan { background: #eef6ea; color: #24452f; border: 1px solid #cce3cf; }
.chat-message.tool-status { background: #fff; border: 1px solid #d7dee8; }
.chat-message.tool-status.running { border-color: #2a6cb6; }
.chat-message.tool-status.done { border-color: #2f855a; }
.chat-message.tool-status.failed { border-color: #c53030; }
.message-label { font-weight: 600; margin-bottom: 4px; }
.message-meta { margin-top: 8px; font-size: 11px; color: #6b7280; display: flex; gap: 8px; }
```

- [ ] **Step 2: Create the Agent Panel shell with loop scaffolding**

Create `src/components/AgentPanel.tsx` and `src/components/AgentPanel.css` with the loop shell, tool execution stubs, and right-side layout described in the spec.

- [ ] **Step 3: Run the frontend build to verify the new components type-check**

Run: `npm run build`
Expected: PASS if the remaining app wiring already matches, or fail only for `App.tsx` not yet using the new component signatures.

- [ ] **Step 4: Commit the panel shell**

```bash
git add src/components/ChatMessage.tsx src/components/ChatMessage.css src/components/AgentPanel.tsx src/components/AgentPanel.css
git commit -m "feat: add agent panel shell and message cards"
```

---

### Task 7: Mount the Agent Panel into the analysis page

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Test: `npm run build`

- [ ] **Step 1: Mount `AgentPanel` in the analysis tab**

Add this import in `src/App.tsx`:

```tsx
import { AgentPanel } from "./components/AgentPanel";
```

Add this callback inside `App()`:

```tsx
const handleAgentAnalysisResult = (nextSamples: Sample[]) => {
  setSamples(nextSamples);
  setSelectedId(nextSamples[0]?.id ?? null);
};
```

Replace the current analysis-page body wrapper with a two-region layout that keeps the existing sidebar and main content on the left and mounts the panel on the right.

- [ ] **Step 2: Add layout CSS for the persistent right panel**

Append to `src/App.css`:

```css
.analysis-workspace {
  display: flex;
  min-height: 0;
  flex: 1;
}

.analysis-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

@media (max-width: 1200px) {
  .analysis-workspace {
    flex-direction: column;
  }

  .agent-panel {
    width: 100%;
    min-width: 0;
    border-left: none;
    border-top: 1px solid #d7dee8;
  }
}
```

- [ ] **Step 3: Run the build to verify the app compiles with the mounted panel**

Run: `npm run build`
Expected: PASS.

- [ ] **Step 4: Commit the mounted layout**

```bash
git add src/App.tsx src/App.css
git commit -m "feat: mount agent panel in analysis workspace"
```

---

### Task 8: Tighten the Agent Panel loop behavior

**Files:**
- Modify: `src/components/AgentPanel.tsx`
- Modify: `src/components/ChatMessage.tsx`
- Test: `npm run build`
