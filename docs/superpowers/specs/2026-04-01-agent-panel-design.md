## BioAgent Desktop Agent Panel Design

### Goal

Add a controlled `Agent Panel` to the analysis page of BioAgent Desktop. The panel should let the user ask questions in natural language, inspect the current analysis context, and trigger a small set of existing actions through a visible agentic loop.

This version is intentionally scoped to a controlled workflow:

- The agent starts from the current analysis page context.
- The agent may query history, samples, sample detail, re-run analysis, and export reports.
- The agent may propose a short plan and execute multiple tools sequentially.
- The agent may not directly modify stored analysis results, rewrite judgments, or silently change settings.

The goal is to make the desktop app feel like an explainable biology analysis assistant, not a fully autonomous general-purpose agent.

### Product Decision Summary

- Priority: complete `Agent Panel / agentic loop`
- Interaction scope: option 2
- Default context: current analysis first, with ability to query history and detail on demand
- Execution style: the agent can produce a short plan and then execute multiple tools sequentially with visible status updates
- UI placement: right-side panel inside the analysis page, not a separate tab
- Design influence adopted from `claw-code`: explicit loop config, explicit stop reasons, typed tool categories, visible failure summaries

## Architecture

The implementation keeps the current Electron + React + Python sidecar structure.

1. React owns the visible chat session, tool status rendering, and loop control.
2. Electron exposes a single `agent-chat` IPC entry plus the existing analysis/history/export IPC capabilities.
3. Python owns prompt construction, tool descriptions, response parsing, and action selection.
4. Tool execution remains in the desktop app layer and existing sidecar commands, not inside the model itself.

This keeps the loop explainable and avoids giving the model direct authority over local system behavior.

## Runtime Configuration

Borrowing the useful idea from `claw-code`, the agent loop should not rely on hidden constants scattered across files. Version one should define a small explicit runtime config object.

```ts
type AgentRuntimeConfig = {
  maxRounds: number;
  maxToolCallsPerTurn: number;
  maxRecentMessages: number;
  allowActionTools: boolean;
  includeUsage: boolean;
};
```

Initial defaults:

- `maxRounds = 3`
- `maxToolCallsPerTurn = 3`
- `maxRecentMessages = 12`
- `allowActionTools = true`
- `includeUsage = true`

The config can stay hardcoded in version one, but it should exist as a named shape so the loop behavior is easy to inspect and adjust later.

## Agent Loop

### Turn Model

Each user request enters a controlled loop:

1. Frontend sends the user message plus current analysis context to Python through `agent-chat`.
2. Python returns either a direct reply or a list of tool calls with a short plan message.
3. Frontend renders the plan, executes tools one by one, and renders each tool status.
4. Tool results are added to the agent context and sent back to Python for the next turn.
5. The loop ends when Python returns a final reply or the frontend reaches the turn limit.

### Termination Rules

- End immediately when Python returns `reply`.
- Hard stop after 3 loop rounds.
- If the limit is reached, show a final assistant message asking the user to narrow or continue the question.
- Every terminal state should carry an explicit `stop_reason`.

### Stop Reasons

To keep the loop inspectable, every completed interaction should resolve to one of these reasons:

- `final_reply`
- `max_rounds_reached`
- `tool_failed`
- `invalid_model_output`
- `permission_denied`
- `aborted`

The UI does not need to surface all of these as raw labels, but the runtime should keep them available for debugging and concise user-facing summaries.

### Why This Shape

This gives the app a real agentic workflow without introducing uncontrolled recursion, hidden side effects, or hard-to-debug cross-process behavior.

## Message Protocol

### Python Response Contract

Python should return one of two actions:

```ts
type AgentTurnResponse =
  | {
      action: "reply";
      content: string;
      usage?: TokenUsage;
      stopReason?: StopReason;
    }
  | {
      action: "tool_calls";
      message: string;
      calls: ToolCall[];
      usage?: TokenUsage;
    };
```

### Tool Call Contract

```ts
type ToolCall = {
  tool: "query_samples" | "query_history" | "get_sample_detail" | "run_analysis" | "export_report";
  args: Record<string, unknown>;
};
```

### Tool Metadata Contract

Borrowing the useful part of `claw-code`'s tool typing, the Python side should maintain metadata for each tool instead of only a bare function map.

```ts
type ToolCategory = "query" | "action";

type ToolSpec = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  category: ToolCategory;
};
```

This lets the prompt clearly distinguish tools that inspect data from tools that trigger work.

### Frontend Context Contract

```ts
type AgentContext = {
  currentAnalysis?: {
    sourcePath?: string;
    samples: Sample[];
    selectedSampleId?: string | null;
  };
  recentToolResults?: ToolResult[];
  history?: ChatMessage[];
  runtime?: AgentRuntimeConfig;
};
```

### Usage Contract

Borrowing another good pattern from `claw-code`, usage should be tracked explicitly instead of being hidden in logs.

```ts
type TokenUsage = {
  input: number;
  output: number;
  total?: number;
};
```

The first version only needs per-turn usage. Aggregated usage across the visible session can be added later in the frontend.

### Error And Permission Contract

Tool and runtime failures should be normalized into explicit summaries.

```ts
type AgentFailure = {
  kind: "tool_failed" | "invalid_model_output" | "permission_denied";
  message: string;
  toolName?: string;
};
```

The permission case is mostly forward-looking for this desktop app, but the protocol should already have a place for it.

### UI Message Types

The React layer should normalize conversation display into four message types:

- `user`
- `agent`
- `plan`
- `tool_status`

These are display-oriented types. They exist to keep the panel readable and auditable.

## Tool Boundaries

Version one should expose exactly five tools.

### 1. `query_samples`

Category: `query`

Purpose:
Read the current sample list or a filtered subset.

Allowed use:
- Filter by status
- Look up a sample by ID
- Inspect a current analysis or a specific stored analysis

No writes.

### 2. `get_sample_detail`

Category: `query`

Purpose:
Read the detailed fields for one sample so the agent can explain a judgment.

Allowed use:
- Identity
- Coverage
- Frameshift
- Mutation summary
- Rule/reason fields

No writes.

### 3. `query_history`

Category: `query`

Purpose:
Read analysis history summaries.

Allowed use:
- List recent analyses
- Compare counts or timestamps
- Find a likely related past run

Do not return full large payloads by default.

### 4. `run_analysis`

Category: `action`

Purpose:
Trigger an existing analysis flow using current app capabilities.

Allowed arguments:
- `ab1Dir`
- `genesDir`
- `plasmid`
- optional `useLLM` flag, default off for this version

This is an explicit action tool, but it only calls existing behavior already available in the app.

### 5. `export_report`

Category: `action`

Purpose:
Export a report for the current sample set.

Allowed behavior:
- Use the current export path workflow
- Return success or failure plus exported path summary

No hidden format changes or extra file generation behavior.

### Global Tool Constraints

- At most 3 tool calls per model turn
- No duplicate call with identical args in the same turn
- `run_analysis` and `export_report` must be preceded by a short plan message
- If existing tool results are sufficient, the next turn should return `reply` instead of exploring further
- If `allowActionTools` is false in runtime config, `run_analysis` and `export_report` must be filtered out before prompt construction

## Frontend Design

### Placement

The `Agent Panel` belongs inside the analysis page as a right-side panel.

Rationale:
- The agent's default context is the current analysis page
- Users should see samples and detailed evidence while chatting
- This avoids making the workflow chat-first and preserves the current primary analysis flow

### Layout

Desktop default:
- Left: sample list
- Center: current sample detail and chromatogram or mutation views
- Right: `Agent Panel`

Narrow window fallback:
- The right panel may collapse into a drawer, but desktop should prefer a persistent panel

### Panel Sections

1. Header
- Panel title
- Context indicator
- Busy or idle indicator
- Clear chat action

2. Message stream
- User messages
- Agent final replies
- Plan cards
- Tool status cards
- Concise failure cards when needed

3. Composer
- Multi-line input
- Send action
- Enter to send, Shift+Enter for newline

### Message Rendering Rules

- `user`: normal user bubble
- `agent`: normal assistant card for direct answers and final summaries
- `plan`: distinct lightweight card for what the agent will do next
- `tool_status`: compact status card showing tool name, arguments summary, state, and short result summary

The first version should avoid streaming tokens, multi-session chat management, or rich markdown-heavy rendering.

### Failure Presentation

Inspired by `claw-code`'s explicit runtime outcomes, the UI should not silently swallow errors.

- Tool failure should render as a visible status card
- Invalid model output should render as a safe assistant error card
- Permission denial should render as a concise assistant explanation
- Final replies may optionally include usage and stop reason in a subtle footer or developer-facing debug view

## Backend Module Design

### `src-python/bioagent/agent_tools.py`

Responsibilities:
- Register tool metadata
- Define parameter schemas
- Dispatch tool handlers
- Build the tool description block used in the prompt
- Filter tools by runtime permission context when needed

This module should not know about Electron.

### `src-python/bioagent/agent_chat.py`

Responsibilities:
- Build system prompt
- Package current context
- Call the LLM
- Parse the LLM response into `reply` or `tool_calls`
- Compact prior turns when needed
- Return explicit stop reason or failure summaries

This module should not execute Electron IPC.

### `src-python/bioagent/main.py`

Responsibilities:
- Add a `--agent-chat` entry
- Accept JSON input
- Return JSON output
- Stay as the single sidecar entry point

This module should remain a CLI boundary, not the home for full agent logic.

## Electron Changes

### `electron/main.js`

Add a new IPC handler:

- `agent-chat`

This handler should:
- accept a JSON payload from the renderer
- invoke Python with `--agent-chat`
- return the JSON response to the renderer

Existing IPC handlers remain responsible for actual local actions such as analysis, history, and export.

## React Changes

### `src/components/AgentPanel.tsx`

Responsibilities:
- manage visible chat state
- own loop execution on the frontend
- call `agent-chat`
- execute returned tool calls sequentially
- append `plan` and `tool_status` messages
- stop when final reply or loop limit is reached
- preserve concise runtime metadata such as `stopReason` and per-turn `usage`

### `src/App.tsx`

Responsibilities:
- mount `AgentPanel` inside the analysis page
- pass current samples, selected sample, and source-path context
- keep the existing workflow intact

### `src/types/index.ts`

Add or update types for:
- `ChatMessage`
- `ToolCall`
- `ToolSpec`
- `ToolCategory`
- `AgentTurnResponse`
- `AgentRuntimeConfig`
- `TokenUsage`
- `StopReason`
- failure summaries used by the panel loop

## Error Handling

The first version should make failure states explicit and visible:

- If a tool fails, show a `tool_status` failure card
- Feed the failure summary back into the next agent turn
- Allow the agent to explain the failure or stop
- If the loop limit is reached, produce a visible terminal message instead of silently failing
- If `agent-chat` returns invalid JSON, render a safe assistant error message and stop the loop
- If a tool is blocked by runtime permission context, return `permission_denied` instead of letting the loop continue ambiguously

## Testing

Testing should focus on behavior, not only rendering.

### Python

- Response parsing for `reply` and `tool_calls`
- Tool registry prompt generation
- Tool filtering by category or runtime permission context
- Duplicate or invalid tool call handling
- Loop-safe behavior for malformed model output
- Explicit stop reason generation

### React

- Message normalization for `user`, `agent`, `plan`, `tool_status`
- Sequential tool execution
- Stop after final reply
- Stop after max loop count
- Failure rendering for tool errors
- Preserve and display runtime metadata when available

### Integration

- Ask a question about current samples and receive a direct answer
- Ask a question that requires sample detail lookup
- Ask a question that requires history lookup
- Ask for a rerun of analysis and see plan plus tool progress
- Ask for report export and see result summary
- Verify that a blocked action tool produces a visible denial instead of silent failure

## Non-Goals For This Version

- Direct editing of judgment results
- Automatic settings mutation
- Fully autonomous unlimited loop execution
- Multi-conversation management
- UI tab switching controlled by the agent
- Rich markdown document generation inside chat
- Plugin or MCP orchestration frameworks

## Implementation Impact

Expected main touched files:

- `src-python/bioagent/main.py`
- `src-python/bioagent/agent_chat.py`
- `src-python/bioagent/agent_tools.py`
- `electron/main.js`
- `src/App.tsx`
- `src/App.css`
- `src/components/AgentPanel.tsx`
- `src/components/AgentPanel.css`
- `src/components/ChatMessage.tsx`
- `src/components/ChatMessage.css`
- `src/types/index.ts`

## Open Decisions Resolved In This Spec

- Agent scope is controlled, not general
- Default context is current analysis first
- Agent can query history or detail on demand
- Agent can propose a short plan and execute multiple tools sequentially
- Panel placement is inside the analysis page
- Tool set is restricted to five actions
- Loop ownership sits in the frontend
- Python owns prompt construction and action parsing
- Runtime behavior is shaped by explicit loop config, stop reasons, and typed tool metadata

## Acceptance Criteria

The feature is complete for this design when all of the following are true:

1. The analysis page includes a visible `Agent Panel`.
2. A user can ask a question about current analysis results and receive an answer.
3. The agent can return a short plan and then execute multiple tools sequentially.
4. Tool execution is visible in the UI as status updates.
5. The agent can query history, sample list, sample detail, rerun analysis, and export a report.
6. The loop always terminates via final reply or max-round stop.
7. The runtime can report explicit stop reasons and per-turn usage.
8. The agent does not directly alter stored results or silently mutate settings.
