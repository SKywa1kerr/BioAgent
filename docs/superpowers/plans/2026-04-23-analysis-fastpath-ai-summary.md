# Analysis Fast Path + AI Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make explicit dataset-analysis commands render results immediately via a local fast path, then automatically append an assistant-style AI summary in the background.

**Architecture:** Split the work into three bounded slices. First, add a deterministic fast-path matcher plus regression tests in `AgentHarness`. Second, wire a background summary stage that runs after analysis without blocking the result panel. Third, teach the React hook/UI to surface the new summary lifecycle while keeping analysis results as the primary output. Keep the legacy LLM tool-routing path intact for open-ended requests.

**Tech Stack:** Electron main-process JS, React 18 + TypeScript, OpenAI-compatible chat completions, Node `node:test`, existing MCP tool bridge.

---

## File Structure

### Modified
- `electron/agent_harness.mjs` — explicit command matcher, fast-path execution, background summary generation, new emitted events
- `src/hooks/useAgentHarness.ts` — consume new summary events and keep analysis-first UI behavior
- `src/components/panels/AnalysisPanel.tsx` — optional non-blocking summary-pending indicator
- `src/i18n.ts` — zh/en strings for summary-pending and summary-failed states
- `tests/test_agent_harness.mjs` — fast-path and event-order regression coverage

### Unchanged by design
- `src-python/bioagent/tools_register.py` — continue reusing `analyze_sequences`
- `electron/main.js` — keep IPC wrapper unchanged unless a trace string is strictly needed
- `core/alignment.py` — out of scope for this latency fix

---

## Pre-flight

### Task P1: Establish baseline

**Files:**
- None

- [ ] **Step P1.1: Confirm the tree state before implementation**

Run: `git status --short`
Expected: only the spec / plan docs are new or committed; no unexpected runtime code edits.

- [ ] **Step P1.2: Run the focused harness test file**

Run: `node --test tests/test_agent_harness.mjs`
Expected: existing tests pass before new coverage is added.

- [ ] **Step P1.3: Record current behavior for the explicit-analysis path**

Run: `python run.py --dataset pro --no-llm`
Expected: local analysis succeeds; this is the backend-only baseline and should remain unchanged after UI routing changes.

---

## Section 1 — Fast-Path Detection

### Task 1: Add regression tests for explicit dataset commands

**Files:**
- Modify: `tests/test_agent_harness.mjs`

- [ ] **Step 1.1: Write a failing test proving explicit dataset commands bypass the first LLM call**

Append this test to `tests/test_agent_harness.mjs`:

```js
test("runTurn fast-paths explicit dataset analysis without first LLM routing call", async () => {
  const harness = createHarness();
  const events = [];
  let llmCalls = 0;

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async () => {
          llmCalls += 1;
          return {
            choices: [{ message: { content: "summary" } }],
          };
        },
      },
    },
  });

  harness.callMcpTool = async (toolName, args) => {
    assert.equal(toolName, "analyze_sequences");
    assert.equal(args.dataset, "pro");
    return {
      ok: true,
      analysis_id: "analysis-1",
      dataset: "pro",
      sample_count: 2,
      detail: {
        analysis_id: "analysis-1",
        dataset: "pro",
        sample_count: 2,
        samples: [{ id: "C1-1" }, { id: "C2-1" }],
      },
      samples: [{ id: "C1-1" }, { id: "C2-1" }],
    };
  };

  await harness.runTurn("分析 pro 数据集", (payload) => events.push(payload));

  assert.equal(llmCalls, 1);
  assert.ok(events.some((event) => event.type === "tool_result" && event.tool === "analyze_sequences"));
});
```

- [ ] **Step 1.2: Write a failing test proving the summary is emitted after the analysis result**

Append this second test:

```js
test("fast-path analysis emits summary_pending and summary_ready after the tool result", async () => {
  const harness = createHarness();
  const events = [];

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async () => ({
          choices: [{ message: { content: "这批样本整体较稳定，建议优先复核 C366-3。" } }],
        }),
      },
    },
  });

  harness.callMcpTool = async () => ({
    ok: true,
    analysis_id: "analysis-1",
    dataset: "pro",
    sample_count: 1,
    detail: {
      analysis_id: "analysis-1",
      dataset: "pro",
      sample_count: 1,
      samples: [{ id: "C366-3", identity: 0.99, cds_coverage: 0.51, aa_changes: [] }],
    },
    samples: [{ id: "C366-3", identity: 0.99, cds_coverage: 0.51, aa_changes: [] }],
  });

  await harness.runTurn("analyze pro dataset", (payload) => events.push(payload));

  const toolIndex = events.findIndex((event) => event.type === "tool_result" && event.tool === "analyze_sequences");
  const pendingIndex = events.findIndex((event) => event.type === "summary_pending");
  const readyIndex = events.findIndex((event) => event.type === "summary_ready");

  assert.ok(toolIndex >= 0);
  assert.ok(pendingIndex > toolIndex);
  assert.ok(readyIndex > pendingIndex);
});
```

- [ ] **Step 1.3: Write a failing guard test proving open-ended requests still use legacy routing**

Append:

```js
test("non-explicit requests still use the legacy LLM routing flow", async () => {
  const harness = createHarness();
  let llmCalls = 0;

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async () => {
          llmCalls += 1;
          return {
            choices: [{ message: { content: "需要进一步分析", tool_calls: [] } }],
          };
        },
      },
    },
  });

  await harness.runTurn("帮我看看 pro 这批数据有什么问题", () => {});

  assert.ok(llmCalls >= 1);
});
```

- [ ] **Step 1.4: Run the focused test file and confirm failure**

Run: `node --test tests/test_agent_harness.mjs`
Expected: new tests fail because the harness has no fast-path matcher or summary events yet.

- [ ] **Step 1.5: Commit the red test state only after implementation**

Do not commit yet. These tests should be committed together with the implementation in Task 2.

---

## Section 2 — Agent Harness Fast Path

### Task 2: Implement deterministic routing and background summary generation

**Files:**
- Modify: `electron/agent_harness.mjs`
- Modify: `tests/test_agent_harness.mjs`

- [ ] **Step 2.1: Add a narrow explicit-command matcher**

In `electron/agent_harness.mjs`, add these helpers near the prompt / utility section:

```js
function normalizeIntentText(input) {
  return String(input || "").trim().toLowerCase();
}

function matchDatasetAnalysisIntent(input) {
  const text = normalizeIntentText(input)
    .replace(/[，。！？,.!?]/g, " ")
    .replace(/\s+/g, " ");

  const patterns = [
    { re: /^(请\s*)?(帮我\s*)?分析\s+(base|pro|promax)\s+数据集$/, lang: "zh" },
    { re: /^(please\s+)?analyze\s+(base|pro|promax)\s+dataset$/, lang: "en" },
  ];

  for (const pattern of patterns) {
    const m = text.match(pattern.re);
    if (!m) continue;
    const dataset = m[m.length - 1];
    return { dataset };
  }
  return null;
}
```

This matcher must stay intentionally narrow. Do not match free-form questions.

- [ ] **Step 2.2: Add a summary prompt builder for assistant-style explanations**

Add a helper like this:

```js
function buildAnalysisSummaryPrompt(result) {
  const samples = Array.isArray(result?.samples)
    ? result.samples
    : Array.isArray(result?.detail?.samples)
      ? result.detail.samples
      : [];

  const compactSamples = samples.slice(0, 12).map((sample) => ({
    id: sample.id || sample.sid,
    identity: sample.identity,
    cds_coverage: sample.cds_coverage ?? sample.coverage,
    frameshift: Boolean(sample.frameshift),
    aa_changes: sample.aa_changes || [],
    other_read_issues: sample.other_read_issues || [],
    status: sample.status || sample.bucket,
  }));

  return [
    "你是 BioAgent 的实验助手，请用简洁、自然、偏助手讲解型的中文总结这次分析结果。",
    "要求：",
    "1. 先说整体情况。",
    "2. 点出最值得关注的样本。",
    "3. 用 coverage / frameshift / aa_changes 等证据解释原因。",
    "4. 最后给出下一步建议。",
    "5. 不要复述原始 JSON，不要写表格。",
    "",
    JSON.stringify({
      dataset: result?.dataset,
      sample_count: result?.sample_count,
      samples: compactSamples,
    }),
  ].join("\n");
}
```

- [ ] **Step 2.3: Add a dedicated summary request helper**

Still in `electron/agent_harness.mjs`, add a helper that makes one direct completion call without tool routing:

```js
async function createAnalysisSummary(client, model, timeout, result) {
  const response = await withRetry(
    () => client.chat.completions.create({
      model,
      temperature: 0.2,
      max_tokens: 500,
      timeout,
      messages: [{ role: "user", content: buildAnalysisSummaryPrompt(result) }],
    }),
  );
  return response?.choices?.[0]?.message?.content || "";
}
```

- [ ] **Step 2.4: Implement a fast-path execution branch inside `runTurn`**

At the top of `runTurn`, after the busy check and before the main `for (let turn...)` loop, add:

```js
const fastPath = matchDatasetAnalysisIntent(userMessage);
if (fastPath) {
  await this.runExplicitDatasetAnalysis(fastPath, onEvent);
  return;
}
```

Then add `runExplicitDatasetAnalysis` as a class method. It should:

1. emit `thinking`
2. emit `tool_calls_start`
3. emit `tool_call` for `analyze_sequences`
4. call `this.callMcpTool("analyze_sequences", { dataset, no_llm: true })`
5. if result is ok and has `analysis_id` but no inline `samples`, fetch `get_analysis_detail`
6. emit `tool_result` immediately after the analysis payload is ready enough for the UI
7. emit `summary_pending`
8. call `createAnalysisSummary(...)`
9. emit `summary_ready` with the returned text
10. append the final assistant summary to `this.messages`

Use this structure:

```js
async runExplicitDatasetAnalysis({ dataset }, onEvent) {
  const client = this.getClient();
  onEvent({ type: "thinking" });
  onEvent({ type: "tool_calls_start", message: "Running tool steps..." });
  onEvent({ type: "tool_call", tool: "analyze_sequences", args: { dataset, no_llm: true } });

  const result = await this.callMcpTool("analyze_sequences", { dataset, no_llm: true });
  if (!result?.ok) {
    const message = result?.error || `Analysis failed for dataset ${dataset}`;
    onEvent({ type: "error", message });
    onEvent({ type: "reply", content: `Run failed: ${message}`, uiAction: "show_text" });
    return;
  }

  let hydrated = result;
  if ((!Array.isArray(result.samples) || result.samples.length === 0) && result.analysis_id) {
    const detail = await this.callMcpTool("get_analysis_detail", { analysis_id: result.analysis_id });
    if (detail?.ok) {
      hydrated = { ...result, detail, samples: detail.samples || result.samples };
    }
  }

  onEvent({ type: "tool_result", tool: "analyze_sequences", result: hydrated });
  onEvent({ type: "summary_pending", dataset, analysis_id: hydrated.analysis_id });

  const summary = await createAnalysisSummary(
    client,
    this.settings.llmModel || DEFAULT_MODEL,
    this.settings.llmTimeoutMs || LLM_TIMEOUT_MS,
    hydrated,
  );

  if (summary) {
    this.messages.push({ role: "assistant", content: summary });
    onEvent({ type: "summary_ready", content: summary, dataset, analysis_id: hydrated.analysis_id });
  }
}
```

- [ ] **Step 2.5: Keep the legacy routing path untouched for non-explicit inputs**

Do not change the existing tool-routing loop except to leave it behind the new early-return branch. This preserves existing AI behavior for open-ended or multi-step requests.

- [ ] **Step 2.6: Run the focused harness tests**

Run: `node --test tests/test_agent_harness.mjs`
Expected: all harness tests pass, including the three new fast-path tests.

- [ ] **Step 2.7: Commit the harness slice**

```bash
git add electron/agent_harness.mjs tests/test_agent_harness.mjs
git commit -m "feat(agent): fast-path explicit dataset analysis with async AI summary"
```

---

## Section 3 — Frontend Event Consumption

### Task 3: Teach the React hook/UI to display summary lifecycle without blocking results

**Files:**
- Modify: `src/hooks/useAgentHarness.ts`
- Modify: `src/components/panels/AnalysisPanel.tsx`
- Modify: `src/i18n.ts`

- [ ] **Step 3.1: Extend the `AgentEvent` union for summary events**

In `src/hooks/useAgentHarness.ts`, extend the union with:

```ts
| { type: "summary_pending"; dataset?: string; analysis_id?: string }
| { type: "summary_ready"; content?: string; dataset?: string; analysis_id?: string }
```

- [ ] **Step 3.2: Add i18n strings for pending/failure states**

Append to `src/i18n.ts`:

```ts
// zh
"app.summary.pending": "助手正在生成本次分析解读...",
"app.summary.failed": "助手解读生成失败，但分析结果已就绪。",

// en
"app.summary.pending": "Assistant summary is being generated...",
"app.summary.failed": "Assistant summary failed, but the analysis result is ready.",
```

- [ ] **Step 3.3: Update `applyAgentEvent` to surface pending/ready summary messages**

In `useAgentHarness.ts`, inside `applyAgentEvent(payload)`:

1. On `summary_pending`, call `updateAnalysisPayload(...)` to set a lightweight marker such as `__summaryPending: true`
2. On `summary_ready`, clear `__summaryPending` and push the assistant summary into chat if it is not already pushed by another path

Use this pattern:

```ts
if (payload.type === "summary_pending") {
  setProgressState("reply", 100, t(lang, "app.progress.completed"));
  if (payload.analysis_id) {
    updateAnalysisPayload(payload.analysis_id, { __summaryPending: true });
  }
  pushAssistant(t(lang, "app.summary.pending"));
  return;
}

if (payload.type === "summary_ready") {
  if (payload.analysis_id) {
    updateAnalysisPayload(payload.analysis_id, { __summaryPending: false });
  }
  if (payload.content) {
    pushAssistant(payload.content);
  }
  return;
}
```

Important: do not replace or delay the analysis panel when these events arrive.

- [ ] **Step 3.4: Surface the pending state in `AnalysisPanel.tsx` without blocking rendering**

In `AnalysisPanel.tsx`, read:

```ts
const summaryPending = Boolean(result?.__summaryPending);
```

If `samples.length > 0`, keep rendering `<ResultsWorkbench ... />` immediately. Above it, optionally render a small helper line:

```tsx
{summaryPending ? (
  <div className="analysis-helper-line">{t(language, "app.summary.pending")}</div>
) : null}
```

This indicator must be additive only. The workbench should still render while the summary is pending.

- [ ] **Step 3.5: Run focused build and tests**

Run: `node --test tests/test_agent_harness.mjs`
Expected: still green.

Run: `npm run build`
Expected: TypeScript + Vite build passes.

- [ ] **Step 3.6: Commit the UI slice**

```bash
git add src/hooks/useAgentHarness.ts src/components/panels/AnalysisPanel.tsx src/i18n.ts
git commit -m "feat(ui): show async assistant summary without blocking analysis results"
```

---

## Section 4 — Verification

### Task 4: End-to-end verification for the new behavior

**Files:**
- None

- [ ] **Step 4.1: Run the focused JS test suite**

Run: `node --test tests/test_agent_harness.mjs`
Expected: pass.

- [ ] **Step 4.2: Run the broader JS suite**

Run: `npm run test:js`
Expected: all JS tests pass.

- [ ] **Step 4.3: Run the production build**

Run: `npm run build`
Expected: passes without new TypeScript or bundling errors.

- [ ] **Step 4.4: Manual smoke test the explicit fast path**

Run: `npm run electron:dev`

Manual check:
- Enter `分析 pro 数据集`
- Confirm the result workbench appears before the assistant summary finishes
- Confirm a temporary “助手正在生成本次分析解读...” message appears
- Confirm the final assistant summary is appended automatically

- [ ] **Step 4.5: Manual smoke test the legacy path**

In the same session:
- Enter `帮我看看 pro 这批数据有什么问题`
- Confirm the app still uses the normal AI route for a more open-ended question

- [ ] **Step 4.6: Verification-before-completion**

Use superpowers:verification-before-completion before claiming the fix is complete. No completion claim without fresh command output.

---

## Notes

- Keep the matcher conservative. Expanding it too aggressively will turn free-form questions into rule-routed commands and weaken the AI UX.
- If `summary_ready` fails due to network/model issues, prefer emitting an error-level assistant message and leaving the analysis panel untouched.
- Do not reintroduce a second LLM round-trip before showing the analysis workbench.
