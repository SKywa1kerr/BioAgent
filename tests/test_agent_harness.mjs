import test from "node:test";
import assert from "node:assert/strict";

import { AgentHarness } from "../electron/agent_harness.mjs";

function createHarness() {
  return new AgentHarness({
    llmApiKey: "",
    llmBaseUrl: "",
    llmModel: "",
    pythonConfig: {
      cmd: "python",
      baseArgs: ["-m", "bioagent.main"],
      cwd: process.cwd(),
      env: process.env,
    },
  });
}

test("_mcpCall rejects immediately when MCP emits an error event", async () => {
  const harness = createHarness();
  harness.mcpProcess = {
    stdin: { write() {} },
  };

  const pending = harness._mcpCall("initialize", {});
  process.nextTick(() => {
    harness.emit("mcp-response", { error: "spawn EPERM" });
  });

  const result = await Promise.race([
    pending.then(
      () => "resolved",
      (error) => ({ kind: "rejected", message: String(error?.message ?? error) }),
    ),
    new Promise((resolve) => setTimeout(() => resolve("timeout"), 250)),
  ]);

  assert.notEqual(result, "timeout");
  assert.equal(result.kind, "rejected");
  assert.match(result.message, /spawn EPERM/i);
});

test("runTurn emits busy event instead of silently returning when another run is in progress", async () => {
  const harness = createHarness();

  let resolveCreate;
  const pendingResponse = new Promise((resolve) => {
    resolveCreate = resolve;
  });

  harness.getClient = () => ({
    chat: {
      completions: {
        create: () => pendingResponse,
      },
    },
  });

  const firstEvents = [];
  const secondEvents = [];

  const firstRun = harness.runTurn("first message", (payload) => firstEvents.push(payload));
  await new Promise((resolve) => setImmediate(resolve));

  await harness.runTurn("second message", (payload) => secondEvents.push(payload));

  assert.ok(firstEvents.some((event) => event.type === "thinking"));
  assert.ok(secondEvents.some((event) => event.type === "busy"));

  resolveCreate({
    choices: [{ message: { content: "done" } }],
  });
  await firstRun;
});

test("runTurn sends timeout to LLM call to avoid indefinite waiting", async () => {
  const harness = createHarness();
  let capturedRequest = null;

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async (request) => {
          capturedRequest = request;
          return {
            choices: [{ message: { content: "ok" } }],
          };
        },
      },
    },
  });

  await harness.runTurn("hello", () => {});

  assert.equal(capturedRequest.timeout, 45000);
});

test("runTurn emits fallback error and reply when max turns are exhausted without final answer", async () => {
  const harness = createHarness();
  const events = [];

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async () => ({
          choices: [
            {
              message: {
                content: "",
                tool_calls: [
                  {
                    id: "call-1",
                    type: "function",
                    function: {
                      name: "analyze_sequences",
                      arguments: "{}",
                    },
                  },
                ],
              },
            },
          ],
        }),
      },
    },
  });

  harness.callMcpTool = async () => ({ ok: true, data: {} });

  await harness.runTurn("hello", (payload) => events.push(payload));

  assert.ok(events.some((event) => event.type === "error"));
  assert.ok(events.some((event) => event.type === "reply"));
});

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
