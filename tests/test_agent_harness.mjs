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
  assert.deepEqual(events.find((event) => event.type === "tool_call"), {
    type: "tool_call",
    tool: "analyze_sequences",
    args: { dataset: "pro", no_llm: true },
  });
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
  const pendingEvent = events[pendingIndex];
  const readyEvent = events[readyIndex];

  assert.ok(toolIndex >= 0);
  assert.ok(pendingIndex > toolIndex);
  assert.ok(readyIndex > pendingIndex);
  assert.equal(pendingEvent.dataset, "pro");
  assert.equal(pendingEvent.analysis_id, "analysis-1");
  assert.equal(readyEvent.dataset, "pro");
  assert.equal(readyEvent.analysis_id, "analysis-1");
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

test("fast-path analyze_sequences failure short-circuits before tool_result and summary", async () => {
  const harness = createHarness();
  const events = [];
  let llmCalls = 0;

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async () => {
          llmCalls += 1;
          return {
            choices: [{ message: { content: "should not happen" } }],
          };
        },
      },
    },
  });

  harness.callMcpTool = async (toolName, args) => {
    assert.equal(toolName, "analyze_sequences");
    assert.deepEqual(args, { dataset: "pro", no_llm: true });
    return { ok: false, error: "analysis failed" };
  };

  await harness.runTurn("analyze pro dataset", (payload) => events.push(payload));

  assert.equal(llmCalls, 0);
  assert.ok(events.some((event) => event.type === "error" && event.message === "analysis failed"));
  assert.ok(events.some((event) => event.type === "reply" && event.content === "Run failed: analysis failed"));
  assert.ok(!events.some((event) => event.type === "tool_result"));
  assert.ok(!events.some((event) => event.type === "summary_pending"));
  assert.ok(!events.some((event) => event.type === "summary_ready"));
});

test("summary generation failure is non-fatal after fast-path tool_result", async () => {
  const harness = createHarness();
  const events = [];
  let llmCalls = 0;

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async () => {
          llmCalls += 1;
          throw new Error("summary service unavailable");
        },
      },
    },
  });

  harness.callMcpTool = async () => ({
    ok: true,
    analysis_id: "analysis-2",
    dataset: "base",
    sample_count: 1,
    samples: [{ id: "B1-1" }],
  });

  await harness.runTurn("analyze base dataset", (payload) => events.push(payload));

  const toolIndex = events.findIndex((event) => event.type === "tool_result" && event.tool === "analyze_sequences");
  const pendingIndex = events.findIndex((event) => event.type === "summary_pending");
  const failedIndex = events.findIndex((event) => event.type === "summary_failed");

  assert.equal(llmCalls, 1);
  assert.ok(toolIndex >= 0);
  assert.ok(pendingIndex > toolIndex);
  assert.ok(failedIndex > pendingIndex);
  assert.equal(events[failedIndex].dataset, "base");
  assert.equal(events[failedIndex].analysis_id, "analysis-2");
  assert.equal(events[failedIndex].message, "summary service unavailable");
  assert.ok(!events.some((event) => event.type === "error"));
  assert.ok(!events.some((event) => event.type === "reply"));
});

test("fast-path hydrates analysis detail when analyze_sequences omits inline samples", async () => {
  const harness = createHarness();
  const events = [];
  const toolCalls = [];

  harness.getClient = () => ({
    chat: {
      completions: {
        create: async () => ({
          choices: [{ message: { content: "hydrated summary" } }],
        }),
      },
    },
  });

  harness.callMcpTool = async (toolName, args) => {
    toolCalls.push({ toolName, args });
    if (toolName === "analyze_sequences") {
      return {
        ok: true,
        analysis_id: "analysis-3",
        dataset: "promax",
        sample_count: 1,
      };
    }
    if (toolName === "get_analysis_detail") {
      assert.deepEqual(args, { analysis_id: "analysis-3" });
      return {
        ok: true,
        analysis_id: "analysis-3",
        dataset: "promax",
        sample_count: 1,
        samples: [{ id: "P9-1", identity: 0.98 }],
      };
    }
    throw new Error(`Unexpected tool ${toolName}`);
  };

  await harness.runTurn("分析 promax 数据集", (payload) => events.push(payload));

  assert.deepEqual(toolCalls, [
    { toolName: "analyze_sequences", args: { dataset: "promax", no_llm: true } },
    { toolName: "get_analysis_detail", args: { analysis_id: "analysis-3" } },
  ]);
  const toolResult = events.find((event) => event.type === "tool_result" && event.tool === "analyze_sequences");
  assert.deepEqual(toolResult.result.samples, [{ id: "P9-1", identity: 0.98 }]);
  assert.equal(toolResult.result.detail.analysis_id, "analysis-3");
});
