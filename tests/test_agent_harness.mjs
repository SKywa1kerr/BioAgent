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
