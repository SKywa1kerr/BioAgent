import { OpenAI } from "openai";
import { spawn } from "child_process";
import { EventEmitter } from "events";

const DEFAULT_MODEL = "deepseek-chat";
const MAX_TURNS = 3;
const MAX_TOOL_CALLS_PER_TURN = 2;
const LLM_TIMEOUT_MS = 45000;
const MAX_HISTORY_NON_TOOL_MESSAGES = 6;
const MAX_PROMPT_MESSAGES = 8;
const MAX_TOOL_CONTENT_CHARS = 1800;

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

export class AgentHarness extends EventEmitter {
  constructor(settings) {
    super();
    this.settings = {
      ...settings,
      llmTimeoutMs: settings?.llmTimeoutMs ?? LLM_TIMEOUT_MS,
    };
    this.messages = [];
    this.mcpProcess = null;
    this.mcpRequestId = 0;
    this.mcpTools = [];
    this.isRunning = false;
  }

  async initMcpServer() {
    if (this.mcpProcess) {
      this.shutdown();
    }

    const { cmd, baseArgs, cwd, env } = this.settings.pythonConfig;
    const args = [...baseArgs, "--mcp-server"];
    this.mcpProcess = spawn(cmd, args, { cwd, env, windowsHide: true });

    let buffer = "";
    this.mcpProcess.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          this.emit("mcp-response", msg);
        } catch (error) {
          this.emit("mcp-response", { error: String(error) });
        }
      }
    });

    this.mcpProcess.stderr.on("data", (chunk) => {
      const text = chunk.toString().trim();
      if (text) {
        console.error("[MCP stderr]", text);
        this.emit("mcp-stderr", text);
      }
    });
    this.mcpProcess.on("error", (error) => {
      this.emit("mcp-response", { error: String(error) });
    });
    this.mcpProcess.on("exit", (code) => {
      this.emit("mcp-response", { error: `MCP process exited: ${code ?? "unknown"}` });
    });

    await this._mcpCall("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "ultimate-bioagent", version: "0.1.0" },
    });
    const toolsResp = await this._mcpCall("tools/list", {});
    this.mcpTools = toolsResp?.result?.tools || [];
    return this.mcpTools;
  }

  async _mcpCall(method, params) {
    return new Promise((resolve, reject) => {
      if (!this.mcpProcess?.stdin) {
        reject(new Error(`MCP process is not ready: ${method}`));
        return;
      }
      const id = `req-${++this.mcpRequestId}`;
      const handler = (msg) => {
        if (msg?.error) {
          clearTimeout(timeoutId);
          this.removeListener("mcp-response", handler);
          const message = typeof msg.error === "string" ? msg.error : msg.error.message || JSON.stringify(msg.error);
          reject(new Error(message));
          return;
        }
        if (msg?.id !== id) return;
        clearTimeout(timeoutId);
        this.removeListener("mcp-response", handler);
        resolve(msg);
      };
      this.on("mcp-response", handler);
      this.mcpProcess.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
      const timeoutId = setTimeout(() => {
        this.removeListener("mcp-response", handler);
        reject(new Error(`MCP timeout: ${method}`));
      }, 30000);
    });
  }

  async callMcpTool(toolName, args) {
    const response = await this._mcpCall("tools/call", { name: toolName, arguments: args || {} });
    return response?.result || { ok: false, error: "No MCP result" };
  }

  buildSystemPrompt() {
    return [
      "You are Ultimate BioAgent.",
      "Always answer briefly and clearly.",
      "Use tools when useful:",
      "- analyze_sequences: run baseline sequence analysis",
      "- detect_mutation_trends: detect hotspot and trend patterns",
      "- generate_lab_suggestions: suggest experiment improvements",
      "- query_history: check prior analyses (only when user asks for history)",
      "- get_analysis_detail: fetch detail for a specific analysis",
      "When user asks to analyze a dataset, call analyze_sequences first and avoid extra tool calls.",
      "Do not call query_history unless explicitly requested.",
      "If a request includes analysis + trends + suggestions, call multiple tools in sequence.",
      "Risky actions must require user confirmation first.",
    ].join("\n");
  }

  getClient() {
    if (!this.settings.llmApiKey) {
      throw new Error("LLM API key is missing. Please configure it in the UI.");
    }
    return new OpenAI({
      apiKey: this.settings.llmApiKey,
      baseURL: this.settings.llmBaseUrl,
    });
  }

  inferUiAction(content) {
    const text = (content || "").toLowerCase();
    if (text.includes("trend") || text.includes("hotspot")) return "show_trends";
    if (text.includes("suggest") || text.includes("experiment")) return "show_suggestions";
    if (text.includes("analysis") || text.includes("result")) return "show_analysis";
    return "show_text";
  }

  compactHistoryForNewTurn() {
    this.messages = this.messages
      .filter((msg) => msg && typeof msg === "object" && msg.role !== "tool")
      .slice(-MAX_HISTORY_NON_TOOL_MESSAGES)
      .map((msg) => ({ role: msg.role, content: msg.content || "" }));
  }

  summarizeToolResult(toolName, result) {
    if (!result || typeof result !== "object") {
      return String(result || "");
    }

    if (toolName === "analyze_sequences") {
      const sampleCount = typeof result.sample_count === "number" ? result.sample_count : undefined;
      const inlineSamples = Array.isArray(result.samples) ? result.samples.length : 0;
      const detailSamples = Array.isArray(result?.detail?.samples) ? result.detail.samples.length : 0;
      return JSON.stringify({
        ok: result.ok,
        analysis_id: result.analysis_id,
        dataset: result.dataset,
        sample_count: sampleCount,
        samples_inlined: inlineSamples || detailSamples,
        detail_loaded: Boolean(result.detail),
        detail_error: result.detail_error,
        used_llm: result.used_llm,
      });
    }

    if (toolName === "get_analysis_detail") {
      return JSON.stringify({
        ok: result.ok,
        analysis_id: result.analysis_id,
        dataset: result.dataset,
        sample_count: result.sample_count,
        has_samples: Array.isArray(result.samples),
        sample_len: Array.isArray(result.samples) ? result.samples.length : 0,
      });
    }

    let text = "";
    try {
      text = JSON.stringify(result);
    } catch {
      text = String(result);
    }

    if (text.length > MAX_TOOL_CONTENT_CHARS) {
      return `${text.slice(0, MAX_TOOL_CONTENT_CHARS)} ...[truncated]`;
    }
    return text;
  }

  async runTurn(userMessage, onEvent) {
    if (this.isRunning) {
      onEvent({ type: "busy", message: "Agent is already processing another request. Please wait." });
      return;
    }

    this.isRunning = true;
    this.compactHistoryForNewTurn();
    this.messages.push({ role: "user", content: userMessage });

    let replied = false;

    try {
      for (let turn = 0; turn < MAX_TURNS; turn += 1) {
        onEvent({ type: "thinking" });

        const client = this.getClient();

        const response = await withRetry(
          () => client.chat.completions.create({
            model: this.settings.llmModel || DEFAULT_MODEL,
            temperature: 0,
            max_tokens: this.settings.maxTokens || 2400,
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

        const message = response.choices[0].message;
        if (message.tool_calls && message.tool_calls.length > 0) {
          const toolCalls = message.tool_calls.slice(0, MAX_TOOL_CALLS_PER_TURN);
          this.messages.push({ role: "assistant", content: message.content || "", tool_calls: toolCalls });
          onEvent({ type: "tool_calls_start", message: "Running tool steps..." });

          for (const toolCall of toolCalls) {
            const toolName = toolCall.function.name;
            let parsedArgs = {};
            try {
              parsedArgs = JSON.parse(toolCall.function.arguments || "{}");
            } catch {
              parsedArgs = {};
            }
            onEvent({ type: "tool_call", tool: toolName, args: parsedArgs });
            const result = await this.callMcpTool(toolName, parsedArgs);
            if (toolName === "analyze_sequences" && result && result.ok && result.analysis_id) {
              try {
                const detail = await this.callMcpTool("get_analysis_detail", { analysis_id: result.analysis_id });
                if (detail && detail.ok) {
                  result.detail = detail;
                  if (Array.isArray(detail.samples)) {
                    result.samples = detail.samples;
                  }
                } else {
                  result.detail_error = (detail && detail.error) || "Failed to load analysis detail";
                }
              } catch (error) {
                result.detail_error = error instanceof Error ? error.message : String(error);
              }
            }
            onEvent({ type: "tool_result", tool: toolName, result });
            this.messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: this.summarizeToolResult(toolName, result),
            });
          }
          continue;
        }

        const content = message.content || "Analysis completed.";
        this.messages.push({ role: "assistant", content });
        const uiAction = this.inferUiAction(content);
        onEvent({ type: "reply", content, uiAction });
        replied = true;
        break;
      }

      if (!replied) {
        const message = `No final reply after ${MAX_TURNS} turns. Retry or switch model.`;
        const content = `Run failed: ${message}`;
        this.messages.push({ role: "assistant", content });
        onEvent({ type: "error", message });
        onEvent({ type: "reply", content, uiAction: "show_text" });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (/contextwindow|context length|maximum context|input_tokens/i.test(message)) {
        this.compactHistoryForNewTurn();
      }
      const content = `Run failed: ${message}`;
      this.messages.push({ role: "assistant", content });
      onEvent({ type: "error", message });
      onEvent({ type: "reply", content, uiAction: "show_text" });
    } finally {
      this.isRunning = false;
    }
  }

  shutdown() {
    if (this.mcpProcess) {
      this.mcpProcess.removeAllListeners();
      this.mcpProcess.kill();
      this.mcpProcess = null;
    }
    this.mcpTools = [];
    this.messages = this.messages.slice(-10);
  }
}
