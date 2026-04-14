import { OpenAI } from "openai";
import { spawn } from "child_process";
import { EventEmitter } from "events";

const DEFAULT_MODEL = "deepseek-chat";
const MAX_TURNS = 5;
const MAX_TOOL_CALLS_PER_TURN = 3;

export class AgentHarness extends EventEmitter {
  constructor(settings) {
    super();
    this.settings = settings;
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

    this.mcpProcess.stderr.on("data", () => {});
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
      const id = `req-${++this.mcpRequestId}`;
      const handler = (msg) => {
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
    return `你是 Ultimate BioAgent，一个可以理解中文指令的生物信息学智能体。

你可以调用这些工具：
- analyze_sequences：先做基础分析
- detect_mutation_trends：分析突变热点和规律
- generate_lab_suggestions：给出实验改进建议
- query_history：查看历史分析
- get_analysis_detail：查看单次分析详情

工作原则：
1. 用户让你“分析数据/看看结果”时，优先先调用 analyze_sequences。
2. 用户问“有没有共同规律/热点/趋势”时，调用 detect_mutation_trends。
3. 用户问“下一步怎么做/实验建议/如何优化”时，调用 generate_lab_suggestions。
4. 如果一个请求同时要求分析、趋势、建议，你应该按顺序自主调用多个工具完成任务。
5. 始终使用中文简洁回复。
6. 删除、覆盖、外发等危险操作必须先确认。

输出要求：
- 有工具可用时优先调用工具，不要空谈。
- 完成后给出总结性中文答复。`;
  }

  getClient() {
    if (!this.settings.llmApiKey) {
      throw new Error("LLM API Key 未配置，请先在界面中填写。");
    }
    return new OpenAI({
      apiKey: this.settings.llmApiKey,
      baseURL: this.settings.llmBaseUrl,
    });
  }

  inferUiAction(content) {
    if (content.includes("趋势") || content.includes("热点") || content.includes("规律")) return "show_trends";
    if (content.includes("建议") || content.includes("优化") || content.includes("实验")) return "show_suggestions";
    if (content.includes("分析") || content.includes("结果") || content.includes("样本")) return "show_analysis";
    return "show_text";
  }

  async runTurn(userMessage, onEvent) {
    if (this.isRunning) return;
    this.isRunning = true;
    this.messages.push({ role: "user", content: userMessage });

    try {
      for (let turn = 0; turn < MAX_TURNS; turn += 1) {
        onEvent({ type: "thinking" });

        const client = this.getClient();

        const response = await client.chat.completions.create({
          model: this.settings.llmModel || DEFAULT_MODEL,
          temperature: 0,
          max_tokens: 1200,
          messages: [
            { role: "system", content: this.buildSystemPrompt() },
            ...this.messages.slice(-10),
          ],
          tools: this.mcpTools.map((tool) => ({
            type: "function",
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.parameters,
            },
          })),
        });

        const message = response.choices[0].message;
        if (message.tool_calls && message.tool_calls.length > 0) {
          const toolCalls = message.tool_calls.slice(0, MAX_TOOL_CALLS_PER_TURN);
          this.messages.push({ role: "assistant", content: message.content || "", tool_calls: toolCalls });
          onEvent({ type: "tool_calls_start", message: "正在执行分析步骤..." });

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
            onEvent({ type: "tool_result", tool: toolName, result });
            this.messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: JSON.stringify(result),
            });
          }
          continue;
        }

        const content = message.content || "分析完成。";
        this.messages.push({ role: "assistant", content });
        const uiAction = this.inferUiAction(content);
        onEvent({ type: "reply", content, uiAction });
        break;
      }
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
