import { FormEvent, KeyboardEvent, useState } from "react";
import type {
  AgentContext,
  AgentSampleSummary,
  AnalysisContextUpdate,
  AgentRuntimeConfig,
  AgentTurnResponse,
  ChatMessage,
  Sample,
  ToolCall,
  ToolResult,
} from "../types";
import { ChatMessageCard } from "./ChatMessage";
import { t } from "../i18n";
import { getDefaultSelectedSampleId } from "../utils/resultSelection";
import "./AgentPanel.css";
import type { AppLanguage } from "../types";
const { invoke } = window.electronAPI;
const VALID_TOOL_NAMES = new Set<ToolCall["tool"]>([
  "query_samples",
  "query_history",
  "get_sample_detail",
  "run_analysis",
  "export_report",
]);

const DEFAULT_RUNTIME: AgentRuntimeConfig = {
  maxRounds: 3,
  maxToolCallsPerTurn: 3,
  maxRecentMessages: 12,
  allowActionTools: true,
  includeUsage: true,
};

interface AgentPanelProps {
  language: AppLanguage;
  samples: Sample[];
  selectedSampleId: string | null;
  sourcePath?: string | null;
  genesDir?: string | null;
  plasmid?: string;
  onAnalysisComplete?: (nextAnalysis: AnalysisContextUpdate) => void;
}

type MessageDraft =
  | Omit<Extract<ChatMessage, { type: "user" }>, "id" | "timestamp">
  | Omit<Extract<ChatMessage, { type: "agent" }>, "id" | "timestamp">
  | Omit<Extract<ChatMessage, { type: "plan" }>, "id" | "timestamp">
  | Omit<Extract<ChatMessage, { type: "tool_status" }>, "id" | "timestamp">;

interface CurrentAnalysisSnapshot {
  sourcePath?: string;
  genesDir?: string;
  plasmid: string;
  samples: Sample[];
  selectedSampleId: string | null;
}

function createMessageId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function createMessage(message: MessageDraft): ChatMessage {
  return {
    ...message,
    id: createMessageId(message.type),
    timestamp: Date.now(),
  } as ChatMessage;
}

function asString(value: unknown) {
  return typeof value === "string" ? value : "";
}

function asBoolean(value: unknown) {
  return typeof value === "boolean" ? value : false;
}

function asNumber(value: unknown) {
  return typeof value === "number" ? value : undefined;
}

function summarizeArgs(language: AppLanguage, args: Record<string, unknown>) {
  const pairs = Object.entries(args).filter(
    ([, value]) => value !== undefined && value !== null && value !== ""
  );
  if (pairs.length === 0) {
    return t(language, "agent.noArguments");
  }

  return pairs
    .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(", ");
}

function translateTemplate(language: AppLanguage, key: string, values: Record<string, string>) {
  let template = t(language, key);
  for (const [name, value] of Object.entries(values)) {
    template = template.split(`{${name}}`).join(value);
  }
  return template;
}

function summarizeSampleForAgent(sample: Sample): AgentSampleSummary {
  return {
    id: sample.id,
    clone: sample.clone,
    status: sample.status,
    reason: sample.reason,
    mutationCount: Array.isArray(sample.mutations) ? sample.mutations.length : 0,
    error: sample.error,
  };
}

function isToolCall(value: unknown): value is ToolCall {
  if (!value || typeof value !== "object") {
    return false;
  }

  const maybeCall = value as Partial<ToolCall>;
  return (
    typeof maybeCall.tool === "string" &&
    VALID_TOOL_NAMES.has(maybeCall.tool as ToolCall["tool"]) &&
    !!maybeCall.args &&
    typeof maybeCall.args === "object" &&
    !Array.isArray(maybeCall.args)
  );
}

function parseAgentTurnResponse(raw: unknown): AgentTurnResponse {
  const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
  if (!parsed || typeof parsed !== "object" || !("action" in parsed)) {
    throw new Error("invalid_model_output");
  }

  const response = parsed as Partial<AgentTurnResponse>;

  if (response.action === "reply" && typeof response.content === "string") {
    return response as AgentTurnResponse;
  }

  if (
    response.action === "tool_calls" &&
    typeof response.message === "string" &&
    Array.isArray(response.calls) &&
    response.calls.every(isToolCall)
  ) {
    return response as AgentTurnResponse;
  }

  throw new Error("invalid_model_output");
}

export function AgentPanel({
  language,
  samples,
  selectedSampleId,
  sourcePath,
  genesDir,
  plasmid = "pet22b",
  onAnalysisComplete,
}: AgentPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const batchLabel = sourcePath ? sourcePath.split(/[\\/]/).filter(Boolean).at(-1) : null;

  const buildContext = (
    currentAnalysis: CurrentAnalysisSnapshot,
    history: ChatMessage[],
    recentToolResults: ToolResult[]
  ): AgentContext => ({
    currentAnalysis: {
      sourcePath: currentAnalysis.sourcePath,
      samples: currentAnalysis.samples.map(summarizeSampleForAgent),
      selectedSampleId: currentAnalysis.selectedSampleId,
    },
    recentToolResults,
    history: history.slice(-DEFAULT_RUNTIME.maxRecentMessages),
    runtime: DEFAULT_RUNTIME,
  });

  const setToolMessageStatus = (
    history: ChatMessage[],
    messageId: string,
    nextContent: string,
    status: "running" | "done" | "failed"
  ) =>
    history.map((message) => {
      if (message.id !== messageId || message.type !== "tool_status") {
        return message;
      }

      return {
        ...message,
        content: nextContent,
        status,
      };
    });

  const executeToolCall = async (
    call: ToolCall,
    currentAnalysis: CurrentAnalysisSnapshot
  ): Promise<ToolResult> => {
    switch (call.tool) {
      case "query_samples": {
        const status = asString(call.args.status);
        const sampleId = asString(call.args.sampleId);

        const filtered = currentAnalysis.samples.filter((sample) => {
          if (status && sample.status !== status) {
            return false;
          }
          if (sampleId && sample.id !== sampleId) {
            return false;
          }
          return true;
        });

        return {
          tool: call.tool,
          ok: true,
          summary: `Found ${filtered.length} sample${filtered.length === 1 ? "" : "s"}.`,
          data: filtered.map((sample) => ({
            id: sample.id,
            clone: sample.clone,
            status: sample.status,
            reason: sample.reason,
            identity: sample.identity,
            coverage: sample.coverage,
            mutationCount: Array.isArray(sample.mutations) ? sample.mutations.length : 0,
          })),
        };
      }
      case "get_sample_detail": {
        const sampleId = asString(call.args.sampleId);
        const sample = currentAnalysis.samples.find((item) => item.id === sampleId);

        if (!sample) {
          return {
            tool: call.tool,
            ok: false,
            summary: sampleId ? `Sample ${sampleId} was not found.` : "No sampleId was provided.",
          };
        }

        return {
          tool: call.tool,
          ok: true,
          summary: `Loaded detail for ${sample.id}.`,
          data: {
            id: sample.id,
            name: sample.name,
            clone: sample.clone,
            status: sample.status,
            reason: sample.reason,
            identity: sample.identity,
            coverage: sample.coverage,
            frameshift: sample.frameshift,
            llmVerdict: sample.llm_verdict,
            mutations: Array.isArray(sample.mutations) ? sample.mutations : [],
          },
        };
      }
      case "query_history": {
        const raw = (await invoke("get-history")) as string;
        const records = JSON.parse(raw) as Array<Record<string, unknown>>;
        const limit = asNumber(call.args.limit);
        const sliced = typeof limit === "number" ? records.slice(0, limit) : records;

        return {
          tool: call.tool,
          ok: true,
          summary: `Loaded ${sliced.length} historical run${sliced.length === 1 ? "" : "s"}.`,
          data: sliced,
        };
      }
      case "run_analysis": {
        const ab1Dir = asString(call.args.ab1Dir) || currentAnalysis.sourcePath || "";
        const nextGenesDir = asString(call.args.genesDir) || currentAnalysis.genesDir || undefined;
        const nextPlasmid = asString(call.args.plasmid) || currentAnalysis.plasmid;
        const useLLM = asBoolean(call.args.useLLM);

        if (!ab1Dir) {
          return {
            tool: call.tool,
            ok: false,
            summary: "No AB1 source path is available for rerunning analysis.",
          };
        }

        const raw = (await invoke("run-analysis", ab1Dir, nextGenesDir, {
          plasmid: nextPlasmid,
          useLLM,
        })) as string;
        const result = JSON.parse(raw) as { samples?: Sample[] };
        const nextSamples = Array.isArray(result.samples) ? result.samples : [];
        const nextSelectedSampleId = getDefaultSelectedSampleId();

        currentAnalysis.samples = nextSamples;
        currentAnalysis.selectedSampleId = nextSelectedSampleId;
        currentAnalysis.sourcePath = ab1Dir;
        currentAnalysis.genesDir = nextGenesDir;
        currentAnalysis.plasmid = nextPlasmid;

        onAnalysisComplete?.({
          sourcePath: ab1Dir,
          genesDir: nextGenesDir,
          plasmid: nextPlasmid,
          samples: nextSamples,
          selectedSampleId: nextSelectedSampleId,
        });

        return {
          tool: call.tool,
          ok: true,
          summary: `Analysis finished with ${nextSamples.length} sample${nextSamples.length === 1 ? "" : "s"}.`,
          data: { sampleCount: nextSamples.length },
        };
      }
      case "export_report": {
        if (currentAnalysis.samples.length === 0) {
          return {
            tool: call.tool,
            ok: false,
            summary: "There are no samples available to export.",
          };
        }

        const result = await invoke(
          "export-excel",
          currentAnalysis.samples,
          currentAnalysis.sourcePath ?? null
        );
        if (!result) {
          return {
            tool: call.tool,
            ok: false,
            summary: "Export was cancelled.",
          };
        }

        return {
          tool: call.tool,
          ok: true,
          summary: "Export completed successfully.",
          data: result,
        };
      }
      default:
        return {
          tool: call.tool,
          ok: false,
          summary: `Unsupported tool: ${call.tool}`,
        };
    }
  };

  const handleSubmit = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    const prompt = draft.trim();
    if (!prompt || isRunning) {
      return;
    }

    const userMessage = createMessage({
      type: "user",
      content: prompt,
    });

    let conversationHistory = [...messages, userMessage];
    let recentToolResults: ToolResult[] = [];
    const currentAnalysis: CurrentAnalysisSnapshot = {
      sourcePath: sourcePath ?? undefined,
      genesDir: genesDir ?? undefined,
      plasmid,
      samples,
      selectedSampleId,
    };

    setDraft("");
    setMessages(conversationHistory);
    setIsRunning(true);

    try {
      for (let round = 0; round < DEFAULT_RUNTIME.maxRounds; round += 1) {
        const response = parseAgentTurnResponse(
          await invoke("agent-chat", {
            message: prompt,
            context: buildContext(currentAnalysis, conversationHistory, recentToolResults),
          })
        );

        if (response.action === "reply") {
          conversationHistory = [
            ...conversationHistory,
            createMessage({
              type: "agent",
              content: response.content,
              stopReason: response.stopReason ?? "final_reply",
              usage: response.usage,
            }),
          ];
          setMessages(conversationHistory);
          setIsRunning(false);
          return;
        }

        const planMessage = createMessage({
          type: "plan",
          content: response.message,
        });
        conversationHistory = [...conversationHistory, planMessage];
        setMessages(conversationHistory);

        const turnToolResults: ToolResult[] = [];
        const seenCalls = new Set<string>();

        for (const call of response.calls.slice(0, DEFAULT_RUNTIME.maxToolCallsPerTurn)) {
          const signature = JSON.stringify(call);
          if (seenCalls.has(signature)) {
            continue;
          }
          seenCalls.add(signature);

          const statusMessage = createMessage({
              type: "tool_status",
              toolName: call.tool,
              status: "running",
              content: translateTemplate(language, "agent.preparingTool", {
                tool: call.tool,
                args: summarizeArgs(language, call.args),
              }),
            });

          conversationHistory = [...conversationHistory, statusMessage];
          setMessages(conversationHistory);

          const result = await executeToolCall(call, currentAnalysis).catch((error: unknown) => ({
            tool: call.tool,
            ok: false,
            summary: error instanceof Error ? error.message : String(error),
          }));

          conversationHistory = setToolMessageStatus(
            conversationHistory,
            statusMessage.id,
            result.summary,
            result.ok ? "done" : "failed"
          );
          turnToolResults.push(result);
          setMessages(conversationHistory);
        }

        recentToolResults = [...recentToolResults, ...turnToolResults].slice(
          -DEFAULT_RUNTIME.maxToolCallsPerTurn * DEFAULT_RUNTIME.maxRounds
        );
      }

      conversationHistory = [
        ...conversationHistory,
        createMessage({
          type: "agent",
          content: t(language, "agent.maxRoundsReached"),
          stopReason: "max_rounds_reached",
        }),
      ];
      setMessages(conversationHistory);
    } catch (error) {
      const rawMessage = error instanceof Error ? error.message : String(error);
      const message =
        error instanceof Error && error.message === "invalid_model_output"
          ? t(language, "agent.invalidResponse")
          : rawMessage.includes("LLM_API_KEY") || rawMessage.includes("not configured")
            ? t(language, "agent.unavailableNoKey")
            : t(language, "agent.unavailable");

      setMessages((current) => [
        ...current,
        createMessage({
          type: "agent",
          content: message,
          stopReason:
            error instanceof Error && error.message === "invalid_model_output"
              ? "invalid_model_output"
              : "aborted",
        }),
      ]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  };

  return (
    <aside className="agent-panel">
      <header className="agent-panel-header">
        <div className="agent-panel-context">
          <span className="agent-panel-kicker">{batchLabel ?? plasmid}</span>
          <h3>{t(language, "agent.title")}</h3>
          <p>
            {samples.length} {t(language, "app.samples")}
            {selectedSampleId ? ` | ${selectedSampleId}` : ""}
          </p>
        </div>
        <div className="agent-panel-actions">
          <span className={`agent-status ${isRunning ? "busy" : "idle"}`}>
            {isRunning ? t(language, "agent.running") : t(language, "agent.idle")}
          </span>
          <button
            type="button"
            className="clear-chat-button"
            onClick={() => setMessages([])}
            disabled={isRunning || messages.length === 0}
          >
            {t(language, "agent.clear")}
          </button>
        </div>
      </header>

      <div className="agent-message-list">
        {messages.length === 0 ? (
          <div className="agent-empty-state">
            <h4>{t(language, "agent.askTitle")}</h4>
            <p>{t(language, "agent.askBody")}</p>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessageCard key={message.id} language={language} message={message} />
          ))
        )}
      </div>

      <form className="agent-composer" onSubmit={handleSubmit}>
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t(language, "agent.placeholder")}
          rows={4}
          disabled={isRunning}
        />
        <div className="agent-composer-footer">
          <span>{t(language, "agent.composerHint")}</span>
          <button type="submit" disabled={isRunning || draft.trim().length === 0}>
            {t(language, "agent.send")}
          </button>
        </div>
      </form>
    </aside>
  );
}




