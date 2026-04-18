import { useEffect, useRef, useState } from "react";
import type { PanelType } from "../components/SmartCanvas";
import type { AgentSettings } from "../lib/settingsStorage";
import { t, type AppLanguage } from "../i18n";

declare global {
  interface Window {
    electronAPI: {
      invoke: (channel: string, ...args: unknown[]) => Promise<any>;
      onAgentEvent: (callback: (payload: any) => void) => () => void;
    };
  }
}

/* ── Constants ───────────────────────────────────────────────────────── */

const MAX_MESSAGES = 60;
const INIT_TIMEOUT_MS = 35000;
const DETAIL_MAX_ATTEMPTS = 7;
const DETAIL_RETRY_MS = 1600;

/* ── Types ───────────────────────────────────────────────────────────── */

export type ProgressState = {
  phase: string;
  progress: number;
  label: string;
};

type AgentResult = Record<string, unknown>;

export type AgentEvent =
  | { type: "lifecycle"; phase?: "init" | "ready" | "run" | "error" | string; message?: string }
  | { type: "thinking" }
  | { type: "tool_calls_start" }
  | { type: "tool_call"; tool?: string }
  | { type: "tool_result"; tool?: string; result?: AgentResult }
  | { type: "reply"; content?: string; uiAction?: "show_trends" | "show_suggestions" | "show_analysis" | string; result?: AgentResult }
  | { type: "busy"; message?: string }
  | { type: "error"; message?: string }
  | { type: "confirm"; message?: string };

type PanelResolution = { panelType: PanelType; panelPayload: unknown; confirmMessage?: string };

/* ── Module-level pure utilities ─────────────────────────────────────── */

function resolvePanelFromEvent(payload: AgentEvent): PanelResolution | null {
  if (payload.type === "confirm") {
    return {
      panelType: "confirmation",
      panelPayload: null,
      confirmMessage: payload.message || "Please confirm this operation.",
    };
  }

  if (payload.type === "tool_result") {
    if (payload.tool === "detect_mutation_trends") return { panelType: "trends", panelPayload: payload.result };
    if (payload.tool === "generate_lab_suggestions") return { panelType: "suggestions", panelPayload: payload.result };
    if (payload.tool === "analyze_sequences") return { panelType: "analysis", panelPayload: payload.result };
  }

  if (payload.type === "reply") {
    if (payload.uiAction === "show_trends" && payload.result) return { panelType: "trends", panelPayload: payload.result };
    if (payload.uiAction === "show_suggestions" && payload.result) return { panelType: "suggestions", panelPayload: payload.result };
    if (payload.uiAction === "show_analysis" && payload.result) return { panelType: "analysis", panelPayload: payload.result };
    return { panelType: "text", panelPayload: payload };
  }

  return null;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout>;
  const timeoutPromise = new Promise<T>((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]).finally(() => {
    clearTimeout(timeoutId);
  }) as Promise<T>;
}

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeApiKey(raw: string): string {
  return raw
    .trim()
    .replace(/^['"]+|['"]+$/g, "")
    .replace(/^apikey\s*[:=]\s*/i, "")
    .replace(/^api[_\s-]?key\s*[:=]\s*/i, "")
    .replace(/^key\s*[:=]\s*/i, "")
    .trim();
}

function maskSecrets(input: string): string {
  return input.replace(/sk-[A-Za-z0-9_-]{8,}/g, "sk-***");
}

function getFriendlyToolName(tool: string, language: AppLanguage): string {
  if (tool === "analyze_sequences") return language === "zh" ? "\u5e8f\u5217\u5206\u6790" : "sequence analysis";
  if (tool === "detect_mutation_trends") return language === "zh" ? "\u7a81\u53d8\u8d8b\u52bf\u5206\u6790" : "mutation trend analysis";
  if (tool === "generate_lab_suggestions") return language === "zh" ? "\u5b9e\u9a8c\u5efa\u8bae\u751f\u6210" : "lab suggestion generation";
  if (tool === "query_history") return language === "zh" ? "\u5386\u53f2\u8bb0\u5f55\u67e5\u8be2" : "history lookup";
  if (tool === "get_analysis_detail") return language === "zh" ? "\u6837\u672c\u8be6\u60c5\u52a0\u8f7d" : "analysis detail loading";
  return tool || (language === "zh" ? "\u5de5\u5177" : "tool");
}

function getLifecycleLabel(language: AppLanguage, phase: string, message: string): string {
  const text = String(message || "").toLowerCase();
  if (phase === "init" && text.includes("init request received")) return t(language, "app.progress.step.initRequest");
  if (phase === "init" && text.includes("creating harness")) return t(language, "app.progress.step.boot");
  if (phase === "init" && text.includes("starting mcp server")) return t(language, "app.progress.step.connectTools");
  if (phase === "run" && text.includes("run started")) return t(language, "app.progress.step.runStarted");
  if (phase === "run" && text.includes("run finished")) return t(language, "app.progress.step.resultReady");
  if (phase === "run" && text.includes("loaded analysis detail")) return t(language, "app.progress.step.visualReady");
  return message;
}

/* ── Hook ─────────────────────────────────────────────────────────────── */

export function useAgentHarness(language: AppLanguage) {
  /* ── State ──────────────────────────────────────────────────────────── */
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [initialized, setInitialized] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [panelType, setPanelType] = useState<PanelType>("text");
  const [panelPayload, setPanelPayload] = useState<unknown>(null);
  const [confirmMessage, setConfirmMessage] = useState("");
  const [statusMessage, setStatusMessage] = useState(t(language, "app.status.needInit"));
  const [progress, setProgress] = useState<ProgressState>({ phase: "idle", progress: 0, label: t(language, "app.progress.idle") });

  /* ── Refs ───────────────────────────────────────────────────────────── */
  const latestEventRef = useRef<AgentEvent | null>(null);
  const assistantMessageCountRef = useRef(0);
  const panelTypeRef = useRef<PanelType>("text");
  const languageRef = useRef<AppLanguage>(language);
  const runTokenRef = useRef(0);

  /* ── Sync refs ──────────────────────────────────────────────────────── */
  useEffect(() => {
    panelTypeRef.current = panelType;
  }, [panelType]);

  useEffect(() => {
    languageRef.current = language;
  }, [language]);

  /* ── Internal helpers ──────────────────────────────────────────────── */

  function pushAssistant(text: string) {
    assistantMessageCountRef.current += 1;
    setMessages((current) => [...current, { role: "assistant", content: maskSecrets(text) }].slice(-MAX_MESSAGES));
  }

  function setProgressState(phase: string, value: number, label: string) {
    const nextProgress = Math.max(0, Math.min(100, value));
    const nextLabel = maskSecrets(label);
    setProgress((current) => {
      if (current.phase === phase && current.progress === nextProgress && current.label === nextLabel) return current;
      return { phase, progress: nextProgress, label: nextLabel };
    });
  }

  function updateAnalysisPayload(analysisId: string, patch: Record<string, unknown>) {
    setPanelPayload((current: unknown) => {
      const obj = current as Record<string, unknown> | null;
      if (!obj || obj.analysis_id !== analysisId) return current;
      return { ...obj, ...patch };
    });
  }

  async function hydrateAnalysisResult(result: AgentResult | undefined, runToken: number) {
    const analysisId = result?.analysis_id;
    if (!analysisId || typeof analysisId !== "string") return;
    if (runTokenRef.current !== runToken) return;

    const detail = result?.detail as { samples?: unknown[] } | undefined;
    if (detail && Array.isArray(detail.samples)) {
      setPanelPayload((current: unknown) => {
        const obj = current as Record<string, unknown> | null;
        const base = obj && obj.analysis_id === analysisId ? obj : (result as Record<string, unknown>);
        return {
          ...base,
          detail,
          samples: detail.samples,
          __detailPending: false,
          __detailError: result?.detail_error,
        };
      });
      return;
    }

    updateAnalysisPayload(analysisId, { __detailPending: true, __detailError: undefined });

    let lastError = "";
    for (let attempt = 1; attempt <= DETAIL_MAX_ATTEMPTS; attempt += 1) {
      if (runTokenRef.current !== runToken) return;
      try {
        const detailResp = await window.electronAPI.invoke("agent-harness-get-analysis-detail", analysisId);
        if (runTokenRef.current !== runToken) return;
        if (detailResp?.ok && detailResp?.detail) {
          setPanelPayload((current: unknown) => {
            const obj = current as Record<string, unknown> | null;
            const base = obj && obj.analysis_id === analysisId ? obj : (result as Record<string, unknown>);
            return {
              ...base,
              detail: detailResp.detail,
              samples: detailResp.detail.samples || [],
              __detailPending: false,
              __detailError: undefined,
            };
          });
          return;
        }
        lastError = detailResp?.error || "Detail not ready";
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
      }

      if (attempt < DETAIL_MAX_ATTEMPTS) {
        await wait(DETAIL_RETRY_MS);
      }
    }

    if (runTokenRef.current !== runToken) return;
    updateAnalysisPayload(analysisId, {
      __detailPending: false,
      __detailError: lastError || t(languageRef.current, "analysis.detailFetchFailed"),
    });
  }

  function applyAgentEvent(payload: AgentEvent) {
    latestEventRef.current = payload;
    const lang = languageRef.current;

    if (payload.type === "lifecycle") {
      const text = payload.message || "lifecycle";
      const lowered = String(text).toLowerCase();
      const friendly = getLifecycleLabel(lang, payload.phase || "", text);
      if (payload.phase === "init") setProgressState("init", 20, friendly);
      if (payload.phase === "ready") setProgressState("ready", 100, friendly);
      if (payload.phase === "run") {
        const isFinished = lowered.includes("run finished") || lowered.includes("finished");
        setProgressState(isFinished ? "reply" : "run", isFinished ? 100 : 45, friendly);
      }
      if (payload.phase === "error") setProgressState("error", 100, friendly);
      return;
    }

    if (payload.type === "thinking") {
      setProgressState("thinking", 58, t(lang, "app.progress.thinking"));
    } else if (payload.type === "tool_calls_start") {
      setProgressState("tool_calls", 70, t(lang, "app.progress.tools"));
    } else if (payload.type === "tool_call") {
      setProgressState("tool_call", 82, t(lang, "app.progress.toolCall", { tool: getFriendlyToolName(payload.tool || "tool", lang) }));
    } else if (payload.type === "tool_result") {
      setProgressState("tool_result", 92, t(lang, "app.progress.done"));
      if (payload.tool === "analyze_sequences") {
        const result = payload.result as AgentResult | undefined;
        const count = result?.sample_count as number | undefined;
        const dataset = result?.dataset as string | undefined;
        const analysisId = result?.analysis_id as string | undefined;
        const resultPayload: AgentResult = result && typeof result === "object" ? result : {};
        const detail = resultPayload.detail as { samples?: unknown[] } | undefined;
        setPanelType("analysis");
        setPanelPayload((current: unknown) => {
          const obj = current as Record<string, unknown> | null;
          return {
            ...(obj && obj.analysis_id === analysisId ? obj : {}),
            ...resultPayload,
            __detailPending: !(detail && Array.isArray(detail.samples)),
            __detailError: resultPayload.detail_error,
          };
        });
        const suffix = typeof count === "number"
          ? t(lang, "app.analysisSuffix", {
            count,
            dataset: dataset ? t(lang, "app.analysisDataset", { dataset }) : "",
            analysisId: analysisId ? t(lang, "app.analysisId", { analysisId }) : "",
          })
          : "";
        pushAssistant(t(lang, "app.analysisFinished", { suffix }));
        void hydrateAnalysisResult(result, runTokenRef.current);
      }
    } else if (payload.type === "reply") {
      setProgressState("reply", 100, t(lang, "app.progress.completed"));
    } else if (payload.type === "busy") {
      setProgressState("busy", 100, payload.message || t(lang, "app.progress.busy"));
    } else if (payload.type === "error") {
      setProgressState("error", 100, payload.message || t(lang, "error.runFailed", { message: "" }));
    }

    if (payload.type === "reply" || payload.type === "error" || payload.type === "busy") {
      const content = payload.type === "reply" ? payload.content : undefined;
      const message = payload.type !== "reply" ? payload.message : undefined;
      const text = content || message || JSON.stringify(payload);
      pushAssistant(text);
    }

    const resolved = resolvePanelFromEvent(payload);
    if (resolved) {
      if (payload.type === "reply" && resolved.panelType === "text" && panelTypeRef.current === "analysis") {
        return;
      }
      setPanelType(resolved.panelType);
      setPanelPayload(resolved.panelPayload);
      if (resolved.confirmMessage) setConfirmMessage(resolved.confirmMessage);
    }
  }

  /* ── IPC event subscription ────────────────────────────────────────── */
  useEffect(() => {
    const unsubscribe = window.electronAPI.onAgentEvent((payload: unknown) => {
      applyAgentEvent(payload as AgentEvent);
    });
    return () => unsubscribe?.();
  }, []);

  function applyTraceFallback(trace: unknown[] | undefined, eventCursor: AgentEvent | null) {
    if (!Array.isArray(trace) || trace.length === 0) return;
    const ipcDelivered = latestEventRef.current !== eventCursor;
    if (!ipcDelivered) trace.forEach((payload) => applyAgentEvent(payload as AgentEvent));
  }

  /* ── Public actions ────────────────────────────────────────────────── */

  async function initialize(settings: AgentSettings): Promise<void> {
    try {
      const cleanedKey = normalizeApiKey(settings.llmApiKey);
      if (!cleanedKey) throw new Error(t(language, "error.emptyApiKey"));

      setProgressState("init", 10, t(language, "app.status.initializing"));
      setStatusMessage(t(language, "app.status.initializing"));

      await window.electronAPI.invoke("agent-harness-shutdown");
      const eventCursor = latestEventRef.current;
      const initResult = await withTimeout(
        window.electronAPI.invoke("agent-harness-init", {
          llmApiKey: cleanedKey,
          llmBaseUrl: settings.llmBaseUrl,
          llmModel: settings.llmModel,
          maxTokens: settings.maxTokens,
        }),
        INIT_TIMEOUT_MS,
        "Initialization",
      );

      applyTraceFallback(initResult?.trace, eventCursor);
      if (!initResult?.ok) throw new Error(initResult?.error || t(language, "error.initFailed"));

      setInitialized(true);
      setStatusMessage(t(language, "app.status.ready"));

      const precheckCursor = latestEventRef.current;
      const precheck = await window.electronAPI.invoke("dataset-precheck");
      applyTraceFallback(precheck?.trace, precheckCursor);
      if (precheck?.ok && precheck?.report && precheck.report.allOk === false) {
        setStatusMessage(t(language, "app.status.readyIncomplete"));
      }

      setProgressState("ready", 100, t(language, "app.progress.initDone"));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatusMessage(t(language, "app.status.initFailed", { message: maskSecrets(message) }));
      setInitialized(false);
      setProgressState("error", 100, t(language, "app.status.initFailed", { message }));
    }
  }

  async function sendMessage(text: string, settings: AgentSettings): Promise<void> {
    const content = text.trim();
    if (!content || isRunning) return;
    if (!initialized) {
      setStatusMessage(t(language, "app.status.needInit"));
      return;
    }

    runTokenRef.current += 1;
    const currentRunToken = runTokenRef.current;

    setMessages((current) => [...current, { role: "user", content }].slice(-MAX_MESSAGES));
    setIsRunning(true);
    setProgressState("run", 38, t(language, "app.progress.run"));

    try {
      const assistantCountBeforeRun = assistantMessageCountRef.current;
      const cleanedKey = normalizeApiKey(settings.llmApiKey);
      const eventCursor = latestEventRef.current;
      const runResult = await window.electronAPI.invoke("agent-harness-run", content, {
        llmApiKey: cleanedKey,
        llmBaseUrl: settings.llmBaseUrl,
        llmModel: settings.llmModel,
        maxTokens: settings.maxTokens,
      });

      if (runTokenRef.current !== currentRunToken) return;
      applyTraceFallback(runResult?.trace, eventCursor);

      if (!runResult?.ok) {
        const err = runResult?.error || "Run failed";
        pushAssistant(t(language, "error.runFailed", { message: err }));
        setProgressState("error", 100, t(language, "error.runFailed", { message: err }));
        return;
      }

      if (Array.isArray(runResult?.events) && runResult.events.length > 0) {
        const ipcDelivered = latestEventRef.current !== eventCursor;
        if (!ipcDelivered) runResult.events.forEach((payload: unknown) => applyAgentEvent(payload as AgentEvent));
      }

      if (assistantMessageCountRef.current === assistantCountBeforeRun) {
        pushAssistant(t(language, "app.runFallback"));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushAssistant(t(language, "error.runFailed", { message }));
      setProgressState("error", 100, t(language, "error.runFailed", { message }));
    } finally {
      if (runTokenRef.current === currentRunToken) {
        setIsRunning(false);
      }
    }
  }

  async function exportDebugLog(): Promise<void> {
    try {
      const resp = await window.electronAPI.invoke("agent-export-debug-log");
      if (resp?.ok && resp?.filePath) {
        pushAssistant(t(languageRef.current, "app.debugExportOk", { filePath: resp.filePath }));
      } else {
        pushAssistant(t(languageRef.current, "app.debugExportFail", { message: resp?.error || "unknown error" }));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushAssistant(t(languageRef.current, "app.debugExportFail", { message }));
    }
  }

  /* ── Return public interface ───────────────────────────────────────── */
  return {
    initialized,
    isRunning,
    messages,
    progress,
    statusMessage,
    panelType,
    panelPayload,
    confirmMessage,
    initialize,
    sendMessage,
    exportDebugLog,
    setPanelType,
    clearMessages: () => setMessages([]),
  };
}
