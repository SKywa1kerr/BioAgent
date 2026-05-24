import { useCallback, useEffect, useRef, useState } from "react";
import { SmartCanvas, type PanelType } from "./components/SmartCanvas";
import { ChatPanel } from "./components/ChatPanel";
import { SettingsModal } from "./components/SettingsModal";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { CommandPalette } from "./components/CommandPalette";
import { OnboardingCoach } from "./components/OnboardingCoach";
import { ShortcutsOverlay } from "./components/ShortcutsOverlay";
import { TitleBar } from "./components/TitleBar";
import { AnalysisPanel } from "./components/panels/AnalysisPanel";
import { MutationTrendPanel } from "./components/panels/MutationTrendPanel";
import { LabSuggestionPanel } from "./components/panels/LabSuggestionPanel";
import { ConfirmationDialog } from "./components/panels/ConfirmationDialog";
import { Sidebar, type DatasetItem } from "./components/sidebar/Sidebar";
import { Splitter, CollapsedRail } from "./components/workbench/Splitter";
import { useChatColumnWidth } from "./hooks/useChatColumnWidth";
import { DropZone } from "./components/DropZone";
import { InitDialog } from "./components/InitDialog";
import { ImportDatasetDialog } from "./components/ImportDatasetDialog";
import { MultiDatasetChooserDialog, type DatasetCandidate } from "./components/MultiDatasetChooserDialog";
import { useAgentHarness, type LastErrorEvent } from "./hooks/useAgentHarness";
import { useAnalysisHistory } from "./hooks/useAnalysisHistory";
import { useConversations, type Conversation } from "./hooks/useConversations";
import { useOnboarding } from "./hooks/useOnboarding";
import { useUpdater, type UpdaterPhase } from "./hooks/useUpdater";
import { useToasts } from "./components/ui/ToastProvider";
import { registerCommand } from "./lib/commands/registry";
import { loadSettings, saveSettings, type AgentSettings } from "./lib/settingsStorage";
import { getProvider } from "./lib/providers.js";
import { t, type AppLanguage } from "./i18n";

const CANVAS_MIN_WIDTH = 360;
const SIDEBAR_COLLAPSE_KEY = "bioagent-sidebar-collapsed";

/* ── Helpers ────────────────────────────────────────────────────────── */

function getLocalStorageValue<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const saved = window.localStorage.getItem(key);
    if (saved && (allowed as readonly string[]).includes(saved)) return saved as T;
  } catch { /* ignore */ }
  return fallback;
}

function loadSidebarCollapsed(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSE_KEY) === "1";
  } catch { return false; }
}

function saveSidebarCollapsed(v: boolean): void {
  try { window.localStorage.setItem(SIDEBAR_COLLAPSE_KEY, v ? "1" : "0"); } catch { /* ignore */ }
}

/* ── App ────────────────────────────────────────────────────────────── */

export function App() {
  const [language, setLanguage] = useState<AppLanguage>(() => getLocalStorageValue("bioagent-language", ["zh", "en"] as const, "zh"));
  const [theme, setTheme] = useState<"light" | "dark">(() => getLocalStorageValue("bioagent-theme", ["light", "dark"] as const, "dark"));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<AgentSettings>(loadSettings);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [prefillText, setPrefillText] = useState<string | null>(null);
  const chatWidth = useChatColumnWidth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => loadSidebarCollapsed());
  const shellRef = useRef<HTMLDivElement | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const onboarding = useOnboarding();

  useEffect(() => {
    saveSidebarCollapsed(sidebarCollapsed);
  }, [sidebarCollapsed]);

  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const goOnline = () => setIsOnline(true);
    const goOffline = () => setIsOnline(false);
    window.addEventListener("online", goOnline);
    window.addEventListener("offline", goOffline);
    return () => {
      window.removeEventListener("online", goOnline);
      window.removeEventListener("offline", goOffline);
    };
  }, []);

  const agent = useAgentHarness(language);
  const toasts = useToasts();
  const updater = useUpdater();

  /* ── Toast wiring (action handlers + error event → toast) ─────────── */

  const agentRef = useRef(agent);
  useEffect(() => { agentRef.current = agent; }, [agent]);

  useEffect(() => {
    const offs = [
      toasts.registerActionHandler("export-debug-log", () => { void agentRef.current.exportDebugLog(); }),
      toasts.registerActionHandler("retry-last", () => { void agentRef.current.retryLast(); }),
      toasts.registerActionHandler("updater-install", () => { updater.install(); }),
    ];
    return () => offs.forEach((off) => off());
  }, [toasts, updater.install]);

  // Dedupe by reference so a language change alone does not re-emit an
  // already-delivered toast. Each new error from the harness is a fresh
  // object literal, so reference equality is the right gate here.
  const lastEmittedErrorRef = useRef<LastErrorEvent | null>(null);
  useEffect(() => {
    const event = agent.lastErrorEvent;
    if (!event || event === lastEmittedErrorRef.current) return;
    lastEmittedErrorRef.current = event;
    const titleKey =
      event.kind === "init" ? "toast.error.initTitle" :
      event.kind === "run" ? "toast.error.runTitle" :
      "toast.error.runtimeTitle";
    const action =
      event.kind === "run"
        ? { label: t(language, "toast.action.retry"), actionId: "retry-last" }
        : { label: t(language, "toast.action.viewLog"), actionId: "export-debug-log" };
    toasts.pushToast({
      kind: "error",
      title: t(language, titleKey),
      description: event.message,
      durationMs: 0,
      action,
    });
  }, [agent.lastErrorEvent, toasts, language]);

  // Updater state → toast. Dedup by phase so download-progress ticks and
  // language toggles don't re-emit. Only "available", "ready", "error" toast.
  const lastEmittedUpdaterPhaseRef = useRef<UpdaterPhase | null>(null);
  useEffect(() => {
    const { phase, version, message } = updater.state;
    if (phase === lastEmittedUpdaterPhaseRef.current) return;
    if (phase !== "available" && phase !== "ready" && phase !== "error") return;
    lastEmittedUpdaterPhaseRef.current = phase;
    if (phase === "available") {
      toasts.pushToast({
        kind: "info",
        title: t(language, "updater.available", { version: version ?? "" }),
      });
    } else if (phase === "ready") {
      toasts.pushToast({
        kind: "success",
        title: t(language, "updater.ready", { version: version ?? "" }),
        durationMs: 0,
        action: { label: t(language, "updater.action.restart"), actionId: "updater-install" },
      });
    } else {
      toasts.pushToast({
        kind: "error",
        title: t(language, "updater.error", { message: message ?? "" }),
        durationMs: 0,
      });
    }
  }, [updater.state, toasts, language]);

  /* ── Panel history cache ──────────────────────────────────────────── */

  const [panelCache, setPanelCache] = useState<Record<string, any>>({});
  const canvasHasContent = !!(panelCache.analysis || panelCache.trends || panelCache.suggestions);
  const [canvasOpen, setCanvasOpen] = useState(false);
  const prevCanvasHasContentRef = useRef(false);
  useEffect(() => {
    // Auto-open the canvas the moment content arrives where there was none.
    // After that the user controls open/closed via the rail / collapse btn.
    if (canvasHasContent && !prevCanvasHasContentRef.current) {
      setCanvasOpen(true);
    }
    prevCanvasHasContentRef.current = canvasHasContent;
  }, [canvasHasContent]);
  const [activeTab, setActiveTab] = useState<PanelType>("text");

  useEffect(() => {
    const type = agent.panelType;
    const payload = agent.panelPayload;
    if (payload && (type === "analysis" || type === "trends" || type === "suggestions")) {
      setPanelCache((prev) => ({ ...prev, [type]: payload }));
      // Only auto-switch to the analysis tab. Trends / suggestions arrive
      // chained after the first analysis call and silently filling their
      // cache (rather than yanking the user's focus) matches what users
      // actually want — they'll click the chip when they want to see them.
      if (type === "analysis") {
        setActiveTab(type);
      }
    }
  }, [agent.panelType, agent.panelPayload]);

  // Chained tool results fill the relevant cache slot WITHOUT switching the
  // active tab — the user expects the analysis tab to stay focused after a
  // dataset run, with trends/suggestions ready underneath.
  useEffect(() => {
    if (agent.chainedTrends?.result) {
      setPanelCache((prev) => ({ ...prev, trends: agent.chainedTrends!.result }));
    }
  }, [agent.chainedTrends]);
  useEffect(() => {
    if (agent.chainedSuggestions?.result) {
      setPanelCache((prev) => ({ ...prev, suggestions: agent.chainedSuggestions!.result }));
    }
  }, [agent.chainedSuggestions]);

  const TAB_TYPES: PanelType[] = ["analysis", "trends", "suggestions"];
  const availableTabs = TAB_TYPES.filter((tab) => panelCache[tab] != null);

  /* ── Dataset list (sidebar) ───────────────────────────────────────── */

  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [importPrefill, setImportPrefill] = useState<{ ab1Dir?: string; gbDir?: string } | null>(null);
  const [datasetCandidates, setDatasetCandidates] = useState<DatasetCandidate[] | null>(null);

  const refreshDatasets = useCallback(async () => {
    if (!agent.initialized) return;
    try {
      const resp = await window.electronAPI.invoke("datasets-list");
      if (resp?.ok) {
        const builtin: DatasetItem[] = (resp.builtin || []).map((d: any) => ({
          id: d.id,
          label: d.label || d.id,
          kind: "builtin",
        }));
        const user: DatasetItem[] = (resp.user || []).map((d: any) => ({
          id: d.id,
          label: d.label || d.id,
          kind: "user",
        }));
        setDatasets([...builtin, ...user]);
      }
    } catch {
      // sidebar dataset list is best-effort
    }
  }, [agent.initialized]);

  useEffect(() => {
    void refreshDatasets();
  }, [refreshDatasets]);

  /* ── Recent analyses rail (history) ───────────────────────────────── */

  const historyApi = useAnalysisHistory({ enabled: agent.initialized, limit: 20 });
  const conversations = useConversations();

  // Auto-sync: whenever the in-flight messages change, push them to the
  // current conversation (creates one on first message). Skip during the
  // brief moment when we're swapping conversations to avoid clobbering.
  const swappingConversationRef = useRef(false);
  useEffect(() => {
    if (swappingConversationRef.current) return;
    if (!agent.initialized) return;
    conversations.syncMessages(agent.messages);
  }, [agent.messages, agent.initialized, conversations]);
  const lastSeenAnalysisIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!agent.initialized) return;
    if (agent.panelType !== "analysis") return;
    const payload = agent.panelPayload as { analysis_id?: string } | null;
    const analysisId = payload?.analysis_id;
    if (!analysisId || analysisId === lastSeenAnalysisIdRef.current) return;
    lastSeenAnalysisIdRef.current = analysisId;
    void historyApi.refresh();
  }, [agent.initialized, agent.panelType, agent.panelPayload, historyApi]);

  const handleHistorySelect = useCallback(async (analysisId: string) => {
    try {
      const bundleResp = await window.electronAPI.invoke("agent-harness-get-analysis-bundle", analysisId);
      if (bundleResp?.ok && bundleResp?.detail) {
        const detail = bundleResp.detail;
        setPanelCache((prev) => ({
          ...prev,
          analysis: {
            analysis_id: analysisId,
            ...detail,
            samples: Array.isArray(detail.samples) ? detail.samples : [],
            __detailPending: false,
            __detailError: undefined,
          },
          ...(bundleResp.trends ? { trends: bundleResp.trends } : {}),
          ...(bundleResp.suggestions ? { suggestions: bundleResp.suggestions } : {}),
        }));
        setActiveTab("analysis");
      } else {
        const message = bundleResp?.error || "unknown error";
        toasts.pushToast({
          kind: "error",
          title: t(language, "history.loadFailed", { message }),
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toasts.pushToast({
        kind: "error",
        title: t(language, "history.loadFailed", { message }),
      });
    }
  }, [toasts, language]);

  const activeAnalysisId = (panelCache.analysis as { analysis_id?: string } | undefined)?.analysis_id ?? null;

  /* ── Persist theme & language ──────────────────────────────────────── */

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { window.localStorage.setItem("bioagent-theme", theme); } catch { /* ignore */ }
  }, [theme]);

  useEffect(() => {
    try { window.localStorage.setItem("bioagent-language", language); } catch { /* ignore */ }
  }, [language]);

  /* ── Global keyboard shortcuts ───────────────────────────────────── */

  const isAnyModalOpen = settingsOpen || !!agent.confirmMessage || shortcutsOpen;

  useEffect(() => {
    function handleGlobalKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey;

      // "?" → shortcuts overlay (only when not in editable + no other modal)
      if (e.key === "?" && !mod && !e.altKey) {
        const target = e.target as HTMLElement | null;
        const editable =
          target?.tagName === "INPUT" ||
          target?.tagName === "TEXTAREA" ||
          target?.isContentEditable === true;
        if (editable) return;
        if (isAnyModalOpen || paletteOpen || shortcutsOpen) return;
        e.preventDefault();
        setShortcutsOpen(true);
        return;
      }

      // Ctrl+K → toggle command palette
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (isAnyModalOpen) return;
        setPaletteOpen((v) => !v);
        return;
      }

      // Ctrl+, → open settings
      if (mod && e.key === ",") {
        e.preventDefault();
        setSettingsOpen(true);
        return;
      }

      // Escape → close topmost panel
      if (e.key === "Escape") {
        if (shortcutsOpen) {
          e.preventDefault();
          setShortcutsOpen(false);
          return;
        }
        if (settingsOpen) {
          e.preventDefault();
          setSettingsOpen(false);
          return;
        }
      }

      // Ctrl+L → focus chat input
      if (mod && e.key === "l") {
        e.preventDefault();
        chatInputRef.current?.focus();
        return;
      }

      // Ctrl+B → toggle sidebar collapse
      if (mod && e.key.toLowerCase() === "b" && !e.shiftKey) {
        e.preventDefault();
        setSidebarCollapsed((v) => !v);
        return;
      }

      // Ctrl+Shift+Delete → clear chat
      if (mod && e.shiftKey && e.key === "Delete") {
        e.preventDefault();
        if (confirm(t(language, "chat.clearConfirm"))) agent.clearMessages();
        return;
      }
    }

    document.addEventListener("keydown", handleGlobalKeyDown);
    return () => document.removeEventListener("keydown", handleGlobalKeyDown);
  }, [settingsOpen, shortcutsOpen, paletteOpen, language, agent, isAnyModalOpen]);

  /* ── Command registry (cross-cutting) ─────────────────────────────── */

  const focusChat = useCallback(() => chatInputRef.current?.focus(), []);
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const toggleTheme = useCallback(() => setTheme((t) => (t === "light" ? "dark" : "light")), []);
  const toggleLanguage = useCallback(() => setLanguage((l) => (l === "zh" ? "en" : "zh")), []);
  const prefillChat = useCallback((text: string) => setPrefillText(text), []);

  useEffect(() => {
    const offs: Array<() => void> = [];

    offs.push(
      registerCommand({
        id: "nav.focus-chat",
        title: t(language, "palette.cmd.focusChat"),
        group: "nav",
        shortcut: "Ctrl+L",
        run: focusChat,
      }),
      registerCommand({
        id: "nav.open-settings",
        title: t(language, "palette.cmd.openSettings"),
        group: "nav",
        shortcut: "Ctrl+,",
        run: openSettings,
      }),
      registerCommand({
        id: "nav.tab-analysis",
        title: t(language, "palette.cmd.tabAnalysis"),
        group: "nav",
        when: () => panelCache.analysis != null,
        run: () => setActiveTab("analysis"),
      }),
      registerCommand({
        id: "nav.tab-trends",
        title: t(language, "palette.cmd.tabTrends"),
        group: "nav",
        when: () => panelCache.trends != null,
        run: () => setActiveTab("trends"),
      }),
      registerCommand({
        id: "nav.tab-suggestions",
        title: t(language, "palette.cmd.tabSuggestions"),
        group: "nav",
        when: () => panelCache.suggestions != null,
        run: () => setActiveTab("suggestions"),
      }),
      registerCommand({
        id: "appearance.toggle-theme",
        title: t(language, "palette.cmd.toggleTheme"),
        group: "appearance",
        run: toggleTheme,
      }),
      registerCommand({
        id: "appearance.toggle-lang",
        title: t(language, "palette.cmd.toggleLang"),
        group: "appearance",
        run: toggleLanguage,
      }),
      registerCommand({
        id: "log.export-debug",
        title: t(language, "palette.cmd.exportDebug"),
        group: "log",
        run: () => void agent.exportDebugLog(),
      }),
      registerCommand({
        id: "examples.analyze-base",
        title: t(language, "palette.cmd.example.base"),
        group: "examples",
        run: () => prefillChat("分析 base 数据集"),
      }),
      registerCommand({
        id: "examples.analyze-pro",
        title: t(language, "palette.cmd.example.pro"),
        group: "examples",
        run: () => prefillChat("分析 pro 数据集"),
      }),
      registerCommand({
        id: "examples.trends",
        title: t(language, "palette.cmd.example.trends"),
        group: "examples",
        run: () => prefillChat("显示突变趋势"),
      }),
      registerCommand({
        id: "examples.suggestions",
        title: t(language, "palette.cmd.example.suggestions"),
        group: "examples",
        run: () => prefillChat("给出实验建议"),
      }),
    );

    return () => { offs.forEach((off) => off()); };
  }, [language, focusChat, openSettings, toggleTheme, toggleLanguage, prefillChat, agent, panelCache.analysis, panelCache.trends, panelCache.suggestions]);

  /* ── Settings save → init ─────────────────────────────────────────── */

  function handleSettingsSave(next: AgentSettings) {
    setSettings(next);
    saveSettings(next);
    setSettingsOpen(false);
    void agent.initialize(next);
  }

  /* ── Send (auto-init if needed) ───────────────────────────────────── */

  function handleSend(text: string) {
    if (!agent.initialized && settings.llmApiKey) {
      void agent.initialize(settings).then(() => agent.sendMessage(text, settings));
      return;
    }
    void agent.sendMessage(text, settings);
  }

  /* ── Compact progress bar (inside canvas) ─────────────────────────── */

  // Per user request: progress bars are removed. The pending-spinner in
  // the chat header already conveys "agent is working" and the canvas
  // doesn't need its own bar.
  function renderCompactProgress() {
    return null;
  }

  /* ── Panel tab bar ────────────────────────────────────────────────── */

  function renderTabBar() {
    if (!agent.initialized || availableTabs.length <= 1) return null;
    const tabLabels: Record<string, string> = {
      analysis: t(language, "panel.tab.analysis"),
      trends: t(language, "panel.tab.trends"),
      suggestions: t(language, "panel.tab.suggestions"),
    };
    return (
      <div className="panel-tab-bar">
        {availableTabs.map((tab) => (
          <button
            key={tab}
            className={`panel-tab${activeTab === tab ? " panel-tab-active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tabLabels[tab] || tab}
          </button>
        ))}
      </div>
    );
  }

  /* ── Panel routing ────────────────────────────────────────────────── */

  function renderPanel() {
    if (!agent.initialized) {
      return (
        <div className="detail-card audience-card">
          <h3>{t(language, "app.ready.title")}</h3>
          <p>{t(language, "app.ready.body")}</p>
        </div>
      );
    }

    if (agent.panelType === "confirmation") {
      return <ConfirmationDialog message={agent.confirmMessage} onConfirm={() => agent.setPanelType("text")} onCancel={() => agent.setPanelType("text")} language={language} />;
    }

    const cachedPayload = panelCache[activeTab];
    if (activeTab === "analysis" && cachedPayload) return <AnalysisPanel result={cachedPayload} language={language} />;
    if (activeTab === "trends" && cachedPayload) return <MutationTrendPanel result={cachedPayload} language={language} />;
    if (activeTab === "suggestions" && cachedPayload) return <LabSuggestionPanel result={cachedPayload} language={language} />;

    const titleKey =
      activeTab === "trends" ? "panel.empty.trends.title"
      : activeTab === "suggestions" ? "panel.empty.suggestions.title"
      : activeTab === "analysis" ? "panel.empty.analysis.title"
      : "app.ready.title";
    const bodyKey =
      activeTab === "trends" ? "panel.empty.trends.body"
      : activeTab === "suggestions" ? "panel.empty.suggestions.body"
      : activeTab === "analysis" ? "panel.empty.analysis.body"
      : "app.ready.body";

    return (
      <div className="detail-card audience-card">
        <h3>{t(language, titleKey)}</h3>
        <p>{t(language, bodyKey)}</p>
      </div>
    );
  }

  /* ── Layout ───────────────────────────────────────────────────────── */

  const handleSplitterResize = useCallback((dx: number) => {
    const containerW = shellRef.current?.clientWidth ?? window.innerWidth;
    chatWidth.applyDelta(dx, containerW, CANVAS_MIN_WIDTH);
  }, [chatWidth]);

  const chatColumnStyle = chatWidth.collapsed
    ? undefined
    : ({ "--chat-w": `${chatWidth.width}px` } as React.CSSProperties);

  const shellContent = (
    <div className="app-shell-content" ref={shellRef} style={chatColumnStyle}>
      <Sidebar
        language={language}
        history={historyApi.items}
        datasets={datasets}
        activeAnalysisId={activeAnalysisId}
        activeTab={activeTab}
        hasAnalysisCache={panelCache.analysis != null}
        hasTrendsCache={panelCache.trends != null}
        hasSuggestionsCache={panelCache.suggestions != null}
        modelLabel={settings.llmModel}
        onSelectHistory={(id) => void handleHistorySelect(id)}
        onSelectTab={setActiveTab}
        onOpenSettings={() => setSettingsOpen(true)}
        onSelectDataset={(name) => setPrefillText(`分析 ${name} 数据集`)}
        onAddDataset={() => { setImportPrefill(null); setImportDialogOpen(true); }}
        onDeleteDataset={async (id, label) => {
          const confirmMsg = language === "zh"
            ? `确定要删除数据集「${label}」吗？仅移除注册，不会删除磁盘上的文件。`
            : `Delete the dataset "${label}"? This only removes the registration; the files on disk are untouched.`;
          if (!confirm(confirmMsg)) return;
          try {
            const resp = await window.electronAPI.invoke("dataset-delete", id);
            if (resp?.ok) {
              void refreshDatasets();
              toasts.pushToast({
                kind: "success",
                title: language === "zh" ? `已删除：${label}` : `Deleted: ${label}`,
              });
            } else {
              toasts.pushToast({
                kind: "error",
                title: resp?.error || (language === "zh" ? "删除失败" : "Delete failed"),
                durationMs: 0,
              });
            }
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            toasts.pushToast({ kind: "error", title: msg, durationMs: 0 });
          }
        }}
        conversations={conversations.conversations.map((c) => ({
          id: c.id,
          title: c.title,
          updatedAt: c.updatedAt,
        }))}
        currentConversationId={conversations.currentId}
        onNewConversation={() => {
          swappingConversationRef.current = true;
          conversations.newConversation();
          agent.clearMessages();
          // Release the swap guard on the next tick so the syncMessages
          // effect doesn't immediately repopulate the just-cleared list.
          Promise.resolve().then(() => { swappingConversationRef.current = false; });
        }}
        onSelectConversation={(id) => {
          if (id === conversations.currentId) return;
          swappingConversationRef.current = true;
          conversations.setCurrent(id);
          const target = conversations.conversations.find((c) => c.id === id);
          agent.loadMessages(target?.messages || []);
          Promise.resolve().then(() => { swappingConversationRef.current = false; });
        }}
        onDeleteConversation={(id, title) => {
          const ok = confirm(language === "zh"
            ? `确定删除对话「${title}」吗？`
            : `Delete conversation "${title}"?`);
          if (!ok) return;
          conversations.remove(id);
          if (conversations.currentId === id) {
            swappingConversationRef.current = true;
            agent.clearMessages();
            Promise.resolve().then(() => { swappingConversationRef.current = false; });
          }
        }}
      />

      {chatWidth.collapsed ? (
        <CollapsedRail onExpand={chatWidth.expand} language={language} />
      ) : (
        <ChatPanel
          messages={agent.messages}
          isRunning={agent.isRunning}
          progress={agent.progress}
          language={language}
          initialized={agent.initialized}
          onSend={handleSend}
          onExportDebug={() => void agent.exportDebugLog()}
          onToggleLanguage={() => setLanguage((l) => (l === "zh" ? "en" : "zh"))}
          onToggleTheme={() => setTheme((v) => (v === "dark" ? "light" : "dark"))}
          onOpenSettings={() => setSettingsOpen(true)}
          onClear={() => { if (confirm(t(language, "chat.clearConfirm"))) agent.clearMessages(); }}
          theme={theme}
          prefillText={prefillText}
          onPrefillConsumed={() => setPrefillText(null)}
          inputRef={chatInputRef}
          onOpenPalette={() => setPaletteOpen(true)}
          modelPicker={{
            currentModel: settings.llmModel || "",
            availableModels: getProvider(settings.provider || "custom").suggestedModels || [],
            providerLabel: getProvider(settings.provider || "custom").label,
            onChange: (model) => {
              const next = { ...settings, llmModel: model };
              handleSettingsSave(next);
              toasts.pushToast({
                kind: "success",
                title: language === "zh" ? `已切换模型：${model}` : `Model switched: ${model}`,
              });
            },
          }}
          onAttach={async () => {
            // Paperclip flow mirrors drag-drop: open the system picker for
            // files OR a folder, then route through inspect-dropped-paths
            // → inspect-dataset-folder so the user gets the same
            // multi-dataset chooser, prefilled import dialog, or analyze
            // path that they would by dropping the same selection.
            const picked = await window.electronAPI.invoke("dialog-pick-attach", {
              title: t(language, "composer.attach"),
            });
            if (!picked || picked.canceled) return;
            const paths: string[] = Array.isArray(picked.paths) ? picked.paths : [];
            if (paths.length === 0) return;

            try {
              const inspect = await window.electronAPI.invoke("inspect-dropped-paths", paths);
              const dirs: string[] = Array.isArray(inspect?.dirs) ? inspect.dirs : [];
              if (dirs.length === 1) {
                const folder = dirs[0]!;
                const inspected = await window.electronAPI.invoke("inspect-dataset-folder", folder);
                if (inspected?.ok && inspected.layout === "multi" && Array.isArray(inspected.candidates) && inspected.candidates.length > 0) {
                  setDatasetCandidates(inspected.candidates);
                  return;
                }
                if (inspected?.ok) {
                  setImportPrefill({ ab1Dir: inspected.ab1Dir, gbDir: inspected.gbDir });
                } else {
                  setImportPrefill({ ab1Dir: folder, gbDir: folder });
                }
                setImportDialogOpen(true);
                return;
              }
              if (dirs.length >= 2) {
                const [d0, d1] = dirs as [string, string];
                const looksAb1 = (s: string) => /ab1/i.test(s);
                const looksGb = (s: string) => /\bgb\b|gbk/i.test(s);
                let ab1Dir = d0;
                let gbDir = d1;
                if (looksAb1(d1) || looksGb(d0)) {
                  ab1Dir = d1;
                  gbDir = d0;
                }
                setImportPrefill({ ab1Dir, gbDir });
                setImportDialogOpen(true);
                return;
              }
              // Files only — hand off to analyze-dropped-files like the
              // drag-drop path does. Separate ab1/gb extensions first.
              const ab1Paths = paths.filter((p) => /\.ab1$/i.test(p));
              const gbPaths = paths.filter((p) => /\.gbk?$/i.test(p));
              if (ab1Paths.length === 0 && gbPaths.length === 0) {
                toasts.pushToast({
                  kind: "warning",
                  title: language === "zh"
                    ? "未识别到 .ab1 或 .gb 文件"
                    : "No .ab1 or .gb files in selection",
                });
                return;
              }
              await window.electronAPI.invoke("analyze-dropped-files", { ab1Paths, gbPaths });
            } catch (err) {
              const msg = err instanceof Error ? err.message : String(err);
              toasts.pushToast({ kind: "error", title: msg, durationMs: 0 });
            }
          }}
        />
      )}

      <Splitter
        onResize={handleSplitterResize}
        onCollapse={chatWidth.collapsed ? chatWidth.expand : chatWidth.collapse}
        ariaLabel={t(language, chatWidth.collapsed ? "splitter.expandChat" : "splitter.collapseChat")}
      />

      <main className="canvas-panel" aria-label="Analysis canvas">
        {canvasOpen ? (
          <SmartCanvas title={t(language, "app.canvasTitle")} panelType={activeTab}>
            <button
              type="button"
              className="canvas-collapse-btn"
              onClick={() => setCanvasOpen(false)}
              aria-label={t(language, "canvas.collapse")}
              title={t(language, "canvas.collapse")}
            >
              ✕
            </button>
            {renderTabBar()}
            {renderCompactProgress()}
            <ErrorBoundary
              fallbackTitle={t(language, "app.ready.title")}
              retryLabel={language === "zh" ? "重试" : "Retry"}
            >
              {renderPanel()}
            </ErrorBoundary>
          </SmartCanvas>
        ) : (
          <button
            type="button"
            className="canvas-rail"
            onClick={() => setCanvasOpen(true)}
            aria-label={t(language, "canvas.expand")}
            title={t(language, canvasHasContent ? "canvas.expand.hasContent" : "canvas.expand")}
          >
            <span className="canvas-rail-icon" aria-hidden>📊</span>
            {canvasHasContent ? <span className="canvas-rail-dot" aria-hidden /> : null}
          </button>
        )}
      </main>
    </div>
  );

  const shellClass =
    "app-shell" +
    (sidebarCollapsed ? " sidebar-collapsed" : "") +
    (chatWidth.collapsed ? " chat-collapsed" : "") +
    (canvasOpen ? "" : " canvas-collapsed");

  return (
    <div className={shellClass}>
      <TitleBar
        title={t(language, "app.title")}
        labels={{
          minimize: t(language, "titlebar.minimize"),
          maximize: t(language, "titlebar.maximize"),
          restore: t(language, "titlebar.restore"),
          close: t(language, "titlebar.close"),
        }}
      />
      {!isOnline ? <div className="offline-banner">{t(language, "app.offline")}</div> : null}
      {agent.initialized ? (
        <DropZone
          language={language}
          onRequestImport={(prefill) => { setImportPrefill(prefill); setImportDialogOpen(true); }}
          onRequestChoose={(candidates) => setDatasetCandidates(candidates)}
        >
          {shellContent}
        </DropZone>
      ) : (
        shellContent
      )}

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSave={handleSettingsSave}
        currentSettings={settings}
        language={language}
        theme={theme}
        onToggleTheme={toggleTheme}
        onToggleLanguage={toggleLanguage}
      />

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        language={language}
      />

      <ShortcutsOverlay
        open={shortcutsOpen}
        onClose={() => setShortcutsOpen(false)}
        language={language}
      />

      <InitDialog
        open={!agent.initialized && !settingsOpen}
        initialSettings={settings}
        language={language}
        statusMessage={agent.statusMessage}
        isInitializing={!agent.initialized && agent.progress.progress > 0 && agent.progress.progress < 100}
        onSubmit={handleSettingsSave}
      />

      <ImportDatasetDialog
        open={importDialogOpen}
        language={language}
        prefill={importPrefill}
        onClose={() => setImportDialogOpen(false)}
        onSuccess={(label) => {
          setImportDialogOpen(false);
          setImportPrefill(null);
          void refreshDatasets();
          toasts.pushToast({
            kind: "success",
            title: t(language, "import.toast.success", { label }),
          });
        }}
        onError={(message) => {
          toasts.pushToast({
            kind: "error",
            title: t(language, "import.toast.failed", { message }),
            durationMs: 0,
          });
        }}
      />

      <MultiDatasetChooserDialog
        open={datasetCandidates !== null && datasetCandidates.length > 0}
        language={language}
        candidates={datasetCandidates || []}
        onCancel={() => setDatasetCandidates(null)}
        onPick={(cand) => {
          setDatasetCandidates(null);
          // For "subdirs" layout the dataset folder has ab1/ + gb/ children;
          // for "flat" both point at the folder itself. Either way the
          // import dialog handles prefill validation.
          if (cand.layout === "subdirs") {
            // We could resolve the actual subdir paths here, but the IPC
            // round-trip on the next dialog open will do that. For UX speed
            // we hand the candidate path through and let inspect-dataset-folder
            // get called once more inside the dialog flow.
            void window.electronAPI.invoke("inspect-dataset-folder", cand.path).then((resolved) => {
              if (resolved?.ok && resolved.layout === "subdirs") {
                setImportPrefill({ ab1Dir: resolved.ab1Dir, gbDir: resolved.gbDir });
              } else {
                setImportPrefill({ ab1Dir: cand.path, gbDir: cand.path });
              }
              setImportDialogOpen(true);
            });
          } else {
            setImportPrefill({ ab1Dir: cand.path, gbDir: cand.path });
            setImportDialogOpen(true);
          }
        }}
      />

      {!onboarding.complete && !settingsOpen && !paletteOpen && !shortcutsOpen && agent.initialized ? (
        <OnboardingCoach language={language} onDismiss={onboarding.finish} />
      ) : null}
    </div>
  );
}
