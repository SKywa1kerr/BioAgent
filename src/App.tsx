import { useCallback, useEffect, useRef, useState } from "react";
import { SmartCanvas, type PanelType } from "./components/SmartCanvas";
import { ChatPanel } from "./components/ChatPanel";
import { SettingsModal } from "./components/SettingsModal";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { CommandPalette } from "./components/CommandPalette";
import { AnalysisPanel } from "./components/panels/AnalysisPanel";
import { MutationTrendPanel } from "./components/panels/MutationTrendPanel";
import { LabSuggestionPanel } from "./components/panels/LabSuggestionPanel";
import { ConfirmationDialog } from "./components/panels/ConfirmationDialog";
import { useAgentHarness } from "./hooks/useAgentHarness";
import { registerCommand } from "./lib/commands/registry";
import { loadSettings, saveSettings, type AgentSettings } from "./lib/settingsStorage";
import { t, type AppLanguage } from "./i18n";

/* ── Helpers ────────────────────────────────────────────────────────── */

function getLocalStorageValue<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const saved = window.localStorage.getItem(key);
    if (saved && (allowed as readonly string[]).includes(saved)) return saved as T;
  } catch { /* ignore */ }
  return fallback;
}

/* ── App ────────────────────────────────────────────────────────────── */

export function App() {
  const [language, setLanguage] = useState<AppLanguage>(() => getLocalStorageValue("bioagent-language", ["zh", "en"] as const, "zh"));
  const [theme, setTheme] = useState<"light" | "dark">(() => getLocalStorageValue("bioagent-theme", ["light", "dark"] as const, "dark"));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<AgentSettings>(loadSettings);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [prefillText, setPrefillText] = useState<string | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);

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

  /* ── Panel history cache ──────────────────────────────────────────── */

  const [panelCache, setPanelCache] = useState<Record<string, any>>({});
  const [activeTab, setActiveTab] = useState<PanelType>("text");

  useEffect(() => {
    const type = agent.panelType;
    const payload = agent.panelPayload;
    if (payload && (type === "analysis" || type === "trends" || type === "suggestions")) {
      setPanelCache((prev) => ({ ...prev, [type]: payload }));
      setActiveTab(type);
    }
  }, [agent.panelType, agent.panelPayload]);

  const TAB_TYPES: PanelType[] = ["analysis", "trends", "suggestions"];
  const availableTabs = TAB_TYPES.filter((tab) => panelCache[tab] != null);

  /* ── Persist theme & language ──────────────────────────────────────── */

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { window.localStorage.setItem("bioagent-theme", theme); } catch { /* ignore */ }
  }, [theme]);

  useEffect(() => {
    try { window.localStorage.setItem("bioagent-language", language); } catch { /* ignore */ }
  }, [language]);

  /* ── Global keyboard shortcuts ───────────────────────────────────── */

  const isAnyModalOpen = settingsOpen || !!agent.confirmMessage;

  useEffect(() => {
    function handleGlobalKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey;

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

      // Escape → close settings
      if (e.key === "Escape" && settingsOpen) {
        e.preventDefault();
        setSettingsOpen(false);
        return;
      }

      // Ctrl+L → focus chat input
      if (mod && e.key === "l") {
        e.preventDefault();
        chatInputRef.current?.focus();
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
  }, [settingsOpen, language, agent, isAnyModalOpen]);

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

  function renderCompactProgress() {
    const running = ["run", "thinking", "tool_calls", "tool_call", "tool_result"].includes(agent.progress.phase) && agent.progress.progress < 100;
    if (!running) return null;
    return (
      <div className="compact-progress">
        <div className="compact-progress-label">{agent.progress.label}</div>
        <div className="compact-progress-track">
          <div className="compact-progress-fill" style={{ width: `${Math.max(8, agent.progress.progress)}%` }} />
        </div>
      </div>
    );
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
        <div className="result-panel">
          <div className="detail-card">
            <h3>{t(language, "app.panel.settings")}</h3>
            <div className="settings-form">
              <label>
                <span>{t(language, "app.field.apiKey")}</span>
                <input type="password" value={settings.llmApiKey} onChange={(e) => setSettings((s) => ({ ...s, llmApiKey: e.target.value }))} placeholder="sk-..." />
              </label>
              <label>
                <span>{t(language, "app.field.baseUrl")}</span>
                <input value={settings.llmBaseUrl} onChange={(e) => setSettings((s) => ({ ...s, llmBaseUrl: e.target.value }))} placeholder="https://models.sjtu.edu.cn/api/v1" />
              </label>
              <label>
                <span>{t(language, "app.field.model")}</span>
                <input value={settings.llmModel} onChange={(e) => setSettings((s) => ({ ...s, llmModel: e.target.value }))} placeholder="deepseek-chat" />
              </label>
              <div className="settings-actions">
                <button className="primary-button" onClick={() => handleSettingsSave(settings)}>{t(language, "app.action.init")}</button>
              </div>
              <div className="status-line">{agent.statusMessage}</div>
            </div>
          </div>
          <div className="detail-card progress-card clean-progress-card">
            <h3>{t(language, "app.progress.cardTitle")}</h3>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${agent.progress.progress}%` }} />
            </div>
            <div className="progress-meta clean-progress-meta">
              <span>{agent.progress.label}</span>
            </div>
          </div>
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

    return (
      <div className="detail-card audience-card">
        <h3>{t(language, "app.ready.title")}</h3>
        <p>{t(language, "app.ready.body")}</p>
      </div>
    );
  }

  /* ── Layout ───────────────────────────────────────────────────────── */

  return (
    <div className="app-shell">
      {!isOnline ? <div className="offline-banner">{t(language, "app.offline")}</div> : null}
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
      />

      <main className="canvas-panel" aria-label="Analysis canvas">
        <SmartCanvas title={t(language, "app.canvasTitle")} panelType={activeTab}>
          {renderTabBar()}
          {renderCompactProgress()}
          <ErrorBoundary
            fallbackTitle={t(language, "app.ready.title")}
            retryLabel={language === "zh" ? "重试" : "Retry"}
          >
            {renderPanel()}
          </ErrorBoundary>
        </SmartCanvas>
      </main>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSave={handleSettingsSave}
        currentSettings={settings}
        language={language}
      />

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        language={language}
      />
    </div>
  );
}
