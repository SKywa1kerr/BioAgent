import { useEffect, useState, useCallback, useRef } from "react";
import {
  AnalysisContextUpdate,
  AnalysisProgressStage,
  AnalysisProgressState,
  AppLanguage,
  AppSettings,
  AppTheme,
  Sample,
  type DatasetImportState,
} from "./types";
import { t } from "./i18n";
import { TabLayout } from "./components/TabLayout";
import { AgentPanel } from "./components/AgentPanel";
import { HistoryPage } from "./components/HistoryPage";
import { SettingsPage } from "./components/SettingsPage";
import { ResultsWorkbench } from "./components/ResultsWorkbench";
import { resolveDatasetPaths } from "./utils/datasetImport";
import { DEFAULT_ANALYSIS_DECISION_MODE, isAiReviewEnabled, validateAiReviewSettings } from "./utils/analysisPreferences";
import { getDefaultSelectedSampleId } from "./utils/resultSelection";
import "./App.css";
const { invoke, onAnalysisProgress } = window.electronAPI;
const PROGRESS_STAGES: AnalysisProgressStage[] = [
  "preparing",
  "scanning",
  "aligning",
  "aggregating",
  "completed",
];

export function resolveDatasetImportState(
  datasetDir: string,
  existingDirs?: Set<string>
): DatasetImportState {
  return resolveDatasetPaths(datasetDir, existingDirs);
}

function buildTabs(language: AppLanguage) {
  return [
    { id: "analysis", label: t(language, "tabs.analysis") },
    { id: "history", label: t(language, "tabs.history") },
    { id: "settings", label: t(language, "tabs.settings") },
  ];
}

function getProgressCopyKey(stage: AnalysisProgressStage) {
  switch (stage) {
    case "preparing":
      return "analysis.progressPreparing";
    case "scanning":
      return "analysis.progressScanning";
    case "aligning":
      return "analysis.progressAligning";
    case "aggregating":
      return "analysis.progressAggregating";
    case "completed":
      return "analysis.progressCompleted";
    case "failed":
      return "analysis.progressFailed";
    default:
      return "analysis.progressPreparing";
  }
}

function normalizeProgressStage(stage: string | undefined): AnalysisProgressStage {
  switch (stage) {
    case "preparing":
    case "scanning":
    case "aligning":
    case "aggregating":
    case "completed":
    case "failed":
      return stage;
    default:
      return "idle";
  }
}

function getProgressStageOrder(stage: AnalysisProgressStage) {
  const index = PROGRESS_STAGES.indexOf(stage);
  return index === -1 ? 0 : index;
}

type NoticeLevel = "info" | "success" | "warning" | "error";

interface AnalysisNotice {
  level: NoticeLevel;
  title: string;
  detail?: string;
}

interface ActivityItem extends AnalysisNotice {
  id: string;
  timestamp: number;
}

function createActivityItem(level: NoticeLevel, title: string, detail?: string): ActivityItem {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    level,
    title,
    detail,
    timestamp: Date.now(),
  };
}

function formatActivityTime(timestamp: number) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(timestamp);
}

function persistInBackground(task: Promise<unknown>, label: string) {
  void task.catch((error) => {
    console.error(label, error);
  });
}

function App() {
  const [activeTab, setActiveTab] = useState("analysis");
  const [language, setLanguage] = useState<AppLanguage>("zh");
  const [theme, setTheme] = useState<AppTheme>("light");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showProgress, setShowProgress] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgressState>({
    stage: "idle",
    percent: 0,
    processedSamples: null,
    totalSamples: null,
  });

  const [ab1Dir, setAb1Dir] = useState<string | null>(null);
  const [genesDir, setGenesDir] = useState<string | null>(null);
  const [plasmid, setPlasmid] = useState("pet22b");
  const [datasetImport, setDatasetImport] = useState<DatasetImportState | null>(null);
  const [showAdvancedImport, setShowAdvancedImport] = useState(false);
  const [analysisNotice, setAnalysisNotice] = useState<AnalysisNotice | null>(null);
  const [activityLog, setActivityLog] = useState<ActivityItem[]>([]);
  const [lastDatasetDir, setLastDatasetDir] = useState<string | null>(null);
  const progressHideTimerRef = useRef<number | null>(null);

  const readSettings = useCallback(async () => {
    const result = (await invoke("load-settings")) as string | null;
    if (!result) {
      return {} as Record<string, unknown>;
    }

    try {
      return JSON.parse(result) as Record<string, unknown>;
    } catch (error) {
      console.warn("Failed to parse settings payload:", error);
      return {} as Record<string, unknown>;
    }
  }, []);

  const writeSettingsPatch = useCallback(
    async (patch: Record<string, unknown>) => {
      const currentSettings = await readSettings();
      const nextSettings = { ...currentSettings, ...patch };
      await invoke("save-settings", JSON.stringify(nextSettings));
      return nextSettings;
    },
    [readSettings]
  );

  const appendActivity = useCallback((level: NoticeLevel, title: string, detail?: string) => {
    setActivityLog((current) => [createActivityItem(level, title, detail), ...current].slice(0, 6));
  }, []);

  const publishNotice = useCallback(
    (level: NoticeLevel, title: string, detail?: string) => {
      setAnalysisNotice({ level, title, detail });
      appendActivity(level, title, detail);
    },
    [appendActivity]
  );

  useEffect(() => {
    let active = true;

    (async () => {
      try {
        const parsed = (await readSettings()) as {
          language?: AppLanguage;
          theme?: AppTheme;
          recentDatasetDir?: string;
        };
        if (!active) return;

        if (parsed.language === "zh" || parsed.language === "en") {
          setLanguage(parsed.language);
        }
        if (parsed.theme === "light" || parsed.theme === "dark") {
          setTheme(parsed.theme);
        }
        if (typeof parsed.recentDatasetDir === "string" && parsed.recentDatasetDir.trim()) {
          setLastDatasetDir(parsed.recentDatasetDir);
        }
      } catch (error) {
        console.error("Failed to load preferences:", error);
      }
    })();

    return () => {
      active = false;
    };
  }, [readSettings]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    return () => {
      if (progressHideTimerRef.current !== null) {
        window.clearTimeout(progressHideTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const unsubscribe = onAnalysisProgress((payload) => {
      if (progressHideTimerRef.current !== null) {
        window.clearTimeout(progressHideTimerRef.current);
        progressHideTimerRef.current = null;
      }

      const nextStage = normalizeProgressStage(payload.stage);
      setShowProgress(true);
      setAnalysisProgress({
        stage: nextStage,
        percent: payload.percent ?? 0,
        processedSamples:
          typeof payload.processedSamples === "number" ? payload.processedSamples : null,
        totalSamples: typeof payload.totalSamples === "number" ? payload.totalSamples : null,
        sampleId: typeof payload.sampleId === "string" ? payload.sampleId : null,
        message: payload.message,
      });

      if (nextStage === "completed" || nextStage === "failed") {
        progressHideTimerRef.current = window.setTimeout(() => {
          setShowProgress(false);
          setAnalysisProgress({
            stage: "idle",
            percent: 0,
            processedSamples: null,
            totalSamples: null,
            sampleId: null,
          });
          progressHideTimerRef.current = null;
        }, nextStage === "completed" ? 1500 : 2400);
      }
    });

    return () => {
      unsubscribe?.();
    };
  }, []);

  const handleLanguageChange = async (nextLanguage: AppLanguage) => {
    const previousLanguage = language;
    setLanguage(nextLanguage);

    try {
      await writeSettingsPatch({ language: nextLanguage });
    } catch (error) {
      setLanguage(previousLanguage);
      console.error("Failed to persist language:", error);
    }
  };

  const handleThemeChange = async (nextTheme: AppTheme) => {
    const previousTheme = theme;
    setTheme(nextTheme);

    try {
      await writeSettingsPatch({ theme: nextTheme });
    } catch (error) {
      setTheme(previousTheme);
      console.error("Failed to persist theme:", error);
    }
  };

  const tabs = buildTabs(language);
  const getAiValidationMessage = (reason: "missing_api_key" | "missing_base_url" | "missing_model") => {
    if (language === "zh") {
      if (reason === "missing_api_key") return "已启用 AI 复核，但尚未配置 API Key。";
      if (reason === "missing_base_url") return "已启用 AI 复核，但尚未配置 Base URL。";
      return "已启用 AI 复核，但尚未选择模型。";
    }

    if (reason === "missing_api_key") return "AI review is enabled, but no API key is configured.";
    if (reason === "missing_base_url") return "AI review is enabled, but no base URL is configured.";
    return "AI review is enabled, but no model is selected.";
  };

  const describeMissingDatasetFolders = useCallback(
    (missing: DatasetImportState["missing"]) =>
      missing
        .map((item) => (item === "ab1" ? t(language, "dataset.ab1Folder") : t(language, "dataset.gbFolder")))
        .join(" / "),
    [language]
  );

  const applyDatasetImport = useCallback(
    async (folder: string) => {
      const existingDirs = (await invoke("inspect-dataset-directory", folder)) as string[];
      const nextDataset = resolveDatasetImportState(folder, new Set(existingDirs));

      setDatasetImport(nextDataset);
      setAb1Dir(nextDataset.ab1Dir);
      setGenesDir(nextDataset.gbDir);
      setLastDatasetDir(folder);
      persistInBackground(writeSettingsPatch({ recentDatasetDir: folder }), "Failed to persist recent dataset");

      if (!nextDataset.valid || nextDataset.missing.length === 2) {
        publishNotice("error", t(language, "dataset.statusInvalid"), t(language, "dataset.invalid"));
        return;
      }

      if (nextDataset.missing.length > 0) {
        publishNotice(
          "warning",
          t(language, "dataset.statusNeedsAttention"),
          `${t(language, "dataset.importWarning")} · ${t(language, "dataset.missingFolders")}: ${describeMissingDatasetFolders(nextDataset.missing)}`
        );
        return;
      }

      publishNotice("success", t(language, "dataset.statusReady"), t(language, "dataset.importComplete"));
    },
    [describeMissingDatasetFolders, language, publishNotice, writeSettingsPatch]
  );

  const handleReuseLastDataset = useCallback(async () => {
    if (!lastDatasetDir) {
      return;
    }

    try {
      await applyDatasetImport(lastDatasetDir);
    } catch (error) {
      publishNotice(
        "error",
        t(language, "dataset.statusInvalid"),
        error instanceof Error ? error.message : String(error)
      );
    }
  }, [applyDatasetImport, language, lastDatasetDir, publishNotice]);

  const runAnalysis = useCallback(
    async (options: { autoImport?: boolean } = {}) => {
      if (!options.autoImport && !ab1Dir) {
        publishNotice("warning", t(language, "dataset.statusNeedsAttention"), t(language, "analysis.selectAb1First"));
        return;
      }

      try {
        const settings = (await readSettings()) as Partial<AppSettings>;
        const effectiveSettings: AppSettings = {
          llmApiKey: settings.llmApiKey || "",
          llmBaseUrl: settings.llmBaseUrl || "",
          llmModel: settings.llmModel || "deepseek-chat",
          plasmid: settings.plasmid || plasmid,
          qualityThreshold: settings.qualityThreshold || 20,
          analysisDecisionMode: settings.analysisDecisionMode || DEFAULT_ANALYSIS_DECISION_MODE,
          language: settings.language,
          theme: settings.theme,
        };
        const aiValidation = validateAiReviewSettings(effectiveSettings);
        if (!aiValidation.ok) {
          publishNotice("error", t(language, "analysis.analysisFailed"), getAiValidationMessage(aiValidation.reason));
          return;
        }

        if (progressHideTimerRef.current !== null) {
          window.clearTimeout(progressHideTimerRef.current);
          progressHideTimerRef.current = null;
        }

        setShowProgress(true);
        setAnalysisProgress({
          stage: "preparing",
          percent: 0,
          processedSamples: null,
          totalSamples: null,
          sampleId: null,
          message: t(language, "analysis.progressPreparing"),
        });
        setIsAnalyzing(true);
        publishNotice("info", t(language, "analysis.analysisStarted"), ab1Dir || undefined);

        const result = (await invoke(
          "run-analysis",
          ab1Dir,
          genesDir,
          {
            ...options,
            useLLM: isAiReviewEnabled(effectiveSettings),
            plasmid: effectiveSettings.plasmid,
            model: effectiveSettings.llmModel,
          }
        )) as string;

        const data = JSON.parse(result);
        setSamples(data.samples);
        setSelectedId(getDefaultSelectedSampleId());
        publishNotice(
          "success",
          t(language, "analysis.analysisCompleted"),
          `${Array.isArray(data.samples) ? data.samples.length : 0} ${t(language, "app.samples")}`
        );
      } catch (error) {
        setAnalysisProgress({
          stage: "failed",
          percent: 100,
          processedSamples: null,
          totalSamples: null,
          sampleId: null,
          message: error instanceof Error ? error.message : String(error),
        });
        progressHideTimerRef.current = window.setTimeout(() => {
          setShowProgress(false);
          setAnalysisProgress({
            stage: "idle",
            percent: 0,
            processedSamples: null,
            totalSamples: null,
            sampleId: null,
          });
          progressHideTimerRef.current = null;
        }, 2400);
        console.error("Analysis failed:", error);
        publishNotice(
          "error",
          t(language, "analysis.analysisFailed"),
          error instanceof Error ? error.message : String(error)
        );
      } finally {
        setIsAnalyzing(false);
      }
    },
    [ab1Dir, genesDir, language, plasmid, publishNotice, readSettings]
  );

  const handleSelectAb1Dir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (!folder) return;

    setAb1Dir(folder);
    persistInBackground(writeSettingsPatch({ recentAb1Dir: folder }), "Failed to persist recent AB1 path");
    setDatasetImport((current) =>
      current
        ? {
            ...current,
            ab1Dir: folder,
            missing: current.missing.filter((item) => item !== "ab1"),
            valid: Boolean(folder || current.gbDir),
          }
        : current
    );
    publishNotice("info", t(language, "dataset.ab1Selected"), folder);
  };

  const handleSelectGenesDir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (!folder) return;

    setGenesDir(folder);
    persistInBackground(writeSettingsPatch({ recentGbDir: folder }), "Failed to persist recent GB path");
    setDatasetImport((current) =>
      current
        ? {
            ...current,
            gbDir: folder,
            missing: current.missing.filter((item) => item !== "gb"),
            valid: Boolean(current.ab1Dir || folder),
          }
        : current
    );
    publishNotice("info", t(language, "dataset.gbSelected"), folder);
  };

  const handleSelectDatasetDir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (!folder) return;

    try {
      await applyDatasetImport(folder);
    } catch (error) {
      publishNotice(
        "error",
        t(language, "dataset.statusInvalid"),
        error instanceof Error ? error.message : String(error)
      );
    }
  };

  const handleExportExcel = useCallback(async () => {
    if (!samples.length) return;

    publishNotice("info", t(language, "analysis.exportStarted"), ab1Dir || undefined);
    try {
      const result = await invoke("export-excel", samples, ab1Dir);
      if (result) {
        publishNotice(
          "success",
          t(language, "analysis.exportComplete"),
          typeof result === "string" ? result : undefined
        );
        return;
      }

      publishNotice("warning", t(language, "analysis.exportFailed"), t(language, "dataset.statusNeedsAttention"));
    } catch (error) {
      publishNotice(
        "error",
        t(language, "analysis.exportFailed"),
        error instanceof Error ? error.message : String(error)
      );
    }
  }, [ab1Dir, language, publishNotice, samples]);

  const handleAnalysisComplete = useCallback((nextAnalysis: AnalysisContextUpdate) => {
    setAb1Dir(nextAnalysis.sourcePath ?? null);
    setGenesDir(nextAnalysis.genesDir ?? null);
    setPlasmid(nextAnalysis.plasmid);
    setSamples(nextAnalysis.samples);
    setSelectedId(nextAnalysis.selectedSampleId);
  }, []);

  const okCount = samples.filter((sample) => sample.status === "ok").length;
  const issueCount = samples.filter((sample) => sample.status === "wrong").length;
  const pendingCount = samples.filter(
    (sample) => sample.status === "uncertain" || sample.status === "processing"
  ).length;
  const progressPercent = Math.max(0, Math.min(100, analysisProgress.percent ?? 0));
  const progressMessage = analysisProgress.message?.trim() || t(language, getProgressCopyKey(analysisProgress.stage));
  const hasProcessedInfo =
    typeof analysisProgress.processedSamples === "number" &&
    typeof analysisProgress.totalSamples === "number" &&
    analysisProgress.totalSamples > 0;
  const progressSampleId = analysisProgress.sampleId?.trim();
  const activeProgressOrder = getProgressStageOrder(analysisProgress.stage);
  const datasetStatus = !datasetImport
    ? null
    : !datasetImport.valid || datasetImport.missing.length === 2
      ? { level: "error" as const, label: t(language, "dataset.statusInvalid") }
      : datasetImport.missing.length > 0
        ? { level: "warning" as const, label: t(language, "dataset.statusNeedsAttention") }
        : { level: "success" as const, label: t(language, "dataset.statusReady") };
  const currentStatus: AnalysisNotice =
    showProgress && analysisProgress.stage !== "idle"
      ? {
          level:
            analysisProgress.stage === "failed"
              ? "error"
              : analysisProgress.stage === "completed"
                ? "success"
                : "info",
          title: t(language, getProgressCopyKey(analysisProgress.stage)),
          detail: progressMessage,
        }
      : analysisNotice ?? {
          level: "info",
          title: t(language, "analysis.readyTitle"),
          detail: t(language, "analysis.readyBody"),
        };
  const sidebarHeader = (
    <div className="sidebar-stack">
      <div className="brand-block">
        <div className="brand-kicker">{t(language, "app.brandKicker")}</div>
        <div className="logo">BioAgent</div>
      </div>
      <div className="sidebar-chip-grid">
        <div className="context-chip">
          <span className="context-label">{t(language, "app.run")}</span>
          <strong>{`${samples.length} ${t(language, "app.samples")}`}</strong>
        </div>
        <div className="context-chip">
          <span className="context-label">{t(language, "app.pass")}</span>
          <strong>{okCount}</strong>
        </div>
        <div className="context-chip">
          <span className="context-label">{t(language, "app.issues")}</span>
          <strong>{issueCount + pendingCount}</strong>
        </div>
      </div>
    </div>
  );
  const sidebarFooter = (
    <div className="sidebar-stack sidebar-stack-compact">
      <div className="sidebar-preferences">
        <div className="language-switch" aria-label={t(language, "app.languageSwitch")}>
          <button
            type="button"
            className={language === "zh" ? "active" : ""}
            onClick={() => handleLanguageChange("zh")}
            aria-pressed={language === "zh"}
          >
            {t(language, "app.languageZh")}
          </button>
          <button
            type="button"
            className={language === "en" ? "active" : ""}
            onClick={() => handleLanguageChange("en")}
            aria-pressed={language === "en"}
          >
            {t(language, "app.languageEn")}
          </button>
        </div>
        <div className="theme-switch" aria-label={t(language, "app.themeSwitch")}>
          <button
            type="button"
            className={theme === "light" ? "active" : ""}
            onClick={() => handleThemeChange("light")}
            aria-pressed={theme === "light"}
          >
            {t(language, "app.themeLight")}
          </button>
          <button
            type="button"
            className={theme === "dark" ? "active" : ""}
            onClick={() => handleThemeChange("dark")}
            aria-pressed={theme === "dark"}
          >
            {t(language, "app.themeDark")}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="app">
      <TabLayout
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        sidebarHeader={sidebarHeader}
        sidebarFooter={sidebarFooter}
      >
        {activeTab === "analysis" && (
          <>
            <div className="analysis-layout">
              <div className="toolbar">
                <div className="dataset-import-panel">
                  <div className="dataset-import-header">
                    <span className="toolbar-kicker">{t(language, "dataset.ready")}</span>
                    <button className="btn-primary" onClick={handleSelectDatasetDir}>
                      {t(language, "dataset.importDataset")}
                    </button>
                  </div>
                  {datasetImport ? (
                    <div className="dataset-summary">
                      <span className="dataset-chip">
                        {t(language, "dataset.datasetFolder")}: {datasetImport.datasetName}
                      </span>
                      <span className="dataset-chip">
                        {t(language, "dataset.ab1Folder")}:{" "}
                        {datasetImport.ab1Dir ?? t(language, "dataset.notSelected")}
                      </span>
                      <span className="dataset-chip">
                        {t(language, "dataset.gbFolder")}:{" "}
                        {datasetImport.gbDir ?? t(language, "dataset.notSelected")}
                      </span>
                    </div>
                  ) : null}
                  <button
                    type="button"
                    className="toolbar-link-button"
                    onClick={() => setShowAdvancedImport((current) => !current)}
                  >
                    {showAdvancedImport
                      ? t(language, "dataset.hideAdvancedMode")
                      : t(language, "dataset.advancedMode")}
                  </button>
                  {showAdvancedImport ? (
                    <div className="path-selectors advanced-path-selectors">
                      <button onClick={handleSelectAb1Dir} title={t(language, "analysis.importAb1")}>
                        {ab1Dir
                          ? `${t(language, "analysis.importAb1")}: ...${ab1Dir.slice(-18)}`
                          : t(language, "analysis.importAb1")}
                      </button>
                      <button
                        onClick={handleSelectGenesDir}
                        title={t(language, "analysis.importReference")}
                      >
                        {genesDir
                          ? `${t(language, "analysis.importReference")}: ...${genesDir.slice(-18)}`
                          : t(language, "analysis.importReference")}
                      </button>
                      <select value={plasmid} onChange={(e) => setPlasmid(e.target.value)}>
                        <option value="pet22b">pET-22b</option>
                        <option value="pet15b">pET-15b</option>
                        <option value="none">None</option>
                      </select>
                    </div>
                  ) : (
                    <p className="dataset-override-hint">{t(language, "dataset.overrideHint")}</p>
                  )}
                  {lastDatasetDir ? (
                    <div className="dataset-recent-actions">
                      <span className="dataset-recent-label">
                        {t(language, "dataset.lastDataset")}: {lastDatasetDir}
                      </span>
                      <button type="button" className="btn-secondary" onClick={handleReuseLastDataset}>
                        {t(language, "dataset.reuseLastDataset")}
                      </button>
                    </div>
                  ) : null}
                </div>
                <div className="action-buttons">
                  <button
                    className="btn-primary"
                    onClick={() => runAnalysis()}
                    disabled={isAnalyzing || !ab1Dir}
                  >
                    {t(language, "analysis.runAnalysis")}
                  </button>
                  <button
                    className="btn-secondary"
                    onClick={() => void handleExportExcel()}
                    disabled={samples.length === 0}
                  >
                    {t(language, "analysis.exportExcel")}
                  </button>
                </div>
              </div>
              {showProgress ? (
                <section className={`analysis-progress-strip stage-${analysisProgress.stage}`} aria-live="polite">
                  <div className="analysis-progress-layout">
                    <div className="analysis-progress-primary">
                      <div className="analysis-progress-head">
                        <span className="analysis-progress-kicker">{t(language, "analysis.progressKicker")}</span>
                        <strong>{Math.round(progressPercent)}%</strong>
                      </div>
                      <div
                        className="analysis-progress-track"
                        role="progressbar"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={Math.round(progressPercent)}
                      >
                        <div className="analysis-progress-fill" style={{ width: `${progressPercent}%` }} />
                      </div>
                      <div className="analysis-progress-meta">
                        <span className="analysis-progress-message">{progressMessage}</span>
                        <span>{t(language, "analysis.progressModeHint")}</span>
                      </div>
                    </div>
                    <div className="analysis-progress-aside">
                      <article className="analysis-progress-stat">
                        <span>{t(language, "analysis.progressLiveStatusLabel")}</span>
                        <strong>{t(language, getProgressCopyKey(analysisProgress.stage))}</strong>
                      </article>
                      <article className="analysis-progress-stat">
                        <span>{t(language, "analysis.progressProcessedLabel")}</span>
                        <strong>
                          {hasProcessedInfo
                            ? `${analysisProgress.processedSamples}/${analysisProgress.totalSamples}`
                            : "0/0"}
                        </strong>
                      </article>
                      <article className="analysis-progress-stat">
                        <span>{t(language, "analysis.progressCurrentSample")}</span>
                        <strong>{progressSampleId || "-"}</strong>
                      </article>
                    </div>
                  </div>
                  <div className="analysis-progress-steps" aria-hidden="true">
                    {PROGRESS_STAGES.map((stage, index) => {
                      const isCurrent = stage === analysisProgress.stage;
                      const isFailed = analysisProgress.stage === "failed";
                      const isReached =
                        !isFailed &&
                        (stage === "completed"
                          ? analysisProgress.stage === "completed"
                          : activeProgressOrder > index || isCurrent);
                      return (
                        <span
                          key={stage}
                          className={`analysis-progress-step${isCurrent ? " is-current" : ""}${isReached ? " is-reached" : ""}${isFailed ? " is-failed" : ""}`}
                        >
                          {t(language, getProgressCopyKey(stage))}
                        </span>
                      );
                    })}
                  </div>
                </section>
              ) : null}
              <section className="analysis-status-strip" aria-label={t(language, "analysis.activityTitle")}>
                <div className={`analysis-status-inline tone-${currentStatus.level}`}>
                  <span className="analysis-status-kicker">{t(language, "analysis.activityLatest")}</span>
                  <strong>{currentStatus.title}</strong>
                  {currentStatus.detail ? <span className="analysis-status-detail">{currentStatus.detail}</span> : null}
                </div>
                <div className="analysis-status-meta">
                  {datasetStatus ? (
                    <span className={`analysis-status-pill tone-${datasetStatus.level}`}>
                      {datasetStatus.label}
                    </span>
                  ) : null}
                  {datasetImport ? (
                    <span className="analysis-status-pill">
                      {t(language, "dataset.datasetFolder")}: {datasetImport.datasetName}
                    </span>
                  ) : null}
                  {datasetImport && datasetImport.missing.length > 0 ? (
                    <span className="analysis-status-pill">
                      {t(language, "dataset.missingFolders")}: {describeMissingDatasetFolders(datasetImport.missing)}
                    </span>
                  ) : null}
                </div>
                <div className="analysis-status-inline analysis-status-inline-log">
                  <span className="analysis-status-kicker">{t(language, "analysis.activityLogKicker")}</span>
                  {activityLog.length > 0 ? (
                    <ul className="recent-activity-strip">
                      {activityLog.slice(0, 2).map((item) => (
                        <li key={item.id} className={`recent-activity-tag tone-${item.level}`}>
                          <strong>{item.title}</strong>
                          <span>{formatActivityTime(item.timestamp)}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <span className="recent-activity-empty">{t(language, "analysis.activityEmpty")}</span>
                  )}
                </div>
              </section>
              <div className="analysis-workspace">
                <div className="analysis-main">
                  <main className="main-content">
                    <ResultsWorkbench
                      language={language}
                      samples={samples}
                      selectedId={selectedId}
                      onSelect={setSelectedId}
                    >
                      <div className="empty-state">
                        <span className="empty-state-kicker">{t(language, "analysis.emptyKicker")}</span>
                        <h3>{t(language, "analysis.emptyTitle")}</h3>
                        <p>{t(language, "analysis.emptyBody")}</p>
                      </div>
                    </ResultsWorkbench>
                  </main>
                </div>
                <AgentPanel
                  language={language}
                  samples={samples}
                  selectedSampleId={selectedId}
                  sourcePath={ab1Dir}
                  genesDir={genesDir}
                  plasmid={plasmid}
                  onAnalysisComplete={handleAnalysisComplete}
                />
              </div>
            </div>
          </>
        )}
        {activeTab === "history" && <HistoryPage language={language} />}
        {activeTab === "settings" && (
          <SettingsPage
            language={language}
            theme={theme}
            onLanguageChange={handleLanguageChange}
            onThemeChange={handleThemeChange}
          />
        )}
      </TabLayout>
    </div>
  );
}

export default App;










