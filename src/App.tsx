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
  const progressHideTimerRef = useRef<number | null>(null);

  useEffect(() => {
    let active = true;

    (async () => {
      try {
        const result = (await invoke("load-settings")) as string;
        if (!active || !result) return;

        const parsed = JSON.parse(result) as { language?: AppLanguage; theme?: AppTheme };
        if (parsed.language === "zh" || parsed.language === "en") {
          setLanguage(parsed.language);
        }
        if (parsed.theme === "light" || parsed.theme === "dark") {
          setTheme(parsed.theme);
        }
      } catch (error) {
        console.error("Failed to load preferences:", error);
      }
    })();

    return () => {
      active = false;
    };
  }, []);

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
      const result = (await invoke("load-settings")) as string;
      let currentSettings: Record<string, unknown> = {};

      if (result) {
        try {
          currentSettings = JSON.parse(result) as Record<string, unknown>;
        } catch (parseError) {
          console.warn("Failed to parse settings while saving language:", parseError);
        }
      }

      await invoke(
        "save-settings",
        JSON.stringify({ ...currentSettings, language: nextLanguage })
      );
    } catch (error) {
      setLanguage(previousLanguage);
      console.error("Failed to persist language:", error);
    }
  };

  const handleThemeChange = async (nextTheme: AppTheme) => {
    const previousTheme = theme;
    setTheme(nextTheme);

    try {
      const result = (await invoke("load-settings")) as string;
      let currentSettings: Record<string, unknown> = {};

      if (result) {
        try {
          currentSettings = JSON.parse(result) as Record<string, unknown>;
        } catch (parseError) {
          console.warn("Failed to parse settings while saving theme:", parseError);
        }
      }

      await invoke("save-settings", JSON.stringify({ ...currentSettings, theme: nextTheme }));
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

  const runAnalysis = useCallback(
    async (options: { autoImport?: boolean } = {}) => {
      if (!options.autoImport && !ab1Dir) {
        alert(t(language, "analysis.selectAb1First"));
        return;
      }

      try {
        const settingsRaw = (await invoke("load-settings")) as string | null;
        const settings = settingsRaw ? (JSON.parse(settingsRaw) as AppSettings) : null;
        const effectiveSettings: AppSettings = {
          llmApiKey: settings?.llmApiKey || "",
          llmBaseUrl: settings?.llmBaseUrl || "",
          llmModel: settings?.llmModel || "deepseek-chat",
          plasmid: settings?.plasmid || plasmid,
          qualityThreshold: settings?.qualityThreshold || 20,
          analysisDecisionMode: settings?.analysisDecisionMode || DEFAULT_ANALYSIS_DECISION_MODE,
          language: settings?.language,
          theme: settings?.theme,
        };
        const aiValidation = validateAiReviewSettings(effectiveSettings);
        if (!aiValidation.ok) {
          alert(getAiValidationMessage(aiValidation.reason));
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
        alert(`${t(language, "analysis.analysisFailed")}: ${error}`);
      } finally {
        setIsAnalyzing(false);
      }
    },
    [ab1Dir, genesDir, language, plasmid]
  );

  const handleSelectAb1Dir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (!folder) return;

    setAb1Dir(folder);
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
  };

  const handleSelectGenesDir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (!folder) return;

    setGenesDir(folder);
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
  };

  const handleSelectDatasetDir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (!folder) return;

    const existingDirs = (await invoke("inspect-dataset-directory", folder)) as string[];
    const nextDataset = resolveDatasetImportState(folder, new Set(existingDirs));

    setDatasetImport(nextDataset);
    setAb1Dir(nextDataset.ab1Dir);
    setGenesDir(nextDataset.gbDir);

    if (!nextDataset.valid || nextDataset.missing.length === 2) {
      alert(t(language, "dataset.invalid"));
      return;
    }

    if (nextDataset.missing[0] === "ab1") {
      alert(t(language, "dataset.missingAb1"));
      return;
    }

    if (nextDataset.missing[0] === "gb") {
      alert(t(language, "dataset.missingGb"));
    }
  };

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

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-kicker">{t(language, "app.brandKicker")}</div>
          <div className="logo">BioAgent</div>
        </div>
          <div className="header-actions">
            <div className="header-preferences">
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
            <div className="header-context">
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
      </header>
      <TabLayout tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab}>
        {activeTab === "analysis" && (
          <>
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
                  onClick={async () => {
                    if (!samples.length) return;
                    try {
                      const result = await invoke("export-excel", samples, ab1Dir) as any;
                      if (result) alert(t(language, "analysis.exportComplete"));
                    } catch (e) {
                      alert(`${t(language, "analysis.exportFailed")}: ${e}`);
                    }
                  }}
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










