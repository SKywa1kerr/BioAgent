import { useEffect, useState, useCallback, useRef } from "react";
import {
  AnalysisContextUpdate,
  AnalysisProgressStage,
  AnalysisProgressState,
  AppLanguage,
  AppSettings,
  AppTheme,
  CommandActionId,
  CommandPlanAction,
  CommandPlanSummary,
  CommandTimelineEvent,
  Sample,
  type DatasetImportState,
} from "./types";
import { t } from "./i18n";
import { TabLayout } from "./components/TabLayout";
import { AssistantPage } from "./components/AssistantPage";
import { HistoryPage } from "./components/HistoryPage";
import { SettingsPage } from "./components/SettingsPage";
import { CommandWorkbench } from "./components/CommandWorkbench";
import { ActionPlanCard } from "./components/ActionPlanCard";
import { ExecutionTimeline } from "./components/ExecutionTimeline";
import { ResultsWorkbench } from "./components/ResultsWorkbench";
import { resolveDatasetPaths } from "./utils/datasetImport";
import { DEFAULT_ANALYSIS_DECISION_MODE, isAiReviewEnabled, validateAiReviewSettings } from "./utils/analysisPreferences";
import { executeCommandPlanSequentially } from "./utils/commandExecution";
import { getDefaultSelectedSampleId } from "./utils/resultSelection";
import { getActionDefinition } from "./utils/actionRegistry";
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
    { id: "assistant", label: t(language, "tabs.assistant") },
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

type CommandIntentAction = {
  id: CommandActionId;
  args: Record<string, unknown>;
};

type CommandIntentPlan = {
  summary: string;
  actions: CommandIntentAction[];
  needsConfirmation: boolean;
};

type ResultFilter = {
  status?: string | null;
  clone?: string | null;
  sampleId?: string | null;
  query?: string | null;
};

type CommandExecutionContext = {
  ab1Dir: string | null;
  genesDir: string | null;
  plasmid: string;
  samples: Sample[];
  selectedId: string | null;
  resultFilter: ResultFilter | null;
  lastExportPath: string | null;
};

function asString(value: unknown) {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function formatActionDetail(actionId: CommandActionId, args: Record<string, unknown>) {
  switch (actionId) {
    case "set_plasmid":
      return asString(args.plasmid);
    case "filter_results":
      return (
        asString(args.status) ||
        asString(args.clone) ||
        asString(args.sampleId) ||
        asString(args.query)
      );
    case "open_sample":
      return (
        asString(args.sampleId) ||
        asString(args.id) ||
        asString(args.name) ||
        asString(args.clone)
      );
    case "set_ab1_dir":
      return asString(args.ab1Dir) || asString(args.path);
    case "set_genes_dir":
      return asString(args.genesDir) || asString(args.path);
    case "import_dataset":
      return asString(args.datasetDir) || asString(args.path);
    default:
      return undefined;
  }
}

function buildCommandPlanActions(
  language: AppLanguage,
  actions: CommandIntentAction[],
  needsConfirmation: boolean
) {
  return actions.map<CommandPlanAction>((action) => {
    const definition = getActionDefinition(action.id);

    return {
      id: action.id,
      title: t(language, definition.labelKey),
      detail: formatActionDetail(action.id, action.args),
      status: needsConfirmation ? "pending" : "ready",
      needsConfirmation: definition.needsConfirmation,
    };
  });
}

function buildCommandTimelineEvents(language: AppLanguage, actions: CommandIntentAction[]) {
  return actions.map<CommandTimelineEvent>((action, index) => {
    const definition = getActionDefinition(action.id);

    return {
      id: `${action.id}-${index}`,
      title: t(language, definition.labelKey),
      detail: formatActionDetail(action.id, action.args),
      status: "queued",
    };
  });
}

function applyResultFilter(samples: Sample[], resultFilter: ResultFilter | null) {
  if (!resultFilter) {
    return samples;
  }

  const status = resultFilter.status?.trim().toLowerCase();
  const clone = resultFilter.clone?.trim().toLowerCase();
  const sampleId = resultFilter.sampleId?.trim().toLowerCase();
  const query = resultFilter.query?.trim().toLowerCase();

  return samples.filter((sample) => {
    if (status && sample.status.toLowerCase() !== status) {
      return false;
    }

    if (clone && !sample.clone.toLowerCase().includes(clone)) {
      return false;
    }

    if (sampleId && !sample.id.toLowerCase().includes(sampleId)) {
      return false;
    }

    if (query && !`${sample.id} ${sample.name} ${sample.clone}`.toLowerCase().includes(query)) {
      return false;
    }

    return true;
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
  const [commandDraft, setCommandDraft] = useState("");
  const [commandPlan, setCommandPlan] = useState<CommandIntentPlan | null>(null);
  const [commandPlanSummary, setCommandPlanSummary] = useState<CommandPlanSummary | null>(null);
  const [commandActions, setCommandActions] = useState<CommandPlanAction[]>([]);
  const [executionEvents, setExecutionEvents] = useState<CommandTimelineEvent[]>([]);
  const [resultFilter, setResultFilter] = useState<ResultFilter | null>(null);
  const [awaitingCommandConfirmation, setAwaitingCommandConfirmation] = useState(false);
  const [isInterpretingCommand, setIsInterpretingCommand] = useState(false);
  const [isExecutingCommand, setIsExecutingCommand] = useState(false);
  const [lastExportPath, setLastExportPath] = useState<string | null>(null);
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
    async (
      options: { autoImport?: boolean } = {},
      overrides: { ab1Dir?: string | null; genesDir?: string | null; plasmid?: string } = {}
    ) => {
      const nextAb1Dir = overrides.ab1Dir ?? ab1Dir;
      const nextGenesDir = overrides.genesDir ?? genesDir;
      const nextPlasmid = overrides.plasmid ?? plasmid;

      if (!options.autoImport && !nextAb1Dir) {
        alert(t(language, "analysis.selectAb1First"));
        return null;
      }

      try {
        const settingsRaw = (await invoke("load-settings")) as string | null;
        const settings = settingsRaw ? (JSON.parse(settingsRaw) as AppSettings) : null;
        const effectiveSettings: AppSettings = {
          llmApiKey: settings?.llmApiKey || "",
          llmBaseUrl: settings?.llmBaseUrl || "",
          llmModel: settings?.llmModel || "deepseek-chat",
          plasmid: nextPlasmid,
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
          nextAb1Dir,
          nextGenesDir,
          {
            ...options,
            useLLM: isAiReviewEnabled(effectiveSettings),
            plasmid: nextPlasmid,
            model: effectiveSettings.llmModel,
          }
        )) as string;

        const data = JSON.parse(result);
        setSamples(data.samples);
        setSelectedId(getDefaultSelectedSampleId());
        return data.samples as Sample[];
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
        return null;
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

  const exportCurrentReport = useCallback(
    async (sourceSamples: Sample[] = samples, sourcePath: string | null = ab1Dir) => {
      if (sourceSamples.length === 0) {
        return null;
      }

      const result = (await invoke("export-excel", sourceSamples, sourcePath)) as
        | { exported?: string | null }
        | null;
      const exportedPath =
        typeof result?.exported === "string" && result.exported.trim().length > 0
          ? result.exported.trim()
          : null;

      if (exportedPath) {
        setLastExportPath(exportedPath);
      }
      return exportedPath;
    },
    [ab1Dir, samples]
  );

  const executeCommandPlan = useCallback(
    async (plan: CommandIntentPlan) => {
      if (plan.actions.length === 0) {
        return;
      }

      const executionContext: CommandExecutionContext = {
        ab1Dir,
        genesDir,
        plasmid,
        samples,
        selectedId,
        resultFilter,
        lastExportPath,
      };

      setAwaitingCommandConfirmation(false);
      setIsExecutingCommand(true);
      setCommandActions((current) =>
        current.map((action) => ({
          ...action,
          status: action.status === "done" ? "done" : "ready",
        }))
      );

      try {
        await executeCommandPlanSequentially<CommandExecutionContext, CommandActionId>(plan, {
          context: executionContext,
          executeAction: async (action, context) => {
            switch (action.id) {
              case "import_dataset": {
                const folder =
                  asString(action.args.datasetDir) ||
                  asString(action.args.path) ||
                  ((await invoke("open-folder-dialog")) as string | null);
                if (!folder) {
                  throw new Error("Dataset import was cancelled.");
                }

                const existingDirs = (await invoke("inspect-dataset-directory", folder)) as string[];
                const nextDataset = resolveDatasetImportState(folder, new Set(existingDirs));
                setDatasetImport(nextDataset);
                setAb1Dir(nextDataset.ab1Dir);
                setGenesDir(nextDataset.gbDir);
                context.ab1Dir = nextDataset.ab1Dir;
                context.genesDir = nextDataset.gbDir;

                if (!nextDataset.valid || nextDataset.missing.length === 2) {
                  throw new Error(t(language, "dataset.invalid"));
                }
                if (nextDataset.missing[0] === "ab1") {
                  throw new Error(t(language, "dataset.missingAb1"));
                }
                if (nextDataset.missing[0] === "gb") {
                  throw new Error(t(language, "dataset.missingGb"));
                }
                return nextDataset.datasetName;
              }
              case "set_ab1_dir": {
                const folder =
                  asString(action.args.ab1Dir) ||
                  asString(action.args.path) ||
                  ((await invoke("open-folder-dialog")) as string | null);
                if (!folder) {
                  throw new Error("AB1 directory selection was cancelled.");
                }

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
                context.ab1Dir = folder;
                return folder;
              }
              case "set_genes_dir": {
                const folder =
                  asString(action.args.genesDir) ||
                  asString(action.args.path) ||
                  ((await invoke("open-folder-dialog")) as string | null);
                if (!folder) {
                  throw new Error("Genes directory selection was cancelled.");
                }

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
                context.genesDir = folder;
                return folder;
              }
              case "set_plasmid": {
                const nextPlasmid = asString(action.args.plasmid);
                if (!nextPlasmid) {
                  throw new Error("No plasmid value was provided.");
                }

                setPlasmid(nextPlasmid);
                context.plasmid = nextPlasmid;
                return nextPlasmid;
              }
              case "run_analysis": {
                const nextSamples = await runAnalysis(
                  {},
                  {
                    ab1Dir: context.ab1Dir,
                    genesDir: context.genesDir,
                    plasmid: context.plasmid,
                  }
                );

                if (!nextSamples) {
                  throw new Error("Analysis did not return samples.");
                }

                context.samples = nextSamples;
                context.selectedId = getDefaultSelectedSampleId();
                return `${nextSamples.length} ${t(language, "app.samples")}`;
              }
              case "filter_results": {
                const nextFilter: ResultFilter = {
                  status: asString(action.args.status),
                  clone: asString(action.args.clone),
                  sampleId: asString(action.args.sampleId),
                  query: asString(action.args.query),
                };
                setResultFilter(nextFilter);
                context.resultFilter = nextFilter;
                return (
                  nextFilter.status ||
                  nextFilter.clone ||
                  nextFilter.sampleId ||
                  nextFilter.query ||
                  t(language, "command.statusDone")
                );
              }
              case "open_sample": {
                const sampleToken =
                  asString(action.args.sampleId) ||
                  asString(action.args.id) ||
                  asString(action.args.name) ||
                  asString(action.args.clone);
                if (!sampleToken) {
                  throw new Error("No sample identifier was provided.");
                }

                const matchedSample = context.samples.find((sample) => {
                  const token = sampleToken.toLowerCase();
                  return (
                    sample.id.toLowerCase() === token ||
                    sample.name.toLowerCase() === token ||
                    sample.clone.toLowerCase() === token
                  );
                });

                if (!matchedSample) {
                  throw new Error(`Sample not found: ${sampleToken}`);
                }

                setSelectedId(matchedSample.id);
                context.selectedId = matchedSample.id;
                return matchedSample.id;
              }
              case "export_report": {
                const filteredSamples = applyResultFilter(context.samples, context.resultFilter);
                if (filteredSamples.length === 0) {
                  throw new Error("There are no samples available to export.");
                }

                const exportedPath = await exportCurrentReport(filteredSamples, context.ab1Dir);
                if (!exportedPath) {
                  throw new Error("Export was cancelled.");
                }

                context.lastExportPath = exportedPath;
                return `${t(language, "command.exportedReady")} ${exportedPath}`;
              }
              case "open_export_folder": {
                if (!context.lastExportPath) {
                  throw new Error("No exported report is available yet.");
                }

                const opened = await window.electronAPI.openExportFolder(context.lastExportPath);
                if (!opened) {
                  throw new Error("Open export folder was cancelled.");
                }
                return `${t(language, "command.openFolder")} ${context.lastExportPath}`;
              }
              default:
                throw new Error(`Unsupported action: ${action.id}`);
            }
          },
          onActionStart: (_action, index) => {
            setCommandActions((current) =>
              current.map((item, itemIndex) => {
                if (itemIndex < index && item.status !== "failed") {
                  return { ...item, status: "done" };
                }
                if (itemIndex === index) {
                  return { ...item, status: "running" };
                }
                return {
                  ...item,
                  status: item.status === "failed" ? "failed" : "ready",
                };
              })
            );
            setExecutionEvents((current) =>
              current.map((event, eventIndex) =>
                eventIndex === index ? { ...event, status: "running" } : event
              )
            );
          },
          onActionSuccess: (_action, index, detail) => {
            setCommandActions((current) =>
              current.map((item, itemIndex) =>
                itemIndex === index
                  ? { ...item, status: "done", detail: detail || item.detail }
                  : item
              )
            );
            setExecutionEvents((current) =>
              current.map((event, eventIndex) =>
                eventIndex === index
                  ? { ...event, status: "done", detail: detail || event.detail }
                  : event
              )
            );
          },
          onActionFailure: (_action, index, error) => {
            const detail = error instanceof Error ? error.message : String(error);
            setCommandActions((current) =>
              current.map((item, itemIndex) =>
                itemIndex === index ? { ...item, status: "failed", detail } : item
              )
            );
            setExecutionEvents((current) =>
              current.map((event, eventIndex) =>
                eventIndex === index ? { ...event, status: "failed", detail } : event
              )
            );
          },
        });
      } finally {
        setIsExecutingCommand(false);
      }
    },
    [
      ab1Dir,
      exportCurrentReport,
      genesDir,
      language,
      lastExportPath,
      plasmid,
      resultFilter,
      runAnalysis,
      samples,
      selectedId,
    ]
  );

  const handleCommandSubmit = useCallback(
    async (nextCommand: string) => {
      const normalizedCommand = nextCommand.trim();
      if (!normalizedCommand || isInterpretingCommand || isExecutingCommand) {
        return;
      }

      let hasStartedExecution = false;
      setIsInterpretingCommand(true);
      setCommandDraft(normalizedCommand);

      try {
        const plan = (await window.electronAPI.interpretCommand(normalizedCommand)) as CommandIntentPlan;
        const nextSummary: CommandPlanSummary = {
          title: normalizedCommand,
          body: plan.summary,
        };
        const nextActions = buildCommandPlanActions(language, plan.actions, plan.needsConfirmation);
        const nextTimeline = buildCommandTimelineEvents(language, plan.actions);

        setCommandPlan(plan);
        setCommandPlanSummary(nextSummary);
        setCommandActions(nextActions);
        setExecutionEvents(nextTimeline);
        setAwaitingCommandConfirmation(plan.needsConfirmation && plan.actions.length > 0);

        if (!plan.needsConfirmation && plan.actions.length > 0) {
          hasStartedExecution = true;
          await executeCommandPlan(plan);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!hasStartedExecution) {
          setCommandPlan(null);
          setCommandPlanSummary({
            title: normalizedCommand,
            body: message,
          });
          setCommandActions([]);
          setExecutionEvents([]);
          setAwaitingCommandConfirmation(false);
        }
      } finally {
        setIsInterpretingCommand(false);
      }
    },
    [executeCommandPlan, isExecutingCommand, isInterpretingCommand, language]
  );

  const handleCancelCommandPlan = useCallback(() => {
    setAwaitingCommandConfirmation(false);
    setCommandPlan(null);
    setCommandPlanSummary(null);
    setCommandActions([]);
    setExecutionEvents([]);
  }, []);

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
  const filteredSamples = applyResultFilter(samples, resultFilter);
  const selectedResultId = filteredSamples.some((sample) => sample.id === selectedId)
    ? selectedId
    : getDefaultSelectedSampleId();
  const progressPercent = Math.max(0, Math.min(100, analysisProgress.percent ?? 0));
  const progressMessage = analysisProgress.message?.trim() || t(language, getProgressCopyKey(analysisProgress.stage));
  const hasProcessedInfo =
    typeof analysisProgress.processedSamples === "number" &&
    typeof analysisProgress.totalSamples === "number" &&
    analysisProgress.totalSamples > 0;
  const progressSampleId = analysisProgress.sampleId?.trim();
  const activeProgressOrder = getProgressStageOrder(analysisProgress.stage);
  const quickPrompts = [
    {
      id: "analyze-wrong",
      label: language === "zh" ? "运行分析并只看异常样本" : "Run analysis and show abnormal samples",
      command: language === "zh" ? "运行分析，只看 wrong 样本" : "run analysis and filter wrong samples",
    },
    {
      id: "import-run",
      label: language === "zh" ? "导入数据集后开始分析" : "Import a dataset and run analysis",
      command: language === "zh" ? "导入新的数据集并开始分析" : "import a dataset and run analysis",
    },
    {
      id: "export-report",
      label: language === "zh" ? "导出并打开报告目录" : "Export and open the report folder",
      command: language === "zh" ? "导出当前报告并打开导出目录" : "export the current report and open the export folder",
    },
  ];
  const batchSummary = {
    label: language === "zh" ? "当前批次" : "Current batch",
    value: datasetImport?.datasetName ?? (ab1Dir ? ab1Dir.split(/[/\\]/).pop() || ab1Dir : t(language, "dataset.notSelected")),
    hint: datasetImport?.datasetDir ?? ab1Dir ?? t(language, "analysis.empty"),
  };
  const plasmidSummary = {
    label: language === "zh" ? "质粒模板" : "Plasmid",
    value: plasmid,
    hint: genesDir ?? t(language, "analysis.importReference"),
  };
  const sampleSummary = {
    label: language === "zh" ? "结果过滤" : "Result filter",
    value: resultFilter ? `${filteredSamples.length}/${samples.length}` : `${samples.length}`,
    hint:
      resultFilter?.status ||
      resultFilter?.clone ||
      resultFilter?.sampleId ||
      resultFilter?.query ||
      (language === "zh" ? "显示全部样本" : "Showing all samples"),
  };

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
                      const exportedPath = await exportCurrentReport();
                      if (exportedPath) {
                        alert(t(language, "analysis.exportComplete"));
                      }
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
            <div className="analysis-workspace analysis-workspace--solo">
              <div className="analysis-main">
                <section className="analysis-command-stage">
                  <div className="analysis-command-stack">
                    <CommandWorkbench
                      language={language}
                      command={commandDraft}
                      onCommandChange={setCommandDraft}
                      onSubmit={(command) => {
                        void handleCommandSubmit(command);
                      }}
                      quickPrompts={quickPrompts}
                      batchSummary={batchSummary}
                      plasmidSummary={plasmidSummary}
                      sampleSummary={sampleSummary}
                      disabled={isInterpretingCommand || isExecutingCommand}
                    />
                    {commandPlanSummary ? (
                      <div className="analysis-command-output-grid">
                        <ActionPlanCard
                          language={language}
                          planSummary={commandPlanSummary}
                          actions={commandActions}
                          needsConfirmation={awaitingCommandConfirmation}
                          onConfirm={
                            awaitingCommandConfirmation && commandPlan
                              ? () => {
                                  void executeCommandPlan(commandPlan);
                                }
                              : undefined
                          }
                          onCancel={awaitingCommandConfirmation ? handleCancelCommandPlan : undefined}
                        />
                        {executionEvents.length > 0 ? (
                          <ExecutionTimeline language={language} events={executionEvents} />
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </section>
                <section className="analysis-main-surface">
                  <main className="main-content">
                    <ResultsWorkbench
                      language={language}
                      samples={filteredSamples}
                      selectedId={selectedResultId}
                      onSelect={setSelectedId}
                    >
                      <div className="empty-state">
                        <span className="empty-state-kicker">{t(language, "analysis.emptyKicker")}</span>
                        <h3>{t(language, "analysis.emptyTitle")}</h3>
                        <p>{t(language, "analysis.emptyBody")}</p>
                      </div>
                    </ResultsWorkbench>
                  </main>
                </section>
              </div>
            </div>
          </>
        )}
        {activeTab === "assistant" && (
          <AssistantPage
            language={language}
            samples={samples}
            selectedSampleId={selectedId}
            sourcePath={ab1Dir}
            genesDir={genesDir}
            plasmid={plasmid}
            onAnalysisComplete={handleAnalysisComplete}
          />
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










