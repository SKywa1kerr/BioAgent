import React, { useEffect, useRef, useState } from "react";
import { AppLanguage, AppSettings, AppTheme } from "../types";
import { t } from "../i18n";
import { DEFAULT_ANALYSIS_DECISION_MODE, isAiReviewEnabled } from "../utils/analysisPreferences";
import "./SettingsPage.css";

const { invoke } = window.electronAPI;

const LLM_MODEL_OPTIONS = [
  { value: "deepseek-chat", label: "DeepSeek V3.2" },
  { value: "deepseek-reasoner", label: "DeepSeek V3.2 Reasoner" },
  { value: "minimax", label: "MiniMax M2.5" },
  { value: "minimax-m2.5", label: "MiniMax M2.5 Alias" },
  { value: "qwen3coder", label: "Qwen3Coder" },
  { value: "qwen3vl", label: "Qwen3VL" },
] as const;

const DEFAULT_SETTINGS: AppSettings = {
  llmApiKey: "",
  llmBaseUrl: "https://models.sjtu.edu.cn/api/v1",
  llmModel: "deepseek-chat",
  plasmid: "pet22b",
  qualityThreshold: 20,
  analysisDecisionMode: "rules",
  language: undefined,
  theme: "light",
};

interface SettingsPageProps {
  language: AppLanguage;
  theme: AppTheme;
  onLanguageChange: (language: AppLanguage) => void | Promise<void>;
  onThemeChange: (theme: AppTheme) => void | Promise<void>;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({
  language,
  theme,
  onLanguageChange,
  onThemeChange,
}) => {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const saveTimerRef = useRef<number | null>(null);

  useEffect(() => {
    let active = true;

    (async () => {
      try {
        const result = (await invoke("load-settings")) as string;
        if (!active) return;

        if (result) {
          setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(result) });
        }
      } catch (e) {
        if (active) {
          console.error("Failed to load settings:", e);
          setLoadError(t(language, "settings.loadError"));
        }
      } finally {
        if (active) setIsLoading(false);
      }
    })();

    return () => {
      active = false;
      if (saveTimerRef.current) {
        window.clearTimeout(saveTimerRef.current);
      }
    };
  }, [language]);

  const updateSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((current) => ({ ...current, [key]: value }));
  };

  const aiEnabled = isAiReviewEnabled(settings);
  const disabledModelSummary =
    language === "zh"
      ? "当前为仅规则模式，不会发起 AI 复核请求。"
      : "Rules-only mode is active, so no AI review requests will be sent.";
  const disabledModelNotice =
    language === "zh"
      ? "如需 AI 参与分析，请先在下方将判定方式切换为“混合 AI 复核”，再填写你自己的 API 配置。"
      : "If you want AI to participate in analysis, switch the decision mode below to Hybrid AI review and provide your own API configuration.";

  const handleSave = async () => {
    try {
      const raw = (await invoke("load-settings")) as string | null;
      const persisted = raw ? (JSON.parse(raw) as Partial<AppSettings>) : {};
      await invoke("save-settings", JSON.stringify({ ...persisted, ...settings }));
      setSaved(true);

      if (saveTimerRef.current) {
        window.clearTimeout(saveTimerRef.current);
      }

      saveTimerRef.current = window.setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      alert(`${t(language, "settings.saveFailed")}: ${e}`);
    }
  };

  const handleReset = () => {
    setSettings((current) => ({ ...DEFAULT_SETTINGS, language: current.language, theme: current.theme }));
    setSaved(false);
  };

  return (
    <div className="settings-page">
      <div className="settings-hero">
        <div>
          <p className="settings-eyebrow">{t(language, "settings.eyebrow")}</p>
          <h2>{t(language, "settings.title")}</h2>
          <p className="settings-intro">{t(language, "settings.intro")}</p>
        </div>
        <div className="settings-status">
          <span className={`status-pill ${saved ? "status-pill--success" : ""}`}>
            {saved
              ? t(language, "settings.saved")
              : isLoading
                ? t(language, "settings.loading")
                : t(language, "settings.ready")}
          </span>
          <span className="settings-status-note">{t(language, "settings.profileNote")}</span>
        </div>
      </div>

      {loadError ? <div className="settings-banner settings-banner--warning">{loadError}</div> : null}

      <div className="settings-grid">
        <section className="settings-card">
          <div className="settings-card__header">
            <div>
              <p className="settings-card__kicker">{t(language, "settings.languageKicker")}</p>
              <h3>{t(language, "settings.languageTitle")}</h3>
            </div>
            <p className="settings-card__summary">{t(language, "settings.languageSummary")}</p>
          </div>

          <div className="settings-field">
            <label htmlFor="language">{t(language, "settings.languageLabel")}</label>
            <select
              id="language"
              value={language}
              onChange={(e) => void onLanguageChange(e.target.value as AppLanguage)}
            >
              <option value="zh">{t(language, "settings.languageOptionZh")}</option>
              <option value="en">{t(language, "settings.languageOptionEn")}</option>
            </select>
            <span className="settings-help">{t(language, "settings.languageHelp")}</span>
          </div>

          <div className="settings-field">
            <label htmlFor="theme">{t(language, "settings.themeLabel")}</label>
            <select
              id="theme"
              value={theme}
              onChange={(e) => void onThemeChange(e.target.value as AppTheme)}
            >
              <option value="light">{t(language, "settings.themeOptionLight")}</option>
              <option value="dark">{t(language, "settings.themeOptionDark")}</option>
            </select>
            <span className="settings-help">{t(language, "settings.themeHelp")}</span>
          </div>
        </section>

        <section className={`settings-card${aiEnabled ? "" : " settings-card--muted"}`}>
          <div className="settings-card__header">
            <div>
              <p className="settings-card__kicker">{t(language, "settings.modelKicker")}</p>
              <h3>{t(language, "settings.modelTitle")}</h3>
            </div>
            <p className="settings-card__summary">
              {aiEnabled ? t(language, "settings.modelSummary") : disabledModelSummary}
            </p>
          </div>

          {!aiEnabled ? <div className="settings-inline-note">{disabledModelNotice}</div> : null}

          <div className="settings-field">
            <label htmlFor="llmApiKey">{t(language, "settings.apiKey")}</label>
            <input
              id="llmApiKey"
              type="password"
              value={settings.llmApiKey}
              onChange={(e) => updateSetting("llmApiKey", e.target.value)}
              placeholder="sk-..."
              autoComplete="off"
              disabled={!aiEnabled}
            />
            <span className="settings-help">{t(language, "settings.apiKeyHelp")}</span>
          </div>

          <div className="settings-field">
            <label htmlFor="llmBaseUrl">{t(language, "settings.baseUrl")}</label>
            <input
              id="llmBaseUrl"
              type="text"
              value={settings.llmBaseUrl}
              onChange={(e) => updateSetting("llmBaseUrl", e.target.value)}
              placeholder="https://api.example.com/v1"
              disabled={!aiEnabled}
            />
            <span className="settings-help">{t(language, "settings.baseUrlHelp")}</span>
          </div>

          <div className="settings-field">
            <label htmlFor="llmModel">{t(language, "settings.modelName")}</label>
            <select
              id="llmModel"
              value={settings.llmModel}
              onChange={(e) => updateSetting("llmModel", e.target.value)}
              disabled={!aiEnabled}
            >
              {LLM_MODEL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label} ({option.value})
                </option>
              ))}
            </select>
            <span className="settings-help">{t(language, "settings.modelNameHelp")}</span>
          </div>
        </section>

        <section className="settings-card">
          <div className="settings-card__header">
            <div>
              <p className="settings-card__kicker">{t(language, "settings.analysisKicker")}</p>
              <h3>{t(language, "settings.analysisTitle")}</h3>
            </div>
            <p className="settings-card__summary">{t(language, "settings.analysisSummary")}</p>
          </div>

          <div className="settings-field">
            <label htmlFor="analysisDecisionMode">{t(language, "settings.analysisDecisionMode")}</label>
            <select
              id="analysisDecisionMode"
              value={settings.analysisDecisionMode || DEFAULT_ANALYSIS_DECISION_MODE}
              onChange={(e) =>
                updateSetting("analysisDecisionMode", e.target.value as AppSettings["analysisDecisionMode"])
              }
            >
              <option value="rules">{t(language, "settings.analysisDecisionModeRules")}</option>
              <option value="hybrid">{t(language, "settings.analysisDecisionModeHybrid")}</option>
            </select>
            <span className="settings-help">{t(language, "settings.analysisDecisionModeHelp")}</span>
          </div>

          <div className="settings-field">
            <label htmlFor="plasmid">{t(language, "settings.plasmid")}</label>
            <select
              id="plasmid"
              value={settings.plasmid}
              onChange={(e) => updateSetting("plasmid", e.target.value)}
            >
              <option value="pet22b">pET-22b</option>
              <option value="pet15b">pET-15b</option>
              <option value="none">None</option>
            </select>
            <span className="settings-help">{t(language, "settings.plasmidHelp")}</span>
          </div>

          <div className="settings-field">
            <div className="settings-field__row">
              <label htmlFor="qualityThreshold">{t(language, "settings.qualityThreshold")}</label>
              <span className="settings-metric">
                {t(language, "settings.phred")} {settings.qualityThreshold}
              </span>
            </div>
            <input
              id="qualityThreshold"
              type="range"
              value={settings.qualityThreshold}
              onChange={(e) => updateSetting("qualityThreshold", parseInt(e.target.value, 10) || 20)}
              min={0}
              max={60}
            />
            <div className="settings-scale">
              <span>0</span>
              <span>60</span>
            </div>
            <span className="settings-help">{t(language, "settings.qualityScaleHelp")}</span>
          </div>
        </section>

        <section className="settings-card settings-card--accent">
          <div className="settings-card__header">
            <div>
              <p className="settings-card__kicker">{t(language, "settings.advancedKicker")}</p>
              <h3>{t(language, "settings.advancedTitle")}</h3>
            </div>
            <p className="settings-card__summary">{t(language, "settings.advancedSummary")}</p>
          </div>

          <div className="settings-note">
            <p>{t(language, "settings.noteOne")}</p>
            <p>{t(language, "settings.noteTwo")}</p>
          </div>

          <div className="settings-actions">
            <button type="button" className="btn-secondary" onClick={handleReset}>
              {t(language, "settings.restoreDefaults")}
            </button>
            <button type="button" className="btn-primary" onClick={handleSave}>
              {saved ? t(language, "settings.saved") : t(language, "settings.saveSettings")}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
};



