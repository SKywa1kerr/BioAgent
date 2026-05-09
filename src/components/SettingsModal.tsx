import { useEffect, useState } from "react";
import type { AgentSettings } from "../lib/settingsStorage";
import { applyProviderSwitch, getProvider, type ProviderId } from "../lib/providers";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import { ProviderSelect } from "./InitDialog/ProviderSelect";
import "./InitDialog.css";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (settings: AgentSettings) => void;
  currentSettings: AgentSettings;
  language: AppLanguage;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onToggleLanguage: () => void;
}

export function SettingsModal({
  open, onClose, onSave, currentSettings, language, theme, onToggleTheme, onToggleLanguage,
}: SettingsModalProps) {
  const [draft, setDraft] = useState<AgentSettings>(currentSettings);
  const [showCustomBaseUrlHint, setShowCustomBaseUrlHint] = useState(false);
  const [showApiKeyHint, setShowApiKeyHint] = useState(false);

  useEffect(() => {
    if (open) {
      setDraft(currentSettings);
      setShowCustomBaseUrlHint(false);
      setShowApiKeyHint(false);
    }
  }, [open, currentSettings]);

  if (!open) return null;

  const provider = getProvider(draft.provider);

  function handleProviderChange(nextId: ProviderId) {
    setDraft((prev) => {
      const r = applyProviderSwitch({
        fromId: prev.provider,
        toId: nextId,
        currentBaseUrl: prev.llmBaseUrl,
        currentModel: prev.llmModel,
      });
      setShowCustomBaseUrlHint(r.baseUrlIsCustom);
      setShowApiKeyHint(true);
      return { ...prev, provider: nextId, llmBaseUrl: r.baseUrl, llmModel: r.model };
    });
  }

  function handleSave() {
    onSave(draft);
  }

  const canSave =
    (provider.requiresApiKey ? draft.llmApiKey.trim().length > 0 : true) &&
    draft.llmBaseUrl.trim().length > 0;

  return (
    <div
      className="settings-modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={t(language, "settings.title")}
    >
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{t(language, "settings.title")}</h3>

        <div className="init-dialog-form">
          <ProviderSelect
            value={draft.provider}
            language={language}
            onChange={handleProviderChange}
          />

          {provider.requiresApiKey ? (
            <label className="init-dialog-field">
              <span className="init-dialog-field-label">{t(language, "app.field.apiKey")}</span>
              <input
                type="password"
                value={draft.llmApiKey}
                onChange={(e) => setDraft((prev) => ({ ...prev, llmApiKey: e.target.value }))}
                placeholder="sk-..."
                autoComplete="off"
                spellCheck={false}
              />
              {showApiKeyHint ? (
                <p className="init-dialog-hint">{t(language, "provider.switchHint.apiKey")}</p>
              ) : null}
            </label>
          ) : null}

          <label className="init-dialog-field">
            <span className="init-dialog-field-label">{t(language, "app.field.baseUrl")}</span>
            <input
              type="text"
              value={draft.llmBaseUrl}
              onChange={(e) => setDraft((prev) => ({ ...prev, llmBaseUrl: e.target.value }))}
              placeholder={provider.defaultBaseUrl || "https://your-proxy.example/v1"}
              spellCheck={false}
            />
            {provider.noteI18nKey ? (
              <p className="init-dialog-hint">{t(language, provider.noteI18nKey)}</p>
            ) : null}
            {showCustomBaseUrlHint ? (
              <p className="init-dialog-hint">{t(language, "provider.switchHint.customBaseUrl")}</p>
            ) : null}
          </label>

          <label className="init-dialog-field">
            <span className="init-dialog-field-label">{t(language, "app.field.model")}</span>
            <input
              type="text"
              value={draft.llmModel}
              onChange={(e) => setDraft((prev) => ({ ...prev, llmModel: e.target.value }))}
              placeholder={provider.suggestedModels[0] ?? "model-id"}
              spellCheck={false}
              list={`settings-model-suggest-${provider.id}`}
            />
            {provider.suggestedModels.length > 0 ? (
              <datalist id={`settings-model-suggest-${provider.id}`}>
                {provider.suggestedModels.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            ) : null}
          </label>

          <label className="init-dialog-field">
            <span className="init-dialog-field-label">{t(language, "app.field.maxTokens")}</span>
            <input
              type="number"
              value={draft.maxTokens}
              onChange={(e) => setDraft((prev) => ({ ...prev, maxTokens: Number(e.target.value) || 2400 }))}
              min={256}
              max={8192}
            />
          </label>

          <div className="settings-quick-row">
            <button
              type="button"
              className="settings-quick-btn"
              onClick={onToggleTheme}
              title={theme === "dark" ? t(language, "app.theme.light") : t(language, "app.theme.dark")}
            >
              {theme === "dark" ? "☼ " : "☾ "}
              {theme === "dark" ? t(language, "app.theme.light") : t(language, "app.theme.dark")}
            </button>
            <button
              type="button"
              className="settings-quick-btn"
              onClick={onToggleLanguage}
              title={t(language, "app.lang")}
            >
              {language === "zh" ? "中 → EN" : "EN → 中"}
            </button>
          </div>

          <div className="settings-actions">
            <button type="button" className="ghost-button" onClick={onClose}>
              {t(language, "settings.cancel")}
            </button>
            <button
              type="button"
              className="primary-button"
              onClick={handleSave}
              disabled={!canSave}
            >
              {t(language, "settings.save")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
