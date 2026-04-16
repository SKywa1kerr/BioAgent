import { useState } from "react";
import type { AgentSettings } from "../lib/settingsStorage";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (settings: AgentSettings) => void;
  currentSettings: AgentSettings;
  language: AppLanguage;
}

export function SettingsModal({ open, onClose, onSave, currentSettings, language }: SettingsModalProps) {
  const [apiKey, setApiKey] = useState(currentSettings.llmApiKey);
  const [baseUrl, setBaseUrl] = useState(currentSettings.llmBaseUrl);
  const [model, setModel] = useState(currentSettings.llmModel);
  const [maxTokens, setMaxTokens] = useState(currentSettings.maxTokens);

  if (!open) return null;

  function handleSave() {
    onSave({ llmApiKey: apiKey, llmBaseUrl: baseUrl, llmModel: model, maxTokens });
  }

  return (
    <div className="settings-modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label={t(language, "settings.title")}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{t(language, "settings.title")}</h3>
        <div className="settings-form">
          <label>
            <span>{t(language, "app.field.apiKey")}</span>
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-..." />
          </label>
          <label>
            <span>{t(language, "app.field.baseUrl")}</span>
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
          </label>
          <label>
            <span>{t(language, "app.field.model")}</span>
            <input value={model} onChange={(e) => setModel(e.target.value)} />
          </label>
          <label>
            <span>{t(language, "app.field.maxTokens")}</span>
            <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value) || 2400)} min={256} max={8192} />
          </label>
          <div className="settings-actions">
            <button className="ghost-button" onClick={onClose}>{t(language, "settings.cancel")}</button>
            <button className="primary-button" onClick={handleSave}>{t(language, "settings.save")}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
