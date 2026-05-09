import { motion, useReducedMotion } from "framer-motion";
import { Atom } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { AgentSettings } from "../lib/settingsStorage";
import { applyProviderSwitch, getProvider, type ProviderId } from "../lib/providers";
import { t, type AppLanguage } from "../i18n";
import { ProviderSelect } from "./InitDialog/ProviderSelect";
import "./InitDialog.css";

interface InitDialogProps {
  open: boolean;
  initialSettings: AgentSettings;
  language: AppLanguage;
  statusMessage: string;
  isInitializing: boolean;
  onSubmit: (settings: AgentSettings) => void;
}

export function InitDialog({
  open,
  initialSettings,
  language,
  statusMessage,
  isInitializing,
  onSubmit,
}: InitDialogProps): JSX.Element | null {
  const [settings, setSettings] = useState(initialSettings);
  const [showCustomBaseUrlHint, setShowCustomBaseUrlHint] = useState(false);
  const apiKeyRef = useRef<HTMLInputElement | null>(null);
  const baseUrlRef = useRef<HTMLInputElement | null>(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (open) {
      setSettings(initialSettings);
      setShowCustomBaseUrlHint(false);
      const id = window.setTimeout(() => {
        if (getProvider(initialSettings.provider).requiresApiKey) {
          apiKeyRef.current?.focus();
        } else {
          baseUrlRef.current?.focus();
        }
      }, 80);
      return () => window.clearTimeout(id);
    }
    return undefined;
  }, [open, initialSettings]);

  if (!open) return null;

  const provider = getProvider(settings.provider);
  const canSubmit =
    (provider.requiresApiKey ? settings.llmApiKey.trim().length > 0 : true) &&
    settings.llmBaseUrl.trim().length > 0 &&
    !isInitializing;

  function handleProviderChange(nextId: ProviderId) {
    setSettings((prev) => {
      const r = applyProviderSwitch({
        fromId: prev.provider,
        toId: nextId,
        currentBaseUrl: prev.llmBaseUrl,
        currentModel: prev.llmModel,
      });
      setShowCustomBaseUrlHint(r.baseUrlIsCustom);
      return { ...prev, provider: nextId, llmBaseUrl: r.baseUrl, llmModel: r.model };
    });
  }

  return (
    <div className="init-dialog-scrim" role="dialog" aria-modal="true" aria-labelledby="init-title">
      <motion.div
        className="init-dialog-card"
        initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.97, y: 6 }}
        animate={reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.18, ease: [0.2, 0.7, 0.2, 1] }}
      >
        <div className="init-dialog-icon" aria-hidden="true">
          <Atom size={26} strokeWidth={1.6} />
        </div>

        <h1 id="init-title" className="init-dialog-title">
          {t(language, "init.title")}
        </h1>
        <p className="init-dialog-subtitle">{t(language, "init.subtitle")}</p>

        <form
          className="init-dialog-form"
          onSubmit={(event) => {
            event.preventDefault();
            if (canSubmit) onSubmit(settings);
          }}
        >
          <ProviderSelect
            value={settings.provider}
            language={language}
            onChange={handleProviderChange}
          />

          {provider.requiresApiKey ? (
            <label className="init-dialog-field">
              <span className="init-dialog-field-label">{t(language, "app.field.apiKey")}</span>
              <input
                ref={apiKeyRef}
                type="password"
                value={settings.llmApiKey}
                onChange={(event) => setSettings((prev) => ({ ...prev, llmApiKey: event.target.value }))}
                placeholder="sk-..."
                autoComplete="off"
                spellCheck={false}
              />
            </label>
          ) : null}

          <label className="init-dialog-field">
            <span className="init-dialog-field-label">{t(language, "app.field.baseUrl")}</span>
            <input
              ref={baseUrlRef}
              type="text"
              value={settings.llmBaseUrl}
              onChange={(event) => setSettings((prev) => ({ ...prev, llmBaseUrl: event.target.value }))}
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
              value={settings.llmModel}
              onChange={(event) => setSettings((prev) => ({ ...prev, llmModel: event.target.value }))}
              placeholder={provider.suggestedModels[0] ?? "model-id"}
              spellCheck={false}
              list={`init-dialog-model-suggest-${provider.id}`}
            />
            {provider.suggestedModels.length > 0 ? (
              <datalist id={`init-dialog-model-suggest-${provider.id}`}>
                {provider.suggestedModels.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            ) : null}
          </label>

          <button type="submit" className="init-dialog-submit" disabled={!canSubmit}>
            {isInitializing ? t(language, "app.status.initializing") : t(language, "app.action.init")}
          </button>
        </form>

        <p className="init-dialog-helper">{t(language, "init.helper")}</p>
        {statusMessage ? (
          <p className="init-dialog-status" role="status">
            {statusMessage}
          </p>
        ) : null}
      </motion.div>
    </div>
  );
}
