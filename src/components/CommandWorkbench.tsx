import { FormEvent, KeyboardEvent } from "react";
import type {
  AppLanguage,
  CommandWorkbenchPrompt,
  CommandWorkbenchSummaryItem,
} from "../types";
import { t } from "../i18n";
import "./CommandWorkbench.css";

interface CommandWorkbenchProps {
  language: AppLanguage;
  command: string;
  onCommandChange: (nextCommand: string) => void;
  onSubmit: (command: string) => void;
  quickPrompts: CommandWorkbenchPrompt[];
  batchSummary: CommandWorkbenchSummaryItem;
  plasmidSummary: CommandWorkbenchSummaryItem;
  sampleSummary: CommandWorkbenchSummaryItem;
  disabled?: boolean;
}

function SummaryTile({ item }: { item: CommandWorkbenchSummaryItem }) {
  return (
    <article className="command-workbench-summary-tile">
      <span className="command-workbench-summary-label">{item.label}</span>
      <strong className="command-workbench-summary-value">{item.value}</strong>
      {item.hint ? <p className="command-workbench-summary-hint">{item.hint}</p> : null}
    </article>
  );
}

export function CommandWorkbench({
  language,
  command,
  onCommandChange,
  onSubmit,
  quickPrompts,
  batchSummary,
  plasmidSummary,
  sampleSummary,
  disabled = false,
}: CommandWorkbenchProps) {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextCommand = command.trim();
    if (!nextCommand || disabled) {
      return;
    }

    onSubmit(nextCommand);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.nativeEvent.isComposing) {
      return;
    }

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      const nextCommand = command.trim();
      if (!nextCommand || disabled) {
        return;
      }

      onSubmit(nextCommand);
    }
  };

  return (
    <section className="command-workbench" aria-label={t(language, "command.title")}>
      <header className="command-workbench-header">
        <div className="command-workbench-copy">
          <span className="command-workbench-kicker">{t(language, "command.subtitle")}</span>
          <h3>{t(language, "command.title")}</h3>
          <p>{t(language, "command.body")}</p>
        </div>
      </header>

      <div className="command-workbench-layout">
        <form className="command-workbench-composer" onSubmit={handleSubmit}>
          {/* large command input */}
          <label className="command-workbench-label" htmlFor="command-workbench-input">
            {t(language, "command.placeholderLabel")}
          </label>
          <textarea
            id="command-workbench-input"
            className="command-workbench-input"
            value={command}
            onChange={(event) => onCommandChange(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t(language, "command.placeholder")}
            rows={8}
            disabled={disabled}
          />
          <div className="command-workbench-composer-footer">
            <span>{t(language, "command.composerHint")}</span>
            <button type="submit" disabled={disabled || command.trim().length === 0}>
              {t(language, "app.run")}
            </button>
          </div>
        </form>

        <aside className="command-workbench-sidecar">
          <section className="command-workbench-quick-prompts">
            {/* quick prompts */}
            <div className="command-workbench-section-head">
              <span>{t(language, "command.quickPromptsTitle")}</span>
              <strong>{quickPrompts.length}</strong>
            </div>
            <div className="command-workbench-prompt-list">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt.id}
                  type="button"
                  className="command-workbench-prompt"
                  onClick={() => onCommandChange(prompt.command)}
                  disabled={disabled}
                >
                  <span>{prompt.label}</span>
                  <code>{prompt.command}</code>
                </button>
              ))}
            </div>
          </section>

          <section className="command-workbench-summary">
            <SummaryTile item={batchSummary} />
            <SummaryTile item={plasmidSummary} />
            <SummaryTile item={sampleSummary} />
          </section>
        </aside>
      </div>
    </section>
  );
}
