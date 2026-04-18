import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import { SHORTCUTS } from "../lib/commands/shortcuts";
import "./ShortcutsOverlay.css";

interface Props {
  open: boolean;
  onClose: () => void;
  language: AppLanguage;
}

export function ShortcutsOverlay({ open, onClose, language }: Props) {
  if (!open) return null;
  return (
    <div className="shortcuts-scrim" onMouseDown={onClose} role="presentation">
      <div
        className="shortcuts-overlay"
        role="dialog"
        aria-modal="true"
        aria-label={t(language, "shortcuts.title")}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="shortcuts-head">
          <h3>{t(language, "shortcuts.title")}</h3>
          <button type="button" className="shortcuts-close" onClick={onClose} aria-label="close">
            ×
          </button>
        </header>
        <ul className="shortcuts-list">
          {SHORTCUTS.map((s) => (
            <li key={s.key}>
              <kbd>{s.key}</kbd>
              <span>{t(language, s.actionKey)}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
