import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface ConfirmationDialogProps {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  language: AppLanguage;
}

export function ConfirmationDialog({ message, onConfirm, onCancel, language }: ConfirmationDialogProps) {
  return (
    <div className="confirm-overlay">
      <div className="confirm-dialog">
        <h3>{t(language, "confirm.title")}</h3>
        <p>{message}</p>
        <div className="confirm-actions">
          <button className="ghost-button" onClick={onCancel}>{t(language, "confirm.cancel")}</button>
          <button className="primary-button" onClick={onConfirm}>{t(language, "confirm.ok")}</button>
        </div>
      </div>
    </div>
  );
}
