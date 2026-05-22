import { useEffect, useState } from "react";
import { FolderOpen } from "lucide-react";
import { t, type AppLanguage } from "../i18n";
import "./InitDialog.css";

interface ImportDatasetDialogProps {
  open: boolean;
  language: AppLanguage;
  prefill: { ab1Dir?: string; gbDir?: string } | null;
  onClose: () => void;
  onSuccess: (label: string) => void;
  onError: (message: string) => void;
}

export function ImportDatasetDialog({
  open,
  language,
  prefill,
  onClose,
  onSuccess,
  onError,
}: ImportDatasetDialogProps): JSX.Element | null {
  const [label, setLabel] = useState("");
  const [ab1Dir, setAb1Dir] = useState("");
  const [gbDir, setGbDir] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (open) {
      setLabel("");
      setAb1Dir(prefill?.ab1Dir ?? "");
      setGbDir(prefill?.gbDir ?? "");
      setSubmitting(false);
    }
  }, [open, prefill]);

  if (!open) return null;

  async function pickFolder(target: "ab1" | "gb") {
    try {
      const resp = await window.electronAPI.invoke("dialog-pick-folder", {
        title: target === "ab1"
          ? t(language, "import.pick.ab1")
          : t(language, "import.pick.gb"),
      });
      if (resp?.canceled) return;
      if (resp?.path) {
        if (target === "ab1") setAb1Dir(resp.path);
        else setGbDir(resp.path);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      onError(message);
    }
  }

  async function handleSubmit() {
    if (!label.trim() || !ab1Dir.trim() || !gbDir.trim()) return;
    setSubmitting(true);
    try {
      const resp = await window.electronAPI.invoke("dataset-register", {
        label: label.trim(),
        ab1Dir: ab1Dir.trim(),
        gbDir: gbDir.trim(),
      });
      if (resp?.ok) {
        onSuccess(label.trim());
      } else {
        onError(resp?.error || "register failed");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      onError(message);
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = label.trim().length > 0 && ab1Dir.trim().length > 0 && gbDir.trim().length > 0 && !submitting;

  return (
    <div
      className="settings-modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={t(language, "import.title")}
    >
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{t(language, "import.title")}</h3>
        <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
          {t(language, "import.body")}
        </p>

        <div className="init-dialog-form">
          <label className="init-dialog-field">
            <span className="init-dialog-field-label">{t(language, "import.label")}</span>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder={t(language, "import.label.placeholder")}
              autoFocus
            />
          </label>

          <label className="init-dialog-field">
            <span className="init-dialog-field-label">{t(language, "import.ab1Dir")}</span>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="text"
                value={ab1Dir}
                onChange={(e) => setAb1Dir(e.target.value)}
                placeholder="C:\\path\\to\\ab1"
                style={{ flex: 1 }}
              />
              <button
                type="button"
                onClick={() => void pickFolder("ab1")}
                className="init-dialog-secondary"
                aria-label={t(language, "import.pick.ab1")}
              >
                <FolderOpen size={14} />
              </button>
            </div>
          </label>

          <label className="init-dialog-field">
            <span className="init-dialog-field-label">{t(language, "import.gbDir")}</span>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="text"
                value={gbDir}
                onChange={(e) => setGbDir(e.target.value)}
                placeholder="C:\\path\\to\\gb"
                style={{ flex: 1 }}
              />
              <button
                type="button"
                onClick={() => void pickFolder("gb")}
                className="init-dialog-secondary"
                aria-label={t(language, "import.pick.gb")}
              >
                <FolderOpen size={14} />
              </button>
            </div>
          </label>
        </div>

        <div className="init-dialog-actions">
          <button type="button" onClick={onClose} className="init-dialog-secondary">
            {t(language, "import.cancel")}
          </button>
          <button
            type="button"
            onClick={() => void handleSubmit()}
            disabled={!canSubmit}
            className="init-dialog-primary"
          >
            {submitting ? t(language, "import.submitting") : t(language, "import.submit")}
          </button>
        </div>
      </div>
    </div>
  );
}
