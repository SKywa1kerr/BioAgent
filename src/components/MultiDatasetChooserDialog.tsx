import { Folder, FolderTree } from "lucide-react";
import { t, type AppLanguage } from "../i18n";
import "./InitDialog.css";

export interface DatasetCandidate {
  name: string;
  path: string;
  layout: "subdirs" | "flat";
}

interface Props {
  open: boolean;
  language: AppLanguage;
  candidates: DatasetCandidate[];
  onCancel: () => void;
  onPick: (candidate: DatasetCandidate) => void;
}

/**
 * Shown when the user drops a folder that contains multiple dataset-shaped
 * subfolders (e.g. dragging the whole `data/` directory which holds
 * `base/`, `batch1/`, `pro/`, ...). The user picks which one to import.
 */
export function MultiDatasetChooserDialog({
  open,
  language,
  candidates,
  onCancel,
  onPick,
}: Props): JSX.Element | null {
  if (!open) return null;

  return (
    <div
      className="settings-modal-overlay"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
      aria-label={t(language, "import.chooser.title")}
    >
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <h3>{t(language, "import.chooser.title")}</h3>
        <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
          {t(language, "import.chooser.body")}
        </p>

        <ul
          style={{
            listStyle: "none",
            padding: 0,
            margin: "12px 0",
            maxHeight: 320,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 6,
          }}
        >
          {candidates.map((cand) => (
            <li key={cand.path}>
              <button
                type="button"
                onClick={() => onPick(cand)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 12px",
                  background: "var(--surface-1)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                  cursor: "pointer",
                  textAlign: "left",
                  color: "var(--text-primary)",
                }}
              >
                {cand.layout === "subdirs" ? (
                  <FolderTree size={18} strokeWidth={1.6} aria-hidden />
                ) : (
                  <Folder size={18} strokeWidth={1.6} aria-hidden />
                )}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 500 }}>{cand.name}</div>
                  <div
                    style={{
                      fontSize: 11,
                      color: "var(--text-muted)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {cand.path}
                  </div>
                </div>
                <span
                  style={{
                    fontSize: 11,
                    color: "var(--text-muted)",
                    background: "var(--surface-2)",
                    padding: "2px 8px",
                    borderRadius: 999,
                  }}
                >
                  {cand.layout === "subdirs"
                    ? t(language, "import.chooser.layout.subdirs")
                    : t(language, "import.chooser.layout.flat")}
                </span>
              </button>
            </li>
          ))}
        </ul>

        <div className="settings-modal-actions">
          <button type="button" onClick={onCancel}>
            {t(language, "confirm.cancel")}
          </button>
        </div>
      </div>
    </div>
  );
}
