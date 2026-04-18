import { useCallback, useEffect, useRef, useState } from "react";
import type { WorkbenchSample } from "./types";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import { runExport, type ExportFormat } from "../../lib/exporters/runExport";
import "./ExportMenu.css";

interface ExportMenuProps {
  samples: WorkbenchSample[];
  filters: { statusFilter: string; searchQuery: string; sortKey: string };
  dataset?: string;
  language: AppLanguage;
}

type Format = ExportFormat;

export function ExportMenu({ samples, filters, dataset, language }: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<Format | null>(null);
  const [message, setMessage] = useState<{ tone: "error" | "info"; text: string } | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const disabled = samples.length === 0;

  useEffect(() => {
    if (!open) return;
    function onMouseDown(ev: MouseEvent) {
      if (rootRef.current && ev.target instanceof Node && !rootRef.current.contains(ev.target)) {
        setOpen(false);
      }
    }
    function onKeyDown(ev: KeyboardEvent) {
      if (ev.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const exportAs = useCallback(
    async (fmt: Format) => {
      setOpen(false);
      setMessage(null);
      setBusy(fmt);
      try {
        await runExport(fmt, {
          samples,
          filters,
          dataset,
          language,
          onWarn: (text) => setMessage({ tone: "info", text }),
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setMessage({ tone: "error", text: t(language, "export.error", { message: msg }) });
      } finally {
        setBusy(null);
      }
    },
    [samples, filters, dataset, language],
  );

  return (
    <div className="export-menu" ref={rootRef}>
      <button
        type="button"
        className="export-menu-trigger"
        disabled={disabled || busy !== null}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        title={disabled ? t(language, "export.empty") : undefined}
      >
        {busy ? "…" : t(language, "export.menu")} ({samples.length})
      </button>
      {open && !disabled ? (
        <ul className="export-menu-list" role="menu">
          <li>
            <button role="menuitem" type="button" onClick={() => exportAs("csv")}>
              {t(language, "export.csv")}
            </button>
          </li>
          <li>
            <button role="menuitem" type="button" onClick={() => exportAs("json")}>
              {t(language, "export.json")}
            </button>
          </li>
          <li>
            <button role="menuitem" type="button" onClick={() => exportAs("pdf")}>
              {t(language, "export.pdf")}
            </button>
          </li>
        </ul>
      ) : null}
      {message ? (
        <div className={`export-menu-message tone-${message.tone}`} role={message.tone === "error" ? "alert" : "status"}>
          {message.text}
        </div>
      ) : null}
    </div>
  );
}
