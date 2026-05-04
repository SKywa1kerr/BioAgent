import { Suspense, lazy, useEffect, useRef, useState } from "react";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, formatPercent } from "./utils";
import { buildAlignmentViewModel, parseAaChanges } from "./alignmentView";
import { buildChromatogramData } from "./normalize";
import { SequenceAlignmentView } from "./SequenceAlignmentView";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import "./DetailDrawer.css";

const ChromatogramCanvas = lazy(async () => {
  const mod = await import("./ChromatogramCanvas");
  return { default: mod.ChromatogramCanvas };
});

const STORAGE_KEY = "bioagent.drawer.width.v1";

function loadWidth(): number {
  try {
    const v = parseInt(localStorage.getItem(STORAGE_KEY) || "", 10);
    return Number.isFinite(v) && v >= 320 && v <= 900 ? v : 480;
  } catch {
    return 480;
  }
}

interface Props {
  sample: WorkbenchSample | null;
  language: AppLanguage;
  onClose(): void;
}

export function DetailDrawer({ sample, language, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const [width, setWidth] = useState<number>(loadWidth);

  useEffect(() => {
    if (!sample) return;
    closeRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && !e.defaultPrevented) {
        e.preventDefault();
        onClose();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sample, onClose]);

  function startDrag(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = width;
    let lastWidth = startW;
    function onMove(ev: MouseEvent) {
      lastWidth = Math.max(320, Math.min(900, startW + (startX - ev.clientX)));
      setWidth(lastWidth);
    }
    function onUp() {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      try {
        localStorage.setItem(STORAGE_KEY, String(lastWidth));
      } catch {
        // ignore quota / disabled storage
      }
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  if (!sample) return null;
  const bucket = bucketSampleStatus(sample);
  const aa = parseAaChanges(sample.aa_changes);
  const chrom = buildChromatogramData(sample);
  const alignmentView = buildAlignmentViewModel(sample);
  const muts = Array.isArray(sample.mutations) ? sample.mutations : [];
  const avgQ = sample.avg_qry_quality ?? sample.avg_quality;

  return (
    <aside className="detail-drawer" role="dialog" aria-modal="false" aria-label={sample.id} style={{ width }}>
      <div className="detail-drawer-resize" onMouseDown={startDrag} aria-hidden="true" />
      <header className="detail-drawer-head">
        <span className="detail-drawer-sid">{sample.id}</span>
        <span className={`detail-drawer-status status-${bucket}`}>
          {t(language, `wb.status.${bucket}`)}
        </span>
        <button
          ref={closeRef}
          className="detail-drawer-close"
          onClick={onClose}
          aria-label={t(language, "wb.drawer.close")}
        >
          ×
        </button>
      </header>
      <div className="detail-drawer-body">
        <section className="detail-drawer-metrics">
          <article>
            <span>{t(language, "table.clone")}</span>
            <strong>{sample.clone || "-"}</strong>
          </article>
          <article>
            <span>{t(language, "table.orientation")}</span>
            <strong>{sample.orientation || "-"}</strong>
          </article>
          <article>
            <span>{t(language, "table.frameshift")}</span>
            <strong>{sample.frameshift ? t(language, "table.yes") : t(language, "table.no")}</strong>
          </article>
          <article>
            <span>{t(language, "table.avgQ")}</span>
            <strong>{typeof avgQ === "number" ? avgQ.toFixed(1) : "-"}</strong>
          </article>
          <article>
            <span>{t(language, "table.identity")}</span>
            <strong>{formatPercent(sample.identity)}</strong>
          </article>
          <article>
            <span>{t(language, "table.coverage")}</span>
            <strong>{formatPercent(sample.cds_coverage ?? sample.coverage)}</strong>
          </article>
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.aaChanges")}</h4>
          {aa.length ? (
            <div className="detail-drawer-aa">{aa.join(" ")}</div>
          ) : (
            <div className="detail-drawer-empty">{t(language, "table.noAa")}</div>
          )}
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.mutationTable")}</h4>
          {muts.length ? (
            <table className="detail-drawer-table">
              <thead>
                <tr>
                  <th>{t(language, "table.pos")}</th>
                  <th>{t(language, "table.ref")}</th>
                  <th>{t(language, "table.query")}</th>
                  <th>{t(language, "table.type")}</th>
                  <th>{t(language, "table.effect")}</th>
                </tr>
              </thead>
              <tbody>
                {muts.map((m, i) => (
                  <tr
                    key={i}
                    className={
                      m.effect === "synonymous"
                        ? "is-synonymous"
                        : m.effect === "single_read"
                        ? "is-single-read"
                        : undefined
                    }
                  >
                    <td>{m.position ?? "-"}</td>
                    <td>{m.refBase ?? "-"}</td>
                    <td>{m.queryBase ?? "-"}</td>
                    <td>{m.type ?? "-"}</td>
                    <td>{m.effect ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="detail-drawer-empty">{t(language, "table.noMutation")}</div>
          )}
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.alignment")}</h4>
          {alignmentView ? (
            <SequenceAlignmentView view={alignmentView} />
          ) : (
            <div className="detail-drawer-empty">{t(language, "table.noAlignment")}</div>
          )}
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.chromatogram")}</h4>
          {chrom ? (
            <Suspense
              fallback={
                <div className="detail-drawer-empty">
                  {t(language, "table.loadingChromatogram")}
                </div>
              }
            >
              <ChromatogramCanvas
                data={chrom}
                startPosition={1}
                endPosition={chrom.baseCalls.length}
                mutations={muts}
              />
            </Suspense>
          ) : (
            <div className="detail-drawer-empty">{t(language, "table.noChromatogram")}</div>
          )}
        </section>
      </div>
    </aside>
  );
}
