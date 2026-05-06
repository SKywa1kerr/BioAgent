import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, formatPercent } from "./utils";
import { buildAlignmentViewModel, parseAaChanges } from "./alignmentView";
import { buildChromatogramData } from "./normalize";
import { SequenceAlignmentView } from "./SequenceAlignmentView";
import { Icon } from "../ui/Icon";
import { useToasts } from "../ui/ToastProvider";
import { useSampleOverrides } from "../../hooks/useSampleOverrides";
import type { SampleStatusOverride } from "../../lib/ui/sampleOverrides";
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
  sample: WorkbenchSample;
  language: AppLanguage;
  analysisId?: string | null;
  onClose(): void;
}

const OVERRIDE_OPTIONS: SampleStatusOverride[] = ["ok", "wrong", "uncertain"];

export function DetailDrawer({ sample, language, analysisId, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const [width, setWidth] = useState<number>(loadWidth);
  const reduceMotion = useReducedMotion() ?? false;
  const overridesApi = useSampleOverrides();
  const toasts = useToasts();
  const existingOverride = analysisId
    ? overridesApi.getOverride(analysisId, sample.id)
    : undefined;
  const [overrideOpen, setOverrideOpen] = useState(false);
  const [draftStatus, setDraftStatus] = useState<SampleStatusOverride>(
    existingOverride?.status ?? "uncertain",
  );
  const [draftReason, setDraftReason] = useState<string>(existingOverride?.reason ?? "");

  useEffect(() => {
    setOverrideOpen(false);
    setDraftStatus(existingOverride?.status ?? "uncertain");
    setDraftReason(existingOverride?.reason ?? "");
  }, [sample.id, analysisId, existingOverride?.status, existingOverride?.reason]);

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

  // Derived values are memoised so that re-renders triggered by drawer
  // width drag or theme toggles do not rebuild the alignment coordinate
  // map (O(n) over the gapped sequence) or re-parse aa_changes.
  const aa = useMemo(
    () => (sample ? parseAaChanges(sample.aa_changes) : []),
    [sample],
  );
  const chrom = useMemo(
    () => (sample ? buildChromatogramData(sample) : null),
    [sample],
  );
  const alignmentView = useMemo(
    () => (sample ? buildAlignmentViewModel(sample) : null),
    [sample],
  );

  function startDrag(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = width;
    let lastWidth = startW;
    let pendingFrame: number | null = null;
    function flush() {
      pendingFrame = null;
      setWidth(lastWidth);
    }
    function onMove(ev: MouseEvent) {
      lastWidth = Math.max(320, Math.min(900, startW + (startX - ev.clientX)));
      if (pendingFrame == null) pendingFrame = requestAnimationFrame(flush);
    }
    function onUp() {
      if (pendingFrame != null) {
        cancelAnimationFrame(pendingFrame);
        pendingFrame = null;
        setWidth(lastWidth);
      }
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

  const bucket = bucketSampleStatus(sample);
  const muts = Array.isArray(sample.mutations) ? sample.mutations : [];
  const avgQ = sample.avg_qry_quality ?? sample.avg_quality;

  const initial = reduceMotion ? { opacity: 0 } : { opacity: 0, x: 24 };
  const animate = reduceMotion ? { opacity: 1 } : { opacity: 1, x: 0 };
  const exit = reduceMotion ? { opacity: 0 } : { opacity: 0, x: 24 };

  return (
    <motion.aside
      className="detail-drawer"
      role="dialog"
      aria-modal="false"
      aria-label={sample.id}
      style={{ width }}
      initial={initial}
      animate={animate}
      exit={exit}
      transition={{ duration: 0.14, ease: [0.2, 0.7, 0.2, 1] }}
    >
      <div className="detail-drawer-resize" onMouseDown={startDrag} aria-hidden="true" />
      <header className="detail-drawer-head">
        <span className="detail-drawer-sid">{sample.id}</span>
        <span className={`detail-drawer-status status-${bucket}`}>
          {t(language, `wb.status.${bucket}`)}
          {existingOverride ? (
            <span className="detail-drawer-override-badge">
              {t(language, "override.badge")}
            </span>
          ) : null}
        </span>
        <button
          ref={closeRef}
          className="detail-drawer-close"
          onClick={onClose}
          aria-label={t(language, "wb.drawer.close")}
        >
          <Icon name="close" size={14} />
        </button>
      </header>
      <div className="detail-drawer-body">
        {analysisId ? (
          <section className="detail-drawer-override">
            <button
              type="button"
              className="detail-drawer-override-toggle"
              aria-expanded={overrideOpen}
              onClick={() => setOverrideOpen((v) => !v)}
            >
              {t(language, "override.title")}
            </button>
            {overrideOpen ? (
              <div className="detail-drawer-override-panel">
                <div className="detail-drawer-override-row" role="radiogroup" aria-label={t(language, "override.title")}>
                  {OVERRIDE_OPTIONS.map((opt) => (
                    <button
                      key={opt}
                      type="button"
                      role="radio"
                      aria-checked={draftStatus === opt}
                      className={`detail-drawer-override-option status-${opt}${draftStatus === opt ? " is-active" : ""}`}
                      onClick={() => setDraftStatus(opt)}
                    >
                      {t(language, `wb.status.${opt}`)}
                    </button>
                  ))}
                </div>
                <label className="detail-drawer-override-reason">
                  <span>{t(language, "override.reason")}</span>
                  <input
                    type="text"
                    value={draftReason}
                    onChange={(e) => setDraftReason(e.target.value)}
                    placeholder={t(language, "override.reasonPlaceholder")}
                  />
                </label>
                <div className="detail-drawer-override-actions">
                  <button
                    type="button"
                    className="detail-drawer-override-save"
                    onClick={() => {
                      overridesApi.setOverride(analysisId, sample.id, draftStatus, draftReason);
                      toasts.pushToast({
                        kind: "info",
                        title: t(language, "override.savedToast"),
                        durationMs: 2400,
                      });
                      setOverrideOpen(false);
                    }}
                  >
                    {t(language, "override.save")}
                  </button>
                  {existingOverride ? (
                    <button
                      type="button"
                      className="detail-drawer-override-clear"
                      onClick={() => {
                        overridesApi.clearOverride(analysisId, sample.id);
                        toasts.pushToast({
                          kind: "info",
                          title: t(language, "override.clearedToast"),
                          durationMs: 2400,
                        });
                        setOverrideOpen(false);
                      }}
                    >
                      {t(language, "override.clear")}
                    </button>
                  ) : null}
                </div>
              </div>
            ) : null}
          </section>
        ) : null}
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
                <div
                  className="chromatogram-skeleton"
                  role="status"
                  aria-label={t(language, "table.loadingChromatogram")}
                >
                  <div className="chromatogram-skeleton-row" />
                  <div className="chromatogram-skeleton-row" />
                  <div className="chromatogram-skeleton-row" />
                </div>
              }
            >
              <ChromatogramCanvas
                data={chrom}
                startPosition={1}
                endPosition={chrom.baseCalls.length}
                mutations={muts}
                language={language}
              />
            </Suspense>
          ) : (
            <div className="detail-drawer-empty">{t(language, "table.noChromatogram")}</div>
          )}
        </section>
      </div>
    </motion.aside>
  );
}
