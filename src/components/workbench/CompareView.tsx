import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, formatPercent } from "./utils";
import { buildAlignmentViewModel, parseAaChanges } from "./alignmentView";
import { buildChromatogramData } from "./normalize";
import { SequenceAlignmentView } from "./SequenceAlignmentView";
import { Icon } from "../ui/Icon";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import "./CompareView.css";

// Mirrors DetailDrawer's lazy loader: the canvas pulls in chromatogramRender
// (and its bundle of trace-drawing math), and the compare view shows two of
// them, so deferring the import keeps the initial bundle slim.
const ChromatogramCanvas = lazy(async () => {
  const mod = await import("./ChromatogramCanvas");
  return { default: mod.ChromatogramCanvas };
});

interface CompareViewProps {
  left: WorkbenchSample;
  right: WorkbenchSample;
  language: AppLanguage;
  onClose(): void;
}

interface ColumnProps {
  sample: WorkbenchSample;
  language: AppLanguage;
  diffOn: boolean;
}

function CompareColumn({ sample, language, diffOn: _diffOn }: ColumnProps) {
  // Same memoisation pattern as DetailDrawer: derive once per sample so that
  // unrelated parent re-renders (diff toggle, motion frames, theme) do not
  // re-parse aa_changes or rebuild the gapped alignment coordinate map.
  const aa = useMemo(() => parseAaChanges(sample.aa_changes), [sample]);
  const chrom = useMemo(() => buildChromatogramData(sample), [sample]);
  const alignmentView = useMemo(() => buildAlignmentViewModel(sample), [sample]);

  const bucket = bucketSampleStatus(sample);
  const muts = Array.isArray(sample.mutations) ? sample.mutations : [];
  const avgQ = sample.avg_qry_quality ?? sample.avg_quality;

  return (
    <div className="compare-column" role="group" aria-label={sample.id}>
      <header className="compare-column-head">
        <span className="compare-column-sid">{sample.id}</span>
        <span className={`compare-column-status status-${bucket}`}>
          {t(language, `wb.status.${bucket}`)}
          {sample.override ? (
            <span className="compare-column-override-badge">
              {t(language, "override.badge")}
            </span>
          ) : null}
        </span>
      </header>
      <div className="compare-column-body">
        <section className="compare-metrics">
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
            <strong>
              {sample.frameshift ? t(language, "table.yes") : t(language, "table.no")}
            </strong>
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

        <section className="compare-section">
          <h4>{t(language, "table.aaChanges")}</h4>
          {aa.length ? (
            <div className="compare-aa">{aa.join(" ")}</div>
          ) : (
            <div className="compare-empty">{t(language, "table.noAa")}</div>
          )}
        </section>

        <section className="compare-section">
          <h4>{t(language, "table.mutationTable")}</h4>
          {muts.length ? (
            <table className="compare-table">
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
            <div className="compare-empty">{t(language, "table.noMutation")}</div>
          )}
        </section>

        <section className="compare-section">
          <h4>{t(language, "table.alignment")}</h4>
          {alignmentView ? (
            <SequenceAlignmentView view={alignmentView} />
          ) : (
            <div className="compare-empty">{t(language, "table.noAlignment")}</div>
          )}
        </section>

        <section className="compare-section">
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
            <div className="compare-empty">{t(language, "table.noChromatogram")}</div>
          )}
        </section>
      </div>
    </div>
  );
}

export function CompareView({ left, right, language, onClose }: CompareViewProps) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const reduceMotion = useReducedMotion() ?? false;
  // Diff highlight is wired but not yet implemented. The toggle ships now so
  // the layout is final; the actual mutation-set comparison math is deferred.
  const [diffOn, setDiffOn] = useState(false);

  useEffect(() => {
    closeRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && !e.defaultPrevented) {
        e.preventDefault();
        onClose();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const initial = reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12 };
  const animate = reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0 };
  const exit = reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12 };

  return (
    <motion.div
      className="compare-view"
      role="dialog"
      aria-modal="true"
      aria-label={t(language, "compare.title")}
      initial={initial}
      animate={animate}
      exit={exit}
      transition={{ duration: 0.14, ease: [0.2, 0.7, 0.2, 1] }}
    >
      <header className="compare-view-head">
        <span className="compare-view-title">
          <Icon name="compare" size={16} aria-hidden="true" />
          {t(language, "compare.title")}
        </span>
        <label className="compare-diff-toggle">
          <input
            type="checkbox"
            checked={diffOn}
            onChange={(e) => setDiffOn(e.target.checked)}
          />
          <span>{t(language, "compare.diffHighlight")}</span>
        </label>
        <button
          ref={closeRef}
          type="button"
          className="compare-view-close"
          onClick={onClose}
          aria-label={t(language, "wb.drawer.close")}
        >
          <Icon name="close" size={14} />
        </button>
      </header>

      {diffOn ? (
        <div className="compare-diff-banner" role="status">
          {t(language, "compare.diffPlaceholder")}
        </div>
      ) : null}

      <div className="compare-view-grid">
        <CompareColumn sample={left} language={language} diffOn={diffOn} />
        <div className="compare-view-divider" aria-hidden="true" />
        <CompareColumn sample={right} language={language} diffOn={diffOn} />
      </div>
    </motion.div>
  );
}
