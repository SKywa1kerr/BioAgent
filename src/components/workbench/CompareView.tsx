import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import type { WorkbenchSample, WorkbenchMutation } from "./types";
import { bucketSampleStatus, formatPercent } from "./utils";
import { buildAlignmentViewModel, parseAaChanges } from "./alignmentView";
import { buildChromatogramData } from "./normalize";
import { SequenceAlignmentView } from "./SequenceAlignmentView";
import { Icon } from "../ui/Icon";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import { diffMutations, type MutationDiffSet } from "../../lib/workbench/diffMutations";
import "./CompareView.css";

// Mirrors DetailDrawer's lazy loader: the canvas pulls in chromatogramRender
// (and its bundle of trace-drawing math), and the compare view shows two of
// them, so deferring the import keeps the initial bundle slim.
const ChromatogramCanvas = lazy(async () => {
  const mod = await import("./ChromatogramCanvas");
  return { default: mod.ChromatogramCanvas };
});

// Builds the className + data-unique attributes for a mutation row, blending
// the existing effect-based styling (synonymous / single_read) with the diff
// classification (unique-to-this-side, shared, or unrelated). When the diff
// toggle is off (`diffSet` is null) the existing effect-based class survives
// unchanged, matching the DetailDrawer rendering.
function mutationRowAttrs(
  m: WorkbenchMutation,
  side: "left" | "right",
  diffSet: MutationDiffSet | null,
): { className?: string; "data-unique"?: "left" | "right" | "shared" } {
  const effectClass =
    m.effect === "synonymous"
      ? "is-synonymous"
      : m.effect === "single_read"
      ? "is-single-read"
      : null;

  let diffClass: string | null = null;
  let dataUnique: "left" | "right" | "shared" | undefined;
  if (diffSet && typeof m.position === "number" && Number.isFinite(m.position)) {
    if (diffSet.sharedPositions.has(m.position)) {
      diffClass = "is-mutation-shared";
      dataUnique = "shared";
    } else if (side === "left" && diffSet.uniqueLeftPositions.has(m.position)) {
      diffClass = "is-mutation-unique";
      dataUnique = "left";
    } else if (side === "right" && diffSet.uniqueRightPositions.has(m.position)) {
      diffClass = "is-mutation-unique";
      dataUnique = "right";
    }
  }

  const classes = [effectClass, diffClass].filter(Boolean).join(" ") || undefined;
  return { className: classes, "data-unique": dataUnique };
}

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
  side: "left" | "right";
  diffSet: MutationDiffSet | null;
}

function CompareColumn({ sample, language, diffOn, side, diffSet }: ColumnProps) {
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
            <table className="compare-table compare-mutation-table">
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
                    {...mutationRowAttrs(m, side, diffOn ? diffSet : null)}
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
  const [diffOn, setDiffOn] = useState(false);

  // Pull mutation refs out so the memo dep array is stable across re-renders
  // that don't change the underlying samples (toggle flips, theme changes).
  const leftMutations = left.mutations;
  const rightMutations = right.mutations;
  const diffSet = useMemo<MutationDiffSet>(
    () => diffMutations(leftMutations ?? [], rightMutations ?? []),
    [leftMutations, rightMutations],
  );

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
        <div className="compare-diff-summary" role="status">
          {t(language, "compare.diffSummary", {
            left: diffSet.uniqueLeftPositions.size,
            right: diffSet.uniqueRightPositions.size,
            shared: diffSet.sharedPositions.size,
          })}
        </div>
      ) : null}

      <div className="compare-view-grid">
        <CompareColumn
          sample={left}
          language={language}
          diffOn={diffOn}
          side="left"
          diffSet={diffSet}
        />
        <div className="compare-view-divider" aria-hidden="true" />
        <CompareColumn
          sample={right}
          language={language}
          diffOn={diffOn}
          side="right"
          diffSet={diffSet}
        />
      </div>
    </motion.div>
  );
}
