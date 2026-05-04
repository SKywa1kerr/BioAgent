import { useEffect, useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { WorkbenchSample, WorkbenchStatus } from "./types";
import { bucketSampleStatus, formatPercent, countSampleMutations } from "./utils";
import { compactRowView } from "../../lib/workbench/compactRow";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface Props {
  samples: WorkbenchSample[];
  language: AppLanguage;
  density: "compact" | "detailed";
  selectedId: string | null;
  onSelect(id: string): void;
  isFiltered?: boolean;
  onClearFilters?: () => void;
}

interface RowView {
  status: WorkbenchStatus;
  aaPills: string[];
  aaOverflow: number;
  identityPct: string;
  coveragePct: string;
  mutationCount: number;
  reason: string;
}

const ROW_COMPACT = 64;
const ROW_DETAILED = 88;
const OVERSCAN = 6;

export function ResultsTable({
  samples,
  language,
  density,
  selectedId,
  onSelect,
  isFiltered,
  onClearFilters,
}: Props) {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const rowHeight = density === "compact" ? ROW_COMPACT : ROW_DETAILED;
  const virtualizer = useVirtualizer({
    count: samples.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => rowHeight,
    overscan: OVERSCAN,
    getItemKey: (i) => samples[i]?.id ?? i,
  });
  useEffect(() => {
    virtualizer.measure();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [density, samples.length]);

  // Precompute the per-row derived view once per samples array. Without this,
  // every parent re-render (filter, sort, density, theme, scroll-induced state)
  // re-parses aa_changes via JSON.parse and walks the mutation list for every
  // visible row. samples reference is stable across unrelated re-renders
  // because the parent memoises buildResultsView, so this map is rebuilt only
  // when the user actually changes the result set.
  const rowViews = useMemo<RowView[]>(
    () =>
      samples.map((sample) => {
        const compact = compactRowView(sample);
        return {
          status: bucketSampleStatus(sample),
          aaPills: compact.aaPills,
          aaOverflow: compact.aaOverflow,
          identityPct: formatPercent(sample.identity),
          coveragePct: formatPercent(sample.cds_coverage ?? sample.coverage),
          mutationCount: countSampleMutations(sample),
          reason:
            sample.reason ||
            sample.review_reason ||
            sample.auto_reason ||
            sample.llm_reason ||
            "",
        };
      }),
    [samples],
  );

  const items = virtualizer.getVirtualItems();

  return (
    <section className="results-table-panel" aria-label={t(language, "table.title")}>
      <div className="results-section-header results-section-header-compact">
        <div>
          <span className="results-kicker">{t(language, "table.kicker")}</span>
          <h3>{t(language, "table.title")}</h3>
        </div>
      </div>

      <div className="sample-details-list" ref={parentRef}>
        {samples.length === 0 ? (
          <div className="results-table-empty">
            {isFiltered ? (
              <>
                <strong>{t(language, "wb.empty.filtered")}</strong>
                {onClearFilters ? (
                  <button
                    type="button"
                    className="sample-toolbar-button"
                    onClick={onClearFilters}
                  >
                    {t(language, "wb.empty.clear")}
                  </button>
                ) : null}
              </>
            ) : (
              <>
                <strong>{t(language, "table.noDataTitle")}</strong>
                <span>{t(language, "table.noDataBody")}</span>
              </>
            )}
          </div>
        ) : (
          <div style={{ height: virtualizer.getTotalSize(), position: "relative", width: "100%" }}>
            {items.map((v) => {
              const sample = samples[v.index];
              const row = rowViews[v.index];
              if (!sample || !row) return null;
              const isSelected = selectedId === sample.id;
              return (
                <button
                  key={v.key}
                  type="button"
                  data-index={v.index}
                  onClick={() => onSelect(sample.id)}
                  className={`sample-compact-row status-${row.status}${isSelected ? " is-selected" : ""}`}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: rowHeight,
                    transform: `translateY(${v.start}px)`,
                  }}
                >
                  <span className="sample-compact-sid" title={sample.id}>{sample.id}</span>
                  <span className={`sample-compact-status status-${row.status}`}>
                    {t(language, `wb.status.${row.status}`)}
                  </span>
                  <span className="sample-compact-aa">
                    {row.aaPills.length === 0 ? (
                      <span className="sample-compact-aa-empty">-</span>
                    ) : (
                      row.aaPills.map((p) => (
                        <span key={p} className="sample-compact-aa-pill">{p}</span>
                      ))
                    )}
                    {row.aaOverflow > 0 ? (
                      <span className="sample-compact-aa-overflow">+{row.aaOverflow}</span>
                    ) : null}
                  </span>
                  <span className="sample-compact-metric">{row.identityPct}</span>
                  <span className="sample-compact-metric">{row.coveragePct}</span>
                  <span className="sample-compact-metric">{row.mutationCount}</span>
                  <span className="sample-compact-chevron" aria-hidden="true">›</span>
                  {density === "detailed" ? (
                    <span className="sample-compact-subline">
                      {row.reason} · q{sample.avg_qry_quality ?? "-"} · {sample.orientation || "-"}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {samples.length > 0 ? (
        <div className="sample-list-footnote">
          {t(language, "table.showing", {
            visible: Math.min(samples.length, items.length || samples.length),
            total: samples.length,
          })}
        </div>
      ) : null}
    </section>
  );
}
