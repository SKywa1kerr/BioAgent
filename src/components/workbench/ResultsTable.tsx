import { useEffect, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { WorkbenchSample } from "./types";
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
              if (!sample) return null;
              const status = bucketSampleStatus(sample);
              const view = compactRowView(sample);
              const isSelected = selectedId === sample.id;
              const reason =
                sample.reason ||
                sample.review_reason ||
                sample.auto_reason ||
                sample.llm_reason ||
                "";
              return (
                <button
                  key={v.key}
                  type="button"
                  data-index={v.index}
                  onClick={() => onSelect(sample.id)}
                  className={`sample-compact-row status-${status}${isSelected ? " is-selected" : ""}`}
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
                  <span className={`sample-compact-status status-${status}`}>
                    {t(language, `wb.status.${status}`)}
                  </span>
                  <span className="sample-compact-aa">
                    {view.aaPills.length === 0 ? (
                      <span className="sample-compact-aa-empty">-</span>
                    ) : (
                      view.aaPills.map((p) => (
                        <span key={p} className="sample-compact-aa-pill">{p}</span>
                      ))
                    )}
                    {view.aaOverflow > 0 ? (
                      <span className="sample-compact-aa-overflow">+{view.aaOverflow}</span>
                    ) : null}
                  </span>
                  <span className="sample-compact-metric">{formatPercent(sample.identity)}</span>
                  <span className="sample-compact-metric">
                    {formatPercent(sample.cds_coverage ?? sample.coverage)}
                  </span>
                  <span className="sample-compact-metric">{countSampleMutations(sample)}</span>
                  <span className="sample-compact-chevron" aria-hidden="true">›</span>
                  {density === "detailed" ? (
                    <span className="sample-compact-subline">
                      {reason} · q{sample.avg_qry_quality ?? "-"} · {sample.orientation || "-"}
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
