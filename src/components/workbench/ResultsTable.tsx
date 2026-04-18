import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, countSampleMutations, formatPercent } from "./utils";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface ResultsTableProps {
  samples: WorkbenchSample[];
  language: AppLanguage;
}

const ROW_ESTIMATE_COLLAPSED = 64;
const ROW_ESTIMATE_EXPANDED = 480;
const OVERSCAN = 6;

const ChromatogramCanvas = lazy(async () => {
  const mod = await import("./ChromatogramCanvas");
  return { default: mod.ChromatogramCanvas };
});

function displayReason(sample: WorkbenchSample) {
  return sample.reason || sample.review_reason || sample.llm_reason || sample.auto_reason || sample.error || "-";
}

function parseAaChanges(value: WorkbenchSample["aa_changes"]): string[] {
  if (Array.isArray(value)) return value.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
    } catch {
      return value.trim() ? [value.trim()] : [];
    }
  }
  return [];
}

function toChromatogramData(sample: WorkbenchSample) {
  if (!sample.traces_a || !sample.traces_t || !sample.traces_g || !sample.traces_c || !sample.query_sequence) return null;
  return {
    traces: { A: sample.traces_a, T: sample.traces_t, G: sample.traces_g, C: sample.traces_c },
    quality: sample.quality || [],
    baseCalls: sample.query_sequence,
    base_locations: sample.base_locations || [],
    mixed_peaks: sample.mixed_peaks || [],
  };
}

function chromatogramUnavailableReason(sample: WorkbenchSample): string {
  if (!sample.query_sequence || sample.query_sequence.length === 0) return "query sequence missing";
  if (!sample.base_locations || sample.base_locations.length === 0) return "base locations missing";
  if (!sample.traces_a || sample.traces_a.length === 0) return "A channel missing";
  if (!sample.traces_t || sample.traces_t.length === 0) return "T channel missing";
  if (!sample.traces_g || sample.traces_g.length === 0) return "G channel missing";
  if (!sample.traces_c || sample.traces_c.length === 0) return "C channel missing";
  return "trace payload unavailable";
}

function mutationType(type?: string) {
  if (type === "substitution") return "Sub";
  if (type === "insertion") return "Ins";
  if (type === "deletion") return "Del";
  return type || "-";
}

function statusLabel(language: AppLanguage, status: "ok" | "wrong" | "uncertain" | "untested") {
  return t(language, `wb.status.${status}`);
}

function exportCsv(samples: WorkbenchSample[]) {
  const headers = ["ID", "Clone", "Status", "Reason", "Identity", "Coverage", "Mutations", "AA Changes", "Frameshift", "Orientation"];
  const rows = samples.map((s) => [
    s.id,
    s.clone || "",
    bucketSampleStatus(s),
    s.reason || s.review_reason || s.llm_reason || s.auto_reason || "",
    typeof s.identity === "number" ? s.identity.toFixed(4) : "",
    typeof (s.cds_coverage ?? s.coverage) === "number" ? (s.cds_coverage ?? s.coverage)!.toFixed(4) : "",
    String(countSampleMutations(s)),
    Array.isArray(s.aa_changes) ? s.aa_changes.join("; ") : (s.aa_changes || ""),
    s.frameshift ? "Yes" : "No",
    s.orientation || "",
  ]);

  const csvContent = [headers, ...rows]
    .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(","))
    .join("\n");

  const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bioagent-samples-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function ResultsTable({ samples, language }: ResultsTableProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const parentRef = useRef<HTMLDivElement | null>(null);

  const virtualizer = useVirtualizer({
    count: samples.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => (expandedIds.has(samples[index]?.id) ? ROW_ESTIMATE_EXPANDED : ROW_ESTIMATE_COLLAPSED),
    overscan: OVERSCAN,
    getItemKey: (index) => samples[index]?.id ?? index,
  });

  useEffect(() => {
    if (parentRef.current) parentRef.current.scrollTop = 0;
    virtualizer.measure();
    // samples identity change should reset scroll + re-measure
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samples]);

  useEffect(() => {
    virtualizer.measure();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expandedIds]);

  function toggleExpand(id: string) {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const virtualItems = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();

  return (
    <section className="results-table-panel" aria-label={t(language, "table.title")}>
      <div className="results-section-header results-section-header-compact">
        <div>
          <span className="results-kicker">{t(language, "table.kicker")}</span>
          <h3>{t(language, "table.title")}</h3>
        </div>
      </div>

      <div className="sample-table-toolbar">
        <button className="sample-toolbar-button" onClick={() => setExpandedIds(new Set(samples.map(s => s.id)))}>
          {t(language, "table.expandAll")}
        </button>
        <button className="sample-toolbar-button" onClick={() => setExpandedIds(new Set())}>
          {t(language, "table.collapseAll")}
        </button>
        <button className="sample-toolbar-button" onClick={() => exportCsv(samples)}>
          {t(language, "table.exportCsv")}
        </button>
      </div>

      <div className="sample-details-list" ref={parentRef}>
        <div className="sample-detail-head-row" style={{ position: "sticky", top: 0, zIndex: 1 }}>
          <span>{t(language, "table.sample")}</span>
          <span>{t(language, "table.status")}</span>
          <span>{t(language, "table.reason")}</span>
          <span>{t(language, "table.identity")}</span>
          <span>{t(language, "table.coverage")}</span>
          <span>{t(language, "table.mut")}</span>
          <span />
        </div>

        {samples.length === 0 ? (
          <div className="results-table-empty">
            <strong>{t(language, "table.noDataTitle")}</strong>
            <span>{t(language, "table.noDataBody")}</span>
          </div>
        ) : (
          <div style={{ height: totalSize, position: "relative", width: "100%" }}>
            {virtualItems.map((virtualRow) => {
              const sample = samples[virtualRow.index];
              if (!sample) return null;
              const status = bucketSampleStatus(sample);
              const isOpen = expandedIds.has(sample.id);
              const aaChanges = isOpen ? parseAaChanges(sample.aa_changes) : [];
              const chromatogram = isOpen ? toChromatogramData(sample) : null;
              const muts = isOpen && Array.isArray(sample.mutations) ? sample.mutations : [];
              const noChromReason = isOpen && !chromatogram ? chromatogramUnavailableReason(sample) : "";

              return (
                <div
                  key={virtualRow.key}
                  data-index={virtualRow.index}
                  ref={virtualizer.measureElement}
                  className={`sample-detail-card${isOpen ? " is-open" : ""}`}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  <div className="sample-detail-summary sample-detail-summary-grid" onClick={() => toggleExpand(sample.id)}>
                    <span className="sample-detail-sample" data-label={t(language, "table.sample")} title={sample.id}>{sample.id}</span>
                    <span className={`sample-detail-status status-${status}`} data-label={t(language, "table.status")}>{statusLabel(language, status)}</span>
                    <span className="sample-detail-reason" data-label={t(language, "table.reason")} title={displayReason(sample)}>{displayReason(sample)}</span>
                    <span className="sample-detail-cell" data-label={t(language, "table.identity")}>{formatPercent(sample.identity)}</span>
                    <span className="sample-detail-cell" data-label={t(language, "table.coverage")}>{formatPercent(sample.cds_coverage ?? sample.coverage)}</span>
                    <span className="sample-detail-cell" data-label={t(language, "table.mut")}>{countSampleMutations(sample)}</span>
                    <span className="sample-detail-toggle-cell" aria-hidden="true" />
                  </div>

                  {isOpen ? (
                    <div className="sample-detail-body">
                      <div className="sample-detail-metrics">
                        <article className="sample-detail-metric-card"><span>{t(language, "table.clone")}</span><strong>{sample.clone || "-"}</strong></article>
                        <article className="sample-detail-metric-card"><span>{t(language, "table.orientation")}</span><strong>{sample.orientation || "-"}</strong></article>
                        <article className="sample-detail-metric-card"><span>{t(language, "table.frameshift")}</span><strong>{sample.frameshift ? t(language, "table.yes") : t(language, "table.no")}</strong></article>
                        <article className="sample-detail-metric-card"><span>{t(language, "table.avgQ")}</span><strong>{typeof (sample.avg_qry_quality ?? sample.avg_quality) === "number" ? (sample.avg_qry_quality ?? sample.avg_quality)?.toFixed(1) : "-"}</strong></article>
                      </div>

                      <div className="sample-detail-section">
                        <div className="sample-detail-section-head"><h4>{t(language, "table.aaChanges")}</h4></div>
                        {aaChanges.length > 0 ? <div className="sample-detail-aa-code">{aaChanges.join(" ")}</div> : <div className="sample-detail-empty">{t(language, "table.noAa")}</div>}
                      </div>

                      <div className="sample-detail-section">
                        <div className="sample-detail-section-head"><h4>{t(language, "table.mutationTable")}</h4></div>
                        {muts.length > 0 ? (
                          <table className="sample-detail-table">
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
                              {muts.map((mutation, idx) => (
                                <tr key={`${sample.id}-${idx}`}>
                                  <td>{mutation.position ?? "-"}</td>
                                  <td className="sample-detail-base">{mutation.refBase ?? "-"}</td>
                                  <td className="sample-detail-base">{mutation.queryBase ?? "-"}</td>
                                  <td>{mutationType(mutation.type)}</td>
                                  <td>{mutation.effect || "-"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : (
                          <div className="sample-detail-empty">{t(language, "table.noMutation")}</div>
                        )}
                      </div>

                      <div className="sample-detail-section">
                        <div className="sample-detail-section-head"><h4>{t(language, "table.alignment")}</h4></div>
                        <div className="sample-detail-aa-code">
                          <div><strong>REF:</strong> {(sample.aligned_ref_g || sample.ref_sequence || "").slice(0, 600)}</div>
                          <div><strong>QRY:</strong> {(sample.aligned_query_g || sample.query_sequence || "").slice(0, 600)}</div>
                        </div>
                      </div>

                      <div className="sample-detail-section">
                        <div className="sample-detail-section-head"><h4>{t(language, "table.chromatogram")}</h4></div>
                        {chromatogram ? (
                          <Suspense fallback={<div className="sample-detail-empty">{t(language, "table.loadingChromatogram")}</div>}>
                            <ChromatogramCanvas data={chromatogram} startPosition={1} endPosition={chromatogram.baseCalls.length} mutations={muts} />
                          </Suspense>
                        ) : (
                          <div className="sample-detail-empty">
                            <div>{t(language, "table.noChromatogram")}</div>
                            <div>{t(language, "table.noChromatogramReason", { reason: noChromReason })}</div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {samples.length > 0 ? (
        <div className="sample-list-footnote">
          {t(language, "table.showing", { visible: Math.min(samples.length, virtualItems.length || samples.length), total: samples.length })}
        </div>
      ) : null}
    </section>
  );
}
