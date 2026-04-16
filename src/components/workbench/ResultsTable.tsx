import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, countSampleMutations, formatPercent } from "./utils";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface ResultsTableProps {
  samples: WorkbenchSample[];
  language: AppLanguage;
}

const PAGE_SIZE = 24;
const INITIAL_RENDER_COUNT = 10;
const RENDER_BATCH_SIZE = 12;
const RENDER_BATCH_INTERVAL_MS = 70;

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
    traces: {
      A: sample.traces_a,
      T: sample.traces_t,
      G: sample.traces_g,
      C: sample.traces_c,
    },
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
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [renderedCount, setRenderedCount] = useState(INITIAL_RENDER_COUNT);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
    setRenderedCount(INITIAL_RENDER_COUNT);
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [samples]);

  const visibleSamples = useMemo(() => samples.slice(0, visibleCount), [samples, visibleCount]);

  useEffect(() => {
    setRenderedCount((current) => {
      const start = Math.min(INITIAL_RENDER_COUNT, visibleSamples.length);
      return current > visibleSamples.length ? start : Math.max(current, start);
    });
  }, [visibleSamples.length]);

  useEffect(() => {
    if (renderedCount >= visibleSamples.length) return;
    const timer = setTimeout(() => {
      setRenderedCount((current) => Math.min(visibleSamples.length, current + RENDER_BATCH_SIZE));
    }, RENDER_BATCH_INTERVAL_MS);
    return () => clearTimeout(timer);
  }, [renderedCount, visibleSamples.length]);

  const progressivelyRenderedSamples = useMemo(
    () => visibleSamples.slice(0, Math.min(renderedCount, visibleSamples.length)),
    [visibleSamples, renderedCount],
  );

  function loadMoreIfNeeded() {
    setVisibleCount((current) => {
      if (current >= samples.length) return current;
      return Math.min(samples.length, current + PAGE_SIZE);
    });
  }

  function handleListScroll() {
    const node = listRef.current;
    if (!node) return;
    const nearBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 240;
    if (nearBottom) loadMoreIfNeeded();
  }

  const pendingSkeletonCount = Math.min(4, Math.max(0, visibleSamples.length - progressivelyRenderedSamples.length));

  function toggleExpand(id: string) {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <section className="results-table-panel" aria-label={t(language, "table.title")}>
      <div className="results-section-header results-section-header-compact">
        <div>
          <span className="results-kicker">{t(language, "table.kicker")}</span>
          <h3>{t(language, "table.title")}</h3>
        </div>
      </div>

      <div className="sample-table-toolbar">
        <button className="sample-toolbar-button" onClick={() => setExpandedIds(new Set(progressivelyRenderedSamples.map(s => s.id)))}>
          {t(language, "table.expandAll")}
        </button>
        <button className="sample-toolbar-button" onClick={() => setExpandedIds(new Set())}>
          {t(language, "table.collapseAll")}
        </button>
        <button className="sample-toolbar-button" onClick={() => exportCsv(samples)}>
          {t(language, "table.exportCsv")}
        </button>
      </div>

      <div className="sample-details-list" ref={listRef} onScroll={handleListScroll}>
        <div className="sample-detail-head-row">
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
          <>
            {progressivelyRenderedSamples.map((sample) => {
              const status = bucketSampleStatus(sample);
              const aaChanges = parseAaChanges(sample.aa_changes);
              const chromatogram = toChromatogramData(sample);
              const muts = Array.isArray(sample.mutations) ? sample.mutations : [];
              const noChromReason = chromatogram ? "" : chromatogramUnavailableReason(sample);

              return (
                <div key={sample.id} className={`sample-detail-card${expandedIds.has(sample.id) ? " is-open" : ""}`}>
                  <div className="sample-detail-summary sample-detail-summary-grid" onClick={() => toggleExpand(sample.id)}>
                    <span className="sample-detail-sample" data-label={t(language, "table.sample")} title={sample.id}>{sample.id}</span>
                    <span className={`sample-detail-status status-${status}`} data-label={t(language, "table.status")}>{statusLabel(language, status)}</span>
                    <span className="sample-detail-reason" data-label={t(language, "table.reason")} title={displayReason(sample)}>{displayReason(sample)}</span>
                    <span className="sample-detail-cell" data-label={t(language, "table.identity")}>{formatPercent(sample.identity)}</span>
                    <span className="sample-detail-cell" data-label={t(language, "table.coverage")}>{formatPercent(sample.cds_coverage ?? sample.coverage)}</span>
                    <span className="sample-detail-cell" data-label={t(language, "table.mut")}>{countSampleMutations(sample)}</span>
                    <span className="sample-detail-toggle-cell" aria-hidden="true" />
                  </div>

                  {expandedIds.has(sample.id) ? (
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
                          <ChromatogramCanvas data={chromatogram} startPosition={1} endPosition={chromatogram.baseCalls.length} />
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

            {pendingSkeletonCount > 0 ? (
              <div className="sample-skeleton-group" aria-live="polite">
                {Array.from({ length: pendingSkeletonCount }, (_, idx) => (
                  <div key={`skeleton-${idx}`} className="sample-skeleton-card" aria-hidden="true">
                    <div className="sample-skeleton-line long" />
                    <div className="sample-skeleton-line short" />
                  </div>
                ))}
                <div className="sample-skeleton-hint">{t(language, "table.loadingSamples")}</div>
              </div>
            ) : null}
          </>
        )}
      </div>

      {samples.length > 0 ? (
        <div className="sample-list-footnote">
          {t(language, "table.showing", { visible: progressivelyRenderedSamples.length, total: samples.length })}
        </div>
      ) : null}
    </section>
  );
}
