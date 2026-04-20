import { useState } from "react";
import type { AppLanguage, ChromatogramData, Sample } from "../types";
import { t } from "../i18n";
import { ChromatogramCanvas } from "./ChromatogramCanvas";
import { SequenceViewer } from "./SequenceViewer";

interface SampleDetailsListProps {
  language: AppLanguage;
  samples: Sample[];
  selectedId: string | null;
  onSelect?: (sampleId: string) => void;
}

interface DecisionTraceEntry {
  id: string;
  label: string;
  statusLabel: string;
  reason: string;
  tone: string;
  sourceLabel?: string;
}

const UNTESTED_REASON = "\u672a\u6d4b\u901a";

function firstNonEmpty(...values: Array<string | undefined>) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function toChromatogramData(sample: Sample): ChromatogramData | null {
  if (
    !sample.traces_a ||
    !sample.traces_t ||
    !sample.traces_g ||
    !sample.traces_c ||
    !sample.quality ||
    !sample.query_sequence
  ) {
    return null;
  }

  return {
    traces: {
      A: sample.traces_a,
      T: sample.traces_t,
      G: sample.traces_g,
      C: sample.traces_c,
    },
    quality: sample.quality,
    baseCalls: sample.query_sequence,
    base_locations: sample.base_locations || [],
    mixed_peaks: sample.mixed_peaks || [],
  };
}

function parseAaChanges(value: Sample["aa_changes"]) {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string" && item.length > 0);
  }

  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value) as unknown;
      return Array.isArray(parsed)
        ? parsed.filter((item): item is string => typeof item === "string" && item.length > 0)
        : [];
    } catch {
      return [];
    }
  }

  return [];
}

function extractAaChangesFromReason(reason?: string) {
  if (!reason) {
    return [];
  }

  const matches = reason.match(/[A-Z*]\d+[A-Z*]/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function getStatusTone(sample: Sample) {
  if (sample.reason === UNTESTED_REASON) {
    return "untested";
  }

  if (sample.status === "ok" || sample.status === "wrong") {
    return sample.status;
  }

  return "uncertain";
}

function getStatusToneFromValues(status?: Sample["status"], reason?: string) {
  if (reason === UNTESTED_REASON) {
    return "untested";
  }

  if (status === "ok" || status === "wrong") {
    return status;
  }

  return "uncertain";
}

function getDisplayStatus(language: AppLanguage, sample: Sample) {
  return t(language, `results.${getStatusTone(sample)}`);
}

function getDisplayReason(language: AppLanguage, sample: Sample) {
  const finalReason = firstNonEmpty(sample.reason);
  if (finalReason) {
    return finalReason;
  }

  const tone = getStatusTone(sample);
  if (tone !== "ok") {
    const fallbackReason = firstNonEmpty(
      sample.review_reason,
      sample.llm_reason,
      sample.auto_reason,
      sample.error
    );
    if (fallbackReason) {
      return fallbackReason;
    }
  }

  return t(language, "results.noReason");
}

function getDecisionTraceEntries(language: AppLanguage, sample: Sample): DecisionTraceEntry[] {
  const entries: DecisionTraceEntry[] = [];
  const finalTone = getStatusTone(sample);
  const finalReason = getDisplayReason(language, sample);

  entries.push({
    id: "final",
    label: t(language, "results.decisionTraceFinal"),
    statusLabel: t(language, `results.${finalTone}`),
    reason: finalReason,
    tone: finalTone,
    sourceLabel: sample.reviewed
      ? t(language, "results.decisionTraceSourceReviewed")
      : sample.llm_status
        ? t(language, "results.decisionTraceSourceAi")
        : t(language, "results.decisionTraceSourceRules"),
  });

  if (sample.reviewed) {
    const reviewedTone = getStatusToneFromValues(sample.review_status, sample.review_reason);
    entries.push({
      id: "reviewed",
      label: t(language, "results.decisionTraceReviewed"),
      statusLabel: t(language, `results.${reviewedTone}`),
      reason: sample.review_reason || t(language, "results.noReason"),
      tone: reviewedTone,
      sourceLabel: t(language, "results.decisionTraceSourceReviewed"),
    });
  }

  if (sample.llm_status) {
    const llmTone = getStatusToneFromValues(sample.llm_status, sample.llm_reason);
    entries.push({
      id: "llm",
      label: t(language, "results.decisionTraceAi"),
      statusLabel: t(language, `results.${llmTone}`),
      reason: sample.llm_reason || t(language, "results.noReason"),
      tone: llmTone,
      sourceLabel: t(language, "results.decisionTraceSourceAi"),
    });
  }

  if (sample.auto_status) {
    const autoTone = getStatusToneFromValues(sample.auto_status, sample.auto_reason);
    entries.push({
      id: "rules",
      label: t(language, "results.decisionTraceRules"),
      statusLabel: t(language, `results.${autoTone}`),
      reason: sample.auto_reason || t(language, "results.noReason"),
      tone: autoTone,
      sourceLabel: t(language, "results.decisionTraceSourceRules"),
    });
  } else if (!sample.reviewed && !sample.llm_status) {
    entries.push({
      id: "rules",
      label: t(language, "results.decisionTraceRules"),
      statusLabel: t(language, `results.${finalTone}`),
      reason: finalReason,
      tone: finalTone,
      sourceLabel: t(language, "results.decisionTraceSourceRules"),
    });
  }

  return entries.filter((entry, index, array) => {
    if (index === 0) return true;
    return !array.slice(0, index).some((previous) => previous.id === entry.id);
  });
}

function formatPercent(value?: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return `${((value || 0) * 100).toFixed(1)}%`;
}

function average(values?: number[]) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }

  const sum = values.reduce((acc, value) => acc + value, 0);
  return sum / values.length;
}

function getCount(sample: Sample, key: "sub" | "ins" | "del") {
  if (key === "sub") {
    return (
      sample.sub_count ?? sample.sub ?? sample.mutations.filter((mutation) => mutation.type === "substitution").length
    );
  }
  if (key === "ins") {
    return sample.ins_count ?? sample.ins ?? sample.mutations.filter((mutation) => mutation.type === "insertion").length;
  }
  return sample.del_count ?? sample.dele ?? sample.mutations.filter((mutation) => mutation.type === "deletion").length;
}

function getMutationTypeLabel(language: AppLanguage, type: Sample["mutations"][number]["type"]) {
  if (type === "substitution") {
    return t(language, "results.sub");
  }
  if (type === "insertion") {
    return t(language, "results.ins");
  }
  return t(language, "results.del");
}

export function SampleDetailsList({
  language,
  samples,
  selectedId,
  onSelect,
}: SampleDetailsListProps) {
  const [openChromatograms, setOpenChromatograms] = useState<Record<string, boolean>>({});

  return (
    <section className="sample-details-panel" aria-label={t(language, "results.detailsTitle")}>
      <div className="results-section-header">
        <div>
          <span className="results-kicker">{t(language, "results.detailsKicker")}</span>
          <h3>{t(language, "results.detailsTitle")}</h3>
        </div>
        <p>{t(language, "results.detailsBody")}</p>
      </div>

      <div className="sample-detail-head-row" aria-hidden="true">
        <span>{t(language, "results.sample")}</span>
        <span>{t(language, "results.status")}</span>
        <span>{t(language, "results.reason")}</span>
        <span>{t(language, "results.identity")}</span>
        <span>{t(language, "results.coverage")}</span>
        <span>{t(language, "results.sub")}</span>
        <span>{t(language, "results.ins")}</span>
        <span>{t(language, "results.del")}</span>
        <span>{t(language, "results.rule")}</span>
        <span>{t(language, "results.quality")}</span>
        <span />
      </div>

      <div className="sample-details-list">
        {samples.map((sample) => {
          const chromatogramData = toChromatogramData(sample);
          const aaChanges = parseAaChanges(sample.aa_changes);
          const isOpen = sample.id === selectedId;
          const statusTone = getStatusTone(sample);
          const displayStatus = getDisplayStatus(language, sample);
          const displayReason = getDisplayReason(language, sample);
          const coverage = sample.cds_coverage ?? sample.coverage;
          const avgQuality = sample.avg_quality ?? sample.avg_qry_quality;
          const chromatogramAvgQuality = avgQuality ?? average(sample.quality);
          const chromatogramBaseCount = chromatogramData?.baseCalls.length ?? 0;
          const mixedPeakCount = chromatogramData?.mixed_peaks.length ?? sample.mixed_peaks?.length ?? 0;
          const isChromatogramOpen = openChromatograms[sample.id] ?? false;
          const autoStatusTone = getStatusToneFromValues(sample.auto_status, sample.auto_reason);
          const autoStatusLabel = sample.auto_status ? t(language, `results.${autoStatusTone}`) : null;
          const llmStatusTone = getStatusToneFromValues(sample.llm_status, sample.llm_reason);
          const llmStatusLabel = sample.llm_status ? t(language, `results.${llmStatusTone}`) : null;
          const decisionTrace = getDecisionTraceEntries(language, sample);
          const displayAaChanges = Array.from(
            new Set([
              ...aaChanges,
              ...extractAaChangesFromReason(sample.reason),
              ...extractAaChangesFromReason(sample.review_reason),
              ...extractAaChangesFromReason(sample.llm_reason),
              ...extractAaChangesFromReason(sample.auto_reason),
            ])
          );

          return (
            <details
              key={sample.id}
              className={`sample-detail-card${isOpen ? " is-active" : ""}`}
              open={isOpen}
            >
              <summary
                className="sample-detail-summary sample-detail-summary-grid"
                aria-current={isOpen ? "true" : undefined}
                onClick={() => onSelect?.(sample.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    onSelect?.(sample.id);
                  }
                }}
              >
                <span className="sample-detail-sample" title={sample.id} data-label={t(language, "results.sample")}>
                  {sample.id}
                </span>
                <span className={`sample-detail-status status-${statusTone}`} data-label={t(language, "results.status")}>
                  {displayStatus}
                </span>
                <span
                  className="sample-detail-reason"
                  title={displayReason}
                  data-label={t(language, "results.reason")}
                >
                  {displayReason}
                </span>
                <span className="sample-detail-cell" data-label={t(language, "results.identity")}>{formatPercent(sample.identity)}</span>
                <span className="sample-detail-cell" data-label={t(language, "results.coverage")}>{formatPercent(coverage)}</span>
                <span className="sample-detail-cell" data-label={t(language, "results.sub")}>{getCount(sample, "sub")}</span>
                <span className="sample-detail-cell" data-label={t(language, "results.ins")}>{getCount(sample, "ins")}</span>
                <span className="sample-detail-cell" data-label={t(language, "results.del")}>{getCount(sample, "del")}</span>
                <span className="sample-detail-cell" data-label={t(language, "results.rule")}>{sample.rule_id ? `R${sample.rule_id}` : "-"}</span>
                <span className="sample-detail-cell" data-label={t(language, "results.quality")}>
                  {typeof avgQuality === "number" ? avgQuality.toFixed(1) : "-"}
                </span>
                <span className="sample-detail-toggle-cell" aria-hidden="true" />
              </summary>

              {isOpen ? (
              <div className="sample-detail-body">
                {sample.status === "error" ? (
                  <div className="sample-detail-error">
                    <strong>{t(language, "analysis.analysisError")}</strong>
                    <p>{sample.error || sample.reason || "-"}</p>
                  </div>
                ) : (
                  <>
                    <div className="sample-detail-metrics">
                      <article className="sample-detail-metric-card">
                        <span>{t(language, "results.clone")}</span>
                        <strong>{sample.clone || "-"}</strong>
                      </article>
                      <article className="sample-detail-metric-card">
                        <span>{t(language, "results.rule")}</span>
                        <strong>{sample.rule_id ? `R${sample.rule_id}` : "-"}</strong>
                      </article>
                      <article className="sample-detail-metric-card">
                        <span>{t(language, "results.orientation")}</span>
                        <strong>{sample.orientation || "-"}</strong>
                      </article>
                      <article className="sample-detail-metric-card">
                        <span>{t(language, "analysis.frameshift")}</span>
                        <strong>
                          {sample.frameshift
                            ? t(language, "results.frameshiftYes")
                            : t(language, "results.frameshiftNo")}
                        </strong>
                      </article>
                      <article className="sample-detail-metric-card">
                        <span>AA</span>
                        <strong>{sample.aa_changes_n ?? aaChanges.length}</strong>
                      </article>
                      <article className="sample-detail-metric-card">
                        <span>{t(language, "results.coverage")}</span>
                        <strong>{formatPercent(coverage)}</strong>
                      </article>
                      <article className="sample-detail-metric-card">
                        <span>{t(language, "results.quality")}</span>
                        <strong>{typeof avgQuality === "number" ? avgQuality.toFixed(1) : "-"}</strong>
                      </article>
                    </div>

                    <div className="sample-detail-section">
                      <div className="sample-detail-section-head">
                        <h4>{t(language, "results.decisionTraceTitle")}</h4>
                      </div>
                      <div className="sample-detail-trace-grid">
                        {decisionTrace.map((entry, index) => (
                          <article
                            key={`${sample.id}-${entry.id}`}
                            className={`sample-detail-trace-card${index === 0 ? " is-final" : ""}`}
                          >
                            <div className="sample-detail-trace-head">
                              <span className="sample-detail-trace-label">{entry.label}</span>
                              <span className={`sample-detail-status status-${entry.tone}`}>{entry.statusLabel}</span>
                            </div>
                            <strong>{entry.reason}</strong>
                            {entry.sourceLabel ? (
                              <span className="sample-detail-trace-source">{entry.sourceLabel}</span>
                            ) : null}
                          </article>
                        ))}
                      </div>
                    </div>

                    {sample.reviewed ? (
                      <div className="sample-detail-audit-note">
                        <span className="sample-detail-audit-badge">{t(language, "results.reviewedBadge")}</span>
                        <div className="sample-detail-audit-copy">
                          <strong>
                            {t(language, "results.reviewedVerdict")}: {displayStatus}
                            {displayReason ? ` / ${displayReason}` : ""}
                          </strong>
                          {autoStatusLabel ? (
                            <p>
                              {t(language, "results.autoVerdict")}: {autoStatusLabel}
                              {sample.auto_reason ? ` / ${sample.auto_reason}` : ""}
                            </p>
                          ) : null}
                        </div>
                      </div>
                    ) : null}

                    {!sample.reviewed && sample.llm_status ? (
                      <div className="sample-detail-audit-note sample-detail-audit-note-ai">
                        <span className="sample-detail-audit-badge sample-detail-audit-badge-ai">
                          {t(language, "results.aiReviewedBadge")}
                        </span>
                        <div className="sample-detail-audit-copy">
                          <strong>
                            {t(language, "results.aiVerdict")}: {llmStatusLabel}
                            {sample.llm_reason ? ` / ${sample.llm_reason}` : ""}
                          </strong>
                          {autoStatusLabel ? (
                            <p>
                              {t(language, "results.rulesVerdict")}: {autoStatusLabel}
                              {sample.auto_reason ? ` / ${sample.auto_reason}` : ""}
                            </p>
                          ) : null}
                        </div>
                      </div>
                    ) : null}

                    <div className="sample-detail-section">
                      <div className="sample-detail-section-head">
                        <h4>{t(language, "results.aaChangesTitle")}</h4>
                      </div>
                      {displayAaChanges.length > 0 ? (
                        <div className="sample-detail-aa-code">{displayAaChanges.join(" ")}</div>
                      ) : (
                        <div className="sample-detail-empty">{t(language, "results.aaChangesNone")}</div>
                      )}
                    </div>

                    <div className="sample-detail-section">
                      <div className="sample-detail-section-head">
                        <h4>{t(language, "results.sequenceTitle")}</h4>
                      </div>
                      <div className="sample-detail-sequence-viewer">
                        <SequenceViewer
                          refSequence={sample.ref_sequence || ""}
                          querySequence={sample.query_sequence || ""}
                          alignedRefG={sample.aligned_ref_g || ""}
                          alignedQueryG={sample.aligned_query_g || ""}
                          alignedQuery={sample.aligned_query || ""}
                          matches={sample.matches || []}
                          mutations={sample.mutations || []}
                          chromatogramData={chromatogramData}
                          cdsStart={sample.cds_start || 0}
                          cdsEnd={sample.cds_end || 0}
                          featureName={sample.clone || "CDS"}
                        />
                      </div>
                    </div>

                    {chromatogramData ? (
                      <div className="sample-detail-section">
                        <details
                          className="sample-detail-chromatogram-disclosure"
                          open={isChromatogramOpen}
                          onToggle={(event) => {
                            const nextOpen = event.currentTarget.open;
                            setOpenChromatograms((current) => ({ ...current, [sample.id]: nextOpen }));
                          }}
                        >
                          <summary className="sample-detail-chromatogram-summary">
                            <div className="sample-detail-section-head">
                              <h4>{t(language, "results.chromatogramTitle")}</h4>
                            </div>
                            <div className="sample-detail-chromatogram-meta">
                              <span>{`${t(language, "results.chromatogramBases")}: ${chromatogramBaseCount}`}</span>
                              <span>
                                {`${t(language, "results.chromatogramQuality")}: ${
                                  typeof chromatogramAvgQuality === "number"
                                    ? chromatogramAvgQuality.toFixed(1)
                                    : "-"
                                }`}
                              </span>
                              <span>{`${t(language, "results.chromatogramMixedPeaks")}: ${mixedPeakCount}`}</span>
                            </div>
                            <span className="sample-detail-chromatogram-toggle">
                              {isChromatogramOpen ? t(language, "results.collapse") : t(language, "results.expand")}
                            </span>
                          </summary>
                          {isChromatogramOpen ? (
                            <div className="sample-detail-chromatogram">
                              <ChromatogramCanvas
                                data={chromatogramData}
                                startPosition={1}
                                endPosition={chromatogramData.baseCalls.length}
                                onHover={() => {}}
                              />
                            </div>
                          ) : null}
                        </details>
                      </div>
                    ) : null}

                    <div className="sample-detail-section">
                      <div className="sample-detail-section-head">
                        <h4>{t(language, "results.mutationsSectionTitle")}</h4>
                      </div>
                      {sample.mutations && sample.mutations.length > 0 ? (
                        <table className="sample-detail-table">
                          <thead>
                            <tr>
                              <th>{t(language, "analysis.pos")}</th>
                              <th>{t(language, "analysis.ref")}</th>
                              <th>{t(language, "analysis.query")}</th>
                              <th>{t(language, "analysis.type")}</th>
                              <th>{t(language, "analysis.effect")}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {sample.mutations.map((mutation, index) => (
                              <tr key={`${sample.id}-${mutation.position}-${index}`}>
                                <td>{mutation.position}</td>
                                <td className="sample-detail-base">{mutation.refBase}</td>
                                <td className="sample-detail-base">{mutation.queryBase}</td>
                                <td>{getMutationTypeLabel(language, mutation.type)}</td>
                                <td>{mutation.effect || "-"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="sample-detail-empty">{t(language, "results.mutationsEmpty")}</div>
                      )}
                    </div>
                  </>
                )}
              </div>
              ) : null}
            </details>
          );
        })}
      </div>
    </section>
  );
}



