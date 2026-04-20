import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { AppLanguage, Sample } from "../types";
import { t } from "../i18n";
import { ResultsCharts } from "./ResultsCharts";
import { ResultsSummary } from "./ResultsSummary";
import { SampleDetailsList } from "./SampleDetailsList";
import {
  buildResultsView,
  bucketSampleStatus,
  type ResultsSortKey,
  type ResultsStatusFilter,
} from "../utils/resultsWorkbench";
import "./ResultsWorkbench.css";

interface ResultsWorkbenchProps {
  language: AppLanguage;
  samples: Sample[];
  selectedId: string | null;
  onSelect?: (sampleId: string) => void;
  children?: ReactNode;
}

export interface ResultBuckets {
  ok: number;
  wrong: number;
  uncertain: number;
  untested: number;
}

export function ResultsWorkbench({
  language,
  samples,
  selectedId,
  onSelect,
  children,
}: ResultsWorkbenchProps) {
  const [statusFilter, setStatusFilter] = useState<ResultsStatusFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey] = useState<ResultsSortKey>("status");
  const buckets = samples.reduce<ResultBuckets>(
    (acc, sample) => {
      acc[bucketSampleStatus(sample)] += 1;
      return acc;
    },
    { ok: 0, wrong: 0, uncertain: 0, untested: 0 }
  );

  const total = samples.length;
  const averageIdentity =
    total > 0 ? samples.reduce((sum, sample) => sum + (sample.identity || 0), 0) / total : 0;
  const averageCoverage =
    total > 0 ? samples.reduce((sum, sample) => sum + (sample.coverage || 0), 0) / total : 0;
  const visibleSamples = useMemo(
    () => buildResultsView(samples, { statusFilter, searchQuery, sortKey }),
    [samples, statusFilter, searchQuery, sortKey]
  );
  const hasActiveControls = statusFilter !== "all" || searchQuery.trim().length > 0 || sortKey !== "status";
  const statusOptions: Array<{ key: ResultsStatusFilter; label: string }> = [
    { key: "all", label: t(language, "results.filterAll") },
    { key: "wrong", label: t(language, "results.wrong") },
    { key: "uncertain", label: t(language, "results.uncertain") },
    { key: "untested", label: t(language, "results.untested") },
    { key: "ok", label: t(language, "results.ok") },
  ];

  return (
    <section className="results-workbench-shell" aria-label={t(language, "results.shellLabel")}>
      <ResultsSummary
        language={language}
        total={total}
        ok={buckets.ok}
        wrong={buckets.wrong}
        uncertain={buckets.uncertain}
        untested={buckets.untested}
        averageIdentity={averageIdentity}
        averageCoverage={averageCoverage}
      />
      <ResultsCharts language={language} samples={samples} />
      {samples.length > 0 ? (
        <>
          <section className="results-toolbar-panel" aria-label={t(language, "results.controlsTitle")}>
            <div className="results-toolbar-copy">
              <span className="results-kicker">{t(language, "results.controlsKicker")}</span>
              <h3>{t(language, "results.controlsTitle")}</h3>
              <p>
                {t(language, "results.controlsBody")} {visibleSamples.length}/{samples.length}
              </p>
            </div>
            <div className="results-toolbar-controls">
              <label className="results-search-field">
                <span>{t(language, "results.searchLabel")}</span>
                <input
                  type="search"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  placeholder={t(language, "results.searchPlaceholder")}
                />
              </label>
              <label className="results-sort-field">
                <span>{t(language, "results.sortLabel")}</span>
                <select value={sortKey} onChange={(event) => setSortKey(event.target.value as ResultsSortKey)}>
                  <option value="status">{t(language, "results.sortStatus")}</option>
                  <option value="sample">{t(language, "results.sortSample")}</option>
                  <option value="identity">{t(language, "results.sortIdentity")}</option>
                  <option value="coverage">{t(language, "results.sortCoverage")}</option>
                  <option value="mutations">{t(language, "results.sortMutations")}</option>
                </select>
              </label>
            </div>
            <div className="results-filter-row">
              {statusOptions.map((option) => (
                <button
                  key={option.key}
                  type="button"
                  className={`results-filter-chip${statusFilter === option.key ? " is-active" : ""}`}
                  onClick={() => setStatusFilter(option.key)}
                >
                  {option.label}
                </button>
              ))}
              {hasActiveControls ? (
                <button
                  type="button"
                  className="results-clear-filters"
                  onClick={() => {
                    setStatusFilter("all");
                    setSearchQuery("");
                    setSortKey("status");
                  }}
                >
                  {t(language, "results.clearFilters")}
                </button>
              ) : null}
            </div>
          </section>
          {visibleSamples.length > 0 ? (
            <SampleDetailsList
              language={language}
              samples={visibleSamples}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ) : (
            <section className="sample-details-panel" aria-label={t(language, "results.detailsTitle")}>
              <div className="results-empty-filter-state">
                <span className="results-kicker">{t(language, "results.controlsKicker")}</span>
                <h3>{t(language, "results.noVisibleSamplesTitle")}</h3>
                <p>{t(language, "results.noVisibleSamplesBody")}</p>
              </div>
            </section>
          )}
        </>
      ) : (
        children ?? null
      )}
    </section>
  );
}
