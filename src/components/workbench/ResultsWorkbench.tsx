import { useMemo } from "react";
import type { WorkbenchSample } from "./types";
import { ResultsCharts } from "./ResultsCharts";
import { ResultsSummary } from "./ResultsSummary";
import { ResultsTable } from "./ResultsTable";
import { ExportMenu } from "./ExportMenu";
import { buildResultsView, bucketSampleStatus } from "./utils";
import { useWorkbenchControls } from "../../hooks/useWorkbenchControls";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import "./ResultsWorkbench.css";

interface ResultsWorkbenchProps {
  samples: WorkbenchSample[];
  language: AppLanguage;
  dataset?: string;
}

export function ResultsWorkbench({ samples, language, dataset }: ResultsWorkbenchProps) {
  const { controls, setStatusFilter, setSearchQuery, setSortKey, setSummaryScope, reset } = useWorkbenchControls();
  const { statusFilter, searchQuery, sortKey, summaryScope } = controls;

  const visibleSamples = useMemo(
    () => buildResultsView(samples, { statusFilter, searchQuery, sortKey }),
    [samples, statusFilter, searchQuery, sortKey],
  );

  const summarySource = summaryScope === "filtered" ? visibleSamples : samples;

  const buckets = useMemo(() => {
    const acc = { ok: 0, wrong: 0, uncertain: 0, untested: 0 };
    for (const s of summarySource) acc[bucketSampleStatus(s)] += 1;
    return acc;
  }, [summarySource]);

  const total = summarySource.length;
  const averageIdentity = useMemo(
    () => (total > 0 ? summarySource.reduce((sum, s) => sum + (s.identity || 0), 0) / total : 0),
    [summarySource, total],
  );
  const averageCoverage = useMemo(
    () => (total > 0 ? summarySource.reduce((sum, s) => sum + (s.cds_coverage ?? s.coverage ?? 0), 0) / total : 0),
    [summarySource, total],
  );

  const hasActiveControls = statusFilter !== "all" || searchQuery.trim().length > 0 || sortKey !== "status";
  const statusOptions = [
    { key: "all", label: t(language, "wb.status.all") },
    { key: "wrong", label: t(language, "wb.status.wrong") },
    { key: "uncertain", label: t(language, "wb.status.uncertain") },
    { key: "untested", label: t(language, "wb.status.untested") },
    { key: "ok", label: t(language, "wb.status.ok") },
  ] as const;

  return (
    <section className="results-workbench-shell" aria-label={t(language, "wb.aria")}>
      <ResultsSummary
        language={language}
        total={total}
        ok={buckets.ok}
        wrong={buckets.wrong}
        uncertain={buckets.uncertain}
        untested={buckets.untested}
        averageIdentity={averageIdentity}
        averageCoverage={averageCoverage}
        scope={summaryScope}
        onScopeChange={setSummaryScope}
        originalTotal={samples.length}
        filteredTotal={visibleSamples.length}
      />
      <ResultsCharts samples={summarySource} language={language} />

      <section className="results-toolbar-panel" aria-label={t(language, "wb.controls")}>
        <div className="results-toolbar-copy">
          <span className="results-kicker">{t(language, "wb.controls")}</span>
          <h3>{t(language, "wb.filterSort")}</h3>
          <p>{t(language, "wb.showing", { visible: visibleSamples.length, total: samples.length })}</p>
        </div>
        <div className="results-toolbar-controls">
          <label className="results-search-field">
            <span>{t(language, "wb.search")}</span>
            <input
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder={t(language, "wb.searchPlaceholder")}
            />
          </label>
          <label className="results-sort-field">
            <span>{t(language, "wb.sort")}</span>
            <select value={sortKey} onChange={(event) => setSortKey(event.target.value as typeof sortKey)}>
              <option value="status">{t(language, "wb.sort.status")}</option>
              <option value="sample">{t(language, "wb.sort.sample")}</option>
              <option value="identity">{t(language, "wb.sort.identity")}</option>
              <option value="coverage">{t(language, "wb.sort.coverage")}</option>
              <option value="mutations">{t(language, "wb.sort.mutations")}</option>
            </select>
          </label>
          <ExportMenu
            samples={visibleSamples}
            filters={{ statusFilter, searchQuery, sortKey }}
            dataset={dataset}
            language={language}
          />
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
            <button type="button" className="results-clear-filters" onClick={reset}>
              {t(language, "wb.clear")}
            </button>
          ) : null}
        </div>
      </section>

      <ResultsTable samples={visibleSamples} language={language} />
    </section>
  );
}
