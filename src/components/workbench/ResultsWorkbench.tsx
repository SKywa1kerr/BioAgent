import { useEffect, useMemo, useState } from "react";
import type { WorkbenchSample } from "./types";
import { ResultsCharts } from "./ResultsCharts";
import { ResultsSummary } from "./ResultsSummary";
import { ResultsTable } from "./ResultsTable";
import { DetailDrawer } from "./DetailDrawer";
import { ExportMenu } from "./ExportMenu";
import { buildResultsView, bucketSampleStatus } from "./utils";
import { useWorkbenchControls } from "../../hooks/useWorkbenchControls";
import { registerCommand } from "../../lib/commands/registry";
import { runExport } from "../../lib/exporters/runExport";
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

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [density, setDensity] = useState<"compact" | "detailed">("compact");

  const visibleSamples = useMemo(
    () => buildResultsView(samples, { statusFilter, searchQuery, sortKey }),
    [samples, statusFilter, searchQuery, sortKey],
  );

  const selectedSample = selectedId
    ? visibleSamples.find((s) => s.id === selectedId) ?? null
    : null;

  useEffect(() => {
    if (selectedId && !visibleSamples.some((s) => s.id === selectedId)) {
      setSelectedId(null);
    }
  }, [visibleSamples, selectedId]);

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

  useEffect(() => {
    const args = {
      samples: visibleSamples,
      filters: { statusFilter, searchQuery, sortKey },
      dataset,
      language,
    };
    const offs: Array<() => void> = [
      registerCommand({
        id: "workbench.export-csv",
        title: t(language, "palette.cmd.exportCsv"),
        group: "workbench",
        keywords: ["export", "csv", "导出"],
        when: () => visibleSamples.length > 0,
        run: () => runExport("csv", args),
      }),
      registerCommand({
        id: "workbench.export-json",
        title: t(language, "palette.cmd.exportJson"),
        group: "workbench",
        keywords: ["export", "json", "导出"],
        when: () => visibleSamples.length > 0,
        run: () => runExport("json", args),
      }),
      registerCommand({
        id: "workbench.export-pdf",
        title: t(language, "palette.cmd.exportPdf"),
        group: "workbench",
        keywords: ["export", "pdf", "报告", "导出"],
        when: () => visibleSamples.length > 0,
        run: () => runExport("pdf", args),
      }),
      registerCommand({
        id: "workbench.clear-filters",
        title: t(language, "palette.cmd.clearFilters"),
        group: "workbench",
        keywords: ["clear", "reset", "清除"],
        when: () => hasActiveControls,
        run: reset,
      }),
    ];
    return () => { offs.forEach((off) => off()); };
  }, [visibleSamples, statusFilter, searchQuery, sortKey, language, dataset, hasActiveControls, reset]);

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
          <div className="results-density-toggle" role="group" aria-label={t(language, "wb.density.detailed")}>
            {(["compact", "detailed"] as const).map((d) => (
              <button
                key={d}
                type="button"
                className={`results-filter-chip${density === d ? " is-active" : ""}`}
                onClick={() => setDensity(d)}
              >
                {t(language, `wb.density.${d}`)}
              </button>
            ))}
          </div>
        </div>
      </section>

      <ResultsTable
        samples={visibleSamples}
        language={language}
        density={density}
        selectedId={selectedId}
        onSelect={setSelectedId}
        isFiltered={hasActiveControls}
        onClearFilters={reset}
      />
      <DetailDrawer
        sample={selectedSample}
        language={language}
        onClose={() => setSelectedId(null)}
      />
    </section>
  );
}
