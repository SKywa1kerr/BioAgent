import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence } from "framer-motion";
import type { WorkbenchSample } from "./types";
import { ResultsCharts } from "./ResultsCharts";
import { ResultsSummary } from "./ResultsSummary";
import { ResultsTable } from "./ResultsTable";
import { DetailDrawer } from "./DetailDrawer";
import { CompareView } from "./CompareView";
import { ExportMenu } from "./ExportMenu";
import { applyOverride, buildResultsView, bucketSampleStatus } from "./utils";
import { useWorkbenchControls } from "../../hooks/useWorkbenchControls";
import { useSampleOverrides } from "../../hooks/useSampleOverrides";
import { registerCommand } from "../../lib/commands/registry";
import { runExport } from "../../lib/exporters/runExport";
import { nextCompareSelection } from "../../lib/workbench/compareSelection";
import { Icon } from "../ui/Icon";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import styles from "./ResultsWorkbench.module.css";

interface ResultsWorkbenchProps {
  samples: WorkbenchSample[];
  language: AppLanguage;
  dataset?: string;
  analysisId?: string | null;
}

export function ResultsWorkbench({ samples, language, dataset, analysisId }: ResultsWorkbenchProps) {
  const {
    controls,
    setStatusFilter,
    setSearchQuery,
    setSortKey,
    setDensity,
    setSummaryScope,
    reset,
  } = useWorkbenchControls();
  const { statusFilter, searchQuery, sortKey, summaryScope, density } = controls;

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const overridesApi = useSampleOverrides();

  const samplesWithOverrides = useMemo(() => {
    if (!analysisId) return samples;
    return samples.map((s) =>
      applyOverride(s, overridesApi.getOverride(analysisId, s.id)),
    );
  }, [samples, analysisId, overridesApi]);

  const visibleSamples = useMemo(
    () => buildResultsView(samplesWithOverrides, { statusFilter, searchQuery, sortKey }),
    [samplesWithOverrides, statusFilter, searchQuery, sortKey],
  );

  type ExportArgs = {
    samples: WorkbenchSample[];
    filters: { statusFilter: typeof statusFilter; searchQuery: string; sortKey: typeof sortKey };
    dataset?: string;
    language: AppLanguage;
  };
  const exportArgsRef = useRef<ExportArgs>({
    samples: visibleSamples,
    filters: { statusFilter, searchQuery, sortKey },
    dataset,
    language,
  });
  const resetRef = useRef(reset);
  useEffect(() => { resetRef.current = reset; }, [reset]);

  const selectedSample = selectedId
    ? visibleSamples.find((s) => s.id === selectedId) ?? null
    : null;

  // Compare picks resolve against the *full* (post-override) sample set so
  // toggling a status filter does not silently invalidate a selection — the
  // user can still open the side-by-side view of two samples that the filter
  // would otherwise hide.
  const compareLeft = compareIds[0]
    ? samplesWithOverrides.find((s) => s.id === compareIds[0]) ?? null
    : null;
  const compareRight = compareIds[1]
    ? samplesWithOverrides.find((s) => s.id === compareIds[1]) ?? null
    : null;
  const compareReady = compareIds.length === 2 && compareLeft != null && compareRight != null;

  useEffect(() => {
    if (selectedId && !visibleSamples.some((s) => s.id === selectedId)) {
      setSelectedId(null);
    }
  }, [visibleSamples, selectedId]);

  // If a compare-selected sample disappears from the underlying dataset
  // entirely (e.g. user re-runs analysis), drop it. We do NOT prune when only
  // the visible set changes — see comment above.
  useEffect(() => {
    setCompareIds((prev) => {
      const next = prev.filter((id) => samplesWithOverrides.some((s) => s.id === id));
      return next.length === prev.length ? prev : next;
    });
  }, [samplesWithOverrides]);

  // Auto-close the compare overlay if either pick disappears.
  useEffect(() => {
    if (compareOpen && !compareReady) setCompareOpen(false);
  }, [compareOpen, compareReady]);

  function handleToggleCompare(id: string) {
    setCompareIds((prev) => nextCompareSelection(prev, id));
  }

  function handleClearCompare() {
    setCompareIds([]);
    setCompareOpen(false);
  }

  const summarySource = summaryScope === "filtered" ? visibleSamples : samplesWithOverrides;

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
    exportArgsRef.current = args;
  });

  useEffect(() => {
    const hasSamples = () => exportArgsRef.current.samples.length > 0;
    const isFiltered = () => {
      const f = exportArgsRef.current.filters;
      return f.statusFilter !== "all" || f.searchQuery.trim().length > 0 || f.sortKey !== "status";
    };
    const offs: Array<() => void> = [
      registerCommand({
        id: "workbench.export-csv",
        title: t(language, "palette.cmd.exportCsv"),
        group: "workbench",
        keywords: ["export", "csv", "导出"],
        when: hasSamples,
        run: () => runExport("csv", exportArgsRef.current),
      }),
      registerCommand({
        id: "workbench.export-json",
        title: t(language, "palette.cmd.exportJson"),
        group: "workbench",
        keywords: ["export", "json", "导出"],
        when: hasSamples,
        run: () => runExport("json", exportArgsRef.current),
      }),
      registerCommand({
        id: "workbench.export-pdf",
        title: t(language, "palette.cmd.exportPdf"),
        group: "workbench",
        keywords: ["export", "pdf", "报告", "导出"],
        when: hasSamples,
        run: () => runExport("pdf", exportArgsRef.current),
      }),
      registerCommand({
        id: "workbench.clear-filters",
        title: t(language, "palette.cmd.clearFilters"),
        group: "workbench",
        keywords: ["clear", "reset", "清除"],
        when: isFiltered,
        run: () => resetRef.current(),
      }),
    ];
    return () => { offs.forEach((off) => off()); };
  }, [language]);

  return (
    <section className={styles.resultsWorkbenchShell} aria-label={t(language, "wb.aria")}>
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

      <section className={styles.resultsToolbarPanel} aria-label={t(language, "wb.controls")}>
        <div className={styles.resultsToolbarCopy}>
          <span className="results-kicker">{t(language, "wb.controls")}</span>
          <h3>{t(language, "wb.filterSort")}</h3>
          <p>{t(language, "wb.showing", { visible: visibleSamples.length, total: samples.length })}</p>
        </div>
        <div className={styles.resultsToolbarControls}>
          <label className={styles.resultsSearchField}>
            <span>{t(language, "wb.search")}</span>
            <input
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder={t(language, "wb.searchPlaceholder")}
            />
          </label>
          <label className={styles.resultsSortField}>
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
        <div className={styles.resultsFilterRow}>
          {statusOptions.map((option) => (
            <button
              key={option.key}
              type="button"
              className={`${styles.resultsFilterChip}${statusFilter === option.key ? " is-active" : ""}`}
              onClick={() => setStatusFilter(option.key)}
            >
              {option.label}
            </button>
          ))}
          {hasActiveControls ? (
            <button type="button" className={styles.resultsClearFilters} onClick={reset}>
              {t(language, "wb.clear")}
            </button>
          ) : null}
          <div className="results-density-toggle" role="group" aria-label={t(language, "wb.density.detailed")}>
            {(["compact", "detailed"] as const).map((d) => (
              <button
                key={d}
                type="button"
                className={`${styles.resultsFilterChip}${density === d ? " is-active" : ""}`}
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
        compareIds={compareIds}
        onToggleCompare={handleToggleCompare}
      />
      <AnimatePresence>
        {selectedSample != null && (
          <DetailDrawer
            key="detail-drawer"
            sample={selectedSample}
            language={language}
            analysisId={analysisId ?? null}
            onClose={() => setSelectedId(null)}
          />
        )}
      </AnimatePresence>

      {compareIds.length > 0 ? (
        <div className={styles.compareBar} role="region" aria-label={t(language, "compare.title")}>
          <span className={styles.compareBarIcon} aria-hidden="true">
            <Icon name="compare" size={14} />
          </span>
          <span className={styles.compareBarLabel}>
            {compareIds.length === 2
              ? t(language, "compare.bar.two")
              : t(language, "compare.bar.one")}
          </span>
          <span className={styles.compareBarIds} title={compareIds.join(", ")}>
            {compareIds.join(" · ")}
          </span>
          <button
            type="button"
            className={`${styles.compareBarButton} is-primary`}
            disabled={!compareReady}
            onClick={() => setCompareOpen(true)}
          >
            {t(language, "compare.button")}
          </button>
          <button
            type="button"
            className={styles.compareBarButton}
            onClick={handleClearCompare}
          >
            {t(language, "compare.clear")}
          </button>
        </div>
      ) : null}

      <AnimatePresence>
        {compareOpen && compareReady && compareLeft && compareRight ? (
          <CompareView
            key="compare-view"
            left={compareLeft}
            right={compareRight}
            language={language}
            onClose={() => setCompareOpen(false)}
          />
        ) : null}
      </AnimatePresence>
    </section>
  );
}
