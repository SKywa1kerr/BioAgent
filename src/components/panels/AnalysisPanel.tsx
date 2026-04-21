import { lazy, Suspense, useMemo } from "react";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import { normalizeSamples } from "../workbench/normalize";

const ResultsWorkbench = lazy(async () => {
  const mod = await import("../workbench/ResultsWorkbench");
  return { default: mod.ResultsWorkbench };
});

interface AnalysisPanelProps {
  result: any;
  language: AppLanguage;
}

function loadingCard(language: AppLanguage, sampleCount: number | string, message: string) {
  return (
    <div className="result-panel">
      <div className="hero-card">
        <div className="hero-label">{t(language, "analysis.samplesAnalyzed")}</div>
        <div className="hero-value">{sampleCount}</div>
        <div className="hero-subtitle">{message}</div>
      </div>
    </div>
  );
}

export function AnalysisPanel({ result, language }: AnalysisPanelProps) {
  const samples = useMemo(() => normalizeSamples(result, language), [result, language]);
  const detailPending = Boolean(result?.__detailPending);
  const detailError = typeof result?.__detailError === "string" ? result.__detailError : "";
  const sampleCount = result?.sample_count ?? result?.totalSamples ?? samples.length ?? 0;
  const displayCount = sampleCount > 0 ? sampleCount : (detailPending ? "..." : 0);

  if (samples.length === 0) {
    return loadingCard(language, displayCount, detailError || (detailPending ? t(language, "analysis.loadingRows") : t(language, "analysis.detailFetchFailed")));
  }

  return (
    <Suspense fallback={loadingCard(language, sampleCount, t(language, "analysis.loadingRows"))}>
      <ResultsWorkbench samples={samples} language={language} />
    </Suspense>
  );
}
