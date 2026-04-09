import type { ReactNode } from "react";
import type { AppLanguage, ResultWorkbenchStatus, Sample } from "../types";
import { t } from "../i18n";
import { ResultsCharts } from "./ResultsCharts";
import { ResultsSummary } from "./ResultsSummary";
import { SampleDetailsList } from "./SampleDetailsList";
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

const UNTESTED_REASON = "\u672a\u6d4b\u901a";

export function bucketSampleStatus(sample: Sample): ResultWorkbenchStatus {
  if (sample.reason === UNTESTED_REASON) {
    return "untested";
  }

  if (sample.status === "ok" || sample.status === "wrong") {
    return sample.status;
  }

  return "uncertain";
}

export function ResultsWorkbench({
  language,
  samples,
  selectedId,
  onSelect,
  children,
}: ResultsWorkbenchProps) {
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
        <SampleDetailsList
          language={language}
          samples={samples}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      ) : (
        children ?? null
      )}
    </section>
  );
}
