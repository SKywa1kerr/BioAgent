import type { AppLanguage, Sample } from "../types";
import type { ResultBuckets } from "./ResultsWorkbench";
import { t } from "../i18n";

interface ResultsStatsProps {
  language: AppLanguage;
  samples: Sample[];
  buckets: ResultBuckets;
}

interface StatRow {
  key: keyof ResultBuckets;
  tone: "success" | "danger" | "warning" | "neutral";
}

const STAT_ROWS: StatRow[] = [
  { key: "ok", tone: "success" },
  { key: "wrong", tone: "danger" },
  { key: "uncertain", tone: "warning" },
  { key: "untested", tone: "neutral" },
];

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function ResultsStats({ language, samples, buckets }: ResultsStatsProps) {
  const total = samples.length;
  const abnormalCount = buckets.wrong + buckets.uncertain + buckets.untested;
  const averageIdentity =
    total > 0 ? samples.reduce((sum, sample) => sum + (sample.identity || 0), 0) / total : 0;
  const averageCoverage =
    total > 0 ? samples.reduce((sum, sample) => sum + (sample.coverage || 0), 0) / total : 0;

  return (
    <section className="results-stats-panel" aria-label={t(language, "results.distributionTitle")}>
      <div className="results-section-header results-section-header-compact">
        <div>
          <span className="results-kicker">{t(language, "results.distributionKicker")}</span>
          <h3>{t(language, "results.distributionTitle")}</h3>
        </div>
        <p>{t(language, "results.distributionBody")}</p>
      </div>
      {total === 0 ? (
        <div className="results-empty-state">{t(language, "results.empty")}</div>
      ) : (
        <>
          <div className="results-overview-grid">
            <article className="results-overview-card tone-danger">
              <span className="results-overview-label">{t(language, "results.abnormalCount")}</span>
              <strong className="results-overview-value">{abnormalCount}</strong>
            </article>
            <article className="results-overview-card">
              <span className="results-overview-label">{t(language, "results.avgIdentity")}</span>
              <strong className="results-overview-value">{formatPercent(averageIdentity)}</strong>
            </article>
            <article className="results-overview-card">
              <span className="results-overview-label">{t(language, "results.avgCoverage")}</span>
              <strong className="results-overview-value">{formatPercent(averageCoverage)}</strong>
            </article>
          </div>
          <div className="results-stats-list">
            {STAT_ROWS.map(({ key, tone }) => {
              const count = buckets[key];
              const share = total > 0 ? count / total : 0;

              return (
                <div className={`results-stat-row tone-${tone}`} key={key}>
                  <div className="results-stat-label">
                    <strong>{t(language, `results.${key}`)}</strong>
                    <span>{`${count} / ${total}`}</span>
                  </div>
                  <div className="results-stat-bar" aria-hidden="true">
                    <span className="results-stat-fill" style={{ width: `${share * 100}%` }} />
                  </div>
                  <div className="results-stat-value">{`${(share * 100).toFixed(1)}%`}</div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
