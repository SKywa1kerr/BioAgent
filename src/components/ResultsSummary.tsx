import type { AppLanguage } from "../types";
import { t } from "../i18n";

interface ResultsSummaryProps {
  language: AppLanguage;
  total: number;
  ok: number;
  wrong: number;
  uncertain: number;
  untested: number;
  averageIdentity: number;
  averageCoverage: number;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

interface SummaryCardProps {
  label: string;
  value: string;
  tone?: "neutral" | "success" | "danger" | "warning" | "accent";
}

function SummaryCard({ label, value, tone = "neutral" }: SummaryCardProps) {
  return (
    <article className={`results-summary-card tone-${tone}`}>
      <span className="results-summary-label">{label}</span>
      <strong className="results-summary-value">{value}</strong>
    </article>
  );
}

export function ResultsSummary({
  language,
  total,
  ok,
  wrong,
  uncertain,
  untested,
  averageIdentity,
  averageCoverage,
}: ResultsSummaryProps) {
  return (
    <section className="results-summary-panel" aria-label={t(language, "results.summaryTitle")}>
      <div className="results-section-header">
        <div>
          <span className="results-kicker">{t(language, "results.summaryKicker")}</span>
          <h3>{t(language, "results.summaryTitle")}</h3>
        </div>
        <p>{t(language, "results.summaryBody")}</p>
      </div>
      <div className="results-summary-grid">
        <SummaryCard label={t(language, "results.total")} value={`${total}`} tone="accent" />
        <SummaryCard label={t(language, "results.ok")} value={`${ok}`} tone="success" />
        <SummaryCard label={t(language, "results.wrong")} value={`${wrong}`} tone="danger" />
        <SummaryCard label={t(language, "results.uncertain")} value={`${uncertain}`} tone="warning" />
        <SummaryCard label={t(language, "results.untested")} value={`${untested}`} tone="neutral" />
        <SummaryCard
          label={t(language, "results.avgIdentity")}
          value={formatPercent(averageIdentity)}
        />
        <SummaryCard
          label={t(language, "results.avgCoverage")}
          value={formatPercent(averageCoverage)}
        />
      </div>
    </section>
  );
}
