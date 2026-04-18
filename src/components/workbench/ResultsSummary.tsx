import { formatPercent } from "./utils";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import type { SummaryScope } from "../../hooks/useWorkbenchControls";
import { SummaryScopeToggle } from "./SummaryScopeToggle";

interface ResultsSummaryProps {
  total: number;
  ok: number;
  wrong: number;
  uncertain: number;
  untested: number;
  averageIdentity: number;
  averageCoverage: number;
  language: AppLanguage;
  scope: SummaryScope;
  onScopeChange: (scope: SummaryScope) => void;
  originalTotal: number;
  filteredTotal: number;
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
  total,
  ok,
  wrong,
  uncertain,
  untested,
  averageIdentity,
  averageCoverage,
  language,
  scope,
  onScopeChange,
  originalTotal,
  filteredTotal,
}: ResultsSummaryProps) {
  return (
    <section className="results-summary-panel" aria-label={t(language, "summary.kicker")}>
      <div className="results-section-header">
        <div>
          <span className="results-kicker">{t(language, "summary.kicker")}</span>
          <h3>{t(language, "summary.title")}</h3>
        </div>
        <p>{t(language, "summary.body")}</p>
        <SummaryScopeToggle
          scope={scope}
          onChange={onScopeChange}
          language={language}
          filteredTotal={filteredTotal}
          originalTotal={originalTotal}
        />
      </div>

      <div className="results-summary-grid">
        <SummaryCard label={t(language, "summary.total")} value={`${total}`} tone="accent" />
        <SummaryCard label={t(language, "summary.ok")} value={`${ok}`} tone="success" />
        <SummaryCard label={t(language, "summary.wrong")} value={`${wrong}`} tone="danger" />
        <SummaryCard label={t(language, "summary.uncertain")} value={`${uncertain}`} tone="warning" />
        <SummaryCard label={t(language, "summary.untested")} value={`${untested}`} tone="neutral" />
        <SummaryCard label={t(language, "summary.avgIdentity")} value={formatPercent(averageIdentity)} />
        <SummaryCard label={t(language, "summary.avgCoverage")} value={formatPercent(averageCoverage)} />
      </div>

      <div className="results-metric-help" role="note" aria-label={t(language, "summary.metricTip")}>
        <span className="results-metric-help-title">{t(language, "summary.metricTip")}</span>
        <p>
          <strong>{t(language, "summary.avgIdentity")}: </strong>
          {t(language, "summary.identityHelp")}
        </p>
        <p>
          <strong>{t(language, "summary.avgCoverage")}: </strong>
          {t(language, "summary.coverageHelp")}
        </p>
      </div>
    </section>
  );
}
