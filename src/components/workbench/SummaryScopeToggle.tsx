import type { SummaryScope } from "../../hooks/useWorkbenchControls";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface Props {
  scope: SummaryScope;
  onChange: (scope: SummaryScope) => void;
  language: AppLanguage;
  filteredTotal: number;
  originalTotal: number;
}

export function SummaryScopeToggle({ scope, onChange, language, filteredTotal, originalTotal }: Props) {
  if (filteredTotal === originalTotal) return null;
  return (
    <div className="summary-scope-toggle" role="radiogroup" aria-label={t(language, "summary.scope.label")}>
      <button
        type="button"
        role="radio"
        aria-checked={scope === "filtered"}
        className={scope === "filtered" ? "is-active" : ""}
        onClick={() => onChange("filtered")}
      >
        {t(language, "summary.scope.filtered")} ({filteredTotal})
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={scope === "all"}
        className={scope === "all" ? "is-active" : ""}
        onClick={() => onChange("all")}
      >
        {t(language, "summary.scope.all")} ({originalTotal})
      </button>
    </div>
  );
}
