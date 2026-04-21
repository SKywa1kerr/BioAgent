import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface LabSuggestionPanelProps {
  result: any;
  language: AppLanguage;
}

export function LabSuggestionPanel({ result, language }: LabSuggestionPanelProps) {
  const diagnoses = result?.diagnoses ?? [];
  const suggestions = result?.suggestions ?? [];

  return (
    <div className="result-panel">
      <div className="hero-card">
        <div className="hero-label">{t(language, "lab.overallHealth")}</div>
        <div className="hero-value small">{result?.overall_health ?? "unknown"}</div>
        <div className="hero-subtitle">{result?.summary ?? t(language, "lab.noSummary")}</div>
      </div>

      <div className="detail-card">
        <h3>{t(language, "lab.diagnosis")}</h3>
        <div className="diagnosis-list">
          {diagnoses.length ? diagnoses.map((item: any, index: number) => (
            <div key={index} className={`diagnosis-item severity-${item.severity}`}>
              <div className="diagnosis-top">
                <strong>{item.clone}</strong>
                <span>{item.issue}</span>
              </div>
              <p>{item.suggestion}</p>
            </div>
          )) : <div className="empty-state">{t(language, "lab.noDiagnosis")}</div>}
        </div>
      </div>

      <div className="detail-card">
        <h3>{t(language, "lab.tags")}</h3>
        <div className="tag-list">
          {suggestions.length ? suggestions.map((item: string, index: number) => (
            <span key={index} className="tag-chip">{item}</span>
          )) : <div className="empty-state">{t(language, "lab.noTags")}</div>}
        </div>
      </div>
    </div>
  );
}
