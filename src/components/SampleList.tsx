import React from "react";
import { AppLanguage, Sample } from "../types";
import { t } from "../i18n";
import "./SampleList.css";

interface SampleListProps {
  language: AppLanguage;
  samples: Sample[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const STATUS_LABELS: Record<Sample["status"] | "error", string> = {
  ok: "OK",
  wrong: "NG",
  processing: "RUN",
  uncertain: "?",
  error: "ERR",
};

const StatusIcon: React.FC<{ status: Sample["status"] | "error" }> = ({ status }) => (
  <span className={`status-icon ${status === "uncertain" ? "warning" : status}`}>
    {STATUS_LABELS[status]}
  </span>
);

export const SampleList: React.FC<SampleListProps> = ({
  language,
  samples,
  selectedId,
  onSelect,
}) => {
  const okCount = samples.filter((sample) => sample.status === "ok").length;
  const flaggedCount = samples.filter((sample) => sample.status === "wrong").length;
  const reviewCount = samples.filter(
    (sample) => sample.status === "uncertain" || sample.status === "processing"
  ).length;

  return (
    <div className="sample-list">
      <div className="sample-list-header">
        <span className="sample-list-kicker">{t(language, "sampleRail.kicker")}</span>
        <h3>
          {t(language, "sampleRail.title")} ({samples.length})
        </h3>
        <div className="sample-list-summary">
          <span>
            {okCount} {t(language, "sampleRail.pass")}
          </span>
          <span>
            {flaggedCount} {t(language, "sampleRail.issue")}
          </span>
          <span>
            {reviewCount} {t(language, "sampleRail.review")}
          </span>
        </div>
      </div>
      <div className="sample-items">
        {samples.map((sample) => (
          <button
            key={sample.id}
            type="button"
            className={`sample-item ${selectedId === sample.id ? "selected" : ""} ${
              sample.status === "error" ? "has-error" : ""
            }`}
            onClick={() => onSelect(sample.id)}
            title={sample.error || sample.reason || ""}
          >
            <StatusIcon status={sample.status} />
            <div className="sample-info-compact">
              <div className="sample-title-row">
                <span className="sample-name">{sample.id}</span>
                {sample.clone ? <span className="sample-clone">{sample.clone}</span> : null}
              </div>
              {sample.reason && !sample.error ? (
                <span className="reason-text">{sample.reason}</span>
              ) : null}
              {sample.error ? <span className="error-text">{sample.error}</span> : null}
            </div>
            {sample.mutations && sample.mutations.length > 0 ? (
              <span className="mutation-count">{sample.mutations.length}</span>
            ) : null}
          </button>
        ))}
      </div>
    </div>
  );
};
