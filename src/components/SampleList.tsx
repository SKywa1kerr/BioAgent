import React from "react";
import { Sample } from "../types";
import "./SampleList.css";

interface SampleListProps {
  samples: Sample[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const StatusIcon: React.FC<{ status: Sample["status"] | "error" }> = ({
  status,
}) => {
  switch (status) {
    case "ok":
      return <span className="status-icon ok">✅</span>;
    case "wrong":
      return <span className="status-icon wrong">❌</span>;
    case "processing":
      return <span className="status-icon processing">⏳</span>;
    case "uncertain":
      return <span className="status-icon warning">⚠️</span>;
    case "error":
      return <span className="status-icon error">🚫</span>;
    default:
      return <span className="status-icon">❓</span>;
  }
};

export const SampleList: React.FC<SampleListProps> = ({
  samples,
  selectedId,
  onSelect,
}) => {
  return (
    <div className="sample-list">
      <h3>Samples ({samples.length})</h3>
      <div className="sample-items">
        {samples.map((sample) => (
          <div
            key={sample.id}
            className={`sample-item ${selectedId === sample.id ? "selected" : ""} ${
              sample.status === "error" ? "has-error" : ""
            }`}
            onClick={() => onSelect(sample.id)}
            title={sample.error}
          >
            <StatusIcon status={sample.status} />
            <div className="sample-info-compact">
              <span className="sample-name">{sample.id}</span>
              {sample.error && <span className="error-text">{sample.error}</span>}
            </div>
            {sample.mutations && sample.mutations.length > 0 && (
              <span className="mutation-count">{sample.mutations.length}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
