import { useMemo } from "react";
import type { Sample } from "../../../shared/types";
import { groupErrorsIntoRegions } from "../../../utils/sequence";
import "./ReportPanel.css";

interface ReportPanelProps {
  samples: Sample[];
  selectedSampleId: string | null;
}

export function ReportPanel({ samples, selectedSampleId }: ReportPanelProps) {
  const selectedSample = samples.find((s) => s.id === selectedSampleId);

  const stats = useMemo(() => {
    if (!selectedSample) return null;

    const totalSamples = samples.length;
    const okSamples = samples.filter((s) => s.status === "ok").length;
    const wrongSamples = samples.filter((s) => s.status === "wrong").length;
    const errorSamples = samples.filter((s) => s.status === "error").length;

    // Calculate error regions if we have alignment data
    let errorRegions = 0;
    if (selectedSample.matches) {
      const mismatchPositions = selectedSample.matches
        .map((m, i) => (m ? -1 : i))
        .filter((i) => i >= 0);
      errorRegions = groupErrorsIntoRegions(mismatchPositions, 5).length;
    }

    return {
      totalSamples,
      okSamples,
      wrongSamples,
      errorSamples,
      identity: selectedSample.identity,
      coverage: selectedSample.coverage,
      mutationCount: selectedSample.mutations?.length || 0,
      errorRegions,
      frameshift: selectedSample.frameshift,
    };
  }, [samples, selectedSample]);

  if (!selectedSample) {
    return (
      <div className="report-panel">
        <header className="report-panel-header">
          <h3>Analysis Report</h3>
        </header>
        <div className="report-empty">
          <p>Select a sample to view detailed report</p>
        </div>
      </div>
    );
  }

  return (
    <div className="report-panel">
      <header className="report-panel-header">
        <h3>Analysis Report</h3>
        <span className="report-sample-name">
          {selectedSample.name || selectedSample.clone || selectedSample.id}
        </span>
      </header>

      <div className="report-content">
        {/* Statistics Section */}
        <section className="report-section">
          <h4>Statistics</h4>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Identity</span>
              <span className="stat-value">
                {(stats?.identity || 0 * 100).toFixed(1)}%
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Coverage</span>
              <span className="stat-value">
                {(stats?.coverage || 0 * 100).toFixed(1)}%
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Error Regions</span>
              <span className="stat-value">{stats?.errorRegions || 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Mutations</span>
              <span className="stat-value">{stats?.mutationCount || 0}</span>
            </div>
          </div>

          {selectedSample.frameshift && (
            <div className="frameshift-warning">
              ⚠️ Frameshift detected - protein translation may be affected
            </div>
          )}

          {selectedSample.llmVerdict && (
            <div className="llm-verdict">
              <h5>AI Analysis</h5>
              <p>{selectedSample.llmVerdict}</p>
            </div>
          )}
        </section>

        {/* Mutations Table */}
        <section className="report-section">
          <h4>Mutations</h4>
          {selectedSample.mutations && selectedSample.mutations.length > 0 ? (
            <table className="mutation-table">
              <thead>
                <tr>
                  <th>Position</th>
                  <th>Ref</th>
                  <th>Query</th>
                  <th>Type</th>
                  <th>Ref AA</th>
                  <th>Query AA</th>
                  <th>Effect</th>
                </tr>
              </thead>
              <tbody>
                {selectedSample.mutations.map((mutation, idx) => (
                  <tr key={idx}>
                    <td>{mutation.position}</td>
                    <td className="base-cell ref">{mutation.refBase}</td>
                    <td className="base-cell query">{mutation.queryBase}</td>
                    <td>{mutation.type}</td>
                    <td>{mutation.refAA || "-"}</td>
                    <td>{mutation.queryAA || "-"}</td>
                    <td>{mutation.effect || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="no-data">No mutations detected</p>
          )}
        </section>

        {/* Batch Summary */}
        <section className="report-section">
          <h4>Batch Summary</h4>
          <div className="batch-stats">
            <div className="batch-stat">
              <span className="batch-stat-value">{stats?.totalSamples || 0}</span>
              <span className="batch-stat-label">Total Samples</span>
            </div>
            <div className="batch-stat ok">
              <span className="batch-stat-value">{stats?.okSamples || 0}</span>
              <span className="batch-stat-label">OK</span>
            </div>
            <div className="batch-stat wrong">
              <span className="batch-stat-value">{stats?.wrongSamples || 0}</span>
              <span className="batch-stat-label">Wrong</span>
            </div>
            <div className="batch-stat error">
              <span className="batch-stat-value">{stats?.errorSamples || 0}</span>
              <span className="batch-stat-label">Error</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
