import { useState, useEffect } from "react";
import "./HistoryPanel.css";

const { invoke } = window.electronAPI;

interface HistoryRecord {
  id: string;
  timestamp: number;
  sourcePath: string;
  sampleCount: number;
  status: "success" | "failed";
}

export function HistoryPanel() {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setIsLoading(true);
    try {
      const raw = (await invoke("get-history")) as string;
      const data = JSON.parse(raw) as HistoryRecord[];
      setRecords(data);
    } catch (error) {
      console.error("Failed to load history:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="history-panel">
      <header className="history-panel-header">
        <h3>Analysis History</h3>
        <button
          onClick={loadHistory}
          disabled={isLoading}
          className="btn-secondary"
        >
          {isLoading ? "Loading..." : "Refresh"}
        </button>
      </header>

      <div className="history-list">
        {records.length === 0 ? (
          <div className="history-empty-state">
            <p>No analysis history yet.</p>
            <p>Run an analysis to see it here.</p>
          </div>
        ) : (
          records.map((record) => (
            <div key={record.id} className="history-item">
              <div className="history-item-main">
                <span className="history-date">{formatDate(record.timestamp)}</span>
                <span className={`history-status ${record.status}`}>
                  {record.status}
                </span>
              </div>
              <div className="history-item-details">
                <span className="history-path">{record.sourcePath}</span>
                <span className="history-count">
                  {record.sampleCount} samples
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
