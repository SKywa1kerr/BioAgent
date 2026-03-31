import React, { useEffect, useState } from "react";
import { AnalysisRecord } from "../types";
import "./HistoryPage.css";

const { invoke } = window.electronAPI;

export const HistoryPage: React.FC = () => {
  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const result = await invoke("get-history") as string;
        setRecords(JSON.parse(result));
      } catch (e) {
        console.error("Failed to load history:", e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <div className="history-page"><p>Loading...</p></div>;

  return (
    <div className="history-page">
      <h2>Analysis History</h2>
      {records.length === 0 ? (
        <p className="empty">No analysis records yet</p>
      ) : (
        <table className="history-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Source</th>
              <th>Total</th>
              <th>OK</th>
              <th>Wrong</th>
              <th>Uncertain</th>
              <th>Pass Rate</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.id}>
                <td>{new Date(r.created_at).toLocaleString()}</td>
                <td title={r.source_path}>
                  {r.source_path ? `...${r.source_path.slice(-30)}` : "-"}
                </td>
                <td>{r.total}</td>
                <td className="ok">{r.ok_count}</td>
                <td className="wrong">{r.wrong_count}</td>
                <td className="uncertain">{r.uncertain_count}</td>
                <td>{r.total > 0 ? `${(r.ok_count / r.total * 100).toFixed(1)}%` : "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};
