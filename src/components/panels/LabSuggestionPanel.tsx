interface LabSuggestionPanelProps {
  result: any;
}

export function LabSuggestionPanel({ result }: LabSuggestionPanelProps) {
  const diagnoses = result?.diagnoses ?? [];
  const suggestions = result?.suggestions ?? [];

  return (
    <div className="result-panel">
      <div className="hero-card">
        <div className="hero-label">整体健康度</div>
        <div className="hero-value small">{result?.overall_health ?? "unknown"}</div>
        <div className="hero-subtitle">{result?.summary ?? "暂无总结"}</div>
      </div>

      <div className="detail-card">
        <h3>诊断项</h3>
        <div className="diagnosis-list">
          {diagnoses.length ? diagnoses.map((item: any, index: number) => (
            <div key={index} className={`diagnosis-item severity-${item.severity}`}>
              <div className="diagnosis-top">
                <strong>{item.clone}</strong>
                <span>{item.issue}</span>
              </div>
              <p>{item.suggestion}</p>
            </div>
          )) : <div className="empty-state">没有异常诊断</div>}
        </div>
      </div>

      <div className="detail-card">
        <h3>建议标签</h3>
        <div className="tag-list">
          {suggestions.length ? suggestions.map((item: string, index: number) => (
            <span key={index} className="tag-chip">{item}</span>
          )) : <div className="empty-state">暂无建议标签</div>}
        </div>
      </div>
    </div>
  );
}
