interface AnalysisPanelProps {
  result: any;
}

export function AnalysisPanel({ result }: AnalysisPanelProps) {
  const sampleCount = result?.sample_count ?? result?.totalSamples ?? 0;
  return (
    <div className="result-panel">
      <div className="hero-card">
        <div className="hero-label">分析结果</div>
        <div className="hero-value">{sampleCount}</div>
        <div className="hero-subtitle">当前识别的样本数量</div>
      </div>

      <div className="detail-card">
        <h3>结果摘要</h3>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </div>
    </div>
  );
}
