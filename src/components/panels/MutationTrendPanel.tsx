import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface MutationTrendPanelProps {
  result: any;
}

export function MutationTrendPanel({ result }: MutationTrendPanelProps) {
  const hotspots = result?.mutation_hotspots ?? [];
  const insights = result?.insights ?? [];

  return (
    <div className="result-panel">
      <div className="hero-grid">
        <div className="hero-card">
          <div className="hero-label">样本总数</div>
          <div className="hero-value">{result?.total_samples ?? 0}</div>
        </div>
        <div className="hero-card">
          <div className="hero-label">突变总数</div>
          <div className="hero-value">{result?.total_mutations ?? 0}</div>
        </div>
        <div className="hero-card">
          <div className="hero-label">热点数</div>
          <div className="hero-value">{hotspots.length}</div>
        </div>
      </div>

      <div className="detail-card chart-card">
        <h3>突变热点</h3>
        {hotspots.length ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={hotspots.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
              <XAxis dataKey="position" stroke="#9fb5ff" />
              <YAxis stroke="#9fb5ff" />
              <Tooltip />
              <Bar dataKey="count" fill="#4f7cff" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-state">暂无重复热点</div>
        )}
      </div>

      <div className="detail-card">
        <h3>分析洞察</h3>
        <ul className="insight-list">
          {insights.map((insight: string, index: number) => (
            <li key={index}>{insight}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
