import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface MutationTrendPanelProps {
  result: any;
  language: AppLanguage;
}

export function MutationTrendPanel({ result, language }: MutationTrendPanelProps) {
  const hotspots = result?.mutation_hotspots ?? [];
  const insights = result?.insights ?? [];

  return (
    <div className="result-panel">
      <div className="hero-grid">
        <div className="hero-card">
          <div className="hero-label">{t(language, "trends.totalSamples")}</div>
          <div className="hero-value">{result?.total_samples ?? 0}</div>
        </div>
        <div className="hero-card">
          <div className="hero-label">{t(language, "trends.totalMutations")}</div>
          <div className="hero-value">{result?.total_mutations ?? 0}</div>
        </div>
        <div className="hero-card">
          <div className="hero-label">{t(language, "trends.hotspots")}</div>
          <div className="hero-value">{hotspots.length}</div>
        </div>
      </div>

      <div className="detail-card chart-card">
        <h3>{t(language, "trends.hotspotTitle")}</h3>
        {hotspots.length ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={hotspots.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid-color)" />
              <XAxis dataKey="position" stroke="var(--chart-axis-color)" />
              <YAxis stroke="var(--chart-axis-color)" />
              <Tooltip />
              <Bar dataKey="count" fill="var(--chart-primary-color)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-state">{t(language, "trends.noHotspot")}</div>
        )}
      </div>

      <div className="detail-card">
        <h3>{t(language, "trends.insights")}</h3>
        <ul className="insight-list">
          {insights.map((insight: string, index: number) => (
            <li key={index}>{insight}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
