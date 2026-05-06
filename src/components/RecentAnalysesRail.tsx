import { useMemo } from "react";
import { Icon } from "./ui/Icon";
import type { HistoryItem } from "../hooks/useAnalysisHistory";
import { t, type AppLanguage } from "../i18n";
import "./RecentAnalysesRail.css";

interface RecentAnalysesRailProps {
  items: HistoryItem[];
  total: number;
  isLoading: boolean;
  activeId?: string | null;
  language: AppLanguage;
  onSelect: (analysisId: string) => void;
  onRefresh: () => void;
}

function formatTimestamp(value: string | undefined, language: AppLanguage): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return language === "zh" ? "刚刚" : "just now";
  if (diffMin < 60) {
    return language === "zh" ? `${diffMin} 分钟前` : `${diffMin}m ago`;
  }

  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) {
    return language === "zh" ? `${diffHr} 小时前` : `${diffHr}h ago`;
  }

  // Within 24h+ : show local time only.
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

export function RecentAnalysesRail({
  items,
  isLoading,
  activeId,
  language,
  onSelect,
  onRefresh,
}: RecentAnalysesRailProps) {
  const safeItems = useMemo(() => (Array.isArray(items) ? items : []), [items]);

  return (
    <section className="recent-rail" aria-label={t(language, "history.recent")}>
      <header className="recent-rail-header">
        <span className="recent-rail-title">
          <Icon name="history" size={14} />
          <span>{t(language, "history.recent")}</span>
        </span>
        <button
          type="button"
          className="recent-rail-refresh"
          onClick={onRefresh}
          aria-label={t(language, "history.refresh")}
          title={t(language, "history.refresh")}
          disabled={isLoading}
        >
          <Icon name="refresh" size={14} />
        </button>
      </header>

      {isLoading && safeItems.length === 0 ? (
        <div className="recent-rail-empty">{t(language, "history.loading")}</div>
      ) : safeItems.length === 0 ? (
        <div className="recent-rail-empty">{t(language, "history.empty")}</div>
      ) : (
        <ul className="recent-rail-list">
          {safeItems.map((item) => {
            const isActive = activeId != null && activeId === item.analysis_id;
            return (
              <li key={item.analysis_id}>
                <button
                  type="button"
                  className={`recent-rail-item${isActive ? " recent-rail-item-active" : ""}`}
                  onClick={() => onSelect(item.analysis_id)}
                  title={item.analysis_id}
                >
                  <div className="recent-rail-item-row">
                    <span className="recent-rail-dataset">{item.dataset}</span>
                    {item.used_llm ? (
                      <span className="recent-rail-llm" aria-label={t(language, "history.llmBadge")}>
                        {t(language, "history.llmBadge")}
                      </span>
                    ) : null}
                  </div>
                  <div className="recent-rail-item-meta">
                    <span className="recent-rail-samples">
                      {t(language, "history.samples", { count: item.sample_count ?? 0 })}
                    </span>
                    {item.created_at ? (
                      <span className="recent-rail-time">{formatTimestamp(item.created_at, language)}</span>
                    ) : null}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
