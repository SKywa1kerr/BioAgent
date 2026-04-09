import React, { useEffect, useMemo, useState } from "react";
import { AnalysisRecord, AppLanguage } from "../types";
import { t } from "../i18n";
import "./HistoryPage.css";

const { invoke } = window.electronAPI;

const formatDateTime = (language: AppLanguage, value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return {
      date: t(language, "history.unknownDate"),
      time: t(language, "history.unknownTime"),
    };
  }

  const locale = language === "zh" ? "zh-CN" : "en-US";

  return {
    date: date.toLocaleDateString(locale, {
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
    time: date.toLocaleTimeString(locale, {
      hour: "2-digit",
      minute: "2-digit",
    }),
  };
};

const formatSourceName = (language: AppLanguage, sourcePath: string) => {
  if (!sourcePath) return t(language, "history.unassignedSource");
  const parts = sourcePath.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || sourcePath;
};

const formatRecordTitle = (
  language: AppLanguage,
  record: AnalysisRecord,
  index: number
) => {
  const sourceName = formatSourceName(language, record.source_path);
  if (record.source_path && sourceName !== t(language, "history.unassignedSource")) {
    return sourceName;
  }
  return `${t(language, "history.runLabel")} ${index + 1}`;
};

interface HistoryPageProps {
  language: AppLanguage;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({ language }) => {
  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const result = (await invoke("get-history")) as string;
        const parsed = JSON.parse(result);
        setRecords(Array.isArray(parsed) ? parsed : []);
        setLoadError(null);
      } catch (e) {
        console.error("Failed to load history:", e);
        setLoadError(t(language, "history.loadError"));
      } finally {
        setLoading(false);
      }
    })();
  }, [language]);

  const sortedRecords = useMemo(
    () =>
      [...records].sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      ),
    [records]
  );

  const stats = useMemo(() => {
    const totalRuns = sortedRecords.length;
    const totalAnalyses = sortedRecords.reduce((sum, record) => sum + record.total, 0);
    const totalPassed = sortedRecords.reduce((sum, record) => sum + record.ok_count, 0);
    const overallPassRate = totalAnalyses > 0 ? (totalPassed / totalAnalyses) * 100 : 0;
    const latestRecord = sortedRecords[0];

    return {
      totalRuns,
      totalAnalyses,
      overallPassRate,
      latestRecord,
    };
  }, [sortedRecords]);

  if (loading) {
    return (
      <div className="history-page">
        <div className="history-shell">
          <div className="history-hero history-loading">
              <div className="history-hero-copy">
              <span className="history-eyebrow">{t(language, "history.eyebrow")}</span>
              <h2>{t(language, "history.loadingTitle")}</h2>
              <p>{t(language, "history.loadingBody")}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="history-page">
      <div className="history-shell">
        <section className="history-hero">
          <div className="history-hero-copy">
            <span className="history-eyebrow">{t(language, "history.eyebrow")}</span>
            <h2>{t(language, "history.title")}</h2>
            <p>{t(language, "history.intro")}</p>
          </div>
          <div className="history-hero-badge">
            <span>{t(language, "history.archivedRuns")}</span>
            <strong>{stats.totalRuns}</strong>
          </div>
        </section>

        <section className="history-summary">
          <article className="summary-card">
            <span className="summary-label">{t(language, "history.runsArchived")}</span>
            <strong>{stats.totalRuns}</strong>
            <p>{t(language, "history.runsArchivedBody")}</p>
          </article>
          <article className="summary-card">
            <span className="summary-label">{t(language, "history.samplesReviewed")}</span>
            <strong>{stats.totalAnalyses}</strong>
            <p>{t(language, "history.samplesReviewedBody")}</p>
          </article>
          <article className="summary-card summary-card-accent">
            <span className="summary-label">{t(language, "history.overallPassRate")}</span>
            <strong>
              {stats.totalAnalyses > 0 ? `${stats.overallPassRate.toFixed(1)}%` : t(language, "history.na")}
            </strong>
            <p>{t(language, "history.overallPassRateBody")}</p>
          </article>
          <article className="summary-card">
            <span className="summary-label">{t(language, "history.latestRun")}</span>
            <strong>
              {stats.latestRecord
                ? formatDateTime(language, stats.latestRecord.created_at).date
                : t(language, "history.na")}
            </strong>
            <p>
              {stats.latestRecord
                ? formatSourceName(language, stats.latestRecord.source_path)
                : t(language, "history.latestRunEmpty")}
            </p>
          </article>
        </section>

        {loadError ? (
          <div className="history-panel history-empty history-error" role="alert">
            <h3>{t(language, "history.archiveUnavailable")}</h3>
            <p>{loadError}</p>
          </div>
        ) : records.length === 0 ? (
          <div className="history-panel history-empty">
            <div className="empty-mark">
              <span />
            </div>
            <h3>{t(language, "history.emptyTitle")}</h3>
            <p>{t(language, "history.emptyBody")}</p>
          </div>
        ) : (
          <div className="history-list">
            {sortedRecords.map((record, index) => {
              const total = record.total || 0;
              const passRate = total > 0 ? (record.ok_count / total) * 100 : 0;
              const timestamp = formatDateTime(language, record.created_at);

              return (
                <article className="history-card" key={record.id}>
                  <div className="history-card-top">
                    <div>
                      <span className="record-index">
                        {t(language, "history.runLabel")} {index + 1}
                      </span>
                      <h3 title={record.source_path || record.name}>
                        {formatRecordTitle(language, record, index)}
                      </h3>
                    </div>
                    <div className="history-time" title={record.created_at}>
                      <span>{timestamp.date}</span>
                      <strong>{timestamp.time}</strong>
                    </div>
                  </div>

                  <div className="history-source" title={record.source_path}>
                    <span className="history-source-label">{t(language, "history.source")}</span>
                    <strong>{formatSourceName(language, record.source_path)}</strong>
                    <span className="history-source-path">
                      {record.source_path || t(language, "history.noSourcePath")}
                    </span>
                  </div>

                  <div className="history-metrics">
                    <div className="metric">
                      <span>{t(language, "history.total")}</span>
                      <strong>{record.total}</strong>
                    </div>
                    <div className="metric metric-ok">
                      <span>{t(language, "history.ok")}</span>
                      <strong>{record.ok_count}</strong>
                    </div>
                    <div className="metric metric-wrong">
                      <span>{t(language, "history.wrong")}</span>
                      <strong>{record.wrong_count}</strong>
                    </div>
                    <div className="metric metric-uncertain">
                      <span>{t(language, "history.uncertain")}</span>
                      <strong>{record.uncertain_count}</strong>
                    </div>
                  </div>

                  <div className="history-passrate">
                    <div className="history-passrate-copy">
                      <span>{t(language, "history.passRate")}</span>
                      <strong>
                        {total > 0 ? `${passRate.toFixed(1)}%` : t(language, "history.na")}
                      </strong>
                    </div>
                    <div className="history-progress" aria-hidden="true">
                      <div
                        className="history-progress-fill"
                        style={{ width: `${Math.max(0, Math.min(passRate, 100))}%` }}
                      />
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
