import type { AppLanguage, Sample } from "../types";
import { t } from "../i18n";

interface ResultsTableProps {
  language: AppLanguage;
  samples: Sample[];
  selectedId: string | null;
  onSelect: (sampleId: string) => void;
}

type TableStatus = "ok" | "wrong" | "uncertain" | "untested";

const UNTESTED_REASON = "\u672a\u6d4b\u901a";

function firstNonEmpty(...values: Array<string | undefined>) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function getStatus(sample: Sample): TableStatus {
  if (sample.reason === UNTESTED_REASON) {
    return "untested";
  }

  if (sample.status === "ok" || sample.status === "wrong") {
    return sample.status;
  }

  return "uncertain";
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function getDisplayReason(sample: Sample) {
  const directReason = firstNonEmpty(sample.reason);
  if (directReason) {
    return directReason;
  }

  if (getStatus(sample) !== "ok") {
    return firstNonEmpty(sample.review_reason, sample.llm_reason, sample.auto_reason, sample.error) || "-";
  }

  return "-";
}

export function ResultsTable({
  language,
  samples,
  selectedId,
  onSelect,
}: ResultsTableProps) {
  return (
    <section className="results-table-panel" aria-label={t(language, "results.tableTitle")}>
      <div className="results-section-header results-section-header-compact">
        <div>
          <span className="results-kicker">{t(language, "results.tableKicker")}</span>
          <h3>{t(language, "results.tableTitle")}</h3>
        </div>
        <p>{t(language, "results.tableBody")}</p>
      </div>

      <div className="results-table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th>{t(language, "results.sample")}</th>
              <th>{t(language, "results.clone")}</th>
              <th>{t(language, "results.status")}</th>
              <th>{t(language, "results.reason")}</th>
              <th>{t(language, "results.identity")}</th>
              <th>{t(language, "results.coverage")}</th>
              <th>{t(language, "results.mutations")}</th>
            </tr>
          </thead>
          <tbody>
            {samples.length === 0 ? (
              <tr className="results-table-empty-row">
                <td colSpan={7}>
                  <div className="results-table-empty">
                    <strong>{t(language, "results.tableEmptyTitle")}</strong>
                    <span>{t(language, "results.tableEmptyBody")}</span>
                  </div>
                </td>
              </tr>
            ) : (
              samples.map((sample) => {
                const status = getStatus(sample);
                const isSelected = selectedId === sample.id;
                const mutationCount = sample.mutations?.length ?? 0;
                const displayReason = getDisplayReason(sample);

                return (
                  <tr
                    key={sample.id}
                    className={isSelected ? "is-selected" : ""}
                    aria-selected={isSelected}
                    tabIndex={0}
                    onClick={() => onSelect(sample.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onSelect(sample.id);
                      }
                    }}
                  >
                    <td className="results-table-sample">{sample.id}</td>
                    <td className="results-table-clone">
                      <span className="results-table-pill">{sample.clone || "-"}</span>
                    </td>
                    <td>
                      <span className={`results-table-status status-${status}`}>
                        {t(language, `results.${status}`)}
                      </span>
                    </td>
                    <td className="results-table-reason">
                      {displayReason !== "-" ? (
                        <>
                          {displayReason === UNTESTED_REASON ? (
                            <span className="results-table-inline-badge status-untested">
                              {t(language, "results.untested")}
                            </span>
                          ) : null}
                          <span>{displayReason}</span>
                        </>
                      ) : (
                        <span className="results-table-muted">-</span>
                      )}
                    </td>
                    <td className="results-table-metric">{formatPercent(sample.identity)}</td>
                    <td className="results-table-metric">{formatPercent(sample.coverage)}</td>
                    <td className="results-table-metric">{mutationCount}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}


