import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { AppLanguage, Sample } from "../types";
import { t } from "../i18n";
import "./ResultsCharts.css";

type StatusBucket = "ok" | "wrong" | "uncertain" | "untested";

const STATUS_COLORS: Record<StatusBucket, string> = {
  ok: "#16a34a",
  wrong: "#dc2626",
  uncertain: "#d97706",
  untested: "#64748b",
};

const UNTESTED_REASON = "\u672a\u6d4b\u901a";
const BIN_COUNT = 10;

interface ResultsChartsProps {
  language: AppLanguage;
  samples: Sample[];
}

interface DistributionBin {
  label: string;
  ok: number;
  wrong: number;
  uncertain: number;
  untested: number;
  total: number;
}

interface ScatterPoint {
  id: string;
  name: string;
  status: StatusBucket;
  identity: number;
  coverage: number;
  size: number;
}

interface StatusSlice {
  name: StatusBucket;
  value: number;
  fill: string;
}

function getStatus(sample: Sample): StatusBucket {
  if (sample.reason === UNTESTED_REASON) {
    return "untested";
  }

  if (sample.status === "ok" || sample.status === "wrong") {
    return sample.status;
  }

  return "uncertain";
}

function normalizePercent(value: number) {
  if (!Number.isFinite(value)) {
    return null;
  }

  return value <= 1.5 ? value * 100 : value;
}

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

function average(values: Array<number | undefined>) {
  const filtered = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (filtered.length === 0) {
    return undefined;
  }

  return filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
}

function getCoverageValue(sample: Sample) {
  return sample.cds_coverage ?? sample.coverage;
}

function getIdentityValue(sample: Sample) {
  return sample.identity;
}

function getQualityScore(sample: Sample) {
  const chromatogramQuality = average(sample.quality ?? []);
  return sample.avg_quality ?? chromatogramQuality ?? sample.aa_changes_n ?? sample.mutations.length ?? 1;
}

function getBubbleSize(score: number) {
  return Math.max(4, Math.min(18, 4 + score / 4));
}

function buildBins(samples: Sample[], accessor: (sample: Sample) => number | undefined | null) {
  const bins: DistributionBin[] = Array.from({ length: BIN_COUNT }, (_, index) => {
    const lower = index * (100 / BIN_COUNT);
    const upper = index === BIN_COUNT - 1 ? 100 : (index + 1) * (100 / BIN_COUNT);

    return {
      label: `${Math.round(lower)}-${Math.round(upper)}%`,
      ok: 0,
      wrong: 0,
      uncertain: 0,
      untested: 0,
      total: 0,
    };
  });

  for (const sample of samples) {
    const rawValue = accessor(sample);
    const value = rawValue == null ? null : normalizePercent(rawValue);

    if (value == null) {
      continue;
    }

    const clamped = Math.min(100, Math.max(0, value));
    const index = Math.min(BIN_COUNT - 1, Math.floor(clamped / (100 / BIN_COUNT)));
    const status = getStatus(sample);

    bins[index][status] += 1;
    bins[index].total += 1;
  }

  return bins;
}

function buildScatterData(samples: Sample[]): ScatterPoint[] {
  return samples
    .map((sample) => {
      const identity = normalizePercent(getIdentityValue(sample));
      const coverage = normalizePercent(getCoverageValue(sample));

      if (identity == null || coverage == null) {
        return null;
      }

      return {
        id: sample.id,
        name: sample.clone || sample.id,
        status: getStatus(sample),
        identity,
        coverage,
        size: getBubbleSize(getQualityScore(sample)),
      };
    })
    .filter((point): point is ScatterPoint => point !== null);
}

function buildStatusSlices(samples: Sample[]): StatusSlice[] {
  const counts: Record<StatusBucket, number> = {
    ok: 0,
    wrong: 0,
    uncertain: 0,
    untested: 0,
  };

  for (const sample of samples) {
    counts[getStatus(sample)] += 1;
  }

  return (Object.keys(counts) as StatusBucket[]).map((status) => ({
    name: status,
    value: counts[status],
    fill: STATUS_COLORS[status],
  }));
}

function MiniMetric({
  language,
  status,
  value,
}: {
  language: AppLanguage;
  status: StatusBucket;
  value: number;
}) {
  return (
    <div className="results-chart-mini">
      <span className="results-chart-mini-label">
        <i className="results-chart-mini-swatch" style={{ backgroundColor: STATUS_COLORS[status] }} />
        {t(language, `results.${status}`)}
      </span>
      <strong className="results-chart-mini-value">{value}</strong>
    </div>
  );
}

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <article className="results-chart-card">
      <h4 className="results-chart-card-title">{title}</h4>
      <div className="results-chart-card-body">{children}</div>
    </article>
  );
}

export function ResultsCharts({ language, samples }: ResultsChartsProps) {
  const hasSamples = samples.length > 0;
  const identityBins = buildBins(samples, getIdentityValue);
  const coverageBins = buildBins(samples, getCoverageValue);
  const scatterData = buildScatterData(samples);
  const statusSlices = buildStatusSlices(samples);
  const total = samples.length;

  return (
    <section className="results-charts-panel" aria-label={t(language, "results.chartsTitle")}>
      <div className="results-section-header results-section-header-compact results-charts-header">
        <div>
          <h3>{t(language, "results.chartsTitle")}</h3>
        </div>
      </div>

      <div className="results-charts-grid">
        <ChartCard title={t(language, "results.identityDistributionTitle")}>
          {hasSamples ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={identityBins} margin={{ top: 8, right: 8, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="var(--results-grid)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="label"
                  interval={0}
                  tickLine={false}
                  axisLine={{ stroke: "var(--results-axis)" }}
                  tick={{ fill: "var(--results-axis)", fontSize: 11 }}
                  angle={-35}
                  textAnchor="end"
                  height={56}
                />
                <YAxis
                  allowDecimals={false}
                  tickLine={false}
                  axisLine={{ stroke: "var(--results-axis)" }}
                  tick={{ fill: "var(--results-axis)", fontSize: 11 }}
                />
                <Tooltip
                  cursor={{ fill: "rgba(148, 163, 184, 0.12)" }}
                  contentStyle={{
                    backgroundColor: "var(--results-tooltip-bg)",
                    border: "1px solid var(--results-tooltip-border)",
                    borderRadius: 12,
                    color: "var(--results-text)",
                    boxShadow: "0 14px 28px rgba(15, 23, 42, 0.12)",
                  }}
                  labelStyle={{ color: "var(--results-text)", fontWeight: 700 }}
                  itemStyle={{ color: "var(--results-text)" }}
                  wrapperStyle={{ color: "var(--results-text)", outline: "none" }}
                />
                <Bar dataKey="ok" stackId="status" fill={STATUS_COLORS.ok} name={t(language, "results.ok")} />
                <Bar
                  dataKey="wrong"
                  stackId="status"
                  fill={STATUS_COLORS.wrong}
                  name={t(language, "results.wrong")}
                />
                <Bar
                  dataKey="uncertain"
                  stackId="status"
                  fill={STATUS_COLORS.uncertain}
                  name={t(language, "results.uncertain")}
                />
                <Bar
                  dataKey="untested"
                  stackId="status"
                  fill={STATUS_COLORS.untested}
                  name={t(language, "results.untested")}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="results-chart-empty">{t(language, "results.chartsEmpty")}</div>
          )}
        </ChartCard>

        <ChartCard title={t(language, "results.coverageDistributionTitle")}>
          {hasSamples ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={coverageBins} margin={{ top: 8, right: 8, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="var(--results-grid)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="label"
                  interval={0}
                  tickLine={false}
                  axisLine={{ stroke: "var(--results-axis)" }}
                  tick={{ fill: "var(--results-axis)", fontSize: 11 }}
                  angle={-35}
                  textAnchor="end"
                  height={56}
                />
                <YAxis
                  allowDecimals={false}
                  tickLine={false}
                  axisLine={{ stroke: "var(--results-axis)" }}
                  tick={{ fill: "var(--results-axis)", fontSize: 11 }}
                />
                <Tooltip
                  cursor={{ fill: "rgba(148, 163, 184, 0.12)" }}
                  contentStyle={{
                    backgroundColor: "var(--results-tooltip-bg)",
                    border: "1px solid var(--results-tooltip-border)",
                    borderRadius: 12,
                    color: "var(--results-text)",
                    boxShadow: "0 14px 28px rgba(15, 23, 42, 0.12)",
                  }}
                  labelStyle={{ color: "var(--results-text)", fontWeight: 700 }}
                  itemStyle={{ color: "var(--results-text)" }}
                  wrapperStyle={{ color: "var(--results-text)", outline: "none" }}
                />
                <Bar dataKey="ok" stackId="status" fill={STATUS_COLORS.ok} name={t(language, "results.ok")} />
                <Bar
                  dataKey="wrong"
                  stackId="status"
                  fill={STATUS_COLORS.wrong}
                  name={t(language, "results.wrong")}
                />
                <Bar
                  dataKey="uncertain"
                  stackId="status"
                  fill={STATUS_COLORS.uncertain}
                  name={t(language, "results.uncertain")}
                />
                <Bar
                  dataKey="untested"
                  stackId="status"
                  fill={STATUS_COLORS.untested}
                  name={t(language, "results.untested")}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="results-chart-empty">{t(language, "results.chartsEmpty")}</div>
          )}
        </ChartCard>

        <ChartCard title={t(language, "results.qualityScatterTitle")}>
          {hasSamples && scatterData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="var(--results-grid)" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="identity"
                  domain={[0, 100]}
                  tickLine={false}
                  axisLine={{ stroke: "var(--results-axis)" }}
                  tick={{ fill: "var(--results-axis)", fontSize: 11 }}
                  tickFormatter={(value) => formatPercent(Number(value))}
                  name={t(language, "results.identity")}
                />
                <YAxis
                  type="number"
                  dataKey="coverage"
                  domain={[0, 100]}
                  tickLine={false}
                  axisLine={{ stroke: "var(--results-axis)" }}
                  tick={{ fill: "var(--results-axis)", fontSize: 11 }}
                  tickFormatter={(value) => formatPercent(Number(value))}
                  name={t(language, "results.coverage")}
                />
                <ZAxis type="number" dataKey="size" range={[40, 220]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "var(--results-tooltip-bg)",
                    border: "1px solid var(--results-tooltip-border)",
                    borderRadius: 12,
                    color: "var(--results-text)",
                    boxShadow: "0 14px 28px rgba(15, 23, 42, 0.12)",
                  }}
                  labelStyle={{ color: "var(--results-text)", fontWeight: 700 }}
                  itemStyle={{ color: "var(--results-text)" }}
                  wrapperStyle={{ color: "var(--results-text)", outline: "none" }}
                  formatter={(value: unknown, name: string | number | undefined) => {
                    const fieldName = String(name ?? "");

                    if (typeof value === "number" && Number.isFinite(value)) {
                      if (fieldName === "identity" || fieldName === "coverage") {
                        return [`${value.toFixed(1)}%`, t(language, `results.${fieldName}`)];
                      }

                      if (fieldName === "size") {
                        return [value.toFixed(1), "size"];
                      }

                      return [value.toFixed(1), fieldName];
                    }

                    return [String(value), fieldName];
                  }}
                />
                <Scatter data={scatterData} isAnimationActive={false}>
                  {scatterData.map((point) => (
                    <Cell key={point.id} fill={STATUS_COLORS[point.status]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="results-chart-empty">{t(language, "results.chartsEmpty")}</div>
          )}
        </ChartCard>

        <ChartCard title={t(language, "results.statusOverviewTitle")}>
          {hasSamples ? (
            <div className="results-status-card">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--results-tooltip-bg)",
                      border: "1px solid var(--results-tooltip-border)",
                      borderRadius: 12,
                      color: "var(--results-text)",
                      boxShadow: "0 14px 28px rgba(15, 23, 42, 0.12)",
                    }}
                    labelStyle={{ color: "var(--results-text)", fontWeight: 700 }}
                  itemStyle={{ color: "var(--results-text)" }}
                  wrapperStyle={{ color: "var(--results-text)", outline: "none" }}
                  />
                  <Pie
                    data={statusSlices}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={58}
                    outerRadius={84}
                    paddingAngle={3}
                    stroke="var(--results-card)"
                  >
                    {statusSlices.map((slice) => (
                      <Cell key={slice.name} fill={slice.fill} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>

              <div className="results-status-legend">
                {statusSlices.map((slice) => (
                  <span className="results-status-legend-item" key={slice.name}>
                    <i className="results-status-swatch" style={{ backgroundColor: slice.fill }} />
                    {t(language, `results.${slice.name}`)}
                  </span>
                ))}
              </div>

              <div className="results-chart-mini-grid">
                <MiniMetric
                  language={language}
                  status="ok"
                  value={statusSlices.find((slice) => slice.name === "ok")?.value ?? 0}
                />
                <MiniMetric
                  language={language}
                  status="wrong"
                  value={statusSlices.find((slice) => slice.name === "wrong")?.value ?? 0}
                />
                <MiniMetric
                  language={language}
                  status="uncertain"
                  value={statusSlices.find((slice) => slice.name === "uncertain")?.value ?? 0}
                />
                <MiniMetric
                  language={language}
                  status="untested"
                  value={statusSlices.find((slice) => slice.name === "untested")?.value ?? 0}
                />
                <div className="results-chart-mini results-chart-mini-total">
                  <span className="results-chart-mini-label">{t(language, "results.total")}</span>
                  <strong className="results-chart-mini-value">{total}</strong>
                </div>
              </div>
            </div>
          ) : (
            <div className="results-chart-empty">{t(language, "results.chartsEmpty")}</div>
          )}
        </ChartCard>
      </div>
    </section>
  );
}


