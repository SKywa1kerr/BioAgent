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
import { useMemo } from "react";
import type { ReactNode } from "react";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, formatPercent } from "./utils";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import "./ResultsCharts.css";

type StatusBucket = "ok" | "wrong" | "uncertain" | "untested";

const FALLBACK_COLORS: Record<StatusBucket, string> = {
  ok: "#1d9c7d",
  wrong: "#d23f5c",
  uncertain: "#d89a2c",
  untested: "#6a7a93",
};

function readToken(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function readPalette(): Record<StatusBucket, string> {
  return {
    ok: readToken("--status-ok", FALLBACK_COLORS.ok),
    wrong: readToken("--status-wrong", FALLBACK_COLORS.wrong),
    uncertain: readToken("--status-uncertain", FALLBACK_COLORS.uncertain),
    untested: readToken("--status-untested", FALLBACK_COLORS.untested),
  };
}

const BIN_COUNT = 10;

interface DistributionBin {
  label: string;
  ok: number;
  wrong: number;
  uncertain: number;
  untested: number;
}

interface ScatterPoint {
  id: string;
  name: string;
  status: StatusBucket;
  identity: number;
  coverage: number;
  size: number;
}

function normalizePercent(value?: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value <= 1.5 ? value * 100 : value;
}

function getCoverageValue(sample: WorkbenchSample) {
  return sample.cds_coverage ?? sample.coverage;
}

function getIdentityValue(sample: WorkbenchSample) {
  return sample.identity;
}

function getBubbleSize(sample: WorkbenchSample) {
  const score =
    sample.mutations?.length ??
    (sample.sub_count ?? 0) + (sample.ins_count ?? 0) + (sample.del_count ?? 0);
  return Math.max(4, Math.min(18, 4 + score / 2));
}

function buildBins(samples: WorkbenchSample[], accessor: (sample: WorkbenchSample) => number | undefined | null) {
  const bins: DistributionBin[] = Array.from({ length: BIN_COUNT }, (_, idx) => {
    const lower = idx * (100 / BIN_COUNT);
    const upper = idx === BIN_COUNT - 1 ? 100 : (idx + 1) * (100 / BIN_COUNT);
    return { label: `${Math.round(lower)}-${Math.round(upper)}%`, ok: 0, wrong: 0, uncertain: 0, untested: 0 };
  });

  for (const sample of samples) {
    const value = normalizePercent(accessor(sample) ?? undefined);
    if (value == null) continue;
    const clamped = Math.min(100, Math.max(0, value));
    const index = Math.min(BIN_COUNT - 1, Math.floor(clamped / (100 / BIN_COUNT)));
    const status = bucketSampleStatus(sample);
    bins[index][status] += 1;
  }

  return bins;
}

function buildScatterData(samples: WorkbenchSample[]): ScatterPoint[] {
  return samples
    .map((sample) => {
      const identity = normalizePercent(getIdentityValue(sample));
      const coverage = normalizePercent(getCoverageValue(sample));
      if (identity == null || coverage == null) return null;
      return {
        id: sample.id,
        name: sample.clone || sample.id,
        status: bucketSampleStatus(sample),
        identity,
        coverage,
        size: getBubbleSize(sample),
      };
    })
    .filter((point): point is ScatterPoint => point !== null);
}

function buildStatusSlices(samples: WorkbenchSample[], palette: Record<StatusBucket, string>) {
  const counts: Record<StatusBucket, number> = { ok: 0, wrong: 0, uncertain: 0, untested: 0 };
  for (const sample of samples) counts[bucketSampleStatus(sample)] += 1;
  return (Object.keys(counts) as StatusBucket[]).map((status) => ({
    name: status,
    value: counts[status],
    fill: palette[status],
  }));
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <article className="results-chart-card">
      <h4 className="results-chart-card-title">{title}</h4>
      <div className="results-chart-card-body">{children}</div>
    </article>
  );
}

export function ResultsCharts({ samples, language }: { samples: WorkbenchSample[]; language: AppLanguage }) {
  const hasSamples = samples.length > 0;
  const themeKey = typeof document !== "undefined" ? document.documentElement.dataset.theme || "light" : "light";
  const palette = useMemo(
    () => readPalette(),
    // Re-read palette when theme changes (requires a reload or data-theme flip on <html>).
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [themeKey],
  );
  const identityBins = useMemo(() => buildBins(samples, getIdentityValue), [samples]);
  const coverageBins = useMemo(() => buildBins(samples, getCoverageValue), [samples]);
  const scatterData = useMemo(() => buildScatterData(samples), [samples]);
  const statusSlices = useMemo(() => buildStatusSlices(samples, palette), [samples, palette]);

  return (
    <section className="results-charts-panel" aria-label={t(language, "charts.title")}>
      <div className="results-section-header results-section-header-compact results-charts-header">
        <div><h3>{t(language, "charts.title")}</h3></div>
      </div>
      <div className="results-charts-grid">
        <ChartCard title={t(language, "charts.identityDist")}>
          {hasSamples ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={identityBins} margin={{ top: 8, right: 8, bottom: 0, left: -8 }} >
                <CartesianGrid stroke="var(--results-grid)" strokeDasharray="3 3" />
                <XAxis dataKey="label" interval={0} tickLine={false} axisLine={{ stroke: "var(--results-axis)" }} tick={{ fill: "var(--results-axis)", fontSize: 11 }} angle={-35} textAnchor="end" height={56} />
                <YAxis allowDecimals={false} tickLine={false} axisLine={{ stroke: "var(--results-axis)" }} tick={{ fill: "var(--results-axis)", fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="ok" stackId="status" fill={palette.ok} name={t(language, "wb.status.ok")} isAnimationActive={false} />
                <Bar dataKey="wrong" stackId="status" fill={palette.wrong} name={t(language, "wb.status.wrong")} isAnimationActive={false} />
                <Bar dataKey="uncertain" stackId="status" fill={palette.uncertain} name={t(language, "wb.status.uncertain")} isAnimationActive={false} />
                <Bar dataKey="untested" stackId="status" fill={palette.untested} name={t(language, "wb.status.untested")} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="results-chart-empty">{t(language, "charts.noData")}</div>}
        </ChartCard>

        <ChartCard title={t(language, "charts.coverageDist")}>
          {hasSamples ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={coverageBins} margin={{ top: 8, right: 8, bottom: 0, left: -8 }} >
                <CartesianGrid stroke="var(--results-grid)" strokeDasharray="3 3" />
                <XAxis dataKey="label" interval={0} tickLine={false} axisLine={{ stroke: "var(--results-axis)" }} tick={{ fill: "var(--results-axis)", fontSize: 11 }} angle={-35} textAnchor="end" height={56} />
                <YAxis allowDecimals={false} tickLine={false} axisLine={{ stroke: "var(--results-axis)" }} tick={{ fill: "var(--results-axis)", fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="ok" stackId="status" fill={palette.ok} name={t(language, "wb.status.ok")} isAnimationActive={false} />
                <Bar dataKey="wrong" stackId="status" fill={palette.wrong} name={t(language, "wb.status.wrong")} isAnimationActive={false} />
                <Bar dataKey="uncertain" stackId="status" fill={palette.uncertain} name={t(language, "wb.status.uncertain")} isAnimationActive={false} />
                <Bar dataKey="untested" stackId="status" fill={palette.untested} name={t(language, "wb.status.untested")} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="results-chart-empty">{t(language, "charts.noData")}</div>}
        </ChartCard>

        <ChartCard title={t(language, "charts.identityVsCoverage")}>
          {hasSamples && scatterData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="var(--results-grid)" strokeDasharray="3 3" />
                <XAxis type="number" dataKey="identity" domain={[0, 100]} tickFormatter={(v) => formatPercent(Number(v))} name={t(language, "charts.identity")} />
                <YAxis type="number" dataKey="coverage" domain={[0, 100]} tickFormatter={(v) => formatPercent(Number(v))} name={t(language, "charts.coverage")} />
                <ZAxis type="number" dataKey="size" range={[40, 220]} />
                <Tooltip />
                <Scatter data={scatterData} >
                  {scatterData.map((point) => (<Cell key={point.id} fill={palette[point.status]} />))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : <div className="results-chart-empty">{t(language, "charts.noData")}</div>}
        </ChartCard>

        <ChartCard title={t(language, "charts.statusOverview")}>
          {hasSamples ? (
            <div className="results-status-card">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Tooltip />
                  <Pie data={statusSlices} dataKey="value" nameKey="name" innerRadius={58} outerRadius={84} paddingAngle={3} >
                    {statusSlices.map((slice) => (<Cell key={slice.name} fill={slice.fill} />))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="results-status-legend">
                {statusSlices.map((slice) => (
                  <span className="results-status-legend-item" key={slice.name}>
                    <i className="results-status-swatch" style={{ backgroundColor: slice.fill }} />
                    {t(language, `wb.status.${slice.name}`)}
                  </span>
                ))}
              </div>
            </div>
          ) : <div className="results-chart-empty">{t(language, "charts.noData")}</div>}
        </ChartCard>
      </div>
    </section>
  );
}


