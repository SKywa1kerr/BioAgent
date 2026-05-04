import type { ChromatogramData } from "./types";

export type ChromatogramBase = "A" | "T" | "G" | "C";

export interface ChromatogramRenderOptions {
  startPosition: number;
  endPosition: number;
  width?: number;
  height?: number;
  maxPointsPerPixel?: number;
  padding?: number;
}

export interface ChromatogramTracePoint {
  traceIndex: number;
  x: number;
  y: number;
  value: number;
}

export interface ChromatogramBaseLabel {
  baseIndex: number;
  traceIndex: number;
  x: number;
  base: ChromatogramBase;
  quality: number | null;
  mixed: boolean;
}

export interface ChromatogramRenderModel {
  width: number;
  height: number;
  padding: number;
  visibleStartTrace: number;
  visibleEndTrace: number;
  maxVal: number;
  step: number;
  tracePoints: Record<ChromatogramBase, ChromatogramTracePoint[]>;
  baseLabels: ChromatogramBaseLabel[];
}

export interface DrawChromatogramOptions {
  dark: boolean;
  mutations?: Array<{ position?: number }>;
  keyboardCursor?: number | null;
}

const BASES: ChromatogramBase[] = ["A", "T", "G", "C"];
const DEFAULT_WIDTH = 1200;
const DEFAULT_HEIGHT = 220;
const DEFAULT_PADDING = 24;
const TRACE_PADDING = 24;

const DARK_TRACE_COLORS: Record<ChromatogramBase, string> = {
  A: "#4ade80",
  T: "#f87171",
  G: "#fbbf24",
  C: "#60a5fa",
};

const LIGHT_TRACE_COLORS: Record<ChromatogramBase, string> = {
  A: "#16a34a",
  T: "#dc2626",
  G: "#b45309",
  C: "#2563eb",
};

export function percentile(values: number[], ratio: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * ratio)));
  return sorted[idx];
}

function emptyTracePoints(): Record<ChromatogramBase, ChromatogramTracePoint[]> {
  return { A: [], T: [], G: [], C: [] };
}

function buildEmptyModel(width: number, height: number, padding: number): ChromatogramRenderModel {
  return {
    width,
    height,
    padding,
    visibleStartTrace: 0,
    visibleEndTrace: 0,
    maxVal: 1,
    step: 1,
    tracePoints: emptyTracePoints(),
    baseLabels: [],
  };
}

function findBucketPeak(trace: number[], start: number, end: number) {
  let traceIndex = start;
  let value = trace[start] ?? 0;

  for (let i = start + 1; i < end; i += 1) {
    const nextValue = trace[i] ?? 0;
    if (Math.abs(nextValue) > Math.abs(value)) {
      traceIndex = i;
      value = nextValue;
    }
  }

  return { traceIndex, value };
}

export function buildChromatogramRenderModel(
  data: ChromatogramData,
  options: ChromatogramRenderOptions,
): ChromatogramRenderModel {
  const width = options.width ?? DEFAULT_WIDTH;
  const height = options.height ?? DEFAULT_HEIGHT;
  const padding = options.padding ?? DEFAULT_PADDING;
  const traceLength = data.traces.A?.length ?? 0;

  if (
    !Number.isFinite(options.startPosition) ||
    !Number.isFinite(options.endPosition) ||
    !data.baseCalls ||
    !data.base_locations?.length ||
    traceLength <= 0
  ) {
    return buildEmptyModel(width, height, padding);
  }

  const startBaseIdx = Math.max(0, options.startPosition - 1);
  const endBaseIdx = Math.min(data.baseCalls.length, Math.max(startBaseIdx, options.endPosition));

  if (startBaseIdx >= endBaseIdx) {
    return buildEmptyModel(width, height, padding);
  }

  const startTraceIdx = data.base_locations[startBaseIdx] ?? 0;
  const endTraceIdx = data.base_locations[endBaseIdx - 1] ?? traceLength - 1;
  const visibleStartTrace = Math.max(0, Math.min(traceLength, startTraceIdx - TRACE_PADDING));
  const visibleEndTrace = Math.max(
    visibleStartTrace,
    Math.min(traceLength, endTraceIdx + TRACE_PADDING),
  );
  const traceRange = visibleEndTrace - visibleStartTrace;

  if (traceRange <= 0) {
    return buildEmptyModel(width, height, padding);
  }

  const budget = width * (options.maxPointsPerPixel ?? 1.5);
  const step = Math.max(1, Math.ceil(traceRange / Math.max(1, budget)));
  const visibleValues: number[] = [];

  for (const base of BASES) {
    const trace = data.traces[base] ?? [];
    for (let i = visibleStartTrace; i < visibleEndTrace; i += step) {
      const bucketEnd = step === 1 ? i + 1 : Math.min(i + step, visibleEndTrace);
      const { value } = findBucketPeak(trace, i, bucketEnd);
      if (value > 0) visibleValues.push(value);
    }
  }

  const robustTop = percentile(visibleValues, 0.985);
  const absoluteTop = visibleValues.length > 0 ? Math.max(...visibleValues) : 0;
  const maxVal = Math.max(robustTop, absoluteTop, 1);
  const xScale = (width - 2 * padding) / traceRange;
  const yScale = (height - 2 * padding) / (maxVal * 1.08);
  const tracePoints = emptyTracePoints();

  for (const base of BASES) {
    const trace = data.traces[base] ?? [];
    const points = tracePoints[base];
    for (let i = visibleStartTrace; i < visibleEndTrace; i += step) {
      const bucketEnd = step === 1 ? i + 1 : Math.min(i + step, visibleEndTrace);
      const { traceIndex, value } = findBucketPeak(trace, i, bucketEnd);
      const clamped = Math.min(value, maxVal * 1.08);
      points.push({
        traceIndex,
        x: padding + (traceIndex - visibleStartTrace) * xScale,
        y: height - padding - clamped * yScale,
        value,
      });
    }
  }

  const mixedPeaks = new Set(data.mixed_peaks ?? []);
  const baseLabels: ChromatogramBaseLabel[] = [];

  for (let i = startBaseIdx; i < endBaseIdx; i += 1) {
    const traceIndex = data.base_locations[i];
    const base = data.baseCalls[i] as ChromatogramBase | undefined;
    if (
      traceIndex === undefined ||
      traceIndex < visibleStartTrace ||
      traceIndex > visibleEndTrace ||
      !BASES.includes(base as ChromatogramBase)
    ) {
      continue;
    }

    baseLabels.push({
      baseIndex: i,
      traceIndex,
      x: padding + (traceIndex - visibleStartTrace) * xScale,
      base: base as ChromatogramBase,
      quality: typeof data.quality?.[i] === "number" ? data.quality[i] : null,
      mixed: mixedPeaks.has(i),
    });
  }

  return {
    width,
    height,
    padding,
    visibleStartTrace,
    visibleEndTrace,
    maxVal,
    step,
    tracePoints,
    baseLabels,
  };
}

export function findNearestBaseIndex(model: ChromatogramRenderModel, x: number) {
  if (model.baseLabels.length === 0) return -1;

  let nearestBaseIndex = -1;
  let minDistance = Number.POSITIVE_INFINITY;

  for (const label of model.baseLabels) {
    const distance = Math.abs(label.x - x);
    if (distance < minDistance) {
      minDistance = distance;
      nearestBaseIndex = label.baseIndex;
    }
  }

  return nearestBaseIndex;
}

export function drawChromatogram(
  ctx: CanvasRenderingContext2D,
  model: ChromatogramRenderModel,
  options: DrawChromatogramOptions,
) {
  const { width, height, padding } = model;
  const traceColors = options.dark ? DARK_TRACE_COLORS : LIGHT_TRACE_COLORS;
  const background = options.dark ? "#0f172a" : "#f8fbff";
  const gridColor = options.dark ? "rgba(148, 163, 184, 0.12)" : "rgba(148, 163, 184, 0.22)";
  const labelColor = options.dark ? "#dbe7f5" : "#334155";
  const mixedPeakColor = options.dark ? "#fde047" : "#ca8a04";
  const mutationColor = options.dark ? "#f87171" : "#dc2626";

  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let row = 1; row <= 4; row += 1) {
    const y = padding + ((height - 2 * padding) / 4) * row;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }

  for (const base of BASES) {
    const points = model.tracePoints[base];
    if (points.length === 0) continue;

    ctx.strokeStyle = traceColors[base];
    ctx.lineWidth = 1.8;
    ctx.beginPath();

    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });

    ctx.stroke();
  }

  ctx.font = '11px "Consolas", monospace';
  ctx.textAlign = "center";

  for (const label of model.baseLabels) {
    const x = label.x;

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, height - padding - 6);
    ctx.lineTo(x, height - padding + 8);
    ctx.stroke();

    ctx.fillStyle = traceColors[label.base] || labelColor;
    ctx.fillText(label.base, x, height - 8);

    if (label.mixed) {
      ctx.strokeStyle = mixedPeakColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, height - 19, 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  const labelsByPosition = new Map(model.baseLabels.map((label) => [label.baseIndex + 1, label]));
  for (const mutation of options.mutations ?? []) {
    if (typeof mutation.position !== "number") continue;
    const label = labelsByPosition.get(mutation.position);
    if (!label) continue;

    ctx.fillStyle = mutationColor;
    ctx.beginPath();
    ctx.moveTo(label.x - 5, padding + 4);
    ctx.lineTo(label.x + 5, padding + 4);
    ctx.lineTo(label.x, padding + 12);
    ctx.closePath();
    ctx.fill();
  }

  if (typeof options.keyboardCursor === "number") {
    const cursorLabel = model.baseLabels.find((label) => label.baseIndex === options.keyboardCursor);
    if (cursorLabel) {
      const cursorColor = options.dark ? "#facc15" : "#0284c7";
      ctx.strokeStyle = cursorColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cursorLabel.x, padding);
      ctx.lineTo(cursorLabel.x, height - padding);
      ctx.stroke();

      ctx.fillStyle = cursorColor;
      ctx.beginPath();
      ctx.arc(cursorLabel.x, padding, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
