const BASES = ["A", "T", "G", "C"];
const DEFAULT_WIDTH = 1200;
const DEFAULT_HEIGHT = 220;
const DEFAULT_PADDING = 24;
const TRACE_PADDING = 24;

export function percentile(values, ratio) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * ratio)));
  return sorted[idx];
}

function emptyTracePoints() {
  return { A: [], T: [], G: [], C: [] };
}

function buildEmptyModel(width, height, padding) {
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

function findBucketPeak(trace, start, end) {
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

export function buildChromatogramRenderModel(data, options) {
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
  const visibleValues = [];

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
  const baseLabels = [];

  for (let i = startBaseIdx; i < endBaseIdx; i += 1) {
    const traceIndex = data.base_locations[i];
    const base = data.baseCalls[i];
    if (
      traceIndex === undefined ||
      traceIndex < visibleStartTrace ||
      traceIndex > visibleEndTrace ||
      !BASES.includes(base)
    ) {
      continue;
    }

    baseLabels.push({
      baseIndex: i,
      traceIndex,
      x: padding + (traceIndex - visibleStartTrace) * xScale,
      base,
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

export function findNearestBaseIndex(model, x) {
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
