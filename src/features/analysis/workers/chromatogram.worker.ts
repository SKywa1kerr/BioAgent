/**
 * Chromatogram Rendering Worker
 * Handles heavy trace coordinate calculation and canvas drawing off-thread.
 */

interface RenderPayload {
  canvas: OffscreenCanvas;
  traces: {
    A: Float32Array;
    T: Float32Array;
    G: Float32Array;
    C: Float32Array;
  };
  quality: Uint8Array;
  baseLocations: Int32Array;
  mixedPeaks: Int32Array;
  gappedToQueryIdx: Int32Array; // -1 encodes null
  totalBases: number;
  alignedQueryG: string;
  baseWidth: number;
  traceHeight: number;
  dpr: number;
  downsampleStep: number;
}

const colors = { A: "#00aa00", T: "#aa0000", G: "#000000", C: "#0000aa" };
const bases = ["A", "T", "G", "C"] as const;

self.onmessage = (e: MessageEvent<RenderPayload>) => {
  const {
    canvas,
    traces,
    quality,
    baseLocations,
    mixedPeaks,
    gappedToQueryIdx,
    totalBases,
    alignedQueryG,
    baseWidth,
    traceHeight,
    dpr,
    // downsampleStep is received but data is already downsampled in main thread
  } = e.data;

  if (!canvas) return;

  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return;

  const width = totalBases * baseWidth;
  const targetWidth = Math.floor(width * dpr);
  const targetHeight = Math.floor(traceHeight * dpr);

  // Always set size and reset transform to handle DPR changes correctly (Bug 2)
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Background
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, traceHeight);

  const totalTracePoints = traces.A?.length || 0;
  if (totalTracePoints === 0) return;

  // Build mixed peaks Set for O(1) lookup (Bug 4)
  const mixedPeakSet = new Set<number>();
  for (let i = 0; i < mixedPeaks.length; i++) {
    mixedPeakSet.add(mixedPeaks[i]);
  }

  // Decode gappedToQueryIdx: -1 means null (Bug 5)
  const gqi = gappedToQueryIdx;

  // Find 99.5th percentile for scaling — loop-based to avoid stack overflow (Bug 3)
  let maxVal = 0;
  {
    // Sample at most ~10000 points for percentile estimation to stay fast
    const sampleStep = Math.max(1, Math.floor(totalTracePoints / 2500));
    const sampled: number[] = [];
    for (const base of bases) {
      const trace = traces[base];
      for (let i = 0; i < trace.length; i += sampleStep) {
        sampled.push(trace[i]);
      }
    }
    sampled.sort((a, b) => a - b);
    maxVal = sampled[Math.floor(sampled.length * 0.995)] || 100;
  }
  if (maxVal === 0) maxVal = 100;

  const yScale = (traceHeight - 40) / (maxVal * 1.05);
  const baselineY = traceHeight - 25;

  // 0. Draw Baseline
  ctx.strokeStyle = "#f1f5f9";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, baselineY);
  ctx.lineTo(width, baselineY);
  ctx.stroke();

  // 1. Draw Quality Bars
  if (quality && quality.length > 0) {
    for (let i = 0; i < totalBases; i++) {
      const qIdx = gqi[i];
      if (qIdx >= 0 && qIdx < quality.length) {
        const q = quality[qIdx];
        if (q >= 40) ctx.fillStyle = "#e2e8f0";
        else if (q >= 20) ctx.fillStyle = "#fef3c7";
        else ctx.fillStyle = "#fee2e2";

        const barHeight = Math.min(1, q / 60) * (traceHeight - 45);
        const x = i * baseWidth + 1;
        ctx.fillRect(x, traceHeight - 25 - barHeight, baseWidth - 2, barHeight);
      }
    }
  }

  // 2. Draw Traces
  if (baseLocations && baseLocations.length > 0) {
    // Build reverse map: query index -> gapped position
    const queryToGapped: number[] = [];
    for (let i = 0; i < gqi.length; i++) {
      const qIdx = gqi[i];
      if (qIdx >= 0) queryToGapped[qIdx] = i;
    }

    for (const base of bases) {
      const trace = traces[base];
      ctx.strokeStyle = colors[base];
      ctx.lineWidth = 1.2;
      ctx.beginPath();

      let b1 = 0;
      let firstPoint = true;

      for (let i = 0; i < totalTracePoints; i++) {
        // Find the base call index for this trace point
        while (b1 < baseLocations.length - 1 && baseLocations[b1 + 1] < i) {
          b1++;
        }

        const t1 = baseLocations[b1];
        // Bug 6: handle last base — extrapolate from spacing instead of dropping
        let t2: number;
        let g2: number | undefined;
        if (b1 + 1 < baseLocations.length) {
          t2 = baseLocations[b1 + 1];
          g2 = queryToGapped[b1 + 1];
        } else {
          // Last base: extrapolate t2 from previous spacing
          const prevSpacing = b1 > 0 ? (t1 - baseLocations[b1 - 1]) : 15;
          t2 = t1 + prevSpacing;
          // Extrapolate g2 from last known gapped position
          const g1 = queryToGapped[b1];
          g2 = g1 !== undefined ? g1 + 1 : undefined;
        }

        const g1 = queryToGapped[b1];

        if (g1 === undefined || g2 === undefined) {
          firstPoint = true;
          continue;
        }

        // Bug 7: guard division by zero
        const dt = t2 - t1;
        const ratio = dt === 0 ? 0.5 : (i - t1) / dt;
        const gappedX = g1 + (g2 - g1) * ratio;
        const x = (gappedX + 0.5) * baseWidth;
        const y = traceHeight - 25 - trace[i] * yScale;

        if (firstPoint) {
          ctx.moveTo(x, y);
          firstPoint = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
  }

  // 3. Draw Base Call Letters + Mixed Peak markers
  ctx.font = "bold 10px Monaco, Consolas, monospace";
  ctx.textAlign = "center";
  for (let i = 0; i < totalBases; i++) {
    const b = alignedQueryG[i];
    if (b === "-" || !b) continue;

    const x = (i + 0.5) * baseWidth;
    ctx.fillStyle = colors[b as keyof typeof colors] || "#888";
    ctx.fillText(b, x, traceHeight - 8);

    const qIdx = gqi[i];
    if (qIdx >= 0 && mixedPeakSet.has(qIdx)) {
      ctx.strokeStyle = "#eab308";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, traceHeight - 18, 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // Create bitmap from canvas and transfer back to main thread
  createImageBitmap(canvas).then(bitmap => {
    (self as unknown as Worker).postMessage({ type: "rendered", bitmap }, [bitmap]);
  });
};
