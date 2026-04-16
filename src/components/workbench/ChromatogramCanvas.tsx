import { useEffect, useRef, useState } from "react";
import type { ChromatogramData } from "./types";
import "./ChromatogramCanvas.css";

interface Props {
  data: ChromatogramData | null;
  startPosition: number;
  endPosition: number;
  mutations?: Array<{ position?: number }>;
}

function percentile(values: number[], ratio: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * ratio)));
  return sorted[idx];
}

export function ChromatogramCanvas({ data, startPosition, endPosition, mutations }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);

  const totalBases = endPosition - startPosition;
  const visibleBases = Math.max(10, Math.floor(totalBases / zoomLevel));
  const effectiveStart = Math.max(startPosition, startPosition + panOffset);
  const effectiveEnd = Math.min(endPosition, effectiveStart + visibleBases);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const isDarkTheme = document.documentElement.dataset.theme === "dark";
    const traces = data.traces;
    const width = canvas.width;
    const height = canvas.height;
    const padding = 24;

    const traceColors = {
      A: isDarkTheme ? "#4ade80" : "#16a34a",
      T: isDarkTheme ? "#f87171" : "#dc2626",
      G: isDarkTheme ? "#fbbf24" : "#b45309",
      C: isDarkTheme ? "#60a5fa" : "#2563eb",
    } as const;

    const background = isDarkTheme ? "#0f172a" : "#f8fbff";
    const gridColor = isDarkTheme ? "rgba(148, 163, 184, 0.12)" : "rgba(148, 163, 184, 0.22)";
    const labelColor = isDarkTheme ? "#dbe7f5" : "#334155";

    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    if (!data.baseCalls || !data.base_locations || data.base_locations.length === 0) return;

    const startBaseIdx = Math.max(0, effectiveStart - 1);
    const endBaseIdx = Math.min(data.baseCalls.length, effectiveEnd);

    const startTraceIdx = data.base_locations[startBaseIdx] || 0;
    const endTraceIdx = data.base_locations[endBaseIdx - 1] || (traces.A?.length || 1) - 1;

    const tracePadding = 24;
    const visibleStartTrace = Math.max(0, startTraceIdx - tracePadding);
    const visibleEndTrace = Math.min(traces.A.length, endTraceIdx + tracePadding);
    const traceRange = visibleEndTrace - visibleStartTrace;
    if (traceRange <= 0) return;

    const visibleValues: number[] = [];
    (["A", "T", "G", "C"] as const).forEach((base) => {
      const trace = traces[base] || [];
      for (let i = visibleStartTrace; i < visibleEndTrace; i += 2) {
        const value = trace[i] || 0;
        if (value > 0) visibleValues.push(value);
      }
    });

    const robustTop = percentile(visibleValues, 0.985);
    const maxVal = robustTop > 0 ? robustTop : 1;

    const xScale = (width - 2 * padding) / traceRange;
    const yScale = (height - 2 * padding) / (maxVal * 1.08);

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    for (let row = 1; row <= 4; row += 1) {
      const y = padding + ((height - 2 * padding) / 4) * row;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    (["A", "T", "G", "C"] as const).forEach((base) => {
      const trace = traces[base] || [];
      ctx.strokeStyle = traceColors[base];
      ctx.lineWidth = 1.8;
      ctx.beginPath();

      for (let i = visibleStartTrace; i < visibleEndTrace; i += 1) {
        const x = padding + (i - visibleStartTrace) * xScale;
        const clamped = Math.min(trace[i] || 0, maxVal * 1.08);
        const y = height - padding - clamped * yScale;

        if (i === visibleStartTrace) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.stroke();
    });

    ctx.font = '11px "Consolas", monospace';
    ctx.textAlign = "center";
    for (let i = startBaseIdx; i < endBaseIdx; i += 1) {
      const traceIdx = data.base_locations[i];
      if (traceIdx < visibleStartTrace || traceIdx > visibleEndTrace) continue;

      const x = padding + (traceIdx - visibleStartTrace) * xScale;
      const base = data.baseCalls[i];

      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, height - padding - 6);
      ctx.lineTo(x, height - padding + 8);
      ctx.stroke();

      ctx.fillStyle = traceColors[(base as "A" | "T" | "G" | "C")] || labelColor;
      ctx.fillText(base, x, height - 8);

      if (data.mixed_peaks.includes(i)) {
        ctx.strokeStyle = isDarkTheme ? "#fde047" : "#ca8a04";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, height - 19, 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Draw mutation markers
    if (mutations && mutations.length > 0) {
      for (const mut of mutations) {
        if (typeof mut.position !== "number") continue;
        const mutBaseIdx = mut.position - 1; // 0-indexed
        if (mutBaseIdx < startBaseIdx || mutBaseIdx >= endBaseIdx) continue;
        const traceIdx = data.base_locations[mutBaseIdx];
        if (traceIdx < visibleStartTrace || traceIdx > visibleEndTrace) continue;
        const x = padding + (traceIdx - visibleStartTrace) * xScale;

        // Draw red triangle marker above the base
        ctx.fillStyle = isDarkTheme ? "#f87171" : "#dc2626";
        ctx.beginPath();
        ctx.moveTo(x - 5, padding + 4);
        ctx.lineTo(x + 5, padding + 4);
        ctx.lineTo(x, padding + 12);
        ctx.closePath();
        ctx.fill();
      }
    }
  }, [data, effectiveStart, effectiveEnd, mutations]);

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!data || !canvasRef.current || !data.base_locations?.length) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const padding = 24;
    const width = canvas.width - 2 * padding;

    const startBaseIdx = Math.max(0, effectiveStart - 1);
    const endBaseIdx = Math.min(data.baseCalls.length, effectiveEnd);
    const startTraceIdx = data.base_locations[startBaseIdx] || 0;
    const endTraceIdx = data.base_locations[endBaseIdx - 1] || (data.traces.A?.length || 1) - 1;
    const tracePadding = 24;
    const visibleStartTrace = Math.max(0, startTraceIdx - tracePadding);
    const visibleEndTrace = Math.min(data.traces.A.length, endTraceIdx + tracePadding);
    const traceRange = Math.max(1, visibleEndTrace - visibleStartTrace);

    const currentTraceIdx = ((x - padding) / width) * traceRange + visibleStartTrace;

    let closestBaseIdx = startBaseIdx;
    let minDistance = Number.POSITIVE_INFINITY;

    for (let i = startBaseIdx; i < endBaseIdx; i += 1) {
      const distance = Math.abs((data.base_locations[i] || 0) - currentTraceIdx);
      if (distance < minDistance) {
        minDistance = distance;
        closestBaseIdx = i;
      }
    }

    const quality = data.quality?.[closestBaseIdx];
    const base = data.baseCalls[closestBaseIdx] || "-";
    const isMixed = data.mixed_peaks.includes(closestBaseIdx);
    setTooltip({
      x: event.clientX + 12,
      y: event.clientY - 18,
      content: `Pos ${closestBaseIdx + 1} | Base ${base} | Q ${typeof quality === "number" ? quality : "-"}${isMixed ? " | Mixed" : ""}`,
    });
  }

  return (
    <div className="chromatogram-container">
      <div className="chromatogram-toolbar">
        <button className="chromatogram-zoom-btn" onClick={() => { setZoomLevel((z) => Math.min(10, z * 1.5)); setPanOffset(0); }} title="Zoom in">+</button>
        <button className="chromatogram-zoom-btn" onClick={() => { setZoomLevel((z) => Math.max(1, z / 1.5)); setPanOffset(0); }} title="Zoom out">&minus;</button>
        <button className="chromatogram-zoom-btn" onClick={() => { setZoomLevel(1); setPanOffset(0); }} title="Reset">Reset</button>
        <span className="chromatogram-zoom-label">{zoomLevel > 1 ? `${zoomLevel.toFixed(1)}x` : ""}</span>
        {zoomLevel > 1 ? (
          <>
            <button className="chromatogram-zoom-btn" onClick={() => setPanOffset((o) => Math.max(0, o - Math.floor(visibleBases / 2)))} disabled={panOffset <= 0}>&larr;</button>
            <button className="chromatogram-zoom-btn" onClick={() => setPanOffset((o) => Math.min(totalBases - visibleBases, o + Math.floor(visibleBases / 2)))} disabled={effectiveEnd >= endPosition}>&rarr;</button>
          </>
        ) : null}
      </div>
      <canvas
        ref={canvasRef}
        width={1200}
        height={220}
        className="chromatogram-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip ? <div className="chromatogram-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.content}</div> : null}
    </div>
  );
}
