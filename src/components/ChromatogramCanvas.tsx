import React, { useRef, useEffect, useState } from "react";
import { ChromatogramData } from "../types";
import "./ChromatogramCanvas.css";

interface ChromatogramCanvasProps {
  data: ChromatogramData | null;
  startPosition: number;
  endPosition: number;
  onHover: (position: number | null, quality: number | null) => void;
}

export const ChromatogramCanvas: React.FC<ChromatogramCanvasProps> = ({
  data,
  startPosition,
  endPosition,
  onHover,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const isDarkTheme = document.documentElement.dataset.theme === "dark";
    const traceColors = {
      A: isDarkTheme ? "#4ade80" : "#16a34a",
      T: isDarkTheme ? "#f87171" : "#dc2626",
      G: isDarkTheme ? "#fbbf24" : "#b45309",
      C: isDarkTheme ? "#60a5fa" : "#2563eb",
    };
    const background = isDarkTheme ? "#0f172a" : "#f8fbff";
    const gridColor = isDarkTheme ? "rgba(148,163,184,0.12)" : "rgba(148,163,184,0.22)";
    const labelColor = isDarkTheme ? "#dbe7f5" : "#334155";

    // Clear canvas
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!data.traces) return;

    const traces = data.traces;
    const width = canvas.width;
    const height = canvas.height;
    const padding = 24;

    if (!data.baseCalls || !data.base_locations) return;

    // Calculate visible range in terms of base calls
    const startBaseIdx = Math.max(0, startPosition - 1);
    const endBaseIdx = Math.min(data.baseCalls.length, endPosition);

    // Map base indices to trace indices using base_locations
    const startTraceIdx = data.base_locations[startBaseIdx] || 0;
    const endTraceIdx =
      data.base_locations[endBaseIdx - 1] || (traces.A?.length || 1) - 1;

    // Add some padding to the trace range
    const tracePadding = 20;
    const visibleStartTrace = Math.max(0, startTraceIdx - tracePadding);
    const visibleEndTrace = Math.min(
      traces.A.length,
      endTraceIdx + tracePadding
    );
    const traceRange = visibleEndTrace - visibleStartTrace;

    if (traceRange <= 0) return;

    // Find max value for scaling in the visible trace range
    let maxVal = 0;
    ["A", "T", "G", "C"].forEach((base) => {
      const trace = traces[base as keyof typeof traces];
      for (let i = visibleStartTrace; i < visibleEndTrace; i++) {
        if (trace[i] > maxVal) maxVal = trace[i];
      }
    });

    // Draw traces
    const xScale = (width - 2 * padding) / traceRange;
    const yScale = (height - 2 * padding) / (maxVal * 1.1);

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    for (let row = 1; row <= 4; row++) {
      const y = padding + ((height - 2 * padding) / 4) * row;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    ["A", "T", "G", "C"].forEach((base) => {
      const trace = traces[base as keyof typeof traces];
      ctx.strokeStyle = traceColors[base as keyof typeof traceColors];
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = visibleStartTrace; i < visibleEndTrace; i++) {
        const x = padding + (i - visibleStartTrace) * xScale;
        const y = height - padding - trace[i] * yScale;

        if (i === visibleStartTrace) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    });

    // Draw base calls and mixed peak indicators
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    for (let i = startBaseIdx; i < endBaseIdx; i++) {
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

      // Draw base letter
      ctx.fillStyle = traceColors[base as keyof typeof traceColors] || labelColor;
      ctx.fillText(base, x, height - 8);

      // Highlight mixed peaks
      if (data.mixed_peaks.includes(i)) {
        ctx.strokeStyle = isDarkTheme ? "#fde047" : "#ca8a04";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, height - 19, 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }, [data, startPosition, endPosition]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const padding = 20;
    const width = canvas.width - 2 * padding;

    if (!data.baseCalls || !data.base_locations) return;

    const startBaseIdx = Math.max(0, startPosition - 1);
    const endBaseIdx = Math.min(data.baseCalls.length, endPosition);

    const startTraceIdx = data.base_locations[startBaseIdx] || 0;
    const endTraceIdx =
      data.base_locations[endBaseIdx - 1] || (data.traces.A?.length || 1) - 1;

    const tracePadding = 20;
    const visibleStartTrace = Math.max(0, startTraceIdx - tracePadding);
    const visibleEndTrace = Math.min(
      data.traces.A.length,
      endTraceIdx + tracePadding
    );
    const traceRange = visibleEndTrace - visibleStartTrace;

    const relativeX = x - padding;
    const currentTraceIdx =
      (relativeX / width) * traceRange + visibleStartTrace;

    // Find the closest base call to this trace index
    let closestBaseIdx = startBaseIdx;
    let minDistance = Math.abs(data.base_locations[startBaseIdx] - currentTraceIdx);

    for (let i = startBaseIdx + 1; i < endBaseIdx; i++) {
      const distance = Math.abs(data.base_locations[i] - currentTraceIdx);
      if (distance < minDistance) {
        minDistance = distance;
        closestBaseIdx = i;
      }
    }

    if (closestBaseIdx >= startBaseIdx && closestBaseIdx < endBaseIdx && data.quality) {
      const quality = data.quality[closestBaseIdx];
      onHover(closestBaseIdx + 1, quality);

      const isMixed = data.mixed_peaks.includes(closestBaseIdx);

      // Show tooltip
      setTooltip({
        x: e.clientX + 10,
        y: e.clientY - 30,
        content: `${data.baseCalls[closestBaseIdx]} · Q${quality}${isMixed ? " · Mixed Peak" : ""}`,
      });
    }
  };

  const handleMouseLeave = () => {
    onHover(null, null);
    setTooltip(null);
  };

  return (
    <div className="chromatogram-container">
      <canvas
        ref={canvasRef}
        width={1200}
        height={220}
        className="chromatogram-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {tooltip && (
        <div
          className="chromatogram-tooltip"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
};
