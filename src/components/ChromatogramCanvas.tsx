import React, { useRef, useEffect, useState } from "react";
import { ChromatogramData } from "../types";
import "./ChromatogramCanvas.css";

interface ChromatogramCanvasProps {
  data: ChromatogramData | null;
  startPosition: number;
  endPosition: number;
  onHover: (position: number | null, quality: number | null) => void;
}

const TRACE_COLORS = {
  A: "#00aa00",
  T: "#aa0000",
  G: "#000000",
  C: "#0000aa",
};

export const ChromatogramCanvas: React.FC<ChromatogramCanvasProps> = ({
  data,
  startPosition,
  endPosition,
  onHover,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!data.traces) return;

    const traces = data.traces;
    const width = canvas.width;
    const height = canvas.height;
    const padding = 20;

    // Calculate visible range
    const startIdx = Math.max(0, startPosition - 1);
    const endIdx = Math.min(traces.A.length, endPosition);
    const range = endIdx - startIdx;

    if (range <= 0) return;

    // Find max value for scaling
    let maxVal = 0;
    ["A", "T", "G", "C"].forEach((base) => {
      const trace = traces[base as keyof typeof traces];
      for (let i = startIdx; i < endIdx; i++) {
        if (trace[i] > maxVal) maxVal = trace[i];
      }
    });

    // Draw traces
    const xScale = (width - 2 * padding) / range;
    const yScale = (height - 2 * padding) / (maxVal * 1.1);

    ["A", "T", "G", "C"].forEach((base) => {
      const trace = traces[base as keyof typeof traces];
      ctx.strokeStyle = TRACE_COLORS[base as keyof typeof TRACE_COLORS];
      ctx.lineWidth = 1.5;
      ctx.beginPath();

      for (let i = startIdx; i < endIdx; i++) {
        const x = padding + (i - startIdx) * xScale;
        const y = height - padding - trace[i] * yScale;

        if (i === startIdx) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    });
  }, [data, startPosition, endPosition]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const padding = 20;
    const width = canvas.width - 2 * padding;

    const startIdx = Math.max(0, startPosition - 1);
    const endIdx = Math.min(data.traces.A.length, endPosition);
    const range = endIdx - startIdx;

    const relativeX = x - padding;
    const positionIdx = Math.floor((relativeX / width) * range) + startIdx;

    if (positionIdx >= startIdx && positionIdx < endIdx && data.quality) {
      const quality = data.quality[positionIdx];
      onHover(positionIdx + 1, quality);

      // Show tooltip after delay
      setTooltip({
        x: e.clientX + 10,
        y: e.clientY - 30,
        content: `Q${quality}`,
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
        width={800}
        height={150}
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
