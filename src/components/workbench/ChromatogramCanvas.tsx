import { useEffect, useRef, useState } from "react";
import type { ChromatogramData } from "./types";
import {
  buildChromatogramRenderModel,
  drawChromatogram,
  findNearestBaseIndex,
} from "./chromatogramRender";
import { useTheme } from "../../hooks/useTheme";
import "./ChromatogramCanvas.css";

interface Props {
  data: ChromatogramData | null;
  startPosition: number;
  endPosition: number;
  mutations?: Array<{ position?: number }>;
}

export function ChromatogramCanvas({ data, startPosition, endPosition, mutations }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderModelRef = useRef<ReturnType<typeof buildChromatogramRenderModel> | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);
  const theme = useTheme();

  const totalBases = endPosition - startPosition;
  const visibleBases = Math.max(10, Math.floor(totalBases / zoomLevel));
  const effectiveStart = Math.max(startPosition, startPosition + panOffset);
  const effectiveEnd = Math.min(endPosition, effectiveStart + visibleBases);

  useEffect(() => {
    if (!data || !canvasRef.current) {
      renderModelRef.current = null;
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      renderModelRef.current = null;
      return;
    }

    const isDarkTheme = theme === "dark";
    const model = buildChromatogramRenderModel(data, {
      startPosition: effectiveStart,
      endPosition: effectiveEnd,
      width: canvas.width,
      height: canvas.height,
    });
    renderModelRef.current = model;

    drawChromatogram(ctx, model, { dark: isDarkTheme, mutations });
  }, [data, effectiveStart, effectiveEnd, mutations, theme]);

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!data || !canvasRef.current || !data.base_locations?.length) return;
    const model = renderModelRef.current;
    if (!model) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const closestBaseIdx = findNearestBaseIndex(model, x);
    if (closestBaseIdx < 0) return;

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
