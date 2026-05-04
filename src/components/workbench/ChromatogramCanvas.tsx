import { useEffect, useMemo, useRef, useState } from "react";
import type { ChromatogramData } from "./types";
import {
  buildChromatogramRenderModel,
  drawChromatogram,
  findNearestBaseIndex,
} from "./chromatogramRender";
import { useTheme } from "../../hooks/useTheme";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import "./ChromatogramCanvas.css";

interface Props {
  data: ChromatogramData | null;
  startPosition: number;
  endPosition: number;
  mutations?: Array<{ position?: number }>;
  language: AppLanguage;
}

export function ChromatogramCanvas({ data, startPosition, endPosition, mutations, language }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderModelRef = useRef<ReturnType<typeof buildChromatogramRenderModel> | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);
  // Logical (CSS-pixel) canvas size. Drives both the render model dimensions
  // and the bitmap-vs-CSS scaling so that the chromatogram stays sharp on
  // high-DPR screens and reflows when the drawer is resized.
  const [size, setSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  // Keyboard navigation cursor (base index in the original baseCalls). Null
  // means no keyboard focus; arrow keys move it, mouse interaction leaves it
  // alone so the two modes coexist.
  const [keyboardCursor, setKeyboardCursor] = useState<number | null>(null);
  const theme = useTheme();

  const totalBases = endPosition - startPosition;
  const visibleBases = Math.max(10, Math.floor(totalBases / zoomLevel));
  const effectiveStart = Math.max(startPosition, startPosition + panOffset);
  const effectiveEnd = Math.min(endPosition, effectiveStart + visibleBases);

  // Track the canvas's CSS-pixel size so the bitmap can be sized to
  // displaySize × devicePixelRatio. ResizeObserver covers drawer resizes
  // (the parent <aside> width is user-draggable) and zoom changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const target = canvas;
    function read() {
      const cssW = target.clientWidth || 0;
      const cssH = target.clientHeight || 0;
      setSize((prev) => (prev.width === cssW && prev.height === cssH ? prev : { width: cssW, height: cssH }));
    }
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(target);
    return () => ro.disconnect();
  }, []);

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

    const cssW = size.width;
    const cssH = size.height;
    if (cssW <= 0 || cssH <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    const bitmapW = Math.max(1, Math.floor(cssW * dpr));
    const bitmapH = Math.max(1, Math.floor(cssH * dpr));
    if (canvas.width !== bitmapW) canvas.width = bitmapW;
    if (canvas.height !== bitmapH) canvas.height = bitmapH;
    // Anchor the drawing space to CSS pixels so render-model coordinates
    // map 1:1 to layout pixels and findNearestBaseIndex stays correct.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const isDarkTheme = theme === "dark";
    const model = buildChromatogramRenderModel(data, {
      startPosition: effectiveStart,
      endPosition: effectiveEnd,
      width: cssW,
      height: cssH,
    });
    renderModelRef.current = model;

    drawChromatogram(ctx, model, { dark: isDarkTheme, mutations, keyboardCursor });
  }, [data, effectiveStart, effectiveEnd, mutations, theme, size, keyboardCursor]);

  // When the user switches to a different sample, drop the keyboard cursor
  // so we never announce a stale position from the previous read.
  useEffect(() => {
    setKeyboardCursor(null);
  }, [data]);

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!data || !canvasRef.current || !data.base_locations?.length) return;
    const model = renderModelRef.current;
    if (!model) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    // The render model is in CSS-pixel space (see drawing effect), so the
    // CSS-pixel offset is what findNearestBaseIndex expects.
    const x = event.clientX - rect.left;
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

  function handleKeyDown(event: React.KeyboardEvent<HTMLCanvasElement>) {
    if (!data || !data.baseCalls.length) return;
    const last = data.baseCalls.length - 1;
    let next: number | null = null;
    switch (event.key) {
      case "ArrowLeft":
        next = Math.max(0, (keyboardCursor ?? 0) - 1);
        break;
      case "ArrowRight":
        next = Math.min(last, (keyboardCursor ?? -1) + 1);
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = last;
        break;
      default:
        return;
    }
    event.preventDefault();
    setKeyboardCursor(next);
  }

  // Live-region announcement; recomputed only when the cursor or data move so
  // assistive tech does not re-announce on every unrelated re-render.
  const liveText = useMemo(() => {
    if (keyboardCursor === null || !data) return "";
    const base = data.baseCalls[keyboardCursor] || "-";
    const quality = data.quality?.[keyboardCursor];
    const isMixed = data.mixed_peaks?.includes(keyboardCursor);
    let text = t(language, "chrom.aria.position", { pos: keyboardCursor + 1, base });
    if (typeof quality === "number") {
      text += t(language, "chrom.aria.quality", { quality });
    }
    if (isMixed) {
      text += t(language, "chrom.aria.mixed");
    }
    return text;
  }, [keyboardCursor, data, language]);

  const ariaLabel = data
    ? t(language, "chrom.aria.label", { bases: data.baseCalls.length })
    : t(language, "table.chromatogram");

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
        className="chromatogram-canvas"
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onMouseDown={(e) => e.currentTarget.focus()}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
        onKeyDown={handleKeyDown}
      />
      <div className="chromatogram-sr-only" role="status" aria-live="polite">{liveText}</div>
      {tooltip ? <div className="chromatogram-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.content}</div> : null}
    </div>
  );
}
