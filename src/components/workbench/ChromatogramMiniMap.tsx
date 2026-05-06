import { useEffect, useMemo, useRef, useState } from "react";
import type { ChromatogramData } from "./types";
import type { ViewportState } from "../../hooks/useChromatogramViewport";
import { useTheme } from "../../hooks/useTheme";
import {
  centerViewport,
  clampViewport,
  CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW,
} from "../../lib/workbench/chromatogramViewport";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

// Below the main chromatogram canvas we draw a thin "mini-map" band that
// shows the entire sequence at once. A draggable rectangle on top of the
// band marks the current viewport — same idea as Photoshop's navigator.
//
// The band itself is a low-resolution silhouette of the per-base quality
// (when present) or, if quality is missing, a flat strip. We picked quality
// over a full trace silhouette because:
//   - the band is only ~36 px tall and would smear the four-channel trace
//     into mush at most sequence lengths;
//   - quality already varies per-base and visualises read confidence —
//     giving the user signal beyond just "where am I in the sequence";
//   - per-base quality is far cheaper to redraw than four trace arrays.
//
// Interactions:
//   - click on the band: centre the viewport on that base position (preserve
//     the current window size).
//   - drag the rectangle: pan the viewport.
//   - drag the rectangle's left/right edges: resize the window (zoom).

interface Props {
  data: ChromatogramData;
  viewport: ViewportState;
  onChange: (next: ViewportState) => void;
  language: AppLanguage;
  minWindow?: number;
}

const HEIGHT = 36;
const EDGE_HANDLE_PX = 6;

type DragMode = "pan" | "resize-left" | "resize-right";

interface DragState {
  mode: DragMode;
  pointerStart: number;     // clientX at mousedown
  windowStart: number;      // viewport.start at mousedown
  windowEnd: number;        // viewport.end at mousedown
  containerLeft: number;    // bounding box left at mousedown
  containerWidth: number;   // bounding box width at mousedown
}

function basesPerPixel(total: number, containerWidth: number): number {
  if (containerWidth <= 0 || total <= 0) return 0;
  return total / containerWidth;
}

export function ChromatogramMiniMap({ data, viewport, onChange, language, minWindow }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const minWindowResolved = minWindow ?? CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW;

  const [size, setSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const theme = useTheme();

  const total = data.baseCalls?.length ?? 0;

  // Track CSS-pixel size — the band reflows with the drawer / compare view.
  useEffect(() => {
    const target = containerRef.current;
    if (!target) return;
    const read = () => {
      const cssW = target.clientWidth || 0;
      setSize((prev) => (prev.width === cssW && prev.height === HEIGHT ? prev : { width: cssW, height: HEIGHT }));
    };
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(target);
    return () => ro.disconnect();
  }, []);

  // Pre-compute the silhouette samples once per data change so resize and
  // viewport drag don't redo the full reduction.
  const silhouette = useMemo<number[] | null>(() => {
    if (!Array.isArray(data.quality) || data.quality.length === 0) return null;
    return data.quality;
  }, [data.quality]);

  // Paint the band + selection rectangle.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cssW = size.width;
    const cssH = size.height;
    if (cssW <= 0 || cssH <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    const bitmapW = Math.max(1, Math.floor(cssW * dpr));
    const bitmapH = Math.max(1, Math.floor(cssH * dpr));
    if (canvas.width !== bitmapW) canvas.width = bitmapW;
    if (canvas.height !== bitmapH) canvas.height = bitmapH;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const isDark = theme === "dark";
    const bandTrack = isDark ? "rgba(148, 163, 184, 0.16)" : "rgba(148, 163, 184, 0.22)";
    const bandFill = isDark ? "rgba(148, 163, 184, 0.42)" : "rgba(71, 85, 105, 0.55)";

    // Background track.
    ctx.fillStyle = bandTrack;
    ctx.fillRect(0, 0, cssW, cssH);

    // Silhouette: bin the quality array into pixel-wide buckets and draw a
    // bar per pixel reaching from the bottom. With no quality available we
    // simply leave the track flat — still useful as a click target.
    if (silhouette && silhouette.length > 0 && total > 0) {
      ctx.fillStyle = bandFill;
      const buckets = cssW;
      const perPixel = total / buckets;
      // Quality scale: clamp at 60 (Phred ceilings).
      const QMAX = 60;
      for (let i = 0; i < buckets; i += 1) {
        const startIdx = Math.floor(i * perPixel);
        const endIdx = Math.max(startIdx + 1, Math.floor((i + 1) * perPixel));
        let max = 0;
        for (let j = startIdx; j < endIdx && j < silhouette.length; j += 1) {
          const q = silhouette[j];
          if (typeof q === "number" && q > max) max = q;
        }
        if (max <= 0) continue;
        const h = Math.min(cssH, (max / QMAX) * cssH);
        ctx.fillRect(i, cssH - h, 1, h);
      }
    }
  }, [size, silhouette, theme, total]);

  // Selection rectangle position (CSS pixels) — recomputed every render so
  // the rectangle tracks the viewport without going through canvas state.
  const selection = useMemo(() => {
    if (size.width <= 0 || total <= 0) {
      return { left: 0, width: 0, visible: false };
    }
    const px = size.width / total;
    const left = Math.max(0, Math.min(size.width, viewport.start * px));
    const right = Math.max(left, Math.min(size.width, viewport.end * px));
    const width = Math.max(2, right - left); // never narrower than 2 px
    return { left, width, visible: true };
  }, [size.width, viewport, total]);

  function onPointerDown(event: React.MouseEvent<HTMLElement>, mode: DragMode) {
    if (!containerRef.current || total <= 0) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = containerRef.current.getBoundingClientRect();
    dragRef.current = {
      mode,
      pointerStart: event.clientX,
      windowStart: viewport.start,
      windowEnd: viewport.end,
      containerLeft: rect.left,
      containerWidth: rect.width,
    };

    function move(ev: MouseEvent) {
      const drag = dragRef.current;
      if (!drag) return;
      const bpp = basesPerPixel(total, drag.containerWidth);
      if (bpp <= 0) return;
      const deltaPx = ev.clientX - drag.pointerStart;
      const deltaBases = deltaPx * bpp;
      let next: ViewportState;
      if (drag.mode === "pan") {
        next = clampViewport(
          { start: drag.windowStart + deltaBases, end: drag.windowEnd + deltaBases },
          total,
          minWindowResolved,
        );
      } else if (drag.mode === "resize-left") {
        const proposedStart = drag.windowStart + deltaBases;
        next = clampViewport(
          { start: proposedStart, end: drag.windowEnd },
          total,
          minWindowResolved,
        );
      } else {
        const proposedEnd = drag.windowEnd + deltaBases;
        next = clampViewport(
          { start: drag.windowStart, end: proposedEnd },
          total,
          minWindowResolved,
        );
      }
      onChangeRef.current(next);
    }
    function up() {
      dragRef.current = null;
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    }
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function onBandClick(event: React.MouseEvent<HTMLDivElement>) {
    if (!containerRef.current || total <= 0) return;
    if (dragRef.current) return; // a real drag finished, swallow the click
    const rect = containerRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const targetBase = (x / Math.max(1, rect.width)) * total;
    onChangeRef.current(centerViewport(viewport, targetBase, total, minWindowResolved));
  }

  return (
    <div
      ref={containerRef}
      className="chromatogram-minimap"
      role="slider"
      aria-label={t(language, "chrom.minimap.aria")}
      aria-valuemin={0}
      aria-valuemax={Math.max(0, total)}
      aria-valuenow={Math.round((viewport.start + viewport.end) / 2)}
      onMouseDown={onBandClick}
      style={{ height: HEIGHT }}
    >
      <canvas ref={canvasRef} className="chromatogram-minimap-canvas" aria-hidden="true" />
      {selection.visible ? (
        <div
          className="chromatogram-minimap-window"
          style={{ left: selection.left, width: selection.width }}
          onMouseDown={(e) => onPointerDown(e, "pan")}
        >
          <span
            className="chromatogram-minimap-handle chromatogram-minimap-handle-left"
            style={{ width: EDGE_HANDLE_PX }}
            onMouseDown={(e) => onPointerDown(e, "resize-left")}
            aria-hidden="true"
          />
          <span
            className="chromatogram-minimap-handle chromatogram-minimap-handle-right"
            style={{ width: EDGE_HANDLE_PX }}
            onMouseDown={(e) => onPointerDown(e, "resize-right")}
            aria-hidden="true"
          />
        </div>
      ) : null}
    </div>
  );
}
