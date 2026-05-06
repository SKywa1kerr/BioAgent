import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChromatogramData } from "./types";
import {
  buildChromatogramRenderModel,
  drawChromatogram,
  findNearestBaseIndex,
} from "./chromatogramRender";
import { ChromatogramMiniMap } from "./ChromatogramMiniMap";
import { useTheme } from "../../hooks/useTheme";
import { useChromatogramViewport } from "../../hooks/useChromatogramViewport";
import { Icon } from "../ui/Icon";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import "./ChromatogramCanvas.css";

interface Props {
  data: ChromatogramData | null;
  // The legacy startPosition / endPosition props become *initial* values
  // only — once the canvas mounts, the viewport is hook-managed so the user
  // can zoom and pan freely. Both props are 1-based to match the existing
  // call sites; we convert internally.
  startPosition: number;
  endPosition: number;
  mutations?: Array<{ position?: number }>;
  language: AppLanguage;
}

const ZOOM_IN_FACTOR = 0.8;     // shrink window 20% → zoom in
const ZOOM_OUT_FACTOR = 1.25;   // grow window 25%   → zoom out

export function ChromatogramCanvas({ data, startPosition, endPosition, mutations, language }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderModelRef = useRef<ReturnType<typeof buildChromatogramRenderModel> | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  // Logical (CSS-pixel) canvas size. Drives both the render model dimensions
  // and the bitmap-vs-CSS scaling so that the chromatogram stays sharp on
  // high-DPR screens and reflows when the drawer is resized.
  const [size, setSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  // Keyboard navigation cursor (base index in the original baseCalls). Null
  // means no keyboard focus; arrow keys move it, mouse interaction leaves it
  // alone so the two modes coexist.
  const [keyboardCursor, setKeyboardCursor] = useState<number | null>(null);
  const theme = useTheme();

  const total = data?.baseCalls?.length ?? 0;
  const { viewport, zoomLevel, setViewport, zoomBy, panBy, fit } = useChromatogramViewport({
    total,
    // The legacy props are 1-based inclusive on both ends; convert to the
    // half-open base-index range the viewport math uses.
    initialStart: Math.max(0, (startPosition ?? 1) - 1),
    initialEnd: Math.max(0, endPosition ?? total),
  });

  // Hover cursor: track the base index under the mouse so Ctrl+wheel zoom
  // can anchor on it. Falls back to keyboardCursor or viewport centre.
  const hoverBaseRef = useRef<number | null>(null);

  // The render model takes 1-based positions, matching its existing API.
  const effectiveStart = viewport.start + 1;
  const effectiveEnd = viewport.end;

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
    hoverBaseRef.current = null;
  }, [data]);

  // Auto-scroll: when the keyboard cursor lands outside the viewport, slide
  // the viewport just enough to bring it back in. Avoids jumpiness mid-drag.
  useEffect(() => {
    if (keyboardCursor === null) return;
    if (keyboardCursor < viewport.start) {
      panBy(keyboardCursor - viewport.start);
    } else if (keyboardCursor >= viewport.end) {
      panBy(keyboardCursor - viewport.end + 1);
    }
    // Pan helpers are stable; we deliberately avoid re-running on viewport
    // change so we don't fight the user's manual pan.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyboardCursor]);

  function getBaseAtClientX(clientX: number): number | null {
    const canvas = canvasRef.current;
    const model = renderModelRef.current;
    if (!canvas || !model) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const idx = findNearestBaseIndex(model, x);
    return idx >= 0 ? idx : null;
  }

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!data || !canvasRef.current || !data.base_locations?.length) return;
    const closestBaseIdx = getBaseAtClientX(event.clientX);
    if (closestBaseIdx === null) return;
    hoverBaseRef.current = closestBaseIdx;

    const quality = data.quality?.[closestBaseIdx];
    const base = data.baseCalls[closestBaseIdx] || "-";
    const isMixed = data.mixed_peaks.includes(closestBaseIdx);
    setTooltip({
      x: event.clientX + 12,
      y: event.clientY - 18,
      content: `Pos ${closestBaseIdx + 1} | Base ${base} | Q ${typeof quality === "number" ? quality : "-"}${isMixed ? " | Mixed" : ""}`,
    });
  }

  function handleMouseLeave() {
    setTooltip(null);
    hoverBaseRef.current = null;
  }

  // Click-drag panning on the main canvas. We capture the initial mouse
  // position and viewport, and recompute pan delta in *base* units using
  // the visible base count. dragMovedRef gates the click-to-focus handler
  // so a drag never also moves the keyboard cursor.
  const dragRef = useRef<{ startX: number; startStart: number; startEnd: number; moved: boolean } | null>(null);

  function handleMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
    event.currentTarget.focus();
    if (!canvasRef.current || total <= 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const drag = {
      startX: event.clientX,
      startStart: viewport.start,
      startEnd: viewport.end,
      moved: false,
    };
    dragRef.current = drag;

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return;
      const dx = ev.clientX - drag.startX;
      if (Math.abs(dx) < 3 && !drag.moved) return; // dead-zone for clicks
      drag.moved = true;
      const window = drag.startEnd - drag.startStart;
      const basesPerPx = rect.width > 0 ? window / rect.width : 0;
      const deltaBases = -dx * basesPerPx;
      setViewport({
        start: drag.startStart + deltaBases,
        end: drag.startEnd + deltaBases,
      });
    };
    const onUp = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      // If the mouse barely moved, treat it as a click → focus the base.
      if (!drag.moved) {
        const idx = getBaseAtClientX(ev.clientX);
        if (idx !== null) setKeyboardCursor(idx);
      }
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    if (!data || total <= 0) return;
    // Always swallow scroll over the chromatogram so the page doesn't jump
    // out from under the user. Plain wheel pans, Ctrl/Meta wheel zooms.
    event.preventDefault();
    if (event.ctrlKey || event.metaKey) {
      const anchor =
        hoverBaseRef.current ??
        keyboardCursor ??
        Math.floor((viewport.start + viewport.end) / 2);
      const factor = event.deltaY < 0 ? ZOOM_IN_FACTOR : ZOOM_OUT_FACTOR;
      zoomBy(factor, anchor);
    } else {
      const window = viewport.end - viewport.start;
      // One wheel tick ≈ 10% of the visible window — feels right on track-
      // pads (lots of small ticks) and discrete mouse wheels alike.
      const step = Math.max(1, Math.round(window * 0.1));
      panBy(event.deltaY > 0 ? step : -step);
    }
  }

  // React's onWheel listener is attached to the React root and may be passive
  // in some bundlers / setups, in which case event.preventDefault() inside
  // handleWheel above is silently dropped and the page scrolls. We attach a
  // native, explicitly-non-passive listener as a belt-and-braces guard so
  // scroll never escapes the chromatogram. The actual handling still lives in
  // handleWheel — this listener only blocks the default page scroll.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    function block(ev: WheelEvent) {
      // Only block when we have data to zoom/pan; otherwise let the page
      // scroll past the empty canvas as before.
      if (!data || total <= 0) return;
      ev.preventDefault();
    }
    canvas.addEventListener("wheel", block, { passive: false });
    return () => canvas.removeEventListener("wheel", block);
  }, [data, total]);

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
      case "+":
      case "=":
        event.preventDefault();
        zoomBy(
          ZOOM_IN_FACTOR,
          keyboardCursor ?? hoverBaseRef.current ?? null,
        );
        return;
      case "-":
      case "_":
        event.preventDefault();
        zoomBy(
          ZOOM_OUT_FACTOR,
          keyboardCursor ?? hoverBaseRef.current ?? null,
        );
        return;
      case "0":
        event.preventDefault();
        fit();
        return;
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

  const onZoomIn = useCallback(() => {
    zoomBy(ZOOM_IN_FACTOR, keyboardCursor ?? hoverBaseRef.current ?? null);
  }, [zoomBy, keyboardCursor]);
  const onZoomOut = useCallback(() => {
    zoomBy(ZOOM_OUT_FACTOR, keyboardCursor ?? hoverBaseRef.current ?? null);
  }, [zoomBy, keyboardCursor]);

  return (
    <div className="chromatogram-container">
      <div className="chromatogram-toolbar" role="toolbar">
        <button
          type="button"
          className="icon-button chromatogram-toolbar-btn"
          onClick={onZoomIn}
          aria-label={t(language, "chrom.zoom.in")}
          title={t(language, "chrom.zoom.in")}
          disabled={!data || total <= 0}
        >
          <Icon name="zoom-in" size={14} />
        </button>
        <button
          type="button"
          className="icon-button chromatogram-toolbar-btn"
          onClick={onZoomOut}
          aria-label={t(language, "chrom.zoom.out")}
          title={t(language, "chrom.zoom.out")}
          disabled={!data || total <= 0}
        >
          <Icon name="zoom-out" size={14} />
        </button>
        <button
          type="button"
          className="icon-button chromatogram-toolbar-btn"
          onClick={fit}
          aria-label={t(language, "chrom.zoom.fit")}
          title={t(language, "chrom.zoom.fit")}
          disabled={!data || total <= 0}
        >
          <Icon name="refresh" size={14} />
        </button>
        <span className="chromatogram-zoom-label">
          {zoomLevel > 1.05 ? `${zoomLevel.toFixed(1)}x` : "1x"}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        className="chromatogram-canvas"
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        onKeyDown={handleKeyDown}
      />
      <div className="chromatogram-sr-only" role="status" aria-live="polite">{liveText}</div>
      {tooltip ? <div className="chromatogram-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.content}</div> : null}
      {data && total > 0 ? (
        <ChromatogramMiniMap
          data={data}
          viewport={viewport}
          onChange={setViewport}
          language={language}
        />
      ) : null}
    </div>
  );
}
