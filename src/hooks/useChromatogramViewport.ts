import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW,
  centerViewport,
  clampViewport,
  panViewport,
  viewportZoomLevel,
  zoomViewport,
} from "../lib/workbench/chromatogramViewport";

// React wrapper around the pure viewport math. The hook owns the viewport
// state, exposes setter helpers (zoomBy / panBy / centerOn / fit), and
// re-clamps whenever `total` changes (e.g. when the user opens a different
// sample) so the canvas never tries to read out-of-range bases.
//
// Anchor convention for zoomBy:
//   - factor < 1 (e.g. 0.8) ZOOMS IN — window shrinks, fewer bases visible
//   - factor > 1 (e.g. 1.25) ZOOMS OUT — window grows, more bases visible
// (this matches the underlying zoomViewport math).
//
// The hook intentionally does NOT clamp on every render; only when total
// actually changes. That keeps an in-progress drag-pan from being clobbered
// by the viewport bouncing back to its initial value.

export interface ViewportState {
  start: number;
  end: number;
}

export interface UseChromatogramViewportArgs {
  total: number;
  initialStart?: number;
  initialEnd?: number;
  minWindow?: number;
}

export interface ChromatogramViewportApi {
  viewport: ViewportState;
  zoomLevel: number;
  setViewport: (next: ViewportState) => void;
  zoomBy: (factor: number, anchor?: number | null) => void;
  panBy: (delta: number) => void;
  centerOn: (base: number) => void;
  fit: () => void;
}

function deriveInitial(args: UseChromatogramViewportArgs): ViewportState {
  const total = Math.max(0, Math.floor(args.total ?? 0));
  const desiredStart = Number.isFinite(args.initialStart) ? Math.floor(args.initialStart!) : 0;
  const desiredEnd = Number.isFinite(args.initialEnd)
    ? Math.floor(args.initialEnd!)
    : total;
  return clampViewport(
    { start: desiredStart, end: desiredEnd },
    total,
    args.minWindow ?? CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW,
  );
}

export function useChromatogramViewport(
  args: UseChromatogramViewportArgs,
): ChromatogramViewportApi {
  const total = Math.max(0, Math.floor(args.total ?? 0));
  const minWindow = args.minWindow ?? CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW;

  // Carry the latest min/total in refs so callbacks stay stable even when
  // the parent re-renders with a fresh `total` (e.g. drawer width drag).
  const totalRef = useRef(total);
  const minWindowRef = useRef(minWindow);
  totalRef.current = total;
  minWindowRef.current = minWindow;

  const [viewport, setViewportState] = useState<ViewportState>(() => deriveInitial(args));

  // When the underlying sample changes (total flips), reset to a sensible
  // window so we don't show bases that no longer exist. We compare against
  // the ref's previous value to avoid resetting on every render.
  const lastTotalRef = useRef(total);
  useEffect(() => {
    if (lastTotalRef.current !== total) {
      lastTotalRef.current = total;
      setViewportState(deriveInitial({ ...args, total }));
    }
    // We deliberately depend only on `total`. The other args are read fresh
    // through deriveInitial above when this branch fires.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [total]);

  const setViewport = useCallback((next: ViewportState) => {
    setViewportState(clampViewport(next, totalRef.current, minWindowRef.current));
  }, []);

  const zoomBy = useCallback((factor: number, anchor?: number | null) => {
    setViewportState((prev) =>
      zoomViewport(
        prev,
        factor,
        anchor ?? null,
        totalRef.current,
        minWindowRef.current,
      ),
    );
  }, []);

  const panBy = useCallback((delta: number) => {
    setViewportState((prev) =>
      panViewport(prev, delta, totalRef.current, minWindowRef.current),
    );
  }, []);

  const centerOn = useCallback((base: number) => {
    setViewportState((prev) =>
      centerViewport(prev, base, totalRef.current, minWindowRef.current),
    );
  }, []);

  const fit = useCallback(() => {
    setViewportState(
      clampViewport(
        { start: 0, end: totalRef.current },
        totalRef.current,
        minWindowRef.current,
      ),
    );
  }, []);

  const zoomLevel = useMemo(
    () => viewportZoomLevel(viewport, total),
    [viewport, total],
  );

  return { viewport, zoomLevel, setViewport, zoomBy, panBy, centerOn, fit };
}
