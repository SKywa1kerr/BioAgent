// Pure helpers for the chromatogram viewport (zoom + pan window).
//
// A "viewport" is a half-open base-index range [start, end) over the full
// sequence [0, total]. The chromatogram canvas only renders bases inside the
// viewport, so the same data set can be inspected at different magnifications
// by shrinking / growing the window.
//
// These helpers are deliberately framework-free so they can be tested without
// React. The hook in `src/hooks/useChromatogramViewport.ts` wraps them with
// useState.
//
// Conventions:
//   - `total` is the count of base calls in the sample (>=0).
//   - `viewport.start` is inclusive, `viewport.end` is exclusive.
//   - `minWindow` is the smallest allowable window size in bases. Defaults to
//     12 — below that the canvas can't fit base labels comfortably.
//   - `factor` for zoomViewport is multiplicative on the *window size*. A
//     factor of 1.25 makes the window 25% larger (zoom OUT, more bases
//     visible); a factor of 0.8 makes it 20% smaller (zoom IN). This matches
//     the way browsers treat scale — but the consumer-facing zoomBy(1.25, ...)
//     in the hook flips this so the typical "+ zooms in" intuition holds.

const DEFAULT_MIN_WINDOW = 12;

function clampInt(value, lo, hi) {
  if (!Number.isFinite(value)) return lo;
  const n = Math.round(value);
  if (n < lo) return lo;
  if (n > hi) return hi;
  return n;
}

// Constrain a viewport so that:
//   - 0 <= start
//   - end <= total
//   - end - start >= min(minWindow, total)
// If the requested window would overflow either edge, slide it inward
// instead of clipping (this is how SnapGene / Geneious behave when you pan
// past the ends — the rectangle stays full-width).
export function clampViewport(viewport, total, minWindow) {
  const min = Math.max(1, Math.floor(minWindow ?? DEFAULT_MIN_WINDOW));
  const totalSafe = Math.max(0, Math.floor(total));
  if (totalSafe <= 0) return { start: 0, end: 0 };

  // Round but DO NOT clamp yet — we want to preserve the requested width
  // while sliding overflowing windows back inside the [0, total] range.
  let start = Number.isFinite(viewport?.start) ? Math.round(viewport.start) : 0;
  let end = Number.isFinite(viewport?.end) ? Math.round(viewport.end) : totalSafe;
  if (end < start) end = start;

  let window = end - start;
  const minWin = Math.min(min, totalSafe);

  // First normalise the *width*: too-narrow → grow to minWin; too-wide → cap
  // at total. The position correction happens in the slide pass below.
  if (window < minWin) {
    end = start + minWin;
    window = minWin;
  }
  if (window > totalSafe) {
    start = 0;
    end = totalSafe;
    window = totalSafe;
  }

  // Slide the window back into [0, total] while preserving its width.
  if (start < 0) {
    end -= start;
    start = 0;
  }
  if (end > totalSafe) {
    start -= end - totalSafe;
    end = totalSafe;
    if (start < 0) start = 0;
  }

  return { start, end };
}

// Zoom a viewport around an anchor base index, keeping that base's relative
// position inside the window stable. Factor < 1 shrinks the window (zoom in),
// factor > 1 grows it (zoom out). Honors minWindow / total as ceilings.
export function zoomViewport(viewport, factor, anchor, total, minWindow) {
  const totalSafe = Math.max(0, Math.floor(total));
  if (totalSafe <= 0) return { start: 0, end: 0 };

  const min = Math.max(1, Math.floor(minWindow ?? DEFAULT_MIN_WINDOW));
  const minWin = Math.min(min, totalSafe);
  const safeFactor = Number.isFinite(factor) && factor > 0 ? factor : 1;

  const currentStart = clampInt(viewport?.start, 0, totalSafe);
  const currentEnd = clampInt(viewport?.end, currentStart, totalSafe);
  const currentWindow = Math.max(1, currentEnd - currentStart);

  // Pick the anchor — fall back to the centre of the current window.
  let anchorBase = Number.isFinite(anchor)
    ? Math.round(anchor)
    : currentStart + currentWindow / 2;
  if (anchorBase < 0) anchorBase = 0;
  if (anchorBase > totalSafe) anchorBase = totalSafe;

  // Where the anchor sits within the current window (0..1). When the anchor
  // is outside the current window we still respect the ratio — that lets the
  // caller centre the viewport on a brand-new position by passing an out-of-
  // range anchor, but for typical "zoom around cursor" the anchor is in-view.
  const anchorRatio = (anchorBase - currentStart) / currentWindow;

  let nextWindow = Math.round(currentWindow * safeFactor);
  if (nextWindow < minWin) nextWindow = minWin;
  if (nextWindow > totalSafe) nextWindow = totalSafe;

  let nextStart = Math.round(anchorBase - anchorRatio * nextWindow);
  let nextEnd = nextStart + nextWindow;

  return clampViewport({ start: nextStart, end: nextEnd }, totalSafe, minWindow);
}

// Translate a viewport by `delta` bases (positive = right, negative = left).
export function panViewport(viewport, delta, total, minWindow) {
  const totalSafe = Math.max(0, Math.floor(total));
  if (totalSafe <= 0) return { start: 0, end: 0 };
  const d = Number.isFinite(delta) ? Math.round(delta) : 0;
  const start = (viewport?.start ?? 0) + d;
  const end = (viewport?.end ?? 0) + d;
  return clampViewport({ start, end }, totalSafe, minWindow);
}

// Recentre the viewport so its midpoint lands on `targetBase`, preserving the
// current window size. Used by mini-map click-to-jump.
export function centerViewport(viewport, targetBase, total, minWindow) {
  const totalSafe = Math.max(0, Math.floor(total));
  if (totalSafe <= 0) return { start: 0, end: 0 };
  const currentStart = clampInt(viewport?.start, 0, totalSafe);
  const currentEnd = clampInt(viewport?.end, currentStart, totalSafe);
  const window = Math.max(1, currentEnd - currentStart);
  const target = Number.isFinite(targetBase) ? Math.round(targetBase) : 0;
  const start = Math.round(target - window / 2);
  const end = start + window;
  return clampViewport({ start, end }, totalSafe, minWindow);
}

// Compute the zoom level (full sequence span / window size, clamped at 1).
export function viewportZoomLevel(viewport, total) {
  const totalSafe = Math.max(0, Math.floor(total));
  if (totalSafe <= 0) return 1;
  const window = Math.max(1, (viewport?.end ?? 0) - (viewport?.start ?? 0));
  const z = totalSafe / window;
  return z < 1 ? 1 : z;
}

export const CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW = DEFAULT_MIN_WINDOW;
