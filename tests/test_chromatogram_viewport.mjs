import test from "node:test";
import assert from "node:assert/strict";
import {
  centerViewport,
  clampViewport,
  panViewport,
  viewportZoomLevel,
  zoomViewport,
  CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW,
} from "../src/lib/workbench/chromatogramViewport.js";

test("clampViewport collapses to empty when total is zero", () => {
  assert.deepEqual(clampViewport({ start: 0, end: 10 }, 0), { start: 0, end: 0 });
});

test("clampViewport keeps a window inside [0, total] when given a valid one", () => {
  assert.deepEqual(clampViewport({ start: 10, end: 50 }, 100), { start: 10, end: 50 });
});

test("clampViewport slides a viewport that overflows the right edge", () => {
  // Window is 30 wide; total is 100. Sliding the requested [80, 110] window
  // to the right edge keeps the same width and respects the ceiling.
  assert.deepEqual(clampViewport({ start: 80, end: 110 }, 100), { start: 70, end: 100 });
});

test("clampViewport grows a window below minWindow", () => {
  // [10, 12] is too small; defaults push it out to minWindow (=12).
  const out = clampViewport({ start: 10, end: 12 }, 100);
  assert.equal(out.end - out.start, CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW);
  assert.equal(out.start, 10);
});

test("clampViewport caps a window above total", () => {
  assert.deepEqual(clampViewport({ start: -50, end: 10000 }, 100), { start: 0, end: 100 });
});

test("zoomViewport in around an anchor preserves the anchor base ratio", () => {
  const before = { start: 0, end: 100 };
  const total = 1000;
  const anchor = 50;
  const after = zoomViewport(before, 0.5, anchor, total);
  // Window halved → anchor was 50% of [0,100]; after zoom it should still
  // sit at ~50% of the new window. With rounding tolerate ±1.
  const ratio = (anchor - after.start) / (after.end - after.start);
  assert.ok(Math.abs(ratio - 0.5) < 0.05, `ratio drifted: ${ratio}`);
  assert.equal(after.end - after.start, 50);
});

test("zoomViewport hits minWindow floor", () => {
  // Repeatedly zooming in past the minimum should cap at minWindow, never go
  // below it.
  let v = { start: 100, end: 200 };
  for (let i = 0; i < 20; i += 1) {
    v = zoomViewport(v, 0.5, 150, 1000, 12);
  }
  assert.equal(v.end - v.start, 12);
});

test("zoomViewport hits total ceiling", () => {
  // Zoom out until we cover the full sequence.
  let v = { start: 100, end: 200 };
  for (let i = 0; i < 20; i += 1) {
    v = zoomViewport(v, 1.5, 150, 500, 12);
  }
  assert.deepEqual(v, { start: 0, end: 500 });
});

test("zoomViewport zoom-in then zoom-out around the same anchor returns near the same window", () => {
  const total = 1000;
  const anchor = 200;
  const before = { start: 100, end: 300 };
  const zoomedIn = zoomViewport(before, 0.5, anchor, total);
  const zoomedOut = zoomViewport(zoomedIn, 2, anchor, total);
  // Tolerate ±2 base rounding drift.
  assert.ok(Math.abs(zoomedOut.start - before.start) <= 2);
  assert.ok(Math.abs(zoomedOut.end - before.end) <= 2);
});

test("panViewport clamps to [0, total]", () => {
  // Trying to pan off the left edge should slide the window back to [0, w].
  const left = panViewport({ start: 10, end: 60 }, -1000, 100);
  assert.deepEqual(left, { start: 0, end: 50 });
  // Off the right edge: clamp to [total-w, total].
  const right = panViewport({ start: 10, end: 60 }, 1000, 100);
  assert.deepEqual(right, { start: 50, end: 100 });
});

test("panViewport with delta 0 is identity (after clamp)", () => {
  assert.deepEqual(panViewport({ start: 10, end: 60 }, 0, 100), { start: 10, end: 60 });
});

test("centerViewport centres the window on the target base", () => {
  const v = { start: 100, end: 200 };
  const after = centerViewport(v, 500, 1000);
  // Window is still 100 wide; it should now straddle 500.
  assert.equal(after.end - after.start, 100);
  assert.equal((after.start + after.end) / 2, 500);
});

test("centerViewport clicking near the right edge slides the window inward", () => {
  // Clicking near base=995 with a 100-wide window in a 1000-base sequence
  // should slide the rectangle to [900, 1000] rather than overflowing.
  const v = { start: 100, end: 200 };
  const after = centerViewport(v, 995, 1000);
  assert.equal(after.end, 1000);
  assert.equal(after.start, 900);
});

test("centerViewport clicking near the left edge slides the window inward", () => {
  const v = { start: 500, end: 600 };
  const after = centerViewport(v, 5, 1000);
  assert.equal(after.start, 0);
  assert.equal(after.end - after.start, 100);
});

test("viewportZoomLevel reports total/window clamped at 1", () => {
  assert.equal(viewportZoomLevel({ start: 0, end: 100 }, 100), 1);
  assert.equal(viewportZoomLevel({ start: 0, end: 50 }, 100), 2);
  assert.equal(viewportZoomLevel({ start: 0, end: 200 }, 100), 1);
  assert.equal(viewportZoomLevel({ start: 0, end: 0 }, 0), 1);
});

test("zoomViewport with non-finite factor leaves the viewport unchanged in size", () => {
  const v = { start: 100, end: 200 };
  const out = zoomViewport(v, NaN, 150, 1000);
  assert.equal(out.end - out.start, 100);
});

test("zoomViewport with a missing anchor falls back to the window centre", () => {
  const before = { start: 100, end: 200 };
  const total = 1000;
  // Without an anchor, halving the window should keep the centre stable.
  const after = zoomViewport(before, 0.5, undefined, total);
  const beforeCentre = (before.start + before.end) / 2;
  const afterCentre = (after.start + after.end) / 2;
  assert.ok(Math.abs(beforeCentre - afterCentre) <= 1);
  assert.equal(after.end - after.start, 50);
});
