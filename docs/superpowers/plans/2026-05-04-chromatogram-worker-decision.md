# Chromatogram Worker Rendering — Decision

**Date:** 2026-05-04
**Closes:** Task 6 Step 5 of `2026-04-25-pr-borrowed-workbench-optimization.md`

## Decision

**Worker rendering deferred.** Main-thread render model, downsampling, and
nearest-base lookup extraction (Tasks 4–5) are sufficient for the dataset
sizes the workbench currently handles. No follow-up worker plan will be
opened until evidence of a freeze appears.

## Rationale

- **Render model is now O(width × maxPointsPerPixel)**, capped at the
  canvas budget regardless of trace length (`chromatogramRender.ts`
  builds a step-quantised set of points; long traces no longer translate
  to long render loops).
- **Hover does not re-allocate a model.** `ChromatogramCanvas` keeps the
  most recently built `ChromatogramRenderModel` in a ref and reuses it
  for `findNearestBaseIndex`, so mousemove cost is O(visible labels)
  rather than O(trace length) (commit `62781fa`).
- **Robust percentile + value clamping** prevent tall artefact peaks
  from compressing the visible signal, which previously made some users
  zoom in repeatedly and indirectly stress the renderer.
- **Build + automated test verification** (`npm run build`,
  `npm run test:js`, `npm run test:py`) all pass on the current branch.

The smoke checklist below should still be exercised once before declaring
this fully closed; if it surfaces a freeze, reopen with a new plan
named `YYYY-MM-DD-chromatogram-worker-rendering.md`.

## Smoke Checklist (Task 6 Step 4 — pending manual run)

Run `npm run electron:dev` and confirm:

- [ ] Run an existing dataset analysis (e.g. `分析 pro 数据集`).
- [ ] Open a sample drawer.
- [ ] Switch between at least two samples — chromatogram repaints without flicker.
- [ ] Toggle theme (light ↔ dark) — chromatogram and charts repaint with the new palette **(verifies fix from commit `0c20fd2`)**.
- [ ] Drag drawer width — width persists across reload, no laggy mousemove **(verifies fix from commit `d8d41d5`)**.
- [ ] Press Escape to close drawer.
- [ ] No console errors, no blank drawer.

## Reopen Trigger

Reopen with a worker plan only if the smoke run, or a real user report,
shows the drawer freezing for >250 ms when opening or interacting with a
chromatogram on a representative dataset.
