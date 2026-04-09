# Results Workbench Inspired By BioAgent Max Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the analysis results surface into a BioAgent Max-style results workbench with summary cards, stats, a sample table, expandable sample details, and a quieter persistent AI chat rail on the right.

**Architecture:** Keep the existing Electron shell and analysis execution pipeline, but replace the current single-sample-centered center panel with a results workbench composed of focused React components. Preserve the Agent rail on the right, pass it richer batch and selected-sample context, and keep the current dataset import flow at the top of the page.

**Tech Stack:** React, TypeScript, Electron, Vite, Python backend sidecar, existing CSS modules

---

## File Map

- Modify: `src/App.tsx`
  - Shift analysis page orchestration from single-sample viewer-first to results-workbench-first.
- Modify: `src/App.css`
  - Restyle analysis center region into summary/statistics/table/expander sections.
- Modify: `src/i18n.ts`
  - Add strings for workbench summary, table labels, filters, empty states, and untested status copy.
- Modify: `src/types/index.ts`
  - Add any small UI-facing status helpers or derived view types if needed.
- Modify: `src/components/AgentPanel.tsx`
  - Adjust header/context copy so the panel behaves as a batch-aware assistant rather than the main interaction surface.
- Modify: `src/components/AgentPanel.css`
  - Further subordinate the AI rail visually relative to the results workbench.
- Create: `src/components/ResultsWorkbench.tsx`
  - Main center workbench wrapper.
- Create: `src/components/ResultsWorkbench.css`
  - Shared layout and visual styles for the workbench.
- Create: `src/components/ResultsSummary.tsx`
  - Summary cards with explicit `未测通` bucket.
- Create: `src/components/ResultsStats.tsx`
  - Lightweight status/coverage/identity overview.
- Create: `src/components/ResultsTable.tsx`
  - Scan-friendly sample table with row selection.
- Create: `src/components/SampleDetailsList.tsx`
  - Expandable sample result items.
- Modify: `src/components/SequenceViewer.tsx`
  - Fix restriction-label clipping and adapt viewer for embedding inside expander detail panels.
- Modify: `src/components/SequenceViewer.css`
  - Prevent label overlap and improve embedded readability.

### Task 1: Add result-workbench data shaping and status bucketing

**Files:**
- Modify: `src/types/index.ts`
- Create: `src/components/ResultsWorkbench.tsx`
- Test: existing `npm.cmd run build`

- [ ] **Step 1: Write the failing test**

Use a TypeScript-first red step by introducing a missing import/usage from `ResultsWorkbench` in `src/App.tsx` before the component exists:

```tsx
import { ResultsWorkbench } from "./components/ResultsWorkbench";
```

and temporarily render:

```tsx
<ResultsWorkbench samples={samples} selectedId={selectedId} />
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm.cmd run build`
Expected: FAIL because `ResultsWorkbench` does not exist and required props/types are unresolved.

- [ ] **Step 3: Write minimal implementation**

Create `src/components/ResultsWorkbench.tsx` with derived helpers:

```tsx
import { Sample } from "../types";

export interface ResultBuckets {
  ok: number;
  wrong: number;
  uncertain: number;
  untested: number;
}

export function bucketSampleStatus(sample: Sample): keyof ResultBuckets {
  if (sample.reason === "未测通") return "untested";
  if (sample.status === "ok") return "ok";
  if (sample.status === "wrong") return "wrong";
  return "uncertain";
}
```

and a minimal renderable shell:

```tsx
export function ResultsWorkbench() {
  return <section className="results-workbench" />;
}
```

Add any minimal shared type needed in `src/types/index.ts`, for example:

```ts
export type ResultWorkbenchStatus = "ok" | "wrong" | "uncertain" | "untested";
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/types/index.ts src/components/ResultsWorkbench.tsx
git commit -m "feat: add results workbench status bucketing"
```

### Task 2: Build summary cards and statistics overview

**Files:**
- Create: `src/components/ResultsSummary.tsx`
- Create: `src/components/ResultsStats.tsx`
- Create: `src/components/ResultsWorkbench.css`
- Modify: `src/components/ResultsWorkbench.tsx`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Write the failing test**

Add imports and usage in `ResultsWorkbench.tsx` for components that do not yet exist:

```tsx
import { ResultsSummary } from "./ResultsSummary";
import { ResultsStats } from "./ResultsStats";
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm.cmd run build`
Expected: FAIL because the summary/stats components and strings do not exist.

- [ ] **Step 3: Write minimal implementation**

Create compact components:

```tsx
<ResultsSummary
  total={samples.length}
  ok={buckets.ok}
  wrong={buckets.wrong}
  uncertain={buckets.uncertain}
  untested={buckets.untested}
  averageIdentity={averageIdentity}
  averageCoverage={averageCoverage}
  language={language}
/>
```

```tsx
<ResultsStats
  buckets={buckets}
  samples={samples}
  language={language}
/>
```

Add i18n keys:

```ts
results: {
  summary: "...",
  total: "...",
  wrong: "...",
  uncertain: "...",
  untested: "...",
  avgIdentity: "...",
  avgCoverage: "...",
  distribution: "...",
  abnormalCount: "...",
}
```

Use simple bars or ratio rows, not a chart library.

- [ ] **Step 4: Run test to verify it passes**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/ResultsSummary.tsx src/components/ResultsStats.tsx src/components/ResultsWorkbench.tsx src/components/ResultsWorkbench.css src/i18n.ts
git commit -m "feat: add results summary and stats overview"
```

### Task 3: Build sample results table

**Files:**
- Create: `src/components/ResultsTable.tsx`
- Modify: `src/components/ResultsWorkbench.tsx`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Write the failing test**

Reference a missing table component from `ResultsWorkbench.tsx`:

```tsx
import { ResultsTable } from "./ResultsTable";
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm.cmd run build`
Expected: FAIL because `ResultsTable` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create a table component with columns:

```tsx
<table className="results-table">
  <thead>
    <tr>
      <th>Sample</th>
      <th>Clone</th>
      <th>Status</th>
      <th>Reason</th>
      <th>Identity</th>
      <th>Coverage</th>
      <th>Mutations</th>
    </tr>
  </thead>
</table>
```

The table must:
- highlight the selected row
- call `onSelect(sample.id)` on row click
- show `untested` as its own visible badge when `reason === "未测通"`

Add localized headers and empty-copy keys in `src/i18n.ts`.

- [ ] **Step 4: Run test to verify it passes**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/ResultsTable.tsx src/components/ResultsWorkbench.tsx src/i18n.ts
git commit -m "feat: add sample results table"
```

### Task 4: Build expandable sample detail list

**Files:**
- Create: `src/components/SampleDetailsList.tsx`
- Modify: `src/components/ResultsWorkbench.tsx`
- Modify: `src/components/SequenceViewer.tsx`
- Modify: `src/components/SequenceViewer.css`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Write the failing test**

Reference a missing details list component:

```tsx
import { SampleDetailsList } from "./SampleDetailsList";
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm.cmd run build`
Expected: FAIL because `SampleDetailsList` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create expandable sample cards:

```tsx
<details className="sample-detail-card" open={sample.id === selectedId}>
  <summary>{sample.id} — {displayStatus} {sample.reason ?? ""}</summary>
  ...
</details>
```

Inside each detail:
- metrics row
- mutation table
- embedded `SequenceViewer`
- embedded chromatogram block when available

Fix label clipping in `SequenceViewer.css` by increasing restriction-label row height and block top spacing, for example:

```css
.enzyme-row {
  min-height: 56px;
  margin-bottom: 8px;
}

.sequence-block {
  padding-top: 10px;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/SampleDetailsList.tsx src/components/ResultsWorkbench.tsx src/components/SequenceViewer.tsx src/components/SequenceViewer.css src/i18n.ts
git commit -m "feat: add expandable sample details"
```

### Task 5: Integrate workbench into analysis page and subordinate AI rail

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Modify: `src/components/AgentPanel.tsx`
- Modify: `src/components/AgentPanel.css`

- [ ] **Step 1: Write the failing test**

Replace the current center single-sample render path in `src/App.tsx` with a `ResultsWorkbench` render that requires props not yet fully wired:

```tsx
<ResultsWorkbench
  samples={samples}
  selectedId={selectedId}
  onSelect={setSelectedId}
  selectedSample={selectedSample}
  chromatogramData={chromatogramData}
  language={language}
/>
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm.cmd run build`
Expected: FAIL until the props and imports align.

- [ ] **Step 3: Write minimal implementation**

In `src/App.tsx`:
- replace the existing `main-content` single-viewer branch with the workbench component
- keep toolbar and right AI rail intact
- pass richer batch context to `AgentPanel`

In `src/App.css`:
- make `main-content` a workbench container rather than a single viewer viewport
- give the center region more width dominance

In `src/components/AgentPanel.tsx` and `.css`:
- adjust the panel header copy to reflect batch context
- keep AI rail lighter and quieter than the center workbench

- [ ] **Step 4: Run test to verify it passes**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.css src/components/AgentPanel.tsx src/components/AgentPanel.css
git commit -m "feat: integrate results workbench with ai rail"
```

### Task 6: Verify behavior and regression coverage

**Files:**
- Modify only if fixes are required after verification

- [ ] **Step 1: Run frontend verification**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 2: Run targeted dataset import verification**

Run: `npm.cmd run test:dataset-import`
Expected: PASS

- [ ] **Step 3: Run existing Python regression tests**

Run: `python -m pytest tests\test_main.py tests\test_agent_tools.py tests\test_agent_chat.py -q`
Expected: PASS

- [ ] **Step 4: Run desktop smoke**

Run: `npm.cmd run electron:dev`

Manual smoke:
- import `D:\Learning\Biology\projects\BioAgent_Desktop\data\base`
- verify summary cards render
- verify sample table renders
- verify detail expanders open
- verify right-side AI chat remains available
- verify restriction enzyme labels no longer overlap the alignment content

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.css src/i18n.ts src/components src/types/index.ts
git commit -m "feat: rebuild analysis page as results workbench"
```

## Self-Review

- Spec coverage:
  - results-first center layout: Tasks 2-5
  - AI chat on right: Task 5
  - explicit untested bucket: Tasks 1-3
  - clipping fix for alignment labels: Task 4
- Placeholder scan:
  - all tasks include exact files, commands, and minimal code shapes
- Type consistency:
  - `ResultsWorkbench`, `ResultsSummary`, `ResultsStats`, `ResultsTable`, and `SampleDetailsList` are used consistently across tasks
