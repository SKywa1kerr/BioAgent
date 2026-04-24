# PR-Borrowed Workbench Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the current workbench by borrowing the merged PR's useful ideas for normalization, coordinate mapping, sequence detail display, and chromatogram performance without merging unrelated history or migrating architecture.

**Architecture:** Keep `ResultsWorkbench` and `DetailDrawer` as the product surface. Add focused pure-function modules beside the existing workbench components, then have React wrappers consume stable view models. Optimize chromatogram rendering in-place first; leave worker rendering as a follow-up decision after build and smoke verification.

**Tech Stack:** React 18, TypeScript, Vite, Electron, Node `node:test`, existing workbench modules under `src/components/workbench`.

---

## File Structure

- Modify: `src/components/workbench/types.ts`
  - Add `CoordinateMap`, `AlignmentViewModel`, `AlignmentRange`, `AlignmentBaseCell`, and `ChromatogramRenderModel`-adjacent type support if needed by exported pure functions.

- Modify: `src/components/workbench/normalize.ts`
  - Export `normalizeMutation`, `normalizeSample`, `normalizeSamples`, `buildChromatogramData`.
  - Accept snake_case and camelCase aliases while preserving the current `WorkbenchSample` output shape.

- Create: `src/components/workbench/alignmentView.ts`
  - Own coordinate mapping, alignment view model creation, match string creation, CDS/mutation ranges, and AA summary parsing.

- Create: `src/components/workbench/SequenceAlignmentView.tsx`
  - Render only the `AlignmentViewModel`; no raw sample parsing.

- Create: `src/components/workbench/SequenceAlignmentView.css`
  - Scope all styles under `.sequence-alignment-view`.

- Modify: `src/components/workbench/DetailDrawer.tsx`
  - Replace local `chromatogramFrom`.
  - Build and render `SequenceAlignmentView`.
  - Keep existing mutation table, metrics, drawer width, focus, Escape, and chromatogram sections.

- Create: `src/components/workbench/chromatogramRender.ts`
  - Own percentile calculation, visible range model, downsampled trace point construction, nearest-base lookup, and canvas drawing.

- Modify: `src/components/workbench/ChromatogramCanvas.tsx`
  - Use `chromatogramRender.ts` for model and drawing.
  - Keep existing toolbar, zoom/pan, tooltip, and public props.

- Create: `tests/test_workbench_normalize.mjs`
  - Cover aliases and chromatogram construction.

- Create: `tests/test_alignment_view.mjs`
  - Cover coordinate mapping, CDS range, mutation ranges, match strings, and AA parsing.

- Create: `tests/test_chromatogram_render.mjs`
  - Cover visible trace model, downsampling bounds, robust percentile, and nearest-base lookup.

---

## Task 1: Normalize Workbench Sample Inputs

**Files:**
- Modify: `src/components/workbench/normalize.ts`
- Test: `tests/test_workbench_normalize.mjs`

- [ ] **Step 1: Write the failing normalize tests**

Create `tests/test_workbench_normalize.mjs`:

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  buildChromatogramData,
  normalizeMutation,
  normalizeSample,
  normalizeSamples,
} from "../src/components/workbench/normalize.js";

test("normalizeMutation accepts snake_case and camelCase aliases", () => {
  assert.deepEqual(
    normalizeMutation({ position: 7, ref_base: "A", qry_base: "G", type: "substitution", effect: "synonymous" }),
    { position: 7, refBase: "A", queryBase: "G", type: "substitution", effect: "synonymous" },
  );
  assert.deepEqual(
    normalizeMutation({ refPos: 9, refBase: "T", queryBase: "C" }),
    { position: 9, refBase: "T", queryBase: "C", type: undefined, effect: undefined },
  );
});

test("normalizeSample preserves current workbench shape while accepting camelCase aliases", () => {
  const sample = normalizeSample({
    sid: "C366-3",
    alignedRefG: "ATG-C",
    alignedQueryG: "ATGTC",
    querySequence: "ATGTC",
    cdsStart: 1,
    cdsEnd: 5,
    tracesA: [0, 10],
    tracesT: [0, 8],
    tracesG: [0, 7],
    tracesC: [0, 5],
    baseLocations: [0, 1],
    mixedPeaks: [1],
    mutations: [{ refPos: 4, refBase: "-", queryBase: "T", type: "insertion" }],
  }, 0, "en");

  assert.equal(sample.id, "C366-3");
  assert.equal(sample.aligned_ref_g, "ATG-C");
  assert.equal(sample.aligned_query_g, "ATGTC");
  assert.equal(sample.query_sequence, "ATGTC");
  assert.equal(sample.cds_start, 1);
  assert.equal(sample.cds_end, 5);
  assert.deepEqual(sample.traces_a, [0, 10]);
  assert.deepEqual(sample.base_locations, [0, 1]);
  assert.deepEqual(sample.mutations, [{ position: 4, refBase: "-", queryBase: "T", type: "insertion", effect: undefined }]);
});

test("normalizeSamples reads detail samples when direct samples are absent", () => {
  const samples = normalizeSamples({ detail: { samples: [{ id: "S1", identity: 1, coverage: 1 }] } }, "en");
  assert.equal(samples.length, 1);
  assert.equal(samples[0].id, "S1");
  assert.equal(samples[0].status, "ok");
});

test("buildChromatogramData returns null when required traces or query sequence are missing", () => {
  assert.equal(buildChromatogramData({ id: "S1", query_sequence: "AT" }), null);
});

test("buildChromatogramData creates current ChromatogramData shape", () => {
  const chrom = buildChromatogramData({
    id: "S1",
    query_sequence: "AT",
    traces_a: [1, 2],
    traces_t: [3, 4],
    traces_g: [5, 6],
    traces_c: [7, 8],
    quality: [40, 35],
    base_locations: [0, 1],
    mixed_peaks: [1],
  });

  assert.deepEqual(chrom, {
    traces: { A: [1, 2], T: [3, 4], G: [5, 6], C: [7, 8] },
    quality: [40, 35],
    baseCalls: "AT",
    base_locations: [0, 1],
    mixed_peaks: [1],
  });
});
```

- [ ] **Step 2: Run the failing normalize tests**

Run: `node tests/test_workbench_normalize.mjs`

Expected: FAIL because `buildChromatogramData`, `normalizeMutation`, and `normalizeSample` are not exported yet.

- [ ] **Step 3: Export alias helpers and sample normalizers**

Modify `src/components/workbench/normalize.ts` to this shape, preserving `deriveStatus` and `deriveReason`:

```ts
import type { ChromatogramData, WorkbenchMutation, WorkbenchSample } from "./types";
import { t, type AppLanguage } from "../../i18n";

function firstDefined<T>(...values: T[]): T | undefined {
  return values.find((value) => value !== undefined && value !== null);
}

function toNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function toArray<T = unknown>(value: unknown): T[] | undefined {
  return Array.isArray(value) ? value as T[] : undefined;
}

export function normalizeMutation(item: any): WorkbenchMutation {
  return {
    position: firstDefined(item?.position, item?.ref_pos, item?.refPos),
    refBase: firstDefined(item?.refBase, item?.ref_base),
    queryBase: firstDefined(item?.queryBase, item?.qry_base, item?.query_base),
    type: item?.type,
    effect: item?.effect,
  };
}

function deriveStatus(item: any, mutationCount: number): "ok" | "wrong" | "uncertain" {
  const identity = typeof item?.identity === "number" ? item.identity : 0;
  const coverage = typeof item?.cds_coverage === "number" ? item.cds_coverage : (typeof item?.coverage === "number" ? item.coverage : 0);
  if (item?.frameshift) return "wrong";
  if (mutationCount > 0) return "wrong";
  if (identity >= 0.99 && coverage >= 0.8) return "ok";
  return "uncertain";
}

function deriveReason(item: any, mutationCount: number, language: AppLanguage): string {
  const identity = typeof item?.identity === "number" ? item.identity : 0;
  const coverage = typeof item?.cds_coverage === "number" ? item.cds_coverage : (typeof item?.coverage === "number" ? item.coverage : 0);
  if (item?.error) return String(item.error);
  if (item?.frameshift) return t(language, "analysis.reason.frameshift");
  if (mutationCount > 0) return t(language, "analysis.reason.detectedMut", { count: mutationCount });
  if (identity >= 0.99 && coverage >= 0.8) return t(language, "analysis.reason.highQuality");
  return t(language, "analysis.reason.review");
}

export function normalizeSample(item: any, idx: number, language: AppLanguage): WorkbenchSample {
  const id = firstDefined(item?.id, item?.sid, item?.name, `sample-${idx + 1}`);
  const mutations = Array.isArray(item?.mutations) ? item.mutations.map(normalizeMutation) : [];
  const mutationCount =
    (item?.sub_count ?? item?.sub ?? 0) +
    (item?.ins_count ?? item?.ins ?? 0) +
    (item?.del_count ?? item?.dele ?? item?.del ?? 0) ||
    mutations.length;
  const status = (item?.status as "ok" | "wrong" | "uncertain" | undefined) || deriveStatus(item, mutationCount);
  const reason = item?.reason || item?.review_reason || item?.reviewReason || item?.llm_reason || item?.llmReason || item?.auto_reason || item?.autoReason || deriveReason(item, mutationCount, language);

  return {
    id: String(id),
    name: item?.name,
    clone: item?.clone,
    status,
    reason,
    review_reason: firstDefined(item?.review_reason, item?.reviewReason),
    llm_reason: firstDefined(item?.llm_reason, item?.llmReason),
    auto_reason: firstDefined(item?.auto_reason, item?.autoReason),
    error: item?.error,
    identity: item?.identity,
    coverage: item?.coverage,
    cds_coverage: firstDefined(item?.cds_coverage, item?.cdsCoverage),
    sub_count: firstDefined(item?.sub_count, item?.subCount),
    ins_count: firstDefined(item?.ins_count, item?.insCount),
    del_count: firstDefined(item?.del_count, item?.delCount),
    sub: item?.sub,
    ins: item?.ins,
    dele: firstDefined(item?.dele, item?.del),
    aa_changes: firstDefined(item?.aa_changes, item?.aaChanges),
    aa_changes_n: firstDefined(item?.aa_changes_n, item?.aaChangesN),
    avg_qry_quality: firstDefined(item?.avg_qry_quality, item?.avgQryQuality),
    avg_quality: firstDefined(item?.avg_quality, item?.avgQuality),
    orientation: item?.orientation,
    frameshift: item?.frameshift,
    mutations,
    ref_sequence: firstDefined(item?.ref_sequence, item?.refSequence),
    query_sequence: firstDefined(item?.query_sequence, item?.querySequence),
    aligned_ref_g: firstDefined(item?.aligned_ref_g, item?.alignedRefG),
    aligned_query_g: firstDefined(item?.aligned_query_g, item?.alignedQueryG),
    aligned_query: firstDefined(item?.aligned_query, item?.alignedQuery),
    matches: item?.matches,
    cds_start: toNumber(firstDefined(item?.cds_start, item?.cdsStart)),
    cds_end: toNumber(firstDefined(item?.cds_end, item?.cdsEnd)),
    traces_a: toArray<number>(firstDefined(item?.traces_a, item?.tracesA)),
    traces_t: toArray<number>(firstDefined(item?.traces_t, item?.tracesT)),
    traces_g: toArray<number>(firstDefined(item?.traces_g, item?.tracesG)),
    traces_c: toArray<number>(firstDefined(item?.traces_c, item?.tracesC)),
    quality: toArray<number>(item?.quality),
    base_locations: toArray<number>(firstDefined(item?.base_locations, item?.baseLocations)),
    mixed_peaks: toArray<number>(firstDefined(item?.mixed_peaks, item?.mixedPeaks)),
    bucket: item?.bucket,
  };
}

export function buildChromatogramData(sample: WorkbenchSample): ChromatogramData | null {
  if (!sample.traces_a || !sample.traces_t || !sample.traces_g || !sample.traces_c || !sample.query_sequence) {
    return null;
  }
  return {
    traces: {
      A: sample.traces_a,
      T: sample.traces_t,
      G: sample.traces_g,
      C: sample.traces_c,
    },
    quality: sample.quality || [],
    baseCalls: sample.query_sequence,
    base_locations: sample.base_locations || [],
    mixed_peaks: sample.mixed_peaks || [],
  };
}

export function normalizeSamples(result: any, language: AppLanguage): WorkbenchSample[] {
  const direct = Array.isArray(result?.samples) ? result.samples : [];
  const detailSamples = Array.isArray(result?.detail?.samples) ? result.detail.samples : [];
  const payload = direct.length > 0 ? direct : detailSamples;

  return payload
    .filter((item: any) => item && typeof item === "object")
    .map((item: any, idx: number) => normalizeSample(item, idx, language));
}
```

- [ ] **Step 4: Run normalize tests**

Run: `node tests/test_workbench_normalize.mjs`

Expected: PASS with 5 passing tests.

- [ ] **Step 5: Run current JS smoke tests**

Run: `npm run test:js`

Expected: PASS. If this fails because generated `.js` files do not exist for new `.ts` modules, run `npm run build` once, then rerun `npm run test:js`.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/components/workbench/normalize.ts tests/test_workbench_normalize.mjs
git commit -m "feat(workbench): normalize sample aliases"
```

---

## Task 2: Add Alignment View Model Pure Functions

**Files:**
- Modify: `src/components/workbench/types.ts`
- Create: `src/components/workbench/alignmentView.ts`
- Test: `tests/test_alignment_view.mjs`

- [ ] **Step 1: Write the failing alignment view tests**

Create `tests/test_alignment_view.mjs`:

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  buildAlignmentViewModel,
  buildCoordinateMap,
  parseAaChanges,
} from "../src/components/workbench/alignmentView.js";

test("buildCoordinateMap maps gapped ref and query coordinates", () => {
  const map = buildCoordinateMap("ATG-C", "A-GTC");
  assert.deepEqual(map.refToGapped, [0, 1, 2, 4]);
  assert.deepEqual(map.queryToGapped, [0, 2, 3, 4]);
  assert.deepEqual(map.gappedToRef, [0, 1, 2, null, 3]);
  assert.deepEqual(map.gappedToQuery, [0, null, 1, 2, 3]);
});

test("buildAlignmentViewModel maps one-based CDS and mutation positions into gapped ranges", () => {
  const view = buildAlignmentViewModel({
    id: "S1",
    ref_sequence: "ATGC",
    query_sequence: "AGTC",
    aligned_ref_g: "ATG-C",
    aligned_query_g: "A-GTC",
    matches: [true, false, true, false, true],
    cds_start: 2,
    cds_end: 4,
    mutations: [{ position: 4, refBase: "C", queryBase: "T", type: "substitution" }],
    aa_changes: ["S2L"],
  });

  assert.equal(view.refLine, "ATG-C");
  assert.equal(view.queryLine, "A-GTC");
  assert.equal(view.matchLine, "| | |");
  assert.deepEqual(view.cdsRange, { start: 1, end: 5 });
  assert.deepEqual(view.mutationRanges, [{ start: 4, end: 5, label: "C>T", type: "substitution", effect: undefined }]);
  assert.deepEqual(view.aaChanges, ["S2L"]);
});

test("buildAlignmentViewModel returns null without usable alignment strings", () => {
  assert.equal(buildAlignmentViewModel({ id: "S1" }), null);
});

test("parseAaChanges accepts arrays, JSON strings, plain strings, and blanks", () => {
  assert.deepEqual(parseAaChanges(["S2L", ""]), ["S2L"]);
  assert.deepEqual(parseAaChanges('["S2L","K4M"]'), ["S2L", "K4M"]);
  assert.deepEqual(parseAaChanges("S2L K4M"), ["S2L K4M"]);
  assert.deepEqual(parseAaChanges(""), []);
});
```

- [ ] **Step 2: Run the failing alignment tests**

Run: `node tests/test_alignment_view.mjs`

Expected: FAIL because `alignmentView.js` does not exist.

- [ ] **Step 3: Add alignment view types**

Append to `src/components/workbench/types.ts`:

```ts
export interface CoordinateMap {
  refToGapped: number[];
  gappedToRef: Array<number | null>;
  gappedToQuery: Array<number | null>;
  queryToGapped: number[];
}

export interface AlignmentRange {
  start: number;
  end: number;
}

export interface MutationRange extends AlignmentRange {
  label: string;
  type?: string;
  effect?: string;
}

export interface AlignmentViewModel {
  sampleId: string;
  refLine: string;
  queryLine: string;
  matchLine: string;
  positionLine: string;
  tickLine: string;
  coordinateMap: CoordinateMap;
  cdsRange: AlignmentRange | null;
  mutationRanges: MutationRange[];
  aaChanges: string[];
}
```

- [ ] **Step 4: Implement `alignmentView.ts`**

Create `src/components/workbench/alignmentView.ts`:

```ts
import type {
  AlignmentRange,
  AlignmentViewModel,
  CoordinateMap,
  MutationRange,
  WorkbenchSample,
} from "./types";

export function buildCoordinateMap(refGapped: string, queryGapped: string): CoordinateMap {
  const length = Math.max(refGapped.length, queryGapped.length);
  const refToGapped: number[] = [];
  const queryToGapped: number[] = [];
  const gappedToRef: Array<number | null> = new Array(length).fill(null);
  const gappedToQuery: Array<number | null> = new Array(length).fill(null);
  let refPos = 0;
  let queryPos = 0;

  for (let gappedPos = 0; gappedPos < length; gappedPos += 1) {
    const refBase = refGapped[gappedPos] || "-";
    const queryBase = queryGapped[gappedPos] || "-";
    if (refBase !== "-") {
      refToGapped[refPos] = gappedPos;
      gappedToRef[gappedPos] = refPos;
      refPos += 1;
    }
    if (queryBase !== "-") {
      queryToGapped[queryPos] = gappedPos;
      gappedToQuery[gappedPos] = queryPos;
      queryPos += 1;
    }
  }

  return { refToGapped, gappedToRef, gappedToQuery, queryToGapped };
}

export function parseAaChanges(value: WorkbenchSample["aa_changes"]): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
  }
  if (typeof value !== "string") return [];
  const trimmed = value.trim();
  if (!trimmed) return [];
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
    }
  } catch {
    return [trimmed];
  }
  return [trimmed];
}

function buildMatchLine(refLine: string, queryLine: string, matches?: boolean[]): string {
  const length = Math.max(refLine.length, queryLine.length);
  if (Array.isArray(matches) && matches.length >= length) {
    return matches.slice(0, length).map((match) => (match ? "|" : " ")).join("");
  }
  return Array.from({ length }, (_, idx) => {
    const refBase = refLine[idx] || "-";
    const queryBase = queryLine[idx] || "-";
    return refBase !== "-" && refBase === queryBase ? "|" : " ";
  }).join("");
}

function buildPositionLine(length: number): string {
  return Array.from({ length }, (_, idx) => {
    const pos = idx + 1;
    if (pos % 10 !== 0) return " ";
    return String(pos).slice(-1);
  }).join("");
}

function buildTickLine(length: number): string {
  return Array.from({ length }, (_, idx) => ((idx + 1) % 10 === 0 ? "+" : ".")).join("");
}

function mapCdsRange(sample: WorkbenchSample, coordinateMap: CoordinateMap): AlignmentRange | null {
  if (typeof sample.cds_start !== "number" || typeof sample.cds_end !== "number") return null;
  const start = coordinateMap.refToGapped[sample.cds_start - 1];
  const endBase = coordinateMap.refToGapped[sample.cds_end - 1];
  if (start === undefined || endBase === undefined) return null;
  return { start, end: endBase + 1 };
}

function mutationLabel(refBase?: string, queryBase?: string): string {
  if (!refBase && !queryBase) return "mutation";
  return `${refBase || "-"}>${queryBase || "-"}`;
}

function mapMutationRanges(sample: WorkbenchSample, coordinateMap: CoordinateMap): MutationRange[] {
  const mutations = Array.isArray(sample.mutations) ? sample.mutations : [];
  return mutations.flatMap((mutation) => {
    if (typeof mutation.position !== "number") return [];
    const start = coordinateMap.refToGapped[mutation.position - 1];
    if (start === undefined) return [];
    return [{
      start,
      end: start + 1,
      label: mutationLabel(mutation.refBase, mutation.queryBase),
      type: mutation.type,
      effect: mutation.effect,
    }];
  });
}

export function buildAlignmentViewModel(sample: WorkbenchSample): AlignmentViewModel | null {
  const refLine = sample.aligned_ref_g || sample.ref_sequence || "";
  const queryLine = sample.aligned_query_g || sample.aligned_query || sample.query_sequence || "";
  if (!refLine || !queryLine) return null;
  const length = Math.max(refLine.length, queryLine.length);
  const paddedRef = refLine.padEnd(length, "-");
  const paddedQuery = queryLine.padEnd(length, "-");
  const coordinateMap = buildCoordinateMap(paddedRef, paddedQuery);

  return {
    sampleId: sample.id,
    refLine: paddedRef,
    queryLine: paddedQuery,
    matchLine: buildMatchLine(paddedRef, paddedQuery, sample.matches),
    positionLine: buildPositionLine(length),
    tickLine: buildTickLine(length),
    coordinateMap,
    cdsRange: mapCdsRange(sample, coordinateMap),
    mutationRanges: mapMutationRanges(sample, coordinateMap),
    aaChanges: parseAaChanges(sample.aa_changes),
  };
}
```

- [ ] **Step 5: Run alignment tests**

Run: `npm run build`

Expected: PASS and emit updated `.js` files for tests.

Run: `node tests/test_alignment_view.mjs`

Expected: PASS with 4 passing tests.

- [ ] **Step 6: Run JS suite**

Run: `npm run test:js`

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/components/workbench/types.ts src/components/workbench/alignmentView.ts tests/test_alignment_view.mjs
git commit -m "feat(workbench): add alignment view model"
```

---

## Task 3: Render Sequence Alignment in Detail Drawer

**Files:**
- Create: `src/components/workbench/SequenceAlignmentView.tsx`
- Create: `src/components/workbench/SequenceAlignmentView.css`
- Modify: `src/components/workbench/DetailDrawer.tsx`
- Test: existing `npm run build`

- [ ] **Step 1: Create the view component**

Create `src/components/workbench/SequenceAlignmentView.tsx`:

```tsx
import type { AlignmentViewModel } from "./types";
import "./SequenceAlignmentView.css";

interface Props {
  view: AlignmentViewModel;
}

function rangeStyle(start: number, end: number) {
  const width = Math.max(1, end - start);
  return {
    left: `${start}ch`,
    width: `${width}ch`,
  };
}

function renderMutationMarkers(view: AlignmentViewModel) {
  return view.mutationRanges.map((range, idx) => (
    <span
      key={`${range.start}-${idx}`}
      className={`sequence-mutation-marker${range.effect === "synonymous" ? " is-synonymous" : ""}`}
      style={rangeStyle(range.start, range.end)}
      title={`${range.label}${range.effect ? ` (${range.effect})` : ""}`}
    />
  ));
}

export function SequenceAlignmentView({ view }: Props) {
  return (
    <div className="sequence-alignment-view" aria-label={`Alignment ${view.sampleId}`}>
      {view.aaChanges.length > 0 ? (
        <div className="sequence-aa-summary" aria-label="AA changes">
          {view.aaChanges.map((change) => (
            <span key={change} className="sequence-aa-pill">{change}</span>
          ))}
        </div>
      ) : null}

      <div className="sequence-scroll">
        <div className="sequence-line-stack" style={{ width: `${view.refLine.length}ch` }}>
          {view.cdsRange ? (
            <span className="sequence-cds-band" style={rangeStyle(view.cdsRange.start, view.cdsRange.end)} />
          ) : null}
          {renderMutationMarkers(view)}

          <div className="sequence-row">
            <span className="sequence-gutter">REF</span>
            <pre>{view.refLine}</pre>
          </div>
          <div className="sequence-row sequence-match-row">
            <span className="sequence-gutter">MATCH</span>
            <pre>{view.matchLine}</pre>
          </div>
          <div className="sequence-row">
            <span className="sequence-gutter">QRY</span>
            <pre>{view.queryLine}</pre>
          </div>
          <div className="sequence-row sequence-position-row">
            <span className="sequence-gutter">POS</span>
            <pre>{view.positionLine}</pre>
          </div>
          <div className="sequence-row sequence-tick-row">
            <span className="sequence-gutter" />
            <pre>{view.tickLine}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add scoped styles**

Create `src/components/workbench/SequenceAlignmentView.css`:

```css
.sequence-alignment-view {
  display: grid;
  gap: 10px;
}

.sequence-aa-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.sequence-aa-pill {
  border: 1px solid var(--status-wrong-border, rgba(248, 113, 113, 0.4));
  border-radius: 6px;
  color: var(--status-wrong-text, #b91c1c);
  background: var(--status-wrong-bg, rgba(248, 113, 113, 0.1));
  font: 600 11px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
  padding: 2px 6px;
}

.sequence-scroll {
  overflow-x: auto;
  border: 1px solid var(--panel-border, rgba(148, 163, 184, 0.24));
  border-radius: 8px;
  background: var(--panel-bg-subtle, rgba(15, 23, 42, 0.03));
}

.sequence-line-stack {
  position: relative;
  min-width: 100%;
  padding: 10px 12px 10px 58px;
  font: 12px/1.65 ui-monospace, SFMono-Regular, Consolas, monospace;
}

.sequence-row {
  position: relative;
  display: flex;
  min-height: 20px;
}

.sequence-row pre {
  position: relative;
  z-index: 1;
  margin: 0;
  white-space: pre;
  letter-spacing: 0;
}

.sequence-gutter {
  position: sticky;
  left: 0;
  z-index: 2;
  width: 46px;
  margin-left: -58px;
  padding-right: 8px;
  color: var(--muted-text, #64748b);
  background: var(--panel-bg-subtle, #f8fafc);
  font-size: 10px;
  text-align: right;
}

.sequence-match-row,
.sequence-position-row,
.sequence-tick-row {
  color: var(--muted-text, #64748b);
}

.sequence-cds-band,
.sequence-mutation-marker {
  position: absolute;
  z-index: 0;
  pointer-events: auto;
}

.sequence-cds-band {
  top: 10px;
  height: 60px;
  background: rgba(59, 130, 246, 0.12);
  border-left: 1px solid rgba(59, 130, 246, 0.35);
  border-right: 1px solid rgba(59, 130, 246, 0.35);
}

.sequence-mutation-marker {
  top: 10px;
  height: 60px;
  background: rgba(239, 68, 68, 0.16);
  border-bottom: 2px solid rgba(239, 68, 68, 0.85);
}

.sequence-mutation-marker.is-synonymous {
  background: rgba(234, 179, 8, 0.16);
  border-bottom-color: rgba(202, 138, 4, 0.9);
}
```

- [ ] **Step 3: Wire the component into `DetailDrawer`**

Modify imports in `src/components/workbench/DetailDrawer.tsx`:

```tsx
import { buildAlignmentViewModel } from "./alignmentView";
import { buildChromatogramData } from "./normalize";
import { SequenceAlignmentView } from "./SequenceAlignmentView";
```

Remove the local `chromatogramFrom` function entirely.

After existing local values:

```tsx
const chrom = buildChromatogramData(sample);
const alignmentView = buildAlignmentViewModel(sample);
```

Replace the alignment section body with:

```tsx
<section className="detail-drawer-section">
  <h4>{t(language, "table.alignment")}</h4>
  {alignmentView ? (
    <SequenceAlignmentView view={alignmentView} />
  ) : (
    <div className="detail-drawer-empty">{t(language, "table.noAlignment")}</div>
  )}
</section>
```

If `table.noAlignment` is not present in `src/i18n.ts`, add:

```ts
"table.noAlignment": "没有可显示的比对结果。",
```

to the Chinese dictionary and:

```ts
"table.noAlignment": "No alignment is available.",
```

to the English dictionary.

- [ ] **Step 4: Run build**

Run: `npm run build`

Expected: PASS. If it fails because `t(language, "table.noAlignment")` is not accepted by the i18n key type, add the key to the same object shape used by existing `table.noChromatogram`.

- [ ] **Step 5: Run JS tests**

Run: `npm run test:js`

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/components/workbench/SequenceAlignmentView.tsx src/components/workbench/SequenceAlignmentView.css src/components/workbench/DetailDrawer.tsx src/i18n.ts
git commit -m "feat(workbench): show aligned sequence details"
```

---

## Task 4: Extract Chromatogram Render Model and Tests

**Files:**
- Create: `src/components/workbench/chromatogramRender.ts`
- Test: `tests/test_chromatogram_render.mjs`

- [ ] **Step 1: Write failing chromatogram render tests**

Create `tests/test_chromatogram_render.mjs`:

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  buildChromatogramRenderModel,
  findNearestBaseIndex,
  percentile,
} from "../src/components/workbench/chromatogramRender.js";

const data = {
  traces: {
    A: [0, 10, 20, 500, 20, 10, 0],
    T: [0, 8, 16, 24, 16, 8, 0],
    G: [0, 5, 10, 15, 10, 5, 0],
    C: [0, 2, 4, 6, 4, 2, 0],
  },
  quality: [40, 35, 20],
  baseCalls: "ATG",
  base_locations: [1, 3, 5],
  mixed_peaks: [1],
};

test("percentile handles empty input and robust rank", () => {
  assert.equal(percentile([], 0.985), 0);
  assert.equal(percentile([1, 2, 100], 0.5), 2);
});

test("buildChromatogramRenderModel clamps visible base range and preserves base labels", () => {
  const model = buildChromatogramRenderModel(data, {
    startPosition: 1,
    endPosition: 3,
    width: 300,
    height: 120,
    maxPointsPerPixel: 1,
  });

  assert.equal(model.visibleStartTrace, 0);
  assert.equal(model.visibleEndTrace, 7);
  assert.equal(model.baseLabels.length, 3);
  assert.deepEqual(model.baseLabels.map((b) => b.base), ["A", "T", "G"]);
  assert.equal(model.maxVal > 0, true);
});

test("buildChromatogramRenderModel downsamples traces when trace range exceeds canvas budget", () => {
  const longTrace = Array.from({ length: 1000 }, (_, i) => i % 50);
  const model = buildChromatogramRenderModel({
    traces: { A: longTrace, T: longTrace, G: longTrace, C: longTrace },
    quality: [],
    baseCalls: "AT",
    base_locations: [10, 990],
    mixed_peaks: [],
  }, {
    startPosition: 1,
    endPosition: 2,
    width: 100,
    height: 120,
    maxPointsPerPixel: 1,
  });

  assert.equal(model.step > 1, true);
  assert.equal(model.tracePoints.A.length <= 101, true);
});

test("findNearestBaseIndex returns nearest base within visible labels", () => {
  const model = buildChromatogramRenderModel(data, {
    startPosition: 1,
    endPosition: 3,
    width: 300,
    height: 120,
  });
  assert.equal(findNearestBaseIndex(model, model.baseLabels[1].x + 1), 1);
});
```

- [ ] **Step 2: Run failing chromatogram tests**

Run: `node tests/test_chromatogram_render.mjs`

Expected: FAIL because `chromatogramRender.js` does not exist.

- [ ] **Step 3: Implement render model functions**

Create `src/components/workbench/chromatogramRender.ts`:

```ts
import type { ChromatogramData } from "./types";

type Base = "A" | "T" | "G" | "C";
const BASES: Base[] = ["A", "T", "G", "C"];

export interface TracePoint {
  x: number;
  y: number;
  value: number;
  traceIndex: number;
}

export interface BaseLabel {
  baseIndex: number;
  traceIndex: number;
  x: number;
  base: string;
  quality?: number;
  mixed: boolean;
}

export interface ChromatogramRenderModel {
  width: number;
  height: number;
  padding: number;
  visibleStartTrace: number;
  visibleEndTrace: number;
  maxVal: number;
  step: number;
  tracePoints: Record<Base, TracePoint[]>;
  baseLabels: BaseLabel[];
}

interface BuildOptions {
  startPosition: number;
  endPosition: number;
  width: number;
  height: number;
  padding?: number;
  maxPointsPerPixel?: number;
}

export function percentile(values: number[], ratio: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * ratio)));
  return sorted[idx];
}

function collectVisibleValues(data: ChromatogramData, start: number, end: number, step: number): number[] {
  const values: number[] = [];
  for (const base of BASES) {
    const trace = data.traces[base] || [];
    for (let i = start; i < end; i += step) {
      const value = trace[i] || 0;
      if (value > 0) values.push(value);
    }
  }
  return values;
}

export function buildChromatogramRenderModel(data: ChromatogramData, options: BuildOptions): ChromatogramRenderModel {
  const padding = options.padding ?? 24;
  const startBaseIdx = Math.max(0, options.startPosition - 1);
  const endBaseIdx = Math.min(data.baseCalls.length, options.endPosition);
  const traces = data.traces;
  const traceLength = traces.A?.length || 0;
  const startTraceIdx = data.base_locations[startBaseIdx] ?? 0;
  const endTraceIdx = data.base_locations[endBaseIdx - 1] ?? Math.max(0, traceLength - 1);
  const tracePadding = 24;
  const visibleStartTrace = Math.max(0, startTraceIdx - tracePadding);
  const visibleEndTrace = Math.min(traceLength, endTraceIdx + tracePadding + 1);
  const traceRange = Math.max(1, visibleEndTrace - visibleStartTrace);
  const budget = Math.max(1, Math.floor(options.width * (options.maxPointsPerPixel ?? 1.5)));
  const step = Math.max(1, Math.ceil(traceRange / budget));
  const visibleValues = collectVisibleValues(data, visibleStartTrace, visibleEndTrace, Math.max(1, step * 2));
  const robustTop = percentile(visibleValues, 0.985);
  const maxVal = robustTop > 0 ? robustTop : 1;
  const xScale = (options.width - 2 * padding) / traceRange;
  const yScale = (options.height - 2 * padding) / (maxVal * 1.08);

  const tracePoints = BASES.reduce((acc, base) => {
    const trace = traces[base] || [];
    acc[base] = [];
    for (let i = visibleStartTrace; i < visibleEndTrace; i += step) {
      const clamped = Math.min(trace[i] || 0, maxVal * 1.08);
      acc[base].push({
        traceIndex: i,
        value: trace[i] || 0,
        x: padding + (i - visibleStartTrace) * xScale,
        y: options.height - padding - clamped * yScale,
      });
    }
    return acc;
  }, {} as Record<Base, TracePoint[]>);

  const mixedSet = new Set(data.mixed_peaks || []);
  const baseLabels: BaseLabel[] = [];
  for (let i = startBaseIdx; i < endBaseIdx; i += 1) {
    const traceIndex = data.base_locations[i];
    if (traceIndex === undefined || traceIndex < visibleStartTrace || traceIndex > visibleEndTrace) continue;
    baseLabels.push({
      baseIndex: i,
      traceIndex,
      x: padding + (traceIndex - visibleStartTrace) * xScale,
      base: data.baseCalls[i] || "-",
      quality: data.quality?.[i],
      mixed: mixedSet.has(i),
    });
  }

  return {
    width: options.width,
    height: options.height,
    padding,
    visibleStartTrace,
    visibleEndTrace,
    maxVal,
    step,
    tracePoints,
    baseLabels,
  };
}

export function findNearestBaseIndex(model: ChromatogramRenderModel, x: number): number {
  let closest = -1;
  let best = Number.POSITIVE_INFINITY;
  for (const label of model.baseLabels) {
    const distance = Math.abs(label.x - x);
    if (distance < best) {
      best = distance;
      closest = label.baseIndex;
    }
  }
  return closest;
}
```

- [ ] **Step 4: Run build and render tests**

Run: `npm run build`

Expected: PASS.

Run: `node tests/test_chromatogram_render.mjs`

Expected: PASS with 4 passing tests.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/components/workbench/chromatogramRender.ts tests/test_chromatogram_render.mjs
git commit -m "feat(workbench): add chromatogram render model"
```

---

## Task 5: Refactor ChromatogramCanvas to Use Render Model

**Files:**
- Modify: `src/components/workbench/ChromatogramCanvas.tsx`
- Modify: `src/components/workbench/chromatogramRender.ts`
- Test: `tests/test_chromatogram_render.mjs`

- [ ] **Step 1: Add draw helper to `chromatogramRender.ts`**

Append this function:

```ts
export function drawChromatogram(
  ctx: CanvasRenderingContext2D,
  model: ChromatogramRenderModel,
  options: {
    dark: boolean;
    mutations?: Array<{ position?: number }>;
  },
) {
  const traceColors: Record<Base, string> = {
    A: options.dark ? "#4ade80" : "#16a34a",
    T: options.dark ? "#f87171" : "#dc2626",
    G: options.dark ? "#fbbf24" : "#b45309",
    C: options.dark ? "#60a5fa" : "#2563eb",
  };
  const background = options.dark ? "#0f172a" : "#f8fbff";
  const gridColor = options.dark ? "rgba(148, 163, 184, 0.12)" : "rgba(148, 163, 184, 0.22)";
  const labelColor = options.dark ? "#dbe7f5" : "#334155";

  ctx.fillStyle = background;
  ctx.fillRect(0, 0, model.width, model.height);

  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let row = 1; row <= 4; row += 1) {
    const y = model.padding + ((model.height - 2 * model.padding) / 4) * row;
    ctx.beginPath();
    ctx.moveTo(model.padding, y);
    ctx.lineTo(model.width - model.padding, y);
    ctx.stroke();
  }

  for (const base of BASES) {
    const points = model.tracePoints[base];
    ctx.strokeStyle = traceColors[base];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    points.forEach((point, idx) => {
      if (idx === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }

  ctx.font = '11px "Consolas", monospace';
  ctx.textAlign = "center";
  for (const label of model.baseLabels) {
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(label.x, model.height - model.padding - 6);
    ctx.lineTo(label.x, model.height - model.padding + 8);
    ctx.stroke();

    ctx.fillStyle = traceColors[label.base as Base] || labelColor;
    ctx.fillText(label.base, label.x, model.height - 8);

    if (label.mixed) {
      ctx.strokeStyle = options.dark ? "#fde047" : "#ca8a04";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(label.x, model.height - 19, 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  const mutationPositions = new Set((options.mutations || [])
    .map((mutation) => typeof mutation.position === "number" ? mutation.position - 1 : null)
    .filter((value): value is number => typeof value === "number"));
  for (const label of model.baseLabels) {
    if (!mutationPositions.has(label.baseIndex)) continue;
    ctx.fillStyle = options.dark ? "#f87171" : "#dc2626";
    ctx.beginPath();
    ctx.moveTo(label.x - 5, model.padding + 4);
    ctx.lineTo(label.x + 5, model.padding + 4);
    ctx.lineTo(label.x, model.padding + 12);
    ctx.closePath();
    ctx.fill();
  }
}
```

- [ ] **Step 2: Refactor `ChromatogramCanvas.tsx` drawing effect**

Replace the local percentile function import and draw loops with:

```tsx
import {
  buildChromatogramRenderModel,
  drawChromatogram,
  findNearestBaseIndex,
} from "./chromatogramRender";
```

Inside `useEffect`, compute:

```tsx
const model = buildChromatogramRenderModel(data, {
  startPosition: effectiveStart,
  endPosition: effectiveEnd,
  width,
  height,
});
drawChromatogram(ctx, model, { dark: isDarkTheme, mutations });
```

Remove the old local `percentile` function and repeated drawing loops from `ChromatogramCanvas.tsx`.

- [ ] **Step 3: Refactor tooltip nearest-base lookup**

In `handleMouseMove`, replace the manual scan over `data.base_locations` with:

```tsx
const model = buildChromatogramRenderModel(data, {
  startPosition: effectiveStart,
  endPosition: effectiveEnd,
  width: canvas.width,
  height: canvas.height,
});
const closestBaseIdx = findNearestBaseIndex(model, x);
if (closestBaseIdx < 0) return;
```

Keep tooltip text:

```tsx
const quality = data.quality?.[closestBaseIdx];
const base = data.baseCalls[closestBaseIdx] || "-";
const isMixed = data.mixed_peaks.includes(closestBaseIdx);
```

- [ ] **Step 4: Run chromatogram tests and build**

Run: `npm run build`

Expected: PASS.

Run: `node tests/test_chromatogram_render.mjs`

Expected: PASS.

Run: `npm run test:js`

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/components/workbench/ChromatogramCanvas.tsx src/components/workbench/chromatogramRender.ts tests/test_chromatogram_render.mjs
git commit -m "refactor(workbench): split chromatogram rendering"
```

---

## Task 6: Full Verification and Worker Decision

**Files:**
- No required code changes.
- Optional future plan only if worker is still needed.

- [ ] **Step 1: Run complete JS test suite**

Run: `npm run test:js`

Expected: PASS.

- [ ] **Step 2: Run build**

Run: `npm run build`

Expected: PASS.

- [ ] **Step 3: Run full test command**

Run: `npm run test`

Expected: PASS. If Python tests fail from unrelated environment dependencies, capture the exact failing command and output in the final report; do not claim full test pass.

- [ ] **Step 4: Manual drawer smoke test**

Run: `npm run electron:dev`

Manual checks:

- Start or initialize the app with the usual settings.
- Run an existing dataset analysis such as `分析 pro 数据集`.
- Open at least one sample drawer.
- Confirm the drawer shows metrics, AA changes, mutation table, sequence alignment view, and chromatogram.
- Switch at least two samples.
- Toggle theme.
- Drag drawer width.
- Press Escape to close drawer.

Expected: The app remains usable, no blank drawer, no console crash, and chromatogram still renders.

- [ ] **Step 5: Decide on worker follow-up**

If the drawer still visibly freezes when opening large chromatograms after Task 5, write a new focused plan for worker rendering. Use:

```text
Decision: Worker rendering still needed because [specific observed freeze or timing].
Next plan: docs/superpowers/plans/YYYY-MM-DD-chromatogram-worker-rendering.md
```

If Task 5 is sufficient, record:

```text
Decision: Worker rendering deferred. Main-thread render model, downsampling, and lookup extraction are sufficient for current sample sizes.
```

- [ ] **Step 6: Commit verification note only if a file was changed**

If a follow-up plan or doc note was created:

```bash
git add docs/superpowers/plans/<new-file>.md
git commit -m "docs: record chromatogram worker decision"
```

If no files changed, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage:
  - Stable data entrance: Task 1.
  - Coordinate mapping and alignment model: Task 2.
  - Drawer sequence visualization: Task 3.
  - Chromatogram render extraction and downsampling: Tasks 4 and 5.
  - Worker as optional follow-up after evidence: Task 6.

- Type consistency:
  - `buildChromatogramData` returns current `ChromatogramData` with `base_locations` and `mixed_peaks`.
  - `AlignmentViewModel` uses current `WorkbenchSample` fields.
  - React component consumes only `AlignmentViewModel`.

- Rollback boundaries:
  - Task 1 can be reverted independently by restoring `normalize.ts`.
  - Task 3 can be reverted by restoring the old alignment `<pre>` section in `DetailDrawer`.
  - Task 5 can be reverted by restoring old drawing logic while keeping Task 4 tests as guidance.
