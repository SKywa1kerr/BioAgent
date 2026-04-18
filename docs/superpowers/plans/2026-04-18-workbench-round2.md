# Workbench Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add export (CSV/JSON/PDF) + filter persistence + unified Summary/Charts scope to `ResultsWorkbench`, with performance audit.

**Architecture:** Extract control state into `useWorkbenchControls` hook (persists to localStorage); Summary/Charts consume filtered view by default; export logic in pure TS modules under `src/lib/exporters/`; PDF uses pdfmake with pre-subsetted CJK font loaded lazily; Electron save dialog falls back to Blob download.

**Tech Stack:** React 18 + TypeScript, Vite, Electron 33, pdfmake (dynamic import), Noto Sans SC (SIL OFL), node:test runner for `.mjs` tests.

**Spec:** `docs/superpowers/specs/2026-04-18-workbench-round2-design.md`

---

## File Structure

**New files:**
- `src/hooks/useWorkbenchControls.ts` — state + localStorage persistence
- `src/lib/exporters/csv.ts` — CSV serialization
- `src/lib/exporters/json.ts` — JSON serialization
- `src/lib/exporters/pdf.ts` — pdfmake doc builder + font loader
- `src/lib/exporters/filename.ts` — filename helper
- `src/lib/exporters/saveFile.ts` — Blob / Electron save dispatcher
- `src/components/workbench/ExportMenu.tsx` — dropdown UI
- `src/components/workbench/SummaryScopeToggle.tsx` — all/filtered switch
- `public/fonts/NotoSansSC-subset.ttf` — pre-subsetted CJK font (checked in)
- `docs/fonts/README.md` — regeneration instructions
- `tests/test_workbench_controls.mjs`
- `tests/test_exporters.mjs`
- `tests/fixtures/generate_large.mjs` — 10k sample fixture generator

**Modified files:**
- `src/components/workbench/ResultsWorkbench.tsx` — use hook, pass filtered view down, mount ExportMenu
- `src/components/workbench/ResultsSummary.tsx` — accept scope mode + raw/filtered counts
- `src/components/workbench/ResultsCharts.tsx` — consume filtered samples when scope=filtered
- `src/components/workbench/ResultsTable.tsx` — `React.memo` on row, `useMemo` on chromatogram data
- `electron/main.js` — add `showSaveDialog` IPC handler
- `electron/preload.js` — no change needed (generic `invoke` already passes through)
- `src/i18n.ts` — new strings (see Task 2.1)

---

# Phase 1: Extract controls into a persisted hook

Goal: refactor `ResultsWorkbench` in-line state into `useWorkbenchControls`, add `summaryScope`, persist to localStorage, wire Summary/Charts to filtered view. No new UI features yet.

### Task 1.1: Create `useWorkbenchControls` hook with failing test

**Files:**
- Create: `src/hooks/useWorkbenchControls.ts`
- Create: `tests/test_workbench_controls.mjs`

- [ ] **Step 1: Write failing test**

Create `tests/test_workbench_controls.mjs`:

```js
import test from "node:test";
import assert from "node:assert/strict";
import { readControls, writeControls, DEFAULT_CONTROLS, CONTROLS_STORAGE_KEY } from "../src/hooks/useWorkbenchControls.js";

function makeStore() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
    _data: data,
  };
}

test("readControls returns defaults when storage empty", () => {
  const store = makeStore();
  assert.deepEqual(readControls(store), DEFAULT_CONTROLS);
});

test("writeControls + readControls round trips", () => {
  const store = makeStore();
  const next = { statusFilter: "wrong", searchQuery: "abc", sortKey: "identity", summaryScope: "all" };
  writeControls(store, next);
  assert.deepEqual(readControls(store), next);
});

test("readControls falls back to defaults on invalid schema", () => {
  const store = makeStore();
  store.setItem(CONTROLS_STORAGE_KEY, JSON.stringify({ statusFilter: "bogus" }));
  assert.deepEqual(readControls(store), DEFAULT_CONTROLS);
});

test("readControls tolerates malformed JSON", () => {
  const store = makeStore();
  store.setItem(CONTROLS_STORAGE_KEY, "{not json");
  assert.deepEqual(readControls(store), DEFAULT_CONTROLS);
});
```

Note: tests import `.js` because we compile-less run via node. We'll export pure functions from a `.ts` file; node can't consume TS directly, so **extract pure read/write to a plain `.js` sibling** `src/hooks/useWorkbenchControls.js` OR run tests through `tsx`. Decision: add `"test": "node --test tests/*.mjs"` to package.json and write the pure functions in a `.ts` file; tests import the `.ts` via `tsx` loader. Simpler: keep the pure schema helpers in a separate `.js` file colocated.

**Revised approach:** Split the hook into:
- `src/hooks/useWorkbenchControls.js` — pure JS exports `readControls`, `writeControls`, `DEFAULT_CONTROLS`, `CONTROLS_STORAGE_KEY`, `validateControls`
- `src/hooks/useWorkbenchControls.ts` — re-exports them with typed signatures + adds the React hook

Tests import from the `.js`.

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/test_workbench_controls.mjs`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Create `src/hooks/useWorkbenchControls.js`**

```js
export const CONTROLS_STORAGE_KEY = "bioagent-workbench-controls-v1";

export const DEFAULT_CONTROLS = Object.freeze({
  statusFilter: "all",
  searchQuery: "",
  sortKey: "status",
  summaryScope: "filtered",
});

const VALID_STATUS = new Set(["all", "ok", "wrong", "uncertain", "untested"]);
const VALID_SORT = new Set(["status", "sample", "identity", "coverage", "mutations"]);
const VALID_SCOPE = new Set(["filtered", "all"]);

export function validateControls(value) {
  if (!value || typeof value !== "object") return null;
  const { statusFilter, searchQuery, sortKey, summaryScope } = value;
  if (!VALID_STATUS.has(statusFilter)) return null;
  if (typeof searchQuery !== "string") return null;
  if (!VALID_SORT.has(sortKey)) return null;
  if (!VALID_SCOPE.has(summaryScope)) return null;
  return { statusFilter, searchQuery, sortKey, summaryScope };
}

export function readControls(storage) {
  try {
    const raw = storage.getItem(CONTROLS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_CONTROLS };
    const parsed = JSON.parse(raw);
    const valid = validateControls(parsed);
    return valid ?? { ...DEFAULT_CONTROLS };
  } catch {
    return { ...DEFAULT_CONTROLS };
  }
}

export function writeControls(storage, value) {
  try {
    const valid = validateControls(value);
    if (!valid) return;
    storage.setItem(CONTROLS_STORAGE_KEY, JSON.stringify(valid));
  } catch {
    /* ignore quota / privacy-mode errors */
  }
}
```

- [ ] **Step 4: Run test to verify passes**

Run: `node --test tests/test_workbench_controls.mjs`
Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add src/hooks/useWorkbenchControls.js tests/test_workbench_controls.mjs
git commit -m "feat(workbench): add pure controls schema + persistence helpers"
```

---

### Task 1.2: Add React hook wrapper (TS)

**Files:**
- Create: `src/hooks/useWorkbenchControls.ts`

- [ ] **Step 1: Write hook**

```ts
import { useCallback, useEffect, useRef, useState } from "react";
import {
  CONTROLS_STORAGE_KEY,
  DEFAULT_CONTROLS,
  readControls,
  writeControls,
  validateControls,
} from "./useWorkbenchControls.js";
import type { ResultsSortKey, ResultsStatusFilter } from "../components/workbench/utils";

export type SummaryScope = "filtered" | "all";

export interface WorkbenchControls {
  statusFilter: ResultsStatusFilter;
  searchQuery: string;
  sortKey: ResultsSortKey;
  summaryScope: SummaryScope;
}

export { CONTROLS_STORAGE_KEY, DEFAULT_CONTROLS, validateControls };

const WRITE_DEBOUNCE_MS = 300;

function getSafeStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

export function useWorkbenchControls() {
  const storageRef = useRef<Storage | null>(null);
  if (storageRef.current === null) storageRef.current = getSafeStorage();

  const [controls, setControls] = useState<WorkbenchControls>(() => {
    const store = storageRef.current;
    return (store ? readControls(store) : { ...DEFAULT_CONTROLS }) as WorkbenchControls;
  });

  const writeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const store = storageRef.current;
    if (!store) return;
    if (writeTimerRef.current) clearTimeout(writeTimerRef.current);
    writeTimerRef.current = setTimeout(() => writeControls(store, controls), WRITE_DEBOUNCE_MS);
    return () => {
      if (writeTimerRef.current) clearTimeout(writeTimerRef.current);
    };
  }, [controls]);

  const setStatusFilter = useCallback((statusFilter: ResultsStatusFilter) => {
    setControls((prev) => ({ ...prev, statusFilter }));
  }, []);
  const setSearchQuery = useCallback((searchQuery: string) => {
    setControls((prev) => ({ ...prev, searchQuery }));
  }, []);
  const setSortKey = useCallback((sortKey: ResultsSortKey) => {
    setControls((prev) => ({ ...prev, sortKey }));
  }, []);
  const setSummaryScope = useCallback((summaryScope: SummaryScope) => {
    setControls((prev) => ({ ...prev, summaryScope }));
  }, []);
  const reset = useCallback(() => setControls({ ...DEFAULT_CONTROLS } as WorkbenchControls), []);

  return { controls, setStatusFilter, setSearchQuery, setSortKey, setSummaryScope, reset };
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useWorkbenchControls.ts
git commit -m "feat(workbench): add React hook wrapper for controls"
```

---

### Task 1.3: Wire hook into `ResultsWorkbench`, compute scoped summaries

**Files:**
- Modify: `src/components/workbench/ResultsWorkbench.tsx`

- [ ] **Step 1: Rewrite component top to use hook**

Replace lines 16–37 (the three `useState` + derived `buckets`/`averages`/`visibleSamples`) with:

```tsx
const { controls, setStatusFilter, setSearchQuery, setSortKey, setSummaryScope, reset } = useWorkbenchControls();
const { statusFilter, searchQuery, sortKey, summaryScope } = controls;

const visibleSamples = useMemo(
  () => buildResultsView(samples, { statusFilter, searchQuery, sortKey }),
  [samples, statusFilter, searchQuery, sortKey],
);

const summarySource = summaryScope === "filtered" ? visibleSamples : samples;

const buckets = useMemo(() => {
  const acc = { ok: 0, wrong: 0, uncertain: 0, untested: 0 };
  for (const s of summarySource) acc[bucketSampleStatus(s)] += 1;
  return acc;
}, [summarySource]);

const total = summarySource.length;
const averageIdentity = useMemo(
  () => (total > 0 ? summarySource.reduce((sum, s) => sum + (s.identity || 0), 0) / total : 0),
  [summarySource, total],
);
const averageCoverage = useMemo(
  () => (total > 0 ? summarySource.reduce((sum, s) => sum + (s.cds_coverage ?? s.coverage ?? 0), 0) / total : 0),
  [summarySource, total],
);

const hasActiveControls = statusFilter !== "all" || searchQuery.trim().length > 0 || sortKey !== "status";
```

Update the imports at top:

```tsx
import { useMemo } from "react";
import { useWorkbenchControls } from "../../hooks/useWorkbenchControls";
```

Remove the `useState` import (no longer used).

- [ ] **Step 2: Update event handlers to use hook setters**

Replace `setStatusFilter(option.key)` call sites — the variable name already matches; just make sure to delete the old `useState` declarations. In the "clear" button:

```tsx
onClick={reset}
```

Replace `setSearchQuery(event.target.value)` and `setSortKey(...)` — these names are now hook setters, should work without change.

- [ ] **Step 3: Pass scoped data to Summary and Charts**

Change:
```tsx
<ResultsSummary ... />
<ResultsCharts samples={samples} language={language} />
```

To:
```tsx
<ResultsSummary
  language={language}
  total={total}
  ok={buckets.ok}
  wrong={buckets.wrong}
  uncertain={buckets.uncertain}
  untested={buckets.untested}
  averageIdentity={averageIdentity}
  averageCoverage={averageCoverage}
  scope={summaryScope}
  onScopeChange={setSummaryScope}
  originalTotal={samples.length}
  filteredTotal={visibleSamples.length}
/>
<ResultsCharts samples={summarySource} language={language} />
```

- [ ] **Step 4: Typecheck + dev run**

Run: `npx tsc --noEmit`
Expected: errors on ResultsSummary props — fixed in Task 1.4.

- [ ] **Step 5: Commit after Task 1.4 typechecks**

(Defer commit; combine with Task 1.4 since types will fail until Summary is updated.)

---

### Task 1.4: Update `ResultsSummary` to accept scope + show toggle

**Files:**
- Modify: `src/components/workbench/ResultsSummary.tsx`
- Create: `src/components/workbench/SummaryScopeToggle.tsx`

- [ ] **Step 1: Create `SummaryScopeToggle.tsx`**

```tsx
import type { SummaryScope } from "../../hooks/useWorkbenchControls";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface Props {
  scope: SummaryScope;
  onChange: (scope: SummaryScope) => void;
  language: AppLanguage;
  filteredTotal: number;
  originalTotal: number;
}

export function SummaryScopeToggle({ scope, onChange, language, filteredTotal, originalTotal }: Props) {
  if (filteredTotal === originalTotal) return null; // no filtering active
  return (
    <div className="summary-scope-toggle" role="radiogroup" aria-label={t(language, "summary.scope.label")}>
      <button
        type="button"
        role="radio"
        aria-checked={scope === "filtered"}
        className={scope === "filtered" ? "is-active" : ""}
        onClick={() => onChange("filtered")}
      >
        {t(language, "summary.scope.filtered")} ({filteredTotal})
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={scope === "all"}
        className={scope === "all" ? "is-active" : ""}
        onClick={() => onChange("all")}
      >
        {t(language, "summary.scope.all")} ({originalTotal})
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Extend `ResultsSummary` props**

Modify `ResultsSummary.tsx`. Add to `ResultsSummaryProps`:

```tsx
scope: SummaryScope;
onScopeChange: (scope: SummaryScope) => void;
originalTotal: number;
filteredTotal: number;
```

Import at top:
```tsx
import type { SummaryScope } from "../../hooks/useWorkbenchControls";
import { SummaryScopeToggle } from "./SummaryScopeToggle";
```

Render the toggle inside `results-section-header` next to the title:

```tsx
<SummaryScopeToggle
  scope={scope}
  onChange={onScopeChange}
  language={language}
  filteredTotal={filteredTotal}
  originalTotal={originalTotal}
/>
```

- [ ] **Step 3: Add i18n strings**

In `src/i18n.ts` add to both `zh` and `en` dicts:
- `"summary.scope.label"`: `"汇总范围"` / `"Summary scope"`
- `"summary.scope.filtered"`: `"筛选后"` / `"Filtered"`
- `"summary.scope.all"`: `"全部"` / `"All"`

- [ ] **Step 4: Add minimal CSS**

In `src/components/workbench/ResultsWorkbench.css` (or `ResultsCharts.css` — pick the one already imported by Summary; if none, create `ResultsSummary.css` and import):

```css
.summary-scope-toggle { display: inline-flex; gap: 4px; margin-left: auto; }
.summary-scope-toggle button {
  padding: 4px 10px; border-radius: 6px; border: 1px solid var(--results-border, #334);
  background: transparent; color: inherit; cursor: pointer; font-size: 12px;
}
.summary-scope-toggle button.is-active {
  background: var(--results-accent, #4c9eff); color: #fff; border-color: transparent;
}
```

- [ ] **Step 5: Typecheck + smoke test**

Run: `npx tsc --noEmit`
Expected: no errors.

Run: `npm run dev` — open app, load a dataset with results, apply a status filter; verify:
- Summary numbers update to filtered set
- Scope toggle appears only when filtering is active
- Toggle to "All" restores unfiltered numbers
- Charts reflect the same scope

- [ ] **Step 6: Commit**

```bash
git add src/components/workbench/ src/hooks/useWorkbenchControls.ts src/i18n.ts
git commit -m "feat(workbench): persist filter state + unify Summary/Charts scope"
```

---

# Phase 2: Core export (CSV + JSON)

### Task 2.1: Filename helper + tests

**Files:**
- Create: `src/lib/exporters/filename.ts`
- Create: `src/lib/exporters/filename.js` (sibling for tests — or use tsx; here we extract pure fn to js)
- Modify: `tests/test_exporters.mjs` (create)

- [ ] **Step 1: Create `src/lib/exporters/filename.js`**

```js
function pad(n) { return n < 10 ? `0${n}` : `${n}`; }

export function formatStamp(date = new Date()) {
  const y = date.getFullYear();
  const m = pad(date.getMonth() + 1);
  const d = pad(date.getDate());
  const hh = pad(date.getHours());
  const mm = pad(date.getMinutes());
  return `${y}${m}${d}-${hh}${mm}`;
}

export function sanitizeSegment(s) {
  if (!s || typeof s !== "string") return "";
  return s.replace(/[^a-zA-Z0-9_\-]+/g, "_").slice(0, 40);
}

export function buildExportFilename({ dataset, ext, date }) {
  const stamp = formatStamp(date);
  const ds = sanitizeSegment(dataset) || "results";
  return `bioagent-${ds}-${stamp}.${ext}`;
}
```

- [ ] **Step 2: Create `src/lib/exporters/filename.ts`** (re-export with types)

```ts
export { formatStamp, sanitizeSegment, buildExportFilename } from "./filename.js";
```

- [ ] **Step 3: Add tests** — create `tests/test_exporters.mjs`:

```js
import test from "node:test";
import assert from "node:assert/strict";
import { buildExportFilename, sanitizeSegment, formatStamp } from "../src/lib/exporters/filename.js";

test("formatStamp yields YYYYMMDD-HHmm", () => {
  const stamp = formatStamp(new Date(2026, 3, 18, 14, 23));
  assert.equal(stamp, "20260418-1423");
});

test("sanitizeSegment strips unsafe chars", () => {
  assert.equal(sanitizeSegment("pro/max:2"), "pro_max_2");
  assert.equal(sanitizeSegment(""), "");
});

test("buildExportFilename falls back to results when dataset missing", () => {
  const name = buildExportFilename({ dataset: "", ext: "csv", date: new Date(2026, 3, 18, 14, 23) });
  assert.equal(name, "bioagent-results-20260418-1423.csv");
});

test("buildExportFilename keeps valid dataset", () => {
  const name = buildExportFilename({ dataset: "promax", ext: "json", date: new Date(2026, 3, 18, 14, 23) });
  assert.equal(name, "bioagent-promax-20260418-1423.json");
});
```

- [ ] **Step 4: Run tests**

Run: `node --test tests/test_exporters.mjs`
Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add src/lib/exporters/filename.* tests/test_exporters.mjs
git commit -m "feat(exporters): add filename helper with tests"
```

---

### Task 2.2: CSV serializer + tests

**Files:**
- Create: `src/lib/exporters/csv.js` + `csv.ts`
- Modify: `tests/test_exporters.mjs`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_exporters.mjs`:

```js
import { samplesToCsv } from "../src/lib/exporters/csv.js";

test("samplesToCsv emits UTF-8 BOM and header", () => {
  const out = samplesToCsv([]);
  assert.ok(out.startsWith("\uFEFF"));
  assert.ok(out.includes("id,name,clone,status,reason"));
});

test("samplesToCsv escapes commas, quotes, newlines", () => {
  const rows = [{ id: "s1", reason: 'has "quote", and,comma\nline' }];
  const out = samplesToCsv(rows);
  assert.ok(out.includes('"has ""quote"", and,comma\nline"'));
});

test("samplesToCsv joins aa_changes arrays with semicolon-space", () => {
  const rows = [{ id: "s2", aa_changes: ["A1T", "G5C"] }];
  const out = samplesToCsv(rows);
  assert.ok(out.includes("A1T; G5C"));
});

test("samplesToCsv uses fallback mutation count fields", () => {
  const rows = [{ id: "s3", sub_count: 2, ins: 1, del_count: 0 }];
  const out = samplesToCsv(rows);
  const line = out.split("\n").find((l) => l.startsWith("s3"));
  assert.ok(line.includes(",2,1,0,"), `unexpected line: ${line}`);
});
```

Run: `node --test tests/test_exporters.mjs` → FAIL (module not found).

- [ ] **Step 2: Implement `src/lib/exporters/csv.js`**

```js
const BOM = "\uFEFF";

const COLUMNS = [
  ["id", (s) => s.id ?? ""],
  ["name", (s) => s.name ?? ""],
  ["clone", (s) => s.clone ?? ""],
  ["status", (s) => s.status ?? ""],
  ["reason", (s) => s.reason ?? s.review_reason ?? s.llm_reason ?? s.auto_reason ?? ""],
  ["identity", (s) => (typeof s.identity === "number" ? s.identity : "")],
  ["cds_coverage", (s) => {
    const v = s.cds_coverage ?? s.coverage;
    return typeof v === "number" ? v : "";
  }],
  ["sub", (s) => pickCount(s.sub_count, s.sub)],
  ["ins", (s) => pickCount(s.ins_count, s.ins)],
  ["del", (s) => pickCount(s.del_count, s.dele)],
  ["aa_changes", (s) => formatAaChanges(s.aa_changes)],
  ["avg_quality", (s) => {
    const v = s.avg_qry_quality ?? s.avg_quality;
    return typeof v === "number" ? v : "";
  }],
];

function pickCount(a, b) {
  if (typeof a === "number") return a;
  if (typeof b === "number") return b;
  return 0;
}

function formatAaChanges(value) {
  if (Array.isArray(value)) return value.filter((x) => typeof x === "string" && x.trim()).join("; ");
  if (typeof value === "string" && value.trim()) return value.trim();
  return "";
}

function escapeCell(value) {
  const s = value == null ? "" : String(value);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function samplesToCsv(samples) {
  const header = COLUMNS.map(([name]) => name).join(",");
  const rows = samples.map((s) => COLUMNS.map(([, get]) => escapeCell(get(s))).join(","));
  return BOM + [header, ...rows].join("\n");
}
```

- [ ] **Step 3: Re-export from TS** — `src/lib/exporters/csv.ts`:

```ts
export { samplesToCsv } from "./csv.js";
```

- [ ] **Step 4: Run tests**

Run: `node --test tests/test_exporters.mjs`
Expected: all passing (8 total so far).

- [ ] **Step 5: Commit**

```bash
git add src/lib/exporters/csv.* tests/test_exporters.mjs
git commit -m "feat(exporters): CSV serializer with RFC 4180 escaping + BOM"
```

---

### Task 2.3: JSON serializer + tests

**Files:**
- Create: `src/lib/exporters/json.js` + `json.ts`

- [ ] **Step 1: Add test**

Append to `tests/test_exporters.mjs`:

```js
import { samplesToJson } from "../src/lib/exporters/json.js";

test("samplesToJson wraps samples with metadata", () => {
  const fixedDate = new Date("2026-04-18T06:23:00Z");
  const out = samplesToJson([{ id: "a" }, { id: "b" }], {
    filters: { statusFilter: "wrong", searchQuery: "", sortKey: "status" },
    date: fixedDate,
  });
  const parsed = JSON.parse(out);
  assert.equal(parsed.count, 2);
  assert.equal(parsed.filters.statusFilter, "wrong");
  assert.equal(parsed.exportedAt, "2026-04-18T06:23:00.000Z");
  assert.equal(parsed.samples.length, 2);
});
```

- [ ] **Step 2: Implement**

```js
// src/lib/exporters/json.js
export function samplesToJson(samples, { filters, date = new Date() } = {}) {
  const payload = {
    exportedAt: date.toISOString(),
    filters: filters ?? null,
    count: samples.length,
    samples,
  };
  return JSON.stringify(payload, null, 2);
}
```

```ts
// src/lib/exporters/json.ts
export { samplesToJson } from "./json.js";
```

- [ ] **Step 3: Run + commit**

```bash
node --test tests/test_exporters.mjs
git add src/lib/exporters/json.* tests/test_exporters.mjs
git commit -m "feat(exporters): JSON serializer with metadata envelope"
```

---

### Task 2.4: `saveFile` dispatcher + Electron `showSaveDialog` IPC

**Files:**
- Create: `src/lib/exporters/saveFile.ts`
- Modify: `electron/main.js`

- [ ] **Step 1: Add Electron IPC handler**

In `electron/main.js`, after existing `ipcMain.handle` block, add:

```js
const { dialog } = require("electron");

ipcMain.handle("export-save-file", async (_event, { defaultPath, filters, data, encoding }) => {
  if (!mainWindow) throw new Error("no window");
  const result = await dialog.showSaveDialog(mainWindow, { defaultPath, filters });
  if (result.canceled || !result.filePath) return { canceled: true };
  const buffer = encoding === "base64" ? Buffer.from(data, "base64") : Buffer.from(data, "utf8");
  fs.writeFileSync(result.filePath, buffer);
  return { canceled: false, filePath: result.filePath };
});
```

- [ ] **Step 2: Create `src/lib/exporters/saveFile.ts`**

```ts
type SaveArgs = {
  filename: string;
  mime: string;
  data: string;          // utf8 text or base64 (when encoding="base64")
  encoding?: "utf8" | "base64";
};

declare global {
  interface Window {
    electronAPI?: {
      invoke: (channel: string, ...args: unknown[]) => Promise<any>;
    };
  }
}

export async function saveFile(args: SaveArgs): Promise<{ filePath?: string; canceled: boolean }> {
  const api = typeof window !== "undefined" ? window.electronAPI : undefined;
  if (api?.invoke) {
    try {
      const filters = filtersForExt(args.filename);
      const result = await api.invoke("export-save-file", {
        defaultPath: args.filename,
        filters,
        data: args.data,
        encoding: args.encoding ?? "utf8",
      });
      if (result && typeof result === "object") return result;
    } catch (err) {
      console.warn("Electron save failed, falling back to Blob:", err);
    }
  }
  downloadAsBlob(args);
  return { canceled: false };
}

function filtersForExt(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = { csv: "CSV", json: "JSON", pdf: "PDF" };
  return [{ name: map[ext] ?? ext.toUpperCase(), extensions: [ext] }];
}

function downloadAsBlob({ filename, mime, data, encoding }: SaveArgs) {
  const blob = encoding === "base64"
    ? new Blob([base64ToBytes(data)], { type: mime })
    : new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
```

- [ ] **Step 3: Typecheck**

Run: `npx tsc --noEmit` → no errors.

- [ ] **Step 4: Commit**

```bash
git add src/lib/exporters/saveFile.ts electron/main.js
git commit -m "feat(exporters): saveFile dispatcher with Electron + Blob fallback"
```

---

### Task 2.5: `ExportMenu` component + mount in workbench

**Files:**
- Create: `src/components/workbench/ExportMenu.tsx`
- Create: `src/components/workbench/ExportMenu.css`
- Modify: `src/components/workbench/ResultsWorkbench.tsx`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Add i18n strings**

Add to both `zh` and `en` dicts in `src/i18n.ts`:
- `"export.menu"`: `"导出"` / `"Export"`
- `"export.count"`: `"导出 {count} 条"` / `"Export {count} rows"`
- `"export.csv"`: `"CSV"` / `"CSV"`
- `"export.json"`: `"JSON"` / `"JSON"`
- `"export.pdf"`: `"PDF 报告"` / `"PDF report"`
- `"export.empty"`: `"无数据可导出"` / `"Nothing to export"`
- `"export.error"`: `"导出失败: {message}"` / `"Export failed: {message}"`

- [ ] **Step 2: Write `ExportMenu.tsx`**

```tsx
import { useRef, useState } from "react";
import type { WorkbenchSample } from "./types";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import { samplesToCsv } from "../../lib/exporters/csv";
import { samplesToJson } from "../../lib/exporters/json";
import { buildExportFilename } from "../../lib/exporters/filename";
import { saveFile } from "../../lib/exporters/saveFile";
import "./ExportMenu.css";

interface Props {
  samples: WorkbenchSample[];
  filters: { statusFilter: string; searchQuery: string; sortKey: string };
  dataset?: string;
  language: AppLanguage;
}

type Format = "csv" | "json" | "pdf";

export function ExportMenu({ samples, filters, dataset, language }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<Format | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const disabled = samples.length === 0;

  async function exportAs(fmt: Format) {
    setOpen(false);
    setError(null);
    setBusy(fmt);
    try {
      if (fmt === "csv") {
        await saveFile({
          filename: buildExportFilename({ dataset, ext: "csv" }),
          mime: "text/csv;charset=utf-8",
          data: samplesToCsv(samples),
        });
      } else if (fmt === "json") {
        await saveFile({
          filename: buildExportFilename({ dataset, ext: "json" }),
          mime: "application/json;charset=utf-8",
          data: samplesToJson(samples, { filters }),
        });
      } else if (fmt === "pdf") {
        const { exportPdf } = await import("../../lib/exporters/pdf");
        await exportPdf({ samples, filters, dataset, language });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="export-menu" ref={rootRef}>
      <button
        type="button"
        className="export-menu-trigger"
        disabled={disabled || busy !== null}
        onClick={() => setOpen((v) => !v)}
        title={disabled ? t(language, "export.empty") : undefined}
      >
        {busy ? "…" : t(language, "export.menu")} ({samples.length})
      </button>
      {open && !disabled ? (
        <ul className="export-menu-list" role="menu">
          <li><button role="menuitem" onClick={() => exportAs("csv")}>{t(language, "export.csv")}</button></li>
          <li><button role="menuitem" onClick={() => exportAs("json")}>{t(language, "export.json")}</button></li>
          <li><button role="menuitem" onClick={() => exportAs("pdf")}>{t(language, "export.pdf")}</button></li>
        </ul>
      ) : null}
      {error ? <div className="export-menu-error" role="alert">{t(language, "export.error", { message: error })}</div> : null}
    </div>
  );
}
```

Click-outside dismiss: add `useEffect` listening `mousedown` on document and closing if target outside `rootRef.current`.

- [ ] **Step 3: Create `ExportMenu.css`**

```css
.export-menu { position: relative; display: inline-block; }
.export-menu-trigger {
  padding: 6px 14px; border-radius: 6px; border: 1px solid var(--results-border, #334);
  background: var(--results-accent, #4c9eff); color: #fff; cursor: pointer; font-size: 13px;
}
.export-menu-trigger:disabled { opacity: 0.5; cursor: not-allowed; }
.export-menu-list {
  position: absolute; top: 100%; right: 0; z-index: 20;
  margin-top: 4px; padding: 4px; min-width: 160px; list-style: none;
  background: var(--results-panel-bg, #1e2533); border: 1px solid var(--results-border, #334);
  border-radius: 6px; box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}
.export-menu-list button {
  width: 100%; text-align: left; padding: 6px 10px; background: transparent;
  border: none; color: inherit; cursor: pointer; border-radius: 4px; font-size: 13px;
}
.export-menu-list button:hover { background: var(--results-row-hover, #2a3142); }
.export-menu-error { margin-top: 6px; color: var(--results-danger, #ff6b6b); font-size: 12px; }
```

- [ ] **Step 4: Mount in `ResultsWorkbench.tsx`**

Inside `results-toolbar-controls` div, after the sort field label, add:

```tsx
<ExportMenu
  samples={visibleSamples}
  filters={{ statusFilter, searchQuery, sortKey }}
  dataset={samples[0]?.clone /* approximation; adjust if dataset id is on a known field */}
  language={language}
/>
```

Note: the `WorkbenchSample` type has no explicit `dataset` field — `clone` sometimes carries it. Simplest: pass undefined and let filename fall back to `results`. Preferred: add `dataset` prop to `ResultsWorkbench` from caller; for this plan, pass `undefined` and address later.

- [ ] **Step 5: Dev smoke test**

Run: `npm run dev`
- Apply a filter → menu shows correct count
- Export CSV → file downloads with UTF-8 BOM (Excel opens correctly)
- Export JSON → valid JSON with metadata
- Electron build: `npm run electron:dev` → save dialog appears

- [ ] **Step 6: Commit**

```bash
git add src/components/workbench/ExportMenu.tsx src/components/workbench/ExportMenu.css src/components/workbench/ResultsWorkbench.tsx src/i18n.ts
git commit -m "feat(workbench): add export menu with CSV/JSON (PDF pending)"
```

---

# Phase 3: PDF export

### Task 3.1: Prepare CJK font subset

**Files:**
- Create: `public/fonts/NotoSansSC-subset.ttf` (binary)
- Create: `public/fonts/NotoSansSC-subset-Bold.ttf` (binary)
- Create: `docs/fonts/README.md`

- [ ] **Step 1: Download Noto Sans SC Regular + Bold** from https://fonts.google.com/noto/specimen/Noto+Sans+SC

- [ ] **Step 2: Install fonttools** — `pip install fonttools brotli`

- [ ] **Step 3: Subset using `pyftsubset`**

```bash
pyftsubset NotoSansSC-Regular.ttf \
  --text-file=docs/fonts/common-chars.txt \
  --output-file=public/fonts/NotoSansSC-subset.ttf \
  --no-hinting --desubroutinize --name-IDs=* --glyph-names

pyftsubset NotoSansSC-Bold.ttf \
  --text-file=docs/fonts/common-chars.txt \
  --output-file=public/fonts/NotoSansSC-subset-Bold.ttf \
  --no-hinting --desubroutinize --name-IDs=* --glyph-names
```

`docs/fonts/common-chars.txt` content: all ASCII printable + `GB2312` common 6763 Chinese chars. Start with a minimal set: copy all strings from `src/i18n.ts` zh dict + common genetics terms. Target output <400KB per weight.

- [ ] **Step 4: Write `docs/fonts/README.md`**

Document:
- Source font + license (SIL OFL 1.1)
- Subset command used
- How to regenerate when adding new UI strings

- [ ] **Step 5: Commit**

```bash
git add public/fonts/ docs/fonts/
git commit -m "chore(fonts): add Noto Sans SC subset for PDF export"
```

---

### Task 3.2: Install pdfmake + implement `pdf.ts`

**Files:**
- Modify: `package.json`
- Create: `src/lib/exporters/pdf.ts`

- [ ] **Step 1: Install pdfmake**

```bash
npm install pdfmake
npm install -D @types/pdfmake
```

- [ ] **Step 2: Write `pdf.ts`**

```ts
import type { WorkbenchSample } from "../../components/workbench/types";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import { bucketSampleStatus } from "../../components/workbench/utils";
import { buildExportFilename } from "./filename";
import { saveFile } from "./saveFile";

const MAX_DETAIL_SAMPLES = 200;
const MAX_REASON_CHARS = 400;

let fontsLoaded = false;
let pdfMakeRef: any = null;

async function ensurePdfMake() {
  if (pdfMakeRef && fontsLoaded) return pdfMakeRef;
  const pdfMakeMod = await import("pdfmake/build/pdfmake");
  const pdfMake = (pdfMakeMod as any).default ?? pdfMakeMod;

  const [regular, bold] = await Promise.all([
    fetchAsBase64("/fonts/NotoSansSC-subset.ttf"),
    fetchAsBase64("/fonts/NotoSansSC-subset-Bold.ttf"),
  ]);
  pdfMake.vfs = {
    "NotoSansSC-Regular.ttf": regular,
    "NotoSansSC-Bold.ttf": bold,
  };
  pdfMake.fonts = {
    NotoSansSC: {
      normal: "NotoSansSC-Regular.ttf",
      bold: "NotoSansSC-Bold.ttf",
      italics: "NotoSansSC-Regular.ttf",
      bolditalics: "NotoSansSC-Bold.ttf",
    },
  };
  pdfMakeRef = pdfMake;
  fontsLoaded = true;
  return pdfMake;
}

async function fetchAsBase64(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Font fetch failed: ${url} (${res.status})`);
  const buf = await res.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export interface BuildDocArgs {
  samples: WorkbenchSample[];
  filters: { statusFilter: string; searchQuery: string; sortKey: string };
  dataset?: string;
  language: AppLanguage;
  detailMode: boolean;
}

export function buildDocDefinition({ samples, filters, dataset, language, detailMode }: BuildDocArgs): any {
  const counts = { ok: 0, wrong: 0, uncertain: 0, untested: 0 };
  for (const s of samples) counts[bucketSampleStatus(s)] += 1;

  const content: any[] = [
    { text: t(language, "export.pdf.title") || "BioAgent Report", style: "title" },
    { text: `${t(language, "export.pdf.dataset") || "Dataset"}: ${dataset ?? "-"}`, margin: [0, 4, 0, 2] },
    { text: `${t(language, "export.pdf.exportedAt") || "Exported"}: ${new Date().toLocaleString()}`, margin: [0, 0, 0, 8] },
    {
      text: `${t(language, "export.pdf.filters") || "Filters"}: status=${filters.statusFilter}, search="${filters.searchQuery}", sort=${filters.sortKey}`,
      style: "meta",
    },
    {
      style: "counts",
      table: {
        widths: ["*", "*", "*", "*", "*"],
        body: [
          ["total", "ok", "wrong", "uncertain", "untested"],
          [String(samples.length), String(counts.ok), String(counts.wrong), String(counts.uncertain), String(counts.untested)],
        ],
      },
    },
  ];

  if (detailMode) {
    for (const s of samples) {
      content.push(
        { text: s.id ?? "(no id)", style: "sampleHeader", pageBreak: "before" },
        {
          table: {
            widths: ["auto", "*"],
            body: [
              ["status", s.status ?? bucketSampleStatus(s)],
              ["identity", typeof s.identity === "number" ? `${s.identity.toFixed(1)}%` : "-"],
              ["coverage", typeof (s.cds_coverage ?? s.coverage) === "number" ? `${(s.cds_coverage ?? s.coverage ?? 0).toFixed(1)}%` : "-"],
              ["sub/ins/del", `${s.sub_count ?? s.sub ?? 0} / ${s.ins_count ?? s.ins ?? 0} / ${s.del_count ?? s.dele ?? 0}`],
              ["aa_changes", formatAa(s.aa_changes)],
            ],
          },
          margin: [0, 4, 0, 4],
        },
        { text: truncate(s.reason ?? s.review_reason ?? s.llm_reason ?? "", MAX_REASON_CHARS), style: "reason" },
      );
    }
  } else {
    content.push({
      text: t(language, "export.pdf.summaryOnly") || "Summary only (sample count exceeds detail threshold).",
      style: "meta",
      margin: [0, 12, 0, 0],
    });
  }

  return {
    content,
    defaultStyle: { font: "NotoSansSC", fontSize: 10 },
    styles: {
      title: { fontSize: 20, bold: true, margin: [0, 0, 0, 8] },
      meta: { fontSize: 9, color: "#666" },
      counts: { margin: [0, 8, 0, 8] },
      sampleHeader: { fontSize: 13, bold: true, margin: [0, 12, 0, 4] },
      reason: { fontSize: 9, italics: true, color: "#444" },
    },
    pageMargins: [40, 40, 40, 40],
  };
}

function formatAa(value: WorkbenchSample["aa_changes"]): string {
  if (Array.isArray(value)) return value.filter((x): x is string => typeof x === "string").join("; ") || "-";
  if (typeof value === "string" && value.trim()) return value.trim();
  return "-";
}

function truncate(s: string, max: number) {
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

export async function exportPdf(args: Omit<BuildDocArgs, "detailMode"> & { onWarn?: (msg: string) => void }): Promise<void> {
  const detailMode = args.samples.length <= MAX_DETAIL_SAMPLES;
  if (!detailMode && args.onWarn) args.onWarn(`Sample count ${args.samples.length} exceeds detail threshold (${MAX_DETAIL_SAMPLES}); emitting summary only.`);

  const pdfMake = await ensurePdfMake();
  const doc = buildDocDefinition({ ...args, detailMode });
  const filename = buildExportFilename({ dataset: args.dataset, ext: "pdf" });

  return new Promise<void>((resolve, reject) => {
    pdfMake.createPdf(doc).getBase64(async (base64: string) => {
      try {
        await saveFile({ filename, mime: "application/pdf", data: base64, encoding: "base64" });
        resolve();
      } catch (err) {
        reject(err);
      }
    });
  });
}
```

- [ ] **Step 3: Add smoke test for `buildDocDefinition`**

Append to `tests/test_exporters.mjs`:

```js
// Skipping — pdf.ts imports pdfmake which is browser-only; run smoke via a separate UI test
```

Actually pdfmake loading in Node is non-trivial. Strategy: **extract `buildDocDefinition` into a pure `.js` module** `src/lib/exporters/pdfDoc.js` that doesn't import pdfmake — then `pdf.ts` uses both `pdfDoc.js` and the pdfmake dynamic import.

- [ ] **Step 4: Refactor into `pdfDoc.js` + test**

Move `buildDocDefinition`, `formatAa`, `truncate`, `MAX_DETAIL_SAMPLES`, `MAX_REASON_CHARS`, and `bucketSampleStatus` import into `src/lib/exporters/pdfDoc.js` (JS, no pdfmake). `pdf.ts` imports `buildDocDefinition` from there and only adds the font/pdfmake plumbing.

Test:
```js
import { buildDocDefinition } from "../src/lib/exporters/pdfDoc.js";

test("buildDocDefinition detail mode produces sample sections", () => {
  const doc = buildDocDefinition({
    samples: [{ id: "s1", status: "ok", identity: 99.5, reason: "ok" }],
    filters: { statusFilter: "all", searchQuery: "", sortKey: "status" },
    dataset: "pro",
    detailMode: true,
    stringsFn: (k) => k,
  });
  assert.ok(Array.isArray(doc.content));
  assert.ok(doc.content.some((c) => c.text === "s1"));
});

test("buildDocDefinition summary mode skips per-sample sections", () => {
  const samples = Array.from({ length: 201 }, (_, i) => ({ id: `s${i}` }));
  const doc = buildDocDefinition({
    samples, filters: { statusFilter: "all", searchQuery: "", sortKey: "status" },
    detailMode: false, stringsFn: (k) => k,
  });
  assert.equal(doc.content.filter((c) => c.style === "sampleHeader").length, 0);
});
```

Adjust `pdfDoc.js` to accept a `stringsFn` fallback instead of hard-depending on `t()`.

- [ ] **Step 5: Run tests + typecheck**

```bash
node --test tests/test_exporters.mjs
npx tsc --noEmit
```

- [ ] **Step 6: Commit**

```bash
git add src/lib/exporters/pdf.ts src/lib/exporters/pdfDoc.js tests/test_exporters.mjs package.json package-lock.json
git commit -m "feat(exporters): PDF report with CJK font + dynamic import"
```

---

### Task 3.3: Wire PDF into ExportMenu + add i18n

**Files:**
- Modify: `src/components/workbench/ExportMenu.tsx`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Add PDF-specific i18n**

Add keys:
- `"export.pdf.title"`: `"BioAgent 分析报告"` / `"BioAgent Analysis Report"`
- `"export.pdf.dataset"`: `"数据集"` / `"Dataset"`
- `"export.pdf.exportedAt"`: `"导出时间"` / `"Exported at"`
- `"export.pdf.filters"`: `"筛选条件"` / `"Filters"`
- `"export.pdf.summaryOnly"`: `"样本数超过 200，仅生成概要报告。请使用 CSV/JSON 获取完整明细。"` / `"Sample count exceeds 200; summary only. Use CSV/JSON for full detail."`
- `"export.pdf.warn.bigBatch"`: `"PDF 概要模式（{count} 条）"` / `"PDF summary mode ({count} rows)"`

- [ ] **Step 2: Wire warning toast in ExportMenu**

`exportAs("pdf")` branch already calls `exportPdf`. Pass `onWarn` callback:

```ts
const { exportPdf } = await import("../../lib/exporters/pdf");
await exportPdf({
  samples, filters, dataset, language,
  onWarn: (msg) => setError(msg),
});
```

Treat warn as info-level (not red). Add a `warning` state if desired; for now reuse `error` with neutral styling.

- [ ] **Step 3: Dev smoke**

Run: `npm run dev` → export PDF on a small dataset → verify CJK text renders correctly (no tofu squares) and layout is not broken.

- [ ] **Step 4: Commit**

```bash
git add src/components/workbench/ExportMenu.tsx src/i18n.ts
git commit -m "feat(workbench): wire PDF export into menu"
```

---

# Phase 4: Performance audit + finishing touches

### Task 4.1: Memoize `ResultsTable` row

**Files:**
- Modify: `src/components/workbench/ResultsTable.tsx`

- [ ] **Step 1: Extract row renderer**

In `ResultsTable.tsx`, find the inline row JSX inside the virtualizer map (around the row render block). Extract it into a `Row` component:

```tsx
interface RowProps {
  sample: WorkbenchSample;
  expanded: boolean;
  onToggle: (id: string) => void;
  language: AppLanguage;
}

const Row = React.memo(function Row({ sample, expanded, onToggle, language }: RowProps) {
  /* original per-row JSX */
});
```

Use `React.memo` with shallow prop compare (default). Ensure `onToggle` is stable via `useCallback` in the parent.

- [ ] **Step 2: Memoize chromatogram data**

```tsx
const chromatogramData = useMemo(() => (expanded ? toChromatogramData(sample) : null), [expanded, sample]);
```

- [ ] **Step 3: Dev verify**

Open React DevTools Profiler → record a filter change → confirm rows outside viewport do not re-render.

- [ ] **Step 4: Commit**

```bash
git add src/components/workbench/ResultsTable.tsx
git commit -m "perf(workbench): memoize table row + chromatogram data"
```

---

### Task 4.2: Memoize charts data

**Files:**
- Modify: `src/components/workbench/ResultsCharts.tsx`

- [ ] **Step 1: Wrap derived data in `useMemo`**

```tsx
const identityBins = useMemo(() => buildBins(samples, getIdentityValue), [samples]);
const coverageBins = useMemo(() => buildBins(samples, getCoverageValue), [samples]);
const scatterData = useMemo(() => buildScatterData(samples), [samples]);
const statusSlices = useMemo(() => buildStatusSlices(samples), [samples]);
```

Add `useMemo` import.

- [ ] **Step 2: Commit**

```bash
git add src/components/workbench/ResultsCharts.tsx
git commit -m "perf(workbench): memoize chart datasets"
```

---

### Task 4.3: Add large-dataset fixture + manual perf log

**Files:**
- Create: `tests/fixtures/generate_large.mjs`
- Create: `docs/perf/2026-04-18-round2-notes.md`

- [ ] **Step 1: Write fixture generator**

```js
// tests/fixtures/generate_large.mjs
import { writeFileSync } from "node:fs";

const STATUS = ["ok", "wrong", "uncertain"];
const samples = Array.from({ length: 10000 }, (_, i) => ({
  id: `sample-${String(i).padStart(5, "0")}`,
  name: `Sample ${i}`,
  clone: `clone-${i % 50}`,
  status: STATUS[i % STATUS.length],
  reason: i % 7 === 0 ? "???" : `reason text ${i}`,
  identity: 85 + (i % 15),
  cds_coverage: 80 + (i % 20),
  sub_count: i % 5,
  ins_count: i % 3,
  del_count: i % 2,
  aa_changes: i % 4 === 0 ? [`A${i}T`] : [],
}));
writeFileSync("tests/fixtures/large-samples.json", JSON.stringify(samples));
console.log(`Wrote ${samples.length} samples`);
```

Run: `node tests/fixtures/generate_large.mjs`

- [ ] **Step 2: Manual perf check**

Temporarily patch `ResultsWorkbench` with `const samples = fixture;` (commented after measuring). Record in `docs/perf/2026-04-18-round2-notes.md`:

- Filter response time (ms) before/after Tasks 4.1 + 4.2
- React DevTools Profiler flamegraph screenshot references
- Scroll FPS (Chrome DevTools Performance tab)

Revert the patch before commit.

- [ ] **Step 3: Commit notes**

```bash
git add tests/fixtures/ docs/perf/
git commit -m "docs(perf): Round 2 baseline + post-memoization measurements"
```

---

### Task 4.4: Empty state for 0-filtered

**Files:**
- Modify: `src/components/workbench/ResultsTable.tsx`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Add i18n**

- `"wb.empty.filtered"`: `"无匹配结果"` / `"No matching results"`
- `"wb.empty.clear"`: `"清除筛选"` / `"Clear filters"`

- [ ] **Step 2: Render empty state**

Near top of `ResultsTable` component body:

```tsx
if (samples.length === 0) {
  return (
    <div className="results-empty" role="status">
      <p>{t(language, "wb.empty.filtered")}</p>
    </div>
  );
}
```

Add corresponding CSS if needed.

- [ ] **Step 3: Commit**

```bash
git add src/components/workbench/ResultsTable.tsx src/i18n.ts
git commit -m "feat(workbench): empty state for filtered-to-zero"
```

---

### Task 4.5: Final regression pass

- [ ] **Step 1: Full test run**

```bash
node --test tests/*.mjs
npx tsc --noEmit
```

All must pass.

- [ ] **Step 2: Electron build smoke**

```bash
npm run build
npm run electron:dev
```

Verify in packaged-style run:
- Filters persist after reload
- CSV/JSON export writes a file via native dialog
- PDF export renders CN + EN characters
- Summary toggle appears only when filtering is active
- 10k-sample fixture scrolls at 60fps

- [ ] **Step 3: Update CLAUDE.md / README if applicable**

Skip unless documentation gap found.

- [ ] **Step 4: Final commit (tag the round)**

```bash
git commit --allow-empty -m "chore: Round 2 workbench upgrade complete"
git tag round-2-workbench
```

---

## Rollback Notes

- Phase 1 regression: restore `useState` triplet in `ResultsWorkbench.tsx`, delete `useWorkbenchControls.*` and `SummaryScopeToggle.tsx`.
- Phases 2–4 are purely additive; revert the specific commits.
- localStorage key is `-v1`; future breaking schema changes must use `-v2` and co-exist briefly.

## Dependency Additions

- Runtime: `pdfmake`
- Dev: `@types/pdfmake`

All other functionality uses existing deps.
