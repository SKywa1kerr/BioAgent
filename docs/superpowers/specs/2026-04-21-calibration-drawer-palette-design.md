# BioAgent Round 4 — Rule Calibration + Detail Drawer + Color Tokens

**Date:** 2026-04-21
**Scope:** Three coordinated changes addressing the current painful gaps — judgment rules that disagree with expert review, a results table that lags under "expand all" and feels cramped, and a monochrome UI. Truth labels are a one-shot development input; no regression harness is shipped.

---

## 1. Judgment Rule Calibration (Core Engine)

### Problem
The Python engine already computes `aa_changes` per sample (`core/alignment.py:aa_changes_from_cds`), but the ok/wrong decision is driven by raw nt substitution counts, indels, and frameshift detection. Against 17 expert-labeled samples in `truth/result*.txt` the tool diverges in at least three ways:

- **False-positive wrong at CDS edge.** `C397-a` — human: *ok*; tool: 3 substitutions at nt 6517 / 6529 / 6591 (all within ~100 bp of CDS end 6614).
- **Dual-read heuristic suppresses real evidence.** `C366-3` — human: *wrong*; tool's best-read merge keeps the FORWARD read (identity=1.0, partial CDS coverage) and the REVERSE-read subs land in `other_read_notes` if the best read is flagged "authoritative".
- **Synonymous-vs-missense not distinguished.** `C402-2` — human: *wrong* with `R171M L334S`; tool reports 8 nt subs but no aa label is surfaced and the status logic treats any sub as potential wrong.
- **No representation of "inconclusive" in engine output.** Human vocabulary includes `未测通`, `重叠峰`, `测序失败` — currently all of these would likely map to wrong or ok incorrectly.

### Design

**One-shot calibration script** — `scripts/calibrate.py` (dev tool, not imported by runtime):

- Parses `truth/result.txt`, `truth/result_pro.txt`, `truth/result_promax.txt`
- Line grammar: `<CLONE-N> gene is (ok|wrong) [AA_CHANGES] [中文备注]`
- Chinese keyword map (used only for truth-side labeling in the diff):
  - `未测通 → untested`
  - `重叠峰` / `生工重叠峰` → `uncertain`
  - `测序失败` / `片段缺失` → `untested`
  - `移码` → `frameshift+wrong`
  - `比对失败` → `uncertain`
- Runs `analyze_dataset` (or reads existing `summary.csv`/`mutations.csv` when samples have already been processed), then prints a per-sample diff: `expected=X actual=Y rule_contributions=[…]`
- Outputs match rate and a list of remaining disagreements

**Rule changes in `core/alignment.py`:**

1. **CDS-edge substitution filter.** Ignore nt substitutions that fall within `EDGE_IGNORE_BP` of `cds_start` or `cds_end` when deciding status (still recorded in `mutations` list for display, but excluded from the wrong decision). Default 20 bp; tune against the truth set.
2. **Synonymous downgrade.** If an nt substitution produces no aa change (ref codon → qry codon translates to same AA), label that mutation `synonymous` in the `mutations` entry. The wrong/ok decision uses `aa_changes` (which already excludes synonymous) as the primary signal; nt-only subs no longer upgrade an otherwise-clean sample to wrong.
3. **Dual-read consensus for single-read mutations.** When a clone has both forward and reverse reads, a mutation observed only in one read at a position that the other read *covered with the reference base* is demoted from `wrong` to `uncertain`. Still shown; label `single_read`.
4. **Quality-gated status buckets.** Introduce explicit bucket decision in `analyze_sample` return:
   - `untested` — `cds_coverage < 0.4` OR `avg_qry_quality < 15`
   - `uncertain` — 0.4 ≤ coverage < 0.8, or single-read-only mutations present, or `avg_qry_quality < 25`
   - `wrong` — has non-synonymous aa change(s) OR frameshift OR ≥2 consensus nt subs inside CDS (not near edge)
   - `ok` — otherwise
   Buckets propagate through the existing `sid_bucket` field; UI already consumes four states.

**Target.** Match rate ≥ 90% on the 17 labeled samples (10 basic + 7 pro). The promax file (12 samples) has no paired tool output in the current workspace — used as a secondary check only if the corresponding dataset is available.

**Explicitly not built:**
- No truth loader in runtime code.
- No `tests/test_truth_set.py` regression.
- No ML model (sample count too low, unnecessary).

### Files Changed
- `core/alignment.py` — edge filter, synonymous labeling, dual-read consensus, bucket decision
- New: `scripts/calibrate.py`
- `truth/result*.txt` — kept in repo as development reference, not imported

---

## 2. Detail Drawer + Compact Row (Workbench)

### Problem
`src/components/workbench/ResultsTable.tsx` expands rows in-place. Each expanded row renders the chromatogram canvas, a 600-char alignment strip, a mutation table, metrics cards, and aa-change text. "Expand all" sets every row open at once. Even with `@tanstack/react-virtual`, the estimated expanded row is 480px and the lazy chromatogram loads per row — N heavy canvases simultaneously froze the page for the user. Secondary problems:

- Expanded area is constrained by the half-width results column → cramped, hard to read full alignment.
- aa changes are only visible after expansion; summary row shows identity/coverage/mut-count without aa.

### Design

**Remove inline expansion. Introduce `DetailDrawer`** — a single right-side overlay panel driven by a `selectedSampleId` state in `ResultsWorkbench`:

- Slides in from the right, viewport-height, default width 480 px, resizable via a drag handle (persisted to `localStorage`), closable via X button or Escape.
- Only one open at a time → only one chromatogram canvas exists at any moment, which removes the "expand all" perf cliff entirely.
- Contents: header (sample ID + status pill + close), metrics grid (clone, orientation, frameshift, avg_q), aa-change block (full list), mutation table (all rows, including `synonymous` and `single_read` labels), alignment strip (full length, not truncated at 600), chromatogram (lazy, only loads when drawer is open for that sample).

**Compact always-visible row** (64 px fixed — matches current `ROW_ESTIMATE_COLLAPSED`):

- Grid columns: `Sample | Status pill | aa pill group | Identity | Coverage | Mut count | chevron`
- aa pill group shows first 3 aa changes as small capsules (e.g., `K171M` `L334S`); if more, shows `+N`. Empty state shows `-`.
- Left border 3 px colored stripe by status (reuses new color tokens from section 3).
- Whole row clickable → sets `selectedSampleId` → opens drawer.

**Replace "Expand all / Collapse all" buttons with a 2-state toolbar toggle:**

- **紧凑** (default): the compact row above.
- **详细**: same row + a secondary muted subline showing `reason · avg_q · orientation`. Still no heavy content inline. Still virtualized.

Virtualizer simplifies: drop the `ROW_ESTIMATE_EXPANDED` branch. Estimate is constant (64 or 88 depending on toolbar toggle).

### Files Changed
- New: `src/components/workbench/DetailDrawer.tsx`
- New: `src/components/workbench/DetailDrawer.css`
- `src/components/workbench/ResultsTable.tsx` — delete inline expand + per-row chromatogram/alignment/mutation-table; emit `onSelect(sample.id)` on row click; drop expandedIds state and ROW_ESTIMATE_EXPANDED
- `src/components/workbench/ResultsWorkbench.tsx` — own `selectedSampleId` state; render `<DetailDrawer sample={selected} />` at shell level
- `src/components/workbench/ResultsWorkbench.css` — compact row styles, drawer overlay layout
- `src/i18n.ts` — add strings for 紧凑/详细 toggle, drawer close label

---

## 3. Chat Rail + Status Color Tokens

### Problem
- Chat column is fixed ~50% of the shell. With a drawer taking another ~480 px on the right, the middle list becomes unreadable.
- Current palette is largely monochrome + one blue accent. `ok/wrong/uncertain/untested` look interchangeable. Mutation types (sub/ins/del/frameshift) are not color-differentiated. `ConfirmationDialog`, `MutationTrendPanel`, `LabSuggestionPanel` repeat the same single accent.

### Design

**Chat rail — 3 states**, toggle button in the chat panel header, state persisted to `localStorage`:

- **Wide** (default, ~50%) — current behavior
- **Narrow** (~30%) — conversation area compressed, composer stays full-height
- **Hidden** — collapses to a 40 px vertical spine pinned to the left edge with an expand chevron. The shell's grid reallocates freed space to the results column. The drawer continues to open on the right independently.

**Color token set** in `src/styles.css`, defined under both `:root` and `[data-theme="dark"]`:

- Status: `--status-ok`, `--status-wrong`, `--status-uncertain`, `--status-untested`
- Mutation: `--mutation-sub`, `--mutation-ins`, `--mutation-del`, `--mutation-frameshift`, `--mutation-synonymous`
- Accents: keep `--accent-primary` (existing blue); add `--accent-secondary` (teal) for secondary CTAs and second chart series

Apply tokens to: status pills, compact-row left border, aa pills (color shifts with synonymous-vs-missense), `ResultsCharts` series, panel titles, chat bubble highlights. No bespoke hex values remain in `panels/*.tsx` or workbench CSS.

Both light and dark mode retain their respective tokens; the existing theme toggle keeps working unchanged.

### Files Changed
- `src/styles.css` — token declarations, light + dark; remove hard-coded status colors
- `src/App.tsx` — chat rail width state + localStorage persistence; grid-template-columns reacts to state
- `src/components/ChatPanel.tsx` (or equivalent) — toggle button + 3 modifier classes
- `src/components/panels/AnalysisPanel.tsx` / `ConfirmationDialog.tsx` / `LabSuggestionPanel.tsx` / `MutationTrendPanel.tsx` — replace hard-coded colors with tokens
- `src/components/workbench/ResultsCharts.tsx` — pull palette from tokens
- `src/components/workbench/ResultsWorkbench.css` — token-driven borders/pills

---

## Implementation Order

1. **Calibration** — `scripts/calibrate.py` + `core/alignment.py` rule changes. Target ≥ 90% match rate. Drives the status vocabulary that sections 2 and 3 render.
2. **DetailDrawer + compact row** — biggest perf and layout win. Requires section 1's status/aa semantics but no UI dependency on section 3.
3. **Chat rail + color tokens** — polish. Naturally splits into:
   - 3a. Color tokens (safe, self-contained; can ship alongside step 2 for immediate visual improvement)
   - 3b. Chat rail 3-state (depends on 3a for drawer-vs-chat boundary styling)

---

## Non-Goals

- **Regression test against truth labels.** One-shot calibration only; truth data is not expected to grow.
- **ML-based classifier** or LLM-backed mutation adjudication. Deterministic rules suffice at this sample volume.
- **User-adjustable thresholds.** Ship one well-tuned default.
- **Promax dataset coverage guarantee.** If the corresponding AB1/GB input is not in the workspace, promax is a secondary target only.
- **Full design-system rewrite.** Only status/mutation/accent tokens are introduced; typography and spacing untouched.
