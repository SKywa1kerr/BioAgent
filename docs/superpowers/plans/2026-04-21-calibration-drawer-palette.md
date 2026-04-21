# Round 4 — Calibration + Detail Drawer + Color Tokens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the judgment engine against expert-labeled samples, replace inline row expansion with a single right-side DetailDrawer, and introduce status / mutation / accent color tokens plus a three-state chat rail.

**Architecture:** Three coordinated subsystems ship in order — engine rules first (they define the `ok | wrong | uncertain | untested` vocabulary the UI renders), then the drawer refactor (biggest perf and layout win), then color tokens + chat rail polish. Each subsystem decomposes into small self-contained commits.

**Tech Stack:** Python 3.11+ (BioPython, pandas), React 18 + TypeScript, Vite, `@tanstack/react-virtual`, `node:test` for JS, `pytest` for Python.

---

## File Structure

### Created
- `scripts/__init__.py` — make `scripts` a package (empty file)
- `scripts/calibrate.py` — dev tool. Parses `truth/result*.txt`, diffs engine output, prints match rate. Not imported at runtime.
- `tests/test_calibration_truth_parser.py` — unit tests for the truth-line parser
- `tests/test_alignment_rules.py` — rule-level tests: edge-ignore, synonymous downgrade, dual-read consensus, bucket decision
- `src/components/workbench/DetailDrawer.tsx` — right-side overlay
- `src/components/workbench/DetailDrawer.css` — drawer-specific styles
- `src/lib/workbench/compactRow.ts` — pure helper building compact-row cells from a `WorkbenchSample` (aa-pill list, first-N-plus-rest)
- `tests/test_compact_row.mjs` — tests for compact-row helper
- `src/lib/ui/chatRailState.ts` — pure helper for 3-state rail persistence
- `tests/test_chat_rail_state.mjs` — tests for rail state

### Modified
- `core/alignment.py` — `analyze_sample` returns bucket, mutations get `synonymous` / `single_read` labels, edge-ignore applied, dual-read consensus demotes mutations, new constants
- `src/components/workbench/ResultsTable.tsx` — drop inline expand state and `ROW_ESTIMATE_EXPANDED`; emit `onSelect(id)`; render compact row + optional detail subline
- `src/components/workbench/ResultsWorkbench.tsx` — own `selectedSampleId`, mount `<DetailDrawer />`, own `density` toggle
- `src/components/workbench/ResultsWorkbench.css` — compact row grid, aa-pill group, detail-subline, drawer layout hooks
- `src/components/workbench/ResultsCharts.tsx` — colors pulled from CSS variables
- `src/styles.css` — status / mutation / accent tokens under `:root` and `[data-theme="dark"]`; remove hard-coded status colors
- `src/App.tsx` — chat rail width state + localStorage; `grid-template-columns` reacts to state
- `src/components/ChatPanel.tsx` — rail-state toggle button, 3 modifier classes (locate the existing file; if still inside `App.tsx`, extract once and use here)
- `src/components/panels/AnalysisPanel.tsx` / `ConfirmationDialog.tsx` / `LabSuggestionPanel.tsx` / `MutationTrendPanel.tsx` — replace any bespoke hex with tokens
- `src/i18n.ts` — keys for `wb.density.compact`, `wb.density.detailed`, `wb.drawer.close`, `wb.chatRail.wide`, `wb.chatRail.narrow`, `wb.chatRail.hidden`, bucket labels already exist

### Out of scope (flagged in spec)
- No truth loader in runtime code
- No regression harness
- No ML adjudication
- Promax dataset is secondary — only run the diff if its AB1/GB inputs are present

---

## Pre-flight

- [ ] **Step P.1: Confirm working tree is clean or WIP is intentional**

Run: `git status --short`

The worktree currently has uncommitted changes to `core/alignment.py`, `core/llm_client.py`, panel files, and four new test files. If these changes are from a prior unfinished session and are **not** part of Round 4, ask the user whether to commit them to a WIP branch, stash them, or continue on top. The plan below assumes a clean tree or that the WIP has been resolved.

- [ ] **Step P.2: Verify test commands work**

Run: `npm run test:js` — expected: existing tests pass
Run: `npm run test:py` — expected: existing tests pass (or tally of known-broken tests so you can tell what you break later)

---

## Section 1 — Calibration

### Task 1.1: Truth-line parser

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/calibrate.py`
- Create: `tests/test_calibration_truth_parser.py`

- [ ] **Step 1.1.1: Write failing parser tests**

```python
# tests/test_calibration_truth_parser.py
from scripts.calibrate import parse_truth_line, TruthRecord


def test_parses_ok():
    rec = parse_truth_line("C379-a gene is ok")
    assert rec == TruthRecord(sid="C379-a", status="ok", aa=[], note=None)


def test_parses_wrong_with_aa_list():
    rec = parse_truth_line("C402-2 gene is wrong R171M L334S")
    assert rec.status == "wrong"
    assert rec.aa == ["R171M", "L334S"]


def test_parses_single_aa():
    rec = parse_truth_line("C397-2 gene is wrong K171M")
    assert rec.aa == ["K171M"]


def test_chinese_note_maps_to_uncertain_for_overlap():
    rec = parse_truth_line("C410-1 重叠峰")
    assert rec.status == "uncertain"


def test_chinese_note_maps_to_untested_for_failed():
    rec = parse_truth_line("C410-2 测序失败")
    assert rec.status == "untested"


def test_frameshift_keyword_wrongs_the_call():
    rec = parse_truth_line("C411-3 移码突变")
    assert rec.status == "wrong"


def test_blank_line_returns_none():
    assert parse_truth_line("") is None
    assert parse_truth_line("   ") is None


def test_comment_line_returns_none():
    assert parse_truth_line("# header") is None
```

- [ ] **Step 1.1.2: Run tests to confirm they fail**

Run: `pytest tests/test_calibration_truth_parser.py -v`
Expected: ImportError or assertion failures — parser does not exist yet.

- [ ] **Step 1.1.3: Implement parser**

```python
# scripts/calibrate.py
"""Truth-set calibration tool. Dev only — not imported by runtime.

Usage:
    python -m scripts.calibrate --dataset base
    python -m scripts.calibrate --dataset pro --data-dir ./data
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CHINESE_KEYWORDS: dict[str, str] = {
    "未测通": "untested",
    "测序失败": "untested",
    "片段缺失": "untested",
    "重叠峰": "uncertain",
    "生工重叠峰": "uncertain",
    "比对失败": "uncertain",
    "移码": "wrong",
}

AA_CHANGE_RE = re.compile(r"^[A-Z*]\d+[A-Z*]$")
LINE_RE = re.compile(r"^(?P<sid>C\d+-\w+)\s+gene\s+is\s+(?P<status>ok|wrong)(?P<rest>.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class TruthRecord:
    sid: str
    status: str  # ok | wrong | uncertain | untested
    aa: list[str] = field(default_factory=list)
    note: str | None = None


def parse_truth_line(line: str) -> TruthRecord | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Chinese-keyword-only lines (no "gene is ok/wrong")
    for kw, mapped in CHINESE_KEYWORDS.items():
        if kw in stripped and "gene is" not in stripped.lower():
            m = re.match(r"^(C\d+-\w+)", stripped)
            sid = m.group(1) if m else stripped.split()[0]
            return TruthRecord(sid=sid, status=mapped, note=stripped)

    m = LINE_RE.match(stripped)
    if not m:
        return None
    sid = m.group("sid")
    status = m.group("status").lower()
    rest = (m.group("rest") or "").strip()
    aa: list[str] = []
    note_tokens: list[str] = []
    for tok in rest.split():
        if AA_CHANGE_RE.match(tok):
            aa.append(tok)
        else:
            note_tokens.append(tok)
    note = " ".join(note_tokens) or None
    # Chinese keywords in note can override status
    if note:
        for kw, mapped in CHINESE_KEYWORDS.items():
            if kw in note:
                status = mapped
                break
    return TruthRecord(sid=sid, status=status, aa=aa, note=note)


def load_truth_file(path: Path) -> list[TruthRecord]:
    records: list[TruthRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        rec = parse_truth_line(raw)
        if rec is not None:
            records.append(rec)
    return records
```

- [ ] **Step 1.1.4: Verify tests pass**

Run: `pytest tests/test_calibration_truth_parser.py -v`
Expected: 8 passed.

- [ ] **Step 1.1.5: Commit**

```bash
git add scripts/__init__.py scripts/calibrate.py tests/test_calibration_truth_parser.py
git commit -m "feat(calibrate): truth-line parser with Chinese keyword mapping"
```

---

### Task 1.2: Edge-ignore substitution filter

**Files:**
- Modify: `core/alignment.py` (new helper + `analyze_sample` uses it)
- Create: `tests/test_alignment_rules.py`

- [ ] **Step 1.2.1: Write failing edge-ignore test**

```python
# tests/test_alignment_rules.py
from core.alignment import is_edge_ignored, EDGE_IGNORE_BP


def test_substitution_within_edge_of_cds_end_is_ignored():
    # cds_end=6614, EDGE_IGNORE_BP=20 default → pos 6600 ignored, 6593 not
    assert is_edge_ignored(pos=6600, cds_start=100, cds_end=6614) is True
    assert is_edge_ignored(pos=6593, cds_start=100, cds_end=6614) is False


def test_substitution_within_edge_of_cds_start_is_ignored():
    assert is_edge_ignored(pos=105, cds_start=100, cds_end=6614) is True
    assert is_edge_ignored(pos=125, cds_start=100, cds_end=6614) is False


def test_none_cds_bounds_never_ignored():
    assert is_edge_ignored(pos=100, cds_start=None, cds_end=None) is False
```

- [ ] **Step 1.2.2: Run to confirm failure**

Run: `pytest tests/test_alignment_rules.py::test_substitution_within_edge_of_cds_end_is_ignored -v`
Expected: ImportError (helper doesn't exist).

- [ ] **Step 1.2.3: Add helper + constant in `core/alignment.py`**

Locate the constants/top section of `core/alignment.py` (roughly the top, after imports). Add:

```python
# ── Rule tuning constants ─────────────────────────────────────────────────

# Substitutions whose 1-based ref position is within this many bp of
# cds_start or cds_end are demoted from wrong to ok for status purposes.
# They are still recorded in the mutations list for display.
EDGE_IGNORE_BP = 20


def is_edge_ignored(pos: int | None, cds_start: int | None, cds_end: int | None) -> bool:
    """True if this substitution position is close enough to either CDS boundary
    that we should exclude it from the wrong decision."""
    if pos is None or cds_start is None or cds_end is None:
        return False
    return (pos - cds_start) < EDGE_IGNORE_BP or (cds_end - pos) < EDGE_IGNORE_BP
```

- [ ] **Step 1.2.4: Run edge test**

Run: `pytest tests/test_alignment_rules.py -v -k edge`
Expected: 3 passed.

- [ ] **Step 1.2.5: Commit**

```bash
git add core/alignment.py tests/test_alignment_rules.py
git commit -m "feat(alignment): edge-ignore helper for CDS-boundary substitutions"
```

---

### Task 1.3: Synonymous labeling

**Files:**
- Modify: `core/alignment.py` — `extract_mutations` (or a downstream enrich step)
- Modify: `tests/test_alignment_rules.py`

- [ ] **Step 1.3.1: Write failing synonymous test**

```python
# append to tests/test_alignment_rules.py
from core.alignment import label_synonymous_mutations


def test_synonymous_substitution_is_labeled():
    # GAA (Glu) -> GAG (Glu) = synonymous
    ref_seq = "ATGGAATAA"  # M-E-*
    mutations = [
        {"type": "substitution", "position": 6, "refBase": "A", "queryBase": "G",
         "effect": ""},
    ]
    result = label_synonymous_mutations(
        mutations, ref_seq=ref_seq, cds_start=1, cds_end=9,
        cds_positions={4: "G", 5: "A", 6: "G", 7: "T", 8: "A", 9: "A"},
    )
    assert result[0]["effect"] == "synonymous"


def test_missense_substitution_is_not_labeled():
    ref_seq = "ATGGAATAA"  # M-E-*
    mutations = [
        {"type": "substitution", "position": 5, "refBase": "A", "queryBase": "T",
         "effect": ""},
    ]
    result = label_synonymous_mutations(
        mutations, ref_seq=ref_seq, cds_start=1, cds_end=9,
        cds_positions={4: "G", 5: "T", 6: "A", 7: "T", 8: "A", 9: "A"},
    )
    assert result[0]["effect"] != "synonymous"


def test_indel_mutations_passthrough_unchanged():
    mutations = [
        {"type": "insertion", "position": 5, "refBase": "-", "queryBase": "A",
         "effect": ""},
        {"type": "deletion", "position": 7, "refBase": "T", "queryBase": "-",
         "effect": ""},
    ]
    result = label_synonymous_mutations(
        mutations, ref_seq="ATGGAATAA", cds_start=1, cds_end=9, cds_positions={},
    )
    assert all(m["effect"] == "" for m in result)
```

- [ ] **Step 1.3.2: Run to confirm failure**

Run: `pytest tests/test_alignment_rules.py -v -k synonymous`
Expected: ImportError.

- [ ] **Step 1.3.3: Implement `label_synonymous_mutations`**

Add after `extract_mutations` in `core/alignment.py`:

```python
def label_synonymous_mutations(mutations: list[dict],
                               ref_seq: str,
                               cds_start: int | None,
                               cds_end: int | None,
                               cds_positions: dict) -> list[dict]:
    """For each substitution inside CDS, translate ref and query codons and
    tag effect='synonymous' when the AA does not change. Non-substitutions
    and out-of-CDS subs pass through unchanged.

    cds_positions: {ref_pos (1-based) -> query_base} built by
                   compute_cds_covered_positions.
    """
    if cds_start is None or cds_end is None:
        return mutations

    out: list[dict] = []
    for mut in mutations:
        if mut.get("type") != "substitution":
            out.append(mut)
            continue
        pos = mut.get("position")
        if pos is None or pos < cds_start or pos > cds_end:
            out.append(mut)
            continue
        codon_idx = (pos - cds_start) // 3
        frame_start = cds_start + codon_idx * 3
        if frame_start + 2 > cds_end:
            out.append(mut)
            continue
        # ref_seq is 0-indexed; ref positions are 1-based
        ref_codon = ref_seq[frame_start - 1:frame_start + 2]
        qry_codon_chars = []
        for p in range(frame_start, frame_start + 3):
            if p == pos:
                qry_codon_chars.append(mut.get("queryBase") or "N")
            else:
                qry_codon_chars.append(cds_positions.get(p, ref_seq[p - 1]))
        qry_codon = "".join(qry_codon_chars)
        ref_aa = translate_codon(ref_codon)
        qry_aa = translate_codon(qry_codon)
        mut_copy = dict(mut)
        if ref_aa is not None and qry_aa is not None and ref_aa == qry_aa:
            mut_copy["effect"] = "synonymous"
        out.append(mut_copy)
    return out
```

- [ ] **Step 1.3.4: Wire it into `analyze_sample`**

In `analyze_sample` (around the existing `mutation_rows = extract_mutations(...)` block), replace:

```python
mutation_rows = extract_mutations(ref_g, qry_g, ref2_s, ref_len)
mutations = [
    {
        "position": row.get("ref_pos"),
        "refBase": row.get("ref_base"),
        "queryBase": row.get("qry_base"),
        "type": str(row.get("type", "")).lower(),
        "effect": "",
    }
    for row in mutation_rows
]
```

with:

```python
mutation_rows = extract_mutations(ref_g, qry_g, ref2_s, ref_len)
mutations = [
    {
        "position": row.get("ref_pos"),
        "refBase": row.get("ref_base"),
        "queryBase": row.get("qry_base"),
        "type": str(row.get("type", "")).lower(),
        "effect": "",
    }
    for row in mutation_rows
]
mutations = label_synonymous_mutations(
    mutations, ref_seq=ref_seq, cds_start=cds_start, cds_end=cds_end,
    cds_positions=cds_positions,
)
```

- [ ] **Step 1.3.5: Run tests**

Run: `pytest tests/test_alignment_rules.py -v -k synonymous`
Expected: 3 passed.

Run: `pytest tests/test_alignment_stdout_clean.py -v`
Expected: still passes.

- [ ] **Step 1.3.6: Commit**

```bash
git add core/alignment.py tests/test_alignment_rules.py
git commit -m "feat(alignment): label synonymous substitutions in mutations list"
```

---

### Task 1.4: Dual-read consensus demotion

**Files:**
- Modify: `core/alignment.py` — `analyze_dataset` dual-read merge section (lines ~836–904)
- Modify: `tests/test_alignment_rules.py`

- [ ] **Step 1.4.1: Write failing dual-read test**

```python
# append to tests/test_alignment_rules.py
from core.alignment import apply_dual_read_consensus


def test_single_read_mutation_demoted_when_other_read_covers_with_ref_base():
    best = {
        "sid": "C123-1",
        "mutations": [
            {"position": 200, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {200: "T", 201: "G", 202: "C"},
    }
    other = {
        "sid": "C123-1",
        "mutations": [],
        "_cds_positions": {200: "A", 199: "C"},  # same position, ref base
    }
    merged = apply_dual_read_consensus(best, [other])
    assert merged["mutations"][0]["effect"] == "single_read"


def test_consensus_mutation_keeps_effect_unchanged():
    best = {
        "sid": "C123-1",
        "mutations": [
            {"position": 200, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {200: "T"},
    }
    other = {
        "sid": "C123-1",
        "mutations": [
            {"position": 200, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {200: "T"},
    }
    merged = apply_dual_read_consensus(best, [other])
    assert merged["mutations"][0]["effect"] == ""


def test_mutation_uncovered_by_other_read_not_demoted():
    best = {
        "sid": "C123-1",
        "mutations": [
            {"position": 500, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {500: "T"},
    }
    other = {
        "sid": "C123-1",
        "mutations": [],
        "_cds_positions": {},  # other read never covers pos 500
    }
    merged = apply_dual_read_consensus(best, [other])
    assert merged["mutations"][0]["effect"] == ""
```

- [ ] **Step 1.4.2: Run to confirm failure**

Run: `pytest tests/test_alignment_rules.py -v -k dual_read`
Expected: ImportError.

- [ ] **Step 1.4.3: Implement `apply_dual_read_consensus`**

Add to `core/alignment.py`:

```python
def apply_dual_read_consensus(best: dict, others: list[dict]) -> dict:
    """For each substitution in `best`, if the same ref position is covered
    by at least one other read with the *reference* base, demote it to
    single_read. Mutates nothing; returns a shallow copy of `best`."""
    best_copy = dict(best)
    mutations = list(best_copy.get("mutations", []))
    new_muts: list[dict] = []
    for mut in mutations:
        if mut.get("type") != "substitution" or mut.get("effect"):
            new_muts.append(mut)
            continue
        pos = mut.get("position")
        ref_base = (mut.get("refBase") or "").upper()
        demote = False
        for other in others:
            other_base = (other.get("_cds_positions") or {}).get(pos)
            if other_base and other_base.upper() == ref_base:
                demote = True
                break
        if demote:
            mut = dict(mut)
            mut["effect"] = "single_read"
        new_muts.append(mut)
    best_copy["mutations"] = new_muts
    return best_copy
```

- [ ] **Step 1.4.4: Wire into `analyze_dataset` dual-read branch**

In `analyze_dataset`, inside the `else:` branch that handles `len(entries) > 1` (around line 853), after the `best = dict(entries_sorted[0])` and before `best.pop("_cds_positions", None)`, add:

```python
best = apply_dual_read_consensus(best, entries_sorted[1:])
```

- [ ] **Step 1.4.5: Run tests**

Run: `pytest tests/test_alignment_rules.py -v -k dual_read`
Expected: 3 passed.

- [ ] **Step 1.4.6: Commit**

```bash
git add core/alignment.py tests/test_alignment_rules.py
git commit -m "feat(alignment): dual-read consensus demotes single-read substitutions"
```

---

### Task 1.5: Quality-gated bucket decision

**Files:**
- Modify: `core/alignment.py` — `analyze_sample` return dict adds `bucket`
- Modify: `tests/test_alignment_rules.py`

The UI already renders four buckets. The engine must emit one explicitly so the UI doesn't re-derive it.

- [ ] **Step 1.5.1: Write failing bucket tests**

```python
# append to tests/test_alignment_rules.py
from core.alignment import decide_bucket


def test_bucket_untested_when_coverage_below_threshold():
    assert decide_bucket(
        cds_coverage=0.3, avg_qry_quality=30.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "untested"


def test_bucket_untested_when_avg_quality_very_low():
    assert decide_bucket(
        cds_coverage=0.9, avg_qry_quality=10.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "untested"


def test_bucket_wrong_when_aa_changes_present():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=False,
        aa_changes=["K171M"], mutations=[], has_single_read=False,
    ) == "wrong"


def test_bucket_wrong_when_frameshift():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=True,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "wrong"


def test_bucket_uncertain_when_single_read_mutations_present():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=True,
    ) == "uncertain"


def test_bucket_uncertain_for_mid_coverage():
    assert decide_bucket(
        cds_coverage=0.6, avg_qry_quality=35.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "uncertain"


def test_bucket_ok_for_clean_sample():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "ok"
```

- [ ] **Step 1.5.2: Run to confirm failure**

Run: `pytest tests/test_alignment_rules.py -v -k bucket`
Expected: ImportError.

- [ ] **Step 1.5.3: Implement `decide_bucket`**

Also extend the bucket tests so every call site passes `cds_start`/`cds_end` — update `tests/test_alignment_rules.py` bucket tests to supply `cds_start=100, cds_end=1000` (or similar) where mutations are involved; other tests can omit them (defaults to `None`).

Add to `core/alignment.py` near the rule-tuning constants (next to `EDGE_IGNORE_BP`):

```python
COVERAGE_UNTESTED = 0.4
COVERAGE_OK = 0.8
QUALITY_UNTESTED = 15.0
QUALITY_UNCERTAIN = 25.0


def decide_bucket(cds_coverage: float,
                  avg_qry_quality: float | None,
                  frameshift: bool,
                  aa_changes: list[str],
                  mutations: list[dict],
                  has_single_read: bool,
                  cds_start: int | None = None,
                  cds_end: int | None = None) -> str:
    q = avg_qry_quality if avg_qry_quality is not None else 0.0
    if cds_coverage < COVERAGE_UNTESTED or q < QUALITY_UNTESTED:
        return "untested"
    if aa_changes or frameshift:
        return "wrong"
    if has_single_read or cds_coverage < COVERAGE_OK or q < QUALITY_UNCERTAIN:
        return "uncertain"
    hard_subs = [
        m for m in mutations
        if m.get("type") == "substitution"
        and m.get("effect") not in ("synonymous", "single_read")
        and not is_edge_ignored(m.get("position"), cds_start, cds_end)
    ]
    if len(hard_subs) >= 2:
        return "wrong"
    return "ok"
```

- [ ] **Step 1.5.4: Wire `decide_bucket` into the dataset-level post-processing**

`analyze_sample` can't know `has_single_read` until dual-read consensus has run, so compute the bucket **after** `apply_dual_read_consensus` in `analyze_dataset`. For single-entry samples, call `decide_bucket` with `has_single_read=False`.

In `analyze_dataset`, both branches of the merge:

```python
# in single-entry branch, after best.pop("_cds_positions", None):
best["bucket"] = decide_bucket(
    cds_coverage=best.get("cds_coverage") or 0.0,
    avg_qry_quality=best.get("avg_qry_quality"),
    frameshift=bool(best.get("frameshift")),
    aa_changes=best.get("aa_changes") or [],
    mutations=best.get("mutations") or [],
    has_single_read=False,
    cds_start=best.get("cds_start"),
    cds_end=best.get("cds_end"),
)
results.append(best)
```

```python
# in dual-read branch, after best = apply_dual_read_consensus(...)
has_single_read = any(
    m.get("effect") == "single_read" for m in best.get("mutations", [])
)
best.pop("_cds_positions", None)
best["bucket"] = decide_bucket(
    cds_coverage=best.get("cds_coverage") or 0.0,
    avg_qry_quality=best.get("avg_qry_quality"),
    frameshift=bool(best.get("frameshift")),
    aa_changes=best.get("aa_changes") or [],
    mutations=best.get("mutations") or [],
    has_single_read=has_single_read,
    cds_start=best.get("cds_start"),
    cds_end=best.get("cds_end"),
)
results.append(best)
```

- [ ] **Step 1.5.5: Update `bucketSampleStatus` in frontend utils to prefer `sample.bucket`**

**File:** `src/components/workbench/utils.ts`

Find `bucketSampleStatus` and change it to prefer an explicit `sample.bucket` field:

```ts
export function bucketSampleStatus(sample: WorkbenchSample): "ok" | "wrong" | "uncertain" | "untested" {
  if (sample.bucket === "ok" || sample.bucket === "wrong"
      || sample.bucket === "uncertain" || sample.bucket === "untested") {
    return sample.bucket;
  }
  // …existing fallback logic unchanged
}
```

Also add `bucket?: "ok" | "wrong" | "uncertain" | "untested";` to `WorkbenchSample` in `src/components/workbench/types.ts`.

- [ ] **Step 1.5.6: Run tests**

Run: `pytest tests/test_alignment_rules.py -v`
Expected: all rule tests pass (edge + synonymous + dual_read + bucket).

Run: `npm run test:js`
Expected: no regressions.

- [ ] **Step 1.5.7: Commit**

```bash
git add core/alignment.py tests/test_alignment_rules.py src/components/workbench/utils.ts src/components/workbench/types.ts
git commit -m "feat(alignment): quality-gated bucket decision + frontend prefers sample.bucket"
```

---

### Task 1.6: Calibrate runner + ≥ 90 % match rate verification

**Files:**
- Modify: `scripts/calibrate.py`

- [ ] **Step 1.6.1: Add the diff runner**

Append to `scripts/calibrate.py`:

```python
def aa_set(aa_changes) -> set[str]:
    return {x.strip() for x in (aa_changes or []) if x and isinstance(x, str)}


def summarize_diff(
    truth: dict[str, TruthRecord],
    engine: dict[str, dict],
) -> dict:
    agree = 0
    total = 0
    mismatches: list[dict] = []
    for sid, rec in truth.items():
        total += 1
        sample = engine.get(sid)
        actual_bucket = (sample or {}).get("bucket") or "unknown"
        actual_aa = aa_set((sample or {}).get("aa_changes"))
        expected_aa = aa_set(rec.aa)
        bucket_match = (rec.status == actual_bucket)
        aa_match = (rec.status != "wrong") or (expected_aa == actual_aa) or not expected_aa
        if bucket_match and aa_match:
            agree += 1
        else:
            mismatches.append({
                "sid": sid,
                "expected": rec.status,
                "actual": actual_bucket,
                "expected_aa": sorted(expected_aa),
                "actual_aa": sorted(actual_aa),
                "note": rec.note,
            })
    return {"agree": agree, "total": total, "mismatches": mismatches}


def run_engine(dataset: str, data_dir: Path) -> dict[str, dict]:
    from core.alignment import analyze_dataset  # lazy import
    results = analyze_dataset(dataset, data_dir)
    return {r["sid"]: r for r in results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calibrate")
    ap.add_argument("--dataset", required=True, choices=["base", "pro", "promax"])
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--truth", type=Path,
                    help="Override truth file path (default: truth/result[_pro|_promax].txt)")
    args = ap.parse_args(argv)

    if args.truth:
        truth_path = args.truth
    else:
        suffix = "" if args.dataset == "base" else f"_{args.dataset}"
        truth_path = Path("truth") / f"result{suffix}.txt"

    if not truth_path.exists():
        print(f"Truth file missing: {truth_path}", file=sys.stderr)
        return 2

    truth_records = load_truth_file(truth_path)
    truth = {r.sid: r for r in truth_records}

    try:
        engine = run_engine(args.dataset, args.data_dir)
    except FileNotFoundError as exc:
        print(f"Engine input missing: {exc}", file=sys.stderr)
        return 3

    summary = summarize_diff(truth, engine)
    print(f"Match rate: {summary['agree']}/{summary['total']} "
          f"= {summary['agree'] / summary['total']:.1%}")
    for m in summary["mismatches"]:
        print(f"  {m['sid']}: expected={m['expected']} actual={m['actual']} "
              f"exp_aa={m['expected_aa']} act_aa={m['actual_aa']} note={m['note']}")
    return 0 if summary["agree"] / summary["total"] >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 1.6.2: Run calibrate against the base dataset**

Run: `python -m scripts.calibrate --dataset base`
Expected: match rate ≥ 90 %. If below, read the mismatches, adjust `EDGE_IGNORE_BP`, `COVERAGE_*`, or `QUALITY_*` constants (one at a time), re-run. **Do not** expand the rule surface — tune the existing knobs.

- [ ] **Step 1.6.3: Run calibrate against the pro dataset**

Run: `python -m scripts.calibrate --dataset pro`
Expected: match rate ≥ 90 % on the 7 labeled pro samples (or justify any unresolved mismatch in the commit message).

- [ ] **Step 1.6.4: Run calibrate against the promax dataset (secondary)**

Run: `python -m scripts.calibrate --dataset promax`
Expected: either ≥ 90 %, or an explicit `Engine input missing` / low-count note. Promax is secondary — **don't** tune against it at the cost of base/pro.

- [ ] **Step 1.6.5: Full test sweep**

Run: `npm run test` (both suites)
Expected: all green.

- [ ] **Step 1.6.6: Commit**

```bash
git add scripts/calibrate.py core/alignment.py
git commit -m "feat(calibrate): truth-set diff runner; base+pro match ≥ 90%"
```

---

## Section 2 — DetailDrawer + compact row

### Task 2.1: Compact-row helper + tests

**Files:**
- Create: `src/lib/workbench/compactRow.ts`
- Create: `tests/test_compact_row.mjs`

- [ ] **Step 2.1.1: Write failing tests**

```js
// tests/test_compact_row.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { compactRowView } from "../src/lib/workbench/compactRow.js";

test("empty aa_changes collapses to dash sentinel", () => {
  const view = compactRowView({ id: "C1-1", aa_changes: [] });
  assert.deepEqual(view.aaPills, []);
  assert.equal(view.aaOverflow, 0);
});

test("up to 3 aa changes render as pills", () => {
  const view = compactRowView({ id: "C1-1", aa_changes: ["K171M", "L334S", "Q131T"] });
  assert.deepEqual(view.aaPills, ["K171M", "L334S", "Q131T"]);
  assert.equal(view.aaOverflow, 0);
});

test("more than 3 aa changes show +N overflow", () => {
  const view = compactRowView({
    id: "C1-1",
    aa_changes: ["K171M", "L334S", "Q131T", "R200W", "V300A"],
  });
  assert.deepEqual(view.aaPills, ["K171M", "L334S", "Q131T"]);
  assert.equal(view.aaOverflow, 2);
});

test("string aa_changes json is parsed", () => {
  const view = compactRowView({ id: "C1-1", aa_changes: '["K171M","L334S"]' });
  assert.deepEqual(view.aaPills, ["K171M", "L334S"]);
});

test("synonymous mutations feed the mutation type set", () => {
  const view = compactRowView({
    id: "C1-1",
    aa_changes: [],
    mutations: [
      { type: "substitution", effect: "synonymous" },
      { type: "insertion", effect: "" },
    ],
  });
  assert.ok(view.mutationTypes.includes("synonymous"));
  assert.ok(view.mutationTypes.includes("insertion"));
});
```

- [ ] **Step 2.1.2: Run to confirm failure**

Run: `npm run test:js`
Expected: module-not-found error.

- [ ] **Step 2.1.3: Implement helper**

```ts
// src/lib/workbench/compactRow.ts
import type { WorkbenchSample } from "../../components/workbench/types";

const MAX_PILLS = 3;

function parseAa(value: WorkbenchSample["aa_changes"]): string[] {
  if (Array.isArray(value)) return value.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
    } catch {
      return value.trim() ? [value.trim()] : [];
    }
  }
  return [];
}

export interface CompactRowView {
  aaPills: string[];
  aaOverflow: number;
  mutationTypes: string[];
}

export function compactRowView(sample: WorkbenchSample): CompactRowView {
  const aa = parseAa(sample.aa_changes);
  const pills = aa.slice(0, MAX_PILLS);
  const overflow = Math.max(0, aa.length - MAX_PILLS);
  const mutTypes = new Set<string>();
  for (const m of sample.mutations ?? []) {
    if (m?.type) mutTypes.add(String(m.type).toLowerCase());
    if (m?.effect) mutTypes.add(String(m.effect).toLowerCase());
  }
  return { aaPills: pills, aaOverflow: overflow, mutationTypes: Array.from(mutTypes) };
}
```

- [ ] **Step 2.1.4: Run tests**

Run: `npm run test:js`
Expected: 5 passes for this file.

- [ ] **Step 2.1.5: Commit**

```bash
git add src/lib/workbench/compactRow.ts tests/test_compact_row.mjs
git commit -m "feat(workbench): compact-row view helper with aa pills + overflow"
```

---

### Task 2.2: DetailDrawer component

**Files:**
- Create: `src/components/workbench/DetailDrawer.tsx`
- Create: `src/components/workbench/DetailDrawer.css`

- [ ] **Step 2.2.1: Write component**

```tsx
// src/components/workbench/DetailDrawer.tsx
import { Suspense, lazy, useEffect, useRef } from "react";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, formatPercent } from "./utils";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import "./DetailDrawer.css";

const ChromatogramCanvas = lazy(async () => {
  const mod = await import("./ChromatogramCanvas");
  return { default: mod.ChromatogramCanvas };
});

interface Props {
  sample: WorkbenchSample | null;
  language: AppLanguage;
  onClose(): void;
}

function parseAa(v: WorkbenchSample["aa_changes"]): string[] {
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
  if (typeof v === "string") {
    try { const p = JSON.parse(v); if (Array.isArray(p)) return p.filter((x): x is string => typeof x === "string"); } catch { return v.trim() ? [v.trim()] : []; }
  }
  return [];
}

function chromatogramFrom(sample: WorkbenchSample) {
  if (!sample.traces_a || !sample.traces_t || !sample.traces_g || !sample.traces_c || !sample.query_sequence) return null;
  return {
    traces: { A: sample.traces_a, T: sample.traces_t, G: sample.traces_g, C: sample.traces_c },
    quality: sample.quality || [],
    baseCalls: sample.query_sequence,
    base_locations: sample.base_locations || [],
    mixed_peaks: sample.mixed_peaks || [],
  };
}

export function DetailDrawer({ sample, language, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!sample) return;
    closeRef.current?.focus();
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sample, onClose]);

  if (!sample) return null;
  const bucket = bucketSampleStatus(sample);
  const aa = parseAa(sample.aa_changes);
  const chrom = chromatogramFrom(sample);
  const muts = Array.isArray(sample.mutations) ? sample.mutations : [];

  return (
    <aside className="detail-drawer" role="dialog" aria-modal="false" aria-label={sample.id}>
      <header className="detail-drawer-head">
        <span className="detail-drawer-sid">{sample.id}</span>
        <span className={`detail-drawer-status status-${bucket}`}>{t(language, `wb.status.${bucket}`)}</span>
        <button ref={closeRef} className="detail-drawer-close" onClick={onClose} aria-label={t(language, "wb.drawer.close")}>×</button>
      </header>
      <div className="detail-drawer-body">
        <section className="detail-drawer-metrics">
          <article><span>{t(language, "table.clone")}</span><strong>{sample.clone || "-"}</strong></article>
          <article><span>{t(language, "table.orientation")}</span><strong>{sample.orientation || "-"}</strong></article>
          <article><span>{t(language, "table.frameshift")}</span><strong>{sample.frameshift ? t(language, "table.yes") : t(language, "table.no")}</strong></article>
          <article><span>{t(language, "table.avgQ")}</span><strong>{typeof (sample.avg_qry_quality ?? sample.avg_quality) === "number" ? (sample.avg_qry_quality ?? sample.avg_quality)!.toFixed(1) : "-"}</strong></article>
          <article><span>{t(language, "table.identity")}</span><strong>{formatPercent(sample.identity)}</strong></article>
          <article><span>{t(language, "table.coverage")}</span><strong>{formatPercent(sample.cds_coverage ?? sample.coverage)}</strong></article>
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.aaChanges")}</h4>
          {aa.length ? <div className="detail-drawer-aa">{aa.join(" ")}</div> : <div className="detail-drawer-empty">{t(language, "table.noAa")}</div>}
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.mutationTable")}</h4>
          {muts.length ? (
            <table className="detail-drawer-table">
              <thead>
                <tr>
                  <th>{t(language, "table.pos")}</th>
                  <th>{t(language, "table.ref")}</th>
                  <th>{t(language, "table.query")}</th>
                  <th>{t(language, "table.type")}</th>
                  <th>{t(language, "table.effect")}</th>
                </tr>
              </thead>
              <tbody>
                {muts.map((m, i) => (
                  <tr key={i} className={m.effect === "synonymous" ? "is-synonymous" : m.effect === "single_read" ? "is-single-read" : undefined}>
                    <td>{m.position ?? "-"}</td>
                    <td>{m.refBase ?? "-"}</td>
                    <td>{m.queryBase ?? "-"}</td>
                    <td>{m.type ?? "-"}</td>
                    <td>{m.effect ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div className="detail-drawer-empty">{t(language, "table.noMutation")}</div>}
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.alignment")}</h4>
          <pre className="detail-drawer-aa">
            <div><strong>REF:</strong> {sample.aligned_ref_g || sample.ref_sequence || ""}</div>
            <div><strong>QRY:</strong> {sample.aligned_query_g || sample.query_sequence || ""}</div>
          </pre>
        </section>

        <section className="detail-drawer-section">
          <h4>{t(language, "table.chromatogram")}</h4>
          {chrom ? (
            <Suspense fallback={<div className="detail-drawer-empty">{t(language, "table.loadingChromatogram")}</div>}>
              <ChromatogramCanvas data={chrom} startPosition={1} endPosition={chrom.baseCalls.length} mutations={muts} />
            </Suspense>
          ) : <div className="detail-drawer-empty">{t(language, "table.noChromatogram")}</div>}
        </section>
      </div>
    </aside>
  );
}
```

- [ ] **Step 2.2.2: Add drawer CSS**

```css
/* src/components/workbench/DetailDrawer.css */
.detail-drawer {
  position: fixed;
  top: 16px;
  right: 16px;
  bottom: 16px;
  width: var(--drawer-width, 480px);
  max-width: calc(100vw - 32px);
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  box-shadow: var(--panel-shadow);
  display: flex;
  flex-direction: column;
  z-index: 50;
}
.detail-drawer-head {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--panel-border);
}
.detail-drawer-sid { font-weight: 700; }
.detail-drawer-close {
  margin-left: auto;
  background: transparent;
  border: none;
  font-size: 22px;
  cursor: pointer;
  color: var(--text-main);
}
.detail-drawer-body { overflow-y: auto; padding: 16px; }
.detail-drawer-section { margin-bottom: 20px; }
.detail-drawer-section h4 { margin: 0 0 8px; font-size: 13px; color: var(--text-muted); }
.detail-drawer-metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 20px; }
.detail-drawer-metrics article { background: var(--card-bg); border: 1px solid var(--card-border); padding: 8px 10px; border-radius: 8px; }
.detail-drawer-metrics article span { color: var(--text-muted); font-size: 12px; display: block; }
.detail-drawer-metrics article strong { font-size: 14px; }
.detail-drawer-aa { font-family: Consolas, Menlo, monospace; font-size: 12px; white-space: pre-wrap; word-break: break-all; background: var(--card-bg); border: 1px solid var(--card-border); padding: 10px; border-radius: 8px; }
.detail-drawer-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.detail-drawer-table th, .detail-drawer-table td { padding: 6px 8px; border-bottom: 1px solid var(--panel-border); text-align: left; }
.detail-drawer-table tr.is-synonymous td { color: var(--text-muted); }
.detail-drawer-table tr.is-single-read td { font-style: italic; color: var(--text-muted); }
.detail-drawer-empty { color: var(--text-muted); padding: 8px; font-size: 13px; }
```

- [ ] **Step 2.2.3: Add new i18n keys**

In `src/i18n.ts`, add to both the `zh` and `en` blocks:

```ts
// zh
"wb.drawer.close": "关闭",
"wb.density.compact": "紧凑",
"wb.density.detailed": "详细",

// en
"wb.drawer.close": "Close",
"wb.density.compact": "Compact",
"wb.density.detailed": "Detailed",
```

- [ ] **Step 2.2.4: Verify TypeScript compiles**

Run: `npm run build`
Expected: no type errors. The drawer is not wired up yet — compilation should still succeed because the module is unused but imports check out.

- [ ] **Step 2.2.5: Commit**

```bash
git add src/components/workbench/DetailDrawer.tsx src/components/workbench/DetailDrawer.css src/i18n.ts
git commit -m "feat(workbench): DetailDrawer component + i18n keys"
```

- [ ] **Step 2.2.6: Drag-to-resize handle + localStorage width**

Spec calls for a resizable drawer with width persisted. Add to `DetailDrawer.tsx`:

```tsx
const STORAGE_KEY = "bioagent.drawer.width.v1";
function loadWidth(): number { try { const v = parseInt(localStorage.getItem(STORAGE_KEY) || "", 10); return Number.isFinite(v) && v >= 320 && v <= 900 ? v : 480; } catch { return 480; } }

const [width, setWidth] = useState<number>(loadWidth);
useEffect(() => { try { localStorage.setItem(STORAGE_KEY, String(width)); } catch {} }, [width]);

function startDrag(e: React.MouseEvent) {
  e.preventDefault();
  const startX = e.clientX;
  const startW = width;
  function onMove(ev: MouseEvent) { setWidth(Math.max(320, Math.min(900, startW + (startX - ev.clientX)))); }
  function onUp() { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); }
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}
```

Add a handle element at the drawer's left edge:
```tsx
<div className="detail-drawer-resize" onMouseDown={startDrag} aria-hidden="true" />
```

Pass width via inline style: `<aside ... style={{ width }}>` (overrides `--drawer-width` fallback).

CSS:
```css
.detail-drawer-resize { position: absolute; left: -4px; top: 0; bottom: 0; width: 8px; cursor: ew-resize; }
```

Manual check: drag to resize, reload, width persists.

```bash
git add src/components/workbench/DetailDrawer.tsx src/components/workbench/DetailDrawer.css
git commit -m "feat(workbench): drawer drag-to-resize with persisted width"
```

---

### Task 2.3: Compact-row refactor of ResultsTable

**Files:**
- Modify: `src/components/workbench/ResultsTable.tsx`
- Modify: `src/components/workbench/ResultsWorkbench.css`

- [ ] **Step 2.3.1: Replace ResultsTable rendering with compact row**

Replace `src/components/workbench/ResultsTable.tsx` with:

```tsx
import { useEffect, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { WorkbenchSample } from "./types";
import { bucketSampleStatus, formatPercent, countSampleMutations } from "./utils";
import { compactRowView } from "../../lib/workbench/compactRow";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface Props {
  samples: WorkbenchSample[];
  language: AppLanguage;
  density: "compact" | "detailed";
  selectedId: string | null;
  onSelect(id: string): void;
  isFiltered?: boolean;
  onClearFilters?: () => void;
}

const ROW_COMPACT = 64;
const ROW_DETAILED = 88;
const OVERSCAN = 6;

export function ResultsTable({ samples, language, density, selectedId, onSelect, isFiltered, onClearFilters }: Props) {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const rowHeight = density === "compact" ? ROW_COMPACT : ROW_DETAILED;
  const virtualizer = useVirtualizer({
    count: samples.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => rowHeight,
    overscan: OVERSCAN,
    getItemKey: (i) => samples[i]?.id ?? i,
  });
  useEffect(() => { virtualizer.measure(); }, [density, samples.length]); // eslint-disable-line

  const items = virtualizer.getVirtualItems();
  return (
    <section className="results-table-panel" aria-label={t(language, "table.title")}>
      <div className="results-section-header results-section-header-compact">
        <div>
          <span className="results-kicker">{t(language, "table.kicker")}</span>
          <h3>{t(language, "table.title")}</h3>
        </div>
      </div>

      <div className="sample-details-list" ref={parentRef}>
        {samples.length === 0 ? (
          <div className="results-table-empty">
            {isFiltered ? (
              <>
                <strong>{t(language, "wb.empty.filtered")}</strong>
                {onClearFilters ? (
                  <button className="sample-toolbar-button" onClick={onClearFilters}>
                    {t(language, "wb.empty.clear")}
                  </button>
                ) : null}
              </>
            ) : (
              <>
                <strong>{t(language, "table.noDataTitle")}</strong>
                <span>{t(language, "table.noDataBody")}</span>
              </>
            )}
          </div>
        ) : (
          <div style={{ height: virtualizer.getTotalSize(), position: "relative", width: "100%" }}>
            {items.map((v) => {
              const sample = samples[v.index];
              if (!sample) return null;
              const status = bucketSampleStatus(sample);
              const view = compactRowView(sample);
              const isSelected = selectedId === sample.id;
              return (
                <button
                  key={v.key}
                  type="button"
                  data-index={v.index}
                  onClick={() => onSelect(sample.id)}
                  className={`sample-compact-row status-${status}${isSelected ? " is-selected" : ""}`}
                  style={{ position: "absolute", top: 0, left: 0, right: 0, height: rowHeight, transform: `translateY(${v.start}px)` }}
                >
                  <span className="sample-compact-sid" title={sample.id}>{sample.id}</span>
                  <span className={`sample-compact-status status-${status}`}>{t(language, `wb.status.${status}`)}</span>
                  <span className="sample-compact-aa">
                    {view.aaPills.length === 0 ? <span className="sample-compact-aa-empty">-</span>
                      : view.aaPills.map((p) => <span key={p} className="sample-compact-aa-pill">{p}</span>)}
                    {view.aaOverflow > 0 ? <span className="sample-compact-aa-overflow">+{view.aaOverflow}</span> : null}
                  </span>
                  <span className="sample-compact-metric">{formatPercent(sample.identity)}</span>
                  <span className="sample-compact-metric">{formatPercent(sample.cds_coverage ?? sample.coverage)}</span>
                  <span className="sample-compact-metric">{countSampleMutations(sample)}</span>
                  <span className="sample-compact-chevron" aria-hidden="true">›</span>
                  {density === "detailed" ? (
                    <span className="sample-compact-subline">
                      {sample.reason || sample.review_reason || sample.auto_reason || ""} · q{sample.avg_qry_quality ?? "-"} · {sample.orientation || "-"}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {samples.length > 0 ? (
        <div className="sample-list-footnote">
          {t(language, "table.showing", { visible: Math.min(samples.length, items.length || samples.length), total: samples.length })}
        </div>
      ) : null}
    </section>
  );
}
```

- [ ] **Step 2.3.2: Add compact-row CSS**

Append to `src/components/workbench/ResultsWorkbench.css`:

```css
.sample-compact-row {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr 1.6fr 0.7fr 0.7fr 0.5fr 16px;
  align-items: center;
  gap: 8px;
  padding: 8px 16px 8px 12px;
  border: 1px solid var(--panel-border);
  border-left-width: 3px;
  background: var(--card-bg);
  text-align: left;
  font: inherit;
  color: inherit;
  cursor: pointer;
  border-radius: 8px;
  margin-bottom: 6px;
  overflow: hidden;
  width: 100%;
}
.sample-compact-row.is-selected { outline: 2px solid var(--accent-primary); }
.sample-compact-row.status-ok    { border-left-color: var(--status-ok); }
.sample-compact-row.status-wrong { border-left-color: var(--status-wrong); }
.sample-compact-row.status-uncertain { border-left-color: var(--status-uncertain); }
.sample-compact-row.status-untested  { border-left-color: var(--status-untested); }
.sample-compact-aa { display: flex; gap: 4px; flex-wrap: nowrap; overflow: hidden; }
.sample-compact-aa-pill, .sample-compact-aa-overflow { font-size: 11px; padding: 2px 6px; background: var(--card-border); border-radius: 4px; white-space: nowrap; }
.sample-compact-aa-pill.is-missense { background: color-mix(in srgb, var(--status-wrong) 20%, var(--card-bg)); color: var(--status-wrong); }
.sample-compact-aa-pill.is-synonymous { background: color-mix(in srgb, var(--mutation-synonymous) 20%, var(--card-bg)); color: var(--mutation-synonymous); }
.sample-compact-aa-empty { color: var(--text-muted); }
.sample-compact-metric { font-variant-numeric: tabular-nums; color: var(--text-muted); }
.sample-compact-chevron { color: var(--text-muted); }
.sample-compact-subline { grid-column: 1 / -1; font-size: 11px; color: var(--text-muted); }
```

- [ ] **Step 2.3.3: Commit**

```bash
git add src/components/workbench/ResultsTable.tsx src/components/workbench/ResultsWorkbench.css
git commit -m "refactor(workbench): compact rows with aa pills; drop inline expand"
```

---

### Task 2.4: ResultsWorkbench wiring

**Files:**
- Modify: `src/components/workbench/ResultsWorkbench.tsx`

- [ ] **Step 2.4.1: Add selectedId + density state, mount drawer, replace expand buttons**

Apply this diff to `ResultsWorkbench.tsx`:

1. Add imports:
```ts
import { useState } from "react";
import { DetailDrawer } from "./DetailDrawer";
```

2. Inside component:
```ts
const [selectedId, setSelectedId] = useState<string | null>(null);
const [density, setDensity] = useState<"compact" | "detailed">("compact");
const selectedSample = selectedId ? visibleSamples.find((s) => s.id === selectedId) ?? null : null;
```

3. In the toolbar section (the `results-filter-row` is a good spot), add a density toggle:
```tsx
<div className="results-density-toggle" role="group" aria-label={t(language, "wb.density.detailed")}>
  {(["compact", "detailed"] as const).map((d) => (
    <button
      key={d}
      type="button"
      className={`results-filter-chip${density === d ? " is-active" : ""}`}
      onClick={() => setDensity(d)}
    >
      {t(language, `wb.density.${d}`)}
    </button>
  ))}
</div>
```

4. Pass new props to `ResultsTable`:
```tsx
<ResultsTable
  samples={visibleSamples}
  language={language}
  density={density}
  selectedId={selectedId}
  onSelect={setSelectedId}
  isFiltered={hasActiveControls}
  onClearFilters={reset}
/>
<DetailDrawer
  sample={selectedSample}
  language={language}
  onClose={() => setSelectedId(null)}
/>
```

5. When `visibleSamples` changes and the selected id is no longer visible, clear:
```ts
useEffect(() => {
  if (selectedId && !visibleSamples.some((s) => s.id === selectedId)) setSelectedId(null);
}, [visibleSamples, selectedId]);
```

- [ ] **Step 2.4.2: Drop the "Expand all / Collapse all" toolbar from ResultsTable**

The new `ResultsTable` already does not render expand-all / collapse-all buttons. Sanity-check nothing else references those strings.

Run: `grep -rn "expandAll\|collapseAll" src/ tests/`
Expected: only the i18n entries remain, which are now unused. Leave them (harmless), or remove them and related tests if found.

- [ ] **Step 2.4.3: Type-check + test**

Run: `npm run build`
Expected: no errors.

Run: `npm run test:js`
Expected: pass (compact-row tests included).

- [ ] **Step 2.4.4: Manual smoke check**

Run: `npm run dev` (in a second shell), click a row, press Escape, click a status filter — drawer should open, close on Esc, clear when filter hides the selected sample. If the UI has a dev-server-backed sample dataset, exercise the chromatogram lazy load at least once.

- [ ] **Step 2.4.5: Commit**

```bash
git add src/components/workbench/ResultsWorkbench.tsx
git commit -m "feat(workbench): mount DetailDrawer; density toggle replaces expand-all"
```

---

## Section 3 — Color tokens + chat rail

### Task 3.1: Status / mutation / accent tokens

**Files:**
- Modify: `src/styles.css`

- [ ] **Step 3.1.1: Add tokens under `:root`**

Inside the existing `:root` block in `src/styles.css`, append:

```css
/* Status (applies to bucket of a sample) */
--status-ok: #1d9c7d;
--status-wrong: #d23f5c;
--status-uncertain: #d89a2c;
--status-untested: #6a7a93;

/* Mutation types */
--mutation-sub: #2c7fb8;
--mutation-ins: #6f42c1;
--mutation-del: #a04050;
--mutation-frameshift: #c5453e;
--mutation-synonymous: #8a93a6;

/* Accents */
--accent-primary: #2c7fb8;
--accent-secondary: #1f9d8d;
```

- [ ] **Step 3.1.2: Add tokens under `[data-theme="dark"]`**

Inside the existing `:root[data-theme="dark"]` block, append:

```css
--status-ok: #4fd4a7;
--status-wrong: #ff6b80;
--status-uncertain: #f0b84a;
--status-untested: #9eb1d8;
--mutation-sub: #6f9bff;
--mutation-ins: #a07bf5;
--mutation-del: #ff7688;
--mutation-frameshift: #ff8177;
--mutation-synonymous: #9eb1d8;
--accent-primary: #6f9bff;
--accent-secondary: #4fd4a7;
```

- [ ] **Step 3.1.3: Add status pill CSS that references the tokens**

Append to `src/styles.css`:

```css
.sample-detail-status.status-ok, .sample-compact-status.status-ok { color: var(--status-ok); }
.sample-detail-status.status-wrong, .sample-compact-status.status-wrong { color: var(--status-wrong); }
.sample-detail-status.status-uncertain, .sample-compact-status.status-uncertain { color: var(--status-uncertain); }
.sample-detail-status.status-untested, .sample-compact-status.status-untested { color: var(--status-untested); }
.detail-drawer-status.status-ok { color: var(--status-ok); }
.detail-drawer-status.status-wrong { color: var(--status-wrong); }
.detail-drawer-status.status-uncertain { color: var(--status-uncertain); }
.detail-drawer-status.status-untested { color: var(--status-untested); }
```

- [ ] **Step 3.1.4: Remove hard-coded status colors**

Run: `grep -n "#[0-9a-f]\{6\}" src/components/panels/*.tsx src/components/workbench/*.css src/components/workbench/*.tsx`

For every hit, replace the bespoke hex with the nearest token. If a panel uses a hex only in one place (e.g. `AnalysisPanel.tsx` uses `#ff4040`) and the concept maps to a token, swap it. Where a true bespoke color is needed (e.g., a PDF report constant), leave it and note in commit message.

- [ ] **Step 3.1.5: Visual check**

Run: `npm run dev`, toggle theme, click through 2–3 samples of each status. Confirm pills and compact-row left-borders differ across buckets in both light and dark themes.

- [ ] **Step 3.1.6: Commit**

```bash
git add src/styles.css src/components/panels src/components/workbench
git commit -m "feat(styles): status/mutation/accent tokens (light + dark); apply to status pills"
```

---

### Task 3.2: ResultsCharts reads token palette

**Files:**
- Modify: `src/components/workbench/ResultsCharts.tsx`

- [ ] **Step 3.2.1: Replace hard-coded chart colors**

`ResultsCharts` currently uses either `--chart-primary-color` or hard-coded hex. Update the chart color prop to read the new tokens via `getComputedStyle`:

```ts
function readToken(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

const palette = {
  ok: readToken("--status-ok", "#1d9c7d"),
  wrong: readToken("--status-wrong", "#d23f5c"),
  uncertain: readToken("--status-uncertain", "#d89a2c"),
  untested: readToken("--status-untested", "#6a7a93"),
  primary: readToken("--accent-primary", "#2c7fb8"),
  secondary: readToken("--accent-secondary", "#1f9d8d"),
};
```

Plug `palette.*` into the recharts `fill` / `stroke` props for each series. For time-series averages, use `palette.primary` and `palette.secondary`.

- [ ] **Step 3.2.2: Re-read palette on theme change**

Because tokens change under `[data-theme="dark"]`, recompute the palette inside a `useMemo` keyed on a theme-change signal. Simplest: read `document.documentElement.dataset.theme` and key the memo on it, plus subscribe to a `themechange` custom event if one is dispatched elsewhere. If no theme-change event exists, page reload on theme toggle is acceptable for this round — note in commit message.

- [ ] **Step 3.2.3: Build + smoke**

Run: `npm run build`
Expected: no errors.

Run: `npm run dev`, toggle theme while on the results page, verify chart colors update or require a reload (note the behavior in commit).

- [ ] **Step 3.2.4: Commit**

```bash
git add src/components/workbench/ResultsCharts.tsx
git commit -m "feat(workbench): charts read status/accent palette from CSS tokens"
```

---

### Task 3.3: Chat rail 3-state helper + tests

**Files:**
- Create: `src/lib/ui/chatRailState.ts`
- Create: `tests/test_chat_rail_state.mjs`

- [ ] **Step 3.3.1: Write failing tests**

```js
// tests/test_chat_rail_state.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { nextRailState, loadRailState, saveRailState } from "../src/lib/ui/chatRailState.js";

test("cycles wide → narrow → hidden → wide", () => {
  assert.equal(nextRailState("wide"), "narrow");
  assert.equal(nextRailState("narrow"), "hidden");
  assert.equal(nextRailState("hidden"), "wide");
});

test("load from storage returns wide by default", () => {
  const store = {};
  const fake = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
  assert.equal(loadRailState(fake), "wide");
});

test("round-trips through save/load", () => {
  const store = {};
  const fake = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
  saveRailState("narrow", fake);
  assert.equal(loadRailState(fake), "narrow");
});

test("invalid stored value falls back to wide", () => {
  const fake = { getItem: () => "lol", setItem: () => {} };
  assert.equal(loadRailState(fake), "wide");
});
```

- [ ] **Step 3.3.2: Run to confirm failure**

Run: `npm run test:js`
Expected: module-not-found.

- [ ] **Step 3.3.3: Implement helper**

```ts
// src/lib/ui/chatRailState.ts
export type ChatRailState = "wide" | "narrow" | "hidden";

const STORAGE_KEY = "bioagent.chatRail.v1";
const ORDER: ChatRailState[] = ["wide", "narrow", "hidden"];

interface Storage { getItem(k: string): string | null; setItem(k: string, v: string): void; }

export function nextRailState(state: ChatRailState): ChatRailState {
  const i = ORDER.indexOf(state);
  return ORDER[(i + 1) % ORDER.length];
}

export function loadRailState(store: Storage = (typeof localStorage !== "undefined" ? localStorage : { getItem: () => null, setItem: () => {} })): ChatRailState {
  const raw = store.getItem(STORAGE_KEY);
  return (ORDER.includes(raw as ChatRailState) ? (raw as ChatRailState) : "wide");
}

export function saveRailState(state: ChatRailState, store: Storage = (typeof localStorage !== "undefined" ? localStorage : { getItem: () => null, setItem: () => {} })): void {
  store.setItem(STORAGE_KEY, state);
}
```

- [ ] **Step 3.3.4: Run tests**

Run: `npm run test:js`
Expected: 4 new passes.

- [ ] **Step 3.3.5: Commit**

```bash
git add src/lib/ui/chatRailState.ts tests/test_chat_rail_state.mjs
git commit -m "feat(ui): chat rail state helper with 3-step cycle + persistence"
```

---

### Task 3.4: Chat rail UI wiring

**Files:**
- Modify: `src/App.tsx` (or the extracted ChatPanel component, whichever owns the rail)
- Modify: `src/styles.css`
- Modify: `src/i18n.ts`

- [ ] **Step 3.4.1: Add state + toggle**

In `App.tsx`:
```ts
import { loadRailState, nextRailState, saveRailState, type ChatRailState } from "./lib/ui/chatRailState";

const [railState, setRailState] = useState<ChatRailState>(() => loadRailState());
useEffect(() => { saveRailState(railState); }, [railState]);

function cycleRail() { setRailState((s) => nextRailState(s)); }
```

Apply a class to the shell:
```tsx
<div className={`app-shell rail-${railState}`}>
  ...
  <section className="chat-panel">
    <button className="chat-rail-toggle" onClick={cycleRail} aria-label={t(language, `wb.chatRail.${railState}`)}>
      {t(language, `wb.chatRail.${railState}`)}
    </button>
    ...
  </section>
  ...
</div>
```

- [ ] **Step 3.4.2: Rail-state CSS**

In `src/styles.css`, replace the static `grid-template-columns` for `.app-shell` with variants:
```css
.app-shell.rail-wide   { grid-template-columns: minmax(520px, 38%) 1fr; }
.app-shell.rail-narrow { grid-template-columns: minmax(300px, 26%) 1fr; }
.app-shell.rail-hidden { grid-template-columns: 40px 1fr; }

.app-shell.rail-hidden .chat-panel > *:not(.chat-rail-toggle) { display: none; }
.app-shell.rail-hidden .chat-panel { align-items: center; padding: 8px 0; }

.chat-rail-toggle {
  position: absolute;
  top: 12px;
  right: 12px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  color: var(--text-main);
  border-radius: 6px;
  padding: 4px 8px;
  cursor: pointer;
}
```

- [ ] **Step 3.4.3: i18n**

```ts
// zh
"wb.chatRail.wide": "宽",
"wb.chatRail.narrow": "窄",
"wb.chatRail.hidden": "隐藏",
// en
"wb.chatRail.wide": "Wide",
"wb.chatRail.narrow": "Narrow",
"wb.chatRail.hidden": "Hidden",
```

- [ ] **Step 3.4.4: Manual verification**

Run: `npm run dev`, cycle the toggle. Verify the results column grows when hidden, drawer still opens cleanly on the right without overlapping, state survives a reload.

- [ ] **Step 3.4.5: Commit**

```bash
git add src/App.tsx src/styles.css src/i18n.ts
git commit -m "feat(ui): 3-state chat rail with localStorage persistence"
```

---

## Wrap-up

- [ ] **Step W.1: Full sweep**

Run: `npm run test` — both suites green.
Run: `npm run build` — type-check + build succeeds.
Run: `python -m scripts.calibrate --dataset base` — still ≥ 90 %.
Run: `python -m scripts.calibrate --dataset pro` — still ≥ 90 %.

- [ ] **Step W.2: Verification-before-completion skill**

Use superpowers:verification-before-completion to double-check every claim in the PR description with an actual command. No green checkmark without evidence.

- [ ] **Step W.3: Request code review**

Use superpowers:requesting-code-review, providing the plan path, spec path, and a list of the commits.

- [ ] **Step W.4: Finishing the branch**

Use superpowers:finishing-a-development-branch to decide merge / PR / cleanup.

---

## Non-goals reminder

- No runtime truth loader.
- No regression harness.
- No ML adjudication.
- No user-adjustable thresholds.
- No design-system rewrite beyond status / mutation / accent.
