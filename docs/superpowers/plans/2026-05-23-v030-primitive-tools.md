# v0.3.0 Primitive Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose six atomic bio-operations as MCP tools so the agent can answer file/sequence questions without requiring `register_dataset` first.

**Architecture:** A new `bio_primitives.py` module wraps the lower-level functions already living in `core/alignment.py` (and reimplements two tiny path-inspection helpers currently only in Electron) with tool-friendly signatures (str paths, structured dict returns, defensive errors). `tools_register.py` imports these and calls `register_tool()` for each. No changes to the Electron/React layer are required for this plan — all six tools execute inside the existing Python sidecar over the existing MCP JSON-RPC channel that `electron/agent_harness.mjs` already speaks.

**Tech Stack:** Python 3.11, biopython (SeqIO/PairwiseAligner), pytest, existing MCP registry in `bioagent/mcp_tools.py`.

**Out of scope for this plan** (separate v0.3.0 sub-plans):
- Streaming output + history compaction in agent loop
- Tool call UI cards / PreToolUse confirmation hooks
- App icon, macOS x64 build matrix

---

## File Structure

| Path | Purpose | Disposition |
|---|---|---|
| `src-python/bioagent/bio_primitives.py` | Pure functions wrapping core.alignment with tool-friendly signatures (str args, dict returns, error envelopes) | **CREATE** |
| `src-python/bioagent/tools_register.py` | Add 6 `register_tool(...)` calls at the bottom of `register_initial_tools()` | **MODIFY** |
| `tests/test_bio_primitives.py` | Unit tests for each wrapper function | **CREATE** |
| `tests/test_tools_register_primitives.py` | Integration tests via `mcp_tools.call_tool(...)` to verify registration + dispatch | **CREATE** |
| `electron/agent_harness.mjs` | Append a 1-paragraph "Primitive tools" section to `buildSystemPrompt()` so the LLM knows when to call them | **MODIFY** (≤20 lines) |

**Why a separate `bio_primitives.py` module rather than registering directly from `core/alignment.py`:** `core/alignment.py` uses `pathlib.Path`, returns tuples, and raises raw exceptions. MCP tools must accept JSON-serializable str args, return `{ok, ...}` dicts, and never raise (callers expect graceful error envelopes). Wrapping in a thin adapter layer keeps `core/alignment.py` pure and lets us unit-test the wrappers independently.

---

## Tool Catalog (target shapes)

| Tool name | Args | Returns on success | Wraps |
|---|---|---|---|
| `inspect_path` | `paths: string \| string[]` | `{ ok, items: [{ path, type, ext?, size?, dataset_hint? }] }` | (reimplemented in Python) |
| `read_sequence_file` | `path: string, max_chars?: int = 2000` | `{ ok, format, sequence_preview, length, features?, quality_stats? }` | `read_ab1_sequence`, `load_genbank`, `SeqIO.read` (FASTA) |
| `compare_sequences` | `ref_seq: string, query_seq: string, ref_cds_start?: int, ref_cds_end?: int` | `{ ok, identity, matches, length, mutations, aa_changes?, frameshift? }` | `build_aligner`, `pick_best_orientation`, `extract_mutations`, `aa_changes_from_cds` |
| `translate_sequence` | `dna: string, frame?: 0\|1\|2 = 0` | `{ ok, protein, stop_codons: [int] }` | `translate_codon` (loop) |
| `analyze_files_adhoc` | `ab1_dir: string, gb_dir: string, output_dir?: string` | `{ ok, analysis_id, samples_count, summary }` | `analyze_dirs` |
| `export_analysis_html` | `analysis_id: string, sample_id: string, output_path?: string` | `{ ok, html_path }` | `write_alignment_html` + lookup in `_ANALYSIS_DETAILS` |

All error cases must return `{ok: false, error: "<reason>"}` rather than raising.

---

## Task 1: Scaffold the new module + first tool (`translate_sequence`)

Start with the smallest tool to verify the wrap-then-register pattern end-to-end before tackling complex ones.

**Files:**
- Create: `src-python/bioagent/bio_primitives.py`
- Create: `tests/test_bio_primitives.py`

- [ ] **Step 1.1: Write the failing test for `translate_sequence`**

Create `tests/test_bio_primitives.py`:

```python
"""Unit tests for bio_primitives wrappers used by the MCP tool layer."""
from bioagent.bio_primitives import translate_sequence


def test_translate_sequence_full_orf():
    result = translate_sequence(dna="ATGGAATAA")  # M-E-*
    assert result["ok"] is True
    assert result["protein"] == "ME*"
    assert result["stop_codons"] == [3]  # stop at amino acid position 3 (1-based)


def test_translate_sequence_handles_frame_shift_arg():
    result = translate_sequence(dna="GATGGAATAA", frame=1)  # frame 1: ATG GAA TAA
    assert result["ok"] is True
    assert result["protein"] == "ME*"


def test_translate_sequence_rejects_empty():
    result = translate_sequence(dna="")
    assert result["ok"] is False
    assert "empty" in result["error"].lower()


def test_translate_sequence_handles_ambiguous_codon():
    # 'NNN' should produce 'X' (unknown), not crash
    result = translate_sequence(dna="NNNATG")
    assert result["ok"] is True
    assert result["protein"].startswith("X")
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/test_bio_primitives.py -v`
Expected: `ModuleNotFoundError: No module named 'bioagent.bio_primitives'`

- [ ] **Step 1.3: Create the module with `translate_sequence`**

Create `src-python/bioagent/bio_primitives.py`:

```python
"""Tool-friendly wrappers over core.alignment + path inspection helpers.

Each function in this module is intended to be wired into the MCP tool
registry. They share these conventions:

- All path args are accepted as ``str`` (not ``pathlib.Path``) because tool
  arguments arrive as JSON.
- All returns are JSON-serializable dicts with an ``ok: bool`` flag plus
  either data fields or an ``error: str`` field. Functions must NOT raise
  for expected error conditions (missing file, malformed input). They may
  still raise for true bugs.
- These wrappers contain no biology logic beyond translating signatures.
  Domain logic lives in ``core/alignment.py``.
"""
from __future__ import annotations

from core.alignment import translate_codon


def translate_sequence(*, dna: str, frame: int = 0) -> dict:
    if not dna:
        return {"ok": False, "error": "dna sequence is empty"}
    if frame not in (0, 1, 2):
        return {"ok": False, "error": f"frame must be 0|1|2, got {frame}"}

    seq = dna.upper().replace("U", "T")[frame:]
    protein_chars: list[str] = []
    stop_positions: list[int] = []

    for codon_start in range(0, len(seq) - 2, 3):
        codon = seq[codon_start:codon_start + 3]
        aa = translate_codon(codon)
        if aa is None:
            # 'X' marker for invalid codons (gaps, ambiguous) keeps the
            # protein length aligned with the DNA reading frame.
            protein_chars.append("X")
        else:
            protein_chars.append(aa)
            if aa == "*":
                stop_positions.append(len(protein_chars))  # 1-based AA pos

    return {
        "ok": True,
        "protein": "".join(protein_chars),
        "stop_codons": stop_positions,
    }
```

- [ ] **Step 1.4: Run tests to verify pass**

Run: `pytest tests/test_bio_primitives.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src-python/bioagent/bio_primitives.py tests/test_bio_primitives.py
git commit -m "feat(tools): add translate_sequence primitive wrapper"
```

---

## Task 2: Add `inspect_path` (reimplements electron path inspectors in Python)

**Files:**
- Modify: `src-python/bioagent/bio_primitives.py`
- Modify: `tests/test_bio_primitives.py`

- [ ] **Step 2.1: Write the failing test**

Append to `tests/test_bio_primitives.py`:

```python
import os
from pathlib import Path

from bioagent.bio_primitives import inspect_path


def test_inspect_path_handles_single_string(tmp_path):
    f = tmp_path / "sample.ab1"
    f.write_bytes(b"\x00\x00")

    result = inspect_path(paths=str(f))
    assert result["ok"] is True
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["type"] == "file"
    assert item["ext"] == ".ab1"
    assert item["size"] == 2


def test_inspect_path_detects_dataset_layout_subdirs(tmp_path):
    ab1 = tmp_path / "ab1"
    gb = tmp_path / "gb"
    ab1.mkdir()
    gb.mkdir()
    (ab1 / "a.ab1").write_bytes(b"")
    (gb / "a.gb").write_text("LOCUS x")

    result = inspect_path(paths=str(tmp_path))
    assert result["ok"] is True
    assert result["items"][0]["dataset_hint"] == "subdirs"


def test_inspect_path_marks_missing(tmp_path):
    result = inspect_path(paths=str(tmp_path / "does_not_exist"))
    assert result["ok"] is True
    assert result["items"][0]["type"] == "missing"


def test_inspect_path_accepts_list(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")

    result = inspect_path(paths=[str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
    assert result["ok"] is True
    assert len(result["items"]) == 2
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `pytest tests/test_bio_primitives.py::test_inspect_path_handles_single_string -v`
Expected: `ImportError: cannot import name 'inspect_path'`

- [ ] **Step 2.3: Implement `inspect_path`**

Append to `src-python/bioagent/bio_primitives.py`:

```python
import os as _os
from pathlib import Path as _Path
from typing import Union as _Union, List as _List

_AB1_SUBDIR_RE = ("ab1", "ab1_files")
_GB_SUBDIR_RE = ("gb", "gbk", "genbank")


def _detect_dataset_hint(p: _Path) -> str | None:
    """Mirror electron/main.js inspect-dataset-folder logic in Python."""
    if not p.is_dir():
        return None
    try:
        children = list(p.iterdir())
    except OSError:
        return None
    subdirs = {c.name.lower(): c for c in children if c.is_dir()}
    has_ab1_sub = any(name in subdirs for name in _AB1_SUBDIR_RE)
    has_gb_sub = any(name in subdirs for name in _GB_SUBDIR_RE)
    if has_ab1_sub and has_gb_sub:
        return "subdirs"
    files = [c for c in children if c.is_file()]
    has_ab1 = any(c.suffix.lower() == ".ab1" for c in files)
    has_gb = any(c.suffix.lower() in (".gb", ".gbk") for c in files)
    if has_ab1 and has_gb:
        return "flat"
    return None


def inspect_path(*, paths: _Union[str, _List[str]]) -> dict:
    if isinstance(paths, str):
        path_list = [paths]
    elif isinstance(paths, list) and all(isinstance(p, str) for p in paths):
        path_list = paths
    else:
        return {"ok": False, "error": "paths must be a string or list of strings"}

    items = []
    for raw in path_list:
        p = _Path(raw)
        if not p.exists():
            items.append({"path": raw, "type": "missing"})
            continue
        if p.is_dir():
            items.append({
                "path": raw,
                "type": "dir",
                "dataset_hint": _detect_dataset_hint(p),
            })
        else:
            try:
                size = p.stat().st_size
            except OSError:
                size = None
            items.append({
                "path": raw,
                "type": "file",
                "ext": p.suffix.lower(),
                "size": size,
            })
    return {"ok": True, "items": items}
```

- [ ] **Step 2.4: Run tests**

Run: `pytest tests/test_bio_primitives.py -v`
Expected: 8 tests PASS (4 from Task 1 + 4 new).

- [ ] **Step 2.5: Commit**

```bash
git add src-python/bioagent/bio_primitives.py tests/test_bio_primitives.py
git commit -m "feat(tools): add inspect_path primitive (Python port of electron inspectors)"
```

---

## Task 3: Add `read_sequence_file`

Handles three formats: AB1, GenBank, FASTA. Each goes through a different biopython path.

**Files:**
- Modify: `src-python/bioagent/bio_primitives.py`
- Modify: `tests/test_bio_primitives.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/test_bio_primitives.py`:

```python
from bioagent.bio_primitives import read_sequence_file


def test_read_sequence_file_genbank(tmp_path):
    gb = tmp_path / "tiny.gb"
    gb.write_text(
        "LOCUS       tiny    9 bp    DNA     linear   UNK 01-JAN-2026\n"
        "FEATURES             Location/Qualifiers\n"
        "     CDS             1..9\n"
        '                     /gene="X"\n'
        "ORIGIN\n"
        "        1 atggaataa\n"
        "//\n"
    )
    result = read_sequence_file(path=str(gb))
    assert result["ok"] is True
    assert result["format"] == "genbank"
    assert result["length"] == 9
    assert result["sequence_preview"].startswith("ATGGAATAA")
    assert result["features"], "expected at least one CDS feature"


def test_read_sequence_file_fasta(tmp_path):
    fa = tmp_path / "x.fa"
    fa.write_text(">seq1\nATGCATGC\n")
    result = read_sequence_file(path=str(fa))
    assert result["ok"] is True
    assert result["format"] == "fasta"
    assert result["length"] == 8


def test_read_sequence_file_missing(tmp_path):
    result = read_sequence_file(path=str(tmp_path / "missing.gb"))
    assert result["ok"] is False
    assert "not found" in result["error"].lower()


def test_read_sequence_file_unknown_extension(tmp_path):
    f = tmp_path / "weird.xyz"
    f.write_text("x")
    result = read_sequence_file(path=str(f))
    assert result["ok"] is False
    assert "unsupported" in result["error"].lower()
```

- [ ] **Step 3.2: Run test to verify failure**

Run: `pytest tests/test_bio_primitives.py::test_read_sequence_file_genbank -v`
Expected: ImportError.

- [ ] **Step 3.3: Implement `read_sequence_file`**

Append to `src-python/bioagent/bio_primitives.py`:

```python
from Bio import SeqIO as _SeqIO

from core.alignment import load_genbank as _load_genbank
from core.alignment import read_ab1_payload as _read_ab1_payload


def read_sequence_file(*, path: str, max_chars: int = 2000) -> dict:
    p = _Path(path)
    if not p.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {path}"}

    ext = p.suffix.lower()
    try:
        if ext == ".ab1":
            seq, qual, _ = _read_ab1_payload(p, do_trim=False)
            return {
                "ok": True,
                "format": "ab1",
                "length": len(seq),
                "sequence_preview": seq[:max_chars],
                "quality_stats": {
                    "min": min(qual) if qual else None,
                    "max": max(qual) if qual else None,
                    "mean": (sum(qual) / len(qual)) if qual else None,
                },
            }
        if ext in (".gb", ".gbk"):
            rec, ref_seq, ref_len, cds_start, cds_end = _load_genbank(p)
            features = []
            for feat in rec.features:
                if feat.type in ("CDS", "gene", "misc_feature"):
                    features.append({
                        "type": feat.type,
                        "start": int(feat.location.start) + 1,
                        "end": int(feat.location.end),
                        "label": str(feat.qualifiers.get("label", [""])[0])
                                 or str(feat.qualifiers.get("gene", [""])[0]),
                    })
            return {
                "ok": True,
                "format": "genbank",
                "length": ref_len,
                "sequence_preview": ref_seq[:max_chars],
                "features": features,
                "cds_start": cds_start,
                "cds_end": cds_end,
            }
        if ext in (".fa", ".fasta", ".fna"):
            rec = _SeqIO.read(str(p), "fasta")
            seq = str(rec.seq).upper()
            return {
                "ok": True,
                "format": "fasta",
                "length": len(seq),
                "sequence_preview": seq[:max_chars],
                "id": rec.id,
            }
    except Exception as exc:
        return {"ok": False, "error": f"parse failed: {exc}"}
    return {"ok": False, "error": f"unsupported extension: {ext}"}
```

- [ ] **Step 3.4: Run tests**

Run: `pytest tests/test_bio_primitives.py -v`
Expected: 12 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src-python/bioagent/bio_primitives.py tests/test_bio_primitives.py
git commit -m "feat(tools): add read_sequence_file (AB1/GenBank/FASTA)"
```

---

## Task 4: Add `compare_sequences`

This is the most complex primitive — it composes the aligner, mutation extractor, and optional CDS-aware AA-change detector.

**Files:**
- Modify: `src-python/bioagent/bio_primitives.py`
- Modify: `tests/test_bio_primitives.py`

- [ ] **Step 4.1: Write the failing test**

Append to `tests/test_bio_primitives.py`:

```python
from bioagent.bio_primitives import compare_sequences


def test_compare_sequences_finds_single_substitution():
    # ref:  ATGGAATAA
    # qry:  ATGGGATAA  (position 5: A -> G)
    result = compare_sequences(ref_seq="ATGGAATAA", query_seq="ATGGGATAA")
    assert result["ok"] is True
    assert result["length"] >= 8
    assert any(m["type"] == "substitution" for m in result["mutations"])
    sub = next(m for m in result["mutations"] if m["type"] == "substitution")
    assert sub["position"] == 5
    assert sub["ref"] == "A"
    assert sub["qry"] == "G"


def test_compare_sequences_with_cds_yields_aa_changes():
    # ref: ATGGAATAA (M-E-*), qry mutates pos 5 -> AAG codon -> still K? Let's
    # pick a clearer change: pos 4 G->A so codon 2 becomes AAA (Lys) not GAA (Glu)
    result = compare_sequences(
        ref_seq="ATGGAATAA",
        query_seq="ATGAAATAA",
        ref_cds_start=1,
        ref_cds_end=9,
    )
    assert result["ok"] is True
    assert result.get("aa_changes"), "expected aa_changes list when CDS provided"


def test_compare_sequences_rejects_empty():
    result = compare_sequences(ref_seq="", query_seq="ATG")
    assert result["ok"] is False


def test_compare_sequences_identical_inputs_zero_mutations():
    result = compare_sequences(ref_seq="ATGGAATAA", query_seq="ATGGAATAA")
    assert result["ok"] is True
    assert result["mutations"] == []
    assert result["identity"] == 1.0
```

- [ ] **Step 4.2: Run test to verify failure**

Run: `pytest tests/test_bio_primitives.py::test_compare_sequences_finds_single_substitution -v`
Expected: ImportError.

- [ ] **Step 4.3: Implement `compare_sequences`**

Append to `src-python/bioagent/bio_primitives.py`:

```python
from core.alignment import (
    aa_changes_from_cds as _aa_changes_from_cds,
    build_aligner as _build_aligner,
    compute_stats as _compute_stats,
    extract_mutations as _extract_mutations,
    pick_best_orientation as _pick_best_orientation,
)


def compare_sequences(*, ref_seq: str, query_seq: str,
                     ref_cds_start: int | None = None,
                     ref_cds_end: int | None = None) -> dict:
    if not ref_seq or not query_seq:
        return {"ok": False, "error": "ref_seq and query_seq must both be non-empty"}

    ref_seq = ref_seq.upper()
    query_seq = query_seq.upper()
    aligner = _build_aligner()

    # Use ref doubled (ref2) for circular tolerance — mirrors analyze_dirs path.
    ref2 = ref_seq + ref_seq
    try:
        _orient, _aln, ref_g, qry_g, ref2_start, _ref2_end, _qry = \
            _pick_best_orientation(ref2, query_seq, aligner)
    except Exception as exc:
        return {"ok": False, "error": f"alignment failed: {exc}"}

    matches, aligned_both, identity, sub, ins, dele = _compute_stats(ref_g, qry_g)
    mutations = _extract_mutations(ref_g, qry_g, ref2_start, len(ref_seq))

    aa_changes = None
    if ref_cds_start is not None and ref_cds_end is not None:
        try:
            ok, changes, has_indel, _raw_n = _aa_changes_from_cds(
                ref_seq, len(ref_seq), ref_cds_start, ref_cds_end,
                ref_g, qry_g, ref2_start,
            )
            if ok:
                aa_changes = changes
        except Exception as exc:
            return {"ok": False, "error": f"aa change extraction failed: {exc}"}

    return {
        "ok": True,
        "length": aligned_both,
        "matches": matches,
        "identity": identity,
        "substitutions": sub,
        "insertions": ins,
        "deletions": dele,
        "mutations": mutations,
        "aa_changes": aa_changes,
    }
```

- [ ] **Step 4.4: Run tests**

Run: `pytest tests/test_bio_primitives.py -v`
Expected: 16 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src-python/bioagent/bio_primitives.py tests/test_bio_primitives.py
git commit -m "feat(tools): add compare_sequences primitive"
```

---

## Task 5: Add `analyze_files_adhoc`

Thin wrapper over the existing `analyze_dirs` in `core/alignment.py`. The key value over the existing `analyze_sequences` tool is: **no dataset registration required**, so the agent can analyze paths the user just dropped without polluting the dataset registry.

**Files:**
- Modify: `src-python/bioagent/bio_primitives.py`
- Modify: `tests/test_bio_primitives.py`

- [ ] **Step 5.1: Write the failing test (using existing test fixtures)**

Check first what fixtures exist for analyze_dirs in `tests/test_analyze_sequences_dropped.py` (already there per `ls tests/`):

```bash
head -80 tests/test_analyze_sequences_dropped.py
```

Copy its fixture pattern. Append to `tests/test_bio_primitives.py`:

```python
from bioagent.bio_primitives import analyze_files_adhoc


def test_analyze_files_adhoc_missing_dirs():
    result = analyze_files_adhoc(ab1_dir="/no/such", gb_dir="/no/such")
    assert result["ok"] is False
    assert "not found" in result["error"].lower() or "not a directory" in result["error"].lower()


# NOTE: a happy-path test requires real AB1+GB fixtures. If the existing
# tests/test_analyze_sequences_dropped.py has a fixture builder, reuse it
# (e.g. `_make_fixture_dirs(tmp_path)` or similar). If no fixture is
# available, mark a TODO and rely on integration testing only.
```

- [ ] **Step 5.2: Run test to verify failure**

Run: `pytest tests/test_bio_primitives.py::test_analyze_files_adhoc_missing_dirs -v`
Expected: ImportError.

- [ ] **Step 5.3: Implement `analyze_files_adhoc`**

Append to `src-python/bioagent/bio_primitives.py`:

```python
from core.alignment import analyze_dirs as _analyze_dirs


def analyze_files_adhoc(*, ab1_dir: str, gb_dir: str,
                       output_dir: str | None = None) -> dict:
    ab1 = _Path(ab1_dir)
    gb = _Path(gb_dir)
    if not ab1.exists() or not ab1.is_dir():
        return {"ok": False, "error": f"ab1_dir not found or not a directory: {ab1_dir}"}
    if not gb.exists() or not gb.is_dir():
        return {"ok": False, "error": f"gb_dir not found or not a directory: {gb_dir}"}

    out = _Path(output_dir) if output_dir else (ab1.parent / "_adhoc_output")
    try:
        result = _analyze_dirs(gb, ab1, out)
    except Exception as exc:
        return {"ok": False, "error": f"analyze failed: {exc}"}

    # Normalize: analyze_dirs returns various shapes depending on version.
    # Expose just a slim summary; full samples are retrievable via the
    # standard `get_analysis_detail` tool if the caller wants depth.
    samples = result.get("samples") if isinstance(result, dict) else None
    return {
        "ok": True,
        "samples_count": len(samples) if samples else 0,
        "summary": result.get("summary") if isinstance(result, dict) else None,
        "output_dir": str(out),
    }
```

- [ ] **Step 5.4: Run tests**

Run: `pytest tests/test_bio_primitives.py -v`
Expected: 17 tests PASS (or 18 if you added a happy-path test).

- [ ] **Step 5.5: Commit**

```bash
git add src-python/bioagent/bio_primitives.py tests/test_bio_primitives.py
git commit -m "feat(tools): add analyze_files_adhoc (no dataset registration)"
```

---

## Task 6: Add `export_analysis_html`

Pulls a stored analysis sample from `tools_register._ANALYSIS_DETAILS` and renders it via the existing `write_alignment_html`.

**Files:**
- Modify: `src-python/bioagent/bio_primitives.py`
- Modify: `tests/test_bio_primitives.py`

- [ ] **Step 6.1: Read what `write_alignment_html` expects**

Run: `grep -A 30 "^def write_alignment_html" core/alignment.py | head -35`

Note the parameter shape; the wrapper must build that shape from a stored sample dict.

- [ ] **Step 6.2: Write the failing test**

Append to `tests/test_bio_primitives.py`:

```python
import bioagent.tools_register as _tools_register
from bioagent.bio_primitives import export_analysis_html


def test_export_analysis_html_unknown_analysis():
    result = export_analysis_html(analysis_id="nope", sample_id="x")
    assert result["ok"] is False
    assert "analysis" in result["error"].lower()


def test_export_analysis_html_writes_file(tmp_path):
    _tools_register._ANALYSIS_DETAILS["test_aid"] = {
        "analysis_id": "test_aid",
        "samples": [
            {
                "id": "S1",
                "aligned_ref_g": "ATGGAATAA",
                "aligned_query_g": "ATGAAATAA",
                "ref2_start": 0,
                "mutations": [
                    {"position": 4, "refBase": "G", "queryBase": "A", "type": "substitution"},
                ],
            },
        ],
    }

    out_path = tmp_path / "alignment.html"
    result = export_analysis_html(
        analysis_id="test_aid",
        sample_id="S1",
        output_path=str(out_path),
    )
    assert result["ok"] is True
    assert out_path.exists()
    assert "ATGGAATAA" in out_path.read_text() or out_path.read_text().strip()
```

- [ ] **Step 6.3: Run test to verify failure**

Run: `pytest tests/test_bio_primitives.py::test_export_analysis_html_unknown_analysis -v`
Expected: ImportError.

- [ ] **Step 6.4: Implement `export_analysis_html`**

Append to `src-python/bioagent/bio_primitives.py`:

```python
from core.alignment import write_alignment_html as _write_alignment_html


def export_analysis_html(*, analysis_id: str, sample_id: str,
                         output_path: str | None = None) -> dict:
    # Local import to avoid circular import at module load.
    from bioagent.tools_register import _ANALYSIS_DETAILS

    detail = _ANALYSIS_DETAILS.get(analysis_id)
    if not detail:
        return {"ok": False, "error": f"analysis not found: {analysis_id}"}
    samples = detail.get("samples") or []
    sample = next((s for s in samples if s.get("id") == sample_id), None)
    if sample is None:
        return {"ok": False, "error": f"sample not found in analysis: {sample_id}"}

    out = _Path(output_path) if output_path else _Path.cwd() / f"alignment_{analysis_id}_{sample_id}.html"
    try:
        _write_alignment_html(
            out,
            title=f"{analysis_id} / {sample_id}",
            ref_g=sample.get("aligned_ref_g", ""),
            qry_g=sample.get("aligned_query_g", ""),
            mutations=sample.get("mutations", []),
        )
    except TypeError as exc:
        # If the real write_alignment_html signature differs, surface the
        # mismatch cleanly so we can adjust the adapter rather than crash.
        return {"ok": False, "error": f"alignment html signature mismatch: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"html write failed: {exc}"}
    return {"ok": True, "html_path": str(out)}
```

> If Step 6.1 revealed `write_alignment_html` has different parameter names (e.g. uses `aligned_ref` instead of `ref_g`), adjust the call site here before re-running tests.

- [ ] **Step 6.5: Run tests**

Run: `pytest tests/test_bio_primitives.py -v`
Expected: 19 tests PASS.

- [ ] **Step 6.6: Commit**

```bash
git add src-python/bioagent/bio_primitives.py tests/test_bio_primitives.py
git commit -m "feat(tools): add export_analysis_html primitive"
```

---

## Task 7: Wire all 6 tools into the registry

Now that every primitive is unit-tested in isolation, register them with the MCP layer and write a dispatch integration test.

**Files:**
- Modify: `src-python/bioagent/tools_register.py` (extend `register_initial_tools`)
- Create: `tests/test_tools_register_primitives.py`

- [ ] **Step 7.1: Write the failing integration test**

Create `tests/test_tools_register_primitives.py`:

```python
"""Integration tests verifying the 6 primitive tools are registered and
dispatchable via the standard mcp_tools.call_tool() interface."""
import bioagent.mcp_tools as mcp_tools
import bioagent.tools_register as tools_register


def setup_module(_module):
    tools_register.register_initial_tools()


def test_inspect_path_is_callable(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    result = mcp_tools.call_tool("inspect_path", {"paths": str(tmp_path / "a.txt")})
    assert result["ok"] is True
    assert result["data"]["ok"] is True


def test_translate_sequence_is_callable():
    result = mcp_tools.call_tool("translate_sequence", {"dna": "ATGGAATAA"})
    assert result["ok"] is True
    assert result["data"]["protein"] == "ME*"


def test_compare_sequences_is_callable():
    result = mcp_tools.call_tool("compare_sequences", {
        "ref_seq": "ATGGAATAA",
        "query_seq": "ATGGGATAA",
    })
    assert result["ok"] is True
    assert result["data"]["ok"] is True


def test_read_sequence_file_is_callable(tmp_path):
    fa = tmp_path / "x.fa"
    fa.write_text(">s\nATGC\n")
    result = mcp_tools.call_tool("read_sequence_file", {"path": str(fa)})
    assert result["ok"] is True
    assert result["data"]["format"] == "fasta"


def test_analyze_files_adhoc_is_callable():
    result = mcp_tools.call_tool("analyze_files_adhoc", {
        "ab1_dir": "/missing",
        "gb_dir": "/missing",
    })
    assert result["ok"] is True       # call dispatched successfully
    assert result["data"]["ok"] is False  # but execution returned error envelope


def test_export_analysis_html_is_callable():
    result = mcp_tools.call_tool("export_analysis_html", {
        "analysis_id": "nope",
        "sample_id": "nope",
    })
    assert result["ok"] is True
    assert result["data"]["ok"] is False


def test_list_tools_includes_all_six_primitives():
    names = {t["name"] for t in mcp_tools.list_tools()}
    expected = {
        "inspect_path", "read_sequence_file", "compare_sequences",
        "translate_sequence", "analyze_files_adhoc", "export_analysis_html",
    }
    assert expected.issubset(names)
```

- [ ] **Step 7.2: Run test to verify failure**

Run: `pytest tests/test_tools_register_primitives.py -v`
Expected: All tests FAIL — the tools aren't registered yet.

- [ ] **Step 7.3: Register the 6 tools**

In `src-python/bioagent/tools_register.py`, add to the imports near the top (around line 19):

```python
from bioagent.bio_primitives import (
    analyze_files_adhoc,
    compare_sequences,
    export_analysis_html,
    inspect_path,
    read_sequence_file,
    translate_sequence,
)
```

Then append inside `register_initial_tools()` (after the existing `delete_dataset` registration around line 472, before the function returns):

```python
    # ── Primitive (atomic) tools ──────────────────────────────────────────
    # These expose lower-level operations so the agent can answer file /
    # sequence questions without first running register_dataset or a full
    # analyze_sequences pass. See bio_primitives.py for the wrappers.

    register_tool(
        name="inspect_path",
        description=(
            "Inspect a path or list of paths. Returns file/dir status, file "
            "extension/size, and a dataset_hint when a directory looks like "
            "a BioAgent ab1+gb dataset. Use this BEFORE analyze_sequences "
            "when the user mentions a path you haven't seen before."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paths": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                },
            },
            "required": ["paths"],
        },
        execute=inspect_path,
    )

    register_tool(
        name="read_sequence_file",
        description=(
            "Read a single AB1, GenBank, or FASTA file and return a summary "
            "(length, sequence preview, features/CDS bounds for GenBank, "
            "quality stats for AB1). Use for one-off file inspection without "
            "registering a dataset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 2000},
            },
            "required": ["path"],
        },
        execute=read_sequence_file,
    )

    register_tool(
        name="compare_sequences",
        description=(
            "Align two DNA sequences and return mutations + identity. If "
            "ref_cds_start/end are provided, also returns amino-acid changes. "
            "Use when the user wants to diff two sequences without going "
            "through a full dataset analysis."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ref_seq": {"type": "string"},
                "query_seq": {"type": "string"},
                "ref_cds_start": {"type": "integer"},
                "ref_cds_end": {"type": "integer"},
            },
            "required": ["ref_seq", "query_seq"],
        },
        execute=compare_sequences,
    )

    register_tool(
        name="translate_sequence",
        description=(
            "Translate a DNA sequence to protein. Frame defaults to 0; pass "
            "1 or 2 for alternate reading frames. Returns the protein string "
            "plus stop-codon AA positions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dna": {"type": "string"},
                "frame": {"type": "integer", "enum": [0, 1, 2], "default": 0},
            },
            "required": ["dna"],
        },
        execute=translate_sequence,
    )

    register_tool(
        name="analyze_files_adhoc",
        description=(
            "Run the full alignment analysis on a pair of folders WITHOUT "
            "registering them as a dataset. Use when the user wants a "
            "one-off look at folders they don't intend to keep."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ab1_dir": {"type": "string"},
                "gb_dir": {"type": "string"},
                "output_dir": {"type": "string"},
            },
            "required": ["ab1_dir", "gb_dir"],
        },
        execute=analyze_files_adhoc,
    )

    register_tool(
        name="export_analysis_html",
        description=(
            "Render the alignment of a single sample from a stored analysis "
            "to an HTML file. Use to give the user a viewable alignment "
            "outside the app UI."
        ),
        parameters={
            "type": "object",
            "properties": {
                "analysis_id": {"type": "string"},
                "sample_id": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["analysis_id", "sample_id"],
        },
        execute=export_analysis_html,
    )
```

- [ ] **Step 7.4: Run all tests**

Run: `pytest tests/test_bio_primitives.py tests/test_tools_register_primitives.py tests/test_tools_register.py tests/test_tools_register_persistence.py -v`
Expected: All PASS. Watch for regressions in the existing tools_register tests.

- [ ] **Step 7.5: Commit**

```bash
git add src-python/bioagent/tools_register.py tests/test_tools_register_primitives.py
git commit -m "feat(tools): register 6 primitive tools in MCP registry"
```

---

## Task 8: Teach the agent about the new tools (system prompt update)

The LLM only uses tools described in the system prompt + tool catalog. Tool descriptions alone aren't enough — the prompt should highlight WHEN to prefer primitives over the heavyweight tools.

**Files:**
- Modify: `electron/agent_harness.mjs` — find `buildSystemPrompt()` (around line 249)

- [ ] **Step 8.1: Locate the system prompt builder**

Run: `grep -n "buildSystemPrompt\|function buildSystemPrompt" electron/agent_harness.mjs | head`

- [ ] **Step 8.2: Add a primitives section to the prompt**

Inside `buildSystemPrompt()`, after the existing tool-usage guidance, append a paragraph (keep it concise — the LLM does not need every detail; tool descriptions cover that):

```javascript
// Primitive tools guidance — keep terse; full schemas come via the tool catalog.
const primitivesGuidance = `

You also have access to **primitive tools** for atomic operations that don't require a registered dataset:
- \`inspect_path\` — examine a file/folder the user just mentioned, BEFORE deciding to register or analyze it.
- \`read_sequence_file\` — peek at a single AB1/GenBank/FASTA file.
- \`compare_sequences\` — diff two sequences inline (no dataset needed).
- \`translate_sequence\` — DNA → protein.
- \`analyze_files_adhoc\` — one-off folder analysis without polluting the dataset registry.
- \`export_analysis_html\` — render a stored sample's alignment to an HTML file the user can open.

Prefer these primitives when the user's question is narrow (e.g. "what's in this file?", "translate this sequence", "diff these two sequences"). Use the full \`analyze_sequences\` + \`register_dataset\` path only when the user is starting a multi-sample investigation they'll want to revisit.`;
```

Concatenate `primitivesGuidance` onto the returned prompt string.

- [ ] **Step 8.3: Run the existing JS test suite to catch regressions**

Run: `npm run test:js`
Expected: All PASS (including `test_agent_harness.mjs` which exercises buildSystemPrompt).

If `test_agent_harness.mjs` snapshots the prompt text, update its expectation to include the new section.

- [ ] **Step 8.4: Commit**

```bash
git add electron/agent_harness.mjs tests/test_agent_harness.mjs
git commit -m "feat(prompt): teach the agent when to prefer primitive tools"
```

---

## Task 9: Smoke test the full path end-to-end

Verify the primitive tools work when called through the real running sidecar, not just unit-tested in isolation.

**Files:** none modified — this is a manual smoke gate before declaring the plan complete.

- [ ] **Step 9.1: Rebuild the sidecar**

Run: `npm run build:sidecar`
Expected: `[OK] sidecar built at: ...\dist-python\bioagent-sidecar`

- [ ] **Step 9.2: Start the dev app**

Run: `npm run electron:dev`

Expected: app launches, agent harness initializes.

- [ ] **Step 9.3: Manually drive each primitive through the chat UI**

Try prompts that should trigger each tool:

| Prompt | Expected tool call |
|---|---|
| "Translate ATGGAATAA for me" | `translate_sequence` |
| "What's in [some local fasta path]?" | `inspect_path` + `read_sequence_file` |
| "Diff these two sequences: ATGGAATAA vs ATGAAATAA" | `compare_sequences` |
| "Take a quick look at this folder without registering it: [path]" | `inspect_path` then `analyze_files_adhoc` |
| "Export the alignment for [analysis_id] [sample_id] as HTML" | `export_analysis_html` |

Confirm in the trace panel / terminal that the right tool is being called, and that the agent's reply correctly uses the tool's output. **Stop and file bugs for anything that misbehaves before declaring done.**

- [ ] **Step 9.4: Final summary commit (if any tweaks were needed)**

```bash
git add <any-fixup-files>
git commit -m "fix(tools): smoke-test corrections from v0.3.0 primitives manual run"
```

---

## Task 10: Bump version + tag for partial release (optional)

If you want to ship just the primitives ahead of streaming/UI work:

- [ ] **Step 10.1: Bump package.json**

```diff
- "version": "0.2.0",
+ "version": "0.3.0-alpha.1",
```

- [ ] **Step 10.2: Tag and push**

```bash
git add package.json
git commit -m "chore(release): bump to 0.3.0-alpha.1 (primitive tools)"
git tag v0.3.0-alpha.1
git push origin worktree-ultimate-bioagent v0.3.0-alpha.1
```

CI will build and publish. Or skip this task entirely and bundle with the other v0.3.0 sub-plans.

---

## Verification Checklist

Before marking this plan complete:

- [ ] `pytest tests/test_bio_primitives.py -v` — all PASS
- [ ] `pytest tests/test_tools_register_primitives.py -v` — all PASS
- [ ] `pytest tests/test_tools_register.py tests/test_tools_register_persistence.py -v` — no regressions
- [ ] `npm run test:js` — no regressions in `test_agent_harness.mjs`
- [ ] `npm run typecheck` — clean
- [ ] Manual smoke test: each of 6 primitives invoked successfully through the running app
- [ ] All 6 tools appear in `mcp_tools.list_tools()` output
- [ ] System prompt contains the primitives guidance paragraph

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `write_alignment_html` signature drift breaks Task 6 | Medium | Step 6.1 inspects the real signature before writing the adapter call. |
| `analyze_dirs` return shape varies | Medium | Task 5 only exposes a slim summary and downcasts shapes; deeper data still flows through `get_analysis_detail`. |
| LLM doesn't switch to primitives, keeps using `analyze_sequences` | Medium | Task 8's prompt update explicitly steers it; verify in Task 9 smoke. If still bad, strengthen examples in the prompt. |
| Circular import between `bio_primitives.py` and `tools_register.py` (Task 6 references `_ANALYSIS_DETAILS`) | Low | Task 6's implementation uses a local import inside the function body. |
| Existing tests broken by new imports | Low | Task 7.4 explicitly re-runs the existing tools_register suite. |

## Why This Plan Excludes UI Changes

The user explicitly asked for "都可以做" but writing-plans guidance is "each plan produces working, testable software on its own". This plan delivers value (the agent can now answer 5x more question types) without touching React. The streaming + UI cards + permission hook + build config items in the master v0.3.0 roadmap belong to follow-up plans because each is a coherent, independently-testable unit on its own. Sequencing them after this plan also lets us observe whether the primitives actually get used in real conversations before investing in UX polish around them.
