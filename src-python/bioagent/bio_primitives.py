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

from pathlib import Path as _Path
from typing import List as _List, Union as _Union

from core.alignment import translate_codon


_AB1_SUBDIRS = ("ab1", "ab1_files")
_GB_SUBDIRS = ("gb", "gbk", "genbank")


def _detect_dataset_hint(p: _Path) -> str | None:
    """Mirror electron/main.js inspect-dataset-folder logic in Python."""
    if not p.is_dir():
        return None
    try:
        children = list(p.iterdir())
    except OSError:
        return None
    subdirs = {c.name.lower(): c for c in children if c.is_dir()}
    has_ab1_sub = any(name in subdirs for name in _AB1_SUBDIRS)
    has_gb_sub = any(name in subdirs for name in _GB_SUBDIRS)
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
            protein_chars.append("X")
        else:
            protein_chars.append(aa)
            if aa == "*":
                stop_positions.append(len(protein_chars))

    return {
        "ok": True,
        "protein": "".join(protein_chars),
        "stop_codons": stop_positions,
    }
