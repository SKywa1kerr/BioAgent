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

from Bio import SeqIO as _SeqIO

from core.alignment import (
    aa_changes_from_cds as _aa_changes_from_cds,
    analyze_dirs as _analyze_dirs,
    build_aligner as _build_aligner,
    compute_stats as _compute_stats,
    extract_mutations as _extract_mutations,
    load_genbank as _load_genbank,
    pick_best_orientation as _pick_best_orientation,
    read_ab1_payload as _read_ab1_payload,
    translate_codon,
    write_alignment_html as _write_alignment_html,
)


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
                    label = ""
                    if "label" in feat.qualifiers:
                        label = str(feat.qualifiers["label"][0])
                    elif "gene" in feat.qualifiers:
                        label = str(feat.qualifiers["gene"][0])
                    features.append({
                        "type": feat.type,
                        "start": int(feat.location.start) + 1,
                        "end": int(feat.location.end),
                        "label": label,
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


def compare_sequences(*, ref_seq: str, query_seq: str,
                      ref_cds_start: int | None = None,
                      ref_cds_end: int | None = None) -> dict:
    if not ref_seq or not query_seq:
        return {"ok": False, "error": "ref_seq and query_seq must both be non-empty"}

    ref_seq = ref_seq.upper()
    query_seq = query_seq.upper()
    aligner = _build_aligner()

    # Use ref doubled (ref2) for circular tolerance — mirrors analyze_dirs.
    ref2 = ref_seq + ref_seq
    try:
        _orient, _aln, ref_g, qry_g, ref2_start, _ref2_end, _qry = \
            _pick_best_orientation(ref2, query_seq, aligner)
    except Exception as exc:
        return {"ok": False, "error": f"alignment failed: {exc}"}

    matches, aligned_both, identity, sub, ins, dele = _compute_stats(ref_g, qry_g)
    try:
        mutations = _extract_mutations(ref_g, qry_g, ref2_start, len(ref_seq))
    except Exception as exc:
        return {"ok": False, "error": f"mutation extraction failed: {exc}"}

    aa_changes = None
    has_indel = False
    if ref_cds_start is not None and ref_cds_end is not None:
        try:
            # NOTE: in aa_changes_from_cds, the first return value `ok`
            # means "no AA changes and no indels found", NOT "ran without
            # error". We always want the changes list — it's authoritative
            # whether ok is True or False.
            _no_changes, changes, has_indel, _raw_n = _aa_changes_from_cds(
                ref_seq, len(ref_seq), ref_cds_start, ref_cds_end,
                ref_g, qry_g, ref2_start,
            )
            aa_changes = changes  # always a list, possibly empty
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
        "frameshift_indel": has_indel,
    }


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
        # analyze_dirs signature: (gb_dir, ab1_dir, out_html_dir)
        samples = _analyze_dirs(gb, ab1, out)
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"analyze failed: {exc}"}

    samples_list = samples if isinstance(samples, list) else []
    return {
        "ok": True,
        "samples_count": len(samples_list),
        "samples": samples_list,
        "output_dir": str(out),
    }


def export_analysis_html(*, analysis_id: str, sample_id: str,
                         output_path: str | None = None) -> dict:
    # Local import: tools_register imports bio_primitives, so a top-level
    # import here would create a circular dependency.
    from bioagent.tools_register import _ANALYSIS_DETAILS

    detail = _ANALYSIS_DETAILS.get(analysis_id)
    if not detail:
        return {"ok": False, "error": f"analysis not found: {analysis_id}"}
    samples = detail.get("samples") or []
    sample = next((s for s in samples if s.get("id") == sample_id), None)
    if sample is None:
        return {"ok": False, "error": f"sample not found in analysis: {sample_id}"}

    ref_g = sample.get("aligned_ref_g")
    qry_g = sample.get("aligned_query_g")
    ref_seq = sample.get("ref_sequence")
    ref_length = sample.get("ref_length") or (len(ref_seq) if ref_seq else None)
    if not ref_g or not qry_g or not ref_length:
        return {
            "ok": False,
            "error": "sample missing aligned_ref_g / aligned_query_g / ref_length needed for HTML render",
        }

    # Recover ref2_start by re-running the orientation pick on the stored
    # ref/query pair. This is the same logic analyze_sample used originally.
    ref2_start = 0
    query_seq = sample.get("query_sequence")
    if ref_seq and query_seq:
        try:
            aligner = _build_aligner()
            _orient, _aln, _rg, _qg, ref2_start, _re, _qry = \
                _pick_best_orientation(ref_seq + ref_seq, query_seq, aligner)
        except Exception:
            ref2_start = 0  # fall back; HTML positions may be slightly off

    out = _Path(output_path) if output_path else \
          _Path.cwd() / f"alignment_{analysis_id}_{sample_id}.html"
    try:
        _write_alignment_html(
            out,
            title=f"{analysis_id} / {sample_id}",
            ref_g=ref_g, qry_g=qry_g,
            ref2_start=ref2_start, ref_len=ref_length,
        )
    except Exception as exc:
        return {"ok": False, "error": f"html write failed: {exc}"}
    return {"ok": True, "html_path": str(out)}


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
