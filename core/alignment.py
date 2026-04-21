#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
core/alignment.py
Bioinformatics core: GB/AB1 parsing, alignment, mutation detection, AA translation.
Merged and simplified from scripts/gb_ab1_mutations.py and scripts/gb_ab1_agent_pro.py.
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from Bio.Data import CodonTable


# ── Rule tuning constants ────────────────────────────────────────────────────

# Substitutions whose 1-based ref position is within this many bp of
# cds_start or cds_end are demoted from wrong to ok for status purposes.
# They are still recorded in the mutations list for display.
EDGE_IGNORE_BP = 20

COVERAGE_UNTESTED = 0.4
COVERAGE_OK = 0.4
QUALITY_UNTESTED = 15.0
QUALITY_UNCERTAIN = 25.0


def is_edge_ignored(pos, cds_start, cds_end) -> bool:
    """True if this substitution position is close enough to either CDS boundary
    that we should exclude it from the wrong decision."""
    if pos is None or cds_start is None or cds_end is None:
        return False
    return (pos - cds_start) < EDGE_IGNORE_BP or (cds_end - pos) < EDGE_IGNORE_BP


def decide_bucket(cds_coverage: float,
                  avg_qry_quality,
                  frameshift: bool,
                  aa_changes: list,
                  mutations: list,
                  has_single_read: bool,
                  cds_start=None,
                  cds_end=None) -> str:
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


# ── Utilities ────────────────────────────────────────────────────────────────

def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{stamp}" + path.suffix)
        df.to_csv(alt, index=False)
        print(f"[WARN] Permission denied writing {path.name}. Saved as {alt.name}.", file=sys.stderr)
        return alt


# ── GenBank reference loading ────────────────────────────────────────────────

def load_genbank(gb_path: Path):
    """Parse GB reference file; locate inserted gene CDS.
    Returns (record, ref_seq, ref_len, cds_start_1based, cds_end_1based).
    """
    rec = SeqIO.read(str(gb_path), "genbank")
    ref_seq = str(rec.seq).upper()
    ref_len = len(ref_seq)

    cds_start, cds_end = None, None
    target = rec.id.upper()

    best = None
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        qs = feat.qualifiers
        label = str(qs.get("label", [""])[0]).upper() if "label" in qs else ""
        gene = str(qs.get("gene", [""])[0]).upper() if "gene" in qs else ""
        note = (" ".join(qs.get("note", [])).upper()) if "note" in qs else ""

        score = 0
        if target and target in label:
            score += 10
        if target and target in gene:
            score += 10
        if "INSERTED GENE" in note or "INSERTED GENE SEQUENCE" in note:
            score += 6
        if "INSERTED" in note:
            score += 2

        if best is None or score > best[0]:
            best = (score, feat)

    if best is not None:
        feat = best[1]
        cds_start = int(feat.location.start) + 1  # 1-based
        cds_end = int(feat.location.end)           # inclusive 1-based
    return rec, ref_seq, ref_len, cds_start, cds_end


# ── AB1 reading & quality trimming ──────────────────────────────────────────

def trim_ab1_by_quality(seq: str, qual: list[int], qmin=20, min_len=80):
    """Keep the longest contiguous region with quality >= qmin.
    Returns (trimmed_seq, trimmed_qual).
    """
    if not qual or len(qual) != len(seq):
        return seq, qual

    good = [1 if q >= qmin else 0 for q in qual]
    best_l, best_r = 0, -1
    cur_l = None

    for i, g in enumerate(good):
        if g and cur_l is None:
            cur_l = i
        if (not g or i == len(good) - 1) and cur_l is not None:
            cur_r = i if g else i - 1
            if cur_r - cur_l + 1 > best_r - best_l + 1:
                best_l, best_r = cur_l, cur_r
            cur_l = None

    if best_r >= best_l and (best_r - best_l + 1) >= min_len:
        return seq[best_l:best_r + 1], qual[best_l:best_r + 1]
    return seq, qual


def _extract_ab1_traces(rec):
    raw = rec.annotations.get("abif_raw", {})

    def _as_list(tag):
        return list(raw.get(tag, []))

    fwo = raw.get("FWO_1", b"GATC")
    if isinstance(fwo, bytes):
        try:
            fwo = fwo.decode("utf-8")
        except Exception:
            fwo = "GATC"

    traces = {"A": [], "T": [], "G": [], "C": []}
    for idx, base in enumerate(str(fwo)[:4]):
        values = _as_list(f"DATA{idx + 1}")
        if base.upper() in traces and values:
            traces[base.upper()] = values

    if not traces["A"] and raw.get("DATA9"):
        traces = {
            "G": _as_list("DATA9"),
            "A": _as_list("DATA10"),
            "T": _as_list("DATA11"),
            "C": _as_list("DATA12"),
        }

    base_locations = list(raw.get("PLOC1", []))
    return traces, base_locations


def read_ab1_payload(ab1_path: Path, do_trim=True, qmin=20, min_len=80):
    """Read AB1 and return trimmed sequence, quality and chromatogram payload."""
    rec = SeqIO.read(str(ab1_path), "abi")
    raw_seq = str(rec.seq).upper()
    raw_qual = list(rec.letter_annotations.get("phred_quality", []))
    traces, raw_base_locations = _extract_ab1_traces(rec)

    start = 0
    end = len(raw_seq)

    lstrip = len(raw_seq) - len(raw_seq.lstrip("N"))
    rstrip = len(raw_seq) - len(raw_seq.rstrip("N"))
    start += lstrip
    if rstrip > 0:
        end -= rstrip

    seq = raw_seq[start:end]
    qual = raw_qual[start:end] if raw_qual else []

    if do_trim and qual and len(qual) == len(seq):
        good = [1 if q >= qmin else 0 for q in qual]
        best_l, best_r = 0, -1
        cur_l = None
        for i, g in enumerate(good):
            if g and cur_l is None:
                cur_l = i
            if (not g or i == len(good) - 1) and cur_l is not None:
                cur_r = i if g else i - 1
                if cur_r - cur_l + 1 > best_r - best_l + 1:
                    best_l, best_r = cur_l, cur_r
                cur_l = None

        if best_r >= best_l and (best_r - best_l + 1) >= min_len:
            start += best_l
            end = start + (best_r - best_l + 1)
            seq = raw_seq[start:end]
            qual = raw_qual[start:end]

        lstrip2 = len(seq) - len(seq.lstrip("N"))
        rstrip2 = len(seq) - len(seq.rstrip("N"))
        start += lstrip2
        end = end - rstrip2 if rstrip2 > 0 else end
        seq = raw_seq[start:end]
        qual = raw_qual[start:end] if raw_qual else []

    base_locations = raw_base_locations[start:end] if raw_base_locations else []

    trace_start = 0
    trace_end = len(traces.get("A", []))
    if base_locations:
        trace_start = max(0, min(base_locations) - 20)
        trace_end = min(trace_end, max(base_locations) + 20)

    def _slice_trace(values):
        if not values:
            return []
        return values[trace_start:trace_end]

    rel_base_locations = [loc - trace_start for loc in base_locations] if base_locations else []

    chrom = {
        "traces_a": _slice_trace(traces.get("A", [])),
        "traces_t": _slice_trace(traces.get("T", [])),
        "traces_g": _slice_trace(traces.get("G", [])),
        "traces_c": _slice_trace(traces.get("C", [])),
        "quality": qual,
        "base_locations": rel_base_locations,
        "mixed_peaks": [],
    }

    return seq, qual, chrom


def read_ab1_sequence(ab1_path: Path, do_trim=True, qmin=20, min_len=80):
    seq, qual, _chrom = read_ab1_payload(ab1_path, do_trim=do_trim, qmin=qmin, min_len=min_len)
    return seq, qual


# ── Alignment (circular plasmid) ────────────────────────────────────────────

def build_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -1.0
    return aligner


def alignment_to_gapped(ref: str, qry: str, aln):
    coords = aln.coordinates
    ref_pos = coords[0]
    qry_pos = coords[1]
    ref_g = []
    qry_g = []

    for i in range(ref_pos.size - 1):
        r0, r1 = int(ref_pos[i]), int(ref_pos[i + 1])
        q0, q1 = int(qry_pos[i]), int(qry_pos[i + 1])
        dr = r1 - r0
        dq = q1 - q0
        if dr > 0 and dq > 0:
            ref_g.append(ref[r0:r1])
            qry_g.append(qry[q0:q1])
        elif dr > 0 and dq == 0:
            ref_g.append(ref[r0:r1])
            qry_g.append("-" * dr)
        elif dr == 0 and dq > 0:
            ref_g.append("-" * dq)
            qry_g.append(qry[q0:q1])

    return "".join(ref_g), "".join(qry_g), int(ref_pos[0]), int(ref_pos[-1])


def pick_best_orientation(ref2: str, qry: str, aligner):
    """Try forward and reverse-complement; return best.
    Returns (orientation, alignment, ref_gapped, qry_gapped,
             ref2_span_start, ref2_span_end, qry_used).
    """
    alns_f = aligner.align(ref2, qry)
    best_f = alns_f[0] if len(alns_f) else None
    score_f = best_f.score if best_f is not None else -1e18

    qry_rc = str(Seq(qry).reverse_complement())
    alns_r = aligner.align(ref2, qry_rc)
    best_r = alns_r[0] if len(alns_r) else None
    score_r = best_r.score if best_r is not None else -1e18

    if score_f >= score_r:
        ref_g, qry_g, s, e = alignment_to_gapped(ref2, qry, best_f)
        return "FORWARD", best_f, ref_g, qry_g, s, e, qry
    else:
        ref_g, qry_g, s, e = alignment_to_gapped(ref2, qry_rc, best_r)
        return "REVERSE", best_r, ref_g, qry_g, s, e, qry_rc


# ── Stats & mutation extraction ──────────────────────────────────────────────

def compute_stats(ref_g: str, qry_g: str):
    sub = ins = dele = matches = aligned_both = 0
    for a, b in zip(ref_g, qry_g):
        if a == "-" and b != "-":
            ins += 1
        elif a != "-" and b == "-":
            dele += 1
        elif a != "-" and b != "-":
            aligned_both += 1
            if a == b:
                matches += 1
            else:
                sub += 1
    identity = matches / aligned_both if aligned_both else 0.0
    return matches, aligned_both, identity, sub, ins, dele


def ref2pos_to_refpos(ref2_pos_0based: int, ref_len: int) -> int:
    """0-based ref2 position -> 1-based original ref position (circular)."""
    return (ref2_pos_0based % ref_len) + 1


def extract_mutations(ref_g: str, qry_g: str, ref2_start: int, ref_len: int):
    rows = []
    ref2_cursor = ref2_start
    last_refpos = None
    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len)
            last_refpos = refpos
            ref2_cursor += 1
        else:
            refpos = last_refpos
        if a == "-" and b != "-":
            rows.append({"type": "Insertion", "ref_pos": refpos if refpos is not None else 1,
                          "ref_base": "-", "qry_base": b})
        elif a != "-" and b == "-":
            rows.append({"type": "Deletion", "ref_pos": refpos, "ref_base": a, "qry_base": "-"})
        elif a != "-" and b != "-" and a != b:
            rows.append({"type": "Substitution", "ref_pos": refpos, "ref_base": a, "qry_base": b})
    return rows


# ── AA-level gene check ─────────────────────────────────────────────────────

STD_TABLE = CodonTable.unambiguous_dna_by_id[1]


def translate_codon(codon: str):
    codon = codon.upper().replace("U", "T")
    if len(codon) != 3 or "-" in codon or "N" in codon:
        return None
    if codon in STD_TABLE.stop_codons:
        return "*"
    return STD_TABLE.forward_table.get(codon, "X")


def label_synonymous_mutations(mutations: list[dict],
                               ref_seq: str,
                               cds_start,
                               cds_end,
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


def aa_changes_from_cds(ref_seq: str, ref_len: int,
                        cds_start: int, cds_end: int,
                        ref_g: str, qry_g: str,
                        ref2_start: int,
                        qry_qual=None, qry_aln_start=0,
                        qual_min=30):
    """Build AA changes within the inserted CDS region.
    Returns (ok, changes_list, has_indel, raw_changes_n).

    When qry_qual is provided, mutations at positions with any codon base
    quality < qual_min are filtered out. raw_changes_n tracks the
    pre-filter count for diagnostics.
    """
    if cds_start is None or cds_end is None:
        return True, [], False, 0

    cds_len = cds_end - cds_start + 1
    if cds_len <= 0:
        return True, [], False, 0

    ref2_cursor = ref2_start
    qry_cursor = qry_aln_start
    qry_by_refpos = [None] * cds_len
    ref_by_refpos = [None] * cds_len
    qry_qual_by_refpos = [None] * cds_len
    has_indel = False

    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len)
            ref2_cursor += 1
            if cds_start <= refpos <= cds_end:
                idx = refpos - cds_start
                ref_by_refpos[idx] = a
                if b != "-":
                    qry_by_refpos[idx] = b
                    if qry_qual is not None and 0 <= qry_cursor < len(qry_qual):
                        qry_qual_by_refpos[idx] = qry_qual[qry_cursor]
                else:
                    has_indel = True
        else:
            if b != "-":
                has_indel = True

        # Advance query cursor when query has a base
        if b != "-":
            qry_cursor += 1

    raw_changes = []
    filtered_changes = []
    aa_pos = 0
    for i in range(0, cds_len, 3):
        aa_pos += 1
        ref_trip = ref_by_refpos[i:i + 3]
        qry_trip = qry_by_refpos[i:i + 3]
        if len(ref_trip) != 3 or any(x is None for x in ref_trip):
            continue
        ref_codon = "".join(ref_trip)
        qry_codon = "".join([c if c is not None else "N" for c in qry_trip])
        ref_aa = translate_codon(ref_codon)
        qry_aa = translate_codon(qry_codon)
        if qry_aa is None or ref_aa is None:
            continue
        if ref_aa != qry_aa:
            change_str = f"{ref_aa}{aa_pos}{qry_aa}"
            raw_changes.append(change_str)

            # Quality filter: skip if any codon base has low quality
            if qry_qual is not None:
                codon_quals = qry_qual_by_refpos[i:i + 3]
                if any(q is not None and q < qual_min for q in codon_quals):
                    continue
            filtered_changes.append(change_str)

    ok = (len(filtered_changes) == 0) and (not has_indel)
    return ok, filtered_changes, has_indel, len(raw_changes)


# ── CDS coverage & frameshift ───────────────────────────────────────────────

def compute_cds_coverage(ref_g: str, qry_g: str, ref2_start: int,
                         ref_len: int, cds_start: int, cds_end: int) -> float:
    """Fraction of CDS positions covered by alignment (query has a base)."""
    if cds_start is None or cds_end is None:
        return 0.0
    cds_len = cds_end - cds_start + 1
    if cds_len <= 0:
        return 0.0

    covered = 0
    ref2_cursor = ref2_start
    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len)
            ref2_cursor += 1
            if cds_start <= refpos <= cds_end and b != "-":
                covered += 1
    return covered / cds_len


def compute_cds_covered_positions(ref_g: str, qry_g: str, ref2_start: int,
                                  ref_len: int, cds_start: int, cds_end: int) -> dict:
    """Return dict mapping CDS refpos -> query base for covered positions."""
    if cds_start is None or cds_end is None:
        return {}
    positions = {}
    ref2_cursor = ref2_start
    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len)
            ref2_cursor += 1
            if cds_start <= refpos <= cds_end and b != "-":
                positions[refpos] = b
    return positions


def detect_frameshift(ref_g: str, qry_g: str, ref2_start: int,
                      ref_len: int, cds_start: int, cds_end: int) -> bool:
    """Check if there are indels in the CDS region that are NOT multiples of 3."""
    if cds_start is None or cds_end is None:
        return False

    cds_ins = 0
    cds_del = 0
    ref2_cursor = ref2_start
    last_refpos = None

    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len)
            last_refpos = refpos
            ref2_cursor += 1
            if cds_start <= refpos <= cds_end and b == "-":
                cds_del += 1
        else:
            if b != "-" and last_refpos is not None:
                if cds_start <= last_refpos <= cds_end:
                    cds_ins += 1

    net_indel = abs(cds_ins - cds_del)
    if (cds_ins + cds_del) > 0 and (net_indel % 3 != 0):
        return True
    return False


# ── File matching ────────────────────────────────────────────────────────────

def parse_clone_from_gb(gb_path: Path) -> str:
    m = re.match(r"^(C\d+)", gb_path.stem, flags=re.IGNORECASE)
    return m.group(1).upper() if m else gb_path.stem.upper()


def find_ab1_for_clone(ab1_dir: Path, clone: str):
    """Recursive search for AB1 files matching clone."""
    clone_low = clone.lower()
    ab1s = []
    for p in ab1_dir.rglob("*.ab1"):
        name = p.name.lower()
        if f"-{clone_low}-" in name or f"_{clone_low}-" in name or f"{clone_low}-" in name:
            ab1s.append(p)
    return sorted(set(ab1s), key=lambda x: x.as_posix().lower())


def sample_id_from_ab1name(clone: str, ab1_name: str) -> str:
    # C123-4 (numeric suffix)
    m = re.search(r"(C\d+)-(\d+)", ab1_name, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}"
    # C123-a (letter suffix — preserve original case)
    m = re.search(r"(C\d+)-([a-zA-Z])", ab1_name, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}"
    return clone


# ── HTML alignment output ───────────────────────────────────────────────────

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{
  background:#111;
  color:#eee;
  font-family: Consolas, Menlo, Monaco, "Courier New", monospace;
  padding: 16px;
}}
pre {{
  font-size: 14px;
  line-height: 1.35;
  white-space: pre;
}}
.mm {{
  color: #ff4040;
  font-weight: 700;
}}
.star {{
  color: #ff4040;
  font-weight: 700;
}}
.pipe {{
  color: #cfcfcf;
}}
</style>
</head>
<body>
<pre>
{content}
</pre>
</body>
</html>
"""


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def trim_to_query_coverage(ref_g: str, qry_g: str):
    left = right = None
    for i, c in enumerate(qry_g):
        if c != "-":
            left = i
            break
    for i in range(len(qry_g) - 1, -1, -1):
        if qry_g[i] != "-":
            right = i
            break
    if left is None or right is None or right < left:
        return ref_g, qry_g
    return ref_g[left:right + 1], qry_g[left:right + 1]


def wrap_mismatch_tokens(ref_seg: str, qry_seg: str, which: str):
    tokens = []
    for a, b in zip(ref_seg, qry_seg):
        c = a if which == "ref" else b
        if a != "-" and b != "-" and a != b:
            tokens.append(f'<span class="mm">{html_escape(c)}</span>')
        else:
            tokens.append(html_escape(c))
    return tokens


def wrap_mid_tokens(ref_seg: str, qry_seg: str):
    tokens = []
    for a, b in zip(ref_seg, qry_seg):
        if a == "-" or b == "-":
            tokens.append(" ")
        elif a == b:
            tokens.append('<span class="pipe">|</span>')
        else:
            tokens.append('<span class="star">*</span>')
    return tokens


def group10_tokens(tokens, group=10):
    chunks = []
    for i in range(0, len(tokens), group):
        chunks.append("".join(tokens[i:i + group]))
    return " ".join(chunks)


def write_alignment_html(out_path: Path, title: str,
                         ref_g: str, qry_g: str,
                         ref2_start: int, ref_len: int,
                         width=80):
    ref_g2, qry_g2 = trim_to_query_coverage(ref_g, qry_g)
    ref2_cursor = ref2_start
    idx = 0
    lines = []

    while idx < len(ref_g2):
        chunk_ref = ref_g2[idx:idx + width]
        chunk_qry = qry_g2[idx:idx + width]
        show_pos = None
        for j, a in enumerate(chunk_ref):
            if a != "-":
                show_pos = ref2pos_to_refpos(ref2_cursor + j, ref_len)
                break
        if show_pos is None:
            show_pos = ref2pos_to_refpos(ref2_cursor, ref_len)
        adv = sum(1 for a in chunk_ref if a != "-")
        ref_tokens = wrap_mismatch_tokens(chunk_ref, chunk_qry, which="ref")
        qry_tokens = wrap_mismatch_tokens(chunk_ref, chunk_qry, which="qry")
        mid_tokens = wrap_mid_tokens(chunk_ref, chunk_qry)
        lines.append(f"REF {show_pos:05d}  {group10_tokens(ref_tokens)}")
        lines.append(f"MID {show_pos:05d}  {group10_tokens(mid_tokens)}")
        lines.append(f"QRY {show_pos:05d}  {group10_tokens(qry_tokens)}")
        lines.append("")
        ref2_cursor += adv
        idx += width

    content = "\n".join(lines)
    html = HTML_TEMPLATE.format(title=html_escape(title), content=content)
    out_path.write_text(html, encoding="utf-8")


# ── Single sample analysis ───────────────────────────────────────────────────

def analyze_sample(gb_path: Path, ab1_path: Path, aligner,
                   do_trim=True, qmin=20, min_len=80,
                   out_html_dir: Path | None = None) -> dict | None:
    """Analyze a single AB1 sample against a GB reference.
    Returns structured dict or None if sample is too short.
    """
    clone = parse_clone_from_gb(gb_path)
    rec, ref_seq, ref_len, cds_start, cds_end = load_genbank(gb_path)
    ref2 = ref_seq + ref_seq

    ab1_name = ab1_path.name
    sid = sample_id_from_ab1name(clone, ab1_name)

    qry_raw, qry_qual, chrom = read_ab1_payload(ab1_path, do_trim=do_trim, qmin=qmin, min_len=min_len)
    if len(qry_raw) < 30:
        return None

    orientation, best_aln, ref_g, qry_g, ref2_s, ref2_e, qry_used =         pick_best_orientation(ref2, qry_raw, aligner)

    if orientation == "REVERSE" and qry_qual:
        qry_qual_used = list(reversed(qry_qual))
    else:
        qry_qual_used = qry_qual

    matches, aligned_both, identity, sub, ins, dele = compute_stats(ref_g, qry_g)

    cds_cov = compute_cds_coverage(ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end)
    frameshift = detect_frameshift(ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end)

    qry_aln_start = int(best_aln.coordinates[1][0])

    ok, changes, has_indel, raw_aa_changes_n = aa_changes_from_cds(
        ref_seq=ref_seq, ref_len=ref_len,
        cds_start=cds_start, cds_end=cds_end,
        ref_g=ref_g, qry_g=qry_g, ref2_start=ref2_s,
        qry_qual=qry_qual_used, qry_aln_start=qry_aln_start,
    )

    avg_qry_quality = round(sum(qry_qual) / len(qry_qual), 1) if qry_qual else None
    cds_positions = compute_cds_covered_positions(ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end)

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

    if ref2_s < ref_len and ref2_e <= ref_len:
        padding_left = ref2_s
        padding_right = ref_len - ref2_e
        full_ref_g = ref_seq[:padding_left] + ref_g + ref_seq[ref2_e:]
        full_qry_g = "-" * padding_left + qry_g + "-" * padding_right
        match_flags = [True] * padding_left
        for a, b in zip(ref_g, qry_g):
            if a != "-" and b != "-":
                match_flags.append(a.upper() == b.upper())
            else:
                match_flags.append(False)
        match_flags.extend([True] * padding_right)
    else:
        full_ref_g = ref_g
        full_qry_g = qry_g
        match_flags = [a.upper() == b.upper() if a != "-" and b != "-" else False for a, b in zip(full_ref_g, full_qry_g)]

    if orientation == "REVERSE":
        for key in ("traces_a", "traces_t", "traces_g", "traces_c"):
            chrom[key] = list(reversed(chrom.get(key, [])))
        if chrom.get("base_locations"):
            trace_len = len(chrom.get("traces_a", []))
            chrom["base_locations"] = [trace_len - 1 - x for x in reversed(chrom["base_locations"])]
        if chrom.get("quality"):
            chrom["quality"] = list(reversed(chrom["quality"]))
        if chrom.get("mixed_peaks"):
            qlen = len(chrom.get("quality", []))
            chrom["mixed_peaks"] = [qlen - 1 - x for x in reversed(chrom["mixed_peaks"]) if isinstance(x, int)]

    if out_html_dir is not None:
        out_html_dir.mkdir(parents=True, exist_ok=True)
        html_name = f"{clone}__{ab1_path.stem}__{orientation}.html"
        html_path = out_html_dir / html_name
        title = f"{clone} | {ab1_name} | {orientation} | id={identity:.4f}"
        write_alignment_html(html_path, title, ref_g, qry_g, ref2_s, ref_len)

    return {
        "sid": sid,
        "id": sid,
        "clone": clone,
        "ab1": ab1_name,
        "gb": gb_path.name,
        "orientation": orientation,
        "identity": round(identity, 6),
        "coverage": round(cds_cov, 4),
        "cds_coverage": round(cds_cov, 4),
        "frameshift": frameshift,
        "aa_changes": changes,
        "aa_changes_n": len(changes),
        "raw_aa_changes_n": raw_aa_changes_n,
        "has_indel": has_indel,
        "sub": sub,
        "ins": ins,
        "del": dele,
        "mutations": mutations,
        "seq_length": len(qry_used),
        "ref_length": ref_len,
        "cds_start": cds_start,
        "cds_end": cds_end,
        "avg_qry_quality": avg_qry_quality,
        "avg_quality": avg_qry_quality,
        "ref_sequence": ref_seq,
        "query_sequence": qry_used,
        "aligned_ref_g": full_ref_g,
        "aligned_query_g": full_qry_g,
        "aligned_query": qry_used,
        "matches": match_flags,
        "_cds_positions": cds_positions,
        **chrom,
    }

# ── Dataset analysis ────────────────────────────────────────────────────────

DATASET_LAYOUTS = {
    "base": [
        {"gb": "base/gb", "ab1": "base/ab1"},
        {"gb": "gb", "ab1": "ab1_files"},
    ],
    "pro": [
        {"gb": "pro/gb", "ab1": "pro/ab1"},
        {"gb": "gb_pro", "ab1": "ab1_files_pro"},
    ],
    "promax": [
        {"gb": "promax/gb", "ab1": "promax/ab1"},
        {"gb": "gb_promax", "ab1": "ab1_files_promax"},
    ],
}


def resolve_dataset_dirs(dataset: str, data_dir: Path) -> tuple[Path, Path]:
    layouts = DATASET_LAYOUTS[dataset]
    candidates: list[tuple[Path, Path]] = []

    for layout in layouts:
        gb_dir = data_dir / Path(layout["gb"])
        ab1_dir = data_dir / Path(layout["ab1"])
        candidates.append((gb_dir, ab1_dir))

    for gb_dir, ab1_dir in candidates:
        if gb_dir.exists() and ab1_dir.exists():
            return gb_dir, ab1_dir

    for gb_dir, ab1_dir in candidates:
        if gb_dir.exists() or ab1_dir.exists():
            return gb_dir, ab1_dir

    return candidates[0]


def analyze_dataset(dataset: str, data_dir: Path,
                    out_html_dir: Path | None = None) -> list[dict]:
    """Analyze all samples in a dataset. Returns list of structured dicts."""
    if dataset not in DATASET_LAYOUTS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_LAYOUTS.keys())}")

    gb_dir, ab1_dir = resolve_dataset_dirs(dataset, data_dir)

    if not gb_dir.exists():
        raise FileNotFoundError(f"GB directory not found: {gb_dir}")
    if not ab1_dir.exists():
        raise FileNotFoundError(f"AB1 directory not found: {ab1_dir}")

    gb_files = sorted(gb_dir.rglob("*.gb")) + sorted(gb_dir.rglob("*.gbk"))
    if not gb_files:
        raise FileNotFoundError(f"No .gb/.gbk files found in {gb_dir}")

    aligner = build_aligner()
    all_results = []

    for gb_path in gb_files:
        clone = parse_clone_from_gb(gb_path)
        ab1_list = find_ab1_for_clone(ab1_dir, clone)
        if not ab1_list:
            continue

        for ab1_path in ab1_list:
            print(f"  Analyzing {ab1_path.name} ...", end=" ", flush=True, file=sys.stderr)
            result = analyze_sample(
                gb_path=gb_path,
                ab1_path=ab1_path,
                aligner=aligner,
                out_html_dir=out_html_dir,
            )
            if result is None:
                print("SKIP (too short)", file=sys.stderr)
                continue
            print(f"id={result['identity']:.4f}  cds_cov={result['cds_coverage']:.3f}", file=sys.stderr)
            all_results.append(result)

    # Group by SID: if multiple AB1 files exist for the same SID,
    # keep the one with the highest identity as primary, but note duplicates
    by_sid = {}
    for r in all_results:
        sid = r["sid"]
        if sid not in by_sid:
            by_sid[sid] = []
        by_sid[sid].append(r)

    results = []
    for sid, entries in by_sid.items():
        if len(entries) == 1:
            best = dict(entries[0])
            best.pop("_cds_positions", None)
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
        else:
            # Pick best by identity; record info about other reads
            entries_sorted = sorted(entries, key=lambda x: x["identity"], reverse=True)
            best = dict(entries_sorted[0])  # copy
            best = apply_dual_read_consensus(best, entries_sorted[1:])
            other_ids = [f"{e['ab1']}(id={e['identity']:.4f},cov={e['cds_coverage']:.3f})" for e in entries_sorted[1:]]
            best["other_reads"] = other_ids
            best["dual_read"] = True

            # Merge CDS coverage from all reads
            cds_start = best.get("cds_start")
            cds_end = best.get("cds_end")
            if cds_start is not None and cds_end is not None:
                merged_positions = {}
                read_conflict = False
                for e in entries_sorted:
                    cds_pos = e.get("_cds_positions", {})
                    for pos, base in cds_pos.items():
                        if pos in merged_positions:
                            if merged_positions[pos] != base:
                                read_conflict = True
                        else:
                            merged_positions[pos] = base
                cds_len = cds_end - cds_start + 1
                total_cds_coverage = round(len(merged_positions) / cds_len, 4) if cds_len > 0 else 0.0
                best["total_cds_coverage"] = total_cds_coverage

                # Only report read_conflict when the best read is not authoritative
                best_authoritative = (best["identity"] >= 0.99 and best["cds_coverage"] >= 0.8)
                if not best_authoritative:
                    best["read_conflict"] = read_conflict

                # If any non-best read has frameshift or AA changes, note it.
                # Only report other_read_issues when best read is incomplete:
                # if best read already covers most of the CDS with high identity,
                # other reads' issues are likely noise.
                other_issues = []
                for e in entries_sorted[1:]:
                    issues = []
                    if e.get("frameshift"):
                        issues.append("frameshift")
                    if e.get("aa_changes"):
                        issues.append(f"aa:{' '.join(e['aa_changes'])}")
                    if issues:
                        other_issues.append(f"{e['ab1']}({','.join(issues)})")
                if other_issues:
                    if best_authoritative:
                        # Best read is authoritative; note but don't flag
                        best["other_read_notes"] = other_issues
                    else:
                        # Best read is incomplete; other reads provide new info
                        best["other_read_issues"] = other_issues

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

    return results
