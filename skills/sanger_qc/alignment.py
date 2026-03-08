#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
core/alignment.py
Bioinformatics core: GB/AB1 parsing, alignment, mutation detection, AA translation.
Merged and simplified from scripts/gb_ab1_mutations.py and scripts/gb_ab1_agent_pro.py.
"""

import re
from pathlib import Path
from datetime import datetime

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from Bio.Data import CodonTable


# ── Utilities ────────────────────────────────────────────────────────────────

def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{stamp}" + path.suffix)
        df.to_csv(alt, index=False)
        print(f"[WARN] Permission denied writing {path.name}. Saved as {alt.name}.")
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


def read_ab1_sequence(ab1_path: Path, do_trim=True, qmin=20, min_len=80):
    """Read AB1 file and return (trimmed_seq, trimmed_qual) tuple."""
    rec = SeqIO.read(str(ab1_path), "abi")
    raw_seq = str(rec.seq).upper()
    raw_qual = list(rec.letter_annotations.get("phred_quality", []))

    # Strip leading/trailing N from raw sequence, sync quality array
    lstrip = len(raw_seq) - len(raw_seq.lstrip("N"))
    rstrip = len(raw_seq) - len(raw_seq.rstrip("N"))
    if rstrip > 0:
        seq = raw_seq[lstrip:-rstrip]
        qual = raw_qual[lstrip:-rstrip] if raw_qual else []
    else:
        seq = raw_seq[lstrip:]
        qual = raw_qual[lstrip:] if raw_qual else []

    if do_trim and qual:
        seq, qual = trim_ab1_by_quality(seq, qual, qmin=qmin, min_len=min_len)
        # Strip N again after trimming
        lstrip2 = len(seq) - len(seq.lstrip("N"))
        rstrip2 = len(seq) - len(seq.rstrip("N"))
        if rstrip2 > 0:
            seq = seq[lstrip2:-rstrip2]
            qual = qual[lstrip2:-rstrip2]
        else:
            seq = seq[lstrip2:]
            qual = qual[lstrip2:]

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

    qry_raw, qry_qual = read_ab1_sequence(ab1_path, do_trim=do_trim, qmin=qmin, min_len=min_len)
    if len(qry_raw) < 30:
        return None

    orientation, best_aln, ref_g, qry_g, ref2_s, ref2_e, qry_used = \
        pick_best_orientation(ref2, qry_raw, aligner)

    # For REVERSE orientation, the quality array must be reversed to match qry_used (RC)
    if orientation == "REVERSE" and qry_qual:
        qry_qual_used = list(reversed(qry_qual))
    else:
        qry_qual_used = qry_qual

    matches, aligned_both, identity, sub, ins, dele = compute_stats(ref_g, qry_g)

    cds_cov = compute_cds_coverage(ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end)
    frameshift = detect_frameshift(ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end)

    # Get query alignment start position from alignment coordinates
    qry_aln_start = int(best_aln.coordinates[1][0])

    ok, changes, has_indel, raw_aa_changes_n = aa_changes_from_cds(
        ref_seq=ref_seq, ref_len=ref_len,
        cds_start=cds_start, cds_end=cds_end,
        ref_g=ref_g, qry_g=qry_g, ref2_start=ref2_s,
        qry_qual=qry_qual_used, qry_aln_start=qry_aln_start,
    )

    # Compute quality metrics
    avg_qry_quality = round(sum(qry_qual) / len(qry_qual), 1) if qry_qual else None

    # Compute CDS covered positions for multi-read merging
    cds_positions = compute_cds_covered_positions(
        ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end
    )

    # HTML output
    if out_html_dir is not None:
        out_html_dir.mkdir(parents=True, exist_ok=True)
        html_name = f"{clone}__{ab1_path.stem}__{orientation}.html"
        html_path = out_html_dir / html_name
        title = f"{clone} | {ab1_name} | {orientation} | id={identity:.4f}"
        write_alignment_html(html_path, title, ref_g, qry_g, ref2_s, ref_len)

    return {
        "sid": sid,
        "clone": clone,
        "ab1": ab1_name,
        "gb": gb_path.name,
        "orientation": orientation,
        "identity": round(identity, 6),
        "cds_coverage": round(cds_cov, 4),
        "frameshift": frameshift,
        "aa_changes": changes,
        "aa_changes_n": len(changes),
        "raw_aa_changes_n": raw_aa_changes_n,
        "has_indel": has_indel,
        "sub": sub,
        "ins": ins,
        "del": dele,
        "seq_length": len(qry_used),
        "ref_length": ref_len,
        "cds_start": cds_start,
        "cds_end": cds_end,
        "avg_qry_quality": avg_qry_quality,
        "_cds_positions": cds_positions,
    }


# ── Dataset analysis ─────────────────────────────────────────────────────────

DATASET_MAP = {
    "base": {"gb": "gb", "ab1": "ab1_files"},
    "pro": {"gb": "gb_pro", "ab1": "ab1_files_pro"},
    "promax": {"gb": "gb_promax", "ab1": "ab1_files_promax"},
}


def analyze_dataset(dataset: str, data_dir: Path,
                    out_html_dir: Path | None = None) -> list[dict]:
    """Analyze all samples in a dataset. Returns list of structured dicts."""
    if dataset not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_MAP.keys())}")

    dirs = DATASET_MAP[dataset]
    gb_dir = data_dir / dirs["gb"]
    ab1_dir = data_dir / dirs["ab1"]

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
            print(f"  Analyzing {ab1_path.name} ...", end=" ", flush=True)
            result = analyze_sample(
                gb_path=gb_path,
                ab1_path=ab1_path,
                aligner=aligner,
                out_html_dir=out_html_dir,
            )
            if result is None:
                print("SKIP (too short)")
                continue
            print(f"id={result['identity']:.4f}  cds_cov={result['cds_coverage']:.3f}")
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
            results.append(best)
        else:
            # Pick best by identity; record info about other reads
            entries_sorted = sorted(entries, key=lambda x: x["identity"], reverse=True)
            best = dict(entries_sorted[0])  # copy
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

            best.pop("_cds_positions", None)
            results.append(best)

    return results
