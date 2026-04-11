"""Fast sequence alignment for Sanger QC."""

import re
from typing import List, Tuple, Optional, Dict
from Bio import Align
from Bio.Seq import Seq
from Bio.Data import CodonTable

from .models import AlignmentResult, Mutation


def create_aligner() -> Align.PairwiseAligner:
    """Create aligner optimized for Sanger sequencing."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -1.0
    return aligner


def alignment_to_gapped(ref: str, qry: str, aln) -> Tuple[str, str, int, int]:
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


def pick_best_orientation(ref2: str, qry: str, aligner) -> Tuple[str, any, str, str, int, int, str]:
    """Try forward and reverse-complement; return best."""
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


def compute_stats(ref_g: str, qry_g: str) -> Tuple[int, int, float, int, int, int]:
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


def ref2pos_to_refpos(ref2_pos_0based: int, ref_len: int, is_circular: bool = True) -> int:
    """0-based ref2 position -> 1-based original ref position."""
    if is_circular:
        return (ref2_pos_0based % ref_len) + 1
    return ref2_pos_0based + 1


def extract_mutations(
    ref_g: str, qry_g: str, ref2_start: int, ref_len: int, is_circular: bool = True
) -> List[Mutation]:
    muts = []
    ref2_cursor = ref2_start
    last_refpos = None
    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len, is_circular)
            last_refpos = refpos
            ref2_cursor += 1
        else:
            refpos = last_refpos
        
        if a == "-" and b != "-":
            muts.append(Mutation(position=refpos or 1, ref_base="-", query_base=b, type="insertion"))
        elif a != "-" and b == "-":
            muts.append(Mutation(position=refpos, ref_base=a, query_base="-", type="deletion"))
        elif a != "-" and b != "-" and a.upper() != b.upper():
            muts.append(Mutation(position=refpos, ref_base=a, query_base=b, type="substitution"))
    return muts


STD_TABLE = CodonTable.unambiguous_dna_by_id[1]


def translate_codon(codon: str) -> Optional[str]:
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
                        qual_min=30,
                        is_circular: bool = True) -> Tuple[bool, List[str], bool, int]:
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
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len, is_circular)
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
            if qry_qual is not None:
                codon_quals = qry_qual_by_refpos[i:i + 3]
                if any(q is not None and q < qual_min for q in codon_quals):
                    continue
            filtered_changes.append(change_str)

    ok = (len(filtered_changes) == 0) and (not has_indel)
    return ok, filtered_changes, has_indel, len(raw_changes)


def compute_cds_coverage(ref_g: str, qry_g: str, ref2_start: int,
                         ref_len: int, cds_start: int, cds_end: int,
                         is_circular: bool = True) -> float:
    if cds_start is None or cds_end is None:
        return 0.0
    cds_len = cds_end - cds_start + 1
    if cds_len <= 0:
        return 0.0

    covered = 0
    ref2_cursor = ref2_start
    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len, is_circular)
            ref2_cursor += 1
            if cds_start <= refpos <= cds_end and b != "-":
                covered += 1
    return covered / cds_len


def compute_cds_covered_positions(ref_g: str, qry_g: str, ref2_start: int,
                                  ref_len: int, cds_start: int, cds_end: int,
                                  is_circular: bool = True) -> Dict[int, str]:
    if cds_start is None or cds_end is None:
        return {}
    positions = {}
    ref2_cursor = ref2_start
    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len, is_circular)
            ref2_cursor += 1
            if cds_start <= refpos <= cds_end and b != "-":
                positions[refpos] = b
    return positions


def detect_frameshift(ref_g: str, qry_g: str, ref2_start: int,
                      ref_len: int, cds_start: int, cds_end: int,
                      is_circular: bool = True) -> bool:
    if cds_start is None or cds_end is None:
        return False

    cds_ins = 0
    cds_del = 0
    ref2_cursor = ref2_start
    last_refpos = None

    for a, b in zip(ref_g, qry_g):
        if a != "-":
            refpos = ref2pos_to_refpos(ref2_cursor, ref_len, is_circular)
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


def analyze_sample(
    sample_id: str,
    ref_seq: str,
    query_seq: str,
    cds_start: int,
    cds_end: int,
    query_qual: Optional[List[int]] = None,
    aligner: Optional[Align.PairwiseAligner] = None,
    is_circular: bool = True,
) -> AlignmentResult:
    """Analyze a single sample with advanced logic."""
    if aligner is None:
        aligner = create_aligner()

    ref_len = len(ref_seq)
    ref2 = ref_seq + ref_seq if is_circular else ref_seq

    orientation, best_aln, ref_g, qry_g, ref2_s, ref2_e, qry_used = \
        pick_best_orientation(ref2, query_seq, aligner)

    if orientation == "REVERSE" and query_qual:
        qry_qual_used = list(reversed(query_qual))
    else:
        qry_qual_used = query_qual

    matches_count, aligned_both, identity, sub, ins, dele = compute_stats(ref_g, qry_g)
    cds_cov = compute_cds_coverage(
        ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end, is_circular=is_circular
    )
    frameshift = detect_frameshift(
        ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end, is_circular=is_circular
    )
    
    qry_aln_start = int(best_aln.coordinates[1][0])
    ok, changes, has_indel, raw_aa_changes_n = aa_changes_from_cds(
        ref_seq=ref_seq, ref_len=ref_len,
        cds_start=cds_start, cds_end=cds_end,
        ref_g=ref_g, qry_g=qry_g, ref2_start=ref2_s,
        qry_qual=qry_qual_used, qry_aln_start=qry_aln_start,
        is_circular=is_circular,
    )

    mutations = extract_mutations(ref_g, qry_g, ref2_s, ref_len, is_circular=is_circular)
    cds_positions = compute_cds_covered_positions(
        ref_g, qry_g, ref2_s, ref_len, cds_start, cds_end, is_circular=is_circular
    )

    avg_qry_quality = round(sum(query_qual) / len(query_qual), 1) if query_qual else None

    if ref2_s < ref_len and ref2_e <= ref_len:
        padding_left = ref2_s
        padding_right = ref_len - ref2_e
        full_ref_g = ref_seq[:padding_left] + ref_g + ref_seq[ref2_e:]
        full_qry_g = "-" * padding_left + qry_g + "-" * padding_right

        matches = []
        matches.extend([True] * padding_left)
        for a, b in zip(ref_g, qry_g):
            if a != "-" and b != "-":
                matches.append(a.upper() == b.upper())
            else:
                matches.append(False)
        matches.extend([True] * padding_right)
    else:
        full_ref_g = ref_g
        full_qry_g = qry_g
        matches = [
            a.upper() == b.upper() if a != "-" and b != "-" else False
            for a, b in zip(full_ref_g, full_qry_g)
        ]

    return AlignmentResult(
        sample_id=sample_id,
        ref_sequence=ref_seq,
        query_sequence=query_seq,
        aligned_ref_g=full_ref_g,
        aligned_query_g=full_qry_g,
        aligned_query=qry_used,
        matches=matches,
        mutations=mutations,
        cds_start=cds_start,
        cds_end=cds_end,
        frameshift=frameshift,
        identity=identity,
        coverage=cds_cov,
        orientation=orientation,
        aa_changes=changes,
        aa_changes_n=len(changes),
        raw_aa_changes_n=raw_aa_changes_n,
        has_indel=has_indel,
        sub=sub,
        ins=ins,
        dele=dele,
        seq_length=len(qry_used),
        ref_length=ref_len,
        avg_qry_quality=avg_qry_quality,
        _cds_positions=cds_positions,
    )
