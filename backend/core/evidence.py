#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
core/evidence.py
Format alignment results into text evidence for LLM consumption.
"""


def format_evidence_for_llm(samples: list[dict]) -> str:
    """Format all sample analysis results as structured text for LLM."""
    lines = []
    lines.append(f"共 {len(samples)} 个样本的Sanger测序比对分析结果：\n")

    for s in samples:
        lines.append(f"--- 样本 {s['sid']} ---")
        lines.append(f"  clone: {s['clone']}")
        lines.append(f"  identity: {s['identity']}")
        lines.append(f"  cds_coverage: {s['cds_coverage']}")
        lines.append(f"  frameshift: {s['frameshift']}")
        lines.append(f"  aa_changes: {s['aa_changes']} (共{s['aa_changes_n']}个)")
        if s.get("raw_aa_changes_n") is not None and s["raw_aa_changes_n"] != s["aa_changes_n"]:
            lines.append(f"  raw_aa_changes_n: {s['raw_aa_changes_n']} (质量过滤前)")
        lines.append(f"  has_indel: {s['has_indel']}")
        lines.append(f"  sub/ins/del: {s['sub']}/{s['ins']}/{s['del']}")
        lines.append(f"  seq_length: {s['seq_length']}")
        lines.append(f"  ref_length: {s['ref_length']}")
        if s.get("avg_qry_quality") is not None:
            lines.append(f"  avg_qry_quality: {s['avg_qry_quality']}")
        if s.get("dual_read"):
            lines.append(f"  dual_read: True")
        if s.get("total_cds_coverage") is not None:
            lines.append(f"  total_cds_coverage: {s['total_cds_coverage']}")
        if s.get("read_conflict") is not None:
            lines.append(f"  read_conflict: {s['read_conflict']}")
        if s.get("other_reads"):
            lines.append(f"  other_reads: {', '.join(s['other_reads'])}")
        if s.get("other_read_issues"):
            lines.append(f"  other_read_issues: {', '.join(s['other_read_issues'])}")
        if s.get("other_read_notes"):
            lines.append(f"  other_read_notes: {', '.join(s['other_read_notes'])} (主读段已充分覆盖，仅供参考)")
        lines.append("")

    return "\n".join(lines)


def format_evidence_table(samples: list[dict]) -> str:
    """Format as a compact table for LLM."""
    header = (
        f"{'SID':<10} {'identity':>8} {'cds_cov':>7} {'frame':>5} "
        f"{'aa_n':>4} {'raw':>4} {'sub':>3} {'ins':>3} {'del':>3} "
        f"{'seqlen':>6} {'avgQ':>5} {'aa_changes'}"
    )
    lines = [header, "-" * len(header)]

    for s in samples:
        aa_str = " ".join(s["aa_changes"]) if s["aa_changes"] else "-"
        frame_str = "YES" if s["frameshift"] else "no"
        raw_n = s.get("raw_aa_changes_n", s["aa_changes_n"])
        avg_q = f"{s['avg_qry_quality']:>5.1f}" if s.get("avg_qry_quality") is not None else "  N/A"
        lines.append(
            f"{s['sid']:<10} {s['identity']:>8.4f} {s['cds_coverage']:>7.3f} {frame_str:>5} "
            f"{s['aa_changes_n']:>4} {raw_n:>4} {s['sub']:>3} {s['ins']:>3} {s['del']:>3} "
            f"{s['seq_length']:>6} {avg_q} {aa_str}"
        )

    return "\n".join(lines)
