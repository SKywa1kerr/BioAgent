"""Format alignment results into text evidence for LLM consumption."""

from typing import Dict, List


def format_evidence_for_llm(samples: List[Dict]) -> str:
    """Format all sample analysis results as structured text for LLM."""
    lines = []
    lines.append(f"共 {len(samples)} 个样本的Sanger测序比对分析结果：\n")

    for s in samples:
        sample_key = s.get("clone") or s.get("sample_id") or s.get("id") or "UNKNOWN"
        lines.append(f"--- 样本 {sample_key} ---")
        lines.append(f"  clone: {s.get('clone', '')}")
        lines.append(f"  identity: {s.get('identity')}")
        lines.append(f"  cds_coverage: {s.get('cds_coverage', s.get('coverage'))}")
        lines.append(f"  frameshift: {s.get('frameshift')}")
        lines.append(f"  aa_changes: {s.get('aa_changes', [])} (共{s.get('aa_changes_n', 0)}个)")
        if s.get("status") is not None:
            lines.append(f"  deterministic_status: {s['status']}")
        if s.get("reason") is not None:
            lines.append(f"  deterministic_reason: {s['reason']}")
        if s.get("rule_id") is not None:
            lines.append(f"  deterministic_rule: {s['rule_id']}")
        if s.get("raw_aa_changes_n") is not None and s.get("raw_aa_changes_n") != s.get("aa_changes_n"):
            lines.append(f"  raw_aa_changes_n: {s['raw_aa_changes_n']} (质量过滤前)")
        lines.append(f"  has_indel: {s.get('has_indel')}")
        lines.append(f"  sub/ins/del: {s.get('sub', 0)}/{s.get('ins', 0)}/{s.get('dele', s.get('del', 0))}")
        lines.append(f"  seq_length: {s.get('seq_length')}")
        lines.append(f"  ref_length: {s.get('ref_length')}")
        if s.get("avg_qry_quality") is not None:
            lines.append(f"  avg_qry_quality: {s['avg_qry_quality']}")
        if s.get("dual_read"):
            lines.append("  dual_read: True")
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


def format_evidence_table(samples: List[Dict]) -> str:
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
        sample_key = s.get("clone") or s.get("sample_id") or s.get("id") or "UNKNOWN"
        lines.append(
            f"{sample_key:<10} {s['identity']:>8.4f} {s['cds_coverage']:>7.3f} {frame_str:>5} "
            f"{s['aa_changes_n']:>4} {raw_n:>4} {s['sub']:>3} {s['ins']:>3} {s['dele']:>3} "
            f"{s['seq_length']:>6} {avg_q} {aa_str}"
        )

    return "\n".join(lines)
