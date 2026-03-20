# backend/mcp_server.py
"""Standalone MCP Server for Claude Code integration."""
import sys
import json
import logging
from pathlib import Path

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server

from core.alignment import analyze_dataset
from core.evidence import format_evidence_for_llm, format_evidence_table
from core.rules import judge_batch, load_thresholds
from db.database import init_db, get_session_factory
from db.models import Analysis, Sample, new_id

logging.basicConfig(level=logging.INFO)
server = Server("bioagent")


@server.tool()
async def analyze_directory(gb_dir: str, ab1_dir: str, name: str = "") -> str:
    """分析指定目录下的所有 Sanger 测序样本。返回判读结果摘要。结果同时存入数据库，可在 Web 端查看。"""
    from datetime import datetime, timezone

    gb_path, ab1_path = Path(gb_dir), Path(ab1_dir)
    if not gb_path.exists():
        return f"错误: GenBank 目录不存在: {gb_dir}"
    if not ab1_path.exists():
        return f"错误: AB1 目录不存在: {ab1_dir}"

    samples = analyze_dataset(gb_path, ab1_path)
    if not samples:
        return "未发现可分析的样本"

    thresholds = load_thresholds()
    judgments = judge_batch(samples, thresholds)

    # Save to database (same as Web frontend)
    init_db()
    Session = get_session_factory()
    session = Session()
    ok = sum(1 for j in judgments if j["status"] == "ok")
    wrong = sum(1 for j in judgments if j["status"] == "wrong")
    uncertain = sum(1 for j in judgments if j["status"] == "uncertain")

    analysis = Analysis(
        id=new_id(),
        name=name or f"MCP 分析 {datetime.now().strftime('%m-%d %H:%M')}",
        source_type="scan", source_path=ab1_dir,
        status="done", total=len(samples),
        ok_count=ok, wrong_count=wrong, uncertain_count=uncertain,
        config_snapshot=json.dumps(thresholds),
        finished_at=datetime.now(timezone.utc),
    )
    session.add(analysis)

    for sd, jd in zip(samples, judgments):
        session.add(Sample(
            id=new_id(), analysis_id=analysis.id,
            sid=sd["sid"], clone=sd.get("clone", ""),
            status=jd["status"], reason=jd.get("reason", ""),
            rule_id=jd.get("rule"),
            identity=sd["identity"], cds_coverage=sd["cds_coverage"],
            frameshift=sd["frameshift"],
            aa_changes=json.dumps(sd.get("aa_changes", [])),
            aa_changes_n=sd.get("aa_changes_n", 0),
            raw_aa_changes_n=sd.get("raw_aa_changes_n", 0),
            orientation=sd.get("orientation", ""),
            seq_length=sd.get("seq_length", 0),
            ref_length=sd.get("ref_length", 0),
            avg_quality=sd.get("avg_qry_quality"),
            sub_count=sd.get("sub", 0), ins_count=sd.get("ins", 0),
            del_count=sd.get("del", 0),
            ref_gapped=sd.get("ref_gapped", ""),
            qry_gapped=sd.get("qry_gapped", ""),
            quality_scores=json.dumps(sd.get("quality_scores", []) or []),
            raw_data=json.dumps(sd, default=str),
        ))
    session.commit()
    session.close()

    # Build response text
    lines = []
    for s, j in zip(samples, judgments):
        lines.append(f"{s['sid']} gene is {j['status']} {j.get('reason', '')}")
    lines.append(f"\n共 {len(samples)} 个样本 (analysis_id: {analysis.id})")
    lines.append(format_evidence_table(samples))
    return "\n".join(lines)


@server.tool()
async def scan_directory(directory: str) -> str:
    """扫描目录，发现可分析的 AB1 和 GB 文件。"""
    base = Path(directory)
    if not base.exists():
        return f"错误: 目录不存在: {directory}"
    gb = sorted([str(p) for p in base.rglob("*.gb")] + [str(p) for p in base.rglob("*.gbk")])
    ab1 = sorted([str(p) for p in base.rglob("*.ab1")])
    return f"GenBank 文件 ({len(gb)}):\n" + "\n".join(gb) + f"\n\nAB1 文件 ({len(ab1)}):\n" + "\n".join(ab1)


@server.tool()
async def analyze_files(ab1_paths: str, gb_path: str) -> str:
    """分析指定的 AB1 文件和 GenBank 参考序列。ab1_paths 为逗号分隔的路径列表。"""
    from core.alignment import analyze_sample, build_aligner, load_genbank
    paths = [Path(p.strip()) for p in ab1_paths.split(",")]
    gb = Path(gb_path)
    if not gb.exists():
        return f"错误: GenBank 文件不存在: {gb_path}"
    aligner = build_aligner()
    results = []
    thresholds = load_thresholds()
    for ab1 in paths:
        if not ab1.exists():
            results.append(f"{ab1.name}: 文件不存在")
            continue
        sample = analyze_sample(gb, ab1, aligner)
        if sample is None:
            results.append(f"{ab1.name}: 序列过短，跳过")
            continue
        judgment = judge_batch([sample], thresholds)[0]
        results.append(f"{sample['sid']} gene is {judgment['status']} {judgment.get('reason', '')}")
    return "\n".join(results)


@server.tool()
async def get_sample_detail(analysis_id: str, sample_id: str) -> str:
    """获取单个样本的详细分析数据。"""
    init_db()
    Session = get_session_factory()
    session = Session()
    from db.models import Sample
    sample = session.query(Sample).filter(
        Sample.analysis_id == analysis_id, Sample.sid == sample_id
    ).first()
    session.close()
    if not sample:
        return f"未找到样本: analysis={analysis_id}, sid={sample_id}"
    return json.dumps({
        "sid": sample.sid, "status": sample.status, "reason": sample.reason,
        "identity": sample.identity, "cds_coverage": sample.cds_coverage,
        "frameshift": sample.frameshift, "aa_changes_n": sample.aa_changes_n,
        "seq_length": sample.seq_length, "avg_quality": sample.avg_quality,
    }, ensure_ascii=False, indent=2)


@server.tool()
async def get_analysis_summary(analysis_id: str) -> str:
    """获取一次分析的汇总结果。"""
    init_db()
    Session = get_session_factory()
    session = Session()
    analysis = session.get(Analysis, analysis_id)
    if not analysis:
        session.close()
        return f"未找到分析记录: {analysis_id}"
    from db.models import Sample
    samples = session.query(Sample).filter(Sample.analysis_id == analysis_id).all()
    session.close()
    lines = [f"分析: {analysis.name} ({analysis.status})",
             f"总计: {analysis.total} | OK: {analysis.ok_count} | Wrong: {analysis.wrong_count} | Uncertain: {analysis.uncertain_count}",
             ""]
    for s in samples:
        lines.append(f"  {s.sid}: {s.status} {s.reason or ''}")
    return "\n".join(lines)


@server.tool()
async def export_report(analysis_id: str, format: str = "csv") -> str:
    """导出分析报告为 CSV 文本。"""
    init_db()
    Session = get_session_factory()
    session = Session()
    from db.models import Sample
    samples = session.query(Sample).filter(Sample.analysis_id == analysis_id).all()
    session.close()
    if not samples:
        return f"未找到样本数据: {analysis_id}"
    lines = ["SID,Status,Reason,Identity,CDS_Coverage,AA_Changes_N"]
    for s in samples:
        lines.append(f"{s.sid},{s.status},{s.reason},{s.identity},{s.cds_coverage},{s.aa_changes_n}")
    return "\n".join(lines)


@server.tool()
async def update_thresholds(overrides: str) -> str:
    """临时调整判读阈值（仅内存，不写入文件）。overrides 为 JSON 字符串。
    如需重新分析，请在调用后使用 analyze_directory。"""
    try:
        updates = json.loads(overrides)
    except json.JSONDecodeError:
        return "错误: overrides 必须是合法 JSON"
    current = load_thresholds()
    current.update(updates)
    return f"阈值已临时更新: {json.dumps(current, indent=2)}\n提示: 请使用 analyze_directory 重新分析以应用新阈值。"


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
