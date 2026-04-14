from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from core.alignment import analyze_dataset
from core.evidence import format_evidence_for_llm, format_evidence_table
from core.llm_client import call_llm, parse_llm_result

from bioagent.mcp_tools import register_tool
from bioagent.mutation_trends import analyze_mutation_trends
from bioagent.lab_suggestions import generate_lab_suggestions


_ANALYSIS_HISTORY: list[dict] = []
_ANALYSIS_DETAILS: dict[str, dict] = {}
_REGISTERED = False
_DEFAULT_MODEL = "google/gemma-3-27b-it:free"


class ToolExecutionError(RuntimeError):
    """Raised when an MCP tool cannot complete successfully."""


def _project_root() -> Path:
    return _PACKAGE_ROOT


def _data_dir() -> Path:
    return _project_root() / "data"


def _normalize_output_dir(output_dir: str | None, dataset: str) -> Path:
    if output_dir:
        return Path(output_dir)
    return _project_root() / "outputs" / dataset


def _store_analysis(detail: dict) -> dict:
    analysis_id = detail["analysis_id"]
    detail_copy = deepcopy(detail)
    _ANALYSIS_DETAILS[analysis_id] = detail_copy

    history_item = {
        "analysis_id": analysis_id,
        "dataset": detail_copy["dataset"],
        "sample_count": detail_copy["sample_count"],
        "created_at": detail_copy["created_at"],
        "output_dir": detail_copy["output_dir"],
        "used_llm": detail_copy["used_llm"],
    }
    _ANALYSIS_HISTORY.insert(0, history_item)
    return history_item


def analyze_sequences(*, dataset: str, output_dir: str | None = None,
                      no_llm: bool = True, model: str = _DEFAULT_MODEL) -> dict:
    resolved_output_dir = _normalize_output_dir(output_dir, dataset)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = resolved_output_dir / "html"

    samples = analyze_dataset(dataset, _data_dir(), out_html_dir=html_dir)
    if not samples:
        raise ToolExecutionError("No samples analyzed. Check dataset inputs.")

    evidence_table = format_evidence_table(samples)
    evidence_path = resolved_output_dir / "evidence.txt"
    evidence_path.write_text(evidence_table, encoding="utf-8")

    result_lines = None
    raw_response = None
    if not no_llm:
        evidence_text = format_evidence_for_llm(samples)
        raw_response = call_llm(evidence_text, model=model)
        result_lines = parse_llm_result(raw_response)
        (resolved_output_dir / "llm_raw.txt").write_text(raw_response, encoding="utf-8")
        (resolved_output_dir / "result.txt").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    analysis_id = f"analysis-{uuid4().hex[:12]}"
    detail = {
        "analysis_id": analysis_id,
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(resolved_output_dir),
        "evidence_path": str(evidence_path),
        "used_llm": not no_llm,
        "model": model if not no_llm else None,
        "sample_count": len(samples),
        "samples": deepcopy(samples),
        "result_lines": result_lines,
        "llm_raw": raw_response,
    }
    history_item = _store_analysis(detail)

    return {
        "analysis_id": analysis_id,
        "dataset": dataset,
        "sample_count": len(samples),
        "output_dir": str(resolved_output_dir),
        "evidence_path": str(evidence_path),
        "used_llm": history_item["used_llm"],
    }


def query_history(*, limit: int = 20) -> dict:
    normalized_limit = max(0, int(limit))
    return {
        "items": deepcopy(_ANALYSIS_HISTORY[:normalized_limit]),
        "total": len(_ANALYSIS_HISTORY),
    }


def get_analysis_detail(*, analysis_id: str) -> dict:
    detail = _ANALYSIS_DETAILS.get(analysis_id)
    if detail is None:
        raise ToolExecutionError(f"Unknown analysis_id: {analysis_id}")
    return deepcopy(detail)


def _latest_analysis_samples() -> list[dict]:
    if not _ANALYSIS_HISTORY:
        return []
    latest_analysis_id = _ANALYSIS_HISTORY[0].get("analysis_id")
    if not latest_analysis_id:
        return []
    detail = _ANALYSIS_DETAILS.get(latest_analysis_id) or {}
    samples = detail.get("samples") or []
    return deepcopy(samples)


def _run_on_latest_samples(analyzer) -> dict:
    return analyzer(_latest_analysis_samples())


def detect_mutation_trends() -> dict:
    return _run_on_latest_samples(analyze_mutation_trends)


def build_lab_suggestions() -> dict:
    return _run_on_latest_samples(generate_lab_suggestions)


def register_initial_tools() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    register_tool(
        name="analyze_sequences",
        description="Analyze one built-in BioAgent dataset and store the result in in-process history.",
        parameters={
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "enum": ["base", "pro", "promax"]},
                "output_dir": {"type": "string"},
                "no_llm": {"type": "boolean", "default": True},
                "model": {"type": "string", "default": _DEFAULT_MODEL},
            },
            "required": ["dataset"],
        },
        execute=analyze_sequences,
    )
    register_tool(
        name="query_history",
        description="List recent analysis runs recorded by the current MCP server process.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 0, "default": 20},
            },
        },
        execute=query_history,
    )
    register_tool(
        name="get_analysis_detail",
        description="Return the stored detail for a prior analysis_id from the current MCP server process.",
        parameters={
            "type": "object",
            "properties": {
                "analysis_id": {"type": "string"},
            },
            "required": ["analysis_id"],
        },
        execute=get_analysis_detail,
    )
    register_tool(
        name="detect_mutation_trends",
        description="Analyze mutation patterns across the latest analysis and identify hotspots and insights.",
        parameters={
            "type": "object",
            "properties": {},
        },
        execute=detect_mutation_trends,
    )
    register_tool(
        name="generate_lab_suggestions",
        description="Generate experiment improvement suggestions from the latest analysis results.",
        parameters={
            "type": "object",
            "properties": {},
        },
        execute=build_lab_suggestions,
    )
    _REGISTERED = True


__all__ = [
    "ToolExecutionError",
    "analyze_sequences",
    "query_history",
    "get_analysis_detail",
    "register_initial_tools",
]
