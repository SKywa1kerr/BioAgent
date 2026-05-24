from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from core.alignment import DATASET_LAYOUTS, analyze_dataset, analyze_dirs
from core.evidence import format_evidence_for_llm, format_evidence_table
from core.llm_client import call_llm, parse_llm_result

from bioagent.bio_primitives import (
    analyze_files_adhoc,
    compare_analyses,
    compare_sequences,
    export_analysis_html,
    inspect_path,
    read_pdf,
    read_sequence_file,
    translate_sequence,
)
from bioagent.mcp_tools import register_tool
from bioagent.mutation_trends import analyze_mutation_trends
from bioagent.lab_suggestions import generate_lab_suggestions


_ANALYSIS_HISTORY: list[dict] = []
_ANALYSIS_DETAILS: dict[str, dict] = {}
_USER_DATASETS: list[dict] = []
_REGISTERED = False
_BOOTSTRAPPED = False
_DEFAULT_MODEL = "google/gemma-3-27b-it:free"


class ToolExecutionError(RuntimeError):
    """Raised when an MCP tool cannot complete successfully."""


def _project_root() -> Path:
    return _PACKAGE_ROOT


def _builtin_data_dir() -> Path:
    return _project_root() / "data"


def _persist_root() -> Path:
    raw = os.environ.get("BIOAGENT_DATA_DIR")
    root = Path(raw) if raw else (Path.home() / ".bioagent")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _history_index_path() -> Path:
    return _persist_root() / "history.json"


def _details_dir() -> Path:
    d = _persist_root() / "details"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _datasets_path() -> Path:
    return _persist_root() / "datasets.json"


def _bootstrap_from_disk() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    if not _ANALYSIS_HISTORY:
        idx = _history_index_path()
        if idx.exists():
            try:
                data = json.loads(idx.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    _ANALYSIS_HISTORY[:] = data
            except Exception:
                pass

    if not _ANALYSIS_DETAILS:
        dd = _details_dir()
        if dd.exists():
            for fp in dd.glob("*.json"):
                try:
                    detail = json.loads(fp.read_text(encoding="utf-8"))
                    aid = detail.get("analysis_id")
                    if aid:
                        _ANALYSIS_DETAILS[aid] = detail
                except Exception:
                    pass

    if not _USER_DATASETS:
        dp = _datasets_path()
        if dp.exists():
            try:
                data = json.loads(dp.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    _USER_DATASETS[:] = data
            except Exception:
                pass


def _persist_history_index() -> None:
    try:
        _history_index_path().write_text(
            json.dumps(_ANALYSIS_HISTORY, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _persist_detail(detail: dict) -> None:
    aid = detail.get("analysis_id")
    if not aid:
        return
    try:
        (_details_dir() / f"{aid}.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def _persist_datasets() -> None:
    try:
        _datasets_path().write_text(
            json.dumps(_USER_DATASETS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


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
    _ANALYSIS_HISTORY[:] = [h for h in _ANALYSIS_HISTORY if h.get("analysis_id") != analysis_id]
    _ANALYSIS_HISTORY.insert(0, history_item)

    _persist_detail(detail_copy)
    _persist_history_index()
    return history_item


def _find_user_dataset(name: str) -> dict | None:
    for ds in _USER_DATASETS:
        if ds.get("id") == name or ds.get("label") == name:
            return ds
    return None


def analyze_sequences(*, dataset: str | None = None,
                      ab1_dir: str | None = None,
                      gb_dir: str | None = None,
                      output_dir: str | None = None,
                      no_llm: bool = True, model: str = _DEFAULT_MODEL) -> dict:
    _bootstrap_from_disk()

    has_dataset = dataset is not None and str(dataset).strip() != ""
    has_ab1 = ab1_dir is not None and str(ab1_dir).strip() != ""
    has_gb = gb_dir is not None and str(gb_dir).strip() != ""
    has_dirs = has_ab1 and has_gb

    if has_dataset and (has_ab1 or has_gb):
        raise ToolExecutionError(
            "Provide either dataset or (ab1_dir + gb_dir), not both."
        )
    if not has_dataset and not has_dirs:
        if has_ab1 ^ has_gb:
            raise ToolExecutionError(
                "Both ab1_dir and gb_dir must be provided together."
            )
        raise ToolExecutionError(
            "Either dataset or both ab1_dir and gb_dir must be provided."
        )

    user_match = _find_user_dataset(dataset) if has_dataset else None
    if has_dataset and user_match is None and dataset not in DATASET_LAYOUTS:
        raise ToolExecutionError(f"Unknown dataset: {dataset}")

    effective_dataset = dataset if has_dataset else "dropped"
    resolved_output_dir = _normalize_output_dir(output_dir, effective_dataset)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = resolved_output_dir / "html"

    if user_match is not None:
        gb_path = Path(user_match["gb_dir"])
        ab1_path = Path(user_match["ab1_dir"])
        samples = analyze_dirs(gb_path, ab1_path, out_html_dir=html_dir)
    elif has_dataset:
        samples = analyze_dataset(dataset, _builtin_data_dir(), out_html_dir=html_dir)
    else:
        samples = analyze_dirs(Path(gb_dir), Path(ab1_dir), out_html_dir=html_dir)

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
        "dataset": effective_dataset,
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
        "dataset": effective_dataset,
        "sample_count": len(samples),
        "output_dir": str(resolved_output_dir),
        "evidence_path": str(evidence_path),
        "used_llm": history_item["used_llm"],
    }


def query_history(*, limit: int = 20) -> dict:
    _bootstrap_from_disk()
    normalized_limit = max(0, int(limit))
    return {
        "items": deepcopy(_ANALYSIS_HISTORY[:normalized_limit]),
        "total": len(_ANALYSIS_HISTORY),
    }


def get_analysis_detail(*, analysis_id: str) -> dict:
    _bootstrap_from_disk()
    detail = _ANALYSIS_DETAILS.get(analysis_id)
    if detail is None:
        raise ToolExecutionError(f"Unknown analysis_id: {analysis_id}")
    return deepcopy(detail)


def _samples_for_analysis(analysis_id: str | None) -> tuple[list[dict], str | None]:
    if analysis_id:
        detail = _ANALYSIS_DETAILS.get(analysis_id)
        if detail is None:
            raise ToolExecutionError(f"Unknown analysis_id: {analysis_id}")
        return deepcopy(detail.get("samples") or []), analysis_id
    if not _ANALYSIS_HISTORY:
        return [], None
    latest_id = _ANALYSIS_HISTORY[0].get("analysis_id")
    if not latest_id:
        return [], None
    detail = _ANALYSIS_DETAILS.get(latest_id) or {}
    return deepcopy(detail.get("samples") or []), latest_id


def _attach_to_detail(analysis_id: str | None, key: str, payload: dict) -> None:
    if not analysis_id:
        return
    detail = _ANALYSIS_DETAILS.get(analysis_id)
    if not detail:
        return
    detail[key] = deepcopy(payload)
    _persist_detail(detail)


def detect_mutation_trends(*, analysis_id: str | None = None) -> dict:
    _bootstrap_from_disk()
    samples, aid = _samples_for_analysis(analysis_id)
    result = analyze_mutation_trends(samples)
    _attach_to_detail(aid, "trends", result)
    return result


def build_lab_suggestions(*, analysis_id: str | None = None) -> dict:
    _bootstrap_from_disk()
    samples, aid = _samples_for_analysis(analysis_id)
    result = generate_lab_suggestions(samples)
    _attach_to_detail(aid, "suggestions", result)
    return result


def list_datasets() -> dict:
    _bootstrap_from_disk()
    # Built-in datasets (base/pro/promax) were originally surfaced here for
    # dev convenience, but they are NOT shipped with the packaged app (the
    # data/ folder lives only in the repo). Exposing them in the sidebar
    # confused users into clicking entries that don't exist in production.
    # Opt-in via BIOAGENT_SHOW_BUILTINS=1 if you really want them back during
    # local development.
    builtin = []
    if os.environ.get("BIOAGENT_SHOW_BUILTINS") == "1":
        builtin_root = _builtin_data_dir()
        if builtin_root.is_dir():
            from core.alignment import resolve_dataset_dirs
            for name in DATASET_LAYOUTS.keys():
                try:
                    gb_dir, ab1_dir = resolve_dataset_dirs(name, builtin_root)
                except Exception:
                    continue
                if gb_dir.exists() and ab1_dir.exists():
                    builtin.append({"id": name, "label": name, "kind": "builtin"})
    user = [
        {
            "id": ds.get("id"),
            "label": ds.get("label", ds.get("id")),
            "kind": "user",
            "ab1_dir": ds.get("ab1_dir"),
            "gb_dir": ds.get("gb_dir"),
            "created_at": ds.get("created_at"),
        }
        for ds in _USER_DATASETS
    ]
    return {"builtin": builtin, "user": user}


def register_dataset(*, label: str, ab1_dir: str, gb_dir: str) -> dict:
    _bootstrap_from_disk()
    if not label or not ab1_dir or not gb_dir:
        raise ToolExecutionError("label, ab1_dir, and gb_dir are required.")
    # Reject labels that are only whitespace or only punctuation (e.g. "·",
    # "---"). Without this the sidebar shows a phantom row with no readable
    # name and clicking it prefills a malformed "分析 · 数据集" prompt.
    label_str = str(label).strip()
    if not any(c.isalnum() for c in label_str):
        raise ToolExecutionError(
            "label must contain at least one letter, digit, or CJK character."
        )
    if not Path(ab1_dir).is_dir():
        raise ToolExecutionError(f"ab1_dir does not exist or is not a directory: {ab1_dir}")
    if not Path(gb_dir).is_dir():
        raise ToolExecutionError(f"gb_dir does not exist or is not a directory: {gb_dir}")

    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in label_str.lower())[:32] or "dataset"
    existing_ids = {ds.get("id") for ds in _USER_DATASETS}
    candidate = base
    i = 1
    while candidate in existing_ids or candidate in DATASET_LAYOUTS:
        candidate = f"{base}_{i}"
        i += 1

    entry = {
        "id": candidate,
        "label": label_str,
        "ab1_dir": str(ab1_dir),
        "gb_dir": str(gb_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _USER_DATASETS.insert(0, entry)
    _persist_datasets()
    return entry


def delete_dataset(*, id: str) -> dict:
    _bootstrap_from_disk()
    before = len(_USER_DATASETS)
    _USER_DATASETS[:] = [ds for ds in _USER_DATASETS if ds.get("id") != id]
    if len(_USER_DATASETS) == before:
        raise ToolExecutionError(f"Unknown dataset id: {id}")
    _persist_datasets()
    return {"id": id, "deleted": True}


def register_initial_tools() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    _bootstrap_from_disk()

    register_tool(
        name="analyze_sequences",
        description=(
            "Analyze a dataset and store the result in history. The 'dataset' parameter accepts "
            "any built-in name (e.g. base/pro/promax) or any user-registered dataset id/label. "
            "Alternatively, omit 'dataset' and pass ab1_dir+gb_dir to analyze any folder pair on disk."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "ab1_dir": {"type": "string"},
                "gb_dir": {"type": "string"},
                "output_dir": {"type": "string"},
                "no_llm": {"type": "boolean", "default": True},
                "model": {"type": "string", "default": _DEFAULT_MODEL},
            },
        },
        execute=analyze_sequences,
    )
    register_tool(
        name="query_history",
        description="List recent analysis runs.",
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
        description="Return the stored detail (samples, trends, suggestions if cached) for a prior analysis_id.",
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
        description="Analyze mutation hotspots and trends. Optional analysis_id targets a specific run; defaults to the most recent.",
        parameters={
            "type": "object",
            "properties": {
                "analysis_id": {"type": "string"},
            },
        },
        execute=detect_mutation_trends,
    )
    register_tool(
        name="generate_lab_suggestions",
        description="Generate experiment improvement suggestions. Optional analysis_id targets a specific run; defaults to the most recent.",
        parameters={
            "type": "object",
            "properties": {
                "analysis_id": {"type": "string"},
            },
        },
        execute=build_lab_suggestions,
    )
    register_tool(
        name="list_datasets",
        description="List datasets available for analysis (built-in plus user-registered).",
        parameters={"type": "object", "properties": {}},
        execute=list_datasets,
    )
    register_tool(
        name="register_dataset",
        description="Register a user dataset by giving it a label and pointing to its ab1 and gb folders.",
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "ab1_dir": {"type": "string"},
                "gb_dir": {"type": "string"},
            },
            "required": ["label", "ab1_dir", "gb_dir"],
        },
        execute=register_dataset,
    )
    register_tool(
        name="delete_dataset",
        description="Remove a user-registered dataset by id.",
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        execute=delete_dataset,
    )

    # ── Primitive (atomic) tools ──────────────────────────────────────────
    # Lower-level operations the agent can compose to answer file/sequence
    # questions without first registering a dataset. See bio_primitives.py
    # for the wrappers.

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

    register_tool(
        name="compare_analyses",
        description=(
            "Compare two previously-run analyses side by side. Pulls samples "
            "from each by analysis_id and returns per-dataset stats "
            "(total_samples, wrong_rate, frameshift_rate, mean_identity, "
            "mean_coverage, top_mutation_positions) plus a delta block. "
            "Use when the user asks to compare datasets (e.g. 'base vs pro') "
            "— first call query_history if you need to look up analysis_ids."
        ),
        parameters={
            "type": "object",
            "properties": {
                "analysis_id_a": {"type": "string"},
                "analysis_id_b": {"type": "string"},
            },
            "required": ["analysis_id_a", "analysis_id_b"],
        },
        execute=compare_analyses,
    )

    register_tool(
        name="read_pdf",
        description=(
            "Extract text from a local PDF (a research paper, protocol, "
            "supplementary file…). Returns up to max_chars characters from "
            "up to max_pages pages, plus metadata (title, total_pages, "
            "truncated flag). Use when the user attaches a PDF or asks "
            "you to read one on disk."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000},
                "max_pages": {"type": "integer", "default": 30},
            },
            "required": ["path"],
        },
        execute=read_pdf,
    )

    _REGISTERED = True


__all__ = [
    "ToolExecutionError",
    "analyze_sequences",
    "query_history",
    "get_analysis_detail",
    "detect_mutation_trends",
    "build_lab_suggestions",
    "list_datasets",
    "register_dataset",
    "delete_dataset",
    "register_initial_tools",
]
