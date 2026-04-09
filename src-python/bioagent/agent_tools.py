from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    category: str


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="query_samples",
        description="Read the current sample list or a filtered subset.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "sampleId": {"type": "string"},
            },
        },
        category="query",
    ),
    ToolSpec(
        name="query_history",
        description="Read analysis history summaries.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "number"},
            },
        },
        category="query",
    ),
    ToolSpec(
        name="get_sample_detail",
        description="Read detailed data for a single sample.",
        parameters={
            "type": "object",
            "properties": {
                "sampleId": {"type": "string"},
            },
            "required": ["sampleId"],
        },
        category="query",
    ),
    ToolSpec(
        name="run_analysis",
        description="Run analysis using the current desktop workflow.",
        parameters={
            "type": "object",
            "properties": {
                "ab1Dir": {"type": "string"},
                "genesDir": {"type": "string"},
                "plasmid": {"type": "string"},
                "useLLM": {"type": "boolean"},
            },
            "required": ["ab1Dir"],
        },
        category="action",
    ),
    ToolSpec(
        name="export_report",
        description="Export a report for the current sample set.",
        parameters={
            "type": "object",
            "properties": {},
        },
        category="action",
    ),
)


def get_tool_specs() -> list[dict[str, Any]]:
    return [asdict(tool) for tool in TOOL_SPECS]


def filter_tool_specs(tool_specs: list[dict[str, Any]], allow_action_tools: bool) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for tool in tool_specs:
        if not isinstance(tool, dict):
            continue
        category = str(tool.get("category") or "query")
        if not allow_action_tools and category == "action":
            continue
        filtered.append(tool)
    return filtered


def build_tools_prompt(tool_specs: list[dict[str, Any]]) -> str:
    lines = ["Available tools:"]
    for tool in tool_specs:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "<unknown>")
        category = str(tool.get("category") or "query")
        description = str(tool.get("description") or "")
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        rendered_parameters = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
        lines.append(f"- {name} [{category}]: {description} params={rendered_parameters}")
    return "\n".join(lines)
