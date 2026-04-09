import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.agent_tools import build_tools_prompt, filter_tool_specs, get_tool_specs


def _tools_by_name():
    return {tool["name"]: tool for tool in get_tool_specs()}


def test_get_tool_specs_exposes_expected_tools():
    specs = get_tool_specs()
    names = [tool["name"] for tool in specs]
    assert names == [
        "query_samples",
        "query_history",
        "get_sample_detail",
        "run_analysis",
        "export_report",
    ]
    categories = {tool["name"]: tool["category"] for tool in specs}
    assert categories == {
        "query_samples": "query",
        "query_history": "query",
        "get_sample_detail": "query",
        "run_analysis": "action",
        "export_report": "action",
    }
    assert _tools_by_name()["query_samples"]["parameters"]["properties"]["status"]["type"] == "string"
    assert "analysisId" not in _tools_by_name()["query_samples"]["parameters"]["properties"]
    assert _tools_by_name()["get_sample_detail"]["parameters"]["required"] == ["sampleId"]


def test_filter_tool_specs_blocks_action_tools():
    filtered = filter_tool_specs(get_tool_specs(), allow_action_tools=False)
    names = [tool["name"] for tool in filtered]
    assert names == [
        "query_samples",
        "query_history",
        "get_sample_detail",
    ]


def test_filter_tool_specs_allows_action_tools():
    filtered = filter_tool_specs(get_tool_specs(), allow_action_tools=True)
    assert [tool["name"] for tool in filtered] == [
        "query_samples",
        "query_history",
        "get_sample_detail",
        "run_analysis",
        "export_report",
    ]


def test_build_tools_prompt_renders_tools_and_handles_malformed_entries():
    prompt = build_tools_prompt(
        filter_tool_specs(get_tool_specs(), allow_action_tools=False)
        + [{"name": "broken", "description": None, "parameters": "oops"}]
    )
    assert prompt.startswith("Available tools:")
    assert "- query_samples [query]:" in prompt
    assert 'params={"properties": {"sampleId": {"type": "string"}, "status": {"type": "string"}}, "type": "object"}' in prompt
    assert "- get_sample_detail [query]: Read detailed data for a single sample." in prompt
    assert "- broken [query]:  params={}" in prompt
