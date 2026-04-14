import bioagent.mcp_tools as mcp_tools
import bioagent.tools_register as tools_register


def test_register_initial_tools_includes_analysis_trends_and_suggestions():
    tools_register.register_initial_tools()

    names = [item["name"] for item in mcp_tools.list_tools()]

    assert "analyze_sequences" in names
    assert "query_history" in names
    assert "get_analysis_detail" in names
    assert "detect_mutation_trends" in names
    assert "generate_lab_suggestions" in names


def test_trend_and_suggestion_tools_use_latest_analysis_detail():
    tools_register._ANALYSIS_HISTORY.append({"analysis_id": "a1"})
    tools_register._ANALYSIS_DETAILS["a1"] = {
        "analysis_id": "a1",
        "samples": [
            {
                "id": "S1",
                "clone": "C1",
                "status": "wrong",
                "identity": 0.4,
                "coverage": 0.4,
                "avg_quality": 12,
                "frameshift": True,
                "mutations": [
                    {"position": 10, "refBase": "A", "queryBase": "G", "type": "substitution", "effect": "missense"},
                    {"position": 10, "refBase": "A", "queryBase": "G", "type": "substitution", "effect": "missense"},
                ],
            }
        ],
    }

    tools_register.register_initial_tools()

    trend_result = mcp_tools.call_tool("detect_mutation_trends", {})
    suggestion_result = mcp_tools.call_tool("generate_lab_suggestions", {})

    assert trend_result["ok"] is True
    assert trend_result["data"]["total_samples"] == 1
    assert suggestion_result["ok"] is True
    assert suggestion_result["data"]["overall_health"] in {"critical", "needs_attention"}
