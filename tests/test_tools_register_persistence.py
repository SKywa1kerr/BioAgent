import json
from pathlib import Path

import pytest

import bioagent.mcp_tools as mcp_tools
import bioagent.tools_register as tools_register


def _seed_two_analyses():
    tools_register._ANALYSIS_DETAILS["a1"] = {
        "analysis_id": "a1",
        "samples": [
            {
                "id": "S1", "clone": "C1", "status": "wrong",
                "identity": 0.4, "coverage": 0.4, "avg_quality": 12,
                "frameshift": True,
                "mutations": [
                    {"position": 10, "refBase": "A", "queryBase": "G", "type": "substitution", "effect": "missense"},
                    {"position": 10, "refBase": "A", "queryBase": "G", "type": "substitution", "effect": "missense"},
                ],
            }
        ],
    }
    tools_register._ANALYSIS_DETAILS["a2"] = {
        "analysis_id": "a2",
        "samples": [
            {
                "id": "S2", "clone": "C2", "status": "ok",
                "identity": 0.99, "coverage": 0.95, "avg_quality": 40,
                "frameshift": False,
                "mutations": [],
            }
        ],
    }
    tools_register._ANALYSIS_HISTORY.append({"analysis_id": "a2"})
    tools_register._ANALYSIS_HISTORY.append({"analysis_id": "a1"})


def test_detect_mutation_trends_targets_specific_analysis_id():
    _seed_two_analyses()
    tools_register.register_initial_tools()

    latest = mcp_tools.call_tool("detect_mutation_trends", {})
    targeted = mcp_tools.call_tool("detect_mutation_trends", {"analysis_id": "a1"})

    assert latest["ok"] is True
    assert latest["data"]["total_samples"] == 1
    assert latest["data"]["total_mutations"] == 0  # a2 is the head of history (latest)

    assert targeted["ok"] is True
    assert targeted["data"]["total_mutations"] == 2  # a1 has 2 mutations


def test_generate_lab_suggestions_targets_specific_analysis_id():
    _seed_two_analyses()
    tools_register.register_initial_tools()

    latest = mcp_tools.call_tool("generate_lab_suggestions", {})
    targeted = mcp_tools.call_tool("generate_lab_suggestions", {"analysis_id": "a1"})

    assert latest["ok"] is True
    assert latest["data"]["overall_health"] == "good"
    assert targeted["ok"] is True
    assert targeted["data"]["overall_health"] in {"critical", "needs_attention"}


def test_trend_result_attaches_to_analysis_detail():
    _seed_two_analyses()
    tools_register.register_initial_tools()

    mcp_tools.call_tool("detect_mutation_trends", {"analysis_id": "a1"})
    mcp_tools.call_tool("generate_lab_suggestions", {"analysis_id": "a1"})

    detail = tools_register._ANALYSIS_DETAILS["a1"]
    assert "trends" in detail
    assert "suggestions" in detail
    assert detail["trends"]["total_mutations"] == 2


def test_register_and_list_user_datasets(tmp_path):
    ab1_dir = tmp_path / "ab1"
    gb_dir = tmp_path / "gb"
    ab1_dir.mkdir()
    gb_dir.mkdir()

    tools_register.register_initial_tools()

    reg = mcp_tools.call_tool(
        "register_dataset",
        {"label": "My Clones", "ab1_dir": str(ab1_dir), "gb_dir": str(gb_dir)},
    )
    assert reg["ok"] is True
    assert reg["data"]["id"] == "my_clones"

    listing = mcp_tools.call_tool("list_datasets", {})
    assert listing["ok"] is True
    user_ids = [d["id"] for d in listing["data"]["user"]]
    assert "my_clones" in user_ids
    builtin_ids = [d["id"] for d in listing["data"]["builtin"]]
    assert "base" in builtin_ids
    assert "pro" in builtin_ids


def test_register_dataset_rejects_missing_directory(tmp_path):
    tools_register.register_initial_tools()
    nonexistent = tmp_path / "does-not-exist"
    with pytest.raises(tools_register.ToolExecutionError, match="ab1_dir"):
        mcp_tools.call_tool(
            "register_dataset",
            {"label": "Bad", "ab1_dir": str(nonexistent), "gb_dir": str(tmp_path)},
        )


def test_delete_dataset(tmp_path):
    ab1_dir = tmp_path / "ab1"
    gb_dir = tmp_path / "gb"
    ab1_dir.mkdir()
    gb_dir.mkdir()
    tools_register.register_initial_tools()
    mcp_tools.call_tool(
        "register_dataset",
        {"label": "Temp", "ab1_dir": str(ab1_dir), "gb_dir": str(gb_dir)},
    )
    result = mcp_tools.call_tool("delete_dataset", {"id": "temp"})
    assert result["ok"] is True

    listing = mcp_tools.call_tool("list_datasets", {})
    assert "temp" not in [d["id"] for d in listing["data"]["user"]]


def test_persistence_round_trip(tmp_path, monkeypatch):
    persist_root = tmp_path / "persist"
    monkeypatch.setenv("BIOAGENT_DATA_DIR", str(persist_root))

    ab1_dir = tmp_path / "ab1"
    gb_dir = tmp_path / "gb"
    ab1_dir.mkdir()
    gb_dir.mkdir()

    tools_register.register_initial_tools()
    mcp_tools.call_tool(
        "register_dataset",
        {"label": "Persisted", "ab1_dir": str(ab1_dir), "gb_dir": str(gb_dir)},
    )

    # Simulate seeding an analysis manually + persisting
    tools_register._store_analysis({
        "analysis_id": "persisted-1",
        "dataset": "persisted",
        "sample_count": 0,
        "created_at": "2026-05-09T00:00:00+00:00",
        "output_dir": str(tmp_path / "out"),
        "evidence_path": str(tmp_path / "out" / "evidence.txt"),
        "used_llm": False,
        "model": None,
        "samples": [],
        "result_lines": None,
        "llm_raw": None,
    })

    assert (persist_root / "history.json").exists()
    assert (persist_root / "details" / "persisted-1.json").exists()
    assert (persist_root / "datasets.json").exists()

    # Wipe in-memory and re-bootstrap to simulate restart
    tools_register._ANALYSIS_HISTORY.clear()
    tools_register._ANALYSIS_DETAILS.clear()
    tools_register._USER_DATASETS.clear()
    tools_register._BOOTSTRAPPED = False
    tools_register._bootstrap_from_disk()

    assert any(h["analysis_id"] == "persisted-1" for h in tools_register._ANALYSIS_HISTORY)
    assert "persisted-1" in tools_register._ANALYSIS_DETAILS
    assert any(d["id"] == "persisted" for d in tools_register._USER_DATASETS)
