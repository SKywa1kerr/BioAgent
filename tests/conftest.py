from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src-python"))

import bioagent.mcp_tools as mcp_tools
import bioagent.tools_register as tools_register


@pytest.fixture(autouse=True)
def reset_in_memory_state(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOAGENT_DATA_DIR", str(tmp_path / "bioagent"))
    mcp_tools._TOOL_REGISTRY.clear()
    tools_register._REGISTERED = False
    tools_register._BOOTSTRAPPED = False
    tools_register._ANALYSIS_HISTORY.clear()
    tools_register._ANALYSIS_DETAILS.clear()
    tools_register._USER_DATASETS.clear()
    yield
