from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src-python"))

import bioagent.mcp_tools as mcp_tools
import bioagent.tools_register as tools_register


@pytest.fixture(autouse=True)
def reset_in_memory_state():
    mcp_tools._TOOL_REGISTRY.clear()
    tools_register._REGISTERED = False
    tools_register._ANALYSIS_HISTORY.clear()
    tools_register._ANALYSIS_DETAILS.clear()
    yield
