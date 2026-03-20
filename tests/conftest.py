import pytest
from pathlib import Path

@pytest.fixture
def openclaw_data_dir():
    p = Path("D:/Learning/Biology/BioAgent_openclaw/data")
    if not p.exists():
        pytest.skip("openclaw data not available")
    return p

@pytest.fixture
def openclaw_truth_dir():
    p = Path("D:/Learning/Biology/BioAgent_openclaw/truth")
    if not p.exists():
        pytest.skip("openclaw truth not available")
    return p
