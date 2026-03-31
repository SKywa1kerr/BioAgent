import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.db_models import save_analysis, list_analyses, get_analysis_samples

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    import bioagent.db as db_mod
    db_mod._engine = None
    db_mod._SessionFactory = None
    from bioagent.db import init_db
    init_db(db_url)
    yield db_url
    db_mod._engine = None
    db_mod._SessionFactory = None

def test_save_and_list(tmp_db):
    samples = [
        {"sample_id": "S1", "clone": "C1", "identity": 1.0, "coverage": 0.95,
         "frameshift": False, "aa_changes": [], "aa_changes_n": 0,
         "seq_length": 800, "ref_length": 1000, "sub": 0, "ins": 0, "dele": 0},
    ]
    judgments = [{"sid": "S1", "status": "ok", "reason": "", "rule": 10}]
    aid = save_analysis(samples, judgments, {}, source_path="/test")
    assert aid

    analyses = list_analyses()
    assert len(analyses) == 1
    assert analyses[0]["total"] == 1
    assert analyses[0]["ok_count"] == 1

    samps = get_analysis_samples(aid)
    assert len(samps) == 1
    assert samps[0]["sid"] == "S1"
    assert samps[0]["status"] == "ok"

def test_empty_save(tmp_db):
    aid = save_analysis([], [], {})
    assert aid
    analyses = list_analyses()
    assert len(analyses) == 1
    assert analyses[0]["total"] == 0
