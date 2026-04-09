import pytest
import sys
from pathlib import Path

# Ensure src-python is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.rules import judge_sample, load_thresholds

@pytest.fixture
def thresholds():
    return load_thresholds()

def test_perfect_sample(thresholds):
    s = {"sample_id": "X1", "identity": 1.0, "coverage": 0.95,
         "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 800}
    result = judge_sample(s, thresholds)
    assert result["status"] == "ok"
    assert result["rule"] == 10

def test_frameshift_is_wrong(thresholds):
    s = {"sample_id": "X2", "identity": 0.99, "coverage": 0.90,
         "frameshift": True, "aa_changes": [], "aa_changes_n": 0, "seq_length": 800}
    result = judge_sample(s, thresholds)
    assert result["status"] == "wrong"
    assert result["rule"] == 5

def test_aa_mutations_wrong(thresholds):
    s = {"sample_id": "X3", "identity": 0.98, "coverage": 0.90,
         "frameshift": False, "aa_changes": ["S334L"], "aa_changes_n": 1, "seq_length": 800}
    result = judge_sample(s, thresholds)
    assert result["status"] == "wrong"
    assert result["rule"] == 6

def test_seq_failure(thresholds):
    s = {"sample_id": "X4", "identity": 0.10, "coverage": 0.0,
         "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 30}
    result = judge_sample(s, thresholds)
    assert result["status"] == "wrong"
    assert result["rule"] == 2

def test_low_coverage_ok(thresholds):
    s = {"sample_id": "X5", "identity": 0.99, "coverage": 0.40,
         "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 300}
    result = judge_sample(s, thresholds)
    assert result["status"] == "ok"
    assert result["rule"] == 9

def test_uncertain_fallthrough(thresholds):
    s = {"sample_id": "X6", "identity": 0.92, "coverage": 0.70,
         "frameshift": False, "aa_changes": ["A1B", "C2D", "E3F"], "aa_changes_n": 3, "seq_length": 800}
    result = judge_sample(s, thresholds)
    assert result["status"] == "uncertain"
    assert result["rule"] == -1


def test_short_high_identity_trace_can_be_synthetic_overlap(thresholds):
    s = {"sample_id": "X7", "identity": 1.0, "coverage": 0.02,
         "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 25}
    result = judge_sample(s, thresholds)
    assert result["status"] == "ok"
    assert result["reason"] == "生工重叠峰"
    assert result["rule"] == 11


def test_short_high_identity_trace_can_be_overlap_failure(thresholds):
    s = {"sample_id": "X8", "identity": 1.0, "coverage": 0.03,
         "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 39}
    result = judge_sample(s, thresholds)
    assert result["status"] == "wrong"
    assert result["reason"] == "重叠峰"
    assert result["rule"] == 12


def test_very_short_low_coverage_fragment_can_be_fragment_loss(thresholds):
    s = {"sample_id": "X9", "identity": 1.0, "coverage": 0.05,
         "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 65}
    result = judge_sample(s, thresholds)
    assert result["status"] == "wrong"
    assert result["reason"] == "片段缺失"
    assert result["rule"] == 13
