"""Regression test: lock decide_bucket calibration against truth/result*.txt.

Whenever the alignment thresholds or rules in core/alignment.py change, this
test re-runs the analyzer against all three shipped datasets and compares the
bucket against the manual truth labels. The intent is not to chase 100%
agreement (C367-1 has no AB1 trace at all), but to prevent silent regressions
on the calibrated set: 28/29 samples must keep matching truth.
"""
from pathlib import Path

import pytest

from core.alignment import analyze_dataset


_TRUTH = {
    "base": {
        "C379-a": "ok", "C397-a": "wrong", "C373-2": "ok", "C410-a": "ok",
        "C397-2": "wrong", "C406-2": "ok", "C376-2": "ok", "C402-2": "wrong",
        "C379-2": "wrong", "C405-2": "ok",
    },
    "pro": {
        "C403-1": "ok", "C366-3": "wrong", "C370-2": "ok", "C377-1": "ok",
        "C381-1": "ok", "C363-3": "ok", "C364-6": "ok",
    },
    "promax": {
        "C374-1": "ok", "C373-1": "ok", "C371-1": "ok", "C370-1": "wrong",
        "C369-1": "ok", "C368-1": "ok", "C367-1": "wrong", "C366-2": "wrong",
        "C364-2": "wrong", "C363-2": "wrong", "C358-1": "ok", "C351-2": "ok",
    },
}

# Truth entries with no AB1 trace shipped — analyze_dataset cannot label them.
_NO_TRACE = {("promax", "C367-1")}


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ["base", "pro", "promax"])
def test_bucket_matches_truth_for_every_traceable_sample(dataset: str, project_root: Path):
    samples = analyze_dataset(dataset, project_root / "data")
    by_sid = {s["sid"]: s for s in samples}

    mismatches = []
    for sid, expected in _TRUTH[dataset].items():
        if (dataset, sid) in _NO_TRACE:
            assert sid not in by_sid, f"{dataset}/{sid} has no AB1 — should not appear"
            continue
        assert sid in by_sid, f"{dataset}/{sid} expected in analysis output"
        actual = by_sid[sid].get("bucket")
        if actual != expected:
            mismatches.append((sid, expected, actual))

    assert not mismatches, (
        f"{dataset} bucket regressions vs truth: " +
        ", ".join(f"{sid} truth={t} bucket={a}" for sid, t, a in mismatches)
    )


def test_pro_C366_3_dual_read_other_issues_routes_to_wrong(project_root: Path):
    """Specifically pin the C366-3 fix: best read is clean, but the secondary
    [T7-TER] read flagged issues — the pair as a whole should land in wrong.
    """
    samples = analyze_dataset("pro", project_root / "data")
    by_sid = {s["sid"]: s for s in samples}
    sample = by_sid["C366-3"]
    assert sample.get("bucket") == "wrong"
    assert sample.get("dual_read") is True
    assert sample.get("other_read_issues"), "fix relies on this signal"


def test_promax_C363_2_severe_misalignment_with_frameshift_routes_to_wrong(project_root: Path):
    """C363-2 has identity≈0.5 + frameshift — sequencing succeeded but the
    result is clearly defective. Should be wrong, not untested.
    """
    samples = analyze_dataset("promax", project_root / "data")
    by_sid = {s["sid"]: s for s in samples}
    sample = by_sid["C363-2"]
    assert sample.get("bucket") == "wrong"
    assert sample.get("frameshift") is True


# Quality-tag mapping — borrowed from truth file vocabulary so the UI can
# show human-readable hints without strict-matching truth.

def _tags(ds: str, sid: str, project_root: Path) -> set:
    samples = analyze_dataset(ds, project_root / "data")
    by_sid = {s["sid"]: s for s in samples}
    return set(by_sid[sid].get("quality_tags") or [])


def test_quality_tag_partial_coverage_on_pro_samples(project_root: Path):
    # All pro samples have cds_coverage ≈ 0.5 → "未测通"-style tag.
    for sid in ("C363-3", "C364-6", "C370-2", "C377-1", "C381-1", "C403-1"):
        assert "partial_coverage" in _tags("pro", sid, project_root), sid


def test_quality_tag_frameshift_on_promax_C370_1(project_root: Path):
    # truth: 移码错误
    assert "frameshift" in _tags("promax", "C370-1", project_root)


def test_quality_tag_alignment_failed_on_promax_C363_2(project_root: Path):
    # truth: 比对失败
    tags = _tags("promax", "C363-2", project_root)
    assert "alignment_failed" in tags
    assert "frameshift" in tags


def test_quality_tag_large_indel_cluster_on_promax_C364_2(project_root: Path):
    # truth: 片段缺失 — surfaces via long contiguous run of aa changes.
    assert "large_indel_cluster" in _tags("promax", "C364-2", project_root)


def test_quality_tag_dual_read_conflict_on_pro_C366_3(project_root: Path):
    assert "dual_read_conflict" in _tags("pro", "C366-3", project_root)


def test_quality_tags_empty_for_clean_full_coverage_samples(project_root: Path):
    # base samples sit at identity=1, cov=1, q=50, no fs — no quality concerns.
    for sid in ("C373-2", "C376-2", "C379-a", "C405-2", "C406-2", "C410-a"):
        assert _tags("base", sid, project_root) == set(), sid
