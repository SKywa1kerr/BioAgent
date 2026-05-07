from pathlib import Path
import shutil
import uuid

import pytest

import core.alignment as alignment
import bioagent.tools_register as tools_register


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_DIR = PROJECT_ROOT / "data"
BASE_GB_DIR = BASE_DATA_DIR / "base" / "gb"
BASE_AB1_DIR = BASE_DATA_DIR / "base" / "ab1"


def _mk_temp_dir() -> Path:
    root = Path(".tmp-tests")
    root.mkdir(exist_ok=True)
    d = root / f"case-{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _sample_summary(samples):
    """Reduce a list of sample dicts to a stable subset for shape comparison."""
    summary = []
    for s in samples:
        summary.append({
            "sid": s.get("sid"),
            "clone": s.get("clone"),
            "ab1": s.get("ab1"),
            "identity": s.get("identity"),
            "cds_coverage": s.get("cds_coverage"),
            "bucket": s.get("bucket"),
            "frameshift": s.get("frameshift"),
        })
    summary.sort(key=lambda d: (d["sid"] or "", d["ab1"] or ""))
    return summary


@pytest.fixture
def tmp_outdir():
    d = _mk_temp_dir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_analyze_dirs_matches_analyze_dataset_shape_for_base():
    """Calling analyze_dirs with the base dataset's explicit dirs must produce
    the same set of samples as analyze_dataset('base', data_dir)."""
    if not BASE_GB_DIR.exists() or not BASE_AB1_DIR.exists():
        pytest.skip("base dataset not available")

    via_dataset = alignment.analyze_dataset("base", BASE_DATA_DIR)
    via_dirs = alignment.analyze_dirs(BASE_GB_DIR, BASE_AB1_DIR)

    assert isinstance(via_dirs, list)
    assert len(via_dirs) == len(via_dataset)
    assert _sample_summary(via_dirs) == _sample_summary(via_dataset)


def test_analyze_sequences_with_explicit_dirs_uses_dropped_label(tmp_outdir):
    if not BASE_GB_DIR.exists() or not BASE_AB1_DIR.exists():
        pytest.skip("base dataset not available")

    result = tools_register.analyze_sequences(
        ab1_dir=str(BASE_AB1_DIR),
        gb_dir=str(BASE_GB_DIR),
        output_dir=str(tmp_outdir),
    )

    assert result["dataset"] == "dropped"
    assert result["sample_count"] >= 1
    assert result["used_llm"] is False
    detail = tools_register.get_analysis_detail(analysis_id=result["analysis_id"])
    assert detail["dataset"] == "dropped"


def test_analyze_sequences_passes_dropped_label_to_normalize_output_dir(monkeypatch):
    """The dropped run should pass dataset='dropped' to _normalize_output_dir
    so that, when output_dir is None, results land in outputs/dropped."""
    if not BASE_GB_DIR.exists() or not BASE_AB1_DIR.exists():
        pytest.skip("base dataset not available")

    captured = {}
    real_normalize = tools_register._normalize_output_dir

    def spy(output_dir, dataset):
        captured["dataset"] = dataset
        return real_normalize(output_dir, dataset)

    monkeypatch.setattr(tools_register, "_normalize_output_dir", spy)

    out = _mk_temp_dir()
    try:
        result = tools_register.analyze_sequences(
            ab1_dir=str(BASE_AB1_DIR),
            gb_dir=str(BASE_GB_DIR),
            output_dir=str(out),
        )
        assert captured["dataset"] == "dropped"
        assert result["dataset"] == "dropped"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_analyze_sequences_without_any_input_raises():
    with pytest.raises(tools_register.ToolExecutionError):
        tools_register.analyze_sequences()


def test_analyze_sequences_with_only_ab1_dir_raises():
    with pytest.raises(tools_register.ToolExecutionError):
        tools_register.analyze_sequences(ab1_dir=str(BASE_AB1_DIR))


def test_analyze_sequences_with_only_gb_dir_raises():
    with pytest.raises(tools_register.ToolExecutionError):
        tools_register.analyze_sequences(gb_dir=str(BASE_GB_DIR))


def test_analyze_sequences_rejects_dataset_plus_dirs():
    with pytest.raises(tools_register.ToolExecutionError):
        tools_register.analyze_sequences(
            dataset="base",
            ab1_dir=str(BASE_AB1_DIR),
            gb_dir=str(BASE_GB_DIR),
        )


def test_analyze_sequences_dataset_path_still_works(tmp_outdir):
    """Smoke: existing dataset='base' callers must keep working unchanged."""
    if not BASE_GB_DIR.exists() or not BASE_AB1_DIR.exists():
        pytest.skip("base dataset not available")

    result = tools_register.analyze_sequences(
        dataset="base",
        output_dir=str(tmp_outdir),
    )
    assert result["dataset"] == "base"
    assert result["sample_count"] >= 1
