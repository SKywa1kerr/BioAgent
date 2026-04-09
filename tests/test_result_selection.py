from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_result_selection_defaults_to_collapsed_state():
    content = _read("src/utils/resultSelection.ts")
    assert 'export function getDefaultSelectedSampleId()' in content
    assert 'return null;' in content
