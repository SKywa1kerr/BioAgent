from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_analysis_app_exposes_status_board_and_recent_dataset_actions():
    app_content = _read("src/App.tsx")
    i18n_content = _read("src/i18n.ts")
    tab_layout_tsx = _read("src/components/TabLayout.tsx")
    tab_layout_css = _read("src/components/TabLayout.css")

    assert "analysis-status-strip" in app_content
    assert "recent-activity-strip" in app_content
    assert "handleReuseLastDataset" in app_content
    assert "dataset.reuseLastDataset" in app_content
    assert "dataset.lastDataset" in app_content
    assert 'className="tab-sidebar"' in tab_layout_tsx
    assert ".tab-sidebar {" in tab_layout_css
    assert "flex-direction: row;" in tab_layout_css
    assert "width: 184px;" in tab_layout_css
    assert "grid-template-columns: minmax(0, 1fr) auto;" in app_content or "grid-template-columns: minmax(0, 1fr) auto;" in _read("src/App.css")
    assert ".toolbar .toolbar-link-button {" in _read("src/App.css")
    assert 'statusReady: "' in i18n_content
    assert 'statusNeedsAttention: "' in i18n_content
