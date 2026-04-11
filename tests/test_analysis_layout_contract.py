from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_analysis_page_keeps_agent_panel_beside_results_on_desktop():
    app_content = _read("src/App.tsx")
    css_content = _read("src/App.css")

    assert '<div className="analysis-layout">' in app_content
    assert '<div className="analysis-workspace">' in app_content
    assert '<main className="main-content">' in app_content
    assert "<AgentPanel" in app_content

    layout_index = app_content.index('<div className="analysis-layout">')
    workspace_index = app_content.index('<div className="analysis-workspace">')
    main_index = app_content.index('<main className="main-content">')
    agent_index = app_content.index("<AgentPanel")
    assert layout_index < workspace_index < main_index < agent_index

    assert ".analysis-layout {" in css_content
    assert "flex-direction: column;" in css_content
    assert "height: 100%;" in css_content
    assert "grid-template-columns: minmax(0, 1fr) minmax(320px, var(--analysis-side-panel-width));" in css_content
    assert ".analysis-workspace > .agent-panel {" in css_content
    assert "position: sticky;" in css_content
    assert "top: 0;" in css_content
