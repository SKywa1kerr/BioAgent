from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_assistant_page_contract():
    app = _read("src/App.tsx")
    assistant_page = _read("src/components/AssistantPage.tsx")

    assert '{ id: "assistant", label: t(language, "tabs.assistant") }' in app
    assert 'activeTab === "assistant"' in app
    assert "<AssistantPage" in app
    assert "<AgentPanel" not in app

    assert "export function AssistantPage(" in assistant_page
    assert "<AgentPanel" in assistant_page
    assert 'className="assistant-page"' in assistant_page


def test_analysis_page_command_workbench_contract():
    app = _read("src/App.tsx")
    command_workbench = _read("src/components/CommandWorkbench.tsx")

    assert "<CommandWorkbench" in app
    assert "<ActionPlanCard" in app
    assert "<ExecutionTimeline" in app
    assert "window.electronAPI" in app
    assert "interpretCommand" in app
    assert "resultFilter" in app
    assert "needsConfirmation" in app
    assert "let hasStartedExecution = false;" in app
    assert "if (!hasStartedExecution)" in app
    assert "plasmid: nextPlasmid" in app
    assert 'throw new Error(t(language, "dataset.missingGb"))' in app
    assert "<AgentPanel" not in app
    assert 'event.key == "Enter" && !event.shiftKey' not in command_workbench
    assert 'event.key === "Enter" && !event.shiftKey' in command_workbench
    assert "nativeEvent.isComposing" in command_workbench
    assert "onKeyDown={handleKeyDown}" in command_workbench
