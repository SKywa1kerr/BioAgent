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
