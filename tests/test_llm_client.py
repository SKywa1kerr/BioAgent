import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.llm_client import DEFAULT_REQUEST_TIMEOUT_SECONDS, call_llm, normalize_llm_base_url


def test_normalize_llm_base_url_strips_chat_completions_suffix():
    assert (
        normalize_llm_base_url("https://models.sjtu.edu.cn/api/v1/chat/completions")
        == "https://models.sjtu.edu.cn/api/v1"
    )


def test_normalize_llm_base_url_keeps_api_prefix():
    assert normalize_llm_base_url("https://models.sjtu.edu.cn/api/v1") == "https://models.sjtu.edu.cn/api/v1"


def test_call_llm_sets_request_timeout_and_returns_content(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "C397-a gene is wrong S334L"})()})]},
            )()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
    monkeypatch.setattr("bioagent.llm_client.OpenAI", FakeOpenAI)

    result = call_llm("evidence", model="demo-model")

    assert result == "C397-a gene is wrong S334L"
    assert captured["client"]["timeout"] == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert captured["client"]["max_retries"] == 0
    assert captured["create"]["timeout"] == DEFAULT_REQUEST_TIMEOUT_SECONDS
