"""Regression tests: core.llm_client.call_llm must not write to stdout.

The MCP server uses stdout for JSON-RPC framing; any print to stdout from the
LLM client corrupts the protocol when analyze_sequences runs with no_llm=False.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import core.llm_client as llm_client


def _fake_message(content: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice])


def _install_fake_openai(monkeypatch, responses):
    queue = list(responses)

    class FakeCompletions:
        def create(self, **_kwargs):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm_client, "OpenAI", FakeClient)


def test_call_llm_success_does_not_write_stdout(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, [_fake_message("C1-1 gene is ok")])

    buf = StringIO()
    with redirect_stdout(buf):
        result = llm_client.call_llm("evidence", model="test-model")

    assert result == "C1-1 gene is ok"
    assert buf.getvalue() == ""


def test_call_llm_rate_limit_retry_does_not_write_stdout(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr("core.llm_client.time.sleep", lambda _s: None)
    _install_fake_openai(
        monkeypatch,
        [
            Exception("429 rate limit exceeded"),
            _fake_message("C1-1 gene is ok"),
        ],
    )

    buf = StringIO()
    with redirect_stdout(buf):
        result = llm_client.call_llm("evidence", model="test-model")

    assert result == "C1-1 gene is ok"
    assert buf.getvalue() == ""


def test_call_llm_system_prompt_fallback_does_not_write_stdout(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    _install_fake_openai(
        monkeypatch,
        [
            Exception("Developer instruction (system role) is not supported"),
            _fake_message("C1-1 gene is ok"),
        ],
    )

    buf = StringIO()
    with redirect_stdout(buf):
        result = llm_client.call_llm("evidence", model="test-model")

    assert result == "C1-1 gene is ok"
    assert buf.getvalue() == ""
