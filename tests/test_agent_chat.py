import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent import main as main_module
from bioagent.agent_chat import (
    DEFAULT_RUNTIME_CONFIG,
    build_agent_prompt,
    parse_agent_response,
    run_agent_turn,
)


def test_parse_reply_response_defaults_stop_reason():
    result = parse_agent_response('{"action":"reply","content":"done"}')
    assert result["action"] == "reply"
    assert result["content"] == "done"
    assert result["stopReason"] == "final_reply"


def test_parse_tool_calls_response():
    raw = '{"action":"tool_calls","message":"Checking sample detail","calls":[{"tool":"get_sample_detail","args":{"sampleId":"S1"}}]}'
    result = parse_agent_response(raw)
    assert result["action"] == "tool_calls"
    assert result["message"] == "Checking sample detail"
    assert result["calls"][0]["tool"] == "get_sample_detail"
    assert result["calls"][0]["args"]["sampleId"] == "S1"


def test_parse_tool_calls_response_respects_runtime_override():
    raw = (
        '{"action":"tool_calls","message":"Checking sample detail","calls":['
        '{"tool":"query_samples","args":{}},'
        '{"tool":"query_history","args":{}},'
        '{"tool":"get_sample_detail","args":{"sampleId":"S1"}}'
        "]}"
    )
    result = parse_agent_response(raw, runtime={"maxToolCallsPerTurn": 2})
    assert len(result["calls"]) == 2
    assert [call["tool"] for call in result["calls"]] == ["query_samples", "query_history"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"action":"tool_calls","message":"oops","calls":[42]}',
        '{"action":"tool_calls","message":"oops","calls":[{"args":{}}]}',
        '{"action":"tool_calls","message":"oops","calls":[{"tool":"get_sample_detail","args":"bad"}]}',
    ],
)
def test_parse_tool_calls_response_rejects_malformed_calls(raw):
    with pytest.raises(ValueError):
        parse_agent_response(raw)


def test_parse_agent_response_rejects_invalid_actions():
    with pytest.raises(ValueError):
        parse_agent_response('{"action":"unknown"}')


def test_build_agent_prompt_filters_action_tools_when_disabled():
    prompt = build_agent_prompt({"runtime": {"allowActionTools": False}})
    assert prompt.startswith("You are BioAgent, a controlled desktop analysis assistant.")
    assert "Available tools:" in prompt
    assert "query_samples [query]" in prompt
    assert "run_analysis [action]" not in prompt
    assert "export_report [action]" not in prompt


def test_build_agent_prompt_includes_action_tools_by_default():
    prompt = build_agent_prompt({})
    assert "run_analysis [action]" in prompt
    assert "export_report [action]" in prompt


def test_build_agent_prompt_supports_slim_sample_summaries():
    prompt = build_agent_prompt(
        {
            "currentAnalysis": {
                "samples": [
                    {
                        "id": "S2",
                        "clone": "C2",
                        "status": "wrong",
                        "reason": "frameshift",
                        "mutationCount": 2,
                    }
                ]
            }
        }
    )

    assert "- S2 clone=C2 status=wrong mutations=2 reason=frameshift" in prompt


def test_run_agent_turn_calls_model_and_returns_tool_calls(monkeypatch):
    captured = {}

    def fake_call_agent_model(payload, messages):
        captured["payload"] = payload
        captured["messages"] = messages
        return (
            '{"action":"tool_calls","message":"Checking current samples","calls":[{"tool":"query_samples","args":{"status":"wrong"}}]}',
            {"input": 12, "output": 8, "total": 20},
            "tool_calls",
        )

    monkeypatch.setattr("bioagent.agent_chat._call_agent_model", fake_call_agent_model)

    payload = {
        "message": "Which samples are wrong?",
        "context": {
            "currentAnalysis": {
                "sourcePath": "D:/runs/demo",
                "selectedSampleId": "S2",
                "samples": [
                    {"id": "S1", "clone": "C1", "status": "ok", "reason": "", "mutations": []},
                    {"id": "S2", "clone": "C2", "status": "wrong", "reason": "frameshift", "mutations": [1, 2]},
                ],
            },
            "recentToolResults": [{"tool": "query_history", "ok": True, "summary": "Loaded 2 runs."}],
            "history": [
                {"type": "user", "content": "Show me current issues."},
                {"type": "agent", "content": "Checking."},
            ],
            "runtime": {"allowActionTools": False, "maxToolCallsPerTurn": 2},
        },
    }

    result = run_agent_turn(payload)

    assert result["action"] == "tool_calls"
    assert result["message"] == "Checking current samples"
    assert result["calls"] == [{"tool": "query_samples", "args": {"status": "wrong"}}]
    assert result["usage"] == {"input": 12, "output": 8, "total": 20}
    assert captured["payload"] == payload
    assert captured["messages"][0]["role"] == "system"
    assert "selected sample: S2" in captured["messages"][0]["content"]
    assert "run_analysis [action]" not in captured["messages"][0]["content"]
    assert captured["messages"][1] == {"role": "user", "content": "Which samples are wrong?"}


def test_run_agent_turn_returns_safe_reply_when_model_output_is_invalid(monkeypatch):
    monkeypatch.setattr(
        "bioagent.agent_chat._call_agent_model",
        lambda payload, messages: ("not-json", {"input": 3, "output": 1}, None),
    )

    result = run_agent_turn({"message": "hello", "context": {}})

    assert result["action"] == "reply"
    assert "invalid response" in result["content"].lower()
    assert result["stopReason"] == "invalid_model_output"


def test_run_agent_turn_returns_safe_reply_when_model_call_fails(monkeypatch):
    def fake_call_agent_model(payload, messages):
        raise RuntimeError("LLM_API_KEY not configured.\nPlease set your API key in the Settings tab.")

    monkeypatch.setattr("bioagent.agent_chat._call_agent_model", fake_call_agent_model)

    result = run_agent_turn({"message": "hello", "context": {}})

    assert result["action"] == "reply"
    assert "not configured" in result["content"]
    assert result["stopReason"] == "aborted"


def test_runtime_defaults_match_spec():
    assert DEFAULT_RUNTIME_CONFIG == {
        "maxRounds": 3,
        "maxToolCallsPerTurn": 3,
        "maxRecentMessages": 12,
        "allowActionTools": True,
        "includeUsage": True,
    }


def test_agent_chat_cli_returns_reply_json():
    payload = {
        "mockResponse": '{"action":"reply","content":"hello from agent chat","stopReason":"final_reply"}'
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bioagent.main",
            "--agent-chat",
            json.dumps(payload),
        ],
        cwd=Path(__file__).parent.parent / "src-python",
        capture_output=True,
        text=True,
        check=True,
    )

    output = json.loads(result.stdout.strip())
    assert output["action"] == "reply"
    assert output["content"] == "hello from agent chat"


def test_agent_chat_cli_returns_fallback_reply_without_mock_response():
    payload = {}
    env = dict(os.environ)
    env.pop("LLM_API_KEY", None)
    env.pop("LLM_BASE_URL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bioagent.main",
            "--agent-chat",
            json.dumps(payload),
        ],
        cwd=Path(__file__).parent.parent / "src-python",
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    output = json.loads(result.stdout.strip())
    assert output["action"] == "reply"
    assert "needs a user message" in output["content"].lower()
    assert output["stopReason"] == "aborted"


def test_agent_chat_cli_preempts_history_flow():
    payload = {}
    env = dict(os.environ)
    env.pop("LLM_API_KEY", None)
    env.pop("LLM_BASE_URL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bioagent.main",
            "--agent-chat",
            json.dumps(payload),
            "--history",
        ],
        cwd=Path(__file__).parent.parent / "src-python",
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    output = json.loads(result.stdout.strip())
    assert output["action"] == "reply"
    assert "needs a user message" in output["content"].lower()
    assert output["stopReason"] == "aborted"


def test_agent_chat_cli_uses_run_agent_turn(monkeypatch, capsys):
    monkeypatch.setattr(
        main_module,
        "run_agent_turn",
        lambda payload: {"action": "reply", "content": f"echo: {payload['message']}", "stopReason": "final_reply"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["bioagent.main", "--agent-chat", json.dumps({"message": "ping", "context": {}})],
    )

    main_module.main()

    output = json.loads(capsys.readouterr().out.strip())
    assert output == {
        "action": "reply",
        "content": "echo: ping",
        "stopReason": "final_reply",
    }
