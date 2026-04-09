from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .agent_tools import build_tools_prompt, filter_tool_specs, get_tool_specs
from .llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, normalize_llm_base_url


DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "maxRounds": 3,
    "maxToolCallsPerTurn": 3,
    "maxRecentMessages": 12,
    "allowActionTools": True,
    "includeUsage": True,
}
DEFAULT_AGENT_MODEL = DEFAULT_MODEL


def build_agent_prompt(context: dict[str, Any]) -> str:
    runtime = {**DEFAULT_RUNTIME_CONFIG, **(context.get("runtime") or {})}
    tool_specs = filter_tool_specs(get_tool_specs(), allow_action_tools=bool(runtime["allowActionTools"]))
    current_analysis = context.get("currentAnalysis") or {}
    recent_tool_results = context.get("recentToolResults") or []
    history = context.get("history") or []

    samples = current_analysis.get("samples") or []
    if not isinstance(samples, list):
        samples = []

    selected_sample_id = current_analysis.get("selectedSampleId")
    source_path = current_analysis.get("sourcePath")
    status_counts: dict[str, int] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        status = str(sample.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    sample_preview: list[str] = []
    for sample in samples[:5]:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("id") or "?")
        clone = str(sample.get("clone") or "?")
        status = str(sample.get("status") or "unknown")
        reason = str(sample.get("reason") or "")
        error = str(sample.get("error") or "")
        mutation_count_value = sample.get("mutationCount")
        if isinstance(mutation_count_value, bool):
            mutation_count = int(mutation_count_value)
        elif isinstance(mutation_count_value, int):
            mutation_count = mutation_count_value
        else:
            mutations = sample.get("mutations")
            mutation_count = len(mutations) if isinstance(mutations, list) else 0
        preview = f"- {sample_id} clone={clone} status={status} mutations={mutation_count}"
        if reason:
            preview += f" reason={reason}"
        elif error:
            preview += f" error={error}"
        sample_preview.append(preview)

    tool_result_preview: list[str] = []
    for result in recent_tool_results[-3:]:
        if not isinstance(result, dict):
            continue
        tool_name = str(result.get("tool") or "unknown")
        ok = bool(result.get("ok"))
        summary = str(result.get("summary") or "")
        tool_result_preview.append(f"- {tool_name} ok={str(ok).lower()} summary={summary}")

    history_preview: list[str] = []
    max_history = int(runtime["maxRecentMessages"])
    for message in history[-max_history:]:
        if not isinstance(message, dict):
            continue
        message_type = str(message.get("type") or "unknown")
        content = " ".join(str(message.get("content") or "").split())
        if content:
            history_preview.append(f"- {message_type}: {content[:240]}")

    lines = [
        "You are BioAgent, a controlled desktop analysis assistant.",
        "Decide whether to answer directly or request frontend tool calls.",
        "Return JSON only. No markdown, prose outside JSON, or code fences.",
        'Reply shape: {"action":"reply","content":"..."}',
        'Tool shape: {"action":"tool_calls","message":"...","calls":[{"tool":"...","args":{...}}]}',
        "Do not invent tools or arguments outside the published tool schema.",
        f"Runtime limits: maxRounds={runtime['maxRounds']}, maxToolCallsPerTurn={runtime['maxToolCallsPerTurn']}, allowActionTools={str(bool(runtime['allowActionTools'])).lower()}",
        f"Current analysis: samples={len(samples)}, selected sample: {selected_sample_id or 'none'}, source path: {source_path or 'none'}",
        f"Status counts: {json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}",
        "Current sample preview:" if sample_preview else "Current sample preview: none",
        *sample_preview,
        "Recent tool results:" if tool_result_preview else "Recent tool results: none",
        *tool_result_preview,
        "Conversation history:" if history_preview else "Conversation history: none",
        *history_preview,
        build_tools_prompt(tool_specs),
    ]
    return "\n\n".join(lines)


def _runtime_max_tool_calls(runtime: dict[str, Any] | None) -> int:
    if not runtime:
        return int(DEFAULT_RUNTIME_CONFIG["maxToolCallsPerTurn"])
    value = runtime.get("maxToolCallsPerTurn", DEFAULT_RUNTIME_CONFIG["maxToolCallsPerTurn"])
    if isinstance(value, bool):
        raise ValueError("invalid_model_output")
    try:
        max_tool_calls = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_model_output") from exc
    if max_tool_calls < 0:
        raise ValueError("invalid_model_output")
    return max_tool_calls


def parse_agent_response(raw: str, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_model_output") from exc

    if not isinstance(data, dict):
        raise ValueError("invalid_model_output")

    action = data.get("action")
    if action == "reply":
        return {
            "action": "reply",
            "content": str(data.get("content", "")),
            "usage": data.get("usage"),
            "stopReason": data.get("stopReason", "final_reply"),
        }

    if action == "tool_calls":
        calls = data.get("calls")
        if not isinstance(calls, list):
            raise ValueError("invalid_model_output")
        max_tool_calls = _runtime_max_tool_calls(runtime)
        normalized_calls: list[dict[str, Any]] = []
        for call in calls[:max_tool_calls]:
            if not isinstance(call, dict):
                raise ValueError("invalid_model_output")
            tool_name = call.get("tool")
            args = call.get("args")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("invalid_model_output")
            if not isinstance(args, dict):
                raise ValueError("invalid_model_output")
            normalized_calls.append({"tool": tool_name, "args": args})
        return {
            "action": "tool_calls",
            "message": str(data.get("message", "")),
            "calls": normalized_calls,
            "usage": data.get("usage"),
        }

    raise ValueError("invalid_model_output")


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("input") or usage.get("prompt_tokens")
        completion_tokens = usage.get("output") or usage.get("completion_tokens")
        total_tokens = usage.get("total") or usage.get("total_tokens")

    if prompt_tokens is None or completion_tokens is None:
        return None

    normalized = {
        "input": int(prompt_tokens),
        "output": int(completion_tokens),
    }
    if total_tokens is not None:
        normalized["total"] = int(total_tokens)
    return normalized


def _call_agent_model(payload: dict[str, Any], messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None, str | None]:
    api_key = os.environ.get("LLM_API_KEY")
    base_url = normalize_llm_base_url(os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))

    if not api_key or api_key == "your-api-key-here":
        raise RuntimeError(
            "LLM_API_KEY not configured.\n"
            "Please set your API key in the Settings tab."
        )

    model = str(payload.get("model") or os.environ.get("LLM_MODEL") or DEFAULT_AGENT_MODEL)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=800,
        messages=messages,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty content")
    return content, _normalize_usage(getattr(response, "usage", None)), response.choices[0].finish_reason


def _include_usage(runtime: dict[str, Any] | None) -> bool:
    if not runtime:
        return bool(DEFAULT_RUNTIME_CONFIG["includeUsage"])
    return bool(runtime.get("includeUsage", DEFAULT_RUNTIME_CONFIG["includeUsage"]))


def _safe_reply(message: str, stop_reason: str) -> dict[str, Any]:
    return {
        "action": "reply",
        "content": message,
        "stopReason": stop_reason,
    }


def _failure_message(exc: Exception) -> str:
    message = str(exc).strip()
    if "LLM_API_KEY" in message and "not configured" in message:
        return "Agent chat is unavailable because the LLM API key is not configured."
    if not message:
        return "Agent chat is unavailable right now."
    return f"Agent chat is unavailable right now: {message.splitlines()[0]}"


def run_agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    context = payload.get("context") or {}
    runtime = context.get("runtime") if isinstance(context, dict) else None

    if "mockResponse" in payload:
        return parse_agent_response(str(payload["mockResponse"]), runtime)

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return _safe_reply("Agent chat needs a user message before it can continue.", "aborted")

    messages = [
        {"role": "system", "content": build_agent_prompt(context if isinstance(context, dict) else {})},
        {"role": "user", "content": message.strip()},
    ]

    try:
        raw, usage, finish_reason = _call_agent_model(payload, messages)
        parsed = parse_agent_response(raw, runtime)
    except ValueError as exc:
        if str(exc) != "invalid_model_output":
            raise
        return _safe_reply(
            "Agent chat returned an invalid response. Please try again or narrow the request.",
            "invalid_model_output",
        )
    except Exception as exc:
        return _safe_reply(_failure_message(exc), "aborted")

    if _include_usage(runtime) and usage is not None:
        parsed["usage"] = usage
    if parsed["action"] == "reply" and finish_reason == "length":
        parsed["stopReason"] = "aborted"
    return parsed
