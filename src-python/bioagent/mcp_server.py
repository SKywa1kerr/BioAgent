from __future__ import annotations

import json
import sys
from typing import TextIO

from bioagent.mcp_tools import call_tool, list_tools
from bioagent.tools_register import ToolExecutionError, register_initial_tools

SERVER_INFO = {"name": "bioagent-mcp", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"


def _success_response(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_request(request: dict) -> dict:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        return _success_response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {}},
            },
        )

    if method == "tools/list":
        return _success_response(request_id, {"tools": list_tools()})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            result = call_tool(name, arguments)
        except ToolExecutionError as exc:
            return _success_response(request_id, {"ok": False, "error": str(exc)})
        except Exception as exc:
            return _success_response(request_id, {"ok": False, "error": str(exc)})
        if result.get("ok"):
            payload = {"ok": True}
            data = result.get("data")
            if isinstance(data, dict):
                payload.update(data)
            elif data is not None:
                payload["data"] = data
            return _success_response(request_id, payload)
        return _success_response(request_id, {"ok": False, "error": result.get("error")})

    return _error_response(request_id, -32601, f"Method not found: {method}")


def run_stdio_server(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    register_initial_tools()
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout

    while True:
        raw_line = input_stream.buffer.readline() if hasattr(input_stream, "buffer") else input_stream.readline()
        if not raw_line:
            break

        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = raw_line.strip()

        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as exc:
            response = _error_response(None, -32700, f"Parse error: {exc.msg}")
        except ToolExecutionError as exc:
            response = _success_response(None, {"ok": False, "error": str(exc)})
        except Exception as exc:  # pragma: no cover
            response = _success_response(None, {"ok": False, "error": str(exc)})

        output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        output_stream.flush()


__all__ = ["SERVER_INFO", "PROTOCOL_VERSION", "handle_request", "run_stdio_server"]
