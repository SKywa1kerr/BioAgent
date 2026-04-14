import bioagent.mcp_server as mcp_server


def test_handle_initialize_returns_server_capabilities():
    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "bioagent-mcp", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        },
    }


def test_handle_tools_list_returns_registered_tools(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "list_tools",
        lambda: [{"name": "analyze_sequences", "description": "Analyze data", "parameters": {}}],
    )

    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": [
                {"name": "analyze_sequences", "description": "Analyze data", "parameters": {}}
            ]
        },
    }


def test_handle_unknown_method_returns_error():
    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "missing/method",
            "params": {},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 3,
        "error": {
            "code": -32601,
            "message": "Method not found: missing/method",
        },
    }


def test_handle_tools_call_flattens_successful_tool_result(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "call_tool",
        lambda name, arguments: {"ok": True, "data": {"analysis_id": "run-1", "samples": 2}},
    )

    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "analyze_sequences", "arguments": {"dataset": "base"}},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 4,
        "result": {
            "ok": True,
            "analysis_id": "run-1",
            "samples": 2,
        },
    }


def test_handle_tools_call_returns_tool_error(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "call_tool",
        lambda name, arguments: {"ok": False, "error": "Unknown tool: missing"},
    )

    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        }
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 5,
        "result": {
            "ok": False,
            "error": "Unknown tool: missing",
        },
    }
