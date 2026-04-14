import bioagent.mcp_tools as mcp_tools


PARAMETERS = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
        "name": {"type": "string"},
    },
}


def test_register_and_list_tools_excludes_execute():
    def add(*, a, b):
        return a + b

    mcp_tools.register_tool(
        name="add",
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        execute=add,
    )

    assert mcp_tools.list_tools() == [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        }
    ]


def test_call_tool_returns_success_result():
    mcp_tools.register_tool(
        name="greet",
        description="Return a greeting.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        execute=lambda *, name: f"Hello, {name}!",
    )

    assert mcp_tools.call_tool("greet", {"name": "BioAgent"}) == {
        "ok": True,
        "data": "Hello, BioAgent!",
    }


def test_call_tool_returns_error_for_unknown_tool():
    assert mcp_tools.call_tool("missing", {"name": "BioAgent"}) == {
        "ok": False,
        "error": "Unknown tool: missing",
    }
