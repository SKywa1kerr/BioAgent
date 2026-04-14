from copy import deepcopy


_TOOL_REGISTRY = {}


def register_tool(name, description, parameters, execute):
    _TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": deepcopy(parameters),
        "execute": execute,
    }


def list_tools():
    tools = []
    for tool in _TOOL_REGISTRY.values():
        tools.append(
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": deepcopy(tool["parameters"]),
            }
        )
    return tools


def call_tool(name, arguments):
    tool = _TOOL_REGISTRY.get(name)
    if tool is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    arguments = arguments or {}
    return {"ok": True, "data": tool["execute"](**arguments)}


__all__ = ["register_tool", "list_tools", "call_tool"]
