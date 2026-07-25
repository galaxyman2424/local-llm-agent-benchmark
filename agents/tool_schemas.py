"""Centralized tool schema definitions.

Both the Reasoner (so it knows what parameters each tool needs when
proposing a plan) and the Actioner (so it can validate a concrete tool
call before executing it) import from this single source of truth,
rather than each hard-coding its own copy of the tool list in a prompt
string that can drift out of sync with what ``Actioner.execute`` actually
implements.
"""

from __future__ import annotations

from typing import Any

# tool_name -> {"required": [...], "optional": [...], "description": "..."}
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": {
        "required": ["path"],
        "optional": [],
        "description": "Read the full contents of a file at `path` (relative to the workspace).",
    },
    "write_to_file": {
        "required": ["path", "content"],
        "optional": [],
        "description": "Create or overwrite a file at `path` with `content`. Only use when explicitly needed -- never to invent a new unrelated project.",
    },
    "replace_in_file": {
        "required": ["path", "search", "replace"],
        "optional": [],
        "description": "Replace the first occurrence of `search` with `replace` inside the file at `path`.",
    },
    "delete_file": {
        "required": ["path"],
        "optional": [],
        "description": "Delete the file at `path`.",
    },
    "list_directory": {
        "required": [],
        "optional": ["path"],
        "description": "List files/directories under `path` (defaults to the workspace root).",
    },
    "search_code": {
        "required": ["query"],
        "optional": ["path"],
        "description": "Search the repository for `query` (optionally scoped to `path`).",
    },
    "run_command": {
        "required": ["command"],
        "optional": [],
        "description": "Run an arbitrary shell command inside the workspace.",
    },
    "run_tests": {
        "required": ["command"],
        "optional": [],
        "description": "Run the test suite using `command`.",
    },
    "get_git_diff": {
        "required": [],
        "optional": ["diff"],
        "description": "Return the current uncommitted (or `diff`-ref-relative) git diff of the workspace.",
    },
}


def schema_prompt_block() -> str:
    """Render TOOL_SCHEMAS as a human-readable block for prompts."""
    lines = []
    for name, schema in TOOL_SCHEMAS.items():
        req = ", ".join(schema["required"]) or "(none)"
        opt = ", ".join(schema["optional"]) or "(none)"
        lines.append(f"- {name}: required=[{req}] optional=[{opt}] -- {schema['description']}")
    return "\n".join(lines)


def validate_action(action: dict[str, Any] | None) -> tuple[bool, str]:
    """Validate a concrete tool-call dict against TOOL_SCHEMAS.

    Returns ``(is_valid, error_message)``. ``error_message`` is empty when
    valid. This is the single gate ``Actioner.execute`` (or its caller)
    should use to reject malformed actions instead of executing them.
    """
    if not isinstance(action, dict):
        return False, "Action is not a JSON object."

    tool = action.get("tool")
    if not tool or not isinstance(tool, str):
        return False, "Action is missing a string 'tool' field."

    if tool not in TOOL_SCHEMAS:
        return False, f"Unknown tool '{tool}'. Valid tools: {', '.join(TOOL_SCHEMAS)}"

    params = action.get("parameters")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return False, "Action 'parameters' must be a JSON object."

    missing = [p for p in TOOL_SCHEMAS[tool]["required"] if p not in params or params[p] in (None, "")]
    if missing:
        return False, f"Tool '{tool}' is missing required parameter(s): {', '.join(missing)}"

    return True, ""
