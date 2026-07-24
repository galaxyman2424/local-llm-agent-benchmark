"""Actioner module providing a unified interface for all agent actions."""

import json
from typing import Any


class Actioner:
    """Handles execution of tool calls and other actions.

    This class provides methods that correspond to the tools available to the agent,
    including file operations, code search, command execution, and test running.
    """

    def __init__(self, model_id: str = "ornith:9b", workspace_dir: str | None = None):
        # model_id is accepted for interface symmetry with Reasoner and so
        # config-driven callers (e.g. experiments/run_experiment.py) can pass
        # an actioner model name; the Actioner itself doesn't call an LLM
        # directly today; it just executes concrete tool calls handed to it.
        self.model_id = model_id
        self.workspace_dir = workspace_dir or "."
        self._history = []

    def _resolve_path(self, path: str) -> str:
        """Resolve a file path against the current workspace.

        Supports an explicit '@workspace:relative/path' prefix, and also
        treats any plain relative path as relative to workspace_dir (rather
        than the process's ambient cwd) since that's what callers actually
        produce in practice. Absolute paths pass through unchanged.
        """
        import os

        if path.startswith("@"):
            parts = path.split(":", 1)
            if len(parts) == 2 and parts[0] == "@workspace":
                return os.path.join(self.workspace_dir, parts[1])
            return path

        if os.path.isabs(path):
            return path

        return os.path.join(self.workspace_dir, path)

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call and return the result."""
        tool_name = action.get("tool", "unknown")
        params = action.get("parameters", {})
        self._history.append(action)

        try:
            if tool_name == "read_file":
                path = self._resolve_path(params.get("path", ""))
                with open(path, 'r') as f:
                    content = f.read()
                return {"tool": tool_name, "result": content}

            elif tool_name == "write_to_file":
                path = self._resolve_path(params.get("path", ""))
                with open(path, 'w') as f:
                    f.write(params.get("content", ""))
                return {"tool": tool_name, "result": f"File written to {path}"}

            elif tool_name in ("replace_in_file", "edit_file"):
                path = self._resolve_path(params.get("path", ""))
                with open(path, 'r') as f:
                    content = f.read()

                # Simple SEARCH/REPLACE logic
                search_text = params.get("search", "")
                replace_text = params.get("replace", "")
                new_content = content.replace(search_text, replace_text)
                if search_text in content:
                    with open(path, 'w') as f:
                        f.write(new_content)
                    return {"tool": tool_name, "result": "Replaced text successfully"}
                else:
                    return {"tool": tool_name, "error": f"SEARCH/REPLACE failed. Text not found in {path}"}

            elif tool_name == "delete_file":
                path = self._resolve_path(params.get("path", ""))
                import os
                if os.path.exists(path):
                    os.remove(path)
                    return {"tool": tool_name, "result": f"Deleted {path}"}
                else:
                    return {"tool": tool_name, "error": f"File not found: {path}"}

            elif tool_name == "list_directory":
                path = self._resolve_path(params.get("path", "."))
                import os
                items = []
                for item in os.listdir(path):
                    full_path = os.path.join(path, item)
                    if os.path.isdir(full_path):
                        items.append({"name": item, "type": "directory"})
                    else:
                        items.append({"name": item, "type": "file"})
                return {"tool": tool_name, "result": items}

            elif tool_name == "search_code":
                path = self._resolve_path(params.get("path", self.workspace_dir))
                query = params.get("query", "")
                import subprocess
                result = subprocess.run(
                    ["grep", "-r", "-n", "--include=*.py", query, path],
                    capture_output=True, text=True
                )
                matches = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                return {"tool": tool_name, "result": matches}

            elif tool_name == "run_command":
                command = params.get("command", "")
                import subprocess
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    cwd=self.workspace_dir,
                )
                return {
                    "tool": tool_name,
                    "result": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif tool_name == "run_tests":
                test_command = params.get("command", "pytest --tb=short")
                import subprocess
                result = subprocess.run(
                    test_command, shell=True, capture_output=True, text=True,
                    cwd=self.workspace_dir,
                )
                return {
                    "tool": tool_name,
                    "result": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

            elif tool_name == "get_git_diff":
                diff_ref = params.get("diff", "")
                import subprocess
                cmd = ["git", "diff"] if not diff_ref else ["git", "diff", diff_ref]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=self.workspace_dir,
                )
                return {"tool": tool_name, "result": result.stdout}

            elif tool_name == "submit_solution":
                instance_id = params.get("instance_id", "")
                solution_path = params.get("solution_path", "")
                # In production, this would submit to SWE-bench API
                return {
                    "tool": tool_name,
                    "result": f"Solution submitted for {instance_id}",
                    "solution_path": solution_path
                }

            else:
                return {"tool": tool_name, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"tool": action.get("tool", ""), "error": str(e)}

    def get_history(self) -> list[dict]:
        """Return the execution history."""
        return self._history
