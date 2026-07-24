"""Actioner module providing a unified interface for all agent actions."""

from __future__ import annotations
import json
from typing import Any
from .ollama_client import OllamaClient


class Actioner:
    """Handles execution of tool calls and other actions.

    This class provides methods that correspond to the tools available to the agent,
    including file operations, code search, command execution, and test running.
    """

    def __init__(
        self,
        model_id="ornith:9b",
        workspace_dir=None,
    ):
        self.model_id = model_id
        self.workspace_dir = workspace_dir or "."
        self.client = OllamaClient(
            model=model_id,
            timeout_seconds=120.0,
        )
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

    def plan_action(
        self,
        task: str,
        reasoner_plan: dict,
        previous_actions: list[dict],
    ) -> dict | None:
        prompt = """
        You are an action executor operating inside the provided workspace.

        You must modify ONLY the current workspace.

        Never invent an absolute path.
        Never use /home/user.
        Never use a path from an example or previous task.

        For file operations, always use paths relative to the workspace.

        Available tools:
        - read_file
        - write_to_file
        - replace_in_file
        - delete_file
        - list_directory
        - search_code
        - run_command
        - run_tests
        - get_git_diff

        Return exactly one JSON object and nothing else.

        The JSON format MUST be:

        {
        "tool": "tool_name",
        "parameters": {
            ...
        }
        }

        Do not put the tool parameters at the top level.

        For write_to_file, only write a file when the reasoner explicitly requests it.
        Prefer replace_in_file for modifying an existing file.
        Never generate unrelated example projects.
        """

        response = self.client.chat(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            json_mode=True,
            num_predict=512,
            think=False,
        )

        text = response["message"]["content"]

        if not text.strip():
            print("[Actioner] Ollama returned empty content.")
            return None

        action = self.actioner.plan_action(
            task=task,
            reasoner_plan=reasoner_plan,
            previous_actions=previous_actions,
        )

        if not action:
            print("[Agent] Actioner failed to produce a valid tool call.")
            continue

        return action
