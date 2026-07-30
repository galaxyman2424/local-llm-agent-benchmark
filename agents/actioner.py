"""Actioner module providing a unified interface for all agent actions.

The Actioner has two distinct responsibilities that must not blur
together:

1. ``plan_action`` -- calls an LLM to translate the Reasoner's plan into
   exactly ONE concrete, schema-valid tool call. It never decides on its
   own what the overall task needs; it only operationalizes what the
   Reasoner already decided.
2. ``execute`` -- a purely deterministic executor. Given a validated tool
   call, it runs it and returns the result. It never calls an LLM.
"""

from __future__ import annotations
import json
from typing import Any

from .ollama_client import OllamaClient
from .tool_schemas import TOOL_SCHEMAS, schema_prompt_block, validate_action
from .json_utils import extract_json_object, repair_truncated_json

# See reasoner.py's DEFAULT_NUM_CTX for why this matters: Ollama's default
# context window is often too small once tool schemas + the Reasoner's
# suggested parameters are embedded in the prompt, causing the model to hit
# the context ceiling mid-JSON well before num_predict would ever apply.
DEFAULT_NUM_CTX = 16384


class Actioner:
    """Handles execution of tool calls and other actions.

    This class provides methods that correspond to the tools available to the agent,
    including file operations, code search, command execution, and test running.
    """

    def __init__(
        self,
        model_id="ornith:9b",
        workspace_dir=None,
        timeout_seconds: float = 120.0,
        num_ctx: int = DEFAULT_NUM_CTX,
    ):
        """``timeout_seconds`` is the max time for a single Ollama request
        (the Actioner's translate-plan-to-tool-call call), independent of
        the agent's overall per-task timeout configured elsewhere.
        """
        self.model_id = model_id
        self.workspace_dir = workspace_dir or "."
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.client = OllamaClient(
            model=model_id,
            timeout_seconds=timeout_seconds,
        )
        self._history = []

    def _fuzzy_replace(self, content: str, search_text: str, replace_text: str) -> str | None:
        """Try to match `search_text` against `content` ignoring per-line
        leading/trailing whitespace, and re-indent `replace_text` to match
        what was actually found. Returns None if still no match.
        """
        content_lines = content.split("\n")
        search_lines = search_text.split("\n")
        if not search_lines:
            return None

        n = len(search_lines)
        stripped_search = [l.strip() for l in search_lines]

        for i in range(len(content_lines) - n + 1):
            window = content_lines[i:i + n]
            if [l.strip() for l in window] == stripped_search:
                # Reuse the indentation of the first matched line
                indent = window[0][:len(window[0]) - len(window[0].lstrip())]
                replace_lines = [indent + l if l.strip() else l for l in replace_text.split("\n")]
                new_lines = content_lines[:i] + replace_lines + content_lines[i + n:]
                return "\n".join(new_lines)
        return None

    def _resolve_path(self, path: str) -> str:
        """Resolve a file path against the current workspace.

        Supports an explicit '@workspace:relative/path' prefix, and also
        treats any plain relative path as relative to workspace_dir (rather
        than the process's ambient cwd) since that's what callers actually
        produce in practice. Absolute paths pass through unchanged.

        The Actioner/LLM is only ever expected to produce paths relative to
        the workspace -- this method (the deterministic executor's
        responsibility, not the LLM's) is what actually anchors them to
        ``self.workspace_dir``.
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
        """Deterministically execute a single validated tool call.

        Callers (the Agent loop) should validate the action with
        :func:`tool_schemas.validate_action` before calling this -- but as
        a safety net, this also re-validates and refuses to run anything
        malformed rather than guessing at intent.
        """
        is_valid, error = validate_action(action)
        if not is_valid:
            return {"tool": (action or {}).get("tool", "unknown"), "error": f"Rejected invalid action: {error}"}

        tool_name = action.get("tool", "unknown")
        params = action.get("parameters", {}) or {}
        self._history.append(action)

        try:
            if tool_name == "read_file":
                path = self._resolve_path(params.get("path", ""))
                with open(path, "r") as f:
                    lines = f.readlines()
                start = params.get("start_line")
                end = params.get("end_line")
                if start or end:
                    s = max((start or 1) - 1, 0)
                    e = end or len(lines)
                    content = "".join(lines[s:e])
                else:
                    content = "".join(lines)
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
                    # NEW: try again ignoring per-line leading/trailing whitespace
                    fuzzy_result = self._fuzzy_replace(content, search_text, replace_text)
                    if fuzzy_result is not None:
                        with open(path, 'w') as f:
                            f.write(fuzzy_result)
                        return {"tool": tool_name, "result": "Replaced text successfully (fuzzy match)"}
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
                return self._search_code(params)

            elif tool_name == "run_command":
                command = params.get("command", "")
                import subprocess
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True,
                        cwd=self.workspace_dir, timeout=300,
                    )
                except subprocess.TimeoutExpired as e:
                    return {"tool": tool_name, "error": f"Command timed out after 120s: {e}", "returncode": None}
                return {"tool": tool_name, "result": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

            elif tool_name == "run_tests":
                test_command = params.get("command", "pytest --tb=short")
                import subprocess
                try:
                    result = subprocess.run(
                        test_command, shell=True, capture_output=True, text=True,
                        cwd=self.workspace_dir, timeout=300,
                    )
                except subprocess.TimeoutExpired as e:
                    return {"tool": tool_name, "error": f"Tests timed out after 120s: {e}", "returncode": None}
                return {"tool": tool_name, "result": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

            elif tool_name == "get_git_diff":
                diff_ref = params.get("diff", "")
                import subprocess
                cmd = ["git", "diff"] if not diff_ref else ["git", "diff", diff_ref]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=self.workspace_dir,
                )
                return {"tool": tool_name, "result": result.stdout}

            else:
                return {"tool": tool_name, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"tool": action.get("tool", ""), "error": str(e)}

    def _search_code(self, params: dict[str, Any]) -> dict[str, Any]:
        """Search the repository broadly, not just *.py files.

        Prefers `git grep` (respects .gitignore, and is fast even on large
        repos) when the workspace is a git repo; falls back to a plain
        `grep -r` that explicitly excludes VCS/build/venv/cache dirs
        otherwise. This matters for SWE-bench issues that live in
        docs/config (.rst, .cfg, .yaml, .toml, .md), which the old
        `--include=*.py`-only search could never find.
        """
        import subprocess

        query = params.get("query", "")
        subpath = params.get("path", "")
        search_root = self._resolve_path(subpath) if subpath else self.workspace_dir

        exclude_dirs = [".git", ".swebench_venv", "__pycache__", "build", "dist", "node_modules"]

        git_grep_cmd = ["git", "grep", "-n", "-I", "--untracked", query, "--", subpath or "."]
        result = subprocess.run(
            git_grep_cmd, capture_output=True, text=True, cwd=self.workspace_dir,
        )
        if result.returncode in (0, 1):  # 0 = matches found, 1 = no matches (still a valid run)
            matches = [line.strip() for line in result.stdout.split("\n") if line.strip()]
            return {"tool": "search_code", "result": matches}

        # Fall back to plain grep (not a git repo, or git grep unavailable)
        exclude_args = []
        for d in exclude_dirs:
            exclude_args += ["--exclude-dir=" + d]
        grep_cmd = ["grep", "-r", "-n", *exclude_args, query, search_root]
        result = subprocess.run(grep_cmd, capture_output=True, text=True)
        matches = [line.strip() for line in result.stdout.split("\n") if line.strip()]
        return {"tool": "search_code", "result": matches}

    def get_history(self) -> list[dict]:
        """Return the execution history."""
        return self._history

    def plan_action(
        self,
        task: str,
        reasoner_plan: dict,
        previous_actions: list[dict],
    ) -> dict | None:
        """Ask the Actioner model to translate ``reasoner_plan`` into exactly
        one concrete, schema-valid tool call.

        This method must NEVER independently decide what the task needs --
        it only operationalizes the Reasoner's ``next_action`` /
        ``parameters`` into the exact JSON shape the deterministic
        executor expects, resolving any ambiguity (e.g. filling in a
        workspace-relative path) without inventing new goals.
        """
        next_action = reasoner_plan.get("next_action", "")
        reasoner_params = reasoner_plan.get("parameters", {}) or {}
        
        next_action = result.get("next_action", result.get("action", "search_code"))
        if next_action == "done" and not _tests_passed(previous_actions):
            print("[Reasoner.plan] Model chose 'done' without a passing run_tests in "
                  "history; overriding to 'run_tests' instead.")
            next_action = "run_tests"

        return {
            "next_action": next_action,
            "parameters": result.get("parameters", result.get("params", {})),
            "expected_outcome": result.get("expected_outcome", result.get("outcome", "")),
        }

        # Fast path: the Reasoner already gave us a valid, schema-shaped action.
        # Don't waste a token-capped LLM call re-typing a full file body back out.
        direct_candidate = {"tool": next_action, "parameters": reasoner_params}
        is_valid, _ = validate_action(direct_candidate)
        if is_valid:
            print(f"[Actioner] Passthrough (no LLM call needed): {next_action}")
            return direct_candidate

        # Slow path: genuine translation needed. Strip any large string values
        # out before they ever enter the prompt/generation budget, and splice
        # them back in afterward -- the Actioner model only ever has to move a
        # placeholder token around, never regenerate file content.
        safe_params, placeholders = _extract_large_values(reasoner_params)

        prompt = f"""You are an action executor operating inside a single, fixed workspace.

You are operating in this workspace:
{self.workspace_dir}

You MUST NOT invent a different workspace.
All relative paths are relative to this workspace.
Do not create unrelated files.
Do not invent projects.
Do not use example paths such as /home/user/project or any absolute path
from an example or a previous, unrelated task.

The Reasoner has already decided WHAT should happen next. Your only job is
to translate that decision into exactly ONE concrete tool call matching
one of the schemas below. Do not solve the underlying task yourself, do
not add extra steps, and do not change the Reasoner's intent.

REASONER'S CHOSEN NEXT ACTION:
{next_action}

REASONER'S SUGGESTED PARAMETERS:
{json.dumps(reasoner_params)}

Available tools and their required/optional parameters:
{schema_prompt_block()}

Return exactly one JSON object and nothing else, no Markdown, no
explanations, no multiple tool calls, and no large generated file
contents unless the tool is specifically write_to_file and the Reasoner
explicitly asked for a new file.

The JSON format MUST be:

{{
"tool": "tool_name",
"parameters": {{
    ...
}}
}}

Do not put the tool parameters at the top level.
"""

        print("=" * 20, "ACTIONER PROMPT", "=" * 20)
        print(prompt)
        print("=" * 60)


        try:
            response = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                json_mode=True,
                num_predict=num_predict,
                num_ctx=self.num_ctx,
                keep_alive=0,
                think=False,
            )
        except RuntimeError as e:
            print(f"[Actioner] Ollama request failed: {e}")
            return None

        message = response.get("message", {})
        text = message.get("content", "")

        if not text.strip():
            print("[Actioner] Ollama returned empty content.")
            print(f"[Actioner] Done reason: {response.get('done_reason', 'unknown')}")
            return None

        candidate = extract_json_object(text)
        if candidate is None:
            # Likely truncated by the context window rather than genuinely
            # malformed -- try to repair (close the unterminated
            # string/braces) before giving up.
            repaired = repair_truncated_json(text)
            action = None
            if repaired is not None:
                try:
                    action = json.loads(repaired)
                    print("[Actioner] Model reply was truncated mid-JSON (likely num_ctx too small "
                          f"for this prompt: prompt_eval_count={response.get('prompt_eval_count', '?')} "
                          f"eval_count={response.get('eval_count', '?')} num_ctx={self.num_ctx}); "
                          "repaired it and continuing.")
                except json.JSONDecodeError:
                    action = None
            if action is None:
                print("[Actioner] Model reply contained no JSON object (and could not be repaired).")
                print(f"[Actioner] done_reason={response.get('done_reason', '?')} "
                      f"prompt_eval_count={response.get('prompt_eval_count', '?')} "
                      f"eval_count={response.get('eval_count', '?')} num_ctx={self.num_ctx}")
                print(f"[Actioner] Raw model response: {text[:2000]!r}")
                return None
        else:
            try:
                action = json.loads(candidate)
            except json.JSONDecodeError as e:
                print(f"[Actioner] Model reply was not valid JSON: {e}")
                print(f"[Actioner] Raw candidate: {candidate[:2000]!r}")
                return None

        # Normalize: some models put parameters at the top level despite
        # instructions. Fold any recognized required/optional keys for the
        # named tool into `parameters` if `parameters` itself is missing.
        if isinstance(action, dict) and "parameters" not in action and action.get("tool") in TOOL_SCHEMAS:
            tool = action["tool"]
            schema = TOOL_SCHEMAS[tool]
            known_keys = set(schema["required"]) | set(schema["optional"])
            folded = {k: v for k, v in action.items() if k in known_keys}
            if folded:
                action = {"tool": tool, "parameters": folded}

        # Swap placeholder tokens back for the real (large) values the model
        # never actually had to see or regenerate.
        action = _restore_large_values(action, placeholders)

        is_valid, error = validate_action(action)
        if not is_valid:
            print(f"[Actioner] Rejected invalid action from model: {error}")
            print(f"[Actioner] Raw action: {action!r}")
            return None


        print(f"[Actioner] Chose: {action}")

        return action


def _extract_large_values(params: dict, threshold: int = 200) -> tuple[dict, dict]:
    safe, placeholders = {}, {}
    for i, (k, v) in enumerate(params.items()):
        if isinstance(v, str) and len(v) > threshold:
            key = f"__PLACEHOLDER_{i}__"
            placeholders[key] = v
            safe[k] = key
        else:
            safe[k] = v
    return safe, placeholders

def _restore_large_values(action: dict, placeholders: dict) -> dict:
    if not placeholders or not isinstance(action, dict):
        return action
    params = action.get("parameters", {}) or {}
    for k, v in params.items():
        if isinstance(v, str) and v in placeholders:
            params[k] = placeholders[v]
    action["parameters"] = params
    return action