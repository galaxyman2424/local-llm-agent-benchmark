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

MAX_READ_LINES = 300


class Actioner:
    """Handles execution of tool calls and other actions.

    This class provides methods that correspond to the tools available to the agent,
    including file operations, code search, command execution, and test running.
    """

class Actioner:
    def __init__(self, model_id="ornith:9b", workspace_dir=None, timeout_seconds=120.0,
             num_ctx=DEFAULT_NUM_CTX, max_read_lines=300, python_bin: str | None = None,
             container_name: str | None = None):
        """``timeout_seconds`` is the max time for a single Ollama request
        (the Actioner's translate-plan-to-tool-call call), independent of
        the agent's overall per-task timeout configured elsewhere.

        ``max_read_lines`` caps how many lines a single read_file call
        returns (see `execute`'s read_file branch) -- keep this in step
        with the model's num_ctx: a bigger context window can afford a
        bigger visible slice per read.

        ``container_name`` -- when set, run_command/run_tests execute
        inside this Docker container (via docker_utils.exec_in_container)
        instead of subprocess.run() on the host. File-editing tools
        (read_file, replace_lines, write_to_file, etc.) are unaffected --
        they operate on workspace_dir directly, which is the same
        directory bind-mounted into the container, so host-side file
        operations against it are already correct in both modes.
        """
        self.model_id = model_id
        self.workspace_dir = workspace_dir or "."
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.max_read_lines = max_read_lines
        self.python_bin = python_bin
        self.container_name = container_name
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
        """Resolve a file path against the current workspace, and refuse to
        let it escape outside workspace_dir.

        Supports an explicit '@workspace:relative/path' prefix, and also
        treats any plain relative path as relative to workspace_dir. Absolute
        paths are only accepted if they're already inside workspace_dir --
        anything else (the model hallucinating a path outside the sandbox,
        e.g. into the shared cloned repo instead of this instance's worktree)
        is rejected rather than silently executed.
        """
        import os

        workspace_root = os.path.realpath(self.workspace_dir)

        if path.startswith("@"):
            parts = path.split(":", 1)
            if len(parts) == 2 and parts[0] == "@workspace":
                candidate = os.path.join(self.workspace_dir, parts[1])
            else:
                candidate = path
        elif os.path.isabs(path):
            candidate = path
        else:
            candidate = os.path.join(self.workspace_dir, path)

        resolved = os.path.realpath(candidate)
        if os.path.commonpath([resolved, workspace_root]) != workspace_root:
            raise ValueError(
                f"Path '{path}' resolves outside the workspace ({workspace_root}). "
                "All file operations must stay inside the workspace."
            )
        return resolved

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
                with open(path, 'r') as f:
                    lines = f.readlines()

                total_lines = len(lines)
                requested_range = params.get("start_line") is not None or params.get("end_line") is not None

                def _to_int(v, default):
                    try:
                        return int(v) if v not in (None, "") else default
                    except (TypeError, ValueError):
                        return default

                start_line = max(1, _to_int(params.get("start_line"), 1))
                end_line = min(total_lines, _to_int(params.get("end_line"), total_lines))
                if end_line < start_line:
                    end_line = start_line

                notice = ""
                if end_line - start_line + 1 > self.max_read_lines:
                    capped_end = start_line + self.max_read_lines - 1
                    notice = (
                        f"\n... [showing lines {start_line}-{capped_end} of {total_lines} total; "
                        f"re-run read_file with start_line/end_line to see more]"
                    )
                    end_line = capped_end
                elif not requested_range and total_lines > self.max_read_lines:
                    end_line = self.max_read_lines
                    notice = (
                        f"\n... [file has {total_lines} lines total; showing 1-{end_line}. "
                        f"Re-run read_file with start_line/end_line to see more]"
                    )

                numbered = "".join(f"{i}: {lines[i-1]}" for i in range(start_line, end_line + 1))
                if numbered and not numbered.endswith("\n"):
                    numbered += "\n"

                return {
                    "tool": tool_name,
                    "result": numbered + notice,
                    "total_lines": total_lines,
                    # Ground truth for the caller's file cache: exactly which
                    # lines were actually shown, and whether this cut off
                    # before the end of the file. Without this, a no-range
                    # request that got capped by max_read_lines looks
                    # indistinguishable from a genuine full-file read, and a
                    # cache built from `result` text alone can't tell the
                    # difference -- see agents/agent.py's _merge_cache_entry.
                    "displayed_start_line": start_line,
                    "displayed_end_line": end_line,
                    "truncated": end_line < total_lines,
                }

            elif tool_name == "replace_lines":
                path = self._resolve_path(params.get("path", ""))
                with open(path, 'r') as f:
                    lines = f.readlines()

                total_lines = len(lines)

                def _to_int(v):
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return None

                start_line = _to_int(params.get("start_line"))
                end_line = _to_int(params.get("end_line"))

                if start_line is None or end_line is None:
                    return {"tool": tool_name, "error": "start_line and end_line must be integers."}
                if start_line < 1 or end_line < start_line:
                    return {"tool": tool_name, "error": f"Invalid range start_line={start_line}, end_line={end_line}."}
                if end_line > total_lines:
                    return {
                        "tool": tool_name,
                        "error": f"end_line={end_line} is beyond the file's {total_lines} lines.",
                    }

                expected_count = params.get("expected_line_count")
                if expected_count not in (None, ""):
                    expected_count = _to_int(expected_count)
                    actual_count = end_line - start_line + 1
                    if expected_count != actual_count:
                        return {
                            "tool": tool_name,
                            "error": (
                                f"expected_line_count={expected_count} doesn't match the "
                                f"range's actual span ({actual_count} lines). The file may "
                                "have changed since you last read it -- re-read it before "
                                "editing to get correct line numbers."
                            ),
                        }

                new_text = params.get("content", "")
                # Preserve a trailing newline on the inserted block so subsequent lines
                # don't get glued onto it, unless the caller's content already ends
                # with one.
                if new_text and not new_text.endswith("\n"):
                    new_text += "\n"
                new_lines = new_text.splitlines(keepends=True)

                updated = lines[:start_line - 1] + new_lines + lines[end_line:]
                with open(path, 'w') as f:
                    f.writelines(updated)

                return {
                    "tool": tool_name,
                    "result": (
                        f"Replaced lines {start_line}-{end_line} ({end_line - start_line + 1} "
                        f"line(s)) in {path} with {len(new_lines)} new line(s)."
                    ),
                }

            elif tool_name == "write_to_file":
                path = self._resolve_path(params.get("path", ""))
                import os
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    f.write(params.get("content", ""))
                return {"tool": tool_name, "result": f"File written to {path}"}

            elif tool_name in ("replace_in_file", "edit_file"):
                path = self._resolve_path(params.get("path", ""))
                with open(path, 'r') as f:
                    content = f.read()

                search_text = params.get("search", "")
                replace_text = params.get("replace", "")

                if not search_text:
                    return {"tool": tool_name, "error": "`search` cannot be empty."}

                occurrences = content.count(search_text)
                match_note = ""

                if occurrences == 0:
                    # The exact search text isn't there. Before giving up,
                    # try two cheap, deterministic recovery strategies rather
                    # than immediately erroring and letting the model retry
                    # the identical (already-failing) search verbatim:
                    #
                    # 1. De-escaped backslashes: small models frequently
                    #    over-escape when generating a `search` string that
                    #    contains regex/backslash sequences (e.g. emitting
                    #    "\\s" for a literal "\s" in the source file). Retry
                    #    with doubled backslashes collapsed.
                    # 2. Fuzzy whitespace match: `_fuzzy_replace` ignores
                    #    per-line leading/trailing whitespace, which covers
                    #    the far more common case of indentation drift
                    #    between what the model remembers and the real file.
                    deescaped_search = search_text.replace("\\\\", "\\")
                    if deescaped_search != search_text and content.count(deescaped_search) == 1:
                        search_text = deescaped_search
                        occurrences = 1
                        match_note = " (matched after collapsing doubled backslashes in `search`)"
                    else:
                        fuzzy_result = self._fuzzy_replace(content, search_text, replace_text)
                        if fuzzy_result is not None:
                            with open(path, 'w') as f:
                                f.write(fuzzy_result)
                            return {
                                "tool": tool_name,
                                "result": (
                                    f"Replaced text in {path} (matched ignoring "
                                    "per-line leading/trailing whitespace)."
                                ),
                            }
                        return {"tool": tool_name, "error": f"SEARCH text not found in {path}."}

                if occurrences > 1:
                    return {
                        "tool": tool_name,
                        "error": (
                            f"SEARCH text appears {occurrences} times in {path}; refusing to "
                            "guess which one. Include more surrounding lines in `search` so "
                            "it matches exactly once."
                        ),
                    }

                new_content = content.replace(search_text, replace_text, 1)
                with open(path, 'w') as f:
                    f.write(new_content)
                return {"tool": tool_name, "result": f"Replaced text in {path} (1 occurrence){match_note}."}

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
                command = self._use_venv_python(params.get("command", ""))
                if self.container_name:
                    from swebench.docker_utils import exec_in_container
                    try:
                        result = exec_in_container(self.container_name, command, timeout_seconds=120.0)
                    except RuntimeError as e:
                        return {"tool": tool_name, "error": str(e), "returncode": None}
                    return {"tool": tool_name, "result": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
                import subprocess
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True,
                        cwd=self.workspace_dir, timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    return {"tool": tool_name, "error": "run_command timed out after 120s", "returncode": None}
                return {"tool": tool_name, "result": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

            elif tool_name == "run_tests":
                test_command = self._use_venv_python(params.get("command", "pytest --tb=short"))
                if self.container_name:
                    from swebench.docker_utils import exec_in_container
                    try:
                        result = exec_in_container(self.container_name, test_command, timeout_seconds=300.0)
                    except RuntimeError as e:
                        return {"tool": tool_name, "error": str(e), "returncode": None}
                    return {"tool": tool_name, "result": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
                import subprocess
                try:
                    result = subprocess.run(
                        test_command, shell=True, capture_output=True, text=True,
                        cwd=self.workspace_dir, timeout=300,
                    )
                except subprocess.TimeoutExpired:
                    return {"tool": tool_name, "error": "run_tests timed out after 300s", "returncode": None}
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
            git_grep_cmd, capture_output=True, text=True, cwd=self.workspace_dir, timeout=30,
        )
        if result.returncode in (0, 1):  # 0 = matches found, 1 = no matches (still a valid run)
            matches = [line.strip() for line in result.stdout.split("\n") if line.strip()]
            return {"tool": "search_code", "result": matches}

        # Fall back to plain grep (not a git repo, or git grep unavailable)
        exclude_args = []
        for d in exclude_dirs:
            exclude_args += ["--exclude-dir=" + d]
        grep_cmd = ["grep", "-r", "-n", *exclude_args, query, search_root]
        result = subprocess.run(grep_cmd, capture_output=True, text=True, timeout=30)
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

        # write_to_file params are stripped of large string values before
        # they ever enter the prompt (see _extract_large_values below), so
        # in the common case this call never has to regenerate file
        # content and 512 tokens is plenty. This bump is a safety net for
        # cases that don't go through the placeholder path (e.g. content
        # under the extraction threshold, or a schema mismatch that forces
        # the model to retype something) rather than the primary defense
        # against truncation.
        num_predict = 4096 if next_action == "write_to_file" else 512

        reasoner_params = reasoner_plan.get("parameters", {}) or {}

        # Common Reasoner confusion: it says next_action="replace_in_file" (or
        # "edit_file") but actually supplies replace_lines-shaped parameters
        # (start_line/end_line/content instead of search/replace) -- likely
        # because it knows the exact line range but reaches for the wrong
        # tool name. Left as-is, this fails schema validation, falls through
        # to the slow LLM-translation path, and small Actioner models
        # frequently can't recover the original intent there at all (seen in
        # practice: it just re-reads the file instead). Since the *shape* of
        # the parameters unambiguously indicates which tool was meant, fix
        # the tool name here -- deterministically, before any LLM call --
        # rather than hoping the translation step guesses it.
        if next_action in ("replace_in_file", "edit_file") and isinstance(reasoner_params, dict):
            has_replace_lines_shape = (
                "search" not in reasoner_params
                and "start_line" in reasoner_params
                and "end_line" in reasoner_params
            )
            if has_replace_lines_shape:
                if "content_str" in reasoner_params and "content" not in reasoner_params:
                    reasoner_params = dict(reasoner_params)
                    reasoner_params["content"] = reasoner_params.pop("content_str")
                print(f"[Actioner] Reasoner asked for '{next_action}' but supplied "
                      "replace_lines-shaped parameters (start_line/end_line/content) "
                      "-- correcting tool to 'replace_lines'.")
                next_action = "replace_lines"

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
Do not use absolute paths under any circumstances -- always give paths
relative to the workspace shown above. Any absolute path outside this
workspace will be rejected.
Do not create unrelated files.
Do not invent projects.

The Reasoner has already decided WHAT should happen next. Your only job is
to translate that decision into exactly ONE concrete tool call matching
one of the schemas below. Do not solve the underlying task yourself, do
not add extra steps, and do not change the Reasoner's intent.

REASONER'S CHOSEN NEXT ACTION:
{next_action}

REASONER'S SUGGESTED PARAMETERS:
{json.dumps(safe_params)}

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
            #print(f"[Actioner] Done reason: {response.get('done_reason', 'unknown')}")
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
                #print(f"[Actioner] Raw model response: {text[:2000]!r}")
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

        # Small local models are prone to using a slightly different key name
        # than the schema expects (e.g. "file_path" instead of "path") even
        # when json_mode constrains the JSON to be syntactically valid -- it
        # doesn't constrain which keys get used. Remap known aliases before
        # validating, rather than rejecting an otherwise-usable action outright.
        PARAM_ALIASES = {
            "read_file": ["file_path", "filepath", "filename", "file"],
            "write_to_file": ["file_path", "filepath", "filename", "file"],
            "delete_file": ["file_path", "filepath", "filename", "file"],
            "replace_in_file": ["file_path", "filepath", "filename", "file"],
        }

        if isinstance(action, dict) and action.get("tool") in PARAM_ALIASES:
            params = action.get("parameters")
            if isinstance(params, dict) and "path" not in params:
                for alias in PARAM_ALIASES[action["tool"]]:
                    if alias in params:
                        params["path"] = params.pop(alias)
                        break
            action["parameters"] = params

        is_valid, error = validate_action(action)
        if not is_valid:
            print(f"[Actioner] Rejected invalid action from model: {error}")
            print(f"[Actioner] Raw action: {action!r}")
            return None

        is_valid, error = validate_action(action)
        if not is_valid:
            print(f"[Actioner] Rejected invalid action from model: {error}")
            print(f"[Actioner] Raw action: {action!r}")
            return None


        print(f"[Actioner] Chose: {action}")

        return action

def _use_venv_python(self, command: str) -> str:
    """Rewrite bare `pytest`/`python(3)` invocations to use this workspace's
    isolated venv interpreter, so mid-run test signal matches the venv the
    benchmark actually evaluates against."""
    import re
    if self.python_bin in (None, "python"):
        return command
    command = re.sub(r'(?<![\w./])pytest\b', f'"{self.python_bin}" -m pytest', command)
    command = re.sub(r'(?<![\w./])python3?\b', f'"{self.python_bin}"', command)
    return command

def _extract_large_values(params: dict, threshold: int = 80) -> tuple[dict, dict]:
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