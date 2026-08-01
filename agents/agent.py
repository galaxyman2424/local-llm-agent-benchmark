"""Agent: orchestrates the reasoner and actioner to solve tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from .tool_schemas import validate_action

# Meta-actions the Reasoner may choose that are NOT tool calls and should
# stop the loop immediately rather than being handed to the Actioner.
STOP_ACTIONS = ("done", "submit_solution")

# How many times in a row the exact same concrete tool call may be
# produced before the loop gives up rather than burning iterations.
MAX_REPEATED_ACTIONS = 4

# Every distinct way the loop can end -- makes "incomplete"/"timeout"
# statuses actually explain themselves instead of just being a dead end.
STOP_REASONS = (
    "tests_passed",
    "reasoner_done",       # Reasoner explicitly chose "done"/"submit_solution"
    "reasoner_failed",     # Reasoner returned no usable plan (after its own retry)
    "actioner_failed",     # Actioner couldn't produce a valid tool call
    "repeated_action",     # Same (tool, parameters) fired too many times
    "max_iterations",      # Ran out of budget
)


@dataclass
class AgentResult:
    """Final result of running the agent on one task.

    This is the ONE result contract every layer (Agent, benchmarks,
    experiments) agrees on. ``Agent.solve`` always returns this dataclass
    -- never a raw dict. Callers that need a JSON-serializable dict should
    use ``dataclasses.asdict(result)`` rather than hand-rolling one.

    Status vocabulary (internal to the Agent -- benchmarks may map these
    to their own "resolved"/"not_resolved" terminology, but must not mix
    vocabularies within this dataclass itself):

        passed      - the configured test_command succeeded
        failed      - the configured test_command ran and failed
        incomplete  - the loop ended without ever running tests
        timeout     - the loop hit max_iterations without resolving
        error       - the loop ended due to an unrecoverable error
    """
    instance_id: str = ""
    repo_name: str = ""
    base_commit: str = ""
    num_iterations: int = 0
    total_tool_calls: int = 0
    test_passed: bool | None = None
    final_patch: str = ""
    status: str = "incomplete"  # passed, failed, incomplete, timeout, error
    exit_reason: str = ""  # one of STOP_REASONS

    # NEW: diagnostic fields showing where the agent left off
    stop_reason: str = ""            # see STOP_REASONS below
    last_reasoner_plan: dict = field(default_factory=dict)
    last_action: dict = field(default_factory=dict)
    last_result: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

class Agent:
    """Top-level agent that runs a task to completion.

    Loops over iterations: the reasoner proposes a plan (WHAT to do next)
    and the actioner translates that into exactly one concrete tool call,
    which is then deterministically executed. The result feeds back into
    the reasoner as part of previous_actions. Repeats until tests pass,
    the reasoner signals completion, or max_iterations is reached.
    """
    def __init__(self, reasoner, actioner, max_iterations: int = 50, timeout: float | None = None):
        self.reasoner = reasoner
        self.actioner = actioner
        self.max_iterations = max_iterations
        self.timeout = timeout

    def solve(self, repo_path: str, task: str, test_command: str = "pytest --tb=short") -> AgentResult:
        """Run the agent on a single SWE-bench-style task."""
        # Point the actioner at this task's repo so file/command tools
        # operate on the right workspace instead of the ambient cwd.
        self.actioner.workspace_dir = repo_path

        previous_actions: list[dict] = []
        current_state: dict = {
            "repo_path": repo_path,
            "test_command": test_command,
            "file_cache": {}
        }

        # blindness on iteration 1.
        current_state["file_tree"] = self._get_initial_file_listing(repo_path)

        num_iterations = 0
        total_tool_calls = 0
        test_passed: bool | None = None
        repeated_action_count = 0
        last_action_key = None

        stop_reason = "max_iterations"          # default if we exhaust the loop
        last_reasoner_plan: dict = {}
        last_action: dict = {}
        last_result: dict = {}

        #oscillation detection window + test-staleness tracking
        action_window: deque = deque(maxlen=8)
        last_test_iteration = -999

        import time
        solve_start = time.time()
        timed_out = False

        #recording exit reason
        exit_reason = ""

        for iteration in range(self.max_iterations):
            num_iterations = iteration + 1

            if self.timeout is not None and (time.time() - solve_start) > self.timeout:
                print(f"[Agent] Overall timeout ({self.timeout}s) exceeded; stopping.")
                timed_out = True
                exit_reason = "timeout"
                break

            # If the last executed action repeated at least once, tell the
            # Reasoner explicitly to avoid it -- don't wait for the
            # 2-repeat _loop_warning threshold in reasoner.py, since by
            # then the model has already committed to the pattern once.
            avoid_action = None
            if repeated_action_count >= 1 and last_action_key is not None:
                avoid_action = last_action_key

            #nudge toward running tests if changes are piling up untested
            has_changes = bool(previous_actions) and any(
                r["action"].get("tool") in ("write_to_file", "replace_in_file")
                for r in previous_actions
            )
            stagnation_hint = None
            if has_changes and (iteration - last_test_iteration) > 6:
                stagnation_hint = (
                    "You have made file changes but have not run tests in over "
                    "6 iterations. Strongly consider choosing run_tests next."
                )

            reasoner_plan = self.reasoner.plan(
                task=task,
                current_state=current_state,
                previous_actions=previous_actions,
                avoid_action=avoid_action,
                stagnation_hint=stagnation_hint,   # NEW
            )

            if not reasoner_plan:
                print("[Agent] Reasoner failed to produce a plan.")
                stop_reason = "reasoner_failed"
                exit_reason = "reasoner_failed"
                break

            # 2. Stop conditions from the Reasoner
            if reasoner_plan.get("next_action") in STOP_ACTIONS:
                print(f"[Agent] Reasoner signaled '{reasoner_plan.get('next_action')}'; stopping.")
                stop_reason = "reasoner_done"
                exit_reason = "reasoner_done"
                break

            # 3. Actioner translates plan into exactly one concrete tool call
            action = self.actioner.plan_action(
                task=task,
                reasoner_plan=reasoner_plan,
                previous_actions=previous_actions,
            )
            
            if not action:
                fallback = {"tool": reasoner_plan.get("next_action"), "parameters": reasoner_plan.get("parameters", {})}
                is_valid, _ = validate_action(fallback)
                if is_valid:
                    print("[Agent] Actioner failed; using Reasoner's plan directly as fallback.")
                    action = fallback
                else:
                    stop_reason = "actioner_failed"
                    exit_reason = "actioner_failed"
                    break

            is_valid, error = validate_action(action)
            if _is_cached_read(current_state, action):
                path = (action.get("parameters") or {}).get("path")

                print(
                    "[Agent] Rejecting redundant read_file for cached file: {}".format(
                        path
                    )
                )

                # Instead of returning an error, present the cached contents
                # as the result so the Reasoner sees the file content in the
                # previous_actions history and does not repeatedly request it.
                file_cache = current_state.get("file_cache", {})
                cached = file_cache.get(path)
                cached_text = ""
                if isinstance(cached, str):
                    cached_text = cached
                elif isinstance(cached, dict):
                    if cached.get("type") == "full":
                        cached_text = cached.get("content", "")
                    elif cached.get("type") == "chunked":
                        parts = []
                        for c in cached.get("chunks", []):
                            parts.append(c.get("content", ""))
                        cached_text = "\n\n".join(parts)

                previous_actions.append({
                    "iteration": iteration,
                    "reasoner_plan": reasoner_plan,
                    "action": action,
                    "result": {
                        "tool": "read_file",
                        "result": cached_text,
                    },
                })

                continue

            # 4. Loop detection: same tool+parameters repeated too often
            action_key = (action.get("tool"), _stable_repr(action.get("parameters", {})))
            if action_key == last_action_key:
                repeated_action_count += 1
            else:
                repeated_action_count = 0
            last_action_key = action_key

            if repeated_action_count >= MAX_REPEATED_ACTIONS:
                print("[Agent] Same action repeated too many times without progress; stopping.")
                stop_reason = "repeated_action"
                exit_reason = "repeated_action"
                break

            # NEW: detect A-B-A-B style oscillation, not just immediate repeats
            action_window.append(action_key)
            window = list(action_window)
            cycle_detected = False
            for period in (2, 3):
                if len(window) >= period * 2 and window[-period:] == window[-2 * period:-period]:
                    cycle_detected = True
                    break
            if cycle_detected:
                print("[Agent] Detected oscillating action cycle; stopping.")
                stop_reason = "repeated_action"
                exit_reason = "repeated_action"
                break

            # 5. Execute the concrete tool call (deterministic)
            try:
                result = self.actioner.execute(action)
            except Exception as e:
                result = {
                    "tool": action.get("tool", ""),
                    "error": str(e),
                }

            total_tool_calls += 1
            last_reasoner_plan = reasoner_plan
            last_action, last_result = action, result

            # 5a. Update the persistent file cache.
            _update_file_cache(current_state, action, result)


            ## 6. Update persistent state, including the file cache.
            _update_file_cache(current_state, action, result)

            current_state["last_plan"] = reasoner_plan
            current_state["last_action"] = action
            current_state["last_result"] = result

            # 7. Save complete history.
            previous_actions.append({
                "iteration": iteration,
                "reasoner_plan": reasoner_plan,
                "action": action,
                "result": result,
            })

            # 8. Check test results
            if action.get("tool") == "run_tests":
                last_test_iteration = iteration 
                test_passed = result.get("returncode") == 0
                current_state["last_test_result"] = {
                    "returncode": result.get("returncode"),
                    "stdout": result.get("result", ""),
                    "stderr": result.get("stderr", ""),
                }
                if test_passed:
                    print("[Agent] Tests passed.")
                    stop_reason = "tests_passed"
                    exit_reason = "tests_passed"
                    break

        # =========================================================
        # LOOP IS FINISHED. Everything below happens ONCE, after the loop.
        # =========================================================

        # Capture final patch (always after the loop, never inside it).
        final_patch = ""
        try:
            diff_result = self.actioner.execute({"tool": "get_git_diff", "parameters": {}})
            final_patch = diff_result.get("result", "") or ""
        except Exception:
            final_patch = ""

        # Determine final status (assigned exactly once, right before
        # constructing the result -- never referenced earlier).
        if test_passed is None:
            status = "timeout" if num_iterations >= self.max_iterations else "incomplete"
        elif test_passed:
            status = "passed"
        else:
            status = "failed"

        return AgentResult(
            instance_id="", repo_name="", base_commit="",
            num_iterations=num_iterations, total_tool_calls=total_tool_calls,
            test_passed=test_passed, final_patch=final_patch, status=status, exit_reason=exit_reason,
            stop_reason=stop_reason,
            last_reasoner_plan=last_reasoner_plan,
            last_action=last_action,
            last_result=last_result,
            history=previous_actions,
        )
    
    def _get_initial_file_listing(self, repo_path: str, max_files: int = 50) -> str:
        """Best-effort shallow listing of the repo so the Reasoner isn't
        guessing blind on its very first plan() call.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=repo_path, capture_output=True, text=True, timeout=10, check=False,
            )
            files = [f for f in result.stdout.splitlines() if f.strip()]
            if not files:
                raise RuntimeError("no git files")
            return "\n".join(files[:max_files])
        except Exception:
            # Fall back to a plain os.listdir if it's not a git repo /
            # git isn't available.
            import os
            try:
                return "\n".join(sorted(os.listdir(repo_path))[:max_files])
            except Exception:
                return "(directory listing unavailable)"

def _stable_repr(value) -> str:
    import json as _json
    try:
        return _json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def _update_file_cache(
    current_state: dict,
    action: dict,
    result: dict,
) -> None:
    """Update cached file contents based on a successfully executed action.

    Successful read_file actions populate the cache, optionally storing only
    the requested line range so the Reasoner can read long files incrementally.
    File-modifying actions invalidate the corresponding cached entry so the
    Reasoner cannot make decisions using stale file contents.
    """
    tool = action.get("tool")
    parameters = action.get("parameters") or {}
    path = parameters.get("path")

    if not path:
        return

    file_cache = current_state.setdefault("file_cache", {})

    # A successful read_file gives us authoritative file contents.
    if tool == "read_file":
        if "error" not in result:
            content = result.get("result", "")
            if isinstance(content, str):
                file_cache[path] = _merge_cache_entry(file_cache.get(path), content, parameters)

        return

    # Any tool that modifies an existing file invalidates the cached version.
    if tool in {
        "replace_in_file",
        "write_to_file",
    }:
        file_cache.pop(path, None)

def _merge_cache_entry(existing_entry: object, content: str, parameters: dict) -> dict:
    """Merge a newly read chunk into the file cache entry for a path."""
    if not isinstance(content, str):
        return existing_entry if isinstance(existing_entry, dict) else {}

    start_line = parameters.get("start_line")
    end_line = parameters.get("end_line")

    # A full-file read replaces any prior partial context for that path.
    if start_line in (None, "") and end_line in (None, ""):
        return {"type": "full", "content": content}

    if isinstance(existing_entry, str):
        existing_entry = {"type": "full", "content": existing_entry}

    if not isinstance(existing_entry, dict):
        existing_entry = {"type": "chunked", "chunks": []}

    if existing_entry.get("type") == "full":
        return existing_entry

    chunk = {
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
    }
    chunks = existing_entry.get("chunks", [])

    if not any(
        c.get("start_line") == chunk["start_line"] and c.get("end_line") == chunk["end_line"]
        for c in chunks
    ):
        chunks.append(chunk)

    return {"type": "chunked", "chunks": chunks}

def _is_cached_read(
    current_state: dict,
    action: dict,
) -> bool:
    """Return True if read_file requests content that is already cached."""
    if action.get("tool") != "read_file":
        return False

    parameters = action.get("parameters") or {}
    path = parameters.get("path")

    if not path:
        return False

    file_cache = current_state.get("file_cache", {})
    entry = file_cache.get(path)

    if not entry:
        return False

    if isinstance(entry, str):
        return True

    if not isinstance(entry, dict):
        return False

    if entry.get("type") == "full":
        return True

    start_line = parameters.get("start_line")
    end_line = parameters.get("end_line")
    if start_line in (None, "") and end_line in (None, ""):
        return True

    for chunk in entry.get("chunks", []):
        if chunk.get("start_line") == start_line and chunk.get("end_line") == end_line:
            return True

    return False
