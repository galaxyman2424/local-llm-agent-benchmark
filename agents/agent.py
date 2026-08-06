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

    def solve(
        self,
        repo_path,
        task,
        test_command="pytest --tb=short",
        python_bin: str | None = None,
        fail_to_pass_tests: list[str] | None = None,
        pass_to_pass_tests: list[str] | None = None,
    ) -> AgentResult:
        """Run the agent on a single SWE-bench-style task.

        Parameters
        ----------
        fail_to_pass_tests, pass_to_pass_tests
            The instance's gold FAIL_TO_PASS / PASS_TO_PASS test node ids,
            when known (e.g. from the SWE-bench Lite dataset). These are
            just test *names*, not the solution -- the gold ``patch`` stays
            hidden -- so there's no reason to withhold them from the agent.
            When provided, they become BOTH the concrete goal stated to the
            Reasoner (instead of a vague "make test_command pass") AND the
            authoritative exit condition: after every ``run_tests`` call,
            these specific node ids are re-checked directly (see
            ``swebench.utils.run_test_ids``), rather than trusting the
            return code of whatever ad hoc command the model happened to
            run. Without this, a model could run `pytest some_unrelated_
            test.py`, get returncode 0, and the loop would wrongly declare
            victory -- the exit signal would be completely disconnected
            from what the benchmark actually scores. When omitted (e.g. a
            synthetic/local task with no gold test lists), the loop falls
            back to the previous behavior of trusting the configured
            ``test_command``'s return code.
        """
        # Point the actioner at this task's repo so file/command tools
        # operate on the right workspace instead of the ambient cwd.
        self.actioner.workspace_dir = repo_path
        if python_bin:
            self.actioner.python_bin = python_bin

        fail_to_pass_tests = fail_to_pass_tests or []
        pass_to_pass_tests = pass_to_pass_tests or []

        previous_actions: list[dict] = []
        current_state: dict = {
            "repo_path": repo_path,
            "test_command": test_command,
            "fail_to_pass_tests": fail_to_pass_tests,
            "pass_to_pass_tests": pass_to_pass_tests,
            "target_tests_passing": None,
            "file_cache": {},
            "files_read": {}, 
        }

        # blindness on iteration 1.
        current_state["file_tree"] = self._get_initial_file_listing(repo_path)

        num_iterations = 0
        total_tool_calls = 0
        test_passed: bool | None = None
        repeated_action_count = 0
        last_action_key = None
        last_action_tuple: tuple[str, str] | None = None

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
            if repeated_action_count >= 1:
                stagnation_hint = (
                    f"NOTE: your last action was repeated {repeated_action_count + 1} time(s) "
                    "in a row without making progress. Try a genuinely different approach."
                )

            # 1. Reasoner decides what to do
            reasoner_plan = self.reasoner.plan(
                task=task,
                current_state=current_state,
                previous_actions=previous_actions,
                avoid_action=last_action_tuple if repeated_action_count >= 1 else None,
                stagnation_hint=stagnation_hint,
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

            # 4. Loop detection: same tool+parameters repeated too often.
            #
            # IMPORTANT: this must run BEFORE the cached-read/cached-listing
            # short-circuits below. Those short-circuits `continue` the loop
            # for read_file/list_directory calls that hit the cache -- if
            # loop detection ran after them, a model stuck alternating
            # between two already-cached reads would never update
            # action_key/repeated_action_count/action_window at all, since
            # every one of those calls would exit via `continue` first. That
            # made cache-hit loops (e.g. re-reading the same file range over
            # and over) completely invisible to this detector, letting them
            # burn the full iteration budget instead of being caught.
            action_key = (action.get("tool"), _stable_repr(action.get("parameters", {})))
            if action_key == last_action_key:
                repeated_action_count += 1
            else:
                repeated_action_count = 0
            last_action_key = action_key
            last_action_tuple = action_key

            if repeated_action_count >= MAX_REPEATED_ACTIONS:
                print("[Agent] Same action repeated too many times without progress; stopping.")
                stop_reason = "repeated_action"
                exit_reason = "repeated_action"
                break

            # Detect A-B-A-B style oscillation, not just immediate repeats.
            #
            # NOTE: period=1 is deliberately excluded here. window[-1:] ==
            # window[-2:-1] for period=1 is exactly the same condition as
            # action_key == last_action_key above -- but it used to run in
            # this same loop over periods (1, 2, 3), which meant it fired on
            # the very first repeat (repeated_action_count == 1) instead of
            # waiting for MAX_REPEATED_ACTIONS in a row. That made the
            # MAX_REPEATED_ACTIONS=4 threshold unreachable in practice: any
            # single exact repeat killed the run immediately via this
            # oscillation check before the counter above could ever reach 4.
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

            if _is_cached_listing(current_state, action):
                listing_path = (action.get("parameters") or {}).get("path") or "."

                print(
                    "[Agent] Rejecting redundant list_directory for cached path: {}".format(
                        listing_path
                    )
                )

                # Same pattern as the cached-read short-circuit above: feed
                # back the cached listing as the result rather than an
                # error, so the Reasoner still sees the directory contents
                # in its history and isn't left guessing why the call was
                # skipped -- it just shouldn't need to list the same,
                # unchanged directory a second time.
                listing_cache = current_state.get("listing_cache", {})
                cached_listing = listing_cache.get(listing_path, [])

                previous_actions.append({
                    "iteration": iteration,
                    "reasoner_plan": reasoner_plan,
                    "action": action,
                    "result": {
                        "tool": "list_directory",
                        "result": cached_listing,
                    },
                })

                continue

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

            # Update the persistent files manifest so the Reasoner always knows what's
            # been read AND what's been modified since, independent of the rolling
            # last-5-actions window (which truncates and can drop this by later
            # iterations). A file read at iteration 2 but modified at iteration 5
            # means the Reasoner's mental model of its exact content/line numbers
            # from that earlier read is now stale.
            WRITE_TOOLS = {"write_to_file", "replace_in_file", "edit_file", "replace_lines", "delete_file"}

            if "error" not in result:
                params = action.get("parameters") or {}
                file_path = params.get("path", "")
                tool = action.get("tool")

                if file_path and (tool == "read_file" or tool in WRITE_TOOLS):
                    entry = current_state.setdefault("files_read", {}).setdefault(file_path, {})

                    if tool == "read_file":
                        total_lines = result.get("total_lines")
                        start = params.get("start_line")
                        end = params.get("end_line")
                        if start or end:
                            range_str = f"{start or 1}-{end or total_lines or '?'}"
                        elif total_lines:
                            range_str = f"1-{total_lines}"
                        else:
                            range_str = "full file"
                        entry["lines"] = total_lines
                        entry["last_read_iteration"] = iteration + 1
                        entry["last_range_read"] = range_str
                        entry.pop("stale_since_last_read", None)  # fresh read clears staleness

                    elif tool == "delete_file":
                        entry["deleted_at_iteration"] = iteration + 1
                        entry.pop("last_modified_iteration", None)

                    else:  # write_to_file, replace_in_file, edit_file, replace_lines
                        entry["last_modified_iteration"] = iteration + 1
                        entry["last_modified_tool"] = tool
                        entry.pop("deleted_at_iteration", None)
                        entry["stale_since_last_read"] = "last_read_iteration" in entry

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

            # Update the persistent files-read manifest so the Reasoner always knows
            # what's already been inspected, independent of the rolling last-5-actions
            # window (which truncates and can drop this entirely by later iterations).
            if action.get("tool") == "read_file" and "error" not in result:
                file_path = (action.get("parameters") or {}).get("path", "")
                if file_path:
                    total_lines = result.get("total_lines")
                    read_params = action.get("parameters") or {}
                    start = read_params.get("start_line")
                    end = read_params.get("end_line")
                    if start or end:
                        range_str = f"{start or 1}-{end or total_lines or '?'}"
                    elif total_lines:
                        range_str = f"1-{total_lines}"
                    else:
                        range_str = "full file"
                    current_state.setdefault("files_read", {})[file_path] = {
                        "lines": total_lines,
                        "last_read_iteration": iteration + 1,
                        "last_range_read": range_str,
                    }

            current_state["last_plan"] = reasoner_plan
            current_state["last_action"] = action
            current_state["last_result"] = result

            # 8. Check test results
            if action.get("tool") == "run_tests":
                last_test_iteration = iteration
                current_state["last_test_result"] = {
                    "returncode": result.get("returncode"),
                    "stdout": result.get("result", ""),
                    "stderr": result.get("stderr", ""),
                }

                if fail_to_pass_tests:
                    # Authoritative check: ignore whether the arbitrary
                    # command the model chose to run happened to return 0
                    # (it may not even have touched the tests that matter)
                    # and instead directly re-run the instance's own
                    # FAIL_TO_PASS/PASS_TO_PASS node ids -- the same check
                    # used at evaluation time, just run live so the loop's
                    # own "am I done" signal is the same thing that will
                    # actually be scored.
                    target_status = _check_target_tests(
                        repo_path=self.actioner.workspace_dir,
                        python_bin=self.actioner.python_bin or "python",
                        fail_to_pass_tests=fail_to_pass_tests,
                        pass_to_pass_tests=pass_to_pass_tests,
                    )
                    current_state["target_test_results"] = target_status
                    current_state["target_tests_passing"] = target_status["all_passing"]
                    test_passed = target_status["all_passing"]
                else:
                    # No gold test list for this task (e.g. a synthetic/
                    # local instance) -- fall back to trusting the
                    # configured test_command's return code, as before.
                    test_passed = result.get("returncode") == 0

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

def _check_target_tests(
    *,
    repo_path: str,
    python_bin: str,
    fail_to_pass_tests: list[str],
    pass_to_pass_tests: list[str],
) -> dict:
    """Run the instance's actual FAIL_TO_PASS/PASS_TO_PASS node ids live and
    report whether every one currently passes.

    This reuses ``swebench.utils.run_test_ids`` -- the exact same
    per-node-id runner used by offline evaluation (``evaluate_fail_to_pass``)
    -- so the loop's own "should I stop" signal matches what the benchmark
    will actually score, rather than an independent, looser approximation.
    Imported locally (like the rest of this codebase's cross-package
    imports) to avoid a module-level dependency between the `agents` and
    `swebench` packages.
    """
    try:
        from swebench.utils import run_test_ids
    except ImportError:
        # swebench.utils isn't importable in this environment (e.g. a unit
        # test or non-SWE-bench caller) -- don't crash the loop, just report
        # "not confirmed" so the fallback returncode-based path never
        # silently activates by accident.
        return {
            "fail_to_pass_results": {},
            "pass_to_pass_results": {},
            "all_passing": False,
            "error": "swebench.utils.run_test_ids unavailable",
        }

    fail_to_pass_results = run_test_ids(repo_path, fail_to_pass_tests, python_bin=python_bin)
    pass_to_pass_results = run_test_ids(repo_path, pass_to_pass_tests, python_bin=python_bin)

    all_passing = (
        bool(fail_to_pass_tests)
        and all(v is True for v in fail_to_pass_results.values())
        and all(v is True for v in pass_to_pass_results.values())
    )

    return {
        "fail_to_pass_results": fail_to_pass_results,
        "pass_to_pass_results": pass_to_pass_results,
        "all_passing": all_passing,
    }


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

    # list_directory's `path` is optional (defaults to the workspace root),
    # so it must be handled before the `if not path: return` guard below --
    # otherwise every list_directory call with no explicit path (the common
    # case) would silently skip caching entirely.
    if tool == "list_directory":
        if "error" not in result:
            listing_cache = current_state.setdefault("listing_cache", {})
            listing_path = parameters.get("path") or "."
            listing_cache[listing_path] = result.get("result", [])
        return

    # write_to_file/delete_file can add or remove entries from a directory
    # listing anywhere in the tree -- rather than tracking exactly which
    # directory each path belongs to, just invalidate every cached listing
    # so a later list_directory reflects the change instead of silently
    # returning a stale snapshot from before the write/delete. Checked here,
    # before the path guard, since these tools always have a path but the
    # listing_cache itself isn't keyed by that path.
    if tool in {"write_to_file", "delete_file"} and "error" not in result:
        current_state["listing_cache"] = {}

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

def _is_cached_listing(
    current_state: dict,
    action: dict,
) -> bool:
    """Return True if list_directory requests a path already cached.

    Mirrors ``_is_cached_read`` for the same reason: a small local model
    frequently re-lists an unchanged directory (e.g. the repo root) several
    times in a row while orienting itself, and unlike read_file this wasn't
    being deduplicated at all before, so it slipped past the file-cache
    short-circuit and only got caught later (if at all) by the loop's
    repeated-action detection.
    """
    if action.get("tool") != "list_directory":
        return False

    parameters = action.get("parameters") or {}
    listing_path = parameters.get("path") or "."

    listing_cache = current_state.get("listing_cache", {})
    return listing_path in listing_cache