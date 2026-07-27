"""Agent: orchestrates the reasoner and actioner to solve tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

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

    def __init__(self, reasoner, actioner, max_iterations: int = 50):
        self.reasoner = reasoner
        self.actioner = actioner
        self.max_iterations = max_iterations

    def solve(self, repo_path: str, task: str, test_command: str = "pytest --tb=short") -> AgentResult:
        """Run the agent on a single SWE-bench-style task."""
        # Point the actioner at this task's repo so file/command tools
        # operate on the right workspace instead of the ambient cwd.
        self.actioner.workspace_dir = repo_path

        previous_actions: list[dict] = []
        current_state: dict = {
            "repo_path": repo_path,
            "test_command": test_command,
        }

        num_iterations = 0
        total_tool_calls = 0
        test_passed: bool | None = None
        repeated_action_count = 0
        last_action_key = None

        stop_reason = "max_iterations"          # default if we exhaust the loop
        last_reasoner_plan: dict = {}
        last_action: dict = {}
        last_result: dict = {}

        for iteration in range(self.max_iterations):
            num_iterations = iteration + 1

            # If the last executed action repeated at least once, tell the
            # Reasoner explicitly to avoid it -- don't wait for the
            # 2-repeat _loop_warning threshold in reasoner.py, since by
            # then the model has already committed to the pattern once.
            avoid_action = None
            if repeated_action_count >= 1 and last_action_key is not None:
                avoid_action = last_action_key

            # 1. Reasoner decides what to do
            reasoner_plan = self.reasoner.plan(
                task=task,
                current_state=current_state,
                previous_actions=previous_actions,
                avoid_action=avoid_action,
            )

            if not reasoner_plan:
                print("[Agent] Reasoner failed to produce a plan.")
                break

            # 2. Stop conditions from the Reasoner
            if reasoner_plan.get("next_action") in STOP_ACTIONS:
                print(f"[Agent] Reasoner signaled '{reasoner_plan.get('next_action')}'; stopping.")
                break

            # 3. Actioner translates plan into exactly one concrete tool call
            action = self.actioner.plan_action(
                task=task,
                reasoner_plan=reasoner_plan,
                previous_actions=previous_actions,
            )

            if not action:
                print("[Agent] Actioner failed to produce a valid tool call.")
                break

            is_valid, error = validate_action(action)
            if not is_valid:
                print(f"[Agent] Actioner produced an invalid tool call, skipping execution: {error}")
                previous_actions.append({
                    "iteration": iteration,
                    "reasoner_plan": reasoner_plan,
                    "action": action,
                    "result": {"tool": action.get("tool", ""), "error": error},
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
                break

            # 5. Execute the concrete tool call (deterministic)
            try:
                result = self.actioner.execute(action)
            except Exception as e:
                result = {"tool": action.get("tool", ""), "error": str(e)}

            total_tool_calls += 1
            last_action, last_result = action, result


            # 6. Save complete history (real tool output, not just the name)
            previous_actions.append({
                "iteration": iteration,
                "reasoner_plan": reasoner_plan,
                "action": action,
                "result": result,
            })

            # 7. Update state
            current_state["repo_path"] = ...       # set once, early
            current_state["test_command"] = ...    # set once, early
            current_state["last_plan"] = reasoner_plan
            current_state["last_action"] = action
            current_state["last_result"] = result   # <-- the full file content lands HERE, last

            # 8. Check test results
            if action.get("tool") == "run_tests":
                test_passed = result.get("returncode") == 0
                current_state["last_test_result"] = {
                    "returncode": result.get("returncode"),
                    "stdout": result.get("result", ""),
                    "stderr": result.get("stderr", ""),
                }
                if test_passed:
                    print("[Agent] Tests passed.")
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
            test_passed=test_passed, final_patch=final_patch, status=status,
            stop_reason=stop_reason,
            last_reasoner_plan=last_reasoner_plan,
            last_action=last_action,
            last_result=last_result,
            history=previous_actions,
        )

def _stable_repr(value) -> str:
    import json as _json
    try:
        return _json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)
