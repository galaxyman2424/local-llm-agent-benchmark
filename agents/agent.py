"""Agent: orchestrates the reasoner and actioner to solve tasks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IterationResult:
    """Record of a single iteration (tool call + plan step)."""
    iteration: int
    tool_call: str  # "read", "write", "execute", etc.
    target: str      # file path or command
    success: bool
    output: str = ""


@dataclass
class AgentResult:
    """Final result of running the agent on one task."""
    instance_id: str
    repo_name: str
    base_commit: str
    num_iterations: int
    total_tool_calls: int
    test_passed: bool | None
    final_patch: str = ""
    status: str = "incomplete"  # passed, failed, timeout


class Agent:
    """Top-level agent that runs a task to completion.

    Loops over iterations: the reasoner proposes an action and the actioner
    carries it out. When no more useful actions can be produced (or the
    maximum number of iterations is reached), the loop ends and we record
    whatever test status we have at that point.
    """

    def __init__(self, reasoner, actioner, max_iterations: int = 50):
        self.reasoner = reasoner
        self.actioner = actioner
        self.max_iterations = max_iterations

    def solve(self, repo_path: str, task: str, test_command: str = "pytest --tb=short") -> AgentResult:
        """Run the agent on a single SWE-bench-style task.

        Loop, per ACTIONPLAN.md section 4.4:
            Reasoner analyzes task -> creates plan (next_action/parameters)
            -> Actioner executes that action
            -> if the action was a test run, check pass/fail
            -> on failure, the outcome is fed back to the Reasoner as part of
               previous_actions so it can propose a revised plan
            -> repeat until tests pass, the reasoner has nothing left to do,
               or max_iterations is reached.
        """
        start_time = time.time()

        # Point the actioner at this task's repo so file/command tools
        # operate on the right workspace instead of the ambient cwd.
        self.actioner.workspace_dir = repo_path

        iterations: list[IterationResult] = []
        previous_actions: list[dict] = []
        total_tool_calls = 0
        final_patch = ""
        test_passed: bool | None = None

        current_state: dict = {
            "repo_path": repo_path,
            "test_command": test_command,
        }

        for i in range(self.max_iterations):
            # 1. Reasoner produces the next action given task + state + history
            plan = self.reasoner.plan(task, current_state, previous_actions)
            if not plan:
                break

            tool_call = plan.get("next_action", "")
            parameters = plan.get("parameters", {})

            if tool_call in ("submit_solution", "done", ""):
                break

            # 2. Actioner executes that action
            action = {"tool": tool_call, "parameters": parameters}
            try:
                result = self.actioner.execute(action)
            except Exception as e:
                result = {"tool": tool_call, "error": str(e)}

            success = "error" not in result
            output = result.get("result", result.get("error", ""))
            if isinstance(output, (dict, list)):
                output = json.dumps(output)

            iterations.append(IterationResult(i, tool_call, json.dumps(parameters), success, str(output)))
            total_tool_calls += 1
            previous_actions.append({**action, "result": result})

            # Track test outcomes as they happen so we know when to stop.
            if tool_call == "run_tests":
                test_passed = result.get("returncode") == 0
                current_state["last_test_result"] = {
                    "returncode": result.get("returncode"),
                    "stdout": str(result.get("result", ""))[:2000],
                    "stderr": str(result.get("stderr", ""))[:2000],
                }
                if test_passed:
                    break

            current_state["last_action"] = tool_call
            current_state["last_output"] = str(output)[:2000]

        # Capture whatever diff the agent produced, regardless of outcome.
        try:
            diff_result = self.actioner.execute({"tool": "get_git_diff", "parameters": {}})
            final_patch = diff_result.get("result", "") or ""
        except Exception:
            final_patch = ""

        if test_passed is None:
            status = "incomplete"
        elif test_passed:
            status = "passed"
        else:
            status = "failed"

        result = AgentResult(
            instance_id="",       # filled by benchmark
            repo_name="",         # filled by benchmark
            base_commit="",       # filled by benchmark
            num_iterations=len(iterations),
            total_tool_calls=total_tool_calls,
            test_passed=test_passed,
            final_patch=final_patch,
            status=status,
        )

        return result
