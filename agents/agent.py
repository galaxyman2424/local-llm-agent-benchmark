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
        total_tool_calls = 0
        test_passed: bool | None = None

        current_state: dict = {
            "repo_path": repo_path,
            "test_command": test_command,
        }

        previous_actions = []
        test_passed = None
        final_patch = ""

        # Main agent loop
        for i in range(self.max_iterations):

            # 1. Reasoner creates plan
            reasoner_plan = self.reasoner.plan(
                task,
                current_state,
                previous_actions,
            )

            if not reasoner_plan:
                print("[Agent] Reasoner failed to produce a plan.")
                break

            # 2. Actioner creates concrete action
            action = self.actioner.plan_action(
                task=task,
                reasoner_plan=reasoner_plan,
                previous_actions=previous_actions,
            )

            if not action:
                print("[Agent] Actioner failed to produce a tool call.")
                break

            # 3. Execute action
            try:
                tool_result = self.actioner.execute(action)
            except Exception as e:
                tool_result = {
                    "tool": action.get("tool", ""),
                    "error": str(e),
                }

            # 4. Save history
            previous_actions.append({
                "reasoner_plan": reasoner_plan,
                "action": action,
                "result": tool_result,
            })

            # 5. Update state
            current_state["last_action"] = action
            current_state["last_result"] = tool_result

            # 6. Check tests
            if action.get("tool") == "run_tests":
                test_passed = tool_result.get("returncode") == 0

                if test_passed:
                    print("[Agent] Tests passed.")
                    break
        # =========================================================
        # LOOP IS FINISHED
        # Everything below happens ONCE, after the loop
        # =========================================================

        # Capture final patch
        try:
            diff_result = self.actioner.execute({
                "tool": "get_git_diff",
                "parameters": {},
            })
            final_patch = diff_result.get("result", "") or ""
        except Exception:
            final_patch = ""

        # Determine final status
        if test_passed is None:
            status = "incomplete"
        elif test_passed:
            status = "passed"
        else:
            status = "failed"

        # Build final AgentResult
        result = AgentResult(
            instance_id="",
            repo_name="",
            base_commit="",
            num_iterations=len(previous_actions),
            total_tool_calls=len(previous_actions),
            test_passed=test_passed,
            final_patch=final_patch,
            status=status,
        )

        return result