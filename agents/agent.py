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

    def solve(self, repo_path: str, task: str) -> AgentResult | None:
        """Run the agent on a single SWE-bench-style task."""
        start_time = time.time()

        iterations: list[IterationResult] = []
        total_tool_calls = 0
        final_patch = ""
        test_passed = None

        # Initial context from the repo state and task description
        context = {
            "repo_path": repo_path,
            "task": task,
            "base_commit": "",   # filled in when benchmark sets it up
            "test_command": "pytest",
            "current_state": "",
        }

        for i in range(self.max_iterations):
            # 1. Reasoner produces a plan step based on current state
            plan = self.reasoner.plan(context)
            if not plan:
                break

            step = plan[0]
            tool_call = step.action_type
            target = step.target

            # 2. Actioner executes the step and records outcome
            success = False
            output = ""
            try:
                if tool_call == "read":
                    content = self.actioner.read_file(target)
                    output = content or "(file not found)"
                    success = True
                elif tool_call == "write":
                    # For write operations, we'd need the full patch/content
                    # This is a simplified version; real impl would parse the plan
                    result = self.actioner.execute_command(f"git apply {target}")
                    output = result.stdout if result else ""
                    success = True
                elif tool_call == "execute":
                    proc = self.actioner.execute_command(target)
                    output = proc.stdout + proc.stderr
                    success = proc.returncode == 0
                else:
                    # Generic fallback
                    proc = self.actioner.execute_command(f"echo {target}")
                    output = proc.stdout if proc else ""
                    success = True

            except Exception as e:
                output = str(e)
                success = False

            iterations.append(IterationResult(i, tool_call, target, success, output))
            total_tool_calls += 1

            # Update context with new state
            context["current_state"] = json.dumps({tool_call: output}, indent=2)

        elapsed = time.time() - start_time

        result = AgentResult(
            instance_id="",       # filled by benchmark
            repo_name="",         # filled by benchmark
            base_commit="",       # filled by benchmark
            num_iterations=len(iterations),
            total_tool_calls=total_tool_calls,
            test_passed=test_passed,
            final_patch=final_patch,
            status="incomplete" if test_passed is None else "passed" if test_passed else "failed",
        )

        # Store result for the benchmark to collect later
        return result
