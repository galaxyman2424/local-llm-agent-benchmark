"""Run a single SWE-bench Lite instance end-to-end.

Orchestrates the full lifecycle: load instance → set up repo → initialize
agent → provide problem statement → allow agent to modify → evaluate → save.

Example usage::

    python swebench/run_instance.py \\
        --instance django__django-12345 \\
        --config configs/qwen_ornith.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# See experiments/run_experiment.py for why this is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class RunResult:
    """Structured result from running a single instance."""
    instance_id: str = ""
    repo_name: str = ""
    base_commit: str = ""
    agent_config: dict = field(default_factory=dict)
    reasoner_model: str = ""
    actioner_model: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    runtime: float = 0.0
    num_iterations: int = 0
    total_tool_calls: int = 0
    test_results: dict = field(default_factory=dict)
    final_patch: str = ""
    status: str = "incomplete"
    fail_to_pass_count: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_count: int = 0
    pass_to_pass_total: int = 0
    fail_to_pass_results: dict = field(default_factory=dict)
    pass_to_pass_results: dict = field(default_factory=dict)


def run_instance(
    instance_id: str,
    config_path: str | None = None,
    seed_repo_dir: str = "seed_repos",
    results_dir: str = "results",
) -> RunResult:
    """Run one SWE-bench Lite instance through the full pipeline.

    Parameters
    ----------
    instance_id :
        The unique identifier (e.g., ``django__django-12345``).
    config_path :
        Optional path to a YAML config file that overrides defaults for
        reasoner/actioner models, max iterations, etc.
    seed_repo_dir :
        Directory where cloned repos are stored.

    Returns
    -------
    RunResult
        The structured result from this run.
    """
    # 1. Load configuration (use defaults if no config file provided)
    agent_config = load_config(config_path)

    # 2. Load the instance from the cached dataset
    from swebench.utils import load_instance, create_workspace, reset_repository, get_git_diff

    instance = load_instance(instance_id)
    repo_name = instance.get("repo", "unknown")
    base_commit = instance.get("base_commit", "")
    task_description = instance.get("task_description", "")

    # 3. Set up repository at the required base commit
    from swebench.utils import find_repository, checkout_commit

    repo_path = find_repository(repo_name, seed_repo_dir)
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repository {repo_name} not found at {repo_path}. "
            "Run download_dataset.py and setup_repos.py first."
        )

    # Ensure workspace is clean before starting
    reset_repository(repo_path)
    checkout_commit(repo_path, base_commit)

    # 4. Create an isolated workspace for this instance
    from swebench.utils import create_workspace as _create_ws
    workspace_dir = _create_ws(repo_path, instance_id, base_commit=base_commit)

    # Install the repo's dependencies into a venv shared across this repo's
    # instances (created once, reused), but editable-install the package
    # from workspace_dir specifically so tests import this instance's own
    # edited files, not a stale copy from the shared repo_path.
    from swebench.utils import ensure_repo_environment
    python_bin = ensure_repo_environment(repo_path, install_path=workspace_dir)

    # 5. Initialize the agent and give it the problem statement
    from agents import Reasoner, Actioner, Agent

    reasoner_model = agent_config.get("reasoner_model") or agent_config.get("reasoner", {}).get("model", "qwen2.5:7b")
    actioner_model = agent_config.get("actioner_model") or agent_config.get("actioner", {}).get("model", reasoner_model)
    max_iterations = agent_config.get("max_iterations") or agent_config.get("agent", {}).get("max_iterations", 50)
    test_cmd = instance.get("test_command", "pytest")

    reasoner_timeout = agent_config.get("timeout") or agent_config.get("agent", {}).get("timeout", 120.0)
    reasoner = Reasoner(model_id=reasoner_model, timeout_seconds=reasoner_timeout)
    actioner = Actioner(model_id=actioner_model, workspace_dir=str(workspace_dir))
    agent = Agent(reasoner, actioner, max_iterations=max_iterations)

    start_time = time.time()

    result = RunResult(
        instance_id=instance_id,
        repo_name=repo_name,
        base_commit=base_commit,
        agent_config=agent_config,
        reasoner_model=reasoner_model,
        actioner_model=actioner_model,
        start_time=start_time,
    )

    # 6. Let the agent modify the repository in the isolated workspace
    agent_result = agent.solve(str(workspace_dir), task_description or instance.get("problem_statement", ""), test_command=test_cmd)

    result.num_iterations = agent_result.num_iterations
    result.total_tool_calls = agent_result.total_tool_calls

    # 7. Capture the final patch produced by the agent
    result.final_patch = agent_result.final_patch or get_git_diff(workspace_dir)

    # 8. Run tests on the patched repo (independent confirmation of pass/fail)
    proc = subprocess.run(
        [python_bin, "-m", "pytest", "--tb=short"],
        cwd=str(workspace_dir),
        capture_output=True, text=True, timeout=60,
    )
    result.test_results = {
        "command": f"{python_bin} -m pytest",
        "returncode": proc.returncode,
        "stdout": proc.stdout[:500],
        "stderr": proc.stderr[:500],
    }

    # 9. Determine status: official FAIL_TO_PASS/PASS_TO_PASS evaluation
    if instance.get("FAIL_TO_PASS") or instance.get("PASS_TO_PASS"):
        from swebench.utils import evaluate_fail_to_pass
        f2p = evaluate_fail_to_pass(workspace_dir, instance, timeout=60, python_bin=python_bin)
        result.status = f2p["status"]
        result.fail_to_pass_count = f2p["fail_to_pass_count"]
        result.fail_to_pass_total = f2p["fail_to_pass_total"]
        result.pass_to_pass_count = f2p["pass_to_pass_count"]
        result.pass_to_pass_total = f2p["pass_to_pass_total"]
        result.fail_to_pass_results = f2p["fail_to_pass_results"]
        result.pass_to_pass_results = f2p["pass_to_pass_results"]
    else:
        result.status = _evaluate(result)

    result.end_time = time.time()
    result.runtime = result.end_time - result.start_time

    return result


def _evaluate(result: RunResult) -> str:
    """Fallback evaluation for instances with no FAIL_TO_PASS/PASS_TO_PASS
    gold test lists -- just check whether the generic test_command run
    passed.
    """
    if result.test_results.get("returncode") != 0:
        return "test_failure"
    return "resolved"


def load_config(config_path: str | None = None) -> dict:
    """Load agent configuration from YAML or use defaults."""
    if config_path is None:
        return {
            "reasoner_model": "qwen2.5:7b",
            "actioner_model": "qwen2.5:7b",
            "max_iterations": 50,
        }

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single SWE-bench instance")
    parser.add_argument("--instance", required=True, help="Instance ID (e.g., django__django-12345)")
    parser.add_argument(
        "--config", default=None, help="Path to YAML config file"
    )

    args = parser.parse_args()
    result = run_instance(args.instance, config_path=args.config)

    print(f"\n[run_instance] Done. Status: {result.status}")
    print(f"[run_instance] Runtime: {result.runtime:.2f}s")
    print(f"[run_instance] Test results: {json.dumps(result.test_results)}")
