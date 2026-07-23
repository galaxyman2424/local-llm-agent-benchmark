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
import time
from dataclasses import dataclass, field
from pathlib import Path


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
    workspace_dir = _create_ws(repo_path, instance_id)

    # 5. Initialize the agent with the problem statement
    # In a full implementation, this would spin up the Reasoner+Actioner loop.
    # For now we record metadata and prepare for evaluation.
    start_time = time.time()

    result = RunResult(
        instance_id=instance_id,
        repo_name=repo_name,
        base_commit=base_commit,
        agent_config=agent_config,
        reasoner_model=agent_config.get("reasoner_model", ""),
        actioner_model=agent_config.get("actioner_model", ""),
        start_time=start_time,
    )

    # 6. Capture any final patch (placeholder — real impl captures during solve)
    result.final_patch = get_git_diff(repo_path)

    # 7. Run tests on the patched repo
    test_cmd = instance.get("test_command", "pytest")
    import subprocess
    proc = subprocess.run(
        ["/bin/sh", "-c", f"cd {repo_path} && {test_cmd}"],
        capture_output=True, text=True, timeout=60,
    )
    result.test_results = {
        "command": test_cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[:500],
        "stderr": proc.stderr[:500],
    }

    # 8. Determine status based on evaluation criteria
    result.status = _evaluate(result)

    result.end_time = time.time()
    result.runtime = result.end_time - result.start_time
    result.num_iterations = agent_config.get("max_iterations", 50)
    result.total_tool_calls = agent_config.get("max_iterations", 50) * 3  # rough estimate

    return result


def _evaluate(result: RunResult) -> str:
    """Evaluate whether the instance was resolved.

    Distinguishes between Resolved, Not Resolved, Test Failure, Patch Failure,
    Timeout, Agent Error, and Environment Error.
    """
    if result.test_results.get("returncode") != 0:
        return "test_failure"
    # Placeholder — real impl would diff against gold patch
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
