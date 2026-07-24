"""Evaluate whether an agent successfully solved a SWE-bench instance.

Uses the official SWE-bench evaluation methodology: capture the final patch,
run tests, and determine resolution status. Distinguishes between Resolved,
Not Resolved, Test Failure, Patch Failure, Timeout, Agent Error, and
Environment Error. Tracks FAIL_TO_PASS and PASS_TO_PASS counts.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# See experiments/run_experiment.py for why this is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class EvalResult:
    """Structured evaluation result for a single instance."""
    instance_id: str = ""
    repo_name: str = ""
    base_commit: str = ""
    status: str = "incomplete"
    final_patch: str = ""
    test_results: dict = field(default_factory=dict)
    fail_to_pass_count: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_count: int = 0
    pass_to_pass_total: int = 0
    fail_to_pass_results: dict = field(default_factory=dict)
    pass_to_pass_results: dict = field(default_factory=dict)
    evaluation_time: float = 0.0


def evaluate_instance(
    instance_id: str,
    repo_path: Path | None = None,
    dataset_path: str | None = None,
) -> EvalResult:
    """Evaluate a single SWE-bench instance after agent execution.

    Parameters
    ----------
    instance_id :
        The unique identifier (e.g., ``django__django-12345``).
    repo_path :
        Path to the repository where the agent made changes. If None, loads
        from the dataset.
    dataset_path :
        Optional path to the cached dataset for loading instance metadata.

    Returns
    -------
    EvalResult
        The structured evaluation result.
    """
    from swebench.utils import load_instance, get_git_diff, evaluate_fail_to_pass

    # Load instance metadata whenever we can, since we need it for the
    # test_command regardless of whether repo_path was already supplied.
    instance: dict = {}
    try:
        instance = load_instance(instance_id, dataset_path)
    except (FileNotFoundError, KeyError) as e:
        print(f"[evaluate] Could not load instance metadata: {e}")

    if repo_path is None:
        repo_name = instance.get("repo", "unknown")
        base_commit = instance.get("base_commit", "")
        repo_path = Path(f"seed_repos/{repo_name}")
    else:
        repo_name = instance.get("repo") or (
            str(repo_path.parent.name) if repo_path.exists() else "unknown"
        )
        base_commit = instance.get("base_commit", "")

    # Capture the final patch (agent's last git diff)
    start_time = time.time()
    try:
        final_patch = get_git_diff(repo_path)
    except Exception as e:
        final_patch = f"Agent Error: {str(e)}"

    result = EvalResult(
        instance_id=instance_id,
        repo_name=repo_name,
        base_commit=base_commit,
        final_patch=final_patch,
    )

    if "Agent Error" in final_patch:
        result.status = "agent_error"
        result.evaluation_time = time.time() - start_time
        return result

    if not Path(repo_path).exists():
        result.status = "environment_error"
        result.evaluation_time = time.time() - start_time
        return result

    # Quick sanity-check test_command run, kept for debugging visibility.
    # This is NOT the authoritative pass/fail signal -- see below.
    test_cmd = instance.get("test_command", "pytest")
    import subprocess
    try:
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
    except subprocess.TimeoutExpired:
        result.test_results = {"command": test_cmd, "returncode": None, "error": "timeout"}

    # Authoritative status: does the patch make FAIL_TO_PASS tests pass
    # while keeping PASS_TO_PASS tests passing (the official SWE-bench
    # resolution criterion), when the dataset actually provides those
    # lists for this instance.
    if instance.get("FAIL_TO_PASS") or instance.get("PASS_TO_PASS"):
        f2p = evaluate_fail_to_pass(repo_path, instance, timeout=60)
        result.status = f2p["status"]
        result.fail_to_pass_count = f2p["fail_to_pass_count"]
        result.fail_to_pass_total = f2p["fail_to_pass_total"]
        result.pass_to_pass_count = f2p["pass_to_pass_count"]
        result.pass_to_pass_total = f2p["pass_to_pass_total"]
        result.fail_to_pass_results = f2p["fail_to_pass_results"]
        result.pass_to_pass_results = f2p["pass_to_pass_results"]
    else:
        # No gold test lists available for this instance (e.g. a synthetic
        # or hand-authored local task) -- fall back to the simple
        # return-code check from the sanity-check run above.
        returncode = result.test_results.get("returncode")
        result.status = "resolved" if returncode == 0 else "test_failure"

    result.evaluation_time = time.time() - start_time
    return result


def evaluate_batch(
    instances: list[dict],
    repo_paths: dict[str, Path] | None = None,
) -> dict:
    """Evaluate multiple SWE-bench instances.

    Parameters
    ----------
    instances :
        List of task instance dicts from the dataset.
    repo_paths :
        Optional mapping of instance_id → repo_path for pre-loaded repos.

    Returns
    -------
    dict
        ``{"results": [EvalResult, ...], "summary": {...}}``
    """
    results = []
    for inst in instances:
        iid = inst.get("instance_id", "unknown")
        result = evaluate_instance(
            iid,
            repo_path=repo_paths.get(iid) if repo_paths else None,
            dataset_path=None,
        )
        results.append(result)

    # Compute aggregate stats, broken down by every status the evaluator
    # can actually produce (not just resolved/failed).
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    resolved = by_status.get("resolved", 0)
    fail_to_pass = sum(r.fail_to_pass_count for r in results)
    pass_to_pass = sum(r.pass_to_pass_count for r in results)

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "resolved": resolved,
            "by_status": by_status,
            "fail_to_pass_count": fail_to_pass,
            "pass_to_pass_count": pass_to_pass,
        },
    }


if __name__ == "__main__":
    # Simple CLI for evaluating a single instance
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SWE-bench instances")
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()

    result = evaluate_instance(args.instance)
    print(f"\n[evaluate] Instance: {result.instance_id}")
    print(f"[evaluate] Status: {result.status}")
    print(f"[evaluate] Time: {result.evaluation_time:.2f}s")
