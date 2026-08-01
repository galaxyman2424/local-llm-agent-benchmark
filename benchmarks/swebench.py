"""SWE-bench benchmark: manages dataset, repo setup, evaluation, and result storage."""

from __future__ import annotations

import json
import time
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InstanceRecord:
    """Raw record for a single SWE-bench instance — never overwritten."""
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

    stop_reason: str = ""
    last_reasoner_plan: dict = field(default_factory=dict)
    last_action: dict = field(default_factory=dict)
    last_result: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    exit_reason: str = "" 


class SWEbenchBenchmark:
    """Manages the full benchmark lifecycle.

    Responsibilities (per section 3.1):
      - Dataset loading and task selection
      - Repository setup from base commit
      - Agent execution orchestration
      - Test execution after agent runs out of iterations
      - Evaluation against expected patches
      - Result storage in results/raw/
    """

    def __init__(self, seed_repo_dir: str = "seed_repos", results_dir: str = "results"):
        self.seed_repo_dir = Path(seed_repo_dir)
        self.results_dir = Path(results_dir)
        self.raw_results_dir = self.results_dir / "raw"
        self.processed_results_dir = self.results_dir / "processed"

    def list_tasks(self, dataset_path: str | None = None) -> list[dict]:
        """Load tasks from the SWE-bench lite JSONL file."""
        if dataset_path is None:
            dataset_path = self.seed_repo_dir / "swe_bench_lite" / "instances.jsonl"
        path = Path(dataset_path)
        if not path.exists():
            return []
        tasks = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    def setup_repo(self, repo_name: str, base_commit: str) -> Path:
        """Clone (if needed) and checkout a seed repo at the specified commit."""
        import subprocess

        repos_dir = self.seed_repo_dir / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)
        # repo_name is typically "owner/repo"; use a flat dir name so nested
        # slashes don't get misread as extra path segments.
        repo_dir_name = repo_name.replace("/", "__")
        repo_path = repos_dir / repo_dir_name

        if not repo_path.exists():
            clone_url = f"https://github.com/{repo_name}.git"
            print(f"[SWEbenchBenchmark] Cloning {clone_url} -> {repo_path}")
            subprocess.run(
                ["git", "clone", clone_url, str(repo_path)],
                capture_output=True, text=True, check=False,
            )

        if base_commit and repo_path.exists():
            subprocess.run(
                ["git", "checkout", "-f", base_commit],
                cwd=str(repo_path), capture_output=True, text=True, check=False,
            )

        return repo_path

    def run_instance(
        self,
        task: dict,
        agent,
        reasoner_model: str | None = None,
        actioner_model: str | None = None,
    ) -> InstanceRecord:
        """Run one SWE-bench instance through the full pipeline."""
        repo_name = task.get("repo", "unknown")
        base_commit = task.get("base_commit", "")

        record = InstanceRecord(
            instance_id=task.get("instance_id", ""),
            repo_name=repo_name,
            base_commit=base_commit,
            reasoner_model=reasoner_model or "qwen2.5:7b",
            actioner_model=actioner_model or "qwen2.5:7b",
        )

        start = time.time()

        # 1. Set up repo at base commit, and ensure it has its own
        #    dependencies installed (a bare git checkout has none of them --
        #    every test would otherwise fail immediately with import errors
        #    regardless of what the agent did).
        record.start_time = start
        repo_path = self.setup_repo(repo_name, base_commit)
        from swebench.utils import ensure_repo_environment
        python_bin, install_ok = ensure_repo_environment(repo_path)

        task_text = task.get("task_description") or task.get("problem_statement", "")
        test_cmd = task.get("test_command", "pytest")
        
        agent_result = agent.solve(str(repo_path), task_text, test_command=test_cmd)

        record.exit_reason = getattr(agent_result, "exit_reason", "") or ""
        record.stop_reason = getattr(agent_result, "stop_reason", "") or ""
        record.last_action = getattr(agent_result, "last_action", {}) or {}
        record.last_result = getattr(agent_result, "last_result", {}) or {}
        record.last_reasoner_plan = getattr(agent_result, "last_reasoner_plan", {}) or {}
        record.history = getattr(agent_result, "history", []) or []
        record.num_iterations = getattr(agent_result, "num_iterations", 0) or 0
        record.total_tool_calls = getattr(agent_result, "total_tool_calls", 0) or 0
        record.final_patch = getattr(agent_result, "final_patch", "") or ""
        
        if not install_ok:
            # Environment setup genuinely failed -- running tests would
            # just fail for everything regardless of the agent, wasting a
            # lot of time across many instances. Mark it distinctly rather
            # than letting it masquerade as "the agent didn't fix the bug".
            record.status = "environment_error"
            record.test_results = {"error": "repo dependency installation failed; see .swebench_venv/pip_install.log"}
            record.end_time = time.time()
            record.runtime = record.end_time - record.start_time
            return record

        # 2. Run tests on the patched repo, using the repo's own venv
        proc = __import__("subprocess").run(
            [python_bin, "-m", "pytest", "--tb=short"],
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=60,
        )
        record.test_results = {
            "command": f"{python_bin} -m pytest",
            "returncode": proc.returncode,
            "stdout": proc.stdout[:500],
            "stderr": proc.stderr[:500],
        }

        # 3. Official evaluation: does the patch make FAIL_TO_PASS tests
        #    pass while keeping PASS_TO_PASS tests passing?
        if task.get("FAIL_TO_PASS") or task.get("PASS_TO_PASS"):
            from swebench.utils import evaluate_fail_to_pass
            f2p = evaluate_fail_to_pass(repo_path, task, timeout=60, python_bin=python_bin)
            record.status = f2p["status"]
            record.fail_to_pass_count = f2p["fail_to_pass_count"]
            record.fail_to_pass_total = f2p["fail_to_pass_total"]
            record.pass_to_pass_count = f2p["pass_to_pass_count"]
            record.pass_to_pass_total = f2p["pass_to_pass_total"]
            record.fail_to_pass_results = f2p["fail_to_pass_results"]
            record.pass_to_pass_results = f2p["pass_to_pass_results"]
        else:
            record.status = self._evaluate(record)

        record.end_time = time.time()
        record.runtime = record.end_time - record.start_time

        return record

    def _evaluate(self, record: InstanceRecord) -> str:
        """Fallback evaluation for tasks with no FAIL_TO_PASS/PASS_TO_PASS
        gold test lists (e.g. a synthetic or hand-authored local task) --
        just check whether the generic test_command run passed.
        """
        if record.test_results.get("returncode") == 0:
            return "resolved"
        return "not_resolved"

    def save_raw(self, results: list[InstanceRecord]) -> Path:
        """Persist raw results — never overwritten."""
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_results_dir / "results.jsonl"
        with open(path, "a") as f:
            for r in results:
                data = dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
                f.write(json.dumps(data) + "\n")
        return path

    def save_processed(self, results: list[InstanceRecord]) -> Path:
        self.processed_results_dir.mkdir(parents=True, exist_ok=True)
        total = len(results)
        resolved = sum(1 for r in results if r.status == "resolved")

        by_status: dict[str, int] = {}
        by_stop_reason: dict[str, int] = {}
        iterations_by_status: dict[str, list[int]] = {}
        for r in results:
            by_status[r.status] = by_status.get(r.status, 0) + 1
            if r.stop_reason:
                by_stop_reason[r.stop_reason] = by_stop_reason.get(r.stop_reason, 0) + 1
            iterations_by_status.setdefault(r.status, []).append(r.num_iterations)

        avg_iterations_by_status = {
            status: round(sum(vals) / len(vals), 2)
            for status, vals in iterations_by_status.items()
        }

        # Lightweight per-instance digest -- enough to see where each run left
        # off without opening the full raw history for every instance.
        instance_digests = [
            {
                "instance_id": r.instance_id,
                "status": r.status,
                "stop_reason": r.stop_reason,
                "num_iterations": r.num_iterations,
                "total_tool_calls": r.total_tool_calls,
                "last_tool": (r.last_action or {}).get("tool"),
                "last_expected_outcome": (r.last_reasoner_plan or {}).get("expected_outcome", ""),
                "fail_to_pass": f"{r.fail_to_pass_count}/{r.fail_to_pass_total}" if r.fail_to_pass_total else None,
                "pass_to_pass": f"{r.pass_to_pass_count}/{r.pass_to_pass_total}" if r.pass_to_pass_total else None,
            }
            for r in results
        ]

        path = self.processed_results_dir / "summary.json"
        with open(path, "w") as f:
            json.dump({
                "total": total,
                "resolved": resolved,
                "resolve_rate": resolved / total if total else 0.0,
                "by_status": by_status,
                "by_stop_reason": by_stop_reason,
                "avg_iterations_by_status": avg_iterations_by_status,
                "instances": instance_digests,
                "results": [r.__dict__ for r in results],  # full detail, incl. history
            }, f, indent=2)
        return path
