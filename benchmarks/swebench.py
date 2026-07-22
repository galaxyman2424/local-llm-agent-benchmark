"""SWE-bench benchmark: manages dataset, repo setup, evaluation, and result storage."""

from __future__ import annotations

import json
import time
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
        """Clone or checkout a seed repo at the specified commit."""
        # For now we just record; real impl would clone + git checkout <commit>
        repo_path = self.seed_repo_dir / "repos" / repo_name
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

        # 1. Set up repo at base commit
        record.start_time = start
        repo_path = self.setup_repo(repo_name, base_commit)
        record.repo_name = str(repo_path.name)

        task_text = task.get("task_description", "")
        agent_result = agent.solve(str(repo_path), task_text)

        record.num_iterations = agent_result.num_iterations if agent_result else 0
        record.total_tool_calls = agent_result.total_tool_calls if agent_result else 0
        record.final_patch = agent_result.final_patch if agent_result else ""

        # 2. Run tests on the patched repo
        test_cmd = task.get("test_command", "pytest")
        proc = __import__("subprocess").run(
            ["/bin/sh", "-c", f"cd {repo_path} && {test_cmd}"],
            capture_output=True, text=True, timeout=60,
        )
        record.test_results = {
            "command": test_cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[:500],
            "stderr": proc.stderr[:500],
        }

        # 3. Compare against expected patch (for pass@1 evaluation)
        record.status = self._evaluate(record)

        record.end_time = time.time()
        record.runtime = record.end_time - record.start_time

        return record

    def _evaluate(self, record: InstanceRecord) -> str:
        """Simple evaluation: check if patch matches expected (placeholder)."""
        # Real impl would diff the final patch against gold; here we just mark passed/failed
        if record.test_results.get("returncode") == 0:
            return "passed"
        return "failed"

    def save_raw(self, results: list[InstanceRecord]) -> Path:
        """Persist raw results — never overwritten."""
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_results_dir / "results.jsonl"
        with open(path, "a") as f:
            for r in results:
                f.write(json.dumps(r.__dict__) + "\n")
        return path

    def save_processed(self, results: list[InstanceRecord]) -> Path:
        """Aggregate and save processed summary."""
        self.processed_results_dir.mkdir(parents=True, exist_ok=True)
        passed = sum(1 for r in results if r.status == "passed")
        total = len(results)
        path = self.processed_results_dir / "summary.json"
        with open(path, "w") as f:
            json.dump({
                "total": total,
                "passed": passed,
                "pass_rate": passed / total if total else 0.0,
                "results": [r.__dict__ for r in results],
            }, f, indent=2)
        return path
