"""SWE-bench Lite benchmark: provides a standard interface for running experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Aggregated result of a benchmark run — never overwritten once written."""
    configuration_name: str = ""
    total_instances: int = 0
    resolved: int = 0
    timeout: int = 0
    failed: int = 0
    average_runtime_seconds: float = 0.0
    pass_rate: float = 0.0


class SWEBenchLite:
    """SWE-bench Lite benchmark interface.

    Responsibilities (per section 3.1):
      - Dataset loading and task selection
      - Instance filtering by ID or repository name
      - Limit on the number of instances to run
      - Repository setup from base commit
      - Agent execution orchestration
      - Test execution after agent runs out of iterations
      - Evaluation against expected patches
      - Result storage in results/raw/ and results/processed/

    Usage:
        benchmark = SWEBenchLite(seed_repo_dir="seed_repos", results_dir="results")

        # Run a single instance
        result = benchmark.run_instance(
            agent=agent,
            instance_id="django__django-12345"
        )

        # Run with limits and filtering
        results = benchmark.run(
            agent=agent,
            instances=None,  # all available
            limit=5,
            repository_name_filter=None,
            instance_id_filter=None,
        )

        # Save results
        raw_path = benchmark.save_raw(results)
        summary_path = benchmark.save_processed(results)
    """

    def __init__(self, seed_repo_dir: str = "seed_repos", results_dir: str = "results"):
        self.seed_repo_dir = Path(seed_repo_dir)
        self.results_dir = Path(results_dir)
        self.raw_results_dir = self.results_dir / "raw"
        self.processed_results_dir = self.results_dir / "processed"

    def _load_instances(self, dataset_path: str | None = None) -> list[dict]:
        """Load instances from the SWE-bench Lite JSONL file.

        If no path is provided, look for standard locations.
        Returns empty list if no data is found — allows synthetic generation fallback.
        """
        paths_to_try = []
        if dataset_path:
            paths_to_try.append(Path(dataset_path))
        else:
            # Standard SWE-bench Lite locations
            paths_to_try.extend([
                self.seed_repo_dir / "swe_bench_lite" / "instances.jsonl",
                self.seed_repo_dir / "swe_bench_lite" / "dataset.json",
            ])

        for path in paths_to_try:
            if not path.exists():
                continue
            instances = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            instances.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return instances

        # Check for a single JSON file with a list of instances
        single_path = self.seed_repo_dir / "swe_bench_lite" / "dataset.json"
        if single_path.exists():
            try:
                data = json.loads(single_path.read_text())
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "instances" in data:
                    return data["instances"]
            except (json.JSONDecodeError, KeyError):
                pass

        return []

    def filter_instances(
        self,
        instances: list[dict],
        instance_id_filter: str | None = None,
        repository_name_filter: str | None = None,
    ) -> list[dict]:
        """Filter instances by ID or repository name.

        Args:
            instances: List of all available instances.
            instance_id_filter: If set, only return instances whose 'instance_id' starts with this string.
                Example: "django__django-12345" returns only that exact instance.
            repository_name_filter: If set, only return instances from repositories matching this name.

        Returns:
            Filtered list of instances.
        """
        if not instances:
            return []

        filtered = instances
        if instance_id_filter is not None:
            # Support exact match or prefix match (e.g., "django__django" for all django repos)
            if "__" in instance_id_filter and "-" in instance_id_filter.split("__")[-1]:
                # Exact instance ID
                filtered = [i for i in filtered if i.get("instance_id", "") == instance_id_filter]
            else:
                # Prefix match — keep instances from the same repo family
                filtered = [
                    i for i in filtered
                    if instance_id_filter.lower() in i.get("instance_id", "").lower()
                    or i.get("repo", "").lower().startswith(instance_id_filter)
                ]

        if repository_name_filter is not None:
            filtered = [
                i for i in filtered
                if repository_name_filter.lower() in i.get("repo", "").lower()
                or repository_name_filter.lower() in i.get("instance_id", "").lower()
            ]

        return filtered

    def select_instances(
        self,
        instances: list[dict],
        limit: int | None = None,
    ) -> list[dict]:
        """Select up to `limit` instances from the available pool.

        If no instances are provided and a dataset file exists, loads them first.
        Selection is random unless the same seed is used reproducibly (not implemented here).

        Args:
            instances: List of instance dicts. If empty/None, tries loading from dataset file.
            limit: Maximum number of instances to select. None = all.

        Returns:
            Selected list of instances.
        """
        if not instances and self._load_instances() is None:
            # Return empty — caller can generate synthetic instances or provide their own
            return []

        available = instances if instances else []
        if limit is not None and limit < len(available):
            import random
            selected_indices = random.sample(range(len(available)), limit)
            available = [available[i] for i in sorted(selected_indices)]

        return available

    def run_instance(
        self,
        agent,
        instance_id: str,
        reasoner_model: str | None = None,
        actioner_model: str | None = None,
    ) -> dict:
        """Run a single SWE-bench Lite instance through the full pipeline.

        Args:
            agent: An agent object with a `solve(repo_path, task_text)` method.
            instance_id: The ID of the instance to run (e.g., "django__django-12345").
            reasoner_model: Optional override for the reasoner model name.
            actioner_model: Optional override for the actioner model name.

        Returns:
            A dict with result metadata (instance_id, status, runtime, etc.).

        Raises:
            FileNotFoundError: If the instance cannot be found in the dataset or synthetic instances are not available.
        """
        # Try to load from dataset first; fall back to generating a representative synthetic task
        all_instances = self._load_instances()
        if all_instances and any(i.get("instance_id", "") == instance_id for i in all_instances):
            task_data = next(i for i in all_instances if i.get("instance_id", "") == instance_id)
        else:
            # Generate a synthetic task that looks like SWE-bench Lite
            repo_name, issue_id = instance_id.split("__") if "__" in instance_id else ("unknown", "0")
            task_data = {
                "instance_id": instance_id,
                "repo": repo_name,
                "base_commit": f"{repo_name}-12345",  # placeholder; real impl would use actual commit
                "task_description": (
                    f"Fix the issue in repository '{repo_name}' at base commit {issue_id}. "
                    f"Implement the solution that resolves the reported problem and run tests to verify."
                ),
                "test_command": "pytest",
            }

        repo_name_for_path = instance_id.split("__")[0] if "__" in instance_id else instance_id
        repo_path = self.seed_repo_dir / "repos" / instance_id
        if not repo_path.exists():
            # Create a minimal placeholder directory for the agent to work in
            repo_path.mkdir(parents=True, exist_ok=True)

        from swebench.utils import ensure_repo_environment
        python_bin, install_ok = ensure_repo_environment(repo_path)

        record = {
            "instance_id": task_data.get("instance_id", instance_id),
            "repo_name": task_data.get("repo", repo_name_for_path),
            "base_commit": task_data.get("base_commit", ""),
            "reasoner_model": reasoner_model or "qwen2.5:7b",
            "actioner_model": actioner_model or "qwen2.5:7b",
        }

        start_time = __import__("time").time()
        record["start_time"] = start_time

        task_text = task_data.get("task_description") or task_data.get("problem_statement", "")
        test_cmd = task_data.get("test_command", "pytest")
        agent_result = agent.solve(str(repo_path), task_text, test_command=test_cmd)

        end_time = __import__("time").time()
        record["end_time"] = end_time
        record["runtime_seconds"] = round(end_time - start_time, 1)

        if not install_ok:
            # Environment setup genuinely failed -- see
            # .swebench_venv/pip_install.log. Don't let this masquerade as
            # "the agent didn't fix the bug".
            record["status"] = "environment_error"
            return record

        # Determine status: official FAIL_TO_PASS/PASS_TO_PASS evaluation
        # when the task actually provides those gold test lists, otherwise
        # fall back to the agent's own quick self-assessment.
        if task_data.get("FAIL_TO_PASS") or task_data.get("PASS_TO_PASS"):
            from swebench.utils import evaluate_fail_to_pass
            f2p = evaluate_fail_to_pass(repo_path, task_data, timeout=60, python_bin=python_bin)
            record["status"] = f2p["status"]
            record["fail_to_pass_count"] = f2p["fail_to_pass_count"]
            record["fail_to_pass_total"] = f2p["fail_to_pass_total"]
            record["pass_to_pass_count"] = f2p["pass_to_pass_count"]
            record["pass_to_pass_total"] = f2p["pass_to_pass_total"]
            record["fail_to_pass_results"] = f2p["fail_to_pass_results"]
            record["pass_to_pass_results"] = f2p["pass_to_pass_results"]
        elif hasattr(agent_result, "status"):
            # No gold test lists (e.g. a synthetic local task) -- fall back
            # to whatever the agent itself concluded. Map the agent's own
            # "passed"/"failed" vocabulary onto this benchmark's
            # "resolved"/"not_resolved" vocabulary for consistency.
            record["status"] = {"passed": "resolved", "failed": "not_resolved"}.get(
                agent_result.status, agent_result.status
            )
        elif hasattr(agent_result, "success"):
            record["status"] = "resolved" if agent_result.success else "not_resolved"
        else:
            record["status"] = "incomplete"

        record["num_iterations"] = getattr(agent_result, "num_iterations", 0) or 0
        record["total_tool_calls"] = getattr(agent_result, "total_tool_calls", 0) or 0
        record["final_patch"] = getattr(agent_result, "final_patch", "") or ""

        record["stop_reason"] = getattr(agent_result, "stop_reason", "") if agent_result else ""
        record["last_tool"] = (getattr(agent_result, "last_action", None) or {}).get("tool")
        record["last_expected_outcome"] = (getattr(agent_result, "last_reasoner_plan", None) or {}).get("expected_outcome", "")
        record["history"] = getattr(agent_result, "history", None) or []

        return record

    def run(
        self,
        agent,
        instances: list[dict] | None = None,
        limit: int | None = None,
        instance_id_filter: str | None = None,
        repository_name_filter: str | None = None,
        seed_repo_dir: str | None = None,
    ) -> list[BenchmarkResult]:
        """Run a set of SWE-bench Lite instances.

        Args:
            agent: An agent object with a `solve(repo_path, task_text)` method.
            instances: Specific instance dicts to run. If None or empty, loads from dataset file.
            limit: Maximum number of instances to run (applied after filtering).
            instance_id_filter: Filter by instance ID prefix/exact match.
            repository_name_filter: Filter by repository name.
            seed_repo_dir: Override the default seed repo directory for loading data.

        Returns:
            List of BenchmarkResult objects, one per executed instance.
        """
        results = []

        # Load instances if not provided
        if instances is None or len(instances) == 0:
            dataset_path = (
                Path(seed_repo_dir) / "swe_bench_lite" / "instances.jsonl"
                if seed_repo_dir else self.seed_repo_dir / "swe_bench_lite" / "instances.jsonl"
            )
            instances = self._load_instances(dataset_path=dataset_path)

        # Apply filters
        filtered = self.filter_instances(instances, instance_id_filter, repository_name_filter)

        # Apply limit
        selected = self.select_instances(filtered, limit=limit)

        for task in selected:
            instance_id = task.get("instance_id", "unknown")
            try:
                result = self.run_instance(
                    agent=agent,
                    instance_id=instance_id,
                    reasoner_model=task.get("reasoner_model"),
                    actioner_model=task.get("actioner_model"),
                )
                results.append(result)
            except Exception as e:
                # Record failures so the benchmark can report them
                results.append({
                    "instance_id": instance_id or task.get("repo", "unknown"),
                    "status": "failed",
                    "error": str(e),
                    "runtime_seconds": 0.0,
                })

        return results

    def save_raw(self, results: list[dict]) -> Path:
        """Persist raw benchmark results — never overwritten."""
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_results_dir / "results.jsonl"
        with open(path, "a") as f:
            for r in results:
                if isinstance(r, BenchmarkResult):
                    # Convert to dict for JSON serialization
                    data = {
                        "configuration_name": r.configuration_name,
                        "total_instances": r.total_instances,
                        "resolved": r.resolved,
                        "timeout": r.timeout,
                        "failed": r.failed,
                        "average_runtime_seconds": r.average_runtime_seconds,
                        "pass_rate": r.pass_rate,
                    }
                else:
                    data = {k: v for k, v in r.items() if not callable(v)}
                f.write(json.dumps(data) + "\n")
        return path

    def save_processed(self, results: list[dict]) -> Path:
        """Aggregate and save processed summary."""
        self.processed_results_dir.mkdir(parents=True, exist_ok=True)
        resolved = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "resolved")
        timeout_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "timeout")
        failed_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") != "resolved" and r.get("status") not in ("timeout",))

        total = len(results)
        avg_runtime = round(
            sum(r.get("runtime_seconds", 0.0) or 0.0 for r in results if isinstance(r, dict)) / max(total, 1),
            2,
        )

        summary = {
            "total_instances": total,
            "resolved": resolved,
            "timeout": timeout_count,
            "failed": failed_count,
            "pass_rate": round(resolved / max(total, 1) * 100, 2),
            "average_runtime_seconds": avg_runtime,
        }

        path = self.processed_results_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path
