"""Run a complete experiment from a YAML configuration file."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load and return the YAML config as a Python dict."""
    with open(config_path, "r") as f:
        import yaml
        return yaml.safe_load(f)


def run_experiment(
    *,
    config_path: str = "configs/qwen_ornith.yaml",
    limit: int | None = None,
    seed_repos_dir: str = "seed_repos/swe_bench_lite",
    results_raw_dir: str = "results/raw",
    results_processed_dir: str = "results/processed",
) -> dict:
    """Run a complete experiment based on the given config.

    Steps:
      1. Load configuration from YAML file
      2. Initialize agents (Reasoner, Actioner, Agent controller)
      3. Initialize SWE-bench benchmark
      4. Select instances (optionally limited by --limit)
      5. Run the benchmark on each instance
      6. Collect results
      7. Save raw results to results/raw/

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    limit : int | None
        Number of instances to evaluate. If None, all instances are run.
    seed_repos_dir : str
        Directory where SWE-bench Lite repos are cloned.
    results_raw_dir : str
        Where per-instance raw results are saved.
    results_processed_dir : str
        Where aggregated summaries are saved.

    Returns
    -------
    dict
        Aggregated experiment result summary.
    """
    config = load_config(config_path)
    print(f"[Experiment] Loaded config: {config_path}")
    print(json.dumps(config, indent=2))

    # Resolve directories
    seed_repos_dir = Path(seed_repos_dir).resolve()
    results_raw_dir = Path(results_raw_dir).resolve()
    results_processed_dir = Path(results_processed_dir).resolve()

    # Load dataset for instance selection
    from swebench.utils import load_dataset, find_repo_path
    dataset = load_dataset(str(seed_repos_dir))
    print(f"[Experiment] Loaded {len(dataset)} instances from dataset")

    # Initialize agents
    from agents import Reasoner, Actioner, Agent
    reasoner_model = config.get("reasoner", {}).get("model", "qwen3.5:9b")
    actioner_model = config.get("actioner", {}).get("model", "ornith:9b")

    print(f"[Experiment] Initializing Reasoner (model={reasoner_model})...")
    reasoner = Reasoner(model_id=reasoner_model)

    print(f"[Experiment] Initializing Actioner (model={actioner_model})...")
    actioner = Actioner(model_id=actioner_model)

    agent_max_iter = config.get("agent", {}).get("max_iterations", 50)
    print(f"[Experiment] Agent max iterations: {agent_max_iter}")
    agent = Agent(reasoner, actioner, max_iterations=agent_max_iter)

    # Get benchmark object
    from benchmarks import SWEbenchBenchmark
    benchmark = SWEbenchBenchmark(
        seed_repo_dir=str(seed_repos_dir),
        results_dir=str(results_raw_dir),
    )

    # Select instances
    instance_ids = list(dataset["instances"].keys())
    if limit is not None:
        selected_instances = instance_ids[:limit]
        print(f"[Experiment] Running with --limit {limit}: first {len(selected_instances)} instances")
    else:
        selected_instances = instance_ids
        print(f"[Experiment] Running all {len(instance_ids)} instances")

    # Run benchmark on each instance
    start_time = time.time()
    results = []
    for i, instance_id in enumerate(selected_instances):
        print(f"\n{'='*60}")
        print(f"[Experiment] Running instance {i+1}/{len(selected_instances)}: {instance_id}")
        print(f"{'='*60}")

        try:
            # Get the repo path for this instance from benchmark
            task = dataset["instances"][instance_id]
            repo_name = task.get("repo", "unknown")
            base_commit = task.get("base_commit", "")

            # Use benchmark's setup_repo to get the path
            repo_path = benchmark.setup_repo(repo_name, base_commit)
            print(f"[Experiment] Repo path for {instance_id}: {repo_path}")

            problem_statement = task.get("problem_statement", "")
            test_command = task.get("test_command", "pytest -x")

            # Run the agent on this instance using benchmark's run_instance
            result = benchmark.run_instance(
                task=task,
                agent=agent,
                reasoner_model=reasoner_model,
                actioner_model=actioner_model,
            )
            results.append(result)
            print(f"[Experiment] Instance {instance_id}: status={result.status}")

        except Exception as e:
            error_result = {
                "instance_id": instance_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            results.append(error_result)
            print(f"[Experiment] Instance {instance_id}: ERROR - {e}")

    # Save raw results using benchmark's save_raw
    run_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[Experiment] All instances complete. Saving raw results...")
    print(f"{'='*60}")

    benchmark.save_raw(results)
    print(f"[Experiment] Raw results saved to {results_raw_dir}/")

    # Save aggregated summary
    from benchmarks.swebench import InstanceRecord
    passed = sum(1 for r in results if isinstance(r, InstanceRecord) and r.status == "passed")
    total = len(selected_instances)
    error_count = sum(1 for r in results if not isinstance(r, InstanceRecord))

    summary = {
        "experiment_config": config_path,
        "total_instances": total,
        "completed": passed,
        "errors": error_count,
        "run_time_seconds": round(run_time, 2),
    }

    summary_path = results_processed_dir / f"{config_path.split('/')[-1]}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Experiment] Summary saved to {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a complete experiment from config file")
    parser.add_argument("--config", type=str, default="configs/qwen_ornith.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = run_experiment(config_path=args.config, limit=args.limit)
    print(json.dumps(result, indent=2))
