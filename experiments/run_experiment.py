"""Run a complete experiment from a YAML configuration file."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Make the project root importable regardless of the current working
# directory or how this script was invoked -- `python script.py` only adds
# the script's own directory to sys.path, not the project root, so
# `from swebench.utils import ...` / `from agents import ...` etc. would
# otherwise raise ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    reasoner_model: str | None = None,
    actioner_model: str | None = None,
    run_name: str | None = None,
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
        Path to the YAML configuration file. Still used for agent/benchmark
        settings (max_iterations, etc.) even if reasoner_model/actioner_model
        override the model names it specifies.
    limit : int | None
        Number of instances to evaluate. If None, all instances are run.
    seed_repos_dir : str
        Directory where SWE-bench Lite repos are cloned.
    results_raw_dir : str
        Where per-instance raw results are saved.
    results_processed_dir : str
        Where aggregated summaries are saved.
    reasoner_model : str | None
        If given, overrides the config file's reasoner model (useful for
        grid-searching many reasoner/actioner pairings without writing a
        YAML file per combination).
    actioner_model : str | None
        Same as above, for the actioner model.
    run_name : str | None
        If given, used as the summary filename stem instead of the config
        file's stem -- needed so multiple combinations sharing one base
        config don't overwrite each other's summary.json.

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
    dataset_list = load_dataset(str(seed_repos_dir / "dataset.jsonl"))
    dataset = {"instances": {inst.get("instance_id", f"instance_{i}"): inst for i, inst in enumerate(dataset_list)}}
    print(f"[Experiment] Loaded {len(dataset_list)} instances from dataset")

    # Initialize agents (explicit overrides win over the config file)
    from agents import Reasoner, Actioner, Agent
    reasoner_model = reasoner_model or config.get("reasoner", {}).get("model", "qwen3.5:9b")
    actioner_model = actioner_model or config.get("actioner", {}).get("model", "ornith:9b")

    # NOTE: these are per-LLM-request timeouts (how long a single Ollama
    # call may take), deliberately distinct from `agent.timeout` below
    # (the budget for the WHOLE task across all iterations). Conflating
    # the two meant a 600s "agent timeout" was silently being used as a
    # 120s single-request timeout, or vice versa.
    reasoner_timeout = config.get("reasoner", {}).get("timeout_seconds", 120.0)
    actioner_timeout = config.get("actioner", {}).get("timeout_seconds", 120.0)
    # num_ctx (context window) is a SEPARATE knob from num_predict -- see
    # agents/reasoner.py's DEFAULT_NUM_CTX docstring. Too small a value here
    # is the actual cause of replies truncating mid-JSON, not num_predict.
    reasoner_num_ctx = config.get("reasoner", {}).get("num_ctx", 16384)
    actioner_num_ctx = config.get("actioner", {}).get("num_ctx", 16384)

    print(f"[Experiment] Initializing Reasoner (model={reasoner_model}, request_timeout={reasoner_timeout}s, "
          f"num_ctx={reasoner_num_ctx})...")
    reasoner = Reasoner(model_id=reasoner_model, timeout_seconds=reasoner_timeout, num_ctx=reasoner_num_ctx)

    print(f"[Experiment] Initializing Actioner (model={actioner_model}, request_timeout={actioner_timeout}s, "
          f"num_ctx={actioner_num_ctx})...")
    actioner = Actioner(model_id=actioner_model, timeout_seconds=actioner_timeout, num_ctx=actioner_num_ctx)

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
    resolved = sum(1 for r in results if isinstance(r, InstanceRecord) and r.status == "resolved")
    total = len(selected_instances)
    error_count = sum(1 for r in results if not isinstance(r, InstanceRecord))
    by_status: dict[str, int] = {}


    by_stop_reason: dict[str, int] = {}
    instance_digests = []
    for r in results:
        if isinstance(r, InstanceRecord):
            if r.stop_reason:
                by_stop_reason[r.stop_reason] = by_stop_reason.get(r.stop_reason, 0) + 1
            instance_digests.append({
                "instance_id": r.instance_id,
                "status": r.status,
                "stop_reason": r.stop_reason,
                "num_iterations": r.num_iterations,
                "last_tool": (r.last_action or {}).get("tool"),
            })
    
    summary = {
        "experiment_config": config_path,
        "reasoner_model": reasoner_model,
        "actioner_model": actioner_model,
        "total_instances": total,
        "resolved": resolved,
        "resolve_rate": resolved / total if total else 0.0,
        "by_status": by_status,
        "errors": error_count,
        "run_time_seconds": round(run_time, 2),
        "by_stop_reason": by_stop_reason,
        "instances": instance_digests,
    }

    results_processed_dir.mkdir(parents=True, exist_ok=True)
    summary_stem = run_name or Path(config_path).stem
    summary_path = results_processed_dir / f"{summary_stem}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Experiment] Summary saved to {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a complete experiment from config file")
    parser.add_argument("--config", type=str, default="configs/qwen_ornith.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reasoner-model", type=str, default=None, help="Override the config's reasoner model")
    parser.add_argument("--actioner-model", type=str, default=None, help="Override the config's actioner model")
    parser.add_argument("--run-name", type=str, default=None, help="Summary filename stem (default: config file stem)")
    args = parser.parse_args()

    result = run_experiment(
        config_path=args.config,
        limit=args.limit,
        reasoner_model=args.reasoner_model,
        actioner_model=args.actioner_model,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2))
