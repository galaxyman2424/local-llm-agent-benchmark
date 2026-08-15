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

def _write_running_summary(
    summary_path: Path,
    config_path: str,
    planner_model: str,
    actioner_model: str,
    results: list,
    total_instances: int,
    elapsed_seconds: float,
) -> None:
    """Write an in-progress summary after each instance, so a crash mid-run
    still leaves a usable partial summary on disk instead of nothing."""
    from benchmarks.swebench import InstanceRecord

    resolved = sum(1 for r in results if isinstance(r, InstanceRecord) and r.status == "resolved")
    errors = sum(1 for r in results if not isinstance(r, InstanceRecord))
    completed = len(results)

    partial_summary = {
        "experiment_config": config_path,
        "planner_model": planner_model,
        "actioner_model": actioner_model,
        "completed_instances": completed,
        "total_instances": total_instances,
        "resolved": resolved,
        "resolve_rate": resolved / completed if completed else 0.0,
        "errors": errors,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "status": "in_progress",
    }
    with open(summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)

def load_config(config_path: str) -> dict:
    """Load and return the YAML config as a Python dict."""
    with open(config_path, "r") as f:
        import yaml
        return yaml.safe_load(f)

def get_model_max_context(model_id: str, base_url: str = "http://localhost:11434") -> int | None:
    """Query Ollama for a model's native max context length."""
    import urllib.request, json
    req = urllib.request.Request(
        f"{base_url}/api/show",
        data=json.dumps({"model": model_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        info = json.loads(resp.read())
    # context length shows up under model_info, key varies by architecture
    # e.g. "qwen2.context_length", "llama.context_length"
    model_info = info.get("model_info", {})
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            return int(value)
    return None       


def run_experiment(
    *,
    config_path: str = "configs/qwen_ornith.yaml",
    limit: int | None = None,
    seed_repos_dir: str = "seed_repos/swe_bench_lite",
    results_raw_dir: str = "results/raw",
    results_processed_dir: str = "results/processed",
    planner_model: str | None = None,
    actioner_model: str | None = None,
    run_name: str | None = None,
    repo_filter: str | None = None,
    pure_python_only: bool = False,
    instances_per_repo: int | None = None,
    use_docker: bool = False,
) -> dict:
    """Run a complete experiment based on the given config.

    Steps:
      1. Load configuration from YAML file
      2. Initialize agents (Planner, Actioner, Agent controller)
      3. Initialize SWE-bench benchmark
      4. Select instances (optionally limited by --limit)
      5. Run the benchmark on each instance
      6. Collect results
      7. Save raw results to results/raw/

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file. Still used for agent/benchmark
        settings (max_iterations, etc.) even if planner_model/actioner_model
        override the model names it specifies.
    limit : int | None
        Number of instances to evaluate. If None, all instances are run.
    seed_repos_dir : str
        Directory where SWE-bench Lite repos are cloned.
    results_raw_dir : str
        Where per-instance raw results are saved.
    results_processed_dir : str
        Where aggregated summaries are saved.
    planner_model : str | None
        If given, overrides the config file's planner model (useful for
        grid-searching many planner/actioner pairings without writing a
        YAML file per combination).
    actioner_model : str | None
        Same as above, for the actioner model.
    run_name : str | None
        If given, used as the summary filename stem instead of the config
        file's stem -- needed so multiple combinations sharing one base
        config don't overwrite each other's summary.json.
    repo_filter : str | None
        If given, only run instances whose ``repo`` field contains this
        substring (case-insensitive), e.g. ``"requests"``. Lets you prove
        out the Planner/Actioner architecture on a specific repo before
        paying the environment-setup cost of the full grid.
    pure_python_only : bool
        If True, restrict to the subset of SWE-bench Lite repos that don't
        require compiling numpy/scipy/pandas/matplotlib-style C extensions
        (see ``swebench.utils.PURE_PYTHON_REPOS``) -- most of
        ``ensure_repo_environment``'s complexity exists to survive that
        build pain, which has nothing to do with whether the agent
        architecture itself works. Combine with a small ``--limit`` for
        fast early iteration on real instances with real scoring, before
        trusting a resolve-rate number from the full (slower) dataset.
    instances_per_repo : int | None
        If given, OVERRIDES ``limit`` and instead selects up to this many
        instances from EACH repo present after filtering (preserving
        dataset order within a repo), rather than the first N instances
        overall (which could all come from a single repo if the dataset
        happens to be sorted that way). Meant for a quick per-repo pilot --
        e.g. ``instances_per_repo=1`` with ``pure_python_only=True`` runs
        exactly one instance from each pure-Python repo, so you can see
        whether failures are repo-specific (environment issues) or
        model-specific (small-model reasoning/context limits) before
        committing to a full run. See ``experiments/pilot_run.py``.
    use_docker : bool
        If True, each instance's repo environment runs in a per-repo
        Docker container (see ``swebench/docker_utils.py``) instead of a
        host venv -- all test execution routes through it too (the
        agent's own run_tests calls, its live FAIL_TO_PASS check, and the
        two evaluation steps in ``SWEbenchBenchmark.run_instance``).

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
    print(f"[Experiment] Loaded {len(dataset_list)} instances from dataset")

    if pure_python_only:
        from swebench.utils import filter_pure_python_instances
        before = len(dataset_list)
        dataset_list = filter_pure_python_instances(dataset_list)
        print(f"[Experiment] --pure-python-only: kept {len(dataset_list)}/{before} instances "
              "(skipping repos that need numpy/scipy/pandas/matplotlib-style C-extension builds)")

    if repo_filter:
        before = len(dataset_list)
        dataset_list = [inst for inst in dataset_list if repo_filter.lower() in inst.get("repo", "").lower()]
        print(f"[Experiment] --repo-filter={repo_filter!r}: kept {len(dataset_list)}/{before} instances")

    dataset = {"instances": {inst.get("instance_id", f"instance_{i}"): inst for i, inst in enumerate(dataset_list)}}

    # Initialize agents (explicit overrides win over the config file)
    from agents import Planner, Actioner, Agent
    planner_model = planner_model or config.get("planner", {}).get("model", "qwen3.5:9b")
    actioner_model = actioner_model or config.get("actioner", {}).get("model", "ornith:9b")

    # NOTE: these are per-LLM-request timeouts (how long a single Ollama
    # call may take), deliberately distinct from `agent.timeout` below
    # (the budget for the WHOLE task across all iterations). Conflating
    # the two meant a 600s "agent timeout" was silently being used as a
    # 120s single-request timeout, or vice versa.
    planner_timeout = config.get("planner", {}).get("timeout_seconds", 120.0)
    actioner_timeout = config.get("actioner", {}).get("timeout_seconds", 120.0)

    # num_ctx (context window) is a SEPARATE knob from num_predict -- see
    # agents/planner.py's DEFAULT_NUM_CTX docstring. Too small a value here
    # is the actual cause of replies truncating mid-JSON, not num_predict.
    planner_num_ctx = config.get("planner", {}).get("num_ctx")
    actioner_num_ctx = config.get("actioner", {}).get("num_ctx")

    actioner_max_read_lines = config.get("actioner", {}).get("max_read_lines", 300)

    print(f"[Experiment] Initializing Planner (model={planner_model}, request_timeout={planner_timeout}s, "
          f"num_ctx={planner_num_ctx})...")
    planner = Planner(model_id=planner_model, timeout_seconds=planner_timeout, num_ctx=planner_num_ctx)

    print(f"[Experiment] Initializing Actioner (model={actioner_model}, request_timeout={actioner_timeout}s, "
        f"num_ctx={actioner_num_ctx}, max_read_lines={actioner_max_read_lines})...")
    actioner = Actioner(
        model_id=actioner_model,
        timeout_seconds=actioner_timeout,
        num_ctx=actioner_num_ctx,
        max_read_lines=actioner_max_read_lines,
    )

    agent_max_iter = config.get("agent", {}).get("max_iterations", 50)
    agent_timeout = config.get("agent", {}).get("timeout_seconds", None)
    print(f"[Experiment] Agent max iterations: {agent_max_iter}")
    print(f"[Experiment] Agent timeout: {agent_timeout}s")
    agent = Agent(planner, actioner, max_iterations=agent_max_iter, timeout=agent_timeout)

    # Get benchmark object
    from benchmarks import SWEbenchBenchmark
    benchmark = SWEbenchBenchmark(
        seed_repo_dir=str(seed_repos_dir),
        results_dir=str(results_raw_dir),
    )

    # Select instances
    instance_ids = list(dataset["instances"].keys())
    if instances_per_repo is not None:
        # Group by repo (preserving each repo's relative order) and keep the
        # first `instances_per_repo` from each group, rather than the first
        # N overall -- a plain `limit` could land entirely inside one repo
        # if the dataset happens to be sorted/clustered by repo, which would
        # defeat the point of a per-repo pilot.
        by_repo: dict[str, list[str]] = {}
        for iid in instance_ids:
            repo = dataset["instances"][iid].get("repo", "unknown")
            by_repo.setdefault(repo, []).append(iid)
        selected_instances = [
            iid for repo_ids in by_repo.values() for iid in repo_ids[:instances_per_repo]
        ]
        print(f"[Experiment] --instances-per-repo {instances_per_repo}: selected "
              f"{len(selected_instances)} instance(s) across {len(by_repo)} repo(s)")
    elif limit is not None:
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

        result = None
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
                planner_model=planner_model,
                actioner_model=actioner_model,
                use_docker=use_docker,
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
            result = None
            print(f"[Experiment] Instance {instance_id}: ERROR - {e}")
            print(error_result["traceback"]) 

        if result is not None:
            benchmark.save_raw([result])

        results_processed_dir.mkdir(parents=True, exist_ok=True)
        summary_stem = run_name or Path(config_path).stem
        summary_path = results_processed_dir / f"{summary_stem}.json"
        _write_running_summary(summary_path, config_path, planner_model,
                                actioner_model, results, len(selected_instances),
                                time.time() - start_time)

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
    by_exit_reason: dict[str, int] = {}
    for r in results:
        if isinstance(r, InstanceRecord):
            if r.stop_reason:
                by_stop_reason[r.stop_reason] = by_stop_reason.get(r.stop_reason, 0) + 1
            if r.exit_reason:
                by_exit_reason[r.exit_reason] = by_exit_reason.get(r.exit_reason, 0) + 1
            instance_digests.append({
                "instance_id": r.instance_id,
                "repo_name": r.repo_name,
                "status": r.status,
                "stop_reason": r.stop_reason,
                "exit_reason": r.exit_reason,
                "num_iterations": r.num_iterations,
                "last_tool": (r.last_action or {}).get("tool"),
            })
    
    timeouts = sum(1 for r in results if isinstance(r, InstanceRecord) and r.status == "timeout")
    repeated_loops = by_exit_reason.get("repeated_action_loop", 0)

    summary = {
        "experiment_config": config_path,
        "planner_model": planner_model,
        "actioner_model": actioner_model,
        "total_instances": total,
        "resolved": resolved,
        "resolve_rate": resolved / total if total else 0.0,
        "by_status": by_status,
        "by_exit_reason": by_exit_reason,
        "timeouts": timeouts,
        "repeated_action_loops": repeated_loops,
        "errors": error_count,
        "run_time_seconds": round(run_time, 2),
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
    parser.add_argument("--planner-model", type=str, default=None, help="Override the config's planner model")
    parser.add_argument("--actioner-model", type=str, default=None, help="Override the config's actioner model")
    parser.add_argument("--run-name", type=str, default=None, help="Summary filename stem (default: config file stem)")
    parser.add_argument("--repo-filter", type=str, default=None,
                         help="Only run instances whose repo contains this substring, e.g. 'requests'")
    parser.add_argument("--pure-python-only", action="store_true",
                         help="Only run repos that don't need C-extension builds (numpy/scipy/pandas/"
                              "matplotlib-style deps) -- cheaper/faster signal for early iteration")
    parser.add_argument("--instances-per-repo", type=int, default=None,
                         help="Select up to N instances from EACH repo (overrides --limit) -- e.g. "
                              "1 to run a quick one-instance-per-repo pilot")
    parser.add_argument("--docker", action="store_true",
                         help="Run each instance's repo environment in Docker instead of a host venv "
                              "(see swebench/docker_utils.py)")
    args = parser.parse_args()

    result = run_experiment(
        config_path=args.config,
        limit=args.limit,
        planner_model=args.planner_model,
        actioner_model=args.actioner_model,
        run_name=args.run_name,
        repo_filter=args.repo_filter,
        pure_python_only=args.pure_python_only,
        instances_per_repo=args.instances_per_repo,
        use_docker=args.docker,
    )
    print(json.dumps(result, indent=2))