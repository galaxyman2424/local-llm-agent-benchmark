"""Run every reasoner x actioner combination from models.txt through SWE-bench
and rank them by resolve rate.

This reuses experiments/run_experiment.py's run_experiment() for each
combination (passing reasoner_model/actioner_model overrides so we don't
need a separate YAML config file per pairing), then builds a leaderboard
from all the resulting summaries.

Example usage::

    # See what would run without actually running anything
    python experiments/run_grid_search.py --dry-run

    # Run the full grid (every reasoner x actioner pair) against 5 instances
    python experiments/run_grid_search.py --limit 5

    # Use a different base config for shared agent/runtime settings
    # (models.txt still decides which reasoner/actioner models are tried)
    python experiments/run_grid_search.py --config configs/qwen_deepseek.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

# See experiments/run_experiment.py for why this is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.model_roster import load_model_roster
from experiments.run_experiment import run_experiment


def _slugify(model_id: str) -> str:
    """Turn a model id like 'qwen3.5:9b' into a filesystem-safe stem."""
    return model_id.replace(":", "-").replace("/", "-")


def run_grid_search(
    *,
    models_path: str = "models.txt",
    config_path: str = "configs/qwn_ornith.yaml",
    limit: int | None = None,
    seed_repos_dir: str = "seed_repos/swe_bench_lite",
    results_raw_dir: str = "results/raw",
    results_processed_dir: str = "results/processed/grid_search",
    dry_run: bool = False,
    repo_filter: str | None = None,
    pure_python_only: bool = False,
) -> list[dict]:
    """Run (or preview) every reasoner x actioner combination.

    Parameters
    ----------
    models_path :
        Path to the models.txt roster (see experiments/model_roster.py).
    config_path :
        Base YAML config supplying shared settings (max_iterations, etc.);
        its own reasoner/actioner models are ignored in favor of each grid
        combination.
    limit :
        Number of SWE-bench instances to run per combination. Keep this
        small (e.g. 3-10) for an initial grid search -- N combinations each
        running M instances is N*M total agent runs.
    dry_run :
        If True, just print the combinations that would run (and how many
        total agent runs that implies) without actually running anything.
        Useful for sanity-checking models.txt and estimating runtime before
        committing to a potentially long grid search.
    repo_filter, pure_python_only :
        Passed straight through to :func:`run_experiment` -- lets you grid-
        search reasoner/actioner combinations against a cheap, fast-
        iterating subset of instances (e.g. skipping astropy/scikit-learn-
        style C-extension builds) before committing to the full dataset.

    Returns
    -------
    list[dict]
        One summary dict per combination, sorted by resolve_rate descending
        (best first). Each combination's full summary is also written to
        ``results_processed_dir/<reasoner>__<actioner>.json``, and the
        whole leaderboard to ``results_processed_dir/leaderboard.json``.
    """
    roster = load_model_roster(models_path)
    reasoners = roster["reasoners"]
    actioners = roster["actioners"]
    combinations = list(itertools.product(reasoners, actioners))

    print(f"[GridSearch] Loaded roster from {models_path}: "
          f"{len(reasoners)} reasoner(s) x {len(actioners)} actioner(s) "
          f"= {len(combinations)} combination(s)")
    for reasoner_model, actioner_model in combinations:
        print(f"  - reasoner={reasoner_model:<40} actioner={actioner_model}")

    if dry_run:
        instances_note = "all available" if limit is None else str(limit)
        print(f"\n[GridSearch] Dry run only -- each combination would run "
              f"against {instances_note} instance(s); "
              f"no agent runs were started.")
        return []

    results_processed_dir = Path(results_processed_dir)
    results_processed_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: list[dict] = []
    grid_start = time.time()

    for i, (reasoner_model, actioner_model) in enumerate(combinations):
        run_name = f"{_slugify(reasoner_model)}__{_slugify(actioner_model)}"
        print(f"\n{'#'*70}")
        print(f"[GridSearch] Combination {i+1}/{len(combinations)}: "
              f"reasoner={reasoner_model} actioner={actioner_model}")
        print(f"{'#'*70}")

        try:
            summary = run_experiment(
                config_path=config_path,
                limit=limit,
                seed_repos_dir=seed_repos_dir,
                results_raw_dir=str(Path(results_raw_dir) / run_name),
                results_processed_dir=str(results_processed_dir),
                reasoner_model=reasoner_model,
                actioner_model=actioner_model,
                run_name=run_name,
                repo_filter=repo_filter,
                pure_python_only=pure_python_only,
            )
        except Exception as e:
            print(f"[GridSearch] Combination {run_name} FAILED: {e}")
            summary = {
                "reasoner_model": reasoner_model,
                "actioner_model": actioner_model,
                "total_instances": 0,
                "resolved": 0,
                "resolve_rate": 0.0,
                "errors": 1,
                "run_time_seconds": 0.0,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        leaderboard.append(summary)

    # Rank best-first by resolve rate; ties broken by fewer errors, then
    # faster runtime (a combo that resolves just as much in less time wins).
    leaderboard.sort(
        key=lambda s: (
            -s.get("resolve_rate", 0.0),
            s.get("errors", 0),
            s.get("run_time_seconds", float("inf")),
        )
    )

    total_time = time.time() - grid_start
    leaderboard_path = results_processed_dir / "leaderboard.json"
    with open(leaderboard_path, "w") as f:
        json.dump({
            "combinations_run": len(combinations),
            "total_time_seconds": round(total_time, 2),
            "leaderboard": leaderboard,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[GridSearch] Done. {len(combinations)} combination(s) in {total_time:.1f}s")
    print(f"[GridSearch] Leaderboard saved to {leaderboard_path}")
    print(f"{'='*70}")
    _print_leaderboard(leaderboard)

    return leaderboard


def _print_leaderboard(leaderboard: list[dict]) -> None:
    """Pretty-print the ranked results as a simple text table."""
    if not leaderboard:
        print("[GridSearch] No results to show.")
        return

    header = f"{'rank':<5}{'reasoner':<28}{'actioner':<38}{'resolved':<10}{'rate':<8}{'errors':<8}"
    print(header)
    print("-" * len(header))
    for rank, s in enumerate(leaderboard, start=1):
        resolved = s.get("resolved", 0)
        total = s.get("total_instances", 0)
        rate = s.get("resolve_rate", 0.0)
        errors = s.get("errors", 0)
        print(
            f"{rank:<5}{s.get('reasoner_model', '?'):<28}{s.get('actioner_model', '?'):<38}"
            f"{f'{resolved}/{total}':<10}{f'{rate:.1%}':<8}{errors:<8}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run every reasoner x actioner combination from models.txt through SWE-bench"
    )
    parser.add_argument("--models", type=str, default="models.txt", help="Path to models.txt roster")
    parser.add_argument("--config", type=str, default="configs/qwen_ornith.yaml",
                         help="Base config for shared settings (max_iterations, etc.)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Number of instances to run per combination (recommended for an initial grid search)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Just print the combinations that would run, without running them")
    parser.add_argument("--repo-filter", type=str, default=None,
                         help="Only run instances whose repo contains this substring, e.g. 'requests'")
    parser.add_argument("--pure-python-only", action="store_true",
                         help="Only run repos that don't need C-extension builds -- cheaper/faster "
                              "signal for early iteration")
    args = parser.parse_args()

    run_grid_search(
        models_path=args.models,
        config_path=args.config,
        limit=args.limit,
        dry_run=args.dry_run,
        repo_filter=args.repo_filter,
        pure_python_only=args.pure_python_only,
    )