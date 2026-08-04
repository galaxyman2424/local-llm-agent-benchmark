"""Quick pilot: run a small, fixed number of instances per repo across every
reasoner x actioner combination in models.txt, and summarize WHY runs
succeeded or failed (stop_reason/exit_reason), broken down by repo and by
model -- instead of just a resolve rate.

The point isn't to measure resolve rate (too few instances for that to mean
anything) -- it's diagnostic: does a small model mostly fail with
`repeated_action`/`reasoner_failed` (a model-capability/context-window
problem no repo choice will fix), or with `environment_error`
(infrastructure tax that filtering to pure-Python repos should mostly
remove)? That distinction is what should actually decide whether "switch to
an easier benchmark" would even help.

Usage
-----
    # One instance from each pure-Python SWE-bench Lite repo, every
    # reasoner x actioner combination in models.txt
    python experiments/pilot_run.py

    # Two instances per repo, and include the full (not just pure-Python) repo set
    python experiments/pilot_run.py --instances-per-repo 2 --all-repos

    # Just one specific repo
    python experiments/pilot_run.py --repo-filter requests
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.model_roster import load_model_roster
from experiments.run_experiment import run_experiment
from experiments.run_grid_search import _slugify


def pilot_run(
    *,
    models_path: str = "models.txt",
    config_path: str = "configs/qwen_ornith.yaml",
    instances_per_repo: int = 1,
    seed_repos_dir: str = "seed_repos/swe_bench_lite",
    results_raw_dir: str = "results/raw/pilot",
    results_processed_dir: str = "results/processed/pilot",
    repo_filter: str | None = None,
    pure_python_only: bool = True,
) -> dict:
    """Run the pilot and return the aggregated diagnostic report (also
    printed and saved to ``results_processed_dir/PILOT_REPORT.json``).
    """
    roster = load_model_roster(models_path)
    combinations = [(r, a) for r in roster["reasoners"] for a in roster["actioners"]]

    print(f"[Pilot] {len(combinations)} reasoner x actioner combination(s), "
          f"{instances_per_repo} instance(s)/repo, pure_python_only={pure_python_only}, "
          f"repo_filter={repo_filter!r}")

    results_processed_dir = Path(results_processed_dir)
    results_processed_dir.mkdir(parents=True, exist_ok=True)

    # Every per-instance digest across every combo, tagged with the models
    # that produced it -- this flat list is what all the breakdowns below
    # get built from.
    all_digests: list[dict] = []
    combo_summaries: list[dict] = []

    start = time.time()
    for i, (reasoner_model, actioner_model) in enumerate(combinations):
        run_name = f"pilot__{_slugify(reasoner_model)}__{_slugify(actioner_model)}"
        print(f"\n{'#'*70}\n[Pilot] {i+1}/{len(combinations)}: "
              f"reasoner={reasoner_model} actioner={actioner_model}\n{'#'*70}")

        try:
            summary = run_experiment(
                config_path=config_path,
                seed_repos_dir=seed_repos_dir,
                results_raw_dir=str(Path(results_raw_dir) / run_name),
                results_processed_dir=str(results_processed_dir),
                reasoner_model=reasoner_model,
                actioner_model=actioner_model,
                run_name=run_name,
                repo_filter=repo_filter,
                pure_python_only=pure_python_only,
                instances_per_repo=instances_per_repo,
            )
        except Exception as e:
            print(f"[Pilot] Combination {run_name} CRASHED: {e}")
            summary = {
                "reasoner_model": reasoner_model, "actioner_model": actioner_model,
                "instances": [], "error": str(e), "traceback": traceback.format_exc(),
            }

        combo_summaries.append(summary)
        for digest in summary.get("instances", []):
            all_digests.append({
                **digest,
                "reasoner_model": reasoner_model,
                "actioner_model": actioner_model,
            })

    report = _build_report(all_digests, combo_summaries, time.time() - start)
    report_path = results_processed_dir / "PILOT_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    print(f"\n[Pilot] Full report saved to {report_path}")
    return report


def _build_report(all_digests: list[dict], combo_summaries: list[dict], elapsed: float) -> dict:
    def _counts(key: str, rows: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = row.get(key) or "(none)"
            counts[value] = counts.get(value, 0) + 1
        return counts

    by_repo: dict[str, dict] = {}
    for repo in sorted({d.get("repo_name", "unknown") for d in all_digests}):
        rows = [d for d in all_digests if d.get("repo_name", "unknown") == repo]
        by_repo[repo] = {
            "total": len(rows),
            "by_status": _counts("status", rows),
            "by_stop_reason": _counts("stop_reason", rows),
            "by_exit_reason": _counts("exit_reason", rows),
        }

    by_model: dict[str, dict] = {}
    for d in all_digests:
        key = f"{d['reasoner_model']} / {d['actioner_model']}"
        by_model.setdefault(key, []).append(d)
    by_model_report = {
        key: {
            "total": len(rows),
            "by_status": _counts("status", rows),
            "by_stop_reason": _counts("stop_reason", rows),
        }
        for key, rows in by_model.items()
    }

    crashed = [
        {"reasoner_model": s.get("reasoner_model"), "actioner_model": s.get("actioner_model"), "error": s.get("error")}
        for s in combo_summaries if s.get("error") and not s.get("instances")
    ]

    return {
        "total_instances_run": len(all_digests),
        "total_combinations": len(combo_summaries),
        "elapsed_seconds": round(elapsed, 1),
        "overall_by_status": _counts("status", all_digests),
        "overall_by_stop_reason": _counts("stop_reason", all_digests),
        "overall_by_exit_reason": _counts("exit_reason", all_digests),
        "by_repo": by_repo,
        "by_model": by_model_report,
        "crashed_combinations": crashed,
    }


def _print_report(report: dict) -> None:
    print(f"\n{'='*70}\n[Pilot] DIAGNOSTIC REPORT "
          f"({report['total_instances_run']} instance-runs, "
          f"{report['total_combinations']} combination(s), "
          f"{report['elapsed_seconds']}s)\n{'='*70}")

    print("\nOverall exit_reason (why did each run actually stop?):")
    for reason, count in sorted(report["overall_by_exit_reason"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4}  {reason}")

    print("\nOverall status:")
    for status, count in sorted(report["overall_by_status"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4}  {status}")

    print("\nBy repo (environment_error here = infra tax; repeated_action/reasoner_failed "
          "here = likely a model-capability issue that repo choice won't fix):")
    for repo, info in report["by_repo"].items():
        top_exit = sorted(info["by_exit_reason"].items(), key=lambda kv: -kv[1])
        top_exit_str = ", ".join(f"{r}={c}" for r, c in top_exit)
        print(f"  {repo:<20} n={info['total']:<3} {top_exit_str}")

    print("\nBy model combination:")
    for combo, info in report["by_model"].items():
        resolved = info["by_status"].get("resolved", 0)
        print(f"  {combo:<45} n={info['total']:<3} resolved={resolved} "
              f"top_exit={max(info['by_stop_reason'].items(), key=lambda kv: kv[1], default=('?', 0))[0]}")

    if report["crashed_combinations"]:
        print("\nCrashed combinations (need a manual look, not just a low score):")
        for c in report["crashed_combinations"]:
            print(f"  {c['reasoner_model']} / {c['actioner_model']}: {c['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic pilot: N instances/repo across the full model roster")
    parser.add_argument("--models", type=str, default="models.txt")
    parser.add_argument("--config", type=str, default="configs/qwen_ornith.yaml")
    parser.add_argument("--instances-per-repo", type=int, default=1)
    parser.add_argument("--repo-filter", type=str, default=None,
                         help="Only run instances whose repo contains this substring, e.g. 'requests'")
    parser.add_argument("--all-repos", action="store_true",
                         help="Include repos that need C-extension builds too (default: pure-Python only)")
    args = parser.parse_args()

    pilot_run(
        models_path=args.models,
        config_path=args.config,
        instances_per_repo=args.instances_per_repo,
        repo_filter=args.repo_filter,
        pure_python_only=not args.all_repos,
    )