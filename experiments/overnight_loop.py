"""Run experiments/grid-search combinations unattended overnight, within a
wall-clock time budget, and produce a single human-readable morning digest.

This does NOT modify any project code -- it only orchestrates repeated calls
to experiments/run_experiment.py (via run_grid_search's building blocks),
survives per-combination crashes, and stops cleanly at a deadline instead of
running indefinitely or getting killed mid-write.

Usage
-----
    # Run until 7:00 AM local time, 3 instances per combination
    python experiments/overnight_loop.py --until 07:00 --limit 3

    # Or run for a fixed duration instead of a clock time
    python experiments/overnight_loop.py --hours 8 --limit 3

    # Resume: safe to re-run any time. Already-completed combinations
    # (present in results/processed/grid_search/<run_name>.json) are
    # skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.model_roster import load_model_roster
from experiments.run_experiment import run_experiment
from experiments.run_grid_search import _slugify


def _parse_until(until: str) -> dt.datetime:
    """Parse HH:MM into the next occurrence of that clock time (today if
    still ahead, otherwise tomorrow)."""
    hh, mm = (int(x) for x in until.split(":"))
    now = dt.datetime.now()
    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate <= now:
        candidate += dt.timedelta(days=1)
    return candidate


def overnight_loop(
    *,
    models_path: str = "models.txt",
    config_path: str = "configs/qwen_ornith.yaml",
    limit: int = 3,
    deadline: dt.datetime,
    seed_repos_dir: str = "seed_repos/swe_bench_lite",
    results_raw_dir: str = "results/raw",
    results_processed_dir: str = "results/processed/grid_search",
    force: bool = False,
    per_combo_safety_margin_seconds: float = 120.0,
    repo_filter: str | None = None,
    pure_python_only: bool = False,
) -> Path:
    """Run every reasoner x actioner combination once, skipping ones that
    already have a saved summary (unless --force), and stop BEFORE starting
    a new combination if there isn't enough time left before `deadline`.

    Returns the path to the morning digest markdown file.
    """
    roster = load_model_roster(models_path)
    combinations = [
        (r, a) for r in roster["reasoners"] for a in roster["actioners"]
    ]

    results_processed_dir = Path(results_processed_dir)
    results_processed_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_processed_dir / "overnight_log.jsonl"

    def _log(event: dict) -> None:
        event["ts"] = dt.datetime.now().isoformat()
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    _log({"event": "start", "deadline": deadline.isoformat(),
          "combinations": len(combinations), "limit": limit})

    last_combo_duration = 0.0

    for i, (reasoner_model, actioner_model) in enumerate(combinations):
        run_name = f"{_slugify(reasoner_model)}__{_slugify(actioner_model)}"
        summary_path = results_processed_dir / f"{run_name}.json"

        if summary_path.exists() and not force:
            _log({"event": "skip_existing", "run_name": run_name})
            continue

        time_left = (deadline - dt.datetime.now()).total_seconds()
        # Use the last combination's actual duration (padded) as the estimate
        # for whether there's room for one more -- far more honest than a
        # fixed guess, since instance count/model speed varies a lot.
        estimated_next = max(last_combo_duration, 60.0) + per_combo_safety_margin_seconds
        if time_left < estimated_next:
            _log({"event": "stop_deadline", "time_left_seconds": round(time_left, 1),
                  "estimated_next_seconds": round(estimated_next, 1),
                  "remaining_combinations": len(combinations) - i})
            break

        print(f"\n[overnight] {dt.datetime.now().strftime('%H:%M:%S')} "
              f"Combination {i+1}/{len(combinations)}: "
              f"reasoner={reasoner_model} actioner={actioner_model} "
              f"(time left: {time_left/3600:.1f}h)")

        combo_start = time.time()
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
            _log({"event": "combo_done", "run_name": run_name,
                  "resolve_rate": summary.get("resolve_rate", 0.0),
                  "errors": summary.get("errors", 0)})
        except Exception as e:
            # A crash in ONE combination must never take down the rest of
            # the overnight run.
            summary = {
                "reasoner_model": reasoner_model,
                "actioner_model": actioner_model,
                "total_instances": 0,
                "resolved": 0,
                "resolve_rate": 0.0,
                "errors": 1,
                "run_time_seconds": time.time() - combo_start,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            _log({"event": "combo_crashed", "run_name": run_name, "error": str(e)})
            print(f"[overnight] Combination {run_name} CRASHED: {e}")

        last_combo_duration = time.time() - combo_start

    _log({"event": "loop_end"})
    return _write_digest(results_processed_dir, log_path)


def _write_digest(results_processed_dir: Path, log_path: Path) -> Path:
    """Build a single morning-readable markdown digest from every saved
    summary + the run log, sorted best-first."""
    summaries = []
    for p in sorted(results_processed_dir.glob("*.json")):
        if p.name == "leaderboard.json":
            continue
        try:
            summaries.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue

    summaries.sort(
        key=lambda s: (
            -s.get("resolve_rate", 0.0),
            s.get("errors", 0),
            s.get("run_time_seconds", float("inf")),
        )
    )

    events = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    crashed = [e for e in events if e.get("event") == "combo_crashed"]
    stopped = [e for e in events if e.get("event") == "stop_deadline"]

    lines = ["# Overnight Run Digest", "",
              f"Generated: {dt.datetime.now().isoformat()}", ""]

    lines.append("## Leaderboard (best first)")
    lines.append("")
    lines.append("| rank | reasoner | actioner | resolved | rate | errors | runtime(s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for rank, s in enumerate(summaries, start=1):
        lines.append(
            f"| {rank} | {s.get('reasoner_model','?')} | {s.get('actioner_model','?')} | "
            f"{s.get('resolved',0)}/{s.get('total_instances',0)} | "
            f"{s.get('resolve_rate',0.0):.1%} | {s.get('errors',0)} | "
            f"{s.get('run_time_seconds',0):.1f} |"
        )

    if crashed:
        lines += ["", "## Crashed combinations (needs manual look, not just a low score)"]
        for e in crashed:
            lines.append(f"- `{e['run_name']}`: {e.get('error','?')}")

    if stopped:
        lines += ["", "## Stopped early due to time budget"]
        for e in stopped:
            lines.append(
                f"- {e['remaining_combinations']} combination(s) not attempted "
                f"({e['time_left_seconds']:.0f}s left, needed ~{e['estimated_next_seconds']:.0f}s)"
            )

    digest_path = results_processed_dir / "MORNING_DIGEST.md"
    digest_path.write_text("\n".join(lines) + "\n")
    print(f"\n[overnight] Digest written to {digest_path}")
    return digest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight time-boxed grid search runner")
    parser.add_argument("--models", type=str, default="models.txt")
    parser.add_argument("--config", type=str, default="configs/qwen_ornith.yaml")
    parser.add_argument("--limit", type=int, default=3,
                         help="Instances per combination (keep small overnight -- this runs unattended)")
    parser.add_argument("--until", type=str, default=None, help="Stop by this clock time, e.g. 07:00")
    parser.add_argument("--hours", type=float, default=None, help="Or: stop after this many hours from now")
    parser.add_argument("--force", action="store_true", help="Re-run combinations that already have a summary")
    parser.add_argument("--repo-filter", type=str, default=None,
                         help="Only run instances whose repo contains this substring, e.g. 'requests'")
    parser.add_argument("--pure-python-only", action="store_true",
                         help="Only run repos that don't need C-extension builds -- cheaper/faster "
                              "signal for early iteration, good for overnight sanity runs")
    args = parser.parse_args()

    if args.until:
        deadline = _parse_until(args.until)
    elif args.hours:
        deadline = dt.datetime.now() + dt.timedelta(hours=args.hours)
    else:
        parser.error("Pass either --until HH:MM or --hours N")

    print(f"[overnight] Deadline: {deadline}")
    overnight_loop(
        models_path=args.models,
        config_path=args.config,
        limit=args.limit,
        deadline=deadline,
        force=args.force,
        repo_filter=args.repo_filter,
        pure_python_only=args.pure_python_only,
    )