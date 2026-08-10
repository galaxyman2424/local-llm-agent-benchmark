# COMMANDS.md

All commands assume you're at the project root. All Python scripts here add
the project root to `sys.path` themselves, so `python path/to/script.py`
works from anywhere without needing `PYTHONPATH` set manually.

## One-time setup

### 1. Download the dataset

```bash
python swebench/download_dataset.py
```

Fetches SWE-bench Lite (`princeton-nlp/SWE-bench_Lite`, `test` split) via
Hugging Face `datasets` and caches it to
`seed_repos/swe_bench_lite/dataset.jsonl`. No-ops if already cached.

Options:
```bash
python swebench/download_dataset.py --url=https://example.com/mirror.jsonl
python swebench/download_dataset.py --cache-dir=/custom/path
```

Requires `pip install datasets --break-system-packages` unless `--url` is
given.

### 2. Check your local Ollama models against the roster

```bash
ollama list
```

Then edit `models.txt`'s `[reasoners]`/`[actioners]` sections to match what
you actually have pulled — the checked-in list only reflects models already
referenced elsewhere in configs, not what's on your machine. Pull anything
missing:

```bash
ollama pull qwen3.5:9b
```

Sanity-check the roster parses correctly:

```bash
python experiments/model_roster.py
python experiments/model_roster.py path/to/other_models.txt
```

Prints the parsed `{reasoners: [...], actioners: [...]}` plus the total
combination count.

## Running a single instance

```bash
python swebench/run_instance.py --instance django__django-12345
python swebench/run_instance.py --instance django__django-12345 --config configs/qwen_ornith.yaml
```

Runs the full pipeline for one instance: load → set up repo/env → apply
`test_patch` → run the agent → evaluate → print status/runtime/test results.
Repo must already be cloneable/reachable (network egress must allow
`github.com`); this does not use `benchmarks.SWEbenchBenchmark`, it's a
standalone script using `swebench/utils.py` directly.

## Running a full experiment (one reasoner/actioner pair)

```bash
python experiments/run_experiment.py --config configs/qwen_ornith.yaml
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/qwen_ornith.yaml` | YAML config for agent/timeout settings |
| `--limit N` | none (all instances) | Cap total instances run |
| `--reasoner-model` | config's `reasoner.model` | Override reasoner model id |
| `--actioner-model` | config's `actioner.model` | Override actioner model id |
| `--run-name` | config file stem | Summary filename stem (needed when reusing one base config across multiple model pairs, so summaries don't overwrite each other) |
| `--repo-filter SUBSTR` | none | Only run instances whose `repo` contains this substring, e.g. `requests` |
| `--pure-python-only` | off | Restrict to `PURE_PYTHON_REPOS` (django, flask, requests, pylint, pytest, sphinx, sympy) — skips numpy/scipy/pandas/matplotlib-style C-extension builds |
| `--instances-per-repo N` | none | **Overrides `--limit`.** Selects up to N instances from *each* repo after filtering, instead of the first N overall |

Examples:
```bash
# Quick smoke test: 3 instances, cheap repos only
python experiments/run_experiment.py --limit 3 --pure-python-only

# One instance per pure-Python repo, specific model pair
python experiments/run_experiment.py \
  --instances-per-repo 1 --pure-python-only \
  --reasoner-model qwen3.5:9b --actioner-model ornith:9b \
  --run-name qwen_ornith_pilot

# Only requests instances
python experiments/run_experiment.py --repo-filter requests --limit 10
```

Output:
- Per-instance raw records → `results/raw/results.jsonl` (append-only)
- Running summary after every instance (crash-safe) + final summary →
  `results/processed/<run_name-or-config-stem>.json`

## Grid search (every reasoner × actioner combination)

```bash
python experiments/run_grid_search.py --dry-run
python experiments/run_grid_search.py --limit 5
```

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `models.txt` | Roster file |
| `--config` | `configs/qwen_ornith.yaml` | Base config for shared settings (models.txt still decides which reasoner/actioner models are actually tried) |
| `--limit N` | none | Instances per combination — keep small (3–10) for an initial sweep |
| `--dry-run` | off | Print combinations and exit without running anything |
| `--repo-filter SUBSTR` | none | Passed through to `run_experiment` |
| `--pure-python-only` | off | Passed through to `run_experiment` |

Always `--dry-run` first when editing `models.txt` — N reasoners × M
actioners × your `--limit` is N×M full agent runs, and it's easy to
underestimate the total.

Output: one summary per combo (`results/processed/grid_search/<reasoner>__<actioner>.json`)
plus a ranked `results/processed/grid_search/leaderboard.json`
(best resolve_rate first; ties broken by fewer errors, then faster runtime),
plus a printed text table.

## Diagnostic pilot (why are models failing, not how well)

```bash
python experiments/pilot_run.py
python experiments/pilot_run.py --instances-per-repo 2 --all-repos
python experiments/pilot_run.py --repo-filter requests
```

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `models.txt` | Roster file |
| `--config` | `configs/qwen_ornith.yaml` | Base config |
| `--instances-per-repo N` | `1` | Instances per repo per combination |
| `--repo-filter SUBSTR` | none | Restrict to one repo |
| `--all-repos` | off | Include C-extension-heavy repos too (default: pure-Python only) |

Run this **before** trusting a grid-search resolve rate on small local
models. It breaks results down by `exit_reason`/`stop_reason` per repo and
per model pair — a model failing mostly with `repeated_action`/
`reasoner_failed` has a context-window/capability problem no repo choice
will fix; a model failing mostly with `environment_error` is hitting infra
tax that `--pure-python-only` should already remove.

Output: `results/processed/pilot/PILOT_REPORT.json` + a console report.

## Overnight / unattended runs

```bash
python experiments/overnight_loop.py --until 07:00 --limit 3
python experiments/overnight_loop.py --hours 8 --limit 3
```

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `models.txt` | Roster file |
| `--config` | `configs/qwen_ornith.yaml` | Base config |
| `--limit N` | `3` | Instances per combination — keep small, this runs unattended |
| `--until HH:MM` | — | Stop by this clock time (today if still ahead, else tomorrow) |
| `--hours N` | — | Or: stop after N hours from now |
| `--force` | off | Re-run combinations that already have a saved summary |
| `--repo-filter SUBSTR` | none | Passed through |
| `--pure-python-only` | off | Passed through |

Exactly one of `--until` / `--hours` is required. Safe to interrupt and
re-run: already-completed combinations (a summary already exists under
`results/processed/grid_search/`) are skipped unless `--force`. Stops
*before* starting a new combination once the estimated time for one more
(based on the last combo's actual duration + a safety margin) exceeds the
remaining budget — it will not get killed mid-write.

Output: same per-combo summaries as grid search, plus
`results/processed/grid_search/overnight_log.jsonl` (event log) and
`results/processed/grid_search/MORNING_DIGEST.md` (human-readable ranked
table + crashed combinations + anything skipped due to the deadline).

## Re-evaluating an instance after the fact

```bash
python swebench/evaluate.py --instance django__django-12345
```

Loads instance metadata, captures whatever git diff currently exists at
`seed_repos/<repo>` (or wherever the agent left it), installs/reuses the
repo's venv, and runs the official FAIL_TO_PASS/PASS_TO_PASS check (or the
generic `test_command` fallback if no gold lists exist). Useful for
re-scoring without re-running the agent.

## Repo/environment maintenance (usually called internally, but runnable directly)

```bash
python swebench/setup_repos.py <repo_name> <base_commit>
```

Clones (if missing), checks out `base_commit`, and creates an isolated
`git worktree` workspace — prints the workspace path. Most pipeline scripts
call this indirectly; run it directly only for debugging a specific repo's
setup.

## Environment notes

- Every command that touches a repo's dependencies goes through
  `ensure_repo_environment` (`swebench/utils.py`), which creates a
  **per-repo** venv under `<repo_path>/.swebench_venv/`. First run per repo
  is slow (C-extension builds for astropy/numpy-adjacent repos can take
  several minutes); subsequent runs reuse it. If installs are failing, check
  `.swebench_venv/pip_install.log` before assuming the agent is at fault —
  `ensure_repo_environment` returns `install_ok=False` and callers mark the
  instance `environment_error` distinctly from a genuine agent failure.
- `--pure-python-only` (django/flask/requests/pylint/pytest/sphinx/sympy)
  is the fast way to sidestep that build pain entirely for early iteration.