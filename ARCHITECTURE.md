# ARCHITECTURE.md

Local-model (Ollama) SWE-bench Lite benchmark harness with a Reasoner/Actioner
agent split. This doc covers module boundaries and data flow; see `AGENT.md`
for the internals of the agent loop itself.

## High-level flow

```
models.txt ──► experiments/model_roster.py ──► {reasoners: [...], actioners: [...]}
                                                        │
                                                        ▼
                                     experiments/run_grid_search.py
                                     experiments/pilot_run.py
                                     experiments/overnight_loop.py
                                                        │  (per reasoner×actioner combo)
                                                        ▼
                                     experiments/run_experiment.py
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        ▼                                ▼                               ▼
              configs/*.yaml (settings)        agents.Reasoner / Actioner       benchmarks.SWEbenchBenchmark
                                                          │                               │
                                                          ▼                               ▼
                                                   agents.Agent.solve() ◄──────── benchmark.run_instance()
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              ▼                                                         ▼
                    Reasoner.plan() [LLM]                                    Actioner.plan_action() [LLM]
                    "what should happen next"                                "translate to 1 concrete tool call"
                                                                                         │
                                                                                         ▼
                                                                              Actioner.execute() [no LLM]
                                                                              (deterministic tool runner)
                                                                                         │
                                                                                         ▼
                                                                        swebench.utils (git, pytest, venvs)
```

## Package responsibilities

### `agents/`
The agent itself — model-agnostic loop logic. See `AGENT.md` for details.

- `ollama_client.py` — single HTTP client for Ollama's `/api/chat` and
  `/api/embeddings`. Both Reasoner and Actioner import this same class
  (not duplicated). Key knobs: `num_predict` vs `num_ctx` (the latter is the
  actual fix for mid-JSON truncation, not the former), `think=False` to
  suppress hidden reasoning traces from eating the token budget, and
  `keep_alive=0` to unload models between grid-search combinations on
  limited VRAM.
- `json_utils.py` — balanced-brace JSON extraction + truncation repair.
  Shared so Reasoner and Actioner never drift into two different parsers.
- `tool_schemas.py` — the single source of truth for tool names/params
  (`read_file`, `write_to_file`, `replace_in_file`, `replace_lines`,
  `delete_file`, `list_directory`, `search_code`, `run_command`,
  `run_tests`, `get_git_diff`). Both the Reasoner's prompt
  (`schema_prompt_block()`) and the Actioner's validator (`validate_action`)
  read from here.
- `reasoner.py` — plans the next action; never executes.
- `actioner.py` — translates a plan into one tool call, then executes it
  deterministically; enforces a workspace sandbox (`_resolve_path` refuses
  paths that escape `workspace_dir`).
- `agent.py` — the loop: caching, loop/oscillation detection, stop
  conditions, and the live FAIL_TO_PASS/PASS_TO_PASS exit check.

### `swebench/`
SWE-bench Lite specific plumbing — dataset acquisition, repo/environment
setup, and official-methodology evaluation. Independent of the `agents/`
package's internals; only imported *by* higher layers.

- `download_dataset.py` — pulls SWE-bench Lite from Hugging Face (or a
  direct URL) into `seed_repos/swe_bench_lite/dataset.jsonl`. Run once.
- `fetch_swebench_lite.py` — a simpler/older standalone variant of the same
  download step (kept for ad hoc use; `download_dataset.py` is the one
  wired into the pipeline scripts).
- `setup_repos.py` — clones a repo if missing, checks out a base commit,
  and creates an isolated `git worktree` per instance so one task's edits
  can't contaminate another.
- `utils.py` — the largest module in the package. Houses:
  - dataset/instance loading (`load_dataset`, `load_instance`)
  - repo/workspace helpers (`create_workspace`, `reset_repository`, git diff/patch)
  - **`ensure_repo_environment`** — creates (or reuses) a per-repo venv,
    editable-installs the package, and papers over a long list of
    SWE-bench-Lite-era build issues (missing declared build requirements,
    a `setuptools==65.5.0` pin because newer setuptools dropped
    `dep_util`, an old-Python fallback search since distutils was removed
    in 3.12, submodule fetching for vendored C libs, `pytest<8` +
    `hypothesis` + `pytest-remotedata` as commonly-missing test deps).
    This function is where most "why did every test fail regardless of
    the agent" incidents come from — check `.swebench_venv/pip_install.log`
    first.
  - `PURE_PYTHON_REPOS` / `filter_pure_python_instances` — a 7-repo subset
    (django, flask, requests, pylint, pytest, sphinx, sympy) with no
    numpy/scipy/pandas/matplotlib-style C-extension chain, for cheap fast
    iteration.
  - `evaluate_fail_to_pass` / `run_test_ids` — the official-methodology
    scorer: apply `test_patch`, run each FAIL_TO_PASS/PASS_TO_PASS node id
    **individually** (more robust than parsing one combined pytest run's
    output across differently-configured repos), and require *all*
    FAIL_TO_PASS to pass and *all* PASS_TO_PASS to still pass.
- `run_instance.py` — CLI: run one instance end-to-end using a YAML config.
- `evaluate.py` — standalone re-evaluation of an instance after the fact
  (given `repo_path`, captures the diff and re-runs the official check).

### `benchmarks/`
Two parallel benchmark interfaces exist — **know which one is actually wired
in** before changing either:

- `benchmarks/swebench.py` (`SWEbenchBenchmark`, `InstanceRecord`) — **this
  is the one `experiments/run_experiment.py` actually uses.** Dataclass-based
  raw records, `save_raw`/`save_processed` write JSONL + a summary JSON with
  full history embedded.
- `benchmarks/swebench_lite.py` (`SWEBenchLite`, `BenchmarkResult`) — a
  dict-based alternative interface with its own `filter_instances`/
  `select_instances`, and its own `--repo-filter` support. Not currently
  called from any `experiments/*.py` entry point. If you're adding
  repo/instance filtering, double check you're not fixing the unused one.

### `configs/`
- `agent_config.py` — a small `AgentConfig` dataclass (defaults reasoner ==
  actioner model unless overridden). Lighter-weight than the YAML configs;
  check current call sites before assuming it's on the main path.
- `qwen_ornith.yaml` (and siblings) — the actual configs consumed by
  `run_experiment.py`'s `load_config`. Sections: `reasoner` (model,
  timeout_seconds, num_ctx), `actioner` (model, timeout_seconds, num_ctx,
  max_read_lines), `agent` (max_iterations, timeout — the **whole-task**
  budget, distinct from the per-request timeouts above it).

### `experiments/`
Orchestration layer — nothing here talks to Ollama or git directly; it all
delegates to `run_experiment.run_experiment()`.

- `model_roster.py` — parses `models.txt`'s `[reasoners]`/`[actioners]`
  sections into a `{"reasoners": [...], "actioners": [...]}` dict.
- `run_experiment.py` — the one real entry point. Loads a YAML config,
  builds `Reasoner`/`Actioner`/`Agent`, iterates selected instances via
  `SWEbenchBenchmark`, writes an in-progress summary after *every* instance
  (`_write_running_summary`) so a crash mid-run still leaves usable partial
  results, then a final aggregated summary keyed by `run_name` (or the
  config file's stem).
- `run_grid_search.py` — cartesian product of every reasoner × actioner in
  `models.txt`, each run via `run_experiment`, ranked into
  `leaderboard.json` (best resolve_rate first, ties broken by fewer errors
  then faster runtime).
- `pilot_run.py` — diagnostic, not scoring: `instances_per_repo` (default 1)
  across the full roster, broken down by `exit_reason`/`stop_reason` per
  repo and per model combo. Meant to answer "is this an infra problem or a
  model-capability problem" before trusting a resolve rate at all.
- `overnight_loop.py` — time-boxed wrapper around the same
  `run_experiment` building blocks: stops *before* starting a new
  combination if there isn't enough time left (estimated from the last
  combo's actual duration + a safety margin), skips already-completed
  combinations unless `--force`, and survives a per-combination crash
  without ending the whole run. Produces `MORNING_DIGEST.md`.

## Data flow for a single instance

1. `run_experiment` loads the dataset (optionally filtered:
   `pure_python_only`, `repo_filter`, `instances_per_repo`).
2. `SWEbenchBenchmark.setup_repo` clones/checks out the repo at the
   instance's `base_commit`.
3. `ensure_repo_environment` creates/reuses the repo's venv and
   editable-installs the package (`install_ok` gates everything downstream —
   an environment failure is recorded as `environment_error`, distinct from
   the agent failing to fix the bug).
4. The instance's `test_patch` (reference tests only, never the fix) is
   applied *before* the agent runs, so both the agent's own `run_tests`
   calls and the final scoring target the real FAIL_TO_PASS/PASS_TO_PASS
   node ids.
5. `Agent.solve` runs the Reasoner/Actioner loop (see `AGENT.md`) until a
   stop condition fires.
6. The final patch is captured via `get_git_diff`; `evaluate_fail_to_pass`
   (or the returncode fallback, if no gold test lists exist) produces the
   authoritative `status`.
7. `InstanceRecord` is appended to `results/raw/results.jsonl`; aggregated
   stats go to `results/processed/summary.json`.

## Result vocabularies (don't mix these across layers)

- **Agent-level** (`AgentResult.status`): `passed` / `failed` / `incomplete`
  / `timeout` / `error`.
- **Benchmark-level** (`InstanceRecord.status`, from `evaluate_fail_to_pass`):
  `resolved` / `not_resolved` / `patch_failure` / `timeout` /
  `environment_error`.
- **`stop_reason`** / **`exit_reason`** (agent-internal diagnostics): see
  `AGENT.md`'s stop-condition table.

`benchmarks/swebench_lite.py`'s `run_instance` does map agent status onto
benchmark vocabulary (`{"passed": "resolved", "failed": "not_resolved"}`) —
but again, that module isn't the one currently wired into the pipeline.

## Known duplication / drift risks

- `find_repository` (`swebench/utils.py`) is aliased as `find_repo_path` for
  callers that import it under that name — both names resolve to the same
  function; don't add a second implementation.
- `experiments/run_grid_search.py`'s default `config_path` argparse default
  (`configs/qwen_ornith.yaml`) and its `run_grid_search()` function default
  (`configs/qwn_ornith.yaml`) are spelled differently (missing an `e`) —
  worth fixing so a bare function call doesn't silently 404 on a config
  file.
- Two benchmark interfaces exist (`benchmarks/swebench.py` vs
  `benchmarks/swebench_lite.py`) — see `benchmarks/` section above.