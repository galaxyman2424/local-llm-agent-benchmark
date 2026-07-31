# Project & Benchmark Conventions

## Directory layout (keep imports matching disk)
```
agents/         ollama_client.py, reasoner.py, actioner.py, agent.py,
                tool_schemas.py, json_utils.py
benchmarks/     swebench.py, swebench_lite.py
experiments/    run_experiment.py, run_grid_search.py, model_roster.py
swebench/       utils.py, run_instance.py, evaluate.py, setup_repos.py,
                download_dataset.py, fetch_swebench_lite.py
configs/        *.yaml (name/reasoner/actioner/agent/benchmark keys)
```
Don't create a second copy of a module under a different path (this project
previously had duplicate `agent_config.py` under both `configs/` and
`experiments/config/` — avoid reintroducing that kind of drift).

## Results are append-only
- `results/raw/` is written with `save_raw(..., "a")` — NEVER overwritten.
  Treat this as the experimental record.
- `results/processed/` holds aggregated summaries and CAN be regenerated.

## Per-repo virtualenvs, not the ambient interpreter
- Tests must always run through `swebench/utils.py::ensure_repo_environment`,
  which creates/reuses `<repo_path>/.swebench_venv` and returns
  `(python_bin, install_ok)`. Never call `subprocess.run(["pytest", ...])`
  against the ambient interpreter — the target repo's own dependencies
  (and often an older Python, since many SWE-bench-era C-extension packages
  don't build on 3.12+) won't be present.
- Always check `install_ok` before treating a test failure as "the agent
  didn't fix the bug" — a failed dependency install is recorded as its own
  `environment_error` status, distinct from `not_resolved`.

## FAIL_TO_PASS / PASS_TO_PASS is the real evaluation signal
- An instance only counts as `resolved` if every FAIL_TO_PASS test passes
  AND every PASS_TO_PASS test still passes (`evaluate_fail_to_pass` in
  `swebench/utils.py`). "The git diff is non-empty" is never sufficient.
- Only fall back to a plain test_command return-code check when the
  instance genuinely has no FAIL_TO_PASS/PASS_TO_PASS lists (e.g. a
  synthetic local task).

## Config files
- `configs/*.yaml` always separate `reasoner.timeout_seconds`,
  `actioner.timeout_seconds`, and `agent.timeout` — see
  `02-ollama-conventions.md`.
- `models.txt` is the roster consumed by `experiments/model_roster.py` for
  `run_grid_search.py`. Format is `[reasoners]` / `[actioners]` sections,
  one Ollama model id per line. It only lists models already referenced
  elsewhere in the project — always run `ollama list` and edit it to match
  what's actually pulled locally before a grid search.

## Removed / rejected patterns (don't reintroduce)
- No `submit_solution` tool — it was a no-op that could let the Agent
  falsely claim a submission occurred. `get_git_diff` + the benchmark's own
  evaluation is the only real submission/scoring mechanism.
- Don't run the full benchmark before a single-instance and small-limit
  (`--limit 1`, then `--limit 5`, then `--limit 10`) run has been verified
  stable, per `ACTIONPLAN.md` section 14.
