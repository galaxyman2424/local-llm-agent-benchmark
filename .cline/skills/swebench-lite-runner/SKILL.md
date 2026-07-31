---
name: swebench-lite-runner
description: Use when running, debugging, or interpreting SWE-bench Lite instance or grid-search runs in this project -- covers swebench/run_instance.py, experiments/run_experiment.py, experiments/run_grid_search.py, dataset download, per-repo venv setup, and how to read status/test_results/leaderboard output. Trigger on mentions of SWE-bench, run_instance, run_experiment, grid search, FAIL_TO_PASS/PASS_TO_PASS, environment_error, or a specific instance_id like "django__django-12345".
---

# SWE-bench Lite Runner

## Getting the dataset
If `seed_repos/swe_bench_lite/dataset.jsonl` is missing:
```bash
pip install datasets --break-system-packages
python swebench/fetch_swebench_lite.py
```
Full detail (including huggingface_hub and official-harness alternatives) is
in `docs/DOWNLOADING_SWEBENCH_LITE.md`.

## Running one instance
```bash
python swebench/run_instance.py --instance <instance_id> --config configs/qwen_ornith.yaml
```
This: loads the instance → resets/checks out the repo at `base_commit` →
creates an isolated `git worktree` workspace → ensures a per-repo venv
(`ensure_repo_environment`) → runs the Agent → captures the final patch →
evaluates via FAIL_TO_PASS/PASS_TO_PASS when the instance provides them,
else a plain test_command check.

## Running an experiment / grid search
```bash
python experiments/run_experiment.py --config configs/qwen_ornith.yaml --limit 1   # smoke test first
python experiments/run_grid_search.py --dry-run                                     # see combinations, no runs
python experiments/run_grid_search.py --limit 3                                     # small real pass
```
`run_grid_search.py` builds every `[reasoners] x [actioners]` pair from
`models.txt`, reuses `run_experiment()` per pair, and writes a ranked
`leaderboard.json` sorted by `resolve_rate` desc, then `errors` asc, then
`run_time_seconds` asc.

## Reading status
Two DIFFERENT status vocabularies exist -- never assume they're the same field:

| Layer | Values |
|---|---|
| `AgentResult.status` (agents/agent.py) | `passed / failed / incomplete / timeout / error` |
| Benchmark `InstanceRecord.status` / grid summaries | `resolved / not_resolved / patch_failure / timeout / environment_error` |

`environment_error` means the repo's own dependency install failed (see
`.swebench_venv/pip_install.log` under the repo's cache dir) -- this is NOT
"the agent failed the task"; the FAIL_TO_PASS check was never meaningful
for that instance. Report it separately from a genuine `not_resolved`.

`patch_failure` means the instance's `test_patch` itself couldn't be
applied (an environment/setup problem, not the agent's fault).

## Common failure modes and what they actually mean
- **Empty/truncated JSON from the Reasoner or Actioner** -- almost always a
  `num_ctx` (context window) problem, not a model capability problem. See
  the `ollama-json-reliability` skill.
- **`environment_error` on an astropy/scikit-learn/etc. instance** -- a
  missing system header or wrong Python version for building C extensions.
  `swebench/utils.py::_diagnose_build_failure` already recognizes the
  common cases and prints a one-line fix hint (e.g.
  `sudo apt install pythonX.Y-dev`) -- check the console output/log before
  guessing.
- **A combination in the grid search leaderboard shows `errors: 1` with
  `resolve_rate: 0.0`** -- this is a CRASHED combination
  (`run_grid_search.py`'s except-block placeholder), not a combination that
  genuinely resolved 0 instances. Check its `error`/`traceback` field.

## Rule of thumb before scaling up
Always verify with `--limit 1`, then `--limit 5`, then `--limit 10` before a
full run (see `ACTIONPLAN.md` section 14) -- a stuck/misconfigured agent
burns a lot of wall-clock time per instance.
