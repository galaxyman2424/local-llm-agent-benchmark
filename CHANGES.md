# Architecture fixes applied

Package layout (imports now match on disk):

```
agents/         ollama_client.py, reasoner.py, actioner.py, agent.py, tool_schemas.py, json_utils.py, __init__.py
benchmarks/     swebench.py, swebench_lite.py, __init__.py
experiments/    run_experiment.py, run_grid_search.py, model_roster.py, config/agent_config.py, __init__.py
swebench/       utils.py, run_instance.py, evaluate.py, setup_repos.py, download_dataset.py, fetch_swebench_lite.py
configs/        single_model.yaml, qwen_ornith.yaml, qwen_deepseek.yaml
```

## 1–2. Reasoner/Actioner separation & agent loop
`agents/agent.py` now runs the exact loop shape requested: Reasoner.plan()
→ stop-condition check (`"done"`/`"submit_solution"`) → Actioner.plan_action()
→ schema validation → Actioner.execute() → history/state update → test check.
Final patch capture and status computation happen **after** the loop, never
inside it.

## 3. AgentResult consistency
`AgentResult` is a `@dataclass` in `agents/agent.py`. `Agent.solve()` always
returns one. `benchmarks/swebench.py` and `benchmarks/swebench_lite.py` read
`agent_result.num_iterations` / `.total_tool_calls` / `.final_patch` /
`.status` as dataclass attributes — no more `dict` vs. dataclass mismatch.
Use `dataclasses.asdict(result)` when JSON-serializing.

## 4–5. `status` assignment & vocabulary
`status` is computed exactly once, immediately before constructing the
`AgentResult`, using only `passed / failed / incomplete / timeout / error`
(the `error` value is reserved for the crash path in `Actioner.execute`
callers). The benchmark layer (`InstanceRecord.status`) keeps its own
separate vocabulary (`resolved / not_resolved / patch_failure / timeout /
environment_error`) and maps the Agent's `passed/failed` onto it explicitly
in `swebench_lite.py` — the two vocabularies are never mixed in one object.

## 6. `num_iterations` vs `total_tool_calls`
Tracked as two independent counters in the loop (`num_iterations` counts
Reasoner→Actioner cycles including ones with no action produced;
`total_tool_calls` only increments after a successful `execute()` call).

## 7, 10–11, 13. JSON reliability + tool schemas
New `agents/tool_schemas.py` is the single source of truth for tool
parameter requirements (`TOOL_SCHEMAS`, `validate_action`,
`schema_prompt_block`). Both the Reasoner's planning prompt and the
Actioner's translation prompt render from this same table, and
`Agent.solve()` validates every action before executing it — invalid
actions (e.g. `read_file` with a `query` instead of a `path`) are logged
and skipped rather than executed or silently coerced.

`agents/json_utils.py` holds one shared, brace-depth-aware JSON extractor
used by both Reasoner and Actioner (replacing the old greedy-regex
approach and the accidental duplicate implementations).

Both `Reasoner._call_model` and `Actioner.plan_action` parse only
`message["content"]`, never `message["thinking"]`.

## 8–9. Actioner hallucination / path handling
`Actioner.plan_action`'s prompt now states the fixed `workspace_dir`
explicitly, forbids inventing other paths/projects, and asks for
workspace-relative paths only. `Actioner._resolve_path` (the deterministic
executor) is solely responsible for anchoring relative paths to
`self.workspace_dir`.

## 12, 32. Actioner scope
The Actioner's prompt makes explicit that it must translate the Reasoner's
`next_action`/`parameters` into ONE schema-valid tool call and must not
invent new goals, extra steps, or unrelated projects.

## 14–15. `OllamaClient` consistency
`chat()` now actually forwards `think` into the Ollama payload (it was
accepted as a parameter but silently dropped before). `keep_alive` was
already supported and remains so. Both `Reasoner` and `Actioner` import
the exact same `agents/ollama_client.OllamaClient`.

## 16–18. Empty content / thinking budget / retry / loop detection
`Reasoner._call_model` always sends `think=False` + `format=json`. On
empty content, `Reasoner.plan` retries once with a shorter prompt before
giving up (returns `None`, never spins forever). `Agent.solve` tracks the
last concrete `(tool, parameters)` action; after `MAX_REPEATED_ACTIONS`
identical repeats it stops the loop. The Reasoner's prompt also receives
an explicit "you are stuck, choose differently" warning once repetition
is detected (`_loop_warning` in `reasoner.py`).

## 19–20. Feedback quality & output truncation
`previous_actions` passed into `Reasoner.plan` always contains the actual
`result` dict (stdout/stderr/returncode/error), not just action names.
Anything embedded back into a prompt is truncated to `MAX_OUTPUT_CHARS`
(8000 chars); full untruncated output is still kept in the in-memory
history for benchmark logs.

## 21–23. Test environment
`swebench/evaluate.py` now calls `ensure_repo_environment` before running
any tests (previously it ran raw `/bin/sh -c test_cmd` against the
ambient interpreter), and passes the resulting venv `python_bin` into
`evaluate_fail_to_pass`. `run_instance.py` / `swebench.py` /
`swebench_lite.py` already did this and are unchanged. Astropy's
`setup.cfg` / `addopts = --doctest-rst` is left untouched; the fix is
installing `pytest-astropy`-family dependencies into the isolated venv
instead (see `ensure_repo_environment` in `swebench/utils.py`).

## 24. Test command plumbing
`current_state["test_command"]` is set at the start of `Agent.solve` so
the Reasoner can see the benchmark-provided command rather than the
Actioner inventing its own.

## 25–27. Evaluation & final patch/status ordering
Benchmark-level resolution (`resolved`/`not_resolved`) still comes from
`evaluate_fail_to_pass`, never from "diff is non-empty". Final patch
capture (`get_git_diff`) and status computation happen strictly after the
Agent's loop, in that order, before constructing `AgentResult`.

## 28–29. `run_command` / `search_code`
`run_command` remains a trusted-local-execution helper scoped to
`workspace_dir` (documented as such). `Actioner._search_code` now prefers
`git grep -n -I --untracked` (works across `.py/.rst/.cfg/.yaml/.toml/.md`
etc., respects `.gitignore`) and falls back to `grep -r` with explicit
`--exclude-dir` for `.git/.swebench_venv/__pycache__/build/dist/node_modules`
when the workspace isn't a git repo.

## 30–31. Search/inspection guidance
The Reasoner's prompt explicitly tells it to extract concrete identifiers
(function/class/file names) for `search_code` queries instead of pasting
the whole problem statement, and to prefer search → read → edit → test
over jumping straight to a write.

## 33–36. Grid search
`run_grid_search.py`'s per-combination failure summary now always
includes `errors: 1` and `run_time_seconds: 0.0` so leaderboard sorting
doesn't silently misrank crashed combinations against the default `0`.
`run_experiment.py` now reads **separate** `reasoner.timeout_seconds` /
`actioner.timeout_seconds` config keys for per-LLM-request timeouts,
distinct from `agent.timeout` (overall task budget) — see the updated
YAML configs under `configs/`.

## 37. `max_iterations`
Left at the existing per-config values (10/50); not blindly raised, per
the request — the reliability fixes above should be validated first.

## Removed
`submit_solution` was removed from `TOOL_SCHEMAS` / the Actioner's
executable tools since it was a no-op placeholder that could make the
Agent falsely claim a submission occurred. The benchmark's own
`get_git_diff` + evaluation is the actual submission/scoring mechanism.
