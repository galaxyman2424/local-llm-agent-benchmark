# Architecture Contract — Reasoner / Actioner / Agent

This project enforces a strict separation between three layers. Do not blur
them when adding features or fixing bugs.

## 1. Reasoner decides WHAT, Actioner decides HOW
- `agents/reasoner.py` (`Reasoner.plan`) may only choose a `next_action` name
  and suggested `parameters`. It never calls a tool and never touches the
  filesystem/subprocess directly.
- `agents/actioner.py` (`Actioner.plan_action`) only translates the
  Reasoner's choice into one schema-valid `{"tool": ..., "parameters": {...}}`
  call. It must not invent a different goal, tool, or path than what the
  Reasoner asked for.
- `Actioner.execute` is the ONLY place that is purely deterministic (no LLM
  call) and actually touches disk/subprocess. Never add an LLM call inside
  `execute`.

## 2. Tool schemas are single-sourced
- `agents/tool_schemas.py::TOOL_SCHEMAS` is the one place that defines every
  tool's required/optional parameters. Both the Reasoner's prompt
  (`schema_prompt_block()`) and `Actioner.execute`'s dispatch must stay in
  sync with it.
- Adding a new tool means: add it to `TOOL_SCHEMAS`, implement its branch in
  `Actioner.execute`, and validate every call through
  `tool_schemas.validate_action` before executing. Never execute an
  unvalidated action.

## 3. Agent loop order is fixed
`agents/agent.py::Agent.solve` loop shape per iteration:
Reasoner.plan → check `STOP_ACTIONS` (`"done"`, `"submit_solution"`) →
Actioner.plan_action → `validate_action` → Actioner.execute → append to
`previous_actions` (with the REAL result, not just the action name) →
update `current_state` → check for `run_tests` result.

`final_patch` (via `get_git_diff`) and the final `status` are computed
**exactly once, after the loop ends** — never inside the loop, and never
referenced before that point.

## 4. Status vocabularies never mix
- `AgentResult.status` (agents/agent.py) uses only:
  `passed / failed / incomplete / timeout / error`
- `InstanceRecord.status` / benchmark-level records (benchmarks/*.py) use:
  `resolved / not_resolved / patch_failure / timeout / environment_error`
- When one layer reports into the other, map explicitly
  (`{"passed": "resolved", "failed": "not_resolved"}` style dict) — never
  reuse a bare string across both vocabularies.

## 5. Loop / repetition safety
- `MAX_REPEATED_ACTIONS` in `agents/agent.py` stops the loop if the exact
  same `(tool, parameters)` repeats too many times in a row. Don't remove
  this without adding an equivalent safeguard — local models get stuck in
  literal repeat loops in practice.
- `agents/reasoner.py::_loop_warning` injects an explicit "you are stuck"
  message into the Reasoner's prompt after `REPEATED_ACTION_THRESHOLD`
  repeats. Keep this in sync if you change the loop-detection window.

When implementing a change, check whether it belongs in Reasoner (decides
what), Actioner (decides how / executes), or Agent (orchestrates) before
writing code — don't add reasoning logic to the Actioner or execution logic
to the Reasoner.
