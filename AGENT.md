# AGENT.md

Guide to the agent loop itself: `agents/agent.py`, `agents/reasoner.py`,
`agents/actioner.py`. Read this before touching any of those three files —
the split between them is load-bearing and easy to accidentally blur.

## The core separation

The agent is deliberately split into two LLM roles plus one deterministic
executor:

- **Reasoner** (`agents/reasoner.py`) — decides **WHAT** should happen next.
  Never executes anything. Returns `{next_action, parameters, expected_outcome}`.
- **Actioner** (`agents/actioner.py`) — translates the Reasoner's plan into
  **exactly one** concrete, schema-valid tool call (`plan_action`), then
  executes it deterministically (`execute`). `execute` never calls an LLM.
- **Agent** (`agents/agent.py`) — orchestrates the loop: Reasoner → Actioner
  → execute → feed result back → repeat, until a stop condition fires.

Do not let `execute` make planning decisions, and do not let `plan`/`plan_action`
run side effects. If you're tempted to do either, the fix is almost always
to move logic into `Agent.solve` instead.

## Stop conditions (`STOP_REASONS` in agent.py)

| stop_reason | meaning |
|---|---|
| `tests_passed` | FAIL_TO_PASS/PASS_TO_PASS (or fallback returncode) confirmed passing |
| `reasoner_done` | Reasoner explicitly chose `"done"`/`"submit_solution"` |
| `reasoner_failed` | Reasoner returned no usable plan, even after its own retry |
| `actioner_failed` | Actioner couldn't produce a valid tool call, and the Reasoner's own plan wasn't directly valid either |
| `repeated_action` | Same (tool, params) fired `MAX_REPEATED_ACTIONS` times, or an A-B-A-B oscillation was detected |
| `max_iterations` | Ran out of budget (default) |

`AgentResult.status` (`passed`/`failed`/`incomplete`/`timeout`/`error`) is a
**separate** vocabulary from `stop_reason` — don't conflate them. Benchmarks
map `status` onto their own `resolved`/`not_resolved` terms; `stop_reason`
stays internal-diagnostic.

## Exit condition = what gets scored

`Agent.solve` accepts `fail_to_pass_tests` / `pass_to_pass_tests` (the
instance's real gold test node ids). When provided, these are BOTH:

1. Stated to the Reasoner as the literal goal (`goal_block` in the prompt), and
2. Re-checked live after every `run_tests` call via `_check_target_tests`
   (which calls `swebench.utils.run_test_ids` — the same runner
   `evaluate_fail_to_pass` uses at scoring time).

This means the loop's own "am I done" signal is the same thing that gets
scored later. **Never** replace this with "trust the returncode of whatever
command the model ran" — that was the bug fixed earlier (see
`SESSION_SUMMARY.md`), and it's easy to reintroduce by adding a new stop
path that checks `returncode == 0` directly.

When no gold test lists exist (synthetic/local task), the loop falls back to
trusting `test_command`'s returncode. Keep that fallback — don't require
FAIL_TO_PASS/PASS_TO_PASS unconditionally.

## State the Reasoner sees each iteration

`current_state` (built in `Agent.solve`) carries, among other things:

- `file_cache` — full or chunked file contents already read (see
  `_update_file_cache`, `_merge_cache_entry`). Read-modify-invalidate:
  writes/replaces invalidate the cached entry for that path.
- `files_read` — a **persistent** manifest (survives past the rolling
  last-5-actions window) tracking last-read iteration, last-modified
  iteration/tool, and a `stale_since_last_read` flag when a file was
  modified after being read. The Reasoner prompt surfaces this explicitly
  so it knows not to trust stale line numbers.
- `listing_cache` — same idea for `list_directory`.
- `target_tests_passing` — authoritative live FAIL_TO_PASS/PASS_TO_PASS result.

If you add a new piece of persistent state, prefer the "manifest that
survives the rolling window" pattern over stuffing more into
`previous_actions`, since only the last 5 actions are rendered in the
Reasoner prompt (`_format_previous_actions(..., last_n=5)`).

## Redundant-call short-circuiting

`Agent.solve` intercepts (before execution) two cases and feeds back the
**cached** result instead of re-running anything:

- `_is_cached_read` — a `read_file` whose exact range (or full-file) is
  already in `file_cache`.
- `_is_cached_listing` — a `list_directory` whose path is already in
  `listing_cache`.

Both `continue` the loop without incrementing `total_tool_calls`. If you add
a new cacheable tool, follow this pattern rather than relying solely on the
repeated-action detector below — small local models re-issue identical reads
far more than they oscillate, and catching it here saves real iterations.

## Loop-avoidance layers (in order of severity)

1. `avoid_action` / `stagnation_hint` — soft: told to the Reasoner as a
   constraint once the same action has fired once already.
2. `MAX_REPEATED_ACTIONS` (4) — hard stop on the exact same (tool, params)
   key repeating.
3. Oscillation window (`action_window`, period 1/2/3) — catches A-B-A-B
   cycles the immediate-repeat check would miss.

## Truncation handling (`agents/json_utils.py`)

Small local models routinely get cut off mid-JSON by `num_ctx`, not
`num_predict` (see the long docstring in `ollama_client.py` — this is a
common enough footgun to repeat here: **`num_ctx` is prompt + generation
combined**, and Ollama's default is small). `extract_json_object` finds the
first *balanced* `{...}` (not a greedy first-`{`-to-last-`}` match, which
would splice unrelated trailing content together).

`repair_truncated_json` is a last resort for genuinely truncated replies. In
`Reasoner._call_model`, a repaired plan is **rejected outright** if it
touches `RISKY_CONTENT_FIELDS` (`content`, `search`, `replace`) — a repaired
`expected_outcome` string is harmless; a repaired `content` field could
silently write truncated code to disk. If you add a new field that holds
file/code content verbatim, add it to `RISKY_CONTENT_FIELDS`.

## Param-name aliasing (Actioner)

Small models frequently use `file_path`/`filename`/`file` instead of `path`
even under `json_mode`. `PARAM_ALIASES` in `actioner.py` remaps these before
validation. If you see a new model consistently using a different alias,
add it there rather than special-casing it in `execute`.

## Adding a new tool

1. Add it to `TOOL_SCHEMAS` in `agents/tool_schemas.py` (single source of
   truth — both Reasoner's prompt and Actioner's validator read from this).
2. Implement the branch in `Actioner.execute`.
3. If it's cacheable (reads something that doesn't change turn-to-turn),
   wire it into `_update_file_cache`/`_is_cached_read` or the listing
   equivalent.
4. If it writes/deletes files, make sure it invalidates the right cache
   entries (see the `write_to_file`/`delete_file` handling in
   `_update_file_cache`, which also clears `listing_cache` entirely since a
   write anywhere could change any directory listing).

## Known rough edges (as of this session)

- `agents/actioner.py` has a duplicate `class Actioner:` definition (an
  empty one immediately followed by the real one) — harmless (Python just
  uses the second), but worth cleaning up if you're in that file anyway.
- `_use_venv_python` and `_extract_large_values`/`_restore_large_values` are
  defined as **module-level** functions in `actioner.py` but are called as
  `self._use_venv_python(...)` / bare calls elsewhere — double check
  indentation/binding if you touch that region.
- `Agent.solve` calls `_update_file_cache(current_state, action, result)`
  twice in a row (once under a `5a.` comment, once under `6.`) — idempotent
  today since it's the same inputs, but only one call is needed.