# Local LLM (Ollama) Coding Conventions

## One client, one JSON extractor
- `agents/ollama_client.py::OllamaClient` is the SINGLE Ollama HTTP client.
  Never write a second copy of it — both Reasoner and Actioner import this
  exact class.
- `agents/json_utils.py::extract_json_object` / `repair_truncated_json` are
  the SINGLE JSON-extraction helpers. Never reintroduce a greedy
  `\{.*\}` regex — it silently splices unrelated braces together whenever a
  model emits more than one `{...}`-looking chunk.
- Only `message["content"]` is ever parsed as the model's answer. Never
  treat `message["thinking"]` (hidden reasoning trace) as the final JSON —
  planning calls always pass `think=False`.

## num_ctx vs num_predict (this matters more than it looks)
- `num_ctx` (total context window: prompt + generation) is usually the REAL
  cause of mid-JSON truncation, not `num_predict`. Ollama's per-model
  default `num_ctx` (often 2048–4096) is frequently too small once tool
  schemas + prior tool results + repo state are embedded in the prompt.
- Any planning/action prompt that embeds tool results, file contents, or
  repo state should set `num_ctx` generously (8192–16384). See
  `DEFAULT_NUM_CTX` in `agents/reasoner.py` and `agents/actioner.py`.
- If you see "Ollama returned an empty content response" or JSON parse
  failures, check `prompt_eval_count` / `eval_count` against `num_ctx` in the
  logged diagnostics before assuming the model just failed — a near-full
  context window is the more likely cause.
- `repair_truncated_json` is a LAST RESORT for a genuinely truncated reply
  (closes an unterminated string/braces). Prefer fixing the underlying
  `num_ctx` budget over relying on this repair path as normal behavior.

## Prompt truncation
- Anything embedded back into a prompt (previous tool output, file reads)
  must go through the shared `MAX_OUTPUT_CHARS` truncation
  (`agents/reasoner.py::_truncate`) — full untruncated output still lives in
  `previous_actions` / benchmark logs, only what's fed back into the LLM is
  capped.

## Path handling
- `Actioner._resolve_path` is the ONLY place responsible for anchoring a
  relative path to `self.workspace_dir`. The LLM-facing prompt in
  `Actioner.plan_action` explicitly forbids inventing other paths/projects —
  don't relax that instruction, and don't add path-resolution logic anywhere
  else.

## Timeouts are two different knobs — don't conflate them
- `reasoner.timeout_seconds` / `actioner.timeout_seconds` (per-config, per
  single Ollama request) are distinct from `agent.timeout` (whole-task
  budget across all iterations). See `configs/*.yaml` and
  `experiments/run_experiment.py`. Never use one where the other is meant.
