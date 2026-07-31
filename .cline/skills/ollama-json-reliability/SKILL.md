---
name: ollama-json-reliability
description: Use when a Reasoner or Actioner call to a local Ollama model returns empty content, truncated/malformed JSON, or when tuning num_ctx, num_predict, or timeout_seconds for local models in this project. Trigger on mentions of "empty content", "done_reason: length", JSON parse errors, num_ctx, num_predict, or a model hanging/timing out.
---

# Ollama JSON Reliability

## The two knobs people confuse
- **`num_predict`** caps how many tokens the model is allowed to GENERATE.
- **`num_ctx`** caps the TOTAL context window (prompt tokens + generated
  tokens). Ollama's per-model default is often only 2048-4096.

In this project, prompts embed tool schemas, several previous tool results,
and repo state -- that alone can consume most of a small default `num_ctx`,
leaving almost no room to generate before hitting the ceiling. The reply
then truncates mid-JSON well before `num_predict` would ever matter, and the
exact cutoff point shifts slightly between retries as prompt content shifts.
**If you're debugging truncated/empty JSON, check `num_ctx` first.**

Both `agents/reasoner.py` and `agents/actioner.py` already set
`DEFAULT_NUM_CTX = 16384` for this reason -- don't lower it without a
specific reason, and raise it further if prompts grow (e.g. embedding larger
file reads).

## Diagnosing an empty-content response
When `message["content"]` is empty, both `Reasoner._call_model` and
`Actioner.plan_action` already log:
```
done_reason=...
prompt_eval_count=... eval_count=...
```
If `eval_count` is near `num_predict`, or `prompt_eval_count + eval_count` is
near `num_ctx`, the context window is too small for that prompt -- raise
`num_ctx`, not `num_predict`.

## The shared JSON extraction path (don't duplicate it)
`agents/json_utils.py` has exactly two functions both Reasoner and Actioner
use:
- `extract_json_object(text)` -- brace-depth-aware scan for the first
  complete, balanced `{...}` object (correctly ignores braces inside quoted
  strings). This deliberately replaces a naive greedy regex, which would
  splice an unrelated trailing chunk of text into the JSON.
- `repair_truncated_json(text)` -- LAST RESORT: closes an unterminated
  string / balances open braces+brackets when `extract_json_object` found
  nothing complete. This is a strong signal the reply was cut off by
  `num_ctx`, not a formatting mistake by the model.

Never add a second JSON-extraction implementation -- both callers must keep
using this shared pair so a fix in one place fixes both.

## Timeout knobs are also two different things
- `reasoner.timeout_seconds` / `actioner.timeout_seconds` (per config file,
  e.g. `configs/qwen_ornith.yaml`) bound a SINGLE Ollama HTTP request.
- `agent.timeout` bounds the WHOLE task across all iterations.
Raising one to fix a problem caused by the other is a common mistake --
confirm which one actually applies before changing a config value.

## `think` mode
Planning calls (both Reasoner and Actioner) always pass `think=False`. If a
model spends its `num_predict` budget on a hidden reasoning trace instead of
the JSON answer, you'll see `done_reason: "length"` with empty `content` --
confirm `think=False` is actually being forwarded (see `OllamaClient.chat`'s
`think` parameter) before assuming the model itself is broken.
