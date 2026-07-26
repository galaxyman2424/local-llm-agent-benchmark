"""Reasoner for task analysis and planning.

The Reasoner is responsible ONLY for deciding WHAT should happen next. It
never executes tools itself -- it hands a structured plan to the Actioner,
which turns that plan into exactly one concrete tool call
(``Actioner.plan_action``), which the deterministic ``Actioner.execute``
then runs. Enforcing this separation is why ``Reasoner.plan`` returns a
``next_action`` / ``parameters`` / ``expected_outcome`` triple rather than
anything that looks like an already-executed result.
"""

from __future__ import annotations

from typing import Any

from .tool_schemas import schema_prompt_block
from .json_utils import extract_json_object, repair_truncated_json

# Truncate any single previous tool result embedded back into the prompt so
# a large file read/test output doesn't blow the context window. Full,
# untruncated output is still kept in `previous_actions`/benchmark logs --
# only what gets fed back into the LLM prompt is capped here.
MAX_OUTPUT_CHARS = 8000

# How many times the same (tool, parameters) pair can repeat in a row
# before we inject explicit "you are stuck" feedback into the prompt.
REPEATED_ACTION_THRESHOLD = 2

# Ollama's default num_ctx (often 2048-4096) is frequently too small once a
# prompt embeds tool schemas + several previous tool results + repo state --
# the model then hits the context ceiling mid-generation and the reply
# truncates well before num_predict would ever kick in. Set this generously
# for every planning call.
DEFAULT_NUM_CTX = 16384


class Reasoner:
    """Reasoning component that analyzes tasks and produces action plans."""

    def __init__(
        self,
        *,
        model_id: str = "qwen3.5:9b",
        timeout_seconds: float = 120.0,
        num_ctx: int = DEFAULT_NUM_CTX,
    ):
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx

    def analyze(self, task_description: str, state_context: dict[str, Any]) -> dict[str, Any]:
        """Analyze the current repository state and generate a high-level plan.

        This is a coarse, non-binding summary (used for logging/telemetry);
        the actual step-by-step decisions come from :meth:`plan`.
        """
        prompt = f"""You are an expert software engineer. Analyze this task and the current state to determine what needs to be done.

TASK DESCRIPTION:
{task_description}

CURRENT STATE CONTEXT:
{_truncate(state_context)}

Provide your analysis as a structured JSON response with these fields:
- "understanding": A clear summary of what the task requires
- "plan": An ordered list of specific actions needed (file reads, edits, test runs)
- "expected_outcome": What should be true after all actions complete"""

        analysis = {
            "understanding": f"Task requires {task_description[:100]}...",
            "plan": [],
            "expected_outcome": "Repository should be in correct state with all changes applied.",
        }

        result = self._call_model(prompt)
        if isinstance(result, dict):
            analysis.update(result)

        return analysis

    def plan(
        self,
        task: str,
        current_state: dict[str, Any],
        previous_actions: list[dict],
        avoid_action: tuple[str, str] | None = None,  # (tool, json_stringified_params)
    ) -> dict[str, Any] | None:

        already_read = _already_read_files(previous_actions)
        already_read_block = (
            f"\nFILES ALREADY READ THIS RUN (do NOT read_file these again unless "
            f"you just edited them and need to re-check): {already_read}\n"
            if already_read else ""
        )

        avoid_block = ""
        if avoid_action is not None:
            avoid_tool, avoid_params = avoid_action
            avoid_block = f"""
        HARD CONSTRAINT: You are FORBIDDEN from choosing this exact action again
        right now -- you already did this and it made no further progress:
            tool={avoid_tool} parameters={avoid_params}
        You MUST pick a genuinely different tool call (or "done"). Re-reading a
        file you already have the full contents of is NOT allowed.
        """

        """Generate the next single action to take.

        Parameters
        ----------
        task
            The problem description from the SWE-bench instance.
        current_state
            Current repository/agent state (workspace, test command, last
            result, etc.).
        previous_actions
            Full history of ``{"iteration", "reasoner_plan", "action",
            "result"}`` records taken so far this run -- actual tool
            outputs, not just action names, so the Reasoner can see what
            actually happened.

        Returns
        -------
        dict | None
            ``{"next_action": ..., "parameters": ..., "expected_outcome": ...}``.
            ``next_action`` is either a tool name from ``TOOL_SCHEMAS`` or
            the meta-action ``"done"`` to signal the Reasoner believes no
            further action is needed (e.g. tests already pass). Returns
            ``None`` if the model could not be reached or produced no
            usable plan after a retry.
        """
        loop_warning = _loop_warning(previous_actions)

        prompt = f"""You are the reasoning component of a software engineering agent.

Your job is to select exactly ONE next action. You do NOT execute tools
yourself -- a separate component (the Actioner) will translate your choice
into a concrete tool call.

TASK:
{task}

CURRENT STATE:
{_truncate(current_state)}

RECENT PREVIOUS ACTIONS AND THEIR RESULTS (most recent last):
{_truncate(_format_previous_actions(previous_actions))}
{loop_warning}
Available tools and their required/optional parameters:
{schema_prompt_block()}

You may also choose the special action "done" (with empty parameters) if
you believe the task is already complete (e.g. the tests already pass and
no further changes are needed).

Guidance:
- Prefer to search and read before you write or edit.
- Use search_code with concrete identifiers extracted from the task
  (function/class names, file names, error messages) -- never paste the
  entire task description as a search query.
- Only choose write_to_file when a brand new file is genuinely required;
  prefer replace_in_file for editing existing files.
- Never invent a project, path, or workspace unrelated to this task.

Return ONLY a JSON object with exactly these fields:

{{
"next_action": "tool_name_or_done",
"parameters": {{}},
"expected_outcome": "short description"
}}

Rules:
- Select exactly one next action.
- Do not output Markdown.
- Do not output explanations outside the JSON.
- Keep the response concise.
- Do not repeat the task description.
"""
        result = self._call_model(prompt)
        if result is None:
            # One retry with a shorter, stripped-down prompt before giving up.
            print("[Reasoner.plan] Empty/invalid model reply, retrying once with a shorter prompt.")
            short_prompt = f"""You are the reasoning component of a software engineering agent.
Select exactly ONE next action as JSON: {{"next_action": "tool_name_or_done", "parameters": {{}}, "expected_outcome": "..."}}

TASK (truncated): {task[:400]}

Available tools:
{schema_prompt_block()}

Return only the JSON object, nothing else."""
            result = self._call_model(short_prompt, num_predict=1024)

        if not isinstance(result, dict):
            print("[Reasoner.plan] No usable plan after retry; giving up for this iteration.")
            return None

        print("=" * 20, "REASONER PROMPT", "=" * 20)
        print(prompt)
        print("=" * 60)

        return {
            "next_action": result.get("next_action", result.get("action", "search_code")),
            "parameters": result.get("parameters", result.get("params", {})),
            "expected_outcome": result.get("expected_outcome", result.get("outcome", "")),
        }

    def _call_model(self, prompt: str, *, num_predict: int = 4096) -> Any:
        """Call the reasoning model and parse its JSON reply into a dict.

        Returns ``None`` (rather than raising) if the model is unreachable,
        returns empty content, or the reply isn't valid JSON even after a
        repair attempt, so callers can retry or fall back to a heuristic.
        Only ``message.content`` is ever parsed as the action JSON --
        ``message.thinking`` (the model's hidden reasoning trace, when
        thinking mode is on) is never treated as the final answer.
        """
        import json as _json

        from .ollama_client import OllamaClient

        client = OllamaClient(model=self.model_id, timeout_seconds=self.timeout_seconds)
        try:
            response = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.6,
                json_mode=True,
                num_predict=num_predict,
                num_ctx=self.num_ctx,
                keep_alive=0,
                think=False,
            )
        except RuntimeError as e:
            print(f"[Reasoner] Ollama request failed: {e}")
            return None

        message = response.get("message", {})
        text = message.get("content", "")

        if not text.strip():
            print("[Reasoner] Ollama returned an empty content response.")
            print("[Reasoner] Done reason: {}".format(response.get("done_reason", "unknown")))
            print("[Reasoner] Thinking length: {}".format(len(message.get("thinking", "") or "")))
            print("[Reasoner] prompt_eval_count={} eval_count={} (if eval_count is near num_predict "
                  "or prompt_eval_count+eval_count is near num_ctx={}, the context window is too "
                  "small for this prompt).".format(
                      response.get("prompt_eval_count", "?"), response.get("eval_count", "?"), self.num_ctx))
            return None

        candidate = extract_json_object(text)

        if candidate is None:
            # Likely a context-window truncation (see DEFAULT_NUM_CTX note
            # above) rather than a genuinely malformed reply -- try to
            # repair it (close the unterminated string/braces) before
            # giving up entirely.
            repaired = repair_truncated_json(text)
            if repaired is not None:
                try:
                    parsed = _json.loads(repaired)
                    print("[Reasoner] Model reply was truncated mid-JSON (likely num_ctx too small "
                          f"for this prompt: prompt_eval_count={response.get('prompt_eval_count', '?')} "
                          f"eval_count={response.get('eval_count', '?')} num_ctx={self.num_ctx}); "
                          "repaired it and continuing.")
                    return parsed
                except _json.JSONDecodeError:
                    pass

            print("[Reasoner] Model reply contained no JSON object (and could not be repaired)")
            print(f"[Reasoner] done_reason={response.get('done_reason', '?')} "
                  f"prompt_eval_count={response.get('prompt_eval_count', '?')} "
                  f"eval_count={response.get('eval_count', '?')} num_ctx={self.num_ctx}")
            print("[Reasoner] Raw model response:")
            print(repr(text[:2000]))
            return None
        try:
            return _json.loads(candidate)
        except _json.JSONDecodeError as e:
            print(f"[Reasoner] Model reply was not valid JSON: {e}")
            return None


def _truncate(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text) - MAX_OUTPUT_CHARS} more chars]"
    return text


def _format_previous_actions(previous_actions: list[dict], *, last_n: int = 5) -> str:
    """Render the most recent previous actions (with real tool results)."""
    if not previous_actions:
        return "(none yet)"
    recent = previous_actions[-last_n:]
    lines = []
    for record in recent:
        action = record.get("action", {})
        result = record.get("result", {})
        result_text = str(result.get("result", result.get("error", "")))
        lines.append(
            f"- tool={action.get('tool')} params={action.get('parameters')} "
            f"-> success={'error' not in result} "
            f"result={result_text[:4000]}"  # was [:500] — was hiding almost all read_file content
        )
    return "\n".join(lines)


def _loop_warning(previous_actions: list[dict]) -> str:
    """Detect immediate repetition of the same (tool, parameters) action and
    inject explicit feedback telling the Reasoner it must change course.
    """
    if len(previous_actions) < REPEATED_ACTION_THRESHOLD:
        return ""

    def _key(record: dict) -> tuple:
        action = record.get("action", {})
        return (action.get("tool"), _json_stable(action.get("parameters", {})))

    tail = previous_actions[-REPEATED_ACTION_THRESHOLD:]
    keys = [_key(r) for r in tail]
    if len(set(keys)) != 1:
        return ""

    last = tail[-1]
    action = last.get("action", {})
    result = last.get("result", {})
    return f"""
WARNING: The previous action did not make progress and was repeated
{REPEATED_ACTION_THRESHOLD} times in a row without changing the outcome.

Previous action:
{action}

Previous result:
{str(result)[:1000]}

You MUST choose a different action or inspect a more specific file/query
than before. Do not repeat the same tool call with the same parameters.
"""
def _already_read_files(previous_actions: list[dict]) -> list[str]:
    """Return paths that were successfully read_file'd anywhere in history
    (not just the last N shown in _format_previous_actions), so the
    Reasoner has an explicit checklist even once a file scrolls out of the
    'recent actions' window.
    """
    seen: list[str] = []
    for record in previous_actions:
        action = record.get("action", {})
        result = record.get("result", {})
        if action.get("tool") == "read_file" and "error" not in result:
            path = (action.get("parameters") or {}).get("path")
            if path and path not in seen:
                seen.append(path)
    return seen

def _json_stable(value: Any) -> str:
    import json as _json
    try:
        return _json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def _already_read_files(previous_actions: list[dict]) -> list[str]:
    """Files successfully read at least once this run."""
    seen = []
    for record in previous_actions:
        action = record.get("action", {})
        result = record.get("result", {})
        if action.get("tool") == "read_file" and "error" not in result:
            path = action.get("parameters", {}).get("path")
            if path and path not in seen:
                seen.append(path)
    return seen
